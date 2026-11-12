/**
 * Families `deepseek_v41` (DeepSeek-V4.1: `DeepseekV41ForCausalLM`; language model under `text_config`) and
 * `deepseek_v4` (DeepSeek-V4: `DeepseekV4ForCausalLM`; flat config), built by ONE builder with a small
 * per-version {@link DeepseekProfile} for the facts no config field states. Not in transformers; the reference is
 * each repo's `inference/model.py` (`Block`, `Attention`, `Compressor`, `Indexer`, `Gate`, [`Engram`],
 * `Transformer.forward`).
 *
 * Every layer (`num_hidden_layers`: 40 in V4.1-Flash, 43 in V4-Flash): [Engram before layers `engram_layer_ids`],
 * then two mHC sublayers (`hc_mult` 4 streams; each with its own `hc_*_fn [(2+N)·N, N·D]`, base, scale, Sinkhorn,
 * and its own pre-norm): compressed sparse attention (CSA) and a MoE (all layers; one shared expert of width
 * `moe_intermediate_size`). After the last layer the streams collapse without a projection (`hc_pre` with the
 * last FFN's pre weights), then the final RMSNorm and the head.
 *
 * CSA layer roles come from the config lists, entry i of each (the first `num_hidden_layers` entries of
 * `compress_ratios`; the rest belong to the MTP stages):
 * - `compress_ratios[i]` r = 0 → sliding window only (no global branch).
 * - i ∈ `kv_source_layer_ids` → owns the compressed global KV (runs the compressor); otherwise it reads the cache
 *   of the latest source layer before i. Without the list (V4) every r > 0 layer owns its compressor.
 * - i ∈ `index_source_layer_ids` → runs its own indexer (keys pooled r tokens per key; V4.1: projected from the
 *   `head_dim`-wide compressed latent; V4: from the hidden state by the indexer's own compressor); it owns indexer
 *   keys iff it is also a KV source ("Full"), else it reads the source's keys ("Reindex"). Indexers after
 *   `candidate_source_layer_id` score at most `candidate_topk_blocks · candidate_block_size` candidates. Any other
 *   r > 0 layer reuses the top-k ("Reuse"). Without the list (V4) a layer runs an indexer iff r = 4 (model.py
 *   `Attention.__init__`: `if self.compress_ratio == 4: self.indexer = Indexer(...)`) and any other r > 0 layer
 *   attends every compressed entry ("All", `get_compress_topk_idxs`).
 * - r = 4 compressors pool overlapping windows (model.py `Compressor.overlap = compress_ratio == 4`).
 * DeepSeek-V4.1-Flash: 2 SWA-only, 4 Full (2, 8, 14, 20), 4 Reindex (24, 28, 32, 36), 30 Reuse, Engram at 1, 14.
 * DeepSeek-V4-Flash-0731: 2 SWA-only (0, 1), 21 Full with r = 4 (even 2..42), 20 All with r = 128 (odd 3..41), no
 * Engram; layers 0..`num_hash_layers`−1 route experts by token id (`tid2eid`), but the router GEMM still runs.
 * Prefill runs every layer on every token, as `model.py` does (no "Bounded Replay").
 *
 * Excluded: the DSpark MTP stages (estimated from the `dspark_*` fields; their count is
 * max(`num_nextn_predict_layers`, `len(compress_ratios)` − `num_hidden_layers`), see {@link dsparkStageCount}) and
 * the vision tower (estimated from `vision_config`). Approximations: the attention sink (H per layer), the router
 * biases (`gate.bias`, `gate.bias_vl`: 2·E per layer) are not counted, and the indexer key norm is counted as a
 * LayerNorm (2·hd_I instead of hd_I); together 960 params per layer at most. V4 also: the compressor APE (fp32,
 * r·coff·hd per owner layer), the final `hc_head` projection and the `tid2eid` table are not counted.
 */

import type { FfnParams, MixerParams } from '../../../ir/builders';
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import type {
  BlockDraftSpec,
  CsaGlobalSpec,
  IndexerSpec,
  LayerSpec,
  QuantConfig,
} from '../../../ir/model';
import type { ConfigReader } from '../reader';
import { createReader, isJsonObject } from '../reader';
import { withDraft } from '../drafter';
import type { FamilyBuildResult, FamilyEntry, JsonObject } from '../types';
import {
  layerParams,
  mhcCollapse,
  mhcResidual,
  readActivation,
  readMoeFfn,
  structuralErrors,
} from './mlaCommon';

/** The CSA fields every layer shares (heads, dims, ranks, window). */
type CsaBase = Omit<Extract<MixerParams, { kind: 'csa' }>, 'global'>;

/**
 * Role of a CSA layer, for notes and tests: window only, own KV + own indexer, own indexer on shared KV, reused
 * KV and top-k, or own KV with every compressed entry attended (DeepSeek-V4 r = 128 layers).
 */
export type CsaRole = 'swa' | 'full' | 'reindex' | 'reuse' | 'all';

/**
 * The CSA role of layer i (see the module header). `indexListGiven` false (no `index_source_layer_ids`, V4): a
 * layer that runs no indexer attends all its entries instead of reusing a top-k.
 */
export function csaRole(
  ratio: number,
  isKvSource: boolean,
  isIndexSource: boolean,
  indexListGiven = true,
): CsaRole {
  if (ratio === 0) return 'swa';
  if (isIndexSource) return isKvSource ? 'full' : 'reindex';
  return indexListGiven ? 'reuse' : 'all';
}

/**
 * What a DeepSeek version's `inference/model.py` fixes that no config field states. One per family entry:
 * {@link V41_PROFILE}, {@link V4_PROFILE}.
 */
export interface DeepseekProfile {
  /**
   * Where an indexer gets its keys. `'compressed_latent'` (V4.1 model.py `Indexer.wk [index_head_dim, head_dim]`):
   * one projection of the layer's `head_dim`-wide compressed latent. `'own_compressor'` (V4 model.py
   * `Indexer.compressor = Compressor(args, compress_ratio, index_head_dim, rotate=True)`): its own gated pooling
   * compressor over the hidden state (`wkv`, `wgate` h → coff·hd_I, APE, RMSNorm), overlapping at r = 4.
   */
  indexerKeys: 'compressed_latent' | 'own_compressor';
  /**
   * The architecture-native (QAT) cache formats, as family quant hints (a cache format the repo's quantization
   * metadata states explicitly wins).
   */
  cacheHints: (c: DeepseekConfig) => Partial<QuantConfig>;
  /** Version-specific approximation notes (what of that model.py the IR leaves out); may be empty. */
  notes: string[];
}

/**
 * DeepSeek-V4.1 (findings R2 §1, V4.1 model.py): window KV fp8 with a ue8m0 scale per 32 (`act_quant(kv,
 * fp8_block_size, …)`) → mxfp8; compressed global KV fp4 e2m1 with an e4m3 scale per 16 → nvfp4; indexer keys fp4
 * with a ue8m0 scale per 32 → mxfp4. Weight-module formats are not hinted: the quant readers derive them per repo
 * (the native checkpoint keeps the compressor and indexer wk / weights_proj in bf16; quant/moduleClass.ts marks
 * them).
 */
export const V41_PROFILE: DeepseekProfile = {
  indexerKeys: 'compressed_latent',
  cacheHints: () => ({
    kvCache: { dtype: 'mxfp8' },
    compressedKvCache: { dtype: 'nvfp4' },
    indexerKeyCache: { dtype: 'mxfp4' },
  }),
  notes: [],
};

/**
 * DeepSeek-V4 (V4 model.py): window KV and compressed KV quantize the hd − dr non-RoPE dims to fp8 e4m3 with a
 * ue8m0 scale per 64 and keep the dr RoPE dims bf16 (`act_quant(kv[..., :-rd], 64, …)` in `Attention` and in
 * `Compressor` with `rotate=False`): 583 B per 512-element vector. Indexer keys and queries: fp4 with a ue8m0 scale
 * per 32 (`fp4_act_quant(kv, 32, True)`) → mxfp4. The vLLM card's `--kv-cache-dtype fp8` and
 * `use_fp4_indexer_cache` agree.
 */
export const V4_PROFILE: DeepseekProfile = {
  indexerKeys: 'own_compressor',
  cacheHints: (c) => {
    const hd = c.csa.headDim;
    const dr = c.csa.ropeDim;
    // Formula: extraBitsPerElem = (dr·(16 − 8) + ((hd − dr)/64)·8)/hd over an fp8 base: RoPE dims bf16, one ue8m0
    //          scale per 64 non-RoPE dims (hd 512, dr 64: (512 + 56)/512 = 1.109375 → 583 B per vector)
    const extraBitsPerElem = (dr * 8 + ((hd - dr) / 64) * 8) / hd;
    const fp8Rope = { dtype: 'fp8_e4m3' as const, extraBitsPerElem };
    return { kvCache: fp8Rope, compressedKvCache: fp8Rope, indexerKeyCache: { dtype: 'mxfp4' } };
  },
  notes: [
    'compressor APE (fp32 positional bias of the pooling weights, r·coff·hd per layer) and the final hc_head projection not counted',
    'indexer key norm costed on the T·hd_I pre-pool keys (model.py normalizes the T/r pooled keys): at most (r−1)/r·T·hd_I extra elementwise elements per indexer layer, negligible',
    'indexer compressor key RoPE and Hadamard rotation (rotate_activation) not emitted, as for the generic indexer; elementwise, negligible',
  ],
};

/** Everything the layer builder reads from the config. */
export interface DeepseekConfig {
  hidden: number;
  csa: CsaBase;
  ratios: number[];
  /**
   * Layers that own a compressed KV (`kv_source_layer_ids`). Default when absent (V4): every layer i < n with
   * `compress_ratios[i]` > 0 (V4 model.py `Attention.__init__`: `if self.compress_ratio: self.compressor = …`).
   */
  kvSources: number[];
  /**
   * Layers that run an indexer (`index_source_layer_ids`). Default when absent (V4): every layer i < n with
   * `compress_ratios[i]` = 4 (V4 model.py `Attention.__init__`: `if self.compress_ratio == 4: self.indexer = …`).
   */
  indexSources: number[];
  /** True when `index_source_layer_ids` is given (V4.1): a non-indexer r > 0 layer reuses the top-k; else 'all'. */
  indexListGiven: boolean;
  candidateSource: number | undefined;
  maxScoredKeys: number | undefined;
  indexer: { heads: number; headDim: number; topK: number };
  streams: number;
  sinkhornIters: number;
  engramLayers: number[];
  engramRows: number[];
  engramHeadRows: number[][];
  engramRowDim: number;
  engramRowsPerToken: number;
}

/**
 * The global branch of layer i, or undefined for an SWA-only layer. Errors (a consumer with no earlier KV source)
 * go to `errors`.
 */
function csaGlobal(
  c: DeepseekConfig,
  p: DeepseekProfile,
  i: number,
  errors: string[],
): CsaGlobalSpec | undefined {
  const r = c.ratios[i];
  if (r === 0) return undefined;
  const ownsKv = c.kvSources.includes(i);
  const earlierSources = c.kvSources.filter((s) => s < i);
  // Formula: kvCacheFromLayer = max{ s ∈ kv_source_layer_ids : s < i } for a non-source layer
  const kvFrom = ownsKv || earlierSources.length === 0 ? undefined : Math.max(...earlierSources);
  if (!ownsKv && kvFrom === undefined)
    errors.push(`kv_source_layer_ids: layer ${i} has compress ratio ${r} but no earlier KV source`);
  let selection: CsaGlobalSpec['selection'];
  if (c.indexSources.includes(i)) {
    const keys: Partial<IndexerSpec> =
      p.indexerKeys === 'compressed_latent'
        ? // wk reads the compressed latent (`indexer.wk [index_head_dim, head_dim]`).
          { keyInputDim: c.csa.headDim }
        : // Its own gated compressor over h (`indexer.compressor.wkv / wgate`, APE), overlapping at r = 4.
          { keyPoolGate: true, ...(r === 4 ? { keyPoolOverlap: true } : {}) };
    const indexer: IndexerSpec = {
      ...c.indexer,
      keyPoolTokens: r,
      ...keys,
      ownsKeys: ownsKv,
      ...(c.candidateSource !== undefined && c.maxScoredKeys !== undefined && i > c.candidateSource
        ? { maxScoredKeys: c.maxScoredKeys }
        : {}),
    };
    selection = { kind: 'indexer', indexer };
  } else if (c.indexListGiven) {
    selection = { kind: 'reuse', topK: c.indexer.topK };
  } else {
    // No indexer in this layer and none to reuse: attend every compressed entry (V4 `get_compress_topk_idxs`).
    selection = { kind: 'all' };
  }
  return {
    compressRatio: r,
    // model.py `Compressor.overlap = compress_ratio == 4` (V4.1 has no r = 4 layer, so its IR is unchanged).
    ...(r === 4 && ownsKv ? { compressorOverlap: true } : {}),
    ...(kvFrom !== undefined ? { kvCacheFromLayer: kvFrom } : {}),
    selection,
  };
}

/** One decoder layer: [Engram], mHC + CSA, mHC + `ffn`. */
function deepseekLayer(
  c: DeepseekConfig,
  i: number,
  ratio: number,
  global: CsaGlobalSpec | undefined,
  ffn: FfnParams,
  name: string,
): LayerSpec {
  const N = c.streams;
  const D = c.hidden;
  const engramIndex = c.engramLayers.indexOf(i);
  const mixer: MixerParams = { ...c.csa, ...(ratio > 0 && global !== undefined ? { global } : {}) };
  return decoderLayer({
    name,
    hidden: D,
    mixer,
    ffn,
    residual: mhcResidual(N, D, c.sinkhornIters),
    ...(engramIndex >= 0
      ? {
          ngram: {
            name: 'engram',
            tableRows: c.engramRows[engramIndex],
            headRows: c.engramHeadRows[engramIndex],
            rowDim: c.engramRowDim,
            rowsPerToken: c.engramRowsPerToken,
            // Formula: wkv: rowsPerToken·rowDim → (N + 1)·D (one key per stream + a shared value)
            projections: [
              {
                name: 'engram.wkv',
                inputDim: c.engramRowsPerToken * c.engramRowDim,
                outputDim: (N + 1) * D,
              },
            ],
            outputDim: N * D,
            // Formula: q_weight, k_weight: 2·N·D
            vectorParams: 2 * N * D,
            // Engram tables are designed to live off-device (model card); UI hint only, the simulator's default
            // placement stays the device (design doc §2.10).
            offDeviceByDesign: true,
          },
        }
      : {}),
  });
}

/**
 * Contiguous Engram hash-head buckets: successive distinct primes ≥ engram_vocab_size, never reused across
 * layers. vLLM EngramLayout (7c9a2fa2, common/engram.py); the checkpoint's tableRows remains authoritative.
 * Undefined when counts or buckets exceed safe integers; the reader reports an invalid config, never a stuck search.
 */
function engramHeadRows(base: number, heads: number, layers: number): number[][] | undefined {
  if (!Number.isSafeInteger(base) || !Number.isSafeInteger(heads) || heads < 0) return undefined;
  const prime = (n: number): boolean => {
    if (n < 2) return false;
    if (n % 2 === 0) return n === 2;
    for (let d = 3; d * d <= n; d += 2) if (n % d === 0) return false;
    return true;
  };
  let next = base;
  const buckets: number[][] = [];
  for (let layer = 0; layer < layers; layer++) {
    const rows: number[] = [];
    for (let head = 0; head < heads; head++) {
      // Above MAX_SAFE_INTEGER, next++ may stop advancing. Reject before searching or emitting such a bucket.
      while (Number.isSafeInteger(next) && !prime(next)) next++;
      if (!Number.isSafeInteger(next)) return undefined;
      rows.push(next++);
    }
    buckets.push(rows);
  }
  return buckets;
}

/** Read the config fields the layers need; `n` = `num_hidden_layers` (for the V4 list defaults). */
function readDeepseekConfig(r: ConfigReader, n: number): DeepseekConfig {
  const hidden = r.int('hidden_size');
  const csa: CsaBase = {
    kind: 'csa',
    heads: r.int('num_attention_heads'),
    headDim: r.int('head_dim'),
    ropeDim: r.int('qk_rope_head_dim'),
    qLoraRank: r.int('q_lora_rank'),
    outGroups: r.int('o_groups'),
    outLoraRank: r.int('o_lora_rank'),
    windowTokens: r.int('sliding_window'),
  };
  const candidateBlocks = r.optInt('candidate_topk_blocks');
  const candidateBlockSize = r.optInt('candidate_block_size');
  const engramLayers = r.optNumArray('engram_layer_ids') ?? [];
  // Formula: rowsPerToken = (engram_max_ngram_size − 1) · engram_n_heads
  const engramRowsPerToken =
    engramLayers.length > 0 ? (r.int('engram_max_ngram_size') - 1) * r.int('engram_n_heads') : 0;
  const headRows =
    engramLayers.length > 0
      ? engramHeadRows(r.int('engram_vocab_size'), engramRowsPerToken, engramLayers.length)
      : [];
  if (headRows === undefined)
    r.errors.push(
      `${r.path ? `${r.path}.` : ''}engram_vocab_size: cannot generate safe integer hash-head buckets`,
    );
  const ratios = r.numArray('compress_ratios');
  const mainLayers = ratios.slice(0, n).map((ratio, i) => ({ ratio, i }));
  const indexList = r.optNumArray('index_source_layer_ids');
  return {
    hidden,
    csa,
    ratios,
    // Default (V4): every r > 0 layer owns its compressor (model.py `if self.compress_ratio: self.compressor = …`).
    kvSources:
      r.optNumArray('kv_source_layer_ids') ?? mainLayers.filter((l) => l.ratio > 0).map((l) => l.i),
    // Default (V4): an indexer exactly at r = 4 (model.py `if self.compress_ratio == 4: self.indexer = …`).
    indexSources: indexList ?? mainLayers.filter((l) => l.ratio === 4).map((l) => l.i),
    indexListGiven: indexList !== undefined,
    candidateSource: r.optInt('candidate_source_layer_id'),
    // Formula: maxScoredKeys = candidate_topk_blocks · candidate_block_size
    maxScoredKeys:
      candidateBlocks !== undefined && candidateBlockSize !== undefined
        ? candidateBlocks * candidateBlockSize
        : undefined,
    indexer: {
      heads: r.int('index_n_heads'),
      headDim: r.int('index_head_dim'),
      topK: r.int('index_topk'),
    },
    streams: r.int('hc_mult'),
    sinkhornIters: r.optInt('hc_sinkhorn_iters') ?? 0,
    engramLayers,
    engramRows: engramLayers.length > 0 ? r.numArray('engram_num_embeddings') : [],
    engramHeadRows: headRows ?? [],
    engramRowDim: engramLayers.length > 0 ? r.int('engram_head_dim') : 0,
    engramRowsPerToken,
  };
}

/**
 * Number of DSpark MTP stages in the checkpoint, and a note when the config fields disagree. With `dspark_block_size`
 * set, `compress_ratios` lists one entry per main layer AND per MTP stage, so its surplus over `num_hidden_layers`
 * also counts the stages. DeepSeek-V4-Flash-0731 says `num_nextn_predict_layers: 1` but lists 46 = 43 + 3 ratios,
 * and its inference/config.json says `n_mtp_layers: 3`; the safetensors I8 count is exactly 46 layers of routed
 * experts, so 3 is right. Without `dspark_block_size`: `num_nextn_predict_layers`. Unit: count.
 */
function dsparkStageCount(
  r: ConfigReader,
  c: DeepseekConfig,
  n: number,
): { stages: number; note?: string } {
  const nextn = r.optInt('num_nextn_predict_layers') ?? 0;
  if (r.optInt('dspark_block_size') === undefined) return { stages: nextn };
  // Formula: stages = max(num_nextn_predict_layers, len(compress_ratios) − num_hidden_layers)
  const fromRatios = Math.max(c.ratios.length - n, 0);
  const stages = Math.max(nextn, fromRatios);
  if (stages === nextn) return { stages };
  return {
    stages,
    note: `num_nextn_predict_layers ${nextn}, but compress_ratios lists ${fromRatios} MTP stages (DeepSeek-V4-Flash-0731: inference/config.json n_mtp_layers 3 agrees); using ${stages}`,
  };
}

/**
 * The DSpark MTP stages (`mtp.*`): `stages` decoder layers ({@link dsparkStageCount}; window-only CSA, MoE with
 * `dspark_n_routed_experts` / `dspark_num_experts_per_tok` when given, else the main MoE, mHC). Empty without stages.
 */
function dsparkStages(
  r: ConfigReader,
  c: DeepseekConfig,
  moe: FfnParams,
  n: number,
  stages: number,
): LayerSpec[] {
  const experts = r.optInt('dspark_n_routed_experts');
  const topK = r.optInt('dspark_num_experts_per_tok');
  const draftMoe: FfnParams =
    moe.kind === 'moe'
      ? { ...moe, ...(experts ? { experts } : {}), ...(topK ? { topK } : {}) }
      : moe;
  // DSpark attention is window-only (`assert self.compress_ratio == 0`); a non-zero ratio is ignored.
  return repeatLayers(stages, (j) =>
    deepseekLayer(
      { ...c, engramLayers: [] },
      n + j,
      c.ratios[n + j] ?? 0,
      undefined,
      draftMoe,
      `mtp.${j}`,
    ),
  );
}

/**
 * Params of the DSpark MTP stages (`mtp.*`), for the param check: the {@link dsparkStages} layers + the first
 * stage's `main_proj` (targets·h → h) and `main_norm` + the last stage's `norm`, Markov head (embed and head,
 * vocab × `dspark_markov_rank`) and confidence projection (h + rank → 1). Embedding and head are shared with the
 * main model. Unit: params.
 */
function dsparkParamsEstimate(
  r: ConfigReader,
  c: DeepseekConfig,
  stages: LayerSpec[],
  vocab: number,
): number {
  if (stages.length === 0) return 0;
  const total = stages.reduce((sum, layer) => sum + layerParams(layer), 0);
  const h = c.hidden;
  const targets = (r.optNumArray('dspark_target_layer_ids') ?? []).length;
  const rank = r.optInt('dspark_markov_rank') ?? 0;
  // Formula: P = Σ_j P_stage + targets·h·h + h (main_proj, main_norm) + h + 2·vocab·rank + (h + rank) (norm, Markov, confidence)
  return total + targets * h * h + h + h + 2 * vocab * rank + (h + rank);
}

/**
 * Vision tower + aligner of DeepSeek-V4.1 from `vision_config`, for the param check: `num_hidden_layers` blocks
 * of `4·d²` (wqkv, wo) + `3·d·ffn` (gated MLP; `w1` holds gate and up) and the aligner `w1: d·s² → h`,
 * `w2: h → h` with s = `downsample_ratio`. Biases, norms and patch embeddings are left out. Unit: params.
 */
function visionParamsEstimate(vision: JsonObject | undefined, hidden: number): number {
  if (vision === undefined) return 0;
  const num = (k: string) => (typeof vision[k] === 'number' ? vision[k] : 0);
  const depth = num('num_hidden_layers');
  const d = num('hidden_size');
  const ffn = num('intermediate_size');
  const s = num('downsample_ratio') || 1;
  // Formula: P_vision = depth·(4·d² + 3·d·ffn) + d·s²·h + h·h
  return depth * (4 * d * d + 3 * d * ffn) + d * s * s * hidden + hidden * hidden;
}

/**
 * The in-checkpoint DSpark drafter as a block drafter (design doc §3.2) with the {@link dsparkStages} as
 * `moeLayers`: window = the CSA `sliding_window`, target features of `dspark_target_layer_ids` (`main_proj`), Markov
 * head of `dspark_markov_rank`, `blockSize = dspark_block_size + 1` (DSpark's block counts draft positions, so the
 * verify width is one more). The dense head/width fields mirror the stages' CSA and expert shapes for display; the
 * lowering uses `moeLayers`. Undefined without stages, block size or target layers.
 */
function dsparkDraft(
  r: ConfigReader,
  c: DeepseekConfig,
  stages: LayerSpec[],
  vocab: number,
  activation: BlockDraftSpec['activation'],
  repoId: string,
): BlockDraftSpec | undefined {
  const block = r.optInt('dspark_block_size');
  const targets = r.optNumArray('dspark_target_layer_ids');
  if (stages.length === 0 || block === undefined || targets === undefined || targets.length === 0)
    return undefined;
  const moe = stages[0].blocks.find((b) => b.kind === 'moe');
  const rank = r.optInt('dspark_markov_rank');
  return {
    kind: 'block_drafter',
    method: 'dspark',
    hidden: c.hidden,
    vocab,
    layers: stages.length,
    heads: c.csa.heads,
    kvHeads: 1,
    headDim: c.csa.headDim,
    intermediate: moe !== undefined && moe.kind === 'moe' ? moe.expertIntermediate : 0,
    gated: true,
    activation,
    windowTokens: c.csa.windowTokens,
    targetLayerIds: targets,
    // Formula: blockSize = dspark_block_size + 1 (k ≤ dspark_block_size drafts, verify width k + 1)
    // See docs/hf-loader.md ("Drafters"), docs/simulation-model.md §13 ("Draft spec")
    blockSize: block + 1,
    ...(rank !== undefined ? { markovRank: rank } : {}),
    moeLayers: stages,
    source: `${repoId} (mtp.* DSpark stages)`,
  };
}

/**
 * Build the DeepSeek-V4 / V4.1 IR from its text config, with the version's {@link DeepseekProfile}.
 */
export function buildDeepseek(
  text: JsonObject,
  top: JsonObject,
  repoId: string,
  profile: DeepseekProfile,
): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const act = readActivation(r);
  const c = readDeepseekConfig(r, n);
  const moe = readMoeFfn(r, act);
  if (c.ratios.length < n)
    r.errors.push(`${r.path}.compress_ratios: ${c.ratios.length} entries for ${n} layers`);
  if (c.engramRows.length < c.engramLayers.length)
    r.errors.push(`${r.path}.engram_num_embeddings: one entry per engram layer needed`);
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const errors: string[] = [];
  const roles: CsaRole[] = [];
  const layers = repeatLayers(n, (i) => {
    const ratio = c.ratios[i];
    roles.push(
      csaRole(ratio, c.kvSources.includes(i), c.indexSources.includes(i), c.indexListGiven),
    );
    return deepseekLayer(c, i, ratio, csaGlobal(c, profile, i, errors), moe, `layer.${i}`);
  });
  if (errors.length > 0) return { kind: 'invalid_config', errors };
  const model = decoderModel({
    name: repoId,
    hidden: c.hidden,
    vocab,
    tieEmbeddings: tie,
    layers,
    residualStreams: c.streams,
    beforeHead: [mhcCollapse(c.streams, c.hidden)],
    source: repoId,
  });
  const structural = structuralErrors(model);
  if (structural.length > 0) return { kind: 'invalid_config', errors: structural };

  const vision = isJsonObject(top.vision_config)
    ? top.vision_config
    : isJsonObject(text.vision_config)
      ? text.vision_config
      : undefined;
  const visionParams = visionParamsEstimate(vision, c.hidden);
  const stageCount = dsparkStageCount(r, c, n);
  const stages = dsparkStages(r, c, moe, n, stageCount.stages);
  const mtp = dsparkParamsEstimate(r, c, stages, vocab);
  const draft = dsparkDraft(r, c, stages, vocab, act, repoId);
  const count = (role: CsaRole) => roles.filter((x) => x === role).length;
  const allNote = count('all') > 0 ? `, ${count('all')} own KV, attend all entries` : '';
  const hashLayers = r.optInt('num_hash_layers') ?? 0;
  const notes = [
    `CSA layers: ${count('swa')} window-only, ${count('full')} own KV + indexer, ${count('reindex')} own indexer on shared KV, ${count('reuse')} reuse KV and top-k${allNote}`,
    `mHC residual: ${c.streams} streams; Engram tables before layers ${c.engramLayers.join(', ') || 'none'} (total params, not active)`,
    'prefill runs every layer on every prompt token (reference model.py; no Bounded Replay)',
    'attention sink, router biases not counted; indexer key norm counted as LayerNorm (< 1 k params per layer)',
    ...(hashLayers > 0
      ? [
          `layers 0–${hashLayers - 1} route by token id (tid2eid); the router GEMM still runs; the table is not counted`,
        ]
      : []),
    ...(stageCount.note !== undefined ? [stageCount.note] : []),
    ...profile.notes,
  ];
  const excludedParts: string[] = [];
  if (mtp > 0) excludedParts.push('DSpark MTP stages');
  if (visionParams > 0) excludedParts.push('vision tower + aligner');
  return {
    kind: 'ok',
    model: withDraft(model, draft),
    notes,
    mtpParamsEstimate: mtp,
    visionParamsEstimate: visionParams,
    excludedParts,
    // Architecture-native cache formats of this version (see the profiles). They are defaults: a cache format the
    // repo's quantization metadata states explicitly wins.
    quantHints: profile.cacheHints(c),
  };
}

/** Family entry for DeepSeek-V4.1 (CSA + hyper-connections + Engram). */
export const deepseekV41Family: FamilyEntry = {
  family: 'deepseek_v41',
  modelTypes: ['deepseek_v41', 'deepseek_v41_text'],
  architectures: ['DeepseekV41ForCausalLM'],
  build: (text, top, files) => buildDeepseek(text, top, files.id, V41_PROFILE),
};

/**
 * Family entry for DeepSeek-V4 (CSA with r = 4 indexer layers and r = 128 attend-all layers, hyper-connections,
 * hash-routed first layers, DSpark MTP; no Engram). Same builder as V4.1 (decision (a) of the V4 support task:
 * the differences are config defaults plus {@link V4_PROFILE}).
 */
export const deepseekV4Family: FamilyEntry = {
  family: 'deepseek_v4',
  modelTypes: ['deepseek_v4'],
  architectures: ['DeepseekV4ForCausalLM'],
  build: (text, top, files) => buildDeepseek(text, top, files.id, V4_PROFILE),
};
