/**
 * The op catalogue: the unit of work the engine costs.
 *
 * An op is **the work of ONE device on its pipeline stage's critical path, for one microbatch step**.
 * Sharding is assumed balanced, so all devices of a stage are symmetric and one device's ops describe the stage.
 * Every count below is therefore per device (after tensor/expert parallel sharding) unless a field says otherwise.
 *
 * Shapes are concrete numbers (elements, tokens, sequences), never symbolic. FLOP and byte counts are derived
 * from these fields by `ir/metrics.ts` (docs/simulation-model.md §3), not stored on the op.
 */

import type { AccumulateType, Dtype, MathFormat, TensorFormat } from './dtype';

/** Any op one device executes. Dispatch with `switch (op.kind)`. */
export type Op =
  | GemmOp
  | AttentionOp
  | IndexerOp
  | RecurrentOp
  | ElementwiseOp
  | EmbeddingOp
  | CollectiveOp
  | TransferOp;

/**
 * A (possibly batched/grouped) matrix multiply against weights:
 * `out[b] (m×n) = in[b] (m×k) · W[b] (k×n)` for `b` in `0..batch-1`.
 * Batch entries have DISTINCT weights (grouped GEMM: MoE experts, MLA per-head absorb matrices), except that a MoE
 * grouped GEMM reads only `distinctWeightMatrices` of them (the expected distinct experts, docs §5).
 * FLOP = 2·batch·m·n·k (dense-equivalent; 2:4 sparsity does not halve the count). See docs §3 "GEMM".
 */
export interface GemmOp {
  kind: 'gemm';
  /** Stable dotted name for reports, e.g. `'attn.qkv_proj'`, `'moe.gate_up'`. */
  name: string;
  /** Number of independent GEMMs, each with its own weight matrix (1 for a plain linear layer). Count. */
  batch: number;
  /**
   * Weight matrices actually read, when fewer than `batch` (≤ batch; real, an expectation): a MoE grouped GEMM's
   * compute runs on a balanced grid of `batch` experts while only the `U` expected distinct experts under uniform
   * routing are touched (docs §5 "Expected distinct experts"). Scales the weight (and bias) bytes only; FLOP, tiles
   * and activations keep `batch`. Absent: `batch`. Unit: weight matrices (count, real).
   */
  distinctWeightMatrices?: number;
  /** Rows of the input/output per batch entry: tokens on this device. Unit: elements (= tokens). */
  m: number;
  /** Output features per batch entry (weight columns on this device, after sharding). Unit: elements. */
  n: number;
  /** Input features per batch entry (the reduction dimension). Unit: elements. */
  k: number;
  /** Storage dtype of the activation input `in`, read from memory. */
  inputDtype: Dtype;
  /** Storage format of the weights `W` (with quantization overhead), read from memory once per step. */
  weight: TensorFormat;
  /** Storage dtype of the output `out`, written to memory. */
  outputDtype: Dtype;
  /** Matmul input class the tensor cores run in (selects the peak; from `resolveGemmMath`). */
  math: MathFormat;
  /** Accumulator type (selects the peak; from `resolveAccumulate`). */
  accumulate: AccumulateType;
  /**
   * Structured sparsity of the weight operand: `'dense'`, or `'2:4'` (2 of every 4 weights kept; uses the
   * device's 2:4 sparse peak and stores half the values plus 2-bit indices per kept value).
   */
  sparsity: 'dense' | '2:4';
  /**
   * Output widths of the physical weight matrices fused into this GEMM's `n` (they sum to `n`), e.g. q, k, v of
   * a `qkv_proj`. Only matters for formats with per-matrix side vectors (EXL3 `perDimBits`): each part carries
   * its own vectors. Absent: one matrix. Unit: elements.
   */
  outParts?: number[];
  /**
   * Total side-vector entries (per-row plus per-column `perDimBits`-wide values, summed over the batch) this op
   * is charged, replacing the shape-derived count. For an op that executes only a slice of a physical matrix:
   * the MLA absorbed `q_absorb` / `v_absorb` split the one stored `kv_b` matrix, and their entries sum to the
   * matrix's `rows + cols` so both forms store the same bytes. Unit: elements.
   */
  sideVectorElems?: number;
  /** Dtype of the bias vector; present iff the GEMM adds a bias of `n` elements per batch entry. */
  biasDtype?: Dtype;
}

/** Which keys each query attends to. */
export type KvSelection =
  /** Every causally visible key (or every key, if not causal). */
  | { kind: 'full' }
  /** Sliding window: each query sees only the last `windowTokens` visible keys. Unit: tokens. */
  | { kind: 'window'; windowTokens: number }
  /** Top-k sparse attention: each query sees `topK` keys chosen by a preceding {@link IndexerOp}. Unit: tokens. */
  | { kind: 'topk'; topK: number };

/** A stream of K/V (or latent) elements read or written per token, per device per layer. */
export interface KvStream {
  /** Storage format of the stream's elements (e.g. the KV-cache format). */
  format: TensorFormat;
  /**
   * Elements per token per device per layer, K and V together (e.g. GQA: 2·kvHeadsLocal·headDim;
   * MLA latent: kvLoraRank + qkRopeDim). 0 = no traffic. Unit: elements/token.
   */
  elemsPerToken: number;
}

/**
 * Fused attention core, `softmax(Q·Kᵀ)·V`, FlashAttention-style: scores never reach memory, so only Q, K/V and
 * the output move. Covers MHA/GQA/MQA, sliding window, top-k sparse, MLA (absorbed: kvHeads 1, qk ≠ v dims), and
 * a second key set in the same softmax ({@link AttentionKvSource}: DeepSeek-V4.1 window + compressed entries).
 * FLOP = 2·seqs·heads·(pairs + pairs_secondary)·(qkHeadDim + vHeadDim), the causal triangle counted exactly.
 * See docs §3 "Attention".
 */
export interface AttentionOp {
  kind: 'attention';
  /** Stable dotted name, e.g. `'attn.core'`. */
  name: string;
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** Query heads on this device (after TP sharding). Count. */
  heads: number;
  /** K/V heads on this device (after TP sharding or replication; 1 for absorbed MLA). Count. */
  kvHeads: number;
  /** Q/K head dimension (absorbed MLA: kvLoraRank + qkRopeDim, e.g. 576). Unit: elements. */
  qkHeadDim: number;
  /** V/output head dimension (absorbed MLA: kvLoraRank, e.g. 512). Unit: elements. */
  vHeadDim: number;
  /** Query tokens per sequence in this step (prefill: prompt chunk; decode: 1). Unit: tokens. */
  qLen: number;
  /** Keys visible to the LAST query of the step (= past tokens + qLen). Unit: tokens. */
  kvLen: number;
  /** True if query i only sees keys at positions ≤ its own (decoder self-attention). */
  causal: boolean;
  /** Which of the visible keys each query attends to (full, sliding window, or top-k). */
  kvSelection: KvSelection;
  /**
   * K/V read per visible-and-selected token: the KV cache, or decompressed K/V produced by a preceding GEMM
   * (MLA prefill, MHA form).
   */
  kvRead: KvStream;
  /** Appended to the KV cache per NEW token (`kvWriteTokens ?? qLen` tokens per sequence); `elemsPerToken` 0 = no cache write. */
  kvWrite: KvStream;
  /**
   * Tokens per sequence whose K/V are appended to the cache this step; absent = `qLen` (every query token). A block
   * drafter attends over `m_ctx + k + 1` new rows but keeps only the `m_ctx` context rows (design doc §3.3).
   * Applies to the primary `kvWrite` and to `secondaryKv.kvWrite`. Unit: tokens.
   */
  kvWriteTokens?: number;
  /** Matmul input class of Q·Kᵀ and P·V (selects the peak). */
  math: MathFormat;
  /** Accumulator type of Q·Kᵀ and P·V (selects the peak). */
  accumulate: AccumulateType;
  /**
   * A second key/value set attended in the SAME softmax (one Q read, one O write): DeepSeek-V4.1 CSA's compressed
   * global entries next to its raw window. Absent = none.
   */
  secondaryKv?: AttentionKvSource;
  /**
   * Decode context parallelism (DCP) degree d: the keys of every query are spread evenly over d ranks along the
   * token axis, and this op is ONE rank's share of them: attended pairs `/ d`, distinct KV tokens read
   * `ceil(kvTokensRead / d)` and tokens written `ceil((kvWriteTokens ?? qLen) / d)` (the busiest rank). Q, O and
   * FLOP per (query, key, head) are unchanged. Absent = 1 = unsharded (never written as 1). Only set with
   * `kvSelection.kind === 'full'` and no `secondaryKv` (lower/dcp.ts). Unit: ranks (count).
   * See docs/simulation-model.md §5 "Decode context parallelism (DCP)".
   */
  kvTokenShards?: number;
  /** Storage dtype of Q (read) and the output O (written). */
  activationDtype: Dtype;
}

/**
 * A second key/value source of an {@link AttentionOp}, whose entries each stand for `tokensPerEntry` tokens
 * (compressed KV). A query at absolute position p sees about p / tokensPerEntry entries.
 */
export interface AttentionKvSource {
  /** Tokens pooled into one entry r (≥ 1). Unit: tokens/entry. */
  tokensPerEntry: number;
  /** Which visible entries each query attends (full, or top-k over entries). Caps are in entries. */
  kvSelection: KvSelection;
  /** K/V read per touched ENTRY (`elemsPerToken` here means per entry). */
  kvRead: KvStream;
  /** Appended per NEW token (the owner of a compressed cache: entryElems / r; a consumer: 0). */
  kvWrite: KvStream;
}

/**
 * Top-k key scorer for sparse attention (DeepSeek-V3.2 DSA "lightning indexer"):
 * `score(q, s) = Σ_h w_h · ReLU(q_h · k_s)`; keeps the `topK` best keys per query. It reads its own key cache
 * (one key vector per indexer key, shared by all indexer heads; a key may pool `keyPoolTokens` tokens), has no V,
 * and writes int32 indices. It scores every visible key (or at most `maxScoredKeys`), with the causal triangle.
 * See docs §3 "Indexer".
 */
export interface IndexerOp {
  kind: 'indexer';
  /** Stable dotted name, e.g. `'attn.indexer'`. */
  name: string;
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** Indexer heads; replicated on every TP rank (not sharded). Count. */
  heads: number;
  /** Indexer head dimension. Unit: elements. */
  headDim: number;
  /** Query tokens per sequence in this step. Unit: tokens. */
  qLen: number;
  /** Keys visible to the last query (= past + qLen). Unit: tokens. */
  kvLen: number;
  /** True if query i only scores keys at positions ≤ its own. */
  causal: boolean;
  /** Indexer keys kept per query (and int32 indices written per query). Unit: indexer keys. */
  topK: number;
  /** Storage format of the indexer key cache. */
  keyCache: TensorFormat;
  /** Indexer key-cache elements per indexer KEY per device per layer (usually `headDim`). Unit: elements/key. */
  keyElemsPerKey: number;
  /** Tokens pooled into one indexer key p (1 = one key per token). Unit: tokens/key. */
  keyPoolTokens: number;
  /** Most keys scored per query (a hierarchical candidate pool); absent = every visible key. Unit: indexer keys. */
  maxScoredKeys?: number;
  /** Key-cache elements appended per NEW token (owner: keyElemsPerKey / p; reads another layer's keys: 0). Unit: elements/token. */
  keyWriteElemsPerToken: number;
  /** True if per-head score weights (`weights_proj` output, one per head and query) are read. */
  headWeights: boolean;
  /** Matmul input class of the q·k scoring (selects the peak). */
  math: MathFormat;
  /** Accumulator type of the scoring (selects the peak). */
  accumulate: AccumulateType;
  /** Storage dtype of the indexer queries and head weights (read). */
  activationDtype: Dtype;
}

/**
 * Linear-attention / SSM token mixer with a FIXED-size recurrent state per sequence (GatedDeltaNet, KDA, Mamba2 SSD,
 * plain linear attention). State per sequence per device = `heads · keyDim · valueDim` elements, independent
 * of context length. See docs §3 "Recurrent".
 */
export interface RecurrentOp {
  kind: 'recurrent';
  /** Stable dotted name, e.g. `'lin.recurrent'`. */
  name: string;
  /** Which recurrence (sets the step-mode FLOP constant, see `stateFlopsPerElemPerToken`). */
  variant: 'gated_delta_net' | 'kda' | 'mamba2' | 'linear';
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** New tokens per sequence in this step. Unit: tokens. */
  qLen: number;
  /** State (value) heads on this device. Count. */
  heads: number;
  /** q/k heads on this device (Mamba2: B/C groups, over which the chunked form computes C·Bᵀ once each). Count. */
  qkHeads: number;
  /** State key dimension dk (Mamba2: d_state). Unit: elements. */
  keyDim: number;
  /** State value dimension dv (Mamba2: head dim). Unit: elements. */
  valueDim: number;
  /** `'chunked'`: prefill, chunk-parallel form (tensor-core friendly); `'step'`: decode, one recurrent update. */
  mode: 'chunked' | 'step';
  /**
   * Tokens per chunk in chunked mode: `min(chunkTokens ?? 64, qLen)` (`LinearAttentionBlock.chunkTokens`); 1 in step
   * mode. Unit: tokens.
   */
  chunkLen: number;
  /**
   * FLOP per state element per token for the state terms (c_s in docs §3: chunked 4; step: gated_delta_net 7,
   * mamba2 5, linear 4). A documented approximation set by lowering. Unit: FLOP/(element·token).
   */
  stateFlopsPerElemPerToken: number;
  /**
   * Gate elements read per token per state head: 2 for GatedDeltaNet / Mamba2 (decay and β, or dt and A); KDA dk + 1
   * (per-channel decay plus β). Unit: elements/(token·head).
   */
  gateElemsPerTokenPerHead: number;
  /** False on the first prefill step of a sequence (no prior state to read); true otherwise. */
  hasPriorState: boolean;
  /**
   * State snapshots written this step; absent = 1. A `k + 1`-token speculative verify keeps one snapshot per
   * position so the state can be rolled back to the accepted prefix (design doc §3.3): `k + 1`. Multiplies the state
   * WRITE bytes only (the prior state is still read once). Count.
   */
  stateSnapshots?: number;
  /** Storage dtype of the recurrent state (read and written once per step). */
  stateDtype: Dtype;
  /** Storage dtype of q, k, v, gates (read) and the output (written). */
  activationDtype: Dtype;
  /** Matmul input class of the chunked form (selects the peak). */
  math: MathFormat;
  /** Accumulator type of the chunked form (selects the peak). */
  accumulate: AccumulateType;
}

/**
 * Memory-bound elementwise or small-reduction op (norms, activations, RoPE, residual adds, conv1d, routing).
 * Runs on the vector units: FLOP = elems · flopsPerElem; bytes are given explicitly by the byte fields
 * (lowering computes them, since they differ per function). See docs §3 "Elementwise".
 */
export interface ElementwiseOp {
  kind: 'elementwise';
  /** Stable dotted name, e.g. `'attn.qk_norm'`. */
  name: string;
  /**
   * Function computed; its FLOP-per-element approximation is in the docs §3 table (`ELEMENTWISE_FLOPS_PER_ELEM` in
   * lower/common.ts). `silu`, `relu` and `relu_mul` cover the ungated-SiLU, ungated-ReLU and gated-ReLU MLPs; `relu2`
   * is the ungated squared ReLU `relu(x)²` (Nemotron-H experts); `situ` / `situ_mul` Kimi K3's tanh-bounded SiLU.
   * Hyper-connections: `hc_collapse` (N streams → 1), `hc_expand` (1 → N, with or without stream mixing), `sinkhorn`
   * (per N×N matrix element); `ngram_gate` (n-gram value gated against the stream); `kv_pool` (softmax-weighted
   * pooling of tokens); `residual_mix` (Attention Residuals: a softmax-weighted mix of `candidates` residual vectors
   * per output element, docs §5 "Attention Residuals").
   * Speculative decoding (design doc §3.3–3.4): `kv_cache_write` (a write-only cache op: a block drafter's context
   * K/V cast into the KV format and stored, its bytes in `kvWriteBytes`); `candidate_score` (the DFlash 2 selector's
   * rank-r dot products, one multiply-add per element).
   */
  fn:
    | 'rmsnorm'
    | 'layernorm'
    | 'residual_add'
    | 'rope'
    | 'qk_norm'
    | 'silu'
    | 'silu_mul'
    | 'gelu_mul'
    | 'gelu'
    | 'relu'
    | 'relu_mul'
    | 'relu2'
    | 'situ'
    | 'situ_mul'
    | 'sigmoid_mul'
    | 'gated_rmsnorm'
    | 'causal_conv1d'
    | 'router_topk'
    | 'moe_combine'
    | 'hc_collapse'
    | 'hc_expand'
    | 'sinkhorn'
    | 'ngram_gate'
    | 'kv_pool'
    | 'residual_mix'
    | 'kv_cache_write'
    | 'candidate_score'
    | 'copy';
  /** Output elements computed on this device. Unit: elements. */
  elems: number;
  /** Vector FLOP per output element (e.g. rmsnorm 4; causal_conv1d 2·kernelWidth). Unit: FLOP/element. */
  flopsPerElem: number;
  /** Storage dtype of the activations (selects the vector peak). */
  dtype: Dtype;
  /** Activation bytes read from memory on this device. Unit: bytes. */
  activationReadBytes: number;
  /** Activation bytes written to memory on this device. Unit: bytes. */
  activationWriteBytes: number;
  /** Weight bytes read on this device (norm gamma, conv taps); 0 if none. Unit: bytes. */
  weightBytes: number;
  /** Per-sequence state bytes read (causal_conv1d rolling state); 0 otherwise. Unit: bytes. */
  stateReadBytes: number;
  /** Per-sequence state bytes written (causal_conv1d rolling state); 0 otherwise. Unit: bytes. */
  stateWriteBytes: number;
  /**
   * KV-cache bytes written by a write-only cache op (a block drafter's context K/V in prefill, design doc §3.4);
   * absent = 0. Counted as `ByteBreakdown.kvWriteBytes`. Unit: bytes.
   */
  kvWriteBytes?: number;
}

/**
 * Token-embedding gather: one row of the embedding table per token. The table is vocab-parallel over TP
 * (each rank holds `vocabShard` rows; a TP all-reduce combines the partial results). FLOP = 0.
 */
export interface EmbeddingOp {
  kind: 'embedding';
  /** Stable dotted name, e.g. `'embed'`. */
  name: string;
  /** Tokens embedded on this device in this step. Unit: tokens. */
  tokens: number;
  /** Actual table rows read by this device in this step; absent = tokens. Masked outputs may contain more rows. Unit: rows. */
  rowsRead?: number;
  /**
   * Useful peer-HBM payload received by this device in this step, including row scales. Balanced ownership means
   * the same payload is served outbound; each direction is costed against the per-direction link bandwidth once.
   * Absent = local only. Unit: bytes. See docs/simulation-model.md §5 "Lookup tables across DP".
   */
  peerReadBytes?: number;
  /** Hidden size (row length). Unit: elements. */
  hidden: number;
  /**
   * Table rows resident on this device: normally ceil(vocab/tp); shared n-gram tables use
   * `lower/ngram.ts:tableShardRows` (row or head ownership over TP×DP). Unit: rows.
   */
  vocabShard: number;
  /** Storage format of the embedding table. */
  weight: TensorFormat;
  /** Storage dtype of the embedded activations written. */
  outputDtype: Dtype;
}

/**
 * A collective among the devices of one parallel group. Its time comes from the alpha-beta model
 * (engine/collectives.ts, docs §6); it moves no HBM bytes in this model.
 *
 * `messageBytes` per device, by collective:
 * - `all_reduce`: the full buffer S (input size = output size);
 * - `all_gather`: the full OUTPUT (after gathering);
 * - `reduce_scatter`: the full INPUT (before scattering);
 * - `all_to_all`: the total this device sends, including the share it keeps for itself;
 * - `send_recv`: the bytes sent to the peer.
 */
export interface CollectiveOp {
  kind: 'collective';
  /** Stable dotted name, e.g. `'attn.tp_all_reduce'`. */
  name: string;
  /** Collective pattern. */
  collective: 'all_reduce' | 'all_gather' | 'reduce_scatter' | 'all_to_all' | 'send_recv';
  /**
   * Parallel group it runs over: tensor (`tp`), expert (`ep`), pipeline (`pp`, send/recv between stages),
   * expert-TP (`etp`: the `tp/ep` ranks of a TP group that hold the same MoE experts, strided by ep; mode A only;
   * its links come from `engine/placement.ts: expertTpGroupSpan`), or decode context-parallel (`dcp`: the d
   * consecutive ranks of a TP group that share one token-sharded KV cache; a contiguous group placed like TP,
   * `engine/placement.ts: groupSpan`; docs/simulation-model.md §5 "Decode context parallelism (DCP)").
   * Shared lookup exchanges use `dp` (same TP index, stride tp) and `tp_dp` (the contiguous TP×DP ranks of one
   * stage); paths from `engine/placement.ts:lookupTableCollectivePaths`, docs §5 "Lookup tables across DP".
   */
  group: 'tp' | 'ep' | 'pp' | 'etp' | 'dcp' | 'dp' | 'tp_dp';
  /** Devices in the group (send_recv: 2). Count. */
  groupSize: number;
  /** Per-device message size, by the convention above. Unit: bytes. */
  messageBytes: number;
  /** Wire dtype (informational); int64 is for lookup IDs only, not a weight or compute format. */
  dtype: Dtype | 'int64';
  /** Live lookup exchange buffers at this op; capacity only, not additional HBM traffic. Absent = 0. Unit: bytes. */
  workspaceBytes?: number;
}

/**
 * Where a {@link TransferOp} reads its bytes from: the node's host RAM, its NVMe SSD, or a remote KV pool reached
 * over the inter-node network (`'remote_pool'`: Mooncake store, LMCache remote backends; KV cache only,
 * docs/simulation-model.md §12 "KV cache off-device" / "Remote KV pool"). `'host_ram'` on a unified-memory device
 * is never emitted: it is device memory.
 */
export type TransferSource = 'host_ram' | 'nvme' | 'remote_pool';

/**
 * A read of off-device bytes (host RAM, NVMe, or the remote KV pool) into this device's memory, issued in front of the op that
 * needs them (deployment knob `offload`, design doc §2). Its time comes from the tier and host-link model
 * (engine/transfer.ts, design doc §2.3), not from the roofline: it has no FLOP, and moves no HBM bytes in this
 * model (the DMA write into device memory, at most `B_link / memBW` of the step, is a documented approximation).
 *
 * Per device, per microbatch step. Every quantity is an EXPECTED value (a real number: no rounding of expected
 * misses to whole requests). Lowering emits one only when it has bytes to move.
 */
export interface TransferOp {
  kind: 'transfer';
  /**
   * Stable dotted name, e.g. `'moe.expert_fetch'`, `'ngram.row_fetch'`, `'embed.row_fetch'`, `'attn.kv_fetch'`
   * (KV from host RAM), `'attn.kv_fetch_ssd'` (KV from the SSD), `'attn.kv_fetch_pool'` (KV from the remote pool).
   */
  name: string;
  /**
   * What is streamed: missed routed experts, gathered n-gram table rows, gathered token-embedding rows (the 1-gram
   * table of `OffloadConfig.embeddings`; its own group because its prefetch share and staging differ,
   * docs/simulation-model.md §12 "Token embedding off-device"), or the off-device share of an attention layer's KV
   * cache (`kv_cache`, `OffloadConfig.kvCache`: the past tokens' K/V or MLA latents the attention reads this step and
   * that are not in device memory; docs §12 "KV cache off-device").
   */
  group: 'routed_experts' | 'lookup_table' | 'token_embedding' | 'kv_cache';
  /**
   * Storage tier the bytes come from ({@link TransferSource}). Weight groups use host RAM or the SSD; only the KV
   * cache may come from the remote pool.
   */
  source: TransferSource;
  /**
   * Physical read classes issued this step: `count` reads of `bytesEach` bytes each (before rounding to the tier's
   * read granularity). Experts: `[{ count: M, bytesEach: W_expert }]`; table rows per the scale layout of design
   * doc §2.10, e.g. separate value and scale reads
   * `[{ count: T·g, bytesEach: 256 }, { count: T·g, bytesEach: 8 }]`; token-embedding rows always as the separate
   * layout (`[{ R, rowBytes }]`, or value and scale reads for a block-scaled format, R = T/tp); KV cache: one
   * contiguous span per sequence for full attention and prefill (`[{ count: b, bytesEach: M_t·kvB }]`), one row per
   * selected token for top-k decode (`[{ count: b·M_t, bytesEach: kvB }]`). `Σ count` pays the
   * per-request (IOPS, Little's-law) terms; each class is rounded up to the tier granularity for the tier-bandwidth
   * term.
   * Unit: `count` requests (expected, real), `bytesEach` bytes.
   */
  reads: { count: number; bytesEach: number }[];
  /**
   * Useful payload that crosses the host link, independent of the read layout (e.g. 264 B per Engram row, whatever
   * the granularity rounding). Unit: bytes.
   */
  usefulBytes: number;
  /**
   * q: the share of this op's transfer time that is prefetched, i.e. issued before the consuming block's router (or
   * before the step, for table rows) and so eligible to hide behind the stage's compute
   * (docs/simulation-model.md §12 "Expert prefetch"). Routed experts: `q = max(p, M/(E_loc − C))`; n-gram table rows:
   * `q = 1` when prefetching is on (`p > 0`), `0` at `p = 0`; token-embedding rows: `q = 1` in prefill (`p > 0`),
   * `q = 0` in decode (the row is the token the previous step just sampled, and its gather is the step's first op);
   * KV cache (`p > 0`): `q = 1` for full attention and prefill (every past token is needed, known before the step),
   * `q = max(p, M/(N_past − N_res))` for top-k decode (the expert rule, with past tokens for experts).
   * Set by `lower/offload.ts: prefetchedFraction`.
   * Dimensionless, in [0, 1].
   */
  prefetchedFraction: number;
}
