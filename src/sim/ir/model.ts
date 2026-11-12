/**
 * Model architecture spec: WHAT the model computes, independent of tokens, batch and parallelism.
 *
 * A model is an ordered list of heterogeneous layers, each an ordered list of blocks. That covers hybrid stacks
 * (e.g. 3 GatedDeltaNet layers per full-attention layer), dense first layers before MoE layers, and alternating
 * sliding-window layers. Lowering (`src/sim/lower/`) turns blocks into per-device ops for a given workload and
 * parallelism. All dimensions are LOGICAL (unsharded) sizes. See docs/simulation-model.md §2.
 *
 * Cross-layer references (`kvCacheFromLayer`, a `'reuse'` sparse selection, `IndexerSpec.ownsKeys: false`) name
 * an EARLIER layer by its index in `ModelSpec.layers`; `modelSpecErrors` (ir/validateModel.ts) checks them.
 */

import type { A2aFormat } from './a2aFormat';
import type { Dtype, TensorFormat } from './dtype';

/** A decoder-only language model. */
export interface ModelSpec {
  /** Display name, e.g. `'Qwen3-8B'`. */
  name: string;
  /** Width of ONE residual stream d_model (the embedding width). Unit: elements. */
  hidden: number;
  /** Vocabulary size (embedding and LM-head rows). Count. */
  vocab: number;
  /** True if the LM head reuses the embedding matrix (stored once). */
  tieEmbeddings: boolean;
  /** Decoder layers in execution order (heterogeneous). */
  layers: LayerSpec[];
  /**
   * Norm applied before the LM head. `'none'`: no final norm (Qwen4Exp, whose final hyper-connection collapse
   * normalizes instead).
   */
  finalNorm: 'rmsnorm' | 'layernorm' | 'none';
  /**
   * Parallel residual streams N (hyper-connections); default 1. Each token carries `residualStreams · hidden`
   * residual elements between blocks and layers (the PP activation send uses that width). Count.
   */
  residualStreams?: number;
  /** Where the dims came from (e.g. a Hugging Face model ID or paper URL). */
  source?: string;
  /**
   * The checkpoint's own speculative-decoding draft head(s) (MTP modules, or a DSpark drafter shipped inside the
   * checkpoint). Costed only when the deployment turns speculative decoding on; it never enters `countParams`, the
   * memory fit or the op lists otherwise. Absent = the checkpoint has none. Design doc §3.2.
   */
  draft?: DraftSpec;
}

/**
 * A speculative-decoding drafter: the module that proposes `k` draft tokens per decode step, which the target then
 * verifies in one `k + 1`-token pass (design doc §3). Dimensions are LOGICAL (unsharded) sizes.
 */
export type DraftSpec = MtpDraftSpec | BlockDraftSpec;

/**
 * Multi-token-prediction module(s) of the checkpoint (DeepSeek-V3 `num_nextn_predict_layers`, GLM, Qwen3.5+): each
 * module is one target-shaped decoder layer fed `[norm(h_prev); norm(emb(next token))]`. Draft tokens come from
 * running the module once over the newly committed tokens, then `k − 1` recursive one-token passes (design doc §3.3).
 */
export interface MtpDraftSpec {
  kind: 'mtp';
  /** The MTP module's decoder layer(s), one target-shaped layer per module, in execution order. */
  layers: LayerSpec[];
  /** True if the module projects `[norm(h_prev); norm(emb)]` (2h → h) before its layer (`eh_proj`). */
  hiddenProj: boolean;
  /**
   * Draft positions the head was trained for; `k` beyond it collapses (warning, and a low tail acceptance when the
   * preset acceptance is kept). Absent = any depth (Qwen3.5+/3.8 multi-step heads). Count.
   */
  maxReliableDepth?: number;
}

/**
 * A block drafter (DFlash 2, DSpark): a small attention stack that drafts a whole block of `k` tokens in ONE pass,
 * conditioned on target hidden states of `targetLayerIds`, sharing the target's embedding and LM head.
 */
export interface BlockDraftSpec {
  kind: 'block_drafter';
  /** Extras to cost: DFlash 2 (dynamic convs + candidate selector) or DSpark (Markov head). */
  method: 'dflash2' | 'dspark';
  /** Drafter width h (= the target's, since embedding and LM head are shared). Unit: elements. */
  hidden: number;
  /** Shared vocabulary V (selector codebooks and Markov matrices are V × rank). Count. */
  vocab: number;
  /** Softmax-attention layers (`num_hidden_layers`; released drafters: 5). Count. */
  layers: number;
  /** Query heads of the DRAFTER's attention (not the target's). Count. */
  heads: number;
  /** KV heads of the drafter's attention. Count. */
  kvHeads: number;
  /** Head dimension of the drafter's attention. Unit: elements. */
  headDim: number;
  /** MLP intermediate width of the drafter. Unit: elements. */
  intermediate: number;
  /** True for a GLU-style (gate and up) drafter MLP. */
  gated: boolean;
  /** Drafter MLP activation function. */
  activation: FfnActivation;
  /**
   * Sliding window of the drafter's attention over its context K/V (`sliding_window` when `layer_types` are
   * sliding); absent = full attention. Unit: tokens.
   */
  windowTokens?: number;
  /** Target layers whose hidden states are concatenated into `fc`'s input (`len · h → h`). Zero-based indices. */
  targetLayerIds: number[];
  /** Block positions per draft pass = k + 1 (anchor + k masks); the deployment's k must satisfy k + 1 ≤ blockSize. Count. */
  blockSize: number;
  /** DFlash 2 `GroupedDynamicCausalConv` (two per layer): taps K (tokens) and channels per group g (elements). */
  conv?: { kernel: number; groupSize: number };
  /** DFlash 2 `CandidateSelector`: candidates per position (count, 16) and codebook rank r (elements, 256). */
  selector?: { topK: number; rank: number };
  /** DSpark Markov head rank r (`W1: V×r`, `W2: r×V`). Unit: elements. */
  markovRank?: number;
  /** DeepSeek-V4-style MoE drafter layers (used instead of the dense fields above when present). */
  moeLayers?: LayerSpec[];
  /** Where the shape came from: a Hugging Face repo id (+ revision), or `'hypothetical'` for the labelled stand-in. */
  source: string;
}

/** One decoder layer: blocks in execution order (norms and residual adds are explicit blocks). */
export interface LayerSpec {
  /** Display name, e.g. `'layer.0'`. */
  name: string;
  /** Blocks in execution order. */
  blocks: BlockSpec[];
}

/** Any block of a layer. Dispatch with `switch (block.kind)`. */
export type BlockSpec =
  | NormBlock
  | ResidualBlock
  | ResidualMixBlock
  | GqaAttentionBlock
  | MlaAttentionBlock
  | CsaAttentionBlock
  | LinearAttentionBlock
  | MlpBlock
  | MoeBlock
  | HcPreBlock
  | HcPostBlock
  | NgramEmbeddingBlock
  | ShortConvBlock;

/**
 * Activation function of a feed-forward block (`relu2`: squared ReLU, ungated only; `situ`: Kimi K3's tanh-bounded
 * SiLU; see {@link MlpBlock.activation}).
 */
export type FfnActivation = 'silu' | 'gelu' | 'relu' | 'relu2' | 'situ';

/** A normalization over the last dimension (per token). */
export interface NormBlock {
  kind: 'norm';
  /** Display name, e.g. `'input_norm'`. */
  name: string;
  /** Norm function (rmsnorm: one gamma vector; layernorm: gamma and beta). */
  norm: 'rmsnorm' | 'layernorm';
  /** Normalized width. Unit: elements. */
  dim: number;
}

/** Residual add: `x = x + f(x)` over the residual stream. */
export interface ResidualBlock {
  kind: 'residual';
  /** Display name, e.g. `'attn_residual'`. */
  name: string;
  /** Residual width. Unit: elements. */
  dim: number;
}

/**
 * Attention Residuals (Kimi K3 `attn_res_block_size`, `modeling_kimi_linear.py: _apply_attn_res`): the input of a
 * sublayer is a softmax-weighted mix of `candidates` vectors per token (the running residual sum and the summaries
 * of the earlier blocks of `attn_res_block_size` layers) instead of the running sum itself. Per token: each
 * candidate is RMS-normalized and scored against one shared `dim`-wide weight (the RMSNorm γ times the
 * `dim → 1` score projection), softmax over the scores, weighted sum. The residual add after the sublayer stays a
 * plain {@link ResidualBlock}. No cross-token state; the candidates are activations of the same forward pass.
 */
export interface ResidualMixBlock {
  kind: 'residual_mix';
  /** Display name, e.g. `'attn_residual_mix'`. */
  name: string;
  /** Residual width h. Unit: elements. */
  dim: number;
  /** Vectors mixed per token: the running sum plus the earlier block summaries (Kimi K3: 2 … 9). Count. */
  candidates: number;
}

/** A dense projection with its own weight matrix `inputDim → outputDim` (k = inputDim, n = outputDim), no bias. */
export interface ProjectionSpec {
  /** Display name, e.g. `'hc.down'`. */
  name: string;
  /** Input features k. Unit: elements. */
  inputDim: number;
  /** Output features n. Unit: elements. */
  outputDim: number;
}

/**
 * Top-k sparse-attention indexer (DeepSeek-V3.2 DSA "lightning indexer" and its relatives: GLM k-pool, Qwen QSA,
 * MiniMax MSA, DeepSeek-V4.1 CSA). Per query it scores the visible indexer keys with `heads` small heads and keeps
 * the top `topK` for the main attention. Defaults reproduce DSA: one key per token, from the hidden state, with
 * head weights, owned by this layer.
 */
export interface IndexerSpec {
  /** Indexer heads H_I. Count. */
  heads: number;
  /** Indexer head dimension hd_I (also the indexer key-cache elements per indexer key). Unit: elements. */
  headDim: number;
  /**
   * Keys the main attention reads per query, in the attention's key units: tokens for gqa/mla (a pooled indexer
   * keeps `topK / keyPoolTokens` pools and expands them to tokens), compressed entries for csa (one indexer key
   * per entry). Unit: tokens (gqa/mla) or entries (csa).
   */
  topK: number;
  /**
   * Tokens pooled into one indexer key p; default 1. Scored pairs and the key cache are divided by p.
   * Qwen QSA `indexer_compress_ratio` 4, GLM `index_kpool` 4, DeepSeek-V4.1: the layer's compress ratio r.
   * Unit: tokens per key.
   */
  keyPoolTokens?: number;
  /**
   * True if pooling uses a learned gate (GLM `index_kpool_compress_gate`): an extra `h → hd_I` GEMM per token and
   * `p · hd_I` positional (APE) params. Default false.
   */
  keyPoolGate?: boolean;
  /**
   * True if the pooled keys come from OVERLAPPING windows (DeepSeek-V4 indexer `Compressor.overlap`, r = 4; needs
   * `keyPoolGate`): the key and gate projections `idx.k`, `idx.pool_gate` produce coff·hd_I = 2·hd_I features per
   * token (half for the previous window, half for the current one) and the APE holds p·2·hd_I params. Default
   * false (coff = 1). Unit: flag.
   */
  keyPoolOverlap?: boolean;
  /**
   * Input width of the key projection kIn; default the block's hidden. DeepSeek-V4.1's `wk` reads the 512-wide
   * compressed KV latent. Unit: elements.
   */
  keyInputDim?: number;
  /**
   * True (default) if this layer computes and caches its own indexer keys. False: it reads the key cache of the
   * nearest earlier layer whose indexer owns keys (DeepSeek-V4.1 "Reindex" layers 24/28/32/36): no key GEMM, no
   * key norm, no key write, no key-cache memory; it still runs its own query GEMM and scoring.
   */
  ownsKeys?: boolean;
  /**
   * Upper bound on the keys scored per query (a hierarchical candidate pool); absent = every visible key.
   * DeepSeek-V4.1 indexers after `candidate_source_layer_id`: `candidate_topk_blocks · candidate_block_size`
   * = 16384. Unit: indexer keys.
   */
  maxScoredKeys?: number;
  /**
   * True (default) if a head-weight projection `h → H_I` (DSA `weights_proj`) combines the per-head scores.
   * False for Qwen QSA (fused `index_qk_proj`) and MiniMax MSA.
   */
  headWeights?: boolean;
}

/** How a sparse-attention layer gets its top-k key selection. */
export type SparseSelection =
  /** Runs its own indexer in this layer. */
  | {
      kind: 'indexer';
      /** The indexer this layer runs. */
      indexer: IndexerSpec;
    }
  /**
   * Reuses the top-k indices of the nearest earlier layer that ran an indexer (GLM-5.3 / Mistral Large 4
   * `indexer_types` 'shared', DeepSeek-V4.1 "Reuse"): no indexer ops, no indexer key cache, no indexer weights.
   */
  | {
      kind: 'reuse';
      /** Keys read per query, in the attention's key units (see {@link IndexerSpec.topK}). */
      topK: number;
    };

/**
 * Multi-head / grouped-query / multi-query attention (MHA when kvHeads = heads, MQA when kvHeads = 1), with
 * optional q/k-norm, output gate, sliding window, top-k sparse selection, a learned relative-position bias and
 * short convs on K and V.
 */
export interface GqaAttentionBlock {
  kind: 'gqa_attention';
  /** Display name, e.g. `'attn'`. */
  name: string;
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** Query heads. Count. */
  heads: number;
  /** Key/value heads. Count. */
  kvHeads: number;
  /** Head dimension (Q, K and V). Unit: elements. */
  headDim: number;
  /** True if the q/k/v projections have biases. */
  qkvBias: boolean;
  /** True if the output projection has a bias. */
  outBias: boolean;
  /** True if q and k are RMS-normalized per head before RoPE (Qwen3, Gemma 3). */
  qkNorm: boolean;
  /** True if rotary position embeddings are applied to q and k. */
  rope: boolean;
  /** True for causal (decoder) attention. */
  causal: boolean;
  /** Gated attention output (Qwen3-Next): `sigmoid(gate) ⊙ attn_out`, the gate from a wider q projection. */
  outputGate: boolean;
  /** Sliding-window size; absent = full attention. Mutually exclusive with `sparse`. Unit: tokens. */
  slidingWindow?: number;
  /** Top-k sparse attention; absent = dense. Mutually exclusive with `slidingWindow`. */
  sparse?: SparseSelection;
  /**
   * Input-dependent relative-position bias added to the attention scores (Inkling, used with `rope: false`): a
   * projection `h → heads · relDim` per token, then per (query, head) a `relDim × extentTokens` product with a
   * learned profile bank, gathered by distance. Absent = none.
   */
  relativeBias?: {
    /** Relative-state width per head d_rel (Inkling `d_rel` 16). Unit: elements. */
    relDim: number;
    /** Distances covered by the bias (Inkling `rel_extent` 1024; sliding-window layers use the window). Unit: tokens. */
    extentTokens: number;
  };
  /**
   * Width of the depthwise causal convs on K and on V after their projections (Inkling `sconv_kernel_size` 4),
   * each with its own `(K−1)`-token rolling state; absent = none. Unit: tokens (taps).
   */
  kvConvKernel?: number;
}

/**
 * Multi-head latent attention (DeepSeek-V2/V3): K/V are compressed to a low-rank latent; the KV cache holds
 * `kvLoraRank + qkRopeDim` elements per token per layer (shared by all heads). `qkRopeDim: 0` is legal (NoPE
 * layers: GLM-5.3-Flash, Mistral Large 4 every 4th layer). Kimi K3 adds a sigmoid output gate (`outputGate`).
 */
export interface MlaAttentionBlock {
  kind: 'mla_attention';
  /** Display name, e.g. `'attn'`. */
  name: string;
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** Attention heads. Count. */
  heads: number;
  /** Query compression rank rq; 0 = no q compression (q projected straight from hidden). Unit: elements. */
  qLoraRank: number;
  /** KV latent rank rkv (cached per token). Unit: elements. */
  kvLoraRank: number;
  /** Per-head q/k dimension without RoPE (dn). Unit: elements. */
  qkNopeDim: number;
  /** Per-head q/k RoPE dimension (dr); the k part is shared by all heads and cached. Unit: elements. */
  qkRopeDim: number;
  /** Per-head value dimension (dv). Unit: elements. */
  vHeadDim: number;
  /** Top-k sparse attention (DSA); absent = dense. */
  sparse?: SparseSelection;
  /**
   * Index of the earlier layer whose latent cache this layer reads (Mistral Large 4 `kv_source_layers`); absent =
   * this layer computes and caches its own latent. A consumer has no `kv_a` projection, no `kv_a_norm`, no
   * cache write and no cache memory, but keeps its own `kv_b` up-projection and its own query path.
   */
  kvCacheFromLayer?: number;
  /**
   * Gated attention output (Kimi K3 `mla_use_output_gate`): `sigmoid(g_proj(x)) ⊙ attn_out` before `o_proj`, the
   * gate from its own projection `g_proj: h → heads·vHeadDim` of the layer input (unlike GQA's gate, which is a
   * wider q projection; MLA's q is low-rank). Default false.
   */
  outputGate?: boolean;
}

/**
 * DeepSeek-V4 and V4.1 compressed sparse attention (CSA). Every token has ONE cached vector of `headDim` elements
 * used as both key and value (K = V). Each query attends the layer's own raw sliding window of `windowTokens` and,
 * when `global` is set, entries of a compressed global KV (r tokens pooled into one entry; the `topK` its selection
 * keeps, or all of them), concatenated into one softmax. q is low-rank (`q_a → q_norm → q_b`); the output projection
 * is low-rank and grouped (`wo_a` is block-diagonal over `outGroups`, then `wo_b`). Source: DeepSeek-V4 and V4.1
 * `inference/model.py` `Attention`, `Compressor`, `Indexer`.
 */
export interface CsaAttentionBlock {
  kind: 'csa_attention';
  /** Display name, e.g. `'attn'`. */
  name: string;
  /** Input/output width (one residual stream). Unit: elements. */
  hidden: number;
  /** Query heads H (64). Count. */
  heads: number;
  /** q/k dim = v dim = the one cached vector per token or entry (`head_dim` 512). Unit: elements. */
  headDim: number;
  /** RoPE part of headDim (`qk_rope_head_dim` 64). Unit: elements. */
  ropeDim: number;
  /** Query compression rank rq (`q_lora_rank` 1280). Unit: elements. */
  qLoraRank: number;
  /** Output groups G (`o_groups` 8): `wo_a` maps each group's `H·hd/G` features to `outLoraRank`. Count. */
  outGroups: number;
  /** Output rank per group R_o (`o_lora_rank` 1024); `wo_b` maps `G·R_o → h`. Unit: elements. */
  outLoraRank: number;
  /** Raw sliding window every layer attends and caches (`sliding_window` 128). Unit: tokens. */
  windowTokens: number;
  /** The compressed global branch; absent when `compress_ratios[i] == 0` (SWA-only layers 0 and 1). */
  global?: CsaGlobalSpec;
}

/** The compressed global branch of a {@link CsaAttentionBlock}. */
export interface CsaGlobalSpec {
  /** Tokens pooled into one global KV entry r ≥ 1 (`compress_ratios[i]`; r = 1 is a plain projection). Count. */
  compressRatio: number;
  /**
   * Index of the earlier layer whose global (compressed) KV cache this layer reads; absent = this layer runs the
   * compressor and owns the cache (a `kv_source_layer_ids` layer). A consumer has no compressor and no global cache.
   */
  kvCacheFromLayer?: number;
  /**
   * True when the compressor pools OVERLAPPING windows (DeepSeek-V4 `Compressor.overlap`, set iff r = 4): `wkv_c`
   * and `wgate_c` project to coff·hd with coff = 2 (one half for the previous window, one for the current one), so
   * each entry pools 2r tokens, and the fp32 pooling state doubles in both dimensions (coff·r rows of coff·hd).
   * Default false (coff = 1). Ignored with `kvCacheFromLayer` (no compressor). Unit: flag.
   */
  compressorOverlap?: boolean;
  /**
   * Which entries each query reads: its own indexer, the top-k of an earlier one (topK in entries, `index_topk`
   * 512), or every visible entry (`{ kind: 'all' }`, see {@link CsaSelection}).
   */
  selection: CsaSelection;
}

/**
 * Entry selection of a CSA global branch: a {@link SparseSelection} (top-k by an indexer, own or reused), or
 * `{ kind: 'all' }`: each query attends EVERY visible compressed entry (about p/r of them at position p, causal), with
 * no indexer ops, no indexer weights and no indexer keys. DeepSeek-V4 r = 128 layers (`inference/model.py`
 * `get_compress_topk_idxs`: the indices are simply 0 … ⌊(p+1)/r⌋ − 1).
 */
export type CsaSelection =
  | SparseSelection
  /** Attend every visible compressed entry (dense over entries); no indexer. */
  | { kind: 'all' };

/**
 * Linear-attention / SSM token mixer (GatedDeltaNet, Kimi Delta Attention, Mamba2, plain linear attention) with a
 * fixed-size state. State per sequence = `vHeads · keyDim · valueDim` elements, plus the conv1d rolling state.
 */
export interface LinearAttentionBlock {
  kind: 'linear_attention';
  /** Display name, e.g. `'linear_attn'`. */
  name: string;
  /**
   * Recurrence variant. `'kda'` (Kimi Delta Attention, GLM-5.3-Flash, Kimi K3): separate full q/k/v projections
   * with their own convs, a per-channel forget gate and an output gate, both low-rank (`gateRank`) unless
   * `fullRankOutputGate`, and a β projection.
   * `'mamba2'` (Mamba-2 SSD, Nemotron-H): one fused in_proj `h → (z, x, B, C, dt)` with ONE dt per head (not two
   * gates); per-head vectors A_log, D and dt_bias (3 per head); the conv runs over x, B and C; the output norm is a
   * grouped RMSNorm after `y·silu(z)` with γ over all `vHeads·valueDim` channels; in the chunked form C·Bᵀ is
   * computed once per B/C group, not per head (docs §5 "Mamba-2").
   */
  variant: 'gated_delta_net' | 'kda' | 'mamba2' | 'linear';
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** q/k heads (Mamba2: B/C groups). Count. */
  qkHeads: number;
  /** Value (state) heads. Count. */
  vHeads: number;
  /** State key dimension dk (Mamba2: d_state). Unit: elements. */
  keyDim: number;
  /** State value dimension dv (Mamba2: head dim). Unit: elements. */
  valueDim: number;
  /** Causal depthwise conv width over the q, k, v channels (Mamba2: x, B, C); 0 = no conv. Unit: tokens (taps). */
  convKernel: number;
  /**
   * True if the conv1d has a bias of one element per conv channel `C = 2·qkHeads·keyDim + vHeads·valueDim`
   * (Mamba2 `use_conv_bias`). Default false. Unit: none (flag).
   */
  convBias?: boolean;
  /**
   * Chunk length Q of the chunked (prefill) form (Mamba2 `chunk_size`, 128 for Nemotron-H); default
   * `RECURRENT_CHUNK_TOKENS` (64, lower/linearAttention.ts). A step uses `min(chunkTokens, qLen)`. Must be a positive
   * integer. Unit: tokens.
   */
  chunkTokens?: number;
  /** True if the output passes a z-gate (gated rmsnorm) before the output projection. Must be true for `'kda'`. */
  outputGate: boolean;
  /**
   * Rank r of the low-rank forget gate (`f_a: h → r`, `f_b: r → vH·dk`) and output gate (`g_a: h → r`,
   * `g_b: r → vH·dv`); set iff `variant` is `'kda'` (GLM-5.3-Flash and Kimi K3: r = head_dim 128). Unit: elements.
   */
  gateRank?: number;
  /**
   * True if the KDA output gate is ONE full-rank projection `g: h → vH·dv` (Kimi K3 `use_full_rank_gate`) instead
   * of the rank-`gateRank` pair `g_a → g_b`; the forget gate stays low-rank. Only with `variant 'kda'`. Default false.
   */
  fullRankOutputGate?: boolean;
}

/** Dense feed-forward block: `down(act(gate(x)) ⊙ up(x))` if gated, else `down(act(up(x)))`. */
export interface MlpBlock {
  kind: 'mlp';
  /** Display name, e.g. `'mlp'`. */
  name: string;
  /** Input/output width. Unit: elements. */
  hidden: number;
  /** Intermediate width f. Unit: elements. */
  intermediate: number;
  /** True for GLU-style (gate and up projections, e.g. SwiGLU). */
  gated: boolean;
  /**
   * Activation function. `relu2`: `relu(x)²`, ungated only (Nemotron-H). `situ` (Kimi K3 `hidden_act`):
   * `β·tanh(gate/β)·sigmoid(gate) ⊙ β'·tanh(up/β')`, a tanh-bounded SiLU on the gate and a tanh bound on the up
   * branch (elementwise; the bounds β, β' are constants).
   */
  activation: FfnActivation;
  /** True if the projections have biases. */
  bias: boolean;
}

/**
 * Mixture-of-experts feed-forward block with a top-k router and optional always-on shared experts. With
 * `latentDim` set it is a latent MoE: the routed experts run at a narrower width l than the residual h.
 */
export interface MoeBlock {
  kind: 'moe';
  /** Display name, e.g. `'moe'`. */
  name: string;
  /** Input/output width. Unit: elements. */
  hidden: number;
  /** Routed experts E. Count. */
  experts: number;
  /** Routed experts per token k. Count. */
  topK: number;
  /** Intermediate width of one routed expert. Unit: elements. */
  expertIntermediate: number;
  /** True for GLU-style experts. */
  gated: boolean;
  /** Expert activation function (see {@link MlpBlock.activation}; Nemotron-H experts use `relu2`). */
  activation: FfnActivation;
  /** Shared experts applied to every token (0 = none). Count. */
  sharedExperts: number;
  /** Intermediate width of EACH shared expert. Unit: elements. */
  sharedIntermediate: number;
  /**
   * Width l the routed experts run at (latent MoE: Muse Spark `moe_hidden_size`, Nemotron `moe_latent_size`,
   * Kimi K3 `routed_expert_hidden_size`); absent = `hidden` with no projections. When set, a projection `h → l`
   * runs before the expert dispatch and `l → h` after the combine (both module `'moe_latent'`; `l → h` is row-split
   * when `expertOutputNorm` splits the expert outputs or with {@link MoeBlock.latentNorm}); the routed experts'
   * input/output width, the EP dispatch/combine messages and the combine are at l. The router reads h and the
   * shared experts stay at h. Otherwise both projections are replicated across TP. Must be a positive integer.
   * Unit: elements.
   */
  latentDim?: number;
  /**
   * True if each routed expert's output (its down projection, width l) passes an RMSNorm before the weighted
   * combine (Muse Spark `post_expert_norm`, one l-wide γ shared by all experts). The norm needs the full l-wide
   * row, so with TP-sharded experts (`etp > 1`) the down partial sums are first reduce-scattered over the
   * expert-TP group, and the per-row sum of squares (one fp32 scalar per token-expert row) is all-reduced; each
   * rank then works on an `l/etp` shard. Absent = false (no norm, no extra communication).
   */
  expertOutputNorm?: boolean;
  /**
   * True if an RMSNorm over l (γ: l params) follows the combine, before the `l → h` projection (Kimi K3
   * `latent_moe_use_norm`). Only with `latentDim`, and not with `expertOutputNorm`. The norm needs the complete
   * expert sum, so in mode A the TP ranks' partial sums are all-reduced at l first; the `l → h` projection is then
   * row-parallel over tp (its partial h sums join the block's one TP all-reduce). Default false (absent). Unit: none
   * (flag).
   */
  latentNorm?: boolean;
}

/**
 * Hyper-connection "pre" step: collapses N residual streams (N·D elements per token) into the D-wide input of one
 * sublayer, with mixing weights computed from the streams themselves. Covers Qwen4Exp "Gated Residual"
 * (normalize, low-rank down/up gate, average, plus the `inject` weights its post step uses) and DeepSeek/GLM mHC
 * (RMS statistic over N·D, one `N·D → (2+N)·N` projection for the pre/post/comb coefficients, Sinkhorn on the
 * N×N comb matrix). Also the final collapse before the LM head (Qwen `hyper_connection_mixer`, DeepSeek `hc_pre`
 * without projection, GLM `hc_head` mean). The coefficients a step computes may be used by a later step
 * (DeepSeek); each projection is counted once where it is computed.
 */
export interface HcPreBlock {
  kind: 'hc_pre';
  /** Display name, e.g. `'attn_hc_pre'`. */
  name: string;
  /** Residual streams N (`hc_count` / `hc_mult`). Count. */
  streams: number;
  /** Width of one stream D (= model hidden). Unit: elements. */
  dim: number;
  /**
   * True if the streams pass a learned RMSNorm γ over N·D (Qwen `hc_norm`, N·D params); false: RMS statistics
   * only, no params (DeepSeek/GLM).
   */
  normGamma: boolean;
  /**
   * The mixing projections, each a GEMM per token. Qwen: `[down N·D→rank, up rank→N·D, inject N·D→N]` (final
   * collapse: no inject); DeepSeek/GLM: `[fn N·D→(2+N)·N]`; final DeepSeek/GLM collapse: `[]`.
   */
  projections: ProjectionSpec[];
  /** Sinkhorn normalization iterations on the N×N comb matrix: 0 for Qwen, `hc_sinkhorn_iters` (20) for mHC. Count. */
  sinkhornIters: number;
  /** Small vectors outside the projections (mHC `base` (2+N)·N + `scale` 3). Unit: params. */
  vectorParams: number;
}

/**
 * Hyper-connection "post" step: writes a sublayer output (D wide) back into the N residual streams, replacing the
 * residual add. mHC: `y_i = post_i · out + Σ_j comb_ij · res_j`; Qwen: `y_i = res_i + w_i · out`. Its coefficients
 * come from the matching {@link HcPreBlock}; no params of its own.
 */
export interface HcPostBlock {
  kind: 'hc_post';
  /** Display name, e.g. `'attn_hc_post'`. */
  name: string;
  /** Residual streams N. Count. */
  streams: number;
  /** Width of one stream D (= model hidden). Unit: elements. */
  dim: number;
  /** True if the streams are mixed by the N×N comb matrix (mHC); false: each stream only adds `w_i · out` (Qwen). */
  mixesStreams: boolean;
}

/**
 * Hashed n-gram embedding written into the residual stream (Qwen4Exp PLE, DeepSeek-V4.1 Engram). Per token, the
 * hashes of its preceding n-grams pick `rowsPerToken` rows of a huge lookup table; the gathered rows
 * (`rowsPerToken · rowDim` elements) pass through `projections` and a gate against the residual stream, and the
 * result (`outputDim` wide) is added to the residual. The table is memory capacity: each token reads only its own
 * rows, so it counts toward total but not active params.
 */
export interface NgramEmbeddingBlock {
  kind: 'ngram_embedding';
  /** Display name, e.g. `'ple'` or `'engram'`. */
  name: string;
  /**
   * Table rows, summed over hash heads. DeepSeek: `engram_num_embeddings[j]`. Qwen: each of the
   * `(ngram_size−1)·heads_per_ngram` heads has a prime-sized slice just above `ngram_vocab_size_base`, padded to a
   * multiple of `make_ngram_vocab_size_divisible_by`; mappers use `heads · ngram_vocab_size_base` rounded up to that
   * multiple (relative error below 1e-4; exact 320,001,536 rows vs 320,000,000 for Qwen3.8-Flash-Next). Count.
   */
  tableRows: number;
  /**
   * Rows in each contiguous hash-head bucket (Engram). Optional physical layout metadata for whole-head
   * distribution; length = rowsPerToken, sum ≤ tableRows. Absent = row-parallel table (PLE/legacy fixtures).
   */
  headRows?: readonly number[];
  /** Elements per table row (Qwen `ple_embed_dim / ((ngram_size−1)·heads_per_ngram)` 160; DeepSeek `engram_head_dim` 256). Unit: elements. */
  rowDim: number;
  /** Rows gathered per token (Qwen `(ngram_size−1)·heads_per_ngram` 16; DeepSeek `(engram_max_ngram_size−1)·engram_n_heads` 24). Count. */
  rowsPerToken: number;
  /**
   * GEMMs on the gathered rows, each with `inputDim = rowsPerToken · rowDim`. Qwen: key 2560→10240, value
   * 2560→2560; DeepSeek: `wkv` 6144→(N+1)·D.
   */
  projections: ProjectionSpec[];
  /** Width written into the residual, N·D (all streams). Unit: elements. */
  outputDim: number;
  /**
   * Depthwise causal conv over `outputDim` after the gate (Qwen `ple_conv_kernel_size` 4, dilation `ngram_size` 3);
   * absent = none.
   */
  conv?: {
    /** Taps. Unit: tokens. */
    kernel: number;
    /** Spacing between taps; the rolling state holds `(kernel−1)·dilation` tokens. Unit: tokens. */
    dilation: number;
  };
  /** Small vectors outside the table and projections (Qwen 3 norm γ of N·D; DeepSeek q/k weights 2·N·D). Unit: params. */
  vectorParams: number;
  /**
   * True if the checkpoint is designed to keep this table off-device (Qwen PLE, DeepSeek Engram; per the model
   * cards). Drives the UI hint and the Host-RAM prefill of the lookup-table placement only; the simulator keeps the
   * table on the device unless `deployment.offload.lookupTables` moves it (design doc §2.10).
   */
  offDeviceByDesign?: boolean;
}

/**
 * Short depthwise causal conv plus residual on a branch output, `y = x + conv(x)` (Inkling `use_sconv` on the
 * attention and MLP outputs), with a `(kernel−1)`-token rolling state per sequence.
 */
export interface ShortConvBlock {
  kind: 'short_conv';
  /** Display name, e.g. `'attn_sconv'`. */
  name: string;
  /** Channels (the branch width). Unit: elements. */
  dim: number;
  /** Taps K (Inkling `sconv_kernel_size` 4). Unit: tokens. */
  kernel: number;
}

/**
 * Linear-layer module classes that quantized checkpoints treat differently (e.g. NVFP4 routed experts with FP8
 * attention and BF16 hyper-connections). Each linear GEMM belongs to exactly one class. Sub-classes fall back to
 * their parent when `QuantConfig.modules` does not list them ({@link WEIGHT_MODULE_FALLBACK}): `attention_kv_down`
 * and `attention_kv_up` → `attention`; `indexer_key` and `indexer_head_weights` → `indexer`; `moe_latent` →
 * `shared_experts`.
 */
export type WeightModule =
  /** GQA q/k/v/o, MLA q path and o_proj, CSA q path, window `wkv` and output projections. */
  | 'attention'
  /**
   * Projections from the hidden state into a cached KV latent: MLA `kv_a` (`kv_a_proj_with_mqa`), the CSA
   * compressor (`wkv`, `wgate`). Falls back to `attention`. (Mistral Large 4 FP8 keeps `wkv_a` BF16.)
   */
  | 'attention_kv_down'
  /**
   * Up-projection of the MLA latent, `kv_b`, and its absorbed slices (`q_absorb` = W_UK, `v_absorb` = W_UV).
   * Falls back to `attention`. (GLM-5.3-Flash FP8 keeps `kv_b_proj` BF16 while q_a/q_b/kv_a/o are FP8.)
   */
  | 'attention_kv_up'
  /** GatedDeltaNet / KDA / Mamba projections. */
  | 'linear_attention'
  /** Sparse-attention indexer query projection (`wq` / `wq_b`); parent of the other indexer classes. */
  | 'indexer'
  /** Indexer key projection `wk` and the k-pool gate. Falls back to `indexer`. */
  | 'indexer_key'
  /** Indexer head-weight projection `weights_proj`. Falls back to `indexer`. */
  | 'indexer_head_weights'
  /** Dense MLP projections. */
  | 'dense_mlp'
  /** MoE shared-expert projections. */
  | 'shared_experts'
  /** MoE routed-expert projections. */
  | 'routed_experts'
  /**
   * Latent-MoE projections around the routed experts, `h → l` and `l → h` ({@link MoeBlock.latentDim}): dense
   * per-token GEMMs like the shared experts, which they fall back to. (Nemotron NVFP4 keeps `fc1/fc2_latent_proj`
   * BF16; Kimi K3 stores `routed_expert_{down,up}_proj` BF16 next to MXFP4 experts.)
   */
  | 'moe_latent'
  /** Hyper-connection projections; default `{ dtype: highPrecision }` (not `weights`), never structured-sparse. */
  | 'residual_mixer'
  /** n-gram / Engram projections (not the lookup table: see `QuantConfig.lookupTables`). */
  | 'ngram';

/** The parent each weight-module sub-class falls back to when `QuantConfig.modules` does not list it. */
export const WEIGHT_MODULE_FALLBACK: Partial<Record<WeightModule, WeightModule>> = {
  attention_kv_down: 'attention',
  attention_kv_up: 'attention',
  indexer_key: 'indexer',
  indexer_head_weights: 'indexer',
  moe_latent: 'shared_experts',
};

/** Numeric format of one {@link WeightModule} class. */
export interface ModuleQuant {
  /** Weight storage format. */
  weights: TensorFormat;
  /** Dtype the GEMM input is quantized to (W8A8, W4A4); default `QuantConfig.activations` (weight-only quant). */
  gemmInput?: Dtype;
}

/** Numeric formats of a deployment (weights, activations, caches). */
export interface QuantConfig {
  /** Linear-layer weights (attention/mixer projections, MLP, experts) of every class not in `modules`. */
  weights: TensorFormat;
  /** GEMM inputs and intermediate activations. */
  activations: Dtype;
  /** KV, MLA latent and CSA window caches; default of `compressedKvCache` and `indexerKeyCache`. */
  kvCache: TensorFormat;
  /**
   * Per-module overrides. An absent class uses its {@link WEIGHT_MODULE_FALLBACK} parent's entry if listed, else
   * `{ weights }` (except `residual_mixer`: `{ dtype: highPrecision }`). Router, LM head, embeddings and lookup
   * tables are not modules (they have their own fields).
   */
  modules?: Partial<Record<WeightModule, ModuleQuant>>;
  /** n-gram / Engram lookup tables; default: the `embeddings` format. */
  lookupTables?: TensorFormat;
  /** CSA global (compressed) KV cache; default `kvCache`. */
  compressedKvCache?: TensorFormat;
  /** Indexer key cache; default `kvCache`. */
  indexerKeyCache?: TensorFormat;
  /** Recurrent (linear-attention/SSM) state; default `'fp32'`. */
  recurrentState?: Dtype;
  /** Embedding table; default `{ dtype: highPrecision }`. */
  embeddings?: TensorFormat;
  /** LM head weights; default: `embeddings`. */
  lmHead?: TensorFormat;
  /** Dtype the LM-head GEMM input is quantized to (e.g. NVFP4 W4A4 heads); default `activations`. */
  lmHeadGemmInput?: Dtype;
  /** Norm weights, router, biases, conv taps; default `'bf16'`. */
  highPrecision?: Dtype;
  /** Attention-core math dtype (Q·Kᵀ, P·V); default: `activations`. */
  attentionMath?: Dtype;
  /** Requested float matmul accumulator; default `'fp32'` (bf16/tf32/fp32 inputs always accumulate in fp32). */
  accumulate?: 'fp32' | 'fp16';
  /** Structured sparsity of linear-layer weights (every module but `residual_mixer`); default `'dense'`. */
  weightSparsity?: 'dense' | '2:4';
}

/**
 * Parallelism degrees. Each is ≥ 1. World size = tp · pp · dp devices; the expert-parallel group (ep) is laid
 * over the tp × dp ranks of a stage (docs §5, MoE modes A/B).
 */
export interface ParallelConfig {
  /** Tensor-parallel degree (devices sharing each layer's weights). Count. */
  tp: number;
  /** Pipeline-parallel degree (stages, contiguous layer ranges). Count. */
  pp: number;
  /** Data-parallel degree (independent replicas, each serving batch / dp sequences). Count. */
  dp: number;
  /** Expert-parallel degree (devices sharing the routed experts of a MoE layer). Count. */
  ep: number;
  /**
   * EP load imbalance ρ: routed (token, expert) assignments on the busiest device of an EP group over the mean
   * device's, averaged over layers and steps (max/mean, ≥ 1; 1 = perfectly balanced). Scales the busiest device's
   * expert GEMM rows, and with {@link ParallelConfig.epImbalanceScalesAllToAll} also its MoE mode B dispatch /
   * combine all-to-all bytes; never the expert weights it reads (docs §5 "EP load imbalance"). A user input: the load
   * balancer that lowers it (EPLB) is not modelled. Only read when ep > 1. Default `DEFAULT_EP_IMBALANCE`
   * (lower/moe.ts, 1.2; absent and an explicit 1.2 are the same). Dimensionless.
   */
  epImbalance?: number;
  /**
   * When true, the MoE mode B dispatch / combine all-to-alls carry the busiest device's `A_max` rows (scaled by ρ:
   * it receives the assignments routed to its experts and the step waits for it); when false or absent (default),
   * they carry the mean volume `Ts·k` (each device's own token copies). Off by default so default results stay the
   * ones from before. Only read when ep > 1 in mode B (docs §5 "EP load imbalance").
   */
  epImbalanceScalesAllToAll?: boolean;
  /**
   * Redundant expert slots per EP rank (EPLB replicas of hot experts): extra full copies of routed experts each
   * device stores in every MoE layer, on top of its `E/ep` own experts. They only add resident weight bytes
   * (`r·W_expert` per device per MoE layer, docs §5 "Redundant experts"); the lower load they buy is the user's ρ
   * ({@link ParallelConfig.epImbalance}), and they change neither the routing nor the expert weights read. Only read
   * when ep > 1. Default 0 (absent). Count per device, integer ≥ 0.
   */
  redundantExpertsPerRank?: number;
  /**
   * Wire format of the MoE mode B dispatch all-to-all (tokens to their experts' devices), block scales included
   * ({@link A2aFormat}; docs §5 "Dispatch and combine precision"). Absent (default) = the activation dtype
   * (`QuantConfig.activations`), the format from before this field existed. Only read in mode B (ep > tp).
   */
  dispatchFormat?: A2aFormat;
  /**
   * Wire format of the MoE mode B combine all-to-all (expert outputs back to the token's device). Absent (default)
   * = the activation dtype, at least 16-bit (`max(bpe(activations), 2)`: reductions are never carried below
   * 16-bit), the format from before this field existed. Only read in mode B (ep > tp).
   */
  combineFormat?: A2aFormat;
  /**
   * Achieved fraction η of the link bandwidth an all-to-all reaches, in (0, 1]: its bandwidth term becomes
   * `(n−1)/n · S/(η·B)`, the latency term is unchanged (docs §6 "All-to-all efficiency"). Stands for the
   * communication backend (DeepEP, NIXL-EP, flashinfer one-sided, MoRI, …) not reaching the link's peak; there is no
   * backend choice, the user supplies η. Absent or 1 (default) = the link's modelled bandwidth, the time from before
   * this field existed. Applies to every all-to-all op of the EP group. Dimensionless.
   */
  allToAllEfficiency?: number;
  /**
   * Group the dense FFN layers (`mlp` blocks) and the LM head are tensor-parallel over; size G (count).
   * - `'tp'` (default, also when absent): the replica's own tp ranks, G = tp. At tp = 1 they are whole on every
   *   device, i.e. data-parallel (vLLM DEP; SGLang `--moe-dense-tp-size 1 --enable-dp-lm-head`).
   * - `'ep'`: the whole EP group, G = ep devices spanning ep / tp replicas (SGLang's default under
   *   `--enable-dp-attention`). Each device's tokens are all-gathered over the group before the sharded GEMMs and
   *   reduce-scattered back after. Requires tp = 1 and no speculative decoding (validate.ts).
   * The embedding stays per replica either way. See docs/simulation-model.md §5 "DP attention + EP".
   */
  denseTpGroup?: 'tp' | 'ep';
  /**
   * Decode context-parallel (DCP) degree d, a count of ranks. The decode KV cache of full-attention layers (MLA
   * and full GQA) is sharded along the TOKEN axis over d consecutive ranks of each TP group: each rank stores and
   * reads ceil(context / d) tokens of KV per sequence, and a Q all-gather plus a partial-output + log-sum-exp
   * reduce-scatter over the DCP group follow each sharded attention layer. Adds no devices. Must divide tp, and
   * for GQA also tp / kvHeads (vLLM: dcp in [1, tp / kvHeads]). Absent or 1 = unsharded, today's behaviour bit
   * for bit. See docs/simulation-model.md §5 "Decode context parallelism (DCP)".
   */
  dcp?: number;
  /**
   * Chunked pipeline prefill (CPP; Mooncake's "chunked pipeline parallelism", arXiv 2407.00079 §5.1): with pp > 1,
   * each microbatch's prompt is split into chunks of `Workload.prefillChunkTokens` (C, default 8192) new tokens per
   * sequence that flow through the pipeline stages one after another, so stage s + 1 starts on chunk j while stage s
   * runs chunk j + 1. Changes the prefill time to first token (`PhaseResult.latencySec`) and, a little, the cycle;
   * decode is unchanged. Absent or false = whole-prompt microbatches, results bit for bit as before; also when the
   * prompt is a single chunk (P_new ≤ C). pp = 1: ignored, with a warning. See docs/simulation-model.md §7
   * "Chunked pipeline prefill".
   */
  chunkedPipelinePrefill?: boolean;
}
