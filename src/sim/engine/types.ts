/**
 * Engine input and result types. A simulation takes a model, its numeric formats, its parallelism, a workload
 * and a hardware stack, and returns the theoretical-maximum prefill and decode rates plus a memory-fit report.
 * See docs/simulation-model.md §7 (stage, overlap, pipeline) and §8 (memory fit).
 *
 * Conventions: times are seconds; rates are tokens/s; bytes are per device unless a field says otherwise.
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { ByteBreakdown } from '../ir/metrics';
import type { BlockDraftSpec, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { Op, TransferSource } from '../ir/ops';
import type { DeviceSpec, HardwareStack, LinkSpec } from '../hw/types';

/** The request load being evaluated. */
export interface Workload {
  /** GLOBAL concurrent sequences; each DP replica serves `B_r = batch / dp` (must divide). Count. */
  batch: number;
  /**
   * Prefill length P: prompt tokens per sequence that prefill processes from an EMPTY KV cache. Prefill tok/s @ P =
   * P ÷ the time for that whole prompt (so attention sees `qLen = kvLen = P`). Unit: tokens.
   */
  promptLen: number;
  /**
   * Decode context N: tokens ALREADY in the KV cache at the evaluated decode step (not counting the token being
   * generated). Decode tok/s @ N = the rate of generating one new token per sequence at that point, so attention
   * sees `qLen = 1, kvLen = N + 1` (the new token attends to itself too). Unit: tokens.
   */
  decodeContextLen: number;
  /**
   * Prefill chunk C: the most prompt tokens one prefill step processes per microbatch (chunked prefill, vLLM
   * `max_num_batched_tokens`, SGLang `--chunked-prefill-size`). Two uses:
   * - memory fit: the fit's prefill activation peak is the one of the LAST chunk (`qLen = min(C, P)`, `kvLen = P`),
   *   not of the whole prompt in one pass (docs/simulation-model.md §8 "Activation peak");
   * - with `ParallelConfig.chunkedPipelinePrefill` on and pp > 1: the pipeline chunk, C new tokens per sequence per
   *   chunk (docs §7 "Chunked pipeline prefill"). Only then does it change the prefill rate and time to first
   *   token; otherwise the prefill rate is always the whole prompt in one pass.
   * Absent = `DEFAULT_PREFILL_CHUNK_TOKENS` (8192, `engine/memory.ts`, sourced there); an integer ≥ 1. Unit: tokens.
   */
  prefillChunkTokens?: number;
  /**
   * Prefix-cache hit ratio h ∈ [0, 1): the fraction of each prompt already in the KV cache from earlier requests
   * (production prefix caching: DeepSeek reports 56%, Mooncake 34–55%). Prefill then computes only the new tokens
   * `P_new = P − floor(P·h)` while their attention still reads the `P − P_new` cached tokens' KV (`qLen = P_new`,
   * `kvLen = P`). The cached KV is in device memory, unless KV-cache offload
   * (`OffloadConfig.kvCache.prefixDeviceFraction`) places a share of it in host RAM / on the SSD: then prefill first
   * reads that share from its tier (docs/simulation-model.md §12 "KV cache off-device"). Absent or 0 = today's
   * behaviour, bit for bit. Dimensionless. See docs/simulation-model.md §7 "Prefix cache".
   */
  prefixCacheHitRatio?: number;
}

/**
 * Prompt tokens served from the prefix cache, `floor(P·h)`. They are never computed; their KV is in device memory, or
 * with KV-cache offload a share `1 − f_pre` of it is fetched from a lower tier at prefill
 * (`lower/kvOffload.ts: kvFetchOps`, docs/simulation-model.md §12 "KV cache off-device"). Unit: tokens.
 */
export function cachedPrefixTokens(
  workload: Pick<Workload, 'promptLen' | 'prefixCacheHitRatio'>,
): number {
  // Formula: cached = floor(P · h)
  return Math.floor(workload.promptLen * (workload.prefixCacheHitRatio ?? 0));
}

/** Prompt tokens prefill actually computes, `P_new = P − floor(P·h)` (≥ 1 since h < 1). Unit: tokens. */
export function computedPromptTokens(
  workload: Pick<Workload, 'promptLen' | 'prefixCacheHitRatio'>,
): number {
  // Formula: P_new = P − cached
  return workload.promptLen - cachedPrefixTokens(workload);
}

/**
 * Overlap class of a collective (docs/simulation-model.md §7 "Stage time and overlap"), by its
 * `CollectiveOp.group` (`engine/overlap.ts: overlapClassOf`):
 * - `'tp'`: tensor-, context- and lookup-table collectives (groups `tp`, `etp`, `dcp`, `dp`, `tp_dp`);
 * - `'ep'`: collectives over the EP group (group `ep`): the MoE dispatch/combine all-to-alls, and with DP attention the
 *   dense all-gather / reduce-scatter and the head's sample all-to-all;
 * - `'pp'`: the activation send to the next pipeline stage (`StageResult.sendToNext`, group `pp`).
 */
export type OverlapClass = 'tp' | 'ep' | 'pp';
/**
 * Documented ideal-vs-none physical assumptions (not inference-engine knobs).
 */
export interface SimOptions {
  /**
   * Communication/compute overlap, the DEFAULT fraction of every overlap class ({@link OverlapClass}): `'ideal'`
   * (default) = 1, `'none'` = 0, unless {@link SimOptions.overlapByClass} sets a class (or two-batch overlap sets EP to
   * 1, {@link DeploymentConfig.twoBatchOverlap}). With every class at its endpoint the stage core time is the former
   * rule: `'ideal'`: collective bandwidth time hides behind compute, latency (α) does not, `t = lat + max(compute,
   * bw)` (the PP send off-stage, on the latency path); `'none'`: `t = compute + lat + bw + p2p`. General fractions:
   * `engine/overlap.ts: coreTime`, docs/simulation-model.md §7 "Stage time and overlap".
   */
  overlap: 'ideal' | 'none';
  /**
   * Per-collective overlap (docs/simulation-model.md §7 "Stage time and overlap"): for each overlap class c, the
   * fraction f_c of the class's collective bandwidth (β) time that may run concurrently with the stage's compute
   * (`tp`: TP / expert-TP / DCP / lookup collectives; `ep`: the EP all-to-alls and other EP-group collectives; `pp`: the
   * share of the send to the next stage that leaves the stage and goes on the latency path). α is never hidden, and
   * hidden time never exceeds compute: `t_core,s = lat_s + max(compute_s, Σ_c f_c·bw_c,s) + Σ_c (1 − f_c)·bw_c,s +
   * (1 − f_pp)·p2p_s`. An absent class takes the default: 1 for `ep` in a step two-batch overlap splits, else 1 under
   * `overlap 'ideal'` and 0 under `'none'`. Absent or `{}` = every class at its default, results bit-identical to
   * before. Dimensionless, each a finite number in [0, 1].
   */
  overlapByClass?: Partial<Record<OverlapClass, number>>;
  /**
   * Elementwise fusion. `'none'` (default): every elementwise op reads and writes its activations from memory.
   * `'ideal'`: elementwise activation traffic is 0 (fused into neighbours); weight and state bytes stay.
   */
  elementwiseFusion: 'none' | 'ideal';
  /**
   * All-reduce algorithm: `'best'` (default; min of ring and tree, plus the in-network reduction of
   * {@link SimOptions.inNetworkReduction} when on), `'ring'` or `'tree'`.
   */
  collectiveAlgorithm: 'best' | 'ring' | 'tree';
  /**
   * In-network reduction (docs/simulation-model.md §6 "In-network reduction (NVLS, SHARP)"): `'on'` lets an
   * all-reduce use the in-switch reduction its link level offers (`LinkSpec.inNetworkReduction`: NVLS on an NVLink
   * Switch, SHARP on an InfiniBand switch) as one more candidate of `collectiveAlgorithm: 'best'`; the fastest of
   * ring, tree and the in-network algorithm wins. Explicit `'ring'` / `'tree'`, links without the capability, and
   * every other collective are unchanged. `'off'` (default) = results bit-identical to before.
   */
  inNetworkReduction: 'off' | 'on';
  /**
   * Per-op latency floor L (a what-if sensitivity setting, docs/simulation-model.md §7 "Per-op latency floor"):
   * every IR op (on-device ops, collectives incl. PP sends, off-device transfers) costs at least L on the critical
   * path, `t_op = max(t_op, L)`; a collective's or transfer's lift goes to its latency term. `0` (default) = no
   * floor, results bit-identical to before. There is no measured default: the hardware floor (DRAM load-to-use
   * latency of a dependent op) is unmeasured (review F1). Finite, ≥ 0. Unit: seconds.
   */
  perOpLatencyFloorSec: number;
}

/**
 * Informational overlays: figures computed NEXT TO the theoretical maximum, never folded into it. The headline
 * rates of a result are identical with or without them. Absent or `{}` = none.
 */
export interface OverlayConfig {
  /**
   * Engine dispatch overlay (docs/simulation-model.md §7 "Engine dispatch overlay"): an inference-engine overhead,
   * the one engine-overhead exception of the project, opt-in only. Adds a fixed CPU/driver dispatch cost per kernel
   * to decode and reports the result as {@link SimResult.decodeWithDispatch}. Absent = off.
   */
  engineDispatch?: EngineDispatchOverlay;
}

/** Parameters of the engine dispatch overlay. */
export interface EngineDispatchOverlay {
  /**
   * Dispatch cost τ per kernel: CPU launch, command-processor and graph-node dispatch. Absent =
   * `DEFAULT_DISPATCH_SEC_PER_KERNEL` (2 µs, `engine/dispatch.ts`, sourced there). Finite, ≥ 0. Unit: seconds.
   */
  perKernelSec?: number;
}

/**
 * Deployment-scenario knobs (design doc §0): explicit opt-ins beyond the hardware/math bound of the bare model.
 * Every field is optional and off by default: `deployment` absent or `{}` gives today's results, bit for bit. With
 * a knob on, the result is still the hardware/math upper bound given the knob's stated parameters.
 */
export interface DeploymentConfig {
  /** Storage placement and lookup-table sharing. Absent = device-resident, unshared tables. */
  offload?: OffloadConfig;
  /** Speculative decoding (draft + verify). Absent = off. */
  speculative?: SpecDecodeConfig;
  /**
   * Two-batch overlap (TBO; docs/simulation-model.md §7 "Two-batch overlap"): each microbatch runs as two halves,
   * lowered separately (`engine/twoBatch.ts`), so one half's EP all-to-alls can run while the other half computes.
   * Sequence split (`ceil(b/2)` and `floor(b/2)` sequences) when a microbatch holds two or more sequences; a causal
   * token split of the prompt for a one-sequence prefill; no split (a warning) for a one-sequence decode. In a step it
   * splits, the EP overlap class's default becomes full (f_ep = 1); a step it cannot split keeps the global default; an
   * explicit `SimOptions.overlapByClass.ep` always wins. The halves pay
   * the costs that come with them: GEMMs at half the rows, every weight read twice, α twice. Needs a MoE model in mode
   * B (`ep > tp`); not modelled together with offloading (validation errors). Absent or false = off, results
   * bit-identical to before.
   */
  twoBatchOverlap?: boolean;
}

/** Off-device storage tier a weight group can be placed on (`hardware.host.ram` / `hardware.host.ssd`). */
export type OffloadTier = 'host_ram' | 'nvme';

/**
 * A tier the KV cache can be placed on: the weight tiers plus the remote KV pool (`'remote_pool'`, reached over the
 * inter-node network; `KvCacheOffloadConfig.remotePool`). The same values as `TransferOp.source`.
 */
export type KvTier = TransferSource;

/** A half-open range of decoder-layer indices `[from, to)` over `ModelSpec.layers`. Zero-based indices. */
export interface LayerRange {
  /** First layer index in the range (inclusive). */
  from: number;
  /** One past the last layer index (exclusive); `to > from`. */
  to: number;
}

/**
 * Expected-hit model h(f) of the device-resident expert cache (design doc §2.4): the probability that a unique
 * routed expert needed this step is already resident, as a function of the resident fraction f. `'uniform'`
 * (default): `h = f` (balanced router); `'skewed_classic'` / `'skewed_fine_grained'`: the skewed-routing fits of
 * design doc §5.2; `'custom'`: the user's `h` per unique expert per layer per step at batch 1 (dimensionless, in
 * [0, 1]), clamped to the physical bounds by the engine.
 */
export type HitRatioModel =
  | { kind: 'uniform' }
  | { kind: 'skewed_classic' }
  | { kind: 'skewed_fine_grained' }
  | { kind: 'custom'; h: number };

/** Storage placement (design doc §2), fetch policy, and device-resident lookup-table sharing (docs §5). */
export interface OffloadConfig {
  /**
   * `'transfer'` (default): missed weights stream to the device, which computes. `'cpu_compute'`: the host computes
   * the missed experts (design doc §2.8); a designed hook, rejected by validation for now.
   */
  mode?: 'transfer' | 'cpu_compute';
  /**
   * Outstanding read requests kept in flight per device (Little's law); default and cap: the tier's
   * `maxOutstanding`. 1 serializes the reads. Integer ≥ 1. Count.
   */
  queueDepth?: number;
  /**
   * SSD large-read bandwidth scenario: absent or 'datasheet' preserves the rated B/s ceiling;
   * 'sustained' applies the drive's measured low-QD fraction to reads at least its measured block size.
   * Small table reads and host RAM are unchanged; missing data warns and uses datasheet bandwidth.
   */
  ssdReadRate?: 'datasheet' | 'sustained';
  /**
   * Prefetch accuracy p: the predictor's recall one layer ahead, the share of the needed routed experts it picks from
   * the previous layer's hidden state. While a stage computes, the link copies predicted experts; a share p of them are
   * needed, the rest of the misses are copied after the router, on the critical path.
   * `0` = no prefetching at all (table rows are fetched on demand too: the former
   * `offloadOverlap 'none'`); `1` = perfect prediction (the former `'ideal'`). Absent = {@link
   * DEFAULT_PREFETCH_ACCURACY} (0.8, estimated). Dimensionless, in [0, 1]. See docs/simulation-model.md §12
   * ("Expert prefetch").
   */
  prefetchAccuracy?: number;
  /**
   * Routed experts of MoE layers kept partly off-device, with a resident expert cache. One rule, or a list of rules
   * over disjoint layer ranges (e.g. layers 0–23 in host RAM, 24–47 on the SSD); layers no rule covers keep their
   * experts on the device. The single object is the one-rule list (same meaning, bit-identical results); `[]` = none.
   * See docs/simulation-model.md §12 ("Placement") and docs/design/offload-specdec-latency.md §2.11.
   */
  routedExperts?: RoutedExpertRule | readonly RoutedExpertRule[];
  /**
   * n-gram lookup-table policy (Qwen PLE, DeepSeek Engram): `acrossDp` shares device tables; `placements` holds
   * one rule or a list over disjoint layer ranges. Layers no rule covers keep their tables on the device. Fetches
   * read individual rows, never the whole table. Legacy rule / rule-list inputs mean `{ placements }`, sharing off;
   * absent / `[]` = device, unshared. See LookupTableConfig, docs §5 "Lookup tables across DP", design doc §2.11.
   */
  lookupTables?: LookupTableConfig | LookupTableRule | readonly LookupTableRule[];
  /**
   * Token-embedding placement (a 1-gram lookup table: each token gathers one row of `h` elements, no matmul); absent
   * or `{ tier: 'device' }` = on the device (identity). The table is not layer-indexed: it lives on pipeline stage 0
   * only, so this is one placement, not a rule list. Rejected by validation for a model with tied embeddings
   * (`ModelSpec.tieEmbeddings`: the LM head reuses the matrix for a GEMM every step). On unified memory a
   * `'host_ram'` placement is dropped with a warning (host RAM is the device pool). See docs/simulation-model.md §12
   * ("Token embedding off-device").
   */
  embeddings?: EmbeddingRule;
  /**
   * KV-cache tiering (vLLM CPU offload, LMCache, SGLang HiCache, Mooncake store): a share of each sequence's growing
   * KV cache lives in host RAM, on the SSD or in a remote KV pool and is read back every step by the attention that
   * needs it. Absent, or a no-op setting (`deviceFraction` 1 and no prefix share off the device) = everything on the
   * device, results bit-identical. See docs/simulation-model.md §12 ("KV cache off-device").
   */
  kvCache?: KvCacheOffloadConfig;
}

/**
 * Hit model of a top-k (sparse-attention) layer's device-resident KV: the probability h that an attended past token
 * is in device memory. `'uniform'` (default): `h = f_dev` (the selected tokens are spread evenly over the context);
 * `'custom'`: the user's h (dimensionless, in [0, 1]; e.g. a recency-biased selector with recent tokens kept on the
 * device), clamped to the physical bounds by the lowering. Full-attention layers read every past token, so h has no
 * effect there.
 */
export type KvHitRatioModel = { kind: 'uniform' } | { kind: 'custom'; h: number };

/**
 * KV-cache placement across the device, host RAM, the SSD and a remote KV pool (docs/simulation-model.md §12 "KV cache
 * off-device", "Remote KV pool").
 * Only KV that grows with the context is tiered: GQA K/V of full and top-k layers, MLA latents. Sliding-window caches,
 * indexer keys, recurrent/conv state, CSA caches and the drafter's KV stay on the device. Fractions are shares of each
 * offloadable layer's per-sequence KV, by token.
 */
export interface KvCacheOffloadConfig {
  /** f_dev ∈ [0, 1]: share of each offloadable layer's per-sequence KV kept in device memory. Dimensionless. */
  deviceFraction: number;
  /**
   * f_ram ∈ [0, 1 − f_dev − f_pool]: share kept in host RAM; the rest, `f_ssd = 1 − f_dev − f_ram − f_pool`, is on
   * the SSD. Absent = `1 − f_dev − f_pool` (all remaining off-device KV in host RAM). On unified memory host RAM is the
   * device pool, so f_ram counts as
   * device-resident (a warning). Dimensionless.
   */
  hostRamFraction?: number;
  /**
   * The remote KV pool and its share f_pool (`RemoteKvPoolConfig.fraction`): a memory pool on other machines reached
   * over the inter-node network (Mooncake store, LMCache remote backends, SGLang HiCache L3). Absent = no pool (f_pool
   * = 0). With `deviceFraction` 1 (a prefix-only placement) a given pool is where the cached prefix is read from.
   */
  remotePool?: RemoteKvPoolConfig;
  /** Hit model of top-k (sparse) attention layers; absent = `'uniform'`. No effect on full-attention layers. */
  hitRatio?: KvHitRatioModel;
  /**
   * f_pre ∈ [0, 1]: share of a prefix-cache hit's KV found in device memory at prefill; the rest is read from the
   * tiers in the `f_ram : f_ssd : f_pool` proportion (all 0 ⇒ the pool when `remotePool` is given, else host RAM when
   * `hostRamFraction` is absent, else the SSD). Absent = f_dev. Only matters with
   * `Workload.prefixCacheHitRatio > 0`. Dimensionless.
   */
  prefixDeviceFraction?: number;
}

/**
 * A remote KV pool: DRAM (or flash) on other machines that holds part of the KV cache, read over this device's
 * inter-node link (RDMA). Only the link's bandwidth and one latency per fetch are costed: the pool side is assumed to
 * serve every reader at line rate (no pool-side bandwidth or sharing limit, an explicit upper bound), there is no
 * RDMA software overhead, and writes of new KV into the pool are not priced. See docs/simulation-model.md §12
 * ("Remote KV pool").
 */
export interface RemoteKvPoolConfig {
  /**
   * f_pool ∈ [0, 1 − f_dev]: share of each offloadable layer's per-sequence KV kept in the pool. Ignored with
   * `deviceFraction` 1 (then only the cached prefix, read from the pool, is off the device). Dimensionless.
   */
  fraction: number;
  /**
   * The link of ONE device to the pool, per direction: `bandwidthPerDirectionBytesPerSec` (B_pool, B/s per device)
   * and `latencySec` (α_pool, s, the one-way small-message floor of the transport). Absent = the stack's inter-node
   * link as one device sees it (`hw/nics.ts: interNodeLinkPerDevice`: the node's NICs shared by its devices), and
   * on a stack without one the default InfiniBand NDR400 NIC per device (`kvTransfer.ts:
   * DEFAULT_KV_TRANSFER_LINK`, 50e9 B/s, 1 µs) with a warning.
   */
  link?: Pick<LinkSpec, 'name' | 'bandwidthPerDirectionBytesPerSec' | 'latencySec'>;
  /**
   * Usable capacity of the pool shared by EVERY device of the deployment (all nodes). Absent = unlimited: the pool's
   * memory-fit entry reports an infinite capacity and always fits. Unit: bytes.
   */
  capacityBytes?: number;
}

/**
 * One routed-expert placement rule: the MoE layers in `layers` keep their full local expert set on `tier`, with a
 * device-resident cache of a fraction f of them (per layer). Rules of one list cover disjoint ranges.
 */
export interface RoutedExpertRule {
  /** Tier holding the full expert set of each offloaded layer of this rule. */
  tier: OffloadTier;
  /**
   * Layers whose experts this rule offloads. Absent = every MoE layer, allowed only when this is the only rule.
   * Layers of the range without a MoE block are skipped.
   */
  layers?: LayerRange;
  /** Fraction f of each offloaded layer's experts kept resident in device memory, in [0, 1]. Dimensionless. */
  residentFraction: number;
  /** Hit-ratio model h(f) of this rule's resident cache. */
  hitRatio: HitRatioModel;
  /**
   * Also offload the drafter's own MoE experts (DeepSeek-V4 DSpark) with this rule's tier, f and h; default false.
   * At most one rule of a list may set it.
   */
  includeDraft?: boolean;
}

/**
 * Target n-gram table policy. Missing placements keep every table on the device. The legacy single-rule and
 * rule-list forms of `OffloadConfig.lookupTables` remain accepted and mean `{ placements }`, with sharing off.
 */
export interface LookupTableConfig {
  /** Share device-resident tables across DP replicas; absent/false = off, dp 1 = identity. */
  acrossDp?: boolean;
  /** Per-layer storage placement; absent or [] = every table on the device. */
  placements?: LookupTableRule | readonly LookupTableRule[];
}

/**
 * One lookup-table placement rule: the n-gram tables of the layers in `layers` are read row by row from `tier`
 * (`'device'` = they stay on the device). Rules of one list cover disjoint ranges.
 */
export interface LookupTableRule {
  /** Where these layers' tables live. */
  tier: 'device' | OffloadTier;
  /**
   * Layers whose tables this rule places. Absent = every layer, allowed only when this is the only rule. Layers of the
   * range without an n-gram block are skipped.
   */
  layers?: LayerRange;
  /**
   * Block-scaled tables only: default `'separate'` on host RAM, `'pinned_host'` on NVMe (the scales pinned in host RAM
   * and counted in its capacity; on unified memory they stay in the unified pool and count in the device total),
   * `'interleaved'` for a repacked table (one read per row).
   */
  scaleLayout?: 'separate' | 'pinned_host' | 'interleaved';
}

/**
 * The token-embedding placement rule: where the stage-0 table shard (`ceil(vocab/tp)·h·bpe(quant.embeddings)` bytes
 * per device) lives. Off the device, each step reads its gathered rows from `tier` (docs/simulation-model.md §12
 * "Token embedding off-device"). There is no scale-layout choice: a block-scaled format reads each row as two
 * requests, its value bytes and its block-scale bytes (the `'separate'` layout of n-gram tables); nothing is pinned.
 */
export interface EmbeddingRule {
  /** Where the token-embedding table lives; `'device'` = on the device (identity). */
  tier: 'device' | OffloadTier;
}

/**
 * Per-token draft acceptance (design doc §3.5). `'iid'`: one probability α in [0, 1) for every draft position
 * (E[τ] = (1 − α^(k+1)) / (1 − α)); `'per_position'`: α_j in [0, 1], the probability that draft j is accepted given
 * drafts 1..j−1 were (a short vector is extended with its last value; α_j = 1 is legal). Dimensionless.
 */
export type AcceptanceModel =
  | { kind: 'iid'; alpha: number }
  | { kind: 'per_position'; alphas: number[] };

/** Speculative decoding (design doc §3): a drafter proposes k tokens per step, the target verifies k + 1. */
export interface SpecDecodeConfig {
  /**
   * `'mtp'` needs `model.draft.kind === 'mtp'`; the block methods need `drafter`. Works at any pp: the drafter runs on
   * the last stage (docs/simulation-model.md §13 "Pipeline parallelism"); with pp > 1 the token embedding must stay
   * on the device.
   */
  method: 'mtp' | 'dflash2' | 'dspark';
  /**
   * The block drafter: a Hugging Face repo whose config the loader read, or the labelled hypothetical stand-in.
   * Ignored for MTP. The engine reads `spec` only; `repoId` is provenance for the UI.
   */
  drafter?:
    | { kind: 'repo'; repoId: string; spec: BlockDraftSpec }
    | { kind: 'hypothetical'; spec: BlockDraftSpec };
  /** Draft tokens per step k (verify width k + 1; at most `blockSize − 1` for block drafters). Integer in 1..16. Count. */
  draftLength: number;
  /** Per-token acceptance of the drafts. */
  acceptance: AcceptanceModel;
}

/** Everything one simulation needs. */
export interface SimInput {
  /** Model architecture. */
  model: ModelSpec;
  /** Numeric formats. */
  quant: QuantConfig;
  /** Parallelism degrees. */
  parallel: ParallelConfig;
  /** Request load. */
  workload: Workload;
  /** Hardware (device, topology, variant modifiers). */
  hardware: HardwareStack;
  /** Assumption switches; missing fields take their defaults. */
  options?: Partial<SimOptions>;
  /** Deployment-scenario knobs (offload, speculative decoding); absent or `{}` = off, identical results. */
  deployment?: DeploymentConfig;
  /** Informational overlays (engine dispatch); absent or `{}` = none. Never change the headline rates. */
  overlays?: OverlayConfig;
}

/**
 * What limits an op: tensor/vector compute, memory bandwidth, communication (collectives), or an off-device
 * transfer (host link / storage tier).
 */
export type Bound = 'compute' | 'memory' | 'comm' | 'transfer';

/**
 * How much of the shared transfer resources one off-device transfer op uses, per device and microbatch step: the
 * device's host link (PCIe, per direction) and the node's host RAM bandwidth (docs/simulation-model.md §12 "Overlap
 * across tiers"). Each time is the op's occupancy of THAT resource alone (bytes / bandwidth, no latency term); the
 * op's own source-tier time, α included, is `OpResult.sec`. Host RAM bandwidth is shared by the node's n_host
 * consumers (host-RAM readers and NVMe stagers, each device once), as in `transferTime`.
 */
export interface TransferResourceUse {
  /** Bytes over this device's host link: the op's `usefulBytes`, 0 on unified memory. Unit: bytes. */
  linkBytes: number;
  /** Time on this device's host link, `linkBytes / B_link` (0 on unified memory). Unit: seconds. */
  linkSec: number;
  /**
   * Bytes through host RAM: host-RAM tier `bytes_tier` (read once); NVMe with staging `2·bytes_tier` (written, then
   * read); 0 otherwise. Unit: bytes.
   */
  hostRamBytes: number;
  /**
   * Time on host RAM bandwidth, B_ram shared by n_host: host-RAM tier `bytes_tier / (B_ram / n_host)`; NVMe with
   * staging `bytes_tier / (B_ram / (2·n_host))`; 0 otherwise. Unit: seconds.
   */
  hostRamSec: number;
}

/** The cost of one op on one device. */
export interface OpResult {
  /** The op costed. */
  op: Op;
  /** Dense-equivalent FLOP of the op. Unit: FLOP. */
  flops: number;
  /** Memory traffic of the op, per device. Unit: bytes. */
  bytes: ByteBreakdown;
  /** Compute time (shape efficiency included). Unit: seconds. */
  computeSec: number;
  /** Memory time = totalBytes / memory bandwidth. Unit: seconds. */
  memorySec: number;
  /** Collective latency (α) terms; 0 for non-collectives. Unit: seconds. */
  commLatencySec: number;
  /** Collective bandwidth (β) terms; 0 for non-collectives. Unit: seconds. */
  commBandwidthSec: number;
  /** Off-device read latency terms (α_tier + α_link) of a transfer op; 0 for other ops. Unit: seconds. */
  transferLatencySec: number;
  /**
   * Off-device service term of a transfer op (the max of its Little's-law, IOPS, tier-, host- and link-bandwidth
   * terms, design doc §2.3); 0 for other ops. Unit: seconds.
   */
  transferBandwidthSec: number;
  /**
   * Transfer ops only: the op's use of the shared link and host RAM, for the overlapped stage transfer
   * (`pipeline.ts: stageTransfer`); absent on other ops.
   */
  transferResources?: TransferResourceUse;
  /** Op time: the roofline max(computeSec, memorySec), the collective time, or the transfer time. Unit: seconds. */
  sec: number;
  /** Which term set `sec`. */
  bound: Bound;
  /** Tensor shape efficiency in (0, 1] (tensor ops only; dimensionless). */
  efficiency?: number;
  /** Peak used for the compute term, per device. Unit: FLOP/s. */
  peakFlopsPerSec?: number;
  /** Input class of the peak actually used (after fallback). */
  usedInput?: MathFormat;
  /** Accumulator of the peak actually used (after fallback). */
  usedAccumulate?: AccumulateType;
  /**
   * Attention ops on an engine with `sfuOpsPerSec`: the softmax time `t_exp + t_vec` (exp2 on the SFU plus the
   * per-score vector ops), which overlaps the MMAs: `computeSec = max(t_mma, softmaxSec)` with
   * `t_mma = flops / (peakFlopsPerSec · efficiency)` (docs/simulation-model.md §7 "Attention softmax"). Absent on
   * other ops. Unit: seconds.
   */
  softmaxSec?: number;
  /**
   * GEMM ops only: `'gemv'` when the engine's GEMV path (`ComputeEngine.gemvMaxRows`, §4 "GEMV path") was faster
   * than the MMA path and set `computeSec` (then `peakFlopsPerSec` is the vector peak and `efficiency` 1); absent =
   * the MMA path.
   */
  computePath?: 'gemv';
  /**
   * Ops with `SimOptions.perOpLatencyFloorSec` > 0 whose own time was below it: the floor L that set `sec` (then
   * `sec = L`; a collective's `commLatencySec` / a transfer's `transferLatencySec` carries the lift; `bound` stays the
   * op's own). Absent otherwise. Unit: seconds.
   */
  latencyFloorSec?: number;
  /** Collectives only: which link level carried it. */
  linkLevel?: 'intra' | 'inter' | 'hierarchical';
}

/** The cost of one lowered layer on one device. */
export interface LayerResult {
  /** `'embed'`, `'layer.<i>'` or `'head'`. */
  name: string;
  /** Index into `ModelSpec.layers`, or `null` for the embedding and the head. */
  layerIndex: number | null;
  /** Per-op costs, in execution order. */
  ops: OpResult[];
  /** Σ op sec, without overlap. Unit: seconds. */
  sec: number;
}

/** The cost of one pipeline stage for one microbatch, on one device. */
export interface StageResult {
  /** Stage index, 0-based. */
  index: number;
  /** Per-layer costs. */
  layers: LayerResult[];
  /** Σ of compute-op times: every op that is neither a collective nor an off-device transfer. Unit: seconds. */
  computeSec: number;
  /** Σ collective latency (α) terms. Unit: seconds. */
  commLatencySec: number;
  /** Σ collective bandwidth (β) terms. Unit: seconds. */
  commBandwidthSec: number;
  /**
   * Communication time not hidden behind compute under the overlap fractions: `t_core,s − compute_s`, where
   * `t_core,s` is the stage time without transfers (docs §7 "Stage time and overlap", `engine/overlap.ts: coreTime`).
   * Unit: seconds.
   */
  exposedCommSec: number;
  /**
   * Stage time per microbatch, docs §7: `t_core,s = lat_s + max(compute_s, H_s) + E_s + (1 − f_pp)·p2p_s` (H_s / E_s
   * the hideable / serial collective β time; every f = 1: lat + max(compute, bw); every f = 0: compute + lat + bw +
   * p2p) extended by the transfers not hidden by prefetching (docs §12 "Expert prefetch"). Unit: seconds.
   */
  sec: number;
  /**
   * The stage's effective transfer time `xfer_s` (`pipeline.ts: stageTransfer`): Σ transfer op times (latency +
   * service); for a stage that prefetches from both host RAM and the SSD, the overlapped `D_s + P_s` (the demand share
   * serial, the prefetched share bounded by the busiest stream or shared resource; docs §12 "Overlap across tiers"),
   * which can be less than Σ op sec. Unit: seconds.
   */
  transferSec: number;
  /** Transfer time not hidden by prefetching: `t_s − t_core,s` (0 when nothing is offloaded). Unit: seconds. */
  exposedTransferSec: number;
  /** Cost of the activation send to the next stage (absent on the last stage). */
  sendToNext?: OpResult;
}

/** The result of one phase (prefill or decode). */
export interface PhaseResult {
  /** Which phase. */
  phase: 'prefill' | 'decode';
  /** Microbatches in flight per replica, `m = min(pp, B_r)`. Count. */
  microbatches: number;
  /** Sequences per microbatch, `b_mb = ceil(B_r / m)`. Count. */
  microbatchSize: number;
  /** Per-stage costs. */
  stages: StageResult[];
  /** Decode-only last-stage → stage-0 uint32 sampled-token send (α and β seconds); absent for pp1 or prefill. */
  loopBack?: OpResult;
  /** Pipeline cycle: time for every microbatch of a replica to pass one step (docs §7). Unit: seconds. */
  cycleSec: number;
  /** Prefill: time to first token of a replica batch; decode: time per output token. Unit: seconds. */
  latencySec: number;
  /** Aggregate rate over all DP replicas (prefill: prompt tokens; decode: generated tokens). Unit: tokens/s. */
  tokensPerSec: number;
  /**
   * Prefill with a prefix-cache hit only: COMPUTED tokens per second, `dp·B_r·P_new / cycle`, counting just the
   * tokens prefill actually ran (`tokensPerSec` counts every prompt token, cached ones included). Absent without a
   * prefix cache (then both are equal). Unit: tokens/s.
   */
  computedTokensPerSec?: number;
  /** Rate seen by one sequence. Unit: tokens/s. */
  perSequenceTokensPerSec: number;
  /**
   * Communication of the phase on its critical path, split into latency and bandwidth (design doc §4.2). Sums of
   * numbers the engine already computes, display only (`cycleSec` does not depend on them):
   * `latencySec = Σ_s lat_s + Σ α_p2p`, `bandwidthSec = Σ_s bw_s + Σ β_p2p`,
   * `exposedSec = Σ_s exposedComm_s + Σ_s f_pp·p2p_s (the share of the PP sends on the latency path; the rest is
   * inside exposedComm_s) + loopBack.sec`. The α/β sums also include loopBack, once, in every mode.
   * Unit: seconds.
   */
  comm: { latencySec: number; bandwidthSec: number; exposedSec: number };
  /**
   * Off-device transfers of the phase (design doc §2.5); absent when nothing is offloaded. `sec` = Σ_s transferSec,
   * `exposedSec` = Σ_s exposedTransferSec (seconds); `bytes` = Σ useful bytes over the host link (bytes);
   * `requests` = Σ expected read requests (count, real) of one microbatch through every stage.
   */
  transfer?: { sec: number; exposedSec: number; bytes: number; requests: number };
  /**
   * Decode with speculative decoding on (design doc §3.5); absent otherwise. The step from verify cost to accepted
   * tokens: one spec-dec step costs `stepCostRatio` plain decode steps and commits E[τ] tokens per sequence, so
   *   speedup = E[τ] / stepCostRatio   (same dp and B_r; exact up to float error)
   * and spec-dec pays iff stepCostRatio < E[τ]. See docs/simulation-model.md §13 ("Accepted tokens", "Break-even
   * batch").
   */
  speculative?: {
    /** Draft share of one step: Σ op times of the `draft.*` layers, without overlap. Unit: seconds. */
    draftSec: number;
    /** Verify share of one step: Σ op times of every other layer, without overlap. Unit: seconds. */
    verifySec: number;
    /** E[τ], tokens committed per sequence per step incl. the bonus token. Count (real, in [1, k + 1]). */
    expectedTokensPerStep: number;
    /** Aggregate decode rate of the same input with speculative decoding off. Unit: tokens/s. */
    offTokensPerSec: number;
    /** Pipeline cycle of the same input with speculative decoding off (docs §7). Unit: seconds. */
    offCycleSec: number;
    /** Cost of one spec-dec step in plain decode steps, `cycleSec / offCycleSec`. Dimensionless. */
    stepCostRatio: number;
    /** `tokensPerSec / offTokensPerSec` (= E[τ] / stepCostRatio). Dimensionless. */
    speedup: number;
  };
  /**
   * Prefill with chunked pipeline prefill in effect (`ParallelConfig.chunkedPipelinePrefill`, pp > 1, K > 1 chunks);
   * absent otherwise. Then `latencySec` is the chunked time to first token of a replica batch, `cycleSec` the
   * chunked cycle, and `stages` hold the per-stage sums over the K chunks of one microbatch
   * (`engine/chunkedPipeline.ts`; docs/simulation-model.md §7 "Chunked pipeline prefill").
   */
  chunkedPipeline?: {
    /** Chunk size C: new prompt tokens per sequence per chunk (the last chunk may be shorter). Unit: tokens. */
    chunkTokens: number;
    /** Chunks per microbatch, K = ceil(P_new / C). Count. */
    chunks: number;
    /**
     * The same prefill with whole-prompt microbatches (chunked pipeline prefill off; the result of before). Rates
     * in tokens/s, times in seconds.
     */
    unchunked: PhaseRates;
  };
}

/** Memory use of one device of one pipeline stage. */
export interface DeviceMemory {
  /** Stage index. */
  stage: number;
  /**
   * Resident weights on this device (incl. embedding / LM-head shards on the first / last stage; the embedding shard
   * only while it is on the device, `OffloadConfig.embeddings`). Unit: bytes.
   */
  weightBytes: number;
  /**
   * Bytes of n-gram lookup-table shards resident on this device (ngramTableShardBytes, including the selected
   * DP distribution), ALREADY INCLUDED in `weightBytes`; 0 when the tables are off-device or the model has none. Display only
   * (the memory card's table segment). Unit: bytes.
   */
  lookupTableBytes: number;
  /**
   * Per-replica sequence state RESIDENT on this device: KV + MLA latent + indexer keys + recurrent/conv state, × B_r.
   * With KV-cache offload (`OffloadConfig.kvCache`) only the device share f_dev of the offloadable KV counts here;
   * the rest is in `TierMemory.breakdown.kvCache`. Unit: bytes.
   */
  sequenceStateBytes: number;
  /**
   * Peak transient activations: the max over the ops of the decode step and of the LAST prefill chunk (chunked
   * prefill, `Workload.prefillChunkTokens`) of activation read + write or live lookup-exchange workspace. Unit: bytes.
   */
  activationPeakBytes: number;
  /**
   * Staging buffer(s) for streamed weight groups (double-buffered for prefetched transfers, `q > 0`; design doc §2.7);
   * a stage that prefetches from both host RAM and the SSD keeps one buffer being read plus one being filled per
   * source (`memory.ts: stageStagingBytes`). 0 when nothing streams. Unit: bytes.
   */
  stagingBytes: number;
  /**
   * Draft weights, the drafter's own token-embedding copy at pp > 1 (`memory.ts: draftEmbeddingCopyBytes`) and draft
   * KV/state (speculative decoding on; last stage); 0 otherwise. Unit: bytes.
   */
  draftBytes: number;
  /**
   * Per-position recurrent/conv state snapshots of a k+1-token verify, × B_r (spec-dec on, hybrid models; design doc
   * §3.3); 0 otherwise. Unit: bytes.
   */
  speculativeTransientBytes: number;
  /**
   * Runtime reserve: bytes the framework and driver keep on the device besides the model (CUDA context, allocator
   * pools, workspaces), `HardwareStack.runtimeReserveBytes`; 0 by default. Unit: bytes.
   */
  runtimeReserveBytes: number;
  /**
   * weightBytes + sequenceStateBytes + activationPeakBytes + runtimeReserveBytes (+ stagingBytes + draftBytes +
   * speculativeTransientBytes, all 0 with the deployment knobs off). Unit: bytes.
   */
  totalBytes: number;
  /**
   * Capacity the check uses: the GPU-usable capacity `f · physicalCapacityBytes` (unified memory: f = the GPU-usable
   * fraction, {@link MemoryReport.gpuUsableFraction}; dedicated memory: f = 1, the whole device). Unit: bytes.
   */
  capacityBytes: number;
  /** Installed memory of the device (the selected capacity option). Unit: bytes. */
  physicalCapacityBytes: number;
  /** True if totalBytes ≤ capacityBytes. */
  fits: boolean;
}

/** The memory-fit report. */
export interface MemoryReport {
  /** Memory kind of the (modified) device; `'unified'` = the pool is shared with the CPU and OS. */
  kind: 'dedicated' | 'unified';
  /**
   * GPU-usable fraction f of each device's memory used for the check: the `HardwareStack.gpuUsableFraction` override,
   * else the device's `MemorySpec.gpuUsableFraction`, else 1; always 1 on dedicated memory. Dimensionless, (0, 1].
   */
  gpuUsableFraction: number;
  /** One entry per pipeline stage (all devices of a stage are symmetric). */
  perStage: DeviceMemory[];
  /** True if every stage fits. */
  fits: boolean;
  /**
   * Most sequences one DP replica could hold at this context length:
   * `floor(min_s (capacity − W_s − act_s − reserve) / state_s)` with the GPU-usable capacity (docs §8). Infinity
   * when no stage keeps per-sequence state; the activation peak is the evaluated batch's. Count.
   */
  maxSequencesPerReplica: number;
  /**
   * Off-device storage tiers holding offloaded weight groups (one entry per tier used; empty when nothing is
   * offloaded). Their `fits` enters `fits` above.
   */
  tiers: TierMemory[];
}

/** What fills an off-device tier, by category. All fields are bytes; they sum to the tier's total. */
export interface TierBytes {
  /** n-gram lookup-table shards (PLE, Engram), plus pinned scales on host RAM. Unit: bytes. */
  lookupTables: number;
  /**
   * ALL local routed experts of the offloaded MoE layers: the tier holds every expert, whatever
   * `residentFraction` is (that is a device-side cache of a subset, a copy). Unit: bytes.
   */
  routedExperts: number;
  /** The speculative drafter's own routed experts (`includeDraft`). Unit: bytes. */
  draftExperts: number;
  /**
   * The token-embedding table shard of a stage-0 device (`OffloadConfig.embeddings` off the device):
   * `ceil(vocab/tp)·h·bpe(quant.embeddings)` per device; 0 on other stages and with the table on the device.
   * Unit: bytes.
   */
  tokenEmbeddings: number;
  /**
   * Off-device KV cache of the live sequences (`OffloadConfig.kvCache`): `B_r · Σ_blocks f_t · KV_block(N)` per device
   * on this tier t (f_ram, f_ssd or f_pool, N the larger of the prompt and decode contexts); 0 without KV-cache offload.
   * Unit: bytes.
   */
  kvCache: number;
}

/**
 * Bytes placed on one off-device storage tier, against its capacity (design doc §2.7): host RAM and the SSD per node
 * (the fullest node), the remote KV pool over the whole deployment (every node's devices share one pool).
 */
export interface TierMemory {
  /** Which tier. */
  tier: KvTier;
  /**
   * Bytes the node's devices place on the tier: the FULL offloaded groups (all experts of the offloaded layers,
   * whole table shards, the token-embedding shard of each stage-0 device), since the device cache is a copy; plus
   * pinned table scales on host RAM. Unit: bytes (per node, summed over its devices; the remote pool: summed over
   * every device of the deployment).
   */
  bytes: number;
  /** `bytes` by category, of the fullest node (its parts sum to `bytes`; the remote pool: of the whole deployment). Unit: bytes. */
  breakdown: TierBytes;
  /**
   * Installed capacity of the tier (`HostMemorySpec.capacityBytes` / `NvmeSpec.capacityBytes` /
   * `RemoteKvPoolConfig.capacityBytes`; `Infinity` for a pool without a stated capacity). Unit: bytes.
   */
  capacityBytes: number;
  /** True if `bytes ≤ capacityBytes`. */
  fits: boolean;
}

/** The full simulation result. */
export interface SimResult {
  /** Prefill phase result. */
  prefill: PhaseResult;
  /** Decode phase result (at `decodeContextLen`). */
  decode: PhaseResult;
  /** Memory fit. */
  memory: MemoryReport;
  /** Devices used, tp · pp · dp. Count. */
  devicesUsed: number;
  /** The device spec actually simulated (after modifiers), for the UI to display. */
  effectiveDevice: DeviceSpec;
  /** Peak fallbacks, unused devices, applied modifiers, unified-memory notes (human-readable lines). */
  warnings: string[];
  /**
   * Prefill with the communication/compute overlap removed (the same input with every overlap fraction 0 and
   * two-batch overlap off, i.e. `overlap: 'none'` on the full-batch lowering), a labelled what-if next to the headline
   * (review F7; docs/simulation-model.md §7 "Prefill without overlap"). Present only when some resolved overlap
   * fraction is above 0 (by default: `overlap: 'ideal'`) and the run's prefill has inter-device communication (any
   * collective or PP send with a nonzero time); absent otherwise.
   */
  prefillWithoutOverlap?: PhaseRates;
  /**
   * Decode with the engine dispatch overlay (`SimInput.overlays.engineDispatch`); absent when the overlay is off.
   * An engine overhead, outside the theoretical maximum: informational only, `decode` is unchanged.
   */
  decodeWithDispatch?: DispatchOverlayResult;
}

/** The rates of one phase, without the per-op detail (a what-if figure next to a headline). */
export interface PhaseRates {
  /** Aggregate rate over all DP replicas. Unit: tokens/s. */
  tokensPerSec: number;
  /** Rate seen by one sequence. Unit: tokens/s. */
  perSequenceTokensPerSec: number;
  /** Prefill: time to first token of a replica batch; decode: time per output token. Unit: seconds. */
  latencySec: number;
  /** Pipeline cycle. Unit: seconds. */
  cycleSec: number;
}

/** Decode rates with the engine dispatch overlay (docs/simulation-model.md §7 "Engine dispatch overlay"). */
export interface DispatchOverlayResult extends PhaseRates {
  /** Dispatch cost τ per kernel used. Unit: seconds. */
  perKernelSec: number;
  /**
   * Kernels per decode step: the IR ops of every stage's layers for one microbatch (one kernel per op, no fusion;
   * PP sends and the token return not counted). Count.
   */
  kernelsPerStep: number;
}
