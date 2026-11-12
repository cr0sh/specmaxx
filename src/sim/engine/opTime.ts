/**
 * Per-op time on one device: the roofline `max(compute, memory)` for compute ops, the vector peak for
 * elementwise ops and recurrent steps, memory only for embedding gathers, the alpha-beta model for
 * collectives, and the tier/link read rule for off-device transfers (`TransferOp`, design doc §2.3). One
 * exhaustive `switch (op.kind)`. Attention adds its softmax (exp on the SFU, overlapped with the MMAs); a GEMM of
 * few rows may take the engine's GEMV path. An optional per-op latency floor (`SimOptions.perOpLatencyFloorSec`, 0 = off)
 * raises every op below it to it, `t = max(t_op, L)` (docs §7 "Per-op latency floor").
 * See docs/simulation-model.md §7 ("Per-op time", "Attention softmax"), §4 ("GEMV path") and §12 ("Transfer path").
 *
 * Inputs are plain specs: device and link modifiers were already applied by `simulate` (docs §4 "Device
 * variants and modifiers"), so nothing here knows about variants.
 */

import {
  attentionPairs,
  attentionScores,
  indexerPairs,
  opBytes,
  opFlops,
  SOFTMAX_VECTOR_FLOP_PER_SCORE,
  type ByteBreakdown,
} from '../ir/metrics';
import type {
  AttentionOp,
  CollectiveOp,
  ElementwiseOp,
  GemmOp,
  IndexerOp,
  Op,
  RecurrentOp,
  TransferOp,
} from '../ir/ops';
import type { AccumulateType, MathFormat } from '../ir/dtype';
import { gemmComputeSec, gemvComputeSec, type GemmShape } from '../hw/efficiency';
import { matmulPeakFlopsPerSec, vectorPeakFlopsPerSec } from '../hw/peaks';
import type { ComputeEngine } from '../hw/types';
import { collectiveTime, type AllReduceAlgorithm, type CollectivePath } from './collectives';
import { remotePoolTransferTime, transferTime, type TransferPaths } from './transfer';
import type { Bound, OpResult, SimOptions } from './types';

/** Everything the per-op cost needs besides the op itself. Plain data, built once per stage by `simulate`. */
export interface OpTimeContext {
  /** The compute engine the run uses (after device modifiers). */
  engine: ComputeEngine;
  /**
   * Memory bandwidth this engine sees, per device: `min(memory.bandwidthBytesPerSec, engine cap)`.
   * Unit: B/s (read + write combined, as vendors quote it).
   */
  memBandwidthBytesPerSec: number;
  /**
   * Where each parallel group's collectives run (from placement): TP group, EP group, this stage's PP send, the
   * expert-TP group (`placement.ts: expertTpCollectivePath`), and the decode context-parallel group (the d
   * consecutive ranks of a TP group, `collectivePath(groupSpan(dcp))`). `etp` is set only when the model lowers
   * expert-TP collectives (a MoE `expertOutputNorm` with etp > 1), `dcp` only when the phase lowers DCP collectives
   * (`parallel.dcp` > 1, decode). Shared lookup tables add `dp` (strided) and `tp_dp` (contiguous), from
   * `placement.ts:lookupTableCollectivePaths`. Any collective without its group's path is an error.
   * See docs/simulation-model.md §5 "Lookup tables across DP".
   */
  collectivePaths: {
    tp: CollectivePath;
    ep: CollectivePath;
    pp: CollectivePath;
    etp?: CollectivePath;
    dcp?: CollectivePath;
    dp?: CollectivePath;
    tp_dp?: CollectivePath;
  };
  /** All-reduce algorithm (SimOptions.collectiveAlgorithm). */
  collectiveAlgorithm: AllReduceAlgorithm;
  /** Elementwise fusion assumption (SimOptions.elementwiseFusion). */
  elementwiseFusion: SimOptions['elementwiseFusion'];
  /**
   * Transfer paths of this stage's devices per source (offloading on; design doc §2.3): tier, host link, queue
   * depth and n_stream for host RAM and the SSD; the pool link for the remote KV pool. Absent when nothing is offloaded.
   */
  transferPaths?: TransferPaths;
  /** Peak-fallback warnings are appended here (human-readable lines; `simulate` de-duplicates them). */
  warnings: string[];
  /**
   * Per-op latency floor L of on-device ops (`SimOptions.perOpLatencyFloorSec`); absent or 0 = no floor (the
   * roofline time is returned as is). Unit: seconds.
   */
  perOpLatencyFloorSec?: number;
  /**
   * Achieved fraction η of the link bandwidth every all-to-all reaches (`ParallelConfig.allToAllEfficiency`, docs §6
   * "All-to-all efficiency"): its bandwidth term is divided by η. Absent = 1 (the modelled link bandwidth; the time
   * is returned untouched). Dimensionless, in (0, 1].
   */
  allToAllEfficiency?: number;
}

/** Memory-effective bandwidth of `engine` on a device with peak memory bandwidth `deviceMemBandwidthBytesPerSec` (B/s). */
export function engineMemBandwidthBytesPerSec(
  engine: ComputeEngine,
  deviceMemBandwidthBytesPerSec: number,
): number {
  // Formula: memBW = min(device.memory.bandwidthBytesPerSec, engine.memBandwidthCapBytesPerSec ?? ∞)
  // See docs/simulation-model.md §7 ("Per-op time")
  return Math.min(deviceMemBandwidthBytesPerSec, engine.memBandwidthCapBytesPerSec ?? Infinity);
}

/**
 * Roofline of one op: the larger of its compute and memory times (seconds) sets the op time; a tie counts as
 * compute-bound.
 */
export function roofline(computeSec: number, memorySec: number): { sec: number; bound: Bound } {
  // Formula: t = max(t_compute, t_memory)
  // See docs/simulation-model.md §7 ("Per-op time")
  return computeSec >= memorySec
    ? { sec: computeSec, bound: 'compute' }
    : { sec: memorySec, bound: 'memory' };
}

/**
 * Cost of one op on one device (see the module doc for the rule per op kind), then the per-op latency floor
 * ({@link withLatencyFloor}) when the context sets one.
 */
export function opTime(op: Op, ctx: OpTimeContext): OpResult {
  const result = opTimeByKind(op, ctx);
  const floorSec = ctx.perOpLatencyFloorSec;
  return floorSec === undefined || floorSec <= 0 ? result : withLatencyFloor(result, floorSec);
}

/**
 * The per-op latency floor L (docs/simulation-model.md §7 "Per-op latency floor"): EVERY IR op whose time is below L
 * costs L, `t_op = max(t_op, L)`, and records `latencyFloorSec = L`. The lift `Δ = L − t_op` is booked where the stage
 * puts the op on its critical path:
 * - on-device ops: `sec` (it enters compute_s);
 * - collectives (incl. PP sends): the latency term, `commLatencySec += Δ` (α is never hidden, `lat_s`); the
 *   bandwidth term, a link-occupancy time, is unchanged;
 * - off-device transfers: the latency term, `transferLatencySec += Δ` (`sec` enters xfer_s).
 * The op's `bound` stays its own. `floorSec` in seconds.
 */
export function withLatencyFloor(result: OpResult, floorSec: number): OpResult {
  if (result.sec >= floorSec) return result;
  // Formula: t_op = max(t_op, L);  Δ = L − t_op
  // See docs/simulation-model.md §7 ("Per-op latency floor")
  const liftSec = floorSec - result.sec;
  const floored: OpResult = { ...result, sec: floorSec, latencyFloorSec: floorSec };
  switch (result.op.kind) {
    case 'collective':
      // Formula: α' = α + Δ   (so α' + β = L)
      floored.commLatencySec = result.commLatencySec + liftSec;
      return floored;
    case 'transfer':
      floored.transferLatencySec = result.transferLatencySec + liftSec;
      return floored;
    default:
      return floored;
  }
}

/** The cost of one op by its kind, without the latency floor. */
function opTimeByKind(op: Op, ctx: OpTimeContext): OpResult {
  switch (op.kind) {
    case 'gemm':
      return gemmOpTime(op, ctx);
    case 'attention':
      return attentionOpTime(op, ctx);
    case 'indexer':
      return tensorOpTime(op, indexerShape(op), op.math, op.accumulate, 'dense', ctx);
    case 'recurrent':
      return op.mode === 'chunked'
        ? tensorOpTime(op, recurrentChunkShape(op), op.math, op.accumulate, 'dense', ctx)
        : vectorOpTime(op, op.stateDtype, opBytes(op), ctx);
    case 'elementwise':
      // Lookup selection copies are materialized data movement, not fusible arithmetic.
      return vectorOpTime(
        op,
        op.dtype,
        op.fn === 'copy' ? opBytes(op) : elementwiseBytesUnderFusion(op, ctx.elementwiseFusion),
        ctx,
      );
    case 'embedding':
      return memoryOnlyOpTime(op, ctx);
    case 'collective':
      return collectiveOpTime(op, ctx);
    case 'transfer':
      return transferOpTime(op, ctx);
    default: {
      const _never: never = op;
      throw new Error(`unknown op ${JSON.stringify(_never)}`);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Tensor ops: GEMM, attention, indexer, chunked recurrent. All use the shape-efficiency model as a batched GEMM.

/** The GEMM shape of a GEMM op: its own (batch, m, n, k). Unit: elements. */
export function gemmShape(op: GemmOp): GemmShape {
  return { batch: op.batch, m: op.m, n: op.n, k: op.k };
}

/**
 * The equivalent batched-GEMM shape of the fused attention core, for the efficiency model: GQA query heads of
 * one KV head are packed into the rows; the keys a query sees on average form the columns.
 */
export function attentionShape(op: AttentionOp): GemmShape {
  const pairs = attentionPairs(op); // primary + secondary keys (one softmax)
  // Formula: batch = seqs·kvHeads, m = qLen·heads/kvHeads, n = ceil(pairs/qLen), k = qkHeadDim
  // See docs/simulation-model.md §4 ("Shape efficiency")
  return {
    batch: op.seqs * op.kvHeads,
    m: (op.qLen * op.heads) / op.kvHeads,
    n: Math.ceil(pairs / op.qLen),
    k: op.qkHeadDim,
  };
}

/** The equivalent batched-GEMM shape of the indexer (all heads share one key per token, so they pack into rows). */
export function indexerShape(op: IndexerOp): GemmShape {
  const pairsFull = indexerPairs(op); // scored (pooled, capped) keys
  // Formula: batch = seqs, m = qLen·heads, n = ceil(pairs_full/qLen), k = headDim
  // See docs/simulation-model.md §4 ("Shape efficiency")
  return {
    batch: op.seqs,
    m: op.qLen * op.heads,
    n: Math.ceil(pairsFull / op.qLen),
    k: op.headDim,
  };
}

/** The equivalent batched-GEMM shape of a chunked recurrent mixer: one chunk × value dim × key dim per (seq, head, chunk). */
export function recurrentChunkShape(op: RecurrentOp): GemmShape {
  // Formula: batch = seqs·heads·ceil(qLen/chunkLen), m = chunkLen, n = valueDim, k = keyDim
  // See docs/simulation-model.md §4 ("Shape efficiency")
  return {
    batch: op.seqs * op.heads * Math.ceil(op.qLen / op.chunkLen),
    m: op.chunkLen,
    n: op.valueDim,
    k: op.keyDim,
  };
}

/**
 * Roofline of a tensor op: the matmul peak for its (math, accumulate, sparsity) key (with fallback), the shape
 * efficiency of `shape`, and its compulsory memory traffic.
 */
function tensorOpTime(
  op: GemmOp | AttentionOp | IndexerOp | RecurrentOp,
  shape: GemmShape,
  math: MathFormat,
  accumulate: AccumulateType,
  sparsity: 'dense' | '2:4',
  ctx: OpTimeContext,
): OpResult {
  const flops = opFlops(op);
  const bytes = opBytes(op);
  const peak = matmulPeakFlopsPerSec(ctx.engine, math, accumulate, sparsity, ctx.warnings);
  const tiling = ctx.engine.gemmTiling[peak.usedInput] ?? ctx.engine.gemmTiling.default;
  const { efficiency } = gemmComputeSec(shape, peak.flopsPerSec, ctx.engine.computeUnits, tiling);
  // Formula: t_compute = F / (peak · efficiency);  t_memory = bytes / memBW
  // See docs/simulation-model.md §7 ("Per-op time")
  const computeSec = flops === 0 ? 0 : flops / (peak.flopsPerSec * efficiency);
  const memorySec = bytes.totalBytes / ctx.memBandwidthBytesPerSec;
  const { sec, bound } = roofline(computeSec, memorySec);
  return {
    op,
    flops,
    bytes,
    computeSec,
    memorySec,
    commLatencySec: 0,
    commBandwidthSec: 0,
    transferLatencySec: 0,
    transferBandwidthSec: 0,
    sec,
    bound,
    efficiency,
    peakFlopsPerSec: peak.flopsPerSec,
    usedInput: peak.usedInput,
    usedAccumulate: peak.usedAccumulate,
  };
}

/** Math formats a GEMV kernel runs on the vector ALUs (the engine lists vector peaks for these; others have none). */
const GEMV_MATH: ReadonlySet<MathFormat> = new Set<MathFormat>(['fp16', 'bf16', 'fp32']);

/**
 * A GEMM: the tensor-core roofline, or the engine's GEMV path when it is faster (docs §4 "GEMV path"). The GEMV
 * path applies when the engine sets `gemvMaxRows`, `m ≤ gemvMaxRows`, the weights are dense and the math is fp16,
 * bf16 or fp32 (int8 / fp8 / fp4 math has no vector peak, so it keeps the MMA path). Then
 * `t_compute = min(t_mma, F / vectorPeak(math))`; the memory term is unchanged.
 */
function gemmOpTime(op: GemmOp, ctx: OpTimeContext): OpResult {
  const mma = tensorOpTime(op, gemmShape(op), op.math, op.accumulate, op.sparsity, ctx);
  const maxRows = ctx.engine.gemvMaxRows;
  if (maxRows === undefined || op.m > maxRows || op.sparsity !== 'dense' || !GEMV_MATH.has(op.math))
    return mma;
  const vectorPeak = vectorPeakFlopsPerSec(ctx.engine, op.math as 'fp16' | 'bf16' | 'fp32');
  // Formula: t_compute = min(t_mma, t_gemv),  t_gemv = F / vectorPeak(math)
  // See docs/simulation-model.md §4 ("GEMV path")
  const gemvSec = gemvComputeSec(mma.flops, vectorPeak);
  if (gemvSec >= mma.computeSec) return mma;
  const { sec, bound } = roofline(gemvSec, mma.memorySec);
  const { usedInput: _input, usedAccumulate: _accumulate, ...rest } = mma;
  void _input;
  void _accumulate;
  return {
    ...rest,
    computeSec: gemvSec,
    sec,
    bound,
    efficiency: 1,
    peakFlopsPerSec: vectorPeak,
    computePath: 'gemv',
  };
}

/**
 * Softmax time of an attention op on `engine` (docs §7 "Attention softmax"): one FP32 exp2 per score on the SFU plus
 * {@link SOFTMAX_VECTOR_FLOP_PER_SCORE} FLOP per score at the vector FP32 peak, the two pipes in series with each
 * other (conservative). `undefined` when the engine has no `sfuOpsPerSec` (softmax not costed). Unit: seconds.
 */
export function attentionSoftmaxSec(op: AttentionOp, engine: ComputeEngine): number | undefined {
  const sfuOpsPerSec = engine.sfuOpsPerSec;
  if (sfuOpsPerSec === undefined) return undefined;
  const scores = attentionScores(op);
  // Formula: S = seqs·heads·pairs;  t_exp = S / sfuOpsPerSec;  t_vec = 5·S / vectorPeak(fp32);  t_softmax = t_exp + t_vec
  // See docs/simulation-model.md §7 ("Attention softmax")
  const expSec = scores / sfuOpsPerSec;
  const vectorSec =
    (SOFTMAX_VECTOR_FLOP_PER_SCORE * scores) / vectorPeakFlopsPerSec(engine, 'fp32');
  return expSec + vectorSec;
}

/**
 * Fused attention: the tensor-core roofline of QKᵀ and PV, with the softmax overlapped with the MMAs
 * (FlashAttention-3/4 ping-pong scheduling: one warpgroup's softmax runs while another's MMAs do).
 * `t_compute = max(t_mma, t_softmax)`; `Math.max` returns `t_mma` itself when the softmax does not bind, so such ops
 * are bit-identical to an engine without `sfuOpsPerSec`. The memory term is unchanged.
 */
function attentionOpTime(op: AttentionOp, ctx: OpTimeContext): OpResult {
  const mma = tensorOpTime(op, attentionShape(op), op.math, op.accumulate, 'dense', ctx);
  const softmaxSec = attentionSoftmaxSec(op, ctx.engine);
  if (softmaxSec === undefined) return mma;
  // Formula: t_compute = max(t_mma, t_exp + t_vec)   (softmax overlapped with the MMAs)
  // See docs/simulation-model.md §7 ("Attention softmax")
  const computeSec = Math.max(mma.computeSec, softmaxSec);
  const { sec, bound } = roofline(computeSec, mma.memorySec);
  return { ...mma, computeSec, sec, bound, softmaxSec };
}

// ---------------------------------------------------------------------------------------------------------------
// Vector ops: elementwise, recurrent step.

/**
 * Memory traffic of an elementwise op under the fusion assumption: `'none'` = its own fields; `'ideal'` = the
 * activation read and write are 0 (fused into a neighbour), weight and state bytes stay. Unit: bytes.
 */
export function elementwiseBytesUnderFusion(
  op: ElementwiseOp,
  fusion: SimOptions['elementwiseFusion'],
): ByteBreakdown {
  const bytes = opBytes(op);
  switch (fusion) {
    case 'none':
      return bytes;
    case 'ideal': {
      // Formula: bytes_ideal = weight + stateRead + stateWrite   (activations stay on chip)
      // See docs/simulation-model.md §7 ("Per-op time")
      const totalBytes = bytes.totalBytes - bytes.activationReadBytes - bytes.activationWriteBytes;
      return { ...bytes, activationReadBytes: 0, activationWriteBytes: 0, totalBytes };
    }
    default: {
      const _never: never = fusion;
      throw new Error(`unknown elementwise fusion ${String(_never)}`);
    }
  }
}

/** Roofline of a vector op: FLOP at the engine's vector peak for `dtype` (no efficiency term) vs memory traffic. */
function vectorOpTime(
  op: ElementwiseOp | RecurrentOp,
  dtype: ElementwiseOp['dtype'],
  bytes: ByteBreakdown,
  ctx: OpTimeContext,
): OpResult {
  const flops = opFlops(op);
  const peakFlopsPerSec = vectorPeakFlopsPerSec(ctx.engine, dtype);
  // Formula: t_compute = F / vectorPeak(dtype);  t_memory = bytes / memBW
  // See docs/simulation-model.md §7 ("Per-op time")
  const computeSec = flops / peakFlopsPerSec;
  const memorySec = bytes.totalBytes / ctx.memBandwidthBytesPerSec;
  const { sec, bound } = roofline(computeSec, memorySec);
  return {
    op,
    flops,
    bytes,
    computeSec,
    memorySec,
    commLatencySec: 0,
    commBandwidthSec: 0,
    transferLatencySec: 0,
    transferBandwidthSec: 0,
    sec,
    bound,
    peakFlopsPerSec,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Embedding, collectives.

/** An embedding gather: no FLOP, memory time only. */
function memoryOnlyOpTime(op: Op, ctx: OpTimeContext): OpResult {
  const bytes = opBytes(op);
  // Formula: t = max(bytes/memBW, peerReadBytes/B_peer); absent peerReadBytes = bytes/memBW unchanged.
  // See docs/simulation-model.md §7 "Per-op time", §5 "Lookup tables across DP".
  let memorySec = bytes.totalBytes / ctx.memBandwidthBytesPerSec;
  if (op.kind === 'embedding' && op.peerReadBytes !== undefined) {
    const path = ctx.collectivePaths.dp;
    if (path?.kind !== 'flat' || path.level !== 'intra') {
      throw new Error(`peer lookup '${op.name}' needs a flat intra-node DP path`);
    }
    // One gather kernel: max(HBM service, peer bytes / fabric bandwidth), not another collective or launch.
    // Remote-load latency is unmeasured; this is a bandwidth ceiling, not a collective's α.
    memorySec = Math.max(memorySec, op.peerReadBytes / path.link.bandwidthPerDirectionBytesPerSec);
  }
  return {
    op,
    flops: 0,
    bytes,
    computeSec: 0,
    memorySec,
    commLatencySec: 0,
    commBandwidthSec: 0,
    transferLatencySec: 0,
    transferBandwidthSec: 0,
    sec: memorySec,
    bound: 'memory',
  };
}

/** The link level a collective path uses (absent for a group of one). */
function linkLevelOf(path: CollectivePath): OpResult['linkLevel'] {
  switch (path.kind) {
    case 'none':
      return undefined;
    case 'flat':
      return path.level;
    case 'hierarchical':
      return 'hierarchical';
    default: {
      const _never: never = path;
      throw new Error(`unknown collective path ${JSON.stringify(_never)}`);
    }
  }
}

/** A collective: alpha-beta time on the path of its group (docs §6); no FLOP, no HBM traffic in this model. */
function collectiveOpTime(op: CollectiveOp, ctx: OpTimeContext): OpResult {
  const path = ctx.collectivePaths[op.group];
  if (path === undefined)
    throw new Error(`collective '${op.name}': no path for group '${op.group}'`);
  const t = collectiveTime(
    op.collective,
    op.messageBytes,
    op.groupSize,
    path,
    ctx.collectiveAlgorithm,
  );
  // Formula: all_to_all: β′ = β/η,  α unchanged,  t = α + β′   (η = allToAllEfficiency; absent = 1 leaves t as is)
  // See docs/simulation-model.md §6 "All-to-all efficiency"
  const eta = op.collective === 'all_to_all' ? ctx.allToAllEfficiency : undefined;
  const bandwidthSec = eta === undefined ? t.bandwidthSec : t.bandwidthSec / eta;
  const totalSec = eta === undefined ? t.totalSec : t.latencySec + bandwidthSec;
  const result: OpResult = {
    op,
    flops: 0,
    bytes: opBytes(op),
    computeSec: 0,
    memorySec: 0,
    commLatencySec: t.latencySec,
    commBandwidthSec: bandwidthSec,
    transferLatencySec: 0,
    transferBandwidthSec: 0,
    sec: totalSec,
    bound: 'comm',
  };
  const linkLevel = linkLevelOf(path);
  if (linkLevel !== undefined) result.linkLevel = linkLevel;
  return result;
}

/** The stage's path for a transfer op's source; throws when the stage has none (a lowering/engine mismatch). */
function requirePath<P>(path: P | undefined, op: TransferOp): P {
  if (path === undefined)
    throw new Error(`transfer op '${op.name}': no transfer path for tier '${op.source}'`);
  return path;
}

/**
 * An off-device read: `t = α_path + service` of {@link transferTime} on the stage's path for the op's source tier
 * (design doc §2.3), or of {@link remotePoolTransferTime} for the remote KV pool (docs/simulation-model.md §12 "Remote
 * KV pool"). No FLOP, no device-memory bytes (the DMA write into device memory is not charged). Also records
 * the op's use of the shared host link and host RAM (`transferResources`) for the overlapped stage transfer
 * (`pipeline.ts: stageTransfer`); the latency floor ({@link withLatencyFloor}) lifts α only and keeps them.
 */
function transferOpTime(op: TransferOp, ctx: OpTimeContext): OpResult {
  const paths = ctx.transferPaths ?? {};
  // Formula: host RAM / SSD: t = α_tier + α_link + max(Little, IOPS/n, tier/n, link, host staging)
  //          remote pool:    t = α_pool + max(useful/B_pool, useful/B_link)
  // See docs/simulation-model.md §12 ("Transfer path", "Remote KV pool")
  const t =
    op.source === 'remote_pool'
      ? remotePoolTransferTime(op.reads, op.usefulBytes, requirePath(paths.remote_pool, op))
      : transferTime(op.reads, op.usefulBytes, requirePath(paths[op.source], op));
  return {
    op,
    flops: 0,
    bytes: opBytes(op),
    computeSec: 0,
    memorySec: 0,
    commLatencySec: 0,
    commBandwidthSec: 0,
    transferLatencySec: t.latencySec,
    transferBandwidthSec: t.serviceSec,
    transferResources: {
      linkBytes: t.linkBytes,
      linkSec: t.linkSec,
      hostRamBytes: t.hostRamBytes,
      hostRamSec: t.hostRamSec,
    },
    sec: t.sec,
    bound: 'transfer',
  };
}
