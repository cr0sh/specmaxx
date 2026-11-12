/**
 * Memory fit: resident weights + per-sequence state (KV, MLA latent, indexer keys, recurrent and conv state)
 * + peak transient activations (decode step and the last prefill chunk) + runtime reserve (+ offload staging) on one
 * device of each pipeline stage, against its GPU-usable capacity (the whole device on dedicated memory, the OS's GPU
 * limit on unified memory); and, with weights offloaded, the bytes each off-device tier of a node holds against its
 * capacity. See docs/simulation-model.md §8 ("Memory fit") and §12 ("Memory tiers").
 */

import { matrixStoredBytes, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { DraftSpec, LayerSpec, ModelSpec } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import { draftResidentWeightBytes, draftSequenceStateBytes } from '../lower/draft';
import {
  embeddingFormat,
  embeddingResidentWeightBytes,
  embeddingShardBytes,
} from '../lower/embedding';
import { lmHeadFormat, lmHeadResidentWeightBytes, lmHeadVocabShard } from '../lower/head';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  stageLayerRange,
  type LoweredModel,
  type LoweringContext,
} from '../lower/lower';
import { moeLocalExpertBytes } from '../lower/moe';
import { lookupTableFormat, ngramTableShardBytes, tableRowBytes } from '../lower/ngram';
import { kvOffloadableStateBytes } from '../lower/kvOffload';
import { draftExpertPlacement, expertPlacementAt, tablePlacementAt } from '../lower/offload';
import { prefetchesFromMultipleTiers } from './transfer';
import type { ResolvedHost } from './host';
import type { DeviceMemory, KvTier, MemoryReport, TierBytes, TierMemory } from './types';

/** Inputs of {@link memoryReport}. */
export interface MemoryInput {
  /** The model. */
  model: ModelSpec;
  /** A lowering context carrying the quantization and parallelism (its phase and token fields are not used). */
  ctx: LoweringContext;
  /** Sequences per DP replica `B_r` (every one keeps its state on every stage). Count. */
  sequencesPerReplica: number;
  /**
   * Context length the state is sized for: `max(promptLen, decodeContextLen + 1)` (the cache after prefill, or after
   * the decode step appended its new token). Unit: tokens.
   */
  contextTokens: number;
  /** The lowered phases (prefill and decode), for the offload staging buffers. */
  lowered: LoweredModel[];
  /**
   * The lowered phases whose ops set the activation peak: the decode step and the LAST prefill chunk
   * ({@link prefillChunkShape}); default `lowered`.
   */
  activationLowered?: LoweredModel[];
  /** Installed memory capacity per device (the selected option of the modified device). Unit: bytes. */
  capacityBytes: number;
  /** Memory kind of the modified device. */
  kind: 'dedicated' | 'unified';
  /**
   * GPU-usable fraction f of `capacityBytes` (unified memory: {@link resolveGpuUsableFraction}); default 1.
   * Dimensionless, (0, 1].
   */
  gpuUsableFraction?: number;
  /** Runtime reserve per device (framework, driver, CUDA context); default 0. Unit: bytes. */
  runtimeReserveBytes?: number;
  /** Devices per node (the host tiers are shared by a node's devices); default 1. Count. */
  devicesPerNode?: number;
  /** The resolved host side when something is offloaded (tier capacities); absent otherwise. */
  host?: ResolvedHost;
}

/**
 * Default prefill chunk C of the memory fit (`Workload.prefillChunkTokens`): 8192 tokens per step. Sources: SGLang's
 * `--chunked-prefill-size` default (8192; https://docs.sglang.ai/advanced_features/server_arguments.html) and vLLM's
 * `max_num_batched_tokens` default for the offline `LLM` class and for the API server on H100-class GPUs (8192;
 * https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py). Smaller budgets (vLLM 2048 on other
 * GPUs, llama.cpp `n_ubatch` 512, mlx-lm `prefill_step_size` 2048) only shrink the peak, so 8192 is the
 * conservative common choice. Unit: tokens.
 */
export const DEFAULT_PREFILL_CHUNK_TOKENS = 8192;

/** The shape of one prefill step of the memory fit (see {@link prefillChunkShape}). */
export interface PrefillChunkShape {
  /** Sequences in the step. Count. */
  seqs: number;
  /** New prompt tokens per sequence in the step. Unit: tokens. */
  qLen: number;
  /** Keys visible to the step's last query (past chunks + this one). Unit: tokens. */
  kvLen: number;
}

/**
 * The prefill step whose activations the memory fit counts, under chunked prefill with a budget of C tokens per
 * step: a microbatch of `seqs` prompts of P tokens, of which `N = newTokens` are computed (`N = P` without a prefix
 * cache; with a prefix-cache hit `N = P − floor(P·h) < P` and the cached tokens only add keys, so `kvLen` stays P).
 * - `seqs·N ≤ C`: the whole microbatch in one step (`seqs, qLen = N, kvLen = P`), as the rate is costed.
 * - `N ≤ C < seqs·N`: as many whole prompts as fit the budget (`floor(C/N)` sequences, `qLen = N`, `kvLen = P`).
 * - `C < N`: one sequence's LAST chunk (`qLen = C`, `kvLen = P`): the chunk that sees the whole prompt.
 * Without a cache (`N = P`) these are `qLen = kvLen = P` as before.
 * Activations of every op grow with the step's tokens `seqs·qLen`, so this step bounds the peak of all chunks.
 */
export function prefillChunkShape(
  promptLen: number,
  seqs: number,
  chunkTokens: number,
  newTokens: number = promptLen,
): PrefillChunkShape {
  // `newTokens` = P_new ≤ P: prompt tokens actually computed (prefix cache); the cached P − P_new tokens only add keys.
  // Formula: seqs·N ≤ C → (seqs, N, P);  N ≤ C → (floor(C/N), N, P);  else (1, C, P)   [N = P_new; P_new = P: no cache]
  // See docs/simulation-model.md §8 ("Activation peak")
  if (seqs * newTokens <= chunkTokens) return { seqs, qLen: newTokens, kvLen: promptLen };
  if (newTokens <= chunkTokens)
    return { seqs: Math.floor(chunkTokens / newTokens), qLen: newTokens, kvLen: promptLen };
  return { seqs: 1, qLen: chunkTokens, kvLen: promptLen };
}

/**
 * The GPU-usable fraction f the fit uses: on unified memory the stack's override (`HardwareStack.gpuUsableFraction`,
 * a raised OS limit), else the device's default (`MemorySpec.gpuUsableFraction`), else 1; on dedicated memory always
 * 1 (the device is the GPU's own). Dimensionless, (0, 1].
 */
export function resolveGpuUsableFraction(
  memory: { kind: 'dedicated' | 'unified'; gpuUsableFraction?: number },
  override: number | undefined,
): number {
  if (memory.kind !== 'unified') return 1;
  // Formula: f = override ?? f_device ?? 1   (unified memory)
  return override ?? memory.gpuUsableFraction ?? 1;
}

/**
 * True if two stored formats are the same bytes on device (dtype and group-quant overhead), so one allocation can
 * serve both the embedding and a tied LM head.
 */
export function sameStoredFormat(a: TensorFormat, b: TensorFormat): boolean {
  return (
    a.dtype === b.dtype &&
    (a.extraBitsPerElem ?? 0) === (b.extraBitsPerElem ?? 0) &&
    (a.valueBitsPerElem ?? 0) === (b.valueBitsPerElem ?? 0) &&
    (a.perDimBits ?? 0) === (b.perDimBits ?? 0)
  );
}

/**
 * Resident weight bytes on one device of stage `s`: the blocks of its layers (all local experts), plus the
 * embedding shard on stage 0 (when the table is on the device) and the LM-head shard (+ final norm) on the last
 * stage. A tied head shares the embedding table only when it is on the same device (pp = 1, and the embedding is not
 * off-device) AND stored in the same format (dtype and overhead); a tied head quantized differently is a second copy
 * and is counted in full. (Validation rejects an off-device embedding on a tied model; the on-device condition is a
 * defense.) With weights offloaded (`ctx.offload`), an offloaded MoE layer keeps `C = f·E_loc` experts, an
 * off-device table keeps its non-table part (design doc §2.7), and an off-device token embedding leaves stage 0.
 * Unit: bytes.
 */
export function stageWeightBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const pp = ctx.parallel.pp;
  const [start, end] = stageLayerRange(model.layers.length, pp, s);
  let layerBytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks)
      layerBytes += blockResidentWeightBytes(block, ctx, i);
  }
  const embedBytes = s === 0 ? embeddingResidentWeightBytes(model, ctx) : 0;
  let headBytes = s === pp - 1 ? lmHeadResidentWeightBytes(model, ctx) : 0;
  if (
    model.tieEmbeddings &&
    pp === 1 &&
    ctx.offload?.embeddings === undefined &&
    sameStoredFormat(embeddingFormat(ctx), lmHeadFormat(ctx))
  ) {
    // The shared matrix is already counted by the embedding (its ceil(vocab/tp) rows hold the head's
    // ceil(vocab/G) shard: G = tp, or G = ep ≥ tp = 1 with denseTpGroup 'ep'); keep only the final-norm part.
    const headMatrixBytes = matrixStoredBytes(
      lmHeadFormat(ctx),
      lmHeadVocabShard(model, ctx),
      model.hidden,
    );
    headBytes -= headMatrixBytes;
  }
  // Formula: W_s = Σ_{blocks of s} W_block + [s = 0 ∧ emb on device] W_emb + [s = pp−1] W_head
  //          (tied, pp = 1, emb on device, same format: matrix once)
  // See docs/simulation-model.md §8 ("Memory fit"), §12 ("Token embedding off-device")
  return layerBytes + embedBytes + headBytes;
}

/**
 * Bytes of the drafter's own token-embedding copy on ONE device of the last stage (speculative decoding at pp > 1).
 * Unit: bytes.
 *
 * The drafter (MTP `embed`, block-drafter `noise_embed`) gathers rows of the target's token embedding on the last
 * stage. At pp = 1 the table is already on the device (0). At pp > 1 it lives on stage 0, so the last stage keeps a
 * copy of the shard `ceil(V/tp)·h·bpe(emb)`, as vLLM's EAGLE proposer does: it shares `embed_tokens` with the target
 * only when the PP world size is 1, else loads its own (vLLM v0.11.0 `vllm/v1/spec_decode/eagle.py: load_model`,
 * https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/spec_decode/eagle.py#L854-L881). Exception: a tied
 * LM head stored in the same format IS that matrix, and {@link stageWeightBytes} already counts it in full on the
 * last stage at pp > 1, so no copy is added (an upper-bound choice; vLLM loads a copy regardless).
 *   W_copy = [pp > 1 ∧ ¬(tied ∧ same format)] · ceil(V/tp)·h·bpe(emb)
 * See docs/simulation-model.md §13 "Pipeline parallelism".
 */
export function draftEmbeddingCopyBytes(model: ModelSpec, ctx: LoweringContext): number {
  if (ctx.parallel.pp === 1) return 0;
  const tiedSameFormat =
    model.tieEmbeddings && sameStoredFormat(embeddingFormat(ctx), lmHeadFormat(ctx));
  // Formula: W_copy = [pp > 1 ∧ ¬(tied ∧ same format)] · ceil(V/tp)·h·bpe(emb)
  return tiedSameFormat ? 0 : embeddingShardBytes(model, ctx);
}

/**
 * Per-sequence state bytes on one device of stage `s` for `contextTokens` of context: the FULL state, wherever it
 * lives (with KV-cache offload, also the off-device part). This is what a prefill→decode KV transfer moves
 * (`kvTransfer.ts`); the device fit uses {@link stageResidentSequenceStateBytesPerSeq}. Unit: bytes per sequence.
 */
export function stageSequenceStateBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  contextTokens: number,
): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks)
      bytes += blockSequenceStateBytes(block, ctx, contextTokens);
  }
  // Formula: state_s = Σ_{blocks of s} blockSequenceStateBytes(ctxTokens)
  // See docs/simulation-model.md §8 ("Memory fit")
  return bytes;
}

/**
 * Per-sequence state bytes RESIDENT on one device of stage `s` for `contextTokens` of context: the full state of
 * {@link stageSequenceStateBytesPerSeq} minus the off-device share `1 − f_dev` of each offloadable block's KV
 * (`lower/kvOffload.ts: kvOffloadableStateBytes`). Without KV-cache offload, exactly the full state.
 * Unit: bytes per sequence.
 */
export function stageResidentSequenceStateBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  contextTokens: number,
): number {
  const full = stageSequenceStateBytesPerSeq(model, ctx, s, contextTokens);
  const kv = ctx.offload?.kvCache;
  if (kv === undefined) return full;
  // Formula: state_dev,s = state_s − (1 − f_dev)·Σ_{blocks of s} S_off(ctxTokens)
  // See docs/simulation-model.md §8 ("Memory fit"), §12 ("KV cache off-device")
  return (
    full - (1 - kv.deviceFraction) * stageKvOffloadableBytesPerSeq(model, ctx, s, contextTokens)
  );
}

/** Σ over the blocks of stage `s` of their offloadable KV per sequence (`kvOffloadableStateBytes`). Unit: bytes. */
function stageKvOffloadableBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  contextTokens: number,
): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks)
      bytes += kvOffloadableStateBytes(block, ctx, contextTokens);
  }
  // Formula: S_off,s = Σ_{blocks of s} N_rank·kvB
  return bytes;
}

/**
 * Peak transient activation bytes of stage `s`: the largest activation read + write or explicit collective
 * workspace of any single op of that
 * stage over the given lowered phases (the inputs and outputs of the op that is running). Unit: bytes.
 */
export function stageActivationPeakBytes(lowered: LoweredModel[], s: number): number {
  let peak = 0;
  for (const model of lowered) {
    for (const layer of model.stages[s].layers) {
      for (const op of layer.ops) {
        const b = opBytes(op);
        // Formula: act_s = max_ops (activationRead + activationWrite, lookup exchange workspace)
        // See docs/simulation-model.md §8 ("Memory fit")
        peak = Math.max(
          peak,
          b.activationReadBytes + b.activationWriteBytes,
          op.kind === 'collective' ? (op.workspaceBytes ?? 0) : 0,
        );
      }
    }
  }
  return peak;
}

/**
 * Bytes of n-gram table shards resident on one device of stage `s` (included in {@link stageWeightBytes}; 0 for
 * off-device tables). Unit: bytes.
 */
export function stageLookupTableBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    if (tablePlacementAt(ctx, i) !== undefined) continue;
    for (const block of model.layers[i].blocks) {
      if (block.kind === 'ngram_embedding') bytes += ngramTableShardBytes(block, ctx);
    }
  }
  // Formula: W_tables,s = Σ_{on-device tables of s} ngramTableShardBytes(block, ctx)
  // The same ownership as lowering (TP×DP with sharing); docs §5 "Lookup tables across DP".
  return bytes;
}

/** The transfer ops of stage `s` of one lowered phase, in op order. */
function stageTransferOps(model: LoweredModel, s: number): TransferOp[] {
  const transfers: TransferOp[] = [];
  for (const layer of model.stages[s].layers) {
    for (const op of layer.ops) if (op.kind === 'transfer') transfers.push(op);
  }
  return transfers;
}

/**
 * One staging allocation of {@link stageStagingBytes}: the bytes a group's fetch holds at once, `bufferBytes`, split by
 * source (`bySource`: host RAM, the SSD, the remote KV pool), and its prefetch share q. Every weight-group transfer is
 * its own unit (one source). The KV-cache fetches of one layer from host RAM, the SSD and the pool feed the same
 * attention op, so all are resident together: they form ONE unit, their bytes summed. Unit: bytes.
 */
interface StagingUnit {
  bufferBytes: number;
  bySource: Record<KvTier, number>;
  prefetchedFraction: number;
}

/**
 * The staging units of stage `s` in one phase: each non-KV transfer op of `ops` alone, and per layer the sum of its
 * KV-cache fetch ops (`kvLayers`: the KV ops grouped by layer).
 */
function stagingUnits(
  ops: TransferOp[],
  kvLayers: TransferOp[][],
  bufferBytes: (op: TransferOp) => number,
): StagingUnit[] {
  const units: StagingUnit[] = ops
    .filter((op) => op.group !== 'kv_cache')
    .map((op) => {
      const buffer = bufferBytes(op);
      return {
        bufferBytes: buffer,
        bySource: {
          host_ram: op.source === 'host_ram' ? buffer : 0,
          nvme: op.source === 'nvme' ? buffer : 0,
          remote_pool: op.source === 'remote_pool' ? buffer : 0,
        },
        prefetchedFraction: op.prefetchedFraction,
      };
    });
  for (const layerOps of kvLayers) {
    if (layerOps.length === 0) continue;
    // Formula: buffer_kv,ℓ = Σ_{tiers t} useful_t,ℓ   (every tier's fetch of layer ℓ is resident together)
    const sumOf = (pick: (op: TransferOp) => boolean): number =>
      layerOps.filter(pick).reduce((sum, op) => sum + op.usefulBytes, 0);
    units.push({
      bufferBytes: sumOf(() => true),
      bySource: {
        host_ram: sumOf((op) => op.source === 'host_ram'),
        nvme: sumOf((op) => op.source === 'nvme'),
        remote_pool: sumOf((op) => op.source === 'remote_pool'),
      },
      prefetchedFraction: Math.max(...layerOps.map((op) => op.prefetchedFraction)),
    });
  }
  return units;
}

/** The KV-cache fetch ops of stage `s` of one lowered phase, grouped by layer. */
function stageKvFetchesByLayer(model: LoweredModel, s: number): TransferOp[][] {
  return model.stages[s].layers.map((layer) =>
    layer.ops.filter((op): op is TransferOp => op.kind === 'transfer' && op.group === 'kv_cache'),
  );
}

/**
 * Staging buffer of stage `s` for streamed weight groups (design doc §2.7): the largest single transfer of the stage
 * over the given phases, double-buffered when it is prefetched (`prefetchedFraction > 0`: one buffer fills while the
 * other is read), single otherwise.
 * A phase whose transfers prefetch from two or more tiers ({@link prefetchesFromMultipleTiers}: host RAM, the SSD, the
 * remote KV pool) has one stream per source in flight (docs/simulation-model.md §12 "Overlap across tiers"): one
 * buffer being read by the device plus one being filled per source, `max_all + Σ_t max_{t,q>0}`. With one source
 * this is today's 2·max.
 * A transfer's buffer is `Σ ceil(count)·bytesEach` for experts (capacity is not an average, so
 * expected misses round up) and its `usefulBytes` for table and token-embedding rows (the gathered rows, not the
 * tier's pages). A layer's KV-cache fetches from host RAM, the SSD and the pool are one buffer, `Σ_t useful_t` (all
 * feed the same attention op, so they are resident together; {@link StagingUnit}), whose per-source parts fill the
 * per-source terms above. Token-embedding transfers are taken from `embeddingLowered` (the decode step and the
 * LAST prefill chunk, as for the activation peak: chunked prefill stages one chunk's rows, not the whole prompt's),
 * every other transfer from `lowered`; phase i of one pairs with phase i of the other (same length), and
 * `embeddingLowered` defaults to `lowered`. 0 when nothing streams. Unit: bytes.
 */
export function stageStagingBytes(
  lowered: LoweredModel[],
  s: number,
  embeddingLowered: LoweredModel[] = lowered,
): number {
  if (embeddingLowered.length !== lowered.length) {
    throw new Error('stageStagingBytes: embeddingLowered must list the same phases as lowered');
  }
  // Formula: buffer = table / embedding rows, KV cache: usefulBytes; else Σ ceil(count_i)·bytes_i
  const bufferBytes = (op: TransferOp): number =>
    op.group === 'lookup_table' || op.group === 'token_embedding' || op.group === 'kv_cache'
      ? op.usefulBytes
      : op.reads.reduce((sum, r) => sum + Math.ceil(r.count) * r.bytesEach, 0);
  let largest = 0;
  for (let phase = 0; phase < lowered.length; phase++) {
    // The phase's transfers: every group but the token embedding from `lowered`, the token embedding from the
    // matching phase of `embeddingLowered` (same source and q, only R per chunk), so the multi-tier test below
    // agrees with the stage time's (`pipeline.ts: stageTransfer`).
    const transfers = [
      ...stageTransferOps(lowered[phase], s).filter((op) => op.group !== 'token_embedding'),
      ...stageTransferOps(embeddingLowered[phase], s).filter(
        (op) => op.group === 'token_embedding',
      ),
    ];
    const units = stagingUnits(transfers, stageKvFetchesByLayer(lowered[phase], s), bufferBytes);
    for (const unit of units) {
      // Formula: slots = q > 0 ? 2 : 1   (double buffer for a prefetched transfer)
      const slots = unit.prefetchedFraction > 0 ? 2 : 1;
      largest = Math.max(largest, slots * unit.bufferBytes);
    }
    if (prefetchesFromMultipleTiers(transfers)) {
      // Formula: staging = max_all buffer + Σ_t max_{t,q>0} buffer   (1 read + 1 fill per source t)
      let maxAll = 0;
      const maxPrefetched: Record<KvTier, number> = { host_ram: 0, nvme: 0, remote_pool: 0 };
      for (const unit of units) {
        maxAll = Math.max(maxAll, unit.bufferBytes);
        if (unit.prefetchedFraction <= 0) continue;
        for (const t of Object.keys(maxPrefetched) as KvTier[])
          maxPrefetched[t] = Math.max(maxPrefetched[t], unit.bySource[t]);
      }
      largest = Math.max(
        largest,
        maxAll + maxPrefetched.host_ram + maxPrefetched.nvme + maxPrefetched.remote_pool,
      );
    }
  }
  // Formula: staging_s = max_phase ( ≥ 2 tiers prefetched ? max_all + max_ram + max_ssd + max_pool
  //                                                        : max_transfer (q > 0 ? 2 : 1)·buffer )
  //          token-embedding buffers from the decode step and the LAST prefill chunk (R·rowBytes per chunk)
  // See docs/simulation-model.md §12 ("Memory tiers", "Overlap across tiers", "Token embedding off-device")
  return largest;
}

/** An all-zero {@link TierBytes}. */
function emptyTierBytes(): TierBytes {
  return { lookupTables: 0, routedExperts: 0, draftExperts: 0, tokenEmbeddings: 0, kvCache: 0 };
}

/** Sum of the categories of a {@link TierBytes}. Unit: bytes. */
export function tierBytesTotal(b: TierBytes): number {
  return b.lookupTables + b.routedExperts + b.draftExperts + b.tokenEmbeddings + b.kvCache;
}

/**
 * Bytes one device of stage `s` places on each off-device tier, split by what they hold (design doc §2.7): the FULL
 * groups, since the device cache is a copy: all local experts of each offloaded MoE layer (`routedExperts`), each
 * off-device table shard (`lookupTables`); plus, for
 * an NVMe table with `scaleLayout 'pinned_host'`, its scales (`ceil(rows/tp)·scaleBytes`) on host RAM
 * (`lookupTables`) — on dedicated memory only: on unified memory (`hostRamIsDevice`) the scales are device bytes counted
 * in {@link stageWeightBytes}, and a `host_ram` group is an error (invariant). The draft's own experts are
 * `draftExperts`. Stage 0 with the token embedding off-device adds its shard `ceil(vocab/tp)·h·bpe(emb)` to its tier
 * (`tokenEmbeddings`). With KV-cache offload, `kv` gives the live sequences and their context: each tier t (host RAM,
 * SSD, remote pool) holds `B_r·f_t·S_off,s` (`kvCache`; f_ram = (1 − f_dev)·ramShare, f_ssd = (1 − f_dev)·nvmeShare,
 * f_pool = (1 − f_dev)·remotePoolShare); absent `kv` = 0. Only the KV cache may land on the remote pool.
 * Unit: bytes per device.
 */
export function stageTierBreakdown(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  kv?: { sequencesPerReplica: number; contextTokens: number },
): Record<KvTier, TierBytes> {
  const bytes: Record<KvTier, TierBytes> = {
    host_ram: emptyTierBytes(),
    nvme: emptyTierBytes(),
    remote_pool: emptyTierBytes(),
  };
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  for (let i = start; i < end; i++) {
    const layer = model.layers[i];
    const experts = expertPlacementAt(ctx, i);
    const tables = tablePlacementAt(ctx, i);
    for (const block of layer.blocks) {
      if (block.kind === 'moe' && experts !== undefined) {
        // Formula: tier += E_loc·W_expert
        bytes[experts.tier].routedExperts += moeLocalExpertBytes(block, ctx);
      } else if (block.kind === 'ngram_embedding' && tables !== undefined) {
        const shard = ngramTableShardBytes(block, ctx);
        const { scaleBytes } = tableRowBytes(lookupTableFormat(ctx), block.rowDim);
        if (tables.scaleLayout === 'pinned_host' && scaleBytes > 0) {
          // Formula: nvme += shard − scales;  host_ram += ceil(rows/tp)·scaleBytes   (dedicated memory)
          // Formula: nvme += shard − scales;  the scales are device bytes             (unified: see stageWeightBytes)
          const scales = Math.ceil(block.tableRows / ctx.parallel.tp) * scaleBytes;
          bytes[tables.tier].lookupTables += shard - scales;
          if (ctx.offload?.hostRamIsDevice !== true) bytes.host_ram.lookupTables += scales;
        } else {
          // Formula: tier += ceil(rows/tp)·d·bpe(lookup)
          bytes[tables.tier].lookupTables += shard;
        }
      }
    }
  }
  const embeddings = ctx.offload?.embeddings;
  if (s === 0 && embeddings !== undefined) {
    // Formula: tier += ceil(vocab/tp)·h·bpe(emb)   (stage 0, token embedding off-device)
    // See docs/simulation-model.md §12 ("Token embedding off-device")
    bytes[embeddings.tier].tokenEmbeddings += embeddingShardBytes(model, ctx);
  }
  // Spec-dec with includeDraft: the drafter's own MoE experts (last stage) live on the tier of the one rule that sets
  // includeDraft (design §3.8).
  const draftExperts = draftExpertPlacement(ctx.offload);
  if (s === ctx.parallel.pp - 1 && ctx.speculative !== undefined && draftExperts !== undefined) {
    for (const layer of draftLayerSpecs(ctx.speculative.draft)) {
      for (const block of layer.blocks) {
        // Formula: tier += E_loc·W_expert per drafter MoE block
        if (block.kind === 'moe')
          bytes[draftExperts.tier].draftExperts += moeLocalExpertBytes(block, ctx);
      }
    }
  }
  const kvPlacement = ctx.offload?.kvCache;
  if (kv !== undefined && kvPlacement !== undefined && kv.sequencesPerReplica > 0) {
    const offBytes = stageKvOffloadableBytesPerSeq(model, ctx, s, kv.contextTokens);
    const off = 1 - kvPlacement.deviceFraction;
    // Formula: tier_t += B_r·f_t·S_off,s,  f_ram = (1 − f_dev)·ramShare,  f_ssd = (1 − f_dev)·nvmeShare,
    //          f_pool = (1 − f_dev)·remotePoolShare
    // See docs/simulation-model.md §12 ("KV cache off-device", "Remote KV pool")
    bytes.host_ram.kvCache += kv.sequencesPerReplica * off * kvPlacement.ramShare * offBytes;
    bytes.nvme.kvCache += kv.sequencesPerReplica * off * kvPlacement.nvmeShare * offBytes;
    bytes.remote_pool.kvCache +=
      kv.sequencesPerReplica * off * kvPlacement.remotePoolShare * offBytes;
  }
  // Invariant: on unified memory host RAM is the device pool, so no group may land on a `host_ram` tier.
  if (ctx.offload?.hostRamIsDevice === true && tierBytesTotal(bytes.host_ram) > 0) {
    throw new Error('memory: invariant violated — a host_ram group on unified memory');
  }
  return bytes;
}

/**
 * Bytes one device of stage `s` places on each off-device tier: {@link stageTierBreakdown} summed over its
 * categories (`kv`: the live sequences for the KV-cache share, as there). Unit: bytes per device.
 */
export function stageTierBytes(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  kv?: { sequencesPerReplica: number; contextTokens: number },
): Record<KvTier, number> {
  const b = stageTierBreakdown(model, ctx, s, kv);
  return {
    host_ram: tierBytesTotal(b.host_ram),
    nvme: tierBytesTotal(b.nvme),
    remote_pool: tierBytesTotal(b.remote_pool),
  };
}

/** The drafter's decoder layers given as `LayerSpec`s: MTP modules, or a DeepSeek-V4-style drafter's `moeLayers`. */
function draftLayerSpecs(draft: DraftSpec): LayerSpec[] {
  return draft.kind === 'mtp' ? draft.layers : (draft.moeLayers ?? []);
}

/**
 * Recurrent and conv state of one sequence on stage `s` that a speculative verify snapshots per position (design doc
 * §3.3): the context-independent state of the blocks whose lowering writes `stateSnapshots` copies (linear attention
 * recurrent + conv, short convs, n-gram convs, GQA K/V convs) = their `blockSequenceStateBytes` at context 0. KV,
 * MLA and CSA caches are not snapshotted (rejected rows are overwritten). Unit: bytes per sequence.
 */
export function stageSnapshotStateBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks) {
      const snapshotted =
        block.kind === 'linear_attention' ||
        block.kind === 'short_conv' ||
        block.kind === 'ngram_embedding' ||
        block.kind === 'gqa_attention';
      // Formula: snap_s = Σ_{state blocks of s} blockSequenceStateBytes(ctx = 0)
      // See docs/simulation-model.md §13 ("Draft memory")
      if (snapshotted) bytes += blockSequenceStateBytes(block, ctx, 0);
    }
  }
  return bytes;
}

/**
 * The node's tier bytes by category, per node (a device of rank r sits on stage `floor(r/(tp·dp))` and node
 * `floor(r/devicesPerNode)`, docs §5 "Rank layout"): `perStage[s][tier]` summed over the node's devices.
 */
function nodeTierBreakdowns(
  perStage: Record<KvTier, TierBytes>[],
  tier: KvTier,
  parallel: { tp: number; dp: number; pp: number },
  devicesPerNode: number,
): TierBytes[] {
  const { tp, dp, pp } = parallel;
  const ranks = tp * dp * pp;
  const nodes = Math.ceil(ranks / devicesPerNode);
  const out: TierBytes[] = [];
  for (let node = 0; node < nodes; node++) {
    const nodeBreakdown = emptyTierBytes();
    for (let r = node * devicesPerNode; r < Math.min(ranks, (node + 1) * devicesPerNode); r++) {
      // Formula: bytes_node = Σ_{ranks r of the node} tierBytes(stage(r)),  stage(r) = floor(r / (tp·dp))
      const device = perStage[Math.floor(r / (tp * dp))][tier];
      for (const k of Object.keys(nodeBreakdown) as (keyof TierBytes)[])
        nodeBreakdown[k] += device[k];
    }
    out.push(nodeBreakdown);
  }
  return out;
}

/**
 * The tier's bytes by category per unit of its capacity: host RAM and the SSD are per node ({@link nodeTierBreakdowns},
 * one entry per node); the remote KV pool is one pool for the whole deployment, so its one entry sums every node.
 */
function tierScopeBreakdowns(
  perStage: Record<KvTier, TierBytes>[],
  tier: KvTier,
  parallel: { tp: number; dp: number; pp: number },
  devicesPerNode: number,
): TierBytes[] {
  const perNode = nodeTierBreakdowns(perStage, tier, parallel, devicesPerNode);
  if (tier !== 'remote_pool') return perNode;
  // Formula: bytes_pool = Σ_{nodes} bytes_node   (one pool shared by every device of the deployment)
  // See docs/simulation-model.md §12 ("Remote KV pool")
  const whole = emptyTierBytes();
  for (const node of perNode)
    for (const k of Object.keys(whole) as (keyof TierBytes)[]) whole[k] += node[k];
  return [whole];
}

/**
 * Capacity of a tier: the node's host RAM or SSD (`0` when the host resolved none), the remote pool's stated capacity
 * (`Infinity` when unlimited or not resolved). Unit: bytes.
 */
function tierCapacityBytes(tier: KvTier, host: ResolvedHost): number {
  switch (tier) {
    case 'host_ram':
      return host.ram?.capacityBytes ?? 0;
    case 'nvme':
      return host.ssd?.capacityBytes ?? 0;
    case 'remote_pool':
      return host.remotePool?.capacityBytes ?? Infinity;
  }
}

/** The tiers of the memory report, in display order. */
const REPORT_TIERS: readonly KvTier[] = ['host_ram', 'nvme', 'remote_pool'];

/**
 * Off-device tier entries of the report (design doc §2.7): per node, `bytes = Σ over the node's devices` of their
 * {@link stageTierBytes} (a device of rank r sits on stage `floor(r/(tp·dp))` and node `floor(r/devicesPerNode)`,
 * docs §5 "Rank layout"), taking the fullest node; the remote KV pool over every node ({@link tierScopeBreakdowns});
 * against the tier's capacity ({@link tierCapacityBytes}). One entry per tier with bytes.
 * `kv`: the live sequences and context for the KV-cache share (absent = none counted).
 * On unified memory no `'host_ram'` group reaches here (it is device memory); {@link stageTierBreakdown} enforces it.
 */
export function tierMemory(
  model: ModelSpec,
  ctx: LoweringContext,
  devicesPerNode: number,
  host: ResolvedHost,
  kv?: { sequencesPerReplica: number; contextTokens: number },
): TierMemory[] {
  const perStage = Array.from({ length: ctx.parallel.pp }, (_, s) =>
    stageTierBreakdown(model, ctx, s, kv),
  );
  const tiers: TierMemory[] = [];
  for (const tier of REPORT_TIERS) {
    let bytes = 0;
    let breakdown = emptyTierBytes();
    for (const nodeBreakdown of tierScopeBreakdowns(perStage, tier, ctx.parallel, devicesPerNode)) {
      const nodeBytes = tierBytesTotal(nodeBreakdown);
      // The fullest node: its breakdown is the one reported, so the parts always sum to `bytes`.
      if (nodeBytes > bytes) {
        bytes = nodeBytes;
        breakdown = nodeBreakdown;
      }
    }
    if (bytes === 0) continue;
    const capacityBytes = tierCapacityBytes(tier, host);
    tiers.push({ tier, bytes, breakdown, capacityBytes, fits: bytes <= capacityBytes });
  }
  return tiers;
}

/**
 * The most sequences per replica the off-device tiers can hold with KV-cache offload: per tier t and node (the remote
 * pool: the whole deployment), the capacity left after the static groups (experts, tables, embeddings) over the
 * node's per-sequence KV share, `floor((cap_t − static_t,node) / Σ_{node devices} f_t·S_off,s)`, the minimum over
 * tiers and nodes (0 when the static groups alone overflow). Infinity when no tier holds KV, or only a pool without a
 * stated capacity. Count.
 */
export function tierMaxSequencesPerReplica(
  model: ModelSpec,
  ctx: LoweringContext,
  devicesPerNode: number,
  host: ResolvedHost,
  contextTokens: number,
): number {
  if (ctx.offload?.kvCache === undefined) return Infinity;
  const pp = ctx.parallel.pp;
  const staticStages = Array.from({ length: pp }, (_, s) => stageTierBreakdown(model, ctx, s));
  const perSeqStages = Array.from({ length: pp }, (_, s) =>
    stageTierBreakdown(model, ctx, s, { sequencesPerReplica: 1, contextTokens }),
  );
  let maxSeq = Infinity;
  for (const tier of REPORT_TIERS) {
    const capacityBytes = tierCapacityBytes(tier, host);
    const statics = tierScopeBreakdowns(staticStages, tier, ctx.parallel, devicesPerNode);
    const perSeqs = tierScopeBreakdowns(perSeqStages, tier, ctx.parallel, devicesPerNode);
    for (let node = 0; node < perSeqs.length; node++) {
      const perSeqBytes = perSeqs[node].kvCache;
      if (!(perSeqBytes > 0)) continue;
      // Formula: maxSeq_t = floor((cap_t − static_t,node) / Σ_{node devices} f_t·S_off,s)
      // See docs/simulation-model.md §12 ("KV cache off-device")
      const freeBytes = capacityBytes - tierBytesTotal(statics[node]);
      maxSeq = Math.min(maxSeq, Math.max(0, Math.floor(freeBytes / perSeqBytes)));
    }
  }
  return maxSeq;
}

/**
 * The memory-fit report: per stage weights + B_r · state + activation peak + runtime reserve + staging against the
 * GPU-usable capacity `cap = f · capacity` ({@link resolveGpuUsableFraction}), and the most sequences per replica that
 * would fit at this context length: `floor(min_s (cap − W_s − act_s − reserve − staging_s) / state_s)` (Infinity
 * when no stage keeps per-sequence state; 0 when the weights alone do not fit). The activation peak is taken at the
 * evaluated batch, so this is an estimate. With weights offloaded, the off-device tiers are checked too and enter
 * `fits`. With KV-cache offload, `state_s` is the resident state only
 * ({@link stageResidentSequenceStateBytesPerSeq}), the tiers hold the rest, and the tiers' own sequence bound
 * ({@link tierMaxSequencesPerReplica}) caps `maxSequencesPerReplica`.
 */
export function memoryReport(input: MemoryInput): MemoryReport {
  const { model, ctx } = input;
  const gpuUsableFraction = input.gpuUsableFraction ?? 1;
  const physicalCapacityBytes = input.capacityBytes;
  // Formula: cap = f · capacity   (f = GPU-usable fraction; 1 on dedicated memory)
  // See docs/simulation-model.md §8 ("GPU-usable capacity")
  const capacityBytes = gpuUsableFraction * physicalCapacityBytes;
  const runtimeReserveBytes = input.runtimeReserveBytes ?? 0;
  const activationLowered = input.activationLowered ?? input.lowered;
  const perStage: DeviceMemory[] = [];
  let maxSequencesPerReplica = Infinity;
  for (let s = 0; s < ctx.parallel.pp; s++) {
    const weightBytes = stageWeightBytes(model, ctx, s);
    const lookupTableBytes = stageLookupTableBytes(model, ctx, s);
    // Resident state only: with KV-cache offload the off-device KV share is on the tiers (equal to the full state
    // without it).
    const statePerSeqBytes = stageResidentSequenceStateBytesPerSeq(
      model,
      ctx,
      s,
      input.contextTokens,
    );
    const activationPeakBytes = stageActivationPeakBytes(activationLowered, s);
    // Deployment terms (design doc §2.7, §3.3): 0 with nothing offloaded / spec-dec off.
    // Token-embedding rows are staged per prefill chunk (the activation phases), other groups per costed phase.
    const stagingBytes = stageStagingBytes(input.lowered, s, activationLowered);
    // Spec-dec (last stage): draft weights (shared embedding / head excluded) + the pp > 1 embedding copy + B_r · draft
    // KV/state at the context; every stage: the verify's (k + 1) per-position state snapshots × B_r.
    const spec = ctx.speculative;
    let draftBytes = 0;
    let draftStatePerSeqBytes = 0;
    let snapshotPerSeqBytes = 0;
    if (spec !== undefined) {
      if (s === ctx.parallel.pp - 1) {
        draftStatePerSeqBytes = draftSequenceStateBytes(spec.draft, ctx, input.contextTokens);
        // Formula: draft_s = W_draft + W_copy + B_r·S_draft(ctx)   (W_copy = 0 at pp = 1)
        // See docs/simulation-model.md §13 ("Draft memory", "Pipeline parallelism")
        draftBytes =
          draftResidentWeightBytes(model, spec.draft, ctx) +
          draftEmbeddingCopyBytes(model, ctx) +
          input.sequencesPerReplica * draftStatePerSeqBytes;
      }
      // Formula: specTransient_s = B_r·(k + 1)·snap_s
      snapshotPerSeqBytes = spec.stateSnapshots * stageSnapshotStateBytesPerSeq(model, ctx, s);
    }
    const speculativeTransientBytes = input.sequencesPerReplica * snapshotPerSeqBytes;
    // Formula: total_s = W_s + B_r·state_s + act_s + reserve + staging_s + draft_s + specTransient_s;  fits = total_s ≤ cap
    // See docs/simulation-model.md §8 ("Memory fit"), §12 ("Memory tiers"), §13 ("Draft memory")
    const sequenceStateBytes = input.sequencesPerReplica * statePerSeqBytes;
    const totalBytes =
      weightBytes +
      sequenceStateBytes +
      activationPeakBytes +
      runtimeReserveBytes +
      stagingBytes +
      draftBytes +
      speculativeTransientBytes;
    perStage.push({
      stage: s,
      weightBytes,
      lookupTableBytes,
      sequenceStateBytes,
      activationPeakBytes,
      stagingBytes,
      draftBytes,
      speculativeTransientBytes,
      runtimeReserveBytes,
      totalBytes,
      capacityBytes,
      physicalCapacityBytes,
      fits: totalBytes <= capacityBytes,
    });
    // Per-sequence bytes that grow with B_r, and the fixed bytes besides them (spec-dec terms 0 when off).
    const perSeqBytes =
      spec === undefined
        ? statePerSeqBytes
        : statePerSeqBytes + draftStatePerSeqBytes + snapshotPerSeqBytes;
    const fixedDraftBytes = draftBytes - input.sequencesPerReplica * draftStatePerSeqBytes;
    if (perSeqBytes > 0) {
      // Formula: maxSeq = floor(min_s (cap − W_s − act_s − reserve − staging_s − W_draft) / (state_s + S_draft + (k+1)·snap_s))
      // See docs/simulation-model.md §8 ("Memory fit"), §13 ("Draft memory")
      const fixedBytes = weightBytes + activationPeakBytes + runtimeReserveBytes + stagingBytes;
      const freeBytes =
        spec === undefined
          ? capacityBytes - fixedBytes
          : capacityBytes - fixedBytes - fixedDraftBytes;
      maxSequencesPerReplica = Math.min(
        maxSequencesPerReplica,
        Math.max(0, Math.floor(freeBytes / perSeqBytes)),
      );
    } else if (
      weightBytes + activationPeakBytes + runtimeReserveBytes + stagingBytes + fixedDraftBytes >
      capacityBytes
    ) {
      maxSequencesPerReplica = 0;
    }
  }
  const devicesPerNode = input.devicesPerNode ?? 1;
  const kv = { sequencesPerReplica: input.sequencesPerReplica, contextTokens: input.contextTokens };
  const tiers =
    input.host === undefined ? [] : tierMemory(model, ctx, devicesPerNode, input.host, kv);
  if (input.host !== undefined && ctx.offload?.kvCache !== undefined) {
    // Formula: maxSeq = min(device bound, per tier floor((cap_t − static_t,node) / Σ_{node devices} perSeq_t))
    // See docs/simulation-model.md §12 ("KV cache off-device")
    maxSequencesPerReplica = Math.min(
      maxSequencesPerReplica,
      tierMaxSequencesPerReplica(model, ctx, devicesPerNode, input.host, input.contextTokens),
    );
  }
  // Formula: fits = ∀ s fits_s ∧ ∀ tiers (bytes ≤ capacity)
  const fits = perStage.every((d) => d.fits) && tiers.every((t) => t.fits);
  return {
    kind: input.kind,
    gpuUsableFraction,
    perStage,
    fits,
    maxSequencesPerReplica,
    tiers,
  };
}

/** Bytes as decimal GB with one decimal, for warnings (`96.0 GB`). */
function gbText(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * Warning lines of a unified-memory report: the GPU-usable share the fit used, and, for a stage that overflows that
 * share but not the installed memory, that it fits only with a raised OS limit ("fits by default" vs "fits with
 * override", docs §8 "GPU-usable capacity").
 */
export function unifiedMemoryWarnings(report: MemoryReport): string[] {
  const first = report.perStage[0];
  if (first === undefined) return [];
  const percent = Math.round(report.gpuUsableFraction * 1000) / 10;
  const lines = [
    `memory: unified memory — the GPU may use ${percent}% of it (${gbText(first.capacityBytes)} of ` +
      `${gbText(first.physicalCapacityBytes)}); the rest stays with the OS and CPU`,
  ];
  const overLimit = report.perStage.find((d) => !d.fits && d.totalBytes <= d.physicalCapacityBytes);
  if (overLimit !== undefined) {
    lines.push(
      `memory: needs ${gbText(overLimit.totalBytes)}, above the GPU limit of ${gbText(overLimit.capacityBytes)}; ` +
        'it fits only with a raised limit (macOS iogpu.wired_limit_mb, Linux ttm.pages_limit)',
    );
  }
  return lines;
}
