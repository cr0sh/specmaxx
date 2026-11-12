/**
 * Input validation for `simulate`: parallel degrees and workload are positive integers, the batch divides
 * into DP replicas, PP fits the layers, every block shards evenly over TP (heads, widths) and EP (MoE modes),
 * the parallelism fits the topology and its links, the effective device (after modifiers, or the base) and the
 * topology pass hardware validation (incl. positive rates), the modifiers are valid, and the engine and memory
 * capacity selections exist, and the model spec is structurally valid (`modelSpecErrors`: cross-layer references,
 * KDA fields, residual-stream widths), the decode context-parallel degree fits TP and the model ({@link dcpErrors};
 * its per-block rules come from lowering, through {@link blockShardingErrors}), and the deployment knobs (offload
 * placement, speculative decoding) are consistent with the model and hardware ({@link deploymentErrors}).
 * See docs/simulation-model.md §5 (Parallelism) and §4 ("Device variants and modifiers"), design doc §2 and §3.
 */

import {
  validateDevice,
  validateDeviceModifiers,
  validateHost,
  validateLinkModifiers,
  validateTopology,
} from '../hw/validate';
import { applyDeviceModifiers, applyLinkModifiers } from '../hw/modifiers';
import { deviceHostLink } from '../hw/presets/links';
import { A2A_FORMATS, isA2aFormat } from '../ir/a2aFormat';
import type { ModelSpec } from '../ir/model';
import { modelSpecErrors } from '../ir/validateModel';
import { lowerBlock, type LoweringContext } from '../lower/lower';
import {
  DCP_BLOCK_DRAFTER_ERROR,
  dcpDivideTpError,
  dcpHasCandidateBlock,
  dcpNoBlockError,
} from '../lower/dcp';
import { modelHasOffloadableKv } from '../lower/kvOffload';
import { moeParallelErrors } from '../lower/moe';
import {
  checkPlacement,
  expertTpCollectivePath,
  lookupTableCollectivePaths,
  lookupTablesUsePeerReads,
} from './placement';
import {
  expertRules,
  kvCacheMovesAnything,
  lookupTableConfig,
  lookupTableRules,
} from './offloadRules';
import type { KvCacheOffloadConfig, LayerRange, SimInput } from './types';

/**
 * The exact error for speculative decoding with pipeline parallelism and the token embedding off the device
 * (`offload.embeddings` on host RAM or NVMe). At pp > 1 the drafter on the last stage gathers from its own copy of the
 * table (docs/simulation-model.md §13 "Pipeline parallelism"); that stage may sit on another node, whose host does
 * not hold the off-device table, so the combination is not modelled.
 */
export const SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR =
  'speculative decoding with pp > 1 and the token embedding off-device is not supported';

/** The exact error for two-batch overlap without MoE expert parallelism across TP groups (docs §7 "Two-batch overlap"). */
export const TWO_BATCH_OVERLAP_MODE_B_ERROR =
  'deployment.twoBatchOverlap needs a MoE model with expert parallelism across TP groups (ep > tp)';

/** The exact error for two-batch overlap together with off-device weights (not modelled; docs §7 "Two-batch overlap"). */
export const TWO_BATCH_OVERLAP_OFFLOAD_ERROR =
  'deployment.twoBatchOverlap with weights offloaded is not modelled';

/**
 * Errors of `options.overlapByClass` (docs §7 "Stage time and overlap"): keys must be overlap classes (`tp`, `ep`,
 * `pp`), values finite numbers in [0, 1] (dimensionless fractions). Absent or `{}`: none.
 */
export function overlapByClassErrors(byClass: unknown): string[] {
  if (byClass === undefined) return [];
  if (typeof byClass !== 'object' || byClass === null || Array.isArray(byClass))
    return ['options.overlapByClass must be an object of fractions by class (tp, ep, pp)'];
  const errors: string[] = [];
  for (const [key, value] of Object.entries(byClass)) {
    if (key !== 'tp' && key !== 'ep' && key !== 'pp') {
      errors.push(`options.overlapByClass.${key}: unknown overlap class (expected tp, ep or pp)`);
    } else if (
      value !== undefined &&
      !(typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1)
    ) {
      errors.push(
        `options.overlapByClass.${key} (${String(value)}) must be a finite number in [0, 1]`,
      );
    }
  }
  return errors;
}

/**
 * The exact error for an off-device token embedding (`deployment.offload.embeddings` on host RAM or NVMe) on a model
 * with tied embeddings: the table is also the LM head's weight, read whole every step, so it stays on the device
 * (docs/simulation-model.md §12 "Token embedding off-device").
 */
export const TIED_EMBEDDINGS_OFFLOAD_ERROR =
  'deployment.offload.embeddings: the model ties its token embedding to the LM head (tie_word_embeddings), so the table stays on the device';

/** Largest draft length k accepted (design doc §3.5). Count. */
export const MAX_DRAFT_LENGTH = 16;

/**
 * The exact error for `denseTpGroup: 'ep'` with attention TP > 1: dense TP over the EP group is modelled only at
 * tp = 1 (docs/simulation-model.md §5 "DP attention + EP", §9).
 */
export const DENSE_TP_EP_NEEDS_TP1_ERROR = "parallel.denseTpGroup 'ep' needs tp = 1";

/**
 * The exact error for `denseTpGroup: 'ep'` (ep > 1) with speculative decoding: the draft heads would need the same
 * group-wide sharding, which is not modelled (docs/simulation-model.md §5 "DP attention + EP", §9).
 */
export const DENSE_TP_EP_SPEC_DEC_ERROR =
  "parallel.denseTpGroup 'ep' with speculative decoding is not supported yet";

/**
 * Errors of `ParallelConfig.denseTpGroup`: an unknown value, `'ep'` with tp ≠ 1, and `'ep'` over a real group
 * (ep > 1) with speculative decoding. The width rule (G divides every dense `mlp` intermediate) comes from
 * {@link blockShardingErrors}. Absent: none. See docs/simulation-model.md §5 "DP attention + EP".
 */
export function denseTpGroupErrors(input: Pick<SimInput, 'parallel' | 'deployment'>): string[] {
  const group = input.parallel.denseTpGroup;
  if (group === undefined || group === 'tp') return [];
  if (group !== 'ep') return [`parallel.denseTpGroup must be 'tp' or 'ep' (got ${String(group)})`];
  const errors: string[] = [];
  if (input.parallel.tp !== 1) errors.push(DENSE_TP_EP_NEEDS_TP1_ERROR);
  if (input.parallel.ep > 1 && input.deployment?.speculative !== undefined)
    errors.push(DENSE_TP_EP_SPEC_DEC_ERROR);
  return errors;
}

/**
 * Global errors of `ParallelConfig.dcp` (decode context parallelism, d): d must divide tp (the DCP group lies inside
 * one TP group), the model needs an attention block DCP applies to (MLA, CSA, or GQA without a sliding window;
 * unsupported ones report their own error), and a block drafter (DFlash 2, DSpark) is rejected; MTP is allowed. The
 * per-block rules (GQA `d | tp / kvHeads`, sparse / CSA / relative-bias / K/V-conv rejections) are thrown by lowering
 * and reported by {@link blockShardingErrors}. Absent, 1, or not an integer ≥ 1 (reported by `validateInput`): none.
 * Exported so the UI can attribute them to the DCP field. See docs/simulation-model.md §5 "Decode context
 * parallelism (DCP)".
 */
export function dcpErrors(input: Pick<SimInput, 'model' | 'parallel' | 'deployment'>): string[] {
  const d = input.parallel.dcp;
  if (d === undefined || !Number.isInteger(d) || d <= 1) return [];
  const errors: string[] = [];
  const tp = input.parallel.tp;
  if (tp % d !== 0) errors.push(dcpDivideTpError(d, tp));
  if (!dcpHasCandidateBlock(input.model)) errors.push(dcpNoBlockError(d));
  const spec = input.deployment?.speculative;
  if (spec !== undefined && spec.method !== 'mtp') errors.push(DCP_BLOCK_DRAFTER_ERROR);
  return errors;
}

/** Errors for a field that must be an integer ≥ 1. */
function positiveIntegerErrors(value: number, label: string): string[] {
  return Number.isInteger(value) && value >= 1
    ? []
    : [`${label} must be an integer ≥ 1 (got ${value})`];
}

/**
 * Sharding errors of every block (heads, KV heads, widths divisible by tp; MoE expert parallelism), found by
 * lowering each block once for a one-token step: the lowering rules throw on an uneven split. Each message is
 * prefixed with the layer name (`'<layer>: <rule message>'`); identical messages are reported once.
 *
 * Exported so the UI can attribute TP / EP errors to their fields (and list the valid degrees) without a full
 * {@link validateInput}, which calls it too. Only `model`, `quant` and `parallel` are read; the workload and
 * hardware do not affect sharding. Rules: docs/simulation-model.md §5 ("TP sharding", "KV-head replication",
 * "MoE modes A/B", and the per-block DCP rules of "Decode context parallelism (DCP)", whose messages contain
 * `dcp (d)`).
 */
export function blockShardingErrors(
  input: Pick<SimInput, 'model' | 'quant' | 'parallel'>,
): string[] {
  const ctx: LoweringContext = {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 2,
    quant: input.quant,
    parallel: input.parallel,
  };
  const errors = new Set<string>();
  for (const layer of input.model.layers) {
    for (const block of layer.blocks) {
      try {
        lowerBlock(block, ctx);
      } catch (e) {
        errors.add(`${layer.name}: ${(e as Error).message}`);
      }
    }
  }
  return [...errors];
}

/**
 * Check a simulation input. Returns human-readable error lines; empty = valid.
 * `simulate()` throws `new Error(errors.join('\n'))` if any.
 */
export function validateInput(input: SimInput): string[] {
  const { model, parallel, workload, hardware } = input;
  const errors: string[] = [];

  // Parallel degrees and workload.
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const)
    errors.push(...positiveIntegerErrors(parallel[key], `parallel.${key}`));
  if (parallel.dcp !== undefined)
    errors.push(...positiveIntegerErrors(parallel.dcp, 'parallel.dcp'));
  const rho = parallel.epImbalance;
  if (rho !== undefined && !(Number.isFinite(rho) && rho >= 1)) {
    errors.push(`parallel.epImbalance must be a finite number ≥ 1 (got ${rho})`);
  }
  const redundant = parallel.redundantExpertsPerRank;
  if (redundant !== undefined && !(Number.isSafeInteger(redundant) && redundant >= 0)) {
    errors.push(`parallel.redundantExpertsPerRank must be a whole number ≥ 0 (got ${redundant})`);
  }
  const scalesA2a = parallel.epImbalanceScalesAllToAll;
  if (scalesA2a !== undefined && typeof scalesA2a !== 'boolean') {
    errors.push(
      `parallel.epImbalanceScalesAllToAll must be true or false (got ${String(scalesA2a)})`,
    );
  }
  const cpp = parallel.chunkedPipelinePrefill;
  if (cpp !== undefined && typeof cpp !== 'boolean') {
    errors.push(`parallel.chunkedPipelinePrefill must be true or false (got ${String(cpp)})`);
  }
  for (const field of ['dispatchFormat', 'combineFormat'] as const) {
    const format = parallel[field];
    if (format !== undefined && !isA2aFormat(format)) {
      errors.push(
        `parallel.${field} must be one of ${A2A_FORMATS.join(', ')} (got ${String(format)})`,
      );
    }
  }
  const eta = parallel.allToAllEfficiency;
  if (eta !== undefined && !(Number.isFinite(eta) && eta > 0 && eta <= 1)) {
    errors.push(`parallel.allToAllEfficiency must be a number in (0, 1] (got ${eta})`);
  }
  errors.push(...positiveIntegerErrors(workload.batch, 'workload.batch'));
  errors.push(...positiveIntegerErrors(workload.promptLen, 'workload.promptLen'));
  errors.push(...positiveIntegerErrors(workload.decodeContextLen, 'workload.decodeContextLen'));
  if (workload.prefillChunkTokens !== undefined) {
    errors.push(
      ...positiveIntegerErrors(workload.prefillChunkTokens, 'workload.prefillChunkTokens'),
    );
  }
  const hit = workload.prefixCacheHitRatio;
  if (hit !== undefined && !(Number.isFinite(hit) && hit >= 0 && hit < 1)) {
    errors.push(`workload.prefixCacheHitRatio (${hit}) must be a number in [0, 1)`);
  }
  const usable = hardware.gpuUsableFraction;
  if (usable !== undefined && !(usable > 0 && usable <= 1)) {
    errors.push(`hardware.gpuUsableFraction (${usable}) must be in (0, 1]`);
  }
  const reserve = hardware.runtimeReserveBytes;
  if (reserve !== undefined && !(Number.isFinite(reserve) && reserve >= 0)) {
    errors.push(`hardware.runtimeReserveBytes (${reserve}) must be a finite number ≥ 0`);
  }
  errors.push(...overlapByClassErrors(input.options?.overlapByClass));
  const floor = input.options?.perOpLatencyFloorSec;
  if (floor !== undefined && !(Number.isFinite(floor) && floor >= 0)) {
    errors.push(`options.perOpLatencyFloorSec (${floor}) must be a finite number ≥ 0`);
  }
  const inNetwork: string | undefined = input.options?.inNetworkReduction;
  if (inNetwork !== undefined && inNetwork !== 'off' && inNetwork !== 'on') {
    errors.push(`options.inNetworkReduction ('${inNetwork}') must be 'off' or 'on'`);
  }
  const dispatchSec = input.overlays?.engineDispatch?.perKernelSec;
  if (dispatchSec !== undefined && !(Number.isFinite(dispatchSec) && dispatchSec >= 0)) {
    errors.push(
      `overlays.engineDispatch.perKernelSec (${dispatchSec}) must be a finite number ≥ 0`,
    );
  }
  if (errors.length > 0) return errors;
  if (workload.batch % parallel.dp !== 0) {
    errors.push(`workload.batch (${workload.batch}) must be divisible by dp (${parallel.dp})`);
  }
  if (parallel.pp > model.layers.length) {
    errors.push(
      `pp (${parallel.pp}) must not exceed the number of layers (${model.layers.length})`,
    );
  }
  const hasMoe = model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
  if (parallel.ep > 1 && !hasMoe)
    errors.push(`ep (${parallel.ep}) > 1 but the model has no MoE block`);
  errors.push(...denseTpGroupErrors(input));
  errors.push(...dcpErrors(input));
  errors.push(...modelSpecErrors(model));
  errors.push(...blockShardingErrors(input));

  // Hardware: the EFFECTIVE device (after modifiers, or the base itself) is always validated: physical sanity,
  // rates used as divisors > 0, every number sourced. Then the engine and capacity selection.
  const deviceModifiers = hardware.deviceModifiers ?? [];
  const deviceErrors =
    deviceModifiers.length > 0
      ? validateDeviceModifiers(hardware.device, deviceModifiers)
      : validateDevice(hardware.device);
  errors.push(...deviceErrors);
  if (deviceErrors.length === 0) {
    const device =
      deviceModifiers.length > 0
        ? applyDeviceModifiers(hardware.device, deviceModifiers)
        : hardware.device;
    if (
      hardware.engineId !== undefined &&
      !device.engines.some((e) => e.id === hardware.engineId)
    ) {
      errors.push(
        `hardware.engineId '${hardware.engineId}' is not an engine of ${device.id} (${device.engines.map((e) => e.id).join(', ')})`,
      );
    }
    if (
      hardware.memCapacityBytes !== undefined &&
      !device.memory.capacityOptionsBytes.includes(hardware.memCapacityBytes)
    ) {
      errors.push(
        `hardware.memCapacityBytes ${hardware.memCapacityBytes} is not one of ${device.id}'s capacity options ` +
          `(${device.memory.capacityOptionsBytes.join(', ')})`,
      );
    }
  }
  // The topology and its base links are always validated; modified links again after their modifiers.
  const topology = hardware.topology;
  errors.push(...validateTopology(topology));
  for (const level of ['intraNode', 'interNode'] as const) {
    const mods = hardware.linkModifiers?.[level] ?? [];
    if (mods.length === 0) continue;
    const link = topology[level];
    if (link === undefined)
      errors.push(`hardware.linkModifiers.${level}: topology ${topology.id} has no ${level} link`);
    else errors.push(...validateLinkModifiers(link, mods).map((e) => `${level}: ${e}`));
  }

  errors.push(...deploymentErrors(input));
  // Placement on the topology (world size, group spans, required links).
  const placementErrors = checkPlacement(parallel, topology).errors;
  errors.push(...placementErrors);
  // The expert-TP group (MoE expertOutputNorm with etp = tp/ep > 1) must have a modeled path on the topology.
  const usesExpertTp =
    parallel.tp % parallel.ep === 0 &&
    parallel.tp / parallel.ep > 1 &&
    model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe' && b.expertOutputNorm === true));
  if (usesExpertTp && placementErrors.length === 0) {
    try {
      expertTpCollectivePath(parallel, topology);
    } catch (e) {
      errors.push(`etp: ${(e as Error).message}`);
    }
  }
  if (
    lookupTableConfig(input.deployment?.offload).acrossDp === true &&
    parallel.dp > 1 &&
    errors.length === 0
  ) {
    const label = 'deployment.offload.lookupTables.acrossDp';
    try {
      // Use the same modified links as simulation; changing P2P can change the table ownership layout.
      const effectiveTopology = { ...topology };
      for (const level of ['intraNode', 'interNode'] as const) {
        const link = topology[level];
        const mods = hardware.linkModifiers?.[level] ?? [];
        if (link !== undefined && mods.length > 0)
          effectiveTopology[level] = applyLinkModifiers(link, mods);
      }
      lookupTableCollectivePaths(parallel, effectiveTopology);
      const peer = lookupTablesUsePeerReads(parallel, effectiveTopology);
      const hasHeadTables = model.layers.some((l) =>
        l.blocks.some((b) => b.kind === 'ngram_embedding' && b.headRows !== undefined),
      );
      if (hasHeadTables && parallel.tp * parallel.pp * parallel.dp > topology.devicesPerNode) {
        errors.push(
          `${label}: Engram currently requires all replicas in one node; node-local DP subgroups and cross-node peer-HBM domains are not modeled`,
        );
      }
      if (!peer) {
        const shards = parallel.tp * parallel.dp;
        for (const layer of model.layers)
          for (const block of layer.blocks) {
            if (block.kind !== 'ngram_embedding' || block.headRows === undefined) continue;
            const heads = block.rowsPerToken;
            if ((shards - 1) * Math.ceil(heads / shards) >= heads) {
              errors.push(
                `${label}: ${layer.name} would leave a rank without hash heads (${heads} heads over ${shards} shards)`,
              );
            }
          }
      }
    } catch (e) {
      errors.push(`${label}: ${(e as Error).message}`);
    }
  }
  return errors;
}

/** Errors of a `[from, to)` layer range: integers, non-empty, inside `[0, L)`. */
function layerRangeErrors(range: LayerRange, layerCount: number, label: string): string[] {
  const { from, to } = range;
  if (!Number.isInteger(from) || !Number.isInteger(to))
    return [`${label} [${from}, ${to}) must be integer layer indices`];
  if (from < 0 || to > layerCount || from >= to) {
    return [`${label} [${from}, ${to}) must be a non-empty range inside [0, ${layerCount})`];
  }
  return [];
}

/**
 * Errors for rules of one group whose layer ranges overlap (rules must cover disjoint ranges; a rule without `layers`
 * is reported elsewhere and skipped here). Names both rule indices: `<label>[i].layers [a, b) overlaps <label>[j].layers
 * [c, d)`. Rules are compared by start layer, so their list order does not matter.
 */
function overlapErrors(rules: readonly { layers?: LayerRange }[], label: string): string[] {
  const ranged = rules
    .flatMap((r, i) => (r.layers === undefined ? [] : [{ i, range: r.layers }]))
    .sort((a, b) => a.range.from - b.range.from);
  const errors: string[] = [];
  for (let k = 0; k < ranged.length; k++) {
    for (let m = k + 1; m < ranged.length; m++) {
      const a = ranged[k]!;
      const b = ranged[m]!;
      // Formula: [a.from, a.to) ∩ [b.from, b.to) ≠ ∅  ⇔  b.from < a.to   (b.from ≥ a.from after the sort)
      if (b.range.from < a.range.to) {
        const [x, y] = a.i < b.i ? [a, b] : [b, a];
        errors.push(
          `${label}[${x.i}].layers [${x.range.from}, ${x.range.to}) overlaps ${label}[${y.i}].layers [${y.range.from}, ${y.range.to})`,
        );
      }
    }
  }
  return errors;
}

/** True if `x` is a finite number in [lo, hi] (or [lo, hi) when `hiOpen`). */
function inRange(x: number, lo: number, hi: number, hiOpen = false): boolean {
  return Number.isFinite(x) && x >= lo && (hiOpen ? x < hi : x <= hi);
}

/**
 * The exact error for KV-cache offload on a model without offloadable KV (`lower/kvOffload.ts: kvOffloadableBlock`):
 * only sliding-window, recurrent, short-conv or CSA state, which stays on the device.
 */
export const KV_CACHE_NO_OFFLOADABLE_ERROR =
  'deployment.offload.kvCache: the model has no offloadable KV cache (only sliding-window, recurrent or CSA state)';

/**
 * Errors of `deployment.offload.kvCache` (docs/simulation-model.md §12 "KV cache off-device"): `deviceFraction`,
 * `prefixDeviceFraction` in [0, 1]; `hostRamFraction` in [0, 1 − deviceFraction]; `remotePool.fraction` in
 * [0, 1 − deviceFraction] with `hostRamFraction + remotePool.fraction ≤ 1 − deviceFraction`, a given pool link with a
 * positive finite bandwidth and a finite latency ≥ 0, a given pool capacity positive; a custom hit ratio `h` in [0, 1],
 * and `h = 0` when no KV is resident (`deviceFraction` 0, the expert rule); and, when it moves anything
 * (`movesAnything`), a model with offloadable KV ({@link KV_CACHE_NO_OFFLOADABLE_ERROR}). Every message names the
 * "KV cache".
 */
export function kvCacheErrors(
  kv: KvCacheOffloadConfig | undefined,
  model: ModelSpec,
  movesAnything: boolean,
): string[] {
  if (kv === undefined) return [];
  const errors: string[] = [];
  const label = 'deployment.offload.kvCache';
  const f = kv.deviceFraction;
  const fOk = Number.isFinite(f) && inRange(f, 0, 1);
  if (!fOk) {
    errors.push(
      `${label}.deviceFraction: the KV cache share on the device must be in [0, 1] (got ${f})`,
    );
  }
  const fRam = kv.hostRamFraction;
  if (fRam !== undefined && fOk && !(Number.isFinite(fRam) && inRange(fRam, 0, 1 - f + 1e-12))) {
    errors.push(
      `${label}.hostRamFraction: the KV cache share in host RAM must be in [0, 1 − deviceFraction] (got ${fRam})`,
    );
  }
  const pool = kv.remotePool;
  if (pool !== undefined) {
    const fPool = pool.fraction;
    if (fOk && !(Number.isFinite(fPool) && inRange(fPool, 0, 1 - f + 1e-12))) {
      errors.push(
        `${label}.remotePool.fraction: the KV cache share in the remote pool must be in [0, 1 − deviceFraction] (got ${fPool})`,
      );
    } else if (fOk && fRam !== undefined && Number.isFinite(fRam) && fRam + fPool > 1 - f + 1e-12) {
      errors.push(
        `${label}: the KV cache shares on the device, in host RAM and in the remote pool exceed 1 (${f} + ${fRam} + ${fPool})`,
      );
    }
    const link = pool.link;
    if (link !== undefined) {
      const bw = link.bandwidthPerDirectionBytesPerSec;
      if (!(Number.isFinite(bw) && bw > 0)) {
        errors.push(
          `${label}.remotePool.link.bandwidthPerDirectionBytesPerSec: the remote KV pool link bandwidth must be a positive number of B/s (got ${bw})`,
        );
      }
      if (!(Number.isFinite(link.latencySec) && link.latencySec >= 0)) {
        errors.push(
          `${label}.remotePool.link.latencySec: the remote KV pool link latency must be a finite number of seconds ≥ 0 (got ${link.latencySec})`,
        );
      }
    }
    const cap = pool.capacityBytes;
    if (cap !== undefined && !(Number.isFinite(cap) && cap > 0)) {
      errors.push(
        `${label}.remotePool.capacityBytes: the remote KV pool capacity must be a positive number of bytes, or absent for unlimited (got ${cap})`,
      );
    }
  }
  const fPre = kv.prefixDeviceFraction;
  if (fPre !== undefined && !(Number.isFinite(fPre) && inRange(fPre, 0, 1))) {
    errors.push(
      `${label}.prefixDeviceFraction: the cached prefix's KV cache share on the device must be in [0, 1] (got ${fPre})`,
    );
  }
  const hit = kv.hitRatio;
  if (hit !== undefined && hit.kind !== 'uniform' && hit.kind !== 'custom') {
    errors.push(`${label}.hitRatio: the KV cache hit model must be 'uniform' or 'custom'`);
  }
  if (hit?.kind === 'custom') {
    const h = hit.h;
    if (!(Number.isFinite(h) && inRange(h, 0, 1))) {
      errors.push(`${label}.hitRatio.h: the KV cache hit ratio must be in [0, 1] (got ${h})`);
    } else if (f === 0 && h > 0) {
      errors.push(
        `${label}.hitRatio: no KV cache is on the device (deviceFraction 0), h must be 0`,
      );
    }
  }
  if (movesAnything && !modelHasOffloadableKv(model)) errors.push(KV_CACHE_NO_OFFLOADABLE_ERROR);
  return errors;
}

/**
 * Errors of the deployment knobs (design doc §2, §3.6). Absent or `{}` deployment: none, and no other field is read.
 *
 * Offload: `mode 'cpu_compute'` is a designed hook, rejected for now (§2.8); `queueDepth` an integer ≥ 1; routed
 * experts need a MoE model, `residentFraction` in [0, 1], a custom `h` in [0, 1] and `h = 0` when `f = 0` (no expert
 * is resident); layer ranges non-empty inside `[0, L)`, disjoint within a group, and required in a list of two or more
 * rules; `includeDraft` on one expert rule at most; an off-device lookup-table tier needs n-gram tables, and
 * `scaleLayout 'pinned_host'` needs the NVMe tier; the removed `denseLayers` is rejected; `embeddings.tier` is
 * `'device'`, `'host_ram'` or `'nvme'`, and off the device only for an untied model
 * ({@link TIED_EMBEDDINGS_OFFLOAD_ERROR}). Any off-device group on a dedicated-memory device needs a host
 * link: `hardware.host.link`, the device's cited `hostLink`, or its PCIe `hostInterface`. `hardware.host`, when given, is validated.
 *
 * Two-batch overlap: needs a MoE model with ep > tp ({@link TWO_BATCH_OVERLAP_MODE_B_ERROR}) and nothing off the
 * device ({@link TWO_BATCH_OVERLAP_OFFLOAD_ERROR}).
 *
 * Speculative decoding: with pp > 1 the token embedding must stay on the device
 * ({@link SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR}); `draftLength` k an integer in
 * 1..16; `'mtp'` needs `model.draft.kind === 'mtp'`; a block method needs a `drafter` of the same method with
 * `k + 1 ≤ blockSize`; acceptance i.i.d. α in [0, 1), per-position α_j in [0, 1] (non-empty); the drafter's own MoE
 * blocks (MTP layers, a block drafter's `moeLayers`) follow the same expert-parallel rules as the target's
 * (`moeParallelErrors`: their expert count may differ, e.g. DeepSeek-V4.1 DSpark's 128 against 384).
 */
export function deploymentErrors(input: SimInput): string[] {
  const { model, parallel, hardware } = input;
  const errors: string[] = [];
  if (hardware.host !== undefined) errors.push(...validateHost(hardware.host, 'hardware.host'));
  const offload = input.deployment?.offload;
  // Anything placed off the device (routed experts, lookup tables, the token embedding, the KV cache); set in the
  // offload block.
  let offDevice = false;
  if (offload !== undefined) {
    const L = model.layers.length;
    if (offload.mode === 'cpu_compute') {
      errors.push("deployment.offload.mode 'cpu_compute' is not supported yet (only 'transfer')");
    }
    if (offload.queueDepth !== undefined) {
      errors.push(...positiveIntegerErrors(offload.queueDepth, 'deployment.offload.queueDepth'));
    }
    if (
      offload.ssdReadRate !== undefined &&
      offload.ssdReadRate !== 'datasheet' &&
      offload.ssdReadRate !== 'sustained'
    ) {
      errors.push("deployment.offload.ssdReadRate must be 'datasheet' or 'sustained'");
    }
    const p = offload.prefetchAccuracy;
    if (p !== undefined && !(Number.isFinite(p) && inRange(p, 0, 1))) {
      errors.push(`deployment.offload.prefetchAccuracy must be in [0, 1] (got ${p})`);
    }
    const experts = expertRules(offload);
    // The object form keeps its original message paths; a list names the rule: `routedExperts[i].…`.
    const expertLabel = (i: number): string =>
      Array.isArray(offload.routedExperts)
        ? `deployment.offload.routedExperts[${i}]`
        : 'deployment.offload.routedExperts';
    if (experts.length > 0) {
      const hasMoe = model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
      if (!hasMoe) errors.push('deployment.offload.routedExperts: the model has no MoE block');
    }
    experts.forEach((rule, i) => {
      const label = expertLabel(i);
      const f = rule.residentFraction;
      if (!inRange(f, 0, 1)) errors.push(`${label}.residentFraction must be in [0, 1] (got ${f})`);
      if (rule.hitRatio.kind === 'custom') {
        const h = rule.hitRatio.h;
        if (!inRange(h, 0, 1)) errors.push(`${label}.hitRatio.h must be in [0, 1] (got ${h})`);
        else if (f === 0 && h > 0) {
          errors.push(`${label}.hitRatio: no experts are resident, h must be 0`);
        }
      }
      if (rule.layers !== undefined) {
        errors.push(...layerRangeErrors(rule.layers, L, `${label}.layers`));
      } else if (experts.length > 1) {
        errors.push(`${label}.layers is required when there is more than one rule`);
      }
    });
    errors.push(...overlapErrors(experts, 'deployment.offload.routedExperts'));
    const drafting = experts.flatMap((rule, i) => (rule.includeDraft === true ? [i] : []));
    if (drafting.length > 1) {
      errors.push(
        `deployment.offload.routedExperts: includeDraft may be set on one rule only (rules ${drafting.join(', ')})`,
      );
    }
    // Whole-layer streaming (`denseLayers`) was removed (docs/simulation-model.md §12 "Placement"); an old input that
    // still carries it is rejected rather than silently simulated as on-device.
    if ('denseLayers' in offload) {
      errors.push(
        'deployment.offload.denseLayers was removed: whole-layer streaming is no longer modeled (offload MoE experts instead)',
      );
    }
    const tables = lookupTableRules(offload);
    const tableConfig = lookupTableConfig(offload);
    if (tableConfig.acrossDp !== undefined && typeof tableConfig.acrossDp !== 'boolean') {
      errors.push('deployment.offload.lookupTables.acrossDp must be a boolean');
    }
    if (tableConfig.acrossDp === true) {
      if (!model.layers.some((l) => l.blocks.some((b) => b.kind === 'ngram_embedding'))) {
        errors.push(
          'deployment.offload.lookupTables.acrossDp: the model has no n-gram lookup table',
        );
      }
      if (tables.some((rule) => rule.tier !== 'device')) {
        errors.push(
          'deployment.offload.lookupTables.acrossDp requires every lookup-table placement to be device',
        );
      }
    }
    // Preserve legacy error paths; the wrapper names its placements explicitly.
    const wrapped =
      offload.lookupTables !== undefined &&
      !Array.isArray(offload.lookupTables) &&
      !('tier' in offload.lookupTables);
    if (wrapped) {
      for (const key of Object.keys(tableConfig)) {
        if (key !== 'acrossDp' && key !== 'placements')
          errors.push(
            `deployment.offload.lookupTables.${key}: unknown key (expected acrossDp or placements)`,
          );
      }
    }
    const tablePrefix = wrapped
      ? 'deployment.offload.lookupTables.placements'
      : 'deployment.offload.lookupTables';
    // The object form keeps its original message paths; a list names the rule: `lookupTables[i].…`.
    const tableLabel = (i: number): string =>
      Array.isArray(tableConfig.placements) ? `${tablePrefix}[${i}]` : tablePrefix;
    const offDeviceTables = tables.some((rule) => rule.tier !== 'device');
    if (offDeviceTables) {
      const hasNgram = model.layers.some((l) => l.blocks.some((b) => b.kind === 'ngram_embedding'));
      if (!hasNgram)
        errors.push('deployment.offload.lookupTables: the model has no n-gram lookup table');
    }
    tables.forEach((rule, i) => {
      const label = tableLabel(i);
      if (rule.tier !== 'device' && rule.scaleLayout === 'pinned_host' && rule.tier !== 'nvme') {
        errors.push(`${label}.scaleLayout 'pinned_host' needs tier 'nvme'`);
      }
      if (rule.layers !== undefined) {
        errors.push(...layerRangeErrors(rule.layers, L, `${label}.layers`));
      } else if (tables.length > 1) {
        errors.push(`${label}.layers is required when there is more than one rule`);
      }
    });
    errors.push(...overlapErrors(tables, tablePrefix));
    // Token embedding (docs/simulation-model.md §12 "Token embedding off-device"): a tied table is also the LM head's
    // GEMM weight, read whole every step, so it must stay on the device; a duplicate copy is not modelled.
    const embeddingTier: unknown = offload.embeddings?.tier;
    if (
      offload.embeddings !== undefined &&
      embeddingTier !== 'device' &&
      embeddingTier !== 'host_ram' &&
      embeddingTier !== 'nvme'
    ) {
      errors.push(
        `deployment.offload.embeddings.tier must be 'device', 'host_ram' or 'nvme' (got ${JSON.stringify(embeddingTier)})`,
      );
    }
    const offDeviceEmbeddings = embeddingTier === 'host_ram' || embeddingTier === 'nvme';
    if (offDeviceEmbeddings && model.tieEmbeddings) {
      errors.push(TIED_EMBEDDINGS_OFFLOAD_ERROR);
    }
    const kvOffDevice = kvCacheMovesAnything(offload.kvCache, input.workload.prefixCacheHitRatio);
    errors.push(...kvCacheErrors(offload.kvCache, model, kvOffDevice));
    offDevice = experts.length > 0 || offDeviceTables || offDeviceEmbeddings || kvOffDevice;
    const device = hardware.device;
    if (
      offDevice &&
      device.memory.kind === 'dedicated' &&
      hardware.host?.link === undefined &&
      deviceHostLink(device) === undefined
    ) {
      errors.push(
        `deployment.offload: no host link known for ${device.id}; set hardware.host.link`,
      );
    }
  }
  // Two-batch overlap (docs §7 "Two-batch overlap"): it exists to hide the EP all-to-alls (MoE mode B, ep > tp), and
  // its interplay with offload transfers (staging, prefetch per half) is not modelled.
  if (input.deployment?.twoBatchOverlap === true) {
    const hasMoe = model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
    if (!hasMoe || !(parallel.ep > parallel.tp)) errors.push(TWO_BATCH_OVERLAP_MODE_B_ERROR);
    if (offDevice) errors.push(TWO_BATCH_OVERLAP_OFFLOAD_ERROR);
  }
  const spec = input.deployment?.speculative;
  if (spec !== undefined) {
    const embeddingTier: unknown = input.deployment?.offload?.embeddings?.tier;
    if (parallel.pp > 1 && (embeddingTier === 'host_ram' || embeddingTier === 'nvme'))
      errors.push(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR);
    const k = spec.draftLength;
    if (!Number.isInteger(k) || k < 1 || k > MAX_DRAFT_LENGTH) {
      errors.push(
        `deployment.speculative.draftLength must be an integer in 1..${MAX_DRAFT_LENGTH} (got ${k})`,
      );
    }
    if (spec.method === 'mtp') {
      if (model.draft?.kind !== 'mtp') {
        errors.push(
          "deployment.speculative.method 'mtp' needs a model with an MTP draft head (model.draft.kind 'mtp')",
        );
      }
    } else if (spec.drafter === undefined) {
      errors.push(`deployment.speculative.method '${spec.method}' needs a drafter`);
    } else {
      const drafter = spec.drafter.spec;
      if (drafter.method !== spec.method) {
        errors.push(
          `deployment.speculative.drafter is a ${drafter.method} drafter, not ${spec.method}`,
        );
      }
      if (k + 1 > drafter.blockSize) {
        errors.push(
          `deployment.speculative.draftLength ${k}: k + 1 must not exceed the drafter's blockSize (${drafter.blockSize})`,
        );
      }
    }
    // The drafter's MoE blocks lower with the same parallel config: each must satisfy moeParallelErrors (E % ep == 0,
    // and tp % ep == 0 or the mode-B rule), else simulate would throw while lowering the draft layers.
    const draftLayers =
      spec.method === 'mtp'
        ? model.draft?.kind === 'mtp'
          ? model.draft.layers
          : []
        : (spec.drafter?.spec.moeLayers ?? []);
    for (const layer of draftLayers) {
      for (const block of layer.blocks) {
        if (block.kind !== 'moe') continue;
        for (const e of moeParallelErrors(block, parallel))
          errors.push(`deployment.speculative drafter ${layer.name}: ${e}`);
      }
    }
    const acc = spec.acceptance;
    if (acc.kind === 'iid') {
      if (!inRange(acc.alpha, 0, 1, true)) {
        errors.push(`deployment.speculative.acceptance.alpha must be in [0, 1) (got ${acc.alpha})`);
      }
    } else if (acc.alphas.length === 0) {
      errors.push('deployment.speculative.acceptance.alphas must not be empty');
    } else if (!acc.alphas.every((a) => inRange(a, 0, 1))) {
      errors.push(
        `deployment.speculative.acceptance.alphas must each be in [0, 1] (got ${acc.alphas.join(', ')})`,
      );
    }
  }
  return errors;
}
