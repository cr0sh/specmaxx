/**
 * Per-collective overlap and two-batch overlap in the UI (docs/simulation-model.md §7 "Stage time and overlap",
 * "Two-batch overlap"): which overlap classes a run has, and the options / deployment flag as the engine sees them.
 *
 * The state keeps what the user chose (`SimOptions.overlapByClass`, `DeploymentChoice.twoBatchOverlap`) even while a
 * class or the two-batch switch does not apply (fewer devices, a dense model); the SimInput carries them only where
 * they apply, so a stale choice changes nothing (as the all-to-all knobs, `parallelism.ts: onModel`). Pure TypeScript.
 */

import type { OverlapClass, SimOptions } from '../../sim/engine/types';
import type { ModelSpec, ParallelConfig } from '../../sim/ir/model';
import type { TopologySpec } from '../../sim/hw/types';
import { lookupTablesUsePeerReads } from '../../sim/engine/placement';
import type { OffloadChoice } from './deployment';
import { hasMoe } from './deploymentInput';

/** Every overlap class, in display and URL order: TP/lookup exchanges, EP exchanges, PP send. */
export const OVERLAP_CLASSES: readonly OverlapClass[] = ['tp', 'ep', 'pp'];

/**
 * Whether shared lookup tables emit collectives on the resolved (modified) topology. PLE always exchanges rows;
 * Engram TP1 peer reads are gather-internal work, not a collective overlap class. Missing hardware means unresolved.
 * See docs/simulation-model.md §5 "Lookup tables across DP", §7 "Stage time and overlap".
 */
export function hasLookupCollectives(
  model: ModelSpec,
  parallel: ParallelConfig,
  offload: OffloadChoice | undefined,
  topology: TopologySpec | undefined,
): boolean {
  if (
    parallel.dp <= 1 ||
    offload?.tables?.acrossDp !== true ||
    offload.tables.tier !== undefined ||
    topology === undefined
  )
    return false;
  const peer = lookupTablesUsePeerReads(parallel, topology);
  return model.layers.some((l) =>
    l.blocks.some((b) => b.kind === 'ngram_embedding' && (!peer || b.headRows === undefined)),
  );
}

/**
 * The overlap classes that have communication in this run: `tp` with TP > 1 (TP, expert-TP and DCP collectives)
 * or lookup collectives ({@link hasLookupCollectives}),
 * `ep` on a MoE model with EP > TP (mode B: the all-to-alls and the DP attention + EP collectives), `pp` with PP > 1
 * (the send to the next stage). In {@link OVERLAP_CLASSES} order.
 */
export function overlapClassesOf(
  model: ModelSpec,
  parallel: ParallelConfig,
  lookupCollectives = false,
): OverlapClass[] {
  const present: Record<OverlapClass, boolean> = {
    tp: parallel.tp > 1 || lookupCollectives,
    ep: hasMoe(model) && parallel.ep > parallel.tp,
    pp: parallel.pp > 1,
  };
  return OVERLAP_CLASSES.filter((c) => present[c]);
}

/**
 * True when two-batch overlap applies: a MoE model in mode B (EP > TP, the layout with the expert all-to-all it
 * hides) with nothing offloaded (the engine does not model it with offload).
 */
export function twoBatchOverlapApplies(
  model: ModelSpec,
  parallel: ParallelConfig,
  offloads: boolean,
): boolean {
  return hasMoe(model) && parallel.ep > parallel.tp && !offloads;
}

/**
 * `options` as the engine sees them: `overlapByClass` keeps only the classes the run has
 * ({@link overlapClassesOf}) and is removed when none is left, so absent overrides give the input from before.
 */
export function effectiveOptions(
  options: Partial<SimOptions>,
  model: ModelSpec,
  parallel: ParallelConfig,
  lookupCollectives = false,
): Partial<SimOptions> {
  const { overlapByClass, ...rest } = options;
  if (overlapByClass === undefined) return options;
  const kept: Partial<Record<OverlapClass, number>> = {};
  for (const c of overlapClassesOf(model, parallel, lookupCollectives))
    if (overlapByClass[c] !== undefined) kept[c] = overlapByClass[c];
  return Object.keys(kept).length > 0 ? { ...rest, overlapByClass: kept } : rest;
}
