/**
 * Per-collective communication/compute overlap: which overlap class a collective belongs to, the resolved overlap
 * fraction of each class, and the core time of one pipeline stage under those fractions.
 * See docs/simulation-model.md §7 ("Stage time and overlap").
 *
 * Notation (per stage s, one microbatch, one device): compute_s = Σ compute-op times, lat_s = Σ collective α terms,
 * bw_s = Σ collective β terms (op order), bw_c,s = the same β sum over the collectives of class c, p2p_s = the full
 * time (α + β) of the activation send to stage s + 1 (0 on the last stage), f_c ∈ [0, 1] the overlap fraction of
 * class c. All times in seconds; fractions dimensionless.
 */

import type { CollectiveOp } from '../ir/ops';
import type { OverlapClass, SimOptions } from './types';

/**
 * Resolved overlap fractions, one per class, every one set: f_c ∈ [0, 1] is the share of class c's collective
 * bandwidth (β) time that may run concurrently with the stage's compute (for `pp`: the share of the send to the next
 * stage that leaves the stage and lies on the latency path instead). Dimensionless.
 */
export interface OverlapFractions {
  /** f_tp: tensor/context/lookup collectives (groups `tp`, `etp`, `dcp`, `dp`, `tp_dp`). Dimensionless, in [0, 1]. */
  tp: number;
  /** f_ep: EP-group collectives (MoE dispatch/combine all-to-alls, DP-attention gathers). Dimensionless, in [0, 1]. */
  ep: number;
  /** f_pp: the send to the next pipeline stage. Dimensionless, in [0, 1]. */
  pp: number;
}

/** Every class fully overlapped: the former `overlap: 'ideal'`. */
export const IDEAL_OVERLAP: Readonly<OverlapFractions> = Object.freeze({ tp: 1, ep: 1, pp: 1 });

/** No class overlapped: the former `overlap: 'none'`. */
export const NO_OVERLAP: Readonly<OverlapFractions> = Object.freeze({ tp: 0, ep: 0, pp: 0 });

/**
 * The overlap class of a collective inside a stage's layers, by its parallel group: `tp`, `etp`, `dcp`, `dp`,
 * `tp_dp` → `'tp'` (shared lookup exchanges follow the existing lookup TP all-reduce);
 * `ep` → `'ep'`. A `pp`-group collective belongs to the stage boundary (`sendToNext` / `sendToFirst`), never inside
 * a stage's layers, so it throws here (an invariant violation of lowering).
 */
export function overlapClassOf(group: CollectiveOp['group']): Exclude<OverlapClass, 'pp'> {
  switch (group) {
    case 'tp':
    case 'etp':
    case 'dcp':
    case 'dp':
    case 'tp_dp':
      return 'tp';
    case 'ep':
      return 'ep';
    case 'pp':
      throw new Error(
        'a pp-group collective inside a stage: PP sends belong to the stage boundary',
      );
    default: {
      const _never: never = group;
      throw new Error(`unknown collective group ${String(_never)}`);
    }
  }
}

/**
 * The overlap fraction of every class for one phase step:
 *   f_c = overlapByClass[c] ?? (c = ep ∧ splitInTwo ? 1 : overlap = 'ideal' ? 1 : 0)
 * The global `overlap` switch is the default of every class; a step that two-batch overlap splits in two halves
 * (`splitInTwo`, `twoBatch.ts: twoBatchHalves`) makes the EP default full (its point is to hide the EP all-to-alls
 * behind the other half). A step it could not split (one-sequence decode) keeps the global default. An explicit class
 * value always wins. Dimensionless.
 */
export function resolveOverlap(
  options: Pick<SimOptions, 'overlap' | 'overlapByClass'>,
  /** True if two-batch overlap is on AND split this step into two halves. */
  splitInTwo: boolean,
): OverlapFractions {
  const globalDefault = options.overlap === 'ideal' ? 1 : 0;
  const byClass = options.overlapByClass ?? {};
  return {
    tp: byClass.tp ?? globalDefault,
    ep: byClass.ep ?? (splitInTwo ? 1 : globalDefault),
    pp: byClass.pp ?? globalDefault,
  };
}

/** True if some class has a fraction above 0 (some communication may hide behind compute). */
export function anyOverlap(f: OverlapFractions): boolean {
  return f.tp > 0 || f.ep > 0 || f.pp > 0;
}

/** The per-stage sums {@link coreTime} needs. Times in seconds, for one microbatch on one device. */
export interface CoreTimeInput {
  /** compute_s: Σ compute-op times (every op that is neither a collective nor a transfer). Unit: seconds. */
  computeSec: number;
  /** lat_s: Σ collective α terms, in op order. Unit: seconds. */
  commLatencySec: number;
  /** bw_s: Σ collective β terms, in op order (the endpoint branches use this sum as is). Unit: seconds. */
  commBandwidthSec: number;
  /** bw_c,s: the collective β terms summed per class (bw_tp,s + bw_ep,s = bw_s up to rounding). Unit: seconds. */
  bwBySec: { tp: number; ep: number };
  /** p2p_s: full time (α + β) of the send to the next stage; 0 when there is none. Unit: seconds. */
  p2pSec: number;
  /** True if the stage has a send to the next stage (pp class active), even one of zero time. */
  hasSendToNext: boolean;
}

/** Result of {@link coreTime}. Times in seconds. */
export interface CoreTime {
  /** t_core,s: the stage time without off-device transfers. Unit: seconds. */
  coreSec: number;
  /** core_s = t_core,s − lat_s: the part the prefetched transfers may hide behind (pipeline.ts). Unit: seconds. */
  coreTermSec: number;
}

/**
 * Core time of one stage (one microbatch, one device) under per-class overlap fractions f. A share f_c of class c's
 * β time may run concurrently with compute, the rest is serial; α is on the dependency chain and never hidden; hidden
 * time can never exceed compute:
 *   H_s      = f_tp·bw_tp,s + f_ep·bw_ep,s                          hideable β
 *   E_s      = (1 − f_tp)·bw_tp,s + (1 − f_ep)·bw_ep,s              serial β
 *   core_s   = max(compute_s, H_s) + E_s + (1 − f_pp)·p2p_s        (the in-stage share of the PP send)
 *   t_core,s = lat_s + core_s
 * i.e. t_core,s = lat_s + compute_s + bw_s + (1 − f_pp)·p2p_s − min(compute_s, H_s). The share f_pp·p2p_s of the send
 * leaves the stage and lies on the latency path (pipeline.ts: phaseResult).
 *
 * A class is ACTIVE in the stage when it has time there (tp, ep: bw_c,s > 0; pp: a send to the next stage). The
 * endpoints keep their former expressions, which sum β in op order rather than per class, so they are bit-identical
 * to before:
 * - every active class at f = 1 (former `'ideal'`): t_core,s = lat_s + max(compute_s, bw_s);  core_s = max(compute_s, bw_s)
 * - every active class at f = 0 (former `'none'`):  t_core,s = compute_s + lat_s + bw_s + p2p_s;  core_s = compute_s + bw_s + p2p_s
 * - otherwise the general formula above.
 * With no active class both endpoints give compute_s (+ lat_s).
 * Always lat_s + max(compute_s, bw_s) ≤ t_core,s ≤ compute_s + lat_s + bw_s + p2p_s (up to rounding), and t_core,s is
 * non-increasing in each f_c. See docs/simulation-model.md §7 ("Stage time and overlap").
 */
export function coreTime(input: CoreTimeInput, f: OverlapFractions): CoreTime {
  const { computeSec, commLatencySec, commBandwidthSec, bwBySec, p2pSec } = input;
  const active: [boolean, number][] = [
    [bwBySec.tp > 0, f.tp],
    [bwBySec.ep > 0, f.ep],
    [input.hasSendToNext, f.pp],
  ];
  if (active.every(([on, fc]) => !on || fc === 1)) {
    // Formula: t_core,s = lat_s + max(compute_s, bw_s);  core_s = max(compute_s, bw_s)   (every active f = 1)
    // See docs/simulation-model.md §7 ("Stage time and overlap")
    const coreTermSec = Math.max(computeSec, commBandwidthSec);
    return { coreSec: commLatencySec + coreTermSec, coreTermSec };
  }
  if (active.every(([on, fc]) => !on || fc === 0)) {
    // Formula: t_core,s = compute_s + lat_s + bw_s + p2p_s;  core_s = compute_s + bw_s + p2p_s   (every active f = 0)
    // See docs/simulation-model.md §7 ("Stage time and overlap")
    return {
      coreSec: computeSec + commLatencySec + commBandwidthSec + p2pSec,
      coreTermSec: computeSec + commBandwidthSec + p2pSec,
    };
  }
  // Formula: H_s = f_tp·bw_tp,s + f_ep·bw_ep,s;  E_s = (1 − f_tp)·bw_tp,s + (1 − f_ep)·bw_ep,s
  //          core_s = max(compute_s, H_s) + E_s + (1 − f_pp)·p2p_s;  t_core,s = lat_s + core_s
  // See docs/simulation-model.md §7 ("Stage time and overlap")
  const hideableSec = f.tp * bwBySec.tp + f.ep * bwBySec.ep;
  const serialSec = (1 - f.tp) * bwBySec.tp + (1 - f.ep) * bwBySec.ep;
  const inStageP2pSec = (1 - f.pp) * p2pSec;
  const coreTermSec = Math.max(computeSec, hideableSec) + serialSec + inStageP2pSec;
  return { coreSec: commLatencySec + coreTermSec, coreTermSec };
}

/**
 * The share of the PP sends on the latency path of a phase, outside every stage: Σ_s f_pp·p2p_s. The endpoints keep
 * their former expressions (bit-identical): f_pp = 1 → Σ_s p2p_s (former `'ideal'`), f_pp = 0 → 0 (former `'none'`:
 * each send is inside its stage). Unit: seconds.
 */
export function sendsOnLatencyPathSec(sendSecs: readonly number[], fpp: number): number {
  // Formula: Σ_s f_pp·p2p_s   (f_pp = 1: Σ p2p_s; f_pp = 0: 0)
  // See docs/simulation-model.md §7 ("Pipeline and DP")
  if (fpp === 0) return 0;
  if (fpp === 1) return sendSecs.reduce((s, x) => s + x, 0);
  return sendSecs.reduce((s, x) => s + fpp * x, 0);
}
