import { describe, expect, it } from 'vitest';
import {
  anyOverlap,
  coreTime,
  IDEAL_OVERLAP,
  NO_OVERLAP,
  overlapClassOf,
  resolveOverlap,
  sendsOnLatencyPathSec,
  type CoreTimeInput,
  type OverlapFractions,
} from './overlap';

/** compute 10, lat 1, bw_tp 4, bw_ep 8 (bw 12), p2p 2 with a send (seconds). */
const STAGE: CoreTimeInput = {
  computeSec: 10,
  commLatencySec: 1,
  commBandwidthSec: 12,
  bwBySec: { tp: 4, ep: 8 },
  p2pSec: 2,
  hasSendToNext: true,
};

describe('coreTime (docs §7 "Stage time and overlap"): compute 10, lat 1, bw_tp 4, bw_ep 8, p2p 2 (seconds)', () => {
  it('every f = 0 (former none): t_core = 10 + 1 + 12 + 2 = 25, core = 24', () => {
    expect(coreTime(STAGE, NO_OVERLAP)).toEqual({ coreSec: 25, coreTermSec: 24 });
  });

  it('every f = 1 (former ideal): t_core = 1 + max(10, 12) = 13, core = 12; the send is off-stage', () => {
    expect(coreTime(STAGE, IDEAL_OVERLAP)).toEqual({ coreSec: 13, coreTermSec: 12 });
  });

  it('mixed f = {tp 0.5, ep 1, pp 0.25}: H = 2 + 8 = 10, E = 2, core = max(10, 10) + 2 + 0.75·2 = 13.5, t_core = 14.5', () => {
    expect(coreTime(STAGE, { tp: 0.5, ep: 1, pp: 0.25 })).toEqual({
      coreSec: 14.5,
      coreTermSec: 13.5,
    });
  });

  it('f_tp = 1 only: H = 4, E = 8, core = max(10, 4) + 8 + 2 = 20 (tp hides fully, ep and the send serial)', () => {
    expect(coreTime(STAGE, { tp: 1, ep: 0, pp: 0 })).toEqual({ coreSec: 21, coreTermSec: 20 });
  });

  it('H > compute: hidden time is capped at compute (compute 3, f = {0.5, 0.5}: H 6, E 6, core = 6 + 6 = 12 = bw)', () => {
    const small: CoreTimeInput = { ...STAGE, computeSec: 3, p2pSec: 0, hasSendToNext: false };
    const r = coreTime(small, { tp: 0.5, ep: 0.5, pp: 0 });
    expect(r).toEqual({ coreSec: 13, coreTermSec: 12 });
    // = lat + compute + bw − min(compute, H) = 1 + 3 + 12 − 3
    expect(r.coreSec).toBe(1 + 3 + 12 - Math.min(3, 6));
  });

  it('an inactive class is ignored: bw_ep = 0 with f_ep = 0 still takes the ideal expression, bit for bit', () => {
    const tpOnly: CoreTimeInput = {
      computeSec: 0.1,
      commLatencySec: 0.3,
      commBandwidthSec: 0.7,
      bwBySec: { tp: 0.7, ep: 0 },
      p2pSec: 0.2,
      hasSendToNext: true,
    };
    const r = coreTime(tpOnly, { tp: 1, ep: 0, pp: 1 });
    expect(r.coreSec).toBe(0.3 + Math.max(0.1, 0.7));
    expect(r).toEqual(coreTime(tpOnly, IDEAL_OVERLAP));
    // No send: f_pp does not matter either.
    const noSend = { ...tpOnly, p2pSec: 0, hasSendToNext: false };
    expect(coreTime(noSend, { tp: 1, ep: 0.3, pp: 0 })).toEqual(coreTime(noSend, IDEAL_OVERLAP));
    expect(coreTime(noSend, { tp: 0, ep: 1, pp: 1 })).toEqual(coreTime(noSend, NO_OVERLAP));
  });

  it('no active class: both endpoints give compute (+ lat)', () => {
    const none: CoreTimeInput = {
      computeSec: 2,
      commLatencySec: 0,
      commBandwidthSec: 0,
      bwBySec: { tp: 0, ep: 0 },
      p2pSec: 0,
      hasSendToNext: false,
    };
    for (const f of [IDEAL_OVERLAP, NO_OVERLAP, { tp: 0.5, ep: 0.2, pp: 0.7 }])
      expect(coreTime(none, f)).toEqual({ coreSec: 2, coreTermSec: 2 });
  });

  it('bounds ideal ≤ mixed ≤ none, and t_core non-increasing in each f', () => {
    const grid = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
    const stages: CoreTimeInput[] = [
      STAGE,
      { ...STAGE, computeSec: 3 },
      { ...STAGE, computeSec: 30 },
      { ...STAGE, bwBySec: { tp: 0, ep: 12 } },
    ];
    for (const st of stages) {
      const lo = coreTime(st, IDEAL_OVERLAP).coreSec;
      const hi = coreTime(st, NO_OVERLAP).coreSec;
      for (const tp of grid)
        for (const ep of grid)
          for (const pp of grid) {
            const f: OverlapFractions = { tp, ep, pp };
            const t = coreTime(st, f).coreSec;
            expect(t).toBeGreaterThanOrEqual(lo - 1e-12);
            expect(t).toBeLessThanOrEqual(hi + 1e-12);
            for (const key of ['tp', 'ep', 'pp'] as const) {
              const up = Math.min(1, f[key] + 0.1);
              expect(coreTime(st, { ...f, [key]: up }).coreSec).toBeLessThanOrEqual(t + 1e-12);
            }
          }
    }
  });
});

describe('overlapClassOf', () => {
  it('tp, etp, dcp → tp; ep → ep; a pp collective inside a stage throws', () => {
    expect(overlapClassOf('tp')).toBe('tp');
    expect(overlapClassOf('etp')).toBe('tp');
    expect(overlapClassOf('dcp')).toBe('tp');
    expect(overlapClassOf('dp')).toBe('tp');
    expect(overlapClassOf('tp_dp')).toBe('tp');
    expect(overlapClassOf('ep')).toBe('ep');
    expect(() => overlapClassOf('pp')).toThrow(/pp-group collective inside a stage/);
  });
});

describe('resolveOverlap: f_c = overlapByClass[c] ?? (ep ∧ TBO ? 1 : ideal ? 1 : 0)', () => {
  it('the global switch is the default of every class', () => {
    expect(resolveOverlap({ overlap: 'ideal' }, false)).toEqual(IDEAL_OVERLAP);
    expect(resolveOverlap({ overlap: 'none' }, false)).toEqual(NO_OVERLAP);
    expect(resolveOverlap({ overlap: 'none', overlapByClass: {} }, false)).toEqual(NO_OVERLAP);
  });
  it('explicit classes override; absent ones keep the default', () => {
    expect(resolveOverlap({ overlap: 'none', overlapByClass: { tp: 0.5 } }, false)).toEqual({
      tp: 0.5,
      ep: 0,
      pp: 0,
    });
    expect(resolveOverlap({ overlap: 'ideal', overlapByClass: { pp: 0, ep: 0.3 } }, false)).toEqual(
      { tp: 1, ep: 0.3, pp: 0 },
    );
  });
  it('two-batch overlap makes the EP default full; an explicit EP value still wins', () => {
    expect(resolveOverlap({ overlap: 'none' }, true)).toEqual({ tp: 0, ep: 1, pp: 0 });
    expect(resolveOverlap({ overlap: 'none', overlapByClass: { ep: 0.4 } }, true)).toEqual({
      tp: 0,
      ep: 0.4,
      pp: 0,
    });
    expect(resolveOverlap({ overlap: 'ideal' }, true)).toEqual(IDEAL_OVERLAP);
  });
  it('anyOverlap', () => {
    expect(anyOverlap(NO_OVERLAP)).toBe(false);
    expect(anyOverlap({ tp: 0, ep: 0, pp: 0.01 })).toBe(true);
  });
});

describe('sendsOnLatencyPathSec: Σ_s f_pp·p2p_s', () => {
  it('endpoints and a partial share', () => {
    expect(sendsOnLatencyPathSec([2, 0, 3], 1)).toBe(5);
    expect(sendsOnLatencyPathSec([2, 0, 3], 0)).toBe(0);
    expect(sendsOnLatencyPathSec([2, 0, 3], 0.25)).toBe(1.25);
  });
});
