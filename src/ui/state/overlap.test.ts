/**
 * Per-collective overlap (`SimOptions.overlapByClass`) and two-batch overlap (`DeploymentChoice.twoBatchOverlap`) in
 * the UI state (docs/simulation-model.md §7 "Stage time and overlap", "Two-batch overlap"): the reducer keeps set
 * entries only, the URL codec round-trips them (`ovt`, `ove`, `ovp` in percent, `tbo=1`) and leaves old URLs alone,
 * and the SimInput carries them only for the classes / layouts the run has. GLM-5.3-Flash on a GB200 NVL72 rack.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, normalizeOptions, reduce } from './appState';
import {
  buildDisaggregatedInput,
  buildSimInput,
  loadedFromHf,
  REFERENCE_MODEL,
  type LoadedModel,
} from './buildInput';
import { effectiveOptions, overlapClassesOf, twoBatchOverlapApplies } from './overlap';
import { decodeState, encodeState } from './url';

const ID = 'zai-org/GLM-5.3-Flash';
function load(id: string): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id }, r);
}
const GLM = load(ID);
const PLE_ID = 'Qwen/Qwen3.8-Flash-Next';
const ENGRAM_ID = 'deepseek-ai/DeepSeek-V4.1-Flash';
const PLE = load(PLE_ID);
const ENGRAM = load(ENGRAM_ID);
const DENSE = REFERENCE_MODEL.model;
const RACK = `m=${ID}&d=gb200&n=72&b=72`;
const EP72 = { tp: 1, pp: 1, dp: 72, ep: 72 };

describe('per-class overlap and two-batch overlap', () => {
  it('normalizeOptions keeps set classes (even at a default value) and drops an empty map', () => {
    expect(normalizeOptions({ overlapByClass: {} })).toEqual({});
    expect(normalizeOptions({ overlapByClass: { tp: undefined } })).toEqual({});
    expect(
      normalizeOptions({ overlap: 'ideal', overlapByClass: { tp: 1, ep: undefined } }),
    ).toEqual({ overlapByClass: { tp: 1 } });
  });

  it('the reducer sets and clears one class; two-batch on clears an explicit EP choice', () => {
    let s = reduce(DEFAULT_STATE, { type: 'setOverlapClass', overlapClass: 'ep', fraction: 0.5 });
    s = reduce(s, { type: 'setOverlapClass', overlapClass: 'tp', fraction: 0 });
    expect(s.options).toEqual({ overlapByClass: { ep: 0.5, tp: 0 } });
    const tbo = reduce(s, { type: 'setTwoBatchOverlap', on: true });
    expect(tbo.options).toEqual({ overlapByClass: { tp: 0 } });
    expect(tbo.deployment).toEqual({ twoBatchOverlap: true });
    const off = reduce(tbo, { type: 'setTwoBatchOverlap', on: false });
    expect(off.deployment).toEqual({});
    expect(
      reduce(off, { type: 'setOverlapClass', overlapClass: 'tp', fraction: undefined }),
    ).toEqual(DEFAULT_STATE);
  });

  it('URL: ovt, ove, ovp in percent and tbo=1, right after ov; defaults omitted; invalid values dropped', () => {
    const q = `m=${ID}&d=gb200&n=72&dp=72&ep=72&b=72&ov=none&ovt=0&ove=37.5&ovp=100&tbo=1`;
    const { state, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(state.options).toEqual({ overlap: 'none', overlapByClass: { tp: 0, ep: 0.375, pp: 1 } });
    expect(state.deployment).toEqual({ twoBatchOverlap: true });
    expect(encodeState(state)).toBe(q);
    // Old URLs are unchanged: nothing set, nothing written.
    expect(encodeState(decodeState(`${RACK}&ov=none`).state)).toBe(`${RACK}&ov=none`);
    expect(encodeState(DEFAULT_STATE)).toBe('');
    // ≤ 4 decimals of percent survive the float round trip.
    expect(encodeState(decodeState('ove=33.3333').state)).toBe('ove=33.3333');
    for (const [key, v] of [
      ['ovt', '-1'],
      ['ove', '101'],
      ['ovp', 'x'],
      ['ovt', ''],
      ['tbo', '0'],
      ['tbo', 'yes'],
    ]) {
      const d = decodeState(`${key}=${v}`);
      expect(d.dropped).toEqual([`${key}=${v}`]);
      expect(d.state).toEqual(DEFAULT_STATE);
    }
  });

  it('overlapClassesOf: tp with TP > 1, ep in MoE mode B, pp with PP > 1', () => {
    expect(overlapClassesOf(GLM.model, EP72)).toEqual(['ep']);
    expect(overlapClassesOf(GLM.model, { tp: 8, pp: 2, dp: 1, ep: 8 })).toEqual(['tp', 'pp']);
    expect(overlapClassesOf(GLM.model, { tp: 2, pp: 2, dp: 4, ep: 8 })).toEqual(['tp', 'ep', 'pp']);
    expect(overlapClassesOf(DENSE, { tp: 1, pp: 1, dp: 8, ep: 8 })).toEqual([]);
    expect(twoBatchOverlapApplies(GLM.model, EP72, false)).toBe(true);
    expect(twoBatchOverlapApplies(GLM.model, EP72, true)).toBe(false);
    expect(twoBatchOverlapApplies(DENSE, EP72, false)).toBe(false);
    expect(
      effectiveOptions({ overlap: 'none', overlapByClass: { tp: 0.5 } }, GLM.model, EP72),
    ).toEqual({
      overlap: 'none',
    });
  });

  it.each([
    [PLE_ID, PLE, 'rtx-5090', 2, 2, true, true],
    [PLE_ID, PLE, 'rtx-5090', 2, 2, false, false],
    [PLE_ID, PLE, 'rtx-5090', 1, 1, true, false],
    [PLE_ID, PLE, 'gb200', 72, 72, true, true],
    [ENGRAM_ID, ENGRAM, 'rtx-5090', 2, 2, true, true],
    [ENGRAM_ID, ENGRAM, 'gb200', 72, 72, true, false],
    [ENGRAM_ID, ENGRAM, 'v100-sxm2', 8, 8, true, true, 'nvlink2-hybrid-cube-mesh-8-gpus'],
    [ENGRAM_ID, ENGRAM, 'v100-sxm2', 4, 4, true, false, 'nvlink2-fully-connected-mesh-4-gpus'],
  ] as const)(
    'lookup overlap case %#: %s',
    (id, model, device, n, dp, shared, active, linkId?: string) => {
      const decoded = decodeState(
        `m=${id}&d=${device}&n=${n}&dp=${dp}&b=${dp}&ovt=0${shared ? '&ltdp=1' : ''}${linkId ? `&il=${linkId}` : ''}`,
      );
      expect(decoded.dropped).toEqual([]);
      const state = decoded.state;
      const built = buildSimInput(state, model);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      expect(built.input.options?.overlapByClass).toEqual(active ? { tp: 0 } : undefined);
      // A hidden override stays in the URL for when the class applies again, but cannot affect this input.
      expect(encodeState(state)).toContain('ovt=0');
    },
  );

  it.each([
    [1, 2],
    [2, 1],
  ])(
    'PD keeps the TP-class lookup override with prefill DP %i / decode DP %i',
    (prefillDp, decodeDp) => {
      const base = decodeState(
        `m=${PLE_ID}&d=rtx-5090&n=${prefillDp}&dp=${prefillDp}&b=2&ltdp=1&ovt=0`,
      ).state;
      let state = reduce(base, { type: 'setDisaggregation', on: true });
      state = reduce(state, {
        type: 'inPool',
        pool: 'decode',
        action: { type: 'setCounts', devicesPerNode: decodeDp },
      });
      state = reduce(state, {
        type: 'inPool',
        pool: 'decode',
        action: { type: 'setParallel', parallel: { tp: 1, dp: decodeDp } },
      });
      const built = buildDisaggregatedInput(state, PLE);
      if (built?.kind !== 'ok') throw new Error(JSON.stringify(built?.issues));
      expect(built.prefill.input.options?.overlapByClass).toEqual(
        prefillDp > 1 ? { tp: 0 } : undefined,
      );
      expect(built.decode.input.options?.overlapByClass).toEqual(
        decodeDp > 1 ? { tp: 0 } : undefined,
      );
      expect(built.input.shared.options?.overlapByClass).toEqual({ tp: 0 });
    },
  );

  it('the SimInput carries them only where they apply; absent choices give the input from before', () => {
    const build = (q: string) => {
      const r = buildSimInput(decodeState(`${RACK}&${q}`).state, GLM);
      if (r.kind !== 'ok') throw new Error(JSON.stringify(r.issues));
      return r;
    };
    const plain = build('dp=72&ep=72').input;
    expect(plain.options).toBeUndefined();
    expect(plain.deployment).toBeUndefined();
    const set = build('dp=72&ep=72&ove=50&tbo=1').input;
    expect(set.options).toEqual({ overlapByClass: { ep: 0.5 } });
    expect(set.deployment).toEqual({ twoBatchOverlap: true });
    // A class the run lacks (TP 1, PP 1) and two-batch overlap on a layout without an all-to-all change nothing.
    expect(build('dp=72&ovt=50&ovp=0&tbo=1').input).toEqual(build('dp=72').input);
    // With offloaded experts two-batch overlap is not modelled: off, with a warning on its switch.
    const offload = build('dp=72&ep=72&tbo=1&ox=ram');
    expect(offload.input.deployment?.twoBatchOverlap).toBeUndefined();
    expect(offload.issues.filter((i) => i.field === 'twoBatchOverlap').map((i) => i.level)).toEqual(
      ['warning'],
    );
  });
});
