import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { decodeState, encodeNumber, encodeState, latencyFloorSecFromMicros } from './url';
import { effectiveCabling, resolveHardware } from './hardware';

/** A state built from the default by reducer actions (so it is in the reducer's canonical form). */
function state(
  over: Omit<Partial<AppState>, 'hw'> & { hw?: Partial<AppState['hw']> } = {},
): AppState {
  return { ...DEFAULT_STATE, ...over, hw: { ...DEFAULT_STATE.hw, ...over.hw } };
}

const STATES: [string, AppState][] = [
  ['default', DEFAULT_STATE],
  ['HF model', state({ model: { kind: 'hf', id: 'Qwen/Qwen3.8-27B' } })],
  [
    'HF model with revision',
    state({ model: { kind: 'hf', id: 'nvidia/Qwen3.8-27B-NVFP4', revision: 'a1b2c3d' } }),
  ],
  [
    'quant overrides',
    state({ quant: { weights: 'nvfp4', activations: 'fp8_e4m3', kvCache: 'fp8_e4m3' } }),
  ],
  ['lookup-table format', state({ quant: { weights: 'nvfp4', lookupTables: 'nvfp4' } })],
  [
    'EXL3 weights with bpw and head bpw',
    state({ quant: { weights: 'exl3', bpw: 2.5, headBpw: 5 } }),
  ],
  ['EXL3 weights at the default bpw', state({ quant: { weights: 'exl3' } })],
  [
    'unlocks + capacity',
    state({
      hw: {
        deviceId: 'rtx-4090',
        unlocks: { 'vram-mod': '48gb', 'pcie-p2p': 'direct' },
        capacityIndex: 0,
      },
    }),
  ],
  [
    'engine + capacity index',
    state({ hw: { deviceId: 'apple-m5-max-40c', engineId: 'gpu-simd', capacityIndex: 2 } }),
  ],
  [
    'device edits',
    state({
      hw: {
        deviceEdits: {
          'memory.bandwidthBytesPerSec': 2e12,
          'engines.0.matmulPeaks.0.flopsPerSec': 1.5e14,
        },
      },
    }),
  ],
  [
    'multi-node with links and link edits',
    state({
      hw: {
        deviceId: 'h100-sxm',
        devicesPerNode: 8,
        nodes: 2,
        intraLinkId: 'nvlink4-nvswitch',
        interLinkId: 'infiniband-xdr800-one-nic-per-gpu',
        intraEdits: { latencySec: 2e-6 },
        interEdits: { bandwidthPerDirectionBytesPerSec: 5e10 },
      },
      parallel: { tp: 8, pp: 2, dp: 1, ep: 1 },
    }),
  ],
  [
    'parallel + workload',
    state({
      parallel: { tp: 2, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 8, promptLen: 4096, decodeContextLen: 32768 },
    }),
  ],
  [
    'MoE with expert parallelism',
    state({
      model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
      parallel: { tp: 4, pp: 1, dp: 1, ep: 2 },
    }),
  ],
  [
    'data parallelism',
    state({
      hw: { devicesPerNode: 4 },
      parallel: { tp: 2, pp: 1, dp: 2, ep: 1 },
      workload: { batch: 4, promptLen: 8192, decodeContextLen: 8192 },
    }),
  ],
  [
    'MoE with mode-B expert parallelism',
    state({
      model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
      parallel: { tp: 1, pp: 1, dp: 4, ep: 4 },
      workload: { batch: 8, promptLen: 8192, decodeContextLen: 8192 },
    }),
  ],
  [
    'MoE with EP imbalance and redundant experts',
    state({
      model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
      parallel: { tp: 1, pp: 1, dp: 4, ep: 4, epImbalance: 1.55, redundantExpertsPerRank: 2 },
      workload: { batch: 8, promptLen: 8192, decodeContextLen: 8192 },
    }),
  ],
  [
    'every option',
    state({
      options: { overlap: 'none', elementwiseFusion: 'ideal', collectiveAlgorithm: 'ring' },
    }),
  ],
  [
    'every offload knob and host field',
    state({
      hw: {
        host: {
          ramId: 'ddr5-6400-8ch',
          ramCapacityBytes: 512e9,
          ssdId: 'nvme-t705-2tb',
          link: { generation: 4, lanes: 8 },
        },
      },
      deployment: {
        offload: {
          experts: {
            tier: 'nvme',
            fromLayer: 3,
            residentFraction: 0.25,
            hitRatio: 'skewed_fine_grained',
          },
          tables: { tier: 'host_ram', scaleLayout: 'interleaved' },
          queueDepth: 32,
          prefetchAccuracy: 0.9,
        },
      },
    }),
  ],
  [
    'custom hit ratio',
    state({ deployment: { offload: { experts: { tier: 'host_ram', hitRatio: 0.6 } } } }),
  ],
  [
    'host-RAM + SSD splits of experts and lookup tables',
    state({
      deployment: {
        offload: {
          experts: {
            tier: 'split',
            split: { ramLayers: 20, ramEnd: 'last' },
            fromLayer: 4,
            residentFraction: 0.1,
          },
          tables: { tier: 'split', split: { ramLayers: 10 }, scaleLayout: 'separate' },
        },
      },
    }),
  ],
  [
    'MTP with k and a use case',
    state({ deployment: { specDec: { method: 'mtp', k: 2, useCase: 'code' } } }),
  ],
  [
    'DFlash 2 with a drafter repo and custom α',
    state({
      deployment: {
        specDec: { method: 'dflash2', drafter: 'incoai/Qwen3.8-27B-DFlash2', k: 7, alpha: 0.8 },
      },
    }),
  ],
  [
    'DSpark with the hypothetical drafter',
    state({ deployment: { specDec: { method: 'dspark', drafter: 'hyp', useCase: 'agentic' } } }),
  ],
  [
    'DGX Spark ×4 ring',
    state({
      hw: { deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' },
      parallel: { tp: 4, pp: 1, dp: 1, ep: 1 },
    }),
  ],
  [
    'Strix Halo ×4 daisy chain',
    state({ hw: { deviceId: 'ryzen-ai-max-plus-395', nodes: 4, cabling: 'chain' } }),
  ],
  [
    'M5 Ultra ×4 mesh, 2 cables per pair',
    state({ hw: { deviceId: 'apple-m5-ultra-80c', capacityIndex: 2, nodes: 4, cabling: 'mesh2' } }),
  ],
  [
    'M3 Ultra ×4 ring (the default is a mesh)',
    state({ hw: { deviceId: 'apple-m3-ultra-80c', nodes: 4, cabling: 'ring' } }),
  ],
  [
    'raised GPU limit and runtime reserve on a Mac',
    state({
      hw: { deviceId: 'apple-m4-max-40c', gpuUsableFraction: 0.9, runtimeReserveBytes: 1.5e9 },
    }),
  ],
  [
    'runtime reserve and prefill chunk on a GPU',
    state({
      hw: { runtimeReserveBytes: 1e9 },
      workload: { ...DEFAULT_STATE.workload, prefillChunkTokens: 2048 },
    }),
  ],
  [
    'per-op latency floor and engine dispatch overlay',
    state({
      options: { perOpLatencyFloorSec: latencyFloorSecFromMicros(0.6) },
      engineDispatchOverlay: true,
    }),
  ],
  [
    '2 × 8 H100 with 4 NICs per machine and in-network reduction',
    state({
      hw: { deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2, nicsPerNode: 4 },
      parallel: { tp: 16, pp: 1, dp: 1, ep: 1 },
      options: { inNetworkReduction: 'on' },
    }),
  ],
];

describe('URL codec', () => {
  it('network: nic (NICs per machine, only where offered; the default absent), inr=on; invalid values dropped', () => {
    const base = 'd=h100-sxm&n=8&nodes=2';
    const nic4 = state({
      hw: { deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2, nicsPerNode: 4 },
    });
    expect(encodeState(nic4)).toBe(`${base}&nic=4`);
    expect(decodeState(`${base}&nic=4`)).toEqual({ state: nic4, dropped: [] });
    expect(decodeState(`${base}&nic=32`).state.hw.nicsPerNode).toBe(32);
    // One per GPU is the default: accepted, not stored.
    expect(decodeState(`${base}&nic=8`)).toEqual({ state: decodeState(base).state, dropped: [] });
    // Not rail-symmetric, not a number, or not offered (one machine; one GPU per machine): dropped.
    expect(decodeState(`${base}&nic=3`).dropped).toEqual(['nic=3']);
    expect(decodeState(`${base}&nic=0`).dropped).toEqual(['nic=0']);
    expect(decodeState(`${base}&nic=x`).dropped).toEqual(['nic=x']);
    expect(decodeState('d=h100-sxm&n=8&nic=4').dropped).toEqual(['nic=4']);
    expect(decodeState('d=dgx-spark&nodes=4&nic=2').dropped).toEqual(['nic=2']);
    // A switchless cabling has no NICs to choose.
    expect(decodeState('d=dgx-spark&nodes=4&fab=ring&nic=1').dropped).toEqual(['nic=1']);
    // In-network reduction: an option key; off is the default.
    const on = reduce(DEFAULT_STATE, { type: 'setOptions', options: { inNetworkReduction: 'on' } });
    expect(encodeState(on)).toBe('inr=on');
    expect(decodeState('inr=on')).toEqual({ state: on, dropped: [] });
    expect(decodeState('inr=off')).toEqual({ state: DEFAULT_STATE, dropped: [] });
    expect(decodeState('inr=yes').dropped).toEqual(['inr=yes']);
    expect(
      encodeState(reduce(on, { type: 'setOptions', options: { inNetworkReduction: 'off' } })),
    ).toBe('');
    // Keys come in URL_KEYS order: nic after fab, inr after ar.
    expect(
      encodeState(
        reduce(nic4, {
          type: 'setOptions',
          options: { inNetworkReduction: 'on', collectiveAlgorithm: 'ring' },
        }),
      ),
    ).toBe(`${base}&nic=4&ar=ring&inr=on`);
  });
  it('per-op costs: lf (µs, 0 = off), ed=1 (dispatch overlay); invalid values dropped', () => {
    const on = reduce(
      reduce(DEFAULT_STATE, {
        type: 'setOptions',
        options: { perOpLatencyFloorSec: latencyFloorSecFromMicros(0.6) },
      }),
      { type: 'setEngineDispatchOverlay', on: true },
    );
    expect(encodeState(on)).toBe('lf=0.6&ed=1');
    expect(decodeState('lf=0.6&ed=1')).toEqual({ state: on, dropped: [] });
    expect(decodeState('lf=1.5').state.options.perOpLatencyFloorSec).toBe(1.5e-6);
    // 0 is the default: accepted, not stored.
    expect(decodeState('lf=0')).toEqual({ state: DEFAULT_STATE, dropped: [] });
    expect(decodeState('lf=-1&ed=yes').dropped).toEqual(['lf=-1', 'ed=yes']);
    expect(decodeState('lf=1001').dropped).toEqual(['lf=1001']);
    // Off again: the default state, nothing in the URL.
    const off = reduce(
      reduce(on, { type: 'setOptions', options: { perOpLatencyFloorSec: undefined } }),
      { type: 'setEngineDispatchOverlay', on: false },
    );
    expect(off).toEqual(DEFAULT_STATE);
    expect(encodeState(off)).toBe('');
  });
  it('memory-fit knobs: gpu (percent, unified only), rr (GB), pc (tokens)', () => {
    expect(
      encodeState(
        state({
          hw: { deviceId: 'apple-m4-max-40c', gpuUsableFraction: 0.9, runtimeReserveBytes: 1.5e9 },
        }),
      ),
    ).toBe('d=apple-m4-max-40c&gpu=90&rr=1.5');
    // The GPU limit means nothing on a dedicated GPU: dropped.
    expect(decodeState('gpu=90').dropped).toEqual(['gpu=90']);
    expect(decodeState('d=apple-m4-max-40c&gpu=0').dropped).toEqual(['gpu=0']);
    expect(decodeState('pc=0').dropped).toEqual(['pc=0']);
    expect(
      resolveHardware(decodeState('d=apple-m4-max-40c&gpu=90&rr=2').state.hw).stack,
    ).toMatchObject({
      gpuUsableFraction: 0.9,
      runtimeReserveBytes: 2e9,
    });
    const set = reduce(DEFAULT_STATE, {
      type: 'setMemoryFit',
      field: 'runtimeReserveBytes',
      value: 1e9,
    });
    expect(encodeState(set)).toBe('rr=1');
    expect(
      reduce(set, { type: 'setMemoryFit', field: 'runtimeReserveBytes', value: undefined }),
    ).toEqual(DEFAULT_STATE);
  });

  it('encodes the default state to the empty string', () => {
    expect(encodeState(DEFAULT_STATE)).toBe('');
    expect(decodeState('')).toEqual({ state: DEFAULT_STATE, dropped: [] });
  });

  it.each(STATES)('round-trips: %s', (_name, s) => {
    const q = encodeState(s);
    const decoded = decodeState(q);
    expect(decoded.dropped).toEqual([]);
    expect(decoded.state).toEqual(s);
    expect(decodeState(`?${q}`).state).toEqual(s);
  });

  it.each([
    'm=zai-org/GLM-5.3&kv=fp8_e4m3&d=rtx-4090&u=vram-mod.48gb&n=2&tp=2&b=4',
    'm=Qwen/Qwen3.8-27B&rev=main2&w=fp8_e4m3&a=fp8_e4m3',
    'u=pcie-p2p.direct,power-clock.400w&n=2&il=pcie-5-0-x16-host-bounce&xi=latencySec:5e-7&tp=2',
    'x=memory.bandwidthBytesPerSec:2e12&b=16&p=4096&c=4096',
    'n=4&nodes=2&xl=10gbe-tcp&xe=bandwidthPerDirectionBytesPerSec:2.5e9&tp=4&ov=none&fu=ideal&ar=tree',
    'm=Qwen/Qwen3.8-Flash-Next&n=4&tp=4&ep=4',
    'n=4&tp=2&dp=2&b=2',
    'm=Qwen/Qwen3.8-Flash-Next&n=4&tp=2&dp=2&ep=4&b=4',
    'm=moonshotai/Kimi-K3&d=b200&n=8&tp=8&dcp=8',
  ])('keeps a canonical query unchanged: %s', (q) => {
    const { state: s, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(encodeState(s)).toBe(q);
  });

  it('loads an old URL with dp and ep as written (dp was forced to 1 while DP was hidden)', () => {
    const { state: s, dropped } = decodeState('tp=2&dp=2&ep=2');
    expect(s.parallel).toEqual({ tp: 2, pp: 1, dp: 2, ep: 2 });
    expect(dropped).toEqual([]);
    expect(encodeState(s)).toBe('tp=2&dp=2&ep=2');
  });

  it('drops a dp or ep that is not a whole number ≥ 1', () => {
    for (const key of ['dp', 'ep'] as const) {
      for (const v of ['0', '1.5', '-2', 'x']) {
        const { state: s, dropped } = decodeState(`tp=2&${key}=${v}`);
        expect(s.parallel[key]).toBe(1);
        expect(dropped).toEqual([`${key}=${v}`]);
      }
    }
  });

  it('EP balance: epi (ρ ≥ 1), epa (1 = ρ on the all-to-alls) and rex (whole ≥ 0) after dn; defaults omitted; invalid values dropped', () => {
    const q = 'm=Qwen/Qwen3.8-Flash-Next&n=4&dp=4&ep=4&epi=1.5&epa=1&rex=32&b=4';
    const { state: s, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(s.parallel).toEqual({
      tp: 1,
      pp: 1,
      dp: 4,
      ep: 4,
      epImbalance: 1.5,
      epImbalanceScalesAllToAll: true,
      redundantExpertsPerRank: 32,
    });
    expect(encodeState(s)).toBe(q);
    // The defaults (ρ 1.2, off, 0 replicas) decode to absent fields and are never written
    expect(decodeState('ep=2&dp=2&epi=1.2&rex=0')).toEqual(decodeState('ep=2&dp=2'));
    const set = reduce(DEFAULT_STATE, {
      type: 'setParallel',
      parallel: { epImbalance: 1.5, epImbalanceScalesAllToAll: true, redundantExpertsPerRank: 1 },
    });
    expect(encodeState(set)).toBe('epi=1.5&epa=1&rex=1');
    const back = reduce(set, {
      type: 'setParallel',
      parallel: { epImbalance: 1.2, epImbalanceScalesAllToAll: false, redundantExpertsPerRank: 0 },
    });
    expect(back).toEqual(DEFAULT_STATE);
    expect(encodeState(back)).toBe('');
    for (const [key, v] of [
      ['epi', '0.9'],
      ['epi', '0'],
      ['epi', 'x'],
      ['epi', '101'],
      ['epa', '0'],
      ['epa', 'true'],
      ['rex', '-1'],
      ['rex', '1.5'],
      ['rex', 'x'],
    ]) {
      const d = decodeState(`ep=2&dp=2&${key}=${v}`);
      expect(d.dropped).toEqual([`${key}=${v}`]);
      expect(d.state.parallel).toEqual({ tp: 1, pp: 1, dp: 2, ep: 2 });
    }
  });

  it('drops EXL3 bpw values that are out of range or lack w=exl3', () => {
    expect(encodeState(DEFAULT_STATE)).not.toContain('lt=');
    expect(decodeState('lt=mxfp8').state.quant).toEqual({ lookupTables: 'mxfp8' });
    expect(decodeState('lt=fp32').dropped.length).toBe(1);
    expect(decodeState('w=exl3&bpw=2.5&hbpw=16').state.quant).toEqual({
      weights: 'exl3',
      bpw: 2.5,
      headBpw: 16,
    });
    const d = decodeState('w=exl3&bpw=12&hbpw=0');
    expect(d.state.quant).toEqual({ weights: 'exl3' });
    expect(d.dropped).toEqual(['bpw=12', 'hbpw=0']);
    expect(decodeState('w=fp8_e4m3&bpw=3').dropped).toEqual(['bpw=3']);
    expect(encodeState(state({ quant: { weights: 'fp8_e4m3', bpw: 3 } }))).toBe('w=fp8_e4m3');
  });

  it('round-trips device-only table sharing and drops it for every off-device table placement', () => {
    const { state: s, dropped } = decodeState('ltdp=1');
    expect(dropped).toEqual([]);
    expect(s.deployment.offload).toEqual({ tables: { acrossDp: true } });
    expect(encodeState(s)).toBe('ltdp=1');
    expect(encodeState(DEFAULT_STATE)).not.toContain('ltdp');
    expect(decodeState('ltdp=0').dropped).toEqual(['ltdp=0']);
    for (const olt of ['ram', 'ssd', 'split']) {
      const d = decodeState(`olt=${olt}&ltdp=1`);
      expect(d.dropped).toContain('ltdp=1');
      expect(d.state.deployment.offload?.tables?.acrossDp).toBeUndefined();
      expect(encodeState(d.state)).not.toContain('ltdp');
    }
    expect(decodeState('ox=ram&ltdp=1').state.deployment.offload).toEqual({
      experts: { tier: 'host_ram' },
      tables: { acrossDp: true },
    });
  });

  it('keeps / : , . readable', () => {
    const q = encodeState(
      state({
        model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
        hw: { deviceEdits: { 'memory.bandwidthBytesPerSec': 2e12 } },
      }),
    );
    expect(q).toBe('m=Qwen/Qwen3.8-Flash-Next&x=memory.bandwidthBytesPerSec:2e12');
  });

  it('drops junk values and unknown keys, never throws', () => {
    const { state: s, dropped } = decodeState(
      'm=not a model&w=fp7&d=gtx-1080&tp=0&pp=-1&b=abc&ov=sometimes&zzz=1&cap=9&eng=npu&n=0',
    );
    expect(s).toEqual(DEFAULT_STATE);
    expect(dropped.sort()).toEqual(
      [
        'm=not a model',
        'w=fp7',
        'd=gtx-1080',
        'tp=0',
        'pp=-1',
        'b=abc',
        'ov=sometimes',
        'zzz=1',
        'cap=9',
        'eng=npu',
        'n=0',
      ].sort(),
    );
  });

  it('drops unknown unlocks, edit paths and links, and link edits for a level that does not exist', () => {
    const { state: s, dropped } = decodeState(
      'u=vram-mod.48gb,nope.x&x=memory.bandwidthBytesPerSec:1e12,memory.bogus:3,engines.0.computeUnits:-1&il=warp-drive&xi=latencySec:1e-6',
    );
    // rtx-5090 has no vram-mod axis.
    expect(s.hw.unlocks).toEqual({});
    expect(s.hw.deviceEdits).toEqual({ 'memory.bandwidthBytesPerSec': 1e12 });
    expect(s.hw.intraLinkId).toBeUndefined();
    expect(s.hw.intraEdits).toEqual({});
    expect(dropped).toEqual([
      'u=vram-mod.48gb,nope.x',
      'x=memory.bogus:3,engines.0.computeUnits:-1',
      'il=warp-drive',
      'xi=latencySec:1e-6',
    ]);
  });

  it('checks the capacity index against the unlocked device', () => {
    // The 4090's 48 GB mod replaces the capacity options with one entry: index 0 only.
    expect(decodeState('d=rtx-4090&u=vram-mod.48gb&cap=1').dropped).toEqual(['cap=1']);
    expect(decodeState('d=apple-m3-ultra-80c&cap=2').state.hw.capacityIndex).toBe(2);
  });

  it('never writes an HF token', () => {
    const q = encodeState(state({ model: { kind: 'hf', id: 'meta/gated' } }));
    expect(q).not.toMatch(/token|hf_/i);
  });

  it('normalizes rev=main and default-valued options away', () => {
    expect(encodeState(decodeState('m=a/b&rev=main&ov=ideal&ar=best').state)).toBe('m=a/b');
    const s = reduce(DEFAULT_STATE, { type: 'setOptions', options: { overlap: 'none' } });
    expect(encodeState(reduce(s, { type: 'setOptions', options: { overlap: 'ideal' } }))).toBe('');
  });

  it('writes numbers in their shortest exact form', () => {
    for (const v of [2e12, 1.792e12, 1024, 0.5, 5e-7, 123456789, 1 / 3]) {
      expect(Number(encodeNumber(v))).toBe(v);
    }
    expect(encodeNumber(2e12)).toBe('2e12');
    expect(encodeNumber(1024)).toBe('1024');
  });

  it('omits the default model and hardware; another model and old links still work', () => {
    expect(encodeState(DEFAULT_STATE)).toBe('');
    expect(encodeState(state({ model: { kind: 'hf', id: 'zai-org/GLM-5.3' } }))).toBe(
      'm=zai-org/GLM-5.3',
    );
    expect(decodeState('m=zai-org/GLM-5.3').state.model).toEqual({
      kind: 'hf',
      id: 'zai-org/GLM-5.3',
    });
    // Old link without m: the default model; `?b=2` keeps the default hardware.
    const old = decodeState('b=2');
    expect(old.state.model).toEqual(DEFAULT_STATE.model);
    expect(old.state.hw).toEqual(DEFAULT_STATE.hw);
    expect(old.state.workload.batch).toBe(2);
  });

  it('prefill length and decode context default to 8192 and are written only when different', () => {
    expect(
      encodeState(state({ workload: { batch: 1, promptLen: 8192, decodeContextLen: 32768 } })),
    ).toBe('c=32768');
  });

  it('writes fab only for a non-default cabling, right after xl', () => {
    expect(encodeState(state({ hw: { deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' } }))).toBe(
      'd=dgx-spark&nodes=4&fab=ring',
    );
    expect(
      encodeState(
        state({
          hw: {
            deviceId: 'dgx-spark',
            nodes: 4,
            interLinkId: 'dgx-spark-connectx-7-200gbe-roce',
            cabling: 'ring',
          },
        }),
      ),
    ).toBe('d=dgx-spark&nodes=4&xl=dgx-spark-connectx-7-200gbe-roce&fab=ring');
    // A switch is the default wherever one exists, so it is never written.
    expect(
      encodeState(state({ hw: { deviceId: 'dgx-spark', nodes: 4, cabling: 'switched' } })),
    ).toBe('d=dgx-spark&nodes=4');
    const explicit = decodeState('d=dgx-spark&nodes=4&fab=switched');
    expect(explicit.dropped).toEqual([]);
    expect(explicit.state.hw.cabling).toBeUndefined();
  });

  it('loads an old URL without fab on the default cabling for its link and counts', () => {
    // Before fabric shapes the 4-Spark URL meant the ConnectX-7 switch; it still does.
    const spark = decodeState(
      'm=Qwen/Qwen3.8-Flash-Next&d=dgx-spark&nodes=4&xl=dgx-spark-connectx-7-200gbe-roce&tp=4',
    );
    expect(spark.dropped).toEqual([]);
    expect(spark.state.hw.cabling).toBeUndefined();
    expect(effectiveCabling(spark.state.hw)).toBe('switched');
    expect(encodeState(spark.state)).toBe(
      'm=Qwen/Qwen3.8-Flash-Next&d=dgx-spark&nodes=4&xl=dgx-spark-connectx-7-200gbe-roce&tp=4',
    );
    // Thunderbolt 5 has no switch: 4 Macs open on the full mesh, 4 Strix Halos (2 USB4 ports) on the ring.
    expect(effectiveCabling(decodeState('d=apple-m3-ultra-80c&nodes=4').state.hw)).toBe('mesh');
    expect(effectiveCabling(decodeState('d=ryzen-ai-max-plus-395&nodes=4').state.hw)).toBe('ring');
  });

  it('drops an unknown fab, and one the device, link and counts do not allow', () => {
    expect(decodeState('d=dgx-spark&nodes=4&fab=star').dropped).toEqual(['fab=star']);
    // A full mesh of 5 needs 4 ports; the M4 Max has 3 usable Thunderbolt 5 ports.
    const mesh5 = decodeState('d=apple-m4-max-40c&nodes=5&fab=mesh');
    expect(mesh5.dropped).toEqual(['fab=mesh']);
    expect(mesh5.state.hw.cabling).toBeUndefined();
    // A ring of three is the full mesh; one machine has no cabling.
    expect(decodeState('d=dgx-spark&nodes=3&fab=ring').dropped).toEqual(['fab=ring']);
    expect(decodeState('d=dgx-spark&fab=ring').dropped).toEqual(['fab=ring']);
    // Several GPUs per machine network through a switch only.
    expect(decodeState('n=2&nodes=2&fab=mesh').dropped).toEqual(['fab=mesh']);
  });

  it('loads an xl link the device catalog does not list, as a switch only', () => {
    const { state: s, dropped } = decodeState(
      'd=ryzen-ai-max-plus-395&nodes=4&xl=infiniband-ndr400-one-nic-per-gpu&fab=ring',
    );
    expect(s.hw.interLinkId).toBe('infiniband-ndr400-one-nic-per-gpu');
    expect(dropped).toEqual(['fab=ring']);
    expect(effectiveCabling(s.hw)).toBe('switched');
  });

  it('Thunderbolt TCP: TB5 Macs use the TB5 bridge, the M3 Max the TB4 bridge; an old TB4 link on a TB5 Mac still loads', () => {
    const tb5 = decodeState('d=apple-m3-ultra-80c&nodes=3&xl=thunderbolt-5-bridge-tcp');
    expect(tb5.dropped).toEqual([]);
    expect(effectiveCabling(tb5.state.hw)).toBe('mesh');
    expect(resolveHardware(tb5.state.hw).topology.interNode?.name).toBe(
      'Thunderbolt 5 bridge (TCP)',
    );
    expect(
      resolveHardware(decodeState('d=apple-m3-max-40c&nodes=2').state.hw).topology.interNode?.name,
    ).toBe('Thunderbolt 4 bridge (TCP)');
    const old = decodeState('d=apple-m3-ultra-80c&nodes=3&xl=thunderbolt-4-bridge-tcp');
    expect(old.dropped).toEqual([]);
    expect(resolveHardware(old.state.hw).topology.interNode?.name).toBe(
      'Thunderbolt 4 bridge (TCP)',
    );
    expect(effectiveCabling(old.state.hw)).toBe('switched');
  });

  it('drops the 2-card NVLink bridge at more than 2 GPUs (the default PCIe link applies)', () => {
    expect(decodeState('d=rtx-3090&n=2&il=rtx-3090-nvlink-bridge').dropped).toEqual([]);
    const four = decodeState('d=rtx-3090&n=4&il=rtx-3090-nvlink-bridge');
    expect(four.dropped).toEqual(['il=rtx-3090-nvlink-bridge']);
    expect(four.state.hw.intraLinkId).toBeUndefined();
  });

  it('CMP 170HX: a PCIe link the decoded unlocks cannot run is dropped; with the unlocks it round-trips', () => {
    const stale = decodeState('d=cmp-170hx&n=2&il=pcie-2-0-x16-host-bounce');
    expect(stale.dropped).toEqual(['il=pcie-2-0-x16-host-bounce']);
    expect(stale.state.hw.intraLinkId).toBeUndefined();
    expect(resolveHardware(stale.state.hw).topology.intraNode?.name).toBe(
      'PCIe 1.0 x4 (host bounce)',
    );
    const capOnly = decodeState(
      'd=cmp-170hx&u=pcie-capacitor-mod.x16&n=2&il=pcie-2-0-x16-host-bounce',
    );
    expect(capOnly.dropped).toEqual(['il=pcie-2-0-x16-host-bounce']);
    expect(resolveHardware(capOnly.state.hw).topology.intraNode?.name).toBe(
      'PCIe 1.0 x16 (host bounce)',
    );
    const q =
      'd=cmp-170hx&u=compute-path.cmpunlocker,pcie-capacitor-mod.x16&n=2&il=pcie-2-0-x16-host-bounce';
    const ok = decodeState(q);
    expect(ok.dropped).toEqual([]);
    expect(ok.state.hw.intraLinkId).toBe('pcie-2-0-x16-host-bounce');
    expect(encodeState(ok.state)).toBe(q);
  });

  it('encodes the deployment and host keys in a readable canonical form', () => {
    const q =
      'hr=ddr5-6400-8ch&hrc=512&ssd=nvme-t705-2tb&hl=4x8&ox=ssd&oxl=3&oxf=0.25&oxh=fine&olt=ram&ols=interleaved&oqd=32&opf=0&sd=dflash2&sdr=hyp&sdk=5&sda=0.8';
    const { state: s, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(encodeState(s)).toBe(q);
    expect(s.deployment.specDec).toEqual({ method: 'dflash2', drafter: 'hyp', k: 5, alpha: 0.8 });
  });

  it('token embeddings oet: ram and ssd round-trip next to the other offload keys; split and orphans are rejected', () => {
    expect(decodeState('oet=ram').state.deployment.offload).toEqual({
      embeddings: { tier: 'host_ram' },
    });
    expect(encodeState(decodeState('oet=ram').state)).toBe('oet=ram');
    expect(encodeState(decodeState('oet=ssd').state)).toBe('oet=ssd');
    // Written after the lookup-table keys and before the queue depth and prefetch keys.
    expect(encodeState(decodeState('opf=0.5&oqd=4&oet=ssd&olt=ram&ox=ram').state)).toBe(
      'ox=ram&olt=ram&oet=ssd&oqd=4&opf=0.5',
    );
    // A single block on stage 0: no split.
    const split = decodeState('oet=split');
    expect(split.dropped).toEqual(['oet=split']);
    expect(split.state.deployment.offload ?? {}).toEqual({});
    // oet alone turns the dependent keys on.
    expect(decodeState('oet=ssd&oqd=8&osr=sustained&opf=0.9').dropped).toEqual([]);
    expect(encodeState(decodeState('oet=ssd&oqd=8').state)).toBe('oet=ssd&oqd=8');
    // Absent = on the device, and the key is never written for it.
    expect(encodeState(decodeState('').state)).toBe('');
  });

  it('prefetch accuracy opf: round trip, range, parent; legacy oo maps to the equivalent p', () => {
    expect(decodeState('ox=ram&opf=0.9').state.deployment.offload).toEqual({
      experts: { tier: 'host_ram' },
      prefetchAccuracy: 0.9,
    });
    expect(encodeState(decodeState('ox=ram&opf=0.9').state)).toBe('ox=ram&opf=0.9');
    // The default 0.8 is not encoded.
    expect(encodeState(decodeState('ox=ram&opf=0.8').state)).toBe('ox=ram');
    // Out of range, or with nothing offloaded: dropped with a notice.
    expect(decodeState('ox=ram&opf=1.5').dropped).toEqual(['opf=1.5']);
    expect(decodeState('opf=0.5').dropped).toEqual(['opf=0.5']);
    // Legacy offload overlap: 'none' was no prefetching (p 0), 'ideal' perfect prediction (p 1).
    const none = decodeState('oo=none&ox=ram');
    expect(none.dropped).toEqual([]);
    expect(none.state.deployment.offload?.prefetchAccuracy).toBe(0);
    expect(encodeState(none.state)).toBe('ox=ram&opf=0');
    expect(encodeState(decodeState('oo=ideal&ox=ram').state)).toBe('ox=ram&opf=1');
    expect(encodeState(decodeState('oo=none&olt=ram').state)).toBe('olt=ram&opf=0');
    // An explicit opf wins over the legacy key.
    expect(encodeState(decodeState('oo=none&ox=ram&opf=0.7').state)).toBe('ox=ram&opf=0.7');
    // Without offload the legacy key had no effect: ignored without a notice, never encoded.
    const alone = decodeState('oo=none');
    expect(alone.dropped).toEqual([]);
    expect(encodeState(alone.state)).toBe('');
    expect(alone.state.options).toEqual({});
  });

  it('omits default deployment values', () => {
    const s = state({
      deployment: {
        offload: {
          prefetchAccuracy: 0.8,
          experts: { tier: 'host_ram', fromLayer: 0, residentFraction: 0, hitRatio: 'uniform' },
          tables: { tier: 'nvme', scaleLayout: 'pinned_host' },
        },
        specDec: { method: 'mtp', useCase: 'prose' },
      },
    });
    expect(encodeState(s)).toBe('ox=ram&olt=ssd&sd=mtp');
  });

  it('drops invalid deployment values and dependent keys without their parent', () => {
    const d = decodeState(
      'ox=gpu&oxf=0.5&olt=ram&ols=zig&oqd=0&sd=eagle&sdk=3&hr=ddr9&hrc=-1&ssd=floppy&hl=6x16&oo=maybe',
    );
    expect(d.dropped).toEqual([
      'hr=ddr9',
      'hrc=-1',
      'ssd=floppy',
      'hl=6x16',
      'ox=gpu',
      'oxf=0.5',
      'ols=zig',
      'oqd=0',
      'oo=maybe',
      'sd=eagle',
      'sdk=3',
    ]);
    expect(d.state.deployment).toEqual({ offload: { tables: { tier: 'host_ram' } } });
    expect(d.state.hw.host).toBeUndefined();
    const range = decodeState('sd=mtp&sdk=17&sda=0.995&sdr=a/b&sdu=poetry&oxh=fine&ols=pinned');
    expect(range.dropped).toEqual([
      'oxh=fine',
      'ols=pinned',
      'sdr=a/b',
      'sdk=17',
      'sdu=poetry',
      'sda=0.995',
    ]);
    expect(range.state.deployment).toEqual({ specDec: { method: 'mtp' } });
    expect(decodeState('ox=ram&oxf=1.5&oxh=1.2').dropped).toEqual(['oxf=1.5', 'oxh=1.2']);
  });

  it('host-RAM + SSD split: ox=split / olt=split with oxr, oxe, olr, ole; orphans dropped; old URLs unchanged', () => {
    const q = 'ox=split&oxl=4&oxr=20&oxe=last&olt=split&ols=interleaved&olr=10&ole=last';
    const d = decodeState(q);
    expect(d.dropped).toEqual([]);
    expect(d.state.deployment.offload).toEqual({
      experts: { tier: 'split', fromLayer: 4, split: { ramLayers: 20, ramEnd: 'last' } },
      tables: {
        tier: 'split',
        split: { ramLayers: 10, ramEnd: 'last' },
        scaleLayout: 'interleaved',
      },
    });
    expect(encodeState(d.state)).toBe(q);
    // `first` is the default: read, never written; a split without its keys is the default split.
    expect(encodeState(decodeState('ox=split&oxr=24&oxe=first').state)).toBe('ox=split&oxr=24');
    expect(decodeState('ox=split').state.deployment.offload).toEqual({
      experts: { tier: 'split' },
    });
    // A split keeps any scale layout (there is no single default; absent = each side's tier default).
    expect(encodeState(decodeState('olt=split&ols=separate').state)).toBe('olt=split&ols=separate');
    // The split keys need their group's split placement; invalid values are dropped.
    expect(decodeState('ox=ram&oxr=3&oxe=last&olt=ssd&olr=2&ole=last').dropped).toEqual([
      'oxr=3',
      'oxe=last',
      'olr=2',
      'ole=last',
    ]);
    expect(decodeState('oxr=3&olr=2').dropped).toEqual(['oxr=3', 'olr=2']);
    expect(decodeState('ox=split&oxr=-1&oxe=middle&olt=split&olr=1.5').dropped).toEqual([
      'oxr=-1',
      'oxe=middle',
      'olr=1.5',
    ]);
    // Old single-tier URLs decode as before.
    expect(decodeState('ox=ssd&oxl=3&olt=ram').state.deployment.offload).toEqual({
      experts: { tier: 'nvme', fromLayer: 3 },
      tables: { tier: 'host_ram' },
    });
    // The split counts as SSD offload for the SSD read rate.
    expect(encodeState(decodeState('ox=split&osr=sustained').state)).toBe('ox=split&osr=sustained');
    expect(encodeState(decodeState('olt=split&osr=sustained').state)).toBe(
      'olt=split&osr=sustained',
    );
  });

  it('loads an old URL with whole-layer offload (odl / odt, removed) and ignores it without a notice', () => {
    const old = decodeState('ox=ram&odl=40&odt=ssd');
    expect(old.dropped).toEqual([]);
    expect(old.state.deployment.offload).toEqual({ experts: { tier: 'host_ram' } });
    expect(encodeState(old.state)).toBe('ox=ram');
    // Alone it offloads nothing, so the offload-only keys that came with it are dropped as usual.
    const alone = decodeState('odl=0&odt=ssd&oqd=8');
    expect(alone.dropped).toEqual(['oqd=8']);
    expect(alone.state.deployment).toEqual({});
    expect(encodeState(alone.state)).toBe('');
  });

  it('drops hostInterface edits from x (set through the host link instead)', () => {
    const d = decodeState('x=hostInterface.lanes:8,memory.bandwidthBytesPerSec:2e12');
    expect(d.dropped).toEqual(['x=hostInterface.lanes:8']);
    expect(d.state.hw.deviceEdits).toEqual({ 'memory.bandwidthBytesPerSec': 2e12 });
  });
});
