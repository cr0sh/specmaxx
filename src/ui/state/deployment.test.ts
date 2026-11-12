import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf';
import type { HfApiModelInfo } from '../../sim/models/hf/types';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import flashApi from '../../sim/models/__fixtures__/Qwen__Qwen3.8-Flash-Next.api.json?raw';
import flashConfig from '../../sim/models/__fixtures__/Qwen__Qwen3.8-Flash-Next.config.json?raw';
import {
  DDR5_6400_8CH,
  DEFAULT_HOST_RAM,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  hostPcieLink,
} from '../../sim/hw/presets';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { buildSimInput, loadedFromHf, type LoadedModel } from './buildInput';
import {
  normalizeDeployment,
  normalizeOffload,
  offloadsAnything,
  offloadsToSsd,
  splitTierRanges,
  type OffloadChoice,
} from './deployment';
import {
  drafterSource,
  lookupTableParams,
  tableLayerIndices,
  tablesOffDeviceByDesign,
} from './deploymentInput';
import { defaultSsdFor, editableFields, findDevice, resolveHardware } from './hardware';

const apply = (s: AppState, ...actions: Parameters<typeof reduce>[1][]) =>
  actions.reduce(reduce, s);

/** Qwen3.8-Flash-Next (512 experts, PLE table off-device by design), mapped from the committed fixtures. */
function flashNext(): LoadedModel {
  const id = 'Qwen/Qwen3.8-Flash-Next';
  const r = mapHfModel({
    id,
    revision: 'main',
    config: JSON.parse(flashConfig) as unknown,
    api: JSON.parse(flashApi) as HfApiModelInfo,
  });
  if (r.kind !== 'ok') throw new Error(r.kind);
  return loadedFromHf({ kind: 'hf', id }, r);
}
const FLASH = flashNext();
const FLASH_STATE: AppState = {
  ...DEFAULT_STATE,
  model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
};

describe('deployment normalizers', () => {
  it('a sharing URL on a model without tables explains sharing, not placement', () => {
    const state = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { tables: { acrossDp: true } },
    });
    const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
    expect(built.kind).toBe('ok');
    expect(built.issues).toContainEqual({
      field: 'offload',
      level: 'warning',
      message: 'The model has no n-gram lookup tables; sharing has no effect.',
    });
    if (built.kind === 'ok') expect(built.input.deployment?.offload?.lookupTables).toBeUndefined();
  });

  it('drop default values, and drop a queue depth when nothing is offloaded', () => {
    expect(
      normalizeOffload({
        experts: { tier: 'host_ram', fromLayer: 0, residentFraction: 0, hitRatio: 'uniform' },
        tables: { tier: 'nvme', scaleLayout: 'pinned_host' },
      }),
    ).toEqual({ experts: { tier: 'host_ram' }, tables: { tier: 'nvme' } });
    expect(normalizeOffload({ queueDepth: 4 })).toEqual({});
    expect(
      normalizeDeployment({
        offload: {},
        specDec: { method: 'mtp', drafter: 'a/b', useCase: 'prose' },
      }),
    ).toEqual({ specDec: { method: 'mtp' } });
    expect(
      normalizeDeployment({ specDec: { method: 'dflash2', useCase: 'code', alpha: 0.8 } }),
    ).toEqual({ specDec: { method: 'dflash2', alpha: 0.8 } });
  });

  it('keep a split only with the split placement, without the default ramEnd', () => {
    expect(
      normalizeOffload({
        experts: { tier: 'nvme', split: { ramLayers: 3, ramEnd: 'last' } },
        tables: { tier: 'split', split: { ramEnd: 'first' }, scaleLayout: 'separate' },
      }),
    ).toEqual({
      experts: { tier: 'nvme' },
      // A split keeps its scale layout: there is no single default to drop.
      tables: { tier: 'split', scaleLayout: 'separate' },
    });
    expect(
      normalizeOffload({ experts: { tier: 'split', split: { ramLayers: 0, ramEnd: 'last' } } }),
    ).toEqual({ experts: { tier: 'split', split: { ramLayers: 0, ramEnd: 'last' } } });
    // A split reads from the SSD, so the SSD read rate stays.
    expect(normalizeOffload({ experts: { tier: 'split' }, ssdReadRate: 'sustained' })).toEqual({
      experts: { tier: 'split' },
      ssdReadRate: 'sustained',
    });
  });

  it('splitTierRanges: r of K layers in host RAM at either end; an empty side is left out', () => {
    expect(splitTierRanges(4, 48, { ramLayers: 20 })).toEqual([
      { tier: 'host_ram', layers: { from: 4, to: 24 } },
      { tier: 'nvme', layers: { from: 24, to: 48 } },
    ]);
    expect(splitTierRanges(4, 48, { ramLayers: 20, ramEnd: 'last' })).toEqual([
      { tier: 'nvme', layers: { from: 4, to: 28 } },
      { tier: 'host_ram', layers: { from: 28, to: 48 } },
    ]);
    // Default: half, rounded down (K = 7 → 3 in RAM).
    expect(splitTierRanges(0, 7, undefined)).toEqual([
      { tier: 'host_ram', layers: { from: 0, to: 3 } },
      { tier: 'nvme', layers: { from: 3, to: 7 } },
    ]);
    expect(splitTierRanges(0, 10, { ramLayers: 0 })).toEqual([
      { tier: 'nvme', layers: { from: 0, to: 10 } },
    ]);
    // More than K: clamped to every offloaded layer.
    expect(splitTierRanges(2, 10, { ramLayers: 99 })).toEqual([
      { tier: 'host_ram', layers: { from: 2, to: 10 } },
    ]);
  });
});

describe('reduce: deployment', () => {
  it('turning offload on prefills host-RAM tables for a model whose table is off-device by design, once', () => {
    const on = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { experts: { tier: 'host_ram' } },
      tablesOffDeviceByDesign: true,
    });
    expect(on.deployment.offload).toEqual({
      experts: { tier: 'host_ram' },
      tables: { tier: 'host_ram' },
    });
    // The user moves the tables back to the device; changing the experts later does not prefill again.
    const back = apply(
      on,
      {
        type: 'setOffload',
        offload: { experts: { tier: 'host_ram' } },
        tablesOffDeviceByDesign: true,
      },
      { type: 'setOffload', offload: { experts: { tier: 'nvme' } }, tablesOffDeviceByDesign: true },
    );
    expect(back.deployment.offload).toEqual({ experts: { tier: 'nvme' } });
    // Without the flag nothing is prefilled.
    expect(
      reduce(DEFAULT_STATE, { type: 'setOffload', offload: { experts: { tier: 'host_ram' } } })
        .deployment.offload,
    ).toEqual({
      experts: { tier: 'host_ram' },
    });
  });

  it('turning everything off leaves deployment {}', () => {
    const s = apply(
      DEFAULT_STATE,
      { type: 'setOffload', offload: { experts: { tier: 'host_ram' }, queueDepth: 8 } },
      { type: 'setSpecDec', specDec: { method: 'mtp' } },
      { type: 'setOffload', offload: { queueDepth: 8 } },
      { type: 'setSpecDec', specDec: undefined },
    );
    expect(s.deployment).toEqual({});
  });

  it('changing the model drops the drafter repo; changing the device keeps host RAM and SSD but not the host link', () => {
    const s = apply(
      DEFAULT_STATE,
      {
        type: 'setSpecDec',
        specDec: { method: 'dflash2', drafter: 'incoai/Qwen3.8-27B-DFlash2', k: 5 },
      },
      {
        type: 'setHost',
        host: { ramId: 'ddr5-6400-8ch', ssdId: 'nvme-t705-2tb', link: { generation: 4, lanes: 8 } },
      },
      { type: 'setModel', model: { kind: 'hf', id: 'Qwen/Qwen3-8B' } },
      { type: 'setDevice', deviceId: 'rtx-4090' },
    );
    expect(s.deployment.specDec).toEqual({ method: 'dflash2', k: 5 });
    expect(s.hw.host).toEqual({ ramId: 'ddr5-6400-8ch', ssdId: 'nvme-t705-2tb' });
    expect(reduce(s, { type: 'setHost', host: {} }).hw.host).toBeUndefined();
  });
});

describe('hardware: host', () => {
  it('excludes hostInterface from the editable device fields', () => {
    const fields = editableFields(findDevice('rtx-5090')!);
    expect(fields.some((f) => f.path.startsWith('hostInterface'))).toBe(false);
    expect(fields.length).toBeGreaterThan(0);
  });

  it('builds stack.host only when something is off-device', () => {
    expect(resolveHardware(DEFAULT_STATE.hw).stack.host).toBeUndefined();
    expect(resolveHardware(DEFAULT_STATE.hw, true).stack.host).toEqual({
      link: hostPcieLink(5, 16),
      ram: DEFAULT_HOST_RAM,
      ssd: DEFAULT_SSD,
    });
    const hw = {
      ...DEFAULT_STATE.hw,
      host: {
        ramId: DDR5_6400_8CH.id,
        ramCapacityBytes: 512e9,
        link: { generation: 4 as const, lanes: 8 as const },
      },
    };
    const host = resolveHardware(hw, true).stack.host!;
    expect(host.link).toEqual(hostPcieLink(4, 8));
    expect(host.ram?.capacityBytes).toBe(512e9);
    expect(host.ram?.bandwidthBytesPerSec).toBe(DDR5_6400_8CH.bandwidthBytesPerSec);
  });

  it("on unified memory gives only the SSD, by default the machine's own", () => {
    const spark = { ...DEFAULT_STATE.hw, deviceId: 'dgx-spark' };
    expect(defaultSsdFor(findDevice('dgx-spark')!)).toBe(DGX_SPARK_SSD_4TB);
    expect(defaultSsdFor(findDevice('rtx-5090')!)).toBe(DEFAULT_SSD);
    expect(resolveHardware(spark, true).stack.host).toEqual({ ssd: DGX_SPARK_SSD_4TB });
  });

  it('gives no host link for a device without a host interface', () => {
    const h100 = { ...DEFAULT_STATE.hw, deviceId: 'h100-sxm' };
    expect(resolveHardware(h100, true).stack.host?.link).toBeUndefined();
  });
});

describe('buildSimInput: deployment', () => {
  it('omits deployment and host with every knob off', () => {
    const built = buildSimInput(FLASH_STATE, FLASH);
    expect(built.kind).toBe('ok');
    if (built.kind !== 'ok') return;
    expect('deployment' in built.input).toBe(false);
    expect(built.input.hardware.host).toBeUndefined();
    // A table placement on a model without tables emits nothing either (a warning).
    const noTables = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { tables: { tier: 'host_ram' } },
    });
    const b2 = buildSimInput(noTables, BUNDLED_DEFAULT_MODEL);
    expect(b2.kind === 'ok' && 'deployment' in b2.input).toBe(false);
    expect(b2.issues.some((i) => i.field === 'offload' && i.level === 'warning')).toBe(true);
  });

  it('emits the token-embedding placement, and for a tied model nothing', () => {
    const embed = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { embeddings: { tier: 'nvme' } },
    });
    expect(BUNDLED_DEFAULT_MODEL.model.tieEmbeddings).toBe(false);
    const built = buildSimInput(embed, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error('not ok');
    expect(built.issues).toEqual([]);
    expect(built.input.deployment).toEqual({ offload: { embeddings: { tier: 'nvme' } } });
    const tied = {
      ...BUNDLED_DEFAULT_MODEL,
      model: { ...BUNDLED_DEFAULT_MODEL.model, tieEmbeddings: true },
    };
    const b2 = buildSimInput(embed, tied);
    expect(b2.kind === 'ok' && 'deployment' in b2.input).toBe(false);
    expect(b2.issues).toEqual([]);
  });

  it('counts the token-embedding table as an offload and keeps it through normalization', () => {
    expect(offloadsAnything({ embeddings: { tier: 'host_ram' } })).toBe(true);
    expect(offloadsToSsd({ embeddings: { tier: 'host_ram' } })).toBe(false);
    expect(offloadsToSsd({ embeddings: { tier: 'nvme' } })).toBe(true);
    expect(normalizeOffload({ embeddings: { tier: 'nvme' }, queueDepth: 8 })).toEqual({
      embeddings: { tier: 'nvme' },
      queueDepth: 8,
    });
  });

  it('emits the offload config and the host when experts are offloaded', () => {
    const s = reduce(FLASH_STATE, {
      type: 'setOffload',
      offload: {
        experts: {
          tier: 'host_ram',
          fromLayer: 4,
          residentFraction: 0.25,
          hitRatio: 'skewed_fine_grained',
        },
        queueDepth: 16,
      },
    });
    const built = buildSimInput(s, FLASH);
    expect(built.issues.filter((i) => i.level === 'error')).toEqual([]);
    if (built.kind !== 'ok') throw new Error('not ok');
    expect(built.input.deployment).toEqual({
      offload: {
        routedExperts: {
          tier: 'host_ram',
          layers: { from: 4, to: FLASH.model.layers.length },
          residentFraction: 0.25,
          hitRatio: { kind: 'skewed_fine_grained' },
        },
        queueDepth: 16,
      },
    });
    expect(built.input.hardware.host?.ram).toBe(DEFAULT_HOST_RAM);
  });

  it('removed whole-layer offload: a state that still carries `dense` places nothing for it', () => {
    const s = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { dense: { tier: 'nvme', layersOnDevice: 40 } } as OffloadChoice,
    });
    expect(s.deployment).toEqual({});
    const built = buildSimInput(s, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    expect(built.input.deployment).toBeUndefined();
  });

  it('a host-RAM + SSD expert split becomes two rules with the same f and h; an empty side gives the object form', () => {
    const L = FLASH.model.layers.length;
    const config = (experts: NonNullable<OffloadChoice['experts']>) => {
      const built = buildSimInput(
        reduce(FLASH_STATE, { type: 'setOffload', offload: { experts } }),
        FLASH,
      );
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      return built.input.deployment?.offload?.routedExperts;
    };
    const cache = { residentFraction: 0.25, hitRatio: { kind: 'skewed_fine_grained' } } as const;
    expect(
      config({
        tier: 'split',
        split: { ramLayers: 10 },
        fromLayer: 4,
        residentFraction: 0.25,
        hitRatio: 'skewed_fine_grained',
      }),
    ).toEqual([
      { tier: 'host_ram', layers: { from: 4, to: 14 }, ...cache },
      { tier: 'nvme', layers: { from: 14, to: L }, ...cache },
    ]);
    expect(config({ tier: 'split', split: { ramLayers: 10, ramEnd: 'last' } })).toEqual([
      {
        tier: 'nvme',
        layers: { from: 0, to: L - 10 },
        residentFraction: 0,
        hitRatio: { kind: 'uniform' },
      },
      {
        tier: 'host_ram',
        layers: { from: L - 10, to: L },
        residentFraction: 0,
        hitRatio: { kind: 'uniform' },
      },
    ]);
    // r = 0 or r = K: the single-tier object form, exactly as with that tier picked.
    expect(config({ tier: 'split', split: { ramLayers: 0 } })).toEqual(config({ tier: 'nvme' }));
    expect(config({ tier: 'split', split: { ramLayers: L } })).toEqual({
      tier: 'host_ram',
      residentFraction: 0,
      hitRatio: { kind: 'uniform' },
    });
    expect(config({ tier: 'split', split: { ramLayers: L - 4 }, fromLayer: 4 })).toEqual(
      config({ tier: 'host_ram', fromLayer: 4 }),
    );
  });

  it('a lookup-table split counts table layers; ranges cover every layer; an empty side gives the object form', () => {
    // DeepSeek-V4.1-Flash has Engram tables at layers 1 and 14 (Qwen3.8-Flash-Next has a single table layer).
    const id = 'deepseek-ai/DeepSeek-V4.1-Flash';
    const mapped = mapHfModel(fixtureRepoFiles(id, { dir: '' }));
    if (mapped.kind !== 'ok') throw new Error(mapped.kind);
    const ds = loadedFromHf({ kind: 'hf', id }, mapped);
    const L = ds.model.layers.length;
    expect(tableLayerIndices(ds.model)).toEqual([1, 14]);
    expect(tableLayerIndices(FLASH.model)).toHaveLength(1);
    const tables = (choice: NonNullable<OffloadChoice['tables']>) => {
      const built = buildSimInput(
        reduce(
          { ...DEFAULT_STATE, model: { kind: 'hf', id } },
          { type: 'setOffload', offload: { tables: choice } },
        ),
        ds,
      );
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      const policy = built.input.deployment?.offload?.lookupTables;
      expect(policy).toHaveProperty('placements');
      return policy !== undefined && 'placements' in policy ? policy.placements : undefined;
    };
    // 1 of the 2 table layers in host RAM (the default half): the boundary is the 2nd table layer, 14.
    expect(tables({ tier: 'split', scaleLayout: 'interleaved' })).toEqual([
      { tier: 'host_ram', layers: { from: 0, to: 14 }, scaleLayout: 'interleaved' },
      { tier: 'nvme', layers: { from: 14, to: L }, scaleLayout: 'interleaved' },
    ]);
    expect(tables({ tier: 'split', split: { ramLayers: 1, ramEnd: 'last' } })).toEqual([
      { tier: 'nvme', layers: { from: 0, to: 14 } },
      { tier: 'host_ram', layers: { from: 14, to: L } },
    ]);
    expect(tables({ tier: 'split', split: { ramLayers: 0 } })).toEqual(tables({ tier: 'nvme' }));
    expect(tables({ tier: 'split', split: { ramLayers: 2 } })).toEqual({ tier: 'host_ram' });
    // Scales pinned in host RAM apply to the SSD side only; the host-RAM side keeps its tier default (its scales are
    // already in RAM). Other layouts apply to both sides.
    expect(tables({ tier: 'split', scaleLayout: 'pinned_host' })).toEqual([
      { tier: 'host_ram', layers: { from: 0, to: 14 } },
      { tier: 'nvme', layers: { from: 14, to: L }, scaleLayout: 'pinned_host' },
    ]);
    expect(tables({ tier: 'split', split: { ramLayers: 2 }, scaleLayout: 'pinned_host' })).toEqual({
      tier: 'host_ram',
    });
    expect(tables({ tier: 'split', split: { ramLayers: 0 }, scaleLayout: 'pinned_host' })).toEqual({
      tier: 'nvme',
      scaleLayout: 'pinned_host',
    });
    expect(tables({ tier: 'split', scaleLayout: 'separate' })).toEqual([
      { tier: 'host_ram', layers: { from: 0, to: 14 }, scaleLayout: 'separate' },
      { tier: 'nvme', layers: { from: 14, to: L }, scaleLayout: 'separate' },
    ]);
  });

  it.each(['host_ram', 'nvme', 'remote_pool'] as const)(
    'shared tables preserve the selected host for KV-only %s offload',
    (tier) => {
      const s = reduce(FLASH_STATE, {
        type: 'setOffload',
        offload: {
          tables: { acrossDp: true },
          kvCache: {
            tier,
            deviceFraction: 0.5,
            ...(tier === 'remote_pool' ? { poolLinkId: 'infiniband-hdr200-one-nic-per-gpu' } : {}),
          },
          queueDepth: 8,
        },
      });
      const built = buildSimInput(s, FLASH);
      expect(built.kind).toBe('ok');
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      expect(built.input.hardware.host).toBeDefined();
      expect(built.input.deployment?.offload?.kvCache?.deviceFraction).toBe(0.5);
      expect(built.input.deployment?.offload?.queueDepth).toBe(8);
      expect(built.input.deployment?.offload?.lookupTables).toEqual({ acrossDp: true });
    },
  );

  it('shared tables with prefix-only KV offload resolve a host only for a cache-hit workload', () => {
    const state = reduce(FLASH_STATE, {
      type: 'setOffload',
      offload: {
        tables: { acrossDp: true },
        kvCache: { tier: 'host_ram', prefixDeviceFraction: 0.25 },
      },
    });
    for (const hit of [0, 0.5]) {
      const built = buildSimInput(
        { ...state, workload: { ...state.workload, prefixCacheHitRatio: hit } },
        FLASH,
      );
      expect(built.kind).toBe('ok');
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      expect(built.input.hardware.host !== undefined).toBe(hit > 0);
    }
  });

  it('shared device tables survive normalization without a host and are not auto-offloaded with experts', () => {
    const shared = { tables: { acrossDp: true } };
    const s = reduce(FLASH_STATE, {
      type: 'setOffload',
      offload: shared,
      tablesOffDeviceByDesign: true,
    });
    expect(s.deployment.offload).toEqual(shared);
    expect(
      normalizeOffload({ ...shared, queueDepth: 8, prefetchAccuracy: 1, ssdReadRate: 'sustained' }),
    ).toEqual(shared);
    const built = buildSimInput(s, FLASH);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    expect(built.input.deployment?.offload).toEqual({ lookupTables: { acrossDp: true } });
    expect(built.input.hardware.host).toBeUndefined();
    expect(offloadsAnything(s.deployment.offload)).toBe(false);
    const experts = reduce(s, {
      type: 'setOffload',
      offload: { ...shared, experts: { tier: 'host_ram' } },
      tablesOffDeviceByDesign: true,
    });
    expect(experts.deployment.offload?.tables).toEqual({ acrossDp: true });
    expect(normalizeOffload({ tables: { acrossDp: false } })).toEqual({});
    for (const tier of ['host_ram', 'nvme', 'split'] as const) {
      const moved = reduce(s, {
        type: 'setOffload',
        offload: { tables: { tier, acrossDp: true } },
      });
      expect(moved.deployment.offload?.tables).toEqual({ tier });
    }
  });

  it('MTP on the default model uses its head, the default k and the prose preset', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setSpecDec', specDec: { method: 'mtp' } });
    const built = buildSimInput(s, BUNDLED_DEFAULT_MODEL);
    if (BUNDLED_DEFAULT_MODEL.model.draft?.kind !== 'mtp') {
      expect(built.issues.some((i) => i.field === 'specDec' && i.level === 'error')).toBe(true);
      return;
    }
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const sd = built.input.deployment?.speculative;
    expect(sd?.method).toBe('mtp');
    expect(sd?.drafter).toBeUndefined();
    expect(sd?.draftLength).toBeGreaterThanOrEqual(1);
    expect(sd?.acceptance.kind).toBeDefined();
    const custom = buildSimInput(
      reduce(s, { type: 'setSpecDec', specDec: { method: 'mtp', k: 2, alpha: 0.6 } }),
      BUNDLED_DEFAULT_MODEL,
    );
    if (custom.kind !== 'ok') throw new Error(JSON.stringify(custom.issues));
    expect(custom.input.deployment?.speculative).toMatchObject({
      draftLength: 2,
      acceptance: { kind: 'iid', alpha: 0.6 },
    });
  });

  it('a block method waits for its drafter repo, then uses the loaded spec; the hypothetical one needs no fetch', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setSpecDec', specDec: { method: 'dflash2' } });
    const source = drafterSource(s.deployment.specDec, BUNDLED_DEFAULT_MODEL.model, s.model.id);
    expect(source).toEqual({ kind: 'repo', repoId: 'incoai/Qwen3.8-27B-DFlash2', known: true });
    const loading = buildSimInput(s, BUNDLED_DEFAULT_MODEL, {
      kind: 'loading',
      repoId: 'incoai/Qwen3.8-27B-DFlash2',
    });
    expect(loading.kind).toBe('issues');
    expect(
      loading.issues.some((i) => i.field === 'specDec' && /Loading drafter/.test(i.message)),
    ).toBe(true);
    const failed = buildSimInput(s, BUNDLED_DEFAULT_MODEL, {
      kind: 'error',
      repoId: 'incoai/Qwen3.8-27B-DFlash2',
      message: 'gone (404)',
    });
    expect(failed.issues.some((i) => /gone \(404\)/.test(i.message))).toBe(true);
    const hyp = reduce(s, { type: 'setSpecDec', specDec: { method: 'dflash2', drafter: 'hyp' } });
    expect(
      drafterSource(hyp.deployment.specDec, BUNDLED_DEFAULT_MODEL.model, hyp.model.id),
    ).toEqual({ kind: 'hypothetical' });
    const built = buildSimInput(hyp, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    expect(built.input.deployment?.speculative?.drafter?.kind).toBe('hypothetical');
    expect('host' in built.input.hardware).toBe(false);
  });
});

describe('lookup tables', () => {
  it('counts the PLE table params of Qwen3.8-Flash-Next and flags it off-device by design', () => {
    expect(lookupTableParams(FLASH.model) / 1e9).toBeCloseTo(51.2, 1);
    const flagged = {
      ...FLASH.model,
      layers: FLASH.model.layers.map((l) => ({
        ...l,
        blocks: l.blocks.map((b) =>
          b.kind === 'ngram_embedding' ? { ...b, offDeviceByDesign: true } : b,
        ),
      })),
    };
    expect(tablesOffDeviceByDesign(flagged)).toBe(true);
    expect(tablesOffDeviceByDesign(BUNDLED_DEFAULT_MODEL.model)).toBe(false);
    expect(lookupTableParams(BUNDLED_DEFAULT_MODEL.model)).toBe(0);
  });
});
