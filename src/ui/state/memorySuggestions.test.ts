import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../../sim/models/hf/index';
import type { PoolAction } from './appState';
import { loadedFromHf } from './buildInput';
import type { OffloadChoice } from './deployment';
import { memorySuggestions } from './memorySuggestions';
import type { MemorySummary, StageMemory } from './summary';

function load(id: string) {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id }, r).model;
}
/** MoE + n-gram tables (Qwen PLE), untied embeddings. */
const MOE_TABLES = load('Qwen/Qwen3.8-Flash-Next');
/** Dense, no tables. */
const DENSE = load('Qwen/Qwen3.8-27B');

const GB = 1e9;

function stage(totalGb: number, capacityGb = 80): StageMemory {
  return {
    stage: 0,
    weightBytes: totalGb * 0.8 * GB,
    sequenceStateBytes: totalGb * 0.1 * GB,
    activationPeakBytes: totalGb * 0.1 * GB,
    totalBytes: totalGb * GB,
    capacityBytes: capacityGb * GB,
    fits: totalGb <= capacityGb,
  };
}

/** A memory report; `tiers` as `[tier, usedGb, capacityGb, breakdownGb]`. */
function memory(
  stages: StageMemory[],
  tiers: [
    'host_ram' | 'nvme',
    number,
    number,
    Partial<{
      lookupTables: number;
      routedExperts: number;
      tokenEmbeddings: number;
      kvCache: number;
    }>?,
  ][] = [],
  kind: 'dedicated' | 'unified' = 'dedicated',
): MemorySummary {
  const t = tiers.map(([name, used, cap, breakdown]) => ({
    tier: name,
    bytes: used * GB,
    capacityBytes: cap * GB,
    fits: used <= cap,
    breakdown: {
      lookupTables: 0,
      routedExperts: 0,
      draftExperts: 0,
      tokenEmbeddings: 0,
      kvCache: 0,
      ...Object.fromEntries(Object.entries(breakdown ?? {}).map(([k, v]) => [k, v * GB])),
    },
  }));
  return {
    kind,
    fits: stages.every((s) => s.fits) && t.every((x) => x.fits),
    maxSequencesPerReplica: 1,
    stages,
    tiers: t,
  };
}

/** The offload choice a probe's actions set (the last `setOffload`). */
function offloadOf(actions: PoolAction[]): OffloadChoice | undefined {
  for (const a of [...actions].reverse()) if (a.type === 'setOffload') return a.offload;
  return undefined;
}

describe('memorySuggestions', () => {
  it('fitting memory has no suggestions', () => {
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(60)]),
      offload: {},
      host: {},
      probe: () => undefined,
    });
    expect(r.suggestions).toEqual([]);
  });

  it('device overflow: moves the tables to host RAM when that fits, smallest change first', () => {
    // Tables hold 30 GB of the 100 GB; experts could free 50 GB.
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      if (o.tables?.tier === 'host_ram') return memory([stage(70)], [['host_ram', 30, 96]]);
      if (o.experts?.tier === 'host_ram') return memory([stage(50)], [['host_ram', 50, 96]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual([
      'n-gram tables → host RAM',
      'MoE experts → host RAM',
    ]);
    expect(r.suggestions.every((s) => s.fits)).toBe(true);
    const first = r.suggestions[0].actions[0];
    expect(first.type === 'setOffload' && first.offload.tables).toEqual({ tier: 'host_ram' });
    expect(r.suggestions[0].help.join(' ')).toContain('Frees 30.0GB per device');
  });

  it('device overflow: shared tables remain eligible for a host-placement fix that clears sharing', () => {
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)]),
      offload: { tables: { acrossDp: true } },
      host: {},
      probe: (actions) =>
        offloadOf(actions)?.tables?.tier === 'host_ram'
          ? memory([stage(70)], [['host_ram', 30, 96]])
          : undefined,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['n-gram tables → host RAM']);
    expect(offloadOf(r.suggestions[0].actions)?.tables).toEqual({ tier: 'host_ram' });
  });

  it('device overflow: uses the SSD when the data does not fit the host RAM; drops fixes that do not help', () => {
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      // Tables: 30 GB; the host RAM holds 16 GB only.
      if (o.tables?.tier === 'host_ram') return memory([stage(70)], [['host_ram', 30, 16]]);
      if (o.tables?.tier === 'nvme') return memory([stage(70)], [['nvme', 30, 2000]]);
      // Embeddings free nothing: not offered.
      if (o.embeddings !== undefined) return memory([stage(100)], [['host_ram', 1, 16]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['n-gram tables → SSD']);
  });

  it('device overflow: shows the closest when nothing fits', () => {
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      if (o.experts?.tier === 'host_ram') return memory([stage(90)], [['host_ram', 10, 96]]);
      if (o.embeddings?.tier === 'host_ram') return memory([stage(95)], [['host_ram', 5, 96]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => [s.label, s.fits])).toEqual([
      ['MoE experts → host RAM', false],
      ['Token embeddings → host RAM', false],
    ]);
  });

  it('host RAM overflow: the smallest preset that holds the data, then tables on the SSD', () => {
    const current = memory([stage(60)], [['host_ram', 100, 96, { lookupTables: 100 }]]);
    const probe = (actions: PoolAction[]) => {
      const a = actions[0];
      if (a.type === 'setHost') return memory([stage(60)], [['host_ram', 100, 128]]);
      const o = offloadOf(actions)!;
      if (o.tables?.tier === 'nvme') return memory([stage(60)], [['nvme', 100, 2000]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: current,
      offload: { tables: { tier: 'host_ram' } },
      host: { ramCapacityBytes: 96 * GB },
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['Host RAM → 128 GB', 'n-gram tables → SSD']);
    const size = r.suggestions[0].actions[0];
    // Same state action the Host RAM picker dispatches: a preset id, the capacity override cleared.
    expect(size).toMatchObject({ type: 'setHost', host: { ramId: 'ddr5-4800-4ch' } });
    expect(size.type === 'setHost' && size.host.ramCapacityBytes).toBeUndefined();
    expect(r.notice).toBeUndefined();
  });

  it('host RAM overflow: no preset is large enough, so only the SSD move and a notice', () => {
    const current = memory([stage(60)], [['host_ram', 900, 96, { routedExperts: 900 }]]);
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      return o.experts?.tier === 'nvme' ? memory([stage(60)], [['nvme', 900, 2000]]) : undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: current,
      offload: { experts: { tier: 'host_ram' } },
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['MoE experts → SSD']);
    expect(r.notice).toBe('No Host RAM size holds this');
    expect(r.suggestions.flatMap((s) => s.actions).some((a) => a.type === 'setHost')).toBe(false);
  });

  it('host RAM overflow: offers the host RAM + SSD split by layer range when it fits', () => {
    const current = memory([stage(60)], [['host_ram', 900, 96, { routedExperts: 900 }]]);
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      if (o.experts?.tier === 'split')
        return memory(
          [stage(60)],
          [
            ['host_ram', 90, 96],
            ['nvme', 810, 2000],
          ],
        );
      if (o.experts?.tier === 'nvme') return memory([stage(60)], [['nvme', 900, 2000]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: current,
      offload: { experts: { tier: 'host_ram' } },
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual([
      'MoE experts → RAM + SSD',
      'MoE experts → SSD',
    ]);
    const split = r.suggestions[0].actions[0];
    const layers = MOE_TABLES.layers.length;
    // ⌊K · (96 GB − 0) / 900 GB⌋ layers stay in RAM.
    expect(split.type === 'setOffload' && split.offload.experts).toEqual({
      tier: 'split',
      split: { ramLayers: Math.floor((layers * 96) / 900) },
    });
  });

  it('unified memory never suggests a host-RAM size-up: groups go to the SSD', () => {
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions);
      if (o?.tables?.tier === 'nvme') return memory([stage(70)], [['nvme', 30, 4000]], 'unified');
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)], [], 'unified'),
      offload: {},
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['n-gram tables → SSD']);
    const all = r.suggestions.flatMap((s) => s.actions);
    expect(all.some((a) => a.type === 'setHost')).toBe(false);
    expect(r.suggestions[0].label).not.toMatch(/host RAM/i);
  });

  it('SSD overflow: nothing to offer', () => {
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(60)], [['nvme', 3000, 2000, { routedExperts: 3000 }]]),
      offload: { experts: { tier: 'nvme' } },
      host: {},
      probe: () => memory([stage(60)], [['nvme', 3000, 2000]]),
    });
    expect(r).toEqual({ suggestions: [] });
  });

  it('only offers what the model supports', () => {
    const asked: string[] = [];
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      probe: (actions) => {
        const o = offloadOf(actions)!;
        asked.push(Object.keys(o).join());
        return o.kvCache?.tier === 'host_ram'
          ? memory([stage(70)], [['host_ram', 30, 96]])
          : undefined;
      },
    });
    // No tables, no experts on a dense model; its KV cache can tier.
    expect(new Set(asked)).toEqual(new Set(['embeddings', 'kvCache']));
    expect(r.suggestions.map((s) => s.label)).toEqual(['KV cache → host RAM']);
  });
  it('raises the offloaded share of a partly offloaded KV cache, keeping its tier', () => {
    const asked: number[] = [];
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)], [['host_ram', 20, 96, { kvCache: 20 }]]),
      offload: { kvCache: { tier: 'host_ram', deviceFraction: 0.5 } },
      host: {},
      probe: (actions) => {
        const kv = offloadOf(actions)?.kvCache;
        if (kv?.tier === 'host_ram' && kv.deviceFraction !== 0.5)
          asked.push(kv.deviceFraction ?? 1);
        return kv?.deviceFraction === 0.25
          ? memory([stage(70)], [['host_ram', 40, 96]])
          : undefined;
      },
    });
    expect(asked).toEqual([0.25]);
    expect(r.suggestions.map((s) => s.label)).toEqual(['KV cache → 25% on GPU']);
    const a = r.suggestions[0].actions[0];
    expect(a.type === 'setOffload' && a.offload.kvCache).toEqual({
      tier: 'host_ram',
      deviceFraction: 0.25,
    });
  });

  it('offers fewer resident experts for partly offloaded experts', () => {
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)], [['host_ram', 50, 96, { routedExperts: 50 }]]),
      offload: { experts: { tier: 'host_ram', residentFraction: 0.5 } },
      host: {},
      probe: (actions) =>
        offloadOf(actions)?.experts?.residentFraction === 0
          ? memory([stage(60)], [['host_ram', 50, 96]])
          : undefined,
    });
    expect(r.suggestions.map((s) => [s.label, s.fits])).toEqual([
      ['MoE experts → 0% resident', true],
    ]);
  });

  it('ranks fitting device fixes by the memory they free, smallest first', () => {
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      if (o.tables?.tier === 'host_ram') return memory([stage(60)], [['host_ram', 40, 96]]);
      if (o.experts?.tier === 'host_ram') return memory([stage(78)], [['host_ram', 22, 96]]);
      return undefined;
    };
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      probe,
    });
    expect(r.suggestions.map((s) => s.label)).toEqual([
      'MoE experts → host RAM',
      'n-gram tables → host RAM',
    ]);
  });

  it('says an offload fix applies to both pools when the offload is shared', () => {
    const run = (sharedOffload: boolean) =>
      memorySuggestions({
        model: MOE_TABLES,
        memory: memory([stage(100)]),
        offload: {},
        host: {},
        sharedOffload,
        probe: (actions) =>
          offloadOf(actions)?.tables?.tier === 'host_ram'
            ? memory([stage(70)], [['host_ram', 30, 96]])
            : undefined,
      }).suggestions[0].help;
    expect(run(true)).toContain('Applies to both pools.');
    expect(run(true).length).toBeLessThanOrEqual(3);
    expect(run(false)).not.toContain('Applies to both pools.');
  });
  it('drops a shared offload fix that makes the other pool overflow, keeps one that does not', () => {
    const probeOther = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      // The other pool fits at 60 GB; tables on host RAM push it to 90 GB, experts leave it alone.
      return o.tables !== undefined ? memory([stage(90)]) : memory([stage(55)]);
    };
    const probe = (actions: PoolAction[]) => {
      const o = offloadOf(actions)!;
      if (o.tables?.tier === 'host_ram') return memory([stage(70)], [['host_ram', 30, 96]]);
      if (o.experts?.tier === 'host_ram') return memory([stage(50)], [['host_ram', 50, 96]]);
      return undefined;
    };
    const base = { model: MOE_TABLES, memory: memory([stage(100)]), offload: {}, host: {}, probe };
    const labels = (extra: object) =>
      memorySuggestions({ ...base, ...extra }).suggestions.map((s) => s.label);
    expect(labels({})).toEqual(['n-gram tables → host RAM', 'MoE experts → host RAM']);
    expect(labels({ sharedOffload: true, otherMemory: memory([stage(60)]), probeOther })).toEqual([
      'MoE experts → host RAM',
    ]);
    // An unusable other-pool probe is as bad as an overflow.
    expect(
      labels({
        sharedOffload: true,
        otherMemory: memory([stage(60)]),
        probeOther: () => undefined,
      }),
    ).toEqual([]);
  });

  it('falls back to the SSD when raising the offloaded KV share on host RAM cannot fit it', () => {
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)], [['host_ram', 10, 96, { kvCache: 10 }]]),
      offload: { kvCache: { tier: 'host_ram', deviceFraction: 0.5 } },
      host: {},
      probe: (actions) => {
        const kv = offloadOf(actions)?.kvCache;
        if (kv?.deviceFraction !== 0.25) return undefined;
        // 40 GB of KV does not fit 16 GB of host RAM, but the SSD holds it.
        return kv.tier === 'host_ram'
          ? memory([stage(70)], [['host_ram', 40, 16]])
          : memory([stage(70)], [['nvme', 40, 2000]]);
      },
    });
    expect(r.suggestions.map((s) => [s.label, s.fits])).toEqual([
      ['KV cache → 25% on GPU, SSD', true],
    ]);
    const a = r.suggestions[0].actions[0];
    expect(a.type === 'setOffload' && a.offload.kvCache).toEqual({
      tier: 'nvme',
      deviceFraction: 0.25,
    });
  });

  it('falls back to the SSD for experts on host RAM when fewer resident experts overflow it', () => {
    const r = memorySuggestions({
      model: MOE_TABLES,
      memory: memory([stage(100)], [['host_ram', 50, 96, { routedExperts: 50 }]]),
      offload: { experts: { tier: 'host_ram', residentFraction: 0.5 } },
      host: {},
      probe: (actions) => {
        const e = offloadOf(actions)?.experts;
        if (e?.residentFraction !== 0) return undefined;
        return e.tier === 'host_ram'
          ? memory([stage(60)], [['host_ram', 120, 96]])
          : memory([stage(60)], [['nvme', 120, 2000]]);
      },
    });
    expect(r.suggestions.map((s) => [s.label, s.fits])).toEqual([
      ['MoE experts → 0% resident, SSD', true],
    ]);
  });
  it('keeps a remote KV pool tier when raising the offloaded KV share', () => {
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)]),
      offload: { kvCache: { tier: 'remote_pool', deviceFraction: 0.5, poolLinkId: 'x' } },
      host: {},
      probe: (actions) => {
        const kv = offloadOf(actions)?.kvCache;
        return kv?.deviceFraction === 0.25 && kv.tier === 'remote_pool'
          ? memory([stage(70)])
          : undefined;
      },
    });
    expect(r.suggestions.map((s) => s.label)).toEqual(['KV cache → 25% on GPU']);
    // The help names the pool the placement keeps, not host RAM.
    expect(r.suggestions[0].help[0]).toBe(
      'Keeps 25% of the KV cache on GPU, the rest in the remote pool.',
    );
    expect(r.suggestions[0].help.join(' ')).not.toMatch(/host RAM/);
    expect(r.suggestions[0].help.length).toBeLessThanOrEqual(3);
  });

  it('names a host RAM + SSD split in the help of a suggestion that keeps it', () => {
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)]),
      offload: { kvCache: { tier: 'split', deviceFraction: 0.5 } },
      host: {},
      probe: (actions) => {
        const kv = offloadOf(actions)?.kvCache;
        return kv?.deviceFraction === 0.25 && kv.tier === 'split' ? memory([stage(70)]) : undefined;
      },
    });
    expect(r.suggestions[0].help[0]).toBe(
      'Keeps 25% of the KV cache on GPU, the rest on host RAM and the SSD.',
    );
    expect(r.suggestions[0].help.length).toBeLessThanOrEqual(3);
  });

  it('keeps a shared KV fix help within 3 lines, ending with the shared note', () => {
    const r = memorySuggestions({
      model: DENSE,
      memory: memory([stage(100)]),
      offload: {},
      host: {},
      sharedOffload: true,
      otherMemory: memory([stage(60)]),
      probeOther: () => memory([stage(55)]),
      probe: (actions) =>
        offloadOf(actions)?.kvCache?.tier === 'host_ram' ? memory([stage(70)]) : undefined,
    });
    const help = r.suggestions[0].help;
    expect(help).toHaveLength(3);
    expect(help[0]).toMatch(/^Keeps 50% of the KV cache on GPU, the rest on host RAM\.$/);
    expect(help[2]).toBe('Applies to both pools.');
  });
});
