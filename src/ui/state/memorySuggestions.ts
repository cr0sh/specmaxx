/**
 * One-click memory fixes for the memory card (pure, no React): given a pool's memory report that does not fit, a
 * ranked short list of state actions that make it fit (or get closest), each checked with a `probe` that returns the
 * memory report the change would give. The probe is the sim itself in the app and a stub in the tests, so the list is
 * computed from the sim's numbers, not from a copy of its formulas.
 *
 * Rules (design: the memory card task):
 * - Device memory overflows: move one off-device-able group (n-gram tables, token embeddings, KV cache, MoE experts)
 *   to host RAM when that fits, else to the SSD. Only changes that fit or shrink the overflow; smallest change first.
 * - Host RAM overflows (dedicated memory only): the smallest Host RAM preset that holds the offloaded data, then a
 *   host RAM + SSD split by layer range, then the whole group on the SSD. No big enough preset: a notice.
 * - Unified memory has no separate host RAM: never a host-RAM size-up, groups go to the SSD.
 * - SSD overflows: nothing to offer.
 */
import type { TierMemory } from '../../sim/engine/types';
import { HOST_RAM_PRESETS } from '../../sim/hw/presets';
import type { HostMemorySpec } from '../../sim/hw/types';
import type { ModelSpec } from '../../sim/ir/model';
import { modelHasOffloadableKv } from '../../sim/lower/kvOffload';
import type { PoolAction } from './appState';
import type { HostChoice, KvCacheChoice, OffloadChoice, TierChoice } from './deployment';
import {
  hasLookupTables,
  hasMoe,
  tableLayerIndices,
  tablesOffDeviceByDesign,
} from './deploymentInput';
import { formatBytes } from './format';
import type { MemorySummary } from './summary';

/** A weight group or the KV cache that can live off the device. */
type Group = 'tables' | 'embeddings' | 'kv' | 'experts';

/** Display names of the groups, as the memory card names them. */
const GROUP_NAME: Record<Group, string> = {
  tables: 'n-gram tables',
  embeddings: 'Token embeddings',
  kv: 'KV cache',
  experts: 'MoE experts',
};

/** Groups in the order they are tried: the least costly to move first (rows read per token, then streamed KV, experts). */
const GROUP_ORDER: readonly Group[] = ['tables', 'embeddings', 'kv', 'experts'];

/** Share of each sequence's KV cache left on the device, tried in order (dimensionless, in (0, 1)). */
const KV_DEVICE_SHARES = [0.5, 0.25, 0] as const;

/** At most this many lines in a suggestion's help. Count. */
const MAX_HELP_LINES = 3;

/** At most this many suggestions are listed. Count. */
export const MAX_SUGGESTIONS = 3;

/** One one-click fix. */
export interface MemorySuggestion {
  /** Stable key, unique within a list. */
  id: string;
  /** Short button label, e.g. `Host RAM → 256 GB`, `n-gram tables → SSD`. */
  label: string;
  /** Lines of the "?" help: what changes and what it costs. One sentence each. */
  help: string[];
  /** The actions to dispatch, in order (the same ones the manual controls dispatch). */
  actions: PoolAction[];
  /** True when the probe says everything (every stage and every tier) fits after the change. */
  fits: boolean;
}

/** The suggestions of one memory report. */
export interface MemorySuggestions {
  /** Ranked fixes: fitting ones first, smallest change first; at most {@link MAX_SUGGESTIONS}. */
  suggestions: MemorySuggestion[];
  /** Short note when a fix is missing, e.g. no Host RAM preset is big enough; absent otherwise. */
  notice?: string;
}

/** Inputs of {@link memorySuggestions}. */
export interface MemorySuggestionInput {
  /** The model (which groups can move off the device, layer counts). */
  model: ModelSpec;
  /** The memory report of the pool, at its own phase and context. */
  memory: MemorySummary;
  /** The pool's current offload choice (`{}` = everything on the device). */
  offload: OffloadChoice;
  /** The pool's current host choice (`{}` = defaults). */
  host: HostChoice;
  /**
   * True with disaggregation on: the offload choice is one for both pools (the state keeps a single
   * `deployment.offload`), so an offload fix also changes the other pool; its help says so.
   */
  sharedOffload?: boolean;
  /** With `sharedOffload`: the other pool's current memory report. */
  otherMemory?: MemorySummary;
  /**
   * With `sharedOffload`: the other pool's memory report after the actions. An offload fix that makes that pool
   * overflow, or overflow more, is dropped.
   */
  probeOther?: (actions: PoolAction[]) => MemorySummary | undefined;
  /**
   * The memory report the pool would have after the actions, or undefined when they give no valid input. In the app:
   * the sim run on the changed state.
   */
  probe: (actions: PoolAction[]) => MemorySummary | undefined;
}

/**
 * The Host RAM presets that would hold `tier`'s data and are bigger than its installed capacity, smallest capacity
 * first (the fastest first among equals). Dedicated memory only. Unit of capacity: bytes.
 */
function biggerRamPresets(tier: TierMemory): HostMemorySpec[] {
  return HOST_RAM_PRESETS.filter(
    (p) => p.capacityBytes >= tier.bytes && p.capacityBytes > tier.capacityBytes,
  ).sort(
    (a, b) => a.capacityBytes - b.capacityBytes || b.bandwidthBytesPerSec - a.bandwidthBytesPerSec,
  );
}

/** Bytes by which `memory` is over capacity, summed over stages (per device) and tiers (per machine). Unit: bytes. */
function overflowBytes(memory: MemorySummary): number {
  const stages = memory.stages.reduce((n, s) => n + Math.max(0, s.totalBytes - s.capacityBytes), 0);
  const tiers = (memory.tiers ?? []).reduce(
    (n, t) => n + Math.max(0, t.bytes - t.capacityBytes),
    0,
  );
  return stages + tiers;
}

/** Largest per-device total over the stages. Unit: bytes. */
function peakDeviceBytes(memory: MemorySummary): number {
  return Math.max(...memory.stages.map((s) => s.totalBytes));
}

/** Short tier name for labels. */
const TIER_NAME: Record<TierChoice, string> = { host_ram: 'host RAM', nvme: 'SSD' };

/** `offload` with `group` placed by `placement` (the manual pickers' shapes); other groups unchanged. */
function withGroup(
  offload: OffloadChoice,
  group: Group,
  placement: OffloadChoice[keyof OffloadChoice],
): OffloadChoice {
  const key = { tables: 'tables', embeddings: 'embeddings', kv: 'kvCache', experts: 'experts' }[
    group
  ] as 'tables';
  return { ...offload, [key]: placement } as OffloadChoice;
}

/** The ranked one-click fixes of a memory report; empty when it fits. See the file comment for the rules. */
export function memorySuggestions(input: MemorySuggestionInput): MemorySuggestions {
  const {
    model,
    memory,
    offload,
    host,
    probe: rawProbe,
    sharedOffload = false,
    otherMemory,
    probeOther,
  } = input;
  if (memory.fits) return { suggestions: [] };
  const unified = memory.kind === 'unified';
  const tiers = memory.tiers ?? [];
  const ramTier = tiers.find((t) => t.tier === 'host_ram');
  const deviceOver = memory.stages.some((s) => !s.fits);
  const ramOver = !unified && ramTier !== undefined && !ramTier.fits;
  const before = overflowBytes(memory);
  const designOff = tablesOffDeviceByDesign(model) && !unified;
  /** Probes `actions`; an offload change that makes the other pool overflow (or overflow more) is rejected. */
  const probe = (actions: PoolAction[]): MemorySummary | undefined => {
    if (
      sharedOffload &&
      otherMemory !== undefined &&
      probeOther !== undefined &&
      actions.some((a) => a.type === 'setOffload')
    ) {
      const other = probeOther(actions);
      if (other === undefined || overflowBytes(other) > overflowBytes(otherMemory))
        return undefined;
    }
    return rawProbe(actions);
  };
  const setOffload = (next: OffloadChoice): PoolAction => ({
    type: 'setOffload',
    offload: next,
    tablesOffDeviceByDesign: designOff,
  });
  const candidates: { suggestion: MemorySuggestion; rank: number; remaining: number }[] = [];
  const seen = new Set<string>();
  const add = (
    suggestion: MemorySuggestion,
    rank: number,
    after: MemorySummary | undefined,
  ): void => {
    if (seen.has(suggestion.id)) return;
    if (!suggestion.fits && (after === undefined || overflowBytes(after) >= before)) return;
    seen.add(suggestion.id);
    const touchesOffload = suggestion.actions.some((a) => a.type === 'setOffload');
    candidates.push({
      suggestion:
        touchesOffload && sharedOffload
          ? {
              ...suggestion,
              // At most 3 lines: the last of a full help gives way to the shared note.
              help: [...suggestion.help.slice(0, MAX_HELP_LINES - 1), 'Applies to both pools.'],
            }
          : suggestion,
      rank,
      remaining: after === undefined ? 0 : overflowBytes(after),
    });
  };

  const supported: Record<Group, boolean> = {
    tables: hasLookupTables(model),
    embeddings: !model.tieEmbeddings,
    kv: modelHasOffloadableKv(model),
    experts: hasMoe(model),
  };
  const onDevice = unified ? 'in memory' : 'on GPU';

  /**
   * The changes to try for one group on `tier`, smallest first: a group still on the device is placed on the tier; a
   * partly offloaded KV cache or expert group offloads more (a lower device share, fewer resident experts, experts from
   * layer 0), on its own tier, or on the SSD when it is on host RAM and `tier` is the SSD. Tables and embeddings are
   * all-or-nothing, so once placed they have no further step.
   */
  const variants = (
    group: Group,
    tier: TierChoice,
  ): { id: string; label: string; placement: OffloadChoice[keyof OffloadChoice] }[] => {
    const fresh = { id: `${group}-${tier}`, label: `${GROUP_NAME[group]} → ${TIER_NAME[tier]}` };
    if (group === 'kv') {
      const kv = offload.kvCache;
      const current = kv?.deviceFraction ?? 1;
      // A split or a remote pool keeps its tier; host RAM may also move to the SSD.
      const fixed = kv?.tier === 'split' || kv?.tier === 'remote_pool';
      const moved = kv !== undefined && !fixed && kv.tier !== tier;
      const own = kv === undefined || fixed ? undefined : kv.tier;
      if (own !== undefined && own !== tier && own !== 'host_ram') return [];
      return KV_DEVICE_SHARES.filter((s) => s < current).map((deviceFraction) => ({
        id:
          kv === undefined
            ? `${fresh.id}-${deviceFraction}`
            : `kv-share-${deviceFraction}${moved ? '-nvme' : ''}`,
        label:
          kv === undefined
            ? fresh.label
            : `KV cache → ${Math.round(deviceFraction * 100)}% ${onDevice}${moved ? ', SSD' : ''}`,
        placement:
          kv === undefined
            ? { tier, deviceFraction }
            : { ...kv, tier: fixed ? kv.tier : tier, deviceFraction },
      }));
    }
    if (group === 'experts') {
      const experts = offload.experts;
      if (experts === undefined) return [{ ...fresh, placement: { tier } }];
      const moved = experts.tier !== 'split' && experts.tier !== tier;
      if (moved && experts.tier !== 'host_ram') return [];
      const at = experts.tier === 'split' ? 'split' : tier;
      const suffix = moved ? '-nvme' : '';
      const tag = moved ? ', SSD' : '';
      const resident = experts.residentFraction ?? 0;
      return [
        ...[resident / 2, 0]
          .filter((r, i, all) => r < resident && all.indexOf(r) === i)
          .map((r) => ({
            id: `experts-resident-${r}${suffix}`,
            label: `MoE experts → ${Math.round(r * 100)}% resident${tag}`,
            placement: { ...experts, tier: at, residentFraction: r } as const,
          })),
        ...((experts.fromLayer ?? 0) > 0
          ? [
              {
                id: `experts-from-0${suffix}`,
                label: `MoE experts → from layer 0${tag}`,
                placement: { ...experts, tier: at, fromLayer: 0 } as const,
              },
            ]
          : []),
      ];
    }
    return offload[group === 'tables' ? 'tables' : 'embeddings']?.tier === undefined
      ? [{ ...fresh, placement: { tier } }]
      : [];
  };

  if (deviceOver) {
    for (const group of GROUP_ORDER) {
      if (!supported[group]) continue;
      // A group already off the device keeps its tier; one still on it goes to host RAM if that fits, else the SSD.
      const current: string | undefined =
        group === 'kv' ? offload.kvCache?.tier : offload[group]?.tier;
      // A group on host RAM may also move to the SSD when more host placement cannot fit.
      const tierOrder: TierChoice[] =
        current === 'nvme'
          ? ['nvme']
          : current === 'split' || current === 'remote_pool'
            ? ['host_ram']
            : unified
              ? ['nvme']
              : ['host_ram', 'nvme'];
      let best: { suggestion: MemorySuggestion; after: MemorySummary; freed: number } | undefined;
      for (const tier of tierOrder) {
        for (const { id, label, placement } of variants(group, tier)) {
          const actions = [setOffload(withGroup(offload, group, placement))];
          const after = probe(actions);
          if (after === undefined) continue;
          const freed = peakDeviceBytes(memory) - peakDeviceBytes(after);
          const suggestion: MemorySuggestion = {
            id,
            label,
            help: [
              // Named from the placement the action sets (a split or a remote pool keeps its tier).
              group === 'kv'
                ? `Keeps ${placementShare(placement)} of the KV cache ${onDevice}, the rest ${placedWhere(placement)}.`
                : current === undefined || tier !== current
                  ? `Moves ${groupText(group)} off the device ${placedWhere(placement)}.`
                  : `Offloads more of ${groupText(group)} ${placedWhere(placement)}.`,
              freed > 0 ? `Frees ${formatBytes(freed)} per device.` : 'Frees no device memory.',
              'Reads from it can slow decode.',
            ],
            actions,
            fits: after.fits,
          };
          if (best === undefined || remainingOf(after) < remainingOf(best.after)) {
            best = { suggestion, after, freed };
          }
          // The smallest step that fits wins within a group.
          if (after.fits) break;
        }
        // Host RAM that fits wins; otherwise the SSD is tried next.
        if (best?.after.fits) break;
      }
      // Fitting device fixes rank by the probed memory they free, smallest first.
      if (best !== undefined) add(best.suggestion, Math.max(0, best.freed), best.after);
    }
  }

  if (ramOver && ramTier !== undefined) {
    // 1. The smallest Host RAM preset that holds the offloaded data.
    const preset = biggerRamPresets(ramTier)[0];
    if (preset !== undefined) {
      const actions: PoolAction[] = [
        { type: 'setHost', host: { ...host, ramId: preset.id, ramCapacityBytes: undefined } },
      ];
      const after = probe(actions);
      add(
        {
          id: `ram-${preset.id}`,
          label: `Host RAM → ${formatCapacity(preset.capacityBytes)}`,
          help: [
            `Picks ${preset.name} as the host RAM.`,
            `It holds ${formatBytes(preset.capacityBytes)}; the data needs ${formatBytes(ramTier.bytes)}.`,
            'Its speed may differ from the current RAM.',
          ],
          actions,
          fits: after?.fits ?? true,
        },
        -3,
        after,
      );
    }
    // 2. Groups on host RAM: a host RAM + SSD split by layer range, then all on the SSD.
    const onRam: { group: Group; bytes: number }[] = [
      { group: 'experts', bytes: ramTier.breakdown.routedExperts },
      { group: 'tables', bytes: ramTier.breakdown.lookupTables },
      { group: 'embeddings', bytes: ramTier.breakdown.tokenEmbeddings },
      { group: 'kv', bytes: ramTier.breakdown.kvCache },
    ];
    for (const { group, bytes } of onRam) {
      if (bytes <= 0) continue;
      const split = splitPlacement(group, bytes);
      if (split !== undefined) {
        const actions = [setOffload(withGroup(offload, group, split))];
        const after = probe(actions);
        if (after !== undefined) {
          add(
            {
              id: `${group}-split`,
              label: `${GROUP_NAME[group]} → RAM + SSD`,
              help: [
                `Keeps the first layers of the ${GROUP_NAME[group].toLowerCase()} in host RAM and puts the rest on the SSD.`,
                'The share in RAM is the most that fits.',
              ],
              actions,
              fits: after.fits,
            },
            -2,
            after,
          );
        }
      }
      const toSsd = ssdPlacement(group);
      if (toSsd !== undefined) {
        const actions = [setOffload(withGroup(offload, group, toSsd))];
        const after = probe(actions);
        if (after !== undefined) {
          add(
            {
              id: `${group}-nvme`,
              label: `${GROUP_NAME[group]} → SSD`,
              help: [
                `Moves the ${GROUP_NAME[group].toLowerCase()} from host RAM to the SSD.`,
                `Frees ${formatBytes(bytes)} of host RAM.`,
                'The SSD is slower than RAM.',
              ],
              actions,
              fits: after.fits,
            },
            -1,
            after,
          );
        }
      }
    }
  }

  // A device fix that fits makes the device fixes that only shrink the overflow noise: they are dropped.
  const deviceFits = candidates.some((c) => c.rank >= 0 && c.suggestion.fits);
  const ranked = candidates
    .filter((c) => !(deviceFits && c.rank >= 0 && !c.suggestion.fits))
    .sort(
      (a, b) =>
        Number(b.suggestion.fits) - Number(a.suggestion.fits) ||
        (a.suggestion.fits ? a.rank - b.rank : a.remaining - b.remaining) ||
        a.rank - b.rank,
    );
  // Fixes that fit come first; partial ones (a smaller overflow) fill the rest of the list.
  const chosen = ranked.slice(0, MAX_SUGGESTIONS);
  const noPreset = ramOver && ramTier !== undefined && biggerRamPresets(ramTier).length === 0;
  return {
    suggestions: chosen.map((c) => c.suggestion),
    ...(noPreset ? { notice: 'No Host RAM size holds this' } : {}),
  };

  /** Overflow of a report, for ranking the partial fixes. Unit: bytes. */
  function remainingOf(report: MemorySummary): number {
    return overflowBytes(report);
  }

  /** The host RAM + SSD split of an offloaded group sized so the RAM part fits, or undefined when none helps. */
  function splitPlacement(
    group: Group,
    groupBytes: number,
  ): OffloadChoice[keyof OffloadChoice] | undefined {
    if (ramTier === undefined) return undefined;
    const layers =
      group === 'experts'
        ? model.layers.length - (offload.experts?.fromLayer ?? 0)
        : group === 'tables'
          ? tableLayerIndices(model).length
          : 0;
    if ((group !== 'experts' && group !== 'tables') || layers < 2) return undefined;
    if (offload[group] === undefined || offload[group]?.tier !== 'host_ram') return undefined;
    // Formula: RAM layers r = ⌊K · (capacity − other groups) / group bytes⌋, in [1, K − 1]
    const available = ramTier.capacityBytes - (ramTier.bytes - groupBytes);
    const ramLayers = Math.floor((layers * available) / groupBytes);
    if (ramLayers < 1 || ramLayers >= layers) return undefined;
    return group === 'experts'
      ? { ...offload.experts!, tier: 'split', split: { ramLayers } }
      : { tier: 'split', split: { ramLayers } };
  }

  /** The group's current choice moved to the SSD, or undefined when it is not on host RAM. */
  function ssdPlacement(group: Group): OffloadChoice[keyof OffloadChoice] | undefined {
    switch (group) {
      case 'experts':
        return offload.experts?.tier === 'host_ram'
          ? { ...offload.experts, tier: 'nvme' }
          : undefined;
      case 'tables':
        return offload.tables?.tier === 'host_ram' ? { tier: 'nvme' } : undefined;
      case 'embeddings':
        return offload.embeddings?.tier === 'host_ram' ? { tier: 'nvme' } : undefined;
      case 'kv':
        return offload.kvCache?.tier === 'host_ram'
          ? { ...offload.kvCache, tier: 'nvme' }
          : undefined;
    }
  }
}

/** A group's name inside a sentence (`MoE experts` keeps its capitals). */
function groupText(group: Group): string {
  return group === 'experts' ? GROUP_NAME[group] : GROUP_NAME[group].toLowerCase();
}

/** Where a placement puts its data, as a phrase: `on host RAM`, `on the SSD`, `on host RAM and the SSD`, `in the remote pool`. */
function placedWhere(placement: unknown): string {
  switch ((placement as { tier?: string }).tier) {
    case 'host_ram':
      return 'on host RAM';
    case 'nvme':
      return 'on the SSD';
    case 'split':
      return 'on host RAM and the SSD';
    case 'remote_pool':
      return 'in the remote pool';
    default:
      return 'off the device';
  }
}

/** The device share of a KV placement as a percentage, e.g. `50%`. */
function placementShare(placement: unknown): string {
  const share = (placement as KvCacheChoice).deviceFraction ?? 1;
  return `${Math.round(share * 100)}%`;
}

/** A host-RAM capacity as the label shows it, e.g. `256 GB`, `1 TB`. Unit in: bytes. */
function formatCapacity(bytes: number): string {
  const gb = Math.round(bytes / 1e9);
  return gb >= 1000 && gb % 1000 === 0 ? `${gb / 1000} TB` : `${gb} GB`;
}
