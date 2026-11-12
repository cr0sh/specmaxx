import type { ReactNode } from 'react';
import { DEFAULT_PREFETCH_ACCURACY } from '../../sim/engine/transfer';
import {
  HOST_PCIE_LINKS,
  HOST_RAM_PRESETS,
  SSD_PRESETS,
  defaultHostRamFor,
  hostPcieLink,
} from '../../sim/hw/presets';
import { modelHasOffloadableKv, modelHasTopKKv } from '../../sim/lower/kvOffload';
import {
  defaultScaleLayout,
  formatBytesPerSec,
  hasLookupTables,
  hasMoe,
  offloadsAnything,
  offloadsToSsd,
  resolveHost,
  tableLayerIndices,
  tablesOffDeviceByDesign,
  TIED_EMBEDDINGS_NOTE,
  type HitRatioChoice,
  type HostChoice,
  KV_LINK_PRESETS,
  kvLinkIdOf,
  type KvCacheChoice,
  type KvTierChoice,
  type OffloadChoice,
  type ScaleLayoutChoice,
  type PlacementChoice,
  type TierChoice,
  type TierSplitChoice,
} from '../state';
import { HelpIcon } from './HelpIcon';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PickerOption } from './fuzzy';
import type { PanelProps } from './types';

/** Picker value of "on the device". */
const DEVICE = 'device';

/** Scale-layout picker value of "each side of a split uses its own tier's default" (`scaleLayout` absent). */
const TIER_DEFAULT = 'tier-default';

/** "?" help of the prefetch accuracy (docs/simulation-model.md §12 "Expert prefetch"). */
const PREFETCH_HELP = helpLines([
  'How often needed experts are guessed one layer early, so they load in advance.',
  'Missed experts load late and slow the step.',
  '100 means perfect guesses, 0 means no prefetching.',
]);

/** Tier labels. */
const TIER_LABELS: Record<PlacementChoice | typeof DEVICE, string> = {
  device: 'Device',
  host_ram: 'Host RAM',
  nvme: 'SSD',
  split: 'Host RAM + SSD',
};

/** "Host RAM holds" picker rows: which end of the offloaded layers is in host RAM. */
const RAM_END_OPTIONS: PickerOption[] = [
  { value: 'first', label: 'Earliest layers' },
  { value: 'last', label: 'Latest layers' },
];

/**
 * The host-RAM / SSD split fields of a group with `tier: 'split'`: how many of its `offloadedLayers` are in host RAM
 * (absent = half) and which end of them.
 */
function SplitFields({
  split,
  offloadedLayers,
  onChange,
}: {
  split: TierSplitChoice | undefined;
  offloadedLayers: number;
  onChange: (next: TierSplitChoice) => void;
}) {
  const k = Math.max(0, offloadedLayers);
  const ramLayers = Math.min(k, split?.ramLayers ?? Math.floor(k / 2));
  return (
    <div className="fields two" data-testid="split-fields">
      <NumberField
        label="Layers in host RAM"
        value={ramLayers}
        min={0}
        max={k}
        onChange={(v) =>
          onChange({ ...split, ramLayers: Number.isInteger(v) ? Math.min(k, Math.max(0, v)) : 0 })
        }
        help={helpLines([
          'How many offloaded layers are kept in host RAM.',
          'The other offloaded layers are on the SSD.',
        ])}
      />
      <Picker
        label="Host RAM holds"
        value={split?.ramEnd ?? 'first'}
        options={RAM_END_OPTIONS}
        onChange={(v) => onChange({ ...split, ramEnd: v as 'first' | 'last' })}
        help={helpLines([
          'Which end of the offloaded layers goes to host RAM.',
          'The other end goes to the SSD.',
        ])}
      />
    </div>
  );
}

/** Hit-ratio model labels (custom is a number in the state). */
const HIT_RATIO_OPTIONS: PickerOption[] = [
  { value: 'uniform', label: 'Uniform (h = f)' },
  { value: 'skewed_classic', label: 'Skewed, ≤ 64 experts' },
  { value: 'skewed_fine_grained', label: 'Skewed, ≥ 128 experts' },
  { value: 'custom', label: 'Custom' },
];

/** Scale layout labels. */
const SCALE_LAYOUT_OPTIONS: PickerOption[] = [
  { value: 'separate', label: 'Separate scale reads' },
  { value: 'pinned_host', label: 'Scales pinned in host RAM' },
  { value: 'interleaved', label: 'Interleaved rows' },
];

/** Scale layout labels on unified memory: the pinned scales stay in the device's own pool. */
const UNIFIED_SCALE_LAYOUT_OPTIONS: PickerOption[] = SCALE_LAYOUT_OPTIONS.map((o) =>
  o.value === 'pinned_host' ? { ...o, label: 'Scales pinned in memory' } : o,
);

/** Picker value of the default host link (the device's own interface). */
const DEFAULT_LINK = 'default';

/** `<gen>x<lanes>` of a host-link spec name `PCIe 5.0 x16 (host link)`. */
function linkKey(name: string): string {
  const m = /PCIe (\d)\.0 x(\d+)/.exec(name);
  return m ? `${m[1]}x${m[2]}` : name;
}

/** "PCIe 5.0 x16, 63.0 GB/s" of a host-link spec. */
function linkLabel(link: { name: string; bandwidthPerDirectionBytesPerSec: number }): string {
  return `${link.name.replace(' (host link)', '')}, ${formatBytesPerSec(link.bandwidthPerDirectionBytesPerSec)}`;
}

/** Host-RAM picker rows: grouped by DDR generation (DDR5 first, then DDR4), fastest first within a group. */
const HOST_RAM_OPTIONS: PickerOption[] = [...HOST_RAM_PRESETS]
  .map((r) => ({ r, generation: r.id.split('-')[0].toUpperCase() }))
  .sort(
    (a, b) =>
      (a.generation < b.generation ? 1 : a.generation > b.generation ? -1 : 0) ||
      b.r.bandwidthBytesPerSec - a.r.bandwidthBytesPerSec,
  )
  .map(({ r, generation }) => ({
    value: r.id,
    label: r.name,
    group: generation,
    description: `${(r.bandwidthBytesPerSec / 1e9).toFixed(1)} GB/s`,
  }));

/** Picker group of a device's own host RAM that is not a catalog preset (the pooled Grace LPDDR5X). */
const OWN_RAM_GROUP = 'This machine';

/** Help of the Host RAM picker on PCIe-attached GPUs and on Grace machines. */
const HOST_RAM_HELP = helpLines([
  'Speed of the machine’s system memory (RAM), from its datasheet.',
  'All GPUs in a machine share it.',
]);
const GRACE_RAM_HELP = helpLines([
  'Memory of the Grace CPUs: one CPU per two GPUs, pooled for the whole machine.',
  'All GPUs in a machine share it.',
]);

/** Help of the fixed host link of a device with a non-PCIe CPU link (Grace NVLink-C2C). */
const OWN_LINK_HELP = helpLines([
  'The link between each GPU and its CPU, built into the device.',
  'Offloaded data cannot stream faster than this link or the RAM allows.',
]);

/** A label with its "?" help and a value as plain text (a field with one possibility), styled like the setup row's. */
function StaticField({
  label,
  help,
  children,
}: {
  label: string;
  help: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <div className="field-head">
        <span className="field-label">{label}</span>
        <HelpIcon term={label}>{help}</HelpIcon>
      </div>
      <p className="field-static" data-testid="host-link-static">
        {children}
      </p>
    </div>
  );
}

/**
 * Shared state of the offload controls (the main section and its Advanced subsection): the model and device, the
 * current offload choice and its setter, and the tier picker rows (`placementOptions` adds "Host RAM + SSD" for the
 * experts and lookup tables). `undefined` before a model and device resolve.
 */
function offloadControls({ state, dispatch, loaded, built }: PanelProps) {
  const model = loaded?.model;
  const device = built.hardware?.device.device;
  if (model === undefined || device === undefined) return undefined;
  const unified = device.memory.kind === 'unified';
  const chosen = state.deployment.offload ?? {};
  // A tied model's token-embedding choice (from a shared URL) places nothing, so it is ignored here: it shows as the
  // default and turns no host control on.
  const offload: OffloadChoice = model.tieEmbeddings
    ? { ...chosen, embeddings: undefined }
    : chosen;
  const designOffDevice = tablesOffDeviceByDesign(model);
  const setOffload = (next: OffloadChoice) =>
    dispatch({
      type: 'setOffload',
      offload: next,
      tablesOffDeviceByDesign: designOffDevice && !unified,
    });
  const tierOptions: PickerOption[] = (
    unified ? [DEVICE, 'nvme'] : [DEVICE, 'host_ram', 'nvme']
  ).map((t) => ({
    value: t,
    label: TIER_LABELS[t as TierChoice | typeof DEVICE],
  }));
  const tier = (v: string): TierChoice | undefined =>
    v === DEVICE ? undefined : (v as TierChoice);
  /**
   * Rows of the experts / lookup-tables picker: the tiers plus the host-RAM + SSD split on dedicated memory when the
   * group has at least 2 layers to split (`splittable`). On unified memory host RAM is the device pool, so the split
   * is listed only while the state holds it (a shared URL); the same for a group with fewer than 2 layers.
   */
  const placementOptions = (
    current: PlacementChoice | undefined,
    splittable = true,
  ): PickerOption[] =>
    (!unified && splittable) || current === 'split'
      ? [...tierOptions, { value: 'split', label: TIER_LABELS.split }]
      : tierOptions;
  const placement = (v: string): PlacementChoice | undefined =>
    v === DEVICE ? undefined : (v as PlacementChoice);
  return {
    model,
    device,
    unified,
    offload,
    designOffDevice,
    setOffload,
    tierOptions,
    tier,
    placementOptions,
    placement,
    anyOff: offloadsAnything(offload),
  };
}

/**
 * Hardware → "Offload" (dedicated memory) / "Storage" (unified memory), design doc §6.2: where the MoE experts and
 * the expert cache live, and the host's RAM, SSD and PCIe link. On unified memory host RAM is the
 * device pool, so only the SSD is a tier and there is no host link. The less common knobs (lookup-table placement,
 * scale layout, queue depth, prefetch accuracy) are {@link OffloadAdvancedFields}, in the panel's Advanced.
 */
export function OffloadSection(props: PanelProps) {
  const { state, dispatch, built } = props;
  const controls = offloadControls(props);
  if (controls === undefined) return null;
  const { model, device, unified, offload, setOffload, placementOptions, placement, anyOff } =
    controls;
  const host = state.hw.host ?? {};
  const setHost = (next: Partial<HostChoice>) =>
    dispatch({ type: 'setHost', host: { ...host, ...next } });
  const resolvedHost = resolveHost(state.hw, device);
  // The device's own host RAM (Grace) is not a catalog preset: list it first, and picking it clears the choice.
  const ownRam = defaultHostRamFor(device, state.hw.devicesPerNode);
  const ownRamListed = HOST_RAM_PRESETS.some((r) => r.id === ownRam.id);
  const ramOptions: PickerOption[] = ownRamListed
    ? HOST_RAM_OPTIONS
    : [
        {
          value: ownRam.id,
          label: ownRam.name,
          group: OWN_RAM_GROUP,
          description: `${(ownRam.bandwidthBytesPerSec / 1e12).toFixed(2)} TB/s`,
        },
        ...HOST_RAM_OPTIONS,
      ];
  const layerCount = model.layers.length;
  const issues = built.issues.filter((i) => i.field === 'offload');
  const experts = offload.experts;
  const hitRatio: HitRatioChoice = experts?.hitRatio ?? 'uniform';
  return (
    <fieldset className="deployment-section" data-testid="offload-section">
      <legend>{unified ? 'Storage' : 'Offload'}</legend>
      {hasMoe(model) && (
        <>
          <Picker
            label="MoE experts"
            value={experts?.tier ?? DEVICE}
            options={placementOptions(experts?.tier)}
            onChange={(v) => {
              const t = placement(v);
              const offloaded = layerCount - (experts?.fromLayer ?? 0);
              setOffload({
                ...offload,
                experts:
                  t === undefined
                    ? undefined
                    : {
                        ...experts,
                        tier: t,
                        ...(t === 'split' && experts?.tier !== 'split'
                          ? { split: { ramLayers: Math.floor(offloaded / 2) } }
                          : {}),
                      },
              });
            }}
            help={helpLines([
              'Experts live in host RAM or SSD; some stay cached on the GPU.',
              'Each token fetches only the few experts it needs, if not cached.',
              'Usually the fastest way to fit a large MoE.',
            ])}
          />
          {experts !== undefined && (
            <div className="fields two">
              <NumberField
                label="From layer"
                value={experts.fromLayer ?? 0}
                min={0}
                max={layerCount - 1}
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: { ...experts, fromLayer: Number.isInteger(v) && v >= 0 ? v : 0 },
                  })
                }
                help={helpLines([
                  'Experts from this layer onward are offloaded.',
                  'Earlier layers keep their experts on the device.',
                ])}
              />
              <NumberField
                label="Resident %"
                value={Math.round((experts.residentFraction ?? 0) * 1000) / 10}
                min={0}
                max={100}
                step="1"
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: { ...experts, residentFraction: Math.min(1, Math.max(0, v / 100)) },
                  })
                }
                help={helpLines([
                  'Share of each offloaded layer’s experts kept in device memory.',
                  'More resident experts mean fewer slow transfers.',
                  'They count toward memory use.',
                ])}
              />
            </div>
          )}
          {experts?.tier === 'split' && (
            <SplitFields
              split={experts.split}
              offloadedLayers={layerCount - (experts.fromLayer ?? 0)}
              onChange={(split) => setOffload({ ...offload, experts: { ...experts, split } })}
            />
          )}
          {experts !== undefined && (
            <>
              <Picker
                label="Hit-ratio model"
                value={typeof hitRatio === 'number' ? 'custom' : hitRatio}
                options={HIT_RATIO_OPTIONS}
                onChange={(v) =>
                  setOffload({
                    ...offload,
                    experts: {
                      ...experts,
                      hitRatio:
                        v === 'custom' ? (experts.residentFraction ?? 0) : (v as HitRatioChoice),
                    },
                  })
                }
                help={helpLines([
                  'How often a needed expert is already on the device.',
                  'Uniform assumes experts are used evenly.',
                  'Skewed options assume some experts are used more often.',
                ])}
              />
              {typeof hitRatio === 'number' && (
                <NumberField
                  label="Hit ratio h"
                  value={hitRatio}
                  step="0.01"
                  min={0}
                  max={1}
                  onChange={(v) =>
                    setOffload({
                      ...offload,
                      experts: { ...experts, hitRatio: Math.min(1, Math.max(0, v)) },
                    })
                  }
                  help={helpLines([
                    'How often a needed expert is already on the device.',
                    '1 means always, 0 means never.',
                    'Higher is faster.',
                  ])}
                />
              )}
            </>
          )}
        </>
      )}
      {issues.length > 0 && (
        <span className="hint" role="status">
          {issues.map((i, n) => (
            <span className={i.level === 'error' ? 'error' : ''} key={n}>
              {i.message}
            </span>
          ))}
        </span>
      )}
      {anyOff && !unified && (
        <>
          <Picker
            label="Host RAM"
            value={resolvedHost.ram.id}
            options={ramOptions}
            searchThreshold={4}
            onChange={(v) =>
              setHost({
                ramId: !ownRamListed && v === ownRam.id ? undefined : v,
                ramCapacityBytes: undefined,
              })
            }
            help={!ownRamListed && resolvedHost.ramDefault ? GRACE_RAM_HELP : HOST_RAM_HELP}
          />
          <NumberField
            label="Host RAM capacity (GB)"
            value={resolvedHost.ram.capacityBytes / 1e9}
            min={1}
            onChange={(v) => setHost({ ramCapacityBytes: v > 0 ? Math.round(v * 1e9) : undefined })}
            help={helpLines([
              'Installed system RAM per machine.',
              'Offloaded weights must fit in it.',
            ])}
          />
        </>
      )}
      {anyOff && (
        <Picker
          label="SSD"
          value={resolvedHost.ssd.id}
          options={SSD_PRESETS.map((s) => ({ value: s.id, label: s.name }))}
          searchThreshold={4}
          onChange={(v) => setHost({ ssdId: v })}
          help={helpLines([
            'Speed of the NVMe SSD, from its datasheet.',
            'All GPUs in a machine share it.',
          ])}
        />
      )}
      {anyOff && !unified && device.hostLink !== undefined && resolvedHost.link !== undefined && (
        <StaticField label="Host link" help={OWN_LINK_HELP}>
          {`${resolvedHost.link.name.replace(/ \(.*\)$/, '')}, ${formatBytesPerSec(resolvedHost.link.bandwidthPerDirectionBytesPerSec)}`}
        </StaticField>
      )}
      {anyOff && !unified && device.hostLink === undefined && (
        <Picker
          label="Host link"
          value={host.link ? `${host.link.generation}x${host.link.lanes}` : DEFAULT_LINK}
          options={[
            {
              value: DEFAULT_LINK,
              label: device.hostInterface
                ? `${linkLabel(hostPcieLink(device.hostInterface.generation, device.hostInterface.lanes))} (device)`
                : 'Not specified by the device',
            },
            ...HOST_PCIE_LINKS.map((l) => ({ value: linkKey(l.name), label: linkLabel(l) })),
          ]}
          onChange={(v) => {
            if (v === DEFAULT_LINK) setHost({ link: undefined });
            else {
              const [g, lanes] = v.split('x').map(Number);
              setHost({ link: { generation: g as 3 | 4 | 5, lanes: lanes as 4 | 8 | 16 } });
            }
          }}
          help={helpLines([
            'The PCIe link between the GPU and system memory.',
            'Offloaded data cannot stream faster than this link or the RAM allows.',
          ])}
        />
      )}
    </fieldset>
  );
}

/** KV-cache tier picker rows: host RAM, the SSD, a split between them, or the remote KV pool. */
const KV_TIER_OPTIONS: PickerOption[] = [
  ...(['host_ram', 'nvme', 'split'] as const).map((t) => ({ value: t, label: TIER_LABELS[t] })),
  { value: 'remote_pool', label: 'Remote pool' },
];

/** The URL value of "the stack's own network link" in the pool-link picker (no preset chosen). */
const STACK_LINK = 'stack';

/** Pool-link picker rows: the stack's network link, then the RDMA NIC presets. */
const POOL_LINK_OPTIONS: PickerOption[] = [
  { value: STACK_LINK, label: 'Network link of the stack' },
  ...KV_LINK_PRESETS.map((l) => ({ value: kvLinkIdOf(l), label: l.name })),
];

/** KV-cache hit-model picker rows (top-k models; custom is a number in the state). */
const KV_HIT_OPTIONS: PickerOption[] = [
  { value: 'uniform', label: 'Uniform (h = GPU share)' },
  { value: 'custom', label: 'Custom' },
];

/** A share in [0, 1] as the percent the fields show, at most one decimal. */
function percentOf(share: number): number {
  return Math.round(share * 1000) / 10;
}

/** A typed percent as a share clamped to [0, `max`]. */
function shareOf(percent: number, max = 1): number {
  return Number.isFinite(percent) ? Math.min(max, Math.max(0, percent / 100)) : max;
}

/**
 * Hardware → Advanced → Offload: the KV-cache tiering fields (docs/simulation-model.md §12 "KV cache off-device",
 * "Remote KV pool"), for models whose KV cache grows with the context. The GPU share (100 = off) turns it on; then its
 * tier (dedicated memory: host RAM, SSD, split, remote pool; unified memory: SSD or remote pool), the host-RAM share of
 * a split, the pool's link and capacity, the hit model of top-k models, and the cached prefix's GPU share while the
 * prefix-cache hit ratio is above 0.
 */
function KvCacheFields(props: PanelProps) {
  const controls = offloadControls(props);
  if (controls === undefined || !modelHasOffloadableKv(controls.model)) return null;
  const { model, unified, offload, setOffload } = controls;
  const kv = offload.kvCache;
  const onDevice = unified ? 'in memory' : 'on GPU';
  const deviceFraction = kv?.deviceFraction ?? 1;
  const offDevice = 1 - deviceFraction;
  const prefixHits = (props.state.workload.prefixCacheHitRatio ?? 0) > 0;
  const setKv = (next: Partial<KvCacheChoice>) =>
    setOffload({
      ...offload,
      kvCache: { ...(kv ?? { tier: unified ? 'nvme' : 'host_ram' }), ...next },
    });
  // On unified memory host RAM is the device pool: the SSD and the remote pool are listed, plus a URL-carried tier.
  const tierOptions = unified
    ? KV_TIER_OPTIONS.filter(
        (o) => o.value === 'nvme' || o.value === 'remote_pool' || o.value === kv?.tier,
      )
    : KV_TIER_OPTIONS;
  const stackLink = props.built.hardware?.topology.interNode;
  const label = unified ? 'KV cache in memory %' : 'KV cache on GPU %';
  return (
    <div className="kv-fields" data-testid="kv-offload">
      <div className="fields two">
        <NumberField
          label={label}
          value={percentOf(deviceFraction)}
          min={0}
          max={100}
          step="1"
          onChange={(v) => setKv({ deviceFraction: shareOf(v) })}
          help={helpLines([
            `Share of each sequence’s KV cache kept ${onDevice}.`,
            unified
              ? 'The rest streams in from the SSD or a remote pool every step.'
              : 'The rest streams in from host RAM, the SSD or a remote pool every step.',
            'More sequences or a longer context fit, but decode can slow to the link speed.',
          ])}
        />
        {kv !== undefined && tierOptions.length > 1 && (
          <Picker
            label="KV cache tier"
            value={kv.tier}
            options={tierOptions}
            onChange={(v) => setKv({ tier: v as KvTierChoice })}
            help={helpLines([
              `Where the KV cache that is not ${onDevice} lives.`,
              'Host RAM is faster; the SSD holds more.',
              'A remote pool is memory on other machines, read over the network card.',
            ])}
          />
        )}
        {kv?.tier === 'remote_pool' && (
          <>
            <Picker
              label="Pool link"
              value={kv.poolLinkId ?? STACK_LINK}
              options={POOL_LINK_OPTIONS}
              onChange={(v) => setKv({ poolLinkId: v === STACK_LINK ? undefined : v })}
              help={helpLines([
                'The network card each GPU reads the pool through.',
                stackLink === undefined
                  ? 'This stack has no network link, so its default is one InfiniBand NDR 400 card per GPU.'
                  : `The stack’s link is ${stackLink.name}, shared by the GPUs of a machine.`,
                'Decode cannot go faster than this card moves the fetched KV.',
              ])}
            />
            <NumberField
              label="Pool capacity GB"
              value={kv.poolCapacityBytes === undefined ? 0 : kv.poolCapacityBytes / 1e9}
              min={0}
              step="1"
              onChange={(v) =>
                setKv({
                  poolCapacityBytes: Number.isFinite(v) && v > 0 ? Math.round(v * 1e9) : undefined,
                })
              }
              help={helpLines([
                'Memory of the pool shared by all GPUs, in GB.',
                '0 means unlimited: the pool always fits.',
              ])}
            />
          </>
        )}
        {kv?.tier === 'split' && (
          <NumberField
            label="KV in host RAM %"
            value={percentOf(Math.min(offDevice, kv.ramFraction ?? offDevice / 2))}
            min={0}
            max={percentOf(offDevice)}
            step="1"
            onChange={(v) => setKv({ ramFraction: shareOf(v, offDevice) })}
            help={helpLines([
              'Share of the KV cache kept in host RAM.',
              `The rest of what is not ${onDevice} is on the SSD.`,
            ])}
          />
        )}
        {kv !== undefined && modelHasTopKKv(model) && (
          <Picker
            label="KV hit model"
            value={kv.hitRatio === undefined ? 'uniform' : 'custom'}
            options={KV_HIT_OPTIONS}
            onChange={(v) => setKv({ hitRatio: v === 'custom' ? deviceFraction : undefined })}
            help={helpLines([
              `How often a token the attention picks is already ${onDevice}.`,
              `Uniform assumes picks are spread evenly, so it equals the share ${onDevice}.`,
            ])}
          />
        )}
        {kv?.hitRatio !== undefined && modelHasTopKKv(model) && (
          <NumberField
            label="KV hit ratio %"
            value={percentOf(kv.hitRatio)}
            min={0}
            max={100}
            step="1"
            onChange={(v) => setKv({ hitRatio: shareOf(v) })}
            help={helpLines([
              `How often a token the attention picks is already ${onDevice}.`,
              'Higher is faster.',
            ])}
          />
        )}
        {prefixHits && (
          <NumberField
            label={unified ? 'Prefix KV in memory %' : 'Prefix KV on GPU %'}
            value={percentOf(kv?.prefixDeviceFraction ?? deviceFraction)}
            min={0}
            max={100}
            step="1"
            onChange={(v) => setKv({ prefixDeviceFraction: shareOf(v) })}
            help={helpLines([
              `Share of a reused prompt prefix’s KV cache found ${onDevice}.`,
              'Prefill reads the rest from its tier (host RAM, SSD or remote pool) first.',
              `Starts equal to the KV cache ${onDevice} share.`,
            ])}
          />
        )}
      </div>
    </div>
  );
}
/**
 * Hardware → Advanced → "Offload" / "Storage": the token-embedding placement (every model; locked on the device for a
 * model with tied embeddings), the lookup-table placement (models with n-gram tables), the scale layout of off-device
 * tables, the queue depth and the prefetch accuracy (with any offload on).
 */
export function OffloadAdvancedFields(props: PanelProps) {
  const { state } = props;
  const controls = offloadControls(props);
  if (controls === undefined) return null;
  const {
    model,
    unified,
    offload,
    designOffDevice,
    setOffload,
    placementOptions,
    placement,
    tierOptions,
    tier,
    anyOff,
  } = controls;
  const tables = offload.tables;
  // A split counts table layers: it needs at least 2 (Qwen PLE has 1, DeepSeek Engram 2).
  const tableLayerCount = tableLayerIndices(model).length;
  const embeddingsTier = offload.embeddings?.tier ?? DEVICE;
  // On unified memory host RAM is the device pool, so a URL-carried host-RAM choice shows as the default row.
  const embeddingsValue = unified && embeddingsTier === 'host_ram' ? DEVICE : embeddingsTier;
  const embeddingsOptions: PickerOption[] = tierOptions.map((o) =>
    o.value === DEVICE ? { ...o, label: unified ? 'In memory' : 'On GPU' } : o,
  );
  const placeTablesOnHost = () =>
    setOffload({ ...offload, tables: { tier: unified ? 'nvme' : 'host_ram' } });
  return (
    <fieldset className="unlock advanced-fields" data-testid="offload-advanced">
      <legend>{unified ? 'Storage' : 'Offload'}</legend>
      <Picker
        label="Token embeddings"
        value={embeddingsValue}
        options={embeddingsOptions}
        disabled={model.tieEmbeddings}
        onChange={(v) => {
          const t = tier(v);
          setOffload({ ...offload, embeddings: t === undefined ? undefined : { tier: t } });
        }}
        help={helpLines([
          'Where the table that turns tokens into vectors lives.',
          ...(model.tieEmbeddings
            ? [TIED_EMBEDDINGS_NOTE]
            : [
                'Each token reads only one small row, so moving it off the GPU costs almost no speed.',
                'Frees GPU memory for a longer context or a bigger quant.',
              ]),
        ])}
      />
      {hasLookupTables(model) && (
        <div className="tables-field">
          <Picker
            label="Lookup tables"
            value={tables?.tier ?? DEVICE}
            options={placementOptions(tables?.tier, tableLayerCount >= 2)}
            onChange={(v) => {
              const t = placement(v);
              setOffload({
                ...offload,
                tables:
                  t === undefined
                    ? tables?.acrossDp === true
                      ? { acrossDp: true }
                      : undefined
                    : t === 'split'
                      ? {
                          tier: t,
                          split: tables?.split ?? {
                            ramLayers: Math.floor(tableLayerCount / 2),
                          },
                        }
                      : { tier: t },
              });
            }}
            help={helpLines([
              'Where the lookup tables are stored.',
              designOffDevice
                ? 'This model is built to keep its tables off the GPU.'
                : 'Only a few rows are read per token, so an SSD delay matters.',
            ])}
          />
          {designOffDevice && tables?.tier === undefined && (
            <button
              type="button"
              className="chip"
              onClick={placeTablesOnHost}
              data-testid="tables-hint"
            >
              {unified ? 'SSD ↗' : 'Host RAM ↗'}
            </button>
          )}
        </div>
      )}
      {hasLookupTables(model) && (state.parallel.dp > 1 || (props.otherPoolDp ?? 1) > 1) && (
        <Picker
          label="Share lookup tables across DP"
          value={tables?.tier === undefined && tables?.acrossDp === true ? 'on' : 'off'}
          options={[
            { value: 'off', label: 'off' },
            { value: 'on', label: 'on' },
          ]}
          disabled={tables?.tier !== undefined}
          onChange={(v) => setOffload({ ...offload, tables: { ...tables, acrossDp: v === 'on' } })}
          help={helpLines([
            props.pool === undefined
              ? 'Shares the tables between DP replicas instead of storing one copy per replica.'
              : 'Shares tables separately in each pool with DP above 1.',
            'Saves device memory but adds lookup traffic between devices.',
            'Set Lookup tables to Device to enable sharing.',
          ])}
        />
      )}
      {tables?.tier === 'split' && (
        <SplitFields
          split={tables.split}
          offloadedLayers={tableLayerCount}
          onChange={(split) => setOffload({ ...offload, tables: { ...tables, split } })}
        />
      )}
      {tables?.tier !== undefined && (
        <Picker
          label="Scale layout"
          value={
            tables.scaleLayout ??
            (tables.tier === 'split' ? TIER_DEFAULT : defaultScaleLayout(tables.tier))
          }
          options={[
            // A split has no single default: each side uses its own tier's layout.
            ...(tables.tier === 'split'
              ? [{ value: TIER_DEFAULT, label: 'Each tier’s default' }]
              : []),
            ...(unified ? UNIFIED_SCALE_LAYOUT_OPTIONS : SCALE_LAYOUT_OPTIONS),
          ]}
          onChange={(v) =>
            setOffload({
              ...offload,
              tables: {
                ...tables,
                scaleLayout: v === TIER_DEFAULT ? undefined : (v as ScaleLayoutChoice),
              },
            })
          }
          help={helpLines([
            'Separate: each row and its scales are two reads.',
            unified
              ? 'Pinned in memory: scales stay there, so only rows are read from the SSD.'
              : 'Pinned in host RAM: scales stay there, so only rows are read from the SSD.',
            'Interleaved: one read per row.',
          ])}
        />
      )}
      <KvCacheFields {...props} />
      {anyOff && (
        <NumberField
          label="Queue depth"
          value={offload.queueDepth ?? NaN}
          min={1}
          onChange={(v) =>
            setOffload({
              ...offload,
              queueDepth: Number.isInteger(v) && v >= 1 ? v : undefined,
            })
          }
          help={helpLines([
            'How many reads are in flight at once.',
            '1 means every read waits for the one before.',
            'Empty uses the drive’s rated maximum.',
          ])}
        />
      )}
      {offloadsToSsd(offload) && (
        <Picker
          label="SSD read rate"
          value={offload.ssdReadRate ?? 'datasheet'}
          options={[
            { value: 'datasheet', label: 'Datasheet' },
            { value: 'sustained', label: 'Sustained' },
          ]}
          onChange={(v) =>
            setOffload({ ...offload, ssdReadRate: v as OffloadChoice['ssdReadRate'] })
          }
          help={helpLines([
            'Datasheet uses the drive’s rated sequential read speed.',
            'Sustained uses a measured speed for large reads with few in flight, which is lower.',
            'Applies to experts and layers read from the SSD.',
          ])}
        />
      )}
      {anyOff && (
        <NumberField
          label="Prefetch accuracy %"
          value={Math.round((offload.prefetchAccuracy ?? DEFAULT_PREFETCH_ACCURACY) * 100)}
          min={0}
          max={100}
          step="1"
          onChange={(v) =>
            setOffload({
              ...offload,
              prefetchAccuracy: Number.isFinite(v) ? Math.min(1, Math.max(0, v / 100)) : undefined,
            })
          }
          help={PREFETCH_HELP}
        />
      )}
    </fieldset>
  );
}
