/**
 * Hardware resolution for the UI: the state's `HardwareChoice` → the effective device, links and cabling → the
 * `HardwareStack` the engine simulates. Also the default links per device (from the setup catalog,
 * `presets/catalog.ts`), the setup row's options (counts, links, cablings), the popular setups, and the editable
 * spec fields with their display units.
 *
 * Order of application (docs/simulation-model.md §4 "Device variants and modifiers", "Unlock axes"):
 * preset → unlock modifiers → user edits. Edits are absolute values at `numericLeafPaths` paths applied
 * LAST, so the user's number wins over an unlock. An edited spec gets id `<id>-custom`, name
 * `<name> (edited)` and one `estimated` entry per edited path ("edited by the user"), so it still passes
 * `validateDevice` / `validateLink` and is honestly marked.
 */

import type {
  DeviceSpec,
  EstimatedField,
  HardwareStack,
  HostInterfaceSpec,
  HostMemorySpec,
  HostSpec,
  LinkSpec,
  NvmeSpec,
  InNetworkReduction,
  TopologySpec,
} from '../../sim/hw/types';
import type { LinkModifier } from '../../sim/hw/modifiers';
import {
  applyDeviceModifiers,
  applyLinkModifiers,
  describeDeviceModifier,
  describeLinkModifier,
} from '../../sim/hw/modifiers';
import { resolveUnlocks } from '../../sim/hw/unlocks';
import { numericLeafPaths } from '../../sim/hw/validate';
import { DEVICE_PRESETS, deviceUnlocks } from '../../sim/hw/presets/index';
import {
  LEGACY_LINK_IDS,
  LINK_PRESETS,
  deviceHostLink,
  hostPcieLink,
} from '../../sim/hw/presets/links';
import {
  APPLE_SSD_M3_M4_MAX_1TB,
  APPLE_SSD_M5_PRO_MAX,
  defaultHostRamFor,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  HOST_RAM_PRESETS,
  NVME_GEN4_X4_GENERIC,
  SSD_PRESETS,
} from '../../sim/hw/presets/storage';
import { linkPresetId, nodeTopology } from '../../sim/hw/presets/topologies';
import {
  POPULAR_SETUPS,
  allowedCablings,
  allowedLocalLinks,
  localLinkAllows,
  localLinkFits,
  defaultCabling,
  fabricOf,
  setupOptionsFor,
  type CablingId,
  type DeviceSetupOptions,
  type NetworkOption,
  type LocalLinkOption,
  type PopularSetup,
  type SetupChoice,
} from '../../sim/hw/presets/catalog';
import type { HardwareChoice, SpecEdits } from './appState';

/** The reason recorded in `estimated[]` for an edited field. */
export const EDITED_REASON = 'edited by the user';

/** The device preset with this id, or undefined. */
export function findDevice(id: string): DeviceSpec | undefined {
  return DEVICE_PRESETS.find((d) => d.id === id);
}

/**
 * The current `linkPresetId` of a link id from the state or an old URL: renamed links map through
 * `LEGACY_LINK_IDS` (old id → current id), every other id is itself.
 */
export function currentLinkId(id: string): string {
  return LEGACY_LINK_IDS[id] ?? id;
}

/** The link preset with this `linkPresetId` (or a legacy id of it, see {@link currentLinkId}), or undefined. */
export function findLink(id: string): LinkSpec | undefined {
  const current = currentLinkId(id);
  return LINK_PRESETS.find((l) => linkPresetId(l) === current);
}

/** Default links of a device: the ones the UI picks until the user chooses another preset. */
export interface DefaultLinks {
  /** Intra-node link (used when devices per node > 1); undefined for single-SoC machines. */
  intraNode?: LinkSpec;
  /** Inter-node link (used when nodes > 1). */
  interNode: LinkSpec;
}

/**
 * Default links of `device`: the first local link that fits `hostInterface` (the device's EFFECTIVE PCIe interface,
 * after unlocks; default: the preset's stock one) and the first network of its setup catalog entry
 * (`setupOptionsFor`). The catalog reproduces the hand-picked defaults the UI had before it (catalog.test.ts).
 */
export function defaultLinks(
  device: Pick<DeviceSpec, 'id'>,
  hostInterface: HostInterfaceSpec | undefined = findDevice(device.id)?.hostInterface,
): DefaultLinks {
  const options = setupOptionsFor(device.id);
  const intraNode = (
    options.localLinks.find((l) => localLinkFits(l, hostInterface)) ?? options.localLinks[0]
  )?.link;
  return { ...(intraNode !== undefined ? { intraNode } : {}), interNode: options.networks[0].link };
}

/**
 * The PCIe host interface of a choice's device after its unlocks (e.g. the CMP 170HX's Gen2 patch and capacitor mod),
 * which bounds the local links offered; undefined for an unknown device or one without a host interface.
 */
export function effectiveHostInterface(
  hw: Pick<HardwareChoice, 'deviceId'> & Partial<Pick<HardwareChoice, 'unlocks'>>,
): HostInterfaceSpec | undefined {
  const base = findDevice(hw.deviceId);
  if (base === undefined) return undefined;
  const mods = resolveUnlocks(deviceUnlocks(base), hw.unlocks ?? {}).deviceModifiers;
  return mods.length > 0 ? applyDeviceModifiers(base, mods).hostInterface : base.hostInterface;
}

/**
 * The intra-node link id in effect: the chosen id unless it names a catalogued local link the effective host
 * interface cannot run (a stale choice after an unlock was removed), else the default. Uncatalogued ids (old URLs)
 * are kept, as before. Undefined when there is no local link.
 */
function intraLinkIdInEffect(hw: HardwareChoice): string | undefined {
  const iface = effectiveHostInterface(hw);
  if (hw.intraLinkId !== undefined) {
    const id = currentLinkId(hw.intraLinkId);
    const opt = setupOptionsOf(hw).localLinks.find((l) => linkPresetId(l.link) === id);
    if (opt === undefined || localLinkFits(opt, iface)) return id;
  }
  const intra = defaultLinks({ id: hw.deviceId }, iface).intraNode;
  return intra && linkPresetId(intra);
}

/**
 * The UI name of a cabling for `nodes` machines: "Switch", "Full mesh", "Full mesh, 2 cables per pair", "Ring",
 * "Daisy chain"; two machines read "One cable" / "Two cables" (a mesh of two is a direct cable).
 */
export function cablingName(c: CablingId, nodes: number): string {
  switch (c) {
    case 'switched':
      return 'Switch';
    case 'mesh':
      return nodes === 2 ? 'One cable' : 'Full mesh';
    case 'mesh2':
      return nodes === 2 ? 'Two cables' : 'Full mesh, 2 cables per pair';
    case 'ring':
      return 'Ring';
    case 'chain':
      return 'Daisy chain';
  }
}

// ---------------------------------------------------------------------------------------------------------
// The setup row: counts, links and cabling the device offers

/** The setup catalog entry of the choice's device. */
export function setupOptionsOf(hw: Pick<HardwareChoice, 'deviceId'>): DeviceSetupOptions {
  return setupOptionsFor(hw.deviceId);
}

/** The offered count nearest to `n` (the smaller one on a tie); `n` itself when nothing is offered. Count. */
export function nearestCount(offered: number[], n: number): number {
  let best: number | undefined;
  for (const c of offered)
    if (
      best === undefined ||
      Math.abs(c - n) < Math.abs(best - n) ||
      (Math.abs(c - n) === Math.abs(best - n) && c < best)
    )
      best = c;
  return best ?? n;
}

/**
 * The effective link ids of a choice, only at the levels that exist: the chosen id, else the device default (for the
 * intra-node link: a chosen catalog link the effective host interface cannot run counts as not chosen).
 */
export function effectiveLinkIds(hw: HardwareChoice): { intra?: string; inter?: string } {
  const defaults = defaultLinks({ id: hw.deviceId });
  const intra = hw.devicesPerNode > 1 ? intraLinkIdInEffect(hw) : undefined;
  const inter =
    hw.nodes > 1
      ? hw.interLinkId !== undefined
        ? currentLinkId(hw.interLinkId)
        : linkPresetId(defaults.interNode)
      : undefined;
  return { ...(intra !== undefined ? { intra } : {}), ...(inter !== undefined ? { inter } : {}) };
}

/**
 * The catalog network of the choice's inter-node link (chosen or default), or undefined when the device's catalog
 * entry does not list that link (an old URL): then only a switch is offered (`allowedCablings`).
 */
export function networkOptionOf(hw: HardwareChoice): NetworkOption | undefined {
  const id = hw.interLinkId !== undefined ? currentLinkId(hw.interLinkId) : undefined;
  const networks = setupOptionsOf(hw).networks;
  return id === undefined ? networks[0] : networks.find((n) => linkPresetId(n.link) === id);
}

/** The catalog local link of the choice's intra-node link (chosen or default), or undefined when not listed. */
export function localLinkOptionOf(hw: HardwareChoice): LocalLinkOption | undefined {
  const id = intraLinkIdInEffect(hw);
  return setupOptionsOf(hw).localLinks.find((l) => linkPresetId(l.link) === id);
}

/**
 * The catalog local links that join the choice's devices per machine (e.g. the 2-card NVLink bridge only at 2) and fit
 * its effective PCIe host interface (the CMP 170HX's PCIe unlocks: stock offers only PCIe 1.0 x4).
 */
export function localLinkChoices(
  hw: Pick<HardwareChoice, 'deviceId' | 'devicesPerNode'> &
    Partial<Pick<HardwareChoice, 'unlocks'>>,
): LocalLinkOption[] {
  return allowedLocalLinks(setupOptionsOf(hw), hw.devicesPerNode, effectiveHostInterface(hw));
}

/**
 * The choice with a chosen intra-node link that cannot join its devices per machine (a catalog link restricted to
 * other counts, e.g. the 2-card NVLink bridge at 4 GPUs) or that its effective host interface cannot run (a CMP 170HX
 * PCIe 2.0 link after the Gen2 unlock is removed) dropped, with its edits, so the default link applies. A link the
 * catalog does not list (an older URL) is kept.
 */
export function normalizeLocalLink(hw: HardwareChoice): HardwareChoice {
  if (hw.intraLinkId === undefined || hw.devicesPerNode <= 1) return hw;
  const id = currentLinkId(hw.intraLinkId);
  const opt = setupOptionsOf(hw).localLinks.find((l) => linkPresetId(l.link) === id);
  if (
    opt === undefined ||
    (localLinkAllows(opt, hw.devicesPerNode) && localLinkFits(opt, effectiveHostInterface(hw)))
  )
    return hw;
  const { intraLinkId: _i, ...rest } = hw;
  void _i;
  return { ...rest, intraEdits: {} };
}

/**
 * Devices per machine for picking the local link `linkId`: the current count when the link joins it, else the
 * nearest count the link supports (e.g. picking the 2-card NVLink bridge at 4 GPUs gives 2). Count.
 */
export function devicesPerNodeForLocalLink(hw: HardwareChoice, linkId: string): number {
  const id = currentLinkId(linkId);
  const opt = setupOptionsOf(hw).localLinks.find((l) => linkPresetId(l.link) === id);
  if (
    opt === undefined ||
    localLinkAllows(opt, hw.devicesPerNode) ||
    opt.devicesPerNodeCounts === undefined
  )
    return hw.devicesPerNode;
  return nearestCount(opt.devicesPerNodeCounts, hw.devicesPerNode);
}

/** The cablings the choice's network, machine count and devices per machine allow ([] for one machine). */
export function cablingOptions(hw: HardwareChoice): CablingId[] {
  if (hw.nodes <= 1) return [];
  return allowedCablings(networkOptionOf(hw), hw.nodes, hw.devicesPerNode);
}

/**
 * The cabling in effect (undefined for one machine): the chosen one while it is allowed, else the default
 * (`defaultCabling`: a switch where one exists, else a full mesh where the ports allow, else a ring, else a chain).
 */
export function effectiveCabling(hw: HardwareChoice): CablingId | undefined {
  if (hw.nodes <= 1) return undefined;
  if (hw.cabling !== undefined && cablingOptions(hw).includes(hw.cabling)) return hw.cabling;
  return defaultCabling(networkOptionOf(hw), hw.nodes, hw.devicesPerNode) ?? 'switched';
}

/**
 * The choice with `cabling` normalized: kept only while it is allowed and differs from the default, so equal
 * setups are equal states (and the URL omits a default `fab`).
 */
export function normalizeCabling(hw: HardwareChoice): HardwareChoice {
  if (hw.cabling === undefined) return hw;
  const keep =
    hw.nodes > 1 &&
    cablingOptions(hw).includes(hw.cabling) &&
    hw.cabling !== defaultCabling(networkOptionOf(hw), hw.nodes, hw.devicesPerNode);
  if (keep) return hw;
  const { cabling: _c, ...rest } = hw;
  void _c;
  return rest;
}

/**
 * The NIC counts per machine the choice can take (`TopologySpec.nicsPerNode`, docs/simulation-model.md §6 "Multi-NIC
 * nodes"): the divisors of the devices per machine (dpn), then 2·dpn and 4·dpn, ascending, so every device gets the
 * same share. Offered only between several machines of more than one device each, joined by a switch (a switchless
 * cabling's ports are its cables); else `[]`. dpn itself is the default (one NIC per device). Counts.
 */
export function nicOptions(hw: HardwareChoice): number[] {
  const dpn = hw.devicesPerNode;
  if (hw.nodes <= 1 || dpn <= 1) return [];
  if (effectiveCabling(hw) !== 'switched') return [];
  const divisors: number[] = [];
  for (let k = 1; k <= dpn; k++) if (dpn % k === 0) divisors.push(k);
  return [...divisors, 2 * dpn, 4 * dpn];
}

/**
 * The picker label of a NIC count on machines of `devicesPerNode` GPUs: the count and the share per GPU, e.g.
 * "4 (1 per 2 GPUs)", "8 (1 per GPU)", "16 (2 per GPU)".
 */
export function nicLabel(nics: number, devicesPerNode: number): string {
  if (nics >= devicesPerNode) return `${nics} (${nics / devicesPerNode} per GPU)`;
  return `${nics} (1 per ${devicesPerNode / nics} GPUs)`;
}

/**
 * The choice with `nicsPerNode` normalized: kept only while it is offered ({@link nicOptions}) and differs from the
 * default (one per device, `devicesPerNode`), so equal setups are equal states and the URL omits a default `nic`.
 */
export function normalizeNics(hw: HardwareChoice): HardwareChoice {
  if (hw.nicsPerNode === undefined) return hw;
  if (hw.nicsPerNode !== hw.devicesPerNode && nicOptions(hw).includes(hw.nicsPerNode)) return hw;
  const { nicsPerNode: _n, ...rest } = hw;
  void _n;
  return rest;
}

/** True if two host interfaces are the same generation and width (both absent counts as the same). */
function sameInterface(
  a: HostInterfaceSpec | undefined,
  b: HostInterfaceSpec | undefined,
): boolean {
  return a?.generation === b?.generation && a?.lanes === b?.lanes;
}
/**
 * The catalog's view of a choice (for `setupLabel`): device, memory per device, counts, the effective link ids,
 * cabling and (when unlocks change it) the effective PCIe host interface. `memCapacityBytes` is the capacity in use
 * (unlocks may change the options). Unit: bytes.
 */
export function setupChoiceOf(hw: HardwareChoice, memCapacityBytes?: number): SetupChoice {
  const links = effectiveLinkIds(hw);
  const cabling = effectiveCabling(hw);
  const iface = effectiveHostInterface(hw);
  return {
    deviceId: hw.deviceId,
    ...(memCapacityBytes !== undefined ? { memCapacityBytes } : {}),
    devicesPerNode: hw.devicesPerNode,
    nodes: hw.nodes,
    ...(links.intra !== undefined ? { localLinkId: links.intra } : {}),
    ...(links.inter !== undefined ? { networkLinkId: links.inter } : {}),
    ...(cabling !== undefined ? { cabling } : {}),
    ...(iface !== undefined && !sameInterface(iface, findDevice(hw.deviceId)?.hostInterface)
      ? { hostInterface: iface }
      : {}),
  };
}

/**
 * The hardware choice of a popular setup: its device (stock, no unlocks, first engine), the index of its memory
 * option (0 when it names none), its counts, its links as explicit preset ids (only at the levels that exist),
 * and its cabling (normalized: absent when it is the default). No edits.
 */
export function popularSetupChoice(p: PopularSetup): HardwareChoice {
  const c = p.choice;
  const device = findDevice(c.deviceId);
  if (device === undefined) throw new Error(`setup '${p.id}': unknown device '${c.deviceId}'`);
  const capacityIndex =
    c.memCapacityBytes === undefined
      ? 0
      : device.memory.capacityOptionsBytes.indexOf(c.memCapacityBytes);
  if (capacityIndex < 0)
    throw new Error(
      `setup '${p.id}': ${c.memCapacityBytes} B is not a memory option of ${device.id}`,
    );
  const defaults = defaultLinks(device);
  const intra = c.localLinkId ?? (defaults.intraNode && linkPresetId(defaults.intraNode));
  const inter = c.networkLinkId ?? linkPresetId(defaults.interNode);
  return normalizeCabling({
    deviceId: device.id,
    unlocks: {},
    capacityIndex,
    devicesPerNode: c.devicesPerNode,
    nodes: c.nodes,
    ...(c.devicesPerNode > 1 && intra !== undefined ? { intraLinkId: intra } : {}),
    ...(c.nodes > 1 ? { interLinkId: inter } : {}),
    ...(c.nodes > 1 && c.cabling !== undefined ? { cabling: c.cabling } : {}),
    deviceEdits: {},
    intraEdits: {},
    interEdits: {},
  });
}

/**
 * The popular setup a hardware choice is, or undefined: same device, memory option, counts, effective links (an
 * absent link id means the device default) and effective cabling, with no unlocks, engine choice or edits. So
 * picking a setup and then changing anything (a count, a link, the cabling, a spec number) shows the plain device.
 */
export function matchingPopularSetup(hw: HardwareChoice): PopularSetup | undefined {
  const stock =
    Object.keys(hw.unlocks).length === 0 &&
    hw.engineId === undefined &&
    Object.keys(hw.deviceEdits).length === 0 &&
    Object.keys(hw.intraEdits).length === 0 &&
    Object.keys(hw.interEdits).length === 0 &&
    normalizeNics(hw).nicsPerNode === undefined;
  if (!stock || findDevice(hw.deviceId) === undefined) return undefined;
  const links = effectiveLinkIds(hw);
  const cabling = effectiveCabling(hw);
  return POPULAR_SETUPS.find((p) => {
    const c = popularSetupChoice(p);
    const cLinks = effectiveLinkIds(c);
    return (
      c.deviceId === hw.deviceId &&
      c.capacityIndex === hw.capacityIndex &&
      c.devicesPerNode === hw.devicesPerNode &&
      c.nodes === hw.nodes &&
      cLinks.intra === links.intra &&
      cLinks.inter === links.inter &&
      effectiveCabling(c) === cabling
    );
  });
}

/** A deep copy of a plain-data spec (presets are JSON-safe: numbers, strings, booleans, arrays, objects). */
function clonePlain<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Sets `value` at dotted `path` in `obj` (mutates the copy it is given). */
function setAtPath(obj: unknown, path: string, value: number): void {
  const keys = path.split('.');
  let node = obj as Record<string, unknown>;
  for (const key of keys.slice(0, -1)) node = node[key] as Record<string, unknown>;
  node[keys[keys.length - 1]] = value;
}

/** Result of {@link applyEdits}. */
export interface EditedSpec<T> {
  /** The spec with the edits applied (the input itself when there are none). */
  spec: T;
  /** Edited paths that are not numeric leaves of the spec (ignored). */
  unknownPaths: string[];
}

/**
 * Applies absolute-value edits at numeric-leaf paths to a copy of `spec`, marking each edited path in
 * `estimated` with {@link EDITED_REASON}. Paths that are not numeric leaves are ignored and reported. The
 * caller renames the spec (devices: `<id>-custom`, `<name> (edited)`).
 */
export function applyEdits<T extends { estimated: EstimatedField[] }>(
  spec: T,
  edits: SpecEdits,
): EditedSpec<T> {
  const leaves = new Set(numericLeafPaths(spec).map((l) => l.path));
  const paths = Object.keys(edits).sort();
  const known = paths.filter((p) => leaves.has(p));
  const unknownPaths = paths.filter((p) => !leaves.has(p));
  if (known.length === 0) return { spec, unknownPaths };
  const copy = clonePlain(spec);
  for (const path of known) setAtPath(copy, path, edits[path]);
  copy.estimated = [...copy.estimated, ...known.map((field) => ({ field, reason: EDITED_REASON }))];
  return { spec: copy, unknownPaths };
}

/** The device a hardware choice resolves to. */
export interface ResolvedDevice {
  /** The preset, unmodified. */
  base: DeviceSpec;
  /** Base + unlock modifiers (before edits): what the spec editor shows as "stock for this selection". */
  unlocked: DeviceSpec;
  /** Base + unlocks + edits: the device actually simulated. */
  device: DeviceSpec;
  /** Intra-node link modifiers of the chosen unlocks (e.g. the P2P driver). */
  intraNodeLinkModifiers: LinkModifier[];
  /** True when at least one device edit applied. */
  edited: boolean;
  /** One "variant: …" line per unlock modifier. */
  notes: string[];
}

/**
 * Resolves the device of a hardware choice: preset → unlock modifiers (`resolveUnlocks`,
 * `applyDeviceModifiers`) → edits. Throws on an unknown device, axis or option id (the URL decoder and the
 * reducer only produce known ids, so this is a UI bug).
 */
export function resolveDevice(
  hw: Pick<HardwareChoice, 'deviceId' | 'unlocks' | 'deviceEdits'>,
): ResolvedDevice {
  const base = findDevice(hw.deviceId);
  if (base === undefined) throw new Error(`unknown device '${hw.deviceId}'`);
  const unlocks = resolveUnlocks(deviceUnlocks(base), hw.unlocks);
  const unlocked =
    unlocks.deviceModifiers.length > 0 ? applyDeviceModifiers(base, unlocks.deviceModifiers) : base;
  const { spec } = applyEdits(unlocked, hw.deviceEdits);
  const edited = spec !== unlocked;
  const device = edited
    ? { ...spec, id: `${base.id}-custom`, name: `${base.name} (edited)` }
    : spec;
  return {
    base,
    unlocked,
    device,
    intraNodeLinkModifiers: unlocks.intraNodeLinkModifiers,
    edited,
    notes: unlocks.deviceModifiers.map(describeDeviceModifier),
  };
}

/** One resolved link level. */
export interface ResolvedLink {
  /** The chosen (or default) preset, unmodified. */
  preset: LinkSpec;
  /** Preset + unlock link modifiers (intra-node only), before edits. */
  unlocked: LinkSpec;
  /** Preset + modifiers + edits: the link actually simulated. */
  link: LinkSpec;
  /** True when at least one edit applied. */
  edited: boolean;
  /** One "link variant: …" line per unlock modifier applied. */
  notes: string[];
}

/** Resolves one link: preset → modifiers → edits (renamed `<name> (edited)` when edited). */
function resolveLink(preset: LinkSpec, mods: LinkModifier[], edits: SpecEdits): ResolvedLink {
  const unlocked = mods.length > 0 ? applyLinkModifiers(preset, mods) : preset;
  const { spec } = applyEdits(unlocked, edits);
  const edited = spec !== unlocked;
  return {
    preset,
    unlocked,
    link: edited ? { ...spec, name: `${preset.name} (edited)` } : spec,
    edited,
    notes: mods.map((m) => `intraNode ${describeLinkModifier(m)}`),
  };
}

/** The links of a hardware choice, present only at the levels that exist (devicesPerNode > 1, nodes > 1). */
export interface ResolvedLinks {
  /** Intra-node link, when devicesPerNode > 1 and the device has one. */
  intra?: ResolvedLink;
  /** Inter-node link, when nodes > 1. */
  inter?: ResolvedLink;
}

/**
 * Resolves the links of a choice: the chosen preset (`intraLinkId` / `interLinkId`) or the device default,
 * the unlocks' intra-node link modifiers (only on a host-bounce link: they enable direct P2P), then the edits.
 * Unknown link ids fall back to the default.
 */
export function resolveLinks(hw: HardwareChoice, resolved: ResolvedDevice): ResolvedLinks {
  const defaults = defaultLinks(resolved.base, resolved.unlocked.hostInterface);
  const out: ResolvedLinks = {};
  if (hw.devicesPerNode > 1) {
    // A chosen catalog link the unlocked host interface cannot run falls back to the default (and drops its edits).
    const id = intraLinkIdInEffect(hw);
    const chosenKept = hw.intraLinkId === undefined || id === currentLinkId(hw.intraLinkId);
    const preset = (id !== undefined ? findLink(id) : undefined) ?? defaults.intraNode;
    if (preset !== undefined) {
      const mods = preset.p2p === 'host_bounce' ? resolved.intraNodeLinkModifiers : [];
      out.intra = resolveLink(preset, mods, chosenKept ? hw.intraEdits : {});
    }
  }
  if (hw.nodes > 1) {
    const preset =
      (hw.interLinkId !== undefined ? findLink(hw.interLinkId) : undefined) ?? defaults.interNode;
    out.inter = resolveLink(preset, [], hw.interEdits);
  }
  return out;
}

/** Everything the UI derives from a hardware choice. */
export interface ResolvedHardware {
  /** The resolved device. */
  device: ResolvedDevice;
  /** The resolved links. */
  links: ResolvedLinks;
  /** The topology built from the counts, the resolved links and the cabling (`nodeTopology`). */
  topology: TopologySpec;
  /** The cabling in effect between machines (`effectiveCabling`); undefined for one machine. */
  cabling?: CablingId;
  /** The catalog network of the inter-node link, when there is one and the device's catalog entry lists it. */
  network?: NetworkOption;
  /** The stack handed to the engine: already-resolved specs, so no modifiers. */
  stack: HardwareStack;
  /** Capacity used, per device: `capacityOptionsBytes[capacityIndex]`, or the default when out of range. Unit: bytes. */
  memCapacityBytes: number;
  /** True when any device or link edit applied (the UI says "edited" wherever the rig is named). */
  edited: boolean;
  /** Modifier description lines (the engine sees no modifiers, so the UI shows these as notes). */
  notes: string[];
}

// ---------------------------------------------------------------------------------------------------------
// The node's host: host RAM, SSD and the host link (offloading, design doc §2.2, §6.2)

/** The host-RAM preset with this id, or undefined. */
export function findHostRam(id: string): HostMemorySpec | undefined {
  return HOST_RAM_PRESETS.find((r) => r.id === id);
}

/** The SSD preset with this id, or undefined. */
export function findSsd(id: string): NvmeSpec | undefined {
  return SSD_PRESETS.find((s) => s.id === id);
}

/**
 * The SSD a device starts with: its own internal drive on unified-memory machines (Apple M5 → the M5 Pro/Max SSD,
 * other Apple chips → the M3/M4 Max SSD, DGX Spark → its 4 TB drive, Strix Halo → a generic PCIe 4.0 x4 drive),
 * else the engine's default, the Samsung 990 PRO 2 TB.
 */
export function defaultSsdFor(device: Pick<DeviceSpec, 'id' | 'memory'>): NvmeSpec {
  if (device.memory.kind === 'unified') {
    if (device.id.startsWith('apple-m5')) return APPLE_SSD_M5_PRO_MAX;
    if (device.id.startsWith('apple-')) return APPLE_SSD_M3_M4_MAX_1TB;
    if (device.id === 'dgx-spark') return DGX_SPARK_SSD_4TB;
    if (device.id === 'ryzen-ai-max-plus-395') return NVME_GEN4_X4_GENERIC;
  }
  return DEFAULT_SSD;
}

/** The resolved host of a choice: what the engine gets as `HardwareStack.host`, and where each part came from. */
export interface ResolvedHost {
  /**
   * The host-RAM preset in use (capacity overridden when `ramCapacityBytes` is set): the user's preset, else the
   * device's default for this many devices per machine (`defaultHostRamFor`: the pooled Grace LPDDR5X of
   * ceil(devicesPerNode / 2) CPUs on GB200 / GB300, the desktop DDR5 default elsewhere). Capacity in bytes and
   * bandwidth in B/s per machine (one scale-up domain).
   */
  ram: HostMemorySpec;
  /** True when `ram` is the device's default (`defaultHostRamFor`), not a preset the user picked. */
  ramDefault: boolean;
  /** The SSD preset in use. */
  ssd: NvmeSpec;
  /**
   * The device's link to the host, per GPU, per direction: on a device with a non-PCIe `hostLink` (Grace
   * NVLink-C2C) always that link; else the user's PCIe override, else the device's PCIe `hostInterface`; undefined
   * when none exists (SXM/OAM modules, SoCs), which the engine rejects while something is offloaded on dedicated
   * memory.
   */
  link?: LinkSpec;
  /**
   * True when `link` is the user's PCIe override, not the device's own link. Always false on a device with a
   * `hostLink`: its GPU reaches the CPU only over that link, so a PCIe override (from a shared URL) is ignored.
   */
  linkOverridden: boolean;
}

/** Resolves the host of a choice for `device` (see {@link ResolvedHost}). Unknown preset ids fall back to the default. */
export function resolveHost(
  hw: Pick<HardwareChoice, 'host' | 'devicesPerNode'>,
  device: DeviceSpec,
): ResolvedHost {
  const host = hw.host ?? {};
  const picked = host.ramId !== undefined ? findHostRam(host.ramId) : undefined;
  const preset = picked ?? defaultHostRamFor(device, hw.devicesPerNode);
  const ram =
    host.ramCapacityBytes !== undefined && host.ramCapacityBytes !== preset.capacityBytes
      ? {
          ...preset,
          capacityBytes: host.ramCapacityBytes,
          estimated: [
            ...preset.estimated.filter((e) => e.field !== 'capacityBytes'),
            { field: 'capacityBytes', reason: EDITED_REASON },
          ],
        }
      : preset;
  const ssd = (host.ssdId !== undefined ? findSsd(host.ssdId) : undefined) ?? defaultSsdFor(device);
  const overridden = device.hostLink === undefined && host.link !== undefined;
  const link =
    overridden && host.link !== undefined
      ? hostPcieLink(host.link.generation, host.link.lanes)
      : deviceHostLink(device);
  return {
    ram,
    ramDefault: picked === undefined,
    ssd,
    ...(link !== undefined ? { link } : {}),
    linkOverridden: overridden,
  };
}

/**
 * The engine's `HardwareStack.host` for a resolved host: on dedicated memory the link (when known), host RAM and
 * the SSD; on unified memory only the SSD (host RAM is the device pool there and the link does not exist).
 */
export function hostSpecOf(host: ResolvedHost, device: DeviceSpec): HostSpec {
  if (device.memory.kind === 'unified') return { ssd: host.ssd };
  return { ...(host.link !== undefined ? { link: host.link } : {}), ram: host.ram, ssd: host.ssd };
}

/**
 * Resolves a hardware choice to the engine's `HardwareStack`. The device and links are passed already
 * resolved (unlocks and edits applied), with `deviceModifiers` and `linkModifiers` empty, so the engine
 * validates exactly the numbers the user sees. Throws on unknown device / unlock ids (see {@link resolveDevice}).
 * `stack.host` is set only when `offDevice` is true (the deployment places something off the device), so a
 * stack without offloading is exactly the one from before host tiers.
 */
export function resolveHardware(hw: HardwareChoice, offDevice = false): ResolvedHardware {
  const device = resolveDevice(hw);
  const links = resolveLinks(hw, device);
  const cabling = effectiveCabling(hw);
  const network = hw.nodes > 1 ? networkOptionOf(hw) : undefined;
  const nics = normalizeNics(hw).nicsPerNode;
  // The fabric is passed for every cabling; nodeTopology omits a non-blocking switch, so the switched
  // topologies are exactly the ones from before fabric shapes (fabric-identity.test.ts).
  const topology = nodeTopology({
    devicesPerNode: hw.devicesPerNode,
    nodes: hw.nodes,
    device: device.device,
    ...(links.intra !== undefined ? { intraNode: links.intra.link } : {}),
    ...(links.inter !== undefined ? { interNode: links.inter.link } : {}),
    ...(cabling !== undefined && links.inter !== undefined ? { fabric: fabricOf(cabling) } : {}),
    // Only a non-default count the choice still offers; nodeTopology drops it otherwise too.
    ...(nics !== undefined ? { nicsPerNode: nics } : {}),
  });
  const capacities = device.device.memory.capacityOptionsBytes;
  const memCapacityBytes = capacities[hw.capacityIndex] ?? capacities[0];
  const stack: HardwareStack = {
    device: device.device,
    topology,
    memCapacityBytes,
    ...(hw.engineId !== undefined ? { engineId: hw.engineId } : {}),
    // Memory-fit knobs (Hardware → Advanced); the GPU limit applies to unified memory only.
    ...(hw.gpuUsableFraction !== undefined && device.device.memory.kind === 'unified'
      ? { gpuUsableFraction: hw.gpuUsableFraction }
      : {}),
    ...(hw.runtimeReserveBytes !== undefined
      ? { runtimeReserveBytes: hw.runtimeReserveBytes }
      : {}),
    ...(offDevice ? { host: hostSpecOf(resolveHost(hw, device.device), device.device) } : {}),
  };
  return {
    device,
    links,
    topology,
    ...(cabling !== undefined ? { cabling } : {}),
    ...(network !== undefined ? { network } : {}),
    stack,
    memCapacityBytes,
    edited: device.edited || (links.intra?.edited ?? false) || (links.inter?.edited ?? false),
    notes: [...device.notes, ...(links.intra?.notes ?? [])],
  };
}

/**
 * The in-network reductions (docs/simulation-model.md §6 "In-network reduction (NVLS, SHARP)") the resolved links
 * offer to an all-reduce, in level order: the intra-node link's (`LinkSpec.inNetworkReduction`, e.g. NVLS on an
 * NVLink Switch) when a machine has more than one device, and the inter-node link's (e.g. SHARP on InfiniBand) when
 * the machines are joined by a switch (a switchless cabling has no switch to reduce in). Duplicates removed; `[]` when
 * nothing offers one, so the UI hides the switch.
 */
export function inNetworkReductionsOf(resolved: ResolvedHardware): InNetworkReduction[] {
  const out: InNetworkReduction[] = [];
  const intra = resolved.links.intra?.link.inNetworkReduction;
  if (resolved.topology.devicesPerNode > 1 && intra !== undefined) out.push(intra);
  const inter = resolved.links.inter?.link.inNetworkReduction;
  if (resolved.topology.nodes > 1 && resolved.cabling === 'switched' && inter !== undefined)
    if (!out.includes(inter)) out.push(inter);
  return out;
}

// ---------------------------------------------------------------------------------------------------------
// Editable fields and display units

/** A display unit for a spec number: `display = si / scale`. */
export interface DisplayUnit {
  /** Unit label shown next to the field, e.g. `'TFLOP/s'`. */
  label: string;
  /** SI value per display unit (e.g. 1e12 for TFLOP/s, 1e-6 for µs). */
  scale: number;
}

/** Display units used by the spec editor. */
export const UNITS = {
  tflops: { label: 'TFLOP/s', scale: 1e12 },
  tops: { label: 'TOPS', scale: 1e12 },
  tOpsPerSec: { label: 'T ops/s', scale: 1e12 },
  gbps: { label: 'GB/s', scale: 1e9 },
  gb: { label: 'GB', scale: 1e9 },
  mb: { label: 'MB', scale: 1e6 },
  us: { label: 'µs', scale: 1e-6 },
  count: { label: '', scale: 1 },
  elements: { label: 'elements', scale: 1 },
} as const satisfies Record<string, DisplayUnit>;

/** SI value → display value. */
export function toDisplay(si: number, unit: DisplayUnit): number {
  return si / unit.scale;
}

/** Display value → SI value. */
export function fromDisplay(display: number, unit: DisplayUnit): number {
  return display * unit.scale;
}

/** Groups of the spec editor; `tiling` is the advanced group (collapsed by default). */
export type FieldGroup = 'memory' | 'matmul' | 'vector' | 'tiling' | 'link';

/** One editable number of a spec. */
export interface EditableField {
  /** Dotted `numericLeafPaths` path, the key in `SpecEdits`. */
  path: string;
  /** Human label, e.g. `'bf16, fp32 accumulate, dense'`. */
  label: string;
  /** Editor group. */
  group: FieldGroup;
  /** Display unit. */
  unit: DisplayUnit;
  /** Current value of the spec given, SI (as the field defines it). */
  value: number;
  /** Help lines for a "?" icon, for fields whose meaning is not obvious (one short sentence per line). */
  help?: string[];
}

/**
 * True if a device leaf path may be edited (spec editor, URL `x`): every numeric leaf except `hostInterface.*`, whose
 * generation and lanes are enumerations set through the Host link picker, not free numbers; `hostLink.*`, the fixed
 * non-PCIe CPU link (Grace NVLink-C2C), a cited link spec rather than a device number; and
 * `memory.gpuUsableFraction`, which has its own field (Advanced → Memory fit → GPU memory limit).
 */
export function isEditableDevicePath(path: string): boolean {
  return (
    !path.startsWith('hostInterface.') &&
    !path.startsWith('hostLink.') &&
    path !== 'memory.gpuUsableFraction'
  );
}
/**
 * The editable numbers of a device: every `numericLeafPaths` leaf, grouped into memory, matmul peaks, vector
 * peaks and tiling (advanced), with human labels and display units. Engine-specific labels carry the engine
 * name when the device has more than one engine.
 */
export function editableFields(device: DeviceSpec): EditableField[] {
  const multiEngine = device.engines.length > 1;
  const fields: EditableField[] = [];
  for (const { path, value } of numericLeafPaths(device)) {
    if (!isEditableDevicePath(path)) continue;
    const parts = path.split('.');
    const field = (label: string, group: FieldGroup, unit: DisplayUnit, help?: string[]) =>
      fields.push({ path, label, group, unit, value, ...(help ? { help } : {}) });
    if (path === 'memory.bandwidthBytesPerSec') field('Memory bandwidth', 'memory', UNITS.gbps);
    else if (parts[0] === 'memory' && parts[1] === 'capacityOptionsBytes') {
      field(
        device.memory.capacityOptionsBytes.length > 1
          ? `Capacity option ${Number(parts[2]) + 1}`
          : 'Capacity',
        'memory',
        UNITS.gb,
      );
    } else if (path === 'l2Bytes')
      field('Last-level cache (not used by the model)', 'tiling', UNITS.mb);
    else if (parts[0] === 'engines') {
      const engine = device.engines[Number(parts[1])];
      const prefix = multiEngine ? `${engine.name}: ` : '';
      const rest = parts.slice(2);
      if (rest[0] === 'matmulPeaks') {
        const peak = engine.matmulPeaks[Number(rest[1])];
        const integer = peak.accumulate === 'int32';
        field(
          `${prefix}${peak.input}, ${peak.accumulate} accumulate, ${peak.sparsity === 'dense' ? 'dense' : '2:4 sparse'}`,
          'matmul',
          integer ? UNITS.tops : UNITS.tflops,
        );
      } else if (rest[0] === 'vectorFlopsPerSec')
        field(`${prefix}vector ${rest[1]}`, 'vector', UNITS.tflops);
      else if (rest[0] === 'memBandwidthCapBytesPerSec')
        field(`${prefix}engine memory-bandwidth cap`, 'memory', UNITS.gbps);
      else if (rest[0] === 'sfuOpsPerSec') {
        field(`${prefix}SFU exp rate`, 'vector', UNITS.tOpsPerSec, [
          'Exponentials per second on the special-function units.',
          'Attention needs one per score, so a low rate slows long prompts.',
        ]);
      } else if (rest[0] === 'gemvMaxRows') {
        field(`${prefix}GEMV path: max rows`, 'tiling', UNITS.count, [
          'Up to this many tokens, matrix products run on the plain vector units.',
          'They then skip the padding of the matrix units.',
        ]);
      } else if (rest[0] === 'computeUnits') field(`${prefix}compute units`, 'tiling', UNITS.count);
      else field(`${prefix}${rest.join(' ')}`, 'tiling', UNITS.elements);
    } else field(path, 'tiling', UNITS.count);
  }
  return fields;
}

/**
 * The editable numbers of a link: bandwidth (GB/s per direction), latency (µs), host staging, and for multi-port
 * links the cap over all ports (GB/s per direction) and the relay latency per hop (µs); full peer-domain size
 * (devices) where the link profile establishes one.
 */
export function editableLinkFields(link: LinkSpec): EditableField[] {
  const labels: Record<string, [string, DisplayUnit]> = {
    bandwidthPerDirectionBytesPerSec: ['Bandwidth per direction, per device', UNITS.gbps],
    latencySec: ['Latency per message', UNITS.us],
    'hostBounce.hostMemBandwidthBytesPerSec': [
      'Host memory bandwidth for staging (shared by the node)',
      UNITS.gbps,
    ],
    'hostBounce.extraLatencySec': ['Extra latency of the host copy', UNITS.us],
    allPortsCapPerDirectionBytesPerSec: ['Bandwidth cap, all ports', UNITS.gbps],
    forwardingLatencySec: ['Relay latency per hop', UNITS.us],
    fullyConnectedDevices: ['Fully connected domain (devices)', UNITS.count],
  };
  return numericLeafPaths(link).map(({ path, value }) => {
    const [label, unit] = labels[path] ?? [path, UNITS.count];
    return { path, label, group: 'link' as const, unit, value };
  });
}
