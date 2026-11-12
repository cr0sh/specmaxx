/**
 * Validation of hardware specs and modifiers: physical sanity (finite, non-negative numbers; sparse ≥ dense;
 * tile shapes aligned to the MMA shape) and the project rule that EVERY number is sourced (covered by a
 * `sources[].fields` path or an `estimated[].field`). The preset-library task reuses these.
 * See docs/simulation-model.md §4 ("Estimated fields", "Device variants and modifiers").
 */

import type { MathFormat } from '../ir/dtype';
import { hasStrippedProvenance } from './provenance';
import {
  applyDeviceModifiers,
  applyLinkModifiers,
  deviceModifierErrors,
  linkModifierErrors,
} from './modifiers';
import { fabricShapeErrors } from './fabric';
import { nicCountErrors } from './nics';
import type { DeviceModifier, LinkModifier } from './modifiers';
import type {
  ComputeEngine,
  DeviceSpec,
  EstimatedField,
  GemmTiling,
  HostInterfaceSpec,
  HostSpec,
  LinkSpec,
  SourceRef,
  StorageTierSpec,
  TopologySpec,
} from './types';

/** One numeric leaf of a spec object. */
export interface NumericLeaf {
  /** Dotted path from the spec root, array indices included (e.g. `'engines.0.matmulPeaks.3.flopsPerSec'`). */
  path: string;
  /** The number found there (unit as the field defines it). */
  value: number;
}

/** Keys that hold provenance, not physical numbers; {@link numericLeafPaths} skips them. */
const PROVENANCE_KEYS = new Set(['sources', 'estimated', 'appliedModifiers']);

/**
 * Every numeric leaf of `obj` (recursing through objects and arrays), with its dotted path, skipping the
 * provenance keys `sources`, `estimated` and `appliedModifiers`. Strings and booleans are not leaves.
 */
export function numericLeafPaths(obj: unknown, prefix = ''): NumericLeaf[] {
  if (typeof obj === 'number') return [{ path: prefix, value: obj }];
  if (obj === null || typeof obj !== 'object') return [];
  const leaves: NumericLeaf[] = [];
  const entries: [string, unknown][] = Array.isArray(obj)
    ? obj.map((v, i) => [String(i), v])
    : Object.entries(obj as Record<string, unknown>);
  for (const [key, value] of entries) {
    if (!Array.isArray(obj) && PROVENANCE_KEYS.has(key)) continue;
    leaves.push(...numericLeafPaths(value, prefix === '' ? key : `${prefix}.${key}`));
  }
  return leaves;
}

/** True if `path` equals one of `covers` or lies in the subtree of one (a prefix covers its subtree). */
export function isPathCovered(path: string, covers: string[]): boolean {
  return covers.some((c) => path === c || path.startsWith(`${c}.`));
}

/** Errors for numbers that are not finite or negative, and for numeric leaves no source or estimate covers. */
function leafErrors(
  obj: unknown,
  sources: SourceRef[],
  estimated: EstimatedField[],
  label: string,
): string[] {
  const covers = [...sources.flatMap((s) => s.fields), ...estimated.map((e) => e.field)];
  const errors: string[] = [];
  for (const leaf of numericLeafPaths(obj)) {
    if (!Number.isFinite(leaf.value) || leaf.value < 0) {
      errors.push(`${label}: ${leaf.path} must be finite and ≥ 0 (got ${leaf.value})`);
    }
    // Bundled presets were citation-checked in source tests. Never skip physical checks.
    if (!hasStrippedProvenance(obj) && !isPathCovered(leaf.path, covers)) {
      errors.push(
        `${label}: ${leaf.path} is not covered by any sources[].fields or estimated[].field`,
      );
    }
  }
  return errors;
}

/** Errors in the citation records themselves (empty URL, empty fields, empty reasons). */
function provenanceRecordErrors(
  sources: SourceRef[],
  estimated: EstimatedField[],
  label: string,
): string[] {
  const errors: string[] = [];
  sources.forEach((s, i) => {
    if (s.url.trim() === '') errors.push(`${label}: sources[${i}] has an empty url`);
    if (s.fields.length === 0) errors.push(`${label}: sources[${i}] (${s.url}) covers no fields`);
  });
  estimated.forEach((e, i) => {
    if (e.reason.trim() === '')
      errors.push(`${label}: estimated[${i}] (${e.field}) has an empty reason`);
  });
  return errors;
}

/** Errors in one tiling: positive MMA shape, ≥ 1 CTA tile, tiles aligned to the MMA shape, ctasPerUnit ≥ 1. */
function tilingErrors(t: GemmTiling, label: string): string[] {
  const errors: string[] = [];
  for (const [name, v] of [
    ['mmaM', t.mmaM],
    ['mmaN', t.mmaN],
    ['mmaK', t.mmaK],
    ['ctasPerUnit', t.ctasPerUnit],
  ] as const) {
    if (!Number.isInteger(v) || v < 1)
      errors.push(`${label}: ${name} must be an integer ≥ 1 (got ${v})`);
  }
  if (t.ctaTiles.length === 0) errors.push(`${label}: ctaTiles is empty`);
  t.ctaTiles.forEach((tile, i) => {
    if (tile.m <= 0 || tile.m % t.mmaM !== 0)
      errors.push(
        `${label}: ctaTiles[${i}].m = ${tile.m} is not a positive multiple of mmaM ${t.mmaM}`,
      );
    if (tile.n <= 0 || tile.n % t.mmaN !== 0)
      errors.push(
        `${label}: ctaTiles[${i}].n = ${tile.n} is not a positive multiple of mmaN ${t.mmaN}`,
      );
  });
  return errors;
}

/** True for integer input classes. */
function isIntegerInput(input: MathFormat): boolean {
  return input === 'int8' || input === 'int4';
}

/**
 * Physical-consistency errors of one engine (provenance is checked on the device). Every rate the engine divides
 * by (matmul peaks, vector peaks, the memory-bandwidth cap, the SFU rate) must be > 0; a GEMV row limit is an
 * integer ≥ 1.
 */
function engineErrors(e: ComputeEngine, label: string): string[] {
  const errors: string[] = [];
  if (!Number.isInteger(e.computeUnits) || e.computeUnits < 1) {
    errors.push(`${label}: computeUnits must be an integer ≥ 1 (got ${e.computeUnits})`);
  }
  e.matmulPeaks.forEach((p, i) => {
    if (!(p.flopsPerSec > 0))
      errors.push(`${label}: matmulPeaks.${i}.flopsPerSec must be > 0 (got ${p.flopsPerSec})`);
  });
  for (const [dtype, v] of Object.entries(e.vectorFlopsPerSec)) {
    if (v !== undefined && !(v > 0))
      errors.push(`${label}: vectorFlopsPerSec.${dtype} must be > 0 (got ${v})`);
  }
  if (e.vectorFlopsPerSec.fp32 === undefined)
    errors.push(`${label}: vectorFlopsPerSec.fp32 is required (the last peak fallback)`);
  if (e.memBandwidthCapBytesPerSec !== undefined && !(e.memBandwidthCapBytesPerSec > 0)) {
    errors.push(
      `${label}: memBandwidthCapBytesPerSec must be > 0 (got ${e.memBandwidthCapBytesPerSec})`,
    );
  }
  if (e.sfuOpsPerSec !== undefined && !(Number.isFinite(e.sfuOpsPerSec) && e.sfuOpsPerSec > 0)) {
    errors.push(`${label}: sfuOpsPerSec must be a finite number > 0 (got ${e.sfuOpsPerSec})`);
  }
  if (e.gemvMaxRows !== undefined && !(Number.isInteger(e.gemvMaxRows) && e.gemvMaxRows >= 1)) {
    errors.push(`${label}: gemvMaxRows must be an integer ≥ 1 (got ${e.gemvMaxRows})`);
  }
  const keys = new Set<string>();
  for (const p of e.matmulPeaks) {
    const key = `${p.input}/${p.accumulate}/${p.sparsity}`;
    if (keys.has(key)) errors.push(`${label}: duplicate matmul peak ${key}`);
    keys.add(key);
    if (isIntegerInput(p.input) !== (p.accumulate === 'int32')) {
      errors.push(
        `${label}: peak ${key}: integer inputs accumulate in int32, float inputs in fp16/fp32`,
      );
    }
    if (p.sparsity === '2:4') {
      const dense = e.matmulPeaks.find(
        (d) => d.input === p.input && d.accumulate === p.accumulate && d.sparsity === 'dense',
      );
      if (dense === undefined) errors.push(`${label}: sparse peak ${key} has no dense counterpart`);
      else if (p.flopsPerSec < dense.flopsPerSec)
        errors.push(`${label}: sparse peak ${key} is below its dense peak`);
    }
  }
  for (const input of new Set(e.matmulPeaks.map((p) => p.input))) {
    if (
      !isIntegerInput(input) &&
      !e.matmulPeaks.some((p) => p.input === input && p.accumulate === 'fp32')
    ) {
      errors.push(`${label}: float input ${input} has peaks but no fp32-accumulate entry`);
    }
  }
  for (const [key, tiling] of Object.entries(e.gemmTiling)) {
    if (tiling !== undefined) errors.push(...tilingErrors(tiling, `${label}.gemmTiling.${key}`));
  }
  return errors;
}

/**
 * Errors of a device spec; empty = valid. Checks: numbers finite and ≥ 0; every numeric leaf covered by a
 * source or an estimate; ≥ 1 engine with unique ids; per engine: sparse ≥ dense for the same key, every float
 * input has an fp32-accumulate entry, tiles are multiples of the MMA shape; `capacityOptionsBytes` non-empty
 * and positive; memory bandwidth > 0; every rate used as a divisor (matmul and vector peaks, an engine's memory
 * cap) > 0, and a vector fp32 peak present (the end of the peak fallback chain); a `hostInterface`, when present,
 * names a PCIe generation 3/4/5 and 4/8/16 lanes; `gpuDirectRdma` only on dedicated memory; `gpuUsableFraction` only on
 * unified memory, in (0, 1].
 */
export function validateDevice(d: DeviceSpec): string[] {
  const label = `device ${d.id}`;
  // The nested link owns its provenance, like links on a topology; do not require duplicate device citations.
  const { hostLink, ...deviceNumbers } = d;
  const errors = [
    ...leafErrors(deviceNumbers, d.sources, d.estimated, label),
    ...provenanceRecordErrors(d.sources, d.estimated, label),
  ];
  if (hostLink !== undefined) errors.push(...validateLink(hostLink));
  if (d.engines.length === 0) errors.push(`${label}: no engines`);
  const ids = new Set<string>();
  d.engines.forEach((e, i) => {
    if (ids.has(e.id)) errors.push(`${label}: duplicate engine id '${e.id}'`);
    ids.add(e.id);
    errors.push(...engineErrors(e, `${label}.engines.${i}`));
  });
  if (d.memory.capacityOptionsBytes.length === 0)
    errors.push(`${label}: memory.capacityOptionsBytes is empty`);
  if (d.memory.capacityOptionsBytes.some((c) => c <= 0))
    errors.push(`${label}: memory.capacityOptionsBytes must be > 0`);
  if (!(d.memory.bandwidthBytesPerSec > 0))
    errors.push(`${label}: memory.bandwidthBytesPerSec must be > 0`);
  if (d.hostInterface !== undefined)
    errors.push(...hostInterfaceErrors(d.hostInterface, `${label}.hostInterface`));
  if (d.memory.kind === 'unified' && d.gpuDirectRdma !== undefined) {
    errors.push(
      `${label}: gpuDirectRdma applies to dedicated memory only (unified memory never stages); leave it absent`,
    );
  }
  const f = d.memory.gpuUsableFraction;
  if (f !== undefined) {
    if (d.memory.kind !== 'unified')
      errors.push(
        `${label}: memory.gpuUsableFraction applies to unified memory only; leave it absent`,
      );
    if (!(f > 0 && f <= 1)) errors.push(`${label}: memory.gpuUsableFraction must be in (0, 1]`);
  }
  return errors;
}

/** Errors of a PCIe host interface: generation one of 1 to 5, lanes one of 4 / 8 / 16 (values the type allows). */
function hostInterfaceErrors(h: HostInterfaceSpec, label: string): string[] {
  const errors: string[] = [];
  if (![1, 2, 3, 4, 5].includes(h.generation))
    errors.push(`${label}: generation must be 1, 2, 3, 4 or 5 (got ${h.generation})`);
  if (![4, 8, 16].includes(h.lanes))
    errors.push(`${label}: lanes must be 4, 8 or 16 (got ${h.lanes})`);
  return errors;
}

/** Errors for a number that must be finite and > 0 (a rate, size or latency the engine divides by or adds). */
function positiveErrors(entries: [string, number | undefined][], label: string): string[] {
  return entries
    .filter(([, v]) => v !== undefined && !(Number.isFinite(v) && v > 0))
    .map(([name, v]) => `${label}: ${name} must be finite and > 0 (got ${v})`);
}

/** Errors for a number that must be an integer ≥ 1 (a count). */
function countErrors(entries: [string, number | undefined][], label: string): string[] {
  return entries
    .filter(([, v]) => v !== undefined && !(Number.isInteger(v) && v >= 1))
    .map(([name, v]) => `${label}: ${name} must be an integer ≥ 1 (got ${v})`);
}

/**
 * Errors of a storage tier (host RAM or NVMe SSD); empty = valid. Checks: every numeric leaf covered by a source or
 * an estimate and the provenance records well formed; every size, rate and latency finite and > 0 (the transfer
 * model divides by the bandwidths and IOPS and adds the latencies); `maxOutstanding`, `readGranularityBytes`,
 * `channels` and the rated `randomRead.queueDepth` / `threads` integers ≥ 1. See docs/simulation-model.md §4
 * ("Host and storage tiers").
 */
export function validateStorageTier(t: StorageTierSpec, label = `${t.kind} ${t.id}`): string[] {
  const errors = [
    ...leafErrors(t, t.sources, t.estimated, label),
    ...provenanceRecordErrors(t.sources, t.estimated, label),
  ];
  if (t.kind === 'host_ram') {
    errors.push(
      ...positiveErrors(
        [
          ['capacityBytes', t.capacityBytes],
          ['bandwidthBytesPerSec', t.bandwidthBytesPerSec],
          ['latencySec', t.latencySec],
        ],
        label,
      ),
      ...countErrors(
        [
          ['channels', t.channels],
          ['readGranularityBytes', t.readGranularityBytes],
          ['maxOutstanding', t.maxOutstanding],
        ],
        label,
      ),
    );
  } else {
    errors.push(
      ...positiveErrors(
        [
          ['capacityBytes', t.capacityBytes],
          ['seqReadBandwidthBytesPerSec', t.seqReadBandwidthBytesPerSec],
          ['randomReadLatencySec', t.randomReadLatencySec],
          ['randomRead.iops', t.randomRead?.iops],
          ['linkBandwidthBytesPerSec', t.linkBandwidthBytesPerSec],
          ['sustainedRead.fraction', t.sustainedRead?.fraction],
          ['sustainedRead.blockBytes', t.sustainedRead?.blockBytes],
        ],
        label,
      ),
      ...countErrors(
        [
          ['readGranularityBytes', t.readGranularityBytes],
          ['maxOutstanding', t.maxOutstanding],
          ['randomRead.queueDepth', t.randomRead?.queueDepth],
          ['randomRead.threads', t.randomRead?.threads],
          ['sustainedRead.queueDepth', t.sustainedRead?.queueDepth],
          ['sustainedRead.threads', t.sustainedRead?.threads],
        ],
        label,
      ),
    );
  }
  if (t.kind === 'nvme' && t.sustainedRead !== undefined) {
    if (t.sustainedRead.fraction > 1) errors.push(`${label}: sustainedRead.fraction must be ≤ 1`);
    if (t.sustainedRead.blockBytes < t.readGranularityBytes) {
      errors.push(`${label}: sustainedRead.blockBytes must be ≥ readGranularityBytes`);
    }
  }
  return errors;
}

/**
 * Errors of a node's host spec; empty = valid. The link must pass {@link validateLink} and each present tier
 * {@link validateStorageTier}; `ram` must be a host-RAM tier and `ssd` an NVMe tier. Whether a host is REQUIRED
 * (something placed off-device) is the engine's check.
 */
export function validateHost(h: HostSpec, label = 'host'): string[] {
  const errors: string[] = [];
  if (h.link !== undefined) errors.push(...validateLink(h.link, `${label}.link`));
  if (h.ram !== undefined) {
    if (h.ram.kind !== 'host_ram')
      errors.push(
        `${label}.ram: must be a 'host_ram' tier (got '${(h.ram as StorageTierSpec).kind}')`,
      );
    errors.push(...validateStorageTier(h.ram, `${label}.ram`));
  }
  if (h.ssd !== undefined) {
    if (h.ssd.kind !== 'nvme')
      errors.push(
        `${label}.ssd: must be an 'nvme' tier (got '${(h.ssd as StorageTierSpec).kind}')`,
      );
    errors.push(...validateStorageTier(h.ssd, `${label}.ssd`));
  }
  return errors;
}

/**
 * Errors of a link spec; empty = valid. Checks: numbers finite and ≥ 0 and sourced; bandwidth > 0;
 * `hostBounce` present iff `p2p === 'host_bounce'`, with host memory bandwidth > 0; an
 * `allPortsCapPerDirectionBytesPerSec` at least the per-port bandwidth; `transport: 'tcp'` only with
 * `gpuDirectRdma: false` (a socket takes host pointers only); `inNetworkReduction`, when present, one of `'nvls'`,
 * `'sharp'`; `fullyConnectedDevices`, when present, a safe integer ≥ 2 (sourced domain size, not a neighbour count).
 */
export function validateLink(l: LinkSpec, label = `link ${l.name}`): string[] {
  const errors = [
    ...leafErrors(l, l.sources, l.estimated, label),
    ...provenanceRecordErrors(l.sources, l.estimated, label),
  ];
  if (!(l.bandwidthPerDirectionBytesPerSec > 0))
    errors.push(`${label}: bandwidthPerDirectionBytesPerSec must be > 0`);
  if ((l.p2p === 'host_bounce') !== (l.hostBounce !== undefined)) {
    errors.push(`${label}: hostBounce must be present iff p2p is 'host_bounce'`);
  }
  if (l.hostBounce !== undefined && !(l.hostBounce.hostMemBandwidthBytesPerSec > 0)) {
    errors.push(`${label}: hostBounce.hostMemBandwidthBytesPerSec must be > 0`);
  }
  if (
    l.fullyConnectedDevices !== undefined &&
    (!Number.isSafeInteger(l.fullyConnectedDevices) || l.fullyConnectedDevices < 2)
  ) {
    errors.push(`${label}: fullyConnectedDevices must be a safe integer ≥ 2`);
  }
  const cap = l.allPortsCapPerDirectionBytesPerSec;
  if (cap !== undefined && !(cap >= l.bandwidthPerDirectionBytesPerSec)) {
    errors.push(
      `${label}: allPortsCapPerDirectionBytesPerSec (${cap}) must be ≥ bandwidthPerDirectionBytesPerSec (${l.bandwidthPerDirectionBytesPerSec}): one port alone reaches its own rate`,
    );
  }
  if (l.transport === 'tcp' && l.gpuDirectRdma !== false) {
    errors.push(
      `${label}: transport 'tcp' needs gpuDirectRdma: false (a socket takes host pointers only)`,
    );
  }
  const inNetwork: string | undefined = l.inNetworkReduction;
  if (inNetwork !== undefined && inNetwork !== 'nvls' && inNetwork !== 'sharp') {
    errors.push(`${label}: inNetworkReduction must be 'nvls' or 'sharp' (got '${inNetwork}')`);
  }
  return errors;
}

/**
 * Errors of a topology; empty = valid. `devicesPerNode` and `nodes` must be integers ≥ 1 (they describe the
 * user's configuration, so they need no citation); each present link must pass {@link validateLink}.
 * Whether a link is REQUIRED depends on the parallelism (checked by `engine/placement.ts`).
 * An `interNodeFabric` needs `interNode` and `nodes > 1`, and must pass the structural rules of
 * `fabricShapeErrors` (one device per node for every shape but `'switched'`, a ring of ≥ 3 nodes, valid
 * `cablesPerPair` / `oversubscription`). A `nicsPerNode` needs `interNode`, `nodes > 1` and a switched fabric (absent
 * or `'switched'`: a switchless shape fixes its ports itself), and must pass `nicCountErrors` (integer ≥ 1, divides
 * `devicesPerNode` or is a multiple of it). Port counts are a machine property the topology does not record; the
 * setup catalog checks them.
 */
export function validateTopology(t: TopologySpec): string[] {
  const label = `topology ${t.id}`;
  const errors: string[] = [];
  if (!Number.isInteger(t.devicesPerNode) || t.devicesPerNode < 1)
    errors.push(`${label}: devicesPerNode must be an integer ≥ 1`);
  if (!Number.isInteger(t.nodes) || t.nodes < 1)
    errors.push(`${label}: nodes must be an integer ≥ 1`);
  if (t.intraNode !== undefined) errors.push(...validateLink(t.intraNode, `${label}.intraNode`));
  if (t.interNode !== undefined) errors.push(...validateLink(t.interNode, `${label}.interNode`));
  if (t.interNodeFabric !== undefined) {
    if (t.interNode === undefined) errors.push(`${label}: interNodeFabric needs an interNode link`);
    if (t.nodes <= 1) errors.push(`${label}: interNodeFabric needs nodes > 1 (got ${t.nodes})`);
    const shapeErrors = fabricShapeErrors(t.interNodeFabric, {
      nodes: t.nodes,
      devicesPerNode: t.devicesPerNode,
    });
    errors.push(...shapeErrors.map((e) => `${label}.interNodeFabric: ${e}`));
  }
  if (t.nicsPerNode !== undefined) {
    if (t.interNode === undefined) errors.push(`${label}: nicsPerNode needs an interNode link`);
    if (t.nodes <= 1) errors.push(`${label}: nicsPerNode needs nodes > 1 (got ${t.nodes})`);
    if (t.interNodeFabric !== undefined && t.interNodeFabric.kind !== 'switched')
      errors.push(
        `${label}: nicsPerNode needs a switched fabric (got ${t.interNodeFabric.kind}, whose ports are its cables)`,
      );
    errors.push(...nicCountErrors(t.nicsPerNode, t.devicesPerNode).map((e) => `${label}: ${e}`));
  }
  return errors;
}

/**
 * Errors of applying `mods` to `base`, in order; empty = valid. Each modifier is checked against the spec as
 * modified by the ones before it (factor > 0, engineId exists, a peakScale filter matches ≥ 1 peak, provenance
 * present), and the final result must pass {@link validateDevice}.
 */
export function validateDeviceModifiers(base: DeviceSpec, mods: DeviceModifier[]): string[] {
  let spec = base;
  for (const [i, mod] of mods.entries()) {
    const errors = deviceModifierErrors(spec, mod).map((e) => `deviceModifiers[${i}]: ${e}`);
    if (errors.length > 0) return errors;
    spec = applyDeviceModifiers(spec, [mod]);
  }
  return validateDevice(spec).map((e) => `after modifiers: ${e}`);
}

/** Errors of applying `mods` to `link`, in order (as {@link validateDeviceModifiers}); empty = valid. */
export function validateLinkModifiers(link: LinkSpec, mods: LinkModifier[]): string[] {
  let current = link;
  for (const [i, mod] of mods.entries()) {
    const errors = linkModifierErrors(current, mod).map((e) => `linkModifiers[${i}]: ${e}`);
    if (errors.length > 0) return errors;
    current = applyLinkModifiers(current, [mod]);
  }
  return validateLink(current).map((e) => `after link modifiers: ${e}`);
}
