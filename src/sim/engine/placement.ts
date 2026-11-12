/**
 * Placement of parallel groups on a topology: which link level each TP / EP group and each pipeline-stage
 * boundary uses, and whether the parallelism fits the devices at all.
 * See docs/simulation-model.md §5 ("Rank layout").
 *
 * Rank layout: `rank = tpIdx + tp · (dpIdx + dp · ppIdx)`; ranks fill node 0 first (`node = floor(rank / dpn)`,
 * dpn = devicesPerNode). TP groups (and EP groups, which are `ep` consecutive ranks) are therefore contiguous.
 *
 * Inter-node fabric (`TopologySpec.interNodeFabric`, docs §6 "Fabric shapes"): node i of a ring or chain is cabled to
 * node i ± 1 in this rank order. A path carries a fabric only when the topology's is not the non-blocking switch, so
 * every pre-fabric topology gets exactly its old paths.
 *
 * Multi-NIC nodes and in-network reduction (docs §6 "Multi-NIC nodes", "In-network reduction (NVLS, SHARP)"): every
 * inter-node path carries the device's share of the node's NICs (`hw/nics.ts: interNodeLinkPerDevice`, the topology's
 * own link object with one NIC per device), and a path carries an in-network reduction only when the run switches it
 * on and the level's link has one, so every topology without them gets exactly its old paths.
 */

import type { ParallelConfig } from '../ir/model';
import type { InNetworkReduction, LinkSpec, TopologySpec } from '../hw/types';
import { groupFabric, isNonBlockingSwitched, nodeHops } from '../hw/fabric';
import { interNodeLinkPerDevice } from '../hw/nics';
import type { CollectivePath } from './collectives';
import type { HostStaging } from './staging';

/** How a contiguous group of devices lies on the topology. */
export interface GroupSpan {
  /** Devices in the group. Count. */
  n: number;
  /** Group members in each node it touches (n1). Count. */
  devicesPerNodeInGroup: number;
  /** Nodes the group touches (n2); n = n1 · n2. Count. */
  nodesSpanned: number;
}

/**
 * Errors placing a contiguous group of `size` devices; empty = valid. A group is intra-node if
 * `size ≤ dpn` and `dpn % size == 0` (it never straddles a node boundary), hierarchical if `size % dpn == 0`
 * and `size / dpn ≤ nodes`; anything else straddles nodes unevenly and is an error.
 */
export function groupSpanErrors(size: number, topo: TopologySpec): string[] {
  const dpn = topo.devicesPerNode;
  if (!Number.isInteger(size) || size < 1)
    return [`group size must be an integer ≥ 1 (got ${size})`];
  if (size <= dpn && dpn % size === 0) return [];
  if (size % dpn === 0 && size / dpn <= topo.nodes) return [];
  if (size % dpn === 0)
    return [
      `a group of ${size} needs ${size / dpn} nodes of ${dpn}; the topology has ${topo.nodes}`,
    ];
  return [
    `a group of ${size} devices does not tile nodes of ${dpn} devices (needs size | ${dpn} or ${dpn} | size)`,
  ];
}

/** Span of a contiguous group of `size` devices (TP and EP groups). Throws if {@link groupSpanErrors} reports any. */
export function groupSpan(size: number, topo: TopologySpec): GroupSpan {
  const errors = groupSpanErrors(size, topo);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const dpn = topo.devicesPerNode;
  if (size <= dpn) return { n: size, devicesPerNodeInGroup: size, nodesSpanned: 1 };
  // Formula: n1 = dpn;  n2 = n / dpn
  return { n: size, devicesPerNodeInGroup: dpn, nodesSpanned: size / dpn };
}

/** How the expert-TP group of a mode-A MoE layer lies on the topology (see {@link expertTpGroupSpan}). */
export interface ExpertTpSpan extends GroupSpan {
  /** Node-index distance between consecutive member nodes: 1 = adjacent nodes, `ep/dpn` when ep > dpn. Count. */
  nodeStride: number;
}

/**
 * Span of the expert-TP group: in MoE mode A (`tp % ep == 0`) the `etp = tp/ep` ranks of a TP group that hold the
 * same experts, each a `fe/etp` slice. EP groups are `ep` consecutive ranks (rank layout above), so the rank with
 * EP index j shares its experts with `j + i·ep`, i < etp: a group strided by ep inside the contiguous, node-aligned
 * TP group. Hence (dpn = devicesPerNode, spans of tp and ep valid):
 * - tp ≤ dpn: one node, `n1 = etp`, `n2 = 1`;
 * - tp > dpn, ep ≤ dpn (ep | dpn): `n1 = dpn/ep` per node over `n2 = tp/dpn` adjacent nodes;
 * - tp > dpn, ep > dpn (dpn | ep): one member per node, `n2 = etp` nodes `ep/dpn` apart.
 * etp = 1 (mode B, or ep = tp) is the one-device span. Throws if the tp or ep span is invalid.
 */
export function expertTpGroupSpan(
  par: Pick<ParallelConfig, 'tp' | 'ep'>,
  topo: TopologySpec,
): ExpertTpSpan {
  const { tp, ep } = par;
  groupSpan(tp, topo);
  groupSpan(ep, topo);
  const dpn = topo.devicesPerNode;
  // Formula: etp = tp/ep (mode A), else 1
  const etp = tp % ep === 0 ? tp / ep : 1;
  if (etp === 1) return { n: 1, devicesPerNodeInGroup: 1, nodesSpanned: 1, nodeStride: 1 };
  if (tp <= dpn) return { n: etp, devicesPerNodeInGroup: etp, nodesSpanned: 1, nodeStride: 1 };
  // Formula: ep ≤ dpn: n1 = dpn/ep, n2 = tp/dpn, stride 1;  ep > dpn: n1 = 1, n2 = etp, stride ep/dpn
  if (ep <= dpn)
    return { n: etp, devicesPerNodeInGroup: dpn / ep, nodesSpanned: tp / dpn, nodeStride: 1 };
  return { n: etp, devicesPerNodeInGroup: 1, nodesSpanned: etp, nodeStride: ep / dpn };
}

/** The run's in-network reduction switch (`SimOptions.inNetworkReduction`): `'off'` (default) or `'on'`. */
export type InNetworkSwitch = 'off' | 'on';

/**
 * The in-network reduction a path on `link` may use: the link's `inNetworkReduction` capability when the switch is
 * `'on'`, else undefined (also for an absent link or a link without the capability).
 */
export function inNetworkCapability(
  link: LinkSpec | undefined,
  inNetwork: InNetworkSwitch,
): InNetworkReduction | undefined {
  return inNetwork === 'on' ? link?.inNetworkReduction : undefined;
}

/**
 * The {@link CollectivePath} of the expert-TP group ({@link expertTpGroupSpan}), built like a contiguous group's by
 * {@link collectivePath}: within a node the intra-node link, across nodes the inter-node link (hierarchical when
 * both). Members on non-adjacent nodes (`nodeStride > 1`) are modeled on a switched or full-mesh fabric (one hop
 * between any two nodes) and rejected on a ring or chain, whose hop counts assume adjacent nodes. `inNetwork` as for
 * {@link collectivePath}.
 */
export function expertTpCollectivePath(
  par: Pick<ParallelConfig, 'tp' | 'ep'>,
  topo: TopologySpec,
  staging?: HostStaging,
  inNetwork: InNetworkSwitch = 'off',
): CollectivePath {
  const span = expertTpGroupSpan(par, topo);
  const shape = topo.interNodeFabric;
  if (span.nodeStride > 1 && (shape?.kind === 'ring' || shape?.kind === 'chain')) {
    throw new Error(
      `expert-TP group of ${span.n} nodes ${span.nodeStride} apart on a ${shape.kind} fabric is not modeled (needs ep ≤ devices per node, or a switched or full-mesh fabric)`,
    );
  }
  return collectivePath(span, topo, staging, inNetwork);
}

/**
 * DP peers have the same TP index: rank = tpIdx + tp·(dpIdx + dp·ppIdx). This is the same strided geometry
 * as expertTpGroupSpan with parent size tp·dp and stride tp, NOT the same communication group as expert-TP.
 * Returns the strided DP path and contiguous TP×DP path, with the same staging, NIC sharing and in-network
 * capabilities as {@link collectivePath}. A whole run inside one node need not tile that node's unused devices.
 * See docs/simulation-model.md §5 "Lookup tables across DP".
 */
export function lookupTableCollectivePaths(
  par: ParallelConfig,
  topo: TopologySpec,
  staging?: HostStaging,
  inNetwork: InNetworkSwitch = 'off',
): { dp: CollectivePath; tp_dp: CollectivePath } {
  // Formula: N = tp·dp; a wholly intra-node run (N·pp ≤ devicesPerNode) has spans (dp, 1) and (N, 1).
  const n = par.tp * par.dp;
  const withinNode = n * par.pp <= topo.devicesPerNode;
  const span = withinNode
    ? { n: par.dp, devicesPerNodeInGroup: par.dp, nodesSpanned: 1, nodeStride: 1 }
    : expertTpGroupSpan({ tp: n, ep: par.tp }, topo);
  const combined = withinNode
    ? { n, devicesPerNodeInGroup: n, nodesSpanned: 1 }
    : groupSpan(n, topo);
  const shape = topo.interNodeFabric;
  if (span.nodeStride > 1 && (shape?.kind === 'ring' || shape?.kind === 'chain')) {
    throw new Error(
      `DP group of ${span.n} nodes ${span.nodeStride} apart on a ${shape.kind} fabric is not modeled`,
    );
  }
  return {
    dp: collectivePath(span, topo, staging, inNetwork),
    tp_dp: collectivePath(combined, topo, staging, inNetwork),
  };
}

/**
 * Direct peer-HBM lookup: TP1 inside one direct NVLink domain whose full connectivity and size are established
 * by the link profile. Unknown/partial connectivity or an oversized node uses the collective path instead;
 * neither `p2p: 'direct'` nor NVLS proves full connectivity. See docs/simulation-model.md §5 "Lookup tables across DP".
 */
export function lookupTablesUsePeerReads(par: ParallelConfig, topo: TopologySpec): boolean {
  const link = topo.intraNode;
  return (
    par.tp === 1 &&
    par.dp > 1 &&
    par.tp * par.dp * par.pp <= topo.devicesPerNode &&
    link?.p2p === 'direct' &&
    topo.devicesPerNode <= (link.fullyConnectedDevices ?? 0) &&
    (link.kind === 'nvlink' || link.kind === 'nvlink_bridge')
  );
}

/**
 * True if the boundary between pipeline stage `stage` and `stage + 1` crosses nodes: some rank r of the stage
 * sends to r + tp·dp in another node (`floor(r / dpn)` differs).
 * `destinationStage` is a zero-based stage index, default stage + 1; 0 selects the decode loop-back.
 */
export function ppBoundaryIsInterNode(
  par: ParallelConfig,
  topo: TopologySpec,
  stage: number,
  destinationStage = stage + 1,
): boolean {
  const dpn = topo.devicesPerNode;
  const stride = par.tp * par.dp;
  for (let i = 0; i < stride; i++) {
    // Formula: r = tpIdx + tp·(dpIdx + dp·s) = s·tp·dp + i,  partner = r + tp·dp
    const rank = stage * stride + i;
    const partner = destinationStage * stride + i;
    if (Math.floor(rank / dpn) !== Math.floor(partner / dpn)) return true;
  }
  return false;
}

/**
 * The {@link CollectivePath} of a group with span `span`: `'none'` for one device, flat intra-node, flat
 * inter-node (one member per node), or hierarchical. Throws if the topology lacks the needed link.
 * A flat inter-node path carries the group's view of the fabric (`groupFabric`: a ring segment is a chain) unless it
 * is the non-blocking switch; a hierarchical path carries an oversubscribed switch (the only shape allowed with
 * several devices per node). `staging` (the run's inter-node host staging, `staging.ts`) is attached to the
 * inter-node level only (`interStaging`); absent = the path without staging, exactly as before.
 * The inter-node link is the device's share of the node's NICs (`hw/nics.ts: interNodeLinkPerDevice`; the
 * topology's own link object when `nicsPerNode` is absent).
 * `inNetwork` (`SimOptions.inNetworkReduction`, default `'off'`): when `'on'`, a level whose link has an
 * `inNetworkReduction` capability gets it on the path (flat `inNetworkReduction`, hierarchical
 * `intraInNetworkReduction` / `interInNetworkReduction`); the inter-node level only without host staging and,
 * when flat, without a fabric shape (switchless shapes and an oversubscribed flat switch keep ring / tree).
 */
export function collectivePath(
  span: GroupSpan,
  topo: TopologySpec,
  staging?: HostStaging,
  inNetwork: InNetworkSwitch = 'off',
): CollectivePath {
  if (span.n <= 1) return { kind: 'none' };
  const needIntra = span.devicesPerNodeInGroup > 1;
  const needInter = span.nodesSpanned > 1;
  if (needIntra && topo.intraNode === undefined)
    throw new Error(`topology ${topo.id} has no intraNode link`);
  const interNode = interNodeLinkPerDevice(topo);
  if (needInter && interNode === undefined)
    throw new Error(`topology ${topo.id} has no interNode link`);
  const shape = topo.interNodeFabric;
  const interStaging = staging === undefined ? {} : { interStaging: staging };
  const intraInSwitch = inNetworkCapability(topo.intraNode, inNetwork);
  // Host staging bounces every message through host memory: no in-switch reduction on a staged inter-node level.
  const interInSwitch =
    staging === undefined ? inNetworkCapability(interNode, inNetwork) : undefined;
  if (needIntra && needInter && topo.intraNode !== undefined && interNode !== undefined) {
    const path: CollectivePath = {
      kind: 'hierarchical',
      devicesPerNode: span.devicesPerNodeInGroup,
      nodes: span.nodesSpanned,
      intra: topo.intraNode,
      inter: interNode,
      ...interStaging,
      ...(intraInSwitch !== undefined ? { intraInNetworkReduction: intraInSwitch } : {}),
      ...(interInSwitch !== undefined ? { interInNetworkReduction: interInSwitch } : {}),
    };
    if (isNonBlockingSwitched(shape) || shape === undefined) return path;
    if (shape.kind !== 'switched')
      throw new Error(`topology ${topo.id}: a ${shape.kind} fabric needs one device per node`);
    return {
      ...path,
      interFabric: { kind: 'switched', oversubscription: shape.oversubscription ?? 1 },
    };
  }
  if (needInter && interNode !== undefined) {
    const fabric = groupFabric(shape, span.nodesSpanned, topo.nodes);
    if (isNonBlockingSwitched(fabric) || fabric === undefined)
      return {
        kind: 'flat',
        level: 'inter',
        link: interNode,
        ...interStaging,
        ...(interInSwitch !== undefined ? { inNetworkReduction: interInSwitch } : {}),
      };
    return { kind: 'flat', level: 'inter', link: interNode, fabric, ...interStaging };
  }
  if (topo.intraNode !== undefined)
    return {
      kind: 'flat',
      level: 'intra',
      link: topo.intraNode,
      ...(intraInSwitch !== undefined ? { inNetworkReduction: intraInSwitch } : {}),
    };
  throw new Error(`topology ${topo.id} has no link for a group of ${span.n}`);
}

/**
 * Links the activation send after stage `stage` crosses on the inter-node fabric: the most over the stage's ranks r
 * of `nodeHops(node(r), node(r + tp·dp))`, node(r) = floor(r / dpn). 1 on a switch or mesh; on a ring or chain, 1 for
 * pure PP (stages on neighbours) and tp·dp for one device per node (e.g. tp 2 × pp 2 on a ring of 4: 2). Count.
 * `destinationStage` is a zero-based stage index (default stage + 1); use 0 for the decode return.
 */
export function ppSendHops(
  par: ParallelConfig,
  topo: TopologySpec,
  stage: number,
  destinationStage = stage + 1,
): number {
  const dpn = topo.devicesPerNode;
  const stride = par.tp * par.dp;
  let hops = 0;
  for (let i = 0; i < stride; i++) {
    // Formula: h = max_i nodeHops(floor((stage·tp·dp+i)/dpn), floor((destinationStage·tp·dp+i)/dpn))
    const rank = stage * stride + i;
    const partner = destinationStage * stride + i;
    hops = Math.max(
      hops,
      nodeHops(topo.interNodeFabric, Math.floor(rank / dpn), Math.floor(partner / dpn), topo.nodes),
    );
  }
  return hops;
}

/**
 * The flat path of the activation send from stage `stage` to `stage + 1` (intra- or inter-node link). Throws if
 * missing. An inter-node send carries the topology's fabric unless it is the non-blocking switch, and `sendHops`
 * ({@link ppSendHops}) when that is not 1.
 * `destinationStage` is a zero-based stage index (default stage + 1); use 0 for the decode return. `staging` (the
 * run's inter-node host staging) is attached only when the send crosses nodes. An inter-node send runs at the
 * device's share of the node's NICs (`hw/nics.ts: interNodeLinkPerDevice`).
 */
export function ppSendPath(
  par: ParallelConfig,
  topo: TopologySpec,
  stage: number,
  destinationStage = stage + 1,
  staging?: HostStaging,
): CollectivePath {
  const inter = ppBoundaryIsInterNode(par, topo, stage, destinationStage);
  const link = inter ? interNodeLinkPerDevice(topo) : topo.intraNode;
  if (link === undefined)
    throw new Error(
      `topology ${topo.id} has no ${inter ? 'interNode' : 'intraNode'} link for the PP boundary after stage ${stage}`,
    );
  const fabric = topo.interNodeFabric;
  // Formula (G8): nActive = min(devicesPerNode, tp·dp), for both forward activations and decode token return.
  // See docs/simulation-model.md §6 "Host bounce".
  const hostSharing =
    link.p2p === 'host_bounce'
      ? { sendActiveDevices: Math.min(topo.devicesPerNode, par.tp * par.dp) }
      : {};
  const interStaging = inter && staging !== undefined ? { interStaging: staging } : {};
  if (!inter || isNonBlockingSwitched(fabric) || fabric === undefined)
    return {
      kind: 'flat',
      level: inter ? 'inter' : 'intra',
      link,
      ...hostSharing,
      ...interStaging,
    };
  const sendHops = ppSendHops(par, topo, stage, destinationStage);
  return {
    kind: 'flat',
    level: 'inter',
    link,
    fabric,
    ...hostSharing,
    ...(sendHops !== 1 ? { sendHops } : {}),
    ...interStaging,
  };
}

/**
 * True when some communication of `par` on `topo` crosses nodes: a TP or EP group spans more than one node, or a
 * pipeline boundary (incl. the decode loop-back from the last stage to stage 0) is inter-node. Used to report host
 * staging only for runs that have inter-node messages, including lookup exchanges when `sharedLookupTables` is on.
 * Assumes {@link checkPlacement} found no errors.
 */
export function hasInterNodeTraffic(
  par: ParallelConfig,
  topo: TopologySpec,
  sharedLookupTables = false,
): boolean {
  if (groupSpan(par.tp, topo).nodesSpanned > 1 || groupSpan(par.ep, topo).nodesSpanned > 1)
    return true;
  if (sharedLookupTables && par.tp * par.dp > topo.devicesPerNode) return true;
  for (let stage = 0; stage + 1 < par.pp; stage++) {
    if (ppBoundaryIsInterNode(par, topo, stage)) return true;
  }
  return par.pp > 1 && ppBoundaryIsInterNode(par, topo, par.pp - 1, 0);
}

/** Result of {@link checkPlacement}: errors make the input invalid; warnings are informational lines. */
export interface PlacementCheck {
  /** Invalid placement (too many devices, a group that straddles nodes unevenly, a missing link). */
  errors: string[];
  /** E.g. unused devices. */
  warnings: string[];
}

/**
 * Checks that `par` fits `topo`: world `tp·pp·dp ≤ dpn·nodes` (unused devices are a warning); TP, EP and DCP groups
 * (DCP: the `dcp ?? 1` consecutive ranks of a TP group, contiguous like TP; size 1 never errs) place per
 * {@link groupSpanErrors}; every link a TP/EP/DCP group or a PP boundary needs exists.
 */
export function checkPlacement(par: ParallelConfig, topo: TopologySpec): PlacementCheck {
  const errors: string[] = [];
  const warnings: string[] = [];
  const world = par.tp * par.pp * par.dp;
  const devices = topo.devicesPerNode * topo.nodes;
  if (world > devices) {
    errors.push(`tp·pp·dp = ${world} devices needed, the topology has ${devices}`);
    return { errors, warnings };
  }
  if (world < devices)
    warnings.push(`${devices - world} of ${devices} devices are unused (tp·pp·dp = ${world})`);
  for (const [name, size] of [
    ['tp', par.tp],
    ['ep', par.ep],
    ['dcp', par.dcp ?? 1],
  ] as const) {
    const spanErrors = groupSpanErrors(size, topo).map((e) => `${name}: ${e}`);
    errors.push(...spanErrors);
    if (spanErrors.length === 0) {
      try {
        collectivePath(groupSpan(size, topo), topo);
      } catch (e) {
        errors.push(`${name}: ${(e as Error).message}`);
      }
    }
  }
  for (let stage = 0; stage + 1 < par.pp; stage++) {
    try {
      ppSendPath(par, topo, stage);
    } catch (e) {
      errors.push(`pp: ${(e as Error).message}`);
    }
  }
  return { errors, warnings };
}
