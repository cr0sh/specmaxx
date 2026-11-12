/**
 * `simulate`: the theoretical-maximum prefill and decode rates of a model on a hardware stack.
 * See docs/simulation-model.md §7 and §8.
 *
 * Steps:
 * 1. {@link validateInput} (throws on any error).
 * 2. Apply the device and link modifiers once (docs §4 "Device variants and modifiers"); everything after sees
 *    plain specs.
 * 3. Resolve the offload placement and the host side, if any (docs §12); `hardware.host` is read only then.
 * 4. For each phase: lower the model (`seqs = b_mb`; with two-batch overlap as two merged halves, `twoBatch.ts`) →
 *    per-op times (transfer ops on their stage's tier paths) → per-stage times under the resolved overlap fractions
 *    (`overlap.ts: resolveOverlap, coreTime`) → pipeline + DP.
 * 4a. Chunked pipeline prefill (docs §7 "Chunked pipeline prefill"), when on with pp > 1 and a prompt of K > 1
 *    chunks: each chunk lowered and costed as above, then the chunked schedule (`chunkedPipeline.ts`); the
 *    whole-prompt prefill stays as `prefill.chunkedPipeline.unchunked`.
 * 4b. Labelled what-if figures next to the headline (docs §7): prefill with every overlap fraction 0 and two-batch
 *    overlap off when some fraction is above 0 and there is inter-device communication, and decode with the engine
 *    dispatch overlay when it is on. Neither changes the
 *    headline phases.
 * 5. Memory fit (device and the off-device tiers).
 */

import {
  applyDeviceModifiers,
  applyLinkModifiers,
  describeDeviceModifier,
  describeLinkModifier,
} from '../hw/modifiers';
import type { DeviceSpec, HardwareStack, TopologySpec } from '../hw/types';
import { crossLayerReferences } from '../ir/validateModel';
import type { DraftSpec, ModelSpec } from '../ir/model';
import type { TransferSource } from '../ir/ops';
import { actBpe, stepTokens } from '../lower/common';
import { lowerDraftDecode, lowerDraftPrefill, ppBoundaryDraftFeatureVectors } from '../lower/draft';
import {
  lowerModel,
  stageLayerRange,
  type LoweredLayer,
  type LoweredModel,
  type LoweringContext,
} from '../lower/lower';
import type { AllReduceAlgorithm, CollectivePath } from './collectives';
import {
  CHUNKED_PIPELINE_SINGLE_STAGE_WARNING,
  chunkedPipelinePhase,
  chunkedPipelinePlan,
  type ChunkedPipelinePlan,
} from './chunkedPipeline';
import {
  offloadTiers,
  remotePoolPath,
  resolveHost,
  resolveOffload,
  resolveRemotePool,
  transferPath,
  type ResolvedHost,
} from './host';
import {
  DEFAULT_PREFILL_CHUNK_TOKENS,
  memoryReport,
  prefillChunkShape,
  resolveGpuUsableFraction,
  unifiedMemoryWarnings,
} from './memory';
import { engineMemBandwidthBytesPerSec, opTime, type OpTimeContext } from './opTime';
import { anyOverlap, coreTime, NO_OVERLAP, resolveOverlap, type OverlapFractions } from './overlap';
import {
  layerResult,
  microbatching,
  phaseResult,
  stageCoreTimeInput,
  stageResult,
  type PhaseShape,
} from './pipeline';
import {
  checkPlacement,
  collectivePath,
  expertTpCollectivePath,
  groupSpan,
  hasInterNodeTraffic,
  lookupTableCollectivePaths,
  lookupTablesUsePeerReads,
  ppSendPath,
} from './placement';
import { resolveSpeculative } from './specdec';
import { resolveHostStaging, type HostStaging } from './staging';
import { stageNStream, type TransferPaths } from './transfer';
import { mergeTwoBatch, twoBatchHalves, TWO_BATCH_NOTHING_TO_SPLIT_WARNING } from './twoBatch';
import { cachedPrefixTokens, computedPromptTokens } from './types';
import type {
  PhaseRates,
  PhaseResult,
  SimInput,
  SimOptions,
  SimResult,
  StageResult,
} from './types';
import { DEFAULT_DISPATCH_SEC_PER_KERNEL, stageKernelCount, withDispatch } from './dispatch';
import { validateInput } from './validate';
import { lookupTableConfig } from './offloadRules';

/**
 * The defaults of {@link SimOptions}: no elementwise fusion, ideal comm/compute overlap, best all-reduce, in-network
 * reduction off. (Offload prefetching is a deployment knob, `OffloadConfig.prefetchAccuracy`.)
 */
export const DEFAULT_SIM_OPTIONS: SimOptions = {
  overlap: 'ideal',
  elementwiseFusion: 'none',
  collectiveAlgorithm: 'best',
  inNetworkReduction: 'off',
  perOpLatencyFloorSec: 0,
};

/** The hardware after modifiers: the device and topology the engine actually simulates, plus info lines. */
export interface EffectiveHardware {
  /** Device after `deviceModifiers` (the base object itself when there are none). */
  device: DeviceSpec;
  /** Topology with `linkModifiers` applied to its links. */
  topology: TopologySpec;
  /** One "variant: …" line per applied modifier. */
  notes: string[];
}

/** Applies a stack's device and link modifiers (in order), without mutating the stack. */
export function effectiveHardware(hardware: HardwareStack): EffectiveHardware {
  const deviceMods = hardware.deviceModifiers ?? [];
  const intraMods = hardware.linkModifiers?.intraNode ?? [];
  const interMods = hardware.linkModifiers?.interNode ?? [];
  const device =
    deviceMods.length > 0 ? applyDeviceModifiers(hardware.device, deviceMods) : hardware.device;
  const topology: TopologySpec = { ...hardware.topology };
  if (topology.intraNode !== undefined && intraMods.length > 0)
    topology.intraNode = applyLinkModifiers(topology.intraNode, intraMods);
  if (topology.interNode !== undefined && interMods.length > 0)
    topology.interNode = applyLinkModifiers(topology.interNode, interMods);
  const notes = [
    ...deviceMods.map(describeDeviceModifier),
    ...intraMods.map((m) => `intraNode ${describeLinkModifier(m)}`),
    ...interMods.map((m) => `interNode ${describeLinkModifier(m)}`),
  ];
  return { device, topology, notes };
}

/**
 * One warning per cross-layer reference (shared KV cache, reused top-k, shared indexer keys) whose source layer
 * sits on an earlier pipeline stage than its consumer. Documented approximation: the consumer's stage is assumed to
 * hold the data (no transfer is costed, and the data is not counted twice in memory). See docs §9.
 */
export function crossStageWarnings(model: ModelSpec, pp: number): string[] {
  const stageOf = (layer: number): number => {
    for (let s = 0; s < pp; s++) {
      const [start, end] = stageLayerRange(model.layers.length, pp, s);
      if (layer >= start && layer < end) return s;
    }
    return -1;
  };
  return crossLayerReferences(model)
    .filter((ref) => stageOf(ref.source) !== stageOf(ref.consumer))
    .map(
      (ref) =>
        `layer ${ref.consumer} reads the ${ref.what.replaceAll('_', ' ')} of layer ${ref.source} on another pipeline stage; ` +
        'its transfer is not modeled',
    );
}

/** The resolved host side of an offloading run, for {@link simulatePhase}. */
interface PhaseOffload {
  /** Host link, RAM, SSD and, with a KV share in it, the remote KV pool. */
  host: ResolvedHost;
  /** The deployment's queue depth request (capped per tier). Count. */
  queueDepth: number | undefined;
  /** SSD bandwidth scenario; absent preserves the datasheet rate. */
  ssdReadRate: 'datasheet' | 'sustained' | undefined;
  /** Prefetching on, i.e. prefetch accuracy p > 0 (more stages stream at once; docs §12 "Expert prefetch"). */
  prefetching: boolean;
}

/**
 * Transfer paths of stage `s` per tier this phase streams from (design doc §2.3): a tier is shared by the
 * concurrently streaming stages of the node ({@link stageNStream}), which depends on which stages hold a transfer op
 * of that tier in this phase, whether prefetching is on (p > 0) and the microbatches `m`.
 *
 * Host RAM is one resource behind two paths: host-RAM-tier reads and the staging hop of NVMe reads (dedicated memory).
 * Its sharing n_host counts every stage that uses it either way, each device once:
 *   hostStreams[s] = streams_ram[s] ∨ streams_nvme[s];  n_host = stageNStream(hostStreams)
 * The host-RAM tier is shared by n_host, and the NVMe staging term by n_host; the SSD's own bandwidth and IOPS stay
 * shared by n_nvme. With one tier streaming, hostStreams = streams_tier, so n_host = n_tier (results unchanged).
 * The remote KV pool's path is the device's own NIC share: no n_stream (docs §12 "Remote KV pool").
 * See docs/simulation-model.md §12 ("Sharing").
 */
function stageTransferPaths(
  lowered: LoweredModel,
  s: number,
  offload: PhaseOffload,
  input: SimInput,
  topology: TopologySpec,
  microbatches: number,
): TransferPaths {
  const paths: TransferPaths = {};
  const streamsFrom = (tier: TransferSource): boolean[] =>
    lowered.stages.map((st) =>
      st.layers.some((l) => l.ops.some((op) => op.kind === 'transfer' && op.source === tier)),
    );
  const nStreamOf = (streams: readonly boolean[]): number =>
    stageNStream(
      input.parallel,
      topology.devicesPerNode,
      streams,
      s,
      offload.prefetching,
      microbatches,
    );
  const streams = { host_ram: streamsFrom('host_ram'), nvme: streamsFrom('nvme') };
  // Formula: hostStreams[s] = streams_ram[s] ∨ streams_nvme[s];  n_host = stageNStream(hostStreams)
  const hostStreams = streams.host_ram.map((ram, i) => ram || streams.nvme[i] === true);
  const n_host = hostStreams[s] === true ? nStreamOf(hostStreams) : 1;
  if (streams.host_ram[s] === true) {
    // Formula: n(host_ram) = n_host
    paths.host_ram = transferPath(
      'host_ram',
      offload.host,
      offload.queueDepth,
      n_host,
      offload.ssdReadRate,
    );
  }
  if (streams.nvme[s] === true) {
    // Formula: n(nvme) = n_nvme (drive bandwidth, IOPS);  staging shares host RAM by n_host
    const n_nvme = nStreamOf(streams.nvme);
    paths.nvme = transferPath(
      'nvme',
      offload.host,
      offload.queueDepth,
      n_nvme,
      offload.ssdReadRate,
      n_host,
    );
  }
  if (streamsFrom('remote_pool')[s] === true) {
    // Formula: t_pool = α_pool + max(useful/B_pool, useful/B_link)   (per device; no sharing term)
    paths.remote_pool = remotePoolPath(offload.host);
  }
  return paths;
}

/**
 * Lowers and costs one phase (prefill or decode) for one microbatch, then applies the pipeline and DP rules. With
 * `dispatchSecPerKernel` (the engine dispatch overlay's τ, seconds) each stage time first gets its kernels' dispatch
 * cost added (`dispatch.ts: withDispatch`); absent = the theoretical stage times.
 */
function simulatePhase(
  lowered: LoweredModel,
  phase: 'prefill' | 'decode',
  base: Omit<OpTimeContext, 'collectivePaths'> & {
    tpPath: CollectivePath;
    epPath: CollectivePath;
    /** The run's inter-node host staging (`staging.ts`), for the PP sends; absent = none. */
    interStaging?: HostStaging;
  },
  input: SimInput,
  topology: TopologySpec,
  options: SimOptions,
  /** Resolved overlap fractions of the run (`overlap.ts: resolveOverlap`). */
  overlap: OverlapFractions,
  shape: {
    sequencesPerReplica: number;
    microbatches: number;
    microbatchSize: number;
    /**
     * Work units circulating through the stages at once, for the offload stream sharing (`transfer.ts: streamSlots`,
     * which bounds the stages streaming at the same time by it). Absent = `microbatches` (m). Chunked pipeline
     * prefill passes m·K: every chunk of every microbatch occupies a stage (docs §7 "Chunked pipeline prefill").
     * Count.
     */
    unitsInFlight?: number;
  },
  offload: PhaseOffload | undefined,
  tokensPerStep?: number,
  dispatchSecPerKernel?: number,
): PhaseResult {
  let loopBack: ReturnType<typeof opTime> | undefined;
  // Expert-TP collectives (MoE expertOutputNorm with etp > 1) get their own strided-group path (placement.ts:
  // expertTpCollectivePath); a model without them builds none, so its contexts are exactly as before.
  const usesExpertTp = lowered.stages.some((st) =>
    st.layers.some((l) => l.ops.some((op) => op.kind === 'collective' && op.group === 'etp')),
  );
  const etpPath = usesExpertTp
    ? {
        etp: expertTpCollectivePath(
          input.parallel,
          topology,
          base.interStaging,
          options.inNetworkReduction,
        ),
      }
    : {};
  // Decode context-parallel collectives (parallel.dcp > 1, decode) run over the d consecutive ranks of a TP group, a
  // contiguous group placed like TP (docs/simulation-model.md §5 "Decode context parallelism (DCP)"); a phase
  // without them builds no path, so its contexts are exactly as before.
  const usesDcp = lowered.stages.some((st) =>
    st.layers.some((l) => l.ops.some((op) => op.kind === 'collective' && op.group === 'dcp')),
  );
  const dcpPath = usesDcp
    ? {
        dcp: collectivePath(
          groupSpan(input.parallel.dcp ?? 1, topology),
          topology,
          base.interStaging,
          options.inNetworkReduction,
        ),
      }
    : {};
  const lookupPaths =
    lookupTableConfig(input.deployment?.offload).acrossDp === true && input.parallel.dp > 1
      ? lookupTableCollectivePaths(
          input.parallel,
          topology,
          base.interStaging,
          options.inNetworkReduction,
        )
      : {};
  const stages = lowered.stages.map((stage) => {
    const ppPath: CollectivePath =
      stage.sendToNext === undefined
        ? { kind: 'none' }
        : ppSendPath(input.parallel, topology, stage.index, stage.index + 1, base.interStaging);
    const ctx: OpTimeContext = {
      engine: base.engine,
      memBandwidthBytesPerSec: base.memBandwidthBytesPerSec,
      collectivePaths: {
        tp: base.tpPath,
        ep: base.epPath,
        pp: ppPath,
        ...etpPath,
        ...dcpPath,
        ...lookupPaths,
      },
      collectiveAlgorithm: base.collectiveAlgorithm,
      elementwiseFusion: base.elementwiseFusion,
      warnings: base.warnings,
      ...(base.perOpLatencyFloorSec === undefined
        ? {}
        : { perOpLatencyFloorSec: base.perOpLatencyFloorSec }),
      ...(base.allToAllEfficiency === undefined
        ? {}
        : { allToAllEfficiency: base.allToAllEfficiency }),
    };
    if (offload !== undefined) {
      ctx.transferPaths = stageTransferPaths(
        lowered,
        stage.index,
        offload,
        input,
        topology,
        shape.unitsInFlight ?? shape.microbatches,
      );
    }
    const layers = stage.layers.map((l) =>
      layerResult(
        l.name,
        l.layerIndex,
        l.ops.map((op) => opTime(op, ctx)),
      ),
    );
    const send = stage.sendToNext === undefined ? undefined : opTime(stage.sendToNext, ctx);
    if (stage.sendToFirst !== undefined) {
      const returnPath = ppSendPath(input.parallel, topology, stage.index, 0, base.interStaging);
      loopBack = opTime(stage.sendToFirst, {
        ...ctx,
        collectivePaths: { ...ctx.collectivePaths, pp: returnPath },
      });
    }
    const original = stageResult(stage.index, layers, send, overlap);
    if (!(overlap.tp > 0) || options.collectiveAlgorithm !== 'best') return original;
    // Formula (G7): choose min_assignment t_core,s (overlap.ts: coreTime; every active f = 1: Σα + max(compute_s, Σβ)).
    // Bounded heuristic: compare per-op min-total, uniform ring, uniform tree; ties keep the original.
    // See docs/simulation-model.md §7 "Stage time and overlap". Explicit algorithms are never overridden.
    const scoreOf = (st: StageResult): number =>
      coreTime(
        stageCoreTimeInput(
          st.layers.flatMap((l) => l.ops),
          send,
        ),
        overlap,
      ).coreSec;
    let best = original;
    let bestCoreSec = scoreOf(best);
    for (const collectiveAlgorithm of ['ring', 'tree'] as const) {
      const candidateLayers = layers.map((layer) =>
        layerResult(
          layer.name,
          layer.layerIndex,
          layer.ops.map((r) =>
            r.op.kind === 'collective' && r.op.collective === 'all_reduce'
              ? opTime(r.op, { ...ctx, collectiveAlgorithm })
              : r,
          ),
        ),
      );
      const candidate = stageResult(stage.index, candidateLayers, send, overlap);
      const coreSec = scoreOf(candidate);
      if (coreSec < bestCoreSec) {
        best = candidate;
        bestCoreSec = coreSec;
      }
    }
    return best;
  });
  const phaseShape: PhaseShape = {
    phase,
    dp: input.parallel.dp,
    sequencesPerReplica: shape.sequencesPerReplica,
    microbatches: shape.microbatches,
    microbatchSize: shape.microbatchSize,
    promptLen: input.workload.promptLen,
    computedTokens: computedPromptTokens(input.workload),
    overlap,
  };
  if (tokensPerStep !== undefined) phaseShape.tokensPerStep = tokensPerStep;
  if (loopBack !== undefined) phaseShape.loopBack = loopBack;
  if (dispatchSecPerKernel === undefined) return phaseResult(stages, phaseShape);
  return phaseResult(
    stages.map((st) => withDispatch(st, dispatchSecPerKernel)),
    phaseShape,
  );
}

/**
 * Adds the drafter to a lowered phase (speculative decoding, design doc §3.3–3.6): appends its `draft.*` layers to the
 * last stage, after the target's head, so the stage sums, overlap and collectives apply to it unchanged; and at
 * pp > 1 widens each stage's activation send by the block drafter's target features that boundary relays
 * (`draft.ts: ppBoundaryDraftFeatureVectors`). `ctx` is the phase's lowering context (the one `lowered` came from).
 *   S_s = T·(N_live + F_s)·h·bpe(act),  T = stepTokens(ctx),  F_s = F(draft, end_s)
 * A stage with F_s = 0 (MTP, every pp = 1 lowering: no send at all) keeps its send object untouched.
 * See docs/simulation-model.md §13 "Pipeline parallelism".
 */
function withDrafter(
  lowered: LoweredModel,
  model: ModelSpec,
  draft: DraftSpec,
  ctx: LoweringContext,
  draftLayers: LoweredLayer[],
): LoweredModel {
  const last = lowered.stages.length - 1;
  const stages = lowered.stages.map((stage, s) => {
    const layers = s === last ? [...stage.layers, ...draftLayers] : stage.layers;
    const send = stage.sendToNext;
    if (send === undefined) return s === last ? { ...stage, layers } : stage;
    const end = stageLayerRange(model.layers.length, lowered.stages.length, s)[1];
    const F = ppBoundaryDraftFeatureVectors(draft, end);
    if (F === 0) return stage;
    // Formula: S_s = T·(N_live + F_s)·h·bpe(act)  (the target's T·N_live·h·bpe plus T·F_s·h·bpe of features)
    // See docs/simulation-model.md §13 "Pipeline parallelism"
    const featureBytes = stepTokens(ctx) * F * model.hidden * actBpe(ctx);
    return { ...stage, sendToNext: { ...send, messageBytes: send.messageBytes + featureBytes } };
  });
  return { stages };
}

/**
 * Speculative-decoding summary of a decode phase (design doc §3.5, §4.2): `draftSec` = Σ op times of the `draft.*`
 * layers, `verifySec` = Σ of every other layer (the verify pass), both without overlap (seconds); E[τ]; the same
 * input's spec-off rate and cycle (`off`); the step cost ratio `cycle / offCycle` (dimensionless); and the speedup
 * over spec-dec off (tokens/s ratio, dimensionless), which equals E[τ] / stepCostRatio.
 */
function speculativeSummary(
  decode: PhaseResult,
  expectedTokensPerStep: number,
  off: PhaseResult,
): NonNullable<PhaseResult['speculative']> {
  let draftSec = 0;
  let verifySec = 0;
  for (const stage of decode.stages) {
    for (const layer of stage.layers) {
      if (layer.name.startsWith('draft.')) draftSec += layer.sec;
      else verifySec += layer.sec;
    }
  }
  // Formula: stepCostRatio = cycle / cycle_off;  speedup = tokens/s(spec-dec) / tokens/s(off) = E[τ] / stepCostRatio
  // See docs/simulation-model.md §13 ("Accepted tokens", "Break-even batch")
  return {
    draftSec,
    verifySec,
    expectedTokensPerStep,
    offTokensPerSec: off.tokensPerSec,
    offCycleSec: off.cycleSec,
    stepCostRatio: decode.cycleSec / off.cycleSec,
    speedup: decode.tokensPerSec / off.tokensPerSec,
  };
}

/** The rates of a phase result, for a what-if figure (tokens/s, seconds). */
function phaseRates(p: PhaseResult): PhaseRates {
  return {
    tokensPerSec: p.tokensPerSec,
    perSequenceTokensPerSec: p.perSequenceTokensPerSec,
    latencySec: p.latencySec,
    cycleSec: p.cycleSec,
  };
}

/**
 * The lowering of a chunk before the last under chunked pipeline prefill: the same, without the last stage's `head`
 * layer (prefill computes logits only for each sequence's last position, which the last chunk holds;
 * `lower/head.ts: headTokens`). Draft layers are added after this (`withDrafter`).
 */
function withoutHead(lowered: LoweredModel): LoweredModel {
  const last = lowered.stages.length - 1;
  return {
    stages: lowered.stages.map((stage, s) =>
      s === last ? { ...stage, layers: stage.layers.filter((l) => l.name !== 'head') } : stage,
    ),
  };
}

/**
 * The prefill {@link PhaseShape} of a run (as `simulatePhase` builds it), for the chunked schedule
 * (`chunkedPipeline.ts: chunkedPipelinePhase`) under overlap fractions `overlap`.
 */
function prefillPhaseShape(
  input: SimInput,
  shape: { sequencesPerReplica: number; microbatches: number; microbatchSize: number },
  overlap: OverlapFractions,
): PhaseShape {
  return {
    phase: 'prefill',
    dp: input.parallel.dp,
    sequencesPerReplica: shape.sequencesPerReplica,
    microbatches: shape.microbatches,
    microbatchSize: shape.microbatchSize,
    promptLen: input.workload.promptLen,
    computedTokens: computedPromptTokens(input.workload),
    overlap,
  };
}

/** `PhaseResult.chunkedPipeline` of a chunked prefill: C, K and the whole-prompt prefill's rates. */
function chunkedPipelineSummary(
  plan: ChunkedPipelinePlan,
  wholePrompt: PhaseResult,
): NonNullable<PhaseResult['chunkedPipeline']> {
  return {
    chunkTokens: plan.chunkTokens,
    chunks: plan.chunks.length,
    unchunked: phaseRates(wholePrompt),
  };
}

/**
 * Per-call options of {@link simulate} that are not part of the input (no URL, no validation): how much of the result
 * a caller needs.
 */
export interface SimulateRunOptions {
  /**
   * With chunked pipeline prefill in effect (`ParallelConfig.chunkedPipelinePrefill`, pp > 1, K > 1): skip the K chunk
   * lowerings and the chunked schedule, for a caller that reads only decode and the memory fit (the UI's batch
   * sweep). `memory` and `decode` are then bit-identical to the full run (the fit still counts the pipeline chunk,
   * §8 "Activation peak"); `prefill` and `prefillWithoutOverlap` are the whole-prompt ones, without
   * `prefill.chunkedPipeline`, and `warnings` may lack a line only the chunk costing raises. Absent or false = the
   * full run. No effect without chunked pipeline prefill.
   */
  skipChunkedPrefillSchedule?: boolean;
}

/**
 * Simulate one input: validate, apply modifiers, cost prefill (the whole prompt from an empty KV cache) and decode
 * (one new token with `decodeContextLen` tokens already cached; with speculative decoding a `k + 1`-token verify plus
 * the drafter), and check the memory fit. Throws `Error(errors.join('\n'))` on an invalid input. `run`: see
 * {@link SimulateRunOptions}.
 */
export function simulate(input: SimInput, run: SimulateRunOptions = {}): SimResult {
  const errors = validateInput(input);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const options: SimOptions = { ...DEFAULT_SIM_OPTIONS, ...input.options };
  const { parallel, workload, model, quant } = input;
  const sharedLookupTables =
    lookupTableConfig(input.deployment?.offload).acrossDp === true && parallel.dp > 1;

  // 2. Modifiers, engine, memory bandwidth, collective paths.
  const hw = effectiveHardware(input.hardware);
  const device = hw.device;
  const engine = device.engines.find(
    (e) => e.id === (input.hardware.engineId ?? device.engines[0].id),
  );
  if (engine === undefined) throw new Error(`engine '${input.hardware.engineId}' not found`);
  const warnings: string[] = [
    ...hw.notes,
    ...checkPlacement(parallel, hw.topology).warnings,
    ...crossStageWarnings(model, parallel.pp),
  ];
  // Inter-node host staging (docs §6 "Inter-node host staging"): a dedicated-memory device without a GPUDirect path
  // over the inter-node link stages every inter-node message through host memory. Absent ⇒ the paths of before.
  const hostStaging = resolveHostStaging(device, hw.topology, input.hardware, parallel);
  const interStaging = hostStaging.staging;
  if (
    interStaging !== undefined &&
    hasInterNodeTraffic(parallel, hw.topology, sharedLookupTables)
  ) {
    warnings.push(...hostStaging.warnings);
  }
  const algorithm: AllReduceAlgorithm = options.collectiveAlgorithm;
  const base = {
    engine,
    memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(
      engine,
      device.memory.bandwidthBytesPerSec,
    ),
    collectiveAlgorithm: algorithm,
    elementwiseFusion: options.elementwiseFusion,
    warnings,
    // In-network reduction (docs §6 "In-network reduction (NVLS, SHARP)"): 'off' builds exactly the paths of before.
    tpPath: collectivePath(
      groupSpan(parallel.tp, hw.topology),
      hw.topology,
      interStaging,
      options.inNetworkReduction,
    ),
    epPath: collectivePath(
      groupSpan(parallel.ep, hw.topology),
      hw.topology,
      interStaging,
      options.inNetworkReduction,
    ),
    ...(interStaging === undefined ? {} : { interStaging }),
    // The per-op latency floor only when set: with 0 the op contexts are exactly as before.
    ...(options.perOpLatencyFloorSec > 0
      ? { perOpLatencyFloorSec: options.perOpLatencyFloorSec }
      : {}),
    // The all-to-all efficiency only when below 1: absent or 1 leaves the op contexts exactly as before.
    ...(parallel.allToAllEfficiency !== undefined && parallel.allToAllEfficiency !== 1
      ? { allToAllEfficiency: parallel.allToAllEfficiency }
      : {}),
  };

  // 3. Offload placement and host (docs §12). Nothing off-device ⇒ offload undefined, hardware.host never read.
  const resolved = resolveOffload(
    model,
    input.deployment?.offload,
    device.memory.kind,
    parallel.ep,
    workload.prefixCacheHitRatio ?? 0,
  );
  warnings.push(...resolved.warnings);
  let phaseOffload: PhaseOffload | undefined;
  if (resolved.offload !== undefined) {
    const host = resolveHost(input.hardware, device, offloadTiers(resolved.offload));
    if (
      input.deployment?.offload?.ssdReadRate === 'sustained' &&
      host.ssd !== undefined &&
      host.ssd.sustainedRead === undefined
    ) {
      warnings.push(
        `offload: no sustained-read data for ${host.ssd.id}; using datasheet SSD read rate`,
      );
    }
    // The remote KV pool (docs §12 "Remote KV pool"): resolved only when a KV share is placed in it, so a placement
    // without one never reads the topology's inter-node link.
    const kvConfig = input.deployment?.offload?.kvCache;
    if (
      (resolved.offload.kvCache?.remotePoolShare ?? 0) > 0 &&
      kvConfig?.remotePool !== undefined
    ) {
      host.remotePool = resolveRemotePool(kvConfig.remotePool, hw.topology, host.warnings);
    }
    warnings.push(...host.warnings);
    phaseOffload = {
      host,
      queueDepth: input.deployment?.offload?.queueDepth,
      ssdReadRate: input.deployment?.offload?.ssdReadRate,
      prefetching: resolved.offload.prefetchAccuracy > 0,
    };
  }
  // 4. Speculative decoding (docs §13). Absent ⇒ plain decode, no draft layers.
  const spec = resolveSpeculative(model, input.deployment?.speculative);
  if (spec !== undefined) warnings.push(...spec.warnings);
  // 5. Phases. Formula: B_r = batch/dp; m = min(pp, B_r); b_mb = ceil(B_r/m)  (docs §7 "Pipeline and DP")
  const sequencesPerReplica = workload.batch / parallel.dp;
  const mb = microbatching(parallel.pp, sequencesPerReplica);
  const shape = { sequencesPerReplica, ...mb };
  // Prefill @ P: P_new = P − floor(P·h) new tokens (h = prefix-cache hit ratio, 0 = empty cache) attend over all P
  // keys (cached past + new), so qLen = P_new, kvLen = P. At h = 0 this is qLen = kvLen = P (docs §7 "Prefix cache").
  const prefillCtx: LoweringContext = {
    phase: 'prefill',
    seqs: mb.microbatchSize,
    qLen: computedPromptTokens(workload),
    kvLen: workload.promptLen,
    quant,
    parallel,
  };
  if (resolved.offload !== undefined) prefillCtx.offload = resolved.offload;
  if (sharedLookupTables) {
    const peerReads = lookupTablesUsePeerReads(parallel, hw.topology);
    prefillCtx.lookupTables = { peerReads };
    if (
      peerReads &&
      model.layers.some((l) =>
        l.blocks.some((b) => b.kind === 'ngram_embedding' && b.headRows !== undefined),
      )
    ) {
      warnings.push(
        'lookup tables: peer-HBM traffic is a bandwidth ceiling; remote-load latency, transaction headers and routing skew are not modeled',
      );
    }
  }
  // Decode @ N: N tokens are already in the KV cache and one new token is generated, so the new token's query
  // sees its own key too.  Formula: qLen = 1; kvLen = past + qLen = N + 1  (docs §7 "Pipeline and DP").
  const decodePastTokens = workload.decodeContextLen;
  const plainDecodeCtx: LoweringContext = {
    ...prefillCtx,
    phase: 'decode',
    qLen: 1,
    kvLen: decodePastTokens + 1,
  };
  // Verify @ N (spec-dec): the k + 1 positions are the bonus token and k drafts.
  // Formula: qLen = k + 1; kvLen = N + k + 1  (docs §13 "Verify")
  const decodeCtx: LoweringContext =
    spec === undefined
      ? plainDecodeCtx
      : {
          ...plainDecodeCtx,
          qLen: spec.lowering.verifyTokens,
          kvLen: decodePastTokens + spec.lowering.verifyTokens,
          speculative: spec.lowering,
        };
  // Two-batch overlap (docs §7 "Two-batch overlap"): each step is lowered as two halves, merged per layer
  // (twoBatch.ts). Off: the full-batch lowering, exactly as before.
  const twoBatch = input.deployment?.twoBatchOverlap === true;
  const lowerStep = (ctx: LoweringContext): LoweredModel => {
    const full = lowerModel(model, ctx);
    if (!twoBatch) return full;
    const split = twoBatchHalves(ctx);
    if (split === undefined) {
      warnings.push(TWO_BATCH_NOTHING_TO_SPLIT_WARNING);
      return full;
    }
    return mergeTwoBatch(
      full,
      lowerModel(model, split.a),
      lowerModel(model, split.b),
      split.tokenSplit,
    );
  };
  // Overlap fractions per class of one phase step (docs §7 "Stage time and overlap", "Two-batch overlap"):
  // f_c = overlapByClass[c] ?? (c = ep ∧ the step is split in two ? 1 : the global default). A step that two-batch
  // overlap could not split (one-sequence decode) keeps the global EP default: there is no other half to hide behind.
  const overlapFor = (ctx: LoweringContext): OverlapFractions =>
    resolveOverlap(options, twoBatch && twoBatchHalves(ctx) !== undefined);
  const prefillOverlap = overlapFor(prefillCtx);
  const decodeOverlap = overlapFor(decodeCtx);
  let prefillLowered = lowerStep(prefillCtx);
  let decodeLowered = lowerStep(decodeCtx);
  if (spec !== undefined) {
    const { draft, contextRows, draftLength } = spec.lowering;
    prefillLowered = withDrafter(
      prefillLowered,
      model,
      draft,
      prefillCtx,
      lowerDraftPrefill(model, draft, prefillCtx),
    );
    decodeLowered = withDrafter(
      decodeLowered,
      model,
      draft,
      decodeCtx,
      lowerDraftDecode(model, draft, decodeCtx, contextRows, draftLength),
    );
  }
  const wholePromptPrefill = simulatePhase(
    prefillLowered,
    'prefill',
    base,
    input,
    hw.topology,
    options,
    prefillOverlap,
    shape,
    phaseOffload,
  );
  // 4a. Chunked pipeline prefill (docs §7 "Chunked pipeline prefill"). Only with the flag on, pp > 1 and K > 1 chunks;
  // otherwise `prefill` is the whole-prompt result above, untouched. Chunk j is lowered as its own step
  // { qLen: q_j, kvLen: kv_j } (the prefix-cache shape: its tokens over every earlier key), with two-batch overlap
  // and the drafter as for a whole prompt; chunks before the last have no LM head (no token is sampled yet).
  if (parallel.chunkedPipelinePrefill === true && parallel.pp === 1) {
    warnings.push(CHUNKED_PIPELINE_SINGLE_STAGE_WARNING);
  }
  const cpp = chunkedPipelinePlan(parallel, workload);
  // The chunks costed and scheduled: none when the caller skips the schedule (the memory fit below still uses `cpp`).
  const schedule = run.skipChunkedPrefillSchedule === true ? undefined : cpp;
  // Each chunk's past = the prefix-cache hit (`prefixCacheTokens`, placed by f_pre) + the earlier chunks (live KV,
  // placed by f_dev): with KV-cache offload every chunk fetches the off-device part of both (docs §7 "Chunked
  // pipeline prefill").
  const prefixCacheTokens = cachedPrefixTokens(workload);
  const chunkContexts: LoweringContext[] =
    schedule?.chunks.map((chunk) => ({
      ...prefillCtx,
      qLen: chunk.qLen,
      kvLen: chunk.kvLen,
      prefixCacheTokens,
    })) ?? [];
  // One chunk's lowering: the target (without the head before the last chunk), then the drafter.
  const lowerPrefillChunk = (
    ctx: LoweringContext,
    isLast: boolean,
    lower: (ctx: LoweringContext) => LoweredModel,
  ): LoweredModel => {
    const target = lower(ctx);
    const lowered = isLast ? target : withoutHead(target);
    if (spec === undefined) return lowered;
    const { draft } = spec.lowering;
    return withDrafter(lowered, model, draft, ctx, lowerDraftPrefill(model, draft, ctx));
  };
  const lowerPrefillChunks = (lower: (ctx: LoweringContext) => LoweredModel): LoweredModel[] =>
    chunkContexts.map((ctx, j) => lowerPrefillChunk(ctx, j === chunkContexts.length - 1, lower));
  // Each chunk costed as one microbatch step, under the given options and overlap fractions, then scheduled. The
  // offload streams are shared as with m·K units in flight: the K chunks of each microbatch occupy stages at the same
  // time, so up to min(m·K, |S_tier|) stages of a node stream together (docs §7 "Chunked pipeline prefill").
  const chunkShape = { ...shape, unitsInFlight: shape.microbatches * chunkContexts.length };
  const chunkedPrefill = (
    chunkLowerings: readonly LoweredModel[],
    chunkOptions: SimOptions,
    overlapOf: (ctx: LoweringContext) => OverlapFractions,
    scheduleOverlap: OverlapFractions,
  ): PhaseResult =>
    chunkedPipelinePhase(
      chunkLowerings.map(
        (lowered, j) =>
          simulatePhase(
            lowered,
            'prefill',
            base,
            input,
            hw.topology,
            chunkOptions,
            overlapOf(chunkContexts[j]),
            chunkShape,
            phaseOffload,
          ).stages,
      ),
      prefillPhaseShape(input, shape, scheduleOverlap),
    );
  const prefillChunkLowerings = schedule === undefined ? [] : lowerPrefillChunks(lowerStep);
  const prefill: PhaseResult =
    schedule === undefined
      ? wholePromptPrefill
      : {
          ...chunkedPrefill(prefillChunkLowerings, options, overlapFor, prefillOverlap),
          chunkedPipeline: chunkedPipelineSummary(schedule, wholePromptPrefill),
        };
  const decode = simulatePhase(
    decodeLowered,
    'decode',
    base,
    input,
    hw.topology,
    options,
    decodeOverlap,
    shape,
    phaseOffload,
    spec?.expectedTokensPerStep,
  );
  if (spec !== undefined) {
    // The same decode with speculative decoding off, for the speedup.
    const off = simulatePhase(
      lowerStep(plainDecodeCtx),
      'decode',
      base,
      input,
      hw.topology,
      options,
      overlapFor(plainDecodeCtx),
      shape,
      phaseOffload,
    );
    decode.speculative = speculativeSummary(decode, spec.expectedTokensPerStep, off);
  }
  // 5b. Labelled what-if figures next to the headline; neither changes `prefill` or `decode`.
  // Prefill without overlap (review F7; docs §7 "Prefill without overlap"): the same prefill with every overlap
  // fraction 0 and two-batch overlap off (the full-batch lowering), when some fraction is above 0 and the prefill has
  // inter-device comm. With no class overrides and TBO off: overlap 'ideal' vs 'none', exactly as before.
  let prefillWithoutOverlap: PhaseRates | undefined;
  if (
    anyOverlap(prefillOverlap) &&
    (prefill.comm.latencySec > 0 || prefill.comm.bandwidthSec > 0)
  ) {
    if (schedule !== undefined) {
      // Chunked pipeline prefill: the same chunks under every fraction 0, without two-batch overlap (the full-batch
      // chunk lowerings; without TBO they are the headline's, reused).
      const noneLowerings = twoBatch
        ? lowerPrefillChunks((ctx) => lowerModel(model, ctx))
        : prefillChunkLowerings;
      const none = chunkedPrefill(
        noneLowerings,
        { ...options, overlap: 'none' },
        () => NO_OVERLAP,
        NO_OVERLAP,
      );
      prefillWithoutOverlap = phaseRates(none);
    } else {
      let fullPrefillLowered = prefillLowered;
      if (twoBatch) {
        fullPrefillLowered = lowerModel(model, prefillCtx);
        if (spec !== undefined)
          fullPrefillLowered = withDrafter(
            fullPrefillLowered,
            model,
            spec.lowering.draft,
            prefillCtx,
            lowerDraftPrefill(model, spec.lowering.draft, prefillCtx),
          );
      }
      const none = simulatePhase(
        fullPrefillLowered,
        'prefill',
        base,
        input,
        hw.topology,
        { ...options, overlap: 'none' },
        NO_OVERLAP,
        shape,
        phaseOffload,
      );
      prefillWithoutOverlap = phaseRates(none);
    }
  }
  // Engine dispatch overlay (review F1b; docs §7 "Engine dispatch overlay"): t_s' = t_s + n_s·τ per decode stage.
  let decodeWithDispatch: SimResult['decodeWithDispatch'];
  const dispatch = input.overlays?.engineDispatch;
  if (dispatch !== undefined) {
    const perKernelSec = dispatch.perKernelSec ?? DEFAULT_DISPATCH_SEC_PER_KERNEL;
    const withEngine = simulatePhase(
      decodeLowered,
      'decode',
      base,
      input,
      hw.topology,
      options,
      decodeOverlap,
      shape,
      phaseOffload,
      spec?.expectedTokensPerStep,
      perKernelSec,
    );
    decodeWithDispatch = {
      ...phaseRates(withEngine),
      perKernelSec,
      kernelsPerStep: decode.stages.reduce((n, st) => n + stageKernelCount(st), 0),
    };
  }

  // 6. Memory fit. The activation peak counts the decode step and the LAST prefill chunk (chunked prefill, docs §8
  // "Activation peak"); the prefill rate above is unaffected. A microbatch within the budget reuses the prefill lowering.
  // With chunked pipeline prefill a pipeline chunk holds b_mb sequences of C new tokens each, so its largest step is
  // { seqs: b_mb, qLen: C, kvLen: P } (the step of the budget rule when b_mb = 1).
  const chunk =
    cpp === undefined
      ? prefillChunkShape(
          workload.promptLen,
          prefillCtx.seqs,
          workload.prefillChunkTokens ?? DEFAULT_PREFILL_CHUNK_TOKENS,
          prefillCtx.qLen,
        )
      : {
          seqs: prefillCtx.seqs,
          qLen: cpp.chunkTokens,
          kvLen: workload.promptLen,
          prefixCacheTokens: cachedPrefixTokens(workload),
        };
  let chunkLowered = prefillLowered;
  if (chunk.seqs !== prefillCtx.seqs || chunk.qLen !== prefillCtx.qLen) {
    const chunkCtx: LoweringContext = { ...prefillCtx, ...chunk };
    chunkLowered = lowerStep(chunkCtx);
    if (spec !== undefined)
      chunkLowered = withDrafter(
        chunkLowered,
        model,
        spec.lowering.draft,
        chunkCtx,
        lowerDraftPrefill(model, spec.lowering.draft, chunkCtx),
      );
  }
  // Chunked pipeline prefill: the last chunk is one more phase of the fit, for its staging buffers. It fetches the most
  // off-device KV of any chunk (the whole prefix and every earlier chunk at (1 − f_dev); docs §7 "Chunked pipeline
  // prefill", §12 "KV cache off-device"); its activations and weight fetches are within the steps above. It is the
  // schedule's own lowering, or (schedule skipped) the same lowering made here, so the fit is the same either way.
  const lastChunkPhase: LoweredModel[] = [];
  if (cpp !== undefined) {
    const last = cpp.chunks[cpp.chunks.length - 1];
    lastChunkPhase.push(
      prefillChunkLowerings.at(-1) ??
        lowerPrefillChunk(
          { ...prefillCtx, qLen: last.qLen, kvLen: last.kvLen, prefixCacheTokens },
          true,
          lowerStep,
        ),
    );
  }
  const capacityBytes = input.hardware.memCapacityBytes ?? device.memory.capacityOptionsBytes[0];
  const gpuUsableFraction = resolveGpuUsableFraction(
    device.memory,
    input.hardware.gpuUsableFraction,
  );
  const memory = memoryReport({
    model,
    ctx: decodeCtx,
    sequencesPerReplica,
    // The cache must hold the prompt after prefill and N + 1 tokens after the decode step (docs §8); with
    // speculative decoding the verify appends k + 1 rows, so N + k + 1 (docs §13 "Draft memory").
    contextTokens: Math.max(workload.promptLen, decodeCtx.kvLen),
    lowered: [prefillLowered, decodeLowered, ...lastChunkPhase],
    activationLowered: [chunkLowered, decodeLowered, ...lastChunkPhase],
    capacityBytes,
    kind: device.memory.kind,
    gpuUsableFraction,
    runtimeReserveBytes: input.hardware.runtimeReserveBytes ?? 0,
    devicesPerNode: hw.topology.devicesPerNode,
    ...(phaseOffload === undefined ? {} : { host: phaseOffload.host }),
  });
  if (device.memory.kind === 'unified') warnings.push(...unifiedMemoryWarnings(memory));

  return {
    prefill,
    decode,
    memory,
    devicesUsed: parallel.tp * parallel.pp * parallel.dp,
    effectiveDevice: device,
    warnings: [...new Set(warnings)],
    ...(prefillWithoutOverlap === undefined ? {} : { prefillWithoutOverlap }),
    ...(decodeWithDispatch === undefined ? {} : { decodeWithDispatch }),
  };
}
