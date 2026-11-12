/**
 * The loaded model and the state → `SimInput` step. The simulation runs only when this returns an input;
 * otherwise the UI shows the issues by field.
 */

import type { SimInput } from '../../sim/engine/types';
import { validateInput } from '../../sim/engine/validate';
import { hasOffDevicePlacement } from '../../sim/engine/offloadRules';
import type { ModelSpec } from '../../sim/ir/model';
import { REF_DENSE_8B } from '../../sim/models/reference';
import type { HfMapResult, QuantDefaults } from '../../sim/models/hf/types';
import { disaggregationErrors, type DisaggregatedInput } from '../../sim/engine/disaggregation';
import { DEFAULT_OUTPUT_LEN, poolView, type AppState, type ModelChoice } from './appState';
import { kvTransferConfig } from './kvLink';
import { resolveHardware, type ResolvedHardware } from './hardware';
import { effectiveParallel, fieldIssues, hasErrors, type FieldIssue } from './parallelism';
import { REFERENCE_QUANT_DEFAULTS, applyQuantOverride } from './quant';
import { deploymentConfig, type DrafterStatus } from './deploymentInput';
import { effectiveOptions, hasLookupCollectives, twoBatchOverlapApplies } from './overlap';

/**
 * The panel an engine validation message belongs to, by its wording (speculative decoding, two-batch overlap,
 * offloading, per-class overlap, else general).
 */
function engineIssueField(message: string): FieldIssue['field'] {
  if (/speculative|draft/i.test(message)) return 'specDec';
  if (/two-batch|twoBatch/i.test(message)) return 'twoBatchOverlap';
  if (
    /offload|host link|host_ram|nvme|queue depth|resident|hit ratio|lookup|cpu_compute|kv cache/i.test(
      message,
    )
  )
    return 'offload';
  if (/overlap/i.test(message)) return 'overlap';
  return 'general';
}

/** A successful `mapHfModel` result. */
export type HfMapOk = Extract<HfMapResult, { kind: 'ok' }>;

/** A model ready to simulate: the built-in reference or a mapped Hugging Face repo. */
export interface LoadedModel {
  /** Identity of the choice it was loaded for ({@link modelKey}); a result for another key is stale. */
  key: string;
  /** Display name: the repo id, or the reference model's name. */
  label: string;
  /** The model IR. */
  model: ModelSpec;
  /** Default formats (before the user's overrides) and the scheme label. */
  quant: QuantDefaults;
  /** Loader notes (unwrap, family, quant, param check); empty for the reference. */
  notes: string[];
  /** The full mapping result for an HF model (family, modelType, architecture, excluded parts, param check). */
  hf?: HfMapOk;
}

/** Identity of a model choice: `'hf:<id>@<revision>'` (revision default `main`). */
export function modelKey(choice: ModelChoice): string {
  return `hf:${choice.id}@${choice.revision ?? 'main'}`;
}

/** The built-in reference model (REF_DENSE_8B, bf16), available without network. */
export const REFERENCE_MODEL: LoadedModel = {
  key: 'reference',
  label: REF_DENSE_8B.name,
  model: REF_DENSE_8B,
  quant: REFERENCE_QUANT_DEFAULTS,
  notes: [],
};

/** Wraps a successful mapping as a {@link LoadedModel} for `choice`. */
export function loadedFromHf(
  choice: Extract<ModelChoice, { kind: 'hf' }>,
  result: HfMapOk,
): LoadedModel {
  return {
    key: modelKey(choice),
    label: choice.id,
    model: result.model,
    quant: result.quant,
    notes: result.notes,
    hf: result,
  };
}

/** Result of {@link buildSimInput}. */
export type BuildResult =
  | {
      kind: 'ok';
      /** The engine input. */
      input: SimInput;
      /** The resolved hardware (for display: rig sentence, spec editor, "edited" marks). */
      hardware: ResolvedHardware;
      /** Warnings by field (e.g. unused devices); never errors. */
      issues: FieldIssue[];
      /** Unlock modifier descriptions and loader notes, for the Notes panel. */
      notes: string[];
    }
  | {
      kind: 'issues';
      /** At least one error; the simulation does not run. */
      issues: FieldIssue[];
      /** The resolved hardware, when it resolved (always, for states the decoder or reducer made). */
      hardware?: ResolvedHardware;
    };

/**
 * State + loaded model (+ the fetch state of a block drafter's config, when spec-dec needs one) → engine input, or
 * the issues that stop it. `loaded` must be the model of `state.model` (same {@link modelKey}); otherwise (not loaded
 * yet, or stale) the result is a general issue. Steps: the deployment config (`deploymentConfig`); resolve the
 * hardware (with `host` only while something is offloaded); apply the quant overrides; `fieldIssues` and the
 * deployment issues (errors stop here); then the engine's `validateInput`, whose remaining errors go to the
 * spec-dec / offload field by their wording, else general. `deployment` is in the input only when a knob is on.
 */
export function buildSimInput(
  state: AppState,
  loaded: LoadedModel | null,
  drafter?: DrafterStatus,
): BuildResult {
  if (loaded === null || loaded.key !== modelKey(state.model)) {
    return {
      kind: 'issues',
      issues: [{ field: 'general', level: 'error', message: 'The model is not loaded yet.' }],
    };
  }
  const deployment = deploymentConfig(state, loaded.model, drafter);
  const offloads = hasOffDevicePlacement(
    deployment.config?.offload,
    state.workload.prefixCacheHitRatio,
  );
  let hardware: ResolvedHardware;
  try {
    hardware = resolveHardware(state.hw, offloads);
  } catch (e) {
    return {
      kind: 'issues',
      issues: [{ field: 'general', level: 'error', message: (e as Error).message }],
    };
  }
  const quant = applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3);
  const parallel = effectiveParallel(state.parallel, loaded.model, state.parallelLayout);
  const issues = [
    ...fieldIssues(
      loaded.model,
      quant,
      parallel,
      state.workload,
      hardware.topology,
      state.parallelLayout,
    ),
    ...deployment.issues,
  ];
  // Two-batch overlap only where it applies (MoE mode B, nothing offloaded); with offload it is not modelled.
  const twoBatchOverlap =
    state.deployment.twoBatchOverlap === true &&
    twoBatchOverlapApplies(loaded.model, parallel, offloads);
  if (
    state.deployment.twoBatchOverlap === true &&
    offloads &&
    twoBatchOverlapApplies(loaded.model, parallel, false)
  )
    issues.push({
      field: 'twoBatchOverlap',
      level: 'warning',
      message: 'Two-batch overlap is not modelled with offloaded weights; it is off for this run.',
    });
  const deploymentInput = twoBatchOverlap
    ? { ...deployment.config, twoBatchOverlap: true }
    : deployment.config;
  // Per-class overlap only for the classes the run has: a stale choice (a hidden row) changes nothing.
  const lookupCollectives = hasLookupCollectives(
    loaded.model,
    parallel,
    state.deployment.offload,
    hardware.topology,
  );
  const options = effectiveOptions(state.options, loaded.model, parallel, lookupCollectives);
  if (hasErrors(issues)) return { kind: 'issues', issues, hardware };
  const input: SimInput = {
    model: loaded.model,
    quant,
    // The user's degrees; EP is the user's on a MoE model, else 1. `denseTpGroup` only when dense TP over the EP
    // group is on (MoE, DP attention + EP), so without it the input is exactly the one from before the layout. The
    // EP balance knobs (ρ, EPLB replicas) only when the user set a non-default value on a MoE model. `dcp` likewise
    // only when DCP is on (> 1): with DCP off the input is exactly the one from before DCP. `chunkedPipelinePrefill`
    // only when on and PP > 1 (the engine ignores it at PP 1, with a warning the UI need not show).
    parallel: {
      tp: parallel.tp,
      pp: parallel.pp,
      dp: parallel.dp,
      ep: parallel.ep,
      ...(parallel.denseTpGroup !== undefined ? { denseTpGroup: parallel.denseTpGroup } : {}),
      ...(parallel.epImbalance !== undefined ? { epImbalance: parallel.epImbalance } : {}),
      ...(parallel.epImbalanceScalesAllToAll === true ? { epImbalanceScalesAllToAll: true } : {}),
      ...(parallel.redundantExpertsPerRank !== undefined
        ? { redundantExpertsPerRank: parallel.redundantExpertsPerRank }
        : {}),
      ...(parallel.dcp !== undefined && parallel.dcp > 1 ? { dcp: parallel.dcp } : {}),
      ...(parallel.chunkedPipelinePrefill === true && parallel.pp > 1
        ? { chunkedPipelinePrefill: true }
        : {}),
      ...(parallel.dispatchFormat !== undefined ? { dispatchFormat: parallel.dispatchFormat } : {}),
      ...(parallel.combineFormat !== undefined ? { combineFormat: parallel.combineFormat } : {}),
      ...(parallel.allToAllEfficiency !== undefined
        ? { allToAllEfficiency: parallel.allToAllEfficiency }
        : {}),
    },
    workload: state.workload,
    hardware: hardware.stack,
    ...(Object.keys(options).length > 0 ? { options } : {}),
    // Only when a knob is on: with every knob off the input is exactly the one from before deployment knobs.
    ...(deploymentInput !== undefined ? { deployment: deploymentInput } : {}),
    // The dispatch overlay adds a labelled figure only; off, the input has no `overlays` at all.
    ...(state.engineDispatchOverlay ? { overlays: { engineDispatch: {} } } : {}),
  };
  const engineErrors = validateInput(input);
  if (engineErrors.length > 0) {
    const general = engineErrors.map((message): FieldIssue => ({
      field: engineIssueField(message),
      level: 'error',
      message,
    }));
    return { kind: 'issues', issues: [...issues, ...general], hardware };
  }
  const drafterNotes =
    input.deployment?.speculative !== undefined && drafter?.kind === 'ok' ? drafter.notes : [];
  return {
    kind: 'ok',
    input,
    hardware,
    issues,
    notes: [...hardware.notes, ...loaded.notes, ...drafterNotes],
  };
}

/** A successful {@link BuildResult}. */
export type BuildOk = Extract<BuildResult, { kind: 'ok' }>;

/** Prefix of the decode pool's issue messages in the combined list. */
export const DECODE_POOL_ISSUE_PREFIX = 'Decode pool: ';

/**
 * Result of {@link buildDisaggregatedInput}. `prefill` and `decode` are each pool's own {@link BuildResult} (the
 * decode one with unprefixed messages, for the panels while they edit that pool); `issues` is the combined list: the
 * prefill pool's issues, the decode pool's prefixed with {@link DECODE_POOL_ISSUE_PREFIX}, and the disaggregation
 * settings' own (`outputLen`, `instances`).
 */
export type DisaggregatedBuildResult =
  | {
      kind: 'ok';
      /** The engine input of `simulateDisaggregated`. */
      input: DisaggregatedInput;
      prefill: BuildOk;
      decode: BuildOk;
      /** Warnings only. */
      issues: FieldIssue[];
    }
  | {
      kind: 'issues';
      prefill: BuildResult;
      decode: BuildResult;
      /** At least one error; nothing runs. */
      issues: FieldIssue[];
    };

/** True for an integer ≥ 1. */
function isCount(x: number): boolean {
  return Number.isInteger(x) && x >= 1;
}

/** What both pools share: `input` without its `hardware` and `parallel` (a new object; `input` is not mutated). */
export function sharedInput(input: SimInput): DisaggregatedInput['shared'] {
  const copy: Partial<SimInput> = { ...input };
  delete copy.hardware;
  delete copy.parallel;
  return copy as DisaggregatedInput['shared'];
}

/**
 * The shared part of both pools' inputs. Everything but two per-run fields comes out the same from either pool's
 * `buildSimInput`; those two are set per pool by its layout, and one `shared` must serve both:
 * - `options.overlapByClass`: each input keeps only the classes its own run has (`effectiveOptions`); the shared one
 *   keeps the union, which the engine ignores for a class a run lacks, so each pool sees its own choices;
 * - `deployment.twoBatchOverlap`: set only where it applies (MoE mode B, no offload), and the engine rejects it
 *   elsewhere; it stays on only when it applies to both pools, else it is off for both with a warning.
 */
function poolsShared(
  prefill: SimInput,
  decode: SimInput,
  issues: FieldIssue[],
): DisaggregatedInput['shared'] {
  const shared = sharedInput(prefill);
  const byClass = { ...decode.options?.overlapByClass, ...prefill.options?.overlapByClass };
  if (Object.keys(byClass).length > 0)
    shared.options = { ...shared.options, overlapByClass: byClass };
  const prefillTbo = prefill.deployment?.twoBatchOverlap === true;
  const decodeTbo = decode.deployment?.twoBatchOverlap === true;
  if (prefillTbo !== decodeTbo) {
    issues.push({
      field: 'twoBatchOverlap',
      level: 'warning',
      message:
        'Two-batch overlap needs expert parallelism across TP groups in both pools; it is off for this run.',
    });
    if (shared.deployment !== undefined) {
      const deployment = { ...shared.deployment };
      delete deployment.twoBatchOverlap;
      if (Object.keys(deployment).length > 0) shared.deployment = deployment;
      else delete shared.deployment;
    }
  }
  return shared;
}

/**
 * The disaggregated input of `state`, or undefined when disaggregation is off (so the off path builds exactly
 * today's single-pool input with {@link buildSimInput} and nothing else). The prefill pool is `state` itself, the
 * decode pool `poolView(state, 'decode')`, each built by {@link buildSimInput}; the shared part is the prefill
 * pool's input without its hardware and parallel. Then the settings' ranges (output length and instances integers
 * ≥ 1), then `disaggregationErrors` (anything the per-pool checks did not catch, as general errors).
 */
export function buildDisaggregatedInput(
  state: AppState,
  loaded: LoadedModel | null,
  drafter?: DrafterStatus,
): DisaggregatedBuildResult | undefined {
  const d = state.disaggregation;
  if (d === undefined) return undefined;
  const prefill = buildSimInput(state, loaded, drafter);
  const decode = buildSimInput(poolView(state, 'decode'), loaded, drafter);
  const outputLen = d.outputLen ?? DEFAULT_OUTPUT_LEN;
  const instances = d.instances ?? { prefill: 1, decode: 1 };
  const settings: FieldIssue[] = [];
  if (!isCount(outputLen))
    settings.push({
      field: 'outputLen',
      level: 'error',
      message: 'Output length must be a whole number ≥ 1.',
    });
  if (!isCount(instances.prefill) || !isCount(instances.decode))
    settings.push({
      field: 'instances',
      level: 'error',
      message: 'Instance counts must be whole numbers ≥ 1.',
    });
  const kvTransfer = kvTransferConfig(d);
  const { bandwidthPerDirectionBytesPerSec: kvBw, latencySec: kvLat } = kvTransfer.link;
  if (!(Number.isFinite(kvBw) && kvBw > 0))
    settings.push({
      field: 'kvBandwidth',
      level: 'error',
      message: 'KV link bandwidth must be above 0.',
    });
  if (!(Number.isFinite(kvLat) && kvLat >= 0))
    settings.push({
      field: 'kvLatency',
      level: 'error',
      message: 'KV link latency must be 0 or more.',
    });
  const issues = [
    ...prefill.issues,
    ...decode.issues.map((i) => ({ ...i, message: DECODE_POOL_ISSUE_PREFIX + i.message })),
    ...settings,
  ];
  if (prefill.kind !== 'ok' || decode.kind !== 'ok' || hasErrors(issues))
    return { kind: 'issues', prefill, decode, issues };
  const shared = poolsShared(prefill.input, decode.input, issues);
  const input: DisaggregatedInput = {
    shared,
    prefillPool: {
      hardware: prefill.input.hardware,
      parallel: prefill.input.parallel,
      batch: prefill.input.workload.batch,
    },
    decodePool: {
      hardware: decode.input.hardware,
      parallel: decode.input.parallel,
      batch: decode.input.workload.batch,
    },
    config: {
      outputLen,
      prefillInstances: instances.prefill,
      decodeInstances: instances.decode,
      kvTransfer,
    },
  };
  const errors = disaggregationErrors(input).map((message): FieldIssue => ({
    field: 'general',
    level: 'error',
    message,
  }));
  if (errors.length > 0) return { kind: 'issues', prefill, decode, issues: [...issues, ...errors] };
  return { kind: 'ok', input, prefill, decode, issues };
}
