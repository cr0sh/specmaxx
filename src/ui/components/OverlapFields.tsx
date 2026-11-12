import { Fragment, useState } from 'react';
import type { OverlapClass } from '../../sim/engine/types';
import { hasOffDevicePlacement } from '../../sim/engine/offloadRules';
import {
  effectiveParallel,
  hasMoe,
  hasLookupCollectives,
  offloadsAnything,
  overlapClassesOf,
  twoBatchOverlapApplies,
} from '../state';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { Segmented } from './Segmented';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/** Picker value of each overlap mode: the default (follows the global switch), none (0), full (1), partial (0..1). */
type OverlapMode = 'default' | 'none' | 'full' | 'partial';

/** Fraction a newly picked "Partial" starts at. Dimensionless. */
const DEFAULT_PARTIAL = 0.5;

/** The labels and help of each class's row. */
const CLASS_ROWS: Record<OverlapClass, { label: string; percentLabel: string; help: string[] }> = {
  tp: {
    label: 'All-reduce (TP)',
    percentLabel: 'All-reduce hidden %',
    help: [
      'Share of the tensor-parallel exchange time that can hide behind compute.',
      'None adds it all to the step; full hides it, up to the compute time.',
      'Default follows Communication overlap under Assumptions.',
    ],
  },
  ep: {
    label: 'All-to-all (EP)',
    percentLabel: 'All-to-all hidden %',
    help: [
      'Share of the expert all-to-all time that can hide behind compute.',
      'None adds it all to the step; full hides it, up to the compute time.',
      'Default follows Communication overlap, or full with two-batch overlap.',
    ],
  },
  pp: {
    label: 'Stage hand-off (PP)',
    percentLabel: 'Hand-off hidden %',
    help: [
      'Share of the hand-off to the next stage that runs while this stage starts its next microbatch.',
      'None adds it all to the stage time; full takes it off the stage.',
      'Default follows Communication overlap under Assumptions.',
    ],
  },
};

/** The existing TP class also controls shared lookup exchanges, including at TP1. */
const LOOKUP_TP_ROW = {
  label: 'TP / lookup exchange',
  percentLabel: 'Exchange hidden %',
  help: [
    'Share of the TP and lookup-table transfer time that can hide behind compute.',
    'Only bandwidth time can overlap; message latency stays on the critical path.',
    'Default follows Communication overlap under Assumptions.',
  ],
};

const PERCENT_HELP = helpLines([
  'Share of this exchange hidden behind compute, in percent.',
  '0% adds it all to the step; 100% hides it all, up to the compute time.',
]);

const TWO_BATCH_HELP = helpLines([
  "Splits each batch in two, so one half's expert exchange hides behind the other half's compute.",
  'Each half reads the weights again, so it only pays when the exchange is not already hidden.',
  'SGLang needs about 64–128 tokens per GPU to gain; that engine limit is not enforced here.',
]);

/**
 * Hardware → Advanced → "Communication overlap": per overlap class (TP/lookup exchanges, EP exchanges, PP send; only the
 * classes the run has, `overlapClassesOf`) the share of its transfer time that may hide behind compute
 * (`SimOptions.overlapByClass`: Default follows Assumptions → Communication overlap, None 0, Full 1, Partial a
 * percent), and two-batch overlap (`DeploymentChoice.twoBatchOverlap`, MoE mode B only; docs/simulation-model.md §7
 * "Stage time and overlap", "Two-batch overlap"). Hidden when the run has no communication between devices.
 */
export function OverlapFields({ state, dispatch, loaded, built }: PanelProps) {
  // Classes whose row shows "Partial" while being edited (a typed 0 or 100 keeps the percent field open).
  const [editing, setEditing] = useState<ReadonlySet<OverlapClass>>(new Set());
  if (loaded === null) return null;
  const parallel = effectiveParallel(state.parallel, loaded.model, state.parallelLayout);
  const lookupCollectives = hasLookupCollectives(
    loaded.model,
    parallel,
    state.deployment.offload,
    built.hardware?.topology,
  );
  const classes = overlapClassesOf(loaded.model, parallel, lookupCollectives);
  if (classes.length === 0) return null;
  const twoBatchOn = state.deployment.twoBatchOverlap === true;
  // Match buildSimInput's effective placement: a saved prefix-only KV choice need not move data in this run.
  const offloads =
    built.kind === 'ok'
      ? hasOffDevicePlacement(built.input.deployment?.offload, state.workload.prefixCacheHitRatio)
      : offloadsAnything(state.deployment.offload);
  const twoBatchShown =
    hasMoe(loaded.model) &&
    (twoBatchOverlapApplies(loaded.model, parallel, offloads) ||
      (twoBatchOn && parallel.ep > parallel.tp));
  const globalLabel = (state.options.overlap ?? 'ideal') === 'ideal' ? 'full' : 'none';
  const byClass = state.options.overlapByClass ?? {};

  const modeOf = (c: OverlapClass): OverlapMode => {
    const f = byClass[c];
    if (f === undefined) return 'default';
    if (editing.has(c) || (f > 0 && f < 1) || !Number.isFinite(f)) return 'partial';
    return f === 0 ? 'none' : f === 1 ? 'full' : 'partial';
  };
  const setFraction = (c: OverlapClass, fraction: number | undefined) =>
    dispatch({ type: 'setOverlapClass', overlapClass: c, fraction });
  const setEditingClass = (c: OverlapClass, on: boolean) =>
    setEditing((prev) => {
      const next = new Set(prev);
      if (on) next.add(c);
      else next.delete(c);
      return next;
    });

  return (
    <fieldset className="unlock advanced-fields" data-testid="overlap-advanced">
      <legend>Communication overlap</legend>
      {classes.map((c) => {
        const row = c === 'tp' && lookupCollectives ? LOOKUP_TP_ROW : CLASS_ROWS[c];
        const mode = modeOf(c);
        const defaultLabel = `Default (${c === 'ep' && twoBatchOn ? 'full' : globalLabel})`;
        const f = byClass[c] ?? DEFAULT_PARTIAL;
        const invalid = !(Number.isFinite(f) && f >= 0 && f <= 1);
        return (
          <Fragment key={c}>
            <Picker
              label={row.label}
              value={mode}
              options={[
                { value: 'default', label: defaultLabel },
                { value: 'none', label: 'None' },
                { value: 'full', label: 'Full' },
                { value: 'partial', label: 'Partial' },
              ]}
              help={helpLines(row.help)}
              onChange={(v) => {
                setEditingClass(c, v === 'partial');
                if (v === 'partial')
                  setFraction(c, mode === 'partial' ? byClass[c] : DEFAULT_PARTIAL);
                else setFraction(c, v === 'none' ? 0 : v === 'full' ? 1 : undefined);
              }}
            />
            {mode === 'partial' && (
              <NumberField
                label={row.percentLabel}
                value={Math.round(f * 1e6) / 1e4}
                min={0}
                max={100}
                step="any"
                onChange={(v) => setFraction(c, v / 100)}
                help={PERCENT_HELP}
                issues={
                  invalid
                    ? [
                        {
                          field: 'overlap',
                          level: 'error',
                          message: 'Must be between 0 and 100%.',
                        },
                      ]
                    : []
                }
              />
            )}
          </Fragment>
        );
      })}
      {twoBatchShown && (
        <Segmented
          label="Two-batch overlap"
          value={twoBatchOn ? 'on' : 'off'}
          segments={[
            { value: 'off', label: 'off' },
            { value: 'on', label: 'on' },
          ]}
          onChange={(v) => {
            if (v === 'on') setEditingClass('ep', false);
            dispatch({ type: 'setTwoBatchOverlap', on: v === 'on' });
          }}
          help={TWO_BATCH_HELP}
          issues={built.issues.filter((i) => i.field === 'twoBatchOverlap')}
        />
      )}
    </fieldset>
  );
}
