import { sameModel } from './state/sameModel';
import { useEffect, useMemo, useReducer, useRef, useState, type CSSProperties } from 'react';
import { loadHfModel } from '../hf/load';
import { getHfToken } from '../hf/token';
import { drafterLoadMessage, loadDrafter } from '../hf/loadDrafter';
import { ModelPanel } from './components/ModelPanel';
import { ScrollArea } from './components/ScrollArea';
import { Splitter, splitBounds, useSplit } from './components/Splitter';
import { MiniResult } from './components/MiniResult';
import { useStacked } from './layout';
import {
  decodeState,
  encodeState,
  reduce,
  modelKey,
  BUNDLED_DEFAULT_MODEL,
  loadedFromHf,
  buildSimInput,
  buildDisaggregatedInput,
  poolView,
  effectivePrefillPhase,
  formatRateAtContext,
  type AppAction,
  type PoolAction,
  type PoolId,
  type MemorySuggestion,
  stateMemorySuggestions,
  drafterSource,
  type DrafterStatus,
  type LoadedModel,
} from './state';
import { workerRunner, type SimRunner, type RunResult } from './sim/runner';
import {
  QuantPanel,
  HardwarePanel,
  ParallelPanel,
  WorkloadPanel,
  AssumptionsPanel,
} from './Panels';
import { ResultHero, BreakdownTable, BatchSweepChart, RooflineChart } from './Results';
import { DisaggregationPanel } from './components/DisaggregationPanel';
import type { PanelProps } from './components/types';
import '../styles/tokens.css';
import './panels.css';
import './app.css';

export type ModelLoader = typeof loadHfModel;
export type DrafterLoader = typeof loadDrafter;
export function App({
  loader = loadHfModel,
  drafterLoader = loadDrafter,
  runner: suppliedRunner,
}: {
  loader?: ModelLoader;
  /** Loads a block drafter's config.json (speculative decoding); injectable for tests. */
  drafterLoader?: DrafterLoader;
  runner?: SimRunner;
}) {
  const initial = useMemo(() => decodeState(location.search), []);
  const [state, dispatch] = useReducer(reduce, initial.state);
  const [dropped, setDropped] = useState(initial.dropped);
  const [id, setId] = useState(initial.state.model.id);
  const [attempt, setAttempt] = useState(0);
  const [modelState, setModelState] = useState<{
    key: string;
    loaded?: LoadedModel;
    error?: string;
    gated?: boolean;
  }>({ key: BUNDLED_DEFAULT_MODEL.key, loaded: BUNDLED_DEFAULT_MODEL });
  const [simulation, setSimulation] = useState<{ key: string; result?: RunResult; error?: string }>(
    { key: '' },
  );
  const [dim, setDim] = useState(false);
  const runner = useMemo(() => suppliedRunner ?? workerRunner(), [suppliedRunner]);
  const choiceKey = modelKey(state.model);
  const isBundled = choiceKey === BUNDLED_DEFAULT_MODEL.key;
  // The bundled default model is used until (and unless) a fresher copy loads; its refresh never errors.
  const loaded =
    modelState.key === choiceKey
      ? (modelState.loaded ?? (isBundled ? BUNDLED_DEFAULT_MODEL : null))
      : isBundled
        ? BUNDLED_DEFAULT_MODEL
        : null;
  const loading =
    !isBundled && (modelState.key !== choiceKey || (!modelState.loaded && !modelState.error));
  // The drafter repo a block spec-dec method needs (none for MTP, the checkpoint's own or the hypothetical drafter).
  const source = loaded
    ? drafterSource(state.deployment.specDec, loaded.model, state.model.id)
    : undefined;
  const drafterRepo = source?.kind === 'repo' ? source.repoId : undefined;
  const [drafterState, setDrafterState] = useState<DrafterStatus | undefined>(undefined);
  const drafter = useMemo<DrafterStatus | undefined>(
    () =>
      drafterRepo === undefined
        ? undefined
        : drafterState?.repoId === drafterRepo
          ? drafterState
          : { kind: 'loading', repoId: drafterRepo },
    [drafterRepo, drafterState],
  );
  useEffect(() => {
    if (drafterRepo === undefined) return;
    let current = true;
    const fail = (message: string) => {
      if (current) setDrafterState({ kind: 'error', repoId: drafterRepo, message });
    };
    drafterLoader(drafterRepo, { token: getHfToken() })
      .then((result) => {
        if (!current) return;
        if (result.kind === 'ok')
          setDrafterState({
            kind: 'ok',
            repoId: drafterRepo,
            spec: result.spec,
            notes: result.notes,
          });
        else fail(drafterLoadMessage(result));
      })
      .catch(() => fail('Drafter load failed. Check your connection.'));
    return () => {
      current = false;
    };
  }, [drafterRepo, drafterLoader]);
  const built = useMemo(() => buildSimInput(state, loaded, drafter), [state, loaded, drafter]);
  // Prefill/decode disaggregation: undefined while off, so the off path is exactly the single-pool one.
  const pdBuilt = useMemo(
    () => buildDisaggregatedInput(state, loaded, drafter),
    [state, loaded, drafter],
  );
  // Which pool the Hardware, Parallelism and batch fields edit (not in the URL); always the prefill pool while off.
  const [editingPoolChoice, setEditingPool] = useState<PoolId>('prefill');
  const editingPool: PoolId = state.disaggregation ? editingPoolChoice : 'prefill';
  // The issues the notice lists and the run waits on: the combined list while on.
  const allIssues = pdBuilt?.issues ?? built.issues;
  const runnable = pdBuilt ? pdBuilt.kind === 'ok' : built.kind === 'ok';
  const query = encodeState(state);
  useEffect(() => {
    const timer = setTimeout(() => {
      const search = query ? `?${query}` : '';
      history.replaceState(null, '', `${location.pathname}${search}${location.hash}`);
      try {
        sessionStorage.setItem('specmaxx:last-query', search);
      } catch {
        /* storage may be disabled */
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);
  useEffect(() => {
    const restore = () => {
      const parsed = decodeState(location.search);
      dispatch({ type: 'replace', state: parsed.state });
      setDropped(parsed.dropped);
      setId(parsed.state.model.id);
    };
    addEventListener('popstate', restore);
    return () => removeEventListener('popstate', restore);
  }, []);
  useEffect(() => {
    const choice = state.model;
    let current = true;
    loader(choice.id, { revision: choice.revision, token: getHfToken() })
      .then((result) => {
        if (!current) return;
        if (result.kind === 'ok') {
          const next = loadedFromHf(choice, result);
          setModelState((previous) => {
            const active =
              previous.key === next.key
                ? (previous.loaded ??
                  (next.key === BUNDLED_DEFAULT_MODEL.key ? BUNDLED_DEFAULT_MODEL : undefined))
                : undefined;
            // Preserve identity: neither buildSimInput nor the memory-fix probe needs to run again.
            return active && sameModel(active, next)
              ? previous.loaded === active
                ? previous
                : { key: next.key, loaded: active }
              : { key: next.key, loaded: next };
          });
        } else {
          const error =
            result.kind === 'unsupported'
              ? `Unsupported architecture: ${result.modelType ?? result.architecture ?? 'unknown'}. Supported: ${result.supported.join(', ')}. ${result.reason ?? ''}`
              : result.kind === 'invalid_config'
                ? result.errors.join('; ')
                : result.message;
          if (modelKey(choice) === BUNDLED_DEFAULT_MODEL.key)
            setModelState({ key: modelKey(choice), loaded: BUNDLED_DEFAULT_MODEL });
          else
            setModelState({
              key: modelKey(choice),
              error,
              gated: result.kind === 'unauthorized' || result.kind === 'forbidden',
            });
        }
      })
      .catch(() => {
        if (!current) return;
        if (modelKey(choice) === BUNDLED_DEFAULT_MODEL.key)
          setModelState({ key: modelKey(choice), loaded: BUNDLED_DEFAULT_MODEL });
        else
          setModelState({
            key: modelKey(choice),
            error: 'Model load failed. Check your connection and try Load again.',
          });
      });
    return () => {
      current = false;
    };
  }, [state.model, loader, attempt]);
  useEffect(() => {
    if (built.kind !== 'ok') return;
    if (pdBuilt !== undefined && pdBuilt.kind !== 'ok') return;
    let current = true;
    const key = query;
    const timer = setTimeout(() => {
      if (current) setDim(true);
    }, 150);
    // With disaggregation on, the single-pool run is the prefill pool's state; the pools run on top of it.
    const run =
      pdBuilt === undefined
        ? runner.run(built.input, built.notes)
        : runner.run(built.input, built.notes, {
            input: pdBuilt.input,
            decodeNotes: pdBuilt.decode.notes,
          });
    run
      .then((result) => {
        if (current) {
          setSimulation({ key, result });
          setDim(false);
        }
      })
      .catch((error: unknown) => {
        if (current) {
          setSimulation({ key, error: error instanceof Error ? error.message : String(error) });
          setDim(false);
        }
      });
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [built, pdBuilt, query, runner]);
  const titleModel = loaded?.label ?? state.model.id;
  const titleResult = runnable ? simulation.result?.summary : undefined;
  const pdResult = runnable && pdBuilt ? simulation.result?.pd : undefined;
  // With disaggregation on: the prefill pool's prefill and the decode pool's decode.
  const titlePrefill = pdResult ? effectivePrefillPhase(pdResult.summary) : titleResult?.prefill;
  const titleDecode = pdResult?.summary.decodePool ?? titleResult;
  useEffect(() => {
    document.title =
      titlePrefill && titleDecode
        ? `${titleModel} · ${formatRateAtContext(titlePrefill.tokensPerSec, state.workload.promptLen)} prefill · ${formatRateAtContext(titleDecode.decode.tokensPerSec, state.workload.decodeContextLen)} decode`
        : `${titleModel} · specmaxx`;
  }, [
    titleModel,
    titlePrefill,
    titleDecode,
    state.workload.promptLen,
    state.workload.decodeContextLen,
  ]);
  useEffect(() => () => runner.dispose?.(), [runner]);
  const props = { state, dispatch, loaded, built };
  // The per-pool panels (Hardware, Parallelism, the batch field) edit the chosen pool's view.
  const poolProps: PanelProps =
    pdBuilt === undefined
      ? props
      : editingPool === 'decode'
        ? {
            state: poolView(state, 'decode'),
            dispatch: (action: AppAction) =>
              dispatch({ type: 'inPool', pool: 'decode', action: action as PoolAction }),
            loaded,
            built: pdBuilt.decode,
            pool: 'decode',
            otherPoolDp: state.parallel.dp,
          }
        : {
            ...props,
            built: pdBuilt.prefill,
            pool: 'prefill',
            otherPoolDp: state.disaggregation?.decodePool.parallel.dp,
          };
  // The memory card's one-click fixes of each pool that does not fit (computed with the sim, so only when needed).
  // They run the same actions as the manual controls; the decode pool's through `inPool`.
  const memoryFixes = useMemo(
    () =>
      loaded !== null && runnable && simulation.key === query && simulation.result
        ? stateMemorySuggestions(
            state,
            loaded,
            drafter,
            simulation.result.pd
              ? {
                  prefill: simulation.result.pd.summary.prefillPool,
                  decode: simulation.result.pd.summary.decodePool,
                }
              : { prefill: simulation.result.summary },
          )
        : undefined,
    [state, loaded, drafter, runnable, simulation, query],
  );
  const applySuggestion = (suggestion: MemorySuggestion, pool: PoolId) => {
    for (const action of suggestion.actions) {
      dispatch(pool === 'decode' ? { type: 'inPool', pool: 'decode', action } : action);
    }
  };
  // The summary the memory card, breakdown, sweep and notes show: with disaggregation on, the edited pool's (the
  // prefill pool's run fits at P and shows only its prefill phase).
  const result = !runnable
    ? undefined
    : pdResult
      ? editingPool === 'decode'
        ? pdResult.summary.decodePool
        : pdResult.summary.prefillPool
      : pdBuilt
        ? undefined
        : simulation.result?.summary;
  const detailPhases: readonly ('prefill' | 'decode')[] | undefined =
    pdResult && editingPool === 'prefill' ? ['prefill'] : undefined;
  const sweep = pdResult
    ? editingPool === 'decode'
      ? pdResult.decodeSweep
      : undefined
    : simulation.result?.sweep;
  // The break-even batch of the sweep shown (spec-dec on): the decode pool's with disaggregation on.
  const breakEvenBatch = pdResult
    ? editingPool === 'decode'
      ? pdResult.decodeBreakEvenBatch
      : undefined
    : simulation.result?.breakEvenBatch;
  const error = modelState.key === choiceKey ? modelState.error : undefined;
  const stacked = useStacked();
  const workspaceRef = useRef<HTMLElement>(null);
  const split = useSplit(workspaceRef);
  // The mini result bar shows once the hero figures are mostly out of view: of the output pane (split layout) or
  // of the window (stacked layout).
  const [outputViewport, setOutputViewport] = useState<HTMLDivElement | null>(null);
  const [figures, setFigures] = useState<HTMLDivElement | null>(null);
  const [figuresVisible, setFiguresVisible] = useState(true);
  useEffect(() => {
    if (!figures || typeof IntersectionObserver !== 'function') return;
    const observer = new IntersectionObserver(
      ([entry]) => setFiguresVisible(entry.intersectionRatio >= 0.5),
      { root: stacked ? null : outputViewport, threshold: [0, 0.5, 1] },
    );
    observer.observe(figures);
    return () => observer.disconnect();
  }, [figures, outputViewport, stacked]);
  function load(typed?: string) {
    const nextId = (typed ?? id).trim();
    if (!nextId) return;
    const unchanged = state.model.id === nextId;
    const choice = unchanged ? state.model : { kind: 'hf' as const, id: nextId };
    if (!unchanged) dispatch({ type: 'setModel', model: choice });
    setModelState({ key: modelKey(choice) });
    setAttempt((n) => n + 1);
  }
  const notices = (
    <>
      {dropped.length > 0 && (
        <aside className="notice">
          Ignored invalid URL values: {dropped.join(', ')}.{' '}
          <button className="quiet" onClick={() => setDropped([])}>
            Dismiss
          </button>
        </aside>
      )}
      {allIssues.length > 0 && (
        <aside className="notice" role="alert">
          <p>
            {!runnable
              ? 'No estimate is computed until the input errors are resolved.'
              : 'Input notes'}
          </p>
          <ul>
            {allIssues.map((issue, index) => (
              <li className={issue.level === 'error' ? 'error' : ''} key={index}>
                {issue.message}
              </li>
            ))}
          </ul>
        </aside>
      )}
      {simulation.error && simulation.key === query && (
        <p role="alert" className="error">
          {simulation.error}
        </p>
      )}
    </>
  );
  const hero = (
    <ResultHero
      state={state}
      loaded={loaded}
      built={
        // While on, an error in either pool (or the settings) keeps the hero empty with the issues message.
        pdBuilt?.kind === 'issues' && built.kind === 'ok'
          ? { kind: 'issues', issues: pdBuilt.issues, hardware: built.hardware }
          : built
      }
      result={result}
      pending={dim && simulation.key !== query}
      suggestions={memoryFixes}
      onSuggestion={applySuggestion}
      roofline={!stacked}
      figuresRef={setFigures}
      pd={
        pdResult && pdBuilt
          ? {
              summary: pdResult.summary,
              decodeState: poolView(state, 'decode'),
              decodeBuilt: pdBuilt.decode,
              phases: detailPhases ?? ['prefill', 'decode'],
            }
          : undefined
      }
    />
  );
  const mini = result && (
    <MiniResult result={result} visible={!figuresVisible} pd={pdResult?.summary} />
  );
  const inputs = (
    <>
      <ModelPanel
        id={id}
        onIdChange={setId}
        onLoad={load}
        onRetry={() => {
          setModelState({ key: choiceKey });
          setAttempt((n) => n + 1);
        }}
        loaded={loaded}
        loading={loading}
        error={error}
        gated={modelState.key === choiceKey && !!modelState.gated}
      />
      <QuantPanel {...props} />
      <DisaggregationPanel
        state={state}
        dispatch={dispatch}
        editingPool={editingPool}
        onEditingPool={setEditingPool}
        issues={allIssues}
      />
      <HardwarePanel {...poolProps} />
      <ParallelPanel {...poolProps} />
      <WorkloadPanel {...poolProps} drafter={drafter} />
      <AssumptionsPanel {...props} />
    </>
  );
  const details = result && (
    <div className="detail-results">
      {stacked && (
        <RooflineChart
          roof={result.roofline}
          contexts={{ prefill: result.prefill.contextTokens, decode: result.decode.contextTokens }}
          phases={detailPhases}
        />
      )}
      <BreakdownTable result={result} phases={detailPhases} />
      {sweep && <BatchSweepChart points={sweep} breakEvenBatch={breakEvenBatch} />}
      {result.warnings.length > 0 && (
        <details className="notes">
          <summary>Notes and approximations ({result.warnings.length})</summary>
          <ul>
            {result.warnings.map((w, n) => (
              <li key={n}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
  return (
    <>
      <header className="app-header">
        <a className="wordmark" href="./">
          specmaxx
          <span className="roof-mark" aria-hidden="true" />
        </a>
        <span className="tagline">Theoretical ceilings for your rig</span>
        <a href="./about.html">How it works</a>
      </header>
      {/* Split layout: output (left) | splitter | inputs (right), each pane scrolling on its own. Stacked layout
          (phone): hero and memory, then the inputs, then the analysis, in one document scroll. */}
      <main
        ref={workspaceRef}
        className="workspace"
        data-layout={stacked ? 'stacked' : 'split'}
        style={stacked ? undefined : ({ '--inputs-fraction': split.fraction } as CSSProperties)}
      >
        {stacked ? (
          <>
            <div className="output">
              {notices}
              {hero}
              {mini}
            </div>
            <section className="input-band" id="inputs-pane" aria-label="Inputs">
              {inputs}
            </section>
            {details}
          </>
        ) : (
          <>
            <div className="output">
              {mini}
              <ScrollArea className="pane-scroll output-scroll" viewportRef={setOutputViewport}>
                {notices}
                {hero}
                {details}
              </ScrollArea>
            </div>
            <Splitter
              fraction={split.fraction}
              bounds={splitBounds(split.width)}
              controls="inputs-pane"
              onChange={split.set}
              onReset={split.reset}
            />
            <section className="input-band" id="inputs-pane" aria-label="Inputs">
              <ScrollArea className="pane-scroll input-band-scroll">{inputs}</ScrollArea>
            </section>
          </>
        )}
      </main>
    </>
  );
}
