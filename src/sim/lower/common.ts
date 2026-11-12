/**
 * Shared lowering helpers: TP sharding rules, op constructors with the quantization rules applied, and the
 * elementwise FLOP-per-element table. See docs/simulation-model.md §5 (Parallelism) and §3 (Elementwise).
 */

import {
  bytesPerElement,
  mathFormatOf,
  resolveAccumulate,
  resolveGemmMath,
  type AccumulateType,
  type Dtype,
  type MathFormat,
  type TensorFormat,
} from '../ir/dtype';
import { WEIGHT_MODULE_FALLBACK, type ModuleQuant, type WeightModule } from '../ir/model';
import type { CollectiveOp, ElementwiseOp, GemmOp } from '../ir/ops';
import type { LoweringContext } from './lower';

/** Elementwise functions whose FLOP per element depend on their shape: their lowering sets `flopsPerElem`. */
export type ShapeDependentFn =
  | 'causal_conv1d'
  | 'hc_collapse'
  | 'hc_expand'
  | 'sinkhorn'
  | 'residual_mix';

/** The elementwise functions whose FLOP per element depend on their shape (set by their lowering). */
export const SHAPE_DEPENDENT_FNS: readonly ShapeDependentFn[] = [
  'causal_conv1d',
  'hc_collapse',
  'hc_expand',
  'sinkhorn',
  'residual_mix',
];

/**
 * Vector FLOP per output element of each elementwise function (documented approximations; they only feed the
 * vector-compute time). Functions whose cost depends on their shape take an explicit `flopsPerElem` from their
 * lowering ({@link SHAPE_DEPENDENT_FNS}): `causal_conv1d` 2·K, `hc_collapse` 2·N, `hc_expand` 2N+2 (mixing) or 2,
 * `sinkhorn` 4·iters per N×N element, `residual_mix` 6·C per output element over C candidates (per candidate: RMS
 * statistic 2, normalize-and-score 2, weighted sum 2; the C-way softmax is per token, not per element). `router_topk`
 * is per logit. `situ` ≈ 9 (β·tanh(x/β) 5, sigmoid 3, multiply 1), `situ_mul` ≈ 15 (situ + the tanh bound on `up` 5 +
 * the multiply); `ngram_gate` ≈ 8 (two norm statistics, a dot product, a sigmoid and the fused add); `kv_pool` ≈ 4
 * (softmax weight, multiply, add); `kv_cache_write` 1 (the cast of each stored K/V element into the cache format);
 * `candidate_score` 2 (one multiply-add per rank element); `relu2` 2 (max, square).
 * Unit: FLOP/element. See docs/simulation-model.md §3 "Elementwise".
 */
export const ELEMENTWISE_FLOPS_PER_ELEM: Record<
  Exclude<ElementwiseOp['fn'], ShapeDependentFn>,
  number
> = {
  rmsnorm: 4,
  layernorm: 8,
  residual_add: 1,
  rope: 3,
  qk_norm: 4,
  silu: 4,
  silu_mul: 5,
  gelu_mul: 9,
  gelu: 8,
  relu: 1,
  relu_mul: 2,
  relu2: 2,
  situ: 9,
  situ_mul: 15,
  sigmoid_mul: 5,
  gated_rmsnorm: 9,
  router_topk: 5,
  moe_combine: 2,
  ngram_gate: 8,
  kv_pool: 4,
  kv_cache_write: 1,
  candidate_score: 2,
  copy: 0,
};

/** Tokens processed per device step: `T = seqs·qLen`. Unit: tokens. */
export function stepTokens(ctx: LoweringContext): number {
  // Formula: T = seqs·qLen
  return ctx.seqs * ctx.qLen;
}

/** Bytes per element of the activation dtype. Unit: bytes/element. */
export function actBpe(ctx: LoweringContext): number {
  return bytesPerElement({ dtype: ctx.quant.activations });
}

/** High-precision dtype for norms, router, biases and conv taps (default 'bf16'). */
export function highPrecision(ctx: LoweringContext): Dtype {
  return ctx.quant.highPrecision ?? 'bf16';
}

/** Bytes per element of the high-precision dtype. Unit: bytes/element. */
export function hpBpe(ctx: LoweringContext): number {
  return bytesPerElement({ dtype: highPrecision(ctx) });
}

/**
 * Bytes per element carried by reductions (all-reduce, reduce-scatter, MoE combine): at least 16-bit.
 * Unit: bytes/element. See docs/simulation-model.md §5 "Reductions dtype".
 */
export function reduceBpe(ctx: LoweringContext): number {
  // Formula: reduceBpe = max(bpe(act), 2)
  return Math.max(actBpe(ctx), 2);
}

/** Wire dtype of reductions: the activation dtype if it is at least 16-bit, else bf16 (see {@link reduceBpe}). */
export function reduceDtype(ctx: LoweringContext): Dtype {
  return actBpe(ctx) >= 2 ? ctx.quant.activations : 'bf16';
}

/**
 * Query (or other evenly sharded) heads per TP rank: `heads / tp`; throws unless `tp` divides `heads`. Count.
 * See docs/simulation-model.md §5 "TP sharding".
 */
export function shardHeads(heads: number, tp: number, what: string): number {
  if (heads % tp !== 0) throw new Error(`${what} (${heads}) must be divisible by tp (${tp})`);
  // Formula: Hl = heads / tp
  return heads / tp;
}

/**
 * KV (or linear-attention q/k, v) heads per TP rank with replication: `kvHeads / tp` when kvHeads ≥ tp,
 * else 1 (each rank holds one replicated head). Requires `kvHeads % tp == 0` or `tp % kvHeads == 0`. Count.
 * See docs/simulation-model.md §5 "KV-head replication".
 */
export function shardKvHeads(kvHeads: number, tp: number, what: string): number {
  if (kvHeads % tp !== 0 && tp % kvHeads !== 0) {
    throw new Error(`${what} (${kvHeads}) and tp (${tp}) must divide one another`);
  }
  // Formula: KVl = kvHeads ≥ tp ? kvHeads / tp : 1
  return kvHeads >= tp ? kvHeads / tp : 1;
}

/** Evenly shard a width over `parts` ranks; throws unless divisible. Unit: elements. */
export function shardWidth(width: number, parts: number, what: string): number {
  if (width % parts !== 0) throw new Error(`${what} (${width}) must be divisible by ${parts}`);
  return width / parts;
}

/**
 * True when the dense FFN layers and the LM head are sharded over the EP group with more than one device in it
 * (`ParallelConfig.denseTpGroup === 'ep'` and ep > 1): their lowering then gathers the group's tokens before the
 * sharded GEMM and scatters the result after. With ep = 1 it is false and every op is exactly as without the field.
 * See docs/simulation-model.md §5 "DP attention + EP".
 */
export function denseOverEp(ctx: LoweringContext): boolean {
  return ctx.parallel.denseTpGroup === 'ep' && ctx.parallel.ep > 1;
}

/**
 * Size G of the group the dense FFN layers (`mlp` blocks) and the LM head are tensor-parallel over: ep when
 * {@link denseOverEp} (the whole EP group, spanning ep / tp replicas; validation requires tp = 1 then), else tp
 * (the replica's own TP ranks; `denseTpGroup` absent or `'tp'`, the rule of before). Count.
 * See docs/simulation-model.md §5 "DP attention + EP".
 */
export function denseTpGroupSize(ctx: LoweringContext): number {
  // Formula: G = (denseTpGroup = 'ep' ∧ ep > 1) ? ep : tp
  return denseOverEp(ctx) ? ctx.parallel.ep : ctx.parallel.tp;
}

/** Math format and accumulator of the attention-like cores (attentionMath ?? activations). */
export function attentionMath(ctx: LoweringContext): {
  math: MathFormat;
  accumulate: AccumulateType;
} {
  const math = mathFormatOf(ctx.quant.attentionMath ?? ctx.quant.activations);
  return { math, accumulate: resolveAccumulate(math, ctx.quant.accumulate ?? 'fp32') };
}

/** Options of {@link linearGemm}. */
export interface GemmOptions {
  /** Grouped-GEMM batch (distinct weights per entry); default 1. Count. */
  batch?: number;
  /**
   * Weight matrices read when fewer than `batch` (see `GemmOp.distinctWeightMatrices`; MoE: the expected distinct
   * experts). Set on the op only when it differs from the batch. Unit: weight matrices (count, real).
   */
  distinctWeightMatrices?: number;
  /**
   * Module class of a linear layer: selects its weight format, GEMM input dtype and sparsity through
   * {@link moduleQuant}. Exclusive with `format`. Neither given: `quant.weights` with `quant.activations` inputs.
   */
  module?: WeightModule;
  /**
   * Explicit format of a non-module GEMM (router, LM head): weights and GEMM input (default `quant.activations`);
   * dense unless `sparsity` says otherwise.
   */
  format?: ModuleQuant;
  /** Weight sparsity; default the module's (see {@link moduleQuant}), `quant.weightSparsity` with neither, 'dense' with `format`. */
  sparsity?: 'dense' | '2:4';
  /** Input dtype of the op, without changing `math` (MLA kv_b reads the cached latent); default the resolved `gemmInput`. */
  inputDtype?: Dtype;
  /** Output widths of the physical matrices fused into `n` (see `GemmOp.outParts`); default one matrix. Unit: elements. */
  outParts?: number[];
  /** Explicit side-vector entries charged to this op (see `GemmOp.sideVectorElems`). Unit: elements. */
  sideVectorElems?: number;
  /** Adds an `n`-element bias per batch entry in the high-precision dtype. */
  bias?: boolean;
}

/** The resolved numeric format of one weight-module class. */
export interface ResolvedModuleQuant {
  /** Weight storage format. */
  weights: TensorFormat;
  /** Dtype the GEMM input is quantized to. */
  gemmInput: Dtype;
  /** Structured sparsity of the weights. */
  sparsity: 'dense' | '2:4';
}

/**
 * Format of a weight-module class: `quant.modules[module]` if given, else the entry of its fallback parent
 * (`WEIGHT_MODULE_FALLBACK`, e.g. `attention_kv_up` → `attention`) if given, else `{ weights: quant.weights }`,
 * except `residual_mixer`, whose default is `{ dtype: highPrecision }`. `gemmInput` defaults to `quant.activations`,
 * also for a sub-class override that gives only `weights`: it does NOT inherit the parent's `gemmInput` (a BF16
 * `attention_kv_up` under W8A8 `attention` is weight-only).
 * `quant.weightSparsity` applies to every module but `residual_mixer` (always dense).
 * See docs/simulation-model.md §2 "Dtypes, math and accumulate".
 */
export function moduleQuant(ctx: LoweringContext, module: WeightModule): ResolvedModuleQuant {
  const parent = WEIGHT_MODULE_FALLBACK[module];
  const override =
    ctx.quant.modules?.[module] ?? (parent === undefined ? undefined : ctx.quant.modules?.[parent]);
  const defaultWeights =
    module === 'residual_mixer' ? { dtype: highPrecision(ctx) } : ctx.quant.weights;
  return {
    weights: override?.weights ?? defaultWeights,
    gemmInput: override?.gemmInput ?? ctx.quant.activations,
    sparsity: module === 'residual_mixer' ? 'dense' : (ctx.quant.weightSparsity ?? 'dense'),
  };
}

/**
 * A GEMM `m×k · k×n` with the quantization rules applied: `math = resolveGemmMath(w, gemmInput)`,
 * `accumulate = resolveAccumulate(math, quant.accumulate ?? 'fp32')`. Shapes in elements (m = tokens).
 * The weight format, GEMM input and sparsity come from `opts.module` ({@link moduleQuant}), or `opts.format`.
 */
export function linearGemm(
  name: string,
  shape: { m: number; n: number; k: number },
  ctx: LoweringContext,
  opts: GemmOptions = {},
): GemmOp {
  if (opts.module !== undefined && opts.format !== undefined) {
    throw new Error(`${name}: give a module or an explicit format, not both`);
  }
  const act = ctx.quant.activations;
  const fmt: ResolvedModuleQuant =
    opts.module !== undefined
      ? moduleQuant(ctx, opts.module)
      : opts.format !== undefined
        ? {
            weights: opts.format.weights,
            gemmInput: opts.format.gemmInput ?? act,
            sparsity: 'dense',
          }
        : {
            weights: ctx.quant.weights,
            gemmInput: act,
            sparsity: ctx.quant.weightSparsity ?? 'dense',
          };
  const weight = fmt.weights;
  const sparsity = opts.sparsity ?? fmt.sparsity;
  const math = resolveGemmMath(weight.dtype, fmt.gemmInput);
  const op: GemmOp = {
    kind: 'gemm',
    name,
    batch: opts.batch ?? 1,
    m: shape.m,
    n: shape.n,
    k: shape.k,
    inputDtype: opts.inputDtype ?? fmt.gemmInput,
    weight,
    outputDtype: act,
    math,
    accumulate: resolveAccumulate(math, ctx.quant.accumulate ?? 'fp32'),
    sparsity,
  };
  if (opts.outParts !== undefined) op.outParts = opts.outParts;
  if (opts.sideVectorElems !== undefined) op.sideVectorElems = opts.sideVectorElems;
  const distinct = opts.distinctWeightMatrices;
  if (distinct !== undefined && distinct !== op.batch) op.distinctWeightMatrices = distinct;
  if (opts.bias === true) op.biasDtype = highPrecision(ctx);
  return op;
}

/** Byte and FLOP fields of an elementwise op (everything but kind/name/fn/dtype). */
export interface ElementwiseFields {
  /** Output elements. Unit: elements. */
  elems: number;
  /** FLOP per output element; default from {@link ELEMENTWISE_FLOPS_PER_ELEM}. Unit: FLOP/element. */
  flopsPerElem?: number;
  /** Activation bytes read. Unit: bytes. */
  activationReadBytes: number;
  /** Activation bytes written. Unit: bytes. */
  activationWriteBytes: number;
  /** Weight bytes read (gamma, taps); default 0. Unit: bytes. */
  weightBytes?: number;
  /** State bytes read; default 0. Unit: bytes. */
  stateReadBytes?: number;
  /** State bytes written; default 0. Unit: bytes. */
  stateWriteBytes?: number;
  /** KV-cache bytes written by a write-only cache op (`fn: 'kv_cache_write'`); absent = none (field omitted). Unit: bytes. */
  kvWriteBytes?: number;
}

/** True for the functions of {@link SHAPE_DEPENDENT_FNS}. */
function isShapeDependent(fn: ElementwiseOp['fn']): fn is ShapeDependentFn {
  return (SHAPE_DEPENDENT_FNS as readonly string[]).includes(fn);
}

/** An elementwise op over activations of dtype `quant.activations`. */
export function elementwise(
  name: string,
  fn: ElementwiseOp['fn'],
  f: ElementwiseFields,
  ctx: LoweringContext,
): ElementwiseOp {
  let flopsPerElem = f.flopsPerElem;
  if (flopsPerElem === undefined) {
    if (isShapeDependent(fn))
      throw new Error(`${fn} needs an explicit flopsPerElem (set by its lowering)`);
    flopsPerElem = ELEMENTWISE_FLOPS_PER_ELEM[fn];
  }
  const op: ElementwiseOp = {
    kind: 'elementwise',
    name,
    fn,
    elems: f.elems,
    flopsPerElem,
    dtype: ctx.quant.activations,
    activationReadBytes: f.activationReadBytes,
    activationWriteBytes: f.activationWriteBytes,
    weightBytes: f.weightBytes ?? 0,
    stateReadBytes: f.stateReadBytes ?? 0,
    stateWriteBytes: f.stateWriteBytes ?? 0,
  };
  // Only a write-only cache op carries the field, so every other op stays exactly as before (identity).
  if (f.kvWriteBytes !== undefined) op.kvWriteBytes = f.kvWriteBytes;
  return op;
}

/**
 * A per-token norm over `tokens × dim` activations: read and write `tokens·dim·bpe(act)`, gamma (and beta for
 * layernorm) `dim·bpe(hp)` each. Replicated on every TP rank (no sequence parallelism).
 */
export function normOp(
  name: string,
  norm: 'rmsnorm' | 'layernorm',
  tokens: number,
  dim: number,
  ctx: LoweringContext,
): ElementwiseOp {
  const elems = tokens * dim;
  const vectors = norm === 'layernorm' ? 2 : 1;
  // Formula: read = write = T·dim·bpe(act); weights = (layernorm ? 2 : 1)·dim·bpe(hp)
  // See docs/simulation-model.md §5 "Norms and residuals"
  return elementwise(
    name,
    norm,
    {
      elems,
      activationReadBytes: elems * actBpe(ctx),
      activationWriteBytes: elems * actBpe(ctx),
      weightBytes: vectors * dim * hpBpe(ctx),
    },
    ctx,
  );
}

/** True unless this step starts the sequence (first prefill chunk): a prior rolling/recurrent state exists. */
export function hasPriorState(ctx: LoweringContext): boolean {
  return ctx.kvLen > ctx.qLen;
}

/**
 * State snapshots a step's recurrent and conv-state ops write per sequence: `k + 1` in a speculative verify
 * (`ctx.speculative.stateSnapshots`, decode only), so the state can be rolled back to the accepted prefix; 1
 * otherwise (design doc §3.3). Multiplies the state WRITE bytes only. Count.
 */
export function stateSnapshots(ctx: LoweringContext): number {
  // Formula: snapshots = decode with spec-dec ? k + 1 : 1
  // See docs/simulation-model.md §13 ("Verify")
  return ctx.phase === 'decode' ? (ctx.speculative?.stateSnapshots ?? 1) : 1;
}

/**
 * A depthwise causal conv over `tokens × channels` activations with `kernel` taps (dilated by `dilation`):
 * FLOP 2·K per element (+1 with a fused residual add), taps `channels·K·bpe(hp)`, and a rolling state of
 * `(K−1)·dilation` tokens per sequence, read when a prior state exists and written every step (once per
 * {@link stateSnapshots} snapshot: `k + 1` times in a speculative verify).
 */
export function causalConvOp(
  name: string,
  tokens: number,
  channels: number,
  conv: { kernel: number; dilation?: number; residual?: boolean },
  ctx: LoweringContext,
): ElementwiseOp {
  const elems = tokens * channels;
  const act = actBpe(ctx);
  const stateBytes = ctx.seqs * channels * convStateTokens(conv) * act;
  // Formula: elems = T·C; flopsPerElem = 2·K (+1 residual); taps C·K·bpe(hp); state seqs·C·(K−1)·dil·bpe(act)
  // Formula: state write = snapshots·state   (snapshots = k + 1 in a speculative verify, else 1)
  // See docs/simulation-model.md §5 "Short convs", §13 ("Verify")
  return elementwise(
    name,
    'causal_conv1d',
    {
      elems,
      flopsPerElem: 2 * conv.kernel + (conv.residual === true ? 1 : 0),
      activationReadBytes: elems * act,
      activationWriteBytes: elems * act,
      weightBytes: channels * conv.kernel * hpBpe(ctx),
      stateReadBytes: hasPriorState(ctx) ? stateBytes : 0,
      stateWriteBytes: stateSnapshots(ctx) * stateBytes,
    },
    ctx,
  );
}

/** Tokens held by a causal conv's rolling state: `(K−1)·dilation`. Unit: tokens. */
export function convStateTokens(conv: { kernel: number; dilation?: number }): number {
  // Formula: S_tokens = (K−1)·dilation
  return Math.max(conv.kernel - 1, 0) * (conv.dilation ?? 1);
}

/**
 * The TP all-reduce of a `tokens × hidden` partial sum, carried at `reduceBpe`; none when tp = 1.
 * messageBytes = the full buffer S per device.
 */
export function tpAllReduce(
  name: string,
  tokens: number,
  hidden: number,
  ctx: LoweringContext,
): CollectiveOp[] {
  const tp = ctx.parallel.tp;
  if (tp === 1) return [];
  // Formula: S = T·h·reduceBpe
  // See docs/simulation-model.md §5 "TP sharding"
  return [
    {
      kind: 'collective',
      name,
      collective: 'all_reduce',
      group: 'tp',
      groupSize: tp,
      messageBytes: tokens * hidden * reduceBpe(ctx),
      dtype: reduceDtype(ctx),
    },
  ];
}
