/**
 * Lowering of a speculative-decoding drafter: the `draft.*` layers appended to the last pipeline stage when
 * speculative decoding is on (design doc §3.3, §3.4; docs/simulation-model.md §13). Every op name starts with
 * `draft.`, so the drafter's ops never merge with the target's in a breakdown.
 *
 * Notation (per microbatch step, per device): b = `ctx.seqs` sequences; k = draft tokens per step; B = block
 * positions per pass (DFlash 2: k + 1, anchor + k masks; DSpark: k, see {@link draftBlockLength}); m = m_ctx =
 * ⌈E[τ]⌉ newly committed tokens of the previous step, whose target hidden
 * states the drafter ingests; N = the decode context before the verify (`ctx.kvLen − ctx.qLen` of the verify
 * context); W = the drafter's window; h = width; V = vocabulary; Hl, KVl = the drafter's query / KV heads per TP
 * rank; hd = head dim.
 *
 * Two drafter kinds:
 * - MTP (`kind: 'mtp'`): the checkpoint's own module(s), lowered through the target's block lowering
 *   ({@link lowerBlock}). `layers` are modules in chain order (DeepSeek-V3: module j drafts position j); pass p of a
 *   step runs module `min(p, L)`; passes p ≤ L catch the module's cache up over the m committed rows (qLen m), later
 *   passes reuse the last module on one row. With one module (every released checkpoint but Inkling): one catch-up
 *   pass of qLen m, then k − 1 one-row passes (design doc §3.3).
 * - Block drafter (`kind: 'block_drafter'`, DFlash 2 / DSpark): one parallel pass over the block, conditioned on the
 *   target features of `targetLayerIds` through `fc` and per-layer context K/V. DeepSeek-V4-style MoE drafters
 *   (`moeLayers`) lower their layers through {@link lowerBlock} with block rows (see {@link lowerMoeDrafterLayer}).
 *
 * The shared embedding and LM head are the target's. The LM head is on the last stage, where the drafter runs; the
 * embedding is too at pp = 1, and at pp > 1 the last stage holds its own copy (memory.ts: `draftEmbeddingCopyBytes`).
 * So their gathers and reads are costed per pass, but they add no resident bytes here ({@link draftResidentWeightBytes}).
 * At pp > 1 a block drafter's target features cross the stage boundaries ({@link ppBoundaryDraftFeatureVectors}). With the target's
 * token embedding off-device (`offload.embeddings`), each shared-embedding gather first fetches its rows from the
 * tier, like the target's (`draft.embed.row_fetch`, `draft.noise_embed.row_fetch`; docs/simulation-model.md §12
 * "Token embedding off-device").
 */

import { bytesPerElement, matrixStoredBytes } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { BlockDraftSpec, DraftSpec, LayerSpec, ModelSpec, MtpDraftSpec } from '../ir/model';
import type { AttentionOp, EmbeddingOp, Op } from '../ir/ops';
import { blockParams } from '../ir/params';
import {
  actBpe,
  attentionMath,
  elementwise,
  hpBpe,
  linearGemm,
  normOp,
  shardHeads,
  shardKvHeads,
  tpAllReduce,
} from './common';
import { embeddingFetchOp, embeddingFormat } from './embedding';
import { lmHeadFormat, samplingExchangeOps } from './head';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  lowerBlock,
  type LoweredLayer,
  type LoweringContext,
} from './lower';
import { lowerDenseFfn } from './mlp';
import { draftExpertPlacement } from './offload';

// ---------------------------------------------------------------------------------------------------------------
// Shared helpers

/** Bytes of one stored int32 token id (the selector's candidate ids). Unit: bytes. */
const TOKEN_ID_BYTES = 4;

/**
 * The lowering context of the drafter's own layers: the given step shape, without the verify's speculative
 * settings (the drafter keeps one state snapshot: only its final state is kept) and without the target's offload
 * placement, except (1) the routed-expert placement whose rule sets `includeDraft` (at most one), which puts the
 * drafter's own MoE experts off-device with that rule's tier, f and h (design doc §3.8), and (2) the token-embedding
 * placement, since the drafter gathers from the target's shared table. The expert rule's layer range is over the
 * TARGET's layers, so it is widened to every drafter layer (`[0, ∞)`), as a one-rule list; drafter blocks are lowered
 * with their own drafter layer (module) index, which the offload lowering needs to place experts.
 */
function draftContext(ctx: LoweringContext, qLen: number, kvLen: number): LoweringContext {
  const experts = draftExpertPlacement(ctx.offload);
  const embeddings = ctx.offload?.embeddings;
  const draftCtx: LoweringContext = { ...ctx, qLen, kvLen };
  delete draftCtx.speculative;
  delete draftCtx.offload;
  // The policy belongs to the target's n-gram tables, not the drafter's own weights.
  delete draftCtx.lookupTables;
  if (experts !== undefined || embeddings !== undefined) {
    draftCtx.offload = { prefetchAccuracy: ctx.offload?.prefetchAccuracy ?? 0 };
    if (experts !== undefined) {
      // Formula: draft expert range = [0, ∞) (every drafter MoE layer is offloaded with includeDraft)
      draftCtx.offload.routedExperts = [{ ...experts, layers: { from: 0, to: Infinity } }];
    }
    if (embeddings !== undefined) draftCtx.offload.embeddings = embeddings;
  }
  return draftCtx;
}

/** The ops with `draft.` prepended to their names (a target block lowered as part of the drafter). */
function asDraftOps(ops: Op[]): Op[] {
  return ops.map((op) => ({ ...op, name: `draft.${op.name}` }));
}

/** A gather of `tokens` rows of `width` elements from a `rows`-row table (EmbeddingOp; FLOP 0). */
function gatherOp(
  name: string,
  tokens: number,
  width: number,
  rows: number,
  ctx: LoweringContext,
): EmbeddingOp {
  return {
    kind: 'embedding',
    name,
    tokens,
    hidden: width,
    vocabShard: rows,
    weight: embeddingFormat(ctx),
    outputDtype: ctx.quant.activations,
  };
}

/**
 * The shared token embedding gathered for `tokens` rows (vocab-parallel: `ceil(V/tp)` rows per rank), then its TP
 * all-reduce of `tokens·h`. With the target's table off-device, the gather's row fetch
 * ({@link embeddingFetchOp}, `<name>.row_fetch`) goes first.
 */
function sharedEmbedOps(
  name: string,
  model: Pick<ModelSpec, 'hidden' | 'vocab'>,
  tokens: number,
  ctx: LoweringContext,
): Op[] {
  // Formula: [off-device: fetch T/tp rows] gather T rows of h from ceil(V/tp) rows; all-reduce T·h·reduceBpe when tp > 1
  // See docs/simulation-model.md §13 "Draft ops", §12 ("Token embedding off-device")
  const vocabShard = Math.ceil(model.vocab / ctx.parallel.tp);
  const fetch = embeddingFetchOp(name, tokens, model.hidden, ctx);
  return [
    ...(fetch === undefined ? [] : [fetch]),
    gatherOp(name, tokens, model.hidden, vocabShard, ctx),
    ...tpAllReduce(`${name}.tp_all_reduce`, tokens, model.hidden, ctx),
  ];
}

/** The shared LM head over `rows` rows: m = rows, n = ceil(V/tp), k = h, in the LM-head format. */
function sharedLmHeadOps(
  name: string,
  hidden: number,
  vocab: number,
  rows: number,
  ctx: LoweringContext,
): Op[] {
  // Formula: lm_head: m = rows, n = ceil(V/tp), k = h
  // See docs/simulation-model.md §13 "Draft ops"
  const format = {
    weights: lmHeadFormat(ctx),
    gemmInput: ctx.quant.lmHeadGemmInput ?? ctx.quant.activations,
  };
  const head = linearGemm(
    name,
    { m: rows, n: Math.ceil(vocab / ctx.parallel.tp), k: hidden },
    ctx,
    { format },
  );
  return [head, ...samplingExchangeOps(rows, ctx, 'draft.sample_all_gather')];
}

/** A residual add over `elems` activations: read 2·elems, write elems. */
function residualOp(name: string, elems: number, ctx: LoweringContext): Op {
  const act = actBpe(ctx);
  return elementwise(
    name,
    'residual_add',
    { elems, activationReadBytes: 2 * elems * act, activationWriteBytes: elems * act },
    ctx,
  );
}

/** An elementwise op that reads and writes `elems` activations once (RoPE, q/k-norm without weights). */
function readWriteOp(
  name: string,
  fn: 'rope' | 'qk_norm',
  elems: number,
  weightBytes: number,
  ctx: LoweringContext,
): Op {
  const act = actBpe(ctx);
  return elementwise(
    name,
    fn,
    { elems, activationReadBytes: elems * act, activationWriteBytes: elems * act, weightBytes },
    ctx,
  );
}

/** Throws unless `ctx` is a decode verify context of width k + 1 and m_ctx is a whole number in [1, k + 1]. */
function checkDecodeArgs(ctx: LoweringContext, contextRows: number, draftLength: number): void {
  if (ctx.phase !== 'decode') throw new Error('lowerDraftDecode needs a decode context');
  if (!Number.isInteger(draftLength) || draftLength < 1)
    throw new Error(`draft length k must be an integer ≥ 1, got ${draftLength}`);
  if (ctx.qLen !== draftLength + 1) {
    throw new Error(
      `lowerDraftDecode needs the verify context (qLen = k + 1 = ${draftLength + 1}), got qLen ${ctx.qLen}`,
    );
  }
  if (!Number.isInteger(contextRows) || contextRows < 1 || contextRows > draftLength + 1) {
    throw new Error(`m_ctx must be an integer in [1, k + 1], got ${contextRows}`);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Public entry points

/**
 * The drafter's layers for one decode step (design doc §3.3), to append to the last stage after the verify pass.
 * `ctx` is the VERIFY context: `qLen = k + 1`, `kvLen = N + k + 1`. `contextRows` = m_ctx = ⌈E[τ]⌉ (tokens);
 * `draftLength` = k (count). The MTP passes attend over `N + k + 1` keys each (documented simplification: the
 * recursive passes' growing context is not tracked).
 */
export function lowerDraftDecode(
  model: ModelSpec,
  draft: DraftSpec,
  ctx: LoweringContext,
  contextRows: number,
  draftLength: number,
): LoweredLayer[] {
  checkDecodeArgs(ctx, contextRows, draftLength);
  switch (draft.kind) {
    case 'mtp':
      return lowerMtpDecode(model, draft, ctx, contextRows, draftLength);
    case 'block_drafter':
      return lowerBlockDrafterDecode(draft, ctx, contextRows, draftLength);
    default: {
      const _never: never = draft;
      throw new Error(`unknown draft ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * The drafter's layers in the prefill step (design doc §3.4): what fills its context cache from the prompt.
 * `ctx` is the prefill context (`qLen` = the prompt chunk P). MTP: the module's ordinary forward over the prompt;
 * block drafter: only the context-K/V projections and a write-only cache op per layer (no attention, MLP or head).
 */
export function lowerDraftPrefill(
  model: ModelSpec,
  draft: DraftSpec,
  ctx: LoweringContext,
): LoweredLayer[] {
  if (ctx.phase !== 'prefill') throw new Error('lowerDraftPrefill needs a prefill context');
  switch (draft.kind) {
    case 'mtp':
      return lowerMtpPrefill(model, draft, ctx);
    case 'block_drafter':
      return lowerBlockDrafterPrefill(draft, ctx);
    default: {
      const _never: never = draft;
      throw new Error(`unknown draft ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Target-feature vectors per token, each `model.hidden` wide, that the PP boundary in front of target layer
 * `boundaryLayer` (the first layer of the next stage) must relay to the drafter on the last stage ON TOP OF the
 * activation send itself. Count (multiply by `T·h·bpe(act)` for bytes).
 *
 * A block drafter (DFlash 2, DSpark) conditions on the hidden states of the target layers `targetLayerIds`. An id i
 * is the OUTPUT of target layer i (DFlash `extract_context_feature`: `hidden_states[layer_id + 1]`, where HF's
 * `hidden_states[0]` is the embedding output; https://github.com/z-lab/dflash/blob/main/dflash/model.py#L37-L43), so
 * it is produced on the stage holding layer i and must cross every boundary from there to the last stage. The output
 * of the boundary's own last layer (`boundaryLayer − 1`) is the state the activation send already carries (the
 * `N_live` vectors the next layer reads, from which that output follows), so it adds nothing there; it becomes an
 * extra vector only at the following boundaries:
 *   F = |{ i ∈ targetLayerIds : i < boundaryLayer − 1 }|
 * Ids at or past the boundary's last layer need no extra vector here (ids ≥ L count as last-stage features).
 * MTP: 0, since it reads only the last stage's final hidden state.
 *
 * The features of all k + 1 verify rows (prefill: all P rows) ride on the existing activation send, one message per
 * boundary. An earlier stage cannot know which rows the last stage will accept, so sending every verify row needs no
 * round trip after acceptance and adds no α. Per-feature-layer sends of only the m_ctx accepted rows (design doc
 * §3.6) would carry fewer bytes but add one α per feature layer and boundary plus that round trip.
 * See docs/simulation-model.md §13 "Pipeline parallelism".
 */
export function ppBoundaryDraftFeatureVectors(draft: DraftSpec, boundaryLayer: number): number {
  if (draft.kind !== 'block_drafter') return 0;
  // Formula: F = |{ i ∈ targetLayerIds : i < boundaryLayer − 1 }|   (layer boundaryLayer − 1's output is the send itself)
  return draft.targetLayerIds.filter((i) => i < boundaryLayer - 1).length;
}

/**
 * Weight bytes the drafter keeps resident on ONE device of the last stage (after TP sharding), NOT counting the
 * shared embedding and LM head (the head is on the last stage already; the embedding is on the device at pp = 1, and
 * its pp > 1 copy is `memory.ts: draftEmbeddingCopyBytes`). MTP: each module's layer (all local experts of a
 * MoE layer), `eh_proj` and its three norms; block drafter: its layers, `fc`, norms, DFlash 2 conv kernels and
 * projections and selector (both codebooks whole), DSpark Markov `W1` and `W2`. Unit: bytes.
 */
export function draftResidentWeightBytes(
  model: ModelSpec,
  draft: DraftSpec,
  ctx: LoweringContext,
): number {
  const oneToken = draftContext({ ...ctx, phase: 'decode', seqs: 1 }, 1, 2);
  const weightBytes = (ops: Op[]) => ops.reduce((sum, op) => sum + opBytes(op).weightBytes, 0);
  const layerBytes = (layer: LayerSpec, j: number) =>
    layer.blocks.reduce((sum, b) => sum + blockResidentWeightBytes(b, oneToken, j), 0);
  switch (draft.kind) {
    case 'mtp': {
      // Formula: W_draft = Σ_modules (W_layer + W_eh_proj + 3·h·bpe(hp)); embedding and LM head shared
      // See docs/simulation-model.md §13 "Draft memory"
      const glue =
        weightBytes(
          mtpInputOps(model, draft, 1, oneToken).filter((op) => op.kind !== 'embedding'),
        ) + weightBytes([mtpFinalNorm(model, 1, oneToken)]);
      return draft.layers.reduce((sum, layer, j) => sum + layerBytes(layer, j) + glue, 0);
    }
    case 'block_drafter': {
      // Formula: W_draft = W_fc + W_norms + Σ_layers W_layer + W_selector (2·V·r + h·r) + W_markov (V·r + r·V)
      // Shared embedding and LM head excluded; the codebooks and W1 are counted whole, not by gathered rows.
      // See docs/simulation-model.md §13 "Draft memory"
      const shape = blockShape(draft, oneToken, 1, 1, 0);
      const input = blockInputOps(draft, shape, oneToken).filter(
        (op) => op.name !== 'draft.noise_embed',
      );
      const head = blockHeadOps(draft, shape, oneToken).filter(
        (op) => op.kind !== 'embedding' && op.name !== 'draft.lm_head',
      );
      const layers =
        draft.moeLayers !== undefined
          ? draft.moeLayers.reduce((sum, layer, j) => sum + layerBytes(layer, j), 0)
          : draft.layers * weightBytes(blockDenseLayerOps(draft, shape, oneToken));
      const fmt = embeddingFormat(ctx);
      const sel = draft.selector;
      const codebooks = sel === undefined ? 0 : 2 * matrixStoredBytes(fmt, draft.vocab, sel.rank);
      const markovW1 =
        draft.markovRank === undefined ? 0 : matrixStoredBytes(fmt, draft.vocab, draft.markovRank);
      return weightBytes(input) + layers + weightBytes(head) + codebooks + markovW1;
    }
    default: {
      const _never: never = draft;
      throw new Error(`unknown draft ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * The drafter's own per-sequence cache on ONE device for a context of `contextTokens`: MTP, every module's layer
 * state over the full context (KV, latent, recurrent; via {@link blockSequenceStateBytes}); block drafter,
 * `layers · min(ctx, W) · 2·KVl·hd · bpe(kv)` (its MoE layers: their own block state). Unit: bytes per sequence.
 */
export function draftSequenceStateBytes(
  draft: DraftSpec,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  const layerState = (layer: LayerSpec) =>
    layer.blocks.reduce((sum, b) => sum + blockSequenceStateBytes(b, ctx, contextTokens), 0);
  switch (draft.kind) {
    case 'mtp':
      // Formula: S_draft = Σ_modules S_layer(ctx)
      // See docs/simulation-model.md §13 "Draft memory"
      return draft.layers.reduce((sum, layer) => sum + layerState(layer), 0);
    case 'block_drafter': {
      if (draft.moeLayers !== undefined)
        return draft.moeLayers.reduce((sum, layer) => sum + layerState(layer), 0);
      const KVl = shardKvHeads(draft.kvHeads, ctx.parallel.tp, 'draft kvHeads');
      const cached =
        draft.windowTokens === undefined
          ? contextTokens
          : Math.min(contextTokens, draft.windowTokens);
      // Formula: S_draft = L · min(ctx, W) · 2·KVl·hd · bpe(kv)
      // See docs/simulation-model.md §13 "Draft memory"
      return draft.layers * cached * 2 * KVl * draft.headDim * bytesPerElement(ctx.quant.kvCache);
    }
    default: {
      const _never: never = draft;
      throw new Error(`unknown draft ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Logical parameters of a block drafter, EXCLUDING the shared embedding and LM head (design doc §3.3 U-10).
 * Dense layer: `q h·H·hd + k, v h·KV·hd each + o H·hd·h + q/k norms 2·hd + MLP (gated ? 3 : 2)·h·f + norms 2·h`,
 * DFlash 2 `+ 2 convs · (base 2·K·h + kernel projection h·2·K·h/g)`; plus `fc |ids|·h·h`, `hidden_norm` and `norm`
 * (2·h), selector `2·V·r + h·r`, Markov `2·V·r`. MoE layers: their block params. The DSpark confidence head and
 * mask embedding are not modelled. Unit: params.
 */
export function blockDrafterParams(spec: BlockDraftSpec): number {
  const h = spec.hidden;
  const hd = spec.headDim;
  // Formula: per conv: base 2·K·h + kernel_projection h·(2·K·h/g)
  const conv =
    spec.conv === undefined
      ? 0
      : 2 * spec.conv.kernel * h + h * ((2 * spec.conv.kernel * h) / spec.conv.groupSize);
  // Formula: P_layer = h·H·hd + 2·h·KV·hd + H·hd·h + 2·hd + (gated ? 3 : 2)·h·f + 2·h + 2·P_conv
  const denseLayer =
    h * spec.heads * hd +
    2 * h * spec.kvHeads * hd +
    spec.heads * hd * h +
    2 * hd +
    (spec.gated ? 3 : 2) * h * spec.intermediate +
    2 * h +
    2 * conv;
  const layers =
    spec.moeLayers !== undefined
      ? spec.moeLayers.reduce(
          (sum, layer) => sum + layer.blocks.reduce((p, b) => p + blockParams(b), 0),
          0,
        )
      : spec.layers * denseLayer;
  const selector =
    spec.selector === undefined ? 0 : 2 * spec.vocab * spec.selector.rank + h * spec.selector.rank;
  const markov = spec.markovRank === undefined ? 0 : 2 * spec.vocab * spec.markovRank;
  // Formula: P = Σ P_layer + |ids|·h·h (fc) + 2·h (hidden_norm, norm) + P_selector + P_markov
  return layers + spec.targetLayerIds.length * h * h + 2 * h + selector + markov;
}

// ---------------------------------------------------------------------------------------------------------------
// MTP

/**
 * One MTP module's input ops over `rows` rows per sequence: the shared embedding of the next tokens (+ its TP
 * all-reduce), then with `hiddenProj` the `enorm`, `hnorm` RMSNorms over `b·rows·h` each and `eh_proj`
 * (m b·rows, n h, k 2h; replicated on every TP rank).
 */
function mtpInputOps(
  model: ModelSpec,
  draft: MtpDraftSpec,
  rows: number,
  ctx: LoweringContext,
): Op[] {
  const T = ctx.seqs * rows;
  const h = model.hidden;
  const ops = sharedEmbedOps('draft.embed', model, T, ctx);
  if (draft.hiddenProj) {
    // Formula: enorm, hnorm over T·h each; eh_proj: m = T, n = h, k = 2h
    // See docs/simulation-model.md §13 "Draft ops"
    ops.push(normOp('draft.enorm', 'rmsnorm', T, h, ctx));
    ops.push(normOp('draft.hnorm', 'rmsnorm', T, h, ctx));
    ops.push(linearGemm('draft.eh_proj', { m: T, n: h, k: 2 * h }, ctx, { module: 'attention' }));
  }
  return ops;
}

/** The MTP module's own final RMSNorm (DeepSeek `shared_head.norm`) over `rows` rows of h. */
function mtpFinalNorm(model: ModelSpec, rows: number, ctx: LoweringContext): Op {
  return normOp('draft.final_norm', 'rmsnorm', rows, model.hidden, ctx);
}

/** The ops of one MTP pass: module input, the module's layer over `qLen` rows, and the head over the last position. */
function mtpPassOps(
  model: ModelSpec,
  draft: MtpDraftSpec,
  module: LayerSpec,
  moduleIndex: number,
  layerCtx: LoweringContext,
  headRows: number,
): Op[] {
  // Formula: pass = embed(T) → [enorm, hnorm, eh_proj](T) → layer(qLen) → final_norm, lm_head over headRows
  // See docs/simulation-model.md §13 "Draft ops"
  return [
    ...mtpInputOps(model, draft, layerCtx.qLen, layerCtx),
    ...asDraftOps(module.blocks.flatMap((block) => lowerBlock(block, layerCtx, moduleIndex))),
    mtpFinalNorm(model, headRows, layerCtx),
    ...sharedLmHeadOps('draft.lm_head', model.hidden, model.vocab, headRows, layerCtx),
  ];
}

/**
 * MTP decode (design doc §3.3): k passes. Pass p (1-based) runs module `min(p, L)`; p ≤ L catches that module's
 * cache up over the m committed rows (`qLen m`, KV write of m rows, causal over `past = N + k + 1 − m`), p > L is
 * one recursive row. Every pass ends with `final_norm` and `lm_head` on the last position only (m = b). Layers
 * `draft.<p−1>`. The tentative rows a pass writes are overwritten by the next step's catch-up (nothing charged
 * back).
 */
function lowerMtpDecode(
  model: ModelSpec,
  draft: MtpDraftSpec,
  ctx: LoweringContext,
  contextRows: number,
  k: number,
): LoweredLayer[] {
  const L = draft.layers.length;
  if (L === 0) throw new Error('an MTP draft needs at least one module layer');
  const layers: LoweredLayer[] = [];
  for (let p = 1; p <= k; p++) {
    // Formula: module = min(p, L); qLen = p ≤ L ? m_ctx : 1; kvLen = N + k + 1 (all passes)
    const qLen = p <= L ? contextRows : 1;
    const layerCtx = draftContext(ctx, qLen, ctx.kvLen);
    layers.push({
      name: `draft.${p - 1}`,
      layerIndex: null,
      ops: mtpPassOps(
        model,
        draft,
        draft.layers[Math.min(p, L) - 1],
        Math.min(p, L) - 1,
        layerCtx,
        ctx.seqs,
      ),
    });
  }
  return layers;
}

/**
 * MTP prefill (design doc §3.4): every module's ordinary forward over the prompt (embed P rows, norms, `eh_proj`,
 * the layer with `qLen = P`, writing its KV), each module consuming the previous one's outputs; one
 * `final_norm` + `lm_head` per sequence (the first draft token) after the last module.
 */
function lowerMtpPrefill(
  model: ModelSpec,
  draft: MtpDraftSpec,
  ctx: LoweringContext,
): LoweredLayer[] {
  const layerCtx = draftContext(ctx, ctx.qLen, ctx.kvLen);
  const layers: LoweredLayer[] = draft.layers.map((module, j) => ({
    name: `draft.${j}`,
    layerIndex: null,
    ops: [
      ...mtpInputOps(model, draft, ctx.qLen, layerCtx),
      ...asDraftOps(module.blocks.flatMap((block) => lowerBlock(block, layerCtx, j))),
    ],
  }));
  const last = layers.at(-1);
  if (last === undefined) throw new Error('an MTP draft needs at least one module layer');
  // Formula: head over Tout = seqs (last prompt position)
  last.ops.push(
    mtpFinalNorm(model, ctx.seqs, layerCtx),
    ...sharedLmHeadOps('draft.lm_head', model.hidden, model.vocab, ctx.seqs, layerCtx),
  );
  return layers;
}

// ---------------------------------------------------------------------------------------------------------------
// Block drafter (DFlash 2 / DSpark)

/**
 * Row counts of one block-drafter decode step. The block a pass embeds differs by method, per the reference code:
 * DFlash 2 embeds the anchor and k masks (B = k + 1) and takes logits at the k masks (`draft_hidden[:, 1−B:]`);
 * DSpark embeds the anchor and k − 1 masks (B = k) and takes logits at all k positions, each predicting the next
 * token (DeepSpec `build_dspark_proposal`: `block_hidden[:, :block_size]`, then `[anchor, drafts]` verified).
 */
interface BlockShape {
  /** Context rows b·m (the committed tokens' target features). Unit: tokens. */
  ctxRows: number;
  /** Block rows b·B (B = k + 1 for DFlash 2, k for DSpark). Unit: tokens. */
  blockRows: number;
  /** Rows with logits b·k (one per draft token). Unit: tokens. */
  headRows: number;
  /** Draft tokens per step k. Count. */
  k: number;
  /** Block positions per sequence B. Unit: tokens. */
  blockLen: number;
  /** Context tokens per sequence m. Unit: tokens. */
  m: number;
  /** Keys each block query sees: `min(N, W) + m + B` (or `N + m + B` without a window). Unit: tokens. */
  kvLen: number;
}

/**
 * Block positions per sequence of one draft pass: DFlash 2 `k + 1` (anchor + k masks), DSpark `k` (anchor + k − 1
 * masks). Unit: tokens.
 */
export function draftBlockLength(draft: BlockDraftSpec, k: number): number {
  // Formula: B = dspark ? k : k + 1
  return draft.method === 'dspark' ? k : k + 1;
}

/** The block shape for `b = ctx.seqs`, m context rows, k drafts and `past` cached tokens N. */
function blockShape(
  draft: BlockDraftSpec,
  ctx: LoweringContext,
  m: number,
  k: number,
  past: number,
): BlockShape {
  const b = ctx.seqs;
  const cachedPast = draft.windowTokens === undefined ? past : Math.min(past, draft.windowTokens);
  const B = draftBlockLength(draft, k);
  // Formula: kvLen = min(N, W) + m + B
  return {
    ctxRows: b * m,
    blockRows: b * B,
    headRows: b * k,
    k,
    blockLen: B,
    m,
    kvLen: cachedPast + m + B,
  };
}

/**
 * Block-drafter input (`draft.input`): `draft.fc` (m b·m, n h, k |ids|·h) + `draft.hidden_norm` over the context
 * features, then `draft.noise_embed`, the shared embedding of the b·B block tokens (+ its TP all-reduce).
 */
function blockInputOps(draft: BlockDraftSpec, s: BlockShape, ctx: LoweringContext): Op[] {
  const h = draft.hidden;
  // Formula: fc: m = b·m_ctx, n = h, k = |targetLayerIds|·h; hidden_norm over b·m_ctx·h
  // See docs/simulation-model.md §13 "Draft ops"
  return [
    linearGemm('draft.fc', { m: s.ctxRows, n: h, k: draft.targetLayerIds.length * h }, ctx, {
      module: 'attention',
    }),
    normOp('draft.hidden_norm', 'rmsnorm', s.ctxRows, h, ctx),
    ...sharedEmbedOps('draft.noise_embed', { hidden: h, vocab: draft.vocab }, s.blockRows, ctx),
  ];
}

/**
 * DFlash 2 `GroupedDynamicCausalConv` around a sublayer (`prefix` `draft.attn_conv` / `draft.mlp_conv`):
 * `kernel_proj` GEMM (m b·B, n 2·K·h/g, k h; module `residual_mixer`) and the `prepare` convolution before the
 * sublayer; the `finish` convolution after it. Each convolution is 4·K FLOP per element over b·B·h (per tap:
 * base multiply-add + dynamic multiply-add, as `_grouped_dynamic_convolve` computes both in `prepare` and `finish`),
 * reads the hidden rows and its half of the dynamic kernel (b·B·K·h/g), and its base taps K·h·bpe(hp). The block
 * conv is zero-padded at the anchor: no rolling state.
 */
function dynamicConvOps(
  prefix: string,
  draft: BlockDraftSpec,
  s: BlockShape,
  ctx: LoweringContext,
): { before: Op[]; after: Op[] } {
  const conv = draft.conv;
  if (conv === undefined) return { before: [], after: [] };
  const h = draft.hidden;
  const K = conv.kernel;
  const act = actBpe(ctx);
  const groups = h / conv.groupSize;
  const elems = s.blockRows * h;
  // Formula: kernel_proj: m = b·B, n = 2·K·h/g, k = h; conv: elems = b·B·h, 4·K FLOP/elem,
  //          read elems + b·B·K·h/g, write elems, taps K·h·bpe(hp)
  // See docs/simulation-model.md §13 "Draft ops"
  const convOp = (name: string) =>
    elementwise(
      name,
      'causal_conv1d',
      {
        elems,
        flopsPerElem: 4 * K,
        activationReadBytes: (elems + s.blockRows * K * groups) * act,
        activationWriteBytes: elems * act,
        weightBytes: K * h * hpBpe(ctx),
      },
      ctx,
    );
  return {
    before: [
      linearGemm(`${prefix}.kernel_proj`, { m: s.blockRows, n: 2 * K * groups, k: h }, ctx, {
        module: 'residual_mixer',
      }),
      convOp(`${prefix}.prepare`),
    ],
    after: [convOp(`${prefix}.finish`)],
  };
}

/**
 * One dense block-drafter layer (design doc §3.3 step 3): input_norm → [attn_conv] → q_proj (block rows) →
 * k_proj, v_proj (ONE GEMM each over context + block rows, weights read once) → qk_norm → rope → attn.core
 * (non-causal, window W, keeps only the m context rows: `kvWriteTokens = m`) → [attn_conv finish] → o_proj →
 * [TP all-reduce] → residual → post_norm → [mlp_conv] → MLP → [TP all-reduce] → [mlp_conv finish] → residual.
 */
function blockDenseLayerOps(draft: BlockDraftSpec, s: BlockShape, ctx: LoweringContext): Op[] {
  const tp = ctx.parallel.tp;
  const h = draft.hidden;
  const hd = draft.headDim;
  const Hl = shardHeads(draft.heads, tp, 'draft heads');
  const KVl = shardKvHeads(draft.kvHeads, tp, 'draft kvHeads');
  const kvRows = s.ctxRows + s.blockRows;
  const attnConv = dynamicConvOps('draft.attn_conv', draft, s, ctx);
  const mlpConv = dynamicConvOps('draft.mlp_conv', draft, s, ctx);
  const attn = { module: 'attention' } as const;
  // Formula: qk_norm and rope over (b·B·Hl + b·(m+B)·KVl)·hd elements
  const qkElems = (s.blockRows * Hl + kvRows * KVl) * hd;
  const kvStream = { format: ctx.quant.kvCache, elemsPerToken: 2 * KVl * hd };
  // Formula: attn.core: qLen B, kvLen min(N, W) + m + B, causal false, kvWriteTokens = m
  // See docs/simulation-model.md §13 "Draft ops"
  const core: AttentionOp = {
    kind: 'attention',
    name: 'draft.attn.core',
    seqs: ctx.seqs,
    heads: Hl,
    kvHeads: KVl,
    qkHeadDim: hd,
    vHeadDim: hd,
    qLen: s.blockLen,
    kvLen: s.kvLen,
    causal: false,
    kvSelection:
      draft.windowTokens === undefined
        ? { kind: 'full' }
        : { kind: 'window', windowTokens: draft.windowTokens },
    kvRead: kvStream,
    kvWrite: kvStream,
    kvWriteTokens: s.m,
    ...attentionMath(ctx),
    activationDtype: ctx.quant.activations,
  };
  const mlp = {
    hidden: h,
    intermediate: draft.intermediate,
    gated: draft.gated,
    activation: draft.activation,
    bias: false,
  };
  // Formula: q_proj: m = b·B, n = Hl·hd, k = h; k_proj, v_proj: m = b·(m+B), n = KVl·hd, k = h; o_proj: n = h, k = Hl·hd
  return [
    normOp('draft.input_norm', 'rmsnorm', s.blockRows, h, ctx),
    ...attnConv.before,
    linearGemm('draft.q_proj', { m: s.blockRows, n: Hl * hd, k: h }, ctx, attn),
    linearGemm('draft.k_proj', { m: kvRows, n: KVl * hd, k: h }, ctx, attn),
    linearGemm('draft.v_proj', { m: kvRows, n: KVl * hd, k: h }, ctx, attn),
    readWriteOp('draft.qk_norm', 'qk_norm', qkElems, 2 * hd * hpBpe(ctx), ctx),
    readWriteOp('draft.rope', 'rope', qkElems, 0, ctx),
    core,
    ...attnConv.after,
    linearGemm('draft.o_proj', { m: s.blockRows, n: h, k: Hl * hd }, ctx, attn),
    ...tpAllReduce('draft.attn.tp_all_reduce', s.blockRows, h, ctx),
    residualOp('draft.attn_residual', s.blockRows * h, ctx),
    normOp('draft.post_norm', 'rmsnorm', s.blockRows, h, ctx),
    ...mlpConv.before,
    ...lowerDenseFfn('draft.mlp', mlp, s.blockRows, tp, 'dense_mlp', ctx),
    ...tpAllReduce('draft.mlp.tp_all_reduce', s.blockRows, h, ctx),
    ...mlpConv.after,
    residualOp('draft.mlp_residual', s.blockRows * h, ctx),
  ];
}

/**
 * One DeepSeek-V4-style MoE drafter layer (`moeLayers`, window-only CSA + MoE + mHC), lowered through
 * {@link lowerBlock} with the B block rows (`qLen B`, `kvLen min(N, W) + m + B`). Two adjustments make it
 * a block-drafter layer: the window K = V projection `csa.wkv` also projects the m context rows (m = b·(m + B),
 * one GEMM), and the attention is non-causal and keeps only the m context rows (`kvWriteTokens = m`). The kv norm
 * and RoPE of the context rows (< h elements per row) are not added. Only `csa_attention` mixers are supported.
 */
function lowerMoeDrafterLayer(
  layer: LayerSpec,
  layerIndex: number,
  s: BlockShape,
  ctx: LoweringContext,
): Op[] {
  const layerCtx = draftContext(ctx, s.blockLen, s.kvLen);
  return layer.blocks.flatMap((block) => {
    if (
      block.kind === 'gqa_attention' ||
      block.kind === 'mla_attention' ||
      block.kind === 'linear_attention'
    ) {
      throw new Error(
        `MoE drafter layer ${layer.name}: ${block.kind} mixers are not supported (csa_attention only)`,
      );
    }
    return asDraftOps(lowerBlock(block, layerCtx, layerIndex)).map((op): Op => {
      // Formula: wkv: m = b·(m_ctx + B) (context + block rows); attn.core: causal false, kvWriteTokens = m_ctx
      if (op.kind === 'gemm' && op.name === 'draft.csa.wkv')
        return { ...op, m: s.ctxRows + s.blockRows };
      if (op.kind === 'attention') return { ...op, causal: false, kvWriteTokens: s.m };
      return op;
    });
  });
}

/**
 * Block-drafter head (`draft.head`): `draft.norm` over the block, the shared `draft.lm_head` over the b·k mask
 * positions, then DFlash 2's candidate selector or DSpark's Markov head (design doc §3.3 steps 4–6). Selector:
 * `selector.topk` over the b·k·V/tp logits (writes topK ids + scores per row), `selector.hidden_proj` (m b·k, n r,
 * k h), `selector.codebooks` (b·k·(1 + topK) rows of r: predecessor + candidates), `selector.score`
 * (b·k·topK·r multiply-adds). Markov: per draft position a `markov.W1` gather (b rows of r), `markov.W2`
 * (m b, n V/tp, k r) and the bias add over b·V/tp. Their sequential walks are latency, not costed; the DSpark
 * confidence head is not costed. Selector and Markov weights use the embeddings format.
 */
function blockHeadOps(draft: BlockDraftSpec, s: BlockShape, ctx: LoweringContext): Op[] {
  const h = draft.hidden;
  const act = actBpe(ctx);
  const Vl = Math.ceil(draft.vocab / ctx.parallel.tp);
  const emb = { format: { weights: embeddingFormat(ctx) } };
  const ops: Op[] = [
    normOp('draft.norm', 'rmsnorm', s.blockRows, h, ctx),
    ...sharedLmHeadOps('draft.lm_head', h, draft.vocab, s.headRows, ctx),
  ];
  const sel = draft.selector;
  if (sel !== undefined) {
    const logits = s.headRows * Vl;
    const scored = s.headRows * sel.topK * sel.rank;
    // Formula: topk over b·k·V/tp logits (5 FLOP/elem), write b·k·topK·(4 + bpe(act)); hidden_proj m b·k, n r, k h;
    //          codebooks b·k·(1 + topK) rows of r; score b·k·topK·r elems, 2 FLOP/elem
    // See docs/simulation-model.md §13 "Draft ops"
    ops.push(
      elementwise(
        'draft.selector.topk',
        'router_topk',
        {
          elems: logits,
          activationReadBytes: logits * act,
          activationWriteBytes: s.headRows * sel.topK * (TOKEN_ID_BYTES + act),
        },
        ctx,
      ),
      linearGemm('draft.selector.hidden_proj', { m: s.headRows, n: sel.rank, k: h }, ctx, emb),
      gatherOp('draft.selector.codebooks', s.headRows * (1 + sel.topK), sel.rank, draft.vocab, ctx),
      elementwise(
        'draft.selector.score',
        'candidate_score',
        {
          elems: scored,
          activationReadBytes: (scored + 2 * s.headRows * sel.rank) * act,
          activationWriteBytes: s.headRows * sel.topK * act,
        },
        ctx,
      ),
    );
  }
  const r = draft.markovRank;
  if (r !== undefined) {
    const b = ctx.seqs;
    // Formula: per position j = 1..k: W1 gather b rows of r; W2: m = b, n = V/tp, k = r; add over b·V/tp
    // See docs/simulation-model.md §13 "Draft ops"
    for (let j = 0; j < s.k; j++) {
      ops.push(
        gatherOp('draft.markov.W1', b, r, draft.vocab, ctx),
        linearGemm('draft.markov.W2', { m: b, n: Vl, k: r }, ctx, emb),
        residualOp('draft.markov.add', b * Vl, ctx),
      );
    }
  }
  return ops;
}

/**
 * Block-drafter decode (design doc §3.3): `draft.input`, then `draft.<i>` per drafter layer, then `draft.head`.
 */
function lowerBlockDrafterDecode(
  draft: BlockDraftSpec,
  ctx: LoweringContext,
  contextRows: number,
  k: number,
): LoweredLayer[] {
  if (k + 1 > draft.blockSize)
    throw new Error(`draft length k = ${k} needs k + 1 ≤ blockSize (${draft.blockSize})`);
  const dctx = draftContext(ctx, ctx.qLen, ctx.kvLen);
  // Formula: N = kvLen − qLen of the verify context
  const s = blockShape(draft, dctx, contextRows, k, ctx.kvLen - ctx.qLen);
  const layerOps =
    draft.moeLayers !== undefined
      ? draft.moeLayers.map((layer, j) => lowerMoeDrafterLayer(layer, j, s, dctx))
      : Array.from({ length: draft.layers }, () => blockDenseLayerOps(draft, s, dctx));
  return [
    { name: 'draft.input', layerIndex: null, ops: blockInputOps(draft, s, dctx) },
    ...layerOps.map((ops, i) => ({ name: `draft.${i}`, layerIndex: null, ops })),
    { name: 'draft.head', layerIndex: null, ops: blockHeadOps(draft, s, dctx) },
  ];
}

/**
 * The write-only cache op of a block-drafter prefill: `elems = b·min(P, W)·E` stored elements (E per token),
 * 1 FLOP/elem (the cast), reading the projected rows (`elems·bpe(act)`), writing `elems·bpe(kv)` as KV-cache
 * bytes, no activation write.
 */
function kvCacheWriteOp(storedTokens: number, elemsPerToken: number, ctx: LoweringContext): Op {
  const elems = storedTokens * elemsPerToken;
  // Formula: elems = b·min(P, W)·E; read elems·bpe(act); kvWrite elems·bpe(kv); 1 FLOP/elem
  // See docs/simulation-model.md §13 "Draft-cache prefill"
  return elementwise(
    'draft.kv_cache_write',
    'kv_cache_write',
    {
      elems,
      activationReadBytes: elems * actBpe(ctx),
      activationWriteBytes: 0,
      kvWriteBytes: elems * bytesPerElement(ctx.quant.kvCache),
    },
    ctx,
  );
}

/**
 * Block-drafter prefill (design doc §3.4): `draft.input` = `fc` (m b·P) + `hidden_norm`; per layer `draft.<i>` =
 * `k_proj`, `v_proj` (m b·P), `qk_norm` and `rope` over the K heads (b·P·KVl·hd), and the write-only
 * `kv_cache_write` of the last min(P, W) tokens. MoE (CSA) drafter layers: `csa.wkv` (m b·P, n hd), `csa.kv_norm`,
 * `csa.k_rope` (b·P·ropeDim) and the cache write of `hd` per token. No attention, MLP, head, selector or Markov op.
 */
function lowerBlockDrafterPrefill(draft: BlockDraftSpec, ctx: LoweringContext): LoweredLayer[] {
  const dctx = draftContext(ctx, ctx.qLen, ctx.kvLen);
  const h = draft.hidden;
  const P = dctx.qLen;
  const rows = dctx.seqs * P;
  // Formula: stored tokens = b·min(P, W)
  const stored =
    dctx.seqs * (draft.windowTokens === undefined ? P : Math.min(P, draft.windowTokens));
  const attn = { module: 'attention' } as const;
  const input: Op[] = [
    // Formula: fc: m = b·P, n = h, k = |ids|·h; hidden_norm over b·P·h
    linearGemm('draft.fc', { m: rows, n: h, k: draft.targetLayerIds.length * h }, dctx, attn),
    normOp('draft.hidden_norm', 'rmsnorm', rows, h, dctx),
  ];
  const denseLayer = (): Op[] => {
    const hd = draft.headDim;
    const KVl = shardKvHeads(draft.kvHeads, dctx.parallel.tp, 'draft kvHeads');
    const kElems = rows * KVl * hd;
    // Formula: k_proj, v_proj: m = b·P, n = KVl·hd, k = h; k_norm, rope over b·P·KVl·hd; write b·min(P, W)·2·KVl·hd
    return [
      linearGemm('draft.k_proj', { m: rows, n: KVl * hd, k: h }, dctx, attn),
      linearGemm('draft.v_proj', { m: rows, n: KVl * hd, k: h }, dctx, attn),
      readWriteOp('draft.qk_norm', 'qk_norm', kElems, hd * hpBpe(dctx), dctx),
      readWriteOp('draft.rope', 'rope', kElems, 0, dctx),
      kvCacheWriteOp(stored, 2 * KVl * hd, dctx),
    ];
  };
  const csaLayer = (layer: LayerSpec): Op[] => {
    const csa = layer.blocks.find((b) => b.kind === 'csa_attention');
    if (csa === undefined || csa.kind !== 'csa_attention') {
      throw new Error(
        `MoE drafter layer ${layer.name}: no csa_attention mixer (csa_attention only)`,
      );
    }
    const hd = csa.headDim;
    const csaStored = dctx.seqs * Math.min(P, csa.windowTokens);
    // Formula: wkv: m = b·P, n = hd, k = h; kv_norm over b·P·hd; k_rope over b·P·dr; write b·min(P, W)·hd
    return [
      linearGemm('draft.csa.wkv', { m: rows, n: hd, k: h }, dctx, attn),
      normOp('draft.csa.kv_norm', 'rmsnorm', rows, hd, dctx),
      readWriteOp('draft.csa.k_rope', 'rope', rows * csa.ropeDim, 0, dctx),
      kvCacheWriteOp(csaStored, hd, dctx),
    ];
  };
  const layerOps =
    draft.moeLayers !== undefined
      ? draft.moeLayers.map(csaLayer)
      : Array.from({ length: draft.layers }, denseLayer);
  return [
    { name: 'draft.input', layerIndex: null, ops: input },
    ...layerOps.map((ops, i) => ({ name: `draft.${i}`, layerIndex: null, ops })),
  ];
}
