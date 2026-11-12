/**
 * Lowering: model spec + workload step + quantization + parallelism → the ops ONE device runs per stage.
 *
 * Conventions (docs/simulation-model.md §5): `T = seqs · qLen` tokens per device step; TP shards heads and
 * intermediate widths (`Hl = heads / tp`; KV heads are replicated when `kvHeads < tp`); PP splits layers into
 * contiguous stages ({@link stageLayerRange}); DP replicas are represented by one lowering, with cross-replica
 * exchanges when EP spans them or lookup-table sharing is enabled.
 */

import { opBytes } from '../ir/metrics';
import type { CollectiveOp, Op } from '../ir/ops';
import type { BlockSpec, DraftSpec, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { actBpe, stepTokens } from './common';
import { csaSequenceStateBytes, lowerCsa } from './csa';
import { lowerEmbedding } from './embedding';
import { gqaSequenceStateBytes, lowerGqa } from './gqa';
import { lowerHead } from './head';
import { lowerHcPost, lowerHcPre } from './hyperConnection';
import { linearAttentionSequenceStateBytes, lowerLinearAttention } from './linearAttention';
import { lowerMla, mlaSequenceStateBytes } from './mla';
import { lowerMlp } from './mlp';
import { lowerMoe, moeResidentWeightBytes } from './moe';
import { lowerNgram, ngramResidentWeightBytes, ngramSequenceStateBytes } from './ngram';
import { lowerNorm, lowerResidual, lowerResidualMix } from './norm';
import { lowerShortConv, shortConvSequenceStateBytes } from './shortConv';

/** One phase step of one microbatch, as seen by the lowering. */
export interface LoweringContext {
  /** Which phase is lowered (prefill: the whole prompt; decode: one new token per sequence). */
  phase: 'prefill' | 'decode';
  /** Sequences in the microbatch (b_mb), per replica. Count. */
  seqs: number;
  /** New query tokens per sequence in this step (prefill: prompt length; decode: 1). Unit: tokens. */
  qLen: number;
  /** Keys visible to the last query (past + qLen). Unit: tokens. */
  kvLen: number;
  /** Numeric formats. */
  quant: QuantConfig;
  /** Parallelism degrees. */
  parallel: ParallelConfig;
  /** Resolved target lookup-table sharing; absent = the existing per-replica path. */
  lookupTables?: {
    /** Head-layout tables may read peer HBM directly (TP1 in a supported fully connected NVLink domain). */
    peerReads: boolean;
  };
  /**
   * Resolved off-device placement (deployment `offload`, design doc §2.1); absent = everything on the device, and
   * lowering emits no `TransferOp` (identity).
   */
  offload?: OffloadLowering;
  /**
   * Resolved speculative-decoding step (deployment `speculative`, design doc §3.3); absent = plain decode (`qLen 1`,
   * no `draft.*` layers, no snapshot overrides).
   */
  speculative?: SpeculativeLowering;
  /**
   * Prefill under chunked pipeline prefill only (docs/simulation-model.md §7 "Chunked pipeline prefill"): of this
   * step's `kvLen − qLen` past tokens, the first `prefixCacheTokens` are the prefix-cache hit (`cached = floor(P·h)`,
   * placed by `prefixDeviceFraction`) and the rest are the earlier chunks of the same prefill, live KV placed by
   * `deviceFraction` as the memory fit keeps it. KV-cache offload fetches the off-device part of both
   * (`lower/kvOffload.ts: kvMisses`). Absent = every past token is prefix-cache hit (the whole-prompt prefill, where
   * `kvLen − qLen = cached`). Unit: tokens.
   */
  prefixCacheTokens?: number;
}

/** One resolved routed-expert placement rule (`RoutedExpertRule` with every default filled in). */
export interface ExpertPlacement {
  /** Tier holding each offloaded layer's full expert set. */
  tier: 'host_ram' | 'nvme';
  /** Offloaded layer range `[from, to)` (all layers when a lone rule gave none); non-MoE layers are skipped. */
  layers: { from: number; to: number };
  /** Fraction f of each offloaded layer's experts resident on the device, in [0, 1]. Dimensionless. */
  residentFraction: number;
  /**
   * h(f): probability that a unique needed expert is resident, already evaluated by this rule's hit-ratio model
   * (design doc §2.4), before the physical clamp of the hit count. Dimensionless, in [0, 1].
   */
  hitRatio: number;
  /** Also offload the drafter's own MoE experts (DSpark MoE drafters) with this placement. */
  includeDraft: boolean;
}

/** One resolved lookup-table placement rule: the n-gram tables of layers `[from, to)` read row by row from `tier`. */
export interface TablePlacement {
  /** Tier holding these layers' tables. */
  tier: 'host_ram' | 'nvme';
  /** Layer range `[from, to)`, zero-based (every layer when a lone rule gave none); layers without a table are skipped. */
  layers: { from: number; to: number };
  /** Scale layout of block-scaled tables (design doc §2.10), default filled in. */
  scaleLayout: 'separate' | 'pinned_host' | 'interleaved';
}

/**
 * The resolved token-embedding placement: the stage-0 table shard lives on `tier`, and each embedding gather reads
 * its `R = T/tp` rows from there (docs/simulation-model.md §12 "Token embedding off-device"). The read layout is
 * always `'separate'` and nothing is pinned, so the tier is the only field.
 */
export interface EmbeddingPlacement {
  /** Tier holding the token-embedding table shard. */
  tier: 'host_ram' | 'nvme';
}

/**
 * The resolved KV-cache placement (`KvCacheOffloadConfig` with every default filled in, docs/simulation-model.md §12
 * "KV cache off-device"): of each offloadable attention layer's per-sequence KV, a share `deviceFraction` is in device
 * memory; the rest is split between host RAM, the SSD and the remote KV pool as `ramShare : nvmeShare :
 * remotePoolShare` (shares of the off-device part, summing to 1). Every attention step reads its non-resident past
 * tokens from those tiers (`lower/kvOffload.ts`).
 */
export interface KvCachePlacement {
  /** f_dev ∈ [0, 1]: share of each offloadable layer's per-sequence KV in device memory (decode). Dimensionless. */
  deviceFraction: number;
  /**
   * Share of the OFF-device KV in host RAM, `f_ram / (1 − f_dev)` (1 when f_dev = 1 and no split or pool was given).
   * Dimensionless, in [0, 1]; `ramShare + nvmeShare + remotePoolShare = 1`.
   */
  ramShare: number;
  /** Share of the OFF-device KV on the SSD, `f_ssd / (1 − f_dev)`. Dimensionless, in [0, 1]. */
  nvmeShare: number;
  /**
   * Share of the OFF-device KV in the remote KV pool, `f_pool / (1 − f_dev)` (1 when f_dev = 1 and a pool was given).
   * 0 without a pool. Dimensionless, in [0, 1].
   */
  remotePoolShare: number;
  /**
   * h: probability that an attended past token of a top-k (sparse) layer is device-resident, already evaluated by
   * the hit model (`'uniform'` → f_dev), before the physical clamp. Unused by full-attention layers.
   * Dimensionless, in [0, 1].
   */
  hitRatio: number;
  /**
   * f_pre ∈ [0, 1]: share of the prefix-cache hit's KV in device memory at prefill (default f_dev). Dimensionless.
   */
  prefixDeviceFraction: number;
}

/**
 * The off-device placement lowering applies, resolved from `OffloadConfig` with every default filled in (built once
 * per run by `simulate`). Each present group adds one `TransferOp` in front of the block that consumes it, only
 * when it has bytes to move. Layer ranges are `[from, to)` over `ModelSpec.layers` (zero-based indices).
 */
export interface OffloadLowering {
  /**
   * Prefetch accuracy p (`OffloadConfig.prefetchAccuracy`, default filled in): sets each transfer op's
   * `prefetchedFraction` (docs §12 "Expert prefetch"). Dimensionless, in [0, 1].
   */
  prefetchAccuracy: number;
  /**
   * Routed-expert placements, one per rule: the MoE layers in each `layers` keep their experts on its `tier`, with a
   * resident cache of a fraction of them. Absent = none; else non-empty, sorted by `layers.from`, ranges disjoint.
   * Layers no placement covers keep their experts on the device.
   */
  routedExperts?: readonly ExpertPlacement[];
  /**
   * Off-device n-gram lookup-table placements, one per rule (`'device'` rules dropped): the tables of the layers in each
   * `layers` are read row by row from its `tier`. Absent = none; else non-empty, sorted by `layers.from`, ranges
   * disjoint. Layers no placement covers keep their tables on the device.
   */
  lookupTables?: readonly TablePlacement[];
  /**
   * Off-device token-embedding placement (`OffloadConfig.embeddings` with a tier other than `'device'`, kept after
   * the unified-memory drop): stage 0's embedding gathers, and the drafter's shared-embedding gathers, read their
   * rows from its `tier`. Absent = the table is on the device.
   */
  embeddings?: EmbeddingPlacement;
  /**
   * Off-device KV-cache placement (`OffloadConfig.kvCache` when it moves anything). Absent = the whole KV cache is on
   * the device, and attention emits no `kv_cache` transfer.
   */
  kvCache?: KvCachePlacement;
  /**
   * Unified memory: host RAM is the device pool, so pinned-host table scales (and any host-RAM group) are device
   * bytes, never a separate tier. Absent/false = dedicated memory (host RAM is its own tier).
   */
  hostRamIsDevice?: boolean;
}

/**
 * One speculative-decoding step as lowering sees it, resolved from `SpecDecodeConfig` (design doc §3.3, §3.5).
 * The verify pass is the target lowered with `qLen = verifyTokens`; the drafter's ops are appended to the last stage
 * as `draft.*` layers.
 */
export interface SpeculativeLowering {
  /** The drafter to lower: the checkpoint's MTP module(s) or the block drafter. */
  draft: DraftSpec;
  /** Draft tokens per step k. Count. */
  draftLength: number;
  /** Target tokens verified per sequence per step, k + 1. Unit: tokens. */
  verifyTokens: number;
  /**
   * Newly committed tokens of the previous step that the drafter ingests, `m_ctx = ceil(E[τ])` (design doc §3.3).
   * Unit: tokens.
   */
  contextRows: number;
  /** Recurrent/conv state snapshots written by the verify's state ops, k + 1 (`RecurrentOp.stateSnapshots`). Count. */
  stateSnapshots: number;
}

/** The ops of one lowered layer on one device of its stage, in execution order. */
export interface LoweredLayer {
  /** `'embed'`, `'layer.<i>'` or `'head'`. */
  name: string;
  /** Index into `ModelSpec.layers`, or `null` for the embedding and the head. */
  layerIndex: number | null;
  /** Per-device ops in execution order. */
  ops: Op[];
}

/** One pipeline stage: its layers (stage 0 starts with `'embed'`, the last stage ends with `'head'`). */
export interface LoweredStage {
  /** Stage index, 0-based. */
  index: number;
  /** Lowered layers of this stage, in execution order. */
  layers: LoweredLayer[];
  /** Activation send to the next stage (absent on the last stage). */
  sendToNext?: CollectiveOp;
  /** Decode-only sampled uint32 token IDs returned from the last stage to stage 0 (b_mb·qLen·4 bytes). */
  sendToFirst?: CollectiveOp;
}

/** A model lowered for one phase step: per-stage, per-device ops. */
export interface LoweredModel {
  /** Pipeline stages, `parallel.pp` of them, in order. */
  stages: LoweredStage[];
}

/** Lower a whole model for one phase step (embedding, all layers split into PP stages, final norm + LM head). */
export function lowerModel(model: ModelSpec, ctx: LoweringContext): LoweredModel {
  const pp = ctx.parallel.pp;
  const L = model.layers.length;
  if (pp < 1 || pp > L)
    throw new Error(`pp (${pp}) must be between 1 and the number of layers (${L})`);
  const stages: LoweredStage[] = [];
  for (let s = 0; s < pp; s++) {
    const layers: LoweredLayer[] = [];
    if (s === 0) layers.push({ name: 'embed', layerIndex: null, ops: lowerEmbedding(model, ctx) });
    const [start, end] = stageLayerRange(L, pp, s);
    for (let i = start; i < end; i++) {
      const ops = model.layers[i].blocks.flatMap((block) => lowerBlock(block, ctx, i));
      layers.push({ name: `layer.${i}`, layerIndex: i, ops });
    }
    const stage: LoweredStage = { index: s, layers };
    if (s === pp - 1) {
      layers.push({ name: 'head', layerIndex: null, ops: lowerHead(model, ctx) });
      if (pp > 1 && ctx.phase === 'decode') {
        // Formula: S_return = b_mb·(k+1)·4 B; plain decode has qLen = 1.
        // See docs/simulation-model.md §5 "PP split" (audit G2).
        stage.sendToFirst = {
          kind: 'collective',
          name: 'pp.sample_return',
          collective: 'send_recv',
          group: 'pp',
          groupSize: 2,
          messageBytes: stepTokens(ctx) * 4,
          dtype: 'int32',
        };
      }
    } else {
      // Formula: send_recv S = T·(N_live·h)·bpe(act), N_live = ppBoundaryResidualVectors(model, end)
      // See docs/simulation-model.md §5 "PP split"
      const vectors = ppBoundaryResidualVectors(model, end);
      stage.sendToNext = {
        kind: 'collective',
        name: 'pp.send_recv',
        collective: 'send_recv',
        group: 'pp',
        groupSize: 2,
        messageBytes: stepTokens(ctx) * vectors * model.hidden * actBpe(ctx),
        dtype: ctx.quant.activations,
      };
    }
    stages.push(stage);
  }
  return { stages };
}

/**
 * Residual vectors per token, each `hidden` wide, that a PP stage boundary before layer `layerIndex` must carry: the
 * activations the layers from `layerIndex` on still read. Count (multiply by `T·h·bpe(act)` for bytes).
 *
 * - Plain residual: 1 (the running sum). Hyper-connections: `residualStreams` N.
 * - Attention Residuals (Kimi K3): the running sum plus every block summary stored so far, `1 + ceil(i/B)`
 *   (`builders.ts: attnResSummariesBefore`); later layers and the final mix read all of them. This equals the
 *   `candidates` of layer i's first `residual_mix` (its pre-mixer mix, which mixes exactly those vectors), so it is
 *   read from there rather than recomputed. Kimi K3 at pp 2 (boundary before layer 46, B 12): 1 + 4 = 5.
 *
 * Formula: N_live = C(first residual_mix of layer i) if it has one, else residualStreams (default 1).
 * See docs/simulation-model.md §5 "PP split".
 */
export function ppBoundaryResidualVectors(model: ModelSpec, layerIndex: number): number {
  const mix = model.layers[layerIndex]?.blocks.find((b) => b.kind === 'residual_mix');
  if (mix !== undefined && mix.kind === 'residual_mix') return mix.candidates;
  return model.residualStreams ?? 1;
}

/**
 * Lower one block to its per-device ops, in execution order (a `switch` on `block.kind` into `lower/<kind>.ts`).
 * `layerIndex` (index into `ModelSpec.layers`) locates the block for the offload placement; without it (draft layers,
 * one-token weight probes) no transfer op is emitted.
 */
export function lowerBlock(block: BlockSpec, ctx: LoweringContext, layerIndex?: number): Op[] {
  switch (block.kind) {
    case 'norm':
      return lowerNorm(block, ctx);
    case 'residual':
      return lowerResidual(block, ctx);
    case 'residual_mix':
      return lowerResidualMix(block, ctx);
    case 'gqa_attention':
      return lowerGqa(block, ctx);
    case 'mla_attention':
      return lowerMla(block, ctx);
    case 'csa_attention':
      return lowerCsa(block, ctx);
    case 'linear_attention':
      return lowerLinearAttention(block, ctx);
    case 'mlp':
      return lowerMlp(block, ctx);
    case 'moe':
      return lowerMoe(block, ctx, layerIndex);
    case 'hc_pre':
      return lowerHcPre(block, ctx);
    case 'hc_post':
      return lowerHcPost(block, ctx);
    case 'ngram_embedding':
      return lowerNgram(block, ctx, layerIndex);
    case 'short_conv':
      return lowerShortConv(block, ctx);
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Weight bytes of one block resident on ONE device (after TP/EP sharding; all local experts, not only the
 * active ones). Unit: bytes.
 *
 * Every weight of a non-MoE block is read exactly once per step whatever the token count, so its resident
 * bytes equal the weight bytes of its ops in any step: we lower a one-token decode step and sum them (this keeps
 * resident and streamed weights consistent by construction). A MoE step touches only active experts, and an
 * n-gram step only its gathered table rows, so those use their own formulas ({@link moeResidentWeightBytes},
 * {@link ngramResidentWeightBytes}), which also drop the off-device part of an offloaded layer `layerIndex`.
 */
export function blockResidentWeightBytes(
  block: BlockSpec,
  ctx: LoweringContext,
  layerIndex?: number,
): number {
  switch (block.kind) {
    case 'moe':
      return moeResidentWeightBytes(block, ctx, layerIndex);
    case 'ngram_embedding':
      return ngramResidentWeightBytes(block, ctx, layerIndex);
    case 'norm':
    case 'residual':
    case 'residual_mix':
    case 'gqa_attention':
    case 'mla_attention':
    case 'linear_attention':
    case 'csa_attention':
    case 'hc_pre':
    case 'hc_post':
    case 'short_conv':
    case 'mlp': {
      const oneTokenStep: LoweringContext = { ...ctx, phase: 'decode', seqs: 1, qLen: 1, kvLen: 2 };
      // Formula: W_block = Σ_ops weightBytes(op) over a one-token step
      // See docs/simulation-model.md §8 "Memory fit"
      return lowerBlock(block, oneTokenStep).reduce((sum, op) => sum + opBytes(op).weightBytes, 0);
    }
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Per-sequence state bytes of one block on ONE device for a context of `contextTokens`: KV / MLA latent /
 * indexer-key caches × contextTokens, plus the fixed recurrent and conv1d state. Hyper-connection steps keep
 * none. Unit: bytes per sequence.
 */
export function blockSequenceStateBytes(
  block: BlockSpec,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  switch (block.kind) {
    case 'gqa_attention':
      return gqaSequenceStateBytes(block, ctx, contextTokens);
    case 'mla_attention':
      return mlaSequenceStateBytes(block, ctx, contextTokens);
    case 'csa_attention':
      return csaSequenceStateBytes(block, ctx, contextTokens);
    case 'linear_attention':
      return linearAttentionSequenceStateBytes(block, ctx);
    case 'ngram_embedding':
      return ngramSequenceStateBytes(block, ctx);
    case 'short_conv':
      return shortConvSequenceStateBytes(block, ctx);
    case 'norm':
    case 'residual':
    case 'residual_mix':
    case 'mlp':
    case 'moe':
    case 'hc_pre':
    case 'hc_post':
      return 0;
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Layer index range `[start, end)` of stage `s` when `L` layers are split into `pp` stages:
 * `[floor(s·L/pp), floor((s+1)·L/pp))`. Counts and indices (no unit).
 */
export function stageLayerRange(L: number, pp: number, s: number): [number, number] {
  // Formula: [floor(s·L/pp), floor((s+1)·L/pp))
  // See docs/simulation-model.md §5 "PP split"
  return [Math.floor((s * L) / pp), Math.floor(((s + 1) * L) / pp)];
}
