/**
 * Structural validation of a model spec, independent of workload, quantization and parallelism: the cross-layer
 * references (shared KV caches, reused top-k, shared indexer keys) point to compatible EARLIER layers, the KDA
 * fields are consistent, and the hyper-connection and n-gram widths agree with the model's residual streams.
 * Pure; the engine's input validation calls it. See docs/simulation-model.md §2 "Model IR".
 */

import type {
  BlockSpec,
  CsaAttentionBlock,
  IndexerSpec,
  LayerSpec,
  ModelSpec,
  SparseSelection,
} from './model';

/** The attention block of a layer that can share caches or selections (gqa, mla, csa), if any. */
type AttentionBlock = Extract<
  BlockSpec,
  { kind: 'gqa_attention' | 'mla_attention' | 'csa_attention' }
>;

/** The first gqa/mla/csa attention block of a layer, or undefined. */
function attentionOf(layer: LayerSpec): AttentionBlock | undefined {
  for (const b of layer.blocks) {
    if (b.kind === 'gqa_attention' || b.kind === 'mla_attention' || b.kind === 'csa_attention')
      return b;
  }
  return undefined;
}

/**
 * The sparse selection of an attention block (csa: its global branch's), or undefined when it is dense. A csa
 * `{ kind: 'all' }` selection (attend every compressed entry) maps to undefined: it references no other layer.
 */
function selectionOf(block: AttentionBlock): SparseSelection | undefined {
  if (block.kind !== 'csa_attention') return block.sparse;
  const sel = block.global?.selection;
  return sel === undefined || sel.kind === 'all' ? undefined : sel;
}

/** The indexer an attention block runs, or undefined (dense or a reused top-k). */
function indexerOf(block: AttentionBlock): IndexerSpec | undefined {
  const sel = selectionOf(block);
  return sel?.kind === 'indexer' ? sel.indexer : undefined;
}

/** The compress ratio of a csa block's global branch (0 when SWA-only); undefined for other kinds. */
function compressRatioOf(block: AttentionBlock): number | undefined {
  return block.kind === 'csa_attention' ? (block.global?.compressRatio ?? 0) : undefined;
}

/** Errors of an MLA `kvCacheFromLayer` (layer i): an earlier MLA latent owner with the same rkv and dr. */
function mlaKvSourceErrors(
  model: ModelSpec,
  i: number,
  src: number,
  rkv: number,
  dr: number,
): string[] {
  const at = `layer ${i}: kvCacheFromLayer ${src}`;
  if (!Number.isInteger(src) || src < 0 || src >= i)
    return [`${at} must be an earlier layer index`];
  const source = attentionOf(model.layers[src]);
  if (source?.kind !== 'mla_attention') return [`${at} has no mla_attention block`];
  if (source.kvCacheFromLayer !== undefined) return [`${at} does not own its latent cache`];
  if (source.kvLoraRank !== rkv || source.qkRopeDim !== dr) {
    return [
      `${at} caches rkv ${source.kvLoraRank} + dr ${source.qkRopeDim}, this layer reads rkv ${rkv} + dr ${dr}`,
    ];
  }
  return [];
}

/** Errors of a CSA global `kvCacheFromLayer` (layer i): an earlier compressor owner with the same r and headDim. */
function csaKvSourceErrors(
  model: ModelSpec,
  i: number,
  block: CsaAttentionBlock,
  src: number,
): string[] {
  const at = `layer ${i}: global.kvCacheFromLayer ${src}`;
  if (!Number.isInteger(src) || src < 0 || src >= i)
    return [`${at} must be an earlier layer index`];
  const source = attentionOf(model.layers[src]);
  if (source?.kind !== 'csa_attention' || source.global === undefined)
    return [`${at} has no csa global branch`];
  if (source.global.kvCacheFromLayer !== undefined)
    return [`${at} does not own its compressed cache`];
  if (
    source.global.compressRatio !== block.global?.compressRatio ||
    source.headDim !== block.headDim
  ) {
    return [
      `${at} has compress ratio ${source.global.compressRatio} and headDim ${source.headDim}, not this layer's`,
    ];
  }
  return [];
}

/**
 * Errors of a sparse selection at layer i: a `'reuse'` needs an earlier layer of the same attention kind (and csa
 * compress ratio) that runs an indexer, as its nearest earlier indexer; `ownsKeys: false` needs an earlier
 * key-owning indexer with the same headDim and keyPoolTokens.
 */
function selectionErrors(
  model: ModelSpec,
  i: number,
  block: AttentionBlock,
  sel: SparseSelection,
): string[] {
  if (sel.kind === 'reuse') {
    for (let j = i - 1; j >= 0; j--) {
      const earlier = attentionOf(model.layers[j]);
      if (earlier === undefined || indexerOf(earlier) === undefined) continue;
      if (earlier.kind !== block.kind || compressRatioOf(earlier) !== compressRatioOf(block)) {
        return [
          `layer ${i}: reuses the top-k of layer ${j}, whose ${earlier.kind} block is not compatible`,
        ];
      }
      return [];
    }
    return [`layer ${i}: reuses a top-k selection but no earlier layer runs an indexer`];
  }
  const ix = sel.indexer;
  if (ix.ownsKeys !== false) return [];
  for (let j = i - 1; j >= 0; j--) {
    const earlier = attentionOf(model.layers[j]);
    const owner = earlier === undefined ? undefined : indexerOf(earlier);
    if (owner === undefined || owner.ownsKeys === false) continue;
    if (owner.headDim !== ix.headDim || (owner.keyPoolTokens ?? 1) !== (ix.keyPoolTokens ?? 1)) {
      return [
        `layer ${i}: reads the indexer keys of layer ${j}, whose headDim or keyPoolTokens differ`,
      ];
    }
    return [];
  }
  return [`layer ${i}: indexer has ownsKeys false but no earlier layer owns indexer keys`];
}

/** Errors of one block of layer i that do not depend on other layers (KDA fields, residual-stream widths, MoE latent width). */
function blockErrors(model: ModelSpec, i: number, block: BlockSpec): string[] {
  const streams = model.residualStreams ?? 1;
  const at = `layer ${i} ${block.name}`;
  switch (block.kind) {
    case 'linear_attention': {
      const errors: string[] = [];
      if ((block.variant === 'kda') !== (block.gateRank !== undefined)) {
        errors.push(`${at}: gateRank must be set exactly when variant is 'kda'`);
      }
      if (block.variant === 'kda' && !block.outputGate)
        errors.push(`${at}: kda needs outputGate true`);
      if (block.fullRankOutputGate === true && block.variant !== 'kda') {
        errors.push(`${at}: fullRankOutputGate is only for variant 'kda'`);
      }
      if (
        block.chunkTokens !== undefined &&
        !(Number.isInteger(block.chunkTokens) && block.chunkTokens > 0)
      ) {
        errors.push(`${at}: chunkTokens ${block.chunkTokens} must be a positive integer`);
      }
      return errors;
    }
    case 'hc_pre':
    case 'hc_post':
      if (block.streams !== streams || block.dim !== model.hidden) {
        return [
          `${at}: ${block.streams} streams × ${block.dim} must equal residualStreams ${streams} × hidden ${model.hidden}`,
        ];
      }
      return [];
    case 'ngram_embedding':
      if (block.outputDim !== streams * model.hidden) {
        return [
          `${at}: outputDim ${block.outputDim} must equal residualStreams·hidden ${streams * model.hidden}`,
        ];
      }
      if (
        block.headRows !== undefined &&
        (block.headRows.length !== block.rowsPerToken ||
          block.headRows.some((n) => !Number.isSafeInteger(n) || n < 1) ||
          block.headRows.reduce((a, b) => a + b, 0) > block.tableRows)
      )
        return [
          `${at}: headRows must contain rowsPerToken positive row counts whose sum does not exceed tableRows`,
        ];
      return [];
    case 'moe':
      if (
        block.latentDim !== undefined &&
        !(Number.isInteger(block.latentDim) && block.latentDim > 0)
      ) {
        return [`${at}: latentDim ${block.latentDim} must be a positive integer`];
      }
      if (block.latentNorm === true && block.latentDim === undefined)
        return [`${at}: latentNorm needs latentDim`];
      if (block.latentNorm === true && block.expertOutputNorm === true) {
        return [`${at}: latentNorm and expertOutputNorm cannot both be set`];
      }
      return [];
    default:
      return [];
  }
}

/** Errors of a feed-forward block's activation: `relu2` (squared ReLU) is defined only ungated. */
function activationErrors(i: number, block: BlockSpec): string[] {
  if (
    (block.kind === 'mlp' || block.kind === 'moe') &&
    block.activation === 'relu2' &&
    block.gated
  ) {
    return [`layer ${i} ${block.name}: activation relu2 needs gated false`];
  }
  return [];
}

/**
 * Structural errors of a model spec (empty = valid):
 * - `kvCacheFromLayer` (mla, csa global) points to an EARLIER layer that owns a compatible cache (same kind; mla:
 *   same rkv and dr; csa: same compress ratio and headDim);
 * - a `'reuse'` selection's nearest earlier indexer-running layer exists and has the same attention kind (and csa
 *   compress ratio);
 * - an indexer with `ownsKeys: false` has an earlier key-owning indexer of the same headDim and keyPoolTokens;
 * - `gateRank` is set iff the linear-attention variant is `'kda'` (which also needs `outputGate`), and only `'kda'`
 *   may set `fullRankOutputGate`;
 * - every `hc_pre` / `hc_post` has `residualStreams` streams of width `hidden`, and every n-gram block writes
 *   `residualStreams · hidden` elements;
 * - a linear-attention `chunkTokens`, when set, is a positive integer, and a `relu2` activation is ungated;
 * - a MoE `latentDim`, when set, is a positive integer; `latentNorm` needs `latentDim` and excludes `expertOutputNorm`.
 * Messages name the layer index.
 */
export function modelSpecErrors(model: ModelSpec): string[] {
  const errors: string[] = [];
  model.layers.forEach((layer, i) => {
    for (const block of layer.blocks)
      errors.push(...blockErrors(model, i, block), ...activationErrors(i, block));
    const attn = attentionOf(layer);
    if (attn === undefined) return;
    if (attn.kind === 'mla_attention' && attn.kvCacheFromLayer !== undefined) {
      errors.push(
        ...mlaKvSourceErrors(model, i, attn.kvCacheFromLayer, attn.kvLoraRank, attn.qkRopeDim),
      );
    }
    if (attn.kind === 'csa_attention' && attn.global?.kvCacheFromLayer !== undefined) {
      errors.push(...csaKvSourceErrors(model, i, attn, attn.global.kvCacheFromLayer));
    }
    const sel = selectionOf(attn);
    if (sel !== undefined) errors.push(...selectionErrors(model, i, attn, sel));
  });
  return errors;
}

/** One cross-layer reference: layer `consumer` reads `what` produced by the earlier layer `source`. */
export interface CrossLayerReference {
  /** Index of the reading layer. */
  consumer: number;
  /** Index of the earlier layer that produces the data. */
  source: number;
  /** What is shared: a latent / compressed KV cache, a top-k selection, or indexer keys. */
  what: 'kv_cache' | 'topk' | 'indexer_keys';
}

/**
 * Every cross-layer reference of a (valid) model: `kvCacheFromLayer` (kv_cache), a `'reuse'` selection's nearest
 * earlier indexer-running layer (topk), and an `ownsKeys: false` indexer's nearest earlier key owner
 * (indexer_keys). The engine warns when consumer and source sit on different pipeline stages.
 */
export function crossLayerReferences(model: ModelSpec): CrossLayerReference[] {
  const refs: CrossLayerReference[] = [];
  const nearest = (i: number, pick: (ix: IndexerSpec) => boolean): number | undefined => {
    for (let j = i - 1; j >= 0; j--) {
      const earlier = attentionOf(model.layers[j]);
      const ix = earlier === undefined ? undefined : indexerOf(earlier);
      if (ix !== undefined && pick(ix)) return j;
    }
    return undefined;
  };
  model.layers.forEach((layer, i) => {
    const attn = attentionOf(layer);
    if (attn === undefined) return;
    const kvSource =
      attn.kind === 'csa_attention'
        ? attn.global?.kvCacheFromLayer
        : attn.kind === 'mla_attention'
          ? attn.kvCacheFromLayer
          : undefined;
    if (kvSource !== undefined) refs.push({ consumer: i, source: kvSource, what: 'kv_cache' });
    const sel = selectionOf(attn);
    const topkSource = sel?.kind === 'reuse' ? nearest(i, () => true) : undefined;
    if (topkSource !== undefined) refs.push({ consumer: i, source: topkSource, what: 'topk' });
    const keySource =
      sel?.kind === 'indexer' && sel.indexer.ownsKeys === false
        ? nearest(i, (ix) => ix.ownsKeys !== false)
        : undefined;
    if (keySource !== undefined)
      refs.push({ consumer: i, source: keySource, what: 'indexer_keys' });
  });
  return refs;
}
