import { describe, expect, it } from 'vitest';
import type { CsaAttentionBlock, LayerSpec, ModelSpec } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild } from '../types';
import { deepseekV41Family } from './deepseekV41';
import { layerParams } from './mlaCommon';

/** Build a fixture repo with the DeepSeek family; fails the test unless the build is ok. */
function build(id: string): FamilyBuild {
  const files = fixtureRepoFiles(id);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const r = deepseekV41Family.build(u.view.text, u.view.top, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const csaOf = (layer: LayerSpec): CsaAttentionBlock => {
  const b = layer.blocks.find((x) => x.kind === 'csa_attention');
  if (b?.kind !== 'csa_attention') throw new Error(`${layer.name}: no csa block`);
  return b;
};

/** The CSA role of a layer, from its block (not from the config). */
function role(layer: LayerSpec): string {
  const g = csaOf(layer).global;
  if (g === undefined) return 'swa';
  if (g.selection.kind === 'reuse' || g.selection.kind === 'all') return g.selection.kind;
  return g.selection.indexer.ownsKeys === false ? 'reindex' : 'full';
}

const ID = 'deepseek-ai/DeepSeek-V4.1-Flash';

describe('deepseek_v41: DeepSeek-V4.1-Flash', () => {
  const b = build(ID);
  const m: ModelSpec = b.model;
  const h = m.hidden;

  it('reads the dims and the mHC stream count', () => {
    expect(m.layers).toHaveLength(40);
    expect(m.hidden).toBe(5120);
    expect(m.vocab).toBe(129280);
    expect(m.tieEmbeddings).toBe(false);
    expect(m.residualStreams).toBe(4);
    expect(m.finalNorm).toBe('rmsnorm');
  });

  it('assigns the CSA layer roles from compress_ratios / kv_source_layer_ids / index_source_layer_ids', () => {
    const roles = m.layers.map(role);
    const count = (r: string) => roles.filter((x) => x === r).length;
    expect(count('swa')).toBe(2);
    expect(count('full')).toBe(4);
    expect(count('reindex')).toBe(4);
    expect(count('reuse')).toBe(30);
    expect([0, 1].map((i) => roles[i])).toEqual(['swa', 'swa']);
    expect([2, 8, 14, 20].map((i) => roles[i])).toEqual(['full', 'full', 'full', 'full']);
    expect([24, 28, 32, 36].map((i) => roles[i])).toEqual([
      'reindex',
      'reindex',
      'reindex',
      'reindex',
    ]);
  });

  it('points each consumer at the latest earlier KV source and bounds the late indexers', () => {
    const g = (i: number) => csaOf(m.layers[i]).global;
    expect(g(2)?.kvCacheFromLayer).toBeUndefined();
    expect(g(3)?.kvCacheFromLayer).toBe(2);
    expect(g(13)?.kvCacheFromLayer).toBe(8);
    expect(g(19)?.kvCacheFromLayer).toBe(14);
    expect(g(24)?.kvCacheFromLayer).toBe(20);
    expect(g(39)?.kvCacheFromLayer).toBe(20);
    expect(g(3)?.compressRatio).toBe(2);
    expect(g(21)?.compressRatio).toBe(1);
    const ix = (i: number) => {
      const s = g(i)?.selection;
      return s?.kind === 'indexer' ? s.indexer : undefined;
    };
    expect(ix(2)).toEqual({
      heads: 32,
      headDim: 128,
      topK: 512,
      keyPoolTokens: 2,
      keyInputDim: 512,
      ownsKeys: true,
    });
    // Layer 20 is the candidate source itself: it scores every key.
    expect(ix(20)?.maxScoredKeys).toBeUndefined();
    expect(ix(24)).toEqual({
      heads: 32,
      headDim: 128,
      topK: 512,
      keyPoolTokens: 1,
      keyInputDim: 512,
      ownsKeys: false,
      maxScoredKeys: 16384,
    });
    expect(g(25)?.selection).toEqual({ kind: 'reuse', topK: 512 });
  });

  it('places the Engram tables before layers 1 and 14', () => {
    const engrams = m.layers.flatMap((l, i) => (l.blocks[0].kind === 'ngram_embedding' ? [i] : []));
    expect(engrams).toEqual([1, 14]);
    const e = m.layers[1].blocks[0];
    expect(e).toMatchObject({
      kind: 'ngram_embedding',
      tableRows: 384006168,
      rowDim: 256,
      rowsPerToken: 24,
      outputDim: 4 * h,
    });
  });

  // Per-layer params from the safetensors headers (findings R2, dsv41/tensors.json, I8 fp4 bytes counted twice,
  // scales excluded). The IR leaves out attn_sink (64) and the router biases gate.bias + gate.bias_vl (2·384),
  // and counts the key-owning indexers' k_norm (RMSNorm, 128) as a LayerNorm (+128).
  it.each([
    [0, 13_754_511_990, 0], // SWA-only
    [1, 112_217_418_358, 0], // SWA-only + Engram
    [2, 13_765_227_766, 128], // Full, r = 2
    [3, 13_754_511_990, 0], // Reuse
    [20, 13_762_606_326, 128], // Full, r = 1
    [21, 13_754_511_990, 0], // Reuse, r = 1
    [24, 13_759_918_710, 0], // Reindex: no wk, no k_norm
  ])('layer %i matches the checkpoint headers', (i, header, keyNormDelta) => {
    expect(layerParams(m.layers[i])).toBe(header - 64 - 2 * 384 + keyNormDelta);
  });

  it('total and active params match the header-derived targets', () => {
    // 763,205,315,794 (safetensors.total) − MTP 14,225,362,530 − vision 485,268,480; includes Engram 196.6B.
    const target = 748_494_684_784;
    expect(Math.abs(countParams(m) / target - 1)).toBeLessThan(0.005);
    // Decode active, excluding the embedding: layers 15,468,670,064 + head 661,913,600 (README: "16B decode").
    const activeTarget = 16_130_583_664;
    const active = activeParamsPerToken(m) - m.vocab * h;
    expect(Math.abs(active / activeTarget - 1)).toBeLessThan(0.015);
  });

  it('estimates the excluded MTP stages and vision tower', () => {
    expect(Math.abs(b.mtpParamsEstimate / 14_225_362_530 - 1)).toBeLessThan(0.001);
    expect(Math.abs(b.visionParamsEstimate / 485_268_480 - 1)).toBeLessThan(0.01);
    expect(b.excludedParts).toEqual(['DSpark MTP stages', 'vision tower + aligner']);
    const files = fixtureRepoFiles(ID);
    const check = checkParams(m, b, files.api, 'fp8');
    expect(check.status).toBe('ok');
    expect(Math.abs(check.relError ?? 1)).toBeLessThan(0.001);
  });

  it('hints the native KV-cache formats', () => {
    expect(b.quantHints).toEqual({
      kvCache: { dtype: 'mxfp8' },
      compressedKvCache: { dtype: 'nvfp4' },
      indexerKeyCache: { dtype: 'mxfp4' },
    });
  });

  it('builds the -1017 checkpoint identically', () => {
    const other = build('deepseek-ai/DeepSeek-V4.1-Flash-1017');
    expect(other.model.layers).toEqual(m.layers);
    expect(countParams(other.model)).toBe(countParams(m));
  });

  it.each([Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER + 1, 1e100])(
    'rejects unsafe Engram bucket sizes without stalling the loader: %s',
    (base) => {
      const files = fixtureRepoFiles(ID);
      const u = unwrapTextConfig(files.config);
      if (u.kind !== 'ok') throw new Error('unwrap failed');
      const text = { ...u.view.text, engram_vocab_size: base };
      const r = deepseekV41Family.build(text, { ...u.view.top, text_config: text }, files);
      expect(r.kind).toBe('invalid_config');
      if (r.kind === 'invalid_config')
        expect(r.errors).toContain(
          'text_config.engram_vocab_size: cannot generate safe integer hash-head buckets',
        );
    },
  );

  it('rejects a consumer layer with no earlier KV source', () => {
    const files = fixtureRepoFiles(ID);
    const u = unwrapTextConfig(files.config);
    if (u.kind !== 'ok') throw new Error('unwrap failed');
    const text = { ...u.view.text, kv_source_layer_ids: [8, 14, 20] };
    const r = deepseekV41Family.build(text, { ...u.view.top, text_config: text }, files);
    expect(r.kind).toBe('invalid_config');
  });
});
