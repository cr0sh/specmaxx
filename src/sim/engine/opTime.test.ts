import { describe, expect, it } from 'vitest';
import { SYNTH_GPU } from '../__fixtures__/synthetic';
import { M3_ULTRA_80C, M5_MAX_40C } from '../hw/presets/apple';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { ComputeEngine } from '../hw/types';
import type { AttentionOp, ElementwiseOp, GemmOp, IndexerOp, RecurrentOp } from '../ir/ops';
import {
  elementwiseBytesUnderFusion,
  engineMemBandwidthBytesPerSec,
  opTime,
  roofline,
  type OpTimeContext,
} from './opTime';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const SYNTH_GPU_ENGINE = SYNTH_GPU.engines[0];
/** SYNTH_GPU: bf16/fp32 dense 100e12 FLOP/s, vector bf16 20e12, 100 units, tile 128×128 (mma 64×8×16), 1e12 B/s. */
const CTX: OpTimeContext = {
  engine: SYNTH_GPU_ENGINE,
  memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(
    SYNTH_GPU_ENGINE,
    SYNTH_GPU.memory.bandwidthBytesPerSec,
  ),
  collectivePaths: { tp: { kind: 'none' }, ep: { kind: 'none' }, pp: { kind: 'none' } },
  collectiveAlgorithm: 'best',
  elementwiseFusion: 'none',
  warnings: [],
};

function gemm(m: number, n: number, k: number): GemmOp {
  return {
    kind: 'gemm',
    name: 'test.gemm',
    batch: 1,
    m,
    n,
    k,
    inputDtype: 'bf16',
    weight: { dtype: 'bf16' },
    outputDtype: 'bf16',
    math: 'bf16',
    accumulate: 'fp32',
    sparsity: 'dense',
  };
}

describe('T6 roofline (SYNTH_GPU)', () => {
  it('a peer lookup without its intra-node DP path fails rather than being priced as a local gather', () => {
    expect(() =>
      opTime(
        {
          kind: 'embedding',
          name: 'peer.lookup',
          tokens: 4,
          hidden: 8,
          vocabShard: 500,
          weight: { dtype: 'bf16' },
          outputDtype: 'bf16',
          peerReadBytes: 32,
        },
        CTX,
      ),
    ).toThrow(/needs a flat intra-node DP path/);
  });

  it('roofline picks the larger term; a tie is compute-bound', () => {
    expect(roofline(1e-3, 2e-3)).toEqual({ sec: 2e-3, bound: 'memory' });
    expect(roofline(1e-3, 1e-3)).toEqual({ sec: 1e-3, bound: 'compute' });
  });

  it('balanced elementwise op: 20e9 FLOP at 20e12 and 1e9 B at 1e12 → both terms 1e-3 s', () => {
    // The plan's "100e9 FLOP / 1e9 B" case is shape-independent only for a vector op, so it is expressed here at
    // the bf16 vector peak (20e12): 1e9 elems · 20 FLOP = 20e9 FLOP → 1e-3 s; 1e9 B / 1e12 B/s → 1e-3 s.
    const op: ElementwiseOp = {
      kind: 'elementwise',
      name: 'test.ew',
      fn: 'residual_add',
      elems: 1e9,
      flopsPerElem: 20,
      dtype: 'bf16',
      activationReadBytes: 0.5e9,
      activationWriteBytes: 0.5e9,
      weightBytes: 0,
      stateReadBytes: 0,
      stateWriteBytes: 0,
    };
    const r = opTime(op, CTX);
    expectClose(r.computeSec, 1e-3);
    expectClose(r.memorySec, 1e-3);
    expectClose(r.sec, 1e-3);
    // Ideal fusion keeps only weight/state bytes: memory term 0, compute-bound.
    const fused = opTime(op, { ...CTX, elementwiseFusion: 'ideal' });
    expect(fused.memorySec).toBe(0);
    expect(fused.bound).toBe('compute');
    expect(elementwiseBytesUnderFusion(op, 'ideal').totalBytes).toBe(0);
  });

  it('GEMM n = k = 4096, m = 1 is memory-bound', () => {
    const r = opTime(gemm(1, 4096, 4096), CTX);
    // memory: 4096·4096·2 B weights + 2·4096·2 B in/out = 33,570,816 B / 1e12 = 33.57 µs
    expectClose(r.memorySec, (4096 * 4096 * 2 + 2 * 4096 * 2) / 1e12);
    // compute: swap-AB, 32 tiles of 128×8 in one wave: 2·128·8·4096 / (100e12/100) = 8.39 µs
    expectClose(r.computeSec, (2 * 128 * 8 * 4096) / 1e12);
    expect(r.bound).toBe('memory');
  });

  it('GEMM m = n = k = 4096 is compute-bound with efficiency 1024/1100', () => {
    const r = opTime(gemm(4096, 4096, 4096), CTX);
    // 32·32 = 1024 tiles of 128×128 on 100 slots → 11 waves = 1100 tile slots → efficiency 1024/1100
    expectClose(r.efficiency ?? 0, 1024 / 1100);
    // compute = 2·4096³ / (100e12 · 1024/1100) = 1.4763 ms; memory = 3·4096²·2 B / 1e12 = 0.1007 ms
    expectClose(r.computeSec, (2 * 4096 ** 3) / (100e12 * (1024 / 1100)));
    expect(r.bound).toBe('compute');
    expect(r.peakFlopsPerSec).toBe(100e12);
    expect(r.usedInput).toBe('bf16');
  });

  it('an engine memory cap lowers the bandwidth: min(device, cap)', () => {
    expect(
      engineMemBandwidthBytesPerSec(
        { ...SYNTH_GPU_ENGINE, memBandwidthCapBytesPerSec: 0.25e12 },
        1e12,
      ),
    ).toBe(0.25e12);
    expect(engineMemBandwidthBytesPerSec(SYNTH_GPU_ENGINE, 1e12)).toBe(1e12);
  });
});

describe('attention softmax: t_compute = max(t_mma, t_exp + t_vec) (docs §7 "Attention softmax")', () => {
  /** 8 heads (MHA), hd 128, non-causal 1024 × 1024: S = 8·1024² scores, F = 2·S·256 FLOP. */
  const attention: AttentionOp = {
    kind: 'attention',
    name: 'test.attn',
    seqs: 1,
    heads: 8,
    kvHeads: 8,
    qkHeadDim: 128,
    vHeadDim: 128,
    qLen: 1024,
    kvLen: 1024,
    causal: false,
    kvSelection: { kind: 'full' },
    kvRead: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    kvWrite: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    math: 'bf16',
    accumulate: 'fp32',
    activationDtype: 'bf16',
  };
  const S = 8 * 1024 * 1024;
  const withSfu = (sfuOpsPerSec: number): OpTimeContext => ({
    ...CTX,
    engine: { ...SYNTH_GPU_ENGINE, sfuOpsPerSec },
  });

  it('no sfuOpsPerSec: softmax not costed, the result is the MMA roofline alone', () => {
    const r = opTime(attention, CTX);
    expect(r.softmaxSec).toBeUndefined();
    expect(r.computeSec).toBe(r.flops / (r.peakFlopsPerSec! * r.efficiency!));
  });

  it('softmax binds (SFU 1e11/s): computeSec is exactly t_exp + t_vec', () => {
    const r = opTime(attention, withSfu(1e11));
    // t_exp = S / 1e11 = 83.9 µs; t_vec = 5·S / vector fp32 10e12 = 4.19 µs; t_mma ≈ 50 µs
    const softmax = S / 1e11 + (5 * S) / 10e12;
    expect(r.softmaxSec).toBe(softmax);
    expect(r.computeSec).toBe(softmax);
    expect(r.bound).toBe('compute');
    expect(r.computeSec).toBeGreaterThan(r.flops / (r.peakFlopsPerSec! * r.efficiency!));
  });

  it('softmax does not bind (SFU 1e13/s): computeSec is bit-identical to the MMA time', () => {
    const before = opTime(attention, CTX);
    const r = opTime(attention, withSfu(1e13));
    expect(r.softmaxSec).toBe(S / 1e13 + (5 * S) / 10e12);
    expect(r.computeSec).toBe(before.computeSec);
    expect(r.sec).toBe(before.sec);
    expect(r.bound).toBe(before.bound);
  });

  it('the indexer and recurrent (linear attention) ops have no softmax term', () => {
    const indexer: IndexerOp = {
      kind: 'indexer',
      name: 'test.idx',
      seqs: 1,
      heads: 4,
      headDim: 32,
      qLen: 1024,
      kvLen: 1024,
      causal: true,
      topK: 64,
      keyCache: { dtype: 'bf16' },
      keyElemsPerKey: 32,
      keyPoolTokens: 1,
      keyWriteElemsPerToken: 32,
      headWeights: true,
      math: 'bf16',
      accumulate: 'fp32',
      activationDtype: 'bf16',
    };
    const chunked: RecurrentOp = {
      kind: 'recurrent',
      name: 'test.gdn',
      variant: 'gated_delta_net',
      seqs: 1,
      qLen: 1024,
      heads: 8,
      qkHeads: 4,
      keyDim: 64,
      valueDim: 64,
      mode: 'chunked',
      chunkLen: 64,
      stateFlopsPerElemPerToken: 4,
      gateElemsPerTokenPerHead: 2,
      hasPriorState: false,
      stateDtype: 'fp32',
      activationDtype: 'bf16',
      math: 'bf16',
      accumulate: 'fp32',
    };
    for (const op of [
      indexer,
      chunked,
      { ...chunked, mode: 'step' as const, qLen: 1, chunkLen: 1 },
    ]) {
      expect(opTime(op, withSfu(1e9))).toEqual(opTime(op, CTX));
    }
  });
});

describe('GEMV path: t_compute = min(t_mma, F / vectorPeak) for m ≤ gemvMaxRows (docs §4 "GEMV path")', () => {
  const m3 = M3_ULTRA_80C.engines[0];
  const ctxOf = (engine: ComputeEngine, device = M3_ULTRA_80C): OpTimeContext => ({
    ...CTX,
    engine,
    memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(
      engine,
      device.memory.bandwidthBytesPerSec,
    ),
  });
  const { gemvMaxRows: _rows, ...mmaOnly } = m3;
  void _rows;
  /** A MoE router: m = 1 token, n = 128 experts, k = 4096, bf16. */
  const router = gemm(1, 128, 4096);

  it('M3 Ultra sets gemvMaxRows 1 (MLX and llama.cpp run m = 1 as a vector GEMV)', () => {
    expect(m3.gemvMaxRows).toBe(1);
  });

  it('m = 1 router GEMM: faster than the padded 8×8 MMA path, at the vector peak, memory-bound', () => {
    const before = opTime(router, ctxOf(mmaOnly));
    const r = opTime(router, ctxOf(m3));
    expect(r.computePath).toBe('gemv');
    expect(r.computeSec).toBe(r.flops / m3.vectorFlopsPerSec.bf16!);
    expect(r.computeSec).toBeLessThan(before.computeSec);
    expect(r.peakFlopsPerSec).toBe(m3.vectorFlopsPerSec.bf16);
    expect(r.efficiency).toBe(1);
    expect(r.memorySec).toBe(before.memorySec);
    expect(r.bound).toBe('memory');
  });

  it('m = gemvMaxRows + 1 keeps the MMA path, unchanged', () => {
    const op = gemm(2, 128, 4096);
    expect(opTime(op, ctxOf(m3))).toEqual(opTime(op, ctxOf(mmaOnly)));
  });

  it('int8 math and 2:4 weights are not eligible (no vector peak / no sparse GEMV)', () => {
    const int8: GemmOp = {
      ...router,
      inputDtype: 'int8',
      weight: { dtype: 'int8' },
      math: 'int8',
      accumulate: 'int32',
    };
    expect(opTime(int8, ctxOf(m3)).computePath).toBeUndefined();
    const sparse: GemmOp = { ...router, sparsity: '2:4' };
    expect(opTime(sparse, ctxOf(m3)).computePath).toBeUndefined();
  });

  it('an engine without gemvMaxRows (RTX 5090) keeps the MMA path at m = 1', () => {
    const engine = RTX_5090.engines[0];
    expect(engine.gemvMaxRows).toBeUndefined();
    expect(opTime(router, ctxOf(engine, RTX_5090)).computePath).toBeUndefined();
  });

  it('M5 Max NAX engine at m = 1 falls back to the SIMD fp16/bf16 vector peak', () => {
    const nax = M5_MAX_40C.engines[0];
    expect(nax.id).toBe('gpu-na');
    const r = opTime(router, ctxOf(nax, M5_MAX_40C));
    expect(r.computePath).toBe('gemv');
    expect(r.peakFlopsPerSec).toBe(M5_MAX_40C.engines[1].vectorFlopsPerSec.bf16);
  });
});
