import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { ModelSpec, QuantConfig } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import {
  stageActivationPeakBytes,
  stageLookupTableBytes,
  stageTierBreakdown,
  stageTierBytes,
} from '../engine/memory';
import { opBytes } from '../ir/metrics';
import { simulate } from '../engine/simulate';
import { transferTime } from '../engine/transfer';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { lowerModel, type LoweringContext, type OffloadLowering } from './lower';
import { lowerNgram, ngramTableShardBytes, tableRowBytes } from './ngram';
import { draftResidentWeightBytes, lowerDraftDecode } from './draft';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';

function hf(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

const PLE = hf('Qwen/Qwen3.8-Flash-Next');
const ENGRAM = hf('deepseek-ai/DeepSeek-V4.1-Flash');
/** A lone table rule's resolved range: every layer. */
const ALL_LAYERS = { from: 0, to: 1000 };

function decodeCtx(quant: QuantConfig, offload?: OffloadLowering): LoweringContext {
  const ctx: LoweringContext = {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 8193,
    quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
  };
  if (offload !== undefined) ctx.offload = offload;
  return ctx;
}

/** Transfer ops of a decode step, with their layer index. */
function fetches(
  m: { model: ModelSpec; quant: QuantConfig },
  offload?: OffloadLowering,
): { layer: number | null; op: TransferOp }[] {
  return lowerModel(m.model, decodeCtx(m.quant, offload)).stages.flatMap((s) =>
    s.layers.flatMap((l) =>
      l.ops.flatMap((op) => (op.kind === 'transfer' ? [{ layer: l.layerIndex, op }] : [])),
    ),
  );
}

const ramPath = { tier: DDR5_6000_2CH, queueDepth: 64, nStream: 1 };
const ssdPath = { tier: NVME_990_PRO_2TB, queueDepth: 512, nStream: 1 };

describe('U-16 n-gram tables off-device: read layouts, tier bytes and PCIe bytes', () => {
  it('lookupTables absent ⇒ no transfer op', () => {
    // Both checkpoints ship their tables to live off-device (model cards); the flag drives the UI hint only.
    for (const m of [PLE, ENGRAM]) {
      const tables = m.model.layers
        .flatMap((l) => l.blocks)
        .filter((b) => b.kind === 'ngram_embedding');
      expect(tables.length).toBeGreaterThan(0);
      expect(
        tables.every((b) => b.kind === 'ngram_embedding' && b.offDeviceByDesign === true),
      ).toBe(true);
    }
    expect(fetches(PLE)).toEqual([]);
    expect(fetches(ENGRAM)).toEqual([]);
  });

  it('PLE (config ple_layer_ids [2], one-based) sits at layer index 1: one 320 B row read, 16 rows per token', () => {
    const f = fetches(PLE, {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
    });
    expect(f.map((x) => x.layer)).toEqual([1]);
    expect(f[0].op.reads).toEqual([{ count: 16, bytesEach: 320 }]); // bf16, no scales: one read whatever the layout
    expect(f[0].op.usefulBytes).toBe(5_120);
  });

  it('Engram (engram_layer_ids [1, 14], zero-based) at indices 1 and 14; useful 264 B per row (256 value + 8 scale)', () => {
    expect(tableRowBytes({ dtype: 'mxfp8' }, 256)).toEqual({
      valueBytes: 256,
      scaleBytes: 8,
      rowBytes: 264,
    });
    const f = fetches(ENGRAM, {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
    });
    expect(f.map((x) => x.layer)).toEqual([1, 14]);
  });

  it('Engram separate on DDR5: reads [{24, 256}, {24, 8}], bytes_tier 24·256 + 24·64 = 7,680, usefulBytes 6,336', () => {
    const [{ op }] = fetches(ENGRAM, {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
    });
    expect(op.reads).toEqual([
      { count: 24, bytesEach: 256 },
      { count: 24, bytesEach: 8 },
    ]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ramPath).tierBytes).toBe(7_680);
  });

  it('Engram pinned_host on NVMe: reads [{24, 256}], bytes_tier 98,304, usefulBytes 6,336; host RAM holds the scales', () => {
    const offload: OffloadLowering = {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'nvme', layers: ALL_LAYERS, scaleLayout: 'pinned_host' }],
    };
    const [{ op }] = fetches(ENGRAM, offload);
    expect(op.source).toBe('nvme');
    expect(op.reads).toEqual([{ count: 24, bytesEach: 256 }]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ssdPath).tierBytes).toBe(98_304);
    // Scales: 384,006,168·8 = 3,072,049,344 B (table 1) + 384,016,682·8 = 3,072,133,456 B (table 2).
    const tiers = stageTierBytes(ENGRAM.model, decodeCtx(ENGRAM.quant, offload), 0);
    expect(tiers.host_ram).toBe(3_072_049_344 + 3_072_133_456);
    expect(tiers.nvme).toBe((384_006_168 + 384_016_682) * 256);
  });

  it('Engram interleaved: reads [{24, 264}], bytes_tier 98,304 on NVMe / 7,680 on DDR5, usefulBytes 6,336', () => {
    const [{ op }] = fetches(ENGRAM, {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'nvme', layers: ALL_LAYERS, scaleLayout: 'interleaved' }],
    });
    expect(op.reads).toEqual([{ count: 24, bytesEach: 264 }]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ssdPath).tierBytes).toBe(98_304);
    expect(transferTime(op.reads, op.usefulBytes, ramPath).tierBytes).toBe(7_680);
  });

  it('the default scale layout is separate on host RAM and pinned_host on NVMe (through simulate)', () => {
    const run = (tier: 'host_ram' | 'nvme') =>
      simulate({
        ...ENGRAM,
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
        workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
        hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
        deployment: { offload: { lookupTables: { tier } } },
      });
    const reads = (tier: 'host_ram' | 'nvme') => {
      const op = run(tier)
        .decode.stages[0].layers.flatMap((l) => l.ops)
        .find((o) => o.op.kind === 'transfer')?.op;
      return op?.kind === 'transfer' ? op.reads.length : 0;
    };
    expect(reads('host_ram')).toBe(2);
    expect(reads('nvme')).toBe(1);
  });

  it('stageTierBreakdown splits tables from routed experts and sums to stageTierBytes', () => {
    const offload: OffloadLowering = {
      prefetchAccuracy: 1,
      lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
      routedExperts: [
        {
          tier: 'host_ram',
          layers: { from: 0, to: 1000 },
          residentFraction: 0.7,
          hitRatio: 0.7,
          includeDraft: false,
        },
      ],
    };
    const ctx = decodeCtx(PLE.quant, offload);
    const b = stageTierBreakdown(PLE.model, ctx, 0).host_ram;
    expect(b.lookupTables).toBe(102_400_000_000);
    expect(b.routedExperts).toBeGreaterThan(0);
    expect(b.draftExperts).toBe(0);
    expect(stageTierBytes(PLE.model, ctx, 0).host_ram).toBe(b.lookupTables + b.routedExperts);
  });

  describe('device tables shared across DP', () => {
    it('Engram TP 2 × DP 2 owns complete hash heads, exchanges IDs/rows, and retains local projections', () => {
      const block = ENGRAM.model.layers[1].blocks.find((b) => b.kind === 'ngram_embedding')!;
      const ctx: LoweringContext = {
        ...decodeCtx(ENGRAM.quant),
        seqs: 3,
        parallel: { tp: 2, pp: 1, dp: 2, ep: 1 },
        lookupTables: { peerReads: false },
      };
      const ops = lowerNgram(block, ctx);
      // Six consecutive prime-sized head buckets, not ceil(totalRows / 4). MXFP8 rows include 8 scale bytes.
      expect(ngramTableShardBytes(block, ctx)).toBe(96_002_514 * 264);
      expect(stageLookupTableBytes(ENGRAM.model, ctx, 0)).toBe((96_002_514 + 96_005_118) * 264);
      expect(ops[0]).toMatchObject({
        group: 'dp',
        collective: 'all_gather',
        messageBytes: 576,
        dtype: 'int32',
      });
      expect(ops[1]).toMatchObject({ kind: 'embedding', tokens: 36, vocabShard: 96_002_514 });
      expect(ops[2]).toMatchObject({
        group: 'dp',
        collective: 'all_gather',
        messageBytes: 36_864,
        workspaceBytes: 74_304,
      });
      expect(ops[3]).toMatchObject({ kind: 'elementwise', fn: 'copy', flopsPerElem: 0 });
      expect(ops[4]).toMatchObject({ group: 'tp', collective: 'all_gather', messageBytes: 36_864 });
      const unshared = lowerNgram(block, { ...ctx, lookupTables: undefined });
      expect(ops.slice(5)).toEqual(unshared.slice(2));
      const lowered = {
        stages: [{ index: 0, layers: [{ name: block.name, layerIndex: 1, ops: ops.slice(0, 5) }] }],
      };
      expect(stageActivationPeakBytes([lowered], 0)).toBe(74_304);
    });

    it('verification shares the target tables while drafter lookup ops and resident bytes stay unchanged', () => {
      const block = ENGRAM.model.layers[1].blocks.find((b) => b.kind === 'ngram_embedding')!;
      const mtp = REF_DENSE_8B_MTP.draft;
      if (mtp?.kind !== 'mtp') throw new Error('MTP fixture needs an MTP head');
      const draft = { ...mtp, layers: [{ name: 'draft.table', blocks: [block] }] };
      const ctx: LoweringContext = {
        ...decodeCtx(ENGRAM.quant),
        qLen: 3,
        parallel: { tp: 1, dp: 2, pp: 1, ep: 1 },
        lookupTables: { peerReads: false },
      };
      const plain = { ...ctx, lookupTables: undefined };
      expect(lowerNgram(block, ctx)[0]).toMatchObject({ group: 'dp', messageBytes: 576 });
      expect(lowerDraftDecode(ENGRAM.model, draft, ctx, 1, 2)).toEqual(
        lowerDraftDecode(ENGRAM.model, draft, plain, 1, 2),
      );
      expect(draftResidentWeightBytes(ENGRAM.model, draft, ctx)).toBe(
        draftResidentWeightBytes(ENGRAM.model, draft, plain),
      );
    });

    it.each(['bf16', 'fp8_e4m3'] as const)(
      'PLE TP 2 × DP 2: %s activations keep reduction buffers at least 16-bit',
      (activations) => {
        const block = PLE.model.layers[1].blocks.find((b) => b.kind === 'ngram_embedding')!;
        const ctx: LoweringContext = {
          ...decodeCtx({ ...PLE.quant, activations }),
          seqs: 3,
          parallel: { tp: 2, pp: 1, dp: 2, ep: 1 },
          lookupTables: { peerReads: false },
        };
        const ops = lowerNgram(block, ctx);
        expect(ngramTableShardBytes(block, ctx)).toBe(102_400_000_000 / 4);
        expect(ops[0]).toMatchObject({
          group: 'dp',
          collective: 'all_gather',
          messageBytes: 768,
          dtype: 'int64',
        });
        expect(ops[1]).toMatchObject({
          kind: 'embedding',
          tokens: 96,
          rowsRead: 24,
          outputDtype: 'bf16',
        });
        expect(opBytes(ops[1])).toMatchObject({ weightBytes: 7_680, activationWriteBytes: 30_720 });
        expect(ops[2]).toMatchObject({
          group: 'tp_dp',
          groupSize: 4,
          collective: 'all_reduce',
          dtype: 'bf16',
          messageBytes: 30_720,
          workspaceBytes: 62_208,
        });
      },
    );
  });

  describe('quant.lookupTables (the lookup-table format picker)', () => {
    const NVFP4 = { dtype: 'nvfp4' } as const;
    const withTables = (
      m: { model: ModelSpec; quant: QuantConfig },
      lookupTables: QuantConfig['lookupTables'],
    ): QuantConfig => ({
      ...m.quant,
      lookupTables,
    });
    const tableParams = (m: { model: ModelSpec }) =>
      m.model.layers
        .flatMap((l) => l.blocks)
        .reduce((n, b) => (b.kind === 'ngram_embedding' ? n + b.tableRows * b.rowDim : n), 0);

    it('PLE nvfp4 table is 51.2e9 params × 4.5 bits / 8 = 28.8e9 B on host RAM; bf16 is 102.4e9 B', () => {
      expect(tableParams(PLE)).toBe(51_200_000_000);
      const off: OffloadLowering = {
        prefetchAccuracy: 1,
        lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
      };
      const bf16 = stageTierBytes(PLE.model, decodeCtx(PLE.quant, off), 0);
      const nvfp4 = stageTierBytes(PLE.model, decodeCtx(withTables(PLE, NVFP4), off), 0);
      expect(bf16.host_ram).toBe(102_400_000_000);
      expect(nvfp4.host_ram).toBe(28_800_000_000);
      // The scale-layout split follows the format: nvfp4 scales are 0.5 bit/elem of the table.
      const pinned = stageTierBreakdown(
        PLE.model,
        decodeCtx(withTables(PLE, NVFP4), {
          prefetchAccuracy: 1,
          lookupTables: [{ tier: 'nvme', layers: ALL_LAYERS, scaleLayout: 'pinned_host' }],
        }),
        0,
      );
      expect(pinned.host_ram.lookupTables).toBe((51_200_000_000 * 0.5) / 8);
      expect(pinned.nvme.lookupTables).toBe((51_200_000_000 * 4) / 8);
    });

    it('resident device bytes follow the override', () => {
      const bf16 = stageLookupTableBytes(PLE.model, decodeCtx(PLE.quant), 0);
      const nvfp4 = stageLookupTableBytes(PLE.model, decodeCtx(withTables(PLE, NVFP4)), 0);
      expect(nvfp4 / bf16).toBeCloseTo(4.5 / 16, 6);
    });

    it('row bytes of the off-device gather follow the override (PLE d 160: bf16 320 B → nvfp4 90 B)', () => {
      const f = fetches(
        { model: PLE.model, quant: withTables(PLE, NVFP4) },
        {
          prefetchAccuracy: 1,
          lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'interleaved' }],
        },
      );
      expect(f[0].op.reads).toEqual([{ count: 16, bytesEach: 90 }]);
      expect(f[0].op.usefulBytes).toBe(16 * 90);
      const sep = fetches(
        { model: PLE.model, quant: withTables(PLE, NVFP4) },
        {
          prefetchAccuracy: 1,
          lookupTables: [{ tier: 'host_ram', layers: ALL_LAYERS, scaleLayout: 'separate' }],
        },
      );
      expect(sep[0].op.reads).toEqual([
        { count: 16, bytesEach: 80 },
        { count: 16, bytesEach: 10 },
      ]);
    });
  });
});
