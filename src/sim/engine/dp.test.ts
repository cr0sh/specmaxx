/**
 * Data parallelism (DP) next to tensor parallelism (TP), end to end through `simulate`, as the UI now offers it.
 *
 * What DP is in the engine. DP runs `dp` independent replicas of the whole model (each its own TP × PP group), so
 * the devices used are `tp·pp·dp` (placement.ts rank layout `rank = tpIdx + tp·(dpIdx + dp·ppIdx)`). The global
 * batch splits evenly: each replica serves `B_r = batch / dp` sequences (validate.ts rejects a batch DP does not
 * divide). Without cross-replica EP or lookup-table sharing the replicas never talk to each other, so
 * one replica at `B_r` runs exactly like a single model instance at `B_r`, and:
 *   aggregate tok/s(dp, batch) = dp · tok/s(one replica at batch/dp)        (pipeline.ts: tokens/s = dp·B_r/cycle)
 *   per-sequence tok/s(dp, batch) = per-sequence tok/s(one replica at batch/dp)
 * The memory fit is per replica: every device of a replica holds its 1/tp shard of the whole model, so DP 2 on
 * 2 GPUs puts the full weights on each GPU where TP 2 puts half.
 *
 * DP 2 against TP 2 on 2 GPUs, dense 8B (REF_DENSE_8B, bf16, decode @ 4k). Per sequence, DP is always slower: a
 * sequence runs on one GPU that reads all 16.4 GB of weights per token, where TP spreads it over both GPUs (each
 * reads 8.2 GB) for the price of two all-reduces per layer. Aggregate decode depends on what bounds the step:
 * - while decode is memory bound (small batch), TP wins on a fast link: both GPUs read half the weights once per
 *   step for the whole batch, DP's GPUs each read all of them for half the batch, so TP moves half the bytes per
 *   token (here 2 × RTX 5090 over PCIe 5.0 P2P, batch 2: TP ≈ 424 tok/s, DP ≈ 228);
 * - DP wins where TP's all-reduces cost more than the weight reads they save: prefill (compute bound, the
 *   all-reduces only add), decode over a slow network (2 machines over 10 GbE TCP, batch ≥ 32), and decode at a
 *   batch large enough to be compute bound (PCIe P2P, batch 512).
 * See docs/simulation-model.md §7 "Pipeline and DP" and docs/modeling-method.md "Multiple devices".
 */
import { describe, expect, it } from 'vitest';
import { ETHERNET_10G_TCP, PCIE_5_X16_DIRECT, NVLINK_2_HCM_8, NVLINK_4 } from '../hw/presets/links';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH } from '../hw/presets/storage';
import { SYNTH_NVLINK } from '../__fixtures__/synthetic';
import { nodeTopology } from '../hw/presets/topologies';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_DENSE_8B } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { simulate } from './simulate';
import type { DeploymentConfig, SimInput, SimResult } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const DENSE = { model: REF_DENSE_8B, quant: BF16 };

function load(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const r = mapHfModel(fixtureRepoFiles(repo));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return { model: r.model, quant: r.quant.quant };
}
const FLASH_NEXT = load('Qwen/Qwen3.8-Flash-Next');
const ENGRAM = load('deepseek-ai/DeepSeek-V4.1-Flash');
const SHARED_TABLES: DeploymentConfig = { offload: { lookupTables: { acrossDp: true } } };

/** n RTX 5090s in one box over direct PCIe 5.0 x16 P2P. */
function box(n: number): SimInput['hardware'] {
  return {
    device: RTX_5090,
    topology: nodeTopology({
      devicesPerNode: n,
      nodes: 1,
      intraNode: PCIE_5_X16_DIRECT,
      device: RTX_5090,
    }),
  };
}

/** n machines of one RTX 5090 each over 10 GbE TCP. */
function overEthernet(n: number): SimInput['hardware'] {
  return {
    device: RTX_5090,
    topology: nodeTopology({
      devicesPerNode: 1,
      nodes: n,
      interNode: ETHERNET_10G_TCP,
      device: RTX_5090,
    }),
  };
}

function run(
  m: { model: ModelSpec; quant: QuantConfig },
  parallel: Partial<ParallelConfig>,
  batch: number,
  hardware: SimInput['hardware'],
  deployment?: DeploymentConfig,
): SimResult {
  const input: SimInput = {
    model: m.model,
    quant: m.quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch, promptLen: 4096, decodeContextLen: 4096 },
    hardware,
    ...(deployment !== undefined ? { deployment } : {}),
  };
  expect(validateInput(input)).toEqual([]);
  return simulate(input);
}

describe('DP 2 on 2 GPUs: two independent replicas', () => {
  it.each([2, 8, 32])(
    'batch %i: aggregate = 2 × one GPU at batch/2; per sequence the same',
    (batch) => {
      const dp = run(DENSE, { dp: 2 }, batch, box(2));
      const one = run(DENSE, {}, batch / 2, box(1));
      expect(dp.devicesUsed).toBe(2);
      for (const phase of ['prefill', 'decode'] as const) {
        expect(dp[phase].tokensPerSec).toBeCloseTo(2 * one[phase].tokensPerSec, 6);
        expect(dp[phase].perSequenceTokensPerSec).toBeCloseTo(
          one[phase].perSequenceTokensPerSec,
          9,
        );
        // No collective: the replicas never talk.
        expect(
          dp[phase].stages
            .flatMap((s) => s.layers.flatMap((l) => l.ops))
            .some((o) => o.op.kind === 'collective'),
        ).toBe(false);
      }
    },
  );

  it('the memory fit is per replica: each GPU holds the whole model under DP, half under TP', () => {
    const dp = run(DENSE, { dp: 2 }, 2, box(2));
    const tp = run(DENSE, { tp: 2 }, 2, box(2));
    const one = run(DENSE, {}, 1, box(1));
    expect(dp.memory.perStage[0].weightBytes).toBe(one.memory.perStage[0].weightBytes);
    expect(tp.memory.perStage[0].weightBytes).toBeLessThan(
      0.51 * dp.memory.perStage[0].weightBytes,
    );
    // Each replica holds B_r = batch / dp sequences; the most it could hold is the single GPU's.
    expect(dp.memory.perStage[0].sequenceStateBytes).toBe(
      one.memory.perStage[0].sequenceStateBytes,
    );
    expect(dp.memory.maxSequencesPerReplica).toBe(one.memory.maxSequencesPerReplica);
  });

  it('a batch DP does not divide is rejected', () => {
    const input: SimInput = {
      model: DENSE.model,
      quant: DENSE.quant,
      parallel: { tp: 1, pp: 1, dp: 2, ep: 1 },
      workload: { batch: 3, promptLen: 4096, decodeContextLen: 4096 },
      hardware: box(2),
    };
    expect(validateInput(input)).toContain('workload.batch (3) must be divisible by dp (2)');
  });
});

describe('DP 2 against TP 2 on 2 GPUs (dense 8B, bf16)', () => {
  // Per sequence, DP is always slower: one GPU reads all the weights for each token (header).
  it.each([2, 8, 32, 128])('batch %i, PCIe P2P: DP is slower per sequence', (batch) => {
    const dp = run(DENSE, { dp: 2 }, batch, box(2));
    const tp = run(DENSE, { tp: 2 }, batch, box(2));
    expect(dp.decode.perSequenceTokensPerSec).toBeLessThan(tp.decode.perSequenceTokensPerSec);
  });

  it('PCIe P2P: TP has the higher aggregate decode while it is memory bound, DP once it is compute bound', () => {
    // Memory bound: TP reads half the weights per GPU for the whole batch, DP all of them for half the batch.
    for (const batch of [2, 8, 32]) {
      const dp = run(DENSE, { dp: 2 }, batch, box(2));
      const tp = run(DENSE, { tp: 2 }, batch, box(2));
      expect(tp.decode.tokensPerSec).toBeGreaterThan(dp.decode.tokensPerSec);
    }
    // Compute bound (batch 512): the weight reads no longer bound the step; TP still pays its all-reduces.
    const dp = run(DENSE, { dp: 2 }, 512, box(2));
    const tp = run(DENSE, { tp: 2 }, 512, box(2));
    expect(dp.decode.tokensPerSec).toBeGreaterThan(tp.decode.tokensPerSec);
    // Prefill is compute bound at any batch: DP avoids the all-reduces.
    expect(dp.prefill.tokensPerSec).toBeGreaterThan(tp.prefill.tokensPerSec);
  });

  it('2 machines over 10 GbE TCP: DP has the higher aggregate decode from batch 32, and higher prefill', () => {
    for (const batch of [32, 128]) {
      const dp = run(DENSE, { dp: 2 }, batch, overEthernet(2));
      const tp = run(DENSE, { tp: 2 }, batch, overEthernet(2));
      expect(dp.decode.tokensPerSec).toBeGreaterThan(tp.decode.tokensPerSec);
      expect(dp.decode.perSequenceTokensPerSec).toBeGreaterThan(tp.decode.perSequenceTokensPerSec);
      expect(dp.prefill.tokensPerSec).toBeGreaterThan(5 * tp.prefill.tokensPerSec);
    }
  });
});

describe('DP with speculative decoding, offload and MoE', () => {
  it('sharing off or DP 1 is identical; legacy table placements and the wrapper simulate identically', () => {
    for (const m of [FLASH_NEXT, ENGRAM]) {
      expect(run(m, {}, 1, box(1), SHARED_TABLES)).toEqual(run(m, {}, 1, box(1)));
      expect(
        run(m, { dp: 2 }, 2, box(2), { offload: { lookupTables: { acrossDp: false } } }),
      ).toEqual(run(m, { dp: 2 }, 2, box(2)));
    }
    const hardware = { ...box(2), host: { ram: DDR5_6000_2CH } };
    const placements = { tier: 'host_ram' } as const;
    const legacy = run(FLASH_NEXT, { dp: 2 }, 2, hardware, {
      offload: { lookupTables: placements },
    });
    expect(
      run(FLASH_NEXT, { dp: 2 }, 2, hardware, { offload: { lookupTables: [placements] } }),
    ).toEqual(legacy);
    expect(
      run(FLASH_NEXT, { dp: 2 }, 2, hardware, { offload: { lookupTables: { placements } } }),
    ).toEqual(legacy);
  });

  it.each([FLASH_NEXT, ENGRAM])(
    'lookup collectives use the existing TP overlap fraction: $model.name',
    (m) => {
      const input: SimInput = {
        ...m,
        parallel: { tp: 1, dp: 2, pp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
        hardware: box(2),
        deployment: SHARED_TABLES,
        options: { collectiveAlgorithm: 'ring' },
      };
      const runWith = (options: SimInput['options']) =>
        simulate({ ...input, options: { ...input.options, ...options } });
      const none = runWith({ overlap: 'none' });
      const tpNone = runWith({ overlapByClass: { tp: 0 } });
      const half = runWith({ overlapByClass: { tp: 0.5 } });
      const full = runWith({ overlapByClass: { tp: 1 } });
      const epNone = runWith({ overlapByClass: { ep: 0 } });
      for (const phase of ['prefill', 'decode'] as const) {
        expect(tpNone[phase].stages).toEqual(none[phase].stages);
        expect(epNone[phase].stages).toEqual(full[phase].stages);
        expect(full[phase].tokensPerSec).toBeGreaterThanOrEqual(half[phase].tokensPerSec);
        expect(half[phase].tokensPerSec).toBeGreaterThanOrEqual(tpNone[phase].tokensPerSec);
        expect(full[phase].stages[0].commLatencySec).toBe(tpNone[phase].stages[0].commLatencySec);
      }
      expect(tpNone.memory).toEqual(full.memory);
    },
  );

  it.each([NVLINK_2_HCM_8, { ...NVLINK_4, fullyConnectedDevices: undefined }])(
    'partial or unknown peer connectivity uses Engram collectives: $name',
    (intraNode) => {
      const hardware = box(8);
      hardware.topology = { ...hardware.topology, intraNode };
      const shared = run(ENGRAM, { dp: 8 }, 8, hardware, SHARED_TABLES);
      for (const phase of [shared.prefill, shared.decode]) {
        const ops = phase.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
        expect(ops.some((o) => o.op.kind === 'collective' && o.op.group === 'dp')).toBe(true);
        expect(ops.some((o) => o.op.kind === 'embedding' && o.op.peerReadBytes !== undefined)).toBe(
          false,
        );
      }
      expect(shared.warnings.some((w) => w.includes('peer-HBM'))).toBe(false);
      expect(
        run(ENGRAM, { dp: 8 }, 8, hardware, { offload: { lookupTables: { acrossDp: false } } }),
      ).toEqual(run(ENGRAM, { dp: 8 }, 8, hardware));
    },
  );

  it('peer-HBM gather time is unchanged by collective overlap overrides', () => {
    const hardware = box(2);
    hardware.topology = { ...hardware.topology, intraNode: SYNTH_NVLINK };
    const input: SimInput = {
      ...ENGRAM,
      parallel: { tp: 1, dp: 2, pp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
      hardware,
      deployment: SHARED_TABLES,
    };
    const plain = simulate(input);
    const serial = simulate({ ...input, options: { overlapByClass: { tp: 0 } } });
    expect(serial.decode.stages).toEqual(plain.decode.stages);
    expect(serial.prefill.stages).toEqual(plain.prefill.stages);
  });

  it.each([FLASH_NEXT, ENGRAM])('sharing runs with PP2 inside one domain: $model.name', (m) => {
    const plain = run(m, { pp: 2, dp: 2 }, 4, box(4));
    const shared = run(m, { pp: 2, dp: 2 }, 4, box(4), SHARED_TABLES);
    expect(shared.memory.perStage).toHaveLength(2);
    expect(shared.decode.stages).toHaveLength(2);
    expect(shared.prefill.tokensPerSec).toBeGreaterThan(0);
    expect(shared.decode.tokensPerSec).toBeGreaterThan(0);
    expect(plain.memory.perStage.some((s) => (s.lookupTableBytes ?? 0) > 0)).toBe(true);
    shared.memory.perStage.forEach((stage, i) => {
      const before = plain.memory.perStage[i];
      if ((before.lookupTableBytes ?? 0) > 0)
        expect(stage.lookupTableBytes).toBeLessThan(before.lookupTableBytes!);
      else expect(stage.lookupTableBytes).toBe(before.lookupTableBytes);
      expect(stage.weightBytes - (stage.lookupTableBytes ?? 0)).toBe(
        before.weightBytes - (before.lookupTableBytes ?? 0),
      );
    });
  });

  it.each(['chunked', 'chunked + TBO', 'chunked + MTP'])(
    'lookup sharing preserves chunk/prefix contexts and workspace peaks: %s',
    (mode) => {
      const mtp = mode === 'chunked + MTP';
      const input: SimInput = {
        ...FLASH_NEXT,
        parallel: { tp: 1, dp: 2, pp: 2, ep: 2, chunkedPipelinePrefill: true },
        workload: {
          batch: 16,
          promptLen: 128,
          decodeContextLen: 256,
          prefillChunkTokens: 32,
          prefixCacheHitRatio: 0.5,
        },
        hardware: box(4),
        deployment: {
          ...SHARED_TABLES,
          ...(mode === 'chunked + TBO' ? { twoBatchOverlap: true } : {}),
          ...(mtp
            ? {
                speculative: {
                  method: 'mtp' as const,
                  draftLength: 2,
                  acceptance: { kind: 'iid' as const, alpha: 0.8 },
                },
              }
            : {}),
        },
      };
      const shared = simulate(input);
      const plain = simulate({ ...input, deployment: { ...input.deployment, offload: undefined } });
      expect(shared.prefill.chunkedPipeline).toBeDefined();
      const g = input.model.layers
        .flatMap((l) => l.blocks)
        .reduce((n, b) => n + (b.kind === 'ngram_embedding' ? b.rowsPerToken : 0), 0);
      for (const phase of ['prefill', 'decode'] as const) {
        const ops = shared[phase].stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
        const ids = ops.flatMap(({ op }) =>
          op.kind === 'collective' && op.name.endsWith('.dp_ids') ? [op] : [],
        );
        const tokens = shared[phase].microbatchSize * (phase === 'prefill' ? 64 : mtp ? 3 : 1);
        expect(ids.reduce((bytes, op) => bytes + op.messageBytes, 0)).toBe(tokens * 2 * g * 8);
        expect(shared[phase].tokensPerSec).toBeGreaterThan(0);
        if (mtp) {
          const draft = (result: SimResult) =>
            result[phase].stages.flatMap((s) =>
              s.layers
                .filter((l) => l.name.startsWith('draft.'))
                .flatMap((l) => l.ops.map((o) => o.op)),
            );
          expect(draft(shared).length).toBeGreaterThan(0);
          expect(draft(shared)).toEqual(draft(plain));
        }
      }
      shared.memory.perStage.forEach((stage, i) => {
        expect(stage.lookupTableBytes).toBe(plain.memory.perStage[i].lookupTableBytes / 2);
        for (const layer of shared.decode.stages[i].layers)
          for (const { op } of layer.ops) {
            if (op.kind === 'collective' && op.workspaceBytes !== undefined)
              expect(stage.activationPeakBytes).toBeGreaterThanOrEqual(op.workspaceBytes);
          }
      });
    },
  );

  it('PLE sharing cuts only table residency, adds inter-node traffic, and needs no offload host', () => {
    const hardware = overEthernet(2);
    const one = run(FLASH_NEXT, { dp: 2 }, 2, hardware);
    const shared = run(FLASH_NEXT, { dp: 2 }, 2, hardware, SHARED_TABLES);
    const before = one.memory.perStage[0],
      after = shared.memory.perStage[0];
    expect(after.lookupTableBytes).toBe(before.lookupTableBytes / 2);
    expect(after.weightBytes - after.lookupTableBytes).toBe(
      before.weightBytes - before.lookupTableBytes,
    );
    expect(after.sequenceStateBytes).toBe(before.sequenceStateBytes);
    expect(after.stagingBytes).toBe(0);
    expect(shared.memory.tiers).toEqual([]);
    expect(shared.devicesUsed).toBe(2);
    for (const phase of [shared.prefill, shared.decode]) {
      const exchange = phase.stages
        .flatMap((s) => s.layers.flatMap((l) => l.ops))
        .filter(
          (o) => o.op.kind === 'collective' && (o.op.group === 'dp' || o.op.group === 'tp_dp'),
        );
      expect(exchange).toHaveLength(2);
      expect(exchange.every((o) => o.linkLevel === 'inter' && o.commBandwidthSec > 0)).toBe(true);
    }
    expect(shared.decode.tokensPerSec).toBeLessThan(one.decode.tokensPerSec);
  });

  it('rejects unmodeled Engram domains and layouts with empty head owners instead of silently dividing by DP', () => {
    const input: SimInput = {
      ...ENGRAM,
      parallel: { tp: 1, dp: 2, pp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
      hardware: overEthernet(2),
      deployment: SHARED_TABLES,
    };
    expect(validateInput(input).some((e) => /Engram.*one node/.test(e))).toBe(true);
    expect(
      validateInput({
        ...input,
        parallel: { tp: 2, dp: 16, pp: 1, ep: 1 },
        workload: { ...input.workload, batch: 16 },
        hardware: box(32),
      }).some((e) => e.includes('without hash heads')),
    ).toBe(true);
  });

  it('Engram TP 1 over NVLink includes peer-HBM bandwidth in the gather, without another op or host tier', () => {
    const hardware = box(2);
    hardware.topology = { ...hardware.topology, intraNode: SYNTH_NVLINK };
    const plain = run(ENGRAM, { dp: 2 }, 2, hardware);
    const shared = run(ENGRAM, { dp: 2 }, 2, hardware, SHARED_TABLES);
    expect(shared.memory.perStage[0].lookupTableBytes).toBe(
      plain.memory.perStage[0].lookupTableBytes / 2,
    );
    expect(shared.memory.tiers).toEqual([]);
    const ops = shared.decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
    const peer = ops.filter((o) => o.op.kind === 'embedding' && o.op.peerReadBytes !== undefined);
    expect(peer).toHaveLength(2);
    for (const p of peer) {
      expect(p.op).toMatchObject({ peerReadBytes: (24 * 264) / 2 });
      expect(p.memorySec).toBe((24 * 264) / 2 / SYNTH_NVLINK.bandwidthPerDirectionBytesPerSec);
      expect(p.commBandwidthSec).toBe(0);
      expect(p.bound).toBe('memory');
    }
    expect(ops.length).toBe(
      plain.decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops)).length,
    );
    expect(ops.some((o) => o.op.kind === 'collective' && o.op.group === 'dp')).toBe(false);
    expect(shared.warnings.some((w) => w.includes('peer-HBM'))).toBe(true);
  });

  it('MTP spec-dec at DP 2: each replica runs the draft and verify of its own sequences', () => {
    const mtp: DeploymentConfig = {
      speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
    };
    const m = { model: REF_DENSE_8B_MTP, quant: BF16 };
    const dp = run(m, { dp: 2 }, 4, box(2), mtp);
    const one = run(m, {}, 2, box(1), mtp);
    expect(dp.decode.speculative?.speedup).toBeCloseTo(one.decode.speculative?.speedup ?? 0, 9);
    expect(dp.decode.tokensPerSec).toBeCloseTo(2 * one.decode.tokensPerSec, 6);
  });

  it('routed experts in host RAM at DP 2: the node holds both replicas\u2019 offloaded experts', () => {
    const offload: DeploymentConfig = {
      offload: {
        routedExperts: { tier: 'host_ram', residentFraction: 0.5, hitRatio: { kind: 'uniform' } },
      },
    };
    const host = { ram: DDR5_6000_2CH };
    const dp = run(FLASH_NEXT, { dp: 2 }, 2, { ...box(2), host }, offload);
    const one = run(FLASH_NEXT, {}, 1, { ...box(1), host }, offload);
    const ram = (r: SimResult) => r.memory.tiers.find((t) => t.tier === 'host_ram')?.bytes ?? 0;
    expect(ram(one)).toBeGreaterThan(0);
    expect(ram(dp)).toBe(2 * ram(one));
    expect(dp.decode.tokensPerSec).toBeLessThanOrEqual(2 * one.decode.tokensPerSec * (1 + 1e-9));
    const shared = run(
      FLASH_NEXT,
      { dp: 2 },
      2,
      { ...box(2), host },
      {
        offload: { ...offload.offload, lookupTables: { acrossDp: true } },
      },
    );
    expect(shared.memory.tiers).toEqual(dp.memory.tiers);
    expect(shared.memory.perStage[0].lookupTableBytes).toBe(
      dp.memory.perStage[0].lookupTableBytes / 2,
    );
  });

  it('MoE mode B: EP 2 at TP 1, DP 2 spans both replicas, with an all-to-all of tokens', () => {
    const r = run(FLASH_NEXT, { dp: 2, ep: 2 }, 2, box(2));
    const ops = r.decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
    expect(ops.some((o) => o.op.kind === 'collective' && o.op.collective === 'all_to_all')).toBe(
      true,
    );
    expect(r.devicesUsed).toBe(2);
  });
});
