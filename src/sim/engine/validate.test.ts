import { describe, expect, it } from 'vitest';
import {
  SYNTH_CLUSTER,
  SYNTH_GPU,
  SYNTH_NVLINK,
  SYNTH_PCIE_HOSTBOUNCE,
  SYNTH_PCIE_HOSTBOUNCE_LINK,
} from '../__fixtures__/synthetic';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { BlockDraftSpec, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B, REF_MOE_30B_A3B, TINY_HC_NGRAM, TINY_HYBRID } from '../models/reference';
import { singleDevice } from '../hw/presets';
import type { TopologySpec } from '../hw/types';
import type {
  DeploymentConfig,
  OffloadConfig,
  RoutedExpertRule,
  SimInput,
  SpecDecodeConfig,
} from './types';
import {
  blockShardingErrors,
  deploymentErrors,
  SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR,
  TIED_EMBEDDINGS_OFFLOAD_ERROR,
  TWO_BATCH_OVERLAP_MODE_B_ERROR,
  TWO_BATCH_OVERLAP_OFFLOAD_ERROR,
  validateInput,
} from './validate';
import { simulate } from './simulate';
import { countParams } from '../ir/params';
import { modelSpecErrors } from '../ir/validateModel';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

function input(parallel: Partial<ParallelConfig> = {}, over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    ...over,
  };
}

/** Asserts exactly that some error line matches `pattern`. */
function expectError(i: SimInput, pattern: RegExp): void {
  const errors = validateInput(i);
  expect(
    errors.some((e) => pattern.test(e)),
    `errors: ${errors.join(' | ')}`,
  ).toBe(true);
}

describe('T17 input errors', () => {
  it('a valid input has no errors', () => {
    expect(validateInput(input())).toEqual([]);
    expect(validateInput(input({ tp: 8 }))).toEqual([]);
  });

  it('shared Engram validates peer capability and modified P2P topology, not just the preset link', () => {
    const model: ModelSpec = {
      ...TINY_HC_NGRAM,
      layers: TINY_HC_NGRAM.layers.map((layer) => ({
        ...layer,
        blocks: layer.blocks.map((b) =>
          b.kind === 'ngram_embedding'
            ? { ...b, headRows: Array.from({ length: b.rowsPerToken }, () => 1) }
            : b,
        ),
      })),
    };
    const i = input(
      { dp: 32 },
      {
        model,
        workload: { batch: 32, promptLen: 128, decodeContextLen: 256 },
        hardware: {
          device: SYNTH_GPU,
          topology: {
            ...SYNTH_CLUSTER,
            devicesPerNode: 72,
            nodes: 1,
            intraNode: { ...SYNTH_NVLINK, fullyConnectedDevices: 72 },
          },
        },
        deployment: { offload: { lookupTables: { acrossDp: true } } },
      },
    );
    expect(validateInput(i)).toEqual([]);
    expectError(
      {
        ...i,
        hardware: {
          ...i.hardware,
          topology: {
            ...i.hardware.topology,
            intraNode: { ...SYNTH_NVLINK, fullyConnectedDevices: undefined },
          },
        },
      },
      /without hash heads/,
    );
    const modified: SimInput = {
      ...i,
      hardware: {
        ...i.hardware,
        linkModifiers: {
          intraNode: [
            {
              kind: 'p2p',
              p2p: 'host_bounce',
              hostBounce: { hostMemBandwidthBytesPerSec: 100e9, extraLatencySec: 1e-6 },
              note: 'Synthetic host-bounce path',
              sources: [],
              estimatedReason: 'Test fixture',
            },
          ],
        },
      },
    };
    expectError(modified, /without hash heads/);
  });

  it.each(['ring', 'chain'] as const)(
    'rejects strided lookup DP groups on a %s through input validation',
    (kind) => {
      expectError(
        input(
          { tp: 2, dp: 2 },
          {
            model: TINY_HC_NGRAM,
            workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
            hardware: {
              device: SYNTH_GPU,
              topology: {
                ...SYNTH_CLUSTER,
                devicesPerNode: 1,
                nodes: 4,
                interNodeFabric: { kind },
              },
            },
            deployment: { offload: { lookupTables: { acrossDp: true } } },
          },
        ),
        /DP group of 2 nodes 2 apart.*not modeled/,
      );
    },
  );

  it('heads % tp', () => {
    // 32 heads over tp 3 (placement also rejects tp 3 on nodes of 8)
    expectError(input({ tp: 3 }), /heads \(32\) must be divisible by tp \(3\)/);
  });

  it('batch % dp', () => {
    expectError(
      input({ dp: 2 }, { workload: { batch: 3, promptLen: 128, decodeContextLen: 256 } }),
      /batch \(3\) must be divisible by dp \(2\)/,
    );
  });

  it('pp > layers', () => {
    const topo: TopologySpec = { ...SYNTH_CLUSTER, nodes: 8 }; // 64 devices, so only the layer rule fails
    expectError(
      input({ pp: 37 }, { hardware: { device: SYNTH_GPU, topology: topo } }),
      /pp \(37\) must not exceed the number of layers \(36\)/,
    );
  });

  it('world > devices', () => {
    expectError(input({ tp: 8, dp: 4 }), /32 devices needed, the topology has 16/);
  });

  it('missing interNode link', () => {
    const noInter: TopologySpec = { ...SYNTH_CLUSTER, interNode: undefined };
    expectError(
      input({ tp: 16 }, { hardware: { device: SYNTH_GPU, topology: noInter } }),
      /no interNode link/,
    );
  });

  it('invalid ep (MoE: 8 experts, ep 3) and ep on a dense model', () => {
    expectError(
      input({ ep: 3 }, { model: TINY_HYBRID }),
      /experts \(8\) must be divisible by ep \(3\)/,
    );
    expectError(input({ ep: 2 }), /model has no MoE block/);
  });

  it('epImbalance is a finite number ≥ 1 (max/mean load); 1 and above are valid', () => {
    for (const bad of [0.9, 0, Number.NaN, Number.POSITIVE_INFINITY]) {
      expectError(
        input({ ep: 2, dp: 2, epImbalance: bad }, { model: TINY_HYBRID }),
        /parallel\.epImbalance must be a finite number ≥ 1/,
      );
    }
    for (const ok of [1, 1.2, 2.56]) {
      const workload = { batch: 2, promptLen: 128, decodeContextLen: 256 };
      expect(
        validateInput(input({ ep: 2, dp: 2, epImbalance: ok }, { model: TINY_HYBRID, workload })),
      ).toEqual([]);
    }
  });

  it('redundantExpertsPerRank is a whole number ≥ 0; 0 and above are valid', () => {
    for (const bad of [-1, 0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expectError(
        input({ ep: 2, dp: 2, redundantExpertsPerRank: bad }, { model: TINY_HYBRID }),
        /parallel\.redundantExpertsPerRank must be a whole number ≥ 0/,
      );
    }
    for (const ok of [0, 1, 32]) {
      const workload = { batch: 2, promptLen: 128, decodeContextLen: 256 };
      expect(
        validateInput(
          input({ ep: 2, dp: 2, redundantExpertsPerRank: ok }, { model: TINY_HYBRID, workload }),
        ),
      ).toEqual([]);
    }
  });

  it('epImbalanceScalesAllToAll is a boolean', () => {
    const bad = input(
      { ep: 2, dp: 2, epImbalanceScalesAllToAll: 1 as unknown as boolean },
      { model: TINY_HYBRID },
    );
    expectError(bad, /parallel\.epImbalanceScalesAllToAll must be true or false/);
    for (const ok of [true, false]) {
      const workload = { batch: 2, promptLen: 128, decodeContextLen: 256 };
      expect(
        validateInput(
          input({ ep: 2, dp: 2, epImbalanceScalesAllToAll: ok }, { model: TINY_HYBRID, workload }),
        ),
      ).toEqual([]);
    }
  });

  it('unknown engineId', () => {
    expectError(
      input(
        {},
        { hardware: { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU), engineId: 'npu' } },
      ),
      /engineId 'npu'/,
    );
  });

  it('non-integer degrees and empty workloads', () => {
    expectError(input({ tp: 0 }), /parallel\.tp must be an integer ≥ 1/);
    expectError(
      input({}, { workload: { batch: 1, promptLen: 0, decodeContextLen: 256 } }),
      /promptLen must be an integer ≥ 1/,
    );
  });

  it('an unmodified device is validated too (review P2): zero memory bandwidth, zero peak, no engines, unsourced', () => {
    const on = (device: SimInput['hardware']['device']): SimInput =>
      input({}, { hardware: { device, topology: singleDevice(device) } });
    expectError(
      on({ ...SYNTH_GPU, memory: { ...SYNTH_GPU.memory, bandwidthBytesPerSec: 0 } }),
      /memory\.bandwidthBytesPerSec must be > 0/,
    );
    const zeroPeak = {
      ...SYNTH_GPU.engines[0],
      matmulPeaks: SYNTH_GPU.engines[0].matmulPeaks.map((p) => ({ ...p, flopsPerSec: 0 })),
    };
    expectError(
      on({ ...SYNTH_GPU, engines: [zeroPeak] }),
      /matmulPeaks\.0\.flopsPerSec must be > 0/,
    );
    expectError(on({ ...SYNTH_GPU, engines: [] }), /no engines/);
    expectError(on({ ...SYNTH_GPU, sources: [] }), /is not covered by any sources/);
  });

  it('a host-bounce link with zero host bandwidth is rejected', () => {
    const link = {
      ...SYNTH_PCIE_HOSTBOUNCE_LINK,
      hostBounce: { hostMemBandwidthBytesPerSec: 0, extraLatencySec: 5e-6 },
    };
    expectError(
      input(
        { tp: 2 },
        {
          hardware: { device: SYNTH_GPU, topology: { ...SYNTH_PCIE_HOSTBOUNCE, intraNode: link } },
        },
      ),
      /hostMemBandwidthBytesPerSec must be > 0/,
    );
  });

  it('value heads must divide tp (review P2): GDN vHeads 2 at tp 4', () => {
    const model = decoderModel({
      name: 'gdn',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: false,
      layers: [
        decoderLayer({
          hidden: 256,
          mixer: {
            kind: 'linear',
            variant: 'gated_delta_net',
            qkHeads: 1,
            vHeads: 2,
            keyDim: 64,
            valueDim: 64,
            outputGate: true,
          },
          ffn: { kind: 'dense', intermediate: 512 },
        }),
      ],
    });
    expectError(input({ tp: 4 }, { model }), /vHeads \(2\) must be divisible by tp \(4\)/);
    expect(validateInput(input({ tp: 2 }, { model }))).toEqual([]);
  });
  it('invalid modifiers', () => {
    expectError(
      input(
        {},
        {
          hardware: {
            device: SYNTH_GPU,
            topology: SYNTH_CLUSTER,
            deviceModifiers: [
              { kind: 'clockScale', factor: 0, note: 'x', sources: [], estimatedReason: 'test' },
            ],
          },
        },
      ),
      /factor must be a finite number > 0/,
    );
    expectError(
      input(
        {},
        {
          hardware: {
            device: SYNTH_GPU,
            topology: singleDevice(SYNTH_GPU),
            linkModifiers: {
              intraNode: [
                {
                  kind: 'linkBandwidth',
                  bandwidthPerDirectionBytesPerSec: 1e9,
                  note: 'x',
                  sources: [],
                  estimatedReason: 'test',
                },
              ],
            },
          },
        },
      ),
      /has no intraNode link/,
    );
  });
});

describe('blockShardingErrors', () => {
  it('reports sharding errors from model, quant and parallel alone, prefixed with the layer name', () => {
    const errors = blockShardingErrors({
      model: REF_DENSE_8B,
      quant: BF16,
      parallel: { tp: 3, pp: 1, dp: 1, ep: 1 },
    });
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toMatch(/^layer\.0: .*must be divisible by tp \(3\)/);
    expect(
      blockShardingErrors({
        model: REF_DENSE_8B,
        quant: BF16,
        parallel: { tp: 8, pp: 1, dp: 1, ep: 1 },
      }),
    ).toEqual([]);
  });

  it('is part of validateInput', () => {
    const sharding = blockShardingErrors({
      model: REF_DENSE_8B,
      quant: BF16,
      parallel: { tp: 3, pp: 1, dp: 1, ep: 1 },
    });
    const all = validateInput(input({ tp: 3 }));
    for (const e of sharding) expect(all).toContain(e);
  });
});

describe('deployment knobs (design doc §2, §3.6; U-3, U-13)', () => {
  /** A DFlash-2-shaped block drafter (blockSize 8: k ≤ 7). */
  const DRAFTER: BlockDraftSpec = {
    kind: 'block_drafter',
    method: 'dflash2',
    hidden: 4096,
    vocab: 151_936,
    layers: 5,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    intermediate: 12_288,
    gated: true,
    activation: 'silu',
    windowTokens: 2048,
    targetLayerIds: [1, 9, 17, 25, 33],
    blockSize: 8,
    source: 'hypothetical',
  };
  const WITH_MTP: ModelSpec = {
    ...REF_DENSE_8B,
    draft: { kind: 'mtp', layers: [REF_DENSE_8B.layers[0]], hiddenProj: true },
  };
  const spec = (over: Partial<SpecDecodeConfig> = {}): SpecDecodeConfig => ({
    method: 'dflash2',
    drafter: { kind: 'hypothetical', spec: DRAFTER },
    draftLength: 7,
    acceptance: { kind: 'iid', alpha: 0.79 },
    ...over,
  });
  /** TINY_HYBRID (MoE) on a dedicated device with an explicit host link. */
  const withHost = { device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: SYNTH_NVLINK } };
  const offload = (o: OffloadConfig, model: ModelSpec = TINY_HYBRID): SimInput =>
    input({}, { model, hardware: withHost, deployment: { offload: o } });
  const experts = (over: Partial<RoutedExpertRule> = {}): RoutedExpertRule => ({
    tier: 'host_ram',
    residentFraction: 0.5,
    hitRatio: { kind: 'uniform' },
    ...over,
  });
  const errorsOf = (i: SimInput) => deploymentErrors(i);

  it('absent or {} deployment: no errors, and validateInput is unchanged', () => {
    for (const deployment of [
      undefined,
      {},
      { offload: {} },
      { offload: { lookupTables: { tier: 'device' } } },
    ] as (DeploymentConfig | undefined)[]) {
      expect(validateInput(input({}, { deployment }))).toEqual([]);
    }
  });

  it('U-13: spec-dec with pp 2 is valid; with the token embedding off-device it gets the exact error', () => {
    expect(validateInput(input({ pp: 2 }, { deployment: { speculative: spec() } }))).toEqual([]);
    const offloaded = input(
      { pp: 2 },
      { deployment: { speculative: spec(), offload: { embeddings: { tier: 'host_ram' } } } },
    );
    expect(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR).toBe(
      'speculative decoding with pp > 1 and the token embedding off-device is not supported',
    );
    expect(errorsOf(offloaded)).toContain(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR);
    expect(errorsOf(input({}, { deployment: { speculative: spec() } }))).toEqual([]);
  });

  it("U-13: offload mode 'cpu_compute' is rejected (designed hook, §2.8)", () => {
    expect(errorsOf(offload({ mode: 'cpu_compute', routedExperts: experts() }))).toEqual([
      "deployment.offload.mode 'cpu_compute' is not supported yet (only 'transfer')",
    ]);
    expect(errorsOf(offload({ mode: 'transfer', routedExperts: experts() }))).toEqual([]);
  });

  it('U-13: k + 1 > blockSize is rejected (blockSize 8: k 7 ok, k 8 not)', () => {
    expect(errorsOf(input({}, { deployment: { speculative: spec({ draftLength: 7 }) } }))).toEqual(
      [],
    );
    expect(errorsOf(input({}, { deployment: { speculative: spec({ draftLength: 8 }) } }))).toEqual([
      "deployment.speculative.draftLength 8: k + 1 must not exceed the drafter's blockSize (8)",
    ]);
  });

  it('draftLength is an integer in 1..16', () => {
    const mtp = (draftLength: number) =>
      errorsOf(
        input(
          {},
          { model: WITH_MTP, deployment: { speculative: spec({ method: 'mtp', draftLength }) } },
        ),
      );
    expect(mtp(1)).toEqual([]);
    expect(mtp(16)).toEqual([]);
    for (const k of [0, 17, 1.5])
      expect(mtp(k)).toEqual([
        `deployment.speculative.draftLength must be an integer in 1..16 (got ${k})`,
      ]);
  });

  it("'mtp' needs the model's MTP head; block methods need a drafter of their own method", () => {
    expect(
      errorsOf(input({}, { deployment: { speculative: spec({ method: 'mtp', draftLength: 1 }) } })),
    ).toEqual([
      "deployment.speculative.method 'mtp' needs a model with an MTP draft head (model.draft.kind 'mtp')",
    ]);
    expect(
      errorsOf(input({}, { deployment: { speculative: spec({ drafter: undefined }) } })),
    ).toEqual(["deployment.speculative.method 'dflash2' needs a drafter"]);
    expect(
      errorsOf(input({}, { deployment: { speculative: spec({ method: 'dspark' }) } })),
    ).toEqual(['deployment.speculative.drafter is a dflash2 drafter, not dspark']);
  });

  it('acceptance ranges: i.i.d. α in [0, 1); per-position α_j in [0, 1], non-empty', () => {
    const acc = (acceptance: SpecDecodeConfig['acceptance']) =>
      errorsOf(input({}, { deployment: { speculative: spec({ acceptance }) } }));
    expect(acc({ kind: 'iid', alpha: 0 })).toEqual([]);
    expect(acc({ kind: 'iid', alpha: 0.999 })).toEqual([]);
    expect(acc({ kind: 'iid', alpha: 1 })).toEqual([
      'deployment.speculative.acceptance.alpha must be in [0, 1) (got 1)',
    ]);
    expect(acc({ kind: 'iid', alpha: -0.1 })).toHaveLength(1);
    expect(acc({ kind: 'iid', alpha: Number.NaN })).toHaveLength(1);
    expect(acc({ kind: 'per_position', alphas: [1] })).toEqual([]); // E[τ] = k + 1 is legal
    expect(acc({ kind: 'per_position', alphas: [0.854, 0.803, 0] })).toEqual([]);
    expect(acc({ kind: 'per_position', alphas: [0.9, 1.1] })).toHaveLength(1);
    expect(acc({ kind: 'per_position', alphas: [] })).toEqual([
      'deployment.speculative.acceptance.alphas must not be empty',
    ]);
  });

  it('U-3: residentFraction in [0, 1]; custom h in [0, 1] and h > 0 with f = 0 rejected; f = 1 accepts any h', () => {
    expect(errorsOf(offload({ routedExperts: experts({ residentFraction: 1.2 }) }))).toEqual([
      'deployment.offload.routedExperts.residentFraction must be in [0, 1] (got 1.2)',
    ]);
    expect(
      errorsOf(
        offload({
          routedExperts: experts({ residentFraction: 0, hitRatio: { kind: 'custom', h: 0.2 } }),
        }),
      ),
    ).toEqual(['deployment.offload.routedExperts.hitRatio: no experts are resident, h must be 0']);
    expect(
      errorsOf(
        offload({
          routedExperts: experts({ residentFraction: 0, hitRatio: { kind: 'custom', h: 0 } }),
        }),
      ),
    ).toEqual([]);
    expect(errorsOf(offload({ routedExperts: experts({ residentFraction: 0 }) }))).toEqual([]); // uniform: h(0) = 0
    expect(
      errorsOf(
        offload({
          routedExperts: experts({ residentFraction: 1, hitRatio: { kind: 'custom', h: 0.3 } }),
        }),
      ),
    ).toEqual([]);
    expect(
      errorsOf(offload({ routedExperts: experts({ hitRatio: { kind: 'custom', h: 1.5 } }) })),
    ).toEqual(['deployment.offload.routedExperts.hitRatio.h must be in [0, 1] (got 1.5)']);
  });

  it('layer ranges are non-empty, integer and inside [0, L); queueDepth is an integer ≥ 1', () => {
    const L = TINY_HYBRID.layers.length;
    expect(errorsOf(offload({ routedExperts: experts({ layers: { from: 0, to: L } }) }))).toEqual(
      [],
    );
    expect(
      errorsOf(
        offload({ lookupTables: { tier: 'nvme', layers: { from: 1, to: 2 } } }, TINY_HC_NGRAM),
      ),
    ).toEqual([]);
    for (const layers of [
      { from: 0, to: 0 },
      { from: -1, to: 2 },
      { from: 0, to: L + 1 },
      { from: 2, to: 1 },
      { from: 0.5, to: 2 },
    ]) {
      expect(
        errorsOf(offload({ lookupTables: { tier: 'host_ram', layers } }, TINY_HC_NGRAM)),
      ).toHaveLength(1);
      expect(errorsOf(offload({ routedExperts: experts({ layers }) }))).toHaveLength(1);
    }
    expect(errorsOf(offload({ queueDepth: 1, routedExperts: experts() }))).toEqual([]);
    for (const queueDepth of [0, 1.5]) {
      expect(errorsOf(offload({ queueDepth, routedExperts: experts() }))).toEqual([
        `deployment.offload.queueDepth must be an integer ≥ 1 (got ${queueDepth})`,
      ]);
    }
  });

  it('routed experts need a MoE model; off-device lookup tables need n-gram tables', () => {
    expect(errorsOf(offload({ routedExperts: experts() }, REF_DENSE_8B))).toEqual([
      'deployment.offload.routedExperts: the model has no MoE block',
    ]);
    expect(errorsOf(offload({ lookupTables: { tier: 'host_ram' } }, REF_DENSE_8B))).toEqual([
      'deployment.offload.lookupTables: the model has no n-gram lookup table',
    ]);
    expect(errorsOf(offload({ lookupTables: { tier: 'device' } }, REF_DENSE_8B))).toEqual([]);
    expect(errorsOf(offload({ lookupTables: { tier: 'host_ram' } }, TINY_HC_NGRAM))).toEqual([]);
    expect(
      errorsOf(
        offload({ lookupTables: { tier: 'nvme', scaleLayout: 'pinned_host' } }, TINY_HC_NGRAM),
      ),
    ).toEqual([]);
    expect(
      errorsOf(
        offload({ lookupTables: { tier: 'host_ram', scaleLayout: 'pinned_host' } }, TINY_HC_NGRAM),
      ),
    ).toEqual(["deployment.offload.lookupTables.scaleLayout 'pinned_host' needs tier 'nvme'"]);
  });

  it.each(['true', 1, null])('rejects non-boolean table sharing: %j', (acrossDp) => {
    expect(
      errorsOf(
        offload({ lookupTables: { acrossDp: acrossDp as unknown as boolean } }, TINY_HC_NGRAM),
      ),
    ).toContain('deployment.offload.lookupTables.acrossDp must be a boolean');
  });

  it.each(['acrossDP', 'placement', 'scaleLayout', 'layers'])(
    'rejects unknown table config key %s rather than treating it as device placement',
    (key) => {
      const lookupTables = { [key]: 'typo' } as OffloadConfig['lookupTables'];
      expect(errorsOf(offload({ lookupTables }, TINY_HC_NGRAM))).toContain(
        `deployment.offload.lookupTables.${key}: unknown key (expected acrossDp or placements)`,
      );
    },
  );

  it('sharing requires device-only table placements, not device-only experts or embeddings', () => {
    const lookupTables = { acrossDp: true };
    expect(errorsOf(offload({ lookupTables }, TINY_HC_NGRAM))).toEqual([]);
    expect(
      errorsOf(input({}, { model: TINY_HC_NGRAM, deployment: { offload: { lookupTables } } })),
    ).toEqual([]);
    expect(
      errorsOf(
        offload(
          { lookupTables: { ...lookupTables, placements: { tier: 'device' } } },
          TINY_HC_NGRAM,
        ),
      ),
    ).toEqual([]);
    expect(
      errorsOf(offload({ lookupTables, embeddings: { tier: 'host_ram' } }, TINY_HC_NGRAM)),
    ).toEqual([]);
    for (const tier of ['host_ram', 'nvme'] as const) {
      expect(
        errorsOf(
          offload({ lookupTables: { ...lookupTables, placements: { tier } } }, TINY_HC_NGRAM),
        ),
      ).toContain(
        'deployment.offload.lookupTables.acrossDp requires every lookup-table placement to be device',
      );
      expect(
        errorsOf(
          offload(
            {
              lookupTables: {
                ...lookupTables,
                placements: [
                  { tier: 'device', layers: { from: 0, to: 1 } },
                  { tier, layers: { from: 1, to: 2 } },
                ],
              },
            },
            TINY_HC_NGRAM,
          ),
        ),
      ).toContain(
        'deployment.offload.lookupTables.acrossDp requires every lookup-table placement to be device',
      );
    }
    expect(errorsOf(offload({ lookupTables }, REF_DENSE_8B))).toEqual([
      'deployment.offload.lookupTables.acrossDp: the model has no n-gram lookup table',
    ]);
  });

  it('an off-device group on a dedicated device without a host link (no host.link, no hostInterface) is rejected', () => {
    const noHost = input(
      {},
      { model: TINY_HYBRID, deployment: { offload: { routedExperts: experts() } } },
    );
    expect(SYNTH_GPU.hostInterface).toBeUndefined();
    expect(errorsOf(noHost)).toEqual([
      'deployment.offload: no host link known for synth-gpu; set hardware.host.link',
    ]);
    // Nothing off-device: no link needed.
    expect(
      errorsOf(input({}, { deployment: { offload: { lookupTables: { tier: 'device' } } } })),
    ).toEqual([]);
    // A cited edge connector is enough.
    const withInterface = { ...SYNTH_GPU, hostInterface: { generation: 5, lanes: 16 } as const };
    expect(
      errorsOf({ ...noHost, hardware: { device: withInterface, topology: SYNTH_CLUSTER } }),
    ).toEqual([]);
  });

  it('token embedding off-device: rejected on a tied model, any tier on an untied one; a bad tier is an error', () => {
    const tied: ModelSpec = { ...REF_DENSE_8B, tieEmbeddings: true };
    expect(TIED_EMBEDDINGS_OFFLOAD_ERROR).toBe(
      'deployment.offload.embeddings: the model ties its token embedding to the LM head (tie_word_embeddings), so the table stays on the device',
    );
    for (const tier of ['host_ram', 'nvme'] as const) {
      expect(errorsOf(offload({ embeddings: { tier } }, tied))).toEqual([
        TIED_EMBEDDINGS_OFFLOAD_ERROR,
      ]);
      expect(errorsOf(offload({ embeddings: { tier } }, REF_DENSE_8B))).toEqual([]);
    }
    expect(errorsOf(offload({ embeddings: { tier: 'device' } }, tied))).toEqual([]);
    const bad = { embeddings: { tier: 'gpu' } } as unknown as OffloadConfig;
    expect(errorsOf(offload(bad, REF_DENSE_8B))).toEqual([
      `deployment.offload.embeddings.tier must be 'device', 'host_ram' or 'nvme' (got "gpu")`,
    ]);
    // An off-device embedding alone needs a host link on a dedicated device too.
    const noHost = input({}, { deployment: { offload: { embeddings: { tier: 'host_ram' } } } });
    expect(errorsOf(noHost)).toEqual([
      'deployment.offload: no host link known for synth-gpu; set hardware.host.link',
    ]);
  });

  describe('placement rule lists (mixed host RAM + SSD by layer range)', () => {
    const ram = (from: number, to: number) => experts({ layers: { from, to } });
    const ssd = (from: number, to: number) => experts({ tier: 'nvme', layers: { from, to } });

    it('disjoint ranges in any order are valid; [] is valid (nothing offloaded, so no host link is needed either)', () => {
      expect(errorsOf(offload({ routedExperts: [ram(0, 2), ssd(2, 4)] }))).toEqual([]);
      expect(errorsOf(offload({ routedExperts: [ssd(2, 4), ram(0, 2)] }))).toEqual([]);
      expect(errorsOf(offload({ routedExperts: [ram(0, 1), ssd(3, 4)] }))).toEqual([]);
      expect(errorsOf(offload({ routedExperts: [] }, REF_DENSE_8B))).toEqual([]);
      expect(errorsOf(offload({ lookupTables: [] }, REF_DENSE_8B))).toEqual([]);
      const noHost = input(
        {},
        { model: TINY_HYBRID, deployment: { offload: { routedExperts: [], lookupTables: [] } } },
      );
      expect(errorsOf(noHost)).toEqual([]);
    });

    it('a one-rule list without layers means every layer, as the object form does', () => {
      expect(errorsOf(offload({ routedExperts: [experts()] }))).toEqual([]);
    });

    it('overlapping ranges are rejected, naming both rules (list order does not matter)', () => {
      expect(errorsOf(offload({ routedExperts: [ram(0, 3), ssd(2, 4)] }))).toEqual([
        'deployment.offload.routedExperts[0].layers [0, 3) overlaps deployment.offload.routedExperts[1].layers [2, 4)',
      ]);
      expect(errorsOf(offload({ routedExperts: [ssd(2, 4), ram(0, 3)] }))).toEqual([
        'deployment.offload.routedExperts[0].layers [2, 4) overlaps deployment.offload.routedExperts[1].layers [0, 3)',
      ]);
      expect(
        errorsOf(
          offload(
            {
              lookupTables: [
                { tier: 'host_ram', layers: { from: 0, to: 2 } },
                { tier: 'nvme', layers: { from: 1, to: 2 } },
              ],
            },
            TINY_HC_NGRAM,
          ),
        ),
      ).toEqual([
        'deployment.offload.lookupTables[0].layers [0, 2) overlaps deployment.offload.lookupTables[1].layers [1, 2)',
      ]);
    });

    it('a rule without layers in a list of two or more is rejected', () => {
      expect(errorsOf(offload({ routedExperts: [experts(), ssd(2, 4)] }))).toEqual([
        'deployment.offload.routedExperts[0].layers is required when there is more than one rule',
      ]);
      expect(
        errorsOf(
          offload(
            { lookupTables: [{ tier: 'host_ram' }, { tier: 'nvme', layers: { from: 1, to: 2 } }] },
            TINY_HC_NGRAM,
          ),
        ),
      ).toEqual([
        'deployment.offload.lookupTables[0].layers is required when there is more than one rule',
      ]);
    });

    it('includeDraft on more than one rule is rejected; on one rule it is valid', () => {
      expect(
        errorsOf(offload({ routedExperts: [{ ...ram(0, 2), includeDraft: true }, ssd(2, 4)] })),
      ).toEqual([]);
      expect(
        errorsOf(
          offload({
            routedExperts: [
              { ...ram(0, 2), includeDraft: true },
              { ...ssd(2, 4), includeDraft: true },
            ],
          }),
        ),
      ).toEqual([
        'deployment.offload.routedExperts: includeDraft may be set on one rule only (rules 0, 1)',
      ]);
    });

    it('per-rule field errors name the rule index; the object form keeps its original paths', () => {
      expect(
        errorsOf(offload({ routedExperts: [ram(0, 2), { ...ssd(2, 4), residentFraction: 1.2 }] })),
      ).toEqual([
        'deployment.offload.routedExperts[1].residentFraction must be in [0, 1] (got 1.2)',
      ]);
      expect(errorsOf(offload({ routedExperts: [ram(0, 2), ssd(2, 5)] }))).toEqual([
        'deployment.offload.routedExperts[1].layers [2, 5) must be a non-empty range inside [0, 4)',
      ]);
      expect(
        errorsOf(
          offload({ lookupTables: [{ tier: 'nvme', layers: { from: 2, to: 2 } }] }, TINY_HC_NGRAM),
        ),
      ).toEqual([
        'deployment.offload.lookupTables[0].layers [2, 2) must be a non-empty range inside [0, 2)',
      ]);
      expect(
        errorsOf(
          offload(
            {
              lookupTables: [
                { tier: 'host_ram', layers: { from: 1, to: 2 }, scaleLayout: 'pinned_host' },
              ],
            },
            TINY_HC_NGRAM,
          ),
        ),
      ).toEqual(["deployment.offload.lookupTables[0].scaleLayout 'pinned_host' needs tier 'nvme'"]);
      // A table list on a model without tables: one error for the group; device rules alone are fine.
      expect(
        errorsOf(
          offload({ lookupTables: [{ tier: 'nvme', layers: { from: 0, to: 1 } }] }, REF_DENSE_8B),
        ),
      ).toEqual(['deployment.offload.lookupTables: the model has no n-gram lookup table']);
      expect(
        errorsOf(
          offload({ lookupTables: [{ tier: 'device', layers: { from: 0, to: 1 } }] }, REF_DENSE_8B),
        ),
      ).toEqual([]);
      expect(errorsOf(offload({ routedExperts: { ...ram(0, 2), residentFraction: 1.2 } }))).toEqual(
        ['deployment.offload.routedExperts.residentFraction must be in [0, 1] (got 1.2)'],
      );
    });

    it('the removed whole-layer placement (denseLayers) is rejected, not silently ignored', () => {
      const old = {
        denseLayers: { tier: 'host_ram', layers: { from: 0, to: 2 } },
      } as unknown as OffloadConfig;
      expect(errorsOf(offload(old))).toEqual([
        'deployment.offload.denseLayers was removed: whole-layer streaming is no longer modeled (offload MoE experts instead)',
      ]);
    });
  });

  it("the model's own draft head stays out of countParams, modelSpecErrors and the result while spec-dec is off", () => {
    expect(countParams(WITH_MTP)).toBe(countParams(REF_DENSE_8B));
    expect(modelSpecErrors(WITH_MTP)).toEqual(modelSpecErrors(REF_DENSE_8B));
    const plain = simulate(input());
    const withDraft = simulate(input({}, { model: WITH_MTP }));
    expect(withDraft).toEqual(plain);
  });

  it('an invalid hardware.host is reported with its path (validateHost)', () => {
    const badLink = { ...SYNTH_NVLINK, bandwidthPerDirectionBytesPerSec: 0 };
    const errors = errorsOf(
      input(
        {},
        { hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: badLink } } },
      ),
    );
    expect(errors.some((e) => e.startsWith('hardware.host.link'))).toBe(true);
  });
});

describe('per-collective overlap and two-batch overlap (docs §7 "Stage time and overlap", "Two-batch overlap")', () => {
  it('options.overlapByClass: classes tp, ep, pp with finite fractions in [0, 1]', () => {
    expect(
      validateInput(input({}, { options: { overlapByClass: { tp: 0, ep: 0.5, pp: 1 } } })),
    ).toEqual([]);
    expectError(
      input({}, { options: { overlapByClass: { tp: 1.5 } } }),
      /^options\.overlapByClass\.tp \(1\.5\) must be a finite number in \[0, 1\]$/,
    );
    expectError(input({}, { options: { overlapByClass: { ep: -0.1 } } }), /overlapByClass\.ep/);
    expectError(
      input({}, { options: { overlapByClass: { pp: NaN } } }),
      /overlapByClass\.pp \(NaN\)/,
    );
    expectError(
      input({}, { options: { overlapByClass: { dp: 1 } as never } }),
      /^options\.overlapByClass\.dp: unknown overlap class \(expected tp, ep or pp\)$/,
    );
  });
  const moe = (parallel: Partial<ParallelConfig>, deployment: DeploymentConfig) =>
    input(parallel, {
      model: REF_MOE_30B_A3B,
      workload: { batch: 4, promptLen: 128, decodeContextLen: 256 },
      deployment,
    });
  it('two-batch overlap is valid for a MoE model in mode B (ep > tp)', () => {
    expect(validateInput(moe({ ep: 2, dp: 2 }, { twoBatchOverlap: true }))).toEqual([]);
    expect(validateInput(moe({}, { twoBatchOverlap: false }))).toEqual([]);
  });
  it('two-batch overlap needs a MoE model with ep > tp', () => {
    expect(validateInput(input({}, { deployment: { twoBatchOverlap: true } }))).toContain(
      TWO_BATCH_OVERLAP_MODE_B_ERROR,
    );
    expect(validateInput(moe({}, { twoBatchOverlap: true }))).toContain(
      TWO_BATCH_OVERLAP_MODE_B_ERROR,
    );
    expect(validateInput(moe({ tp: 2, ep: 2 }, { twoBatchOverlap: true }))).toContain(
      TWO_BATCH_OVERLAP_MODE_B_ERROR,
    );
  });
  it('two-batch overlap with weights offloaded is not modelled', () => {
    const offload: OffloadConfig = {
      routedExperts: { tier: 'host_ram', residentFraction: 0.5, hitRatio: { kind: 'uniform' } },
    };
    expect(validateInput(moe({ ep: 2, dp: 2 }, { twoBatchOverlap: true, offload }))).toContain(
      TWO_BATCH_OVERLAP_OFFLOAD_ERROR,
    );
    // An offload block that keeps everything on the device is fine.
    expect(
      validateInput(
        moe({ ep: 2, dp: 2 }, { twoBatchOverlap: true, offload: { prefetchAccuracy: 1 } }),
      ),
    ).toEqual([]);
  });
});

describe('network options (docs §6 "In-network reduction (NVLS, SHARP)", "Multi-NIC nodes")', () => {
  it("inNetworkReduction must be 'off' or 'on'", () => {
    expect(validateInput(input({}, { options: { inNetworkReduction: 'on' } }))).toEqual([]);
    expect(validateInput(input({}, { options: { inNetworkReduction: 'off' } }))).toEqual([]);
    expect(
      validateInput(input({}, { options: { inNetworkReduction: 'yes' as unknown as 'on' } })),
    ).toEqual(["options.inNetworkReduction ('yes') must be 'off' or 'on'"]);
  });

  it('an invalid nicsPerNode is a topology error', () =>
    expectError(
      input(
        {},
        { hardware: { device: SYNTH_GPU, topology: { ...SYNTH_CLUSTER, nicsPerNode: 3 } } },
      ),
      /nicsPerNode \(3\) must divide devicesPerNode/,
    ));
});
