import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_IB, SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import type {
  DeviceSpec,
  HostMemorySpec,
  HostSpec,
  LinkSpec,
  NvmeSpec,
  TopologySpec,
} from './types';
import {
  validateDevice,
  validateHost,
  validateLink,
  validateStorageTier,
  validateTopology,
} from './validate';

/** Four single-device nodes on {@link SYNTH_IB}. */
const FOUR_NODES: TopologySpec = {
  id: 'four',
  name: 'four',
  devicesPerNode: 1,
  nodes: 4,
  interNode: SYNTH_IB,
  sources: [],
};

describe('validateTopology: interNodeFabric', () => {
  it('accepts every shape on single-device nodes, and an oversubscribed switch on multi-device nodes', () => {
    for (const f of [
      { kind: 'switched' },
      { kind: 'full_mesh', cablesPerPair: 2 },
      { kind: 'ring' },
      { kind: 'chain' },
    ] as const) {
      expect(validateTopology({ ...FOUR_NODES, interNodeFabric: f })).toEqual([]);
    }
    expect(
      validateTopology({
        ...SYNTH_CLUSTER,
        interNodeFabric: { kind: 'switched', oversubscription: 2 },
      }),
    ).toEqual([]);
  });

  it('a switchless shape with devicesPerNode > 1 is an error', () => {
    const errors = validateTopology({ ...SYNTH_CLUSTER, interNodeFabric: { kind: 'ring' } });
    expect(errors).toHaveLength(2); // one device per node; a ring of 2 nodes
    expect(errors.some((e) => e.includes('one device per node'))).toBe(true);
  });

  it('oversubscription < 1 is an error', () => {
    expect(
      validateTopology({
        ...FOUR_NODES,
        interNodeFabric: { kind: 'switched', oversubscription: 0.5 },
      }),
    ).toHaveLength(1);
  });

  it('a fabric without an interNode link, or on one node, is an error', () => {
    const { interNode: _dropped, ...noLink } = FOUR_NODES;
    void _dropped;
    expect(validateTopology({ ...noLink, interNodeFabric: { kind: 'chain' } })).toEqual([
      'topology four: interNodeFabric needs an interNode link',
    ]);
    expect(
      validateTopology({ ...FOUR_NODES, nodes: 1, interNodeFabric: { kind: 'switched' } }),
    ).toEqual(['topology four: interNodeFabric needs nodes > 1 (got 1)']);
  });

  it('a ring of 2 nodes is an error', () => {
    expect(
      validateTopology({ ...FOUR_NODES, nodes: 2, interNodeFabric: { kind: 'ring' } }),
    ).toHaveLength(1);
  });
});

describe('validateTopology: nicsPerNode', () => {
  it('accepts divisors and multiples of devicesPerNode on a switched network', () => {
    for (const nicsPerNode of [1, 2, 4, 8, 16])
      expect(validateTopology({ ...SYNTH_CLUSTER, nicsPerNode })).toEqual([]);
    expect(
      validateTopology({
        ...SYNTH_CLUSTER,
        nicsPerNode: 4,
        interNodeFabric: { kind: 'switched', oversubscription: 2 },
      }),
    ).toEqual([]);
  });

  it('3 NICs on 8 GPUs is an error (rail symmetry)', () => {
    const errors = validateTopology({ ...SYNTH_CLUSTER, nicsPerNode: 3 });
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/must divide devicesPerNode \(8\)/);
    expect(validateTopology({ ...SYNTH_CLUSTER, nicsPerNode: 0 })).toHaveLength(1);
  });

  it('without an interNode link, on one node, or on a switchless fabric, is an error', () => {
    const { interNode: _dropped, ...noLink } = SYNTH_CLUSTER;
    void _dropped;
    expect(validateTopology({ ...noLink, nicsPerNode: 4 })).toEqual([
      'topology synth-cluster: nicsPerNode needs an interNode link',
    ]);
    expect(validateTopology({ ...SYNTH_CLUSTER, nodes: 1, nicsPerNode: 4 })).toEqual([
      'topology synth-cluster: nicsPerNode needs nodes > 1 (got 1)',
    ]);
    expect(
      validateTopology({ ...FOUR_NODES, nicsPerNode: 2, interNodeFabric: { kind: 'ring' } }),
    ).toEqual([
      'topology four: nicsPerNode needs a switched fabric (got ring, whose ports are its cables)',
    ]);
  });
});

describe('validateLink: inNetworkReduction', () => {
  it("accepts 'nvls' and 'sharp', rejects anything else", () => {
    expect(validateLink({ ...SYNTH_IB, inNetworkReduction: 'sharp' })).toEqual([]);
    expect(validateLink({ ...SYNTH_IB, inNetworkReduction: 'nvls' })).toEqual([]);
    expect(
      validateLink({ ...SYNTH_IB, inNetworkReduction: 'shrap' as unknown as 'sharp' }),
    ).toEqual([
      "link Synthetic InfiniBand: inNetworkReduction must be 'nvls' or 'sharp' (got 'shrap')",
    ]);
  });
});

describe('validateLink: the new fabric fields', () => {
  const sourced = (l: Partial<LinkSpec>): LinkSpec => ({
    ...SYNTH_IB,
    ...l,
    sources: [
      {
        ...SYNTH_IB.sources[0],
        fields: [
          'bandwidthPerDirectionBytesPerSec',
          'latencySec',
          'allPortsCapPerDirectionBytesPerSec',
          'forwardingLatencySec',
          'fullyConnectedDevices',
        ],
      },
    ],
  });

  it('full peer-domain size must be a sourced safe integer of at least two devices', () => {
    expect(validateLink(sourced({ fullyConnectedDevices: 72 }))).toEqual([]);
    expect(
      validateLink({ ...SYNTH_IB, fullyConnectedDevices: 8 }).some((e) =>
        /fullyConnectedDevices.*not covered/.test(e),
      ),
    ).toBe(true);
    for (const size of [
      0,
      1,
      2.5,
      NaN,
      Infinity,
      Number.MAX_SAFE_INTEGER + 1,
      '8' as unknown as number,
    ]) {
      expect(validateLink(sourced({ fullyConnectedDevices: size }))).toContain(
        'link Synthetic InfiniBand: fullyConnectedDevices must be a safe integer ≥ 2',
      );
    }
  });

  it('accepts a cap ≥ the per-port bandwidth and a relay latency', () => {
    expect(
      validateLink(
        sourced({ allPortsCapPerDirectionBytesPerSec: 50e9, forwardingLatencySec: 1e-6 }),
      ),
    ).toEqual([]);
    expect(validateLink(sourced({ allPortsCapPerDirectionBytesPerSec: 100e9 }))).toEqual([]);
  });

  it('a cap below the per-port bandwidth is an error', () => {
    const errors = validateLink(sourced({ allPortsCapPerDirectionBytesPerSec: 25e9 }));
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain('allPortsCapPerDirectionBytesPerSec');
  });

  it('the new fields must be sourced or estimated, finite and ≥ 0', () => {
    const unsourced: LinkSpec = {
      ...SYNTH_IB,
      allPortsCapPerDirectionBytesPerSec: 50e9,
      forwardingLatencySec: 1e-6,
    };
    expect(validateLink(unsourced).filter((e) => e.includes('not covered'))).toHaveLength(2);
    expect(
      validateLink(sourced({ forwardingLatencySec: -1 })).some((e) =>
        e.includes('forwardingLatencySec must be finite'),
      ),
    ).toBe(true);
  });
});

/** A synthetic host-RAM tier, every number marked estimated (one subtree estimate per field). */
const SYNTH_RAM: HostMemorySpec = {
  kind: 'host_ram',
  id: 'synth-ram',
  name: 'synthetic RAM',
  capacityBytes: 64e9,
  bandwidthBytesPerSec: 100e9,
  latencySec: 100e-9,
  channels: 2,
  readGranularityBytes: 64,
  maxOutstanding: 64,
  sources: [],
  estimated: [
    'capacityBytes',
    'bandwidthBytesPerSec',
    'latencySec',
    'channels',
    'readGranularityBytes',
    'maxOutstanding',
  ].map((field) => ({ field, reason: 'synthetic' })),
};

/** A synthetic SSD tier: 10 GB/s, 50 µs, 1M IOPS at QD32 × 16, a 7.88 GB/s slot. */
const SYNTH_SSD: NvmeSpec = {
  kind: 'nvme',
  id: 'synth-ssd',
  name: 'synthetic SSD',
  capacityBytes: 1e12,
  seqReadBandwidthBytesPerSec: 10e9,
  randomReadLatencySec: 50e-6,
  randomRead: { iops: 1e6, queueDepth: 32, threads: 16 },
  maxOutstanding: 512,
  readGranularityBytes: 4096,
  linkBandwidthBytesPerSec: 7.88e9,
  sources: [
    {
      url: 'https://example.com/ssd',
      title: 'synthetic datasheet',
      fields: ['capacityBytes', 'seqReadBandwidthBytesPerSec', 'randomRead'],
    },
  ],
  estimated: [
    'randomReadLatencySec',
    'maxOutstanding',
    'readGranularityBytes',
    'linkBandwidthBytesPerSec',
  ].map((field) => ({
    field,
    reason: 'synthetic',
  })),
};

describe('validateStorageTier', () => {
  it('accepts well-formed host-RAM and SSD tiers, with or without randomRead', () => {
    expect(validateStorageTier(SYNTH_RAM)).toEqual([]);
    expect(validateStorageTier(SYNTH_SSD)).toEqual([]);
    const { randomRead: _r, ...noIops } = SYNTH_SSD;
    void _r;
    expect(validateStorageTier(noIops)).toEqual([]);
  });

  it('rejects zero, negative and non-finite rates, sizes and latencies', () => {
    expect(validateStorageTier({ ...SYNTH_RAM, bandwidthBytesPerSec: 0 })).toEqual([
      'host_ram synth-ram: bandwidthBytesPerSec must be finite and > 0 (got 0)',
    ]);
    expect(
      validateStorageTier({ ...SYNTH_RAM, latencySec: Infinity }).some((e) =>
        e.includes('latencySec must be finite and > 0'),
      ),
    ).toBe(true);
    expect(
      validateStorageTier({ ...SYNTH_SSD, seqReadBandwidthBytesPerSec: -1 }).some((e) =>
        e.includes('seqReadBandwidthBytesPerSec must be finite and > 0'),
      ),
    ).toBe(true);
    expect(
      validateStorageTier({ ...SYNTH_SSD, randomRead: { iops: 0, queueDepth: 32, threads: 16 } }),
    ).toEqual(['nvme synth-ssd: randomRead.iops must be finite and > 0 (got 0)']);
    expect(validateStorageTier({ ...SYNTH_SSD, linkBandwidthBytesPerSec: 0 })).toEqual([
      'nvme synth-ssd: linkBandwidthBytesPerSec must be finite and > 0 (got 0)',
    ]);
  });

  it('maxOutstanding, granularity, channels and the rated condition must be integers ≥ 1', () => {
    expect(validateStorageTier({ ...SYNTH_SSD, maxOutstanding: 0 })).toEqual([
      'nvme synth-ssd: maxOutstanding must be an integer ≥ 1 (got 0)',
    ]);
    expect(validateStorageTier({ ...SYNTH_RAM, maxOutstanding: 1.5 })).toEqual([
      'host_ram synth-ram: maxOutstanding must be an integer ≥ 1 (got 1.5)',
    ]);
    expect(validateStorageTier({ ...SYNTH_RAM, channels: 0 })).toEqual([
      'host_ram synth-ram: channels must be an integer ≥ 1 (got 0)',
    ]);
    expect(
      validateStorageTier({
        ...SYNTH_SSD,
        randomRead: { iops: 1e6, queueDepth: 0.5, threads: 16 },
      }),
    ).toEqual(['nvme synth-ssd: randomRead.queueDepth must be an integer ≥ 1 (got 0.5)']);
  });

  it('every number must be sourced or estimated', () => {
    const errors = validateStorageTier({ ...SYNTH_SSD, estimated: [] });
    expect(errors).toHaveLength(4);
    expect(errors.every((e) => e.includes('not covered'))).toBe(true);
  });
});

describe('validateHost', () => {
  it('accepts an empty host and a full one', () => {
    expect(validateHost({})).toEqual([]);
    expect(validateHost({ link: SYNTH_IB, ram: SYNTH_RAM, ssd: SYNTH_SSD })).toEqual([]);
  });

  it('labels errors by slot and rejects a tier in the wrong slot', () => {
    expect(validateHost({ ram: { ...SYNTH_RAM, latencySec: 0 } })).toEqual([
      'host.ram: latencySec must be finite and > 0 (got 0)',
    ]);
    const swapped = { ram: SYNTH_SSD } as unknown as HostSpec;
    expect(validateHost(swapped)).toContain("host.ram: must be a 'host_ram' tier (got 'nvme')");
    expect(
      validateHost({ link: { ...SYNTH_IB, bandwidthPerDirectionBytesPerSec: 0 } }).some((e) =>
        e.startsWith('host.link:'),
      ),
    ).toBe(true);
  });
});

describe('validateDevice: hostInterface', () => {
  const withInterface = (hostInterface: unknown): DeviceSpec =>
    ({
      ...SYNTH_GPU,
      hostInterface,
      sources: [
        ...SYNTH_GPU.sources,
        { url: 'https://example.com/card', title: 'synthetic', fields: ['hostInterface'] },
      ],
    }) as DeviceSpec;

  it('accepts a cited PCIe interface', () => {
    expect(validateDevice(withInterface({ generation: 5, lanes: 16 }))).toEqual([]);
    expect(validateDevice(withInterface({ generation: 1, lanes: 4 }))).toEqual([]); // CMP 170HX stock
  });

  it('rejects an unknown generation or lane count, and an uncited one', () => {
    expect(validateDevice(withInterface({ generation: 6, lanes: 2 }))).toEqual([
      'device synth-gpu.hostInterface: generation must be 1, 2, 3, 4 or 5 (got 6)',
      'device synth-gpu.hostInterface: lanes must be 4, 8 or 16 (got 2)',
    ]);
    expect(
      validateDevice({ ...SYNTH_GPU, hostInterface: { generation: 4, lanes: 8 } }).filter((e) =>
        e.includes('not covered'),
      ),
    ).toHaveLength(2);
  });
});

describe('validateDevice: gpuDirectRdma', () => {
  it('is accepted on dedicated memory and rejected on unified memory', () => {
    expect(validateDevice({ ...SYNTH_GPU, gpuDirectRdma: false })).toEqual([]);
    expect(validateDevice(SYNTH_UNIFIED)).toEqual([]);
    expect(validateDevice({ ...SYNTH_UNIFIED, gpuDirectRdma: true })).toEqual([
      `device ${SYNTH_UNIFIED.id}: gpuDirectRdma applies to dedicated memory only (unified memory never stages); leave it absent`,
    ]);
  });
});

describe('validateDevice: sfuOpsPerSec and gemvMaxRows', () => {
  const withEngine = (extra: object): DeviceSpec => ({
    ...SYNTH_GPU,
    engines: [{ ...SYNTH_GPU.engines[0], ...extra }],
  });
  it('accepts a positive SFU rate and an integer row limit (covered by the engines source)', () => {
    expect(validateDevice(withEngine({ sfuOpsPerSec: 1e12, gemvMaxRows: 4 }))).toEqual([]);
  });
  it('rejects a zero, negative or NaN SFU rate', () => {
    for (const v of [0, -1, NaN, Infinity]) {
      expect(
        validateDevice(withEngine({ sfuOpsPerSec: v })).some((e) =>
          e.includes('sfuOpsPerSec must be a finite number > 0'),
        ),
      ).toBe(true);
    }
  });
  it('rejects a row limit that is not an integer ≥ 1', () => {
    for (const v of [0, 1.5, -2]) {
      expect(
        validateDevice(withEngine({ gemvMaxRows: v })).some((e) =>
          e.includes('gemvMaxRows must be an integer ≥ 1'),
        ),
      ).toBe(true);
    }
  });
});
