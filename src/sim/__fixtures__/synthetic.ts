/**
 * Synthetic hardware for tests: round made-up numbers so expected values can be derived by hand. NOT presets
 * (never offered in the UI); every number is covered by the `synthetic://test` source so `validateDevice`
 * still passes on them.
 */

import type {
  ComputeEngine,
  DeviceSpec,
  GemmTiling,
  LinkSpec,
  SourceRef,
  TopologySpec,
} from '../hw/types';

/** Citation used by every fixture: covers whole subtrees (made-up numbers). */
function syntheticSource(fields: string[]): SourceRef {
  return {
    url: 'synthetic://test',
    title: 'Synthetic test fixture (made-up round numbers)',
    fields,
  };
}

/** Tiling of the synthetic GPU: MMA m64 n8 k16, one 128×128 CTA tile, one CTA per unit. */
export const SYNTH_TILING: GemmTiling = {
  mmaM: 64,
  mmaN: 8,
  mmaK: 16,
  ctaTiles: [{ m: 128, n: 128 }],
  ctasPerUnit: 1,
};

/** The synthetic GPU engine: 100 units; bf16/fp16 fp32-acc 100e12 dense / 200e12 2:4; fp16-acc, fp8, int8 200e12. */
const SYNTH_GPU_ENGINE: ComputeEngine = {
  id: 'gpu',
  name: 'Synthetic GPU',
  kind: 'gpu',
  computeUnits: 100,
  matmulPeaks: [
    { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 100e12 },
    { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: 200e12 },
    { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 100e12 },
    { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: 200e12 },
    { input: 'fp16', accumulate: 'fp16', sparsity: 'dense', flopsPerSec: 200e12 },
    { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 200e12 },
    { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: 200e12 },
  ],
  vectorFlopsPerSec: { fp32: 10e12, bf16: 20e12 },
  gemmTiling: { default: SYNTH_TILING },
};

/** Synthetic dedicated-memory GPU: 100 units, 100 GB, 1 TB/s (see {@link SYNTH_GPU_ENGINE} for the peaks). */
export const SYNTH_GPU: DeviceSpec = {
  id: 'synth-gpu',
  name: 'Synthetic GPU',
  vendor: 'Synthetic',
  engines: [SYNTH_GPU_ENGINE],
  memory: {
    kind: 'dedicated',
    technology: 'synthetic',
    capacityOptionsBytes: [100e9],
    bandwidthBytesPerSec: 1e12,
  },
  sources: [syntheticSource(['engines', 'memory'])],
  estimated: [],
};

/**
 * Synthetic unified-memory SoC: options 32/64/128 GB, 0.5 TB/s shared. Engines: `gpu` (bf16 50e12, vector
 * fp32 5e12) and `npu` (int8 100e12, NO bf16 peak, vector fp32 1e12, memory capped at 0.25 TB/s).
 */
export const SYNTH_UNIFIED: DeviceSpec = {
  id: 'synth-unified',
  name: 'Synthetic unified SoC',
  vendor: 'Synthetic',
  engines: [
    {
      id: 'gpu',
      name: 'Synthetic SoC GPU',
      kind: 'gpu',
      computeUnits: 40,
      matmulPeaks: [{ input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 50e12 }],
      vectorFlopsPerSec: { fp32: 5e12 },
      gemmTiling: { default: SYNTH_TILING },
    },
    {
      id: 'npu',
      name: 'Synthetic NPU',
      kind: 'npu',
      computeUnits: 8,
      matmulPeaks: [{ input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: 100e12 }],
      vectorFlopsPerSec: { fp32: 1e12 },
      gemmTiling: { default: SYNTH_TILING },
      memBandwidthCapBytesPerSec: 0.25e12,
    },
  ],
  memory: {
    kind: 'unified',
    technology: 'synthetic',
    capacityOptionsBytes: [32e9, 64e9, 128e9],
    bandwidthBytesPerSec: 0.5e12,
  },
  sources: [syntheticSource(['engines', 'memory'])],
  estimated: [],
};

/** Synthetic fully connected 8-device NVLink domain: 100 GB/s per direction, α 1 µs, direct P2P. */
export const SYNTH_NVLINK: LinkSpec = {
  kind: 'nvlink',
  name: 'Synthetic NVLink',
  bandwidthPerDirectionBytesPerSec: 100e9,
  latencySec: 1e-6,
  p2p: 'direct',
  fullyConnectedDevices: 8,
  sources: [
    syntheticSource(['bandwidthPerDirectionBytesPerSec', 'latencySec', 'fullyConnectedDevices']),
  ],
  estimated: [],
};

/** Synthetic InfiniBand-class inter-node link: 50 GB/s per direction, α 5 µs, direct (RDMA). */
export const SYNTH_IB: LinkSpec = {
  kind: 'infiniband',
  name: 'Synthetic InfiniBand',
  bandwidthPerDirectionBytesPerSec: 50e9,
  latencySec: 5e-6,
  p2p: 'direct',
  sources: [syntheticSource(['bandwidthPerDirectionBytesPerSec', 'latencySec'])],
  estimated: [],
};

/** 2 nodes × 8 devices: {@link SYNTH_NVLINK} inside a node, {@link SYNTH_IB} between nodes. */
export const SYNTH_CLUSTER: TopologySpec = {
  id: 'synth-cluster',
  name: 'Synthetic 2 × 8 cluster',
  devicesPerNode: 8,
  nodes: 2,
  intraNode: SYNTH_NVLINK,
  interNode: SYNTH_IB,
  sources: [syntheticSource(['devicesPerNode', 'nodes'])],
};

/**
 * PCIe 5.0 x16 without P2P, staged through host memory: 64 GB/s per direction (PCI-SIG: 32 GT/s × 16 lanes
 * ≈ 64e9 B/s, 128b/130b encoding ignored), α 2 µs; host staging 80 GB/s shared, +5 µs per message.
 * Fixture only: host memory bandwidth is platform-specific.
 */
export const SYNTH_PCIE_HOSTBOUNCE_LINK: LinkSpec = {
  kind: 'pcie',
  name: 'PCIe 5.0 x16 (host bounce)',
  bandwidthPerDirectionBytesPerSec: 64e9,
  latencySec: 2e-6,
  p2p: 'host_bounce',
  hostBounce: { hostMemBandwidthBytesPerSec: 80e9, extraLatencySec: 5e-6 },
  sources: [syntheticSource(['bandwidthPerDirectionBytesPerSec', 'latencySec', 'hostBounce'])],
  estimated: [],
};

/** 1 node × 2 devices over {@link SYNTH_PCIE_HOSTBOUNCE_LINK}. */
export const SYNTH_PCIE_HOSTBOUNCE: TopologySpec = {
  id: 'synth-pcie-hostbounce',
  name: 'Synthetic 2 × PCIe (no P2P)',
  devicesPerNode: 2,
  nodes: 1,
  intraNode: SYNTH_PCIE_HOSTBOUNCE_LINK,
  sources: [syntheticSource(['devicesPerNode', 'nodes'])],
};
