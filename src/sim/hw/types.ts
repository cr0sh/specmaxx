/**
 * Hardware spec schema: devices (compute engines + memory), interconnect links, topologies, and the stack a
 * simulation runs on. Plain data; presets live in `hw/presets/`, variants are expressed as modifiers
 * (`hw/modifiers.ts`). See docs/simulation-model.md §4.
 *
 * Conventions:
 * - All numbers are SI decimal (see `hw/units.ts`).
 * - Compute peaks and memory bandwidth are PER DEVICE.
 * - Link bandwidth is PER DEVICE, PER DIRECTION (unidirectional injection rate), never the bidirectional sum.
 * - Sparse peaks count dense-equivalent FLOP (the vendor convention: a 2:4 peak is ~2× the dense one).
 * - Every number in a preset is covered by a `sources[].fields` path or an `estimated[]` entry.
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceModifier, LinkModifier } from './modifiers';

/** A citation for spec numbers. */
export interface SourceRef {
  /** Where the numbers come from (vendor datasheet, whitepaper, spec). */
  url: string;
  /** Human title of the source document. */
  title: string;
  /**
   * Dotted field paths this source covers, relative to the spec object it sits on (e.g.
   * `'memory.bandwidthBytesPerSec'`, `'engines.0.matmulPeaks'`). A path covers its whole subtree.
   */
  fields: string[];
  /** Optional note (page, table, how a number was derived from the quoted one). */
  note?: string;
  /**
   * What kind of evidence the source is. Absent means `'vendor'`.
   * - `'vendor'`: a datasheet, whitepaper, ISA manual or official spec page of the vendor.
   * - `'review'`: an independent technical review or database that measured or compiled the number
   *   (Chips and Cheese, TechPowerUp, ServeTheHome, ...).
   * - `'community'`: a community measurement or report (forum post, GitHub issue, benchmark repo, seller
   *   listing), i.e. NOT a datasheet number. The UI should flag numbers that rest on these.
   */
  kind?: 'vendor' | 'review' | 'community';
}

/** A spec number no vendor publishes, with the reason for the chosen value. */
export interface EstimatedField {
  /** Dotted field path (covers its subtree), as in {@link SourceRef.fields}. */
  field: string;
  /** Why this value was chosen and from what. */
  reason: string;
}

/**
 * One matmul peak of an engine, keyed by (input, accumulate, sparsity).
 * Per device. Unit: FLOP/s for float inputs, OP/s for integer inputs. Sparse (`'2:4'`) entries count
 * dense-equivalent FLOP/s (vendor convention).
 */
export interface MatmulPeak {
  /** Matmul input class. */
  input: MathFormat;
  /** Accumulator type. */
  accumulate: AccumulateType;
  /** `'dense'`, or `'2:4'` structured sparsity of one operand. */
  sparsity: 'dense' | '2:4';
  /** Peak rate, per device. Unit: FLOP/s (OP/s for integer inputs). */
  flopsPerSec: number;
}

/**
 * Tensor-core tiling parameters for the shape-efficiency model (docs §4, `hw/efficiency.ts`). Dimensions in
 * elements of the GEMM's m, n, k.
 */
export interface GemmTiling {
  /** M granularity of one MMA instruction (rows are padded up to it). Unit: elements. */
  mmaM: number;
  /** N granularity of one MMA instruction. Unit: elements. */
  mmaN: number;
  /** K granularity of one MMA instruction. Unit: elements. */
  mmaK: number;
  /** Candidate CTA (thread-block) output tiles; the model takes the fastest. Each m/n a multiple of mmaM/mmaN. Unit: elements. */
  ctaTiles: { m: number; n: number }[];
  /** Concurrent CTAs per compute unit (wave size = computeUnits · ctasPerUnit). Count. */
  ctasPerUnit: number;
}

/** One matrix engine of a device (GPU tensor cores, an NPU, a TPU MXU, CPU matrix units). A run uses exactly one. */
export interface ComputeEngine {
  /** Stable id, unique within the device (e.g. `'gpu'`, `'npu'`). */
  id: string;
  /** Display name. */
  name: string;
  /** Engine family (informational). */
  kind: 'gpu' | 'npu' | 'tpu' | 'cpu' | 'other';
  /** Compute units (SMs, CUs, cores): the wave granularity of the efficiency model. Count. */
  computeUnits: number;
  /** Matmul peaks; every float input listed must have an `fp32`-accumulate entry. Per device. */
  matmulPeaks: MatmulPeak[];
  /** Non-tensor (vector/SIMT) peaks by dtype, per device. Unit: FLOP/s. */
  vectorFlopsPerSec: Partial<Record<'fp64' | 'fp32' | 'fp16' | 'bf16', number>>;
  /** Tiling parameters: `default`, optionally overridden per input format (MMA k differs by format). */
  gemmTiling: { default: GemmTiling } & Partial<Record<MathFormat, GemmTiling>>;
  /**
   * If this engine cannot reach the device memory peak (e.g. an NPU on a narrower fabric), its own cap on
   * memory bandwidth, per device. Unit: B/s (read + write combined, like {@link MemorySpec.bandwidthBytesPerSec}).
   */
  memBandwidthCapBytesPerSec?: number;
  /**
   * Special-function-unit peak (NVIDIA SFU / MUFU, AMD transcendental unit, Intel extended-math, Apple's SIMD
   * transcendental path) for FP32 `exp2`, per device. Unit: results/s (ops/s; one op = one exp2 result).
   * Costs the softmax exponential of attention: one exp2 per score (the e^x → 2^(x·log2 e) scale folds into the
   * preceding FMA), on a pipe separate from the vector FMA lanes counted in `vectorFlopsPerSec`
   * (docs/simulation-model.md §7 "Attention softmax"). Absent = softmax not costed, the convention before it was
   * modelled; every preset sets it (enforced by a test).
   */
  sfuOpsPerSec?: number;
  /**
   * Largest GEMM row count m (tokens per batch entry) this engine runs as a GEMV kernel: SIMD dot products on the
   * vector ALUs, no MMA instruction, so m is not padded to the MMA shape and no tile / wave quantization applies.
   * Such a GEMM (dense, fp16 / bf16 / fp32 math) costs `min(MMA time, F / vectorPeak(math))`
   * (docs/simulation-model.md §4 "GEMV path"). Absent = the MMA path only. Unit: rows (tokens).
   */
  gemvMaxRows?: number;
}

/**
 * Device memory. `'dedicated'` = memory private to the accelerator (GDDR, HBM); `'unified'` = one pool shared
 * by CPU and accelerators (Apple M-series, DGX Spark, Strix Halo), so the OS takes part of it.
 */
export interface MemorySpec {
  /** Dedicated or unified. */
  kind: 'dedicated' | 'unified';
  /** Memory technology, e.g. `'GDDR7'`, `'HBM3'`, `'LPDDR5X'` (informational). */
  technology: string;
  /** Capacity of each SKU configuration, per device; `[0]` is the default. Non-empty. Unit: bytes. */
  capacityOptionsBytes: number[];
  /** Peak memory bandwidth per device, read + write combined, as vendors quote it. Unit: B/s. */
  bandwidthBytesPerSec: number;
  /**
   * Unified memory only: the share of the pool the GPU may allocate BY DEFAULT, as GPU-usable bytes ÷ installed bytes
   * (the OS keeps the rest): the macOS GPU wired limit (`recommendedMaxWorkingSetSize`, ≈ 75% on Macs with 36 GB or
   * more), the Linux amdgpu GTT limit (≈ 50% of RAM) on Strix Halo, what CUDA reports as total on DGX Spark. The fit
   * checks against `f · capacity` unless `HardwareStack.gpuUsableFraction` overrides it (a raised limit). Absent =
   * 1 (the whole pool). Not allowed on dedicated memory. Dimensionless, (0, 1]. See docs/simulation-model.md §8.
   */
  gpuUsableFraction?: number;
}

/** One accelerator device (or one SoC). All numbers per device. */
export interface DeviceSpec {
  /** Stable id, e.g. `'rtx-5090'`. */
  id: string;
  /** Display name. */
  name: string;
  /** Vendor name, e.g. `'NVIDIA'`. */
  vendor: string;
  /** Matrix engines (≥ 1); `engines[0]` is the default. */
  engines: ComputeEngine[];
  /** Device memory. */
  memory: MemorySpec;
  /** Last-level cache size. Recorded for display; NOT used by the v1 model. Unit: bytes. */
  l2Bytes?: number;
  /**
   * The card's PCIe edge connector to the host (generation and ELECTRICAL lane count, e.g. Arc Pro B60: x16
   * connector, x8 electrical ⇒ `lanes: 8`). Cited preset data, used to build `HostSpec.link` with
   * `hostPcieLink(generation, lanes)` when the stack gives none. Absent on SXM/OAM modules and SoCs, whose host
   * attachment no datasheet states as a slot (the engine then rejects off-device placement unless
   * `HardwareStack.host.link` is set). See docs/simulation-model.md §4 ("Host and storage tiers").
   */
  hostInterface?: HostInterfaceSpec;
  /**
   * Non-PCIe CPU attachment (e.g. Grace NVLink-C2C), per GPU, per direction in B/s; latency in seconds.
   * Preferred over hostInterface when hardware.host.link is absent. Its own sources/estimates cover its numbers.
   * Host RAM remains one pooled tier per scale-up domain; NUMA placement is not modeled (docs/simulation-model.md §4 "Host and storage tiers").
   */
  hostLink?: LinkSpec;
  /**
   * Dedicated-memory devices: whether an RDMA NIC may DMA straight into this device's memory (NVIDIA GPUDirect
   * RDMA: data-center and RTX/Quadro parts, not GeForce; AMD Instinct: PeerDirect). `false` ⇒ every inter-node
   * message is staged through host memory, whatever the link (docs/simulation-model.md §6 "Inter-node host
   * staging", `engine/staging.ts`). Absent on unified-memory devices (the NIC reads the shared pool, nothing to
   * stage). Absent on a dedicated device = true, the convention before host staging was modelled; every dedicated
   * preset sets it (enforced by a test). A capability flag, no unit.
   */
  gpuDirectRdma?: boolean;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
  /** Set by `applyDeviceModifiers`: the modifiers this spec was derived with (provenance). Absent on base presets. */
  appliedModifiers?: DeviceModifier[];
}

/**
 * A PCIe generation: Gen1 2.5, Gen2 5, Gen3 8, Gen4 16, Gen5 32 GT/s per lane (Gen1 / Gen2 with 8b/10b line coding,
 * Gen3 and later with 128b/130b; `hw/presets/links.ts: pcieLineRateBytesPerSec`). Gen1 / Gen2 exist for cards whose
 * link is fused down (NVIDIA CMP 170HX: Gen1 stock, Gen2 after a driver patch). Integer, no unit.
 */
export type PcieGeneration = 1 | 2 | 3 | 4 | 5;

/** Electrical PCIe lane count of a link or edge connector: 4, 8 or 16. Count. */
export type PcieLanes = 4 | 8 | 16;

/**
 * A PCIe host interface: generation ({@link PcieGeneration}) and electrical lane count.
 * Integers, no unit; the bandwidth they imply is `hostPcieLink(generation, lanes).bandwidthPerDirectionBytesPerSec`.
 * A `hostInterface` device modifier (`hw/modifiers.ts`) can change either field (a PCIe speed or width unlock).
 */
export interface HostInterfaceSpec {
  /** PCIe generation (1–5). */
  generation: PcieGeneration;
  /** Electrical lanes (4, 8 or 16). Count. */
  lanes: PcieLanes;
}

/**
 * One interconnect level between devices. Bandwidth is PER DEVICE, PER DIRECTION (the unidirectional rate one
 * device can inject), in B/s; latency is the per-message/per-step α of the alpha-beta model.
 */
export interface LinkSpec {
  /** Link family (informational). */
  kind:
    | 'nvlink'
    | 'nvlink_bridge'
    | 'xgmi'
    | 'pcie'
    | 'infiniband'
    | 'roce'
    | 'ethernet'
    | 'thunderbolt'
    | 'usb4'
    | 'ici'
    | 'other';
  /** Display name, e.g. `'PCIe 5.0 x16'`, `'ConnectX-7 200GbE'`. */
  name: string;
  /**
   * Per-device, per-direction bandwidth. On a multi-port inter-node fabric shape (`full_mesh`, `ring`, `chain` of
   * {@link FabricShape}) it is the bandwidth of ONE port (one cable), per direction. As a switched inter-node link it
   * is ONE NIC, per direction; the topology's `nicsPerNode` (absent = one NIC per device) sets how many NICs the
   * node's devices share (`hw/nics.ts`). Unit: B/s.
   */
  bandwidthPerDirectionBytesPerSec: number;
  /**
   * Per-message (per algorithm step) latency α. For inter-node links, one definition for every transport: the one-way
   * small-message floor as a collective library sees it = half the average (or median) round trip of a tuned
   * small-message ping-pong over that transport (`ib_write_lat` / `ibv_uc_pingpong` for RDMA, a 1-byte TCP
   * ping-pong for TCP); never an ICMP ping, never the minimum sample. Unit: seconds.
   */
  latencySec: number;
  /** `'direct'` = device-to-device transfers; `'host_bounce'` = staged through host memory (e.g. GeForce without P2P). */
  p2p: 'direct' | 'host_bounce';
  /**
   * Largest fully connected device domain established for this link profile, including the local device.
   * Every device can directly address every other device's memory (through a switch or a full mesh).
   * A positive claim about the whole domain, NOT the number of neighbours on a partial mesh. Absent = unknown
   * or partial connectivity: no peer-HBM lookup fast path. A smaller DP group does not certify an oversized
   * modeled node. This is a profile limit, not a limit of the NVLink generation. Unit: devices, safe integer ≥ 2.
   * Physical capability independent of `p2p` driver state and NVLS; `sources[]` must cover it.
   * See docs/simulation-model.md §4 "Links and topologies", §5 "Lookup tables across DP".
   */
  fullyConnectedDevices?: number;
  /** Host staging parameters; present iff `p2p === 'host_bounce'`. */
  hostBounce?: {
    /** Host memory bandwidth available to staging, shared by all devices of the node. Unit: B/s. */
    hostMemBandwidthBytesPerSec: number;
    /** Extra latency per message added by the host copy. Unit: seconds. */
    extraLatencySec: number;
  };
  /**
   * Cap on the sum over all ports of this link that one device drives at once, per direction (egress and ingress
   * each), e.g. the NIC's host PCIe attachment (DGX Spark: two ConnectX-7 ports share ~25 GB/s). Used by the
   * multi-port fabric shapes (`full_mesh`, `ring`, `chain` of {@link FabricShape}), where bytes a node relays count
   * against it, and by multi-NIC nodes (`TopologySpec.nicsPerNode`, `hw/nics.ts`), where it caps one device's share
   * of the node's NICs. Absent = the ports add up (e.g. Apple Thunderbolt 5: one controller per port). Must be ≥
   * `bandwidthPerDirectionBytesPerSec`. Unit: B/s.
   */
  allPortsCapPerDirectionBytesPerSec?: number;
  /**
   * Extra latency a node adds when it relays a message to the next hop on a switchless multi-hop path (`ring`,
   * `chain`), on top of that hop's `latencySec`. Absent = 0 (cut-through relay: the hardware floor). Unit: seconds.
   */
  forwardingLatencySec?: number;
  /**
   * Inter-node links: whether this transport can move a message straight between the NIC and an accelerator's
   * dedicated memory (GPUDirect RDMA / AMD PeerDirect on an RDMA NIC). TCP transports never can: NCCL's socket
   * transport takes host pointers only (`net_socket.cc`, `ptrSupport = NCCL_PTR_HOST`), as do Gloo and MLX.
   * `false` ⇒ a dedicated-memory device stages each message through host memory (docs/simulation-model.md §6
   * "Inter-node host staging"). Ignored on intra-node links and on unified-memory devices. Absent = true, the
   * convention before host staging was modelled; every inter-node preset sets it (enforced by a test). A
   * capability flag, no unit.
   */
  gpuDirectRdma?: boolean;
  /**
   * Inter-node links: how the collective library moves bytes over this link.
   * - `'rdma'`: verbs / RDMA (InfiniBand, RoCE, RDMA over Thunderbolt 5 or USB4): the NIC reads and writes the
   *   registered user buffer directly, zero-copy and kernel-bypass.
   * - `'tcp'`: kernel sockets (NCCL's socket transport, Gloo, MLX's ring): `send()` copies each message from the user
   *   buffer into kernel socket buffers and `recv()` copies it out again, a CPU memcpy at each end that the engine
   *   charges as the TCP socket copy (docs/simulation-model.md §6 "TCP socket copy"). A socket takes host pointers
   *   only, so `'tcp'` requires `gpuDirectRdma: false` (`validateLink`).
   * Absent = no socket copy, the convention before the copy was modelled; intra-node links leave it absent, and every
   * inter-node preset sets it (enforced by a test). On one physical medium, RDMA's `latencySec` never exceeds TCP's
   * (`LINK_MEDIA` in `presets/links.ts`, checked by a test). No unit.
   */
  transport?: 'rdma' | 'tcp';
  /**
   * In-switch reduction the switch of this link level offers for all-reduce ({@link InNetworkReduction}): `'nvls'` on
   * an NVLink Switch (intra-node), `'sharp'` on an InfiniBand switch (inter-node). Used only when
   * `SimOptions.inNetworkReduction` is `'on'`, as one more candidate of `collectiveAlgorithm: 'best'`
   * (docs/simulation-model.md §6 "In-network reduction (NVLS, SHARP)"). Absent = no in-network reduction, the
   * convention before it was modelled. A capability flag, no unit; its `sources[]` entry names the field.
   */
  inNetworkReduction?: InNetworkReduction;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
  /** Set by `applyLinkModifiers`: the modifiers this link was derived with (provenance). */
  appliedModifiers?: LinkModifier[];
}

/**
 * In-switch (in-network) reduction for all-reduce: the switch itself adds up the devices' buffers, so each device
 * sends its buffer about once and receives the result about once, instead of the ≈ 2× of a ring or tree.
 * - `'nvls'`: NVLink SHARP (NVLS) of an NVLink Switch (3rd-gen NVSwitch, Hopper, and later): each GPU reads its
 *   reduced shard from the switch with a multimem load-reduce (`multimem.ld_reduce`) and writes it back to every GPU
 *   with one multicast store (`multimem.st`); cost `engine/collectives.ts: nvlsAllReduceSec`.
 * - `'sharp'`: InfiniBand SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) of a Quantum-class IB
 *   switch: each NIC streams its buffer up the switch's aggregation tree and gets the reduced buffer back; cost
 *   `engine/collectives.ts: sharpAllReduceSec`.
 * See docs/simulation-model.md §6 "In-network reduction (NVLS, SHARP)".
 */
export type InNetworkReduction = 'nvls' | 'sharp';
/**
 * Shape of the network between nodes (docs/simulation-model.md §6 "Fabric shapes"; geometry in `hw/fabric.ts`).
 *
 * Notation: N = nodes in the group; B_L = `interNode.bandwidthPerDirectionBytesPerSec` = ONE port / one cable, per
 * direction; C = `interNode.allPortsCapPerDirectionBytesPerSec` ?? ∞ = the most one node can send (and, separately,
 * receive) over all its ports at once, relayed bytes included.
 *
 * Every shape except `'switched'` needs `devicesPerNode = 1` (`validateTopology`): one SoC or box per node, so a
 * node is a device.
 */
export type FabricShape =
  /**
   * Each node has ONE port into a switch. Ports per node: 1. Injection per node: B_L (C does not apply: one port).
   * Hops between any two nodes: 1 (the switch traversal is inside `latencySec`). An absent
   * `TopologySpec.interNodeFabric` means this, non-blocking: the convention of every topology before fabric shapes.
   */
  | {
      kind: 'switched';
      /**
       * Oversubscription of the switch tier, finite ≥ 1 (absent or 1 = non-blocking). Every inter-node flow gets
       * B_L / oversubscription (worst case: every flow crosses the oversubscribed tier). Dimensionless.
       */
      oversubscription?: number;
    }
  /**
   * Every pair of nodes cabled directly with `cablesPerPair` cables, striped as one link of k·B_L (e.g. JACCL over
   * Thunderbolt 5). Ports per node: (N−1)·k. Per pair k·B_L; per node min((N−1)·k·B_L, C). Hops: 1. N = 2 is
   * "direct cable(s)".
   */
  | {
      kind: 'full_mesh';
      /** Cables between each pair, integer ≥ 1 (absent = 1). Count. */
      cablesPerPair?: number;
    }
  /**
   * N ≥ 3 nodes, node i cabled to i ± 1 (mod N). Ports per node: 2. Per link B_L; per node min(2·B_L, C). Hops: up
   * to ⌊N/2⌋ (shortest direction). Each relaying node adds `forwardingLatencySec`, and the bytes it relays count
   * against its C (store-and-forward through the host: no switching inside these NICs). At N = 3 the cabling equals
   * a `full_mesh` of 3.
   */
  | { kind: 'ring' }
  /**
   * Daisy chain 0 – 1 – … – (N−1), the ends not joined. Ports per node: 2 inside, 1 at the ends. Per link B_L; per
   * node min(2·B_L, C) inside, B_L at the ends. Hops: up to N − 1; relays as for `'ring'`.
   */
  | { kind: 'chain' };

/**
 * How devices are connected. A "node" is one scale-up domain (an NVLink island, a PCIe box, one Mac); devices
 * in a node talk over `intraNode`, nodes over `interNode`.
 */
export interface TopologySpec {
  /** Stable id. */
  id: string;
  /** Display name. */
  name: string;
  /** Devices per node. Count. */
  devicesPerNode: number;
  /** Nodes. Count. */
  nodes: number;
  /** Link between devices of one node; required when a parallel group spans > 1 device of a node. */
  intraNode?: LinkSpec;
  /**
   * Link between nodes; required when a parallel group spans > 1 node. On a switched network it describes ONE NIC
   * (per direction); the default is one NIC per device (see `nicsPerNode`).
   */
  interNode?: LinkSpec;
  /**
   * Inter-node NICs (rails) per node, each an `interNode` link, integer ≥ 1. The node's devices all communicate at
   * once and share these NICs evenly, so one device gets B_dev = min(N_nic · B_nic / devicesPerNode, C) per direction
   * (`hw/nics.ts: interNodeBandwidthPerDeviceBytesPerSec`; B_nic = `interNode.bandwidthPerDirectionBytesPerSec`,
   * C = `interNode.allPortsCapPerDirectionBytesPerSec` ?? ∞). Absent = one NIC per device (= `devicesPerNode`), the
   * convention of every topology before multi-NIC nodes. Allowed only with `interNode`, `nodes > 1`, and a switched
   * fabric (absent or `'switched'`); N_nic must divide `devicesPerNode` or be a multiple of it (rail symmetry,
   * `validateTopology`). See docs/simulation-model.md §6 "Multi-NIC nodes". Count.
   */
  nicsPerNode?: number;
  /**
   * Shape of the inter-node network; absent = `{ kind: 'switched' }`, non-blocking (the pre-fabric convention).
   * Allowed only when `interNode` is present and `nodes > 1`; shapes other than `'switched'` need
   * `devicesPerNode = 1`.
   */
  interNodeFabric?: FabricShape;
  /** Citations for the topology itself. */
  sources: SourceRef[];
}

/**
 * Host DRAM as a storage tier for offloaded weights (docs/simulation-model.md §4 "Host and storage tiers").
 * ONE per node, shared by every device of the node. On a unified-memory device host RAM IS the device pool, so this
 * tier is only used on dedicated-memory devices.
 */
export interface HostMemorySpec {
  /** Discriminant of {@link StorageTierSpec}. */
  kind: 'host_ram';
  /** Stable id, e.g. `'ddr5-6000-2ch'`. */
  id: string;
  /** Display name. */
  name: string;
  /** Installed capacity of the node. Unit: bytes. */
  capacityBytes: number;
  /**
   * Theoretical peak of the whole node, read + write combined: MT/s × 8 B per channel transfer × channels.
   * Unit: B/s.
   */
  bandwidthBytesPerSec: number;
  /** Unloaded load-to-use latency of one random read (the α of a host-RAM read). Unit: seconds. */
  latencySec: number;
  /** Memory channels (informational; already folded into `bandwidthBytesPerSec`). Count. */
  channels: number;
  /** Smallest unit a random read moves (one cache line, 64). Unit: bytes. */
  readGranularityBytes: number;
  /** Random reads the host keeps in flight (memory-level parallelism; the queue-depth cap of Little's law). Count. */
  maxOutstanding: number;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
}

/**
 * An NVMe SSD as a storage tier for offloaded weights (docs/simulation-model.md §4 "Host and storage tiers").
 * ONE per node, shared by every device of the node. Datasheet ceilings, not achieved fractions.
 */
export interface NvmeSpec {
  /** Discriminant of {@link StorageTierSpec}. */
  kind: 'nvme';
  /** Stable id, e.g. `'nvme-990-pro-2tb'`. */
  id: string;
  /** Display name. */
  name: string;
  /** Usable capacity. Unit: bytes. */
  capacityBytes: number;
  /** Datasheet sequential read (large blocks, deep queue). Unit: B/s. */
  seqReadBandwidthBytesPerSec: number;
  /**
   * Opt-in low-concurrency large-read scenario; absent means no measured fraction.
   * A 1 MiB QD1 measurement is pessimistic for 10–20 MB experts (2 MB QD3 can reach 91–95%).
   */
  sustainedRead?: {
    /** Low-QD read rate / same-run deep-queue sequential rate. Dimensionless, in (0, 1]. */
    fraction: number;
    /** Minimum individual read size to apply this scenario. Unit: bytes. */
    blockBytes: number;
    /** Queue depth per thread of the measurement (not the deployment's queue cap). Count. */
    queueDepth: number;
    /** Submission threads of the measurement. Count. */
    threads: number;
  };
  /** Random 4 KiB read latency at QD1 (1 / QD1 IOPS when only the IOPS is published). Unit: seconds. */
  randomReadLatencySec: number;
  /**
   * Datasheet maximum random 4 KiB read rate and its measurement condition. OPTIONAL: when absent the IOPS term is
   * dropped and the request rate is bounded by Little's law alone (`maxOutstanding / randomReadLatencySec`).
   */
  randomRead?: {
    /** Random 4 KiB reads per second at the rated condition. Unit: requests/s. */
    iops: number;
    /** Queue depth per thread of the rated condition. Count. */
    queueDepth: number;
    /** Threads (submission queues) of the rated condition. Count. */
    threads: number;
  };
  /**
   * Read requests the drive keeps in flight (queueDepth × threads of the rated condition, or an explicit estimate in
   * `estimated[]`): the queue-depth cap of Little's law. Count.
   */
  maxOutstanding: number;
  /** Smallest unit a read moves (the LBA / NAND page, 4096): a 320 B row read costs this much. Unit: bytes. */
  readGranularityBytes: number;
  /**
   * The slot's own ceiling when it binds below the datasheet rate (e.g. PCIe 4.0 x4 payload = 7.20e9). Absent = the
   * datasheet rate is the bound. Per direction. Unit: B/s.
   */
  linkBandwidthBytesPerSec?: number;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
}

/** A storage tier an offloaded weight group can be placed on: host DRAM or an NVMe SSD. */
export type StorageTierSpec = HostMemorySpec | NvmeSpec;

/**
 * The host side of a node. `link` is PER DEVICE (each device's own PCIe connection, per direction); `ram` and `ssd`
 * are ONE per node, shared by the node's devices. Every field is optional: the engine fills an absent one with a
 * default (link from `DeviceSpec.hostInterface`, `DDR5_6000_2CH`, `NVME_990_PRO_2TB`) and warns.
 */
export interface HostSpec {
  /** The device's link to host memory (PCIe), per device, per direction. Ignored on unified memory. */
  link?: LinkSpec;
  /** Host DRAM of the node. On unified memory it is the device pool itself and not a separate tier. */
  ram?: HostMemorySpec;
  /** The node's NVMe SSD. */
  ssd?: NvmeSpec;
}

/**
 * Host CPU compute, for the CPU-compute offload mode (llama.cpp `-ot exps=CPU`). A designed HOOK: validated, never
 * costed in the first implementation (`OffloadConfig.mode: 'cpu_compute'` is rejected by validation). Per node.
 */
export interface HostComputeSpec {
  /** Matmul peaks of the host CPU (AVX-512 / AMX / SME), per node. Unit: FLOP/s (OP/s for integer inputs). */
  matmulPeaks: MatmulPeak[];
  /** Non-matrix vector peaks by dtype, per node. Unit: FLOP/s. */
  vectorFlopsPerSec: Partial<Record<'fp32' | 'bf16', number>>;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
}

/** The hardware a simulation runs on: a device type replicated over a topology, plus variant modifiers. */
export interface HardwareStack {
  /** Base device spec (before `deviceModifiers`). */
  device: DeviceSpec;
  /** Topology (devices per node, nodes, links). */
  topology: TopologySpec;
  /** Variant modifiers applied to `device` before simulation, in order. */
  deviceModifiers?: DeviceModifier[];
  /** Modifiers applied to the topology's links before simulation (e.g. enabling P2P over PCIe), in order. */
  linkModifiers?: { intraNode?: LinkModifier[]; interNode?: LinkModifier[] };
  /** Engine to run on; default: `engines[0].id` of the modified device. */
  engineId?: string;
  /** Memory capacity per device; must be one of the modified `memory.capacityOptionsBytes`; default `[0]`. Unit: bytes. */
  memCapacityBytes?: number;
  /**
   * Override of the device's GPU-usable fraction of unified memory (`MemorySpec.gpuUsableFraction`), e.g. after
   * raising the macOS limit with `sysctl iogpu.wired_limit_mb` or the Linux GTT size with `ttm.pages_limit`.
   * Ignored on dedicated memory. Absent = the device's default. Dimensionless, (0, 1].
   */
  gpuUsableFraction?: number;
  /**
   * Runtime reserve per device: memory the framework and driver keep besides the model (CUDA context, allocator
   * pools, cuBLAS / attention workspaces), counted in the fit. No vendor publishes it (it depends on the engine and
   * driver), so absent = 0, the hardware-only bound. Unit: bytes.
   */
  runtimeReserveBytes?: number;
  /**
   * The node's host: the per-device host link and the shared host RAM / SSD tiers that offloaded weights stream
   * from. Only read when the deployment offloads something; absent = defaults (see {@link HostSpec}).
   */
  host?: HostSpec;
}
