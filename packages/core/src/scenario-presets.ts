import {
  SCENARIO_SCHEMA_VERSION,
  type AllocationClass,
  type CommunicatorGroupSpec,
  type EvidenceProvenance,
  type MemoryDomainSpec,
  type NetworkResourceSpec,
  type NetworkTransportMode,
  type PartitionPlacement,
  type SimDeviceSpec,
  type SimLinkSpec,
  type SimulationScenario,
  type TransferRequirement,
} from "./scenario-types.js";
import { assertValidScenario } from "./scenario.js";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;
const TiB = 1024 ** 4;
const GBps = 1_000_000_000;

const HEURISTIC: EvidenceProvenance = {
  confidence: "heuristic",
  source: "built-in illustrative preset; calibrate before performance use",
};

export const SCENARIO_PRESET_NAMES = [
  "cpu-only",
  "single-gpu-cpu",
  "multi-gpu",
  "gpu-npu",
  "unified-memory",
  "multi-node",
] as const;

export const COMPUTER_PRESET_NAMES = [
  "rtx-4090-desktop",
  "rtx-5090-desktop",
  "mac-mini-m4-pro-64gb",
  "mac-studio-m3-ultra-512gb",
  "ryzen-ai-max-395-128gb",
] as const;

export const ALL_SCENARIO_PRESET_NAMES = [
  ...COMPUTER_PRESET_NAMES,
  ...SCENARIO_PRESET_NAMES,
] as const;

export type ScenarioPresetName =
  typeof ALL_SCENARIO_PRESET_NAMES[number];

const PRESET_FACTORIES: Readonly<
  Record<ScenarioPresetName, () => SimulationScenario>
> = {
  "cpu-only": buildCpuOnly,
  "single-gpu-cpu": buildSingleGpu,
  "multi-gpu": buildMultiGpu,
  "gpu-npu": buildGpuNpu,
  "unified-memory": buildUnified,
  "multi-node": buildMultiNode,
  "rtx-4090-desktop": () => buildDiscreteComputer({
    id: "rtx-4090-desktop",
    gpuId: "desktop:rtx4090",
    gpuComputeProfileId: "nvidia-geforce-rtx-4090",
    hostCapacityBytes: 64 * GiB,
    hostResourceLimitBytes: 56 * GiB,
    hostBandwidthBytesPerSec: 83 * GBps,
    vramCapacityBytes: 24 * GiB,
    vramResourceLimitBytes: 22 * GiB,
    vramBandwidthBytesPerSec: 1_008 * GBps,
    pcieBandwidthBytesPerSec: 32 * GBps,
    storageCapacityBytes: 2 * TiB,
  }),
  "rtx-5090-desktop": () => buildDiscreteComputer({
    id: "rtx-5090-desktop",
    gpuId: "desktop:rtx5090",
    gpuComputeProfileId: "nvidia-geforce-rtx-5090",
    hostCapacityBytes: 128 * GiB,
    hostResourceLimitBytes: 112 * GiB,
    hostBandwidthBytesPerSec: 90 * GBps,
    vramCapacityBytes: 32 * GiB,
    vramResourceLimitBytes: 30 * GiB,
    vramBandwidthBytesPerSec: 1_792 * GBps,
    pcieBandwidthBytesPerSec: 64 * GBps,
    storageCapacityBytes: 4 * TiB,
  }),
  "mac-mini-m4-pro-64gb": () => buildUnifiedComputer({
    id: "mac-mini-m4-pro-64gb",
    nodeId: "mac-mini",
    gpuId: "mac-mini:m4-pro-gpu",
    npuId: "mac-mini:m4-pro-neural-engine",
    gpuComputeProfileId: "apple-m4-pro-gpu",
    npuComputeProfileId: "apple-m4-pro-neural-engine",
    capacityBytes: 64 * GiB,
    resourceLimitBytes: 56 * GiB,
    bandwidthBytesPerSec: 273 * GBps,
    storageCapacityBytes: 2 * TiB,
    executionProvider: "CoreMLExecutionProvider",
  }),
  "mac-studio-m3-ultra-512gb": () => buildUnifiedComputer({
    id: "mac-studio-m3-ultra-512gb",
    nodeId: "mac-studio",
    gpuId: "mac-studio:m3-ultra-gpu",
    npuId: "mac-studio:m3-ultra-neural-engine",
    gpuComputeProfileId: "apple-m3-ultra-gpu",
    npuComputeProfileId: "apple-m3-ultra-neural-engine",
    capacityBytes: 512 * GiB,
    resourceLimitBytes: 480 * GiB,
    bandwidthBytesPerSec: 819 * GBps,
    storageCapacityBytes: 8 * TiB,
    executionProvider: "CoreMLExecutionProvider",
  }),
  "ryzen-ai-max-395-128gb": () => buildUnifiedComputer({
    id: "ryzen-ai-max-395-128gb",
    nodeId: "ryzen-ai",
    gpuId: "ryzen-ai:radeon-8060s",
    npuId: "ryzen-ai:xdna2-npu",
    capacityBytes: 128 * GiB,
    resourceLimitBytes: 112 * GiB,
    bandwidthBytesPerSec: 256 * GBps,
    storageCapacityBytes: 2 * TiB,
    executionProvider: "DirectMLExecutionProvider",
  }),
};

export function buildScenarioPreset(name: ScenarioPresetName): SimulationScenario {
  const factory = PRESET_FACTORIES[name];
  if (!factory) {
    throw new Error(
      `unknown scenario preset ${String(name)}; available: ${ALL_SCENARIO_PRESET_NAMES.join(", ")}`,
    );
  }
  const scenario = factory();
  assertValidScenario(scenario);
  return scenario;
}

interface DiscreteComputerConfig {
  readonly id: string;
  readonly gpuId: string;
  readonly gpuComputeProfileId: string;
  readonly hostCapacityBytes: number;
  readonly hostResourceLimitBytes: number;
  readonly hostBandwidthBytesPerSec: number;
  readonly vramCapacityBytes: number;
  readonly vramResourceLimitBytes: number;
  readonly vramBandwidthBytesPerSec: number;
  readonly pcieBandwidthBytesPerSec: number;
  readonly storageCapacityBytes: number;
}

function buildDiscreteComputer(
  config: DiscreteComputerConfig,
): SimulationScenario {
  const nodeId = "desktop";
  const hostDomainId = `${nodeId}:host`;
  const vramDomainId = `${config.gpuId}:vram`;
  const cpu = device(
    `${nodeId}:cpu`,
    nodeId,
    "cpu",
    "CPUExecutionProvider",
    [hostDomainId],
    ["copy", "sampling"],
  );
  const gpu = device(
    config.gpuId,
    nodeId,
    "gpu",
    "CUDAExecutionProvider",
    [vramDomainId, hostDomainId],
    ["attention", "ffn", "collective", "copy", "sampling", "draft"],
    config.gpuComputeProfileId,
  );
  return scenario({
    id: config.id,
    family: "single_discrete",
    storageCapacityBytes: config.storageCapacityBytes,
    domains: [
      {
        ...domain(
          hostDomainId,
          nodeId,
          "host",
          config.hostCapacityBytes,
          config.hostBandwidthBytesPerSec,
          ["pageable", "pinned"],
          [cpu.id, gpu.id],
          { kind: "host", nodeId },
        ),
        resourceLimitBytes: config.hostResourceLimitBytes,
      },
      {
        ...domain(
          vramDomainId,
          nodeId,
          "device",
          config.vramCapacityBytes,
          config.vramBandwidthBytesPerSec,
          ["device"],
          [gpu.id],
          { kind: "device", deviceId: gpu.id },
        ),
        resourceLimitBytes: config.vramResourceLimitBytes,
      },
    ],
    devices: [cpu, gpu],
    links: bidirectionalLink(
      `${nodeId}:pcie`,
      hostDomainId,
      vramDomainId,
      "pcie",
      config.pcieBandwidthBytesPerSec,
      1_500,
    ),
    placements: [
      placement("target", gpu.id, ["attention", "ffn"], [
        allocation(
          "target-weights",
          vramDomainId,
          8 * GiB,
          "device",
          "weights",
        ),
        allocation("target-kv", vramDomainId, 2 * GiB, "device", "kv"),
        allocation(
          "target-workspace",
          vramDomainId,
          256 * MiB,
          "device",
          "workspace",
        ),
        allocation(
          "offload-staging",
          hostDomainId,
          2 * GiB,
          "pinned",
          "staging",
        ),
      ]),
    ],
    transfers: [
      transfer("offload-to-gpu", hostDomainId, vramDomainId, 2 * GiB),
    ],
    groups: [group("world", [[0, gpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 1,
      expert: 1,
      data: 1,
    },
  });
}

interface UnifiedComputerConfig {
  readonly id: string;
  readonly nodeId: string;
  readonly gpuId: string;
  readonly npuId: string;
  readonly gpuComputeProfileId?: string;
  readonly npuComputeProfileId?: string;
  readonly capacityBytes: number;
  readonly resourceLimitBytes: number;
  readonly bandwidthBytesPerSec: number;
  readonly storageCapacityBytes: number;
  readonly executionProvider: string;
}

function buildUnifiedComputer(
  config: UnifiedComputerConfig,
): SimulationScenario {
  const unifiedDomainId = `${config.nodeId}:unified`;
  const cpu = device(
    `${config.nodeId}:cpu`,
    config.nodeId,
    "cpu",
    "CPUExecutionProvider",
    [unifiedDomainId],
    ["attention", "ffn", "copy", "sampling", "draft"],
  );
  const gpu = device(
    config.gpuId,
    config.nodeId,
    "gpu",
    config.executionProvider,
    [unifiedDomainId],
    ["attention", "ffn", "copy", "sampling", "draft"],
    config.gpuComputeProfileId,
  );
  const npu = device(
    config.npuId,
    config.nodeId,
    "npu",
    config.executionProvider,
    [unifiedDomainId],
    ["attention", "copy", "draft"],
    config.npuComputeProfileId,
  );
  return scenario({
    id: config.id,
    family: "unified",
    storageCapacityBytes: config.storageCapacityBytes,
    domains: [
      {
        ...domain(
          unifiedDomainId,
          config.nodeId,
          "unified",
          config.capacityBytes,
          config.bandwidthBytesPerSec,
          ["unified"],
          [cpu.id, gpu.id, npu.id],
          { kind: "host", nodeId: config.nodeId },
          true,
        ),
        resourceLimitBytes: config.resourceLimitBytes,
      },
    ],
    devices: [cpu, gpu, npu],
    placements: [
      placement("target", gpu.id, ["attention", "ffn"], [
        allocation(
          "target-weights",
          unifiedDomainId,
          Math.min(60 * GiB, config.capacityBytes / 2),
          "unified",
          "weights",
        ),
        allocation(
          "target-kv",
          unifiedDomainId,
          Math.min(16 * GiB, config.capacityBytes / 8),
          "unified",
          "kv",
        ),
        allocation(
          "target-workspace",
          unifiedDomainId,
          256 * MiB,
          "unified",
          "workspace",
        ),
      ]),
    ],
    groups: [group("world", [[0, gpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 1,
      expert: 1,
      data: 1,
    },
  });
}

function buildCpuOnly(): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["attention", "ffn", "copy", "sampling", "draft"],
  );
  return scenario({
    id: "cpu-only",
    family: "cpu_only",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        128 * GiB,
        100 * GBps,
        ["pageable", "pinned"],
        [cpu.id],
        { kind: "host", nodeId: "node0" },
      ),
    ],
    devices: [cpu],
    placements: [
      placement("target", cpu.id, ["attention", "ffn"], [
        allocation("target-weights", "node0:host", 8 * GiB, "pageable", "weights"),
        allocation("target-kv", "node0:host", 2 * GiB, "pageable", "kv"),
        allocation(
          "target-workspace",
          "node0:host",
          256 * MiB,
          "pageable",
          "workspace",
        ),
      ]),
    ],
    groups: [group("world", [[0, cpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 1,
      expert: 1,
      data: 1,
    },
  });
}

function buildSingleGpu(): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["copy", "sampling"],
  );
  const gpu = device(
    "node0:gpu0",
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    ["node0:gpu0:vram", "node0:host"],
    ["attention", "ffn", "collective", "copy", "sampling", "draft"],
  );
  const links = bidirectionalLink(
    "node0:pcie0",
    "node0:host",
    "node0:gpu0:vram",
    "pcie",
    32 * GBps,
    1_500,
  );
  return scenario({
    id: "single-gpu-cpu",
    family: "single_discrete",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        256 * GiB,
        100 * GBps,
        ["pageable", "pinned"],
        [cpu.id, gpu.id],
        { kind: "host", nodeId: "node0" },
      ),
      domain(
        "node0:gpu0:vram",
        "node0",
        "device",
        80 * GiB,
        2_000 * GBps,
        ["device"],
        [gpu.id],
        { kind: "device", deviceId: gpu.id },
      ),
    ],
    devices: [cpu, gpu],
    links,
    placements: [
      placement("target", gpu.id, ["attention", "ffn"], [
        allocation(
          "target-weights",
          "node0:gpu0:vram",
          40 * GiB,
          "device",
          "weights",
        ),
        allocation("target-kv", "node0:gpu0:vram", 8 * GiB, "device", "kv"),
        allocation(
          "target-workspace",
          "node0:gpu0:vram",
          256 * MiB,
          "device",
          "workspace",
        ),
        allocation(
          "offload-staging",
          "node0:host",
          4 * GiB,
          "pinned",
          "staging",
        ),
      ]),
    ],
    transfers: [
      transfer("offload-to-gpu", "node0:host", "node0:gpu0:vram", 4 * GiB),
    ],
    groups: [group("world", [[0, gpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 1,
      expert: 1,
      data: 1,
    },
  });
}

export function buildMultiGpuRingScenario(
  gpuCount: number,
): SimulationScenario {
  if (!Number.isSafeInteger(gpuCount) || gpuCount < 2 || gpuCount > 64) {
    throw new Error(
      `multi-GPU ring count must be a safe integer from 2 through 64; got ${gpuCount}`,
    );
  }
  const result = buildMultiGpuRing(gpuCount, `multi-gpu-ring-${gpuCount}`);
  assertValidScenario(result);
  return result;
}

function buildMultiGpu(): SimulationScenario {
  return buildMultiGpuRing(2, "multi-gpu");
}

function buildMultiGpuRing(
  gpuCount: number,
  id: string,
): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["copy", "sampling"],
  );
  const gpus = Array.from({ length: gpuCount }, (_, index) => device(
    `node0:gpu${index}`,
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    [`node0:gpu${index}:vram`, "node0:host"],
    ["attention", "ffn", "collective", "copy", "sampling", "draft"],
  ));
  const ringLinks = gpuCount === 2
    ? bidirectionalLink(
        "node0:nvlink",
        "node0:gpu0:vram",
        "node0:gpu1:vram",
        "nvlink",
        600 * GBps,
        500,
      )
    : gpus.flatMap((_, index) => {
        const next = (index + 1) % gpuCount;
        return bidirectionalLink(
          `node0:nvlink${index}${next}`,
          `node0:gpu${index}:vram`,
          `node0:gpu${next}:vram`,
          "nvlink",
          600 * GBps,
          500,
        );
      });
  return scenario({
    id,
    family: "multi_gpu",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        512 * GiB,
        120 * GBps,
        ["pageable", "pinned"],
        [cpu.id, ...gpus.map((gpu) => gpu.id)],
        { kind: "host", nodeId: "node0" },
      ),
      ...gpus.map((gpu, index) => domain(
        `node0:gpu${index}:vram`,
        "node0",
        "device",
        80 * GiB,
        3_000 * GBps,
        ["device"],
        [gpu.id],
        { kind: "device", deviceId: gpu.id },
      )),
    ],
    devices: [cpu, ...gpus],
    links: [
      ...gpus.flatMap((_, index) => bidirectionalLink(
        `node0:pcie${index}`,
        "node0:host",
        `node0:gpu${index}:vram`,
        "pcie",
        32 * GBps,
        1_500,
      )),
      ...ringLinks,
    ],
    placements: gpus.map((gpu, index) => placement(
      `target-shard-${index}`,
      gpu.id,
      ["attention", "ffn", "collective"],
      [
        allocation(
          `weights-${index}`,
          `node0:gpu${index}:vram`,
          36 * GiB,
          "device",
          "weights",
        ),
        allocation(
          `kv-${index}`,
          `node0:gpu${index}:vram`,
          8 * GiB,
          "device",
          "kv",
        ),
        allocation(
          `workspace-${index}`,
          `node0:gpu${index}:vram`,
          128 * MiB,
          "device",
          "workspace",
        ),
        ...(index === 0
          ? [
              allocation(
                "expert-cache-host",
                "node0:host",
                256 * MiB,
                "pinned",
                "staging",
              ),
            ]
          : []),
      ],
    )),
    transfers: [
      transfer(
        "tensor-parallel",
        "node0:gpu0:vram",
        "node0:gpu1:vram",
        64 * 1024 ** 2,
      ),
    ],
    groups: [group(
      "tp",
      gpus.map((gpu, index) => [index, gpu.id] as const),
    )],
    parallelism: {
      composition: "overlap_by_capability",
      tensor: gpuCount,
      pipeline: 1,
      expert: gpuCount,
      data: 1,
    },
  });
}

function buildGpuNpu(): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["copy", "sampling"],
  );
  const gpu = device(
    "node0:gpu0",
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    ["node0:gpu0:vram", "node0:host"],
    ["ffn", "collective", "copy", "sampling", "draft"],
  );
  const npu = device(
    "node0:npu0",
    "node0",
    "npu",
    "QNNExecutionProvider",
    ["node0:npu0:memory", "node0:host"],
    ["attention", "copy"],
  );
  return scenario({
    id: "gpu-npu",
    family: "gpu_npu",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        128 * GiB,
        80 * GBps,
        ["pageable", "pinned"],
        [cpu.id, gpu.id, npu.id],
        { kind: "host", nodeId: "node0" },
      ),
      domain(
        "node0:gpu0:vram",
        "node0",
        "device",
        48 * GiB,
        900 * GBps,
        ["device"],
        [gpu.id],
        { kind: "device", deviceId: gpu.id },
      ),
      domain(
        "node0:npu0:memory",
        "node0",
        "device",
        16 * GiB,
        200 * GBps,
        ["device"],
        [npu.id],
        { kind: "device", deviceId: npu.id },
      ),
    ],
    devices: [cpu, gpu, npu],
    links: [
      ...bidirectionalLink(
        "node0:gpu-pcie",
        "node0:host",
        "node0:gpu0:vram",
        "pcie",
        32 * GBps,
        1_500,
      ),
      ...bidirectionalLink(
        "node0:npu-pcie",
        "node0:host",
        "node0:npu0:memory",
        "pcie",
        16 * GBps,
        2_000,
      ),
    ],
    placements: [
      placement("attention", npu.id, ["attention"], [
        allocation(
          "attention-weights",
          "node0:npu0:memory",
          8 * GiB,
          "device",
          "weights",
        ),
        allocation("attention-kv", "node0:npu0:memory", 4 * GiB, "device", "kv"),
        allocation(
          "attention-workspace",
          "node0:npu0:memory",
          256 * MiB,
          "device",
          "workspace",
        ),
        allocation(
          "npu-staging",
          "node0:host",
          2 * GiB,
          "pinned",
          "staging",
        ),
      ]),
      placement("ffn", gpu.id, ["ffn"], [
        allocation(
          "ffn-weights",
          "node0:gpu0:vram",
          30 * GiB,
          "device",
          "weights",
        ),
        allocation(
          "ffn-workspace",
          "node0:gpu0:vram",
          256 * MiB,
          "device",
          "workspace",
        ),
        allocation(
          "gpu-staging",
          "node0:host",
          2 * GiB,
          "pinned",
          "staging",
        ),
      ]),
    ],
    transfers: [
      transfer(
        "npu-to-gpu-activation",
        "node0:npu0:memory",
        "node0:gpu0:vram",
        256 * 1024 ** 2,
        true,
        ["npu-staging"],
      ),
    ],
    groups: [group("pipeline", [[0, npu.id], [1, gpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 2,
      expert: 1,
      data: 1,
    },
  });
}

function buildUnified(): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:unified"],
    ["attention", "ffn", "copy", "sampling", "draft"],
  );
  const gpu = device(
    "node0:gpu0",
    "node0",
    "gpu",
    "CoreMLExecutionProvider",
    ["node0:unified"],
    ["attention", "ffn", "copy", "sampling", "draft"],
  );
  return scenario({
    id: "unified-memory",
    family: "unified",
    domains: [
      domain(
        "node0:unified",
        "node0",
        "unified",
        128 * GiB,
        500 * GBps,
        ["unified"],
        [cpu.id, gpu.id],
        { kind: "host", nodeId: "node0" },
        true,
      ),
    ],
    devices: [cpu, gpu],
    placements: [
      placement("target", gpu.id, ["attention", "ffn"], [
        allocation(
          "target-weights",
          "node0:unified",
          60 * GiB,
          "unified",
          "weights",
        ),
        allocation("target-kv", "node0:unified", 16 * GiB, "unified", "kv"),
        allocation(
          "target-workspace",
          "node0:unified",
          256 * MiB,
          "unified",
          "workspace",
        ),
      ]),
    ],
    groups: [group("world", [[0, gpu.id]])],
    parallelism: {
      composition: "cartesian",
      tensor: 1,
      pipeline: 1,
      expert: 1,
      data: 1,
    },
  });
}

function buildMultiNode(): SimulationScenario {
  const generated = buildMultiNodeLanScenario(2, {
    linkKind: "infiniband",
    transport: "rdma_host",
    bandwidthBytesPerSec: 50 * GBps,
    latencyNs: 3_000,
  });
  return {
    ...generated,
    links: generated.links.map((entry) => {
      if (entry.id === "lan:node0:node1") {
        const { transport: _transport, ...legacy } = entry;
        return { ...legacy, id: "cluster:rdma:forward" };
      }
      if (entry.id === "lan:node1:node0") {
        const { transport: _transport, ...legacy } = entry;
        return { ...legacy, id: "cluster:rdma:reverse" };
      }
      return entry;
    }),
  };
}

export type MultiNodeLanNodeCount = 2 | 3 | 4;

export interface MultiNodeLanOptions {
  readonly advanced?: boolean;
  readonly linkKind?: "ethernet" | "infiniband";
  readonly transport?: NetworkTransportMode;
  readonly bandwidthBytesPerSec?: number;
  readonly latencyNs?: number;
  readonly linkConcurrencyLanes?: number;
  readonly nicBandwidthBytesPerSec?: number;
  readonly nicLatencyNs?: number;
  readonly nicConcurrencyLanes?: number;
  readonly fabricBandwidthBytesPerSec?: number;
  readonly fabricLatencyNs?: number;
  readonly fabricConcurrencyLanes?: number;
}

export function buildMultiNodeLanScenario(
  nodeCount: MultiNodeLanNodeCount,
  options: MultiNodeLanOptions = {},
): SimulationScenario {
  if (![2, 3, 4].includes(nodeCount)) {
    throw new Error("small LAN node count must be 2, 3, or 4");
  }
  const advanced = options.advanced ?? false;
  const linkKind = options.linkKind ?? "ethernet";
  const transport = options.transport
    ?? (linkKind === "ethernet" ? "tcp" : "rdma_host");
  if (transport === "gpudirect_rdma" && !advanced) {
    throw new Error("GPUDirect RDMA requires advanced LAN modeling");
  }
  const nodes = Array.from({ length: nodeCount }, (_, index) => {
    const nodeId = `node${index}`;
    const cpu = device(
      `${nodeId}:cpu0`,
      nodeId,
      "cpu",
      "CPUExecutionProvider",
      [`${nodeId}:host`],
      ["copy", "sampling"],
    );
    const gpu = device(
      `${nodeId}:gpu0`,
      nodeId,
      "gpu",
      "CUDAExecutionProvider",
      [`${nodeId}:gpu0:vram`, `${nodeId}:host`],
      ["attention", "ffn", "collective", "copy", "draft"],
    );
    return { index, nodeId, cpu, gpu };
  });
  const networkResources: NetworkResourceSpec[] = advanced
    ? [
        ...nodes.map(({ nodeId }): NetworkResourceSpec => ({
          id: `${nodeId}:nic0`,
          kind: "nic",
          nodeId,
          bandwidthBytesPerSec:
            options.nicBandwidthBytesPerSec ?? 25 * GBps,
          latencyNs: options.nicLatencyNs ?? 700,
          concurrencyLanes: options.nicConcurrencyLanes ?? 2,
          supportedTransports: [transport],
          directMemoryDomainIds: transport === "gpudirect_rdma"
            ? [`${nodeId}:gpu0:vram`]
            : [],
          provenance: HEURISTIC,
        })),
        {
          id: "lan:fabric0",
          kind: "switch",
          bandwidthBytesPerSec:
            options.fabricBandwidthBytesPerSec ?? 50 * GBps,
          latencyNs: options.fabricLatencyNs ?? 500,
          concurrencyLanes: options.fabricConcurrencyLanes ?? nodeCount,
          supportedTransports: [transport],
          directMemoryDomainIds: [],
          provenance: HEURISTIC,
        },
      ]
    : [];
  const networkLinks: SimLinkSpec[] = [];
  for (let source = 0; source < nodes.length; source++) {
    for (let target = 0; target < nodes.length; target++) {
      if (source === target) {
        continue;
      }
      const sourceNode = nodes[source];
      const targetNode = nodes[target];
      const endpoint = (nodeId: string): string => (
        transport === "gpudirect_rdma"
          ? `${nodeId}:gpu0:vram`
          : `${nodeId}:host`
      );
      networkLinks.push({
        id: `lan:${sourceNode.nodeId}:${targetNode.nodeId}`,
        sourceDomainId: endpoint(sourceNode.nodeId),
        targetDomainId: endpoint(targetNode.nodeId),
        kind: linkKind,
        bandwidthBytesPerSec: options.bandwidthBytesPerSec
          ?? (linkKind === "ethernet" ? 10 * GBps : 50 * GBps),
        latencyNs: options.latencyNs
          ?? (linkKind === "ethernet" ? 10_000 : 3_000),
        concurrencyLanes: options.linkConcurrencyLanes ?? 1,
        transport,
        ...(advanced
          ? {
              networkResourceIds: [
                `${sourceNode.nodeId}:nic0`,
                "lan:fabric0",
                `${targetNode.nodeId}:nic0`,
              ],
            }
          : {}),
        provenance: HEURISTIC,
      });
    }
  }
  return scenario({
    id: nodeCount === 2 ? "multi-node" : `multi-node-${nodeCount}`,
    family: "multi_node",
    domains: nodes.flatMap(({ nodeId, cpu, gpu }) => [
      domain(
        `${nodeId}:host`,
        nodeId,
        "host",
        256 * GiB,
        100 * GBps,
        ["pageable", "pinned"],
        [cpu.id, gpu.id],
        { kind: "host", nodeId },
      ),
      domain(
        `${nodeId}:gpu0:vram`,
        nodeId,
        "device",
        80 * GiB,
        2_000 * GBps,
        ["device"],
        [gpu.id],
        { kind: "device", deviceId: gpu.id },
      ),
    ]),
    devices: nodes.flatMap(({ cpu, gpu }) => [cpu, gpu]),
    ...(advanced ? { networkResources } : {}),
    links: [
      ...nodes.flatMap(({ nodeId }) => bidirectionalLink(
        `${nodeId}:pcie`,
        `${nodeId}:host`,
        `${nodeId}:gpu0:vram`,
        "pcie",
        32 * GBps,
        1_500,
      )),
      ...networkLinks,
    ],
    placements: nodes.map(({ index, nodeId, gpu }) => (
      placement(
        `target-shard-${index}`,
        gpu.id,
        ["attention", "ffn", "collective"],
        [
          allocation(
            `weights-${index}`,
            `${nodeId}:gpu0:vram`,
            Math.floor(72 * GiB / nodeCount),
            "device",
            "weights",
          ),
          allocation(
            `kv-${index}`,
            `${nodeId}:gpu0:vram`,
            Math.floor(16 * GiB / nodeCount),
            "device",
            "kv",
          ),
          allocation(
            `workspace-${index}`,
            `${nodeId}:gpu0:vram`,
            128 * MiB,
            "device",
            "workspace",
          ),
          allocation(
            `staging-${index}`,
            `${nodeId}:host`,
            1 * GiB,
            "pinned",
            "staging",
          ),
        ],
      )
    )),
    transfers: [
      transfer(
        "cross-node-collective",
        "node0:gpu0:vram",
        "node1:gpu0:vram",
        64 * 1024 ** 2,
        transport !== "gpudirect_rdma",
        transport === "gpudirect_rdma" ? [] : ["staging-0", "staging-1"],
      ),
    ],
    groups: [group(
      "tp",
      nodes.map(({ index, gpu }) => [index, gpu.id]),
    )],
    parallelism: {
      composition: "overlap_by_capability",
      tensor: nodeCount,
      pipeline: 1,
      expert: nodeCount,
      data: 1,
    },
  });
}

export function configureSmallLanNetwork(
  input: SimulationScenario,
  options: MultiNodeLanOptions,
): SimulationScenario {
  const nodeIds = [...new Set([
    ...input.devices.map((entry) => entry.nodeId),
    ...input.memoryDomains.map((entry) => entry.nodeId),
  ])].sort();
  if (
    nodeIds.length < 2
    || nodeIds.length > 4
  ) {
    throw new Error("small LAN configuration requires 2 through 4 systems");
  }
  const advanced = options.advanced ?? false;
  const linkKind = options.linkKind ?? "ethernet";
  const transport = options.transport
    ?? (linkKind === "ethernet" ? "tcp" : "rdma_host");
  if (transport === "gpudirect_rdma" && !advanced) {
    throw new Error("GPUDirect RDMA requires advanced LAN modeling");
  }
  const endpointByNode = new Map<string, {
    readonly host: MemoryDomainSpec;
    readonly device: MemoryDomainSpec;
  }>();
  for (const nodeId of nodeIds) {
    const hosts = input.memoryDomains.filter((domain) => (
      domain.nodeId === nodeId && domain.kind === "host"
    ));
    const devices = input.memoryDomains.filter((domain) => (
      domain.nodeId === nodeId && domain.kind === "device"
    ));
    if (hosts.length !== 1 || devices.length !== 1) {
      throw new Error(
        `small LAN system ${nodeId} requires exactly one host and one device memory domain`,
      );
    }
    endpointByNode.set(nodeId, { host: hosts[0], device: devices[0] });
  }
  const existingNetworkLinks = input.links.filter((entry) => {
    if (entry.kind !== "ethernet" && entry.kind !== "infiniband") {
      return false;
    }
    const source = input.memoryDomains.find(
      (domain) => domain.id === entry.sourceDomainId,
    );
    const target = input.memoryDomains.find(
      (domain) => domain.id === entry.targetDomainId,
    );
    return source !== undefined
      && target !== undefined
      && source.nodeId !== target.nodeId;
  });
  const provenance = existingNetworkLinks[0]?.provenance ?? HEURISTIC;
  const networkResources: NetworkResourceSpec[] = advanced
    ? [
        ...nodeIds.map((nodeId): NetworkResourceSpec => ({
          id: `${nodeId}:nic0`,
          kind: "nic",
          nodeId,
          bandwidthBytesPerSec:
            options.nicBandwidthBytesPerSec ?? 25 * GBps,
          latencyNs: options.nicLatencyNs ?? 700,
          concurrencyLanes: options.nicConcurrencyLanes ?? 2,
          supportedTransports: [transport],
          directMemoryDomainIds: transport === "gpudirect_rdma"
            ? [endpointByNode.get(nodeId)!.device.id]
            : [],
          provenance,
        })),
        {
          id: "lan:fabric0",
          kind: "switch",
          bandwidthBytesPerSec:
            options.fabricBandwidthBytesPerSec ?? 50 * GBps,
          latencyNs: options.fabricLatencyNs ?? 500,
          concurrencyLanes: options.fabricConcurrencyLanes ?? nodeIds.length,
          supportedTransports: [transport],
          directMemoryDomainIds: [],
          provenance,
        },
      ]
    : [];
  const networkLinks = nodeIds.flatMap((sourceNodeId) => (
    nodeIds.flatMap((targetNodeId): SimLinkSpec[] => {
      if (sourceNodeId === targetNodeId) {
        return [];
      }
      const source = endpointByNode.get(sourceNodeId)!;
      const target = endpointByNode.get(targetNodeId)!;
      return [{
        id: `lan:${sourceNodeId}:${targetNodeId}`,
        sourceDomainId: transport === "gpudirect_rdma"
          ? source.device.id
          : source.host.id,
        targetDomainId: transport === "gpudirect_rdma"
          ? target.device.id
          : target.host.id,
        kind: linkKind,
        bandwidthBytesPerSec: options.bandwidthBytesPerSec
          ?? (linkKind === "ethernet" ? 10 * GBps : 50 * GBps),
        latencyNs: options.latencyNs
          ?? (linkKind === "ethernet" ? 10_000 : 3_000),
        concurrencyLanes: options.linkConcurrencyLanes ?? 1,
        transport,
        ...(advanced
          ? {
              networkResourceIds: [
                `${sourceNodeId}:nic0`,
                "lan:fabric0",
                `${targetNodeId}:nic0`,
              ],
            }
          : {}),
        provenance,
      }];
    })
  ));
  const domainById = new Map(
    input.memoryDomains.map((domain) => [domain.id, domain]),
  );
  const stagingByNode = new Map<string, string[]>();
  for (const placement of input.placements) {
    for (const allocation of placement.allocations) {
      if (
        allocation.purpose !== "staging"
        || allocation.allocationClass !== "pinned"
      ) {
        continue;
      }
      const nodeId = domainById.get(allocation.domainId)?.nodeId;
      if (nodeId !== undefined) {
        stagingByNode.set(nodeId, [
          ...(stagingByNode.get(nodeId) ?? []),
          allocation.physicalAllocationId,
        ]);
      }
    }
  }
  const transfers = input.transfers.map((entry) => {
    const sourceNodeId = domainById.get(entry.sourceDomainId)?.nodeId;
    const targetNodeId = domainById.get(entry.targetDomainId)?.nodeId;
    if (
      sourceNodeId === undefined
      || targetNodeId === undefined
      || sourceNodeId === targetNodeId
    ) {
      return entry;
    }
    return {
      ...entry,
      requiresPinnedStaging: transport !== "gpudirect_rdma",
      stagingAllocationIds: transport === "gpudirect_rdma"
        ? []
        : [
            ...(stagingByNode.get(sourceNodeId) ?? []),
            ...(stagingByNode.get(targetNodeId) ?? []),
          ],
    };
  });
  const {
    networkResources: _networkResources,
    ...withoutNetworkResources
  } = input;
  const configured: SimulationScenario = {
    ...withoutNetworkResources,
    ...(advanced ? { networkResources } : {}),
    links: [
      ...input.links.filter((entry) => !existingNetworkLinks.includes(entry)),
      ...networkLinks,
    ],
    transfers,
  };
  assertValidScenario(configured);
  return configured;
}

interface ScenarioParts {
  id: string;
  family: SimulationScenario["family"];
  domains: readonly MemoryDomainSpec[];
  devices: readonly SimDeviceSpec[];
  networkResources?: readonly NetworkResourceSpec[];
  links?: readonly SimLinkSpec[];
  placements: readonly PartitionPlacement[];
  transfers?: readonly TransferRequirement[];
  groups: readonly CommunicatorGroupSpec[];
  parallelism: SimulationScenario["execution"]["parallelism"];
  storageCapacityBytes?: number;
  storageBandwidthBytesPerSec?: number;
}

function scenario(parts: ScenarioParts): SimulationScenario {
  const storage = storageTiers(parts);
  const rankedDevices = new Set(
    parts.groups.flatMap((entry) => (
      entry.orderedRanks.map((rank) => rank.deviceId)
    )),
  );
  const hostGroups = storage.devices
    .filter((candidate) => (
      candidate.kind === "cpu" && !rankedDevices.has(candidate.id)
    ))
    .map((candidate): CommunicatorGroupSpec => ({
      id: `host-control:${candidate.id}`,
      orderedRanks: [{
        rankId: `host-rank:${candidate.id}`,
        deviceId: candidate.id,
      }],
    }));
  return {
    schemaVersion: SCENARIO_SCHEMA_VERSION,
    id: parts.id,
    family: parts.family,
    memoryDomains: storage.domains,
    devices: storage.devices,
    ...(parts.networkResources === undefined
      ? {}
      : { networkResources: parts.networkResources }),
    links: storage.links,
    placements: storage.placements,
    transfers: parts.transfers ?? [],
    groups: [...parts.groups, ...hostGroups],
    workload: {
      batchSize: 1,
      inputSequenceLength: 2048,
      outputSequenceLength: 256,
    },
    execution: {
      topologyEpoch: 0,
      seed: 42,
      maxEvents: 1_000_000,
      features: {
        ssdStreaming: true,
      },
      parallelism: parts.parallelism,
    },
    calibration: { coefficients: [] },
  };
}

function storageTiers(parts: ScenarioParts): {
  readonly domains: readonly MemoryDomainSpec[];
  readonly devices: readonly SimDeviceSpec[];
  readonly links: readonly SimLinkSpec[];
  readonly placements: readonly PartitionPlacement[];
} {
  const nodes = [...new Set(parts.devices.map((entry) => entry.nodeId))].sort();
  const domains = [...parts.domains];
  const devices = [...parts.devices];
  const links = [...(parts.links ?? [])];
  const placements = parts.placements.map((entry) => {
    if (!entry.requiredCapabilities.includes("ffn")) {
      return entry;
    }
    const workspace = entry.allocations.find(
      (allocation) => allocation.purpose === "workspace",
    );
    if (workspace === undefined) {
      throw new Error(
        `preset FFN placement ${entry.partitionId} requires a workspace`,
      );
    }
    return {
      ...entry,
      allocations: [
        ...entry.allocations,
        allocation(
          `expert-hot-cache:${entry.partitionId}`,
          workspace.domainId,
          8 * GiB,
          workspace.allocationClass,
          "cache",
        ),
      ],
    };
  });
  for (const nodeId of nodes) {
    const cpuIndex = devices.findIndex((candidate) => (
      candidate.nodeId === nodeId && candidate.kind === "cpu"
    ));
    const warmDomain = domains.find((candidate) => (
      candidate.nodeId === nodeId
      && (candidate.kind === "host" || candidate.kind === "unified")
    ));
    if (cpuIndex < 0 || warmDomain === undefined) {
      throw new Error(
        `preset node ${nodeId} requires a CPU and host/unified warm domain`,
      );
    }
    const cpu = devices[cpuIndex];
    const storageDomainId = `${nodeId}:storage`;
    domains.push({
      id: storageDomainId,
      nodeId,
      kind: "storage",
      capacityBytes: parts.storageCapacityBytes ?? 2 * TiB,
      resourceLimitBytes: parts.storageCapacityBytes ?? 2 * TiB,
      bandwidthBytesPerSec: parts.storageBandwidthBytesPerSec ?? 7 * GBps,
      latencyNs: 50_000,
      coherent: false,
      allocationClasses: ["storage"],
      accessibleBy: [cpu.id],
      governor: { kind: "none" },
      provenance: HEURISTIC,
    });
    devices[cpuIndex] = {
      ...cpu,
      memoryDomainIds: [...cpu.memoryDomainIds, storageDomainId],
    };
    links.push(link(
      `${nodeId}:storage-read`,
      storageDomainId,
      warmDomain.id,
      "storage",
      parts.storageBandwidthBytesPerSec ?? 7 * GBps,
      50_000,
    ));
    placements.push(placement(
      `expert-tier:${nodeId}`,
      cpu.id,
      [],
      [
        allocation(
          `expert-backing:${nodeId}`,
          storageDomainId,
          512 * GiB,
          "storage",
          "backing",
        ),
        allocation(
          `expert-warm-cache:${nodeId}`,
          warmDomain.id,
          8 * GiB,
          warmDomain.kind === "unified" ? "unified" : "pinned",
          "cache",
        ),
      ],
    ));
  }
  return { domains, devices, links, placements };
}

function domain(
  id: string,
  nodeId: string,
  kind: MemoryDomainSpec["kind"],
  capacityBytes: number,
  bandwidthBytesPerSec: number,
  allocationClasses: readonly AllocationClass[],
  accessibleBy: readonly string[],
  governor: MemoryDomainSpec["governor"],
  coherent = false,
): MemoryDomainSpec {
  return {
    id,
    nodeId,
    kind,
    capacityBytes,
    resourceLimitBytes: capacityBytes,
    bandwidthBytesPerSec,
    latencyNs: kind === "host" ? 100 : 80,
    coherent,
    allocationClasses,
    accessibleBy,
    governor,
    provenance: HEURISTIC,
  };
}

function device(
  id: string,
  nodeId: string,
  kind: SimDeviceSpec["kind"],
  executionProvider: string,
  memoryDomainIds: readonly string[],
  capabilities: SimDeviceSpec["capabilities"],
  computeProfileId?: string,
): SimDeviceSpec {
  return {
    id,
    nodeId,
    kind,
    executionProvider,
    memoryDomainIds,
    capabilities: kind === "cpu" && !capabilities.includes("lookup")
      ? [...capabilities, "lookup"]
      : capabilities,
    supportedDtypes: ["fp16", "fp8", "int8"],
    ...(computeProfileId === undefined ? {} : { computeProfileId }),
    maxConcurrentCompute: 1,
    provenance: HEURISTIC,
  };
}

function bidirectionalLink(
  id: string,
  left: string,
  right: string,
  kind: SimLinkSpec["kind"],
  bandwidthBytesPerSec: number,
  latencyNs: number,
): readonly SimLinkSpec[] {
  return [
    link(`${id}:forward`, left, right, kind, bandwidthBytesPerSec, latencyNs),
    link(`${id}:reverse`, right, left, kind, bandwidthBytesPerSec, latencyNs),
  ];
}

function link(
  id: string,
  sourceDomainId: string,
  targetDomainId: string,
  kind: SimLinkSpec["kind"],
  bandwidthBytesPerSec: number,
  latencyNs: number,
): SimLinkSpec {
  return {
    id,
    sourceDomainId,
    targetDomainId,
    kind,
    bandwidthBytesPerSec,
    latencyNs,
    concurrencyLanes: 1,
    provenance: HEURISTIC,
  };
}

function placement(
  partitionId: string,
  deviceId: string,
  requiredCapabilities: PartitionPlacement["requiredCapabilities"],
  allocations: readonly ReturnType<typeof allocation>[],
): PartitionPlacement {
  return { partitionId, deviceId, requiredCapabilities, allocations };
}

function allocation(
  physicalAllocationId: string,
  domainId: string,
  bytes: number,
  allocationClass: AllocationClass,
  purpose: PartitionPlacement["allocations"][number]["purpose"],
): PartitionPlacement["allocations"][number] {
  return {
    physicalAllocationId,
    domainId,
    bytes,
    allocationClass,
    purpose,
  };
}

function transfer(
  id: string,
  sourceDomainId: string,
  targetDomainId: string,
  bytes: number,
  requiresPinnedStaging = false,
  stagingAllocationIds: readonly string[] = [],
): TransferRequirement {
  return {
    id,
    sourceDomainId,
    targetDomainId,
    bytes,
    requiresPinnedStaging,
    stagingAllocationIds,
  };
}

function group(
  id: string,
  ranks: readonly (readonly [number, string])[],
): CommunicatorGroupSpec {
  return {
    id,
    orderedRanks: ranks.map(([rank, deviceId]) => ({
      rankId: `rank-${rank}`,
      deviceId,
    })),
  };
}
