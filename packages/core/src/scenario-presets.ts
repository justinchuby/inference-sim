import {
  SCENARIO_SCHEMA_VERSION,
  type AllocationClass,
  type CommunicatorGroupSpec,
  type EvidenceProvenance,
  type MemoryDomainSpec,
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

export type ScenarioPresetName = typeof SCENARIO_PRESET_NAMES[number];

const PRESET_FACTORIES: Readonly<
  Record<ScenarioPresetName, () => SimulationScenario>
> = {
  "cpu-only": buildCpuOnly,
  "single-gpu-cpu": buildSingleGpu,
  "multi-gpu": buildMultiGpu,
  "gpu-npu": buildGpuNpu,
  "unified-memory": buildUnified,
  "multi-node": buildMultiNode,
};

export function buildScenarioPreset(name: ScenarioPresetName): SimulationScenario {
  const factory = PRESET_FACTORIES[name];
  if (!factory) {
    throw new Error(
      `unknown scenario preset ${String(name)}; available: ${SCENARIO_PRESET_NAMES.join(", ")}`,
    );
  }
  const scenario = factory();
  assertValidScenario(scenario);
  return scenario;
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
    parallelism: { tensor: 1, pipeline: 1, expert: 1, data: 1 },
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
    parallelism: { tensor: 1, pipeline: 1, expert: 1, data: 1 },
  });
}

function buildMultiGpu(): SimulationScenario {
  const cpu = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["copy", "sampling"],
  );
  const gpu0 = device(
    "node0:gpu0",
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    ["node0:gpu0:vram", "node0:host"],
    ["attention", "ffn", "collective", "copy", "sampling", "draft"],
  );
  const gpu1 = device(
    "node0:gpu1",
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    ["node0:gpu1:vram", "node0:host"],
    ["attention", "ffn", "collective", "copy", "sampling", "draft"],
  );
  return scenario({
    id: "multi-gpu",
    family: "multi_gpu",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        512 * GiB,
        120 * GBps,
        ["pageable", "pinned"],
        [cpu.id, gpu0.id, gpu1.id],
        { kind: "host", nodeId: "node0" },
      ),
      domain(
        "node0:gpu0:vram",
        "node0",
        "device",
        80 * GiB,
        3_000 * GBps,
        ["device"],
        [gpu0.id],
        { kind: "device", deviceId: gpu0.id },
      ),
      domain(
        "node0:gpu1:vram",
        "node0",
        "device",
        80 * GiB,
        3_000 * GBps,
        ["device"],
        [gpu1.id],
        { kind: "device", deviceId: gpu1.id },
      ),
    ],
    devices: [cpu, gpu0, gpu1],
    links: [
      ...bidirectionalLink(
        "node0:pcie0",
        "node0:host",
        "node0:gpu0:vram",
        "pcie",
        32 * GBps,
        1_500,
      ),
      ...bidirectionalLink(
        "node0:pcie1",
        "node0:host",
        "node0:gpu1:vram",
        "pcie",
        32 * GBps,
        1_500,
      ),
      ...bidirectionalLink(
        "node0:nvlink",
        "node0:gpu0:vram",
        "node0:gpu1:vram",
        "nvlink",
        600 * GBps,
        500,
      ),
    ],
    placements: [
      placement("target-shard-0", gpu0.id, ["attention", "ffn", "collective"], [
        allocation("weights-0", "node0:gpu0:vram", 36 * GiB, "device", "weights"),
        allocation("kv-0", "node0:gpu0:vram", 8 * GiB, "device", "kv"),
        allocation(
          "workspace-0",
          "node0:gpu0:vram",
          128 * MiB,
          "device",
          "workspace",
        ),
        allocation(
          "expert-cache-host",
          "node0:host",
          256 * MiB,
          "pinned",
          "staging",
        ),
      ]),
      placement("target-shard-1", gpu1.id, ["attention", "ffn", "collective"], [
        allocation("weights-1", "node0:gpu1:vram", 36 * GiB, "device", "weights"),
        allocation("kv-1", "node0:gpu1:vram", 8 * GiB, "device", "kv"),
        allocation(
          "workspace-1",
          "node0:gpu1:vram",
          128 * MiB,
          "device",
          "workspace",
        ),
      ]),
    ],
    transfers: [
      transfer(
        "tensor-parallel",
        "node0:gpu0:vram",
        "node0:gpu1:vram",
        64 * 1024 ** 2,
      ),
    ],
    groups: [group("tp", [[0, gpu0.id], [1, gpu1.id]])],
    parallelism: { tensor: 2, pipeline: 1, expert: 1, data: 1 },
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
    parallelism: { tensor: 1, pipeline: 2, expert: 1, data: 1 },
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
    parallelism: { tensor: 1, pipeline: 1, expert: 1, data: 1 },
  });
}

function buildMultiNode(): SimulationScenario {
  const cpu0 = device(
    "node0:cpu0",
    "node0",
    "cpu",
    "CPUExecutionProvider",
    ["node0:host"],
    ["copy", "sampling"],
  );
  const gpu0 = device(
    "node0:gpu0",
    "node0",
    "gpu",
    "CUDAExecutionProvider",
    ["node0:gpu0:vram", "node0:host"],
    ["attention", "ffn", "collective", "copy", "draft"],
  );
  const cpu1 = device(
    "node1:cpu0",
    "node1",
    "cpu",
    "CPUExecutionProvider",
    ["node1:host"],
    ["copy", "sampling"],
  );
  const gpu1 = device(
    "node1:gpu0",
    "node1",
    "gpu",
    "CUDAExecutionProvider",
    ["node1:gpu0:vram", "node1:host"],
    ["attention", "ffn", "collective", "copy", "draft"],
  );
  return scenario({
    id: "multi-node",
    family: "multi_node",
    domains: [
      domain(
        "node0:host",
        "node0",
        "host",
        256 * GiB,
        100 * GBps,
        ["pageable", "pinned"],
        [cpu0.id, gpu0.id],
        { kind: "host", nodeId: "node0" },
      ),
      domain(
        "node0:gpu0:vram",
        "node0",
        "device",
        80 * GiB,
        2_000 * GBps,
        ["device"],
        [gpu0.id],
        { kind: "device", deviceId: gpu0.id },
      ),
      domain(
        "node1:host",
        "node1",
        "host",
        256 * GiB,
        100 * GBps,
        ["pageable", "pinned"],
        [cpu1.id, gpu1.id],
        { kind: "host", nodeId: "node1" },
      ),
      domain(
        "node1:gpu0:vram",
        "node1",
        "device",
        80 * GiB,
        2_000 * GBps,
        ["device"],
        [gpu1.id],
        { kind: "device", deviceId: gpu1.id },
      ),
    ],
    devices: [cpu0, gpu0, cpu1, gpu1],
    links: [
      ...bidirectionalLink(
        "node0:pcie",
        "node0:host",
        "node0:gpu0:vram",
        "pcie",
        32 * GBps,
        1_500,
      ),
      ...bidirectionalLink(
        "node1:pcie",
        "node1:host",
        "node1:gpu0:vram",
        "pcie",
        32 * GBps,
        1_500,
      ),
      ...bidirectionalLink(
        "cluster:rdma",
        "node0:host",
        "node1:host",
        "infiniband",
        50 * GBps,
        3_000,
      ),
    ],
    placements: [
      placement("target-shard-0", gpu0.id, ["attention", "ffn", "collective"], [
        allocation("weights-0", "node0:gpu0:vram", 36 * GiB, "device", "weights"),
        allocation("kv-0", "node0:gpu0:vram", 8 * GiB, "device", "kv"),
        allocation(
          "workspace-0",
          "node0:gpu0:vram",
          128 * MiB,
          "device",
          "workspace",
        ),
        allocation("staging-0", "node0:host", 1 * GiB, "pinned", "staging"),
      ]),
      placement("target-shard-1", gpu1.id, ["attention", "ffn", "collective"], [
        allocation("weights-1", "node1:gpu0:vram", 36 * GiB, "device", "weights"),
        allocation("kv-1", "node1:gpu0:vram", 8 * GiB, "device", "kv"),
        allocation(
          "workspace-1",
          "node1:gpu0:vram",
          128 * MiB,
          "device",
          "workspace",
        ),
        allocation("staging-1", "node1:host", 1 * GiB, "pinned", "staging"),
      ]),
    ],
    transfers: [
      transfer(
        "cross-node-collective",
        "node0:gpu0:vram",
        "node1:gpu0:vram",
        64 * 1024 ** 2,
        true,
        ["staging-0", "staging-1"],
      ),
    ],
    groups: [group("tp", [[0, gpu0.id], [1, gpu1.id]])],
    parallelism: { tensor: 2, pipeline: 1, expert: 1, data: 1 },
  });
}

interface ScenarioParts {
  id: string;
  family: SimulationScenario["family"];
  domains: readonly MemoryDomainSpec[];
  devices: readonly SimDeviceSpec[];
  links?: readonly SimLinkSpec[];
  placements: readonly PartitionPlacement[];
  transfers?: readonly TransferRequirement[];
  groups: readonly CommunicatorGroupSpec[];
  parallelism: SimulationScenario["execution"]["parallelism"];
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
  const placements = [...parts.placements];
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
      capacityBytes: 2 * TiB,
      bandwidthBytesPerSec: 7 * GBps,
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
      7 * GBps,
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
