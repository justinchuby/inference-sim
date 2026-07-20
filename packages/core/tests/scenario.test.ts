import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  ScenarioValidationError,
  assertValidScenario,
  buildMultiGpuRingScenario,
  buildScenarioPreset,
  calculateScenarioMemoryLedger,
  findTransferPath,
  findTransferRoute,
  validateScenario,
  type MemoryDomainSpec,
  type SimLinkSpec,
  type SimulationScenario,
  type TransferRequirement,
} from "../src/index.js";

const ROUTE_PROVENANCE = {
  confidence: "heuristic" as const,
  source: "route-selection test fixture",
};

function routeDomain(
  id: string,
  allocationClasses: MemoryDomainSpec["allocationClasses"],
): MemoryDomainSpec {
  return {
    id,
    nodeId: "node0",
    kind: allocationClasses.includes("device") ? "device" : "host",
    capacityBytes: 1024 ** 3,
    bandwidthBytesPerSec: 100_000_000_000,
    latencyNs: 0,
    coherent: false,
    allocationClasses,
    accessibleBy: [],
    governor: { kind: "none" },
    provenance: ROUTE_PROVENANCE,
  };
}

function routeLink(
  id: string,
  sourceDomainId: string,
  targetDomainId: string,
  bandwidthBytesPerSec: number,
  latencyNs: number,
): SimLinkSpec {
  return {
    id,
    sourceDomainId,
    targetDomainId,
    kind: "pcie",
    bandwidthBytesPerSec,
    latencyNs,
    concurrencyLanes: 1,
    provenance: ROUTE_PROVENANCE,
  };
}

function routeRequirement(
  bytes: number,
  requiresPinnedStaging = false,
): TransferRequirement {
  return {
    id: "route",
    sourceDomainId: "source",
    targetDomainId: "target",
    bytes,
    requiresPinnedStaging,
    stagingAllocationIds: [],
  };
}

describe("scenario presets", () => {
  it("validates all six required topology families", () => {
    const families = SCENARIO_PRESET_NAMES.map((name) => {
      const scenario = buildScenarioPreset(name);
      expect(validateScenario(scenario)).toEqual({ valid: true, issues: [] });
      for (const placement of scenario.placements.filter((candidate) => (
        candidate.requiredCapabilities.includes("ffn")
      ))) {
        const workspace = placement.allocations.find(
          (allocation) => allocation.purpose === "workspace",
        );
        const hotCaches = placement.allocations.filter(
          (allocation) => allocation.purpose === "cache",
        );
        expect(hotCaches, placement.partitionId).toHaveLength(1);
        expect(hotCaches[0].domainId, placement.partitionId)
          .toBe(workspace?.domainId);
      }
      return scenario.family;
    });

    expect(families).toEqual([
      "cpu_only",
      "single_discrete",
      "multi_gpu",
      "gpu_npu",
      "unified",
      "multi_node",
    ]);
  });

  it("rejects scenarios from the previous routing contract revision", () => {
    const scenario = buildScenarioPreset("single-gpu-cpu");
    const result = validateScenario({
      ...scenario,
      schemaVersion: 3 as typeof scenario.schemaVersion,
    });

    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual({
      code: "schema_version",
      path: "schemaVersion",
      message: "expected 4, got 3",
    });
  });

  it("keeps unified compute memory distinct from cold storage", () => {
    const scenario = buildScenarioPreset("unified-memory");
    const unified = scenario.memoryDomains.filter(
      (domain) => domain.kind === "unified",
    );
    const storage = scenario.memoryDomains.filter(
      (domain) => domain.kind === "storage",
    );

    expect(unified).toHaveLength(1);
    expect(storage).toHaveLength(1);
    expect(unified[0].accessibleBy).toEqual([
      "node0:cpu0",
      "node0:gpu0",
    ]);
    expect(
      scenario.placements
        .filter((placement) => placement.partitionId === "target")
        .flatMap((placement) => placement.allocations)
        .every((allocation) => allocation.domainId === "node0:unified"),
    ).toBe(true);
    expect(calculateScenarioMemoryLedger(scenario)).toEqual([
      {
        domainId: "node0:storage",
        capacityBytes: 2 * 1024 ** 4,
        reservedBytes: 512 * 1024 ** 3,
        freeBytes: 1536 * 1024 ** 3,
      },
      {
        domainId: "node0:unified",
        capacityBytes: 128 * 1024 ** 3,
        reservedBytes: 92 * 1024 ** 3 + 256 * 1024 ** 2,
        freeBytes: 36 * 1024 ** 3 - 256 * 1024 ** 2,
      },
    ]);
  });

  it("finds mandatory pinned staging for GPU/NPU and multi-node transfers", () => {
    const heterogeneous = buildScenarioPreset("gpu-npu");
    const heteroPath = findTransferPath(
      heterogeneous,
      heterogeneous.transfers[0],
    );
    expect(heteroPath).toEqual([
      "node0:npu0:memory",
      "node0:host",
      "node0:gpu0:vram",
    ]);

    const multiNode = buildScenarioPreset("multi-node");
    expect(findTransferPath(multiNode, multiNode.transfers[0])).toEqual([
      "node0:gpu0:vram",
      "node0:host",
      "node1:host",
      "node1:gpu0:vram",
    ]);
  });

  it("selects a deterministic message-size-aware transfer route", () => {
    const memoryDomains = [
      routeDomain("source", ["pageable"]),
      routeDomain("relay", ["pageable"]),
      routeDomain("target", ["device"]),
    ];
    const links = [
      routeLink("direct", "source", "target", 100_000_000_000, 10_000),
      routeLink("relay-in", "source", "relay", 1_000_000_000, 0),
      routeLink("relay-out", "relay", "target", 1_000_000_000, 0),
    ];
    const small = findTransferRoute(
      { memoryDomains, links },
      routeRequirement(1),
    );
    const large = findTransferRoute(
      { memoryDomains, links },
      routeRequirement(1_000_000_000),
    );

    expect(small).toEqual({
      domainIds: ["source", "relay", "target"],
      linkIds: ["relay-in", "relay-out"],
      declaredDurationNs: 2,
    });
    expect(large).toEqual({
      domainIds: ["source", "target"],
      linkIds: ["direct"],
      declaredDurationNs: 10_010_000,
    });
  });

  it("chooses the stable fastest parallel link independent of input order", () => {
    const memoryDomains = [
      routeDomain("source", ["pageable"]),
      routeDomain("target", ["device"]),
    ];
    const links = [
      routeLink("z-slow", "source", "target", 1_000_000, 50),
      routeLink("z-fast", "source", "target", 1_000_000_000, 50),
      routeLink("a-fast", "source", "target", 1_000_000_000, 50),
    ];
    const forward = findTransferRoute(
      { memoryDomains, links },
      routeRequirement(1_000),
    );
    const reversed = findTransferRoute(
      { memoryDomains, links: [...links].reverse() },
      routeRequirement(1_000),
    );

    expect(forward?.linkIds).toEqual(["a-fast"]);
    expect(reversed).toEqual(forward);
    expect(findTransferPath(
      { memoryDomains, links },
      routeRequirement(1_000),
    )).toEqual(["source", "target"]);
  });

  it("requires a real pinned intermediate rather than a target cycle", () => {
    const memoryDomains = [
      routeDomain("source", ["pageable"]),
      routeDomain("relay", ["pinned"]),
      routeDomain("target", ["device"]),
    ];
    const direct = routeLink(
      "direct",
      "source",
      "target",
      100_000_000_000,
      0,
    );
    const relayLinks = [
      routeLink("relay-in", "source", "relay", 1_000_000_000, 0),
      routeLink("relay-out", "relay", "target", 1_000_000_000, 0),
    ];
    const pinned = findTransferRoute(
      { memoryDomains, links: [direct, ...relayLinks] },
      routeRequirement(1_000, true),
    );

    expect(pinned?.domainIds).toEqual(["source", "relay", "target"]);
    expect(findTransferRoute(
      {
        memoryDomains,
        links: [
          direct,
          routeLink("target-relay", "target", "relay", 1_000_000_000, 0),
          routeLink("relay-target", "relay", "target", 1_000_000_000, 0),
        ],
      },
      routeRequirement(1_000, true),
    )).toBeUndefined();
    expect(findTransferRoute(
      {
        memoryDomains,
        links: [
          direct,
          routeLink("source-relay", "source", "relay", 1_000_000_000, 0),
          routeLink("relay-source", "relay", "source", 1_000_000_000, 0),
        ],
      },
      routeRequirement(1_000, true),
    )).toBeUndefined();
  });

  it("retains a costlier partial route when the cheapest label blocks completion", () => {
    const memoryDomains = [
      routeDomain("source", ["pageable"]),
      routeDomain("block", ["pageable"]),
      routeDomain("cheap-pinned", ["pinned"]),
      routeDomain("costly-pinned", ["pinned"]),
      routeDomain("join", ["pageable"]),
      routeDomain("target", ["device"]),
    ];
    const links = [
      routeLink("cheap-a", "source", "block", 1_000_000_000, 0),
      routeLink("cheap-b", "block", "cheap-pinned", 1_000_000_000, 0),
      routeLink("cheap-c", "cheap-pinned", "join", 1_000_000_000, 0),
      routeLink("costly-a", "source", "costly-pinned", 1_000_000_000, 100),
      routeLink("costly-b", "costly-pinned", "join", 1_000_000_000, 0),
      routeLink("finish-a", "join", "block", 1_000_000_000, 0),
      routeLink("finish-b", "block", "target", 1_000_000_000, 0),
    ];

    expect(findTransferRoute(
      { memoryDomains, links },
      routeRequirement(1, true),
    )?.linkIds).toEqual([
      "costly-a",
      "costly-b",
      "finish-a",
      "finish-b",
    ]);
  });

  it("fails closed for invalid transfer sizes and impossible local staging", () => {
    const memoryDomains = [
      routeDomain("source", ["pinned"]),
      routeDomain("target", ["device"]),
    ];
    const links = [
      routeLink("direct", "source", "target", 1_000_000_000, 0),
    ];

    expect(findTransferRoute(
      { memoryDomains, links },
      routeRequirement(0),
    )).toBeUndefined();
    expect(findTransferRoute(
      { memoryDomains, links },
      {
        ...routeRequirement(1, true),
        targetDomainId: "source",
      },
    )).toBeUndefined();
  });

  it("builds validated parameterized multi-GPU rings", () => {
    for (const gpuCount of [2, 4, 8]) {
      const scenario = buildMultiGpuRingScenario(gpuCount);
      const targetPlacements = scenario.placements.filter((placement) => (
        placement.requiredCapabilities.includes("attention")
      ));
      const group = scenario.groups.find((candidate) => candidate.id === "tp");
      const ringLinks = scenario.links.filter((link) => link.kind === "nvlink");

      expect(validateScenario(scenario)).toEqual({ valid: true, issues: [] });
      expect(scenario.id).toBe(`multi-gpu-ring-${gpuCount}`);
      expect(targetPlacements).toHaveLength(gpuCount);
      expect(group?.orderedRanks).toHaveLength(gpuCount);
      expect(ringLinks).toHaveLength(gpuCount === 2 ? 2 : gpuCount * 2);
      expect(scenario.execution.parallelism).toMatchObject({
        composition: "overlap_by_capability",
        tensor: gpuCount,
        expert: gpuCount,
      });
    }
  });

  it("rejects unsafe multi-GPU ring sizes", () => {
    for (const gpuCount of [1, 2.5, 65, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => buildMultiGpuRingScenario(gpuCount)).toThrow(
        "multi-GPU ring count must be a safe integer from 2 through 64",
      );
    }
  });
});

describe("validateScenario", () => {
  it("allows a shared-KV alias without charging the physical allocation twice", () => {
    const base = buildScenarioPreset("unified-memory");
    const scenario: SimulationScenario = {
      ...base,
      placements: [
        ...base.placements,
        {
          partitionId: "draft",
          deviceId: "node0:cpu0",
          requiredCapabilities: ["draft"],
          allocations: [],
          sharedAllocationIds: ["target-kv"],
        },
      ],
    };

    expect(validateScenario(scenario)).toEqual({ valid: true, issues: [] });
    expect(calculateScenarioMemoryLedger(scenario).find(
      (entry) => entry.domainId === "node0:unified",
    )?.reservedBytes).toBe(
      92 * 1024 ** 3 + 256 * 1024 ** 2,
    );
  });

  it("rejects a placement on a device without the required capability", () => {
    const base = buildScenarioPreset("gpu-npu");
    const scenario: SimulationScenario = {
      ...base,
      placements: base.placements.map((placement) => (
        placement.partitionId === "attention"
          ? { ...placement, deviceId: "node0:gpu0" }
          : placement
      )),
    };

    const result = validateScenario(scenario);
    expect(result.valid).toBe(false);
    expect(result.issues.some((issue) => issue.code === "missing_capability")).toBe(
      true,
    );
  });

  it("rejects duplicate physical allocations even across partitions", () => {
    const base = buildScenarioPreset("multi-gpu");
    const duplicate = base.placements[0].allocations[0].physicalAllocationId;
    const scenario: SimulationScenario = {
      ...base,
      placements: [
        base.placements[0],
        {
          ...base.placements[1],
          allocations: base.placements[1].allocations.map((allocation, index) => (
            index === 0 ? { ...allocation, physicalAllocationId: duplicate } : allocation
          )),
        },
      ],
    };

    expect(
      validateScenario(scenario).issues.some(
        (issue) => issue.code === "duplicate_allocation",
      ),
    ).toBe(true);
  });

  it("rejects domain overcommit with exact integer accounting", () => {
    const base = buildScenarioPreset("single-gpu-cpu");
    const scenario: SimulationScenario = {
      ...base,
      placements: base.placements.map((placement) => ({
        ...placement,
        allocations: placement.allocations.map((allocation) => (
          allocation.domainId === "node0:gpu0:vram"
            ? { ...allocation, bytes: 70 * 1024 ** 3 }
            : allocation
        )),
      })),
    };

    expect(
      validateScenario(scenario).issues.some(
        (issue) => issue.code === "domain_over_capacity",
      ),
    ).toBe(true);
  });

  it("rejects a pinned-staging transfer if the staging class disappears", () => {
    const base = buildScenarioPreset("gpu-npu");
    const scenario: SimulationScenario = {
      ...base,
      memoryDomains: base.memoryDomains.map((domain) => (
        domain.id === "node0:host"
          ? { ...domain, allocationClasses: ["pageable"] }
          : domain
      )),
    };

    const result = validateScenario(scenario);
    expect(result.issues.some((issue) => issue.code === "no_transfer_path")).toBe(
      true,
    );
    expect(() => assertValidScenario(scenario)).toThrowError(
      ScenarioValidationError,
    );
  });

  it("rejects a staging allocation smaller than the transfer extent", () => {
    const base = buildScenarioPreset("gpu-npu");
    const scenario: SimulationScenario = {
      ...base,
      placements: base.placements.map((placement) => ({
        ...placement,
        allocations: placement.allocations.map((allocation) => (
          allocation.physicalAllocationId === "npu-staging"
            ? { ...allocation, bytes: 1 }
            : allocation
        )),
      })),
    };

    expect(
      validateScenario(scenario).issues.some(
        (issue) => issue.code === "invalid_staging",
      ),
    ).toBe(true);
  });

  it("rejects asymmetric domain accessibility", () => {
    const base = buildScenarioPreset("single-gpu-cpu");
    const scenario: SimulationScenario = {
      ...base,
      memoryDomains: base.memoryDomains.map((domain) => (
        domain.id === "node0:host"
          ? { ...domain, accessibleBy: ["node0:cpu0"] }
          : domain
      )),
    };

    expect(
      validateScenario(scenario).issues.some(
        (issue) => issue.code === "asymmetric_access",
      ),
    ).toBe(true);
  });

  it("rejects storage domains with memory governors or non-storage classes", () => {
    const base = buildScenarioPreset("cpu-only");
    const scenario: SimulationScenario = {
      ...base,
      memoryDomains: base.memoryDomains.map((domain) => (
        domain.kind === "storage"
          ? {
              ...domain,
              allocationClasses: ["pinned"],
              governor: { kind: "host", nodeId: "node0" },
            }
          : domain
      )),
    };
    const issues = validateScenario(scenario).issues;

    expect(issues.some((issue) => issue.code === "storage_class")).toBe(true);
    expect(issues.some((issue) => issue.code === "storage_governor")).toBe(true);
  });

  it("rejects ambiguous overlap and undersized Cartesian parallelism", () => {
    const base = buildScenarioPreset("multi-gpu");
    const unequal: SimulationScenario = {
      ...base,
      execution: {
        ...base.execution,
        parallelism: {
          ...base.execution.parallelism,
          expert: 1,
        },
      },
    };
    const cartesian: SimulationScenario = {
      ...base,
      execution: {
        ...base.execution,
        parallelism: {
          ...base.execution.parallelism,
          composition: "cartesian",
        },
      },
    };
    const missingGroup: SimulationScenario = {
      ...base,
      groups: [],
    };

    expect(validateScenario(unequal).issues.some(
      (issue) => issue.code === "parallelism_composition",
    )).toBe(true);
    expect(validateScenario(cartesian).issues.some(
      (issue) => issue.code === "parallelism_participants",
    )).toBe(true);
    expect(validateScenario(missingGroup).issues.some(
      (issue) => (
        issue.code === "parallelism_composition"
        && issue.message.includes("2-rank communicator")
      ),
    )).toBe(true);
  });
});
