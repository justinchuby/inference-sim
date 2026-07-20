import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  ScenarioValidationError,
  assertValidScenario,
  buildMultiGpuRingScenario,
  buildScenarioPreset,
  calculateScenarioMemoryLedger,
  findTransferPath,
  validateScenario,
  type SimulationScenario,
} from "../src/index.js";

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
