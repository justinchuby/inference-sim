import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  ScenarioValidationError,
  assertValidScenario,
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

  it("uses one physical domain for unified host and device access", () => {
    const scenario = buildScenarioPreset("unified-memory");

    expect(scenario.memoryDomains).toHaveLength(1);
    expect(scenario.memoryDomains[0].kind).toBe("unified");
    expect(scenario.memoryDomains[0].accessibleBy).toEqual([
      "node0:cpu0",
      "node0:gpu0",
    ]);
    expect(
      scenario.placements.flatMap((placement) => placement.allocations)
        .every((allocation) => allocation.domainId === "node0:unified"),
    ).toBe(true);
    expect(calculateScenarioMemoryLedger(scenario)).toEqual([{
      domainId: "node0:unified",
      capacityBytes: 128 * 1024 ** 3,
      reservedBytes: 76 * 1024 ** 3 + 256 * 1024 ** 2,
      freeBytes: 52 * 1024 ** 3 - 256 * 1024 ** 2,
    }]);
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
    expect(calculateScenarioMemoryLedger(scenario)[0].reservedBytes).toBe(
      76 * 1024 ** 3 + 256 * 1024 ** 2,
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
});
