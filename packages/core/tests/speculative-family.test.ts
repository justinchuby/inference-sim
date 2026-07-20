import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  SpeculativeEligibilityError,
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  defaultSpeculativeEligibility,
  simulateSpeculativeWorkload,
  simulateTopologyWorkload,
  speculativeFamilyContract,
  topologyProfileFromSpeculative,
  validateSpeculativeEligibility,
  type SpeculativeProposerFamily,
} from "../src/index.js";

const FAMILIES: readonly SpeculativeProposerFamily[] = [
  "prompt_lookup",
  "draft_model",
  "mtp",
  "eagle3",
  "shared_kv",
  "self_speculative",
];

function simulateFamily(family: SpeculativeProposerFamily) {
  const initialTokenLength = 32;
  const outputTokenCount = 9;
  const maxAdditionalTokens = 3;
  return simulateSpeculativeWorkload({
    family,
    eligibility: defaultSpeculativeEligibility(family),
    initialTokenLength,
    outputTokenCount,
    maxAdditionalTokens,
    stateGroups: buildSpeculativeStateGroups(
      family,
      initialTokenLength + outputTokenCount + maxAdditionalTokens,
      maxAdditionalTokens,
    ),
    acceptance: {
      kind: "conditional_heuristic",
      matchProbabilityByPosition: [1, 1, 1],
      seed: 7,
    },
    pagedKv: {
      pageSizeTokens: 4,
      bytesPerToken: 128,
      capacityBytes: 64 * 128,
    },
  });
}

describe("speculative family contracts", () => {
  it("preserves target-only committed state for every proposer family", () => {
    for (const family of FAMILIES) {
      const result = simulateFamily(family);
      expect(result.finalTokenLength, family).toBe(41);
      expect(result.targetOnlyFinalTokenLength, family).toBe(41);
      expect(result.pagedKv?.snapshot.logicalTokenLength, family).toBe(41);
      for (const group of result.stateGroups) {
        expect(group.logicalLength, `${family}:${group.role}`).toBe(
          group.lifetime === "committed_prefix" ? 41 : 0,
        );
      }
    }
  });

  it("models family-owned state instead of assigning generic proposer KV", () => {
    const prompt = buildSpeculativeStateGroups("prompt_lookup", 64, 4);
    expect(prompt.map((group) => group.role)).toEqual(["target_kv"]);

    const draft = buildSpeculativeStateGroups("draft_model", 64, 4);
    expect(draft.map((group) => group.role)).toEqual([
      "target_kv",
      "draft_kv",
    ]);

    const mtp = buildSpeculativeStateGroups("mtp", 64, 4);
    expect(mtp.filter((group) => group.lifetime === "proposal_local"))
      .toHaveLength(2);

    const shared = buildSpeculativeStateGroups("shared_kv", 64, 4);
    expect(shared.some((group) => group.role === "shared_kv_lease")).toBe(true);
    expect(shared.some((group) => group.role === "draft_kv")).toBe(false);
  });

  it("matches onnx-genai family-specific proposal prefixes", () => {
    expect(speculativeFamilyContract("prompt_lookup").proposalPrefix).toBe("none");
    expect(speculativeFamilyContract("draft_model").proposalPrefix).toBe("none");
    expect(speculativeFamilyContract("mtp").proposalPrefix)
      .toBe("guaranteed_target");
    expect(speculativeFamilyContract("eagle3").proposalPrefix)
      .toBe("guaranteed_target");
    expect(speculativeFamilyContract("shared_kv").proposalPrefix)
      .toBe("guaranteed_target");
    expect(speculativeFamilyContract("self_speculative").proposalPrefix)
      .toBe("none");
  });

  it("fails closed on current onnx-genai eligibility restrictions", () => {
    const valid = defaultSpeculativeEligibility("mtp");
    expect(() => validateSpeculativeEligibility("mtp", {
      ...valid,
      grammarActive: true,
    })).toThrowError(SpeculativeEligibilityError);
    expect(() => validateSpeculativeEligibility("mtp", {
      ...valid,
      decoding: "sampling",
    })).toThrow("requires greedy or temperature-zero");
    expect(() => validateSpeculativeEligibility("eagle3", {
      ...defaultSpeculativeEligibility("eagle3"),
      targetHiddenOutputCount: 2,
    })).toThrow("requires 3 target hidden output");
    expect(() => validateSpeculativeEligibility("shared_kv", {
      ...defaultSpeculativeEligibility("shared_kv"),
      sharedKvGroupCount: 0,
    })).toThrow("requires at least one declared shared-KV group");
    expect(() => simulateSpeculativeWorkload({
      family: "mtp",
      eligibility: defaultSpeculativeEligibility("mtp"),
      initialTokenLength: 0,
      outputTokenCount: 1,
      maxAdditionalTokens: 1,
      stateGroups: [{
        id: "target-only",
        owner: "target",
        role: "target_kv",
        lifetime: "committed_prefix",
        capacityTokens: 2,
        rollbackProtection: { kind: "non_destructive_tail" },
      }],
      acceptance: { kind: "replay", acceptedDraftTokens: [0] },
    })).toThrow("requires proposer sidecar_kv");
  });

  it("marks self speculative as design-only and validates its layer split", () => {
    expect(speculativeFamilyContract("self_speculative").support)
      .toBe("design_only");
    expect(() => validateSpeculativeEligibility("self_speculative", {
      ...defaultSpeculativeEligibility("self_speculative"),
      allowDesignOnly: false,
    })).toThrow("requires allowDesignOnly");
    expect(() => validateSpeculativeEligibility("self_speculative", {
      ...defaultSpeculativeEligibility("self_speculative"),
      earlyExitLayer: 32,
    })).toThrow("earlyExitLayer < targetLayerCount");
  });

  it("executes all six families across all six device topologies", () => {
    for (const family of FAMILIES) {
      const profile = topologyProfileFromSpeculative(simulateFamily(family));
      for (const scenarioName of SCENARIO_PRESET_NAMES) {
        const result = simulateTopologyWorkload(
          buildScenarioPreset(scenarioName),
          profile,
        );
        expect(result.execution.status, `${family}:${scenarioName}`)
          .toBe("succeeded");
        expect(result.metrics.committedTokens, `${family}:${scenarioName}`)
          .toBe(9);
      }
    }
  });

  it("places prompt lookup on host CPU and target-coupled sidecars on target", () => {
    const scenario = buildScenarioPreset("single-gpu-cpu");
    const prompt = simulateTopologyWorkload(
      scenario,
      topologyProfileFromSpeculative(simulateFamily("prompt_lookup")),
    );
    const mtp = simulateTopologyWorkload(
      scenario,
      topologyProfileFromSpeculative(simulateFamily("mtp")),
    );
    const computeDevices = (result: typeof prompt) => (
      result.plan.steps.flatMap((step) => (
        step.operation.kind === "compute"
          && (step.operation.capability === "lookup"
            || step.operation.capability === "draft")
          ? [step.operation.deviceId]
          : []
      ))
    );

    expect(computeDevices(prompt)).toContain("node0:cpu0");
    expect(computeDevices(mtp)).toContain("node0:gpu0");
  });

  it("honors an explicit separate draft-model placement", () => {
    const base = buildScenarioPreset("single-gpu-cpu");
    const scenario = {
      ...base,
      devices: base.devices.map((device) => (
        device.id === "node0:cpu0"
          ? { ...device, capabilities: [...device.capabilities, "draft" as const] }
          : device
      )),
      placements: [
        ...base.placements,
        {
          partitionId: "draft-model",
          deviceId: "node0:cpu0",
          requiredCapabilities: ["draft" as const],
          allocations: [
            {
              physicalAllocationId: "draft-weights",
              domainId: "node0:host",
              bytes: 1024,
              allocationClass: "pageable" as const,
              purpose: "weights" as const,
            },
            {
              physicalAllocationId: "draft-workspace",
              domainId: "node0:host",
              bytes: 1024,
              allocationClass: "pageable" as const,
              purpose: "workspace" as const,
            },
          ],
        },
      ],
    };
    const result = simulateTopologyWorkload(
      scenario,
      topologyProfileFromSpeculative(simulateFamily("draft_model")),
    );
    const draftDevices = result.plan.steps.flatMap((step) => (
      step.operation.kind === "compute"
      && step.operation.capability === "draft"
        ? [step.operation.deviceId]
        : []
    ));

    expect(draftDevices.length).toBeGreaterThan(0);
    expect(draftDevices.every((deviceId) => deviceId === "node0:cpu0"))
      .toBe(true);
  });
});
