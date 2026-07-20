import { describe, expect, it } from "vitest";
import {
  FROZEN_PLAN_ARTIFACT_REVISION,
  PLAN_CONTRACT_REVISION,
  buildMultiGpuRingScenario,
  buildScenarioPreset,
  canonicalJsonFingerprint,
  compileTopologyWorkloadPlan,
  createFrozenPlanArtifact,
  executeFrozenPlan,
  parseFrozenPlanArtifact,
  replayPlanTrace,
  serializeFrozenPlanArtifact,
  targetOnlyTopologyProfile,
} from "../src/index.js";

function fixture() {
  const scenario = buildScenarioPreset("multi-gpu");
  const plan = compileTopologyWorkloadPlan(
    scenario,
    targetOnlyTopologyProfile(3),
  );
  return { scenario, plan };
}

describe("FrozenPlan artifacts", () => {
  it("round-trips deterministically and executes with exact replay", () => {
    const { scenario, plan } = fixture();
    const first = createFrozenPlanArtifact(scenario, plan);
    const second = createFrozenPlanArtifact(scenario, plan);

    expect(first).toEqual(second);
    expect(serializeFrozenPlanArtifact(first)).toBe(
      serializeFrozenPlanArtifact(second),
    );

    const parsed = parseFrozenPlanArtifact(
      JSON.parse(serializeFrozenPlanArtifact(first)),
    );
    const execution = executeFrozenPlan(parsed.scenario, parsed.plan);
    const replay = replayPlanTrace(
      parsed.scenario,
      parsed.plan,
      execution.trace,
    );
    expect(execution.status).toBe("succeeded");
    expect(replay).toMatchObject({
      status: "succeeded",
      completedAtNs: execution.completedAtNs,
    });
    expect(replay.appliedEvents).toBe(execution.trace.operations.length + 1);
  });

  it("rejects tampering at the narrowest fingerprint boundary", () => {
    const { scenario, plan } = fixture();
    const artifact = createFrozenPlanArtifact(scenario, plan);
    const tampered = structuredClone(artifact);
    (tampered.plan.steps[0].operation as { durationNs: number }).durationNs++;

    expect(() => parseFrozenPlanArtifact(tampered))
      .toThrow("plan fingerprint mismatch");
  });

  it("rejects stale revisions even when the envelope is re-fingerprinted", () => {
    const { scenario, plan } = fixture();
    const artifact = createFrozenPlanArtifact(scenario, plan);
    const stale = {
      ...artifact,
      revision: FROZEN_PLAN_ARTIFACT_REVISION + 1,
    };
    const { artifactFingerprint: _ignored, ...unsigned } = stale;
    const resigned = {
      ...stale,
      artifactFingerprint: canonicalJsonFingerprint(unsigned),
    };

    expect(() => parseFrozenPlanArtifact(resigned))
      .toThrow("artifact revision must be 1");
  });

  it("rejects malformed operations before semantic validation", () => {
    const { scenario, plan } = fixture();
    const malformedPlan = structuredClone(plan) as unknown as {
      steps: Array<{ operation: Record<string, unknown> }>;
    };
    malformedPlan.steps[0].operation.unownedField = true;
    const malformed = {
      ...createFrozenPlanArtifact(scenario, plan),
      plan: malformedPlan,
      planFingerprint: canonicalJsonFingerprint(malformedPlan),
    };
    const unsigned = {
      kind: malformed.kind,
      revision: malformed.revision,
      scenarioSchemaVersion: malformed.scenarioSchemaVersion,
      planContractRevision: malformed.planContractRevision,
      scenarioFingerprint: malformed.scenarioFingerprint,
      planFingerprint: malformed.planFingerprint,
      scenario: malformed.scenario,
      plan: malformed.plan,
    };

    expect(() => parseFrozenPlanArtifact({
      ...malformed,
      artifactFingerprint: canonicalJsonFingerprint(unsigned),
    })).toThrow("unknown fields unownedField");
  });

  it("rejects unknown fields nested inside the embedded scenario", () => {
    const { scenario, plan } = fixture();
    const artifact = createFrozenPlanArtifact(scenario, plan);
    const malformed = structuredClone(artifact) as unknown as {
      scenario: {
        devices: Array<Record<string, unknown>>;
      };
    };
    malformed.scenario.devices[0].implicitRuntimeDefault = true;

    expect(() => parseFrozenPlanArtifact(malformed))
      .toThrow("unknown fields implicitRuntimeDefault");
  });

  it("rejects a valid plan rebound to a different topology", () => {
    const { plan } = fixture();
    const scenario = buildScenarioPreset("cpu-only");

    expect(() => createFrozenPlanArtifact(scenario, plan))
      .toThrow("semantic validation failed");
  });

  it("records the exact current plan revision", () => {
    const { scenario, plan } = fixture();
    expect(createFrozenPlanArtifact(scenario, plan).planContractRevision)
      .toBe(PLAN_CONTRACT_REVISION);
  });

  it("round-trips the maximum supported parameterized topology", () => {
    const scenario = buildMultiGpuRingScenario(64);
    const plan = compileTopologyWorkloadPlan(
      scenario,
      targetOnlyTopologyProfile(1),
    );
    const parsed = parseFrozenPlanArtifact(
      createFrozenPlanArtifact(scenario, plan),
    );

    expect(parsed.scenario.groups.find((group) => group.id === "tp")
      ?.orderedRanks).toHaveLength(64);
    expect(executeFrozenPlan(parsed.scenario, parsed.plan).status)
      .toBe("succeeded");
  });
});
