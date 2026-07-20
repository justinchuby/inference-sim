import { describe, expect, it } from "vitest";
import {
  buildMultiGpuRingScenario,
  compileTopologyWorkloadPlan,
  createFrozenPlanArtifact,
  serializeFrozenPlanArtifact,
  targetOnlyTopologyProfile,
} from "@inference-sim/core";
import { parseFrozenPlanArtifactFileText } from "./frozen-plan-import.js";
import { executeFrozenPlanWorkerRun } from "./frozen-plan-worker-run.js";

function fixtureText(tokenCount = 3): string {
  const scenario = buildMultiGpuRingScenario(4);
  const plan = compileTopologyWorkloadPlan(
    scenario,
    targetOnlyTopologyProfile(tokenCount),
  );
  return serializeFrozenPlanArtifact(
    createFrozenPlanArtifact(scenario, plan),
    true,
  );
}

describe("FrozenPlan browser import", () => {
  it("parses the shared core artifact contract", () => {
    const artifact = parseFrozenPlanArtifactFileText(
      fixtureText(),
      "decode-plan.json",
    );

    expect(artifact.kind).toBe("inference-sim/frozen-plan");
    expect(artifact.scenario.id).toBe("multi-gpu-ring-4");
    expect(artifact.plan.steps.length).toBeGreaterThan(0);
  });

  it("executes and independently replays the artifact in the Worker boundary", () => {
    const result = executeFrozenPlanWorkerRun(
      fixtureText(8),
      "decode-plan.json",
    );

    expect(result.execution.status).toBe("succeeded");
    expect(result.replay).toMatchObject({
      status: "succeeded",
      completedAtNs: result.execution.completedAtNs,
      appliedEvents: result.execution.operationCount + 1,
      exact: true,
    });
    expect(result.plan.operationCounts.collective).toBe(8);
    expect(result.execution.operationPreview.length).toBeLessThanOrEqual(100);
  });

  it("rejects wrong extensions and tampering before execution", () => {
    expect(() => parseFrozenPlanArtifactFileText(
      fixtureText(),
      "decode-plan.yaml",
    )).toThrow("must use .json");

    const tampered = JSON.parse(fixtureText()) as {
      plan: { steps: Array<{ operation: { durationNs: number } }> };
    };
    tampered.plan.steps[0].operation.durationNs++;
    expect(() => executeFrozenPlanWorkerRun(
      JSON.stringify(tampered),
      "decode-plan.json",
    )).toThrow("plan fingerprint mismatch");
  });
});
