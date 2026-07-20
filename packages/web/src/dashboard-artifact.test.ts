import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import {
  buildScenarioPreset,
  createSimulationResultArtifact,
  serializeSimulationResultArtifact,
} from "@inference-sim/core";
import {
  compareDashboardArtifact,
  createDashboardArtifact,
  dashboardArtifactContracts,
  parseDashboardArtifactFileText,
} from "./dashboard-artifact.js";
import { createBuiltinModelBinding } from "./model-binding.js";
import { parseCalibrationFileText } from "./calibration-import.js";
import { simulateDashboardExecution } from "./dashboard-simulation.js";
import { executeDashboardWorkerRun } from "./dashboard-worker-run.js";
import { parseTokenTraceFileText } from "./token-trace-import.js";
import type {
  DashboardRunConfig,
  WorkerRunProgress,
} from "./types.js";

const config: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  multiGpuRanks: 2,
  multiNodeCount: 2,
  mode: "speculative",
  seed: 42,
  speculative: {
    family: "mtp",
    outputTokens: 8,
    draftWidth: 2,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    compareTopologies: false,
    useExpertCache: false,
    decodeMode: "mtp",
    draftWidth: 2,
    firstPositionAcceptance: 0.82,
    requestCount: 2,
    arrivalGapUs: 100,
    promptTokens: 32,
    outputTokens: 4,
    maxBatchSize: 2,
    maxBatchTokens: 16,
    prefillChunkTokens: 16,
  },
  expertCache: {
    placementStrategy: "contiguous",
    tokenCount: 8,
    topK: 2,
    expertCount: 8,
    hotSlots: 4,
    warmSlots: 4,
    adaptivePrefetch: true,
  },
};

describe("dashboard result artifact import", () => {
  const pipelineConfig: DashboardRunConfig = {
    ...config,
    mode: "pipeline",
    modelBinding: {
      ...createBuiltinModelBinding("llama-3-8b"),
      source: "local_model_package",
      displayName: "audio codec pipeline",
      modelFingerprints: ["fnv1a32:12345678", "fnv1a32:90abcdef"],
      targetModelFingerprint: "fnv1a32:90abcdef",
      componentCount: 2,
      totalParameters: 2_000,
      weightBytes: 4_000,
      pipelineStrategy: "composite",
      pipelineExecution: {
        strategyKind: "composite",
        replacesTarget: true,
        components: [
          {
            id: "encoder",
            role: "audio_encoder",
            phase: "prompt_only",
            strategyKind: "single_pass",
            invocationMultiplier: 1,
            weightBytes: 2_000,
            isPrimary: false,
            order: 0,
          },
          {
            id: "vocoder",
            role: "vocoder",
            phase: "prompt_only",
            strategyKind: "single_pass",
            invocationMultiplier: 1,
            weightBytes: 2_000,
            isPrimary: true,
            order: 1,
          },
        ],
        edges: [{
          fromComponent: "encoder",
          toComponent: "vocoder",
          deviceTransfer: false,
        }],
      },
      executionCoverage: {
        fidelity: "complete",
        scope: "full_model",
        modeledComponentIds: ["encoder", "vocoder"],
        unmodeledComponentIds: [],
        limitations: [],
      },
    },
  };

  it("binds only contracts used by the selected execution path", () => {
    expect(Object.keys(dashboardArtifactContracts(config)).sort()).toEqual([
      "frozen_plan",
      "paged_kv",
      "scenario_schema",
      "speculative_family",
      "speculative_iteration",
      "topology_cost_model",
    ]);
    expect(Object.keys(dashboardArtifactContracts({
      ...config,
      mode: "serving",
      serving: {
        ...config.serving,
        decodeMode: "target_only",
        useExpertCache: false,
      },
    })).sort()).toEqual([
      "frozen_plan",
      "scenario_schema",
      "serving_trace",
      "topology_cost_model",
    ]);
    expect(Object.keys(dashboardArtifactContracts({
      ...config,
      mode: "serving",
      serving: {
        ...config.serving,
        useExpertCache: true,
      },
    })).sort()).toEqual([
      "concurrent_plan_trace",
      "expert_cache",
      "frozen_plan",
      "scenario_schema",
      "serving_expert_cache",
      "serving_trace",
      "speculative_family",
      "speculative_iteration",
      "topology_cost_model",
    ]);
  });

  it("validates and replays a current deterministic artifact", () => {
    const artifact = createDashboardArtifact(
      config,
      simulateDashboardExecution(config),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact, true),
      "run.json",
    );
    const rerun = createDashboardArtifact(
      parsed.config,
      simulateDashboardExecution(parsed.config),
    );
    const replay = compareDashboardArtifact(rerun, parsed.expectation);

    expect(parsed.config).toEqual(config);
    expect(replay).toMatchObject({
      sourceFileName: "run.json",
      inputMatches: true,
      outputMatches: true,
      matches: true,
    });
    expect(replay.actualArtifactFingerprint).toBe(
      artifact.artifactFingerprint,
    );
  });

  it("round-trips model dtype and quantization evidence", () => {
    const quantizedConfig: DashboardRunConfig = {
      ...config,
      mode: "serving",
      modelBinding: createBuiltinModelBinding("llama-3-8b", "int4"),
      serving: {
        ...config.serving,
        decodeMode: "target_only",
      },
    };
    const artifact = createDashboardArtifact(
      quantizedConfig,
      simulateDashboardExecution(quantizedConfig),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "int4-run.json",
    );

    expect(parsed.config.modelBinding?.modelFormat).toEqual(
      quantizedConfig.modelBinding?.modelFormat,
    );
    expect(parsed.config.modelBinding?.weightBytes).toBe(
      quantizedConfig.modelBinding?.weightBytes,
    );
  });

  it("strictly round-trips and replays a selected multi-node count", () => {
    const multiNodeConfig: DashboardRunConfig = {
      ...config,
      scenarioName: "multi-node",
      multiNodeCount: 4,
    };
    const artifact = createDashboardArtifact(
      multiNodeConfig,
      simulateDashboardExecution(multiNodeConfig),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "multi-node.json",
    );

    expect(parsed.config).toEqual(multiNodeConfig);
    expect(executeDashboardWorkerRun(
      parsed.config,
      parsed.expectation,
    ).artifactReplay?.matches).toBe(true);
  });

  it("binds and revalidates a complete custom scenario", () => {
    const customScenario = {
      ...buildScenarioPreset("single-gpu-cpu"),
      id: "custom-single-gpu",
      family: "custom" as const,
    };
    const customConfig: DashboardRunConfig = {
      ...config,
      scenarioName: "custom",
      customScenario,
    };
    const artifact = createDashboardArtifact(
      customConfig,
      simulateDashboardExecution(customConfig),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "custom.json",
    );

    expect(parsed.config.customScenario).toEqual(customScenario);
    expect(parsed.config.scenarioName).toBe("custom");
    expect(executeDashboardWorkerRun(
      parsed.config,
      parsed.expectation,
    ).artifactReplay?.matches).toBe(true);
  });

  it("round-trips model capability evidence in deterministic input", () => {
    const boundConfig: DashboardRunConfig = {
      ...config,
      modelBinding: {
        ...createBuiltinModelBinding("llama-3-8b"),
        source: "local_model_package",
        modelFingerprints: [
          "fnv1a32:12345678",
          "fnv1a32:90abcdef",
        ],
        targetModelFingerprint: "fnv1a32:12345678",
        componentCount: 2,
        pipelineStrategy: "composite",
        speculativeFamilies: ["draft_model", "mtp"],
      },
    };
    const artifact = createDashboardArtifact(
      boundConfig,
      simulateDashboardExecution(boundConfig),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "model-bound.json",
    );

    expect(parsed.config.modelBinding).toEqual(boundConfig.modelBinding);
    expect(executeDashboardWorkerRun(
      parsed.config,
      parsed.expectation,
    ).artifactReplay?.matches).toBe(true);
  });

  it("replays all dashboard modes through the Worker execution boundary", async () => {
    const configs: DashboardRunConfig[] = [
      config,
      pipelineConfig,
      { ...config, mode: "expert-cache" },
      { ...config, mode: "serving" },
      {
        ...config,
        mode: "serving",
        serving: { ...config.serving, compareTopologies: true },
      },
    ];
    for (const replayConfig of configs) {
      const artifact = createDashboardArtifact(
        replayConfig,
        simulateDashboardExecution(replayConfig),
      );
      const parsed = parseDashboardArtifactFileText(
        serializeSimulationResultArtifact(artifact),
        `${replayConfig.mode}.json`,
      );
      const result = executeDashboardWorkerRun(
        parsed.config,
        parsed.expectation,
      );
      const exported = JSON.parse(await result.artifact.blob.text()) as {
        artifactFingerprint: string;
      };

      expect(result.artifactReplay?.matches).toBe(true);
      expect(result.artifact.artifactFingerprint).toBe(
        parsed.expectation.artifactFingerprint,
      );
      expect(exported.artifactFingerprint).toBe(
        parsed.expectation.artifactFingerprint,
      );
    }
  });

  it("round-trips component-tagged pipeline execution", () => {
    const output = simulateDashboardExecution(pipelineConfig);
    const artifact = createDashboardArtifact(
      pipelineConfig,
      output,
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "pipeline.json",
    );
    const replay = executeDashboardWorkerRun(
      parsed.config,
      parsed.expectation,
    );
    expect(parsed.config.modelBinding?.pipelineExecution).toEqual(
      pipelineConfig.modelBinding?.pipelineExecution,
    );
    expect(replay.artifactReplay?.matches).toBe(true);
    const operations = output.evidence.kind === "pipeline"
      ? output.evidence.topology.plan.steps.flatMap((step) => (
          step.operation.kind === "compute"
            ? [step.operation.componentId]
            : []
        ))
      : [];
    expect(operations).toEqual([
      "encoder",
      "vocoder",
      "encoder",
      "vocoder",
    ]);
  });

  it("reports execution-bound progress without changing the artifact", async () => {
    const progress: WorkerRunProgress[] = [];
    const baseline = executeDashboardWorkerRun(config);
    const observed = executeDashboardWorkerRun(
      config,
      undefined,
      (update) => progress.push(update),
    );

    assertMonotonicProgress(progress);
    expect(observed.summary).toEqual(baseline.summary);
    expect(observed.artifact.artifactFingerprint).toBe(
      baseline.artifact.artifactFingerprint,
    );
    expect(await observed.artifact.blob.text()).toBe(
      await baseline.artifact.blob.text(),
    );
  });

  it("restores embedded calibration and token evidence before replay", async () => {
    const calibrationText = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const traceText = await readFile(new URL(
      "../../../examples/speculative-token-trace-mtp.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      calibrationText,
      "calibration.yaml",
    );
    const tokenTrace = await parseTokenTraceFileText(
      traceText,
      "trace.yaml",
    );
    const embeddedConfig: DashboardRunConfig = {
      ...config,
      calibration: calibration.dataset,
      speculative: {
        ...config.speculative,
        family: tokenTrace.trace.family,
        outputTokens: tokenTrace.trace.expectedOutputTokenIds.length,
        draftWidth: tokenTrace.trace.maxAdditionalTokens,
        trace: tokenTrace.trace,
      },
    };
    const artifact = createDashboardArtifact(
      embeddedConfig,
      simulateDashboardExecution(embeddedConfig),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "evidence.json",
    );

    expect(parsed.calibration?.fit.datasetFingerprint).toBe(
      calibration.fit.datasetFingerprint,
    );
    expect(parsed.tokenTrace?.preview.committedOutputTokenIds).toEqual(
      tokenTrace.preview.committedOutputTokenIds,
    );
    expect(executeDashboardWorkerRun(
      parsed.config,
      parsed.expectation,
    ).artifactReplay?.matches).toBe(true);
    expect(Object.keys(dashboardArtifactContracts(parsed.config))).toContain(
      "calibration_dataset",
    );
    expect(Object.keys(dashboardArtifactContracts(parsed.config))).toContain(
      "speculative_token_trace",
    );
  });

  it("reports a deterministic output mismatch after implementation drift", () => {
    const artifact = createDashboardArtifact(
      config,
      simulateDashboardExecution(config),
    );
    const parsed = parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(artifact),
      "run.json",
    );
    const changed = {
      ...artifact,
      output: {
        ...artifact.output,
        summary: {
          ...artifact.output.summary,
          topology: {
            ...artifact.output.summary.topology,
            planSteps: artifact.output.summary.topology.planSteps + 1,
          },
        },
      },
    };
    const rerun = createDashboardArtifact(
      config,
      changed.output,
    );
    const replay = compareDashboardArtifact(rerun, parsed.expectation);

    expect(replay.inputMatches).toBe(true);
    expect(replay.outputMatches).toBe(false);
    expect(replay.matches).toBe(false);

    const workerMismatch = executeDashboardWorkerRun(
      { ...config, seed: config.seed + 1 },
      parsed.expectation,
    );
    expect(workerMismatch.artifactReplay).toMatchObject({
      inputMatches: false,
      matches: false,
    });
  });

  it("rejects tampering, stale contracts, and malformed dashboard input", () => {
    const artifact = createDashboardArtifact(
      config,
      simulateDashboardExecution(config),
    );
    const tampered = JSON.parse(
      serializeSimulationResultArtifact(artifact),
    ) as Record<string, unknown>;
    (tampered.output as {
      summary: { topology: { planSteps: number } };
    }).summary.topology.planSteps++;
    expect(() => parseDashboardArtifactFileText(
      JSON.stringify(tampered),
      "tampered.json",
    )).toThrow("output fingerprint mismatch");

    const stale = createSimulationResultArtifact(
      artifact.runKind,
      {
        ...dashboardArtifactContracts(config),
        topology_cost_model:
          dashboardArtifactContracts(config).topology_cost_model - 1,
      },
      artifact.input,
      artifact.output,
    );
    expect(() => parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(stale),
      "stale.json",
    )).toThrow("topology_cost_model requires revision");

    const malformed = createSimulationResultArtifact(
      artifact.runKind,
      dashboardArtifactContracts(config),
      { ...artifact.input, multiGpuRanks: 3 },
      artifact.output,
    );
    expect(() => parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(malformed),
      "malformed.json",
    )).toThrow("multiGpuRanks is unsupported");

    const invalidNodeCount = createSimulationResultArtifact(
      artifact.runKind,
      dashboardArtifactContracts(config),
      { ...artifact.input, multiNodeCount: 5 },
      artifact.output,
    );
    expect(() => parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(invalidNodeCount),
      "invalid-node-count.json",
    )).toThrow("multiNodeCount is unsupported");

    const advancedNicField = createSimulationResultArtifact(
      artifact.runKind,
      dashboardArtifactContracts(config),
      { ...artifact.input, nicBandwidthBytesPerSec: 25_000_000_000 },
      artifact.output,
    );
    expect(() => parseDashboardArtifactFileText(
      serializeSimulationResultArtifact(advancedNicField),
      "advanced-nic.json",
    )).toThrow("unknown keys nicBandwidthBytesPerSec");
  });

  it("rejects a mismatched run kind and non-JSON extension", () => {
    const artifact = createSimulationResultArtifact(
      "dashboard/serving",
      dashboardArtifactContracts(config),
      config,
      simulateDashboardExecution(config),
    );
    const text = serializeSimulationResultArtifact(artifact);

    expect(() => parseDashboardArtifactFileText(text, "run.json"))
      .toThrow("run kind must be dashboard/speculative");
    expect(() => parseDashboardArtifactFileText(text, "run.yaml"))
      .toThrow("must use .json");
  });
});

function assertMonotonicProgress(
  progress: readonly WorkerRunProgress[],
): void {
  expect(progress.length).toBeGreaterThan(0);
  expect(progress.at(-1)).toEqual({
    progress: 94,
    phase: "Finalizing replay evidence",
  });
  expect(progress.every((entry, index) => (
    entry.phase.length > 0
    && entry.progress >= 0
    && entry.progress <= 99
    && (index === 0 || entry.progress > progress[index - 1]!.progress)
  ))).toBe(true);
}
