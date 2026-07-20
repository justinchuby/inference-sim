import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import {
  buildScenarioPreset,
  parseSimulationResultArtifact,
  serializeSimulationResultArtifact,
} from "@inference-sim/core";
import { parseCalibrationFileText } from "./calibration-import.js";
import {
  createDashboardArtifact,
  dashboardArtifactFileName,
} from "./dashboard-artifact.js";
import { parseTokenTraceFileText } from "./token-trace-import.js";
import { cachePartitionRows } from "./ResultCharts.js";
import {
  simulateDashboard,
  simulateDashboardExecution,
} from "./dashboard-simulation.js";
import type { DashboardRunConfig } from "./types.js";

const base: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  multiGpuRanks: 2,
  mode: "speculative",
  seed: 42,
  speculative: {
    family: "mtp",
    outputTokens: 64,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    compareTopologies: false,
    useExpertCache: false,
    decodeMode: "mtp",
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
    requestCount: 8,
    arrivalGapUs: 100,
    promptTokens: 128,
    outputTokens: 16,
    maxBatchSize: 4,
    maxBatchTokens: 64,
    prefillChunkTokens: 32,
  },
  expertCache: {
    placementStrategy: "contiguous",
    tokenCount: 32,
    topK: 2,
    expertCount: 12,
    hotSlots: 4,
    warmSlots: 6,
    adaptivePrefetch: true,
  },
};

describe("simulateDashboard", () => {
  it("enforces imported model speculative capabilities at execution", () => {
    const config: DashboardRunConfig = {
      ...base,
      modelBinding: {
        source: "local_model_package",
        modelFingerprints: ["fnv1a32:12345678"],
        componentCount: 1,
        speculativeFamilies: ["mtp"],
      },
      speculative: {
        ...base.speculative,
        family: "draft_model",
      },
    };
    expect(() => simulateDashboard(config)).toThrow(
      "does not declare speculative family draft_model",
    );
  });

  it("runs bounded speculative simulation with paged KV metrics", () => {
    const result = simulateDashboard(base);

    expect(result.scenario.deviceCount).toBe(3);
    expect(result.speculative?.finalTokenLength).toBe(2112);
    expect(result.speculative?.family).toBe("mtp");
    expect(result.speculative?.metrics.kvPagesAllocated).toBeGreaterThan(0);
    expect(result.topology.confidence).toBe("heuristic");
    expect(result.topology.metrics.totalDurationNs).toBeGreaterThan(0);
    expect(result.topology.operationCounts.collective).toBeGreaterThan(0);
  });

  it("exports deterministic full protocol evidence without wall-clock timing", () => {
    const first = createDashboardArtifact(
      base,
      simulateDashboardExecution(base),
    );
    const second = createDashboardArtifact(
      base,
      simulateDashboardExecution(base),
    );
    const serialized = serializeSimulationResultArtifact(first, true);
    const parsed = parseSimulationResultArtifact(JSON.parse(serialized));

    expect(first).toEqual(second);
    expect(first.output.evidence.kind).toBe("speculative");
    if (first.output.evidence.kind !== "speculative") {
      throw new Error("expected speculative artifact evidence");
    }
    expect(first.output.evidence.workload.iterations.length).toBeGreaterThan(0);
    expect(first.output.evidence.topology.execution.trace.operations.length)
      .toBeGreaterThan(0);
    expect(serialized).not.toContain("\"durationMs\"");
    expect(parsed.artifactFingerprint).toBe(first.artifactFingerprint);
    expect(dashboardArtifactFileName(first)).toMatch(
      /^inference-sim-multi-gpu-speculative-[0-9a-f]{8}\.json$/,
    );
  });

  it("exports replay evidence for expert, serving, and comparison modes", () => {
    const compactServing = {
      ...base.serving,
      requestCount: 2,
      promptTokens: 32,
      outputTokens: 4,
      maxBatchSize: 2,
      maxBatchTokens: 16,
      prefillChunkTokens: 16,
    };
    const configs: DashboardRunConfig[] = [
      { ...base, mode: "expert-cache" },
      {
        ...base,
        mode: "serving",
        serving: { ...compactServing, useExpertCache: true },
      },
      {
        ...base,
        mode: "serving",
        serving: {
          ...compactServing,
          compareTopologies: true,
          useExpertCache: false,
        },
      },
    ];

    const artifacts = configs.map((config) => createDashboardArtifact(
      config,
      simulateDashboardExecution(config),
    ));
    expect(artifacts.map((artifact) => artifact.output.evidence.kind)).toEqual([
      "expert_cache",
      "serving",
      "serving_comparison",
    ]);
    for (const artifact of artifacts) {
      expect(() => parseSimulationResultArtifact(JSON.parse(
        serializeSimulationResultArtifact(artifact),
      ))).not.toThrow();
    }
    const serving = artifacts[1].output.evidence;
    if (serving.kind !== "serving") {
      throw new Error("expected serving artifact evidence");
    }
    expect(serving.serving.serving.trace.length).toBeGreaterThan(0);
    expect(serving.serving.physical?.execution.trace.operations.length)
      .toBeGreaterThan(0);
    const comparison = artifacts[2].output.evidence;
    if (comparison.kind !== "serving_comparison") {
      throw new Error("expected serving-comparison artifact evidence");
    }
    expect(comparison.comparison.runs).toHaveLength(6);
    expect(comparison.comparison.runs.every(
      (run) => run.result.serving.trace.length > 0,
    )).toBe(true);
  });

  it("runs every selectable proposer family through the shared core contract", () => {
    const families = [
      "prompt_lookup",
      "draft_model",
      "mtp",
      "eagle3",
      "shared_kv",
      "self_speculative",
    ] as const;
    for (const family of families) {
      const result = simulateDashboard({
        ...base,
        speculative: { ...base.speculative, family },
      });
      expect(result.speculative?.family).toBe(family);
      expect(result.speculative?.finalTokenLength).toBe(2112);
    }
  });

  it("runs the selected workload on parameterized multi-GPU rings", () => {
    for (const multiGpuRanks of [4, 8] as const) {
      const result = simulateDashboard({ ...base, multiGpuRanks });

      expect(result.scenario.id).toBe(
        `multi-gpu-ring-${multiGpuRanks}`,
      );
      expect(result.scenario.deviceCount).toBe(multiGpuRanks + 1);
      expect(result.topology.operationCounts.allReduce).toBeGreaterThan(0);
      expect(result.topology.metrics.linkUtilization.some((resource) => (
        resource.resourceId.includes("nvlink")
      ))).toBe(true);
      expect(result.topology.metrics.committedTokens).toBe(
        base.speculative.outputTokens,
      );
    }
  });

  it("runs a strictly embedded custom device scenario", () => {
    const customScenario = {
      ...buildScenarioPreset("gpu-npu"),
      id: "custom-gpu-npu",
      family: "custom" as const,
    };
    const result = simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario,
    });

    expect(result.scenario).toMatchObject({
      id: "custom-gpu-npu",
      family: "custom",
      deviceCount: customScenario.devices.length,
      linkCount: customScenario.links.length,
    });
    expect(result.topology.metrics.committedTokens).toBe(
      base.speculative.outputTokens,
    );
  });

  it("rejects missing, irrelevant, and malformed custom scenarios", () => {
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
    })).toThrow("dashboard custom scenario is missing");
    expect(() => simulateDashboard({
      ...base,
      customScenario: buildScenarioPreset("cpu-only"),
    })).toThrow("must only be set when scenarioName is custom");
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario: {
        ...buildScenarioPreset("cpu-only"),
        family: "invented" as "custom",
      },
    })).toThrow("family: must be one of");
  });

  it("rejects an untrusted dashboard GPU-rank value", () => {
    expect(() => simulateDashboard({
      ...base,
      multiGpuRanks: 3 as DashboardRunConfig["multiGpuRanks"],
    })).toThrow("dashboard multi-GPU ranks must be 2, 4, or 8; got 3");
  });

  it("partitions routed experts across selected GPU ranks", () => {
    const result = simulateDashboard({
      ...base,
      mode: "expert-cache",
      multiGpuRanks: 4,
    });

    expect(result.scenario.id).toBe("multi-gpu-ring-4");
    expect(result.expertCache?.hotPartitions).toHaveLength(4);
    expect(result.topology.operationCounts.allToAll).toBeGreaterThan(0);
    expect(result.topology.assumptions.some((assumption) => (
      assumption.includes("exact round-robin token-source to expert-owner")
    ))).toBe(true);
  });

  it("runs imported token evidence through value and state parity", async () => {
    const text = await readFile(new URL(
      "../../../examples/speculative-token-trace-mtp.yaml",
      import.meta.url,
    ), "utf8");
    const imported = await parseTokenTraceFileText(text, "trace.yaml");
    const result = simulateDashboard({
      ...base,
      speculative: {
        ...base.speculative,
        trace: imported.trace,
      },
    });

    expect(result.speculative?.tokenTrace).toMatchObject({
      traceId: "mtp-correction-bonus-tail",
      source: "synthetic-example",
      runtimeRevision: "onnx-genai-synthetic",
      matchesTargetOnly: true,
      comparedTokenCount: 8,
    });
    expect(result.speculative?.iterations.map((iteration) => iteration.outcome))
      .toEqual(["correction", "bonus", "accepted_tail"]);
    expect(result.topology.metrics.committedTokens).toBe(8);
  });

  it("preserves a well-formed token mismatch as diagnostic output", async () => {
    const text = await readFile(new URL(
      "../../../examples/speculative-token-trace-mtp.yaml",
      import.meta.url,
    ), "utf8");
    const imported = await parseTokenTraceFileText(text, "trace.yaml");
    const result = simulateDashboard({
      ...base,
      speculative: {
        ...base.speculative,
        trace: {
          ...imported.trace,
          expectedOutputTokenIds: [
            10, 999, 21, 30, 31, 32, 40, 41,
          ],
        },
      },
    });

    expect(result.speculative?.tokenTrace).toMatchObject({
      matchesTargetOnly: false,
      firstMismatch: {
        outputIndex: 1,
        expectedTokenId: 999,
        actualTokenId: 20,
      },
    });
    expect(result.topology.metrics.committedTokens).toBe(8);
  });

  it("runs deterministic expert cache simulation", () => {
    const config = { ...base, mode: "expert-cache" as const };
    const first = simulateDashboard(config);
    const second = simulateDashboard(config);

    expect(first).toEqual(second);
    expect(first.expertCache?.routes).toHaveLength(32);
    expect(first.expertCache?.metrics.hotHitRate).toBeGreaterThanOrEqual(0);
    expect(first.expertCache?.metrics.adaptivePrefetchDecisions)
      .toBeGreaterThan(0);
    expect(first.topology.operationCounts.transfer).toBeGreaterThan(0);
    expect(first.topology.assumptions.some(
      (assumption) => assumption.includes("explicit contiguous owner mapping"),
    )).toBe(true);
    expect(first.expertCache?.hotPartitions).toHaveLength(2);
    expect(first.expertCache?.hotPartitions.every((partition) => (
      partition.capacityBytes === 4 * 64 * 1024 ** 2
      && partition.residentBytes + partition.reservedBytes
        <= partition.capacityBytes
    ))).toBe(true);
    expect(first.expertCache?.warmPartitions).toHaveLength(1);
    const partitionRows = cachePartitionRows(first.expertCache);
    expect(partitionRows.map((row) => row.name)).toEqual([
      "H owner 0",
      "H owner 1",
      "W node 0",
    ]);
    expect(partitionRows.every((row) => (
      Math.abs(row.resident + row.reserved + row.free - 100) < 1e-9
    ))).toBe(true);

    const roundRobin = simulateDashboard({
      ...config,
      expertCache: {
        ...config.expertCache,
        placementStrategy: "round_robin",
      },
    });
    expect(roundRobin.topology.assumptions.some(
      (assumption) => assumption.includes("explicit round_robin owner mapping"),
    )).toBe(true);
    expect(roundRobin.expertCache?.hotPartitions.map(
      (partition) => partition.id,
    )).toEqual(["target-shard-0", "target-shard-1"]);
  });

  it("runs continuous serving with replayed request timing", () => {
    const config = {
      ...base,
      mode: "serving" as const,
      serving: { ...base.serving, useExpertCache: true },
    };
    const result = simulateDashboard(config);

    expect(result.serving?.requests).toHaveLength(8);
    expect(result.serving?.metrics.outputTokens).toBe(128);
    expect(result.serving?.metrics.p95TimeToFirstTokenNs).toBeGreaterThan(0);
    expect(result.serving?.metrics.kvHighWaterTokens).toBeGreaterThan(0);
    expect(result.serving?.decodeMode).toBe("mtp");
    expect(result.serving?.metrics.proposedDraftTokens).toBeGreaterThan(0);
    expect(result.serving?.metrics.acceptedDraftTokens).toBeGreaterThan(0);
    expect(result.serving?.batches.length).toBeGreaterThan(1);
    expect(result.expertCache?.routes.length).toBeGreaterThan(0);
    expect(result.expertCache?.metrics.routes).toBe(
      result.expertCache?.routes.length,
    );
    expect(result.expertCache?.hotPartitions).toHaveLength(2);
    expect(result.serving?.physicalReplayEvents).toBeGreaterThan(0);
    expect(result.serving?.maximumConcurrentPlans).toBeGreaterThan(0);
    expect(result.serving?.physicalDrainNs).toBe(
      result.topology.metrics.backgroundDrainNs,
    );
    expect(result.topology.topResources.every((resource) => (
      resource.utilization >= 0 && resource.utilization <= 1
    ))).toBe(true);
    expect(result.topology.operationCounts.allToAll).toBeGreaterThan(0);
    expect(result.topology.planSteps).toBeGreaterThan(0);
  });

  it("preserves an explicit target-only serving baseline", () => {
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        useExpertCache: false,
      },
    });

    expect(result.serving?.support).toBe("target_only");
    expect(result.serving?.metrics.proposedDraftTokens).toBe(0);
    expect(result.serving?.metrics.acceptedDraftTokens).toBe(0);
    expect(result.serving?.metrics.committedTokensPerTargetForward).toBe(1);
  });

  it("compares the same serving workload across all six topologies", () => {
    const config: DashboardRunConfig = {
      ...base,
      mode: "serving",
      serving: { ...base.serving, compareTopologies: true },
    };
    const first = simulateDashboard(config);
    const second = simulateDashboard(config);

    expect(first).toEqual(second);
    expect(first.comparison).toHaveLength(6);
    expect(first.comparison?.map((entry) => entry.rank))
      .toEqual([1, 2, 3, 4, 5, 6]);
    expect(first.comparison?.[0]).toMatchObject({
      scenarioId: "multi-gpu",
      relativeToFastest: 1,
    });
    expect(first.comparison?.at(-1)?.scenarioId).toBe("cpu-only");
    expect(first.scenario.id).toBe("multi-gpu");
    expect(first.comparison?.every((entry, index, entries) => (
      index === 0
      || entry.totalDurationNs >= entries[index - 1].totalDurationNs
    ))).toBe(true);
  });

  it("changes modeled latency when the same workload changes topology", () => {
    const multiGpu = simulateDashboard(base);
    const cpu = simulateDashboard({ ...base, scenarioName: "cpu-only" });

    expect(cpu.topology.metrics.totalDurationNs).toBeGreaterThan(
      multiGpu.topology.metrics.totalDurationNs,
    );
    expect(cpu.topology.metrics.tokensPerSecond).toBeLessThan(
      multiGpu.topology.metrics.tokensPerSecond,
    );
  });

  it("runs an imported calibration through the dashboard worker contract", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );
    const result = simulateDashboard({
      ...base,
      calibration: calibration.dataset,
    });

    expect(result.calibration).toMatchObject({
      datasetId: "synthetic-linear-example",
      datasetFingerprint: calibration.fit.datasetFingerprint,
      evidenceKind: "synthetic",
      fitConfidence: "heuristic",
    });
    expect(result.calibration?.diagnostics).toHaveLength(15);
    expect(result.calibration?.transportDiagnostics).toHaveLength(20);
    expect(result.topology.assumptions[0]).toContain(
      calibration.fit.datasetFingerprint,
    );
  });

  it("runs imported transport curves across all six serving topologies", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      calibration: calibration.dataset,
      serving: {
        ...base.serving,
        compareTopologies: true,
      },
    });

    expect(result.comparison).toHaveLength(6);
    expect(result.topology.assumptions).toContain(
      "transport timing uses exact-path calibration curves without extrapolation",
    );
  });

  it("rejects a revision-3 AllToAllV observation without a traffic signature", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );
    expect(() => simulateDashboard({
      ...base,
      mode: "expert-cache",
      calibration: {
        ...calibration.dataset,
        transportObservations:
          calibration.dataset.transportObservations.map((observation) => (
            observation.algorithm === "all_to_all_v"
              ? { ...observation, trafficSignature: undefined }
              : observation
          )),
      },
    })).toThrow(
      "requires an AllToAllV traffic signature",
    );
  });

  it("rejects adaptive prefetch when storage calibration is missing", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );

    expect(() => simulateDashboard({
      ...base,
      scenarioName: "single-gpu-cpu",
      mode: "expert-cache",
      calibration: {
        ...calibration.dataset,
        transportObservations:
          calibration.dataset.transportObservations?.filter(
            (observation) => !observation.linkIds.some(
              (linkId) => linkId.endsWith(":storage-read"),
            ),
          ),
      },
    })).toThrow("no calibrated transport curve");
  });

  it("rejects routed experts when AllToAllV calibration is missing", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );

    expect(() => simulateDashboard({
      ...base,
      mode: "expert-cache",
      calibration: {
        ...calibration.dataset,
        transportObservations:
          calibration.dataset.transportObservations?.filter(
            (observation) => observation.algorithm !== "all_to_all_v",
          ),
      },
    })).toThrow(
      "no calibrated transport curve",
    );
  });

  it("rejects dashboard work outside imported interpolation ranges", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );

    expect(() => simulateDashboard({
      ...base,
      mode: "serving",
      calibration: calibration.dataset,
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        promptTokens: 512,
        maxBatchTokens: 512,
        prefillChunkTokens: 512,
      },
    })).toThrow("outside calibrated range 1..128");
  });
});
