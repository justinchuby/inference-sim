import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import {
  COMPUTER_PRESET_NAMES,
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
import { createBuiltinModelBinding } from "./model-binding.js";
import { interpretRoofline } from "./roofline-interpretation.js";
import type { DashboardModelPreset } from "./model-binding.js";
import type { DashboardRunConfig } from "./types.js";

const base: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  multiGpuRanks: 2,
  multiNodeCount: 2,
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
  fault: {
    failedNodeId: "",
    faultAtUs: 50,
    quiesceTimeoutUs: 250,
    executionCount: 4,
  },
  modality: "text",
  mediaItemsPerRequest: 1,
    coResidency: {
      models: [
        { preset: "qwen3-8b", weightDtype: "int4" as const, contextTokens: 8192, pinned: false, requestCount: 2 },
        { preset: "qwen3-0.6b", weightDtype: "fp16" as const, contextTokens: 1024, pinned: false, requestCount: 2 },
      ],
      requestGapMs: 4000,
      promptTokens: 256,
      outputTokens: 16,
    },
};

describe("simulateDashboard", () => {
  it("emits hierarchical roofline evidence for serving phases", () => {
    const result = simulateDashboard({
      ...base,
      scenarioName: "rtx-4090-desktop",
      modelBinding: createBuiltinModelBinding("llama-3-8b", "fp16"),
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 2,
        promptTokens: 32,
        outputTokens: 3,
        maxBatchSize: 2,
        maxBatchTokens: 16,
        prefillChunkTokens: 16,
      },
    });

    expect(result.roofline).toMatchObject({
      revision: 2,
      status: "available",
      computeRoof: { dtype: "fp16" },
    });
    expect(result.roofline!.bandwidthRoofs.some(
      (roof) => roof.kind === "device_memory",
    )).toBe(true);
    expect(new Set(result.roofline!.points.map((point) => point.phase)))
      .toContain("prefill");
    expect(result.roofline!.points.every((point) => (
      point.arithmeticIntensity > 0
      && point.predictedFlopsPerSecond > 0
      && point.notes.some((note) => note.includes("simulated replay"))
    ))).toBe(true);
  });

  it("uses the activation roof instead of inventing a low-bit weight roof", () => {
    const result = simulateDashboard({
      ...base,
      scenarioName: "rtx-5090-desktop",
      modelBinding: createBuiltinModelBinding("llama-3-8b", "int2"),
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 1,
        promptTokens: 8,
        outputTokens: 2,
        maxBatchSize: 1,
        maxBatchTokens: 8,
        prefillChunkTokens: 8,
      },
    });

    expect(result.roofline?.status).toBe("available");
    expect(result.roofline?.computeRoof).toMatchObject({
      dtype: "fp16",
      evidence: "vendor_peak",
      flopsPerSecond: 209.5e12,
    });
    expect(result.roofline?.bandwidthRoofs.length).toBeGreaterThan(0);
    expect(result.roofline?.assumptions.some((assumption) => (
      assumption.includes("Weight-only quantization")
    ))).toBe(true);
  });

  it("uses user-declared dense compute peaks without labeling them official", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const result = simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario: {
        ...scenario,
        devices: scenario.devices.map((device) => ({
          ...device,
          customComputePeaks: [{
            dtype: "fp16",
            operationsPerSecond: 10e12,
          }],
        })),
      },
      modelBinding: createBuiltinModelBinding("llama-3-8b", "fp16"),
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
      },
    });

    expect(result.roofline?.computeRoof).toMatchObject({
      label: "User-declared dense compute",
      dtype: "fp16",
      evidence: "user_declared",
    });
    expect(result.roofline?.assumptions.some((assumption) => (
      assumption.includes("not vendor-verified")
    ))).toBe(true);
  });

  it("labels speculative verification as its own roofline phase", () => {
    const result = simulateDashboard({
      ...base,
      modelBinding: createBuiltinModelBinding("llama-3-8b", "int4"),
      speculative: {
        ...base.speculative,
        family: "prompt_lookup",
        outputTokens: 8,
        draftWidth: 3,
      },
    });
    expect(result.roofline?.points.map((point) => point.phase))
      .toContain("spec_verify");
    expect(result.roofline?.assumptions.some(
      (assumption) => assumption.includes("proposer work is excluded"),
    )).toBe(true);
  });

  it("fails closed when roofline model evidence is absent", () => {
    const result = simulateDashboard({ ...base, mode: "expert-cache" });
    expect(result.roofline).toMatchObject({
      revision: 2,
      status: "unavailable",
    });
    expect(result.roofline?.unavailableReason).toContain("select a model");
  });

  it("runs the default bound model on every computer preset", () => {
    const modelBinding = createBuiltinModelBinding("llama-3-8b");
    for (const scenarioName of COMPUTER_PRESET_NAMES) {
      const result = simulateDashboard({
        ...base,
        scenarioName,
        modelBinding,
        mode: "serving",
        serving: {
          ...base.serving,
          compareTopologies: false,
          useExpertCache: false,
          decodeMode: "target_only",
          requestCount: 1,
          promptTokens: 16,
          outputTokens: 2,
          maxBatchSize: 1,
          maxBatchTokens: 16,
          prefillChunkTokens: 16,
        },
      });
      expect(result.scenario.id).toBe(scenarioName);
      expect(result.serving?.metrics.outputTokens).toBe(2);
      expect(result.topology.metrics.totalDurationNs).toBeGreaterThan(0);
    }
  });

  it("enforces imported model speculative capabilities at execution", () => {
    const config: DashboardRunConfig = {
      ...base,
      modelBinding: {
        ...createBuiltinModelBinding("llama-3-8b"),
        source: "local_model_package",
        modelFingerprints: ["fnv1a32:12345678"],
        targetModelFingerprint: "fnv1a32:12345678",
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

  it("keeps prompt lookup available for legacy built-in bindings", () => {
    const result = simulateDashboard({
      ...base,
      modelBinding: {
        ...createBuiltinModelBinding("llama-3-8b"),
        speculativeFamilies: [],
      },
      speculative: {
        ...base.speculative,
        family: "prompt_lookup",
        outputTokens: 8,
        draftWidth: 2,
      },
    });

    expect(result.speculative?.family).toBe("prompt_lookup");
    expect(result.speculative?.metrics.committedTokens).toBe(8);
  });

  it("binds model weight traffic into CPU serving throughput", () => {
    const modelBinding = createBuiltinModelBinding("llama-3-8b");
    const result = simulateDashboard({
      ...base,
      scenarioName: "cpu-only",
      modelBinding,
      mode: "serving",
      serving: {
        ...base.serving,
        compareTopologies: false,
        useExpertCache: false,
        decodeMode: "target_only",
        requestCount: 1,
        promptTokens: 16,
        outputTokens: 8,
        maxBatchSize: 1,
        maxBatchTokens: 16,
        prefillChunkTokens: 16,
      },
    });

    expect(result.model).toMatchObject({
      name: modelBinding.displayName,
      totalParameters: modelBinding.totalParameters,
      weightBytes: modelBinding.weightBytes,
    });
    expect(result.serving!.metrics.throughputTokensPerSecond).toBeLessThan(10);
    expect(result.topology.assumptions).toContain(
      `model ${modelBinding.displayName} contributes an active-weight bandwidth floor for every target attention and FFN invocation`,
    );
  });

  it("rejects a model that cannot fit target memory", () => {
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "cpu-only",
      modelBinding: createBuiltinModelBinding("llama-3-70b"),
      mode: "serving",
      serving: {
        ...base.serving,
        compareTopologies: false,
        useExpertCache: false,
        decodeMode: "target_only",
      },
    })).toThrow(
      "model Llama-3-70B requires",
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

  it("runs the selected workload on deterministic small LAN scenarios", () => {
    for (const multiNodeCount of [2, 3, 4] as const) {
      const first = simulateDashboard({
        ...base,
        scenarioName: "multi-node",
        multiNodeCount,
      });
      const second = simulateDashboard({
        ...base,
        scenarioName: "multi-node",
        multiNodeCount,
      });

      expect(first).toEqual(second);
      expect(first.scenario.id).toBe(
        multiNodeCount === 2 ? "multi-node" : `multi-node-${multiNodeCount}`,
      );
      expect(first.scenario.deviceCount).toBe(multiNodeCount * 2);
      expect(first.scenario.linkCount).toBeGreaterThanOrEqual(
        multiNodeCount * (multiNodeCount - 1),
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

  it("stops the failed node at the fault and bounds quiescence by the deadline", () => {
    const result = simulateDashboard({
      ...base,
      scenarioName: "multi-node",
      mode: "fault",
      modelBinding: createBuiltinModelBinding("llama-3-8b"),
      fault: {
        failedNodeId: "node1",
        faultAtUs: 1,
        quiesceTimeoutUs: 50,
        executionCount: 4,
      },
    });
    const fault = result.fault!;

    expect(fault.failedNodeId).toBe("node1");
    expect(fault.abortDeadlineNs).toBe(fault.faultAtNs + fault.quiesceTimeoutNs);
    expect(fault.quiescedAtNs).toBeLessThanOrEqual(fault.abortDeadlineNs);
    expect(fault.quiescedAtNs).toBeGreaterThanOrEqual(fault.faultAtNs);
    // Queued work on the dead node never runs, so operations are dropped.
    expect(fault.droppedOperations).toBeGreaterThan(0);
    expect(fault.retainedOperations + fault.droppedOperations)
      .toBe(fault.plannedOperations);
    // Every rank on the failed node fails exactly at the fault instant.
    const failedNodeRanks = fault.rankStates.filter(
      (state) => state.onFailedNode,
    );
    expect(failedNodeRanks.length).toBeGreaterThan(0);
    for (const state of failedNodeRanks) {
      expect(state.status).toBe("failed");
      expect(state.terminalAtNs).toBe(fault.faultAtNs);
    }
    expect(fault.replayAppliedEvents).toBeGreaterThan(0);
  });

  it("caps quiescence when survivors would drain past the abort deadline", () => {
    const run = (quiesceTimeoutUs: number) => simulateDashboard({
      ...base,
      scenarioName: "multi-node",
      mode: "fault",
      modelBinding: createBuiltinModelBinding("llama-3-8b"),
      fault: {
        failedNodeId: "node1",
        faultAtUs: 1,
        quiesceTimeoutUs,
        executionCount: 4,
      },
    }).fault!;

    const tight = run(1);
    const loose = run(5_000);

    expect(tight.quiescedAtNs).toBe(tight.abortDeadlineNs);
    expect(tight.drainedAtNs).toBeGreaterThan(tight.quiescedAtNs);
    // A deadline beyond the drain point stops being the binding constraint.
    expect(loose.quiescedAtNs).toBeLessThan(loose.abortDeadlineNs);
  });

  it("falls back to a node that participates in the compiled plan", () => {
    const fault = simulateDashboard({
      ...base,
      scenarioName: "multi-node",
      mode: "fault",
      modelBinding: createBuiltinModelBinding("llama-3-8b"),
      fault: {
        failedNodeId: "node-that-does-not-exist",
        faultAtUs: 1,
        quiesceTimeoutUs: 50,
        executionCount: 2,
      },
    }).fault!;

    expect(fault.failedNodeId).not.toBe("node-that-does-not-exist");
    expect(fault.rankStates.some((state) => state.onFailedNode)).toBe(true);
  });

  it("expands the prompt and KV budget by the media a request carries", () => {
    const run = (modality: "text" | "image", mediaItemsPerRequest: number) => {
      const modelBinding = createBuiltinModelBinding(
        "qwen3-vl-8b", "fp16", "fp16", modality,
      );
      return {
        binding: modelBinding,
        result: simulateDashboard({
          ...base,
          mode: "serving",
          modality,
          mediaItemsPerRequest,
          modelBinding,
          serving: {
            ...base.serving,
            useExpertCache: false,
            decodeMode: "target_only",
            requestCount: 2,
            promptTokens: 256,
            outputTokens: 16,
          },
        }),
      };
    };
    const kvBytes = (result: ReturnType<typeof run>["result"]) => (
      result.scenario.memoryLedger.reduce(
        (sum, entry) => sum + (entry.reservedByPurpose.kv ?? 0),
        0,
      )
    );

    const textOnly = run("text", 2);
    const withMedia = run("image", 2);
    const perItem = withMedia.binding.mediaInputs!.find(
      (input) => input.modality === "image",
    )!.decoderTokensPerItem;

    expect(perItem).toBeGreaterThan(0);
    // Two items per request on two requests, all of them prompt positions.
    expect(withMedia.result.serving!.kvBudgetTokens
      - textOnly.result.serving!.kvBudgetTokens)
      .toBe(2 * 2 * perItem);
    expect(kvBytes(withMedia.result)).toBeGreaterThan(kvBytes(textOnly.result));
    // Selecting text-only must ignore the media count entirely.
    expect(run("text", 8).result.serving!.kvBudgetTokens)
      .toBe(textOnly.result.serving!.kvBudgetTokens);
  });

  it("reserves KV from the run's own token budget, not a preset constant", () => {
    const modelBinding = createBuiltinModelBinding("llama-3-8b");
    const bytesPerToken = modelBinding.executionProfile.kvCacheBytesPerToken!;
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      modelBinding,
      serving: {
        ...base.serving,
        useExpertCache: false,
        decodeMode: "target_only",
        requestCount: 4,
        promptTokens: 256,
        outputTokens: 32,
      },
    });

    const budgetTokens = 4 * (256 + 32 - 1);
    const kvBytes = result.scenario.memoryLedger.reduce(
      (sum, entry) => sum + (entry.reservedByPurpose.kv ?? 0),
      0,
    );
    expect(kvBytes).toBe(budgetTokens * bytesPerToken);

    // Doubling the requests must double the KV reservation.
    const doubled = simulateDashboard({
      ...base,
      mode: "serving",
      modelBinding,
      serving: {
        ...base.serving,
        useExpertCache: false,
        decodeMode: "target_only",
        requestCount: 8,
        promptTokens: 256,
        outputTokens: 32,
      },
    });
    const doubledKvBytes = doubled.scenario.memoryLedger.reduce(
      (sum, entry) => sum + (entry.reservedByPurpose.kv ?? 0),
      0,
    );
    expect(doubledKvBytes).toBe(kvBytes * 2);
  });

  it("uses resource-manager limits instead of physical capacity", () => {
    const preset = buildScenarioPreset("cpu-only");
    // The host domain keeps 128 GiB of physical capacity but only 12 GiB of
    // allocatable extent. Expert caches are dropped so the scenario's own
    // reservations still fit inside that limit.
    const customScenario = {
      ...preset,
      id: "cpu-limited",
      family: "custom" as const,
      memoryDomains: preset.memoryDomains.map((domain) => (
        domain.kind === "host"
          ? { ...domain, resourceLimitBytes: 12 * 1024 ** 3 }
          : domain
      )),
      placements: preset.placements.map((placement) => ({
        ...placement,
        allocations: placement.allocations.filter(
          (allocation) => allocation.purpose !== "cache",
        ),
      })),
    };
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario,
      modelBinding: createBuiltinModelBinding("llama-3-8b"),
      mode: "serving",
      serving: {
        ...base.serving,
        useExpertCache: false,
        decodeMode: "target_only",
      },
    })).toThrow(
      "model Llama-3-8B requires",
    );
  });

  it("fails closed when expert routing needs disabled SSD streaming", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const customScenario = {
      ...preset,
      id: "ssd-disabled",
      family: "custom" as const,
      execution: {
        ...preset.execution,
        features: { ssdStreaming: false },
      },
    };
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario,
      mode: "expert-cache",
      expertCache: {
        ...base.expertCache,
        expertCount: 32,
        hotSlots: 4,
        warmSlots: 0,
      },
    })).toThrow(
      "expert cache leaves 28 experts cold but SSD streaming is disabled",
    );
  });

  it("enforces the SSD allocation limit for expert backing", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const customScenario = {
      ...preset,
      id: "ssd-limited",
      family: "custom" as const,
      memoryDomains: preset.memoryDomains.map((domain) => (
        domain.kind === "storage"
          ? { ...domain, resourceLimitBytes: 1024 ** 3 }
          : domain
      )),
      // The backing reservation is trimmed to the new SSD limit so the
      // scenario itself is valid and the expert-backing demand is what fails.
      placements: preset.placements.map((placement) => ({
        ...placement,
        allocations: placement.allocations.map((allocation) => (
          allocation.purpose === "backing"
            ? { ...allocation, bytes: 1024 ** 3 }
            : allocation
        )),
      })),
    };
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario,
      mode: "expert-cache",
      expertCache: {
        ...base.expertCache,
        expertCount: 32,
        hotSlots: 4,
        warmSlots: 0,
      },
    })).toThrow(
      "expert backing requires 2.0 GiB but the resource manager allows 1.0 GiB of SSD",
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

  it("rejects an untrusted dashboard multi-node count", () => {
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "multi-node",
      multiNodeCount: 5 as DashboardRunConfig["multiNodeCount"],
    })).toThrow("dashboard multi-node count must be 2, 3, or 4; got 5");
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

  it("preserves a million-token prompt through chunked prefill", () => {
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 1,
        promptTokens: 1_048_576,
        outputTokens: 1,
        maxBatchSize: 1,
        maxBatchTokens: 512,
        prefillChunkTokens: 512,
      },
    });

    expect(result.serving?.metrics.prefillTokens).toBe(1_048_576);
    expect(result.serving?.metrics.outputTokens).toBe(1);
    expect(result.serving?.batches).toHaveLength(2_048);
  }, 30_000);

  it("preserves a 32K exact output trace", () => {
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 1,
        promptTokens: 16,
        outputTokens: 32_768,
        maxBatchSize: 1,
        maxBatchTokens: 16,
        prefillChunkTokens: 16,
      },
    });

    expect(result.serving?.metrics.outputTokens).toBe(32_768);
    expect(result.serving?.requests[0]?.tokenTimestampsNs)
      .toHaveLength(32_768);
  }, 30_000);

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
  it("reports the co-residency roster rather than the single-model binding", () => {
    // The sidebar shows one model selector in every other mode, so a
    // co-residency run must not silently inherit or be steered by it.
    const withLlama = simulateDashboard({
      ...base,
      scenarioName: "rtx-4090-desktop",
      mode: "co-residency",
      modelBinding: createBuiltinModelBinding("llama-3-8b", "fp16"),
    });
    const withoutBinding = simulateDashboard({
      ...base,
      scenarioName: "rtx-4090-desktop",
      mode: "co-residency",
      modelBinding: undefined,
    });

    expect(withLlama.coResidency?.metrics.tenants.map(
      (tenant) => tenant.displayName,
    )).toStrictEqual(["Qwen3-0.6B", "Qwen3-8B"]);
    expect(withLlama.coResidency?.metrics).toStrictEqual(
      withoutBinding.coResidency?.metrics,
    );
  });

  it("runs a sparse model too large for memory by leaving experts on storage", () => {
    // 109.5 GiB of INT4 weights on a 56 GiB machine. Only 3.7 GiB of that is
    // touched on every token; the routed experts are read as routing reaches
    // them, which is how these models are actually served on a laptop.
    const config: DashboardRunConfig = {
      ...base,
      scenarioName: "mac-mini-m4-pro-64gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("qwen-3-235b", "int4"),
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 1,
        maxBatchSize: 1,
        outputTokens: 4,
      },
    };
    const result = simulateDashboard(config);

    // It runs, and it is slow, because most of what a token reads crosses a
    // 7 GB/s link rather than 273 GB/s memory. Reporting memory-bandwidth
    // speed here would overstate it by more than an order of magnitude.
    const rate = result.serving!.metrics.throughputTokensPerSecond;
    expect(rate).toBeGreaterThan(0.5);
    expect(rate).toBeLessThan(20);

    // The same model on a machine that holds it is far faster, which is the
    // comparison the offload exists to make legible.
    const resident = simulateDashboard({
      ...config,
      scenarioName: "mac-studio-m3-ultra-512gb",
    });
    expect(resident.serving!.metrics.throughputTokensPerSecond)
      .toBeGreaterThan(rate * 3);
  });

  it("keeps failing closed for a dense model that does not fit", () => {
    // A dense model has no expert to leave behind: every byte is read every
    // token, so streaming cannot rescue it and it must still be rejected.
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "mac-mini-m4-pro-64gb",
      mode: "serving",
      serving: { ...base.serving, decodeMode: "target_only" },
      modelBinding: createBuiltinModelBinding("llama-3-70b", "fp16"),
    })).toThrow("requires 131.4 GiB of weights");
  });

  it("refuses to stream when the scenario forbids it, and says so", () => {
    const preset = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const withoutSsd = {
      ...preset,
      id: `${preset.id}-no-ssd`,
      execution: {
        ...preset.execution,
        features: { ...preset.execution.features, ssdStreaming: false },
      },
    };

    // Fails closed, and names the reason rather than leaving the reader to
    // guess that a switch they already found would have changed the answer.
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "custom",
      customScenario: withoutSsd,
      mode: "serving",
      serving: { ...base.serving, decodeMode: "target_only" },
      modelBinding: createBuiltinModelBinding("qwen-3-235b", "int4"),
    })).toThrow(/routed experts, which SSD streaming would leave on storage/);
  });

  it("does not charge a bound model's experts to the cache study's backing", () => {
    // The standalone cache study schedules a synthetic expert set of its own
    // and charges it to the same storage allocation. If the selected model's
    // routed experts were offloaded as well, the two would describe the same
    // bytes and the backing reservation would be tens of GiB of double count.
    const study = simulateDashboard({
      ...base,
      scenarioName: "mac-mini-m4-pro-64gb",
      mode: "expert-cache",
      modelBinding: createBuiltinModelBinding("qwen-3-235b", "int4"),
    });
    const dense = simulateDashboard({
      ...base,
      scenarioName: "mac-mini-m4-pro-64gb",
      mode: "expert-cache",
      modelBinding: createBuiltinModelBinding("llama-3-8b", "int4"),
    });

    const backing = (result: typeof study) => result.scenario.memoryLedger
      .reduce((sum, entry) => sum + (entry.reservedByPurpose.backing ?? 0), 0);
    // The study is synthetic, so its backing cannot depend on which model is
    // selected beside it.
    expect(backing(study)).toBe(backing(dense));
    expect(backing(study)).toBeLessThan(8 * 1024 ** 3);
  });

  it("sizes an offload against the tightest domain, not their total", () => {
    // gpu-npu puts attention weights on a 16 GiB NPU and the FFN on a 48 GiB
    // GPU. Capacity is enforced per domain, so a plan sized against the 64 GiB
    // total overran the GPU and the run was rejected by the resource manager
    // with a message about allocations rather than about residency.
    const result = simulateDashboard({
      ...base,
      scenarioName: "gpu-npu",
      mode: "serving",
      serving: { ...base.serving, decodeMode: "target_only" },
      modelBinding: createBuiltinModelBinding("qwen-3-235b", "int4"),
    });

    const offload = result.model!.expertOffload!;
    expect(offload.residentExperts).toBeGreaterThan(0);
    expect(offload.residentExperts).toBeLessThan(offload.totalExperts);

    // Every domain must still fit, which is the property the total-based
    // sizing violated.
    for (const entry of result.scenario.memoryLedger) {
      if (!entry.enabled) continue;
      expect(entry.freeBytes, entry.domainId).toBeGreaterThanOrEqual(0);
    }
  });

  it("leaves a single-domain machine's offload untouched", () => {
    // The tightest domain is the only domain there, so the stricter sizing
    // must not cost a personal machine any residency.
    const result = simulateDashboard({
      ...base,
      scenarioName: "mac-mini-m4-pro-64gb",
      mode: "serving",
      serving: { ...base.serving, decodeMode: "target_only" },
      modelBinding: createBuiltinModelBinding("qwen-3-235b", "int4"),
    });

    expect(result.model!.expertOffload!.residentExperts).toBe(62);
  });

  it("turns a bandwidth-bound decode into throughput by batching", () => {
    // A wider batch reads the weights once for every sequence in it, so
    // aggregate tokens per second rise even though nothing about the memory
    // system changed. This is the only lever that moves a decode which is
    // bound by bandwidth rather than by capacity.
    const run = (maxBatchSize: number) => simulateDashboard({
      ...base,
      scenarioName: "panther-lake-x9-388h-32gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("gemma-4-e2b", "int4"),
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 32,
        maxBatchSize,
        maxBatchTokens: 1024,
        promptTokens: 512,
        outputTokens: 32,
        arrivalGapUs: 1,
      },
    }).serving!.metrics.throughputTokensPerSecond;

    const single = run(1);
    const wide = run(16);
    expect(wide).toBeGreaterThan(single * 4);

    // And it costs almost nothing in memory, which is the point: this model's
    // KV is small enough that width is free where bandwidth is not.
    const held = (batch: number) => simulateDashboard({
      ...base,
      scenarioName: "panther-lake-x9-388h-32gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("gemma-4-e2b", "int4"),
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 32,
        maxBatchSize: batch,
        maxBatchTokens: 1024,
        promptTokens: 512,
        outputTokens: 32,
        arrivalGapUs: 1,
      },
    }).scenario.memoryLedger.reduce((sum, entry) => sum + entry.reservedBytes, 0);
    expect(held(16)).toBeLessThan(held(1) * 1.5);
  });

  it("blames the reservation, not the model, when KV alone will not fit", () => {
    // Raising the request ceiling made this reachable: the KV arena can
    // exceed the machine on its own, which left the weight comparison
    // reporting a negative budget and reading as though a 0.3 GiB model were
    // too large for a 64 GB desktop.
    expect(() => simulateDashboard({
      ...base,
      scenarioName: "arrow-lake-s-285k-64gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("qwen3-0.6b", "int4"),
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        requestCount: 128,
        maxBatchSize: 64,
        outputTokens: 4096,
      },
    })).toThrow(/reserves .* GiB of KV .* leaves no room/);
  });

  it("measures what speculation bought instead of asserting it", () => {
    const run = (decodeMode: "target_only" | "prompt_lookup") => simulateDashboard({
      ...base,
      scenarioName: "panther-lake-x9-388h-32gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("gemma-4-e2b", "int4"),
      serving: {
        ...base.serving,
        decodeMode,
        draftWidth: 4,
        firstPositionAcceptance: 0.82,
        requestCount: 4,
        maxBatchSize: 1,
        outputTokens: 32,
      },
    });

    const speculative = run("prompt_lookup");
    const gain = speculative.speculativeGain!;
    // The reported baseline has to be the same run without speculation, not a
    // figure derived from the acceptance ladder.
    expect(gain.baselineTokensPerSecond).toBeCloseTo(
      run("target_only").serving!.metrics.throughputTokensPerSecond,
      6,
    );
    expect(gain.speedup).toBeCloseTo(
      gain.speculativeTokensPerSecond / gain.baselineTokensPerSecond,
      6,
    );
    expect(gain.speedup).toBeGreaterThan(1.5);
    // The acceptance travels with the number so it is not mistaken for one.
    expect(gain.firstPositionAcceptance).toBe(0.82);

    // A run that did not speculate has nothing to report.
    expect(run("target_only").speculativeGain).toBeUndefined();
  });

  it("shows speculation and batching buying the same thing", () => {
    // Both amortise one weight read over several tokens, so they do not stack:
    // the same acceptance that is worth 2.6x on a single stream is worth
    // almost nothing once a wide batch has already saturated memory. Reporting
    // the speedup without this would invite stacking them in a plan.
    const speedupAt = (maxBatchSize: number) => simulateDashboard({
      ...base,
      scenarioName: "panther-lake-x9-388h-32gb",
      mode: "serving",
      modelBinding: createBuiltinModelBinding("gemma-4-e2b", "int4"),
      serving: {
        ...base.serving,
        decodeMode: "prompt_lookup",
        draftWidth: 4,
        firstPositionAcceptance: 0.82,
        requestCount: Math.max(8, maxBatchSize * 2),
        maxBatchSize,
        maxBatchTokens: 2048,
        outputTokens: 32,
        arrivalGapUs: 1,
      },
    }).speculativeGain!;

    const narrow = speedupAt(1);
    const wide = speedupAt(64);
    expect(narrow.speedup).toBeGreaterThan(2);
    expect(wide.speedup).toBeLessThan(1.2);
    // The acceptance did not change; only what it was worth did. The two runs
    // draft over different request counts, so the committed length is sampled
    // slightly differently, but it stays within a few percent while the
    // speedup falls by more than half.
    const drift = Math.abs(
      wide.committedTokensPerTargetForward
        - narrow.committedTokensPerTargetForward,
    ) / narrow.committedTokensPerTargetForward;
    expect(drift).toBeLessThan(0.1);
    expect(wide.speedup / narrow.speedup).toBeLessThan(0.5);
  });

  it("caps the speedup at what the acceptance allows", () => {
    // The ceiling is what a reader needs when a claim of five-fold arrives
    // without a model or a platform attached: a verification pass commits what
    // it accepts and no more, so no drafter and no machine can beat the
    // accepted length. A run short of its ceiling is paying for drafting; a
    // ceiling short of the claim means the acceptance was never going to
    // allow it.
    const run = (firstPositionAcceptance: number, draftWidth: number) =>
      simulateDashboard({
        ...base,
        scenarioName: "mac-studio-m3-ultra-512gb",
        mode: "serving",
        modelBinding: createBuiltinModelBinding("deepseek-v3", "int4"),
        serving: {
          ...base.serving,
          decodeMode: "mtp",
          draftWidth,
          firstPositionAcceptance,
          outputTokens: 32,
        },
      }).speculativeGain!;

    const modest = run(0.72, 4);
    expect(modest.speedup).toBeLessThanOrEqual(modest.ceiling + 1e-9);
    expect(modest.ceiling).toBeCloseTo(
      modest.committedTokensPerTargetForward,
      6,
    );

    // Even an acceptance nobody measures, at the widest draft offered, stays
    // under five: the ladder decays faster than width can compensate for.
    const generous = run(0.95, 8);
    expect(generous.ceiling).toBeGreaterThan(modest.ceiling);
    expect(generous.speedup).toBeLessThan(5);
  });

  it("never predicts a rate the silicon cannot issue", () => {
    // Two separate faults put one model's prefill at 73 times its own compute
    // roof, and the chart could only report that as an unexplained conflict.
    //
    // The first was arithmetic: FLOPs came from the whole parameter count, but
    // a per-layer embedding table is gathered, not multiplied, and on this
    // checkpoint that table is about half of every parameter present.
    //
    // The second was timing: the per-token costs are normalized constants that
    // do not scale with model size, so nothing stopped a wide prefill from
    // implying an unbounded rate. That is invisible on a discrete accelerator
    // and badly wrong on an integrated one.
    const computers = [
      "rtx-5090-desktop",
      "rtx-4090-desktop",
      "mac-mini-m4-pro-64gb",
      "ryzen-ai-max-395-128gb",
      "panther-lake-x9-388h-32gb",
      "arrow-lake-s-285k-64gb",
    ] as const;
    const models = [
      "gemma-4-e2b",
      "qwen3-0.6b",
      "llama-3-8b",
      "qwen3-30b-a3b",
    ] as const;
    for (const scenarioName of computers) {
      for (const preset of models) {
        const roofline = simulateDashboard({
          ...base,
          scenarioName,
          mode: "serving",
          serving: { ...base.serving, decodeMode: "target_only" },
          modelBinding: createBuiltinModelBinding(preset, "int4", "int4"),
        }).roofline;
        if (roofline === undefined || roofline.status === "unavailable") {
          continue;
        }
        const roof = roofline.bandwidthRoofs[0];
        expect(roof).toBeDefined();
        const verdict = interpretRoofline(roofline, roof!, roofline.points);
        expect(
          `${preset} on ${scenarioName}: ${verdict.verdict}`,
        ).not.toContain("Evidence conflict");

        // A device that publishes a peak for this dtype must bound the run.
        // Where none is published there is nothing to bound it with, which
        // the chart already reports rather than papering over.
        if (roofline.computeRoof !== undefined) {
          for (const point of roofline.points) {
            expect(point.predictedFlopsPerSecond)
              .toBeLessThanOrEqual(roofline.computeRoof.flopsPerSecond * 1.05);
          }
        }
      }
    }
  });

  it("charges gathered embeddings as bytes and not as arithmetic", () => {
    // The per-layer table only distinguishes this checkpoint from its
    // siblings, so comparing intensities is what shows the lookup is no
    // longer being multiplied: a table half the size of the model would
    // otherwise put this one at roughly twice everything near its size.
    const intensity = (preset: DashboardModelPreset) => {
      const profile = createBuiltinModelBinding(preset, "int4")
        .executionProfile;
      return profile.forwardFlopsPerToken / (
        profile.attentionWeightBytesPerToken + profile.ffnWeightBytesPerToken
      );
    };
    const withTable = intensity("gemma-4-e2b");
    const peers = [
      intensity("qwen3-0.6b"),
      intensity("llama-3.2-1b"),
      intensity("qwen3-4b"),
    ];
    for (const peer of peers) {
      expect(withTable / peer).toBeLessThan(1.35);
    }
  });
});
