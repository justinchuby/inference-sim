import { describe, expect, it } from "vitest";
import { buildScenarioPreset } from "@inference-sim/core";
import { estimateContextCapacity } from "./context-capacity.js";
import { createBuiltinModelBinding } from "./model-binding.js";
import type { DashboardRunConfig } from "./types.js";

const config: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  multiGpuRanks: 2,
  mode: "serving",
  seed: 42,
  modelBinding: createBuiltinModelBinding("llama-3-8b"),
  speculative: {
    family: "prompt_lookup",
    outputTokens: 128,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    compareTopologies: false,
    useExpertCache: false,
    decodeMode: "target_only",
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
    requestCount: 12,
    arrivalGapUs: 250,
    promptTokens: 512,
    outputTokens: 64,
    maxBatchSize: 8,
    maxBatchTokens: 128,
    prefillChunkTokens: 64,
  },
  expertCache: {
    placementStrategy: "contiguous",
    tokenCount: 96,
    topK: 2,
    expertCount: 16,
    hotSlots: 6,
    warmSlots: 8,
    adaptivePrefetch: true,
  },
};

describe("context capacity estimate", () => {
  it("derives per-request and single-sequence limits from KV placement", () => {
    const estimate = estimateContextCapacity(
      config,
      buildScenarioPreset("multi-gpu"),
    );
    expect(estimate.status).toBe("available");
    if (estimate.status !== "available") return;
    expect(estimate.kvCacheBytesPerToken).toBe(128 * 1024);
    expect(estimate.maxContextTokensPerRequest).toBe(
      Math.floor(estimate.maxSingleSequenceTokens / 12),
    );
    expect(estimate.bottleneckDomainId).toBe("node0:gpu0:vram");
    expect(estimate.fitsConfiguredContext).toBe(true);
  });

  it("reports unavailable evidence when no model is bound", () => {
    const { modelBinding: _removed, ...withoutModel } = config;
    expect(estimateContextCapacity(
      withoutModel,
      buildScenarioPreset("multi-gpu"),
    )).toMatchObject({ status: "unavailable" });
  });
});
