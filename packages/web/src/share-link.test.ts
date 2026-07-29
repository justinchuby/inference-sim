import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "./App.js";
import {
  decodeDashboardShareLink,
  encodeDashboardShareLink,
} from "./share-link.js";
import { createBuiltinModelBinding } from "./model-binding.js";
import { buildScenarioPreset } from "@inference-sim/core";
import type { DashboardRunConfig } from "./types.js";

function roundTrip(config: DashboardRunConfig): DashboardRunConfig {
  const { search } = encodeDashboardShareLink(config, DEFAULT_CONFIG);
  return decodeDashboardShareLink(search, DEFAULT_CONFIG).config;
}

describe("share link", () => {
  it("writes nothing when nothing was changed", () => {
    // A link to the default run is the bare address. Restating every control
    // would make the common case the longest one.
    expect(encodeDashboardShareLink(DEFAULT_CONFIG, DEFAULT_CONFIG).search)
      .toBe("");
  });

  it("carries only what differs, and reads it back", () => {
    const config: DashboardRunConfig = {
      ...DEFAULT_CONFIG,
      scenarioName: "panther-lake-x9-388h-32gb",
      modelBinding: createBuiltinModelBinding("gemma-4-e2b", "int4"),
      serving: {
        ...DEFAULT_CONFIG.serving,
        maxBatchSize: 16,
        requestCount: 32,
      },
    };
    const { search } = encodeDashboardShareLink(config, DEFAULT_CONFIG);

    // Readable rather than an opaque blob, so a reader can see what a link
    // will change before opening it.
    expect(search).toContain("scenario=panther-lake-x9-388h-32gb");
    expect(search).toContain("model=gemma-4-e2b");
    expect(search).toContain("batch=16");
    expect(search).not.toContain("seed");
    expect(search).not.toContain("prompt");

    const decoded = decodeDashboardShareLink(search, DEFAULT_CONFIG).config;
    expect(decoded.scenarioName).toBe("panther-lake-x9-388h-32gb");
    expect(decoded.serving.maxBatchSize).toBe(16);
    expect(decoded.serving.requestCount).toBe(32);
    expect(decoded.modelBinding!.executionProfile.modelId).toBe("gemma-4-e2b");
    expect(decoded.modelBinding!.modelFormat!.weightDtypes[0]).toBe("int4");
    // Untouched fields keep the reader's defaults rather than the sender's.
    expect(decoded.serving.promptTokens)
      .toBe(DEFAULT_CONFIG.serving.promptTokens);
  });

  it("round-trips every mode and its own controls", () => {
    const configs: readonly DashboardRunConfig[] = [
      { ...DEFAULT_CONFIG, mode: "speculative", speculative: {
        ...DEFAULT_CONFIG.speculative,
        family: "mtp",
        draftWidth: 6,
        firstPositionAcceptance: 0.75,
      } },
      { ...DEFAULT_CONFIG, mode: "fault", fault: {
        ...DEFAULT_CONFIG.fault,
        failedNodeId: "node0:gpu1",
        faultAtUs: 120,
        executionCount: 8,
      } },
      { ...DEFAULT_CONFIG, mode: "co-residency", coResidency: {
        ...DEFAULT_CONFIG.coResidency,
        models: [
          { preset: "gemma-4-e2b", weightDtype: "int4", contextTokens: 4096, pinned: true, requestCount: 2 },
          { preset: "whisper-large-v3", weightDtype: "fp16", contextTokens: 1024, pinned: false, requestCount: 3 },
        ],
        requestGapMs: 2000,
      } },
      { ...DEFAULT_CONFIG, mode: "expert-cache", modality: "image",
        mediaItemsPerRequest: 4 },
    ];

    for (const config of configs) {
      const decoded = roundTrip(config);
      expect(decoded.mode, config.mode).toBe(config.mode);
      expect(decoded.speculative, config.mode).toStrictEqual(config.speculative);
      expect(decoded.fault, config.mode).toStrictEqual(config.fault);
      expect(decoded.coResidency, config.mode)
        .toStrictEqual(config.coResidency);
      expect(decoded.modality, config.mode).toBe(config.modality);
      expect(decoded.mediaItemsPerRequest, config.mode)
        .toBe(config.mediaItemsPerRequest);
    }
  });

  it("refuses values the controls could not have produced", () => {
    // A link is untrusted input. Bounds are the run's own, so a hand-edited
    // one cannot reach a state the form cannot.
    for (const [search, key] of [
      ["batch=4096", "batch"],
      ["batch=-1", "batch"],
      ["batch=abc", "batch"],
      ["requests=99999", "requests"],
      ["mode=demolish", "mode"],
      ["scenario=made-up-machine", "scenario"],
      ["specAccept=5", "specAccept"],
      ["input=smell", "input"],
    ] as const) {
      const decoded = decodeDashboardShareLink(search, DEFAULT_CONFIG);
      expect(decoded.warnings.join(" "), search).toContain(key);
      // And the rest of the configuration survives one bad key.
      expect(decoded.config.serving.maxBatchSize, search)
        .toBe(DEFAULT_CONFIG.serving.maxBatchSize);
    }
  });

  it("keeps the good half of a link with one bad key", () => {
    const decoded = decodeDashboardShareLink(
      "scenario=arrow-lake-s-285k-64gb&batch=99999&requests=12",
      DEFAULT_CONFIG,
    );

    expect(decoded.warnings).toHaveLength(1);
    expect(decoded.config.scenarioName).toBe("arrow-lake-s-285k-64gb");
    expect(decoded.config.serving.requestCount).toBe(12);
    expect(decoded.config.serving.maxBatchSize)
      .toBe(DEFAULT_CONFIG.serving.maxBatchSize);
  });

  it("rejects a malformed co-residency roster rather than half-applying it", () => {
    // A roster is one field, so a bad entry has to discard the field. Applying
    // the entries that parsed would silently change how many models are being
    // served, which is the whole subject of that mode.
    const decoded = decodeDashboardShareLink(
      "crModels=gemma-4-e2b:int4:4096:1:2,not-a-model:int4:4096:0:1",
      DEFAULT_CONFIG,
    );

    expect(decoded.warnings.join(" ")).toContain("crModels");
    expect(decoded.config.coResidency.models)
      .toStrictEqual(DEFAULT_CONFIG.coResidency.models);
  });

  it("says what it could not carry instead of dropping it silently", () => {
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const { omitted } = encodeDashboardShareLink({
      ...DEFAULT_CONFIG,
      scenarioName: "custom",
      customScenario: scenario,
    }, DEFAULT_CONFIG);

    expect(omitted.map((entry) => entry.field)).toContain("customScenario");
    expect(omitted[0]!.reason).toMatch(/larger than a URL/);
  });

  it("stays short enough to paste", () => {
    // Every field at once, which no real link carries, still has to fit
    // comfortably inside what mail clients and chat apps will keep intact.
    const config: DashboardRunConfig = {
      ...DEFAULT_CONFIG,
      scenarioName: "multi-node",
      mode: "co-residency",
      seed: 12345,
      modality: "video",
      mediaItemsPerRequest: 3,
      modelBinding: createBuiltinModelBinding("qwen3-vl-8b", "int8", "int8"),
      serving: {
        ...DEFAULT_CONFIG.serving,
        requestCount: 64, arrivalGapUs: 1234, promptTokens: 4096,
        outputTokens: 512, maxBatchSize: 32, maxBatchTokens: 1024,
        prefillChunkTokens: 256, compareTopologies: true,
        useExpertCache: true, decodeMode: "mtp",
      },
    };
    const { search } = encodeDashboardShareLink(config, DEFAULT_CONFIG);

    expect(search.length).toBeLessThan(600);
    expect(roundTrip(config).serving).toStrictEqual(config.serving);
  });
});
