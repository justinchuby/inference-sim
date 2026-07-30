import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "./App.js";
import { simulateDashboard } from "./dashboard-simulation.js";
import { buildScenarioPreset } from "@inference-sim/core";
import {
  createBuiltinModelBinding,
  modelSupportsSpeculativeFamily,
  presetsShippingProposers,
} from "./model-binding.js";
import type { ScenarioPresetName } from "@inference-sim/core";

describe("opening configuration", () => {
  it("starts on one personal computer with nothing batched", () => {
    // The first screen is what a new reader learns the model from, so it must
    // be the simplest run that is still real: one machine, one request, one
    // sequence per batch, prefilled in a single pass.
    expect(DEFAULT_CONFIG.scenarioName).toBe("mac-mini-m4-pro-64gb");
    expect(DEFAULT_CONFIG.mode).toBe("serving");
    expect(DEFAULT_CONFIG.serving.requestCount).toBe(1);
    expect(DEFAULT_CONFIG.serving.maxBatchSize).toBe(1);
    expect(DEFAULT_CONFIG.serving.prefillChunkTokens)
      .toBeGreaterThanOrEqual(DEFAULT_CONFIG.serving.promptTokens);

    // Every optional mechanism is off, so each one can be turned on alone.
    expect(DEFAULT_CONFIG.serving.decodeMode).toBe("target_only");
    expect(DEFAULT_CONFIG.serving.useExpertCache).toBe(false);
    expect(DEFAULT_CONFIG.serving.compareTopologies).toBe(false);
    expect(DEFAULT_CONFIG.modality).toBe("text");
  });

  it("opens on a single-device topology with no links to reason about", () => {
    // Narrowed rather than cast: a default of "custom" would have no preset to
    // build, and that is a failure worth seeing here.
    const name = DEFAULT_CONFIG.scenarioName;
    expect(name).not.toBe("custom");
    const scenario = buildScenarioPreset(name as ScenarioPresetName);

    // One node, and unified memory, so there is no host-to-device copy to
    // explain before anything else makes sense.
    expect(new Set(scenario.devices.map((device) => device.nodeId)).size).toBe(1);
    expect(scenario.memoryDomains.filter(
      (domain) => domain.kind !== "storage",
    )).toHaveLength(1);
  });

  it("fits the machine it ships with and reports a plausible rate", () => {
    const result = simulateDashboard(DEFAULT_CONFIG);

    // A quantized 8B is what a laptop actually holds; at FP16 it would be 15
    // GiB and four times slower, which is a worse introduction.
    expect(result.model!.weightBytes).toBeLessThan(5 * 1024 ** 3);
    const allocatable = result.scenario.memoryLedger.reduce(
      (sum, entry) => sum + entry.capacityBytes,
      0,
    );
    expect(result.model!.weightBytes).toBeLessThan(allocatable);

    // Roughly what an M4 Pro measures on an INT4 8B. A wide band: this pins
    // the order of magnitude, not the cost model.
    const rate = result.serving!.metrics.throughputTokensPerSecond;
    expect(rate).toBeGreaterThan(20);
    expect(rate).toBeLessThan(120);
  });

  it("does not claim to batch when every batch held one sequence", () => {
    const result = simulateDashboard(DEFAULT_CONFIG);

    expect(result.serving!.batches.length).toBeGreaterThan(0);
    for (const batch of result.serving!.batches) {
      expect(batch.sequenceCount).toBe(1);
    }

    // Turning batching on is the first thing a reader tries, and it must
    // actually batch, or the unbatched wording above would be reporting the
    // configuration rather than what the run did.
    const batched = simulateDashboard({
      ...DEFAULT_CONFIG,
      serving: {
        ...DEFAULT_CONFIG.serving,
        requestCount: 8,
        maxBatchSize: 4,
      },
    });
    expect(batched.serving!.batches.some((batch) => batch.sequenceCount > 1))
      .toBe(true);
  });

  it("offers only prompt lookup, and can say which models offer more", () => {
    // The opening model ships no drafter, so the decode-mode menu is two
    // entries long. That reads as an unfinished tool unless the reason is
    // stated, and the reason is a fact about the checkpoint: prompt lookup
    // needs nothing from the weights, everything else is a second set of
    // weights the publisher either released or did not.
    const families = ["mtp", "eagle3", "draft_model", "shared_kv"] as const;
    for (const family of families) {
      expect(modelSupportsSpeculativeFamily(DEFAULT_CONFIG.modelBinding!, family))
        .toBe(false);
    }
    expect(
      modelSupportsSpeculativeFamily(DEFAULT_CONFIG.modelBinding!, "prompt_lookup"),
    ).toBe(true);

    // The pointer the note gives has to be real, and has to be read out of the
    // specs so that declaring a head on a new preset never leaves a hand-typed
    // list behind saying otherwise.
    const shipped = presetsShippingProposers();
    expect(shipped.length).toBeGreaterThan(0);
    expect(shipped.map((entry) => entry.preset)).toContain("deepseek-v3");
    expect(shipped.map((entry) => entry.preset)).not.toContain("llama-3-8b");
    for (const entry of shipped) {
      expect(entry.displayName.length).toBeGreaterThan(0);
      for (const family of entry.families) {
        expect(
          modelSupportsSpeculativeFamily(
            createBuiltinModelBinding(entry.preset, "int4"),
            family,
          ),
        ).toBe(true);
      }
    }
  });
});
