import { describe, expect, it } from "vitest";
import {
  buildModelProfile,
  planExpertResidency,
  topExpertMass,
} from "../src/index.js";

describe("expert residency", () => {
  it("weights the hit rate by how often each expert is actually asked for", () => {
    // Half the experts resident. Under uniform routing that is half the reads;
    // under skew the resident half is the popular half and hits far more, so a
    // plain count ratio would badly understate a real cache.
    const uniform = topExpertMass({ kind: "uniform" }, 64, 128);
    const skewed = topExpertMass({ kind: "zipf", s: 1.05 }, 64, 128);

    expect(uniform).toBeCloseTo(0.5, 10);
    expect(skewed).toBeGreaterThan(0.85);
    expect(skewed).toBeLessThan(1);
  });

  it("returns no mass for nothing resident and all of it for everything", () => {
    for (const distribution of [
      { kind: "uniform" } as const,
      { kind: "zipf", s: 1.05 } as const,
      { kind: "clustered", hotExperts: 8, hotFrequency: 0.7 } as const,
      { kind: "empirical", frequencies: [5, 3, 1, 1] } as const,
    ]) {
      expect(topExpertMass(distribution, 0, 128)).toBe(0);
      expect(topExpertMass(distribution, 128, 128)).toBe(1);
      // Monotone: holding one more expert can never hit less often.
      let previous = 0;
      for (const count of [1, 2, 8, 32, 96, 127]) {
        const mass = topExpertMass(distribution, count, 128);
        expect(mass).toBeGreaterThanOrEqual(previous);
        expect(mass).toBeLessThanOrEqual(1);
        previous = mass;
      }
    }
  });

  it("credits a clustered cache with its hot set before any cold expert", () => {
    const distribution = {
      kind: "clustered",
      hotExperts: 8,
      hotFrequency: 0.8,
    } as const;

    // Exactly the hot set resident captures exactly the hot frequency.
    expect(topExpertMass(distribution, 8, 64)).toBeCloseTo(0.8, 10);
    // Half the hot set captures half of it.
    expect(topExpertMass(distribution, 4, 64)).toBeCloseTo(0.4, 10);
  });

  it("holds experts whole and leaves the remainder on storage", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const perExpert =
      model.moe!.expertBytesPerLayer * model.architecture.numLayers;

    // A budget of two and a half experts holds two: a half-resident expert
    // still has to be read from storage.
    const plan = planExpertResidency(model, perExpert * 2.5)!;
    expect(plan.residentExpertsPerLayer).toBe(2);
    expect(plan.residentExpertBytes).toBe(perExpert * 2);
    expect(plan.residentExpertBytes + plan.streamedExpertBytes).toBe(
      perExpert * model.moe!.numExperts,
    );
  });

  it("streams nothing once every expert fits", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const plan = planExpertResidency(model, Number.MAX_SAFE_INTEGER)!;

    expect(plan.residentExpertsPerLayer).toBe(model.moe!.numExperts);
    expect(plan.streamedExpertBytes).toBe(0);
    expect(plan.residentHitFraction).toBe(1);
    expect(plan.streamedBytesPerToken).toBe(0);
  });

  it("streams every routed read when nothing fits", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const plan = planExpertResidency(model, 0)!;

    expect(plan.residentExpertsPerLayer).toBe(0);
    expect(plan.residentHitFraction).toBe(0);
    // The whole active routed extent crosses storage on every token.
    const activeRouted = model.moe!.activeExpertsPerToken
      * model.moe!.expertBytesPerLayer
      * model.architecture.numLayers;
    expect(plan.streamedBytesPerToken).toBeCloseTo(activeRouted, 0);
  });

  it("offers no plan for a dense model", () => {
    // A dense model has no expert to leave behind, so it cannot be made to fit
    // by streaming and must keep failing against a machine that is too small.
    expect(planExpertResidency(
      buildModelProfile("llama-3-8b", "int4", "fp16"),
      0,
    )).toBeUndefined();
  });
});
