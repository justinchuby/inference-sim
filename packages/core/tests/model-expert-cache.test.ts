import { describe, expect, it } from "vitest";
import {
  buildModelProfile,
  topExpertMass,
  buildScenarioPreset,
  expertCacheFromModel,
  expertCacheTierDomains,
  expertRoutingWeights,
  simulateExpertCacheWorkload,
} from "../src/index.js";

const GiB = 1024 ** 3;

describe("model-bound expert cache", () => {
  it("takes its expert set and routed width from the checkpoint", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const bound = expertCacheFromModel(model, scenario, {
      hotCapacityBytes: 8 * GiB,
      warmCapacityBytes: 4 * GiB,
      routingSeed: 42,
    })!;

    // Not a knob: the checkpoint declares 128 experts and routes 8 per token.
    expect(bound.config.experts).toHaveLength(model.moe!.numExperts);
    expect(bound.topK).toBe(model.moe!.activeExpertsPerToken);
    expect(bound.bytesPerExpert).toBe(
      model.moe!.expertBytesPerLayer * model.architecture.numLayers,
    );
    // Every expert has the model's real extent, not a 64 MiB placeholder.
    for (const expert of bound.config.experts) {
      expect(expert.bytes).toBe(bound.bytesPerExpert);
    }
  });

  it("derives promotion cost from the tier the bytes actually come from", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const bound = expertCacheFromModel(model, scenario, {
      hotCapacityBytes: 8 * GiB,
      warmCapacityBytes: 0,
      routingSeed: 42,
    })!;
    const storage = scenario.memoryDomains.find(
      (domain) => domain.kind === "storage",
    )!;

    // An 850 MiB expert over a 7 GB/s SSD is over a tenth of a second. The
    // constant this replaced was 2.2 ms, which is off by more than 50x and
    // could not have been right for both this and a 64 MiB expert.
    const link = scenario.links.find((candidate) => (
      candidate.sourceDomainId === storage.id
    ))!;
    // Both hops count: the read out of storage and the link it crosses. The
    // narrower of the two rates binds, and both latencies are paid.
    const bandwidth = Math.min(
      storage.bandwidthBytesPerSec,
      link.bandwidthBytesPerSec,
    );
    const expected = storage.latencyNs
      + link.latencyNs
      + (bound.bytesPerExpert / bandwidth) * 1e9;
    expect(bound.config.coldToHotLatencyNs).toBeCloseTo(expected, -4);
    expect(bound.config.coldToHotLatencyNs).toBeGreaterThan(50_000_000);
  });

  it("scales promotion cost with the model, not with a constant", () => {
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const options = {
      hotCapacityBytes: 8 * GiB,
      warmCapacityBytes: 0,
      routingSeed: 42,
    };
    const big = expertCacheFromModel(
      buildModelProfile("qwen-3-235b", "int4", "fp16"),
      scenario,
      options,
    )!;
    const small = expertCacheFromModel(
      buildModelProfile("qwen3-30b-a3b", "int4", "fp16"),
      scenario,
      options,
    )!;

    expect(big.bytesPerExpert).toBeGreaterThan(small.bytesPerExpert * 4);
    expect(big.config.coldToHotLatencyNs)
      .toBeGreaterThan(small.config.coldToHotLatencyNs * 4);
  });

  it("holds the most-requested experts first", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const bound = expertCacheFromModel(model, scenario, {
      hotCapacityBytes: 8 * GiB,
      warmCapacityBytes: 8 * GiB,
      routingSeed: 42,
    })!;

    const weightOf = new Map(
      bound.config.experts.map((expert) => [expert.id, expert.routingWeight!]),
    );
    const hot = bound.config.initialHotExpertIds!;
    const warm = bound.config.initialWarmExpertIds!;
    expect(hot.length).toBe(bound.hotExperts);
    expect(warm.length).toBe(bound.warmExperts);
    // Preloading the cache with arbitrary experts would understate a real
    // runtime, which converges on keeping the popular ones.
    const coldest = Math.min(...hot.map((id) => weightOf.get(id)!));
    const warmest = Math.max(...warm.map((id) => weightOf.get(id)!));
    expect(coldest).toBeGreaterThanOrEqual(warmest);
  });

  it("routes according to the declared skew when simulated", () => {
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const bound = expertCacheFromModel(model, scenario, {
      hotCapacityBytes: 16 * GiB,
      warmCapacityBytes: 0,
      routingSeed: 7,
    })!;
    const result = simulateExpertCacheWorkload({
      cache: bound.config,
      tokenCount: 200,
      topK: bound.topK,
      tokenIntervalNs: 1_000_000,
    });

    // Zipf-skewed routing means the resident head absorbs most requests. A
    // uniform draw over 128 experts with 19 resident would hit about 15%.
    expect(result.snapshot.metrics.routedExperts).toBe(200 * bound.topK);
    expect(result.snapshot.metrics.hotHitRate).toBeGreaterThan(0.4);
  });

  it("offers nothing to bind for a dense checkpoint", () => {
    expect(expertCacheFromModel(
      buildModelProfile("llama-3-8b", "int4", "fp16"),
      buildScenarioPreset("mac-mini-m4-pro-64gb"),
      { hotCapacityBytes: 8 * GiB, warmCapacityBytes: 0, routingSeed: 42 },
    )).toBeUndefined();
  });

  it("weights experts by rank under every declared distribution", () => {
    expect(expertRoutingWeights({ kind: "uniform" }, 4))
      .toStrictEqual([1, 1, 1, 1]);
    // Strictly decreasing under skew, so rank order is meaningful.
    const zipf = expertRoutingWeights({ kind: "zipf", s: 1.05 }, 8);
    for (let index = 1; index < zipf.length; index++) {
      expect(zipf[index]!).toBeLessThan(zipf[index - 1]!);
    }
    // Clustered splits its declared mass across the hot set and the rest.
    const clustered = expertRoutingWeights(
      { kind: "clustered", hotExperts: 2, hotFrequency: 0.8 },
      4,
    );
    expect(clustered[0]).toBeCloseTo(0.4, 10);
    expect(clustered[3]).toBeCloseTo(0.1, 10);
    // Every weight must stay positive or the sampler cannot draw the expert.
    for (const distribution of [
      { kind: "uniform" } as const,
      { kind: "zipf", s: 1.2 } as const,
      { kind: "clustered", hotExperts: 2, hotFrequency: 0.9 } as const,
      { kind: "empirical", frequencies: [4, 2, 1] } as const,
    ]) {
      for (const weight of expertRoutingWeights(distribution, 16)) {
        expect(weight).toBeGreaterThan(0);
      }
    }
  });

  it("moves nothing when every expert is already resident", () => {
    // A cache large enough for the whole checkpoint must never fetch. This is
    // about traffic only: switching routing on legitimately raises FFN compute
    // for a sparse model, because a token really does run several expert FFNs.
    // That cost is where the model's expert width is expressed and is not a
    // cache effect, so it is not asserted away here.
    const scenario = buildScenarioPreset("mac-mini-m4-pro-64gb");
    const model = buildModelProfile("qwen3-30b-a3b", "int4", "fp16");
    const bound = expertCacheFromModel(model, scenario, {
      hotCapacityBytes: 64 * GiB,
      warmCapacityBytes: 0,
      routingSeed: 42,
    })!;

    // Every expert resident, so nothing is ever fetched.
    expect(bound.hotExperts).toBe(model.moe!.numExperts);
    const result = simulateExpertCacheWorkload({
      cache: bound.config,
      tokenCount: 64,
      topK: bound.topK,
      tokenIntervalNs: 1_000_000,
    });
    expect(result.snapshot.metrics.hotHitRate).toBe(1);
    expect(result.snapshot.metrics.bytesMoved).toBe(0);
    expect(result.snapshot.metrics.stallNs).toBe(0);
  });

  it("keeps the same hit rate whichever granularity the experts are held at", () => {
    // This bounds the known limitation. Caching whole-depth experts rather
    // than expert-layer units gets the reload volume wrong, but capacity and
    // demand both scale by the layer count, so the resident share and the hit
    // rate it produces are identical. Only the volume needs fixing, which is
    // why the binding is sound to build on.
    const model = buildModelProfile("qwen-3-235b", "int4", "fp16");
    const moe = model.moe!;
    const layers = model.architecture.numLayers;
    const budget = 40 * GiB;

    const fullDepthResident = Math.floor(
      budget / (moe.expertBytesPerLayer * layers),
    );
    const perLayerResident = Math.floor(
      budget / moe.expertBytesPerLayer / layers,
    );
    expect(fullDepthResident).toBe(perLayerResident);
    expect(fullDepthResident).toBeGreaterThan(0);
    expect(fullDepthResident).toBeLessThan(moe.numExperts);

    expect(
      topExpertMass(moe.activationDistribution, fullDepthResident, moe.numExperts),
    ).toBe(
      topExpertMass(moe.activationDistribution, perLayerResident, moe.numExperts),
    );
  });

  it("charges a promotion at the link when the link is the narrow part", () => {
    // A discrete-GPU machine promotes from host DRAM over PCIe. DRAM reads far
    // faster than PCIe carries, so charging the source domain's own rate would
    // understate every promotion on any machine whose tiers are really
    // separate. The shipped unified-memory preset cannot expose this, so the
    // narrow link is constructed here.
    const preset = buildScenarioPreset("rtx-4090-desktop");
    const model = buildModelProfile("qwen3-30b-a3b", "int4", "fp16");
    const tiers = expertCacheTierDomains(preset)!;
    expect(tiers.warmDomainId).not.toBe(tiers.hotDomainId);

    const warm = preset.memoryDomains.find(
      (domain) => domain.id === tiers.warmDomainId,
    )!;
    const narrowed = {
      ...preset,
      links: preset.links.map((link) => (
        link.sourceDomainId === tiers.warmDomainId
          && link.targetDomainId === tiers.hotDomainId
          ? { ...link, bandwidthBytesPerSec: Math.floor(
              warm.bandwidthBytesPerSec / 4,
            ) }
          : link
      )),
    };

    const wide = expertCacheFromModel(model, preset, {
      hotCapacityBytes: 4 * GiB,
      warmCapacityBytes: 4 * GiB,
      routingSeed: 42,
    })!;
    const narrow = expertCacheFromModel(model, narrowed, {
      hotCapacityBytes: 4 * GiB,
      warmCapacityBytes: 4 * GiB,
      routingSeed: 42,
    })!;

    expect(narrow.config.warmToHotLatencyNs)
      .toBeGreaterThan(wide.config.warmToHotLatencyNs);
    // Cold promotions come from storage and are unaffected by this link.
    expect(narrow.config.coldToHotLatencyNs)
      .toBe(wide.config.coldToHotLatencyNs);
  });
});
