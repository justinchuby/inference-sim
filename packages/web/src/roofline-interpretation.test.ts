import { describe, expect, it } from "vitest";
import { interpretRoofline } from "./roofline-interpretation.js";
import type { DashboardRooflineResult } from "./types.js";

const memoryRoof = {
  id: "memory:vram",
  label: "GPU VRAM",
  kind: "device_memory" as const,
  bytesPerSecond: 1e12,
  confidence: "heuristic" as const,
};

function result(
  intensity: number,
  predictedFlopsPerSecond: number,
  compute = 1e14,
): DashboardRooflineResult {
  return {
    revision: 2,
    status: "available",
    confidence: "heuristic",
    assumptions: [],
    computeRoof: {
      label: "Effective compute",
      flopsPerSecond: compute,
      evidence: "heuristic_effective",
      dtype: "fp16",
    },
    bandwidthRoofs: [memoryRoof],
    points: [{
      id: "decode",
      label: "Decode",
      phase: "decode",
      deviceIds: ["gpu"],
      workFlops: 1e12,
      activeBytes: 1e10,
      durationNs: 1e7,
      arithmeticIntensity: intensity,
      predictedFlopsPerSecond,
      limitingRoofId: "unresolved",
      confidence: "heuristic",
      notes: [],
    }],
  };
}

describe("interpretRoofline", () => {
  it("explains bandwidth and compute sides of the knee", () => {
    expect(interpretRoofline(result(10, 5e12), memoryRoof, result(10, 5e12).points).verdict)
      .toContain("bandwidth-sensitive");
    expect(interpretRoofline(result(200, 5e13), memoryRoof, result(200, 5e13).points).verdict)
      .toBe("Effective compute-sensitive");
  });

  it("flags predicted rates above the selected roof", () => {
    const roofline = result(10, 2e13);
    const interpretation = interpretRoofline(
      roofline,
      memoryRoof,
      roofline.points,
    );
    expect(interpretation.tone).toBe("danger");
    expect(interpretation.verdict).toBe("Evidence conflict");
  });

  it("does not claim a bottleneck when bandwidth alone cannot explain the rate", () => {
    // Half the bandwidth ceiling: something holds this work back, and without
    // a compute ceiling there is nothing to test that something against.
    const roofline = { ...result(10, 5e12), computeRoof: undefined };
    const interpretation = interpretRoofline(
      roofline,
      memoryRoof,
      roofline.points,
    );
    expect(interpretation.verdict).toBe("Limiting resource unproven");
    expect(interpretation.tone).toBe("warning");
    expect(interpretation.explanation).toContain("50%");
  });

  it("concludes bandwidth bound without a compute ceiling when work is at the roof", () => {
    // A compute ceiling can only sit higher than the work, so it cannot be
    // what limits work already at its bandwidth ceiling. That is provable
    // with no compute evidence at all, and refusing to say so was the defect.
    const roofline = { ...result(10, 9.6e12), computeRoof: undefined };
    const interpretation = interpretRoofline(
      roofline,
      memoryRoof,
      roofline.points,
    );
    expect(interpretation.verdict).toBe("Bandwidth bound");
    expect(interpretation.tone).toBe("neutral");
    expect(interpretation.explanation).toContain("can only be higher");
  });

  it("says why the compute ceiling is missing, not merely that it is", () => {
    // The causes need opposite advice, so each is named. Telling a reader to
    // try another dtype is wrong when the vendor publishes none at all.
    const base = { ...result(10, 5e12), computeRoof: undefined };
    const read = (
      absence: DashboardRooflineResult["computeRoofAbsence"],
    ) => interpretRoofline(
      { ...base, ...(absence === undefined ? {} : { computeRoofAbsence: absence }) },
      memoryRoof,
      base.points,
    );

    const publishesNone = read({
      reason: "profile_publishes_none",
      dtype: "int4",
      deviceLabels: ["mac-mini:m4-pro-gpu"],
      publishedDtypes: [],
    });
    expect(publishesNone.explanation).toContain("mac-mini:m4-pro-gpu");
    expect(publishesNone.explanation).toContain("no absolute compute peak for any dtype");
    expect(publishesNone.nextStep).not.toContain("different dtype");

    const missingDtype = read({
      reason: "dtype_not_published",
      dtype: "int4",
      deviceLabels: ["desktop:rtx5090"],
      publishedDtypes: ["fp16", "fp8"],
    });
    // Names what IS published, so the reader can pick one.
    expect(missingDtype.explanation).toContain("fp16, fp8");
    expect(missingDtype.nextStep).toContain("int4");

    const unbound = read({
      reason: "no_profile",
      dtype: "fp16",
      deviceLabels: ["node0:gpu0"],
      publishedDtypes: [],
    });
    expect(unbound.explanation).toContain("no compute profile bound");
    expect(unbound.nextStep).toContain("machine preset");

    const mixed = read({
      reason: "mixed_devices",
      dtype: "fp16",
      deviceLabels: ["desktop:cpu"],
      publishedDtypes: [],
    });
    expect(mixed.explanation).toContain("cannot be summed");
  });

  it("rejects aggregate work against a single-device roof", () => {
    const roofline = result(200, 5e13);
    const aggregatePoint = {
      ...roofline.points[0]!,
      deviceIds: ["gpu0", "gpu1"],
    };
    const interpretation = interpretRoofline(
      roofline,
      memoryRoof,
      [aggregatePoint],
    );
    expect(interpretation.verdict).toBe("Resource scope mismatch");
  });

  it("credits a declared peak so the advice it gives actually works", () => {
    // The remedy the chart suggests has to be reachable. A custom peak
    // replaces the vendor profile rather than joining it, which is why the
    // wording names the control instead of saying "declare a peak".
    const withPeak = {
      ...result(10, 5e12),
      computeRoof: {
        label: "User-declared dense compute",
        flopsPerSecond: 6e13,
        evidence: "user_declared" as const,
        dtype: "fp16",
      },
    };
    const interpretation = interpretRoofline(
      withPeak,
      memoryRoof,
      withPeak.points,
    );
    expect(interpretation.verdict).not.toBe("Limiting resource unproven");
    expect(interpretation.verdict).not.toBe("Bandwidth bound");
  });
});
