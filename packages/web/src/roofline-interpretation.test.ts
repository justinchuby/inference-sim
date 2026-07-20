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
    revision: 1,
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

  it("does not claim a bottleneck without a compute ceiling", () => {
    const roofline = { ...result(10, 5e12), computeRoof: undefined };
    const interpretation = interpretRoofline(
      roofline,
      memoryRoof,
      roofline.points,
    );
    expect(interpretation.verdict).toBe("Bandwidth ceiling only");
    expect(interpretation.explanation).toContain("cannot be proven");
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
});
