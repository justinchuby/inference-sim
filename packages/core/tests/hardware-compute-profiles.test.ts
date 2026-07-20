import { describe, expect, it } from "vitest";
import {
  HARDWARE_COMPUTE_PROFILES,
  denseHardwareComputePeak,
  hardwareComputeProfile,
} from "../src/index.js";

describe("hardware compute registry", () => {
  it("contains unique, sourced, post-2021 profiles with valid peaks", () => {
    expect(HARDWARE_COMPUTE_PROFILES.length).toBeGreaterThanOrEqual(30);
    expect(new Set(HARDWARE_COMPUTE_PROFILES.map((profile) => profile.id)).size)
      .toBe(HARDWARE_COMPUTE_PROFILES.length);
    for (const profile of HARDWARE_COMPUTE_PROFILES) {
      expect(Date.parse(profile.releaseDate)).not.toBeNaN();
      expect(Number(profile.releaseDate.slice(0, 4))).toBeGreaterThanOrEqual(2022);
      expect(profile.sources.length).toBeGreaterThan(0);
      for (const itemSource of profile.sources) {
        expect(itemSource.url).toMatch(/^https:\/\//);
      }
      for (const peak of profile.peaks) {
        expect(peak.operationsPerSecond).toBeGreaterThan(0);
        expect(profile.sources[peak.sourceIndex]).toBeDefined();
      }
    }
  });

  it("selects a conservative dense accumulator mode", () => {
    expect(denseHardwareComputePeak(
      "nvidia-geforce-rtx-4090",
      "fp16",
    )).toMatchObject({
      operationsPerSecond: 165.2e12,
      accumulationDtype: "fp32",
      sparsity: "dense",
    });
  });

  it("does not reinterpret generic TOPS as dtype-specific compute", () => {
    expect(hardwareComputeProfile("apple-m4-neural-engine")?.peaks[0])
      .toMatchObject({ dtype: "vendor_ai", sparsity: "unspecified" });
    expect(denseHardwareComputePeak("apple-m4-neural-engine", "int8"))
      .toBeUndefined();
  });

  it("does not treat unspecified density as a dense roof", () => {
    expect(denseHardwareComputePeak("intel-gaudi-3", "bf16"))
      .toBeUndefined();
  });
});
