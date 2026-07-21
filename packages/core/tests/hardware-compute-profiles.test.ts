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

  it("offers sourced CPU profiles without inventing undisclosed peaks", () => {
    const cpuProfiles = HARDWARE_COMPUTE_PROFILES.filter((profile) => (
      profile.deviceKind === "cpu"
    ));
    expect(cpuProfiles).toHaveLength(40);
    expect(new Set(cpuProfiles.map((profile) => profile.vendor)))
      .toEqual(new Set(["AMD", "Intel", "Apple", "Qualcomm"]));
    expect(cpuProfiles.every((profile) => profile.deviceKind === "cpu")).toBe(true);
    expect(cpuProfiles.every((profile) => Number(profile.releaseDate.slice(0, 4)) >= 2022))
      .toBe(true);
    expect(denseHardwareComputePeak("amd-epyc-9654-cpu", "fp32"))
      .toMatchObject({ operationsPerSecond: 7.3728e12 });
    expect(hardwareComputeProfile("apple-m4-max-cpu")?.peaks)
      .toEqual([]);
    expect(hardwareComputeProfile("qualcomm-snapdragon-x2-elite-extreme-cpu")?.peaks)
      .toEqual([]);
  });

  it("covers representative CPU families and performance tiers", () => {
    const ids = new Set(HARDWARE_COMPUTE_PROFILES.filter((profile) => (
      profile.deviceKind === "cpu"
    )).map((profile) => profile.id));
    const expectedFamilies = [
      ["apple-m2-cpu", "apple-m2-pro-cpu", "apple-m2-max-cpu", "apple-m2-ultra-cpu"],
      ["apple-m3-cpu", "apple-m3-pro-cpu", "apple-m3-max-cpu", "apple-m3-ultra-cpu"],
      ["apple-m4-cpu", "apple-m4-pro-cpu", "apple-m4-max-cpu"],
      ["intel-core-i5-13600k-cpu", "intel-core-i7-13700k-cpu", "intel-core-i9-13900k-cpu"],
      ["intel-core-i5-14600k-cpu", "intel-core-i7-14700k-cpu", "intel-core-i9-14900k-cpu"],
      ["intel-core-ultra-5-125h-cpu", "intel-core-ultra-7-165h-cpu", "intel-core-ultra-9-185h-cpu"],
      ["intel-core-ultra-5-245k-cpu", "intel-core-ultra-7-265k-cpu", "intel-core-ultra-9-285k-cpu"],
      ["intel-xeon-platinum-8480-plus-cpu", "intel-xeon-platinum-8592-plus-cpu", "intel-xeon-6980p-cpu"],
      ["amd-ryzen-5-7600x-cpu", "amd-ryzen-7-7700x-cpu", "amd-ryzen-9-7950x-cpu"],
      ["amd-ryzen-5-9600x-cpu", "amd-ryzen-7-9700x-cpu", "amd-ryzen-9-9950x-cpu"],
      ["amd-threadripper-pro-7995wx-cpu", "amd-threadripper-pro-9995wx-cpu"],
      ["amd-epyc-9654-cpu", "amd-epyc-9965-cpu"],
      ["qualcomm-snapdragon-x-plus-cpu", "qualcomm-snapdragon-x-elite-cpu"],
      ["qualcomm-snapdragon-x2-elite-cpu", "qualcomm-snapdragon-x2-elite-extreme-cpu"],
    ];
    for (const family of expectedFamilies) {
      expect(family.every((id) => ids.has(id)), `missing CPU family member from ${family.join(", ")}`)
        .toBe(true);
    }
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
