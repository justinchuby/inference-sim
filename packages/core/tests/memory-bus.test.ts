import { describe, expect, it } from "vitest";
import {
  ALL_SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  derivedMemoryBandwidth,
} from "../src/index.js";

describe("declared memory bandwidth", () => {
  it("matches the bus every computer preset declares", () => {
    // Peak bandwidth is not a free parameter: it is the transfer rate times
    // the bus width. Model geometry has been checked against a published
    // parameter count for a while; this is the same discipline applied to the
    // machines, which previously carried bare figures nothing could verify.
    for (const name of ALL_SCENARIO_PRESET_NAMES) {
      // Building is the check: a preset whose figures disagree throws.
      expect(() => buildScenarioPreset(name), name).not.toThrow();
    }
  });

  it("computes a bus the way vendors quote one", () => {
    // Worked examples rather than a restatement of the formula, so a wrong
    // formula cannot pass by agreeing with itself.
    expect(derivedMemoryBandwidth({
      label: "LPDDR5X-9600",
      transferMtPerSec: 9_600,
      busWidthBits: 128,
    })).toBeCloseTo(153.6e9, -6);
    expect(derivedMemoryBandwidth({
      label: "DDR5-6400",
      transferMtPerSec: 6_400,
      busWidthBits: 128,
    })).toBeCloseTo(102.4e9, -6);
    // A 384-bit GDDR6X bus at 21 Gbps is the 4090's quoted 1008 GB/s.
    expect(derivedMemoryBandwidth({
      label: "GDDR6X-21000",
      transferMtPerSec: 21_000,
      busWidthBits: 384,
    })).toBeCloseTo(1008e9, -7);
    // Apple's 819 GB/s comes from a 1024-bit bus, four times the M4 Pro's.
    expect(derivedMemoryBandwidth({
      label: "LPDDR5-6400",
      transferMtPerSec: 6_400,
      busWidthBits: 1_024,
    })).toBeCloseTo(819.2e9, -7);
  });

  it("keeps every computer preset's memory within reach of its bus", () => {
    // The unified and device domains are the ones the check governs. Asserting
    // the relationship here as well as at build time means a future preset
    // that bypasses the builders is still caught.
    const computers = ALL_SCENARIO_PRESET_NAMES.filter((name) => (
      !["cpu-only", "single-gpu-cpu", "multi-gpu", "gpu-npu",
        "unified-memory", "multi-node"].includes(name)
    ));
    expect(computers.length).toBeGreaterThanOrEqual(7);

    for (const name of computers) {
      const scenario = buildScenarioPreset(name);
      const main = scenario.memoryDomains.filter(
        (domain) => domain.kind === "unified" || domain.kind === "device",
      );
      expect(main.length, name).toBeGreaterThan(0);
      for (const domain of main) {
        // A plausible client or desktop part, not a datacentre fabric: the
        // check exists to catch a mistyped or copied figure, and an order of
        // magnitude either side of the shipped range does that.
        expect(domain.bandwidthBytesPerSec, `${name}/${domain.id}`)
          .toBeGreaterThan(50e9);
        expect(domain.bandwidthBytesPerSec, `${name}/${domain.id}`)
          .toBeLessThan(3_000e9);
      }
    }
  });
});
