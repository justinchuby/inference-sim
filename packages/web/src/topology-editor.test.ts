import { describe, expect, it } from "vitest";
import { buildScenarioPreset } from "@inference-sim/core";
import {
  bytesToGibibytes,
  bytesToGigabytesPerSecond,
  finalizeEditedTopology,
  gibibytesToBytes,
  gigabytesPerSecondToBytes,
} from "./topology-editor.js";

describe("topology editor boundary", () => {
  it("round-trips displayed memory and bandwidth units", () => {
    expect(bytesToGibibytes(gibibytesToBytes(80))).toBe(80);
    expect(bytesToGigabytesPerSecond(
      gigabytesPerSecondToBytes(600),
    )).toBe(600);
  });

  it("marks edited topology evidence and advances its epoch", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const edited = finalizeEditedTopology({
      ...preset,
      links: preset.links.map((link, index) => (
        index === 0
          ? { ...link, bandwidthBytesPerSec: 48_000_000_000 }
          : link
      )),
    });
    expect(edited.id).toBe("single-gpu-cpu-custom");
    expect(edited.family).toBe("custom");
    expect(edited.execution.topologyEpoch)
      .toBe(preset.execution.topologyEpoch + 1);
    expect(edited.links[0].bandwidthBytesPerSec).toBe(48_000_000_000);
    expect(edited.links[0].provenance.source).toContain("user-edited");
  });

  it("rejects invalid edits through the shared scenario parser", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    expect(() => finalizeEditedTopology({
      ...preset,
      devices: preset.devices.map((device, index) => (
        index === 0 ? { ...device, maxConcurrentCompute: 0 } : device
      )),
    })).toThrow("scenario validation failed");
    expect(() => gibibytesToBytes(0)).toThrow("positive safe integer");
  });
});
