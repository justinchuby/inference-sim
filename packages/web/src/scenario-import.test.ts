import { describe, expect, it } from "vitest";
import { buildScenarioPreset } from "@inference-sim/core";
import {
  MAX_SCENARIO_FILE_BYTES,
  parseScenarioFileText,
} from "./scenario-import.js";

describe("scenario browser import", () => {
  it("parses strict JSON and YAML custom scenarios", async () => {
    const scenario = {
      ...buildScenarioPreset("gpu-npu"),
      id: "custom-gpu-npu",
      family: "custom" as const,
    };

    expect((await parseScenarioFileText(
      JSON.stringify(scenario),
      "scenario.json",
    )).scenario).toEqual(scenario);
    expect((await parseScenarioFileText(
      JSON.stringify(scenario),
      "scenario.yaml",
    )).scenario).toEqual(scenario);
  });

  it("rejects unknown fields, extensions, and oversized input", async () => {
    const scenario = {
      ...buildScenarioPreset("cpu-only"),
      implicitRuntimeDefault: true,
    };
    await expect(parseScenarioFileText(
      JSON.stringify(scenario),
      "scenario.json",
    )).rejects.toThrow("unknown fields implicitRuntimeDefault");
    await expect(parseScenarioFileText(
      JSON.stringify(buildScenarioPreset("cpu-only")),
      "scenario.txt",
    )).rejects.toThrow("must use .json, .yaml, or .yml");
    await expect(parseScenarioFileText(
      " ".repeat(MAX_SCENARIO_FILE_BYTES + 1),
      "scenario.yaml",
    )).rejects.toThrow("exceeds the 4 MiB limit");
  });
});
