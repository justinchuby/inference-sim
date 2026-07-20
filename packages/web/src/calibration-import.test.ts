import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { parse as parseYaml } from "yaml";
import {
  MAX_CALIBRATION_FILE_BYTES,
  parseCalibrationFileText,
} from "./calibration-import.js";

const exampleUrl = new URL(
  "../../../examples/calibration-synthetic.yaml",
  import.meta.url,
);

describe("calibration file import", () => {
  it("imports YAML through the shared core contract", async () => {
    const text = await readFile(exampleUrl, "utf8");
    const result = await parseCalibrationFileText(text, "calibration.yaml");

    expect(result.dataset.id).toBe("synthetic-linear-example");
    expect(result.fit.confidence).toBe("heuristic");
    expect(result.fit.diagnostics).toHaveLength(15);
    expect(result.fit.costModel.validWorkItemRanges?.gpu.attention).toEqual({
      minWorkItems: 1,
      maxWorkItems: 128,
    });
  });

  it("imports equivalent JSON with the same fingerprint", async () => {
    const text = await readFile(exampleUrl, "utf8");
    const yaml = await parseCalibrationFileText(text, "calibration.yml");
    const json = await parseCalibrationFileText(
      JSON.stringify(parseYaml(text)),
      "calibration.json",
    );

    expect(json.fit.datasetFingerprint).toBe(yaml.fit.datasetFingerprint);
  });

  it("rejects unsupported and oversized files before fitting", async () => {
    await expect(parseCalibrationFileText("{}", "calibration.txt"))
      .rejects.toThrow("must use .yaml, .yml, or .json");
    await expect(parseCalibrationFileText(
      "x".repeat(MAX_CALIBRATION_FILE_BYTES + 1),
      "calibration.yaml",
    )).rejects.toThrow("exceeds the 1 MiB limit");
  });
});
