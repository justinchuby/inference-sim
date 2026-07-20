import {
  fitTopologyCostModel,
  parseCalibrationDataset,
  type CalibrationDataset,
  type CalibrationFitResult,
} from "@inference-sim/core";

export const MAX_CALIBRATION_FILE_BYTES = 1024 * 1024;

export interface ParsedCalibrationFile {
  readonly dataset: CalibrationDataset;
  readonly fit: CalibrationFitResult;
}

export async function parseCalibrationFileText(
  text: string,
  fileName: string,
): Promise<ParsedCalibrationFile> {
  if (new TextEncoder().encode(text).byteLength > MAX_CALIBRATION_FILE_BYTES) {
    throw new Error("calibration file exceeds the 1 MiB limit");
  }
  const extension = fileName.split(".").at(-1)?.toLowerCase();
  let input: unknown;
  if (extension === "json") {
    input = JSON.parse(text) as unknown;
  } else if (extension === "yaml" || extension === "yml") {
    const { parse: parseYaml } = await import("yaml");
    input = parseYaml(text) as unknown;
  } else {
    throw new Error("calibration file must use .yaml, .yml, or .json");
  }
  const dataset = parseCalibrationDataset(input);
  return {
    dataset,
    fit: fitTopologyCostModel(dataset),
  };
}
