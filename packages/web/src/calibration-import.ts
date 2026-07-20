import {
  fitTopologyCostModel,
  parseCalibrationDataset,
  type CalibrationDataset,
  type CalibrationFitResult,
} from "@inference-sim/core";

export const MAX_CALIBRATION_FILE_BYTES = 1024 * 1024;
export const MAX_IMPORT_FILE_BYTES = MAX_CALIBRATION_FILE_BYTES;

export interface ParsedCalibrationFile {
  readonly dataset: CalibrationDataset;
  readonly fit: CalibrationFitResult;
}

export async function parseCalibrationFileText(
  text: string,
  fileName: string,
): Promise<ParsedCalibrationFile> {
  const input = await parseYamlOrJsonFileText(text, fileName, "calibration");
  const dataset = parseCalibrationDataset(input);
  return {
    dataset,
    fit: fitTopologyCostModel(dataset),
  };
}

export async function parseYamlOrJsonFileText(
  text: string,
  fileName: string,
  kind: string,
): Promise<unknown> {
  if (new TextEncoder().encode(text).byteLength > MAX_IMPORT_FILE_BYTES) {
    throw new Error(`${kind} file exceeds the 1 MiB limit`);
  }
  const extension = fileName.split(".").at(-1)?.toLowerCase();
  if (extension === "json") {
    return JSON.parse(text) as unknown;
  }
  if (extension === "yaml" || extension === "yml") {
    const { parse: parseYaml } = await import("yaml");
    return parseYaml(text) as unknown;
  }
  throw new Error(`${kind} file must use .yaml, .yml, or .json`);
}
