import {
  parseSpeculativeTokenTrace,
  simulateSpeculativeTokenTrace,
  type SpeculativeTokenTrace,
  type SpeculativeTokenTraceResult,
} from "@inference-sim/core";
import {
  MAX_IMPORT_FILE_BYTES,
  parseYamlOrJsonFileText,
} from "./calibration-import.js";

export const MAX_TOKEN_TRACE_FILE_BYTES = MAX_IMPORT_FILE_BYTES;

export interface ParsedTokenTraceFile {
  readonly trace: SpeculativeTokenTrace;
  readonly preview: SpeculativeTokenTraceResult;
}

export async function parseTokenTraceFileText(
  text: string,
  fileName: string,
): Promise<ParsedTokenTraceFile> {
  const input = await parseYamlOrJsonFileText(text, fileName, "token trace");
  const trace = parseSpeculativeTokenTrace(input);
  return {
    trace,
    preview: simulateSpeculativeTokenTrace(trace),
  };
}
