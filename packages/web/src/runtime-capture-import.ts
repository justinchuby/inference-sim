import {
  bindSpeculativeRuntimeCaptures,
  parseRuntimeTokenCapture,
  type SpeculativeRuntimeCapture,
  type SpeculativeTokenTrace,
  type SpeculativeTokenTraceResult,
  type TargetOnlyRuntimeCapture,
} from "@inference-sim/core";
import {
  MAX_IMPORT_FILE_BYTES,
  parseYamlOrJsonFileText,
} from "./calibration-import.js";

export const MAX_RUNTIME_CAPTURE_FILE_BYTES = MAX_IMPORT_FILE_BYTES;

export interface RuntimeCaptureFileText {
  readonly fileName: string;
  readonly text: string;
}

export interface ParsedRuntimeCapturePair {
  readonly trace: SpeculativeTokenTrace;
  readonly preview: SpeculativeTokenTraceResult;
  readonly targetOnly: {
    readonly fileName: string;
    readonly capture: TargetOnlyRuntimeCapture;
  };
  readonly speculative: {
    readonly fileName: string;
    readonly capture: SpeculativeRuntimeCapture;
  };
}

export async function parseRuntimeCapturePairFileTexts(
  files: readonly RuntimeCaptureFileText[],
): Promise<ParsedRuntimeCapturePair> {
  if (files.length !== 2) {
    throw new Error("runtime evidence import requires exactly two files");
  }
  let targetOnly:
    | ParsedRuntimeCapturePair["targetOnly"]
    | undefined;
  let speculative:
    | ParsedRuntimeCapturePair["speculative"]
    | undefined;
  for (const file of files) {
    const input = await parseYamlOrJsonFileText(
      file.text,
      file.fileName,
      "runtime capture",
    );
    const capture = parseRuntimeTokenCapture(input);
    if (capture.role === "target_only") {
      if (targetOnly) {
        throw new Error("runtime evidence contains two target-only captures");
      }
      targetOnly = { fileName: file.fileName, capture };
    } else {
      if (speculative) {
        throw new Error("runtime evidence contains two speculative captures");
      }
      speculative = { fileName: file.fileName, capture };
    }
  }
  if (!targetOnly || !speculative) {
    throw new Error(
      "runtime evidence requires one target-only and one speculative capture",
    );
  }
  const bound = bindSpeculativeRuntimeCaptures(
    targetOnly.capture,
    speculative.capture,
  );
  return {
    trace: bound.trace,
    preview: bound.result,
    targetOnly,
    speculative,
  };
}
