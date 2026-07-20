import {
  parseOnnxModelManifest,
  type OnnxModelManifest,
} from "@inference-sim/core";

export const MAX_ONNX_MANIFEST_FILE_BYTES = 64 * 1024 * 1024;

export function parseOnnxManifestFileText(
  text: string,
  fileName: string,
): OnnxModelManifest {
  if (
    new TextEncoder().encode(text).byteLength
      > MAX_ONNX_MANIFEST_FILE_BYTES
  ) {
    throw new Error("ONNX manifest exceeds the 64 MiB limit");
  }
  if (!fileName.toLowerCase().endsWith(".json")) {
    throw new Error("ONNX manifest file must use .json");
  }
  let input: unknown;
  try {
    input = JSON.parse(text) as unknown;
  } catch (error) {
    throw new Error(
      `invalid ONNX manifest JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  return parseOnnxModelManifest(input);
}
