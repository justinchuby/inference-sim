import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { basename, dirname, resolve, sep } from "node:path";
import type { OnnxModelManifest } from "@inference-sim/core";
import {
  inspectOnnxModelBytes,
  type OnnxExternalDataSource,
} from "@inference-sim/onnx-inspector";

export async function inspectOnnxModel(
  modelPath: string,
  metadata?: unknown,
): Promise<OnnxModelManifest> {
  const resolvedModelPath = resolve(modelPath);
  const modelDirectory = dirname(resolvedModelPath);
  return inspectOnnxModelBytes({
    modelFileName: basename(modelPath),
    modelBytes: await readFile(resolvedModelPath),
    metadata,
    sha256: async (bytes) => (
      createHash("sha256").update(bytes).digest("hex")
    ),
    resolveExternalData: async (location) => {
      const filePath = resolve(modelDirectory, ...location.split("/"));
      if (
        filePath !== modelDirectory
        && !filePath.startsWith(`${modelDirectory}${sep}`)
      ) {
        throw new Error(
          `external-data location escapes model directory: ${location}`,
        );
      }
      const fileStat = await stat(filePath);
      if (!fileStat.isFile()) {
        throw new Error(`external-data location is not a file: ${location}`);
      }
      return {
        byteLength: fileStat.size,
        sha256: () => sha256File(filePath),
      } satisfies OnnxExternalDataSource;
    },
  });
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest("hex");
}
