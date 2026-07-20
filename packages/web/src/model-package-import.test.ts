import { fromJson, toBinary } from "@bufbuild/protobuf";
import { ModelProtoSchema } from "onnx-buf";
import { describe, expect, it } from "vitest";
import {
  inspectBrowserModelPackage,
  type BrowserPackageFile,
} from "./model-package-import.js";

function tinyOnnxModel(
  externalLocation: string,
  externalLength: number,
): Uint8Array {
  return toBinary(ModelProtoSchema, fromJson(ModelProtoSchema, {
    irVersion: "11",
    producerName: "inference-sim-web-test",
    graph: {
      name: "tiny",
      node: [{
        opType: "MatMul",
        input: ["input", "weight"],
        output: ["output"],
      }],
      initializer: [{
        name: "weight",
        dims: ["2", "2"],
        dataType: 1,
        externalData: [
          { key: "location", value: externalLocation },
          { key: "offset", value: "0" },
          { key: "length", value: String(externalLength) },
        ],
        dataLocation: 1,
      }],
      input: [{ name: "input" }],
      output: [{ name: "output" }],
    },
  }));
}

function packageFile(
  path: string,
  bytes: Uint8Array | string,
): BrowserPackageFile {
  const data = typeof bytes === "string"
    ? new TextEncoder().encode(bytes)
    : bytes;
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);
  const blob = new Blob([copy.buffer]);
  return {
    name: path.split("/").at(-1)!,
    size: blob.size,
    webkitRelativePath: path,
    arrayBuffer: () => blob.arrayBuffer(),
    stream: () => blob.stream(),
  };
}

describe("browser model package import", () => {
  it("parses local protobufs, pipeline metadata, and external sidecars", async () => {
    const metadata = `
pipeline:
  models:
    decoder:
      filename: decoder.onnx
      type: decoder
  strategy:
    kind: autoregressive
    decoder: decoder
speculative:
  proposal_type: mtp
  num_speculative_tokens: 3
`;
    const result = await inspectBrowserModelPackage([
      packageFile("model/inference_metadata.yaml", metadata),
      packageFile(
        "model/decoder.onnx",
        tinyOnnxModel("decoder.onnx.data", 16),
      ),
      packageFile("model/decoder.onnx.data", new Uint8Array(16).fill(7)),
    ]);

    expect(result.metadata.pipelineStrategy).toBe("autoregressive");
    expect(result.metadata.speculative.availableFamilies).toEqual(["mtp"]);
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      fileName: "decoder.onnx",
      componentIds: ["decoder"],
      manifest: {
        graph: { nodeCount: 1 },
        totals: { externalInitializerBytes: 16 },
      },
    });
    expect(result.models[0]!.manifest.externalDataFiles[0]).toMatchObject({
      location: "decoder.onnx.data",
      byteLength: 16,
      referencedByteLength: 16,
    });
    expect(result.models[0]!.manifest.externalDataFiles[0]!.sha256)
      .toMatch(/^[0-9a-f]{64}$/);
  });

  it("rejects missing model components and sidecars", async () => {
    await expect(inspectBrowserModelPackage([
      packageFile(
        "model/inference_metadata.json",
        JSON.stringify({
          pipeline: {
            models: {
              decoder: { filename: "missing.onnx", type: "decoder" },
            },
            strategy: { kind: "autoregressive", decoder: "decoder" },
          },
        }),
      ),
      packageFile(
        "model/decoder.onnx",
        tinyOnnxModel("decoder.onnx.data", 16),
      ),
    ])).rejects.toThrow("is missing missing.onnx");

    await expect(inspectBrowserModelPackage([
      packageFile(
        "model/decoder.onnx",
        tinyOnnxModel("decoder.onnx.data", 16),
      ),
    ])).rejects.toThrow("references missing external data");
  });

  it("rejects ambiguous metadata roots", async () => {
    await expect(inspectBrowserModelPackage([
      packageFile("model/a/inference_metadata.yaml", "{}"),
      packageFile("model/b/inference_metadata.json", "{}"),
      packageFile("model/decoder.onnx", tinyOnnxModel("data.bin", 16)),
      packageFile("model/data.bin", new Uint8Array(16)),
    ])).rejects.toThrow("multiple inference_metadata");
  });
});
