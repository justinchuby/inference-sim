import { describe, expect, it } from "vitest";
import {
  ONNX_MODEL_MANIFEST_KIND,
  ONNX_MODEL_MANIFEST_REVISION,
  canonicalJsonFingerprint,
  createOnnxModelManifest,
  parseOnnxModelManifest,
  resolveOnnxModelProfile,
  serializeOnnxModelManifest,
  type OnnxModelManifestUnsigned,
} from "../src/index.js";

function unsigned(): OnnxModelManifestUnsigned {
  return {
    kind: ONNX_MODEL_MANIFEST_KIND,
    revision: ONNX_MODEL_MANIFEST_REVISION,
    source: {
      modelFileName: "model.onnx",
      modelByteLength: 108,
      sha256: "a".repeat(64),
    },
    model: {
      irVersion: "11",
      producerName: "test",
      producerVersion: "",
      domain: "",
      modelVersion: "0",
    },
    graph: {
      name: "graph",
      nodeCount: 1,
      initializerCount: 1,
      inputNames: ["input_ids"],
      outputNames: ["logits"],
      operators: [{ domain: "ai.onnx", opType: "MatMul", count: 1 }],
    },
    initializers: [{
      name: "weight",
      dataType: "float",
      dimensions: [2, 2],
      elementCount: 4,
      logicalByteLength: 16,
      storage: {
        kind: "external",
        byteLength: 16,
        location: "model.onnx.data",
        offset: 0,
      },
    }],
    externalDataFiles: [{
      location: "model.onnx.data",
      byteLength: 16,
      referencedByteLength: 16,
      sha256: "b".repeat(64),
    }],
    architecture: {
      source: "onnx_genai_manifest",
      modelType: "TinyCausalLM",
      hiddenSize: 2,
      intermediateSize: 4,
      numHiddenLayers: 1,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDimension: 2,
      vocabSize: 8,
    },
    totals: {
      initializerElements: 4,
      initializerLogicalBytes: 16,
      inlineInitializerBytes: 0,
      externalInitializerBytes: 16,
    },
    profileReadiness: {
      ready: true,
      missingFields: [],
    },
  };
}

describe("ONNX model manifest", () => {
  it("creates a deterministic revisioned manifest", () => {
    const first = createOnnxModelManifest(unsigned());
    const second = createOnnxModelManifest(unsigned());

    expect(first).toEqual(second);
    expect(serializeOnnxModelManifest(first)).toBe(
      serializeOnnxModelManifest(second),
    );
    expect(parseOnnxModelManifest(
      JSON.parse(serializeOnnxModelManifest(first)),
    )).toEqual(first);
  });

  it("rejects fingerprint tampering", () => {
    const manifest = structuredClone(createOnnxModelManifest(unsigned()));
    manifest.initializers[0].logicalByteLength++;

    expect(() => parseOnnxModelManifest(manifest))
      .toThrow("totals do not match initializer inventory");
  });

  it("rejects re-signed inconsistent totals", () => {
    const manifest = structuredClone(createOnnxModelManifest(unsigned()));
    manifest.totals.initializerLogicalBytes++;
    const { manifestFingerprint: _ignored, ...tamperedUnsigned } = manifest;
    manifest.manifestFingerprint = canonicalJsonFingerprint(tamperedUnsigned);

    expect(() => parseOnnxModelManifest(manifest))
      .toThrow("totals do not match initializer inventory");
  });

  it("rejects architecture readiness drift and unsafe external paths", () => {
    const incomplete = unsigned();
    delete (incomplete.architecture as { headDimension?: number }).headDimension;
    expect(() => createOnnxModelManifest(incomplete))
      .toThrow("profile readiness does not match architecture evidence");

    const unsafe = unsigned();
    (unsafe.initializers[0].storage as { location: string }).location =
      "../weights.data";
    expect(() => createOnnxModelManifest(unsafe))
      .toThrow("must remain inside the model package");
  });

  it("resolves a capacity-preserving model profile with explicit assumptions", () => {
    const manifest = createOnnxModelManifest(unsigned());
    const profile = resolveOnnxModelProfile(manifest, {
      kvCacheQuantization: "fp8",
    });

    expect(profile).toMatchObject({
      name: "TinyCausalLM",
      architecture: {
        kind: "dense",
        numLayers: 1,
        hiddenDim: 2,
      },
      totalParams: 4,
      quantization: {
        weights: "fp32",
        kvCache: "fp8",
        activations: "fp16",
      },
      provenance: {
        evidence: "heuristic",
        source: `ONNX manifest ${manifest.manifestFingerprint}`,
      },
    });
    expect(profile.layers[0].attentionBytes + profile.layers[0].ffnBytes)
      .toBe(manifest.totals.initializerLogicalBytes);
    expect(profile.layers[0].kvCachePerToken).toBe(4);
  });

  it("requires explicit per-layer expert capacity before declaring MoE ready", () => {
    const input = unsigned();
    input.architecture = {
      ...input.architecture,
      numExperts: 8,
      activeExpertsPerToken: 2,
    };
    input.profileReadiness = {
      ready: false,
      missingFields: [
        "expertBytesPerLayer",
        "sharedExpertBytesPerLayer",
      ],
    };

    const manifest = createOnnxModelManifest(input);
    expect(manifest.profileReadiness).toEqual(input.profileReadiness);
    expect(() => resolveOnnxModelProfile(manifest))
      .toThrow("ONNX model profile is incomplete");
  });
});
