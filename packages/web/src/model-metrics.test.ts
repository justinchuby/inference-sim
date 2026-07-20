import { describe, expect, it } from "vitest";
import {
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  createOnnxModelManifest,
} from "@inference-sim/core";
import type { ImportedModelPackage } from "./model-package-import.js";
import {
  calculateIdealRoofline,
  summarizeModelPackage,
} from "./model-metrics.js";
import { createImportedModelBinding } from "./model-binding.js";

describe("model UI metrics", () => {
  it("keeps exact inventory separate from heuristic work and bandwidth bounds", () => {
    const modelPackage = packageWithDenseModel();
    const scenario = buildScenarioPreset("multi-gpu");
    const metrics = summarizeModelPackage(modelPackage, scenario);

    expect(metrics.packageBytes).toBe(12_000);
    expect(metrics.weightBytes).toBe(4_000);
    expect(metrics.parameterCount).toBe(2_000);
    expect(metrics.graphNodes).toBe(3);
    expect(metrics.forwardFlopsPerToken).toBe(4_000);
    expect(metrics.activeWeightBytesPerToken).toBe(4_000);
    expect(metrics.hotMemoryBandwidthBytesPerSec).toBe(6_000_000_000_000);
    expect(metrics.bandwidthCeilingTokensPerSec).toBe(1_500_000_000);
  });

  it("calculates an ideal hardware roofline without utilization factors", () => {
    const model = buildModelProfile("llama-3-70b");
    const topology = buildTopology("rtx-4090-2x");
    const roofline = calculateIdealRoofline(model, topology, 2);

    expect(roofline.aggregatePeakComputeFlops).toBe(330e12);
    expect(roofline.aggregateMemoryBandwidthBytesPerSec).toBe(2e12);
    expect(roofline.rooflineCeilingTokensPerSec).toBe(
      Math.min(
        roofline.computeCeilingTokensPerSec!,
        roofline.bandwidthCeilingTokensPerSec,
      ),
    );
  });

  it("does not invent aggregate token work for a multi-model pipeline", () => {
    const single = packageWithDenseModel();
    const multi: ImportedModelPackage = {
      ...single,
      models: [
        ...single.models,
        {
          ...single.models[0]!,
          fileName: "draft.onnx",
          componentIds: ["draft"],
        },
      ],
    };
    const metrics = summarizeModelPackage(
      multi,
      buildScenarioPreset("multi-gpu"),
    );

    expect(metrics.components).toHaveLength(2);
    expect(metrics.components.every(
      (component) => component.forwardFlopsPerToken === 4_000,
    )).toBe(true);
    expect(metrics.forwardFlopsPerToken).toBeUndefined();
    expect(metrics.bandwidthCeilingTokensPerSec).toBeUndefined();
  });

  it("turns a complete local target into executable dashboard work", () => {
    const binding = createImportedModelBinding(packageWithDenseModel());

    expect(binding).toMatchObject({
      source: "local_model_package",
      displayName: "test-dense",
      totalParameters: 2_000,
      weightBytes: 4_000,
      executionProfile: {
        modelName: "test-dense",
        forwardFlopsPerToken: 4_000,
      },
    });
    expect(
      binding.executionProfile.attentionWeightBytesPerToken
        + binding.executionProfile.ffnWeightBytesPerToken,
    ).toBe(4_000);
  });
});

function packageWithDenseModel(): ImportedModelPackage {
  const manifest = createOnnxModelManifest({
    kind: "inference-sim/onnx-model",
    revision: 2,
    source: {
      modelFileName: "decoder.onnx",
      modelByteLength: 8_000,
      sha256: "a".repeat(64),
    },
    model: {
      irVersion: "10",
      producerName: "test",
      producerVersion: "1",
      domain: "",
      modelVersion: "1",
    },
    graph: {
      name: "decoder",
      nodeCount: 3,
      initializerCount: 1,
      inputNames: ["input_ids"],
      outputNames: ["logits"],
      operators: [
        { domain: "ai.onnx", opType: "MatMul", count: 2 },
        { domain: "ai.onnx", opType: "Softmax", count: 1 },
      ],
    },
    initializers: [{
      name: "weight",
      dataType: "float16",
      dimensions: [1_000, 2],
      elementCount: 2_000,
      logicalByteLength: 4_000,
      storage: { kind: "inline", byteLength: 4_000 },
    }],
    externalDataFiles: [],
    architecture: {
      source: "inference_metadata",
      modelType: "test-dense",
      hiddenSize: 2,
      intermediateSize: 4,
      numHiddenLayers: 1,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDimension: 2,
      vocabSize: 8,
    },
    totals: {
      initializerElements: 2_000,
      initializerLogicalBytes: 4_000,
      inlineInitializerBytes: 4_000,
      externalInitializerBytes: 0,
    },
    profileReadiness: {
      ready: true,
      missingFields: [],
    },
  });
  return {
    metadata: {
      components: [],
      edges: [],
      stages: [],
      requiredCapabilities: [],
      hardware: {
        requiredDtypes: [],
        beneficialDtypes: [],
      },
      warnings: [],
      speculative: {
        availableFamilies: [],
        evidence: [],
        unrecognizedDeclarations: [],
      },
    },
    models: [{
      fileName: "decoder.onnx",
      componentIds: [],
      manifest,
    }],
    fileCount: 2,
    packageByteLength: 12_000,
    unboundOnnxFiles: ["decoder.onnx"],
  };
}
