import { describe, expect, it } from "vitest";
import {
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  createOnnxModelManifest,
  parseInferenceMetadata,
} from "@inference-sim/core";
import type { ImportedModelPackage } from "./model-package-import.js";
import {
  calculateIdealRoofline,
  summarizeModelPackage,
} from "./model-metrics.js";
import {
  createBuiltinModelBinding,
  createImportedModelBinding,
} from "./model-binding.js";

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
    expect(binding.executionCoverage).toMatchObject({
      fidelity: "complete",
      scope: "full_model",
      unmodeledComponentIds: [],
    });
  });

  it("reports Gemma 4 VLM prompt components as unmodeled execution", () => {
    const modelPackage = packageWithComponents({
      metadata: {
        pipeline: {
          models: {
            vision_encoder: {
              filename: "vision.onnx",
              type: "vision_encoder",
              device_preference: "cuda",
            },
            embedding: {
              filename: "embedding.onnx",
              type: "encoder",
            },
            decoder: {
              filename: "decoder.onnx",
              type: "decoder",
            },
          },
          phases: {
            vision_encoder: { run_on: "prompt_only" },
            embedding: { run_on: "prompt_only" },
            decoder: { run_on: "every_step" },
          },
          dataflow: [
            {
              from: "vision_encoder.image_features",
              to: "embedding.image_features",
              device_transfer: true,
            },
            {
              from: "embedding.inputs_embeds",
              to: "decoder.inputs_embeds",
              device_transfer: true,
            },
          ],
          strategy: {
            kind: "composite",
            stages: [
              {
                name: "vision",
                strategy: { kind: "single_pass", model: "vision_encoder" },
                run_on: "prompt_only",
              },
              {
                name: "fusion",
                strategy: { kind: "single_pass", model: "embedding" },
                run_on: "prompt_only",
              },
              {
                name: "decode",
                strategy: { kind: "autoregressive", decoder: "decoder" },
                run_on: "every_step",
              },
            ],
          },
        },
      },
      components: [
        ["vision.onnx", "vision_encoder"],
        ["embedding.onnx", "embedding"],
        ["decoder.onnx", "decoder"],
      ],
    });

    const binding = createImportedModelBinding(modelPackage);

    expect(binding.executionCoverage).toMatchObject({
      fidelity: "partial",
      scope: "target_component_only",
      modeledComponentIds: ["decoder"],
      unmodeledComponentIds: ["embedding", "vision_encoder"],
    });
    expect(binding.executionCoverage.limitations).toEqual(expect.arrayContaining([
      "non_target_components_not_scheduled",
      "pipeline_dataflow_transfers_not_scheduled",
      "component_device_preferences_not_enforced",
    ]));
  });

  it("reports an independent draft model as heuristic proposer work", () => {
    const modelPackage = packageWithComponents({
      metadata: {
        strategy: {
          kind: "speculative",
          draft: { producer: "draft_model", session: "draft" },
          tokens_per_step: 4,
        },
        pipeline: {
          models: {
            target: { filename: "target.onnx", type: "target" },
            draft: { filename: "draft.onnx", type: "draft" },
          },
        },
      },
      components: [
        ["target.onnx", "target"],
        ["draft.onnx", "draft"],
      ],
    });

    const binding = createImportedModelBinding(modelPackage);

    expect(binding.speculativeFamilies).toContain("draft_model");
    expect(binding.executionCoverage.unmodeledComponentIds).toEqual(["draft"]);
    expect(binding.executionCoverage.limitations).toContain(
      "draft_model_profile_not_bound_to_proposer_cost",
    );
  });

  it("does not claim model-bound EP for representative MoE presets", () => {
    for (const preset of ["mixtral-8x22b", "deepseek-v2"] as const) {
      const binding = createBuiltinModelBinding(preset);
      expect(binding.executionCoverage).toMatchObject({
        fidelity: "partial",
        scope: "full_model",
      });
      expect(binding.executionCoverage.limitations).toContain(
        "model_moe_routing_not_bound_to_expert_workload",
      );
    }
  });
});

function packageWithComponents({
  metadata,
  components,
}: {
  readonly metadata: unknown;
  readonly components: readonly (readonly [string, string])[];
}): ImportedModelPackage {
  const base = packageWithDenseModel();
  return {
    ...base,
    metadata: parseInferenceMetadata(metadata),
    models: components.map(([fileName, componentId]) => ({
      ...base.models[0]!,
      fileName,
      componentIds: [componentId],
    })),
    fileCount: components.length,
    unboundOnnxFiles: [],
  };
}

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
