import { describe, expect, it } from "vitest";
import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  createOnnxModelManifest,
  parseInferenceMetadata,
  simulateTopologyServingWorkload,
  listModelPresets,
  simulateTopologyWorkload,
  topologyProfileFromPipeline,
} from "@inference-sim/core";
import type { ImportedModelPackage } from "./model-package-import.js";
import {
  calculateIdealRoofline,
  summarizeModelPackage,
} from "./model-metrics.js";
import {
  DASHBOARD_MODEL_PRESETS,
  createBuiltinModelBinding,
  createImportedModelBinding,
} from "./model-binding.js";

describe("model UI metrics", () => {
  it("replaces the decoder with an iterative denoiser for image models", () => {
    const binding = createBuiltinModelBinding("flux-1-schnell");
    const pipeline = binding.pipelineExecution!;

    // No autoregressive target survives, so the pipeline owns the whole run.
    expect(pipeline.replacesTarget).toBe(true);
    expect(pipeline.strategyKind).toBe("iterative");
    expect(binding.executionProfile.kvCacheBytesPerToken ?? 0).toBe(0);

    const denoiser = pipeline.components.find((c) => c.isPrimary)!;
    expect(denoiser.role).toBe("denoiser");
    expect(denoiser.phase).toBe("every_step");
    // Timestep-distilled: four steps, no guidance doubling.
    expect(denoiser.invocationMultiplier).toBe(4);

    // Text towers condition once; the latent decoder runs at the end.
    const roles = pipeline.components.map((c) => `${c.role}:${c.phase}`);
    expect(roles).toContain("text_encoder:prompt_only");
    expect(roles).toContain("vae_decoder:final_only");
  });

  it("doubles denoiser invocations under classifier-free guidance", () => {
    const guided = createBuiltinModelBinding("stable-diffusion-3.5-large");
    const distilled = createBuiltinModelBinding("flux-1-dev");
    const primary = (b: typeof guided) => (
      b.pipelineExecution!.components.find((c) => c.isPrimary)!
    );

    // 40 steps with guidance runs the denoiser twice per step.
    expect(primary(guided).invocationMultiplier).toBe(80);
    // Guidance-distilled: 50 steps, one pass each.
    expect(primary(distilled).invocationMultiplier).toBe(50);
  });

  it("only runs media components when the run selects media", () => {
    const withImages = createBuiltinModelBinding(
      "qwen3-vl-8b", "fp16", "fp16", "image",
    );
    const textOnly = createBuiltinModelBinding(
      "qwen3-vl-8b", "fp16", "fp16", "text",
    );

    expect(withImages.pipelineExecution).toBeDefined();
    expect(withImages.pipelineExecution!.components.map((c) => c.role))
      .toContain("vision_encoder");
    // Text-only keeps the tower resident but never schedules it.
    expect(textOnly.pipelineExecution).toBeUndefined();
    expect(textOnly.weightBytes).toBe(withImages.weightBytes);
    // Both still report what each modality would cost.
    expect(textOnly.mediaInputs).toStrictEqual(withImages.mediaInputs);
    expect(textOnly.mediaInputs!.map((input) => input.modality))
      .toStrictEqual(["image", "video"]);
    // The two runs are distinct experiments.
    expect(textOnly.modelFingerprints).not.toEqual(withImages.modelFingerprints);
  });

  it("offers only the modalities a checkpoint accepts", () => {
    // Sourced per release: the encoder-free Gemma-4 takes audio, the MoE
    // sibling has no audio stack at all, and Whisper is audio only.
    const accepted = (preset: Parameters<typeof createBuiltinModelBinding>[0]) =>
      (createBuiltinModelBinding(preset).mediaInputs ?? [])
        .map((input) => input.modality);

    expect(accepted("gemma-4-12b")).toStrictEqual(["image", "audio", "video"]);
    expect(accepted("gemma-4-26b-a4b")).toStrictEqual(["image", "video"]);
    expect(accepted("whisper-large-v3")).toStrictEqual(["audio"]);
    expect(accepted("llama-3.2-11b-vision")).toStrictEqual(["image"]);
  });

  it("reports no media cost for a text-only model", () => {
    const binding = createBuiltinModelBinding("llama-3-8b");

    expect(binding.mediaInputs).toBeUndefined();
    expect(binding.pipelineExecution).toBeUndefined();
  });

  it("charges no decoder tokens for a cross-attending adapter", () => {
    // Llama-3.2-Vision cross-attends image features, so media costs encoder
    // work but adds no prompt positions.
    const binding = createBuiltinModelBinding(
      "llama-3.2-11b-vision", "fp16", "fp16", "image",
    );

    expect(binding.mediaInputs![0]!.decoderTokensPerItem).toBe(0);
    expect(binding.pipelineExecution).toBeDefined();
  });

  it("binds KV cache dtype independently of the weight dtype", () => {
    const fp16 = createBuiltinModelBinding("qwen3-0.6b", "fp16", "fp16");
    const fp8Kv = createBuiltinModelBinding("qwen3-0.6b", "fp16", "fp8");
    const int4Weights = createBuiltinModelBinding("qwen3-0.6b", "int4", "fp16");

    const fp16KvBytes = fp16.executionProfile.kvCacheBytesPerToken!;

    expect(fp16.modelFormat?.kvCacheDtype).toBe("fp16");
    expect(fp8Kv.modelFormat?.kvCacheDtype).toBe("fp8");
    expect(fp8Kv.executionProfile.kvCacheBytesPerToken).toBe(fp16KvBytes / 2);
    // Weight quantization must not move the KV cache figure, and vice versa.
    expect(int4Weights.executionProfile.kvCacheBytesPerToken)
      .toBe(fp16KvBytes);
    expect(fp8Kv.weightBytes).toBe(fp16.weightBytes);
    expect(fp8Kv.modelFingerprints).not.toEqual(fp16.modelFingerprints);
  });

  it("offers every built-in preset and binds each one", () => {
    expect([...DASHBOARD_MODEL_PRESETS].sort())
      .toEqual([...listModelPresets()].sort());

    const bindings = DASHBOARD_MODEL_PRESETS.map((preset) =>
      createBuiltinModelBinding(preset));
    for (const binding of bindings) {
      expect(binding.totalParameters).toBeGreaterThan(0);
      expect(binding.weightBytes).toBeGreaterThan(0);
      expect(binding.executionProfile.forwardFlopsPerToken).toBeGreaterThan(0);
      // A denoiser caches no KV; everything autoregressive must.
      const generatesImages =
        binding.pipelineExecution?.replacesTarget === true;
      expect(binding.executionProfile.kvCacheBytesPerToken ?? 0)
        .toBeGreaterThan(generatesImages ? -1 : 0);
    }
    expect(new Set(bindings.map((binding) => binding.displayName)).size)
      .toBe(bindings.length);
  });

  it("binds weight dtype to storage cost and experiment identity", () => {
    const fp16 = createBuiltinModelBinding("llama-3-8b", "fp16");
    const int4 = createBuiltinModelBinding("llama-3-8b", "int4");
    const int2 = createBuiltinModelBinding("llama-3-8b", "int2");
    const int1 = createBuiltinModelBinding("llama-3-8b", "int1");

    expect(fp16.modelFormat).toMatchObject({
      weightDtypes: ["fp16"],
      weightQuantization: "none",
      kvCacheDtype: "fp16",
      activationDtype: "fp16",
      evidence: "preset_declared",
    });
    expect(int4.modelFormat).toMatchObject({
      weightDtypes: ["int4"],
      weightQuantization: "int4",
    });
    expect(int4.weightBytes).toBe(fp16.weightBytes / 4);
    expect(int2.modelFormat).toMatchObject({
      weightDtypes: ["int2"],
      weightQuantization: "int2",
    });
    expect(int1.modelFormat).toMatchObject({
      weightDtypes: ["int1"],
      weightQuantization: "int1",
    });
    expect(int2.weightBytes).toBe(fp16.weightBytes / 8);
    expect(int1.weightBytes).toBe(fp16.weightBytes / 16);
    expect(int4.executionProfile.attentionWeightBytesPerToken).toBe(
      fp16.executionProfile.attentionWeightBytesPerToken / 4,
    );
    expect(new Set([
      fp16.targetModelFingerprint,
      int4.targetModelFingerprint,
      int2.targetModelFingerprint,
      int1.targetModelFingerprint,
    ]).size).toBe(4);
    expect(fp16.speculativeFamilies).toEqual(["prompt_lookup"]);
  });

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
    expect(metrics.components[0]?.weightDtypes).toEqual(["float16"]);
    expect(createImportedModelBinding(modelPackage).modelFormat).toMatchObject({
      weightDtypes: ["float16"],
      weightQuantization: "none",
      kvCacheDtype: "fp16",
      activationDtype: "fp16",
      evidence: "onnx_inferred",
      runtimeDtypesDefaulted: true,
    });
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

  it("binds and schedules the complete Gemma 4 VLM prompt pipeline", () => {
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
      fidelity: "complete",
      scope: "full_model",
      modeledComponentIds: ["decoder", "embedding", "vision_encoder"],
      unmodeledComponentIds: [],
    });
    expect(binding.weightBytes).toBe(12_000);
    expect(binding.pipelineExecution).toMatchObject({
      strategyKind: "composite",
      replacesTarget: false,
      components: [
        { id: "vision_encoder", phase: "prompt_only" },
        {
          id: "embedding",
          phase: "prompt_only",
          rerunEveryStep: true,
        },
        { id: "decoder", phase: "every_step", isPrimary: true },
      ],
    });
    const serving = simulateTopologyServingWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      {
        requests: [{
          id: "vlm",
          arrivalNs: 0,
          promptTokens: 32,
          outputTokens: 2,
        }],
        maxBatchSize: 1,
        maxBatchTokens: 8,
        prefillChunkTokens: 8,
        maxKvTokens: 64,
      },
      DEFAULT_TOPOLOGY_COST_MODEL,
      undefined,
      binding.executionProfile,
      binding.pipelineExecution,
    );
    const componentIds = serving.batches.flatMap((batch) => (
      batch.topology.plan.steps.flatMap((step) => (
        step.operation.kind === "compute" && step.operation.componentId
          ? [step.operation.componentId]
          : []
      ))
    ));
    expect(componentIds.filter(
      (componentId) => componentId === "vision_encoder",
    )).toHaveLength(1);
    expect(componentIds.filter(
      (componentId) => componentId === "embedding",
    )).toHaveLength(2);
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
          strategy: { kind: "autoregressive", decoder: "target" },
        },
      },
      components: [
        ["target.onnx", "target"],
        ["draft.onnx", "draft"],
      ],
    });

    const binding = createImportedModelBinding(modelPackage);

    expect(binding.speculativeFamilies).toContain("draft_model");
    expect(binding.executionCoverage.unmodeledComponentIds).toEqual([]);
    expect(binding.executionCoverage.limitations).toContain(
      "draft_model_profile_not_bound_to_proposer_cost",
    );
  });

  it("runs a pure any-to-any codec as an ordered one-shot pipeline", () => {
    const modelPackage = packageWithComponents({
      metadata: {
        pipeline: {
          models: {
            encoder: { filename: "encoder.onnx", type: "audio_encoder" },
            vocoder: { filename: "vocoder.onnx", type: "vocoder" },
          },
          phases: {
            encoder: { run_on: "prompt_only" },
            vocoder: { run_on: "prompt_only" },
          },
          dataflow: [{
            from: "encoder.codes",
            to: "vocoder.codes",
            device_transfer: false,
          }],
          strategy: {
            kind: "composite",
            stages: [
              {
                name: "encode",
                strategy: { kind: "single_pass", model: "encoder" },
              },
              {
                name: "decode",
                strategy: { kind: "single_pass", model: "vocoder" },
              },
            ],
          },
        },
      },
      components: [
        ["encoder.onnx", "encoder"],
        ["vocoder.onnx", "vocoder"],
      ],
    });
    const binding = createImportedModelBinding(modelPackage);
    expect(binding.pipelineExecution?.replacesTarget).toBe(true);
    const result = simulateTopologyWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      topologyProfileFromPipeline(binding.pipelineExecution!, 3),
    );
    const componentIds = result.plan.steps.flatMap((step) => (
      step.operation.kind === "compute" && step.operation.componentId
        ? [step.operation.componentId]
        : []
    ));
    expect(componentIds.filter(
      (componentId) => componentId === "encoder",
    )).toHaveLength(3);
    expect(componentIds.filter(
      (componentId) => componentId === "vocoder",
    )).toHaveLength(3);
    expect(result.metrics.committedTokens).toBe(3);
  });

  it("runs Whisper encoders before decode and TTS vocoders after completion", () => {
    const run = (finalRole: "none" | "vocoder") => {
      const models: Record<string, unknown> = {
        encoder: { filename: "encoder.onnx", type: "audio_encoder" },
        decoder: { filename: "decoder.onnx", type: "decoder" },
      };
      const phases: Record<string, unknown> = {
        encoder: { run_on: "prompt_only" },
        decoder: { run_on: "every_step" },
      };
      const dataflow: unknown[] = [{
        from: "encoder.encoder_hidden_states",
        to: "decoder.encoder_hidden_states",
      }];
      const components: Array<readonly [string, string]> = [
        ["encoder.onnx", "encoder"],
        ["decoder.onnx", "decoder"],
      ];
      if (finalRole === "vocoder") {
        models.vocoder = { filename: "vocoder.onnx", type: "vocoder" };
        phases.vocoder = { run_on: "final_only" };
        dataflow.push({
          from: "decoder.output_ids",
          to: "vocoder.codes",
        });
        components.push(["vocoder.onnx", "vocoder"]);
      }
      const binding = createImportedModelBinding(packageWithComponents({
        metadata: {
          pipeline: {
            models,
            phases,
            dataflow,
            strategy: {
              kind: "autoregressive",
              decoder: "decoder",
            },
          },
        },
        components,
      }));
      return simulateTopologyServingWorkload(
        buildScenarioPreset("single-gpu-cpu"),
        {
          requests: [{
            id: "audio",
            arrivalNs: 0,
            promptTokens: 24,
            outputTokens: 3,
          }],
          maxBatchSize: 1,
          maxBatchTokens: 8,
          prefillChunkTokens: 8,
          maxKvTokens: 64,
        },
        DEFAULT_TOPOLOGY_COST_MODEL,
        undefined,
        binding.executionProfile,
        binding.pipelineExecution,
      );
    };
    const whisperComponents = run("none").batches.flatMap(componentOperations);
    expect(whisperComponents.filter((id) => id === "encoder")).toHaveLength(1);
    const ttsComponents = run("vocoder").batches.flatMap(componentOperations);
    expect(ttsComponents.filter((id) => id === "encoder")).toHaveLength(1);
    expect(ttsComponents.filter((id) => id === "vocoder")).toHaveLength(1);
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

function componentOperations(
  batch: ReturnType<typeof simulateTopologyServingWorkload>["batches"][number],
): string[] {
  return batch.topology.plan.steps.flatMap((step) => (
    step.operation.kind === "compute" && step.operation.componentId
      ? [step.operation.componentId]
      : []
  ));
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
