import { describe, expect, it } from "vitest";
import {
  InferenceMetadataError,
  parseInferenceMetadata,
} from "../src/index.js";

describe("inference metadata", () => {
  it("normalizes a composite multi-model pipeline", () => {
    const parsed = parseInferenceMetadata({
      future_section: { accepted: true },
      hardware_requirements: {
        min_memory_gb: 24,
        required_dtypes: ["fp16"],
      },
      pipeline: {
        models: {
          encoder: {
            filename: "encoder.onnx",
            type: "encoder",
          },
          decoder: {
            filename: "decoder.onnx",
            type: "decoder",
            tokenizer: "tokenizer.json",
            device_preference: "cuda",
          },
        },
        dataflow: [{
          from: "encoder.hidden_states",
          to: "decoder.encoder_hidden_states",
          dtype: "fp32",
          device_transfer: false,
        }],
        strategy: {
          kind: "composite",
          stages: [
            {
              name: "encode",
              strategy: { kind: "single_pass", model: "encoder" },
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
    });

    expect(parsed.components.map((component) => component.id)).toEqual([
      "decoder",
      "encoder",
    ]);
    expect(parsed.components.find(
      (component) => component.id === "decoder",
    )?.runOn).toBeUndefined();
    expect(parsed.edges[0]).toMatchObject({
      fromComponent: "encoder",
      toComponent: "decoder",
      deviceTransfer: false,
    });
    expect(parsed.pipelineStrategy).toBe("composite");
    expect(parsed.stages.map((stage) => stage.name)).toEqual([
      "pipeline",
      "encode",
      "decode",
    ]);
    expect(parsed.stages[1]).toMatchObject({
      parentName: "pipeline",
      runOn: "prompt_only",
    });
    expect(parsed.hardware.minimumMemoryGiB).toBe(24);
    expect(parsed.speculative.availableFamilies).toEqual([]);
  });

  it("preserves iterative and nested loop bounds", () => {
    const parsed = parseInferenceMetadata({
      pipeline: {
        models: {
          denoiser: { filename: "denoiser.onnx", type: "denoiser" },
        },
        strategy: {
          kind: "iterative",
          denoiser: "denoiser",
          num_steps: 12,
          start_step: 3,
          max_tokens: 64,
          num_code_groups: 8,
        },
      },
    });
    expect(parsed.stages[0]).toMatchObject({
      kind: "iterative",
      numSteps: 12,
      startStep: 3,
      maxTokens: 64,
      numCodeGroups: 8,
    });
  });

  it("preserves component phase gates for execution planning", () => {
    const parsed = parseInferenceMetadata({
      pipeline: {
        models: {
          vision: { filename: "vision.onnx", type: "vision_encoder" },
          decoder: { filename: "decoder.onnx", type: "decoder" },
        },
        phases: {
          vision: { run_on: "prompt_only" },
          decoder: { run_on: "every_step" },
        },
        strategy: { kind: "autoregressive", decoder: "decoder" },
      },
    });

    expect(parsed.components).toMatchObject([
      { id: "decoder", runOn: "every_step" },
      { id: "vision", runOn: "prompt_only" },
    ]);
  });

  it("maps only explicit supported speculative evidence", () => {
    expect(parseInferenceMetadata({
      speculative: {
        proposal_type: "eagle-3",
        num_speculative_tokens: 5,
      },
    }).speculative).toMatchObject({
      availableFamilies: ["eagle3"],
      evidence: [{
        family: "eagle3",
        maximumDraftTokens: 5,
      }],
    });

    const unknown = parseInferenceMetadata({
      speculative: { proposal_type: "eagle" },
    });
    expect(unknown.speculative.availableFamilies).toEqual([]);
    expect(unknown.speculative.unrecognizedDeclarations).toEqual([
      "speculative.proposal_type=eagle",
    ]);
    expect(unknown.warnings).toHaveLength(1);
  });

  it("maps generic strategy producers and self-speculative depth", () => {
    const parsed = parseInferenceMetadata({
      strategy: {
        kind: "speculative",
        draft: { producer: "draft_model", session: "draft" },
        tokens_per_step: 4,
      },
      model: {
        speculative: { self_speculative_depth: 12 },
      },
    });
    expect(parsed.speculative.availableFamilies).toEqual([
      "draft_model",
      "self_speculative",
    ]);
  });

  it("does not infer a family from unclassified draft heads", () => {
    const parsed = parseInferenceMetadata({
      model: { speculative: { has_draft_heads: true } },
    });
    expect(parsed.speculative.availableFamilies).toEqual([]);
    expect(parsed.warnings[0]).toContain("without identifying");
  });

  it("rejects unsafe paths and dangling component references", () => {
    expect(() => parseInferenceMetadata({
      pipeline: {
        models: {
          decoder: { filename: "../decoder.onnx", type: "decoder" },
        },
        strategy: { kind: "autoregressive", decoder: "decoder" },
      },
    })).toThrow(InferenceMetadataError);
    expect(() => parseInferenceMetadata({
      pipeline: {
        models: {
          decoder: { filename: "decoder.onnx", type: "decoder" },
        },
        dataflow: [{
          from: "encoder.output",
          to: "decoder.input",
        }],
        strategy: { kind: "autoregressive", decoder: "decoder" },
      },
    })).toThrow("unknown component encoder");
  });

  it("rejects conflicting aliases and malformed known fields", () => {
    expect(() => parseInferenceMetadata({
      speculative: { proposal_type: "mtp" },
      speculator_config: { method: "mtp" },
    })).toThrow("cannot both be present");
    expect(() => parseInferenceMetadata({
      hardware_requirements: { required_dtypes: "fp16" },
    })).toThrow("must be an array");
    expect(() => parseInferenceMetadata({
      pipeline: {
        models: {
          encoder: { filename: "encoder.onnx", type: "encoder" },
        },
        phases: { encoder: { run_on: "sometimes" } },
      },
    })).toThrow("unsupported phase sometimes");
  });
});
