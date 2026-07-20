import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fromJson, toBinary } from "@bufbuild/protobuf";
import { ModelProtoSchema } from "onnx-buf";
import { describe, expect, it } from "vitest";
import { buildScenarioPreset } from "@inference-sim/core";
import { runCli, type CliIo } from "../src/main.js";

function captureIo(): {
  io: CliIo;
  stdout: () => string;
  stderr: () => string;
} {
  let stdout = "";
  let stderr = "";
  return {
    io: {
      stdout: (text) => {
        stdout += text;
      },
      stderr: (text) => {
        stderr += text;
      },
    },
    stdout: () => stdout,
    stderr: () => stderr,
  };
}

function tinyOnnxModel(
  externalLocation: string,
  externalLength: number,
): Uint8Array {
  const model = fromJson(ModelProtoSchema, {
    irVersion: "11",
    producerName: "inference-sim-test",
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
  });
  return toBinary(ModelProtoSchema, model);
}

describe("CLI", () => {
  it("lists scenario, hardware, and model presets", async () => {
    const capture = captureIo();
    expect(await runCli(["presets"], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenarios: string[];
      parameterizedScenario: string;
      customScenario: string;
      hardware: { topologies: string[] };
      models: string[];
    };
    expect(output.scenarios).toContain("gpu-npu");
    expect(output.scenarios).toContain("mac-mini-m4-pro-64gb");
    expect(output.scenarios).toContain("rtx-5090-desktop");
    expect(output.parameterizedScenario).toBe("multi-gpu-ring-<2..64>");
    expect(output.customScenario).toBe("<scenario.yaml|json>");
    expect(output.hardware.topologies).toContain("dgx-h100");
    expect(output.models).toContain("deepseek-v2");
  });

  it("prints a validated scenario and exact ledger", async () => {
    const capture = captureIo();
    expect(await runCli(["scenario", "unified-memory"], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenario: { family: string };
      memoryLedger: Array<{ domainId: string; reservedBytes: number }>;
    };
    expect(output.scenario.family).toBe("unified");
    expect(output.memoryLedger.find(
      (entry) => entry.domainId === "node0:unified",
    )?.reservedBytes).toBe(
      92 * 1024 ** 3 + 256 * 1024 ** 2,
    );
    expect(output.memoryLedger.find(
      (entry) => entry.domainId === "node0:storage",
    )?.reservedBytes).toBe(
      512 * 1024 ** 3,
    );
  });

  it("extracts a deterministic ONNX manifest with verified external data", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-onnx-"));
    const modelPath = join(directory, "model.onnx");
    const weightsPath = join(directory, "model.onnx.data");
    const metadataPath = join(directory, "manifest.json");
    await writeFile(modelPath, tinyOnnxModel("model.onnx.data", 16));
    await writeFile(weightsPath, new Uint8Array(16).fill(7));
    await writeFile(metadataPath, JSON.stringify({
      architecture: "TinyCausalLM",
      vocab_size: 8,
      hidden_size: 2,
      intermediate_size: 4,
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 2,
    }), "utf8");
    const first = captureIo();
    const second = captureIo();

    expect(await runCli(
      ["onnx-inspect", modelPath, metadataPath],
      first.io,
    )).toBe(0);
    expect(await runCli(
      ["onnx-inspect", modelPath, metadataPath],
      second.io,
    )).toBe(0);
    expect(first.stdout()).toBe(second.stdout());
    const manifest = JSON.parse(first.stdout()) as {
      kind: string;
      graph: { nodeCount: number; operators: unknown[] };
      totals: { externalInitializerBytes: number };
      externalDataFiles: Array<{
        location: string;
        referencedByteLength: number;
      }>;
      profileReadiness: { ready: boolean; missingFields: string[] };
    };
    expect(manifest).toMatchObject({
      kind: "inference-sim/onnx-model",
      graph: { nodeCount: 1 },
      totals: { externalInitializerBytes: 16 },
      profileReadiness: { ready: true, missingFields: [] },
    });
    expect(manifest.graph.operators).toHaveLength(1);
    expect(manifest.externalDataFiles).toEqual([
      expect.objectContaining({
        location: "model.onnx.data",
        referencedByteLength: 16,
      }),
    ]);
  });

  it("runs static analysis from an inspected ONNX package", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-onnx-static-"));
    const modelPath = join(directory, "model.onnx");
    const metadataPath = join(directory, "manifest.json");
    const configPath = join(directory, "config.json");
    await writeFile(modelPath, tinyOnnxModel("model.onnx.data", 16));
    await writeFile(
      join(directory, "model.onnx.data"),
      new Uint8Array(16).fill(3),
    );
    await writeFile(metadataPath, JSON.stringify({
      architecture: "TinyCausalLM",
      vocab_size: 8,
      hidden_size: 2,
      intermediate_size: 4,
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 2,
    }));
    await writeFile(configPath, JSON.stringify({
      hardware: { preset: "rtx-4090-2x" },
      quantization: { kv_cache: "fp8", activations: "fp16" },
      pipeline: {
        batch_size: 1,
        input_seq_len: 16,
        output_seq_len: 4,
        parallelism: {
          tensor_parallel: 1,
          pipeline_parallel: 1,
          expert_parallel: 1,
          data_parallel: 1,
        },
      },
      memory: {
        kv_cache_budget: 0.4,
        expert_cache_budget: 0,
        pinned_pool: 0.05,
        offload: "none",
        prefetch_ahead: 0,
        pressure_threshold: 0.9,
      },
    }));
    const capture = captureIo();

    expect(await runCli(
      ["onnx-static", configPath, modelPath, metadataPath],
      capture.io,
    )).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      manifest: {
        fingerprint: string;
        initializerLogicalBytes: number;
      };
      model: {
        quantization: { weights: string; kvCache: string };
        provenance: { evidence: string; source: string };
      };
      analysis: { feasible: boolean };
    };
    expect(output.manifest.initializerLogicalBytes).toBe(16);
    expect(output.model.quantization).toMatchObject({
      weights: "fp32",
      kvCache: "fp8",
    });
    expect(output.model.provenance).toMatchObject({
      evidence: "heuristic",
      source: `ONNX manifest ${output.manifest.fingerprint}`,
    });
    expect(output.analysis.feasible).toBe(true);
  });

  it("exhaustively ranks a bounded ONNX configuration space", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-onnx-search-"));
    const modelPath = join(directory, "model.onnx");
    const metadataPath = join(directory, "manifest.json");
    const configPath = join(directory, "search.json");
    await writeFile(modelPath, tinyOnnxModel("model.onnx.data", 16));
    await writeFile(
      join(directory, "model.onnx.data"),
      new Uint8Array(16).fill(5),
    );
    await writeFile(metadataPath, JSON.stringify({
      architecture: "TinyCausalLM",
      vocab_size: 8,
      hidden_size: 2,
      intermediate_size: 4,
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 2,
    }));
    await writeFile(configPath, JSON.stringify({
      search: {
        objective: "decode_throughput",
        top_k: 3,
        max_candidates: 16,
        require_feasible: true,
        maximum_device_used_fraction: 0.9,
      },
      hardware: { presets: ["dgx-h100", "rtx-4090-2x"] },
      quantization: {
        kv_cache: ["fp16", "fp8"],
        activations: ["fp16"],
      },
      pipeline: {
        batch_sizes: [1, 4],
        input_seq_lens: [128],
        output_seq_lens: [32],
        parallelism: {
          tensor_parallel: [1, 2],
          pipeline_parallel: [1],
          expert_parallel: [1],
          data_parallel: [1],
        },
      },
      memory: {
        kv_cache_budget: 0.3,
        expert_cache_budget: 0,
        pinned_pool: 0.05,
        offload: ["none"],
        prefetch_ahead: 0,
        pressure_threshold: 0.9,
      },
    }));
    const first = captureIo();
    const second = captureIo();

    expect(await runCli(
      ["onnx-search", configPath, modelPath, metadataPath],
      first.io,
    )).toBe(0);
    expect(await runCli(
      ["onnx-search", configPath, modelPath, metadataPath],
      second.io,
    )).toBe(0);
    expect(first.stdout()).toBe(second.stdout());
    const output = JSON.parse(first.stdout()) as {
      manifest: { fingerprint: string };
      search: {
        exhaustive: boolean;
        declaredCandidateCount: number;
        evaluatedCandidateCount: number;
        eligibleCandidateCount: number;
        candidates: Array<{
          rank: number;
          candidateId: string;
          feasible: boolean;
        }>;
      };
    };
    expect(output.manifest.fingerprint).toMatch(/^fnv1a32:/);
    expect(output.search).toMatchObject({
      exhaustive: true,
      declaredCandidateCount: 16,
      evaluatedCandidateCount: 16,
      eligibleCandidateCount: 16,
    });
    expect(output.search.candidates.map((candidate) => candidate.rank))
      .toEqual([1, 2, 3]);
    expect(output.search.candidates.every((candidate) => candidate.feasible))
      .toBe(true);
  });

  it("rejects unsafe or truncated ONNX external-data references", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-onnx-"));
    const modelPath = join(directory, "model.onnx");
    await writeFile(modelPath, tinyOnnxModel("../weights.data", 16));
    const unsafe = captureIo();
    expect(await runCli(["onnx-inspect", modelPath], unsafe.io)).toBe(1);
    expect(unsafe.stderr()).toContain("unsafe or missing external-data location");

    await writeFile(modelPath, tinyOnnxModel("weights.data", 16));
    await writeFile(join(directory, "weights.data"), new Uint8Array(8));
    const truncated = captureIo();
    expect(await runCli(["onnx-inspect", modelPath], truncated.io)).toBe(1);
    expect(truncated.stderr()).toContain("external-data range exceeds");
  });

  it("materializes a parameterized multi-GPU scenario target", async () => {
    const capture = captureIo();

    expect(await runCli(
      ["scenario", "multi-gpu-ring-4"],
      capture.io,
    )).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenario: {
        id: string;
        devices: unknown[];
        groups: Array<{ id: string; orderedRanks: unknown[] }>;
      };
      memoryLedger: Array<{ domainId: string }>;
    };
    expect(output.scenario.id).toBe("multi-gpu-ring-4");
    expect(output.scenario.devices).toHaveLength(5);
    expect(output.scenario.groups.find(
      (group) => group.id === "tp",
    )?.orderedRanks).toHaveLength(4);
    expect(output.memoryLedger.filter(
      (entry) => entry.domainId.endsWith(":vram"),
    )).toHaveLength(4);

    const rejected = captureIo();
    expect(await runCli(
      ["scenario", "multi-gpu-ring-65"],
      rejected.io,
    )).toBe(1);
    expect(rejected.stderr()).toContain(
      "multi-GPU ring count must be a safe integer from 2 through 64",
    );
  });

  it("runs a strictly parsed custom device scenario across CLI surfaces", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-custom-"));
    const scenarioPath = join(directory, "scenario.json");
    const workloadPath = join(directory, "target-only.yaml");
    const scenario = {
      ...buildScenarioPreset("multi-gpu"),
      id: "custom-dual-gpu",
      family: "custom" as const,
    };
    await writeFile(scenarioPath, JSON.stringify(scenario), "utf8");
    await writeFile(workloadPath, `
target_only:
  token_count: 3
`, "utf8");

    const materialized = captureIo();
    expect(await runCli(
      ["scenario", scenarioPath],
      materialized.io,
    )).toBe(0);
    const scenarioOutput = JSON.parse(materialized.stdout()) as {
      scenario: { id: string; family: string };
      memoryLedger: unknown[];
    };
    expect(scenarioOutput.scenario).toMatchObject({
      id: "custom-dual-gpu",
      family: "custom",
    });
    expect(scenarioOutput.memoryLedger.length).toBeGreaterThan(0);

    const run = captureIo();
    expect(await runCli(
      ["run", scenarioPath, workloadPath],
      run.io,
    )).toBe(0);
    expect(JSON.parse(run.stdout())).toMatchObject({
      scenarioId: "custom-dual-gpu",
      status: "succeeded",
      metrics: { committedTokens: 3 },
    });

    const exported = captureIo();
    expect(await runCli(
      ["plan-export", scenarioPath, workloadPath],
      exported.io,
    )).toBe(0);
    const artifact = JSON.parse(exported.stdout()) as {
      scenario: { id: string; family: string };
      plan: { steps: unknown[] };
    };
    expect(artifact.scenario).toMatchObject({
      id: "custom-dual-gpu",
      family: "custom",
    });
    expect(artifact.plan.steps.length).toBeGreaterThan(0);
  });

  it("rejects unknown fields and enum values in custom scenarios", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-custom-"));
    const scenario = buildScenarioPreset("single-gpu-cpu");
    const unknownPath = join(directory, "unknown.json");
    const unknown = structuredClone(scenario) as unknown as Record<
      string,
      unknown
    >;
    unknown.implicitRuntimeDefault = true;
    await writeFile(unknownPath, JSON.stringify(unknown), "utf8");
    const unknownCapture = captureIo();

    expect(await runCli(["scenario", unknownPath], unknownCapture.io)).toBe(1);
    expect(unknownCapture.stderr()).toContain(
      "unknown fields implicitRuntimeDefault",
    );
    const validationCapture = captureIo();
    expect(await runCli(["validate", unknownPath], validationCapture.io))
      .toBe(2);
    expect(JSON.parse(validationCapture.stdout())).toMatchObject({
      valid: false,
      issues: [{
        code: "scenario_boundary",
        message: expect.stringContaining(
          "unknown fields implicitRuntimeDefault",
        ),
      }],
    });

    const invalidPath = join(directory, "invalid.json");
    const invalid = {
      ...scenario,
      family: "single_accelerator_maybe",
    };
    await writeFile(invalidPath, JSON.stringify(invalid), "utf8");
    const invalidCapture = captureIo();

    expect(await runCli(["scenario", invalidPath], invalidCapture.io)).toBe(1);
    expect(invalidCapture.stderr()).toContain("family: must be one of");
  });

  it("runs a speculative YAML workload deterministically", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "speculative.yaml");
    await writeFile(path, `
speculative:
  family: mtp
  initial_token_length: 10
  output_token_count: 6
  max_additional_tokens: 2
  acceptance:
    kind: replay
    accepted_draft_tokens: [2, 1]
  paged_kv:
    page_size_tokens: 4
    bytes_per_token: 128
    capacity_bytes: 4096
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["speculative", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      finalTokenLength: number;
      metrics: { committedTokens: number };
      pagedKv: {
        snapshot: {
          logicalTokenLength: number;
          reservedBytes: number;
        };
      };
    };
    expect(output.finalTokenLength).toBe(16);
    expect(output.metrics.committedTokens).toBe(6);
    expect(output.pagedKv.snapshot.logicalTokenLength).toBe(16);
    expect(output.pagedKv.snapshot.reservedBytes).toBe(2048);
  });

  it("verifies token values and optionally executes their topology", async () => {
    const path = new URL(
      "../../../examples/speculative-token-trace-mtp.yaml",
      import.meta.url,
    ).pathname;
    const capture = captureIo();

    expect(await runCli([
      "speculative-trace",
      path,
      "single-gpu-cpu",
    ], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      trace: {
        differential: { matchesTargetOnly: boolean };
        committedOutputTokenIds: number[];
      };
      topology: {
        execution: { status: string };
        metrics: { committedTokens: number };
      };
    };
    expect(output.trace.differential.matchesTargetOnly).toBe(true);
    expect(output.trace.committedOutputTokenIds)
      .toEqual([10, 20, 21, 30, 31, 32, 40, 41]);
    expect(output.topology.execution.status).toBe("succeeded");
    expect(output.topology.metrics.committedTokens).toBe(8);
  });

  it("returns status 2 with the first token differential", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "mismatch.yaml");
    await writeFile(path, `
speculative_token_trace:
  revision: 2
  id: mismatch
  provenance:
    source: synthetic-test
    runtime_revision: onnx-genai-test
    model_fingerprint: target-model-test
    proposer_fingerprint: draft-model-test
    tokenizer_fingerprint: tokenizer-test
    generation_config_fingerprint: greedy-test
    target_only_run_id: target-only-test
    speculative_run_id: speculative-test
  family: draft_model
  prompt_token_ids: [1]
  expected_output_token_ids: [7]
  max_additional_tokens: 1
  iterations:
    - id: tail
      proposal_token_ids: [8]
      target_token_ids: [8]
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["speculative-trace", path], capture.io)).toBe(2);
    const output = JSON.parse(capture.stdout()) as {
      differential: {
        matchesTargetOnly: boolean;
        firstMismatch: {
          outputIndex: number;
          expectedTokenId: number;
          actualTokenId: number;
        };
      };
    };
    expect(output.differential).toEqual({
      matchesTargetOnly: false,
      comparedTokenCount: 1,
      firstMismatch: {
        outputIndex: 0,
        expectedTokenId: 7,
        actualTokenId: 8,
      },
    });
  });

  it("binds independent runtime captures and executes their topology", async () => {
    const targetPath = new URL(
      "../../../examples/runtime-capture-target-only.yaml",
      import.meta.url,
    ).pathname;
    const speculativePath = new URL(
      "../../../examples/runtime-capture-speculative.yaml",
      import.meta.url,
    ).pathname;
    const capture = captureIo();

    expect(await runCli([
      "speculative-capture",
      targetPath,
      speculativePath,
      "single-gpu-cpu",
    ], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      targetOnlyCaptureId: string;
      speculativeCaptureId: string;
      trace: {
        differential: { matchesTargetOnly: boolean };
        iterations: Array<{ outcome: string }>;
      };
      topology: { execution: { status: string } };
    };
    expect(output.targetOnlyCaptureId).toBe("target-only-synthetic-001");
    expect(output.speculativeCaptureId).toBe("speculative-synthetic-001");
    expect(output.trace.differential.matchesTargetOnly).toBe(true);
    expect(output.trace.iterations.map((iteration) => iteration.outcome))
      .toEqual(["correction", "bonus", "accepted_tail"]);
    expect(output.topology.execution.status).toBe("succeeded");
  });

  it("returns a nonzero status with a concise error", async () => {
    const capture = captureIo();
    expect(await runCli(["scenario", "does-not-exist"], capture.io)).toBe(1);
    expect(capture.stderr()).toContain("unknown scenario preset");
  });

  it("runs an exact-capacity expert-cache workload", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "expert-cache.yaml");
    await writeFile(path, `
expert_cache:
  placement_strategy: round_robin
  hot_capacity_bytes: 144
  warm_capacity_bytes: 144
  warm_to_hot_latency_ns: 5
  cold_to_hot_latency_ns: 20
  cold_to_warm_latency_ns: 12
  routing_seed: 7
  initial_hot_expert_ids: [e0]
  experts:
    - { id: e0, bytes: 48 }
    - { id: e1, bytes: 64 }
    - { id: e2, bytes: 80 }
workload:
  token_count: 3
  top_k: 2
  token_interval_ns: 10
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["expert-cache", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      routes: unknown[];
      snapshot: {
        metrics: { routes: number };
        hotResidentBytes: number;
      };
    };
    expect(output.routes).toHaveLength(3);
    expect(output.snapshot.metrics.routes).toBe(3);
    expect(output.snapshot.hotResidentBytes).toBeLessThanOrEqual(144);

    const topologyCapture = captureIo();
    expect(await runCli(
      ["run", "multi-gpu", path],
      topologyCapture.io,
    )).toBe(0);
    const topology = JSON.parse(topologyCapture.stdout()) as {
      operationCounts: {
        collective: number;
        allReduce: number;
        allToAll: number;
      };
      assumptions: string[];
    };
    expect(topology.operationCounts.allReduce).toBe(3);
    expect(topology.operationCounts.allToAll).toBe(6);
    expect(topology.operationCounts.collective).toBe(9);
    expect(topology.assumptions.some(
      (assumption) => assumption.includes("explicit round_robin owner mapping"),
    )).toBe(true);

    const comparisonCapture = captureIo();
    expect(await runCli(["compare", path], comparisonCapture.io)).toBe(0);
    const comparison = JSON.parse(comparisonCapture.stdout()) as {
      comparison: Array<{
        scenarioId: string;
        rank: number;
        relativeToFastest: number;
      }>;
    };
    expect(comparison.comparison).toHaveLength(6);
    expect(comparison.comparison[0].relativeToFastest).toBe(1);
    expect(new Set(comparison.comparison.map((entry) => entry.scenarioId)).size)
      .toBe(6);
  });

  it("runs continuous serving through a selected topology", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 4
  prefill_chunk_tokens: 2
  max_kv_tokens: 20
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 3 }
    - { id: b, arrival_ns: 10, prompt_tokens: 2, output_tokens: 2, priority: 1 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(
      ["serving", "multi-gpu-ring-4", path],
      capture.io,
    )).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenarioId: string;
      serving: {
        metrics: { requests: number; outputTokens: number };
        replay: { completedRequests: number };
      };
      batches: unknown[];
    };
    expect(output.scenarioId).toBe("multi-gpu-ring-4");
    expect(output.serving.metrics.requests).toBe(2);
    expect(output.serving.metrics.outputTokens).toBe(5);
    expect(output.serving.replay.completedRequests).toBe(2);
    expect(output.batches.length).toBeGreaterThan(1);
  });

  it("runs speculative continuous serving with request-scoped acceptance", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving-speculative.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 8
  prefill_chunk_tokens: 4
  max_kv_tokens: 24
  speculative:
    family: mtp
    max_additional_tokens: 2
    acceptance:
      kind: replay
      accepted_draft_tokens_by_request:
        a: [2]
        b: [0, 1]
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 5 }
    - { id: b, arrival_ns: 10, prompt_tokens: 4, output_tokens: 5 }
expert_cache:
  contract_revision: 3
  top_k: 1
  hot_capacity_bytes: 67108864
  warm_capacity_bytes: 67108864
  warm_to_hot_latency_ns: 400000
  cold_to_hot_latency_ns: 2200000
  cold_to_warm_latency_ns: 1500000
  routing_seed: 7
  experts:
    - { id: e0, bytes: 67108864 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["serving", "multi-gpu", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      serving: {
        metrics: {
          outputTokens: number;
          proposedDraftTokens: number;
          acceptedDraftTokens: number;
          targetForwards: number;
        };
        replay: { completedRequests: number; finalKvTokens: number };
      };
      batches: Array<{
        work: {
          decode: Array<{ mode: string; outcome: string }>;
        };
      }>;
      expertCache: {
        contractRevision: number;
        metrics: { coldMisses: number; hotHits: number };
        routes: number;
        replayAppliedEvents: number;
      };
      physical: {
        maximumConcurrentExecutions: number;
        replayAppliedEvents: number;
      };
    };
    expect(output.serving.metrics).toMatchObject({
      outputTokens: 10,
      proposedDraftTokens: 8,
      acceptedDraftTokens: 6,
      targetForwards: 3,
    });
    expect(output.serving.replay).toMatchObject({
      completedRequests: 2,
      finalKvTokens: 0,
    });
    expect(output.batches.some((batch) => (
      batch.work.decode.some((decode) => decode.mode === "speculative")
    ))).toBe(true);
    expect(output.expertCache.contractRevision).toBe(3);
    expect(output.expertCache.metrics.coldMisses).toBe(1);
    expect(output.expertCache.metrics.hotHits)
      .toBe(output.expertCache.routes - 1);
    expect(output.expertCache.replayAppliedEvents).toBeGreaterThan(0);
    expect(output.physical.maximumConcurrentExecutions).toBeGreaterThan(0);
    expect(output.physical.replayAppliedEvents).toBeGreaterThan(0);
  });

  it("compares one serving workload across all topology presets", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving-compare.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 8
  prefill_chunk_tokens: 4
  max_kv_tokens: 24
  speculative:
    family: mtp
    max_additional_tokens: 2
    acceptance:
      kind: conditional_empirical
      match_probability_by_position: [0.8, 0.6]
      seed: 42
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 5 }
    - { id: b, arrival_ns: 10, prompt_tokens: 4, output_tokens: 5 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["serving-compare", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      comparison: Array<{
        rank: number;
        scenarioId: string;
        relativeToFastest: number;
        replayAppliedEvents: number;
      }>;
    };
    expect(output.comparison).toHaveLength(6);
    expect(output.comparison.map((entry) => entry.rank))
      .toEqual([1, 2, 3, 4, 5, 6]);
    expect(output.comparison[0]).toMatchObject({
      scenarioId: "multi-gpu",
      relativeToFastest: 1,
    });
    expect(output.comparison.at(-1)?.scenarioId).toBe("cpu-only");
    expect(output.comparison.every((entry) => (
      entry.replayAppliedEvents > 0
    ))).toBe(true);
  });

  it("fits and imports a scoped calibration dataset", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const calibrationPath = join(directory, "calibration.json");
    const servingPath = join(directory, "serving.yaml");
    await writeFile(
      calibrationPath,
      JSON.stringify(calibrationConfig(["multi-gpu"])),
      "utf8",
    );
    await writeFile(servingPath, `
serving:
  max_batch_size: 1
  max_batch_tokens: 4
  prefill_chunk_tokens: 2
  max_kv_tokens: 12
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 2, output_tokens: 2 }
`, "utf8");

    const fitCapture = captureIo();
    expect(await runCli(
      ["calibrate", calibrationPath],
      fitCapture.io,
    )).toBe(0);
    const fit = JSON.parse(fitCapture.stdout()) as {
      confidence: string;
      datasetFingerprint: string;
      diagnostics: unknown[];
    };
    expect(fit.confidence).toBe("heuristic");
    expect(fit.datasetFingerprint).toMatch(/^fnv1a32:/);
    expect(fit.diagnostics).toHaveLength(15);

    const runCapture = captureIo();
    expect(await runCli(
      ["serving", "multi-gpu", servingPath, calibrationPath],
      runCapture.io,
    )).toBe(0);
    const run = JSON.parse(runCapture.stdout()) as {
      confidence: string;
      assumptions: string[];
    };
    expect(run.confidence).toBe("heuristic");
    expect(run.assumptions[0]).toContain("cli-calibration-fixture");

    const rejectedCapture = captureIo();
    expect(await runCli(
      ["serving", "cpu-only", servingPath, calibrationPath],
      rejectedCapture.io,
    )).toBe(1);
    expect(rejectedCapture.stderr()).toContain(
      "cost model is not applicable to scenario cpu-only",
    );
  });

  it("runs a workload through topology resources", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "target-only.yaml");
    await writeFile(path, `
target_only:
  token_count: 8
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["run", "multi-gpu", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      status: string;
      operationCounts: { collective: number };
      metrics: { committedTokens: number; totalDurationNs: number };
    };
    expect(output.status).toBe("succeeded");
    expect(output.operationCounts.collective).toBe(8);
    expect(output.metrics.committedTokens).toBe(8);
    expect(output.metrics.totalDurationNs).toBeGreaterThan(0);

    const eightGpuCapture = captureIo();
    expect(await runCli(
      ["run", "multi-gpu-ring-8", path],
      eightGpuCapture.io,
    )).toBe(0);
    const eightGpu = JSON.parse(eightGpuCapture.stdout()) as {
      scenarioId: string;
      status: string;
      operationCounts: { collective: number };
    };
    expect(eightGpu).toMatchObject({
      scenarioId: "multi-gpu-ring-8",
      status: "succeeded",
      operationCounts: { collective: 8 },
    });
  });

  it("exports and replays a self-contained FrozenPlan artifact", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const workloadPath = join(directory, "target-only.yaml");
    const artifactPath = join(directory, "plan.json");
    await writeFile(workloadPath, `
target_only:
  token_count: 4
`, "utf8");
    const firstExport = captureIo();
    const secondExport = captureIo();

    expect(await runCli(
      ["plan-export", "multi-gpu-ring-4", workloadPath],
      firstExport.io,
    )).toBe(0);
    expect(await runCli(
      ["plan-export", "multi-gpu-ring-4", workloadPath],
      secondExport.io,
    )).toBe(0);
    expect(firstExport.stdout()).toBe(secondExport.stdout());

    const artifact = JSON.parse(firstExport.stdout()) as {
      kind: string;
      scenario: { id: string };
      plan: { steps: unknown[] };
    };
    expect(artifact.kind).toBe("inference-sim/frozen-plan");
    expect(artifact.scenario.id).toBe("multi-gpu-ring-4");
    expect(artifact.plan.steps.length).toBeGreaterThan(0);

    await writeFile(artifactPath, firstExport.stdout(), "utf8");
    const run = captureIo();
    expect(await runCli(["plan-run", artifactPath], run.io)).toBe(0);
    const result = JSON.parse(run.stdout()) as {
      scenarioId: string;
      execution: {
        status: string;
        completedAtNs: number;
        trace: { operations: unknown[] };
      };
      replay: {
        status: string;
        completedAtNs: number;
        appliedEvents: number;
      };
    };
    expect(result.scenarioId).toBe("multi-gpu-ring-4");
    expect(result.execution.status).toBe("succeeded");
    expect(result.replay).toMatchObject({
      status: "succeeded",
      completedAtNs: result.execution.completedAtNs,
      appliedEvents: result.execution.trace.operations.length + 1,
    });
  });

  it("rejects a tampered FrozenPlan artifact before execution", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const workloadPath = join(directory, "target-only.yaml");
    const artifactPath = join(directory, "tampered-plan.json");
    await writeFile(workloadPath, `
target_only:
  token_count: 1
`, "utf8");
    const exported = captureIo();
    expect(await runCli(
      ["plan-export", "multi-gpu", workloadPath],
      exported.io,
    )).toBe(0);
    const artifact = JSON.parse(exported.stdout()) as {
      plan: { steps: Array<{ operation: { durationNs: number } }> };
    };
    artifact.plan.steps[0].operation.durationNs++;
    await writeFile(artifactPath, JSON.stringify(artifact), "utf8");

    const run = captureIo();
    expect(await runCli(["plan-run", artifactPath], run.io)).toBe(1);
    expect(run.stderr()).toContain("plan fingerprint mismatch");
    expect(run.stdout()).toBe("");
  });

  it("compares one workload across all topology presets", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "target-only.yaml");
    await writeFile(path, `
target_only:
  token_count: 8
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["compare", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      comparison: Array<{ scenarioId: string; rank: number }>;
    };
    expect(output.comparison).toHaveLength(6);
    expect(output.comparison[0].scenarioId).toBe("multi-gpu");
    expect(
      output.comparison.find((entry) => entry.scenarioId === "cpu-only")?.rank,
    ).toBe(6);
  });

  it("runs and independently replays a deterministic fault campaign", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "fault-workload.yaml");
    await writeFile(path, `
target_only:
  token_count: 2
`, "utf8");
    const firstCapture = captureIo();
    const secondCapture = captureIo();

    expect(
      await runCli(
        ["fault-campaign", "multi-gpu-ring-4", path],
        firstCapture.io,
      ),
    ).toBe(0);
    expect(
      await runCli(
        ["fault-campaign", "multi-gpu-ring-4", path],
        secondCapture.io,
      ),
    ).toBe(0);
    expect(firstCapture.stdout()).toBe(secondCapture.stdout());
    const output = JSON.parse(firstCapture.stdout()) as {
      baseline: { status: string; submittedSteps: number };
      cases: Array<{
        id: string;
        status: string;
        replayAppliedEvents: number;
      }>;
    };
    expect(output.baseline.status).toBe("succeeded");
    expect(output.baseline.submittedSteps).toBeGreaterThan(0);
    expect(output.cases.map((entry) => entry.id)).toEqual([
      "node:node0",
      "device:node0:gpu0",
      "device:node0:gpu1",
      "device:node0:gpu2",
      "device:node0:gpu3",
      "link:node0:nvlink01:forward",
      "link:node0:nvlink12:forward",
      "link:node0:nvlink23:forward",
      "link:node0:nvlink30:forward",
      "epoch:1",
    ]);
    expect(output.cases.every((entry) => (
      entry.status !== "succeeded" && entry.replayAppliedEvents > 0
    ))).toBe(true);
  });

  it("runs a seeded concurrent campaign over shared plan resources", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "concurrent-workload.yaml");
    await writeFile(path, `
concurrent_campaign:
  execution_count: 12
  seed: 24301
  arrival_window_ns: 100000
target_only:
  token_count: 2
`, "utf8");
    const firstCapture = captureIo();
    const secondCapture = captureIo();

    expect(
      await runCli(
        ["concurrent-campaign", "multi-gpu", path],
        firstCapture.io,
      ),
    ).toBe(0);
    expect(
      await runCli(
        ["concurrent-campaign", "multi-gpu", path],
        secondCapture.io,
      ),
    ).toBe(0);
    expect(firstCapture.stdout()).toBe(secondCapture.stdout());
    const output = JSON.parse(firstCapture.stdout()) as {
      executionCount: number;
      maximumConcurrentExecutions: number;
      submittedOperations: number;
      replayAppliedEvents: number;
      replayExecutions: number;
      assumptions: readonly string[];
      latencyNs: { minimum: number; p95: number; maximum: number };
    };
    expect(output.executionCount).toBe(12);
    expect(output.maximumConcurrentExecutions).toBeGreaterThan(1);
    expect(output.submittedOperations).toBeGreaterThan(12);
    expect(output.replayAppliedEvents).toBe(
      output.submittedOperations + output.executionCount * 2,
    );
    expect(output.replayExecutions).toBe(12);
    expect(output.assumptions).toContain(
      "physical allocation ids remain shared across executions; conflicting writes are lease-serialized",
    );
    expect(output.latencyNs.minimum).toBeLessThanOrEqual(
      output.latencyNs.p95,
    );
    expect(output.latencyNs.p95).toBeLessThanOrEqual(
      output.latencyNs.maximum,
    );
  });

  it("fails over a node only after old-epoch quiescence", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "node-failover.yaml");
    await writeFile(path, `
node_failover:
  failed_node_id: node1
  fault_at_ns: 1
  reason: node1 heartbeat expired
target_only:
  token_count: 2
`, "utf8");
    const capture = captureIo();

    expect(await runCli([
      "node-failover",
      "multi-node",
      "single-gpu-cpu",
      path,
    ], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      completedAtNs: number;
      handoff: {
        oldTopologyEpoch: number;
        newTopologyEpoch: number;
        oldExecutionQuiescedAtNs: number;
        recoveryAdmittedAtNs: number;
      };
      failedExecution: {
        status: string;
        completedAtNs: number;
        replayAppliedEvents: number;
      };
      recoveryExecution: {
        status: string;
        completedAtNs: number;
        replayAppliedEvents: number;
      };
    };
    expect(output.failedExecution.status).toBe("failed");
    expect(output.recoveryExecution.status).toBe("succeeded");
    expect(output.handoff).toMatchObject({
      oldTopologyEpoch: 0,
      newTopologyEpoch: 1,
      oldExecutionQuiescedAtNs: output.failedExecution.completedAtNs,
      recoveryAdmittedAtNs: output.failedExecution.completedAtNs,
    });
    expect(output.completedAtNs).toBe(
      output.failedExecution.completedAtNs
        + output.recoveryExecution.completedAtNs,
    );
    expect(output.failedExecution.replayAppliedEvents).toBeGreaterThan(0);
    expect(output.recoveryExecution.replayAppliedEvents).toBeGreaterThan(0);
  });

  it("fans a node failure out across a concurrent campaign", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "concurrent-node-failure.yaml");
    await writeFile(path, `
concurrent_campaign:
  execution_count: 8
  seed: 24301
  arrival_window_ns: 0
node_failure:
  node_id: node1
  at_ns: 1
  reason: node1 heartbeat expired
target_only:
  token_count: 2
`, "utf8");
    const capture = captureIo();

    expect(await runCli([
      "concurrent-node-failure",
      "multi-node",
      path,
    ], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      executionCount: number;
      failedExecutions: number;
      submittedOperations: number;
      replayAppliedEvents: number;
      fault: { kind: string; nodeId: string; atNs: number };
      terminalsPreview: Array<{
        status: string;
        failureAtNs: number;
      }>;
    };
    expect(output.executionCount).toBe(8);
    expect(output.failedExecutions).toBe(8);
    expect(output.submittedOperations).toBeGreaterThan(0);
    expect(output.replayAppliedEvents).toBe(
      output.executionCount * 2 + output.submittedOperations,
    );
    expect(output.fault).toEqual({
      kind: "node_failure",
      nodeId: "node1",
      atNs: 1,
      reason: "node1 heartbeat expired",
    });
    expect(output.terminalsPreview).toHaveLength(8);
    expect(output.terminalsPreview.every(
      (terminal) => (
        terminal.status === "failed" && terminal.failureAtNs === 1
      ),
    )).toBe(true);
  });
});

function calibrationConfig(scenarioIds: readonly string[]) {
  const costs = {
    cpu: {
      invocation: 400_000,
      attention: 220_000,
      ffn: 300_000,
      draft: 110_000,
      lookup: 8_000,
    },
    gpu: {
      invocation: 120_000,
      attention: 28_000,
      ffn: 38_000,
      draft: 17_000,
      lookup: 8_000,
    },
    npu: {
      invocation: 100_000,
      attention: 22_000,
      ffn: 52_000,
      draft: 24_000,
      lookup: 8_000,
    },
  } as const;
  const observations = Object.entries(costs).flatMap(([deviceKind, device]) => [
    {
      id: `${deviceKind}-invocation-0`,
      device_kind: deviceKind,
      capability: "invocation",
      work_items: 0,
      durations_ns: [device.invocation - 1, device.invocation, device.invocation + 1],
      regime: "test no-op",
    },
    ...(["attention", "ffn", "draft", "lookup"] as const).flatMap(
      (capability) => [1, 8].map((workItems) => {
        const center = device.invocation + device[capability] * workItems;
        return {
          id: `${deviceKind}-${capability}-${workItems}`,
          device_kind: deviceKind,
          capability,
          work_items: workItems,
          durations_ns: [center - 1, center, center + 1],
          regime: "test batch",
        };
      }),
    ),
  ]);
  return {
    calibration: {
      revision: 3,
      id: "cli-calibration-fixture",
      provenance: {
        kind: "synthetic",
        source: "CLI test fixture",
        software_stack: "test stack",
        model_artifact: "test model",
      },
      applicability: {
        scenario_ids: scenarioIds,
        device_kind_labels: {
          cpu: "test CPU",
          gpu: "test GPU",
          npu: "test NPU",
        },
      },
      model_constants: {
        activation_bytes_per_token: 1_048_576,
        collective_bytes_per_token: 524_288,
        cold_load_byte_multiplier: 2,
      },
      quality: {
        min_samples_per_point: 3,
        max_normalized_rmse: 0.01,
        max_p95_relative_error: 0.01,
      },
      observations,
      transport_observations: [
        {
          id: "multi-gpu-collective-small",
          scenario_id: "multi-gpu",
          operation: "collective",
          link_ids: ["node0:nvlink:forward", "node0:nvlink:reverse"],
          participant_count: 2,
          algorithm: "all_reduce_ring",
          bytes: 524_288,
          durations_ns: [999, 1_000, 1_001],
          regime: "two-rank fixture",
        },
        {
          id: "multi-gpu-collective-large",
          scenario_id: "multi-gpu",
          operation: "collective",
          link_ids: ["node0:nvlink:forward", "node0:nvlink:reverse"],
          participant_count: 2,
          algorithm: "all_reduce_ring",
          bytes: 67_108_864,
          durations_ns: [99_900, 100_000, 100_100],
          regime: "two-rank fixture",
        },
      ],
    },
  };
}
