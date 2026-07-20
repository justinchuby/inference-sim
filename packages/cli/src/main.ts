#!/usr/bin/env node
import {
  SCENARIO_PRESET_NAMES,
  analyzeStatic,
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  calculateScenarioMemoryLedger,
  listModelPresets,
  listPresets,
  simulateSpeculativeWorkload,
  validateScenario,
  type MemoryPolicyConfig,
  type ParallelismConfig,
  type PipelineConfig,
  type ScenarioPresetName,
  type SimulationScenario,
  type SpeculativeAcceptanceModel,
  type SpeculativeProposerFamily,
  type SpeculativeWorkloadConfig,
} from "@inference-sim/core";
import {
  optionalNumber,
  optionalString,
  readConfigFile,
  requireNumber,
  requireNumberArray,
  requireRecord,
  requireString,
} from "./config.js";

export interface CliIo {
  readonly stdout: (text: string) => void;
  readonly stderr: (text: string) => void;
}

const DEFAULT_IO: CliIo = {
  stdout: (text) => process.stdout.write(text),
  stderr: (text) => process.stderr.write(text),
};

export async function runCli(
  args: readonly string[],
  io: CliIo = DEFAULT_IO,
): Promise<number> {
  try {
    const [command = "help", argument] = args;
    switch (command) {
      case "help":
      case "--help":
      case "-h":
        io.stdout(helpText());
        return 0;
      case "presets":
        printJson(io, {
          scenarios: SCENARIO_PRESET_NAMES,
          hardware: listPresets(),
          models: listModelPresets(),
        });
        return 0;
      case "scenario": {
        if (!argument) {
          throw new Error("scenario requires a preset name");
        }
        if (!SCENARIO_PRESET_NAMES.includes(argument as ScenarioPresetName)) {
          throw new Error(`unknown scenario preset ${argument}`);
        }
        const scenario = buildScenarioPreset(argument as ScenarioPresetName);
        printJson(io, {
          scenario,
          memoryLedger: calculateScenarioMemoryLedger(scenario),
        });
        return 0;
      }
      case "validate": {
        const config = await loadRequiredConfig(argument, "validate");
        const scenario = config as unknown as SimulationScenario;
        const validation = validateScenario(scenario);
        printJson(io, validation);
        return validation.valid ? 0 : 2;
      }
      case "static": {
        const config = await loadRequiredConfig(argument, "static");
        const { topology, model, pipeline } = parseStaticConfig(config);
        printJson(io, analyzeStatic(topology, model, pipeline));
        return 0;
      }
      case "speculative": {
        const config = await loadRequiredConfig(argument, "speculative");
        printJson(
          io,
          simulateSpeculativeWorkload(parseSpeculativeConfig(config)),
        );
        return 0;
      }
      default:
        throw new Error(`unknown command ${command}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr(`inference-sim: ${message}\n`);
    return 1;
  }
}

async function loadRequiredConfig(
  path: string | undefined,
  command: string,
): Promise<Record<string, unknown>> {
  if (!path) {
    throw new Error(`${command} requires a YAML or JSON config path`);
  }
  return requireRecord(await readConfigFile(path), "config");
}

function parseStaticConfig(config: Record<string, unknown>) {
  const hardware = requireRecord(config.hardware, "hardware");
  const modelConfig = requireRecord(config.model, "model");
  const quantization = requireRecord(config.quantization, "quantization");
  const pipelineConfig = requireRecord(config.pipeline, "pipeline");
  const parallelismConfig = requireRecord(
    pipelineConfig.parallelism,
    "pipeline.parallelism",
  );
  const memoryConfig = requireRecord(config.memory, "memory");
  const topology = buildTopology(
    requireString(hardware, "preset", "hardware"),
  );
  const model = buildModelProfile(
    requireString(modelConfig, "preset", "model"),
    optionalString(quantization, "weights", "fp16", "quantization"),
    optionalString(quantization, "kv_cache", "fp16", "quantization"),
  );
  const parallelism: ParallelismConfig = {
    tensorParallel: requireNumber(
      parallelismConfig,
      "tensor_parallel",
      "pipeline.parallelism",
    ),
    pipelineParallel: requireNumber(
      parallelismConfig,
      "pipeline_parallel",
      "pipeline.parallelism",
    ),
    expertParallel: requireNumber(
      parallelismConfig,
      "expert_parallel",
      "pipeline.parallelism",
    ),
    dataParallel: requireNumber(
      parallelismConfig,
      "data_parallel",
      "pipeline.parallelism",
    ),
  };
  const memory: MemoryPolicyConfig = {
    kvCacheBudgetFraction: requireNumber(
      memoryConfig,
      "kv_cache_budget",
      "memory",
    ),
    expertCacheBudgetFraction: requireNumber(
      memoryConfig,
      "expert_cache_budget",
      "memory",
    ),
    pinnedPoolFraction: requireNumber(memoryConfig, "pinned_pool", "memory"),
    offloadStrategy: requireOffloadStrategy(
      requireString(memoryConfig, "offload", "memory"),
    ),
    prefetchAhead: requireNumber(memoryConfig, "prefetch_ahead", "memory"),
    pressureThreshold: requireNumber(
      memoryConfig,
      "pressure_threshold",
      "memory",
    ),
    reclaimBatchSize: optionalNumber(
      memoryConfig,
      "reclaim_batch_size",
      4,
      "memory",
    ),
  };
  const pipeline: PipelineConfig = {
    batchSize: requireNumber(pipelineConfig, "batch_size", "pipeline"),
    inputSeqLen: requireNumber(pipelineConfig, "input_seq_len", "pipeline"),
    outputSeqLen: requireNumber(pipelineConfig, "output_seq_len", "pipeline"),
    parallelism,
    memory,
  };
  return { topology, model, pipeline };
}

function parseSpeculativeConfig(
  config: Record<string, unknown>,
): SpeculativeWorkloadConfig {
  const speculative = requireRecord(
    config.speculative ?? config,
    "speculative",
  );
  const acceptanceConfig = requireRecord(
    speculative.acceptance,
    "speculative.acceptance",
  );
  const initialTokenLength = optionalNumber(
    speculative,
    "initial_token_length",
    0,
    "speculative",
  );
  const outputTokenCount = requireNumber(
    speculative,
    "output_token_count",
    "speculative",
  );
  const maxAdditionalTokens = requireNumber(
    speculative,
    "max_additional_tokens",
    "speculative",
  );
  const acceptance = parseAcceptance(acceptanceConfig);
  const capacityTokens =
    initialTokenLength + outputTokenCount + maxAdditionalTokens;
  const pagedKvConfig = speculative.paged_kv === undefined
    ? undefined
    : requireRecord(speculative.paged_kv, "speculative.paged_kv");
  return {
    family: requireFamily(
      requireString(speculative, "family", "speculative"),
    ),
    initialTokenLength,
    outputTokenCount,
    maxAdditionalTokens,
    maxIterations: optionalNumber(
      speculative,
      "max_iterations",
      Math.max(1, outputTokenCount),
      "speculative",
    ),
    acceptance,
    ...(pagedKvConfig
      ? {
          pagedKv: {
            sequenceId: optionalString(
              pagedKvConfig,
              "sequence_id",
              "target",
              "speculative.paged_kv",
            ),
            pageSizeTokens: requireNumber(
              pagedKvConfig,
              "page_size_tokens",
              "speculative.paged_kv",
            ),
            bytesPerToken: requireNumber(
              pagedKvConfig,
              "bytes_per_token",
              "speculative.paged_kv",
            ),
            capacityBytes: requireNumber(
              pagedKvConfig,
              "capacity_bytes",
              "speculative.paged_kv",
            ),
          },
        }
      : {}),
    stateGroups: [
      {
        id: "target-kv",
        owner: "target",
        capacityTokens,
        rollbackProtection: { kind: "non_destructive_tail" },
      },
      {
        id: "proposer-state",
        owner: "proposer",
        capacityTokens,
        rollbackProtection: {
          kind: "bounded_snapshot",
          maxRollbackTokens: maxAdditionalTokens,
        },
      },
    ],
  };
}

function parseAcceptance(
  config: Record<string, unknown>,
): SpeculativeAcceptanceModel {
  const kind = requireString(config, "kind", "speculative.acceptance");
  if (kind === "replay") {
    return {
      kind,
      acceptedDraftTokens: requireNumberArray(
        config,
        "accepted_draft_tokens",
        "speculative.acceptance",
      ),
    };
  }
  if (kind === "conditional_empirical" || kind === "conditional_heuristic") {
    return {
      kind,
      matchProbabilityByPosition: requireNumberArray(
        config,
        "match_probability_by_position",
        "speculative.acceptance",
      ),
      seed: optionalNumber(config, "seed", 42, "speculative.acceptance"),
    };
  }
  throw new Error(`unsupported acceptance kind ${kind}`);
}

function requireFamily(value: string): SpeculativeProposerFamily {
  const families: readonly SpeculativeProposerFamily[] = [
    "prompt_lookup",
    "draft_model",
    "mtp",
    "eagle3",
    "shared_kv",
    "self_speculative",
  ];
  if (!families.includes(value as SpeculativeProposerFamily)) {
    throw new Error(`unsupported speculative family ${value}`);
  }
  return value as SpeculativeProposerFamily;
}

function requireOffloadStrategy(
  value: string,
): MemoryPolicyConfig["offloadStrategy"] {
  if (value !== "none" && value !== "partial" && value !== "full") {
    throw new Error(`unsupported offload strategy ${value}`);
  }
  return value;
}

function printJson(io: CliIo, value: unknown): void {
  io.stdout(`${JSON.stringify(value, (_key, entry) => (
    typeof entry === "number" && !Number.isFinite(entry)
      ? String(entry)
      : entry
  ), 2)}\n`);
}

function helpText(): string {
  return `inference-sim

Usage:
  inference-sim presets
  inference-sim scenario <preset>
  inference-sim validate <scenario.yaml|json>
  inference-sim static <config.yaml|json>
  inference-sim speculative <config.yaml|json>
`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await runCli(process.argv.slice(2));
}
