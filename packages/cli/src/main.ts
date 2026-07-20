#!/usr/bin/env node
import {
  SCENARIO_PRESET_NAMES,
  analyzeStatic,
  buildSpeculativeStateGroups,
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  calculateScenarioMemoryLedger,
  compareTopologyWorkloads,
  compileTopologyWorkloadPlan,
  defaultSpeculativeEligibility,
  listModelPresets,
  listPresets,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateTopologyServingWorkload,
  simulateTopologyWorkload,
  runPlanFaultCampaign,
  targetOnlyTopologyProfile,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
  type ExpertCacheWorkloadConfig,
  type ExpertLoadTarget,
  validateScenario,
  type MemoryPolicyConfig,
  type ParallelismConfig,
  type PipelineConfig,
  type ScenarioPresetName,
  type SimulationScenario,
  type SpeculativeAcceptanceModel,
  type SpeculativeEligibility,
  type SpeculativeProposerFamily,
  type SpeculativeWorkloadConfig,
  type ServingSchedulerConfig,
  type ServingSpeculativeConfig,
  type TopologyServingResult,
  type TopologyWorkloadProfile,
} from "@inference-sim/core";
import {
  optionalBoolean,
  optionalNumber,
  optionalString,
  optionalStringArray,
  readConfigFile,
  requireNumber,
  requireNumberArray,
  requireRecordArray,
  requireRecord,
  requireString,
  requireStringArray,
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
    const [command = "help", argument, secondArgument] = args;
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
      case "expert-cache": {
        const config = await loadRequiredConfig(argument, "expert-cache");
        printJson(
          io,
          simulateExpertCacheWorkload(parseExpertCacheConfig(config)),
        );
        return 0;
      }
      case "serving": {
        if (!argument) {
          throw new Error("serving requires a scenario preset name");
        }
        if (!SCENARIO_PRESET_NAMES.includes(argument as ScenarioPresetName)) {
          throw new Error(`unknown scenario preset ${argument}`);
        }
        const config = await loadRequiredConfig(secondArgument, "serving");
        printJson(
          io,
          summarizeServingRun(simulateTopologyServingWorkload(
            buildScenarioPreset(argument as ScenarioPresetName),
            parseServingConfig(config),
          )),
        );
        return 0;
      }
      case "run": {
        if (!argument) {
          throw new Error("run requires a scenario preset name");
        }
        if (!SCENARIO_PRESET_NAMES.includes(argument as ScenarioPresetName)) {
          throw new Error(`unknown scenario preset ${argument}`);
        }
        const config = await loadRequiredConfig(secondArgument, "run");
        const scenario = buildScenarioPreset(argument as ScenarioPresetName);
        printJson(
          io,
          summarizeTopologyRun(
            simulateTopologyWorkload(
              scenario,
              buildTopologyProfile(config),
            ),
          ),
        );
        return 0;
      }
      case "compare": {
        const config = await loadRequiredConfig(argument, "compare");
        const profile = buildTopologyProfile(config);
        printJson(io, {
          profileId: profile.id,
          comparison: compareTopologyWorkloads(
            SCENARIO_PRESET_NAMES.map(buildScenarioPreset),
            profile,
          ),
        });
        return 0;
      }
      case "fault-campaign": {
        if (!argument) {
          throw new Error("fault-campaign requires a scenario preset name");
        }
        if (!SCENARIO_PRESET_NAMES.includes(argument as ScenarioPresetName)) {
          throw new Error(`unknown scenario preset ${argument}`);
        }
        const config = await loadRequiredConfig(
          secondArgument,
          "fault-campaign",
        );
        const scenario = buildScenarioPreset(argument as ScenarioPresetName);
        const plan = compileTopologyWorkloadPlan(
          scenario,
          buildTopologyProfile(config),
        );
        printJson(
          io,
          summarizeFaultCampaign(runPlanFaultCampaign(scenario, plan)),
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
  const family = requireFamily(
      requireString(speculative, "family", "speculative"),
  );
  return {
    family,
    eligibility: parseSpeculativeEligibility(
      family,
      speculative,
      "speculative",
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
    stateGroups: buildSpeculativeStateGroups(
      family,
      capacityTokens,
      maxAdditionalTokens,
    ),
  };
}

function parseExpertCacheConfig(
  config: Record<string, unknown>,
): ExpertCacheWorkloadConfig {
  const cache = requireRecord(config.expert_cache, "expert_cache");
  const workload = requireRecord(config.workload, "workload");
  const experts = requireRecordArray(
    cache,
    "experts",
    "expert_cache",
  ).map((expert, index) => ({
    id: requireString(expert, "id", `expert_cache.experts[${index}]`),
    bytes: requireNumber(expert, "bytes", `expert_cache.experts[${index}]`),
    routingWeight: optionalNumber(
      expert,
      "routing_weight",
      1,
      `expert_cache.experts[${index}]`,
    ),
  }));
  const prefetch = config.initial_prefetch === undefined
    ? undefined
    : requireRecord(config.initial_prefetch, "initial_prefetch");
  return {
    cache: {
      experts,
      hotCapacityBytes: requireNumber(
        cache,
        "hot_capacity_bytes",
        "expert_cache",
      ),
      warmCapacityBytes: requireNumber(
        cache,
        "warm_capacity_bytes",
        "expert_cache",
      ),
      warmToHotLatencyNs: requireNumber(
        cache,
        "warm_to_hot_latency_ns",
        "expert_cache",
      ),
      coldToHotLatencyNs: requireNumber(
        cache,
        "cold_to_hot_latency_ns",
        "expert_cache",
      ),
      coldToWarmLatencyNs: requireNumber(
        cache,
        "cold_to_warm_latency_ns",
        "expert_cache",
      ),
      routingSeed: optionalNumber(
        cache,
        "routing_seed",
        42,
        "expert_cache",
      ),
      initialHotExpertIds: optionalStringArray(
        cache,
        "initial_hot_expert_ids",
        [],
        "expert_cache",
      ),
      initialWarmExpertIds: optionalStringArray(
        cache,
        "initial_warm_expert_ids",
        [],
        "expert_cache",
      ),
    },
    tokenCount: requireNumber(workload, "token_count", "workload"),
    topK: requireNumber(workload, "top_k", "workload"),
    startAtNs: optionalNumber(workload, "start_at_ns", 0, "workload"),
    tokenIntervalNs: requireNumber(
      workload,
      "token_interval_ns",
      "workload",
    ),
    ...(prefetch
      ? {
          initialPrefetch: {
            expertIds: requireStringArray(
              prefetch,
              "expert_ids",
              "initial_prefetch",
            ),
            targetTier: requireLoadTarget(
              requireString(prefetch, "target_tier", "initial_prefetch"),
            ),
            leadTimeNs: requireNumber(
              prefetch,
              "lead_time_ns",
              "initial_prefetch",
            ),
          },
        }
      : {}),
  };
}

function parseServingConfig(
  config: Record<string, unknown>,
): ServingSchedulerConfig {
  const serving = requireRecord(config.serving ?? config, "serving");
  const requests = requireRecordArray(serving, "requests", "serving").map(
    (request, index) => ({
      id: requireString(request, "id", `serving.requests[${index}]`),
      arrivalNs: requireNumber(
        request,
        "arrival_ns",
        `serving.requests[${index}]`,
      ),
      promptTokens: requireNumber(
        request,
        "prompt_tokens",
        `serving.requests[${index}]`,
      ),
      outputTokens: requireNumber(
        request,
        "output_tokens",
        `serving.requests[${index}]`,
      ),
      priority: optionalNumber(
        request,
        "priority",
        0,
        `serving.requests[${index}]`,
      ),
    }),
  );
  return {
    requests,
    maxBatchSize: requireNumber(
      serving,
      "max_batch_size",
      "serving",
    ),
    maxBatchTokens: requireNumber(
      serving,
      "max_batch_tokens",
      "serving",
    ),
    prefillChunkTokens: requireNumber(
      serving,
      "prefill_chunk_tokens",
      "serving",
    ),
    maxKvTokens: requireNumber(serving, "max_kv_tokens", "serving"),
    ...(serving.speculative === undefined
      ? {}
      : {
          speculative: parseServingSpeculativeConfig(
            requireRecord(serving.speculative, "serving.speculative"),
          ),
        }),
    maxEvents: optionalNumber(
      serving,
      "max_events",
      1_000_000,
      "serving",
    ),
  };
}

function parseServingSpeculativeConfig(
  speculative: Record<string, unknown>,
): ServingSpeculativeConfig {
  const family = requireFamily(
    requireString(speculative, "family", "serving.speculative"),
  );
  const acceptanceConfig = requireRecord(
    speculative.acceptance,
    "serving.speculative.acceptance",
  );
  return {
    family,
    eligibility: parseSpeculativeEligibility(
      family,
      speculative,
      "serving.speculative",
    ),
    maxAdditionalTokens: requireNumber(
      speculative,
      "max_additional_tokens",
      "serving.speculative",
    ),
    acceptance: parseServingAcceptance(acceptanceConfig),
  };
}

function parseSpeculativeEligibility(
  family: SpeculativeProposerFamily,
  parent: Record<string, unknown>,
  context: string,
): SpeculativeEligibility {
  const defaults = defaultSpeculativeEligibility(family);
  const eligibilityContext = `${context}.eligibility`;
  const eligibility = parent.eligibility === undefined
    ? {}
    : requireRecord(parent.eligibility, eligibilityContext);
  const decoding = optionalString(
    eligibility,
    "decoding",
    defaults.decoding,
    eligibilityContext,
  );
  if (decoding !== "greedy" && decoding !== "sampling") {
    throw new Error(`${eligibilityContext}.decoding must be greedy or sampling`);
  }
  return {
    proposerAvailable: optionalBoolean(
      eligibility,
      "proposer_available",
      defaults.proposerAvailable,
      eligibilityContext,
    ),
    decoding,
    grammarActive: optionalBoolean(
      eligibility,
      "grammar_active",
      defaults.grammarActive,
      eligibilityContext,
    ),
    targetKvAvailable: optionalBoolean(
      eligibility,
      "target_kv_available",
      defaults.targetKvAvailable,
      eligibilityContext,
    ),
    targetHiddenOutputCount: optionalNumber(
      eligibility,
      "target_hidden_output_count",
      defaults.targetHiddenOutputCount,
      eligibilityContext,
    ),
    sharedKvGroupCount: optionalNumber(
      eligibility,
      "shared_kv_group_count",
      defaults.sharedKvGroupCount,
      eligibilityContext,
    ),
    ...(family === "self_speculative"
      ? {
          targetLayerCount: optionalNumber(
            eligibility,
            "target_layer_count",
            defaults.targetLayerCount ?? 32,
            eligibilityContext,
          ),
          earlyExitLayer: optionalNumber(
            eligibility,
            "early_exit_layer",
            defaults.earlyExitLayer ?? 16,
            eligibilityContext,
          ),
          allowDesignOnly: optionalBoolean(
            eligibility,
            "allow_design_only",
            defaults.allowDesignOnly ?? false,
            eligibilityContext,
          ),
        }
      : {}),
  };
}

function parseServingAcceptance(
  config: Record<string, unknown>,
): ServingSpeculativeConfig["acceptance"] {
  const context = "serving.speculative.acceptance";
  const kind = requireString(config, "kind", context);
  if (kind === "replay") {
    return {
      kind,
      acceptedDraftTokensByRequest: Object.fromEntries(
        Object.entries(requireRecord(
          config.accepted_draft_tokens_by_request,
          `${context}.accepted_draft_tokens_by_request`,
        )).map(([requestId, values]) => [
          requestId,
          requireNumberArray(
            { values },
            "values",
            `${context}.${requestId}`,
          ),
        ]),
      ),
    };
  }
  if (kind === "conditional_empirical" || kind === "conditional_heuristic") {
    return {
      kind,
      matchProbabilityByPosition: requireNumberArray(
        config,
        "match_probability_by_position",
        context,
      ),
      seed: optionalNumber(config, "seed", 42, context),
    };
  }
  throw new Error(`unsupported serving speculative acceptance kind ${kind}`);
}

function buildTopologyProfile(
  config: Record<string, unknown>,
): TopologyWorkloadProfile {
  if (config.speculative !== undefined) {
    return topologyProfileFromSpeculative(
      simulateSpeculativeWorkload(parseSpeculativeConfig(config)),
    );
  }
  if (config.expert_cache !== undefined) {
    const workload = parseExpertCacheConfig(config);
    const result = simulateExpertCacheWorkload(workload);
    const expertBytes = workload.cache.experts[0]?.bytes;
    if (
      expertBytes === undefined
      || workload.cache.experts.some((expert) => expert.bytes !== expertBytes)
    ) {
      throw new Error(
        "topology expert-cache profiles currently require equal expert byte sizes",
      );
    }
    return topologyProfileFromExpertCache(result, expertBytes);
  }
  if (config.target_only !== undefined) {
    const targetOnly = requireRecord(config.target_only, "target_only");
    return targetOnlyTopologyProfile(
      requireNumber(targetOnly, "token_count", "target_only"),
    );
  }
  throw new Error(
    "topology workload requires speculative, expert_cache, or target_only config",
  );
}

function summarizeTopologyRun(
  result: ReturnType<typeof simulateTopologyWorkload>,
) {
  return {
    scenarioId: result.scenarioId,
    profileId: result.profileId,
    confidence: result.confidence,
    assumptions: result.assumptions,
    status: result.execution.status,
    planSteps: result.plan.steps.length,
    operationCounts: countTopologyOperations(result),
    metrics: result.metrics,
  };
}

function summarizeFaultCampaign(
  result: ReturnType<typeof runPlanFaultCampaign>,
) {
  return {
    baseline: {
      status: result.baseline.status,
      completedAtNs: result.baseline.completedAtNs,
      submittedSteps: result.baseline.trace.operations.length,
      replayAppliedEvents: result.baselineReplay.appliedEvents,
    },
    cases: result.cases.map((entry) => {
      const unsubmitted =
        entry.execution.trace.terminal.unsubmittedStepIds ?? [];
      return {
        id: entry.id,
        fault: entry.fault,
        status: entry.execution.status,
        completedAtNs: entry.execution.completedAtNs,
        submittedSteps: entry.execution.trace.operations.length,
        unsubmittedSteps: unsubmitted.length,
        unsubmittedStepIdsPreview: unsubmitted.slice(0, 16),
        rankStates: entry.execution.rankStates,
        replayAppliedEvents: entry.replay.appliedEvents,
      };
    }),
  };
}

function summarizeServingRun(result: TopologyServingResult) {
  return {
    scenarioId: result.scenarioId,
    confidence: result.confidence,
    assumptions: result.assumptions,
    metrics: result.metrics,
    serving: {
      metrics: result.serving.metrics,
      requests: result.serving.requests,
      trace: result.serving.trace,
      replay: result.serving.replay,
    },
    batches: result.batches.map((batch) => ({
      batchId: batch.batchId,
      work: batch.work,
      durationNs: batch.topology.metrics.totalDurationNs,
      planSteps: batch.topology.plan.steps.length,
      operationCounts: countTopologyOperations(batch.topology),
    })),
  };
}

function countTopologyOperations(
  result: ReturnType<typeof simulateTopologyWorkload>,
) {
  const counts = {
    compute: 0,
    transfer: 0,
    collective: 0,
  };
  for (const event of result.execution.trace.operations) {
    counts[event.kind]++;
  }
  return counts;
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

function requireLoadTarget(value: string): ExpertLoadTarget {
  if (value !== "hot" && value !== "warm") {
    throw new Error(`unsupported expert load target ${value}`);
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
  inference-sim expert-cache <config.yaml|json>
  inference-sim serving <scenario-preset> <config.yaml|json>
  inference-sim run <scenario-preset> <workload.yaml|json>
  inference-sim compare <workload.yaml|json>
  inference-sim fault-campaign <scenario-preset> <workload.yaml|json>
`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await runCli(process.argv.slice(2));
}
