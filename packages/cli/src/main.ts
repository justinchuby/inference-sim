#!/usr/bin/env node
import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  analyzeStatic,
  bindParsedRuntimeCaptures,
  buildMultiGpuRingScenario,
  buildSpeculativeStateGroups,
  buildModelProfile,
  buildScenarioPreset,
  buildTopology,
  calculateScenarioMemoryLedger,
  compareTopologyServingWorkloads,
  compareTopologyWorkloads,
  compileTopologyWorkloadPlan,
  createFrozenPlanArtifact,
  defaultSpeculativeEligibility,
  executeFrozenPlan,
  expertCacheConfigForTopology,
  fitTopologyCostModel,
  parseCalibrationDataset,
  parseFrozenPlanArtifact,
  parseSpeculativeTokenTrace,
  resolveOnnxModelProfile,
  listModelPresets,
  listPresets,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateSpeculativeTokenTrace,
  simulateTopologyServingWorkload,
  simulateTopologyWorkload,
  runPlanFaultCampaign,
  runNodeFailoverCampaign,
  runSeededConcurrentNodeFailureCampaign,
  runSeededConcurrentPlanCampaign,
  replayPlanTrace,
  serializeFrozenPlanArtifact,
  serializeOnnxModelManifest,
  searchStaticConfigurations,
  targetOnlyTopologyProfile,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
  type ExpertCacheWorkloadConfig,
  type ExpertLoadTarget,
  validateScenario,
  type MemoryPolicyConfig,
  type ParallelismConfig,
  type PipelineConfig,
  type QuantType,
  type StaticSearchObjective,
  type StaticSearchRequest,
  type ScenarioPresetName,
  type SimulationScenario,
  type SpeculativeAcceptanceModel,
  type SpeculativeEligibility,
  type SpeculativeProposerFamily,
  type SpeculativeWorkloadConfig,
  type ServingSchedulerConfig,
  type ServingSpeculativeConfig,
  type TopologyServingResult,
  type TopologyServingExpertCacheConfig,
  type TopologyCostModel,
  type TopologyExpertPlacement,
  type TopologyWorkloadProfile,
  type ConcurrentPlanCampaignOptions,
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
import { inspectOnnxModel } from "./onnx-reader.js";

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
    const [
      command = "help",
      argument,
      secondArgument,
      thirdArgument,
      fourthArgument,
    ] = args;
    switch (command) {
      case "help":
      case "--help":
      case "-h":
        io.stdout(helpText());
        return 0;
      case "presets":
        printJson(io, {
          scenarios: SCENARIO_PRESET_NAMES,
          parameterizedScenario: "multi-gpu-ring-<2..64>",
          hardware: listPresets(),
          models: listModelPresets(),
        });
        return 0;
      case "scenario": {
        const scenario = resolveScenarioTarget(argument, "scenario");
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
      case "speculative-trace": {
        const config = await loadRequiredConfig(
          argument,
          "speculative-trace",
        );
        const result = simulateSpeculativeTokenTrace(
          parseSpeculativeTokenTrace(config),
        );
        if (secondArgument) {
          const scenario = resolveScenarioTarget(
            secondArgument,
            "speculative-trace",
          );
          const costModel = await loadCostModel(thirdArgument);
          printJson(io, {
            trace: result,
            topology: simulateTopologyWorkload(
              scenario,
              topologyProfileFromSpeculative(result.workload),
              costModel,
            ),
          });
        } else {
          printJson(io, result);
        }
        return result.differential.matchesTargetOnly ? 0 : 2;
      }
      case "speculative-capture": {
        const targetOnlyConfig = await loadRequiredConfig(
          argument,
          "speculative-capture target-only",
        );
        const speculativeConfig = await loadRequiredConfig(
          secondArgument,
          "speculative-capture speculative",
        );
        const bound = bindParsedRuntimeCaptures(
          targetOnlyConfig,
          speculativeConfig,
        );
        const summary = {
          targetOnlyCaptureId: bound.targetOnlyCaptureId,
          speculativeCaptureId: bound.speculativeCaptureId,
          trace: bound.result,
        };
        if (thirdArgument) {
          const scenario = resolveScenarioTarget(
            thirdArgument,
            "speculative-capture",
          );
          const costModel = await loadCostModel(fourthArgument);
          printJson(io, {
            ...summary,
            topology: simulateTopologyWorkload(
              scenario,
              topologyProfileFromSpeculative(bound.result.workload),
              costModel,
            ),
          });
        } else {
          printJson(io, summary);
        }
        return bound.result.differential.matchesTargetOnly ? 0 : 2;
      }
      case "expert-cache": {
        const config = await loadRequiredConfig(argument, "expert-cache");
        printJson(
          io,
          simulateExpertCacheWorkload(parseExpertCacheConfig(config)),
        );
        return 0;
      }
      case "calibrate": {
        const config = await loadRequiredConfig(argument, "calibrate");
        printJson(io, fitTopologyCostModel(parseCalibrationDataset(config)));
        return 0;
      }
      case "onnx-inspect": {
        if (argument === undefined) {
          throw new Error("onnx-inspect requires an ONNX model path");
        }
        const metadata = secondArgument === undefined
          ? undefined
          : await readConfigFile(secondArgument);
        printJson(
          io,
          JSON.parse(serializeOnnxModelManifest(
            await inspectOnnxModel(argument, metadata),
            true,
          )),
        );
        return 0;
      }
      case "onnx-static": {
        const config = await loadRequiredConfig(argument, "onnx-static");
        if (secondArgument === undefined) {
          throw new Error("onnx-static requires an ONNX model path");
        }
        const metadata = thirdArgument === undefined
          ? undefined
          : await readConfigFile(thirdArgument);
        const manifest = await inspectOnnxModel(secondArgument, metadata);
        const { topology, pipeline, kvCacheQuantization, activationQuantization } =
          parseOnnxStaticConfig(config);
        const model = resolveOnnxModelProfile(manifest, {
          kvCacheQuantization,
          activationQuantization,
        });
        printJson(io, {
          manifest: {
            fingerprint: manifest.manifestFingerprint,
            modelFileName: manifest.source.modelFileName,
            initializerLogicalBytes:
              manifest.totals.initializerLogicalBytes,
            profileReadiness: manifest.profileReadiness,
          },
          model,
          analysis: analyzeStatic(topology, model, pipeline),
        });
        return 0;
      }
      case "onnx-search": {
        const config = await loadRequiredConfig(argument, "onnx-search");
        if (secondArgument === undefined) {
          throw new Error("onnx-search requires an ONNX model path");
        }
        const metadata = thirdArgument === undefined
          ? undefined
          : await readConfigFile(thirdArgument);
        const manifest = await inspectOnnxModel(secondArgument, metadata);
        const baseQuantization = requireRecord(
          config.quantization,
          "quantization",
        );
        const model = resolveOnnxModelProfile(manifest, {
          kvCacheQuantization: requireQuantType(
            requireStringArray(
              baseQuantization,
              "kv_cache",
              "quantization",
            )[0],
          ),
          activationQuantization: requireQuantType(
            requireStringArray(
              baseQuantization,
              "activations",
              "quantization",
            )[0],
          ),
        });
        printJson(io, {
          manifest: {
            fingerprint: manifest.manifestFingerprint,
            modelFileName: manifest.source.modelFileName,
            initializerLogicalBytes:
              manifest.totals.initializerLogicalBytes,
          },
          search: searchStaticConfigurations(
            model,
            parseOnnxSearchConfig(config),
          ),
        });
        return 0;
      }
      case "serving": {
        const scenario = resolveScenarioTarget(argument, "serving");
        const config = await loadRequiredConfig(secondArgument, "serving");
        const costModel = await loadCostModel(thirdArgument);
        const expertCache = parseServingExpertCacheConfig(config);
        printJson(
          io,
          summarizeServingRun(simulateTopologyServingWorkload(
            scenario,
            parseServingConfig(config),
            costModel,
            expertCache,
          )),
        );
        return 0;
      }
      case "serving-compare": {
        const config = await loadRequiredConfig(argument, "serving-compare");
        const costModel = await loadCostModel(secondArgument);
        const expertCache = parseServingExpertCacheConfig(config);
        printJson(
          io,
          summarizeServingComparison(compareTopologyServingWorkloads(
            SCENARIO_PRESET_NAMES.map(buildScenarioPreset),
            parseServingConfig(config),
            costModel,
            expertCache,
          )),
        );
        return 0;
      }
      case "run": {
        const scenario = resolveScenarioTarget(argument, "run");
        const config = await loadRequiredConfig(secondArgument, "run");
        const costModel = await loadCostModel(thirdArgument);
        printJson(
          io,
          summarizeTopologyRun(
            simulateTopologyWorkload(
              scenario,
              buildTopologyProfile(config, scenario),
              costModel,
            ),
          ),
        );
        return 0;
      }
      case "plan-export": {
        const scenario = resolveScenarioTarget(argument, "plan-export");
        const config = await loadRequiredConfig(
          secondArgument,
          "plan-export",
        );
        const costModel = await loadCostModel(thirdArgument);
        const plan = compileTopologyWorkloadPlan(
          scenario,
          buildTopologyProfile(config, scenario),
          costModel,
        );
        io.stdout(`${serializeFrozenPlanArtifact(
          createFrozenPlanArtifact(scenario, plan),
          true,
        )}\n`);
        return 0;
      }
      case "plan-run": {
        const config = await loadRequiredConfig(argument, "plan-run");
        const artifact = parseFrozenPlanArtifact(config);
        const execution = executeFrozenPlan(
          artifact.scenario,
          artifact.plan,
        );
        const replay = replayPlanTrace(
          artifact.scenario,
          artifact.plan,
          execution.trace,
        );
        printJson(io, {
          artifact: {
            kind: artifact.kind,
            revision: artifact.revision,
            artifactFingerprint: artifact.artifactFingerprint,
            scenarioFingerprint: artifact.scenarioFingerprint,
            planFingerprint: artifact.planFingerprint,
          },
          scenarioId: artifact.scenario.id,
          planId: artifact.plan.id,
          execution,
          replay,
        });
        return 0;
      }
      case "compare": {
        const config = await loadRequiredConfig(argument, "compare");
        const scenarios = SCENARIO_PRESET_NAMES.map(buildScenarioPreset);
        const profile = buildTopologyProfile(config, scenarios[0]);
        const costModel = await loadCostModel(secondArgument);
        printJson(io, {
          profileId: profile.id,
          costModel: summarizeCostModel(costModel),
          comparison: compareTopologyWorkloads(
            scenarios,
            (scenario) => buildTopologyProfile(config, scenario),
            costModel,
          ),
        });
        return 0;
      }
      case "fault-campaign": {
        const scenario = resolveScenarioTarget(argument, "fault-campaign");
        const config = await loadRequiredConfig(
          secondArgument,
          "fault-campaign",
        );
        const costModel = await loadCostModel(thirdArgument);
        const plan = compileTopologyWorkloadPlan(
          scenario,
          buildTopologyProfile(config, scenario),
          costModel,
        );
        printJson(
          io,
          summarizeFaultCampaign(runPlanFaultCampaign(scenario, plan)),
        );
        return 0;
      }
      case "concurrent-campaign": {
        const scenario = resolveScenarioTarget(
          argument,
          "concurrent-campaign",
        );
        const config = await loadRequiredConfig(
          secondArgument,
          "concurrent-campaign",
        );
        const costModel = await loadCostModel(thirdArgument);
        const plan = compileTopologyWorkloadPlan(
          scenario,
          buildTopologyProfile(config, scenario),
          costModel,
        );
        printJson(
          io,
          summarizeConcurrentCampaign(runSeededConcurrentPlanCampaign(
            scenario,
            plan,
            parseConcurrentCampaignOptions(config, scenario.execution.seed),
          )),
        );
        return 0;
      }
      case "node-failover": {
        const failedScenario = resolveScenarioTarget(
          argument,
          "node-failover failed scenario",
        );
        const recoveryBase = resolveScenarioTarget(
          secondArgument,
          "node-failover recovery scenario",
        );
        const config = await loadRequiredConfig(thirdArgument, "node-failover");
        const failover = requireRecord(
          config.node_failover,
          "node_failover",
        );
        const recoveryScenario = {
          ...recoveryBase,
          execution: {
            ...recoveryBase.execution,
            topologyEpoch: failedScenario.execution.topologyEpoch + 1,
          },
        };
        const costModel = await loadCostModel(fourthArgument);
        const failedProfile = buildTopologyProfile(config, failedScenario);
        const recoveryProfile = buildTopologyProfile(config, recoveryScenario);
        const failedPlan = compileTopologyWorkloadPlan(
          failedScenario,
          failedProfile,
          costModel,
        );
        const replannedPlan = compileTopologyWorkloadPlan(
          recoveryScenario,
          recoveryProfile,
          costModel,
        );
        printJson(
          io,
          summarizeNodeFailover(runNodeFailoverCampaign(
            failedScenario,
            failedPlan,
            {
              failedNodeId: requireString(
                failover,
                "failed_node_id",
                "node_failover",
              ),
              faultAtNs: requireNumber(
                failover,
                "fault_at_ns",
                "node_failover",
              ),
              reason: optionalString(
                failover,
                "reason",
                "node heartbeat expired",
                "node_failover",
              ),
              recoveryScenario,
              replannedPlan,
            },
          )),
        );
        return 0;
      }
      case "concurrent-node-failure": {
        const scenario = resolveScenarioTarget(
          argument,
          "concurrent-node-failure",
        );
        const config = await loadRequiredConfig(
          secondArgument,
          "concurrent-node-failure",
        );
        const failure = requireRecord(config.node_failure, "node_failure");
        const costModel = await loadCostModel(thirdArgument);
        const plan = compileTopologyWorkloadPlan(
          scenario,
          buildTopologyProfile(config, scenario),
          costModel,
        );
        printJson(
          io,
          summarizeConcurrentNodeFailure(
            runSeededConcurrentNodeFailureCampaign(
              scenario,
              plan,
              parseConcurrentCampaignOptions(config, scenario.execution.seed),
              {
                kind: "node_failure",
                atNs: requireNumber(
                  failure,
                  "at_ns",
                  "node_failure",
                ),
                nodeId: requireString(
                  failure,
                  "node_id",
                  "node_failure",
                ),
                reason: optionalString(
                  failure,
                  "reason",
                  "node heartbeat expired",
                  "node_failure",
                ),
              },
            ),
          ),
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

function resolveScenarioTarget(
  value: string | undefined,
  command: string,
): SimulationScenario {
  if (value === undefined) {
    throw new Error(`${command} requires a scenario target`);
  }
  if (SCENARIO_PRESET_NAMES.includes(value as ScenarioPresetName)) {
    return buildScenarioPreset(value as ScenarioPresetName);
  }
  const ringMatch = /^multi-gpu-ring-([1-9]\d*)$/.exec(value);
  if (ringMatch !== null) {
    return buildMultiGpuRingScenario(Number(ringMatch[1]));
  }
  throw new Error(`unknown scenario preset or ring target ${value}`);
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

async function loadCostModel(
  path: string | undefined,
): Promise<TopologyCostModel> {
  if (path === undefined) {
    return DEFAULT_TOPOLOGY_COST_MODEL;
  }
  return fitTopologyCostModel(
    parseCalibrationDataset(await readConfigFile(path)),
  ).costModel;
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

function parseOnnxStaticConfig(config: Record<string, unknown>) {
  const hardware = requireRecord(config.hardware, "hardware");
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
  return {
    topology,
    pipeline,
    kvCacheQuantization: requireQuantType(
      optionalString(quantization, "kv_cache", "fp16", "quantization"),
    ),
    activationQuantization: requireQuantType(
      optionalString(quantization, "activations", "fp16", "quantization"),
    ),
  };
}

function parseOnnxSearchConfig(
  config: Record<string, unknown>,
): StaticSearchRequest {
  const search = requireRecord(config.search, "search");
  const hardware = requireRecord(config.hardware, "hardware");
  const quantization = requireRecord(config.quantization, "quantization");
  const pipeline = requireRecord(config.pipeline, "pipeline");
  const parallelism = requireRecord(
    pipeline.parallelism,
    "pipeline.parallelism",
  );
  const memory = requireRecord(config.memory, "memory");
  const offloadStrategies = requireStringArray(
    memory,
    "offload",
    "memory",
  ).map(requireOffloadStrategy);
  return {
    objective: requireStaticSearchObjective(
      requireString(search, "objective", "search"),
    ),
    topK: requireNumber(search, "top_k", "search"),
    maxCandidates: requireNumber(search, "max_candidates", "search"),
    constraints: {
      requireFeasible: optionalBoolean(
        search,
        "require_feasible",
        true,
        "search",
      ),
      ...(search.maximum_device_used_fraction === undefined
        ? {}
        : {
            maximumDeviceUsedFraction: requireNumber(
              search,
              "maximum_device_used_fraction",
              "search",
            ),
          }),
      ...(search.maximum_ttft_ms === undefined
        ? {}
        : {
            maximumTimeToFirstTokenMs: requireNumber(
              search,
              "maximum_ttft_ms",
              "search",
            ),
          }),
      ...(search.maximum_itl_ms === undefined
        ? {}
        : {
            maximumInterTokenLatencyMs: requireNumber(
              search,
              "maximum_itl_ms",
              "search",
            ),
          }),
    },
    space: {
      topologies: requireStringArray(
        hardware,
        "presets",
        "hardware",
      ).map((id) => ({ id, topology: buildTopology(id) })),
      kvCacheQuantizations: requireStringArray(
        quantization,
        "kv_cache",
        "quantization",
      ).map(requireQuantType),
      activationQuantizations: requireStringArray(
        quantization,
        "activations",
        "quantization",
      ).map(requireQuantType),
      batchSizes: requireNumberArray(
        pipeline,
        "batch_sizes",
        "pipeline",
      ),
      inputSeqLens: requireNumberArray(
        pipeline,
        "input_seq_lens",
        "pipeline",
      ),
      outputSeqLens: requireNumberArray(
        pipeline,
        "output_seq_lens",
        "pipeline",
      ),
      tensorParallel: requireNumberArray(
        parallelism,
        "tensor_parallel",
        "pipeline.parallelism",
      ),
      pipelineParallel: requireNumberArray(
        parallelism,
        "pipeline_parallel",
        "pipeline.parallelism",
      ),
      expertParallel: requireNumberArray(
        parallelism,
        "expert_parallel",
        "pipeline.parallelism",
      ),
      dataParallel: requireNumberArray(
        parallelism,
        "data_parallel",
        "pipeline.parallelism",
      ),
      memoryPolicies: offloadStrategies.map((offloadStrategy) => ({
        kvCacheBudgetFraction: requireNumber(
          memory,
          "kv_cache_budget",
          "memory",
        ),
        expertCacheBudgetFraction: requireNumber(
          memory,
          "expert_cache_budget",
          "memory",
        ),
        pinnedPoolFraction: requireNumber(memory, "pinned_pool", "memory"),
        offloadStrategy,
        prefetchAhead: requireNumber(memory, "prefetch_ahead", "memory"),
        pressureThreshold: requireNumber(
          memory,
          "pressure_threshold",
          "memory",
        ),
        reclaimBatchSize: optionalNumber(
          memory,
          "reclaim_batch_size",
          4,
          "memory",
        ),
      })),
    },
  };
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
  const adaptivePrefetch = cache.adaptive_prefetch === undefined
    ? undefined
    : requireRecord(
        cache.adaptive_prefetch,
        "expert_cache.adaptive_prefetch",
      );
  const adaptiveTarget = adaptivePrefetch === undefined
    ? undefined
    : requireString(
        adaptivePrefetch,
        "target_tier",
        "expert_cache.adaptive_prefetch",
      );
  if (adaptiveTarget !== undefined && adaptiveTarget !== "warm") {
    throw new Error("expert_cache.adaptive_prefetch.target_tier must be warm");
  }
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
      ...(adaptivePrefetch === undefined
        ? {}
        : {
            adaptivePrefetch: {
              targetTier: "warm" as const,
              minObservations: requireNumber(
                adaptivePrefetch,
                "min_observations",
                "expert_cache.adaptive_prefetch",
              ),
              intervalTokens: requireNumber(
                adaptivePrefetch,
                "interval_tokens",
                "expert_cache.adaptive_prefetch",
              ),
              maxExpertsPerDecision: requireNumber(
                adaptivePrefetch,
                "max_experts_per_decision",
                "expert_cache.adaptive_prefetch",
              ),
            },
          }),
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

function parseServingExpertCacheConfig(
  config: Record<string, unknown>,
): TopologyServingExpertCacheConfig | undefined {
  if (config.expert_cache === undefined) {
    return undefined;
  }
  const cache = requireRecord(config.expert_cache, "expert_cache");
  const revision = requireNumber(
    cache,
    "contract_revision",
    "expert_cache",
  );
  const topK = requireNumber(cache, "top_k", "expert_cache");
  const parsed = parseExpertCacheConfig({
    expert_cache: cache,
    workload: {
      token_count: 1,
      top_k: topK,
      token_interval_ns: 0,
    },
  });
  return {
    contractRevision:
      revision as typeof SERVING_EXPERT_CACHE_CONTRACT_REVISION,
    cache: parsed.cache,
    topK,
    placementStrategy: requireExpertPlacementStrategy(optionalString(
      cache,
      "placement_strategy",
      "contiguous",
      "expert_cache",
    )),
  };
}

function requireExpertPlacementStrategy(
  value: string,
): TopologyExpertPlacement["strategy"] {
  if (value !== "contiguous" && value !== "round_robin") {
    throw new Error(
      `expert_cache.placement_strategy must be contiguous or round_robin; got ${value}`,
    );
  }
  return value;
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
  scenario?: SimulationScenario,
): TopologyWorkloadProfile {
  if (config.speculative !== undefined) {
    return topologyProfileFromSpeculative(
      simulateSpeculativeWorkload(parseSpeculativeConfig(config)),
    );
  }
  if (config.expert_cache !== undefined) {
    const workload = parseExpertCacheConfig(config);
    const cache = requireRecord(config.expert_cache, "expert_cache");
    const placementStrategy = requireExpertPlacementStrategy(optionalString(
      cache,
      "placement_strategy",
      "contiguous",
      "expert_cache",
    ));
    const placement: TopologyExpertPlacement = {
      strategy: placementStrategy,
      expertIds: workload.cache.experts.map((expert) => expert.id),
    };
    const effectiveWorkload = scenario === undefined
      ? workload
      : {
          ...workload,
          cache: expertCacheConfigForTopology(
            scenario,
            workload.cache,
            placement,
          ),
        };
    const result = simulateExpertCacheWorkload(effectiveWorkload);
    return topologyProfileFromExpertCache(
      result,
      placementStrategy,
    );
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

function parseConcurrentCampaignOptions(
  config: Record<string, unknown>,
  defaultSeed: number,
): ConcurrentPlanCampaignOptions {
  const campaign = config.concurrent_campaign === undefined
    ? {}
    : requireRecord(config.concurrent_campaign, "concurrent_campaign");
  return {
    executionCount: optionalNumber(
      campaign,
      "execution_count",
      32,
      "concurrent_campaign",
    ),
    seed: optionalNumber(
      campaign,
      "seed",
      defaultSeed,
      "concurrent_campaign",
    ),
    arrivalWindowNs: optionalNumber(
      campaign,
      "arrival_window_ns",
      1_000_000,
      "concurrent_campaign",
    ),
  };
}

function summarizeConcurrentCampaign(
  result: ReturnType<typeof runSeededConcurrentPlanCampaign>,
) {
  const arrivals = new Map(
    result.requests.map((request) => [
      request.executionId,
      request.arrivalNs,
    ]),
  );
  const latencies = result.execution.executions
    .map((execution) => (
      execution.completedAtNs - (arrivals.get(execution.executionId) ?? 0)
    ))
    .sort((left, right) => left - right);
  const totalOperations = result.execution.trace.operations.length;
  return {
    options: result.options,
    assumptions: [
      "physical allocation ids remain shared across executions; conflicting writes are lease-serialized",
      "this campaign stresses shared execution resources and is not a substitute for paged-KV request serving",
    ],
    completedAtNs: result.execution.completedAtNs,
    executionCount: result.execution.executions.length,
    maximumConcurrentExecutions:
      result.execution.maximumConcurrentExecutions,
    submittedOperations: totalOperations,
    replayAppliedEvents: result.replay.appliedEvents,
    replayExecutions: result.replay.executions.length,
    latencyNs: {
      minimum: latencies[0] ?? 0,
      average: latencies.length === 0
        ? 0
        : latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
      p95: percentile(latencies, 0.95),
      maximum: latencies.at(-1) ?? 0,
    },
    admissionsPreview: result.requests.slice(0, 16),
  };
}

function summarizeNodeFailover(
  result: ReturnType<typeof runNodeFailoverCampaign>,
) {
  return {
    handoff: result.handoff,
    completedAtNs: result.completedAtNs,
    failedExecution: {
      status: result.failedExecution.status,
      completedAtNs: result.failedExecution.completedAtNs,
      submittedOperations: result.failedExecution.trace.operations.length,
      rankStates: result.failedExecution.rankStates,
      replayAppliedEvents: result.failedReplay.appliedEvents,
    },
    recoveryExecution: {
      status: result.recoveryExecution.status,
      completedAtNs: result.recoveryExecution.completedAtNs,
      submittedOperations: result.recoveryExecution.trace.operations.length,
      rankStates: result.recoveryExecution.rankStates,
      replayAppliedEvents: result.recoveryReplay.appliedEvents,
    },
  };
}

function summarizeConcurrentNodeFailure(
  result: ReturnType<typeof runSeededConcurrentNodeFailureCampaign>,
) {
  return {
    options: result.options,
    fault: result.fault,
    assumptions: [
      "all campaign executions must be admitted before the node-fault timestamp",
      "the node fault atomically closes new submission for every old-epoch execution",
      "only the shared global schedule prefix submitted before the fault may quiesce",
    ],
    completedAtNs: result.execution.completedAtNs,
    executionCount: result.execution.executions.length,
    maximumConcurrentExecutions:
      result.execution.maximumConcurrentExecutions,
    submittedOperations: result.execution.trace.operations.length,
    replayAppliedEvents: result.replay.appliedEvents,
    failedExecutions: result.execution.executions.filter(
      (execution) => execution.status === "failed",
    ).length,
    terminalsPreview: result.execution.trace.terminals.slice(0, 8)
      .map((terminal) => ({
        executionId: terminal.executionId,
        status: terminal.status,
        timestampNs: terminal.timestampNs,
        failureAtNs: terminal.failureAtNs,
        submittedOperations: terminal.sourceSequence,
        unsubmittedOperations: terminal.unsubmittedStepIds?.length ?? 0,
        rankStates: terminal.rankStates,
      })),
  };
}

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) {
    return 0;
  }
  return values[Math.min(
    values.length - 1,
    Math.max(0, Math.ceil(values.length * quantile) - 1),
  )];
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
      durationNs: batch.durationNs,
      topologyDurationNs: batch.topology.metrics.totalDurationNs,
      cacheConstraintNs: batch.cacheConstraintNs,
      expertRoutes: batch.expertRoutes.length,
      planSteps: batch.topology.plan.steps.length,
      operationCounts: countTopologyOperations(batch.topology),
    })),
    ...(result.expertCache === undefined
      ? {}
      : {
          expertCache: {
            contractRevision: result.expertCache.contractRevision,
            metrics: result.expertCache.snapshot.metrics,
            hotExpertIds: result.expertCache.snapshot.hotExpertIds,
            warmExpertIds: result.expertCache.snapshot.warmExpertIds,
            hotPartitions: result.expertCache.snapshot.hotPartitions,
            warmPartitions: result.expertCache.snapshot.warmPartitions,
            routes: result.expertCache.routes.length,
            traceEvents: result.expertCache.trace.length,
            replayAppliedEvents: result.expertCache.replay.appliedEvents,
          },
        }),
    ...(result.physical === undefined
      ? {}
      : {
          physical: {
            completedAtNs: result.physical.execution.completedAtNs,
            maximumConcurrentExecutions:
              result.physical.execution.maximumConcurrentExecutions,
            operations: result.physical.execution.trace.operations.length,
            replayAppliedEvents: result.physical.replay.appliedEvents,
          },
        }),
  };
}

function summarizeServingComparison(
  comparison: ReturnType<typeof compareTopologyServingWorkloads>,
) {
  return {
    assumptions: comparison.runs[0]?.result.assumptions ?? [],
    comparison: comparison.runs.map((run) => ({
      rank: run.rank,
      scenarioId: run.result.scenarioId,
      relativeToFastest: run.relativeToFastest,
      confidence: run.result.confidence,
      totalDurationNs: run.result.metrics.totalDurationNs,
      throughputTokensPerSecond:
        run.result.serving.metrics.throughputTokensPerSecond,
      p95TimeToFirstTokenNs:
        run.result.serving.metrics.p95TimeToFirstTokenNs,
      p95InterTokenLatencyNs:
        run.result.serving.metrics.p95InterTokenLatencyNs,
      averageRequestLatencyNs:
        run.result.serving.metrics.averageRequestLatencyNs,
      batches: run.result.serving.metrics.batches,
      kvHighWaterTokens: run.result.serving.metrics.kvHighWaterTokens,
      targetForwards: run.result.serving.metrics.targetForwards,
      proposedDraftTokens:
        run.result.serving.metrics.proposedDraftTokens,
      acceptedDraftTokens:
        run.result.serving.metrics.acceptedDraftTokens,
      replayAppliedEvents: run.result.serving.replay.appliedEvents,
    })),
  };
}

function summarizeCostModel(costModel: TopologyCostModel) {
  return {
    revision: costModel.revision,
    confidence: costModel.confidence,
    source: costModel.source,
    transportCurveCount: costModel.transportCurves?.length ?? 0,
    ...(costModel.applicability === undefined
      ? {}
      : { applicability: costModel.applicability }),
  };
}

function countTopologyOperations(
  result: ReturnType<typeof simulateTopologyWorkload>,
) {
  const counts = {
    compute: 0,
    transfer: 0,
    collective: 0,
    allReduce: 0,
    allToAll: 0,
  };
  for (const event of result.execution.trace.operations) {
    counts[event.kind]++;
    if (event.collectiveAlgorithm === "all_reduce_ring") {
      counts.allReduce++;
    } else if (event.collectiveAlgorithm === "all_to_all_v") {
      counts.allToAll++;
    }
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

function requireQuantType(value: string): QuantType {
  const types: readonly QuantType[] = [
    "fp32",
    "fp16",
    "bf16",
    "fp8",
    "int8",
    "int4",
    "nf4",
  ];
  if (!types.includes(value as QuantType)) {
    throw new Error(`unsupported quantization type ${value}`);
  }
  return value as QuantType;
}

function requireStaticSearchObjective(
  value: string,
): StaticSearchObjective {
  const objectives: readonly StaticSearchObjective[] = [
    "decode_throughput",
    "prefill_throughput",
    "time_to_first_token",
    "inter_token_latency",
    "device_headroom",
  ];
  if (!objectives.includes(value as StaticSearchObjective)) {
    throw new Error(`unsupported static search objective ${value}`);
  }
  return value as StaticSearchObjective;
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
  inference-sim scenario <scenario-target>
  inference-sim validate <scenario.yaml|json>
  inference-sim static <config.yaml|json>
  inference-sim speculative <config.yaml|json>
  inference-sim speculative-trace <trace.yaml|json> [scenario-target] [calibration.yaml|json]
  inference-sim speculative-capture <target-only.yaml|json> <speculative.yaml|json> [scenario-target] [calibration.yaml|json]
  inference-sim expert-cache <config.yaml|json>
  inference-sim calibrate <calibration.yaml|json>
  inference-sim onnx-inspect <model.onnx> [metadata.yaml|json]
  inference-sim onnx-static <config.yaml|json> <model.onnx> [metadata.yaml|json]
  inference-sim onnx-search <search.yaml|json> <model.onnx> [metadata.yaml|json]
  inference-sim serving <scenario-target> <config.yaml|json> [calibration.yaml|json]
  inference-sim serving-compare <config.yaml|json> [calibration.yaml|json]
  inference-sim run <scenario-target> <workload.yaml|json> [calibration.yaml|json]
  inference-sim plan-export <scenario-target> <workload.yaml|json> [calibration.yaml|json]
  inference-sim plan-run <frozen-plan.json>
  inference-sim compare <workload.yaml|json> [calibration.yaml|json]
  inference-sim fault-campaign <scenario-target> <workload.yaml|json> [calibration.yaml|json]
  inference-sim concurrent-campaign <scenario-target> <workload.yaml|json> [calibration.yaml|json]
  inference-sim node-failover <failed-target> <recovery-target> <workload.yaml|json> [calibration.yaml|json]
  inference-sim concurrent-node-failure <scenario-target> <workload.yaml|json> [calibration.yaml|json]

Scenario target:
  one of the listed presets, or multi-gpu-ring-N for N=2..64
`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exitCode = await runCli(process.argv.slice(2));
}
