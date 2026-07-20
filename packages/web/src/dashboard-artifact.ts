import {
  CALIBRATION_DATASET_REVISION,
  CONCURRENT_PLAN_TRACE_REVISION,
  EXPERT_CACHE_CONTRACT_REVISION,
  PAGED_KV_CONTRACT_REVISION,
  PLAN_CONTRACT_REVISION,
  SCENARIO_SCHEMA_VERSION,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  SERVING_TRACE_CONTRACT_REVISION,
  SPECULATIVE_FAMILY_CONTRACT_REVISION,
  SPECULATIVE_ITERATION_CONTRACT_REVISION,
  SPECULATIVE_TOKEN_TRACE_REVISION,
  TOPOLOGY_COST_MODEL_REVISION,
  createSimulationResultArtifact,
  fitTopologyCostModel,
  parseSimulationResultArtifact,
  parseSimulationScenario,
  simulateSpeculativeTokenTrace,
  type CalibrationDataset,
  type SpeculativeProposerFamily,
  type SpeculativeTokenTrace,
} from "@inference-sim/core";
import type { ParsedCalibrationFile } from "./calibration-import.js";
import type { ParsedTokenTraceFile } from "./token-trace-import.js";
import type {
  DashboardArtifact,
  DashboardArtifactExpectation,
  DashboardArtifactOutput,
  DashboardArtifactReplay,
  DashboardRunConfig,
} from "./types.js";
import { ROOFLINE_SUMMARY_REVISION } from "./roofline.js";

export const MAX_DASHBOARD_ARTIFACT_FILE_BYTES = 128 * 1024 * 1024;
export const PIPELINE_EXECUTION_CONTRACT_REVISION = 1;

const DASHBOARD_BASE_CONTRACTS = {
  frozen_plan: PLAN_CONTRACT_REVISION,
  scenario_schema: SCENARIO_SCHEMA_VERSION,
  topology_cost_model: TOPOLOGY_COST_MODEL_REVISION,
} as const;

export interface ParsedDashboardArtifact {
  readonly config: DashboardRunConfig;
  readonly expectation: DashboardArtifactExpectation;
  readonly calibration?: ParsedCalibrationFile;
  readonly tokenTrace?: ParsedTokenTraceFile;
}

export function createDashboardArtifact(
  config: DashboardRunConfig,
  output: DashboardArtifactOutput,
): DashboardArtifact {
  return createSimulationResultArtifact(
    dashboardRunKind(config),
    dashboardArtifactContracts(config),
    config,
    output,
  );
}

export function dashboardArtifactFileName(
  artifact: DashboardArtifact,
): string {
  const fingerprint = artifact.artifactFingerprint.slice("fnv1a32:".length);
  const scenario = artifact.output.summary.scenario.id.replaceAll(
    /[^a-zA-Z0-9_-]/g,
    "-",
  );
  return `inference-sim-${scenario}-${artifact.input.mode}-${fingerprint}.json`;
}

export function compareDashboardArtifact(
  artifact: DashboardArtifact,
  expected: DashboardArtifactExpectation,
): DashboardArtifactReplay {
  const inputMatches =
    artifact.inputFingerprint === expected.inputFingerprint;
  const outputMatches =
    artifact.outputFingerprint === expected.outputFingerprint;
  return {
    sourceFileName: expected.sourceFileName,
    expectedInputFingerprint: expected.inputFingerprint,
    actualInputFingerprint: artifact.inputFingerprint,
    expectedOutputFingerprint: expected.outputFingerprint,
    actualOutputFingerprint: artifact.outputFingerprint,
    expectedArtifactFingerprint: expected.artifactFingerprint,
    actualArtifactFingerprint: artifact.artifactFingerprint,
    inputMatches,
    outputMatches,
    matches: inputMatches
      && outputMatches
      && artifact.artifactFingerprint === expected.artifactFingerprint,
  };
}

export function dashboardArtifactContracts(
  config: DashboardRunConfig,
): Readonly<Record<string, number>> {
  const speculative = config.mode === "speculative"
    || (
      config.mode === "serving"
      && config.serving.decodeMode !== "target_only"
    );
  const expertCache = config.mode === "expert-cache"
    || (config.mode === "serving" && config.serving.useExpertCache);
  return {
    ...DASHBOARD_BASE_CONTRACTS,
    roofline_summary: ROOFLINE_SUMMARY_REVISION,
    ...(config.calibration === undefined
      ? {}
      : { calibration_dataset: CALIBRATION_DATASET_REVISION }),
    ...(config.modelBinding?.pipelineExecution === undefined
      ? {}
      : { pipeline_execution: PIPELINE_EXECUTION_CONTRACT_REVISION }),
    ...(config.mode === "serving"
      ? { serving_trace: SERVING_TRACE_CONTRACT_REVISION }
      : {}),
    ...(config.mode === "serving" && config.serving.useExpertCache
      ? {
          concurrent_plan_trace: CONCURRENT_PLAN_TRACE_REVISION,
          serving_expert_cache: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        }
      : {}),
    ...(speculative
      ? {
          speculative_family: SPECULATIVE_FAMILY_CONTRACT_REVISION,
          speculative_iteration: SPECULATIVE_ITERATION_CONTRACT_REVISION,
        }
      : {}),
    ...(config.mode === "speculative"
      ? { paged_kv: PAGED_KV_CONTRACT_REVISION }
      : {}),
    ...(config.mode === "speculative" && config.speculative.trace !== undefined
      ? { speculative_token_trace: SPECULATIVE_TOKEN_TRACE_REVISION }
      : {}),
    ...(expertCache
      ? { expert_cache: EXPERT_CACHE_CONTRACT_REVISION }
      : {}),
  };
}

export function parseDashboardArtifactFileText(
  text: string,
  fileName: string,
): ParsedDashboardArtifact {
  if (
    new TextEncoder().encode(text).byteLength
    > MAX_DASHBOARD_ARTIFACT_FILE_BYTES
  ) {
    throw new Error("result artifact exceeds the 128 MiB limit");
  }
  if (fileName.split(".").at(-1)?.toLowerCase() !== "json") {
    throw new Error("result artifact file must use .json");
  }
  let input: unknown;
  try {
    input = JSON.parse(text) as unknown;
  } catch (error) {
    throw new Error(`invalid result artifact JSON: ${errorMessage(error)}`);
  }
  const artifact = parseSimulationResultArtifact(input);
  const config = parseDashboardRunConfig(artifact.input);
  assertCurrentDashboardContracts(artifact.contracts, config);
  const expectedRunKind = dashboardRunKind(config);
  if (artifact.runKind !== expectedRunKind) {
    throw new Error(
      `result artifact run kind must be ${expectedRunKind}; got ${artifact.runKind}`,
    );
  }
  const calibration = config.calibration === undefined
    ? undefined
    : {
        dataset: config.calibration,
        fit: fitTopologyCostModel(config.calibration),
      };
  const tokenTrace = config.speculative.trace === undefined
    ? undefined
    : {
        trace: config.speculative.trace,
        preview: simulateSpeculativeTokenTrace(config.speculative.trace),
      };
  return {
    config,
    expectation: {
      sourceFileName: fileName,
      inputFingerprint: artifact.inputFingerprint,
      outputFingerprint: artifact.outputFingerprint,
      artifactFingerprint: artifact.artifactFingerprint,
    },
    ...(calibration === undefined ? {} : { calibration }),
    ...(tokenTrace === undefined ? {} : { tokenTrace }),
  };
}

function dashboardRunKind(config: DashboardRunConfig): string {
  const comparisonSuffix =
    config.mode === "serving" && config.serving.compareTopologies
      ? "/comparison"
      : "";
  return `dashboard/${config.mode}${comparisonSuffix}`;
}

function assertCurrentDashboardContracts(
  contracts: Readonly<Record<string, number>>,
  config: DashboardRunConfig,
): void {
  const expectedContracts = dashboardArtifactContracts(config);
  const expectedNames = Object.keys(expectedContracts).sort();
  const actualNames = Object.keys(contracts).sort();
  if (
    expectedNames.length !== actualNames.length
    || expectedNames.some((name, index) => name !== actualNames[index])
  ) {
    throw new Error(
      "result artifact contract set does not match the current dashboard",
    );
  }
  for (const name of expectedNames) {
    const expected = expectedContracts[name];
    if (contracts[name] !== expected) {
      throw new Error(
        `result artifact contract ${name} requires revision ${expected}; got ${String(contracts[name])}`,
      );
    }
  }
}

function parseDashboardRunConfig(input: unknown): DashboardRunConfig {
  try {
    const config = requireRecord(input, "artifact input");
    assertOnlyKeys(config, [
      "scenarioName",
      "multiGpuRanks",
      "multiNodeCount",
      "customScenario",
      "modelBinding",
      "mode",
      "seed",
      "calibration",
      "speculative",
      "serving",
      "expertCache",
    ], "artifact input");
    const scenarioName = requireEnum(
      config.scenarioName,
      [
        "rtx-4090-desktop",
        "rtx-5090-desktop",
        "mac-mini-m4-pro-64gb",
        "mac-studio-m3-ultra-512gb",
        "ryzen-ai-max-395-128gb",
        "cpu-only",
        "single-gpu-cpu",
        "multi-gpu",
        "gpu-npu",
        "unified-memory",
        "multi-node",
        "custom",
      ] as const,
      "artifact input scenarioName",
    );
    const multiGpuRanks = requireEnum(
      config.multiGpuRanks,
      [2, 4, 8] as const,
      "artifact input multiGpuRanks",
    );
    const multiNodeCount = config.multiNodeCount === undefined
      ? undefined
      : requireEnum(
          config.multiNodeCount,
          [2, 3, 4] as const,
          "artifact input multiNodeCount",
        );
    const customScenario = config.customScenario === undefined
      ? undefined
      : parseSimulationScenario(config.customScenario);
    if (scenarioName === "custom" && customScenario === undefined) {
      throw new Error("artifact input customScenario is required");
    }
    if (scenarioName !== "custom" && customScenario !== undefined) {
      throw new Error(
        "artifact input customScenario requires scenarioName custom",
      );
    }
    const mode = requireEnum(
      config.mode,
      ["serving", "pipeline", "speculative", "expert-cache"] as const,
      "artifact input mode",
    );
    const seed = requireInteger(config.seed, 0, 0xffff_ffff, "artifact input seed");
    const speculative = parseSpeculativeConfig(config.speculative);
    const serving = parseServingConfig(config.serving);
    const expertCache = parseExpertCacheConfig(config.expertCache);
    const modelBinding = config.modelBinding === undefined
      ? undefined
      : parseModelBinding(config.modelBinding);
    if (
      mode === "pipeline"
      && modelBinding?.pipelineExecution?.replacesTarget !== true
    ) {
      throw new Error(
        "artifact input pipeline mode requires a replacing pipeline execution",
      );
    }
    const calibration = config.calibration === undefined
      ? undefined
      : config.calibration as CalibrationDataset;
    if (calibration !== undefined) {
      fitTopologyCostModel(calibration);
    }
    return {
      scenarioName,
      multiGpuRanks,
      ...(multiNodeCount === undefined ? {} : { multiNodeCount }),
      ...(customScenario === undefined ? {} : { customScenario }),
      ...(modelBinding === undefined ? {} : { modelBinding }),
      mode,
      seed,
      speculative,
      serving,
      expertCache,
      ...(calibration === undefined ? {} : { calibration }),
    };
  } catch (error) {
    throw new Error(`invalid dashboard artifact input: ${errorMessage(error)}`);
  }
}

function parseModelBinding(
  input: unknown,
): NonNullable<DashboardRunConfig["modelBinding"]> {
  const binding = requireRecord(input, "modelBinding");
  assertOnlyKeys(binding, [
    "source",
    "displayName",
    "modelFingerprints",
    "targetModelFingerprint",
    "componentCount",
    "totalParameters",
    "weightBytes",
    "modelFormat",
    "executionProfile",
    "pipelineExecution",
    "executionCoverage",
    "pipelineStrategy",
    "speculativeFamilies",
  ], "modelBinding");
  const modelFingerprints = requireStringArray(
    binding.modelFingerprints,
    "modelBinding modelFingerprints",
  );
  if (modelFingerprints.length === 0) {
    throw new Error("modelBinding modelFingerprints must not be empty");
  }
  const speculativeFamilies = requireArray(
    binding.speculativeFamilies,
    "modelBinding speculativeFamilies",
  ).map((family) => requireFamily(
    family,
    "modelBinding speculative family",
  ));
  if (new Set(speculativeFamilies).size !== speculativeFamilies.length) {
    throw new Error(
      "modelBinding speculativeFamilies must not contain duplicates",
    );
  }
  const pipelineStrategy = binding.pipelineStrategy === undefined
    ? undefined
    : requireString(binding.pipelineStrategy, "modelBinding pipelineStrategy");
  const pipelineExecution = binding.pipelineExecution === undefined
    ? undefined
    : parsePipelineExecution(binding.pipelineExecution);
  const modelFormat = binding.modelFormat === undefined
    ? undefined
    : parseModelFormat(binding.modelFormat);
  return {
    source: requireEnum(
      binding.source,
      ["builtin_model", "local_model_package"] as const,
      "modelBinding source",
    ),
    displayName: requireString(
      binding.displayName,
      "modelBinding displayName",
    ),
    modelFingerprints,
    targetModelFingerprint: requireString(
      binding.targetModelFingerprint,
      "modelBinding targetModelFingerprint",
    ),
    componentCount: requireInteger(
      binding.componentCount,
      0,
      512,
      "modelBinding componentCount",
    ),
    totalParameters: requireInteger(
      binding.totalParameters,
      1,
      Number.MAX_SAFE_INTEGER,
      "modelBinding totalParameters",
    ),
    weightBytes: requireInteger(
      binding.weightBytes,
      1,
      Number.MAX_SAFE_INTEGER,
      "modelBinding weightBytes",
    ),
    ...(modelFormat === undefined ? {} : { modelFormat }),
    executionProfile: parseModelExecutionProfile(binding.executionProfile),
    ...(pipelineExecution === undefined ? {} : { pipelineExecution }),
    executionCoverage: parseModelExecutionCoverage(
      binding.executionCoverage,
    ),
    ...(pipelineStrategy === undefined ? {} : { pipelineStrategy }),
    speculativeFamilies,
  };
}

function parseModelFormat(
  input: unknown,
): NonNullable<
  NonNullable<DashboardRunConfig["modelBinding"]>["modelFormat"]
> {
  const format = requireRecord(input, "modelBinding modelFormat");
  assertOnlyKeys(format, [
    "weightDtypes",
    "weightQuantization",
    "kvCacheDtype",
    "activationDtype",
    "evidence",
    "runtimeDtypesDefaulted",
  ], "modelBinding modelFormat");
  const weightDtypes = requireStringArray(
    format.weightDtypes,
    "modelBinding modelFormat weightDtypes",
  );
  if (weightDtypes.length === 0) {
    throw new Error("modelBinding modelFormat weightDtypes must not be empty");
  }
  const quantTypes = [
    "fp32",
    "fp16",
    "bf16",
    "fp8",
    "int8",
    "int4",
    "int2",
    "int1",
    "nf4",
    "unknown",
  ] as const;
  return {
    weightDtypes,
    weightQuantization: requireEnum(
      format.weightQuantization,
      [
        "none",
        "fp8",
        "int8",
        "int4",
        "int2",
        "int1",
        "nf4",
        "mixed",
        "unknown",
      ] as const,
      "modelBinding modelFormat weightQuantization",
    ),
    kvCacheDtype: requireEnum(
      format.kvCacheDtype,
      quantTypes,
      "modelBinding modelFormat kvCacheDtype",
    ),
    activationDtype: requireEnum(
      format.activationDtype,
      quantTypes,
      "modelBinding modelFormat activationDtype",
    ),
    evidence: requireEnum(
      format.evidence,
      ["preset_declared", "onnx_inferred"] as const,
      "modelBinding modelFormat evidence",
    ),
    runtimeDtypesDefaulted: requireBoolean(
      format.runtimeDtypesDefaulted,
      "modelBinding modelFormat runtimeDtypesDefaulted",
    ),
  };
}

function parsePipelineExecution(
  input: unknown,
): NonNullable<
  NonNullable<DashboardRunConfig["modelBinding"]>["pipelineExecution"]
> {
  const pipeline = requireRecord(input, "modelBinding pipelineExecution");
  assertOnlyKeys(pipeline, [
    "strategyKind",
    "replacesTarget",
    "components",
    "edges",
  ], "modelBinding pipelineExecution");
  const components = requireArray(
    pipeline.components,
    "modelBinding pipelineExecution components",
  ).map((input, index) => {
    const component = requireRecord(
      input,
      `modelBinding pipelineExecution components[${index}]`,
    );
    assertOnlyKeys(component, [
      "id",
      "role",
      "phase",
      "strategyKind",
      "invocationMultiplier",
      "weightBytes",
      "devicePreference",
      "isPrimary",
      "order",
      "rerunEveryStep",
    ], `modelBinding pipelineExecution components[${index}]`);
    return {
      id: requireString(component.id, "pipeline component id"),
      role: requireString(component.role, "pipeline component role"),
      phase: requireEnum(
        component.phase,
        ["prompt_only", "every_step", "final_only", "on_demand"] as const,
        "pipeline component phase",
      ),
      strategyKind: requireString(
        component.strategyKind,
        "pipeline component strategyKind",
      ),
      invocationMultiplier: requireInteger(
        component.invocationMultiplier,
        1,
        Number.MAX_SAFE_INTEGER,
        "pipeline component invocationMultiplier",
      ),
      weightBytes: requireInteger(
        component.weightBytes,
        1,
        Number.MAX_SAFE_INTEGER,
        "pipeline component weightBytes",
      ),
      ...(component.devicePreference === undefined
        ? {}
        : {
            devicePreference: requireString(
              component.devicePreference,
              "pipeline component devicePreference",
            ),
          }),
      isPrimary: requireBoolean(component.isPrimary, "pipeline component isPrimary"),
      order: requireInteger(
        component.order,
        0,
        Number.MAX_SAFE_INTEGER,
        "pipeline component order",
      ),
      ...(component.rerunEveryStep === undefined
        ? {}
        : {
            rerunEveryStep: requireBoolean(
              component.rerunEveryStep,
              "pipeline component rerunEveryStep",
            ),
          }),
    };
  });
  const edges = requireArray(
    pipeline.edges,
    "modelBinding pipelineExecution edges",
  ).map((input, index) => {
    const edge = requireRecord(
      input,
      `modelBinding pipelineExecution edges[${index}]`,
    );
    assertOnlyKeys(edge, [
      "fromComponent",
      "toComponent",
      "deviceTransfer",
    ], `modelBinding pipelineExecution edges[${index}]`);
    return {
      fromComponent: requireString(
        edge.fromComponent,
        "pipeline edge fromComponent",
      ),
      toComponent: requireString(
        edge.toComponent,
        "pipeline edge toComponent",
      ),
      ...(edge.deviceTransfer === undefined
        ? {}
        : {
            deviceTransfer: requireBoolean(
              edge.deviceTransfer,
              "pipeline edge deviceTransfer",
            ),
      }),
    };
  });
  const componentIds = components.map((component) => component.id);
  if (new Set(componentIds).size !== componentIds.length) {
    throw new Error("modelBinding pipelineExecution component ids must be unique");
  }
  if (components.filter((component) => component.isPrimary).length !== 1) {
    throw new Error(
      "modelBinding pipelineExecution requires exactly one primary component",
    );
  }
  const known = new Set(componentIds);
  for (const edge of edges) {
    if (!known.has(edge.fromComponent) || !known.has(edge.toComponent)) {
      throw new Error(
        "modelBinding pipelineExecution edge references an unknown component",
      );
    }
  }
  assertAcyclicPipeline(componentIds, edges);
  return {
    strategyKind: requireString(
      pipeline.strategyKind,
      "pipeline strategyKind",
    ),
    replacesTarget: requireBoolean(
      pipeline.replacesTarget,
      "pipeline replacesTarget",
    ),
    components,
    edges,
  };
}

function assertAcyclicPipeline(
  componentIds: readonly string[],
  edges: readonly {
    readonly fromComponent: string;
    readonly toComponent: string;
  }[],
): void {
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (id: string): void => {
    if (visiting.has(id)) {
      throw new Error("modelBinding pipelineExecution dataflow contains a cycle");
    }
    if (visited.has(id)) {
      return;
    }
    visiting.add(id);
    for (const edge of edges) {
      if (edge.fromComponent === id && edge.toComponent !== id) {
        visit(edge.toComponent);
      }
    }
    visiting.delete(id);
    visited.add(id);
  };
  componentIds.forEach(visit);
}

function parseModelExecutionCoverage(
  input: unknown,
): NonNullable<
  DashboardRunConfig["modelBinding"]
>["executionCoverage"] {
  const coverage = requireRecord(
    input,
    "modelBinding executionCoverage",
  );
  assertOnlyKeys(coverage, [
    "fidelity",
    "scope",
    "modeledComponentIds",
    "unmodeledComponentIds",
    "limitations",
  ], "modelBinding executionCoverage");
  const modeledComponentIds = requireStringArray(
    coverage.modeledComponentIds,
    "modelBinding executionCoverage modeledComponentIds",
  );
  if (modeledComponentIds.length === 0) {
    throw new Error(
      "modelBinding executionCoverage modeledComponentIds must not be empty",
    );
  }
  const unmodeledComponentIds = requireStringArray(
    coverage.unmodeledComponentIds,
    "modelBinding executionCoverage unmodeledComponentIds",
  );
  const limitations = requireStringArray(
    coverage.limitations,
    "modelBinding executionCoverage limitations",
  );
  return {
    fidelity: requireEnum(
      coverage.fidelity,
      ["complete", "partial"] as const,
      "modelBinding executionCoverage fidelity",
    ),
    scope: requireEnum(
      coverage.scope,
      ["full_model", "target_component_only"] as const,
      "modelBinding executionCoverage scope",
    ),
    modeledComponentIds,
    unmodeledComponentIds,
    limitations,
  };
}

function parseModelExecutionProfile(
  input: unknown,
): NonNullable<
  DashboardRunConfig["modelBinding"]
>["executionProfile"] {
  const profile = requireRecord(input, "modelBinding executionProfile");
  assertOnlyKeys(profile, [
    "modelId",
    "modelName",
    "attentionWeightBytesPerToken",
    "ffnWeightBytesPerToken",
    "forwardFlopsPerToken",
  ], "modelBinding executionProfile");
  return {
    modelId: requireString(
      profile.modelId,
      "modelBinding executionProfile modelId",
    ),
    modelName: requireString(
      profile.modelName,
      "modelBinding executionProfile modelName",
    ),
    attentionWeightBytesPerToken: requireInteger(
      profile.attentionWeightBytesPerToken,
      1,
      Number.MAX_SAFE_INTEGER,
      "modelBinding executionProfile attentionWeightBytesPerToken",
    ),
    ffnWeightBytesPerToken: requireInteger(
      profile.ffnWeightBytesPerToken,
      1,
      Number.MAX_SAFE_INTEGER,
      "modelBinding executionProfile ffnWeightBytesPerToken",
    ),
    forwardFlopsPerToken: requireInteger(
      profile.forwardFlopsPerToken,
      1,
      Number.MAX_SAFE_INTEGER,
      "modelBinding executionProfile forwardFlopsPerToken",
    ),
  };
}

function parseSpeculativeConfig(
  input: unknown,
): DashboardRunConfig["speculative"] {
  const config = requireRecord(input, "speculative");
  assertOnlyKeys(config, [
    "family",
    "outputTokens",
    "draftWidth",
    "firstPositionAcceptance",
    "trace",
  ], "speculative");
  const family = requireFamily(config.family, "speculative family");
  const outputTokens = requireInteger(
    config.outputTokens,
    1,
    512,
    "speculative outputTokens",
  );
  const draftWidth = requireInteger(
    config.draftWidth,
    1,
    8,
    "speculative draftWidth",
  );
  const firstPositionAcceptance = requireNumber(
    config.firstPositionAcceptance,
    0.05,
    0.99,
    "speculative firstPositionAcceptance",
  );
  const trace = config.trace === undefined
    ? undefined
    : config.trace as SpeculativeTokenTrace;
  if (trace !== undefined) {
    simulateSpeculativeTokenTrace(trace);
    if (
      trace.family !== family
      || trace.expectedOutputTokenIds.length !== outputTokens
      || trace.maxAdditionalTokens !== draftWidth
    ) {
      throw new Error(
        "speculative trace does not match family, outputTokens, and draftWidth",
      );
    }
  }
  return {
    family,
    outputTokens,
    draftWidth,
    firstPositionAcceptance,
    ...(trace === undefined ? {} : { trace }),
  };
}

function parseServingConfig(
  input: unknown,
): DashboardRunConfig["serving"] {
  const config = requireRecord(input, "serving");
  assertOnlyKeys(config, [
    "compareTopologies",
    "useExpertCache",
    "decodeMode",
    "draftWidth",
    "firstPositionAcceptance",
    "requestCount",
    "arrivalGapUs",
    "promptTokens",
    "outputTokens",
    "maxBatchSize",
    "maxBatchTokens",
    "prefillChunkTokens",
  ], "serving");
  const decodeMode = config.decodeMode === "target_only"
    ? "target_only" as const
    : requireFamily(config.decodeMode, "serving decodeMode");
  return {
    compareTopologies: requireBoolean(
      config.compareTopologies,
      "serving compareTopologies",
    ),
    useExpertCache: requireBoolean(
      config.useExpertCache,
      "serving useExpertCache",
    ),
    decodeMode,
    draftWidth: requireInteger(
      config.draftWidth,
      1,
      8,
      "serving draftWidth",
    ),
    firstPositionAcceptance: requireNumber(
      config.firstPositionAcceptance,
      0.05,
      0.99,
      "serving firstPositionAcceptance",
    ),
    requestCount: requireInteger(
      config.requestCount,
      1,
      32,
      "serving requestCount",
    ),
    arrivalGapUs: requireInteger(
      config.arrivalGapUs,
      0,
      10_000,
      "serving arrivalGapUs",
    ),
    promptTokens: requireInteger(
      config.promptTokens,
      16,
      4096,
      "serving promptTokens",
    ),
    outputTokens: requireInteger(
      config.outputTokens,
      1,
      512,
      "serving outputTokens",
    ),
    maxBatchSize: requireInteger(
      config.maxBatchSize,
      1,
      16,
      "serving maxBatchSize",
    ),
    maxBatchTokens: requireInteger(
      config.maxBatchTokens,
      8,
      512,
      "serving maxBatchTokens",
    ),
    prefillChunkTokens: requireInteger(
      config.prefillChunkTokens,
      8,
      512,
      "serving prefillChunkTokens",
    ),
  };
}

function parseExpertCacheConfig(
  input: unknown,
): DashboardRunConfig["expertCache"] {
  const config = requireRecord(input, "expertCache");
  assertOnlyKeys(config, [
    "placementStrategy",
    "tokenCount",
    "topK",
    "expertCount",
    "hotSlots",
    "warmSlots",
    "adaptivePrefetch",
  ], "expertCache");
  const expertCount = requireInteger(
    config.expertCount,
    4,
    64,
    "expertCache expertCount",
  );
  const topK = requireInteger(
    config.topK,
    1,
    expertCount,
    "expertCache topK",
  );
  return {
    placementStrategy: requireEnum(
      config.placementStrategy,
      ["contiguous", "round_robin"] as const,
      "expertCache placementStrategy",
    ),
    tokenCount: requireInteger(
      config.tokenCount,
      1,
      512,
      "expertCache tokenCount",
    ),
    topK,
    expertCount,
    hotSlots: requireInteger(
      config.hotSlots,
      topK,
      expertCount,
      "expertCache hotSlots",
    ),
    warmSlots: requireInteger(
      config.warmSlots,
      0,
      expertCount,
      "expertCache warmSlots",
    ),
    adaptivePrefetch: requireBoolean(
      config.adaptivePrefetch,
      "expertCache adaptivePrefetch",
    ),
  };
}

function requireFamily(
  value: unknown,
  label: string,
): SpeculativeProposerFamily {
  return requireEnum(value, [
    "prompt_lookup",
    "draft_model",
    "mtp",
    "eagle3",
    "shared_kv",
    "self_speculative",
  ] as const, label);
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
  ) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function assertOnlyKeys(
  record: Record<string, unknown>,
  keys: readonly string[],
  label: string,
): void {
  const allowed = new Set(keys);
  const unknown = Object.keys(record).filter((key) => !allowed.has(key));
  const required = keys.filter((key) => (
    key !== "calibration"
    && key !== "customScenario"
    && key !== "multiNodeCount"
    && key !== "modelBinding"
    && key !== "pipelineStrategy"
    && key !== "pipelineExecution"
    && key !== "devicePreference"
    && key !== "deviceTransfer"
    && key !== "rerunEveryStep"
    && key !== "trace"
    && !(key in record)
  ));
  if (unknown.length > 0 || required.length > 0) {
    throw new Error([
      unknown.length > 0 ? `unknown keys ${unknown.sort().join(", ")}` : "",
      required.length > 0 ? `missing keys ${required.join(", ")}` : "",
    ].filter(Boolean).join("; ").replace(/^/, `${label}: `));
  }
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireStringArray(value: unknown, label: string): string[] {
  const result = requireArray(value, label).map((entry, index) => (
    requireString(entry, `${label}[${index}]`)
  ));
  if (new Set(result).size !== result.length) {
    throw new Error(`${label} must not contain duplicates`);
  }
  return result;
}

function requireEnum<T extends string | number>(
  value: unknown,
  allowed: readonly T[],
  label: string,
): T {
  if (!allowed.includes(value as T)) {
    throw new Error(`${label} is unsupported: ${String(value)}`);
  }
  return value as T;
}

function requireInteger(
  value: unknown,
  minimum: number,
  maximum: number,
  label: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value < minimum
    || value > maximum
  ) {
    throw new Error(
      `${label} must be a safe integer from ${minimum} through ${maximum}`,
    );
  }
  return value;
}

function requireNumber(
  value: unknown,
  minimum: number,
  maximum: number,
  label: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isFinite(value)
    || value < minimum
    || value > maximum
  ) {
    throw new Error(`${label} must be from ${minimum} through ${maximum}`);
  }
  return value;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be boolean`);
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
