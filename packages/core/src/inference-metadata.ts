import type { SpeculativeProposerFamily } from "./speculative-workload.js";

export interface InferencePipelineComponent {
  readonly id: string;
  readonly filename: string;
  readonly type: string;
  readonly tokenizer?: string;
  readonly devicePreference?: string;
  readonly runOn?: string;
}

export interface InferencePipelineEdge {
  readonly from: string;
  readonly to: string;
  readonly fromComponent: string;
  readonly fromPort: string;
  readonly toComponent: string;
  readonly toPort: string;
  readonly dtype?: string;
  readonly deviceTransfer?: boolean;
}

export interface InferencePipelineStage {
  readonly name: string;
  readonly kind: string;
  readonly runOn?: string;
  readonly componentIds: readonly string[];
  readonly parentName?: string;
  readonly maxTokens?: number;
  readonly numSteps?: number;
  readonly startStep?: number;
  readonly numCodeGroups?: number;
  readonly bindings: Readonly<Record<string, string>>;
}

export interface InferenceMetadataSpeculativeEvidence {
  readonly family: SpeculativeProposerFamily;
  readonly source:
    | "speculative.proposal_type"
    | "speculative.method"
    | "strategy.draft.producer"
    | "model.speculative.self_speculative_depth";
  readonly value: string;
  readonly maximumDraftTokens?: number;
}

export interface InferenceMetadataSummary {
  readonly components: readonly InferencePipelineComponent[];
  readonly edges: readonly InferencePipelineEdge[];
  readonly pipelineStrategy?: string;
  readonly stages: readonly InferencePipelineStage[];
  readonly requiredCapabilities: readonly string[];
  readonly vision?: {
    readonly imagePlaceholderTokenId?: number;
    readonly tokensPerTile?: number;
  };
  readonly hardware: {
    readonly minimumMemoryGiB?: number;
    readonly minimumTensorParallelDegree?: number;
    readonly supportsTensorParallel?: boolean;
    readonly requiredDtypes: readonly string[];
    readonly beneficialDtypes: readonly string[];
    readonly kvCacheMemoryPer1kTokensMiB?: number;
  };
  readonly speculative: {
    readonly availableFamilies: readonly SpeculativeProposerFamily[];
    readonly evidence: readonly InferenceMetadataSpeculativeEvidence[];
    readonly unrecognizedDeclarations: readonly string[];
  };
  readonly warnings: readonly string[];
}

export class InferenceMetadataError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InferenceMetadataError";
  }
}

export function parseInferenceMetadata(
  input: unknown,
): InferenceMetadataSummary {
  const root = record(input, "inference metadata");
  const warnings: string[] = [];
  const pipeline = optionalRecord(root.pipeline, "pipeline");
  const parsedComponents = parseComponents(pipeline?.models);
  const componentIds = new Set(
    parsedComponents.map((component) => component.id),
  );
  const phaseRunOn = parsePhases(pipeline?.phases, componentIds);
  const components = parsedComponents.map((component) => ({
    ...component,
    ...(phaseRunOn.get(component.id) === undefined
      ? {}
      : { runOn: phaseRunOn.get(component.id)! }),
  }));
  const edges = parseEdges(pipeline?.dataflow, componentIds);
  const strategy = optionalRecord(pipeline?.strategy, "pipeline.strategy");
  if (pipeline !== undefined && strategy === undefined) {
    fail("pipeline.strategy is required when pipeline is declared");
  }
  const stages = strategy === undefined
    ? []
    : parsePipelineStrategy(strategy, "pipeline.strategy", componentIds);
  const evidence: InferenceMetadataSpeculativeEvidence[] = [];
  const unrecognizedDeclarations: string[] = [];
  parseStandaloneSpeculative(root, evidence, unrecognizedDeclarations);
  parseGenericStrategy(root.strategy, root.model, evidence, unrecognizedDeclarations);
  const availableFamilies = [...new Set(
    evidence.map((item) => item.family),
  )].sort();

  if (
    components.some((component) => component.type === "draft")
    && !availableFamilies.includes("draft_model")
  ) {
    warnings.push(
      "pipeline contains a draft component but no speculative strategy binds it",
    );
  }
  const model = optionalRecord(root.model, "model");
  const modelSpeculative = optionalRecord(
    model?.speculative,
    "model.speculative",
  );
  if (
    modelSpeculative?.has_draft_heads === true
    && evidence.length === 0
  ) {
    warnings.push(
      "model advertises draft heads without identifying a supported proposer family",
    );
  }
  if (unrecognizedDeclarations.length > 0) {
    warnings.push(
      "one or more speculative declarations cannot be mapped without guessing",
    );
  }

  return {
    components,
    edges,
    ...(strategy === undefined
      ? {}
      : {
          pipelineStrategy: nonEmptyString(
            strategy.kind,
            "pipeline.strategy.kind",
          ),
        }),
    stages,
    ...parseVision(pipeline?.vision),
    requiredCapabilities: stringArray(
      root.required_capabilities,
      "required_capabilities",
    ),
    hardware: parseHardware(root.hardware_requirements),
    speculative: {
      availableFamilies,
      evidence,
      unrecognizedDeclarations: [...new Set(unrecognizedDeclarations)].sort(),
    },
    warnings,
  };
}

function parseVision(
  input: unknown,
): { readonly vision?: InferenceMetadataSummary["vision"] } {
  const vision = optionalRecord(input, "pipeline.vision");
  if (vision === undefined) {
    return {};
  }
  const parsed = {
    ...optionalIntegerProperty(
      vision.image_placeholder_token_id,
      "pipeline.vision.image_placeholder_token_id",
      "imagePlaceholderTokenId",
      Number.MIN_SAFE_INTEGER,
    ),
    ...optionalIntegerProperty(
      vision.tokens_per_tile,
      "pipeline.vision.tokens_per_tile",
      "tokensPerTile",
      1,
    ),
  };
  return { vision: parsed };
}

function parsePhases(
  input: unknown,
  componentIds: ReadonlySet<string>,
): ReadonlyMap<string, string> {
  const phases = optionalRecord(input, "pipeline.phases");
  if (phases === undefined) {
    return new Map();
  }
  const result = new Map<string, string>();
  for (const [componentId, value] of Object.entries(phases)) {
    if (!componentIds.has(componentId)) {
      fail(`pipeline.phases references unknown component ${componentId}`);
    }
    const phase = record(value, `pipeline.phases.${componentId}`);
    const runOn = optionalPhase(
      phase.run_on,
      `pipeline.phases.${componentId}.run_on`,
    );
    if (runOn === undefined) {
      fail(`pipeline.phases.${componentId}.run_on is required`);
    }
    result.set(componentId, runOn);
  }
  return result;
}

function parseComponents(
  input: unknown,
): InferencePipelineComponent[] {
  if (input === undefined || input === null) {
    return [];
  }
  const models = record(input, "pipeline.models");
  if (Object.keys(models).length === 0) {
    fail("pipeline.models must contain at least one component");
  }
  return Object.entries(models)
    .map(([id, value]) => {
      if (id.length === 0 || id.includes(".")) {
        fail(`pipeline component id must be non-empty and contain no dot: ${id}`);
      }
      const component = record(value, `pipeline.models.${id}`);
      return {
        id,
        filename: relativePackagePath(
          component.filename,
          `pipeline.models.${id}.filename`,
        ),
        type: nonEmptyString(component.type, `pipeline.models.${id}.type`),
        ...optionalRelativePath(
          component.tokenizer,
          `pipeline.models.${id}.tokenizer`,
          "tokenizer",
        ),
        ...optionalStringProperty(
          component.device_preference,
          `pipeline.models.${id}.device_preference`,
          "devicePreference",
        ),
      };
    })
    .sort((left, right) => left.id.localeCompare(right.id));
}

function parseEdges(
  input: unknown,
  componentIds: ReadonlySet<string>,
): InferencePipelineEdge[] {
  if (input === undefined || input === null) {
    return [];
  }
  if (!Array.isArray(input)) {
    fail("pipeline.dataflow must be an array");
  }
  return input.map((value, index) => {
    const edge = record(value, `pipeline.dataflow[${index}]`);
    const from = endpoint(
      edge.from,
      `pipeline.dataflow[${index}].from`,
      componentIds,
    );
    const to = endpoint(
      edge.to,
      `pipeline.dataflow[${index}].to`,
      componentIds,
    );
    return {
      from: `${from.component}.${from.port}`,
      to: `${to.component}.${to.port}`,
      fromComponent: from.component,
      fromPort: from.port,
      toComponent: to.component,
      toPort: to.port,
      ...optionalStringProperty(
        edge.dtype,
        `pipeline.dataflow[${index}].dtype`,
        "dtype",
      ),
      ...optionalBooleanProperty(
        edge.device_transfer,
        `pipeline.dataflow[${index}].device_transfer`,
        "deviceTransfer",
      ),
    };
  });
}

function parsePipelineStrategy(
  strategy: Record<string, unknown>,
  path: string,
  componentIds: ReadonlySet<string>,
  stageName = "pipeline",
  parentName?: string,
): InferencePipelineStage[] {
  const kind = nonEmptyString(strategy.kind, `${path}.kind`);
  const referenced = ["model", "decoder", "denoiser", "outer", "inner"]
    .flatMap((field) => {
      const value = strategy[field];
      if (value === undefined || value === null) {
        return [];
      }
      const componentId = nonEmptyString(value, `${path}.${field}`);
      if (!componentIds.has(componentId)) {
        fail(`${path}.${field} references unknown component ${componentId}`);
      }
      return [componentId];
    });
  const current: InferencePipelineStage = {
    name: stageName,
    kind,
    componentIds: referenced,
    bindings: Object.fromEntries(
      ["model", "decoder", "denoiser", "outer", "inner"].flatMap((field) => (
        typeof strategy[field] === "string"
          ? [[field, strategy[field] as string] as const]
          : []
      )),
    ),
    ...(parentName === undefined ? {} : { parentName }),
    ...optionalPositiveIntegerProperty(
      strategy.max_tokens,
      `${path}.max_tokens`,
      "maxTokens",
    ),
    ...optionalPositiveIntegerProperty(
      strategy.num_steps,
      `${path}.num_steps`,
      "numSteps",
    ),
    ...optionalNonNegativeIntegerProperty(
      strategy.start_step,
      `${path}.start_step`,
      "startStep",
    ),
    ...optionalPositiveIntegerProperty(
      strategy.num_code_groups,
      `${path}.num_code_groups`,
      "numCodeGroups",
    ),
  };
  if (strategy.stages === undefined || strategy.stages === null) {
    return [current];
  }
  if (!Array.isArray(strategy.stages)) {
    fail(`${path}.stages must be an array`);
  }
  const childNames = new Set<string>();
  const children = strategy.stages.flatMap((value, index) => {
    const child = record(value, `${path}.stages[${index}]`);
    const name = nonEmptyString(child.name, `${path}.stages[${index}].name`);
    if (childNames.has(name)) {
      fail(`${path}.stages contains duplicate stage ${name}`);
    }
    childNames.add(name);
    const childStrategy = record(
      child.strategy,
      `${path}.stages[${index}].strategy`,
    );
    const parsed = parsePipelineStrategy(
      childStrategy,
      `${path}.stages[${index}].strategy`,
      componentIds,
      name,
      stageName,
    );
    const runOn = optionalPhase(
      child.run_on,
      `${path}.stages[${index}].run_on`,
    );
    return runOn === undefined
      ? parsed
      : [{ ...parsed[0], runOn }, ...parsed.slice(1)];
  });
  return [current, ...children];
}

function optionalPhase(
  input: unknown,
  path: string,
): string | undefined {
  const value = optionalString(input, path);
  if (
    value !== undefined
    && !["prompt_only", "every_step", "always", "final_only", "on_demand"]
      .includes(value)
  ) {
    fail(`${path} has unsupported phase ${value}`);
  }
  return value === "always" ? "every_step" : value;
}

function optionalPositiveIntegerProperty(
  input: unknown,
  path: string,
  key: string,
): Record<string, number> {
  const value = optionalPositiveInteger(input, path);
  return value === undefined ? {} : { [key]: value };
}

function optionalNonNegativeIntegerProperty(
  input: unknown,
  path: string,
  key: string,
): Record<string, number> {
  if (input === undefined || input === null) {
    return {};
  }
  if (!Number.isSafeInteger(input) || (input as number) < 0) {
    fail(`${path} must be a non-negative integer`);
  }
  return { [key]: input as number };
}

function parseStandaloneSpeculative(
  root: Record<string, unknown>,
  evidence: InferenceMetadataSpeculativeEvidence[],
  unrecognized: string[],
): void {
  if (root.speculative !== undefined && root.speculator_config !== undefined) {
    fail("speculative and deprecated speculator_config cannot both be present");
  }
  const field = root.speculative !== undefined
    ? "speculative"
    : root.speculator_config !== undefined
      ? "speculator_config"
      : undefined;
  if (field === undefined || root[field] === null) {
    return;
  }
  const config = record(root[field], field);
  if (config.proposal_type !== undefined && config.method !== undefined) {
    fail(`${field}.proposal_type and ${field}.method cannot both be present`);
  }
  const declarationField = config.proposal_type !== undefined
    ? "proposal_type"
    : config.method !== undefined
      ? "method"
      : undefined;
  if (declarationField === undefined) {
    fail(`${field} must declare proposal_type or method`);
  }
  const declaration = nonEmptyString(
    config[declarationField],
    `${field}.${declarationField}`,
  );
  const family = standaloneFamily(declaration);
  const maximumDraftTokens = optionalPositiveInteger(
    config.num_speculative_tokens ?? config.tokens_per_step,
    `${field}.num_speculative_tokens`,
  );
  if (
    config.num_speculative_tokens !== undefined
    && config.tokens_per_step !== undefined
  ) {
    fail(
      `${field}.num_speculative_tokens and ${field}.tokens_per_step cannot both be present`,
    );
  }
  if (family === undefined) {
    unrecognized.push(`${field}.${declarationField}=${declaration}`);
    return;
  }
  evidence.push({
    family,
    source: declarationField === "method"
      ? "speculative.method"
      : "speculative.proposal_type",
    value: declaration,
    ...(maximumDraftTokens === undefined ? {} : { maximumDraftTokens }),
  });
}

function parseGenericStrategy(
  input: unknown,
  modelInput: unknown,
  evidence: InferenceMetadataSpeculativeEvidence[],
  unrecognized: string[],
): void {
  const strategy = optionalRecord(input, "strategy");
  if (strategy !== undefined) {
    const kind = nonEmptyString(strategy.kind, "strategy.kind");
    if (kind === "speculative") {
      const draft = record(strategy.draft, "strategy.draft");
      const producer = nonEmptyString(
        draft.producer,
        "strategy.draft.producer",
      );
      const family = genericProducerFamily(producer);
      if (family === undefined) {
        unrecognized.push(`strategy.draft.producer=${producer}`);
      } else {
        evidence.push({
          family,
          source: "strategy.draft.producer",
          value: producer,
          ...optionalDraftTokens(strategy.tokens_per_step),
        });
      }
    }
  }
  const model = optionalRecord(modelInput, "model");
  const speculative = optionalRecord(model?.speculative, "model.speculative");
  const depth = optionalPositiveInteger(
    speculative?.self_speculative_depth,
    "model.speculative.self_speculative_depth",
  );
  if (depth !== undefined) {
    evidence.push({
      family: "self_speculative",
      source: "model.speculative.self_speculative_depth",
      value: String(depth),
    });
  }
}

function standaloneFamily(
  declaration: string,
): SpeculativeProposerFamily | undefined {
  switch (declaration.toLowerCase()) {
    case "mtp":
      return "mtp";
    case "eagle3":
    case "eagle-3":
      return "eagle3";
    case "shared_kv":
    case "shared-kv":
      return "shared_kv";
    default:
      return undefined;
  }
}

function genericProducerFamily(
  producer: string,
): SpeculativeProposerFamily | undefined {
  switch (producer.toLowerCase()) {
    case "draft_model":
      return "draft_model";
    case "ngram":
      return "prompt_lookup";
    case "self_speculative":
      return "self_speculative";
    default:
      return undefined;
  }
}

function parseHardware(input: unknown): InferenceMetadataSummary["hardware"] {
  const hardware = optionalRecord(input, "hardware_requirements");
  if (hardware === undefined) {
    return { requiredDtypes: [], beneficialDtypes: [] };
  }
  return {
    ...optionalNumberProperty(
      hardware.min_memory_gb,
      "hardware_requirements.min_memory_gb",
      "minimumMemoryGiB",
      0,
    ),
    ...optionalIntegerProperty(
      hardware.min_tp_degree,
      "hardware_requirements.min_tp_degree",
      "minimumTensorParallelDegree",
      1,
    ),
    ...optionalBooleanProperty(
      hardware.supports_tensor_parallel,
      "hardware_requirements.supports_tensor_parallel",
      "supportsTensorParallel",
    ),
    requiredDtypes: stringArray(
      hardware.required_dtypes,
      "hardware_requirements.required_dtypes",
    ),
    beneficialDtypes: stringArray(
      hardware.beneficial_dtypes,
      "hardware_requirements.beneficial_dtypes",
    ),
    ...optionalNumberProperty(
      hardware.kv_cache_memory_per_1k_tokens_mb,
      "hardware_requirements.kv_cache_memory_per_1k_tokens_mb",
      "kvCacheMemoryPer1kTokensMiB",
      0,
    ),
  };
}

function endpoint(
  value: unknown,
  label: string,
  componentIds: ReadonlySet<string>,
): { readonly component: string; readonly port: string } {
  const text = nonEmptyString(value, label);
  const separator = text.indexOf(".");
  if (
    separator <= 0
    || separator === text.length - 1
    || text.indexOf(".", separator + 1) !== -1
  ) {
    fail(`${label} must use component.port form`);
  }
  const component = text.slice(0, separator);
  if (!componentIds.has(component)) {
    fail(`${label} references unknown component ${component}`);
  }
  return { component, port: text.slice(separator + 1) };
}

function relativePackagePath(value: unknown, label: string): string {
  const path = nonEmptyString(value, label).replaceAll("\\", "/");
  if (
    path.startsWith("/")
    || /^[A-Za-z]:/.test(path)
    || path.split("/").some((segment) => (
      segment.length === 0 || segment === "." || segment === ".."
    ))
  ) {
    fail(`${label} must be a safe path relative to the model package`);
  }
  return path;
}

function optionalRelativePath(
  value: unknown,
  label: string,
  key: "tokenizer",
): { readonly tokenizer?: string } {
  return value === undefined || value === null
    ? {}
    : { [key]: relativePackagePath(value, label) };
}

function optionalDraftTokens(
  value: unknown,
): { readonly maximumDraftTokens?: number } {
  const parsed = optionalPositiveInteger(value, "strategy.tokens_per_step");
  return parsed === undefined ? {} : { maximumDraftTokens: parsed };
}

function stringArray(value: unknown, label: string): string[] {
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    fail(`${label} must be an array`);
  }
  const parsed = value.map((item, index) => (
    nonEmptyString(item, `${label}[${index}]`)
  ));
  if (new Set(parsed).size !== parsed.length) {
    fail(`${label} must not contain duplicates`);
  }
  return parsed;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function optionalRecord(
  value: unknown,
  label: string,
): Record<string, unknown> | undefined {
  return value === undefined || value === null ? undefined : record(value, label);
}

function nonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function optionalString(value: unknown, label: string): string | undefined {
  return value === undefined || value === null
    ? undefined
    : nonEmptyString(value, label);
}

function optionalPositiveInteger(
  value: unknown,
  label: string,
): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    fail(`${label} must be a positive safe integer`);
  }
  return value as number;
}

function optionalStringProperty<Key extends string>(
  value: unknown,
  label: string,
  key: Key,
): { readonly [K in Key]?: string } {
  const parsed = optionalString(value, label);
  return parsed === undefined ? {} : { [key]: parsed } as {
    readonly [K in Key]?: string;
  };
}

function optionalBooleanProperty<Key extends string>(
  value: unknown,
  label: string,
  key: Key,
): { readonly [K in Key]?: boolean } {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "boolean") {
    fail(`${label} must be a boolean`);
  }
  return { [key]: value } as { readonly [K in Key]?: boolean };
}

function optionalNumberProperty<Key extends string>(
  value: unknown,
  label: string,
  key: Key,
  minimum: number,
): { readonly [K in Key]?: number } {
  if (value === undefined || value === null) {
    return {};
  }
  if (
    typeof value !== "number"
    || !Number.isFinite(value)
    || value < minimum
  ) {
    fail(`${label} must be a finite number >= ${minimum}`);
  }
  return { [key]: value } as { readonly [K in Key]?: number };
}

function optionalIntegerProperty<Key extends string>(
  value: unknown,
  label: string,
  key: Key,
  minimum: number,
): { readonly [K in Key]?: number } {
  if (value === undefined || value === null) {
    return {};
  }
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    fail(`${label} must be a safe integer >= ${minimum}`);
  }
  return { [key]: value } as { readonly [K in Key]?: number };
}

function fail(message: string): never {
  throw new InferenceMetadataError(message);
}
