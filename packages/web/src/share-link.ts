import {
  BUILTIN_KV_CACHE_DTYPES,
  BUILTIN_WEIGHT_DTYPES,
  DASHBOARD_MODEL_PRESETS,
  createBuiltinModelBinding,
  type DashboardModelPreset,
} from "./model-binding.js";
import type { DashboardRunConfig } from "./types.js";
import type { QuantType } from "@inference-sim/core";

/**
 * State that cannot travel in a link, and why.
 *
 * A share link carries the choices a reader could have made themselves. It
 * cannot carry a file they imported: a custom topology, a local ONNX package,
 * a calibration dataset or a token trace are all far larger than a URL and are
 * not reconstructible from an identifier. Saying which one was dropped is the
 * difference between a link that is understood to be partial and one that
 * silently shows a different run.
 */
export interface ShareLinkOmission {
  readonly field: string;
  readonly reason: string;
}

export interface EncodedShareLink {
  /** Query string without the leading `?`. Empty when nothing differs. */
  readonly search: string;
  readonly omitted: readonly ShareLinkOmission[];
}

export interface DecodedShareLink {
  readonly config: DashboardRunConfig;
  /** Parameters that were present but unusable, with the reason. */
  readonly warnings: readonly string[];
}

const MODES: readonly DashboardRunConfig["mode"][] = [
  "serving",
  "speculative",
  "expert-cache",
  "pipeline",
  "fault",
  "co-residency",
];

const MODALITIES: readonly DashboardRunConfig["modality"][] = [
  "text",
  "image",
  "audio",
  "video",
];

const SCENARIOS: readonly DashboardRunConfig["scenarioName"][] = [
  "rtx-4090-desktop",
  "rtx-5090-desktop",
  "mac-mini-m4-pro-64gb",
  "mac-studio-m3-ultra-512gb",
  "ryzen-ai-max-395-128gb",
  "panther-lake-x9-388h-32gb",
  "arrow-lake-s-285k-64gb",
  "cpu-only",
  "single-gpu-cpu",
  "multi-gpu",
  "gpu-npu",
  "unified-memory",
  "multi-node",
];

const SPECULATIVE_FAMILIES = [
  "prompt_lookup",
  "draft_model",
  "mtp",
  "eagle3",
  "shared_kv",
  "self_speculative",
] as const;

const DECODE_MODES = ["target_only", ...SPECULATIVE_FAMILIES] as const;

/**
 * One shareable field: how to read it out of a config and put it back.
 *
 * Declared rather than hand-written per field so that adding a control cannot
 * quietly forget one half of the round trip.
 */
interface Field {
  readonly key: string;
  readonly read: (config: DashboardRunConfig) => string | undefined;
  readonly write: (
    value: string,
    config: DashboardRunConfig,
  ) => DashboardRunConfig | undefined;
}

function integerField(
  key: string,
  read: (config: DashboardRunConfig) => number,
  write: (config: DashboardRunConfig, value: number) => DashboardRunConfig,
  minimum: number,
  maximum: number,
): Field {
  return {
    key,
    read: (config) => String(read(config)),
    write: (raw, config) => {
      const value = Number(raw);
      // Bounds are the same ones the run itself enforces, so a hand-edited
      // link cannot reach a state the controls could not produce.
      if (
        !Number.isSafeInteger(value)
        || value < minimum
        || value > maximum
      ) {
        return undefined;
      }
      return write(config, value);
    },
  };
}

function fractionField(
  key: string,
  read: (config: DashboardRunConfig) => number,
  write: (config: DashboardRunConfig, value: number) => DashboardRunConfig,
): Field {
  return {
    key,
    read: (config) => read(config).toFixed(4).replace(/0+$/, "").replace(/\.$/, ""),
    write: (raw, config) => {
      const value = Number(raw);
      if (!Number.isFinite(value) || value <= 0 || value > 1) {
        return undefined;
      }
      return write(config, value);
    },
  };
}

function booleanField(
  key: string,
  read: (config: DashboardRunConfig) => boolean,
  write: (config: DashboardRunConfig, value: boolean) => DashboardRunConfig,
): Field {
  return {
    key,
    read: (config) => (read(config) ? "1" : "0"),
    write: (raw, config) => (
      raw === "1" || raw === "0"
        ? write(config, raw === "1")
        : undefined
    ),
  };
}

function enumField<T extends string>(
  key: string,
  values: readonly T[],
  read: (config: DashboardRunConfig) => T,
  write: (config: DashboardRunConfig, value: T) => DashboardRunConfig,
): Field {
  return {
    key,
    read: (config) => read(config),
    write: (raw, config) => (
      values.includes(raw as T) ? write(config, raw as T) : undefined
    ),
  };
}

const FIELDS: readonly Field[] = [
  enumField(
    "scenario",
    SCENARIOS,
    (config) => config.scenarioName as (typeof SCENARIOS)[number],
    (config, scenarioName) => ({ ...config, scenarioName }),
  ),
  integerField(
    "gpus",
    (config) => config.multiGpuRanks,
    (config, value) => ({
      ...config,
      multiGpuRanks: value as DashboardRunConfig["multiGpuRanks"],
    }),
    2,
    8,
  ),
  integerField(
    "nodes",
    (config) => config.multiNodeCount ?? 2,
    (config, value) => ({
      ...config,
      multiNodeCount: value as DashboardRunConfig["multiNodeCount"],
    }),
    2,
    4,
  ),
  enumField(
    "mode",
    MODES,
    (config) => config.mode,
    (config, mode) => ({ ...config, mode }),
  ),
  integerField(
    "seed",
    (config) => config.seed,
    (config, seed) => ({ ...config, seed }),
    0,
    0xffff_ffff,
  ),
  enumField(
    "input",
    MODALITIES,
    (config) => config.modality,
    (config, modality) => ({ ...config, modality }),
  ),
  integerField(
    "media",
    (config) => config.mediaItemsPerRequest,
    (config, mediaItemsPerRequest) => ({ ...config, mediaItemsPerRequest }),
    0,
    8,
  ),

  integerField(
    "requests",
    (config) => config.serving.requestCount,
    (config, requestCount) => ({
      ...config,
      serving: { ...config.serving, requestCount },
    }),
    1,
    128,
  ),
  integerField(
    "gap",
    (config) => config.serving.arrivalGapUs,
    (config, arrivalGapUs) => ({
      ...config,
      serving: { ...config.serving, arrivalGapUs },
    }),
    1,
    5_000_000,
  ),
  integerField(
    "prompt",
    (config) => config.serving.promptTokens,
    (config, promptTokens) => ({
      ...config,
      serving: { ...config.serving, promptTokens },
    }),
    16,
    1_048_576,
  ),
  integerField(
    "output",
    (config) => config.serving.outputTokens,
    (config, outputTokens) => ({
      ...config,
      serving: { ...config.serving, outputTokens },
    }),
    1,
    32_768,
  ),
  integerField(
    "batch",
    (config) => config.serving.maxBatchSize,
    (config, maxBatchSize) => ({
      ...config,
      serving: { ...config.serving, maxBatchSize },
    }),
    1,
    64,
  ),
  integerField(
    "batchTokens",
    (config) => config.serving.maxBatchTokens,
    (config, maxBatchTokens) => ({
      ...config,
      serving: { ...config.serving, maxBatchTokens },
    }),
    8,
    2048,
  ),
  integerField(
    "prefillChunk",
    (config) => config.serving.prefillChunkTokens,
    (config, prefillChunkTokens) => ({
      ...config,
      serving: { ...config.serving, prefillChunkTokens },
    }),
    8,
    1_048_576,
  ),
  enumField(
    "decode",
    DECODE_MODES,
    (config) => config.serving.decodeMode,
    (config, decodeMode) => ({
      ...config,
      serving: { ...config.serving, decodeMode },
    }),
  ),
  booleanField(
    "compare",
    (config) => config.serving.compareTopologies,
    (config, compareTopologies) => ({
      ...config,
      serving: { ...config.serving, compareTopologies },
    }),
  ),
  booleanField(
    "expertCache",
    (config) => config.serving.useExpertCache,
    (config, useExpertCache) => ({
      ...config,
      serving: { ...config.serving, useExpertCache },
    }),
  ),

  enumField(
    "specFamily",
    SPECULATIVE_FAMILIES,
    (config) => config.speculative.family,
    (config, family) => ({
      ...config,
      speculative: { ...config.speculative, family },
    }),
  ),
  integerField(
    "specOutput",
    (config) => config.speculative.outputTokens,
    (config, outputTokens) => ({
      ...config,
      speculative: { ...config.speculative, outputTokens },
    }),
    1,
    32_768,
  ),
  integerField(
    "specDraft",
    (config) => config.speculative.draftWidth,
    (config, draftWidth) => ({
      ...config,
      speculative: { ...config.speculative, draftWidth },
    }),
    1,
    16,
  ),
  fractionField(
    "specAccept",
    (config) => config.speculative.firstPositionAcceptance,
    (config, firstPositionAcceptance) => ({
      ...config,
      speculative: { ...config.speculative, firstPositionAcceptance },
    }),
  ),

  integerField(
    "faultAt",
    (config) => config.fault.faultAtUs,
    (config, faultAtUs) => ({ ...config, fault: { ...config.fault, faultAtUs } }),
    0,
    10_000_000,
  ),
  integerField(
    "faultQuiesce",
    (config) => config.fault.quiesceTimeoutUs,
    (config, quiesceTimeoutUs) => ({
      ...config,
      fault: { ...config.fault, quiesceTimeoutUs },
    }),
    0,
    10_000_000,
  ),
  integerField(
    "faultRuns",
    (config) => config.fault.executionCount,
    (config, executionCount) => ({
      ...config,
      fault: { ...config.fault, executionCount },
    }),
    1,
    64,
  ),
  {
    key: "faultNode",
    read: (config) => config.fault.failedNodeId,
    write: (raw, config) => (
      // Device identifiers are opaque here; the run validates them against the
      // topology it builds, so length is the only guard this layer can add.
      raw.length <= 128
        ? { ...config, fault: { ...config.fault, failedNodeId: raw } }
        : undefined
    ),
  },

  integerField(
    "crGap",
    (config) => config.coResidency.requestGapMs,
    (config, requestGapMs) => ({
      ...config,
      coResidency: { ...config.coResidency, requestGapMs },
    }),
    1,
    600_000,
  ),
  integerField(
    "crPrompt",
    (config) => config.coResidency.promptTokens,
    (config, promptTokens) => ({
      ...config,
      coResidency: { ...config.coResidency, promptTokens },
    }),
    16,
    1_048_576,
  ),
  integerField(
    "crOutput",
    (config) => config.coResidency.outputTokens,
    (config, outputTokens) => ({
      ...config,
      coResidency: { ...config.coResidency, outputTokens },
    }),
    1,
    32_768,
  ),
  {
    key: "crModels",
    read: (config) => config.coResidency.models
      .map((model) => [
        model.preset,
        model.weightDtype,
        model.contextTokens,
        model.pinned ? 1 : 0,
        model.requestCount,
      ].join(":"))
      .join(","),
    write: (raw, config) => {
      const models = raw.split(",").map((entry) => {
        const [preset, dtype, context, pinned, requests] = entry.split(":");
        if (
          preset === undefined
          || !(DASHBOARD_MODEL_PRESETS as readonly string[]).includes(preset)
          || !(BUILTIN_WEIGHT_DTYPES as readonly string[]).includes(dtype ?? "")
        ) {
          return undefined;
        }
        const contextTokens = Number(context);
        const requestCount = Number(requests);
        if (
          !Number.isSafeInteger(contextTokens)
          || contextTokens < 64
          || contextTokens > 1_048_576
          || !Number.isSafeInteger(requestCount)
          || requestCount < 1
          || requestCount > 16
          || (pinned !== "0" && pinned !== "1")
        ) {
          return undefined;
        }
        return {
          preset,
          weightDtype: dtype as QuantType,
          contextTokens,
          pinned: pinned === "1",
          requestCount,
        };
      });
      if (models.length === 0 || models.some((model) => model === undefined)) {
        return undefined;
      }
      return {
        ...config,
        coResidency: {
          ...config.coResidency,
          models: models as NonNullable<typeof models[number]>[],
        },
      };
    },
  },
];

/** Model identity, kept apart because it rebuilds a derived binding. */
function readModel(config: DashboardRunConfig): {
  readonly preset?: string;
  readonly weights?: string;
  readonly kv?: string;
} {
  const binding = config.modelBinding;
  if (binding?.source !== "builtin_model") {
    return {};
  }
  return {
    preset: binding.executionProfile.modelId,
    weights: binding.modelFormat?.weightDtypes[0],
    kv: binding.modelFormat?.kvCacheDtype,
  };
}

/**
 * Encode the choices in this configuration as a query string.
 *
 * Only what differs from the reader's defaults is written, so a link stays
 * short and says what it changes rather than restating the whole form.
 */
export function encodeDashboardShareLink(
  config: DashboardRunConfig,
  defaults: DashboardRunConfig,
): EncodedShareLink {
  const params = new URLSearchParams();
  const omitted: ShareLinkOmission[] = [];

  const model = readModel(config);
  const defaultModel = readModel(defaults);
  if (config.modelBinding !== undefined && model.preset === undefined) {
    omitted.push({
      field: "model",
      reason: "an imported local model package cannot travel in a link",
    });
  }
  // The three travel together or not at all. A format that happens to match
  // the reader's default is still meaningful once the model beside it differs,
  // so diffing them independently would drop it and silently rebind the shared
  // model at the wrong precision.
  const modelDiffers = model.preset !== defaultModel.preset
    || model.weights !== defaultModel.weights
    || model.kv !== defaultModel.kv;
  if (model.preset !== undefined && modelDiffers) {
    params.set("model", model.preset);
    if (model.weights !== undefined) {
      params.set("weights", model.weights);
    }
    if (model.kv !== undefined) {
      params.set("kv", model.kv);
    }
  }

  if (config.customScenario !== undefined) {
    omitted.push({
      field: "customScenario",
      reason: "an edited topology is larger than a URL can carry",
    });
  }
  if (config.calibration !== undefined) {
    omitted.push({
      field: "calibration",
      reason: "an imported calibration dataset cannot travel in a link",
    });
  }
  if (config.speculative.trace !== undefined) {
    omitted.push({
      field: "speculative.trace",
      reason: "an imported token trace cannot travel in a link",
    });
  }

  for (const field of FIELDS) {
    const value = field.read(config);
    const fallback = field.read(defaults);
    if (value !== undefined && value !== fallback) {
      params.set(field.key, value);
    }
  }
  return { search: params.toString(), omitted };
}

/**
 * Rebuild a configuration from a query string, starting from the reader's
 * defaults so that anything the link does not mention keeps its usual value.
 *
 * Every value is bounds-checked against the same limits the controls enforce.
 * A parameter that fails is reported and skipped rather than rejected wholesale
 * or silently coerced, so one stale key in a shared link does not discard the
 * rest of it.
 */
export function decodeDashboardShareLink(
  search: string,
  defaults: DashboardRunConfig,
): DecodedShareLink {
  const params = new URLSearchParams(search);
  const warnings: string[] = [];
  let config = defaults;

  const preset = params.get("model");
  if (preset !== null) {
    if ((DASHBOARD_MODEL_PRESETS as readonly string[]).includes(preset)) {
      const weights = params.get("weights") ?? undefined;
      const kv = params.get("kv") ?? undefined;
      const weightDtype = weights !== undefined
        && (BUILTIN_WEIGHT_DTYPES as readonly string[]).includes(weights)
        ? weights as QuantType
        : undefined;
      const kvDtype = kv !== undefined
        && (BUILTIN_KV_CACHE_DTYPES as readonly string[]).includes(kv)
        ? kv as QuantType
        : undefined;
      if (weights !== undefined && weightDtype === undefined) {
        warnings.push(`weights ${weights} is not a supported weight format`);
      }
      if (kv !== undefined && kvDtype === undefined) {
        warnings.push(`kv ${kv} is not a supported KV cache format`);
      }
      config = {
        ...config,
        modelBinding: createBuiltinModelBinding(
          preset as DashboardModelPreset,
          weightDtype,
          kvDtype,
          // Applied below by the modality field, once it has been validated.
          config.modality,
        ),
      };
    } else {
      warnings.push(`model ${preset} is not a built-in preset`);
    }
  }

  for (const field of FIELDS) {
    const raw = params.get(field.key);
    if (raw === null) {
      continue;
    }
    const next = field.write(raw, config);
    if (next === undefined) {
      warnings.push(`${field.key} ${raw} is out of range and was ignored`);
      continue;
    }
    config = next;
  }

  // The binding carries the modality, so it is rebuilt once the modality that
  // came out of the link is known and validated.
  const bound = readModel(config);
  if (bound.preset !== undefined) {
    config = {
      ...config,
      modelBinding: createBuiltinModelBinding(
        bound.preset as DashboardModelPreset,
        bound.weights as QuantType | undefined,
        bound.kv as QuantType | undefined,
        config.modality,
      ),
    };
  }
  return { config, warnings };
}
