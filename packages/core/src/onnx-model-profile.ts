import {
  parseOnnxModelManifest,
  type OnnxModelManifest,
} from "./onnx-manifest.js";
import type {
  LayerProfile,
  ModelProfile,
  QuantType,
} from "./types.js";

export interface OnnxModelProfileOptions {
  readonly kvCacheQuantization?: QuantType;
  readonly activationQuantization?: QuantType;
}

export function resolveOnnxModelProfile(
  input: OnnxModelManifest,
  options: OnnxModelProfileOptions = {},
): ModelProfile {
  const manifest = parseOnnxModelManifest(input);
  if (!manifest.profileReadiness.ready) {
    throw new Error(
      `ONNX model profile is incomplete: ${manifest.profileReadiness.missingFields.join(", ")}`,
    );
  }
  const architecture = manifest.architecture;
  const numLayers = architecture.numHiddenLayers!;
  const weightQuantization = dominantWeightQuantization(manifest);
  const kvCacheQuantization = options.kvCacheQuantization ?? "fp16";
  const activationQuantization = options.activationQuantization ?? "fp16";
  const expertBytes = architecture.numExperts === undefined
    ? 0
    : numLayers * (
        architecture.numExperts * architecture.expertBytesPerLayer!
        + architecture.sharedExpertBytesPerLayer!
      );
  if (expertBytes > manifest.totals.initializerLogicalBytes) {
    throw new Error(
      "ONNX MoE expert bytes exceed the initializer logical-byte inventory",
    );
  }
  const denseBytes = manifest.totals.initializerLogicalBytes - expertBytes;
  const layers = buildLayers(
    denseBytes,
    numLayers,
    architecture.hiddenSize!,
    architecture.intermediateSize!,
    architecture.numAttentionHeads!,
    architecture.numKeyValueHeads!,
    architecture.headDimension!,
    architecture.numExperts !== undefined,
    kvCacheQuantization,
  );
  const assumptions = [
    "Initializer logical bytes and element counts are exact package inventory.",
    "Non-expert initializer bytes are distributed across transformer layers.",
  ];
  if (architecture.numExperts === undefined) {
    assumptions.push(
      "Dense attention and FFN bytes are split by architecture-derived parameter ratios.",
    );
  } else {
    assumptions.push(
      "All non-expert initializer bytes are modeled as dense attention/router capacity.",
    );
  }
  if (options.kvCacheQuantization === undefined) {
    assumptions.push("KV-cache dtype defaults to fp16 because ONNX weights do not encode runtime KV storage.");
  }
  if (options.activationQuantization === undefined) {
    assumptions.push("Activation dtype defaults to fp16 because ONNX weights do not encode runtime activation storage.");
  }

  return {
    name: architecture.modelType ?? manifest.source.modelFileName.replace(/\.onnx$/i, ""),
    architecture: {
      kind: architecture.numExperts === undefined ? "dense" : "moe",
      numLayers,
      hiddenDim: architecture.hiddenSize!,
      numHeads: architecture.numAttentionHeads!,
      numKVHeads: architecture.numKeyValueHeads!,
      vocabSize: architecture.vocabSize!,
      intermediateSize: architecture.intermediateSize!,
    },
    totalParams: manifest.totals.initializerElements,
    quantization: {
      weights: weightQuantization,
      kvCache: kvCacheQuantization,
      activations: activationQuantization,
    },
    layers,
    ...(architecture.numExperts === undefined
      ? {}
      : {
          moe: {
            numExperts: architecture.numExperts,
            activeExpertsPerToken: architecture.activeExpertsPerToken!,
            expertBytesPerLayer: architecture.expertBytesPerLayer!,
            sharedExpertBytesPerLayer: architecture.sharedExpertBytesPerLayer!,
            activationDistribution: { kind: "uniform" as const },
          },
        }),
    provenance: {
      evidence: "heuristic",
      source: `ONNX manifest ${manifest.manifestFingerprint}`,
      assumptions,
    },
  };
}

function buildLayers(
  denseBytes: number,
  numLayers: number,
  hiddenSize: number,
  intermediateSize: number,
  numAttentionHeads: number,
  numKeyValueHeads: number,
  headDimension: number,
  isMoe: boolean,
  kvCacheQuantization: QuantType,
): LayerProfile[] {
  const layerBytes = distributeInteger(denseBytes, numLayers);
  const attentionParameters =
    hiddenSize * headDimension * (numAttentionHeads + 2 * numKeyValueHeads)
    + hiddenSize * hiddenSize;
  const ffnParameters = isMoe ? 0 : 3 * hiddenSize * intermediateSize;
  const denominator = attentionParameters + ffnParameters;
  const kvBytes = 2 * numKeyValueHeads * headDimension
    * bytesPerElement(kvCacheQuantization);
  return layerBytes.map((bytes, index) => {
    const attentionBytes = denominator === 0
      ? bytes
      : Math.round(bytes * attentionParameters / denominator);
    return {
      index,
      attentionBytes,
      ffnBytes: bytes - attentionBytes,
      kvCachePerToken: kvBytes,
    };
  });
}

function distributeInteger(total: number, buckets: number): number[] {
  const base = Math.floor(total / buckets);
  const remainder = total - base * buckets;
  return Array.from(
    { length: buckets },
    (_, index) => base + (index < remainder ? 1 : 0),
  );
}

function dominantWeightQuantization(
  manifest: OnnxModelManifest,
): QuantType {
  const bytesByType = new Map<QuantType, number>();
  for (const initializer of manifest.initializers) {
    const quant = onnxDataTypeToQuant(initializer.dataType);
    if (quant === undefined || initializer.dimensions.length < 2) {
      continue;
    }
    bytesByType.set(
      quant,
      (bytesByType.get(quant) ?? 0) + initializer.logicalByteLength,
    );
  }
  const dominant = [...bytesByType.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0];
  if (dominant === undefined) {
    throw new Error("ONNX model has no rank-2-or-higher initializer with a simulatable weight dtype");
  }
  return dominant[0];
}

function onnxDataTypeToQuant(dataType: string): QuantType | undefined {
  switch (dataType) {
    case "float": return "fp32";
    case "float16": return "fp16";
    case "bfloat16": return "bf16";
    case "float8e4m3fn":
    case "float8e4m3fnuz":
    case "float8e5m2":
    case "float8e5m2fnuz":
    case "float8e8m0":
      return "fp8";
    case "int8":
    case "uint8":
      return "int8";
    case "int4":
    case "uint4":
      return "int4";
    default:
      return undefined;
  }
}

function bytesPerElement(quant: QuantType): number {
  switch (quant) {
    case "fp32": return 4;
    case "fp16":
    case "bf16": return 2;
    case "fp8":
    case "int8": return 1;
    case "int4":
    case "nf4": return 0.5;
  }
}
