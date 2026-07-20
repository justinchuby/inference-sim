/**
 * Model presets — built-in profiles for popular LLMs.
 */
import type { ModelProfile, LayerProfile } from "./types.js";

const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

function bytesPerParam(quant: string): number {
  switch (quant) {
    case "fp32": return 4;
    case "fp16": case "bf16": return 2;
    case "fp8": return 1;
    case "int8": return 1;
    case "int4": case "nf4": return 0.5;
    case "int2": return 0.25;
    case "int1": return 0.125;
    default: return 2;
  }
}

function buildLayers(
  numLayers: number,
  attentionBytesPerLayer: number,
  ffnBytesPerLayer: number,
  kvPerTokenPerLayer: number,
): LayerProfile[] {
  return Array.from({ length: numLayers }, (_, i) => ({
    index: i,
    attentionBytes: attentionBytesPerLayer,
    ffnBytes: ffnBytesPerLayer,
    kvCachePerToken: kvPerTokenPerLayer,
  }));
}

// KV cache per token per layer = 2 * numKVHeads * headDim * bytesPerElement
function kvPerToken(numKVHeads: number, headDim: number, kvQuant: string): number {
  return 2 * numKVHeads * headDim * bytesPerParam(kvQuant);
}

function expertBytesFromParameterBudget(
  totalParams: number,
  bytesPerParameter: number,
  denseBytes: number,
  numLayers: number,
  numExperts: number,
  hasSharedExpert: boolean,
): number {
  const expertSlotsPerLayer = numExperts + (hasSharedExpert ? 1 : 0);
  return Math.max(
    0,
    (totalParams * bytesPerParameter - denseBytes)
      / numLayers
      / expertSlotsPerLayer,
  );
}

export const MODEL_PRESETS: Record<string, (weightQuant: string, kvQuant: string) => ModelProfile> = {
  "llama-3-8b": (wq = "fp16", kvq = "fp16") => {
    const headDim = 128;
    const numLayers = 32;
    const hiddenDim = 4096;
    const numHeads = 32;
    const numKVHeads = 8;
    const intermediate = 14336;
    const bpp = bytesPerParam(wq);
    const attnBytes = (hiddenDim * hiddenDim * 4) * bpp; // Q, K, V, O
    const ffnBytes = (hiddenDim * intermediate * 3) * bpp; // gate, up, down
    return {
      name: "Llama-3-8B",
      architecture: { kind: "dense", numLayers, hiddenDim, numHeads, numKVHeads, vocabSize: 128256, intermediateSize: intermediate },
      totalParams: 8e9,
      quantization: { weights: wq as any, kvCache: kvq as any, activations: "fp16" },
      layers: buildLayers(numLayers, attnBytes, ffnBytes, kvPerToken(numKVHeads, headDim, kvq)),
      provenance: presetProvenance("llama-3-8b"),
    };
  },

  "llama-3-70b": (wq = "fp16", kvq = "fp16") => {
    const headDim = 128;
    const numLayers = 80;
    const hiddenDim = 8192;
    const numHeads = 64;
    const numKVHeads = 8;
    const intermediate = 28672;
    const bpp = bytesPerParam(wq);
    const attnBytes = (hiddenDim * headDim * (numHeads + 2 * numKVHeads) + hiddenDim * hiddenDim) * bpp;
    const ffnBytes = (hiddenDim * intermediate * 3) * bpp;
    return {
      name: "Llama-3-70B",
      architecture: { kind: "dense", numLayers, hiddenDim, numHeads, numKVHeads, vocabSize: 128256, intermediateSize: intermediate },
      totalParams: 70e9,
      quantization: { weights: wq as any, kvCache: kvq as any, activations: "fp16" },
      layers: buildLayers(numLayers, attnBytes, ffnBytes, kvPerToken(numKVHeads, headDim, kvq)),
      provenance: presetProvenance("llama-3-70b"),
    };
  },

  "mixtral-8x22b": (wq = "fp16", kvq = "fp16") => {
    const headDim = 128;
    const numLayers = 56;
    const hiddenDim = 6144;
    const numHeads = 48;
    const numKVHeads = 8;
    const intermediate = 16384;
    const totalParams = 141e9;
    const bpp = bytesPerParam(wq);
    const attnBytes = (hiddenDim * headDim * (numHeads + 2 * numKVHeads) + hiddenDim * hiddenDim) * bpp;
    const denseBytes = attnBytes * numLayers;
    const expertBytes = expertBytesFromParameterBudget(
      totalParams,
      bpp,
      denseBytes,
      numLayers,
      8,
      false,
    );
    return {
      name: "Mixtral-8x22B",
      architecture: { kind: "moe", numLayers, hiddenDim, numHeads, numKVHeads, vocabSize: 32000, intermediateSize: intermediate },
      totalParams,
      quantization: { weights: wq as any, kvCache: kvq as any, activations: "fp16" },
      layers: buildLayers(numLayers, attnBytes, 0, kvPerToken(numKVHeads, headDim, kvq)),
      moe: {
        numExperts: 8,
        activeExpertsPerToken: 2,
        expertBytesPerLayer: expertBytes,
        sharedExpertBytesPerLayer: 0,
        activationDistribution: { kind: "uniform" },
      },
      provenance: presetProvenance("mixtral-8x22b"),
    };
  },

  "deepseek-v2": (wq = "fp16", kvq = "fp16") => {
    const headDim = 128;
    const numLayers = 60;
    const hiddenDim = 5120;
    const numHeads = 128;
    const numKVHeads = 128; // MLA compressed
    const intermediate = 12288;
    const totalParams = 236e9;
    const bpp = bytesPerParam(wq);
    const attnBytes = (hiddenDim * hiddenDim * 2 + hiddenDim * 512 * 2) * bpp; // MLA
    const denseBytes = attnBytes * numLayers;
    const expertBytes = expertBytesFromParameterBudget(
      totalParams,
      bpp,
      denseBytes,
      numLayers,
      160,
      true,
    );
    return {
      name: "DeepSeek-V2",
      architecture: { kind: "moe", numLayers, hiddenDim, numHeads, numKVHeads, vocabSize: 102400, intermediateSize: intermediate },
      totalParams,
      quantization: { weights: wq as any, kvCache: kvq as any, activations: "fp16" },
      layers: buildLayers(numLayers, attnBytes, 0, kvPerToken(2, 512, kvq)), // MLA compressed KV
      moe: {
        numExperts: 160,
        activeExpertsPerToken: 6,
        expertBytesPerLayer: expertBytes,
        sharedExpertBytesPerLayer: expertBytes,
        activationDistribution: { kind: "zipf", s: 1.1 },
      },
      provenance: presetProvenance("deepseek-v2"),
    };
  },

  "qwen-3-235b": (wq = "fp16", kvq = "fp16") => {
    const headDim = 128;
    const numLayers = 94;
    const hiddenDim = 4096;
    const numHeads = 64;
    const numKVHeads = 4;
    const intermediate = 12288;
    const totalParams = 235e9;
    const bpp = bytesPerParam(wq);
    const attnBytes = (hiddenDim * headDim * (numHeads + 2 * numKVHeads) + hiddenDim * hiddenDim) * bpp;
    const denseBytes = attnBytes * numLayers;
    const expertBytes = expertBytesFromParameterBudget(
      totalParams,
      bpp,
      denseBytes,
      numLayers,
      128,
      true,
    );
    return {
      name: "Qwen3-235B-A22B",
      architecture: { kind: "moe", numLayers, hiddenDim, numHeads, numKVHeads, vocabSize: 151936, intermediateSize: intermediate },
      totalParams,
      quantization: { weights: wq as any, kvCache: kvq as any, activations: "fp16" },
      layers: buildLayers(numLayers, attnBytes, 0, kvPerToken(numKVHeads, headDim, kvq)),
      moe: {
        numExperts: 128,
        activeExpertsPerToken: 8,
        expertBytesPerLayer: expertBytes,
        sharedExpertBytesPerLayer: expertBytes,
        activationDistribution: { kind: "zipf", s: 1.05 },
      },
      provenance: presetProvenance("qwen-3-235b"),
    };
  },
};

function presetProvenance(preset: string): ModelProfile["provenance"] {
  return {
    evidence: "heuristic",
    source: `built-in preset:${preset}`,
    assumptions: [
      "Architecture constants are uncalibrated built-in estimates.",
      "MoE expert bytes are normalized to the declared total parameter count.",
    ],
  };
}

export function buildModelProfile(preset: string, weightQuant = "fp16", kvQuant = "fp16"): ModelProfile {
  const builder = MODEL_PRESETS[preset];
  if (!builder) {
    throw new Error(`Unknown model preset: ${preset}. Available: ${Object.keys(MODEL_PRESETS).join(", ")}`);
  }
  return builder(weightQuant, kvQuant);
}

export function listModelPresets(): string[] {
  return Object.keys(MODEL_PRESETS);
}
