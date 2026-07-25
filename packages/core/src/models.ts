/**
 * Model presets — architecture-derived profiles for representative LLMs.
 *
 * Every preset is declared as its published architecture and all byte figures
 * are derived from that declaration, so a preset cannot silently disagree with
 * itself. `derivedTotalParams` recomputes the parameter count from the same
 * geometry; `tests/models.test.ts` checks it against the published count.
 */
import type {
  ExpertDistribution,
  LayerProfile,
  ModelComponentPhase,
  ModelComponentProfile,
  ModelProfile,
  QuantType,
} from "./types.js";

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

// ============================================================
// Attention geometry
// ============================================================

/** Multi-head or grouped-query attention with an explicit head dimension. */
export interface GqaAttention {
  readonly kind: "gqa";
  readonly numHeads: number;
  readonly numKVHeads: number;
  readonly headDim: number;
}

/** DeepSeek-style multi-head latent attention with a compressed KV latent. */
export interface MlaAttention {
  readonly kind: "mla";
  readonly numHeads: number;
  /** 0 selects an uncompressed query projection. */
  readonly qLoraRank: number;
  readonly kvLoraRank: number;
  readonly qkNopeHeadDim: number;
  readonly qkRopeHeadDim: number;
  readonly vHeadDim: number;
}

/**
 * Gated linear / delta-rule attention. The recurrent state is constant in the
 * sequence length, so these layers contribute no per-token KV cache.
 */
export interface LinearAttention {
  readonly kind: "linear";
  readonly numKeyHeads: number;
  readonly keyHeadDim: number;
  readonly numValueHeads: number;
  readonly valueHeadDim: number;
  readonly convKernelDim: number;
}

export type AttentionGeometry = GqaAttention | MlaAttention | LinearAttention;

function attentionParams(
  hiddenDim: number,
  attention: AttentionGeometry,
): number {
  switch (attention.kind) {
    case "gqa": {
      const { numHeads, numKVHeads, headDim } = attention;
      const qkv = hiddenDim * headDim * (numHeads + 2 * numKVHeads);
      const out = numHeads * headDim * hiddenDim;
      return qkv + out;
    }
    case "mla": {
      const {
        numHeads,
        qLoraRank,
        kvLoraRank,
        qkNopeHeadDim,
        qkRopeHeadDim,
        vHeadDim,
      } = attention;
      const qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
      const query = qLoraRank > 0
        ? hiddenDim * qLoraRank + qLoraRank * numHeads * qkHeadDim
        : hiddenDim * numHeads * qkHeadDim;
      const kvDown = hiddenDim * (kvLoraRank + qkRopeHeadDim);
      const kvUp = kvLoraRank * numHeads * (qkNopeHeadDim + vHeadDim);
      const out = numHeads * vHeadDim * hiddenDim;
      return query + kvDown + kvUp + out;
    }
    case "linear": {
      const {
        numKeyHeads,
        keyHeadDim,
        numValueHeads,
        valueHeadDim,
        convKernelDim,
      } = attention;
      const keyWidth = numKeyHeads * keyHeadDim;
      const valueWidth = numValueHeads * valueHeadDim;
      const projections = hiddenDim * (2 * keyWidth + valueWidth)
        + valueWidth * hiddenDim;
      const shortConv = convKernelDim * (2 * keyWidth + valueWidth);
      // Decay and beta gates are per value head.
      const gates = hiddenDim * numValueHeads * 2;
      return projections + shortConv + gates;
    }
  }
}

/** KV cache elements cached per token, per layer. */
function kvElementsPerToken(attention: AttentionGeometry): number {
  switch (attention.kind) {
    case "gqa":
      return 2 * attention.numKVHeads * attention.headDim;
    case "mla":
      // MLA caches one compressed latent plus the shared RoPE key, not K and V.
      return attention.kvLoraRank + attention.qkRopeHeadDim;
    case "linear":
      return 0;
  }
}

/**
 * Feed-forward block. Gated (SwiGLU-style) blocks use gate, up, and down
 * projections; classic transformer blocks use up and down only.
 */
function ffnParams(
  hiddenDim: number,
  intermediateSize: number,
  gated = true,
): number {
  return (gated ? 3 : 2) * hiddenDim * intermediateSize;
}

// ============================================================
// Multimodal component geometry
// ============================================================

/** Vision transformer tower that turns pixels into decoder-visible tokens. */
export interface VisionEncoderSpec {
  readonly kind: "vit";
  readonly numLayers: number;
  readonly hiddenDim: number;
  readonly numHeads: number;
  readonly intermediateSize: number;
  /** SwiGLU towers use three matrices; classic ViT MLPs use two. */
  readonly gatedMlp: boolean;
  readonly patchSize: number;
  readonly temporalPatchSize: number;
  readonly inChannels: number;
}

/** Whisper-style convolutional front end plus a transformer encoder. */
export interface AudioEncoderSpec {
  readonly kind: "conv_transformer";
  readonly numLayers: number;
  readonly hiddenDim: number;
  readonly numHeads: number;
  readonly intermediateSize: number;
  readonly melBins: number;
  readonly convKernelSize: number;
  readonly convLayers: number;
}

/** Dense connector that maps encoder features into the decoder embedding. */
export interface ProjectorSpec {
  readonly kind: "mlp";
  readonly inputDim: number;
  readonly hiddenDim: number;
  readonly outputDim: number;
}

export type EncoderSpec = VisionEncoderSpec | AudioEncoderSpec;

function encoderLayerParams(
  hiddenDim: number,
  numHeads: number,
  intermediateSize: number,
  gatedMlp: boolean,
): number {
  // Encoder attention is full multi-head: Q, K, V, and the output projection.
  const attention = 4 * hiddenDim * hiddenDim;
  const mlp = gatedMlp
    ? 3 * hiddenDim * intermediateSize
    : 2 * hiddenDim * intermediateSize;
  return attention + mlp;
}

function encoderParams(spec: EncoderSpec): number {
  switch (spec.kind) {
    case "vit": {
      const patchEmbedding = spec.inChannels
        * spec.patchSize * spec.patchSize
        * spec.temporalPatchSize
        * spec.hiddenDim;
      return patchEmbedding + spec.numLayers * encoderLayerParams(
        spec.hiddenDim,
        spec.numHeads,
        spec.intermediateSize,
        spec.gatedMlp,
      );
    }
    case "conv_transformer": {
      // Two strided convolutions map mel frames to the model width.
      const firstConv = spec.melBins * spec.hiddenDim * spec.convKernelSize;
      const laterConvs = (spec.convLayers - 1)
        * spec.hiddenDim * spec.hiddenDim * spec.convKernelSize;
      return firstConv + laterConvs + spec.numLayers * encoderLayerParams(
        spec.hiddenDim,
        spec.numHeads,
        spec.intermediateSize,
        false,
      );
    }
  }
}

function projectorParams(spec: ProjectorSpec): number {
  return spec.inputDim * spec.hiddenDim + spec.hiddenDim * spec.outputDim;
}

// ============================================================
// Preset declarations
// ============================================================

interface MoESpec {
  readonly numExperts: number;
  readonly activeExpertsPerToken: number;
  readonly expertIntermediateSize: number;
  readonly sharedExperts: number;
  readonly sharedExpertIntermediateSize: number;
  /** Layers that carry experts; the rest use the dense FFN. */
  readonly moeLayers: number;
  readonly activationDistribution: ExpertDistribution;
}

export interface ModelSpec {
  readonly name: string;
  /** Published parameter count, used only to validate the derived geometry. */
  readonly publishedTotalParams: number;
  readonly numLayers: number;
  readonly hiddenDim: number;
  readonly vocabSize: number;
  readonly tiedEmbeddings: boolean;
  /** Attention geometry, uniform or per layer for hybrid stacks. */
  readonly attention: AttentionGeometry | ((layer: number) => AttentionGeometry);
  /** Dense FFN intermediate size; 0 for layers whose FFN is fully routed. */
  readonly intermediateSize: number | ((layer: number) => number);
  /** False selects a classic two-matrix FFN instead of a gated one. */
  readonly gatedFfn?: boolean;
  /**
   * Cross-attention added to a layer, for encoder-decoder stacks and for
   * vision adapters that attend to encoder features instead of injecting
   * tokens into the sequence.
   */
  readonly crossAttention?:
    | AttentionGeometry
    | ((layer: number) => AttentionGeometry | undefined);
  readonly moe?: MoESpec;
  readonly multimodal?: MultimodalSpec;
  readonly assumptions?: readonly string[];
}

interface MultimodalComponentSpec {
  readonly id: string;
  readonly role: string;
  readonly phase: ModelComponentPhase;
  readonly encoder?: EncoderSpec;
  readonly projector?: ProjectorSpec;
  /** Decoder tokens one media item expands into after any token merging. */
  readonly tokensPerItem?: number;
}

interface MultimodalSpec {
  readonly components: readonly MultimodalComponentSpec[];
}

function componentParams(spec: MultimodalComponentSpec): number {
  return (spec.encoder === undefined ? 0 : encoderParams(spec.encoder))
    + (spec.projector === undefined ? 0 : projectorParams(spec.projector));
}

function attentionAt(spec: ModelSpec, layer: number): AttentionGeometry {
  return typeof spec.attention === "function"
    ? spec.attention(layer)
    : spec.attention;
}

function intermediateAt(spec: ModelSpec, layer: number): number {
  return typeof spec.intermediateSize === "function"
    ? spec.intermediateSize(layer)
    : spec.intermediateSize;
}

function crossAttentionAt(
  spec: ModelSpec,
  layer: number,
): AttentionGeometry | undefined {
  return typeof spec.crossAttention === "function"
    ? spec.crossAttention(layer)
    : spec.crossAttention;
}

/** Self-attention plus any cross-attention carried by the same layer. */
function layerAttentionParams(spec: ModelSpec, layer: number): number {
  const cross = crossAttentionAt(spec, layer);
  return attentionParams(spec.hiddenDim, attentionAt(spec, layer))
    + (cross === undefined ? 0 : attentionParams(spec.hiddenDim, cross));
}

function layerFfnParams(spec: ModelSpec, layer: number): number {
  return ffnParams(
    spec.hiddenDim,
    intermediateAt(spec, layer),
    spec.gatedFfn ?? true,
  );
}

/** Attention geometry of the first full-attention layer, for reporting. */
function representativeAttention(spec: ModelSpec): AttentionGeometry {
  for (let layer = 0; layer < spec.numLayers; layer++) {
    const attention = attentionAt(spec, layer);
    if (attention.kind !== "linear") {
      return attention;
    }
  }
  return attentionAt(spec, 0);
}

function reportedHeadCounts(
  attention: AttentionGeometry,
): { numHeads: number; numKVHeads: number } {
  switch (attention.kind) {
    case "gqa":
      return { numHeads: attention.numHeads, numKVHeads: attention.numKVHeads };
    case "mla":
      return { numHeads: attention.numHeads, numKVHeads: attention.numHeads };
    case "linear":
      return {
        numHeads: attention.numValueHeads,
        numKVHeads: attention.numKeyHeads,
      };
  }
}

function expertParamsPerMoELayer(spec: ModelSpec): number {
  const moe = spec.moe;
  if (moe === undefined) {
    return 0;
  }
  return moe.numExperts * ffnParams(spec.hiddenDim, moe.expertIntermediateSize)
    + moe.sharedExperts
      * ffnParams(spec.hiddenDim, moe.sharedExpertIntermediateSize);
}

function embeddingParams(spec: ModelSpec): number {
  return spec.vocabSize * spec.hiddenDim * (spec.tiedEmbeddings ? 1 : 2);
}

/**
 * Parameter count recomputed from the declared geometry. Normalization and
 * bias terms are omitted; they are far below the tolerance of this model.
 */
export function derivedTotalParams(spec: ModelSpec): number {
  let total = embeddingParams(spec);
  for (let layer = 0; layer < spec.numLayers; layer++) {
    total += layerAttentionParams(spec, layer);
    total += layerFfnParams(spec, layer);
  }
  if (spec.moe !== undefined) {
    total += spec.moe.moeLayers * expertParamsPerMoELayer(spec);
  }
  for (const component of spec.multimodal?.components ?? []) {
    total += componentParams(component);
  }
  return total;
}

function buildComponents(
  multimodal: MultimodalSpec,
  bytesPerParameter: number,
): readonly ModelComponentProfile[] {
  return multimodal.components.map((component) => {
    const params = componentParams(component);
    return {
      id: component.id,
      role: component.role,
      phase: component.phase,
      params,
      weightBytes: params * bytesPerParameter,
      ...(component.tokensPerItem === undefined
        ? {}
        : { tokensPerItem: component.tokensPerItem }),
    };
  });
}

function presetProvenance(
  presetId: string,
  spec: ModelSpec,
  totalParams: number,
): ModelProfile["provenance"] {
  return {
    evidence: "heuristic",
    source: `built-in preset:${presetId}`,
    assumptions: [
      "Weight bytes are derived from the published architecture; normalization"
        + " and bias terms are omitted.",
      `Derived parameter count is ${totalParams.toExponential(3)} against a`
        + ` published ${spec.publishedTotalParams.toExponential(3)}.`,
      ...(spec.assumptions ?? []),
    ],
  };
}

function buildProfile(
  presetId: string,
  spec: ModelSpec,
  weightQuant: string,
  kvQuant: string,
): ModelProfile {
  const bpp = bytesPerParam(weightQuant);
  const kvBpp = bytesPerParam(kvQuant);
  const layers: LayerProfile[] = [];
  for (let index = 0; index < spec.numLayers; index++) {
    const attention = attentionAt(spec, index);
    layers.push({
      index,
      attentionBytes: layerAttentionParams(spec, index) * bpp,
      ffnBytes: layerFfnParams(spec, index) * bpp,
      kvCachePerToken: kvElementsPerToken(attention) * kvBpp,
    });
  }
  const { numHeads, numKVHeads } = reportedHeadCounts(
    representativeAttention(spec),
  );
  const totalParams = derivedTotalParams(spec);
  const profile: ModelProfile = {
    name: spec.name,
    architecture: {
      kind: spec.moe === undefined ? "dense" : "moe",
      numLayers: spec.numLayers,
      hiddenDim: spec.hiddenDim,
      numHeads,
      numKVHeads,
      vocabSize: spec.vocabSize,
      intermediateSize: spec.moe === undefined
        ? intermediateAt(spec, spec.numLayers - 1)
        : spec.moe.expertIntermediateSize,
    },
    totalParams,
    embeddingBytes: embeddingParams(spec) * bpp,
    quantization: {
      weights: weightQuant as QuantType,
      kvCache: kvQuant as QuantType,
      activations: "fp16",
    },
    layers,
    ...(spec.multimodal === undefined
      ? {}
      : { components: buildComponents(spec.multimodal, bpp) }),
    provenance: presetProvenance(presetId, spec, totalParams),
  };
  if (spec.moe === undefined) {
    return profile;
  }
  // MoE consumers apply expert bytes uniformly across every layer. Scale the
  // per-layer figure by the routed-layer fraction so the aggregate stays exact
  // for stacks whose first layers are dense.
  const moeLayerFraction = spec.moe.moeLayers / spec.numLayers;
  return {
    ...profile,
    moe: {
      numExperts: spec.moe.numExperts,
      activeExpertsPerToken: spec.moe.activeExpertsPerToken,
      expertBytesPerLayer:
        ffnParams(spec.hiddenDim, spec.moe.expertIntermediateSize)
        * bpp * moeLayerFraction,
      sharedExpertBytesPerLayer: spec.moe.sharedExperts
        * ffnParams(spec.hiddenDim, spec.moe.sharedExpertIntermediateSize)
        * bpp * moeLayerFraction,
      activationDistribution: spec.moe.activationDistribution,
    },
  };
}

const UNIFORM: ExpertDistribution = { kind: "uniform" };

const DYNAMIC_IMAGE_TOKENS =
  "Image token count is dynamic; the quoted expansion is one 512x512 image."
  + " Per-request tile counts are not modeled.";

/**
 * Qwen-style vision stack: a ViT whose patches are merged 2x2 and projected
 * into the decoder width by a two-layer MLP.
 */
function qwenVisionComponents(
  visionHiddenDim: number,
  depth: number,
  intermediateSize: number,
  decoderHiddenDim: number,
  gatedMlp = false,
  patchSize = 16,
  // Reference image edge, chosen as a multiple of patchSize * mergeSize so the
  // quoted token count is exact.
  referenceImagePx = 512,
): MultimodalSpec {
  const mergedDim = 4 * visionHiddenDim;
  return {
    components: [
      {
        id: "vision_encoder",
        role: "vision_encoder",
        phase: "prompt_only",
        encoder: {
          kind: "vit",
          numLayers: depth,
          hiddenDim: visionHiddenDim,
          numHeads: 16,
          intermediateSize,
          gatedMlp,
          patchSize,
          temporalPatchSize: 2,
          inChannels: 3,
        },
      },
      {
        id: "vision_merger",
        role: "projector",
        phase: "prompt_only",
        projector: {
          kind: "mlp",
          inputDim: mergedDim,
          hiddenDim: mergedDim,
          outputDim: decoderHiddenDim,
        },
        tokensPerItem: (referenceImagePx / patchSize / 2) ** 2,
      },
    ],
  };
}

/** True on the last layer of every `period`-layer hybrid attention group. */
function isGlobalLayer(period: number) {
  return (layer: number): boolean => layer % period === period - 1;
}

const SLIDING_WINDOW_KV_UPPER_BOUND =
  "Sliding-window attention is charged full per-token KV, so KV figures are an"
  + " upper bound beyond the local window.";

const TEXT_DECODER_ONLY =
  "The released checkpoint is natively multimodal; this profile covers the text"
  + " decoder only. Encoder and projector work is not modeled. Import the ONNX"
  + " package to simulate a full multimodal pipeline.";

const MXFP4_UNIFORM_DTYPE =
  "Released weights quantize experts to MXFP4 while attention and embeddings"
  + " stay wider; one uniform weight dtype is applied here.";

export const MODEL_SPECS: Record<string, ModelSpec> = {
  // -------------------------------------------------------- dense, on-device
  "qwen3-0.6b": {
    name: "Qwen3-0.6B",
    publishedTotalParams: 0.6e9,
    numLayers: 28,
    hiddenDim: 1024,
    vocabSize: 151936,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 16, numKVHeads: 8, headDim: 128 },
    intermediateSize: 3072,
  },

  "llama-3.2-1b": {
    name: "Llama-3.2-1B",
    publishedTotalParams: 1.24e9,
    numLayers: 16,
    hiddenDim: 2048,
    vocabSize: 128256,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 64 },
    intermediateSize: 8192,
  },

  "phi-4-mini": {
    name: "Phi-4-mini",
    publishedTotalParams: 3.8e9,
    numLayers: 32,
    hiddenDim: 3072,
    vocabSize: 200064,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 24, numKVHeads: 8, headDim: 128 },
    intermediateSize: 8192,
    assumptions: [
      "Partial rotary embedding does not change weight or KV byte extents.",
    ],
  },

  "qwen3-4b": {
    name: "Qwen3-4B",
    publishedTotalParams: 4.0e9,
    numLayers: 36,
    hiddenDim: 2560,
    vocabSize: 151936,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 9728,
  },

  // -------------------------------------------------------------- dense, mid
  "mistral-7b": {
    name: "Mistral-7B-v0.3",
    publishedTotalParams: 7.25e9,
    numLayers: 32,
    hiddenDim: 4096,
    vocabSize: 32768,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 14336,
  },

  "llama-3-8b": {
    name: "Llama-3-8B",
    publishedTotalParams: 8.03e9,
    numLayers: 32,
    hiddenDim: 4096,
    vocabSize: 128256,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 14336,
  },

  "qwen3-8b": {
    name: "Qwen3-8B",
    publishedTotalParams: 8.2e9,
    numLayers: 36,
    hiddenDim: 4096,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 12288,
  },

  "gemma-4-12b": {
    name: "Gemma-4-12B",
    publishedTotalParams: 12e9,
    numLayers: 48,
    hiddenDim: 3840,
    vocabSize: 262144,
    tiedEmbeddings: true,
    // Five sliding-window local layers per global layer. Local layers use a
    // 256-wide head; global layers use a 512-wide head and fewer KV heads.
    attention: (layer) => (isGlobalLayer(6)(layer)
      ? { kind: "gqa", numHeads: 16, numKVHeads: 1, headDim: 512 }
      : { kind: "gqa", numHeads: 16, numKVHeads: 8, headDim: 256 }),
    intermediateSize: 15360,
    multimodal: {
      components: [{
        id: "patch_embedder",
        role: "projector",
        phase: "prompt_only",
        // Encoder free: raw 16x16 patches and audio samples are projected
        // straight into the decoder embedding space.
        projector: {
          kind: "mlp",
          inputDim: 16 * 16 * 3,
          hiddenDim: 3840,
          outputDim: 3840,
        },
        tokensPerItem: 280,
      }],
    },
    assumptions: [
      SLIDING_WINDOW_KV_UPPER_BOUND,
      "This release is encoder free: image patches and audio samples are"
        + " linearly projected instead of passing through a tower.",
      "Every image costs a fixed 280 decoder tokens after 3x3 pooling.",
    ],
  },

  "phi-4": {
    name: "Phi-4-14B",
    publishedTotalParams: 14.7e9,
    numLayers: 40,
    hiddenDim: 5120,
    vocabSize: 100352,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 40, numKVHeads: 10, headDim: 128 },
    intermediateSize: 17920,
  },

  "qwen3.6-27b": {
    name: "Qwen3.6-27B",
    publishedTotalParams: 27e9,
    numLayers: 64,
    hiddenDim: 5120,
    vocabSize: 248320,
    tiedEmbeddings: false,
    // Three gated linear-attention layers per full-attention layer. Only the
    // full-attention layers hold a growing KV cache.
    attention: (layer) => (isGlobalLayer(4)(layer)
      ? { kind: "gqa", numHeads: 24, numKVHeads: 4, headDim: 256 }
      : {
          kind: "linear",
          numKeyHeads: 16,
          keyHeadDim: 128,
          numValueHeads: 48,
          valueHeadDim: 128,
          convKernelDim: 4,
        }),
    intermediateSize: 17408,
    multimodal: qwenVisionComponents(1152, 27, 4304, 5120),
    assumptions: [
      "Only the 16 full-attention layers cache KV; the 48 linear-attention"
        + " layers carry a constant-size recurrent state that is not modeled.",
      DYNAMIC_IMAGE_TOKENS,
    ],
  },

  "qwen3-32b": {
    name: "Qwen3-32B",
    publishedTotalParams: 32.8e9,
    numLayers: 64,
    hiddenDim: 5120,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 64, numKVHeads: 8, headDim: 128 },
    intermediateSize: 25600,
  },

  "gemma-4-31b": {
    name: "Gemma-4-31B",
    publishedTotalParams: 31e9,
    numLayers: 60,
    hiddenDim: 5376,
    vocabSize: 262144,
    tiedEmbeddings: true,
    attention: (layer) => (isGlobalLayer(6)(layer)
      ? { kind: "gqa", numHeads: 32, numKVHeads: 4, headDim: 512 }
      : { kind: "gqa", numHeads: 32, numKVHeads: 16, headDim: 256 }),
    intermediateSize: 21504,
    assumptions: [SLIDING_WINDOW_KV_UPPER_BOUND, TEXT_DECODER_ONLY],
  },

  "llama-3-70b": {
    name: "Llama-3-70B",
    publishedTotalParams: 70.6e9,
    numLayers: 80,
    hiddenDim: 8192,
    vocabSize: 128256,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 64, numKVHeads: 8, headDim: 128 },
    intermediateSize: 28672,
  },

  // ------------------------------------------------------------- multimodal
  "qwen3-vl-4b": {
    name: "Qwen3-VL-4B",
    publishedTotalParams: 4.4e9,
    numLayers: 36,
    hiddenDim: 2560,
    vocabSize: 151936,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 9728,
    multimodal: qwenVisionComponents(1024, 24, 4096, 2560),
    assumptions: [DYNAMIC_IMAGE_TOKENS],
  },

  "qwen2.5-vl-7b": {
    name: "Qwen2.5-VL-7B",
    publishedTotalParams: 8.29e9,
    numLayers: 28,
    hiddenDim: 3584,
    vocabSize: 152064,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 28, numKVHeads: 4, headDim: 128 },
    intermediateSize: 18944,
    // This tower uses a gated MLP and 14-pixel patches.
    multimodal: qwenVisionComponents(1280, 32, 3420, 3584, true, 14, 448),
    assumptions: [
      "Image token count is dynamic; the quoted expansion assumes a 448x448"
        + " image. Per-request tile counts are not modeled.",
      "The tower's windowed attention is charged as full attention.",
    ],
  },

  "qwen3-vl-8b": {
    name: "Qwen3-VL-8B",
    publishedTotalParams: 8.8e9,
    numLayers: 36,
    hiddenDim: 4096,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 12288,
    multimodal: qwenVisionComponents(1152, 27, 4304, 4096),
    assumptions: [DYNAMIC_IMAGE_TOKENS],
  },

  "llama-3.2-11b-vision": {
    name: "Llama-3.2-11B-Vision",
    publishedTotalParams: 10.6e9,
    numLayers: 40,
    hiddenDim: 4096,
    vocabSize: 128256,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 14336,
    // Vision features are cross-attended by eight decoder layers instead of
    // being injected into the token sequence, so they cost no prompt tokens.
    crossAttention: (layer) => ((layer - 3) % 5 === 0 && layer <= 38
      ? { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 }
      : undefined),
    multimodal: {
      components: [
        {
          id: "vision_encoder",
          role: "vision_encoder",
          phase: "prompt_only",
          encoder: {
            kind: "vit",
            numLayers: 40,
            hiddenDim: 1280,
            numHeads: 16,
            intermediateSize: 5120,
            gatedMlp: false,
            patchSize: 14,
            temporalPatchSize: 1,
            inChannels: 3,
          },
        },
        {
          id: "multi_modal_projector",
          role: "projector",
          phase: "prompt_only",
          projector: {
            kind: "mlp",
            inputDim: 7680,
            hiddenDim: 4096,
            outputDim: 4096,
          },
          // Cross-attention consumes patches directly; nothing is appended to
          // the decoder sequence.
          tokensPerItem: 0,
        },
      ],
    },
    assumptions: [
      "Vision features are cross-attended by eight decoder layers and add no"
        + " decoder tokens or KV positions.",
      "Cross-attention key and value projections are sized from the decoder"
        + " width rather than the concatenated vision feature width.",
      "Tile expansion up to four 560x560 tiles is not modeled.",
    ],
  },

  "whisper-large-v3": {
    name: "Whisper-large-v3",
    publishedTotalParams: 1.55e9,
    numLayers: 32,
    hiddenDim: 1280,
    vocabSize: 51866,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 20, numKVHeads: 20, headDim: 64 },
    intermediateSize: 5120,
    // Classic transformer decoder: ungated FFN, and every layer cross-attends
    // to the audio encoder output.
    gatedFfn: false,
    crossAttention: { kind: "gqa", numHeads: 20, numKVHeads: 20, headDim: 64 },
    multimodal: {
      components: [{
        id: "audio_encoder",
        role: "audio_encoder",
        phase: "prompt_only",
        encoder: {
          kind: "conv_transformer",
          numLayers: 32,
          hiddenDim: 1280,
          numHeads: 20,
          intermediateSize: 5120,
          melBins: 128,
          convKernelSize: 3,
          convLayers: 2,
        },
        // A 30 second window is always 1500 encoder frames.
        tokensPerItem: 1500,
      }],
    },
    assumptions: [
      "Encoder-decoder model: the encoder runs once per 30 second audio"
        + " window and its output is cross-attended, not appended to the"
        + " decoder sequence.",
      "Cross-attention KV is computed once per window; it is charged here as"
        + " per-layer weights only.",
    ],
  },

  // --------------------------------------------------------------------- MoE
  "gpt-oss-20b": {
    name: "gpt-oss-20B",
    publishedTotalParams: 20.9e9,
    numLayers: 24,
    hiddenDim: 2880,
    vocabSize: 201088,
    tiedEmbeddings: false,
    // Alternating 128-token sliding and full attention, same geometry in both.
    attention: { kind: "gqa", numHeads: 64, numKVHeads: 8, headDim: 64 },
    intermediateSize: 0,
    moe: {
      numExperts: 32,
      activeExpertsPerToken: 4,
      expertIntermediateSize: 2880,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 24,
      activationDistribution: UNIFORM,
    },
    assumptions: [SLIDING_WINDOW_KV_UPPER_BOUND, MXFP4_UNIFORM_DTYPE],
  },

  "gemma-4-26b-a4b": {
    name: "Gemma-4-26B-A4B",
    publishedTotalParams: 26e9,
    numLayers: 30,
    hiddenDim: 2816,
    vocabSize: 262144,
    tiedEmbeddings: true,
    attention: (layer) => (isGlobalLayer(6)(layer)
      ? { kind: "gqa", numHeads: 16, numKVHeads: 2, headDim: 512 }
      : { kind: "gqa", numHeads: 16, numKVHeads: 8, headDim: 256 }),
    // Gemma-4 MoE layers keep a narrow dense MLP beside the routed experts.
    intermediateSize: 2112,
    moe: {
      numExperts: 128,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 704,
      sharedExperts: 1,
      sharedExpertIntermediateSize: 2112,
      moeLayers: 30,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
    multimodal: {
      components: [{
        id: "vision_encoder",
        role: "vision_encoder",
        phase: "prompt_only",
        encoder: {
          kind: "vit",
          numLayers: 27,
          hiddenDim: 1152,
          numHeads: 16,
          intermediateSize: 4304,
          gatedMlp: false,
          patchSize: 16,
          temporalPatchSize: 1,
          inChannels: 3,
        },
        // Average pooling fixes the image at 280 soft tokens.
        tokensPerItem: 280,
      }],
    },
    assumptions: [
      SLIDING_WINDOW_KV_UPPER_BOUND,
      "Every image costs a fixed 280 decoder tokens after 3x3 pooling.",
    ],
  },

  "qwen3-30b-a3b": {
    name: "Qwen3-30B-A3B",
    publishedTotalParams: 30.5e9,
    numLayers: 48,
    hiddenDim: 2048,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 4, headDim: 128 },
    intermediateSize: 0,
    moe: {
      numExperts: 128,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 768,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 48,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
  },

  "qwen3-vl-30b-a3b": {
    name: "Qwen3-VL-30B-A3B",
    publishedTotalParams: 30.9e9,
    numLayers: 48,
    hiddenDim: 2048,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 4, headDim: 128 },
    intermediateSize: 0,
    moe: {
      numExperts: 128,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 768,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 48,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
    multimodal: qwenVisionComponents(1152, 27, 4304, 2048),
    assumptions: [DYNAMIC_IMAGE_TOKENS],
  },

  "qwen3.6-35b-a3b": {
    name: "Qwen3.6-35B-A3B",
    publishedTotalParams: 35e9,
    numLayers: 40,
    hiddenDim: 2048,
    vocabSize: 248320,
    tiedEmbeddings: false,
    attention: (layer) => (isGlobalLayer(4)(layer)
      ? { kind: "gqa", numHeads: 16, numKVHeads: 2, headDim: 256 }
      : {
          kind: "linear",
          numKeyHeads: 16,
          keyHeadDim: 128,
          numValueHeads: 32,
          valueHeadDim: 128,
          convKernelDim: 4,
        }),
    intermediateSize: 0,
    moe: {
      numExperts: 256,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 512,
      sharedExperts: 1,
      sharedExpertIntermediateSize: 512,
      moeLayers: 40,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
    multimodal: qwenVisionComponents(1152, 27, 4304, 2048),
    assumptions: [
      "Only the 10 full-attention layers cache KV; the 30 linear-attention"
        + " layers carry a constant-size recurrent state that is not modeled.",
      "Linear-attention head counts are inferred from the 27B sibling and are"
        + " not confirmed by the released configuration.",
      DYNAMIC_IMAGE_TOKENS,
    ],
  },

  "mixtral-8x7b": {
    name: "Mixtral-8x7B",
    publishedTotalParams: 46.7e9,
    numLayers: 32,
    hiddenDim: 4096,
    vocabSize: 32000,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 32, numKVHeads: 8, headDim: 128 },
    intermediateSize: 0,
    moe: {
      numExperts: 8,
      activeExpertsPerToken: 2,
      expertIntermediateSize: 14336,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 32,
      activationDistribution: UNIFORM,
    },
  },

  "gpt-oss-120b": {
    name: "gpt-oss-120B",
    publishedTotalParams: 117e9,
    numLayers: 36,
    hiddenDim: 2880,
    vocabSize: 201088,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 64, numKVHeads: 8, headDim: 64 },
    intermediateSize: 0,
    moe: {
      numExperts: 128,
      activeExpertsPerToken: 4,
      expertIntermediateSize: 2880,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 36,
      activationDistribution: UNIFORM,
    },
    assumptions: [SLIDING_WINDOW_KV_UPPER_BOUND, MXFP4_UNIFORM_DTYPE],
  },

  "mixtral-8x22b": {
    name: "Mixtral-8x22B",
    publishedTotalParams: 141e9,
    numLayers: 56,
    hiddenDim: 6144,
    vocabSize: 32000,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 48, numKVHeads: 8, headDim: 128 },
    intermediateSize: 0,
    moe: {
      numExperts: 8,
      activeExpertsPerToken: 2,
      expertIntermediateSize: 16384,
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 56,
      activationDistribution: UNIFORM,
    },
  },

  "qwen-3-235b": {
    name: "Qwen3-235B-A22B",
    publishedTotalParams: 235e9,
    numLayers: 94,
    hiddenDim: 4096,
    vocabSize: 151936,
    tiedEmbeddings: false,
    attention: { kind: "gqa", numHeads: 64, numKVHeads: 4, headDim: 128 },
    intermediateSize: 0,
    moe: {
      numExperts: 128,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 1536,
      // Qwen3-MoE has no shared expert.
      sharedExperts: 0,
      sharedExpertIntermediateSize: 0,
      moeLayers: 94,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
  },

  "deepseek-v2": {
    name: "DeepSeek-V2",
    publishedTotalParams: 236e9,
    numLayers: 60,
    hiddenDim: 5120,
    vocabSize: 102400,
    tiedEmbeddings: false,
    attention: {
      kind: "mla",
      numHeads: 128,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
    },
    // The first layer is dense; the remaining 59 are routed.
    intermediateSize: (layer) => (layer < 1 ? 12288 : 0),
    moe: {
      numExperts: 160,
      activeExpertsPerToken: 6,
      expertIntermediateSize: 1536,
      sharedExperts: 2,
      sharedExpertIntermediateSize: 1536,
      moeLayers: 59,
      activationDistribution: { kind: "zipf", s: 1.1 },
    },
  },

  "deepseek-v3": {
    name: "DeepSeek-V3",
    publishedTotalParams: 671e9,
    numLayers: 61,
    hiddenDim: 7168,
    vocabSize: 129280,
    tiedEmbeddings: false,
    attention: {
      kind: "mla",
      numHeads: 128,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
    },
    intermediateSize: (layer) => (layer < 3 ? 18432 : 0),
    moe: {
      numExperts: 256,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 2048,
      sharedExperts: 1,
      sharedExpertIntermediateSize: 2048,
      moeLayers: 58,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
  },

  "kimi-k2": {
    name: "Kimi-K2",
    publishedTotalParams: 1.03e12,
    numLayers: 61,
    hiddenDim: 7168,
    vocabSize: 163840,
    tiedEmbeddings: false,
    attention: {
      kind: "mla",
      numHeads: 64,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeHeadDim: 128,
      qkRopeHeadDim: 64,
      vHeadDim: 128,
    },
    intermediateSize: (layer) => (layer < 1 ? 18432 : 0),
    moe: {
      numExperts: 384,
      activeExpertsPerToken: 8,
      expertIntermediateSize: 2048,
      sharedExperts: 1,
      sharedExpertIntermediateSize: 2048,
      moeLayers: 60,
      activationDistribution: { kind: "zipf", s: 1.05 },
    },
  },
};

export const MODEL_PRESETS: Record<
  string,
  (weightQuant: string, kvQuant: string) => ModelProfile
> = Object.fromEntries(
  Object.entries(MODEL_SPECS).map(([presetId, spec]) => [
    presetId,
    (weightQuant = "fp16", kvQuant = "fp16") =>
      buildProfile(presetId, spec, weightQuant, kvQuant),
  ]),
);

export function buildModelProfile(
  preset: string,
  weightQuant = "fp16",
  kvQuant = "fp16",
): ModelProfile {
  const builder = MODEL_PRESETS[preset];
  if (!builder) {
    throw new Error(
      `Unknown model preset: ${preset}. Available: ${
        listModelPresets().join(", ")
      }`,
    );
  }
  return builder(weightQuant, kvQuant);
}

export function listModelPresets(): string[] {
  return Object.keys(MODEL_PRESETS);
}
