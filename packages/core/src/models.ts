/**
 * Model presets — architecture-derived profiles for representative LLMs.
 *
 * Every preset is declared as its published architecture and all byte figures
 * are derived from that declaration, so a preset cannot silently disagree with
 * itself. `derivedTotalParams` recomputes the parameter count from the same
 * geometry; `tests/models.test.ts` checks it against the published count.
 */
import type {
  DiffusionProfile,
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

/** Convolutional latent decoder, as used by a diffusion VAE. */
export interface LatentDecoderSpec {
  readonly kind: "conv_decoder";
  readonly latentChannels: number;
  readonly outputChannels: number;
  /** Channel width per resolution level, finest first. */
  readonly blockOutChannels: readonly number[];
  readonly layersPerBlock: number;
}

export type EncoderSpec =
  | VisionEncoderSpec
  | AudioEncoderSpec
  | LatentDecoderSpec;

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
    case "conv_decoder": {
      const widths = [...spec.blockOutChannels].reverse();
      const widest = widths[0]!;
      let total = 9 * spec.latentChannels * widest;
      // Middle block: two residual blocks around one self-attention.
      total += 2 * residualBlockParams(widest, widest, 0) + 4 * widest * widest;
      let previous = widest;
      for (const [index, width] of widths.entries()) {
        for (let block = 0; block <= spec.layersPerBlock; block++) {
          total += residualBlockParams(previous, width, 0);
          previous = width;
        }
        if (index < widths.length - 1) {
          total += 9 * width * width; // upsample convolution
        }
      }
      return total + 9 * widths.at(-1)! * spec.outputChannels;
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
// Diffusion denoiser geometry
// ============================================================

/**
 * One MMDiT block. Dual-stream blocks carry a separate image and text tower
 * that attend jointly; single-stream blocks fuse the two after the streams
 * merge. Both carry adaptive-layernorm modulation, which is a large enough
 * share of a denoiser to matter.
 */
export interface MmditBlockSpec {
  readonly stream: "dual" | "single";
  readonly mlpRatio: number;
  /** Modulation vectors the block conditions on, each a hidden-wide matrix. */
  readonly modulationChunks: number;
}

function mmditBlockParams(hiddenDim: number, block: MmditBlockSpec): number {
  const square = hiddenDim * hiddenDim;
  if (block.stream === "dual") {
    // Per stream: QKV and output projection, an up/down MLP, and modulation.
    const perStream = 4 + 2 * block.mlpRatio + block.modulationChunks;
    return 2 * perStream * square;
  }
  // Fused: one projection produces QKV and the MLP input, another consumes the
  // concatenated attention and MLP output.
  const fusedIn = 3 + block.mlpRatio;
  const fusedOut = 1 + block.mlpRatio;
  return (fusedIn + fusedOut + block.modulationChunks) * square;
}

/** Attention share of a block, for the attention/FFN split consumers expect. */
function mmditBlockAttentionParams(
  hiddenDim: number,
  block: MmditBlockSpec,
): number {
  const square = hiddenDim * hiddenDim;
  return block.stream === "dual" ? 8 * square : 4 * square;
}

interface DiffusionCommonSpec {
  /** VAE spatial downsample factor. */
  readonly vaeFactor: number;
  readonly defaultResolutionPx: number;
  readonly denoisingSteps: number;
  /** Classifier-free guidance doubles the denoiser batch. */
  readonly classifierFreeGuidance: boolean;
}

/** Uniform transformer denoiser over a patchified latent. */
interface MmditDenoiserSpec extends DiffusionCommonSpec {
  readonly kind: "mmdit";
  readonly dualBlocks: number;
  readonly singleBlocks: number;
  readonly mlpRatio: number;
  readonly dualModulationChunks: number;
  readonly singleModulationChunks: number;
  /** Denoiser patch size over the latent. */
  readonly patchSize: number;
}

/**
 * Convolutional encoder/decoder denoiser. Unlike a transformer stack its width
 * and attention depth change per resolution stage, so each stage is described
 * separately and becomes its own layer entry.
 */
interface UnetDenoiserSpec extends DiffusionCommonSpec {
  readonly kind: "unet";
  /** Channel width at each resolution stage, finest first. */
  readonly blockOutChannels: readonly number[];
  /** Transformer depth inside each stage's attention block. */
  readonly transformerLayersPerBlock: readonly number[];
  readonly crossAttentionDim: number;
  /** Residual blocks per stage on the down path; the up path adds one. */
  readonly resnetLayersPerBlock: number;
  /** Whether each down-path stage carries a spatial transformer. */
  readonly downBlockAttention: readonly boolean[];
  /** Same for the up path, ordered coarsest first as the model runs it. */
  readonly upBlockAttention: readonly boolean[];
  readonly latentChannels: number;
  readonly timeEmbedDim: number;
  /** Extra conditioning projected into the time embedding, if any. */
  readonly additionalEmbedDim?: number;
}

type DiffusionSpec = MmditDenoiserSpec | UnetDenoiserSpec;

/** One resolution stage of a UNet, flattened into a layer entry. */
interface UnetStage {
  readonly label: string;
  readonly width: number;
  readonly attentionParams: number;
  readonly residualParams: number;
}

function residualBlockParams(
  inChannels: number,
  outChannels: number,
  timeEmbedDim: number,
): number {
  // Two 3x3 convolutions, the time-embedding projection, and a 1x1 shortcut
  // whenever the block changes width.
  return 9 * inChannels * outChannels
    + 9 * outChannels * outChannels
    + timeEmbedDim * outChannels
    + (inChannels === outChannels ? 0 : inChannels * outChannels);
}

/** Attention weights of a spatial transformer: self-attention plus cross. */
function spatialTransformerAttentionParams(
  width: number,
  crossAttentionDim: number,
  depth: number,
): number {
  const square = width * width;
  const selfAttention = 4 * square;
  const crossAttention = 2 * square + 2 * width * crossAttentionDim;
  return depth * (selfAttention + crossAttention);
}

/** Everything else in a spatial transformer: the gated FFN and 1x1 projections. */
function spatialTransformerResidualParams(width: number, depth: number): number {
  const square = width * width;
  // GEGLU expands to 4x through a doubled gate projection, then projects back.
  return 2 * square + depth * 12 * square;
}

/**
 * Flattens a UNet into ordered stages: down path, middle, then up path. Each
 * stage keeps its own width, which is exactly what a uniform layer stack
 * cannot express.
 */
function unetStages(spec: UnetDenoiserSpec): UnetStage[] {
  const {
    blockOutChannels: widths,
    transformerLayersPerBlock: depths,
    crossAttentionDim,
    resnetLayersPerBlock,
    downBlockAttention,
    upBlockAttention,
    timeEmbedDim,
  } = spec;
  const stages: UnetStage[] = [];
  const last = widths.length - 1;

  let previous = widths[0]!;
  for (const [index, width] of widths.entries()) {
    let attention = 0;
    let residual = 0;
    for (let block = 0; block < resnetLayersPerBlock; block++) {
      residual += residualBlockParams(
        block === 0 ? previous : width,
        width,
        timeEmbedDim,
      );
      if (downBlockAttention[index]) {
        attention += spatialTransformerAttentionParams(
          width,
          crossAttentionDim,
          depths[index]!,
        );
        residual += spatialTransformerResidualParams(width, depths[index]!);
      }
    }
    previous = width;
    if (index < last) {
      residual += 9 * width * width; // strided downsample
    }
    stages.push({
      label: `down-${index}`,
      width,
      attentionParams: attention,
      residualParams: residual,
    });
  }

  const bottom = widths[last]!;
  stages.push({
    label: "mid",
    width: bottom,
    attentionParams: spatialTransformerAttentionParams(
      bottom,
      crossAttentionDim,
      depths[last]!,
    ),
    residualParams: 2 * residualBlockParams(bottom, bottom, timeEmbedDim)
      + spatialTransformerResidualParams(bottom, depths[last]!),
  });

  const upWidths = [...widths].reverse();
  const upDepths = [...depths].reverse();
  for (const [index, width] of upWidths.entries()) {
    let attention = 0;
    let residual = 0;
    // The up path takes one extra block to consume the skip connection, and
    // every block concatenates a skip of the same width.
    const coarser = upWidths[Math.min(index + 1, upWidths.length - 1)]!;
    for (let block = 0; block <= resnetLayersPerBlock; block++) {
      const skipWidth = block < resnetLayersPerBlock ? width : coarser;
      residual += residualBlockParams(width + skipWidth, width, timeEmbedDim);
      if (upBlockAttention[index]) {
        attention += spatialTransformerAttentionParams(
          width,
          crossAttentionDim,
          upDepths[index]!,
        );
        residual += spatialTransformerResidualParams(width, upDepths[index]!);
      }
    }
    if (index < upWidths.length - 1) {
      residual += 9 * width * width; // upsample convolution
    }
    stages.push({
      label: `up-${index}`,
      width,
      attentionParams: attention,
      residualParams: residual,
    });
  }
  return stages;
}

/** Stem weights that sit outside the resolution stages. */
function unetStemParams(spec: UnetDenoiserSpec): number {
  const first = spec.blockOutChannels[0]!;
  const timeEmbedding = first * spec.timeEmbedDim
    + spec.timeEmbedDim * spec.timeEmbedDim;
  const additional = spec.additionalEmbedDim === undefined
    ? 0
    : spec.additionalEmbedDim * spec.timeEmbedDim
      + spec.timeEmbedDim * spec.timeEmbedDim;
  return 9 * spec.latentChannels * first
    + 9 * first * spec.latentChannels
    + timeEmbedding
    + additional;
}

function diffusionBlockAt(
  spec: MmditDenoiserSpec,
  layer: number,
): MmditBlockSpec {
  return layer < spec.dualBlocks
    ? {
        stream: "dual",
        mlpRatio: spec.mlpRatio,
        modulationChunks: spec.dualModulationChunks,
      }
    : {
        stream: "single",
        mlpRatio: spec.mlpRatio,
        modulationChunks: spec.singleModulationChunks,
      };
}

/** Latent positions the denoiser attends over at the default resolution. */
export function diffusionLatentTokens(spec: DiffusionSpec): number {
  const latentSide = spec.defaultResolutionPx / spec.vaeFactor;
  if (spec.kind === "mmdit") {
    const side = latentSide / spec.patchSize;
    return Math.round(side * side);
  }
  // A UNet attends at several resolutions. Report the finest one that carries
  // attention, which is the largest and dominates the cost.
  const firstAttentionStage = spec.downBlockAttention.indexOf(true);
  const stride = 2 ** (firstAttentionStage < 0 ? 0 : firstAttentionStage);
  const side = latentSide / stride;
  return Math.round(side * side);
}

/** Layer entries a spec produces; a UNet derives them from its stages. */
export function specLayerCount(spec: ModelSpec): number {
  return spec.diffusion?.kind === "unet"
    ? unetStages(spec.diffusion).length
    : spec.numLayers ?? 0;
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
  /** Omitted by stacks that derive their layer count, such as a UNet. */
  readonly numLayers?: number;
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
  /**
   * Present for image-generation models. A diffusion denoiser is not
   * autoregressive: it caches no KV, has no vocabulary, and runs once per
   * denoising step over a fixed latent grid.
   */
  readonly diffusion?: DiffusionSpec;
  readonly assumptions?: readonly string[];
}

interface MultimodalComponentSpec {
  readonly id: string;
  readonly role: string;
  readonly phase: ModelComponentPhase;
  readonly encoder?: EncoderSpec;
  readonly projector?: ProjectorSpec;
  /** Token embedding table of a text tower, as vocabulary times width. */
  readonly tokenEmbedding?: { readonly vocabSize: number; readonly width: number };
  /** Decoder tokens one media item expands into after any token merging. */
  readonly tokensPerItem?: number;
}

interface MultimodalSpec {
  readonly components: readonly MultimodalComponentSpec[];
}

function componentParams(spec: MultimodalComponentSpec): number {
  return (spec.encoder === undefined ? 0 : encoderParams(spec.encoder))
    + (spec.projector === undefined ? 0 : projectorParams(spec.projector))
    + (spec.tokenEmbedding === undefined
      ? 0
      : spec.tokenEmbedding.vocabSize * spec.tokenEmbedding.width);
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
  if (spec.diffusion?.kind === "unet") {
    return unetStages(spec.diffusion)[layer]!.attentionParams;
  }
  if (spec.diffusion !== undefined) {
    return mmditBlockAttentionParams(
      spec.hiddenDim,
      diffusionBlockAt(spec.diffusion, layer),
    );
  }
  const cross = crossAttentionAt(spec, layer);
  return attentionParams(spec.hiddenDim, attentionAt(spec, layer))
    + (cross === undefined ? 0 : attentionParams(spec.hiddenDim, cross));
}

function layerFfnParams(spec: ModelSpec, layer: number): number {
  if (spec.diffusion?.kind === "unet") {
    const stages = unetStages(spec.diffusion);
    // The stem is not a stage, so it rides along with the first one.
    return stages[layer]!.residualParams
      + (layer === 0 ? unetStemParams(spec.diffusion) : 0);
  }
  if (spec.diffusion !== undefined) {
    const block = diffusionBlockAt(spec.diffusion, layer);
    return mmditBlockParams(spec.hiddenDim, block)
      - mmditBlockAttentionParams(spec.hiddenDim, block);
  }
  return ffnParams(
    spec.hiddenDim,
    intermediateAt(spec, layer),
    spec.gatedFfn ?? true,
  );
}

/** Attention geometry of the first full-attention layer, for reporting. */
function representativeAttention(spec: ModelSpec): AttentionGeometry {
  for (let layer = 0; layer < specLayerCount(spec); layer++) {
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
  for (let layer = 0; layer < specLayerCount(spec); layer++) {
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

function buildDiffusionProfile(spec: DiffusionSpec): DiffusionProfile {
  const batchPerStep = spec.classifierFreeGuidance ? 2 : 1;
  return {
    denoisingSteps: spec.denoisingSteps,
    latentTokens: diffusionLatentTokens(spec),
    defaultResolutionPx: spec.defaultResolutionPx,
    classifierFreeGuidance: spec.classifierFreeGuidance,
    denoiserInvocations: spec.denoisingSteps * batchPerStep,
  };
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
  for (let index = 0; index < specLayerCount(spec); index++) {
    const attention = attentionAt(spec, index);
    layers.push({
      index,
      attentionBytes: layerAttentionParams(spec, index) * bpp,
      ffnBytes: layerFfnParams(spec, index) * bpp,
      // A denoiser is not autoregressive, so it caches nothing per token.
      kvCachePerToken: spec.diffusion === undefined
        ? kvElementsPerToken(attention) * kvBpp
        : 0,
    });
  }
  const { numHeads, numKVHeads } = reportedHeadCounts(
    representativeAttention(spec),
  );
  const totalParams = derivedTotalParams(spec);
  const profile: ModelProfile = {
    name: spec.name,
    architecture: {
      kind: spec.diffusion !== undefined
        ? "diffusion"
        : spec.moe === undefined ? "dense" : "moe",
      numLayers: specLayerCount(spec),
      hiddenDim: spec.hiddenDim,
      numHeads,
      numKVHeads,
      vocabSize: spec.vocabSize,
      intermediateSize: spec.moe === undefined
        ? intermediateAt(spec, specLayerCount(spec) - 1)
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
    ...(spec.diffusion === undefined
      ? {}
      : { diffusion: buildDiffusionProfile(spec.diffusion) }),
    provenance: presetProvenance(presetId, spec, totalParams),
  };
  if (spec.moe === undefined) {
    return profile;
  }
  // MoE consumers apply expert bytes uniformly across every layer. Scale the
  // per-layer figure by the routed-layer fraction so the aggregate stays exact
  // for stacks whose first layers are dense.
  const moeLayerFraction = spec.moe.moeLayers / specLayerCount(spec);
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

/** CLIP ViT-L/14 text tower, the conditioning encoder every SD-family model uses. */
const CLIP_L_ENCODER: MultimodalComponentSpec = {
  id: "text_encoder_clip_l",
  role: "text_encoder",
  phase: "prompt_only",
  encoder: {
    kind: "vit",
    numLayers: 12,
    hiddenDim: 768,
    numHeads: 12,
    intermediateSize: 3072,
    gatedMlp: false,
    // A text tower has no patch embedding; the token embedding is charged
    // through the projector entry instead.
    patchSize: 0,
    temporalPatchSize: 0,
    inChannels: 0,
  },
  tokenEmbedding: { vocabSize: 49408, width: 768 },
};

/** T5 v1.1 XXL encoder, gated-GELU. */
const T5_XXL_ENCODER: MultimodalComponentSpec = {
  id: "text_encoder_t5xxl",
  role: "text_encoder",
  phase: "prompt_only",
  encoder: {
    kind: "vit",
    numLayers: 24,
    hiddenDim: 4096,
    numHeads: 64,
    intermediateSize: 10240,
    gatedMlp: true,
    patchSize: 0,
    temporalPatchSize: 0,
    inChannels: 0,
  },
  tokenEmbedding: { vocabSize: 32128, width: 4096 },
};

const OPENCLIP_BIGG_ENCODER: MultimodalComponentSpec = {
  id: "text_encoder_openclip_bigg",
  role: "text_encoder",
  phase: "prompt_only",
  encoder: {
    kind: "vit",
    numLayers: 32,
    hiddenDim: 1280,
    numHeads: 20,
    intermediateSize: 5120,
    gatedMlp: false,
    patchSize: 0,
    temporalPatchSize: 0,
    inChannels: 0,
  },
  tokenEmbedding: { vocabSize: 49408, width: 1280 },
};

/**
 * Latent decoder, run once at the end of a generation. Only the decoder half
 * of the autoencoder runs during generation; the encoder is for inversion.
 */
function vaeDecoder(latentChannels: number): MultimodalComponentSpec {
  return {
    id: "vae_decoder",
    role: "vae_decoder",
    phase: "final_only",
    encoder: {
      kind: "conv_decoder",
      latentChannels,
      outputChannels: 3,
      blockOutChannels: [128, 256, 512, 512],
      layersPerBlock: 2,
    },
  };
}

const DIFFUSION_UNMODELED =
  "Scheduler arithmetic, guidance embedding, and VAE tiling are not modeled;"
  + " the denoiser and its per-step invocation count are.";

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
        // A 30 second window is always 1500 encoder frames, but the decoder
        // cross-attends them instead of reading them as positions, so they
        // expand the prompt by nothing. tokensPerItem counts decoder tokens.
        tokensPerItem: 0,
      }],
    },
    assumptions: [
      "Encoder-decoder model: the encoder runs once per 30 second audio"
        + " window and its output is cross-attended, not appended to the"
        + " decoder sequence.",
      "A 30 second window is 1500 encoder frames. They drive encoder work but"
        + " add no decoder positions, so they cost no prefill or KV.",
      "Cross-attention KV is computed once per window; it is charged here as"
        + " per-layer weights only.",
    ],
  },

  // ------------------------------------------------------- image generation
  "flux-1-schnell": {
    name: "FLUX.1-schnell",
    publishedTotalParams: 17e9,
    numLayers: 57,
    hiddenDim: 3072,
    // A denoiser has no vocabulary; conditioning arrives from the text towers.
    vocabSize: 0,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 24, numKVHeads: 24, headDim: 128 },
    intermediateSize: 0,
    diffusion: {
      kind: "mmdit",
      dualBlocks: 19,
      singleBlocks: 38,
      mlpRatio: 4,
      dualModulationChunks: 6,
      singleModulationChunks: 3,
      vaeFactor: 8,
      patchSize: 2,
      defaultResolutionPx: 1024,
      denoisingSteps: 4,
      classifierFreeGuidance: false,
    },
    multimodal: {
      components: [CLIP_L_ENCODER, T5_XXL_ENCODER, vaeDecoder(16)],
    },
    assumptions: [
      "Timestep-distilled: four denoising steps and no classifier-free"
        + " guidance, so the denoiser runs once per step.",
      DIFFUSION_UNMODELED,
    ],
  },

  "flux-1-dev": {
    name: "FLUX.1-dev",
    publishedTotalParams: 17e9,
    numLayers: 57,
    hiddenDim: 3072,
    vocabSize: 0,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 24, numKVHeads: 24, headDim: 128 },
    intermediateSize: 0,
    diffusion: {
      kind: "mmdit",
      dualBlocks: 19,
      singleBlocks: 38,
      mlpRatio: 4,
      dualModulationChunks: 6,
      singleModulationChunks: 3,
      vaeFactor: 8,
      patchSize: 2,
      defaultResolutionPx: 1024,
      denoisingSteps: 50,
      classifierFreeGuidance: false,
    },
    multimodal: {
      components: [CLIP_L_ENCODER, T5_XXL_ENCODER, vaeDecoder(16)],
    },
    assumptions: [
      "Guidance-distilled: guidance is embedded in the forward pass, so the"
        + " denoiser runs once per step rather than twice.",
      DIFFUSION_UNMODELED,
    ],
  },

  "stable-diffusion-1.5": {
    name: "Stable-Diffusion-1.5",
    publishedTotalParams: 1.07e9,
    // Widest stage, for reporting. A UNet has no single width.
    hiddenDim: 1280,
    vocabSize: 0,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 8, numKVHeads: 8, headDim: 40 },
    intermediateSize: 0,
    diffusion: {
      kind: "unet",
      blockOutChannels: [320, 640, 1280, 1280],
      transformerLayersPerBlock: [1, 1, 1, 1],
      crossAttentionDim: 768,
      resnetLayersPerBlock: 2,
      downBlockAttention: [true, true, true, false],
      upBlockAttention: [false, true, true, true],
      latentChannels: 4,
      timeEmbedDim: 1280,
      vaeFactor: 8,
      defaultResolutionPx: 512,
      denoisingSteps: 20,
      classifierFreeGuidance: true,
    },
    multimodal: {
      components: [CLIP_L_ENCODER, vaeDecoder(4)],
    },
    assumptions: [
      "Classifier-free guidance doubles the denoiser batch at every step.",
      "Attention runs at several resolutions; the reported latent grid is the"
        + " finest one, which dominates the cost.",
      DIFFUSION_UNMODELED,
    ],
  },

  "stable-diffusion-xl": {
    name: "Stable-Diffusion-XL",
    publishedTotalParams: 3.5e9,
    hiddenDim: 1280,
    vocabSize: 0,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 20, numKVHeads: 20, headDim: 64 },
    intermediateSize: 0,
    diffusion: {
      kind: "unet",
      blockOutChannels: [320, 640, 1280],
      // The deepest stage carries ten transformer layers per block.
      transformerLayersPerBlock: [1, 2, 10],
      crossAttentionDim: 2048,
      resnetLayersPerBlock: 2,
      downBlockAttention: [false, true, true],
      upBlockAttention: [true, true, false],
      latentChannels: 4,
      timeEmbedDim: 1280,
      // Pooled text plus the size and crop conditioning SDXL adds.
      additionalEmbedDim: 2816,
      vaeFactor: 8,
      defaultResolutionPx: 1024,
      denoisingSteps: 30,
      classifierFreeGuidance: true,
    },
    multimodal: {
      components: [CLIP_L_ENCODER, OPENCLIP_BIGG_ENCODER, vaeDecoder(4)],
    },
    assumptions: [
      "Classifier-free guidance doubles the denoiser batch at every step.",
      "Attention runs at several resolutions; the reported latent grid is the"
        + " finest one, which dominates the cost.",
      DIFFUSION_UNMODELED,
    ],
  },

  "stable-diffusion-3.5-large": {
    name: "Stable-Diffusion-3.5-Large",
    publishedTotalParams: 13.7e9,
    numLayers: 38,
    hiddenDim: 2432,
    vocabSize: 0,
    tiedEmbeddings: true,
    attention: { kind: "gqa", numHeads: 38, numKVHeads: 38, headDim: 64 },
    intermediateSize: 0,
    diffusion: {
      kind: "mmdit",
      dualBlocks: 38,
      singleBlocks: 0,
      mlpRatio: 4,
      dualModulationChunks: 6,
      singleModulationChunks: 3,
      vaeFactor: 8,
      patchSize: 2,
      defaultResolutionPx: 1024,
      denoisingSteps: 40,
      classifierFreeGuidance: true,
    },
    multimodal: {
      components: [
        CLIP_L_ENCODER,
        OPENCLIP_BIGG_ENCODER,
        T5_XXL_ENCODER,
        vaeDecoder(16),
      ],
    },
    assumptions: [
      "Classifier-free guidance doubles the denoiser batch at every step.",
      "The extra attention block on the first 13 layers is not modeled"
        + " separately.",
      DIFFUSION_UNMODELED,
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
