// ============================================================
// Hardware Topology Types
// ============================================================

export interface HardwareTopology {
  nodes: NodeSpec[];
  interNodeLinks: InterconnectSpec[];
}

export interface NodeSpec {
  id: string;
  devices: DeviceSpec[];
  hostMemory: MemorySpec;
  interDeviceLinks: InterconnectSpec[];
}

export interface DeviceSpec {
  id: string;
  kind: "gpu" | "npu" | "unified";
  memory: MemorySpec;
  compute: ComputeSpec;
}

export interface MemorySpec {
  capacityBytes: number;
  bandwidthBytesPerSec: number;
  latencyNs: number;
}

export interface ComputeSpec {
  fp16Flops: number;
  fp8Flops: number;
  int8Flops: number;
}

export interface InterconnectSpec {
  endpoints: [string, string];
  bandwidthBytesPerSec: number;
  latencyNs: number;
  kind: "nvlink" | "pcie" | "infiniband" | "ethernet" | "thunderbolt" | "on-chip";
}

// ============================================================
// Model Profile Types
// ============================================================

export interface ModelProfile {
  name: string;
  architecture: ModelArchitecture;
  totalParams: number;
  /**
   * Token embedding plus untied output-projection weight bytes. These are not
   * part of any `layers` entry and dominate residency for small models.
   */
  embeddingBytes?: number;
  quantization: Quantization;
  layers: LayerProfile[];
  moe?: MoEProfile;
  /**
   * Non-decoder components of a multimodal package, such as a vision tower and
   * its projector. The decoder itself is described by `layers` and is not
   * repeated here.
   */
  components?: readonly ModelComponentProfile[];
  /** Media modalities this checkpoint accepts. Absent when it is text only. */
  mediaInputs?: readonly MediaInputProfile[];
  /**
   * Speculative drafting the released weights can do unaided. Absent when the
   * checkpoint ships no drafter, which is the common case.
   */
  speculative?: {
    readonly families: readonly string[];
    readonly drafterParams?: number;
    readonly draftWidth?: number;
  };
  /** Present for iterative denoisers. */
  diffusion?: DiffusionProfile;
  provenance: ModelProfileProvenance;
}

export interface DiffusionProfile {
  readonly denoisingSteps: number;
  /** Latent positions the denoiser attends over at the default resolution. */
  readonly latentTokens: number;
  readonly defaultResolutionPx: number;
  /** Classifier-free guidance doubles the denoiser batch at every step. */
  readonly classifierFreeGuidance: boolean;
  /** Denoiser forward passes for one image. */
  readonly denoiserInvocations: number;
}

export type ModelComponentPhase =
  | "prompt_only"
  | "every_step"
  | "final_only"
  | "on_demand";

export interface ModelComponentProfile {
  readonly id: string;
  /** Reported role, for example `vision_encoder` or `projector`. */
  readonly role: string;
  /** When the component runs relative to decoding. */
  readonly phase: ModelComponentPhase;
  readonly params: number;
  readonly weightBytes: number;
  /**
   * Decoder tokens one media item expands into. The simulator charges this as
   * prompt work; per-request tile counts are not modeled.
   */
  readonly tokensPerItem?: number;
}

/** A kind of non-text input a checkpoint accepts. */
export type MediaModality = "image" | "audio" | "video";

/**
 * One media modality a model accepts, and what attaching one item costs the
 * decoder. Models differ on both: a checkpoint may take images but not audio,
 * and an adapter that cross-attends injects no decoder positions at all.
 */
export interface MediaInputProfile {
  readonly modality: MediaModality;
  /**
   * Decoder positions one item injects into the prompt, which the simulator
   * charges as prompt work. Zero when the adapter cross-attends instead of
   * expanding the sequence.
   */
  readonly decoderTokensPerItem: number;
  /** What one item is, for display: `image`, `second of audio`, `30s window`. */
  readonly unit: string;
  /** Components that run when this modality is attached. */
  readonly componentIds: readonly string[];
}

export interface ModelProfileProvenance {
  evidence: "exact" | "calibrated" | "heuristic";
  source: string;
  assumptions: readonly string[];
}

export interface ModelArchitecture {
  /** `diffusion` denotes an iterative denoiser: no vocabulary and no KV cache. */
  kind: "dense" | "moe" | "diffusion";
  numLayers: number;
  hiddenDim: number;
  numHeads: number;
  numKVHeads: number;
  vocabSize: number;
  intermediateSize: number;
}

export interface MoEProfile {
  numExperts: number;
  activeExpertsPerToken: number;
  /** Weight bytes for one routed expert in one transformer layer. */
  expertBytesPerLayer: number;
  /** Weight bytes for the shared expert in one transformer layer. */
  sharedExpertBytesPerLayer: number;
  activationDistribution: ExpertDistribution;
}

export type ExpertDistribution =
  | { kind: "uniform" }
  | { kind: "zipf"; s: number }
  | { kind: "empirical"; frequencies: number[] }
  | { kind: "clustered"; hotExperts: number; hotFrequency: number };

export interface LayerProfile {
  index: number;
  attentionBytes: number;
  ffnBytes: number;
  kvCachePerToken: number;
}

export interface Quantization {
  weights: QuantType;
  kvCache: QuantType;
  activations: QuantType;
}

export type QuantType =
  | "fp32"
  | "fp16"
  | "bf16"
  | "fp8"
  | "int8"
  | "int4"
  | "int2"
  | "int1"
  | "nf4";

// ============================================================
// Pipeline Configuration Types
// ============================================================

export interface PipelineConfig {
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  parallelism: ParallelismConfig;
  memory: MemoryPolicyConfig;
}

export interface ParallelismConfig {
  tensorParallel: number;
  pipelineParallel: number;
  expertParallel: number;
  dataParallel: number;
}

export interface MemoryPolicyConfig {
  kvCacheBudgetFraction: number;
  expertCacheBudgetFraction: number;
  pinnedPoolFraction: number;
  offloadStrategy: "none" | "partial" | "full";
  prefetchAhead: number;
  pressureThreshold: number;
  reclaimBatchSize: number;
}

// ============================================================
// Simulation Output Types
// ============================================================

export interface StaticAnalysisResult {
  feasible: boolean;
  memoryBreakdown: DeviceMemoryBreakdown[];
  hostMemoryBreakdown: HostMemoryBreakdown;
  bottleneck: "compute" | "memory_bandwidth" | "interconnect" | "capacity";
  estimatedThroughput: ThroughputEstimate;
  recommendations: string[];
}

export interface DeviceMemoryBreakdown {
  deviceId: string;
  totalBytes: number;
  weights: number;
  kvCache: number;
  expertCache: number;
  activations: number;
  free: number;
}

export interface HostMemoryBreakdown {
  totalBytes: number;
  offloadedWeights: number;
  warmExperts: number;
  kvOverflow: number;
  free: number;
}

export interface ThroughputEstimate {
  prefillToksPerSec: number;
  decodeToksPerSec: number;
  timeToFirstTokenMs: number;
  interTokenLatencyMs: number;
}

// ============================================================
// Simulation Events (for Phase 2+)
// ============================================================

export type SimEvent =
  | { kind: "token_start"; tokenIdx: number; timestampNs: number }
  | { kind: "layer_compute"; layerIdx: number; phase: "attention" | "ffn"; durationNs: number }
  | { kind: "expert_route"; layerIdx: number; expertIds: number[] }
  | { kind: "expert_cache_hit"; expertId: number; deviceId: string }
  | { kind: "expert_cache_miss"; expertId: number; loadFromTier: "warm" | "cold" }
  | { kind: "expert_load"; expertId: number; bytes: number; durationNs: number }
  | { kind: "expert_evict"; expertId: number; deviceId: string }
  | { kind: "collective"; op: string; bytes: number; durationNs: number }
  | { kind: "pressure_request"; deviceId: string; bytesNeeded: number }
  | { kind: "pressure_grant"; deviceId: string; bytesGranted: number; latencyNs: number }
  | { kind: "memory_snapshot"; allocations: DeviceMemoryBreakdown[] }
  | { kind: "token_complete"; tokenIdx: number; latencyNs: number };

export interface SimTrace {
  events: SimEvent[];
  summary: SimSummary;
}

export interface SimSummary {
  totalTokens: number;
  totalTimeNs: number;
  avgDecodeLatencyNs: number;
  expertCacheHitRate: number;
  pressureEvents: number;
  avgPressureLatencyNs: number;
}
