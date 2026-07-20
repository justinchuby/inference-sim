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
  quantization: Quantization;
  layers: LayerProfile[];
  moe?: MoEProfile;
  provenance: ModelProfileProvenance;
}

export interface ModelProfileProvenance {
  evidence: "exact" | "calibrated" | "heuristic";
  source: string;
  assumptions: readonly string[];
}

export interface ModelArchitecture {
  kind: "dense" | "moe";
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
