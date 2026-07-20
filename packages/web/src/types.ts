import type {
  ConfidenceClass,
  ExpertCacheMetrics,
  ExpertRouteResult,
  ScenarioMemoryLedgerEntry,
  SpeculativeWorkloadIteration,
  SpeculativeWorkloadMetrics,
  SpeculativeProposerFamily,
  ServingMetrics,
  ServingRequestResult,
  TopologyResourceUtilization,
  TopologyWorkloadMetrics,
} from "@inference-sim/core";

export type WorkloadMode = "serving" | "speculative" | "expert-cache";

export interface DashboardRunConfig {
  readonly scenarioName:
    | "cpu-only"
    | "single-gpu-cpu"
    | "multi-gpu"
    | "gpu-npu"
    | "unified-memory"
    | "multi-node";
  readonly mode: WorkloadMode;
  readonly seed: number;
  readonly speculative: {
    readonly family: SpeculativeProposerFamily;
    readonly outputTokens: number;
    readonly draftWidth: number;
    readonly firstPositionAcceptance: number;
  };
  readonly serving: {
    readonly decodeMode: "target_only" | SpeculativeProposerFamily;
    readonly draftWidth: number;
    readonly firstPositionAcceptance: number;
    readonly requestCount: number;
    readonly arrivalGapUs: number;
    readonly promptTokens: number;
    readonly outputTokens: number;
    readonly maxBatchSize: number;
    readonly maxBatchTokens: number;
    readonly prefillChunkTokens: number;
  };
  readonly expertCache: {
    readonly tokenCount: number;
    readonly topK: number;
    readonly expertCount: number;
    readonly hotSlots: number;
    readonly warmSlots: number;
  };
}

export interface DashboardResult {
  readonly scenario: {
    readonly id: string;
    readonly family: string;
    readonly deviceCount: number;
    readonly linkCount: number;
    readonly memoryLedger: readonly ScenarioMemoryLedgerEntry[];
  };
  readonly mode: WorkloadMode;
  readonly durationMs: number;
  readonly topology: {
    readonly confidence: ConfidenceClass;
    readonly assumptions: readonly string[];
    readonly planSteps: number;
    readonly operationCounts: {
      readonly compute: number;
      readonly transfer: number;
      readonly collective: number;
    };
    readonly metrics: TopologyWorkloadMetrics;
    readonly topResources: readonly TopologyResourceUtilization[];
  };
  readonly speculative?: {
    readonly family: SpeculativeProposerFamily;
    readonly support: "onnx_genai_current" | "design_only";
    readonly metrics: SpeculativeWorkloadMetrics;
    readonly iterations: readonly SpeculativeWorkloadIteration[];
    readonly finalTokenLength: number;
  };
  readonly expertCache?: {
    readonly metrics: ExpertCacheMetrics;
    readonly routes: readonly ExpertRouteResult[];
    readonly hotResidentBytes: number;
    readonly warmResidentBytes: number;
    readonly hotCapacityBytes: number;
    readonly warmCapacityBytes: number;
  };
  readonly serving?: {
    readonly decodeMode: "target_only" | SpeculativeProposerFamily;
    readonly support: "onnx_genai_current" | "design_only" | "target_only";
    readonly metrics: ServingMetrics;
    readonly requests: readonly ServingRequestResult[];
    readonly batches: readonly {
      readonly batchId: number;
      readonly sequenceCount: number;
      readonly tokenWork: number;
      readonly prefillSequences: number;
      readonly decodeSequences: number;
      readonly durationNs: number;
    }[];
  };
}

export type WorkerRequest = {
  readonly type: "run";
  readonly runId: number;
  readonly config: DashboardRunConfig;
};

export type WorkerResponse =
  | {
      readonly type: "progress";
      readonly runId: number;
      readonly progress: number;
      readonly phase: string;
    }
  | {
      readonly type: "result";
      readonly runId: number;
      readonly result: DashboardResult;
    }
  | {
      readonly type: "error";
      readonly runId: number;
      readonly message: string;
    };
