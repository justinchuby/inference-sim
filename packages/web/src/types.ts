import type {
  CalibrationDataset,
  CalibrationEvidenceKind,
  CalibrationFitDiagnostic,
  TransportCalibrationFitDiagnostic,
  ConfidenceClass,
  ExpertCacheMetrics,
  ExpertCachePartitionSnapshot,
  ExpertRouteResult,
  ExpertCacheWorkloadResult,
  ScenarioMemoryLedgerEntry,
  SimulationResultArtifact,
  SpeculativeWorkloadIteration,
  SpeculativeWorkloadMetrics,
  SpeculativeWorkloadResult,
  SpeculativeProposerFamily,
  SpeculativeTokenMismatch,
  SpeculativeTokenTrace,
  ServingMetrics,
  ServingRequestResult,
  TopologyServingComparisonResult,
  TopologyServingResult,
  TopologyResourceUtilization,
  TopologyWorkloadMetrics,
  TopologyWorkloadResult,
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
  readonly multiGpuRanks: 2 | 4 | 8;
  readonly mode: WorkloadMode;
  readonly seed: number;
  readonly calibration?: CalibrationDataset;
  readonly speculative: {
    readonly family: SpeculativeProposerFamily;
    readonly outputTokens: number;
    readonly draftWidth: number;
    readonly firstPositionAcceptance: number;
    readonly trace?: SpeculativeTokenTrace;
  };
  readonly serving: {
    readonly compareTopologies: boolean;
    readonly useExpertCache: boolean;
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
    readonly placementStrategy: "contiguous" | "round_robin";
    readonly tokenCount: number;
    readonly topK: number;
    readonly expertCount: number;
    readonly hotSlots: number;
    readonly warmSlots: number;
    readonly adaptivePrefetch: boolean;
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
  readonly calibration?: {
    readonly datasetId: string;
    readonly datasetFingerprint: string;
    readonly evidenceKind: CalibrationEvidenceKind;
    readonly fitConfidence: ConfidenceClass;
    readonly diagnostics: readonly CalibrationFitDiagnostic[];
    readonly transportDiagnostics:
      readonly TransportCalibrationFitDiagnostic[];
  };
  readonly topology: {
    readonly confidence: ConfidenceClass;
    readonly assumptions: readonly string[];
    readonly planSteps: number;
    readonly operationCounts: {
      readonly compute: number;
      readonly transfer: number;
      readonly collective: number;
      readonly allReduce: number;
      readonly allToAll: number;
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
    readonly tokenTrace?: {
      readonly traceId: string;
      readonly source: string;
      readonly runtimeRevision: string;
      readonly modelFingerprint: string;
      readonly targetOnlyRunId: string;
      readonly speculativeRunId: string;
      readonly promptTokenCount: number;
      readonly comparedTokenCount: number;
      readonly matchesTargetOnly: boolean;
      readonly firstMismatch?: SpeculativeTokenMismatch;
      readonly expectedOutputTokenIds: readonly number[];
      readonly committedOutputTokenIds: readonly number[];
    };
  };
  readonly expertCache?: {
    readonly metrics: ExpertCacheMetrics;
    readonly routes: readonly ExpertRouteResult[];
    readonly hotResidentBytes: number;
    readonly warmResidentBytes: number;
    readonly hotCapacityBytes: number;
    readonly warmCapacityBytes: number;
    readonly hotPartitions: readonly ExpertCachePartitionSnapshot[];
    readonly warmPartitions: readonly ExpertCachePartitionSnapshot[];
  };
  readonly serving?: {
    readonly decodeMode: "target_only" | SpeculativeProposerFamily;
    readonly support: "onnx_genai_current" | "design_only" | "target_only";
    readonly metrics: ServingMetrics;
    readonly requests: readonly ServingRequestResult[];
    readonly physicalReplayEvents?: number;
    readonly maximumConcurrentPlans?: number;
    readonly physicalDrainNs?: number;
    readonly batches: readonly {
      readonly batchId: number;
      readonly sequenceCount: number;
      readonly tokenWork: number;
      readonly prefillSequences: number;
      readonly decodeSequences: number;
      readonly durationNs: number;
      readonly cacheConstraintNs: number;
      readonly expertRoutes: number;
    }[];
  };
  readonly comparison?: readonly {
    readonly rank: number;
    readonly scenarioId: string;
    readonly relativeToFastest: number;
    readonly totalDurationNs: number;
    readonly throughputTokensPerSecond: number;
    readonly p95TimeToFirstTokenNs: number;
    readonly p95InterTokenLatencyNs: number;
    readonly averageRequestLatencyNs: number;
    readonly kvHighWaterTokens: number;
    readonly batches: number;
    readonly confidence: ConfidenceClass;
  }[];
}

export type DashboardCoreEvidence =
  | {
      readonly kind: "speculative";
      readonly workload: SpeculativeWorkloadResult;
      readonly topology: TopologyWorkloadResult;
    }
  | {
      readonly kind: "expert_cache";
      readonly workload: ExpertCacheWorkloadResult;
      readonly topology: TopologyWorkloadResult;
    }
  | {
      readonly kind: "serving";
      readonly serving: TopologyServingResult;
    }
  | {
      readonly kind: "serving_comparison";
      readonly comparison: TopologyServingComparisonResult;
    };

export interface DashboardArtifactOutput {
  readonly summary: Omit<DashboardResult, "durationMs">;
  readonly evidence: DashboardCoreEvidence;
}

export type DashboardArtifact = SimulationResultArtifact<
  DashboardRunConfig,
  DashboardArtifactOutput
>;

export interface DashboardArtifactDownload {
  readonly blob: Blob;
  readonly fileName: string;
  readonly artifactFingerprint: string;
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
      readonly summary: Omit<DashboardResult, "durationMs">;
      readonly artifact: DashboardArtifactDownload;
      readonly durationMs: number;
    }
  | {
      readonly type: "error";
      readonly runId: number;
      readonly message: string;
    };
