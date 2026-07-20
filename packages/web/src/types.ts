import type {
  CalibrationDataset,
  CalibrationEvidenceKind,
  CalibrationFitDiagnostic,
  TransportCalibrationFitDiagnostic,
  ConfidenceClass,
  ExpertCacheMetrics,
  ExpertCachePartitionSnapshot,
  ExpertRouteResult,
  PlanTraceEvent,
  RankCompletion,
  RankTerminalState,
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
  ModelProfile,
  StaticAnalysisResult,
  MemoryPolicyConfig,
  ParallelismConfig,
  QuantType,
  StaticSearchObjective,
  StaticSearchResult,
  SimulationScenario,
  TopologyPipelineWork,
} from "@inference-sim/core";

export type WorkloadMode =
  | "serving"
  | "pipeline"
  | "speculative"
  | "expert-cache";

export interface DashboardModelBinding {
  readonly source: "builtin_model" | "local_model_package";
  readonly displayName: string;
  readonly modelFingerprints: readonly string[];
  readonly targetModelFingerprint: string;
  readonly componentCount: number;
  readonly totalParameters: number;
  readonly weightBytes: number;
  readonly modelFormat?: DashboardModelFormat;
  readonly executionProfile: DashboardModelExecutionProfile;
  readonly pipelineExecution?: TopologyPipelineWork;
  readonly executionCoverage: DashboardModelExecutionCoverage;
  readonly pipelineStrategy?: string;
  readonly speculativeFamilies: readonly SpeculativeProposerFamily[];
}

export interface DashboardModelFormat {
  readonly weightDtypes: readonly string[];
  readonly weightQuantization:
    | "none"
    | "fp8"
    | "int8"
    | "int4"
    | "nf4"
    | "mixed"
    | "unknown";
  readonly kvCacheDtype: QuantType | "unknown";
  readonly activationDtype: QuantType | "unknown";
  readonly evidence: "preset_declared" | "onnx_inferred";
  readonly runtimeDtypesDefaulted: boolean;
}

export interface DashboardModelExecutionCoverage {
  readonly fidelity: "complete" | "partial";
  readonly scope: "full_model" | "target_component_only";
  readonly modeledComponentIds: readonly string[];
  readonly unmodeledComponentIds: readonly string[];
  readonly limitations: readonly string[];
}

export interface DashboardModelExecutionProfile {
  readonly modelId: string;
  readonly modelName: string;
  readonly attentionWeightBytesPerToken: number;
  readonly ffnWeightBytesPerToken: number;
  readonly forwardFlopsPerToken: number;
}

export interface DashboardRunConfig {
  readonly scenarioName:
    | "rtx-4090-desktop"
    | "rtx-5090-desktop"
    | "mac-mini-m4-pro-64gb"
    | "mac-studio-m3-ultra-512gb"
    | "ryzen-ai-max-395-128gb"
    | "cpu-only"
    | "single-gpu-cpu"
    | "multi-gpu"
    | "gpu-npu"
    | "unified-memory"
    | "multi-node"
    | "custom";
  readonly multiGpuRanks: 2 | 4 | 8;
  readonly multiNodeCount?: 2 | 3 | 4;
  readonly customScenario?: SimulationScenario;
  readonly modelBinding?: DashboardModelBinding;
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
  readonly model?: {
    readonly name: string;
    readonly source: DashboardModelBinding["source"];
    readonly fingerprint: string;
    readonly totalParameters: number;
    readonly weightBytes: number;
    readonly modelFormat?: DashboardModelFormat;
  };
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
  readonly pipelineExecution?: {
    readonly components: readonly {
      readonly id: string;
      readonly phase: string;
      readonly deviceId: string;
    }[];
    readonly transferOperations: number;
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
      readonly kind: "pipeline";
      readonly topology: TopologyWorkloadResult;
    }
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

export interface DashboardArtifactExpectation {
  readonly sourceFileName: string;
  readonly inputFingerprint: string;
  readonly outputFingerprint: string;
  readonly artifactFingerprint: string;
}

export interface DashboardArtifactReplay {
  readonly sourceFileName: string;
  readonly expectedInputFingerprint: string;
  readonly actualInputFingerprint: string;
  readonly expectedArtifactFingerprint: string;
  readonly actualArtifactFingerprint: string;
  readonly expectedOutputFingerprint: string;
  readonly actualOutputFingerprint: string;
  readonly inputMatches: boolean;
  readonly outputMatches: boolean;
  readonly matches: boolean;
}

export interface FrozenPlanBrowserResult {
  readonly sourceFileName: string;
  readonly artifact: {
    readonly revision: number;
    readonly artifactFingerprint: string;
    readonly scenarioFingerprint: string;
    readonly planFingerprint: string;
  };
  readonly scenario: {
    readonly id: string;
    readonly family: string;
    readonly devices: number;
    readonly links: number;
    readonly ranks: number;
  };
  readonly plan: {
    readonly id: string;
    readonly executionId: string;
    readonly topologyEpoch: number;
    readonly steps: number;
    readonly operationCounts: {
      readonly compute: number;
      readonly transfer: number;
      readonly collective: number;
    };
  };
  readonly execution: {
    readonly status: "succeeded" | "failed" | "aborted";
    readonly completedAtNs: number;
    readonly rankCompletions: readonly RankCompletion[];
    readonly rankStates: readonly RankTerminalState[];
    readonly operationPreview: readonly PlanTraceEvent[];
    readonly operationCount: number;
  };
  readonly replay: {
    readonly status: "succeeded" | "failed" | "aborted";
    readonly completedAtNs: number;
    readonly appliedEvents: number;
    readonly exact: boolean;
  };
}

export interface OnnxStaticBrowserConfig {
  readonly hardwarePreset:
    | "dgx-h100"
    | "dgx-h200"
    | "2x-dgx-h100"
    | "4x-mac-studio-m4"
    | "a100-4x"
    | "rtx-4090-2x";
  readonly kvCacheQuantization: QuantType;
  readonly activationQuantization: QuantType;
  readonly batchSize: number;
  readonly inputSeqLen: number;
  readonly outputSeqLen: number;
  readonly parallelism: ParallelismConfig;
  readonly memory: MemoryPolicyConfig;
}

export interface OnnxStaticBrowserResult {
  readonly sourceFileName: string;
  readonly manifest: {
    readonly revision: number;
    readonly fingerprint: string;
    readonly modelFileName: string;
    readonly modelSha256: string;
    readonly graphName: string;
    readonly nodeCount: number;
    readonly initializerCount: number;
    readonly initializerLogicalBytes: number;
    readonly externalDataFiles: number;
    readonly architectureSource: string;
    readonly profileReadiness: {
      readonly ready: boolean;
      readonly missingFields: readonly string[];
    };
  };
  readonly config: OnnxStaticBrowserConfig;
  readonly model: ModelProfile;
  readonly analysis: StaticAnalysisResult;
}

export interface OnnxSearchBrowserConfig {
  readonly objective: StaticSearchObjective;
  readonly topologyScope: "selected" | "all";
  readonly kvCacheScope: "selected" | "fp16_fp8";
  readonly batchScope: "selected" | "common";
  readonly parallelismScope: "selected" | "common";
  readonly offloadScope: "selected" | "none_partial";
  readonly maximumDeviceUsedFraction: number;
  readonly topK: number;
  readonly maxCandidates: number;
}

export interface OnnxSearchBrowserResult {
  readonly sourceFileName: string;
  readonly manifest: {
    readonly fingerprint: string;
    readonly modelFileName: string;
  };
  readonly modelName: string;
  readonly baseConfig: OnnxStaticBrowserConfig;
  readonly searchConfig: OnnxSearchBrowserConfig;
  readonly result: StaticSearchResult;
}

export interface WorkerRunProgress {
  readonly progress: number;
  readonly phase: string;
}

export type WorkerRunProgressReporter = (
  update: WorkerRunProgress,
) => void;

export type WorkerRequest =
  | {
      readonly type: "run";
      readonly runId: number;
      readonly config: DashboardRunConfig;
      readonly expectedArtifact?: DashboardArtifactExpectation;
    }
  | {
      readonly type: "run-frozen-plan";
      readonly runId: number;
      readonly sourceFileName: string;
      readonly artifactText: string;
    }
  | {
      readonly type: "run-onnx-static";
      readonly runId: number;
      readonly sourceFileName: string;
      readonly artifactText: string;
      readonly config: OnnxStaticBrowserConfig;
    }
  | {
      readonly type: "run-onnx-search";
      readonly runId: number;
      readonly sourceFileName: string;
      readonly artifactText: string;
      readonly baseConfig: OnnxStaticBrowserConfig;
      readonly searchConfig: OnnxSearchBrowserConfig;
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
      readonly artifactReplay?: DashboardArtifactReplay;
      readonly durationMs: number;
    }
  | {
      readonly type: "frozen-plan-result";
      readonly runId: number;
      readonly result: FrozenPlanBrowserResult;
      readonly durationMs: number;
    }
  | {
      readonly type: "onnx-static-result";
      readonly runId: number;
      readonly result: OnnxStaticBrowserResult;
      readonly durationMs: number;
    }
  | {
      readonly type: "onnx-search-result";
      readonly runId: number;
      readonly result: OnnxSearchBrowserResult;
      readonly durationMs: number;
    }
  | {
      readonly type: "error";
      readonly runId: number;
      readonly message: string;
    };
