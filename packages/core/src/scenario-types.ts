export const SCENARIO_SCHEMA_VERSION = 5;

export type ConfidenceClass = "exact" | "bounded" | "calibrated" | "heuristic";
export type SimDeviceKind = "cpu" | "gpu" | "npu";
export type MemoryDomainKind = "host" | "device" | "unified" | "storage";
export type AllocationClass =
  | "pageable"
  | "pinned"
  | "device"
  | "unified"
  | "storage";
export type ComputeCapability =
  | "attention"
  | "ffn"
  | "collective"
  | "copy"
  | "sampling"
  | "draft"
  | "lookup";

export interface EvidenceProvenance {
  readonly confidence: ConfidenceClass;
  readonly source: string;
  readonly measuredAt?: string;
  readonly notes?: string;
}

export type GovernorOwner =
  | { readonly kind: "host"; readonly nodeId: string }
  | { readonly kind: "device"; readonly deviceId: string }
  | { readonly kind: "none" };

export interface MemoryDomainSpec {
  readonly id: string;
  readonly nodeId: string;
  readonly kind: MemoryDomainKind;
  readonly capacityBytes: number;
  readonly resourceLimitBytes: number;
  readonly bandwidthBytesPerSec: number;
  readonly latencyNs: number;
  readonly coherent: boolean;
  readonly allocationClasses: readonly AllocationClass[];
  readonly accessibleBy: readonly string[];
  readonly governor: GovernorOwner;
  readonly provenance: EvidenceProvenance;
}

export interface SimDeviceSpec {
  readonly id: string;
  readonly nodeId: string;
  readonly kind: SimDeviceKind;
  readonly executionProvider: string;
  readonly memoryDomainIds: readonly string[];
  readonly capabilities: readonly ComputeCapability[];
  readonly supportedDtypes: readonly string[];
  readonly maxConcurrentCompute: number;
  readonly provenance: EvidenceProvenance;
}

export interface SimLinkSpec {
  readonly id: string;
  readonly sourceDomainId: string;
  readonly targetDomainId: string;
  readonly kind:
    | "on-chip"
    | "pcie"
    | "nvlink"
    | "ethernet"
    | "infiniband"
    | "thunderbolt"
    | "storage";
  readonly bandwidthBytesPerSec: number;
  readonly latencyNs: number;
  readonly concurrencyLanes: number;
  readonly provenance: EvidenceProvenance;
}

export interface AllocationReservation {
  readonly physicalAllocationId: string;
  readonly domainId: string;
  readonly bytes: number;
  readonly allocationClass: AllocationClass;
  readonly purpose:
    | "weights"
    | "kv"
    | "workspace"
    | "staging"
    | "cache"
    | "backing"
    | "checkpoint"
    | "sidecar";
}

export interface PartitionPlacement {
  readonly partitionId: string;
  readonly deviceId: string;
  readonly requiredCapabilities: readonly ComputeCapability[];
  readonly allocations: readonly AllocationReservation[];
  readonly sharedAllocationIds?: readonly string[];
}

export interface TransferRequirement {
  readonly id: string;
  readonly sourceDomainId: string;
  readonly targetDomainId: string;
  readonly bytes: number;
  readonly requiresPinnedStaging: boolean;
  readonly stagingAllocationIds: readonly string[];
}

export interface TransferRoute {
  readonly domainIds: readonly string[];
  readonly linkIds: readonly string[];
  readonly declaredDurationNs: number;
}

export interface CommunicatorRankSpec {
  readonly rankId: string;
  readonly deviceId: string;
}

export interface CommunicatorGroupSpec {
  readonly id: string;
  readonly orderedRanks: readonly CommunicatorRankSpec[];
}

export interface ScenarioWorkloadSpec {
  readonly batchSize: number;
  readonly inputSequenceLength: number;
  readonly outputSequenceLength: number;
  readonly speculative?: {
    readonly family:
      | "prompt_lookup"
      | "draft_model"
      | "mtp"
      | "eagle3"
      | "shared_kv"
      | "self_speculative";
    readonly maxAdditionalTokens: number;
  };
}

export interface ScenarioExecutionPolicy {
  readonly topologyEpoch: number;
  readonly seed: number;
  readonly maxEvents: number;
  readonly features: {
    readonly ssdStreaming: boolean;
  };
  readonly parallelism: {
    readonly composition: "cartesian" | "overlap_by_capability";
    readonly tensor: number;
    readonly pipeline: number;
    readonly expert: number;
    readonly data: number;
  };
}

export interface CalibrationCoefficient {
  readonly id: string;
  readonly value: number;
  readonly unit: string;
  readonly provenance: EvidenceProvenance;
}

export interface CalibrationSet {
  readonly coefficients: readonly CalibrationCoefficient[];
}

export interface SimulationScenario {
  readonly schemaVersion: typeof SCENARIO_SCHEMA_VERSION;
  readonly id: string;
  readonly family:
    | "cpu_only"
    | "single_discrete"
    | "multi_gpu"
    | "gpu_npu"
    | "unified"
    | "multi_node"
    | "custom";
  readonly memoryDomains: readonly MemoryDomainSpec[];
  readonly devices: readonly SimDeviceSpec[];
  readonly links: readonly SimLinkSpec[];
  readonly placements: readonly PartitionPlacement[];
  readonly transfers: readonly TransferRequirement[];
  readonly groups: readonly CommunicatorGroupSpec[];
  readonly workload: ScenarioWorkloadSpec;
  readonly execution: ScenarioExecutionPolicy;
  readonly calibration: CalibrationSet;
}

export interface ScenarioValidationIssue {
  readonly code: string;
  readonly path: string;
  readonly message: string;
}

export interface ScenarioValidationResult {
  readonly valid: boolean;
  readonly issues: readonly ScenarioValidationIssue[];
}

export interface ScenarioMemoryLedgerEntry {
  readonly domainId: string;
  readonly enabled: boolean;
  readonly physicalCapacityBytes: number;
  readonly capacityBytes: number;
  readonly reservedBytes: number;
  readonly freeBytes: number;
}

export interface ScenarioMemoryLedgerOptions {
  readonly allocationBytes?: Readonly<Record<string, number>>;
}
