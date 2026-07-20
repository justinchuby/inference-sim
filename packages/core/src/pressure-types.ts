export const PROTOCOL_CONTRACT_REVISION = 2;

export type PressureTicketState =
  | "pending"
  | "granted"
  | "claimed"
  | "cancelled"
  | "failed"
  | "completed";

export type HostAllocationState = "granted" | "claimed";

export interface HostGovernorConfig {
  readonly capacityBytes: number;
  readonly fixedChargeBytes: number;
  readonly reclaimableBytesByDevice?: Readonly<Record<string, number>>;
  readonly sourceId?: string;
}

export interface PressureTicketSnapshot {
  readonly requestId: string;
  readonly deviceId: string;
  readonly bytes: number;
  readonly generation: number;
  readonly state: PressureTicketState;
  readonly allocationId?: string;
}

export interface HostAllocationSnapshot {
  readonly allocationId: string;
  readonly requestId: string;
  readonly deviceId: string;
  readonly bytes: number;
  readonly state: HostAllocationState;
}

export interface HostGovernorSnapshot {
  readonly capacityBytes: number;
  readonly fixedChargeBytes: number;
  readonly freeBytes: number;
  readonly configurationGeneration: number;
  readonly reclaimableBytesByDevice: Readonly<Record<string, number>>;
  readonly pendingReclaimNotices: readonly string[];
  readonly tickets: readonly PressureTicketSnapshot[];
  readonly allocations: readonly HostAllocationSnapshot[];
}

interface PressureTraceEnvelope {
  readonly contractRevision: typeof PROTOCOL_CONTRACT_REVISION;
  readonly sourceId: string;
  readonly sourceSequence: number;
  readonly timestampNs: number;
}

export type PressureTraceEvent = PressureTraceEnvelope & PressureTracePayload;

export type PressureTracePayload =
  | {
      readonly kind: "submit";
      readonly requestId: string;
      readonly deviceId: string;
      readonly bytes: number;
      readonly generation: number;
    }
  | {
      readonly kind: "grant";
      readonly requestId: string;
      readonly allocationId: string;
      readonly bytes: number;
    }
  | {
      readonly kind: "claim";
      readonly requestId: string;
      readonly allocationId: string;
    }
  | {
      readonly kind: "release";
      readonly requestId: string;
      readonly allocationId: string;
      readonly bytes: number;
    }
  | {
      readonly kind: "cancel";
      readonly requestId: string;
      readonly previousState: "pending" | "granted";
      readonly allocationId?: string;
    }
  | {
      readonly kind: "timeout";
      readonly requestId: string;
      readonly previousState: "pending" | "granted";
      readonly allocationId?: string;
    }
  | {
      readonly kind: "reconfigure";
      readonly generation: number;
      readonly failedRequestIds: readonly string[];
    }
  | {
      readonly kind: "reclaim_notice";
      readonly deviceId: string;
    }
  | {
      readonly kind: "reclaim";
      readonly deviceId: string;
      readonly bytes: number;
    };

export interface PressureReplayResult {
  readonly appliedEvents: number;
  readonly snapshot: HostGovernorSnapshot;
}
