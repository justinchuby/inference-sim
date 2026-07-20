import {
  PROTOCOL_CONTRACT_REVISION,
  type HostAllocationSnapshot,
  type HostGovernorConfig,
  type HostGovernorSnapshot,
  type PressureTicketSnapshot,
  type PressureTicketState,
  type PressureTraceEvent,
  type PressureTracePayload,
} from "./pressure-types.js";

interface MutableTicket {
  requestId: string;
  deviceId: string;
  bytes: number;
  generation: number;
  state: PressureTicketState;
  allocationId?: string;
}

interface MutableAllocation {
  allocationId: string;
  requestId: string;
  deviceId: string;
  bytes: number;
  state: "granted" | "claimed";
}

export class PressureProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PressureProtocolError";
  }
}

/**
 * Deterministic HostGovernor pressure state machine.
 *
 * Every public mutation is one abstract protocol transition and emits one
 * lossless trace event after its invariants have been checked.
 */
export class HostGovernorSimulator {
  private readonly capacityBytes: number;
  private readonly fixedChargeBytes: number;
  private readonly sourceId: string;
  private readonly reclaimableBytes = new Map<string, number>();
  private readonly pendingReclaimNotices = new Set<string>();
  private readonly tickets = new Map<string, MutableTicket>();
  private readonly allocations = new Map<string, MutableAllocation>();
  private readonly events: PressureTraceEvent[] = [];
  private freeBytes: number;
  private configurationGeneration = 0;
  private nextRequestId = 1;
  private nextAllocationId = 1;
  private nextSourceSequence = 0;
  private lastTimestampNs = 0;

  constructor(config: HostGovernorConfig) {
    assertPositiveSafeInteger(config.capacityBytes, "capacityBytes");
    assertNonNegativeSafeInteger(config.fixedChargeBytes, "fixedChargeBytes");
    if (config.fixedChargeBytes > config.capacityBytes) {
      throw new PressureProtocolError(
        "fixedChargeBytes must not exceed capacityBytes",
      );
    }
    this.capacityBytes = config.capacityBytes;
    this.fixedChargeBytes = config.fixedChargeBytes;
    this.sourceId = config.sourceId ?? "host-governor";
    if (this.sourceId.length === 0) {
      throw new PressureProtocolError("sourceId must not be empty");
    }

    let reclaimableTotal = 0;
    for (const [deviceId, bytes] of Object.entries(
      config.reclaimableBytesByDevice ?? {},
    )) {
      if (deviceId.length === 0) {
        throw new PressureProtocolError("reclaimable device id must not be empty");
      }
      assertNonNegativeSafeInteger(bytes, `reclaimable bytes for ${deviceId}`);
      reclaimableTotal = checkedAdd(reclaimableTotal, bytes, "reclaimable total");
      this.reclaimableBytes.set(deviceId, bytes);
    }

    const initialCharge = checkedAdd(
      this.fixedChargeBytes,
      reclaimableTotal,
      "initial host charge",
    );
    if (initialCharge > this.capacityBytes) {
      throw new PressureProtocolError(
        `initial host charge ${initialCharge} exceeds capacity ${this.capacityBytes}`,
      );
    }
    this.freeBytes = this.capacityBytes - initialCharge;
    this.assertInvariants();
  }

  submit(deviceId: string, bytes: number, timestampNs: number): string {
    this.validateTimestamp(timestampNs);
    if (deviceId.length === 0) {
      throw new PressureProtocolError("deviceId must not be empty");
    }
    assertPositiveSafeInteger(bytes, "request bytes");
    if (bytes > this.capacityBytes - this.fixedChargeBytes) {
      throw new PressureProtocolError(
        `request ${bytes} can never fit usable host capacity`,
      );
    }

    const requestId = `pressure-${this.nextRequestId++}`;
    const ticket: MutableTicket = {
      requestId,
      deviceId,
      bytes,
      generation: this.configurationGeneration,
      state: "pending",
    };
    this.tickets.set(requestId, ticket);
    this.commit(timestampNs, {
      kind: "submit",
      requestId,
      deviceId,
      bytes,
      generation: ticket.generation,
    });
    return requestId;
  }

  grant(requestId: string, timestampNs: number): string {
    this.validateTimestamp(timestampNs);
    const ticket = this.requireTicket(requestId);
    if (ticket.state !== "pending") {
      throw new PressureProtocolError(
        `grant requires pending ticket; ${requestId} is ${ticket.state}`,
      );
    }
    if (ticket.generation !== this.configurationGeneration) {
      throw new PressureProtocolError(
        `cannot grant stale generation ${ticket.generation}`,
      );
    }
    if (this.freeBytes < ticket.bytes) {
      throw new PressureProtocolError(
        `cannot grant ${ticket.bytes} bytes with ${this.freeBytes} free`,
      );
    }

    const allocationId = `host-allocation-${this.nextAllocationId++}`;
    this.freeBytes -= ticket.bytes;
    ticket.state = "granted";
    ticket.allocationId = allocationId;
    this.allocations.set(allocationId, {
      allocationId,
      requestId,
      deviceId: ticket.deviceId,
      bytes: ticket.bytes,
      state: "granted",
    });
    this.commit(timestampNs, {
      kind: "grant",
      requestId,
      allocationId,
      bytes: ticket.bytes,
    });
    return allocationId;
  }

  claim(requestId: string, timestampNs: number): string {
    this.validateTimestamp(timestampNs);
    const ticket = this.requireTicket(requestId);
    if (ticket.state !== "granted" || !ticket.allocationId) {
      throw new PressureProtocolError(
        `claim requires granted ticket; ${requestId} is ${ticket.state}`,
      );
    }
    const allocation = this.requireAllocation(ticket.allocationId);
    if (allocation.state !== "granted") {
      throw new PressureProtocolError(
        `allocation ${allocation.allocationId} was already claimed`,
      );
    }
    allocation.state = "claimed";
    ticket.state = "claimed";
    this.commit(timestampNs, {
      kind: "claim",
      requestId,
      allocationId: allocation.allocationId,
    });
    return allocation.allocationId;
  }

  release(requestId: string, timestampNs: number): void {
    this.validateTimestamp(timestampNs);
    const ticket = this.requireTicket(requestId);
    if (ticket.state !== "claimed" || !ticket.allocationId) {
      throw new PressureProtocolError(
        `release requires claimed ticket; ${requestId} is ${ticket.state}`,
      );
    }
    const allocation = this.requireAllocation(ticket.allocationId);
    if (allocation.state !== "claimed") {
      throw new PressureProtocolError(
        `allocation ${allocation.allocationId} is not claimed`,
      );
    }
    this.allocations.delete(allocation.allocationId);
    this.freeBytes = checkedAdd(this.freeBytes, allocation.bytes, "free bytes");
    ticket.state = "completed";
    delete ticket.allocationId;
    this.commit(timestampNs, {
      kind: "release",
      requestId,
      allocationId: allocation.allocationId,
      bytes: allocation.bytes,
    });
  }

  cancel(requestId: string, timestampNs: number): void {
    this.terminateUnclaimed(requestId, timestampNs, "cancel");
  }

  timeout(requestId: string, timestampNs: number): void {
    this.terminateUnclaimed(requestId, timestampNs, "timeout");
  }

  reconfigure(timestampNs: number): readonly string[] {
    this.validateTimestamp(timestampNs);
    this.configurationGeneration++;
    const failedRequestIds: string[] = [];
    for (const ticket of this.tickets.values()) {
      if (ticket.state === "pending") {
        ticket.state = "failed";
        failedRequestIds.push(ticket.requestId);
      }
    }
    this.commit(timestampNs, {
      kind: "reconfigure",
      generation: this.configurationGeneration,
      failedRequestIds,
    });
    return failedRequestIds;
  }

  sendReclaimNotice(deviceId: string, timestampNs: number): void {
    this.validateTimestamp(timestampNs);
    if ((this.reclaimableBytes.get(deviceId) ?? 0) <= 0) {
      throw new PressureProtocolError(
        `device ${deviceId} has no reclaimable host bytes`,
      );
    }
    if (this.pendingReclaimNotices.has(deviceId)) {
      throw new PressureProtocolError(
        `device ${deviceId} already has a pending reclaim notice`,
      );
    }
    const needsPressure = [...this.tickets.values()].some(
      (ticket) => ticket.state === "pending" && ticket.bytes > this.freeBytes,
    );
    if (!needsPressure) {
      throw new PressureProtocolError("no pending ticket currently requires reclaim");
    }
    this.pendingReclaimNotices.add(deviceId);
    this.commit(timestampNs, { kind: "reclaim_notice", deviceId });
  }

  reclaim(deviceId: string, bytes: number, timestampNs: number): void {
    this.validateTimestamp(timestampNs);
    assertPositiveSafeInteger(bytes, "reclaim bytes");
    if (!this.pendingReclaimNotices.has(deviceId)) {
      throw new PressureProtocolError(
        `device ${deviceId} has no pending reclaim notice`,
      );
    }
    const reclaimable = this.reclaimableBytes.get(deviceId) ?? 0;
    if (bytes > reclaimable) {
      throw new PressureProtocolError(
        `cannot reclaim ${bytes} bytes from ${deviceId}; only ${reclaimable} available`,
      );
    }
    this.reclaimableBytes.set(deviceId, reclaimable - bytes);
    this.pendingReclaimNotices.delete(deviceId);
    this.freeBytes = checkedAdd(this.freeBytes, bytes, "free bytes");
    this.commit(timestampNs, { kind: "reclaim", deviceId, bytes });
  }

  trace(): readonly PressureTraceEvent[] {
    return this.events.map((event) => (
      event.kind === "reconfigure"
        ? { ...event, failedRequestIds: [...event.failedRequestIds] }
        : { ...event }
    ));
  }

  snapshot(): HostGovernorSnapshot {
    const reclaimableBytesByDevice = Object.fromEntries(
      [...this.reclaimableBytes.entries()].sort(([left], [right]) =>
        left.localeCompare(right)
      ),
    );
    const tickets: PressureTicketSnapshot[] = [...this.tickets.values()]
      .map((ticket) => ({ ...ticket }))
      .sort((left, right) => left.requestId.localeCompare(right.requestId));
    const allocations: HostAllocationSnapshot[] = [...this.allocations.values()]
      .map((allocation) => ({ ...allocation }))
      .sort((left, right) => left.allocationId.localeCompare(right.allocationId));
    return {
      capacityBytes: this.capacityBytes,
      fixedChargeBytes: this.fixedChargeBytes,
      freeBytes: this.freeBytes,
      configurationGeneration: this.configurationGeneration,
      reclaimableBytesByDevice,
      pendingReclaimNotices: [...this.pendingReclaimNotices].sort(),
      tickets,
      allocations,
    };
  }

  assertInvariants(): void {
    assertNonNegativeSafeInteger(this.freeBytes, "free bytes");
    let used = checkedAdd(this.fixedChargeBytes, this.freeBytes, "ledger total");
    for (const bytes of this.reclaimableBytes.values()) {
      assertNonNegativeSafeInteger(bytes, "reclaimable bytes");
      used = checkedAdd(used, bytes, "ledger total");
    }
    for (const allocation of this.allocations.values()) {
      assertPositiveSafeInteger(allocation.bytes, "allocation bytes");
      used = checkedAdd(used, allocation.bytes, "ledger total");
    }
    if (used !== this.capacityBytes) {
      throw new PressureProtocolError(
        `capacity invariant failed: ledger ${used} != capacity ${this.capacityBytes}`,
      );
    }

    for (const ticket of this.tickets.values()) {
      const allocation = ticket.allocationId
        ? this.allocations.get(ticket.allocationId)
        : undefined;
      if (ticket.state === "pending" && ticket.generation !== this.configurationGeneration) {
        throw new PressureProtocolError(
          `pending ticket ${ticket.requestId} uses stale generation`,
        );
      }
      if (ticket.state === "granted" || ticket.state === "claimed") {
        if (!allocation || allocation.state !== ticket.state) {
          throw new PressureProtocolError(
            `${ticket.state} ticket ${ticket.requestId} lacks matching allocation`,
          );
        }
        if (
          allocation.requestId !== ticket.requestId
          || allocation.deviceId !== ticket.deviceId
          || allocation.bytes !== ticket.bytes
        ) {
          throw new PressureProtocolError(
            `allocation ownership mismatch for ${ticket.requestId}`,
          );
        }
      } else if (allocation) {
        throw new PressureProtocolError(
          `terminal/pending ticket ${ticket.requestId} retains allocation`,
        );
      }
    }

    for (const allocation of this.allocations.values()) {
      const ticket = this.tickets.get(allocation.requestId);
      if (!ticket || ticket.allocationId !== allocation.allocationId) {
        throw new PressureProtocolError(
          `orphan allocation ${allocation.allocationId}`,
        );
      }
    }
  }

  private terminateUnclaimed(
    requestId: string,
    timestampNs: number,
    kind: "cancel" | "timeout",
  ): void {
    this.validateTimestamp(timestampNs);
    const ticket = this.requireTicket(requestId);
    if (ticket.state !== "pending" && ticket.state !== "granted") {
      throw new PressureProtocolError(
        `${kind} requires pending/granted ticket; ${requestId} is ${ticket.state}`,
      );
    }
    const previousState = ticket.state;
    const allocationId = ticket.allocationId;
    if (previousState === "granted") {
      if (!allocationId) {
        throw new PressureProtocolError(
          `granted ticket ${requestId} has no allocation`,
        );
      }
      const allocation = this.requireAllocation(allocationId);
      this.allocations.delete(allocationId);
      this.freeBytes = checkedAdd(this.freeBytes, allocation.bytes, "free bytes");
      delete ticket.allocationId;
    }
    ticket.state = kind === "cancel" ? "cancelled" : "failed";
    this.commit(timestampNs, {
      kind,
      requestId,
      previousState,
      ...(allocationId ? { allocationId } : {}),
    });
  }

  private requireTicket(requestId: string): MutableTicket {
    const ticket = this.tickets.get(requestId);
    if (!ticket) {
      throw new PressureProtocolError(`unknown pressure request ${requestId}`);
    }
    return ticket;
  }

  private requireAllocation(allocationId: string): MutableAllocation {
    const allocation = this.allocations.get(allocationId);
    if (!allocation) {
      throw new PressureProtocolError(`unknown host allocation ${allocationId}`);
    }
    return allocation;
  }

  private validateTimestamp(timestampNs: number): void {
    assertNonNegativeSafeInteger(timestampNs, "timestampNs");
    if (timestampNs < this.lastTimestampNs) {
      throw new PressureProtocolError(
        `timestamp ${timestampNs} precedes last transition ${this.lastTimestampNs}`,
      );
    }
  }

  private commit(timestampNs: number, payload: PressureTracePayload): void {
    this.assertInvariants();
    this.lastTimestampNs = timestampNs;
    this.events.push({
      contractRevision: PROTOCOL_CONTRACT_REVISION,
      sourceId: this.sourceId,
      sourceSequence: this.nextSourceSequence++,
      timestampNs,
      ...payload,
    } as PressureTraceEvent);
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new PressureProtocolError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new PressureProtocolError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new PressureProtocolError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}
