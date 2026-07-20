import {
  PROTOCOL_CONTRACT_REVISION,
  type HostAllocationSnapshot,
  type HostGovernorConfig,
  type HostGovernorSnapshot,
  type PressureReplayResult,
  type PressureTicketSnapshot,
  type PressureTicketState,
  type PressureTraceEvent,
} from "./pressure-types.js";

interface ReplayTicket {
  requestId: string;
  deviceId: string;
  bytes: number;
  generation: number;
  state: PressureTicketState;
  allocationId?: string;
}

interface ReplayAllocation {
  allocationId: string;
  requestId: string;
  deviceId: string;
  bytes: number;
  state: "granted" | "claimed";
}

export class PressureReplayError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PressureReplayError";
  }
}

/**
 * Independently validates and replays a pressure protocol trace.
 *
 * This deliberately does not call HostGovernorSimulator. Keeping a second
 * transition implementation lets replay expose producer bugs instead of
 * reproducing them.
 */
export function replayPressureTrace(
  config: HostGovernorConfig,
  events: readonly PressureTraceEvent[],
): PressureReplayResult {
  const state = createInitialState(config);

  for (let index = 0; index < events.length; index++) {
    const event = events[index];
    try {
      validateEnvelope(state, event, index);
      applyEvent(state, event);
      assertReplayInvariants(state);
      state.lastTimestampNs = event.timestampNs;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new PressureReplayError(`event ${index}: ${message}`);
    }
  }

  return {
    appliedEvents: events.length,
    snapshot: snapshot(state),
  };
}

interface ReplayState {
  capacityBytes: number;
  fixedChargeBytes: number;
  sourceId: string;
  freeBytes: number;
  configurationGeneration: number;
  lastTimestampNs: number;
  reclaimableBytes: Map<string, number>;
  pendingReclaimNotices: Set<string>;
  tickets: Map<string, ReplayTicket>;
  allocations: Map<string, ReplayAllocation>;
}

function createInitialState(config: HostGovernorConfig): ReplayState {
  assertPositiveSafeInteger(config.capacityBytes, "capacityBytes");
  assertNonNegativeSafeInteger(config.fixedChargeBytes, "fixedChargeBytes");
  if (config.fixedChargeBytes > config.capacityBytes) {
    throw new PressureReplayError(
      "fixedChargeBytes must not exceed capacityBytes",
    );
  }

  const sourceId = config.sourceId ?? "host-governor";
  if (sourceId.length === 0) {
    throw new PressureReplayError("sourceId must not be empty");
  }

  const reclaimableBytes = new Map<string, number>();
  let initialCharge = config.fixedChargeBytes;
  for (const [deviceId, bytes] of Object.entries(
    config.reclaimableBytesByDevice ?? {},
  )) {
    if (deviceId.length === 0) {
      throw new PressureReplayError("reclaimable device id must not be empty");
    }
    assertNonNegativeSafeInteger(bytes, `reclaimable bytes for ${deviceId}`);
    initialCharge = checkedAdd(initialCharge, bytes, "initial host charge");
    reclaimableBytes.set(deviceId, bytes);
  }
  if (initialCharge > config.capacityBytes) {
    throw new PressureReplayError(
      `initial host charge ${initialCharge} exceeds capacity ${config.capacityBytes}`,
    );
  }

  return {
    capacityBytes: config.capacityBytes,
    fixedChargeBytes: config.fixedChargeBytes,
    sourceId,
    freeBytes: config.capacityBytes - initialCharge,
    configurationGeneration: 0,
    lastTimestampNs: 0,
    reclaimableBytes,
    pendingReclaimNotices: new Set(),
    tickets: new Map(),
    allocations: new Map(),
  };
}

function validateEnvelope(
  state: ReplayState,
  event: PressureTraceEvent,
  index: number,
): void {
  if (event.contractRevision !== PROTOCOL_CONTRACT_REVISION) {
    fail(
      `contract revision ${event.contractRevision} does not match ${PROTOCOL_CONTRACT_REVISION}`,
    );
  }
  if (event.sourceId !== state.sourceId) {
    fail(`source ${event.sourceId} does not match ${state.sourceId}`);
  }
  if (event.sourceSequence !== index) {
    fail(`source sequence ${event.sourceSequence} does not match ${index}`);
  }
  assertNonNegativeSafeInteger(event.timestampNs, "timestampNs");
  if (event.timestampNs < state.lastTimestampNs) {
    fail(
      `timestamp ${event.timestampNs} precedes ${state.lastTimestampNs}`,
    );
  }
}

function applyEvent(state: ReplayState, event: PressureTraceEvent): void {
  switch (event.kind) {
    case "submit": {
      if (event.requestId.length === 0 || state.tickets.has(event.requestId)) {
        fail(`duplicate or empty request id ${event.requestId}`);
      }
      if (event.deviceId.length === 0) {
        fail("deviceId must not be empty");
      }
      assertPositiveSafeInteger(event.bytes, "request bytes");
      if (event.bytes > state.capacityBytes - state.fixedChargeBytes) {
        fail(`request ${event.bytes} can never fit usable host capacity`);
      }
      if (event.generation !== state.configurationGeneration) {
        fail(
          `submit generation ${event.generation} does not match ${state.configurationGeneration}`,
        );
      }
      state.tickets.set(event.requestId, {
        requestId: event.requestId,
        deviceId: event.deviceId,
        bytes: event.bytes,
        generation: event.generation,
        state: "pending",
      });
      return;
    }

    case "grant": {
      const ticket = requireTicket(state, event.requestId);
      if (ticket.state !== "pending") {
        fail(`grant requires pending ticket; ${event.requestId} is ${ticket.state}`);
      }
      if (ticket.generation !== state.configurationGeneration) {
        fail(`cannot grant stale generation ${ticket.generation}`);
      }
      if (event.bytes !== ticket.bytes) {
        fail(`grant bytes ${event.bytes} do not match request ${ticket.bytes}`);
      }
      if (event.allocationId.length === 0 || state.allocations.has(event.allocationId)) {
        fail(`duplicate or empty allocation id ${event.allocationId}`);
      }
      if (state.freeBytes < event.bytes) {
        fail(`grant ${event.bytes} exceeds ${state.freeBytes} free bytes`);
      }
      state.freeBytes -= event.bytes;
      ticket.state = "granted";
      ticket.allocationId = event.allocationId;
      state.allocations.set(event.allocationId, {
        allocationId: event.allocationId,
        requestId: event.requestId,
        deviceId: ticket.deviceId,
        bytes: event.bytes,
        state: "granted",
      });
      return;
    }

    case "claim": {
      const ticket = requireTicket(state, event.requestId);
      if (ticket.state !== "granted" || ticket.allocationId !== event.allocationId) {
        fail(`claim does not match granted ticket ${event.requestId}`);
      }
      const allocation = requireAllocation(state, event.allocationId);
      if (allocation.state !== "granted") {
        fail(`allocation ${event.allocationId} was already claimed`);
      }
      allocation.state = "claimed";
      ticket.state = "claimed";
      return;
    }

    case "release": {
      const ticket = requireTicket(state, event.requestId);
      if (ticket.state !== "claimed" || ticket.allocationId !== event.allocationId) {
        fail(`release does not match claimed ticket ${event.requestId}`);
      }
      const allocation = requireAllocation(state, event.allocationId);
      if (allocation.state !== "claimed" || allocation.bytes !== event.bytes) {
        fail(`release bytes/state do not match allocation ${event.allocationId}`);
      }
      state.allocations.delete(event.allocationId);
      state.freeBytes = checkedAdd(state.freeBytes, event.bytes, "free bytes");
      ticket.state = "completed";
      delete ticket.allocationId;
      return;
    }

    case "cancel":
    case "timeout": {
      const ticket = requireTicket(state, event.requestId);
      if (
        (ticket.state !== "pending" && ticket.state !== "granted")
        || ticket.state !== event.previousState
      ) {
        fail(
          `${event.kind} previous state ${event.previousState} does not match ${ticket.state}`,
        );
      }
      if (ticket.state === "granted") {
        if (!event.allocationId || ticket.allocationId !== event.allocationId) {
          fail(`${event.kind} lacks matching granted allocation`);
        }
        const allocation = requireAllocation(state, event.allocationId);
        if (allocation.state !== "granted") {
          fail(`${event.kind} cannot revoke claimed allocation`);
        }
        state.allocations.delete(event.allocationId);
        state.freeBytes = checkedAdd(
          state.freeBytes,
          allocation.bytes,
          "free bytes",
        );
        delete ticket.allocationId;
      } else if (event.allocationId !== undefined) {
        fail(`${event.kind} of pending ticket must not name an allocation`);
      }
      ticket.state = event.kind === "cancel" ? "cancelled" : "failed";
      return;
    }

    case "reconfigure": {
      if (event.generation !== state.configurationGeneration + 1) {
        fail(
          `generation ${event.generation} does not follow ${state.configurationGeneration}`,
        );
      }
      const expected = [...state.tickets.values()]
        .filter((ticket) => ticket.state === "pending")
        .map((ticket) => ticket.requestId);
      if (!sameStringSet(expected, event.failedRequestIds)) {
        fail(
          `failed request ids [${event.failedRequestIds.join(",")}] do not match [${expected.join(",")}]`,
        );
      }
      if (new Set(event.failedRequestIds).size !== event.failedRequestIds.length) {
        fail("failed request ids contain duplicates");
      }
      state.configurationGeneration = event.generation;
      for (const requestId of expected) {
        requireTicket(state, requestId).state = "failed";
      }
      return;
    }

    case "reclaim_notice": {
      if ((state.reclaimableBytes.get(event.deviceId) ?? 0) <= 0) {
        fail(`device ${event.deviceId} has no reclaimable host bytes`);
      }
      if (state.pendingReclaimNotices.has(event.deviceId)) {
        fail(`device ${event.deviceId} already has a reclaim notice`);
      }
      const needsPressure = [...state.tickets.values()].some(
        (ticket) => ticket.state === "pending" && ticket.bytes > state.freeBytes,
      );
      if (!needsPressure) {
        fail("reclaim notice has no unsatisfied pending request");
      }
      state.pendingReclaimNotices.add(event.deviceId);
      return;
    }

    case "reclaim": {
      assertPositiveSafeInteger(event.bytes, "reclaim bytes");
      if (!state.pendingReclaimNotices.has(event.deviceId)) {
        fail(`device ${event.deviceId} has no pending reclaim notice`);
      }
      const reclaimable = state.reclaimableBytes.get(event.deviceId) ?? 0;
      if (event.bytes > reclaimable) {
        fail(
          `reclaim ${event.bytes} exceeds ${reclaimable} bytes for ${event.deviceId}`,
        );
      }
      state.reclaimableBytes.set(event.deviceId, reclaimable - event.bytes);
      state.pendingReclaimNotices.delete(event.deviceId);
      state.freeBytes = checkedAdd(state.freeBytes, event.bytes, "free bytes");
      return;
    }
  }
}

function assertReplayInvariants(state: ReplayState): void {
  assertNonNegativeSafeInteger(state.freeBytes, "free bytes");
  let ledger = checkedAdd(state.fixedChargeBytes, state.freeBytes, "ledger");
  for (const bytes of state.reclaimableBytes.values()) {
    assertNonNegativeSafeInteger(bytes, "reclaimable bytes");
    ledger = checkedAdd(ledger, bytes, "ledger");
  }
  for (const allocation of state.allocations.values()) {
    assertPositiveSafeInteger(allocation.bytes, "allocation bytes");
    ledger = checkedAdd(ledger, allocation.bytes, "ledger");
  }
  if (ledger !== state.capacityBytes) {
    fail(`capacity ledger ${ledger} does not equal ${state.capacityBytes}`);
  }

  for (const ticket of state.tickets.values()) {
    const allocation = ticket.allocationId
      ? state.allocations.get(ticket.allocationId)
      : undefined;
    if (ticket.state === "pending" && ticket.generation !== state.configurationGeneration) {
      fail(`pending ticket ${ticket.requestId} has stale generation`);
    }
    if (ticket.state === "granted" || ticket.state === "claimed") {
      if (
        !allocation
        || allocation.state !== ticket.state
        || allocation.requestId !== ticket.requestId
        || allocation.deviceId !== ticket.deviceId
        || allocation.bytes !== ticket.bytes
      ) {
        fail(`${ticket.state} ticket ${ticket.requestId} has no exact allocation`);
      }
    } else if (allocation) {
      fail(`inactive ticket ${ticket.requestId} retains an allocation`);
    }
  }

  for (const allocation of state.allocations.values()) {
    const ticket = state.tickets.get(allocation.requestId);
    if (!ticket || ticket.allocationId !== allocation.allocationId) {
      fail(`orphan allocation ${allocation.allocationId}`);
    }
  }
}

function snapshot(state: ReplayState): HostGovernorSnapshot {
  const reclaimableBytesByDevice = Object.fromEntries(
    [...state.reclaimableBytes.entries()].sort(([left], [right]) =>
      left.localeCompare(right)
    ),
  );
  const tickets: PressureTicketSnapshot[] = [...state.tickets.values()]
    .map((ticket) => ({ ...ticket }))
    .sort((left, right) => left.requestId.localeCompare(right.requestId));
  const allocations: HostAllocationSnapshot[] = [...state.allocations.values()]
    .map((allocation) => ({ ...allocation }))
    .sort((left, right) => left.allocationId.localeCompare(right.allocationId));

  return {
    capacityBytes: state.capacityBytes,
    fixedChargeBytes: state.fixedChargeBytes,
    freeBytes: state.freeBytes,
    configurationGeneration: state.configurationGeneration,
    reclaimableBytesByDevice,
    pendingReclaimNotices: [...state.pendingReclaimNotices].sort(),
    tickets,
    allocations,
  };
}

function requireTicket(state: ReplayState, requestId: string): ReplayTicket {
  const ticket = state.tickets.get(requestId);
  if (!ticket) {
    fail(`unknown pressure request ${requestId}`);
  }
  return ticket;
}

function requireAllocation(
  state: ReplayState,
  allocationId: string,
): ReplayAllocation {
  const allocation = state.allocations.get(allocationId);
  if (!allocation) {
    fail(`unknown host allocation ${allocationId}`);
  }
  return allocation;
}

function sameStringSet(
  left: readonly string[],
  right: readonly string[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const rightSet = new Set(right);
  return left.every((value) => rightSet.has(value));
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    fail(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail(`${label} must be a non-negative safe integer; got ${value}`);
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail(`${label} must be a positive safe integer; got ${value}`);
  }
}

function fail(message: string): never {
  throw new Error(message);
}
