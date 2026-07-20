import { describe, expect, it } from "vitest";
import {
  HostGovernorSimulator,
  PressureProtocolError,
  PressureReplayError,
  replayPressureTrace,
  type HostGovernorConfig,
  type PressureTraceEvent,
} from "../src/index.js";

const config: HostGovernorConfig = {
  capacityBytes: 100,
  fixedChargeBytes: 10,
  reclaimableBytesByDevice: {
    gpu0: 40,
  },
  sourceId: "test-governor",
};

describe("HostGovernorSimulator", () => {
  it("round-trips an exact allocation ledger through independent replay", () => {
    const governor = new HostGovernorSimulator(config);
    const requestId = governor.submit("gpu1", 30, 1);
    const allocationId = governor.grant(requestId, 2);
    expect(allocationId).toBe("host-allocation-1");
    governor.claim(requestId, 3);
    governor.release(requestId, 4);

    const replay = replayPressureTrace(config, governor.trace());
    expect(replay.appliedEvents).toBe(4);
    expect(replay.snapshot).toEqual(governor.snapshot());
    expect(replay.snapshot.freeBytes).toBe(50);
  });

  it("models reclaim by the requesting device and generation failure", () => {
    const governor = new HostGovernorSimulator(config);
    const blocked = governor.submit("gpu0", 70, 1);

    expect(() => governor.grant(blocked, 2)).toThrowError(
      "cannot grant 70 bytes with 50 free",
    );
    governor.sendReclaimNotice("gpu0", 2);
    governor.reclaim("gpu0", 25, 3);
    governor.grant(blocked, 4);

    const stale = governor.submit("gpu1", 5, 5);
    expect(governor.reconfigure(6)).toEqual([stale]);

    const replay = replayPressureTrace(config, governor.trace());
    expect(replay.snapshot).toEqual(governor.snapshot());
    expect(replay.snapshot.configurationGeneration).toBe(1);
    expect(
      replay.snapshot.tickets.find((ticket) => ticket.requestId === stale)?.state,
    ).toBe("failed");
  });

  it("returns an unclaimed grant exactly once on cancel or timeout", () => {
    const governor = new HostGovernorSimulator(config);
    const cancelled = governor.submit("gpu1", 20, 1);
    governor.grant(cancelled, 2);
    governor.cancel(cancelled, 3);

    const timedOut = governor.submit("gpu1", 15, 4);
    governor.grant(timedOut, 5);
    governor.timeout(timedOut, 6);

    expect(governor.snapshot().freeBytes).toBe(50);
    expect(() => governor.cancel(cancelled, 7)).toThrowError(
      PressureProtocolError,
    );
    expect(replayPressureTrace(config, governor.trace()).snapshot).toEqual(
      governor.snapshot(),
    );
  });

  it("emits byte-for-byte deterministic traces", () => {
    function createTrace(): readonly PressureTraceEvent[] {
      const governor = new HostGovernorSimulator(config);
      const requestId = governor.submit("gpu1", 10, 10);
      governor.grant(requestId, 10);
      governor.claim(requestId, 11);
      return governor.trace();
    }

    expect(JSON.stringify(createTrace())).toBe(JSON.stringify(createTrace()));
  });

  it("allows a valid zero-free initial ledger", () => {
    const zeroFreeConfig: HostGovernorConfig = {
      capacityBytes: 100,
      fixedChargeBytes: 60,
      reclaimableBytesByDevice: { gpu0: 40 },
    };
    const governor = new HostGovernorSimulator(zeroFreeConfig);
    const requestId = governor.submit("gpu0", 40, 1);
    governor.sendReclaimNotice("gpu0", 2);
    governor.reclaim("gpu0", 40, 3);
    governor.grant(requestId, 4);

    expect(governor.snapshot().freeBytes).toBe(0);
    expect(replayPressureTrace(zeroFreeConfig, governor.trace()).snapshot).toEqual(
      governor.snapshot(),
    );
  });
});

describe("replayPressureTrace", () => {
  it("rejects a corrupted grant at the shortest bad prefix", () => {
    const governor = new HostGovernorSimulator(config);
    const requestId = governor.submit("gpu1", 30, 1);
    governor.grant(requestId, 2);
    const trace = governor.trace().map((event) => ({ ...event }));
    const grant = trace[1];
    if (grant.kind !== "grant") {
      throw new Error("test fixture did not produce grant");
    }
    trace[1] = { ...grant, bytes: grant.bytes + 1 };

    expect(() => replayPressureTrace(config, trace)).toThrowError(
      /^event 1: grant bytes 31 do not match request 30$/,
    );
  });

  it("rejects source sequence gaps before applying the event", () => {
    const governor = new HostGovernorSimulator(config);
    governor.submit("gpu1", 10, 1);
    const trace = governor.trace().map((event) => ({ ...event }));
    trace[0] = { ...trace[0], sourceSequence: 3 } as PressureTraceEvent;

    expect(() => replayPressureTrace(config, trace)).toThrowError(
      PressureReplayError,
    );
    expect(() => replayPressureTrace(config, trace)).toThrowError(
      "event 0: source sequence 3 does not match 0",
    );
  });

  it("rejects duplicate reconfiguration failure ids", () => {
    const governor = new HostGovernorSimulator(config);
    const requestId = governor.submit("gpu1", 10, 1);
    governor.reconfigure(2);
    const trace = governor.trace().map((event) => (
      event.kind === "reconfigure"
        ? { ...event, failedRequestIds: [requestId, requestId] }
        : { ...event }
    ));

    expect(() => replayPressureTrace(config, trace)).toThrowError(/^event 1:/);
  });
});
