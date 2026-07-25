import { describe, expect, it } from "vitest";
import {
  ServingProtocolError,
  defaultSpeculativeEligibility,
  replayServingTrace,
  simulateServingWorkload,
  type ServingBatchDurationEstimator,
  type ServingSchedulerConfig,
} from "../src/index.js";

const duration: ServingBatchDurationEstimator = (batch) => (
  10 + batch.tokenWork * 5
);

function config(): ServingSchedulerConfig {
  return {
    requests: [
      {
        id: "a",
        arrivalNs: 0,
        promptTokens: 6,
        outputTokens: 3,
      },
      {
        id: "b",
        arrivalNs: 8,
        promptTokens: 4,
        outputTokens: 2,
      },
      {
        id: "priority",
        arrivalNs: 8,
        promptTokens: 2,
        outputTokens: 2,
        priority: 1,
      },
    ],
    maxBatchSize: 2,
    maxBatchTokens: 5,
    prefillChunkTokens: 3,
    maxKvTokens: 20,
  };
}

describe("continuous serving scheduler", () => {
  it("overlaps arrivals with chunked prefill and decode-first batches", () => {
    const result = simulateServingWorkload(config(), duration);
    const starts = result.trace.filter((event) => event.kind === "batch_start");

    expect(result.replay.appliedEvents).toBe(result.trace.length);
    expect(result.metrics).toMatchObject({
      requests: 3,
      outputTokens: 7,
      prefillTokens: 12,
      decodeTokens: 4,
    });
    expect(result.requests.map((request) => request.id)).toEqual([
      "a",
      "priority",
      "b",
    ]);
    expect(starts.some((event) => (
      event.kind === "batch_start"
      && event.batch.decode.length > 0
      && event.batch.prefill.length > 0
    ))).toBe(true);
    expect(result.requests.every((request) => (
      request.firstTokenNs >= request.arrivalNs
      && request.completedAtNs >= request.firstTokenNs
    ))).toBe(true);
    expect(result.metrics.kvHighWaterTokens).toBeLessThanOrEqual(20);
    expect(result.trace.at(-1)?.kind).toBe("terminal");
  });

  it("is byte deterministic and independently replayable", () => {
    const first = simulateServingWorkload(config(), duration);
    const second = simulateServingWorkload(config(), duration);

    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    expect(replayServingTrace(config(), first.trace, duration)).toEqual(
      first.replay,
    );
  });

  it("binds duration estimation and replay to the batch start time", () => {
    const calls: Array<{ batchId: number; startedAtNs: number }> = [];
    const timedDuration: ServingBatchDurationEstimator = (
      batch,
      startedAtNs,
    ) => {
      calls.push({ batchId: batch.batchId, startedAtNs });
      return duration(batch, startedAtNs);
    };
    const result = simulateServingWorkload(config(), timedDuration);
    const starts = result.trace.flatMap((event) => (
      event.kind === "batch_start"
        ? [{ batchId: event.batch.batchId, startedAtNs: event.atNs }]
        : []
    ));

    expect(calls).toEqual([...starts, ...starts]);
    expect(starts.some((start) => start.startedAtNs > 0)).toBe(true);
  });

  it("rejects the shortest mutated scheduler decision", () => {
    const result = simulateServingWorkload(config(), duration);
    const index = result.trace.findIndex((event) => event.kind === "batch_start");
    const event = result.trace[index];
    if (event?.kind !== "batch_start") {
      throw new Error("missing batch start");
    }
    const mutated = result.trace.map((entry, entryIndex) => (
      entryIndex === index
        ? {
            ...event,
            batch: {
              ...event.batch,
              tokenWork: event.batch.tokenWork + 1,
            },
          }
        : entry
    ));

    expect(() => replayServingTrace(config(), mutated, duration))
      .toThrow("violates scheduler decision");
  });

  it("enforces transient KV capacity and frees completed requests", () => {
    const constrained: ServingSchedulerConfig = {
      requests: [
        { id: "first", arrivalNs: 0, promptTokens: 4, outputTokens: 2 },
        { id: "second", arrivalNs: 0, promptTokens: 4, outputTokens: 2 },
      ],
      maxBatchSize: 2,
      maxBatchTokens: 4,
      prefillChunkTokens: 8,
      maxKvTokens: 5,
    };
    const result = simulateServingWorkload(constrained, duration);
    const firstBatch = result.trace.find(
      (event) => event.kind === "batch_start",
    );

    expect(result.metrics.kvHighWaterTokens).toBe(5);
    expect(
      firstBatch?.kind === "batch_start"
        ? firstBatch.batch.prefill.map((entry) => entry.requestId)
        : [],
    ).toEqual(["first"]);
    expect(result.requests[1].firstTokenNs).toBeGreaterThanOrEqual(
      result.requests[0].completedAtNs,
    );
  });

  it("rejects requests that can never fit and invalid duration models", () => {
    expect(() => simulateServingWorkload({
      ...config(),
      maxKvTokens: 7,
    }, duration)).toThrow("requires 8 KV tokens");
    expect(() => simulateServingWorkload(config(), () => 0))
      .toThrowError(ServingProtocolError);
    expect(() => simulateServingWorkload({
      ...config(),
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 1,
        acceptance: {
          kind: "replay",
          acceptedAdditionalTokensByRequest: {
            a: [0],
            b: [0],
            priority: [0],
            typo: [0],
          },
        },
      },
    }, duration)).toThrow("unknown request typo");
  });

  it("commits speculative bursts through per-request transactions", () => {
    const speculative: ServingSchedulerConfig = {
      requests: [
        { id: "bonus", arrivalNs: 0, promptTokens: 4, outputTokens: 5 },
        { id: "mixed", arrivalNs: 0, promptTokens: 4, outputTokens: 5 },
      ],
      maxBatchSize: 2,
      maxBatchTokens: 6,
      prefillChunkTokens: 4,
      maxKvTokens: 16,
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 2,
        acceptance: {
          kind: "replay",
          acceptedAdditionalTokensByRequest: {
            bonus: [2],
            mixed: [0, 1],
          },
        },
      },
    };
    const result = simulateServingWorkload(speculative, duration);
    const decode = result.trace.flatMap((event) => (
      event.kind === "batch_start" ? event.batch.decode : []
    ));

    expect(result.replay.completedRequests).toBe(2);
    expect(result.metrics).toMatchObject({
      outputTokens: 10,
      decodeTokens: 8,
      targetForwards: 3,
      targetVerificationTokens: 11,
      guaranteedTargetTokens: 3,
      proposedAdditionalTokens: 5,
      acceptedAdditionalTokens: 3,
      proposedDraftTokens: 8,
      acceptedDraftTokens: 6,
      rejectedDraftTokens: 2,
      targetAuthoritativeTokens: 2,
      committedTokensPerTargetForward: 8 / 3,
    });
    expect(decode.map((entry) => entry.outcome)).toEqual([
      "bonus",
      "correction",
      "accepted_tail",
    ]);
    expect(result.trace.some((event) => (
      event.kind === "batch_finish"
      && event.emittedTokens.some((token) => (
        token.source === "speculative_accepted"
      ))
      && event.emittedTokens.some((token) => (
        token.source === "speculative_authoritative"
      ))
    ))).toBe(true);
    expect(result.requests.every((request) => (
      request.tokenTimestampsNs.length === 5
    ))).toBe(true);
  });

  it("rejects a mutated speculative accepted prefix during replay", () => {
    const speculative: ServingSchedulerConfig = {
      requests: [
        { id: "request", arrivalNs: 0, promptTokens: 4, outputTokens: 4 },
      ],
      maxBatchSize: 1,
      maxBatchTokens: 4,
      prefillChunkTokens: 4,
      maxKvTokens: 7,
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 2,
        acceptance: {
          kind: "replay",
          acceptedAdditionalTokensByRequest: { request: [1, 0] },
        },
      },
    };
    const result = simulateServingWorkload(speculative, duration);
    const index = result.trace.findIndex((event) => (
      event.kind === "batch_start" && event.batch.decode.length > 0
    ));
    const event = result.trace[index];
    if (event?.kind !== "batch_start" || event.batch.decode[0] === undefined) {
      throw new Error("missing speculative batch");
    }
    const firstDecode = event.batch.decode[0];
    const mutated = result.trace.map((entry, entryIndex) => (
      entryIndex === index
        ? {
            ...event,
            batch: {
              ...event.batch,
              decode: [{
                ...firstDecode,
                acceptedDraftTokens: 0,
                committedTokens: 1,
                outcome: "correction" as const,
              }],
              expectedOutputTokens: 1,
            },
          }
        : entry
    ));

    expect(() => replayServingTrace(speculative, mutated, duration))
      .toThrow("violates scheduler decision");
  });

  it("never admits prefill that would strand KV between requests", () => {
    // Two requests each peak at 6 KV tokens (5 prompt + 1 uncharged prefill
    // token + 1 decode token) against a 10-token pool. Admitting both prompts
    // fills the pool and leaves neither able to decode.
    const contended: ServingSchedulerConfig = {
      requests: [
        { id: "a", arrivalNs: 0, promptTokens: 5, outputTokens: 2 },
        { id: "b", arrivalNs: 0, promptTokens: 5, outputTokens: 2 },
      ],
      maxBatchSize: 2,
      maxBatchTokens: 16,
      prefillChunkTokens: 8,
      maxKvTokens: 10,
    };
    const result = simulateServingWorkload(contended, duration);

    expect(result.metrics.requests).toBe(2);
    expect(result.metrics.outputTokens).toBe(4);
    expect(result.metrics.kvHighWaterTokens).toBeLessThanOrEqual(10);
    expect(result.replay.completedRequests).toBe(2);
    expect(result.replay.finalKvTokens).toBe(0);
    const firstBatch = result.trace.find((event) => event.kind === "batch_start");
    if (firstBatch?.kind !== "batch_start") {
      throw new Error("missing batch");
    }
    expect(firstBatch.batch.prefill).toHaveLength(1);
  });

  it("serializes admission when every request needs most of the KV pool", () => {
    const contended: ServingSchedulerConfig = {
      requests: Array.from({ length: 6 }, (_, index) => ({
        id: `r${index}`,
        arrivalNs: 0,
        promptTokens: 12,
        outputTokens: 5,
      })),
      maxBatchSize: 4,
      maxBatchTokens: 32,
      prefillChunkTokens: 4,
      maxKvTokens: 40,
    };
    const result = simulateServingWorkload(contended, duration);

    expect(result.metrics.requests).toBe(6);
    expect(result.metrics.outputTokens).toBe(30);
    expect(result.metrics.kvHighWaterTokens).toBeLessThanOrEqual(40);
    expect(result.replay.finalKvTokens).toBe(0);
  });
});
