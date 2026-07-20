import { describe, expect, it } from "vitest";
import {
  ServingProtocolError,
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
      && event.batch.decodeRequestIds.length > 0
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
  });
});
