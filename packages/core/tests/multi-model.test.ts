import { describe, expect, it } from "vitest";
import {
  MultiModelProtocolError,
  buildModelProfile,
  replayMultiModelTrace,
  simulateMultiModelWorkload,
  tenantFootprintBytes,
  type MultiModelBatchDurationEstimator,
  type MultiModelConfig,
  type MultiModelTenantSpec,
} from "../src/index.js";

const GiB = 1024 ** 3;

// Cheap linear cost model: the point of these tests is residency behaviour,
// not timing fidelity.
const duration: MultiModelBatchDurationEstimator = (batch) => (
  1_000 + batch.tokenWork * 20_000
);

function tenant(
  id: string,
  weightGiB: number,
  overrides: Partial<MultiModelTenantSpec> = {},
): MultiModelTenantSpec {
  return {
    id,
    displayName: id,
    weightBytes: Math.round(weightGiB * GiB),
    kvBytesPerToken: 128 * 1024,
    maxKvTokens: 1_024,
    pinned: false,
    requests: [
      { id: `${id}-r0`, arrivalNs: 0, promptTokens: 64, outputTokens: 8 },
    ],
    ...overrides,
  };
}

function config(overrides: Partial<MultiModelConfig> = {}): MultiModelConfig {
  return {
    tenants: [tenant("chat", 4), tenant("draft", 1)],
    deviceMemoryBytes: 24 * GiB,
    loadBandwidthBytesPerSec: 25e9,
    maxBatchTokens: 128,
    prefillChunkTokens: 64,
    ...overrides,
  };
}

describe("co-residency", () => {
  it("keeps every model resident when the working set fits", () => {
    const result = simulateMultiModelWorkload(config(), duration);

    expect(result.metrics.fitsWithoutSwapping).toBe(true);
    expect(result.metrics.reloadsPerRequest).toBe(0);
    expect(result.metrics.totalEvictions).toBe(0);
    // Each model is loaded exactly once, which is unavoidable.
    for (const model of result.metrics.tenants) {
      expect(model.loads, model.tenantId).toBe(1);
      expect(model.evictions, model.tenantId).toBe(0);
      expect(model.rePrefilledTokens, model.tenantId).toBe(0);
    }
    expect(result.metrics.peakResidentBytes)
      .toBeLessThanOrEqual(result.metrics.deviceMemoryBytes);
    expect(result.replay.completedRequests).toBe(2);
  });

  it("swaps models when the working set does not fit", () => {
    // Two 10 GiB models on a 16 GiB device: only one can be resident.
    const contended = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("chat", 10, {
          requests: [
            { id: "c0", arrivalNs: 0, promptTokens: 64, outputTokens: 4 },
            // Arrives after the image model has taken the device, so the chat
            // model has to be loaded a second time.
            { id: "c1", arrivalNs: 3_000_000_000, promptTokens: 64, outputTokens: 4 },
          ],
        }),
        tenant("image", 10, {
          requests: [
            { id: "i0", arrivalNs: 500_000_000, promptTokens: 64, outputTokens: 4 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(contended, duration);

    expect(result.metrics.fitsWithoutSwapping).toBe(false);
    expect(result.metrics.totalEvictions).toBeGreaterThan(0);
    expect(result.metrics.totalLoads).toBeGreaterThan(2);
    expect(result.metrics.reloadsPerRequest).toBeGreaterThan(0);
    // Swapping is visible as time requests spent waiting to be resident.
    expect(
      result.metrics.tenants.reduce((sum, t) => sum + t.residencyWaitNs, 0),
    ).toBeGreaterThan(0);
    expect(result.replay.completedRequests).toBe(3);
  });

  it("never exceeds the device budget or runs an evicted model", () => {
    const contended = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("a", 9, {
          requests: [
            { id: "a0", arrivalNs: 0, promptTokens: 64, outputTokens: 6 },
            { id: "a1", arrivalNs: 60_000_000, promptTokens: 64, outputTokens: 6 },
          ],
        }),
        tenant("b", 9, {
          requests: [
            { id: "b0", arrivalNs: 10_000_000, promptTokens: 64, outputTokens: 6 },
            { id: "b1", arrivalNs: 80_000_000, promptTokens: 64, outputTokens: 6 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(contended, duration);

    const resident = new Set<string>();
    let bytes = 0;
    for (const event of result.trace) {
      if (event.kind === "load_complete") {
        resident.add(event.tenantId);
        bytes = event.residentBytes;
        expect(bytes).toBeLessThanOrEqual(contended.deviceMemoryBytes);
      }
      if (event.kind === "evict") {
        expect(resident.has(event.tenantId)).toBe(true);
        resident.delete(event.tenantId);
        bytes -= event.bytes;
      }
      // The device can only compute for a model it is currently holding.
      if (event.kind === "batch_start") {
        expect(resident.has(event.batch.tenantId)).toBe(true);
      }
      if (event.kind === "batch_finish") {
        expect(resident.has(event.tenantId)).toBe(true);
      }
    }
    expect(bytes).toBe(result.replay.finalResidentBytes);
  });

  it("charges a re-prefill when an eviction discards partial work", () => {
    const contended = config({
      deviceMemoryBytes: 16 * GiB,
      maxBatchTokens: 64,
      prefillChunkTokens: 32,
      tenants: [
        // One batch retires the short request and starts the long one, so the
        // model becomes evictable while a prefill is still part way through.
        tenant("a", 9, {
          requests: [
            { id: "a0", arrivalNs: 0, promptTokens: 32, outputTokens: 1 },
            { id: "a1", arrivalNs: 0, promptTokens: 512, outputTokens: 8 },
          ],
        }),
        tenant("b", 9, {
          requests: [
            { id: "b0", arrivalNs: 5_000_000, promptTokens: 64, outputTokens: 4 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(contended, duration);

    expect(result.metrics.totalEvictions).toBeGreaterThan(0);
    // Losing the arena mid-prefill means those prompt tokens are recomputed.
    expect(
      result.metrics.tenants.reduce((sum, t) => sum + t.rePrefilledTokens, 0),
    ).toBeGreaterThan(0);
    const discarded = result.trace
      .filter((event) => event.kind === "evict")
      .some((event) => event.kind === "evict" && event.discardedKvTokens > 0);
    expect(discarded).toBe(true);
  });

  it("never evicts a pinned model", () => {
    const withPinned = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("assistant", 8, { pinned: true }),
        tenant("image", 6, {
          requests: [
            { id: "i0", arrivalNs: 10_000_000, promptTokens: 64, outputTokens: 4 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(withPinned, duration);

    expect(
      result.trace.some(
        (event) => event.kind === "evict" && event.tenantId === "assistant",
      ),
    ).toBe(false);
    const pinned = result.metrics.tenants.find(
      (model) => model.tenantId === "assistant",
    )!;
    expect(pinned.loads).toBe(1);
    expect(pinned.evictions).toBe(0);
  });

  it("rejects a model that could never be resident", () => {
    expect(() => simulateMultiModelWorkload(
      config({
        deviceMemoryBytes: 4 * GiB,
        tenants: [tenant("huge", 8)],
      }),
      duration,
    )).toThrow(/needs .* resident bytes but the device holds/);
  });

  it("rejects a pinned set that starves an unpinned model", () => {
    // Pinning 14 GiB of a 16 GiB device leaves no room for the 6 GiB model,
    // which could then never be made resident however much is evicted.
    expect(() => simulateMultiModelWorkload(
      config({
        deviceMemoryBytes: 16 * GiB,
        tenants: [
          tenant("pinned-a", 7, { pinned: true }),
          tenant("pinned-b", 7, { pinned: true }),
          tenant("image", 6),
        ],
      }),
      duration,
    )).toThrow(/too little for the largest unpinned model/);
  });

  it("rejects a request that cannot fit its model's KV arena", () => {
    expect(() => simulateMultiModelWorkload(
      config({
        tenants: [
          tenant("chat", 4, {
            maxKvTokens: 32,
            requests: [
              { id: "big", arrivalNs: 0, promptTokens: 64, outputTokens: 8 },
            ],
          }),
        ],
      }),
      duration,
    )).toThrow(/KV tokens but its arena holds/);
  });

  it("hides transfer behind another model's compute", () => {
    // One model computes a long batch while the other loads over the link.
    const overlapped = config({
      deviceMemoryBytes: 24 * GiB,
      tenants: [
        tenant("chat", 4, {
          requests: [
            { id: "c0", arrivalNs: 0, promptTokens: 128, outputTokens: 16 },
          ],
        }),
        tenant("draft", 2, {
          requests: [
            { id: "d0", arrivalNs: 1_000, promptTokens: 64, outputTokens: 4 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(overlapped, duration);

    expect(result.metrics.hiddenTransferNs).toBeGreaterThan(0);
    expect(result.metrics.hiddenTransferNs)
      .toBeLessThanOrEqual(result.metrics.transferServiceNs);
  });

  it("replays a trace independently and rejects a mutated one", () => {
    const scenario = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("a", 9, {
          requests: [
            { id: "a0", arrivalNs: 0, promptTokens: 64, outputTokens: 6 },
          ],
        }),
        tenant("b", 9, {
          requests: [
            { id: "b0", arrivalNs: 20_000_000, promptTokens: 64, outputTokens: 6 },
          ],
        }),
      ],
    });
    const result = simulateMultiModelWorkload(scenario, duration);

    expect(replayMultiModelTrace(scenario, result.trace, duration))
      .toEqual(result.replay);

    // Dropping an eviction leaves the ledger over capacity.
    const withoutEvict = result.trace.filter(
      (event) => event.kind !== "evict",
    );
    expect(() => replayMultiModelTrace(scenario, withoutEvict, duration))
      .toThrow(MultiModelProtocolError);

    // A batch attributed to a model that is not resident must be rejected.
    const firstBatch = result.trace.findIndex(
      (event) => event.kind === "batch_start",
    );
    const mutated = result.trace.map((event, index) => (
      index === firstBatch && event.kind === "batch_start"
        ? { ...event, batch: { ...event.batch, tenantId: "b" } }
        : event
    ));
    expect(() => replayMultiModelTrace(scenario, mutated, duration))
      .toThrow(MultiModelProtocolError);
  });

  it("is deterministic across repeated runs", () => {
    const scenario = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("a", 9, {
          requests: [
            { id: "a0", arrivalNs: 0, promptTokens: 64, outputTokens: 6 },
            { id: "a1", arrivalNs: 50_000_000, promptTokens: 64, outputTokens: 6 },
          ],
        }),
        tenant("b", 9, {
          requests: [
            { id: "b0", arrivalNs: 10_000_000, promptTokens: 64, outputTokens: 6 },
          ],
        }),
      ],
    });

    expect(simulateMultiModelWorkload(scenario, duration).trace)
      .toEqual(simulateMultiModelWorkload(scenario, duration).trace);
  });

  it("bounds loads by batches, so a contended device still progresses", () => {
    // Every load must be followed by at least one batch before that model can
    // be evicted again, which is what rules out reloading without running.
    const thrashing = config({
      deviceMemoryBytes: 16 * GiB,
      tenants: [
        tenant("a", 9, {
          requests: Array.from({ length: 4 }, (_, index) => ({
            id: `a${index}`,
            arrivalNs: index * 30_000_000,
            promptTokens: 64,
            outputTokens: 4,
          })),
        }),
        tenant("b", 9, {
          requests: Array.from({ length: 4 }, (_, index) => ({
            id: `b${index}`,
            arrivalNs: index * 30_000_000 + 15_000_000,
            promptTokens: 64,
            outputTokens: 4,
          })),
        }),
      ],
    });
    const result = simulateMultiModelWorkload(thrashing, duration);
    const batches = result.trace.filter(
      (event) => event.kind === "batch_start",
    ).length;

    expect(result.metrics.totalLoads)
      .toBeLessThanOrEqual(batches + thrashing.tenants.length);
    expect(result.replay.completedRequests).toBe(8);
  });

  it("reports the swap cost of a real local pair that does not fit", () => {
    // Qwen3-32B int4 beside SDXL on a 24 GiB consumer card. Together their
    // footprints exceed the device, so the pair cannot stay resident.
    const chat = buildModelProfile("qwen3-32b", "int4", "fp16");
    const image = buildModelProfile("stable-diffusion-xl", "fp16", "fp16");
    const weightBytes = (model: typeof chat) => (
      model.layers.reduce(
        (sum, layer) => sum + layer.attentionBytes + layer.ffnBytes,
        0,
      )
      + (model.components ?? []).reduce(
        (sum, component) => sum + component.weightBytes,
        0,
      )
      + (model.embeddingBytes ?? 0)
    );
    const kvPerToken = chat.layers.reduce(
      (sum, layer) => sum + layer.kvCachePerToken,
      0,
    );
    const local: MultiModelConfig = {
      deviceMemoryBytes: 24 * GiB,
      loadBandwidthBytesPerSec: 25e9,
      maxBatchTokens: 256,
      prefillChunkTokens: 128,
      tenants: [
        {
          id: "chat",
          displayName: chat.name,
          weightBytes: Math.round(weightBytes(chat)),
          kvBytesPerToken: kvPerToken,
          maxKvTokens: 16_384,
          pinned: false,
          requests: [
            { id: "c0", arrivalNs: 0, promptTokens: 512, outputTokens: 16 },
            { id: "c1", arrivalNs: 3_000_000_000, promptTokens: 512, outputTokens: 16 },
          ],
        },
        {
          id: "image",
          displayName: image.name,
          weightBytes: Math.round(weightBytes(image)),
          kvBytesPerToken: 0,
          maxKvTokens: 64,
          pinned: false,
          requests: [
            { id: "i0", arrivalNs: 1_500_000_000, promptTokens: 64, outputTokens: 1 },
          ],
        },
      ],
    };
    const chatFootprint = tenantFootprintBytes(local.tenants[0]!);
    const imageFootprint = tenantFootprintBytes(local.tenants[1]!);

    // The pair does not fit together, but each fits alone.
    expect(chatFootprint + imageFootprint)
      .toBeGreaterThan(local.deviceMemoryBytes);
    expect(chatFootprint).toBeLessThan(local.deviceMemoryBytes);

    const result = simulateMultiModelWorkload(local, duration);
    expect(result.metrics.fitsWithoutSwapping).toBe(false);
    expect(result.metrics.totalLoadedBytes)
      .toBeGreaterThan(chatFootprint + imageFootprint);
  });
});
