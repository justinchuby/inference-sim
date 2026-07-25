import { describe, expect, it } from "vitest";
import {
  MODEL_SPECS,
  buildModelProfile,
  derivedTotalParams,
  listModelPresets,
} from "../src/index.js";

describe("model presets", () => {
  it("derives a parameter count that matches every published figure", () => {
    const drift = Object.entries(MODEL_SPECS).map(([preset, spec]) => {
      const derived = derivedTotalParams(spec);
      return {
        preset,
        ratio: derived / spec.publishedTotalParams,
      };
    });
    for (const entry of drift) {
      // Normalization, bias, and multimodal towers are outside the model.
      expect(entry.ratio, entry.preset).toBeGreaterThan(0.88);
      expect(entry.ratio, entry.preset).toBeLessThan(1.08);
    }
  });

  it("keeps every preset internally consistent", () => {
    for (const preset of listModelPresets()) {
      const model = buildModelProfile(preset, "fp16", "fp16");
      const spec = MODEL_SPECS[preset]!;

      expect(model.layers, preset).toHaveLength(spec.numLayers);
      expect(model.totalParams, preset).toBe(derivedTotalParams(spec));
      expect(model.embeddingBytes, preset).toBeGreaterThan(0);
      expect(model.architecture.numKVHeads, preset)
        .toBeLessThanOrEqual(model.architecture.numHeads);
      for (const layer of model.layers) {
        expect(layer.attentionBytes, preset).toBeGreaterThan(0);
        expect(layer.ffnBytes, preset).toBeGreaterThanOrEqual(0);
        expect(layer.kvCachePerToken, preset).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(layer.attentionBytes), preset).toBe(true);
      }
      // Every layer holds either dense FFN weights or routed experts.
      const denseFfn = model.layers.reduce(
        (sum, layer) => sum + layer.ffnBytes,
        0,
      );
      expect(denseFfn > 0 || model.moe !== undefined, preset).toBe(true);
      // At least one layer must cache KV or the model cannot decode.
      expect(
        model.layers.some((layer) => layer.kvCachePerToken > 0),
        preset,
      ).toBe(true);
    }
  });

  it("reconstructs total weight bytes from layers, experts, and embeddings", () => {
    for (const preset of listModelPresets()) {
      const model = buildModelProfile(preset, "fp16", "fp16");
      const spec = MODEL_SPECS[preset]!;
      const dense = model.layers.reduce(
        (sum, layer) => sum + layer.attentionBytes + layer.ffnBytes,
        0,
      );
      const experts = model.moe === undefined
        ? 0
        : spec.numLayers * (
          model.moe.numExperts * model.moe.expertBytesPerLayer
          + model.moe.sharedExpertBytesPerLayer
        );
      const bytes = dense + experts + (model.embeddingBytes ?? 0);
      expect(bytes / 2, preset).toBeCloseTo(model.totalParams, -3);
    }
  });

  it("scales weight bytes with the weight dtype and KV with the KV dtype", () => {
    const fp16 = buildModelProfile("qwen3-0.6b", "fp16", "fp16");
    const int4 = buildModelProfile("qwen3-0.6b", "int4", "fp16");
    const fp8Kv = buildModelProfile("qwen3-0.6b", "fp16", "fp8");

    expect(int4.layers[0]!.attentionBytes)
      .toBe(fp16.layers[0]!.attentionBytes / 4);
    expect(int4.embeddingBytes).toBe(fp16.embeddingBytes! / 4);
    expect(int4.totalParams).toBe(fp16.totalParams);
    expect(int4.layers[0]!.kvCachePerToken)
      .toBe(fp16.layers[0]!.kvCachePerToken);
    expect(fp8Kv.layers[0]!.kvCachePerToken)
      .toBe(fp16.layers[0]!.kvCachePerToken / 2);
  });

  it("models grouped-query KV geometry rather than full multi-head KV", () => {
    // Llama-3-8B is 32 query heads over 8 KV heads of width 128.
    const model = buildModelProfile("llama-3-8b", "fp16", "fp16");
    expect(model.layers[0]!.kvCachePerToken).toBe(2 * 8 * 128 * 2);
    expect(model.architecture.numHeads).toBe(32);
    expect(model.architecture.numKVHeads).toBe(8);
  });

  it("caches one compressed latent per token for MLA models", () => {
    // DeepSeek caches kv_lora_rank + qk_rope_head_dim, not two full KV heads.
    for (const preset of ["deepseek-v2", "deepseek-v3", "kimi-k2"]) {
      const model = buildModelProfile(preset, "fp16", "fp16");
      expect(model.layers[0]!.kvCachePerToken, preset).toBe((512 + 64) * 2);
    }
  });

  it("charges no KV to linear-attention layers in hybrid stacks", () => {
    const model = buildModelProfile("qwen3.6-27b", "fp16", "fp16");
    const withKv = model.layers.filter((layer) => layer.kvCachePerToken > 0);

    expect(model.layers).toHaveLength(64);
    expect(withKv).toHaveLength(16);
    expect(withKv.map((layer) => layer.index % 4)).toEqual(
      Array.from({ length: 16 }, () => 3),
    );
  });

  it("uses a wider global head on Gemma hybrid attention layers", () => {
    const model = buildModelProfile("gemma-4-12b", "fp16", "fp16");
    const local = model.layers[0]!;
    const global = model.layers[5]!;

    expect(local.kvCachePerToken).toBe(2 * 8 * 256 * 2);
    expect(global.kvCachePerToken).toBe(2 * 1 * 512 * 2);
  });

  it("keeps aggregate expert bytes exact when leading layers are dense", () => {
    const model = buildModelProfile("deepseek-v3", "fp16", "fp16");
    const spec = MODEL_SPECS["deepseek-v3"]!;
    const routedBytes = spec.numLayers * spec.moe!.numExperts
      * model.moe!.expertBytesPerLayer;
    const expected = spec.moe!.moeLayers * spec.moe!.numExperts
      * 3 * spec.hiddenDim * spec.moe!.expertIntermediateSize * 2;

    expect(routedBytes).toBe(expected);
    // Only the three dense layers carry a non-routed FFN.
    expect(model.layers.filter((layer) => layer.ffnBytes > 0)).toHaveLength(3);
  });

  it("rejects unknown presets with the available list", () => {
    expect(() => buildModelProfile("not-a-model"))
      .toThrow(/Unknown model preset: not-a-model/);
  });

  it("spans every parameter-count decade from sub-1B to over 1T", () => {
    const sizes = listModelPresets()
      .map((preset) => buildModelProfile(preset).totalParams)
      .sort((left, right) => left - right);

    expect(sizes[0]).toBeLessThan(1e9);
    expect(sizes.at(-1)).toBeGreaterThan(1e12);
    for (const decade of [1e9, 1e10, 1e11, 1e12]) {
      expect(
        sizes.some((size) => size >= decade && size < decade * 10),
        `decade ${decade}`,
      ).toBe(true);
    }
  });
});
