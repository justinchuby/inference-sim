/**
 * Hardware presets — built-in specs for common GPUs and topologies.
 */
import type { DeviceSpec, HardwareTopology, NodeSpec, InterconnectSpec } from "./types.js";

// ============================================================
// GPU Presets
// ============================================================

const GiB = 1024 ** 3;
const TiB = 1024 ** 4;
const TB_s = 1e12; // bytes/sec
const TFLOPS = 1e12;
const GB_s = 1e9;
const Gbps = 1e9 / 8; // bits/sec → bytes/sec

export const GPU_PRESETS: Record<string, Omit<DeviceSpec, "id">> = {
  "h100-sxm": {
    kind: "gpu",
    memory: { capacityBytes: 80 * GiB, bandwidthBytesPerSec: 3.35 * TB_s, latencyNs: 100 },
    compute: { fp16Flops: 990 * TFLOPS, fp8Flops: 1979 * TFLOPS, int8Flops: 1979 * TFLOPS },
  },
  "h200-sxm": {
    kind: "gpu",
    memory: { capacityBytes: 141 * GiB, bandwidthBytesPerSec: 4.8 * TB_s, latencyNs: 100 },
    compute: { fp16Flops: 990 * TFLOPS, fp8Flops: 1979 * TFLOPS, int8Flops: 1979 * TFLOPS },
  },
  "a100-80g": {
    kind: "gpu",
    memory: { capacityBytes: 80 * GiB, bandwidthBytesPerSec: 2.0 * TB_s, latencyNs: 100 },
    compute: { fp16Flops: 312 * TFLOPS, fp8Flops: 624 * TFLOPS, int8Flops: 624 * TFLOPS },
  },
  "a100-40g": {
    kind: "gpu",
    memory: { capacityBytes: 40 * GiB, bandwidthBytesPerSec: 1.6 * TB_s, latencyNs: 100 },
    compute: { fp16Flops: 312 * TFLOPS, fp8Flops: 624 * TFLOPS, int8Flops: 624 * TFLOPS },
  },
  "l40s": {
    kind: "gpu",
    memory: { capacityBytes: 48 * GiB, bandwidthBytesPerSec: 864 * GB_s, latencyNs: 150 },
    compute: { fp16Flops: 366 * TFLOPS, fp8Flops: 733 * TFLOPS, int8Flops: 733 * TFLOPS },
  },
  "rtx-4090": {
    kind: "gpu",
    memory: { capacityBytes: 24 * GiB, bandwidthBytesPerSec: 1.0 * TB_s, latencyNs: 120 },
    compute: { fp16Flops: 165 * TFLOPS, fp8Flops: 330 * TFLOPS, int8Flops: 330 * TFLOPS },
  },
  "rtx-5090": {
    kind: "gpu",
    memory: { capacityBytes: 32 * GiB, bandwidthBytesPerSec: 1.79 * TB_s, latencyNs: 100 },
    compute: { fp16Flops: 420 * TFLOPS, fp8Flops: 838 * TFLOPS, int8Flops: 838 * TFLOPS },
  },
  "b200": {
    kind: "gpu",
    memory: { capacityBytes: 192 * GiB, bandwidthBytesPerSec: 8.0 * TB_s, latencyNs: 80 },
    compute: { fp16Flops: 2250 * TFLOPS, fp8Flops: 4500 * TFLOPS, int8Flops: 4500 * TFLOPS },
  },
  // Apple Silicon (unified memory)
  "m1-max": {
    kind: "unified",
    memory: { capacityBytes: 64 * GiB, bandwidthBytesPerSec: 400 * GB_s, latencyNs: 80 },
    compute: { fp16Flops: 10.4 * TFLOPS, fp8Flops: 10.4 * TFLOPS, int8Flops: 10.4 * TFLOPS },
  },
  "m2-ultra": {
    kind: "unified",
    memory: { capacityBytes: 192 * GiB, bandwidthBytesPerSec: 800 * GB_s, latencyNs: 80 },
    compute: { fp16Flops: 27.2 * TFLOPS, fp8Flops: 27.2 * TFLOPS, int8Flops: 27.2 * TFLOPS },
  },
  "m4-max": {
    kind: "unified",
    memory: { capacityBytes: 128 * GiB, bandwidthBytesPerSec: 546 * GB_s, latencyNs: 70 },
    compute: { fp16Flops: 38 * TFLOPS, fp8Flops: 38 * TFLOPS, int8Flops: 38 * TFLOPS },
  },
  "m4-ultra": {
    kind: "unified",
    memory: { capacityBytes: 512 * GiB, bandwidthBytesPerSec: 1.09 * TB_s, latencyNs: 70 },
    compute: { fp16Flops: 76 * TFLOPS, fp8Flops: 76 * TFLOPS, int8Flops: 76 * TFLOPS },
  },
};

// ============================================================
// Topology Presets
// ============================================================

function makeGpuNode(
  nodeId: string,
  gpuPreset: string,
  numGpus: number,
  hostRamBytes: number,
  interDeviceKind: InterconnectSpec["kind"],
  interDeviceBw: number,
): NodeSpec {
  const gpuSpec = GPU_PRESETS[gpuPreset];
  if (!gpuSpec) throw new Error(`Unknown GPU preset: ${gpuPreset}`);

  const devices: DeviceSpec[] = Array.from(
    { length: numGpus },
    (_, i) => instantiateDevice(`${nodeId}:gpu${i}`, gpuSpec),
  );

  // Full-mesh interconnect between devices
  const interDeviceLinks: InterconnectSpec[] = [];
  for (let i = 0; i < numGpus; i++) {
    for (let j = i + 1; j < numGpus; j++) {
      interDeviceLinks.push({
        endpoints: [devices[i].id, devices[j].id],
        bandwidthBytesPerSec: interDeviceBw,
        latencyNs: interDeviceKind === "nvlink" ? 500 : 1500,
        kind: interDeviceKind,
      });
    }
  }

  return {
    id: nodeId,
    devices,
    hostMemory: { capacityBytes: hostRamBytes, bandwidthBytesPerSec: 50 * GB_s, latencyNs: 80 },
    interDeviceLinks,
  };
}

export function buildTopology(preset: string): HardwareTopology {
  switch (preset) {
    case "dgx-h100":
    case "h100-8x":
      return {
        nodes: [makeGpuNode("node0", "h100-sxm", 8, 2 * TiB, "nvlink", 900 * GB_s)],
        interNodeLinks: [],
      };

    case "dgx-h200":
    case "h200-8x":
      return {
        nodes: [makeGpuNode("node0", "h200-sxm", 8, 2 * TiB, "nvlink", 900 * GB_s)],
        interNodeLinks: [],
      };

    case "2x-dgx-h100":
      return {
        nodes: [
          makeGpuNode("node0", "h100-sxm", 8, 2 * TiB, "nvlink", 900 * GB_s),
          makeGpuNode("node1", "h100-sxm", 8, 2 * TiB, "nvlink", 900 * GB_s),
        ],
        interNodeLinks: [{
          endpoints: ["node0", "node1"],
          bandwidthBytesPerSec: 400 * Gbps,
          latencyNs: 1000,
          kind: "infiniband",
        }],
      };

    case "4x-mac-studio-m4":
      return {
        nodes: Array.from({ length: 4 }, (_, i) => ({
          id: `node${i}`,
          devices: [
            instantiateDevice(`node${i}:soc`, GPU_PRESETS["m4-max"]),
          ],
          hostMemory: { capacityBytes: 128 * GiB, bandwidthBytesPerSec: 546 * GB_s, latencyNs: 70 },
          interDeviceLinks: [], // single SoC, no inter-device
        })),
        interNodeLinks: Array.from({ length: 3 }, (_, i) => ({
          endpoints: [`node${i}`, `node${i + 1}`] as [string, string],
          bandwidthBytesPerSec: 40 * Gbps,
          latencyNs: 2000,
          kind: "thunderbolt" as const,
        })),
      };

    case "a100-4x":
      return {
        nodes: [makeGpuNode("node0", "a100-80g", 4, 1 * TiB, "nvlink", 600 * GB_s)],
        interNodeLinks: [],
      };

    case "rtx-4090-2x":
      return {
        nodes: [makeGpuNode("node0", "rtx-4090", 2, 128 * GiB, "pcie", 32 * GB_s)],
        interNodeLinks: [],
      };

    default:
      throw new Error(`Unknown topology preset: ${preset}. Available: dgx-h100, dgx-h200, 2x-dgx-h100, 4x-mac-studio-m4, a100-4x, rtx-4090-2x`);
  }
}

function instantiateDevice(
  id: string,
  spec: Omit<DeviceSpec, "id">,
): DeviceSpec {
  return {
    id,
    kind: spec.kind,
    memory: { ...spec.memory },
    compute: { ...spec.compute },
  };
}

export function listPresets(): { gpus: string[]; topologies: string[] } {
  return {
    gpus: Object.keys(GPU_PRESETS),
    topologies: ["dgx-h100", "h100-8x", "dgx-h200", "h200-8x", "2x-dgx-h100", "4x-mac-studio-m4", "a100-4x", "rtx-4090-2x"],
  };
}
