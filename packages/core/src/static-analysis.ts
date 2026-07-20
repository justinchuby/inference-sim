/**
 * Static analysis — Phase 1 of the simulator.
 * Given hardware + model + pipeline config, compute memory breakdown,
 * feasibility, throughput estimates, and bottleneck identification.
 *
 * No event loop — pure math, instant results.
 */
import type {
  HardwareTopology,
  ModelProfile,
  PipelineConfig,
  StaticAnalysisResult,
  DeviceMemoryBreakdown,
  HostMemoryBreakdown,
  ThroughputEstimate,
  DeviceSpec,
  QuantType,
} from "./types.js";

// ============================================================
// Helpers
// ============================================================

function bytesPerElement(quant: QuantType): number {
  switch (quant) {
    case "fp32": return 4;
    case "fp16": case "bf16": return 2;
    case "fp8": case "int8": return 1;
    case "int4": case "nf4": return 0.5;
  }
}

function sumLayerWeights(model: ModelProfile): number {
  return model.layers.reduce((sum, l) => sum + l.attentionBytes + l.ffnBytes, 0);
}

function totalExpertWeights(model: ModelProfile): number {
  if (!model.moe) return 0;
  return model.moe.numExperts * model.moe.expertSize + model.moe.sharedExpertSize * model.architecture.numLayers;
}

function kvCachePerToken(model: ModelProfile): number {
  return model.layers.reduce((sum, l) => sum + l.kvCachePerToken, 0);
}

function activationMemory(model: ModelProfile, batchSize: number, seqLen: number): number {
  // Rough: 2 * hidden * batch * seq * bytes_per_activation (for intermediate activations)
  const bpe = bytesPerElement(model.quantization.activations);
  // Peak activation ≈ batch * seq * intermediate * 2 (for gate/up in FFN or MoE)
  return batchSize * seqLen * model.architecture.intermediateSize * 2 * bpe;
}

// ============================================================
// Main Analysis
// ============================================================

export function analyzeStatic(
  topology: HardwareTopology,
  model: ModelProfile,
  pipeline: PipelineConfig,
): StaticAnalysisResult {
  const { parallelism, memory: memPolicy } = pipeline;
  const totalDevices = topology.nodes.reduce((sum, n) => sum + n.devices.length, 0);

  // Validate parallelism config
  const totalParallel = parallelism.tensorParallel * parallelism.pipelineParallel *
    parallelism.expertParallel * parallelism.dataParallel;
  if (totalParallel > totalDevices) {
    return {
      feasible: false,
      memoryBreakdown: [],
      hostMemoryBreakdown: { totalBytes: 0, offloadedWeights: 0, warmExperts: 0, kvOverflow: 0, free: 0 },
      bottleneck: "capacity",
      estimatedThroughput: { prefillToksPerSec: 0, decodeToksPerSec: 0, timeToFirstTokenMs: Infinity, interTokenLatencyMs: Infinity },
      recommendations: [`Need ${totalParallel} devices but topology only has ${totalDevices}`],
    };
  }

  // --- Weight distribution ---
  const denseWeights = sumLayerWeights(model);
  const layersPerPPStage = Math.ceil(model.architecture.numLayers / parallelism.pipelineParallel);

  // Dense weights per device (TP sharded, PP staged)
  const denseWeightsPerDevice = denseWeights / parallelism.tensorParallel / parallelism.pipelineParallel;

  // Expert weights per device (EP sharded)
  let expertWeightsPerDevice = 0;
  let totalExpertBytes = 0;
  let hotExpertsPerDevice = 0;
  if (model.moe) {
    totalExpertBytes = totalExpertWeights(model);
    const expertsPerEPRank = Math.ceil(model.moe.numExperts / parallelism.expertParallel);
    expertWeightsPerDevice = expertsPerEPRank * model.moe.expertSize;
    hotExpertsPerDevice = expertsPerEPRank;
  }

  // --- KV Cache ---
  const maxSeqLen = pipeline.inputSeqLen + pipeline.outputSeqLen;
  const kvPerToken = kvCachePerToken(model);
  const kvCacheTotal = kvPerToken * maxSeqLen * pipeline.batchSize;
  // KV cache is replicated per TP rank but split across PP stages
  const kvCachePerDevice = kvCacheTotal / parallelism.pipelineParallel;

  // --- Activations ---
  const activations = activationMemory(model, pipeline.batchSize, pipeline.inputSeqLen);
  const activationsPerDevice = activations / parallelism.tensorParallel;

  // --- Memory breakdown per device ---
  const firstDevice = topology.nodes[0]?.devices[0];
  if (!firstDevice) {
    return {
      feasible: false,
      memoryBreakdown: [],
      hostMemoryBreakdown: { totalBytes: 0, offloadedWeights: 0, warmExperts: 0, kvOverflow: 0, free: 0 },
      bottleneck: "capacity",
      estimatedThroughput: { prefillToksPerSec: 0, decodeToksPerSec: 0, timeToFirstTokenMs: Infinity, interTokenLatencyMs: Infinity },
      recommendations: ["No devices in topology"],
    };
  }

  const deviceCapacity = firstDevice.memory.capacityBytes;
  const totalUsedPerDevice = denseWeightsPerDevice + expertWeightsPerDevice + kvCachePerDevice + activationsPerDevice;

  // Build breakdown for all devices
  const memoryBreakdown: DeviceMemoryBreakdown[] = [];
  for (const node of topology.nodes) {
    for (const device of node.devices) {
      memoryBreakdown.push({
        deviceId: device.id,
        totalBytes: device.memory.capacityBytes,
        weights: denseWeightsPerDevice,
        kvCache: kvCachePerDevice,
        expertCache: expertWeightsPerDevice,
        activations: activationsPerDevice,
        free: device.memory.capacityBytes - totalUsedPerDevice,
      });
    }
  }
  const deviceFeasible = memoryBreakdown.every((breakdown) => breakdown.free >= 0);

  // --- Host memory ---
  // A legacy unified-memory node exposes the same physical capacity through
  // hostMemory and its unified device. It is not an extra offload tier.
  const hostCapacity = topology.nodes.reduce((sum, node) => (
    node.devices.some((device) => device.kind === "unified")
      ? sum
      : sum + node.hostMemory.capacityBytes
  ), 0);
  let offloadedWeights = 0;
  let warmExperts = 0;
  if (!deviceFeasible && memPolicy.offloadStrategy !== "none") {
    // This legacy analyzer treats the overage as an offload requirement. The
    // composable scenario validator performs the exact accessibility check.
    offloadedWeights = memoryBreakdown.reduce(
      (sum, breakdown) => sum + Math.max(0, -breakdown.free),
      0,
    );
  }
  if (model.moe && memPolicy.offloadStrategy === "partial") {
    // Warm experts that don't fit on device go to host
    const hotBudget = deviceCapacity * memPolicy.expertCacheBudgetFraction;
    const hotExperts = Math.floor(hotBudget / model.moe.expertSize);
    warmExperts = Math.max(0, hotExpertsPerDevice - hotExperts) * model.moe.expertSize;
    hotExpertsPerDevice = Math.min(hotExpertsPerDevice, hotExperts);
  }

  const hostMemoryBreakdown: HostMemoryBreakdown = {
    totalBytes: hostCapacity,
    offloadedWeights,
    warmExperts,
    kvOverflow: 0,
    free: hostCapacity - offloadedWeights - warmExperts,
  };
  const hasUnifiedOverflow = topology.nodes.some((node) => (
    node.devices.some((device) => device.kind === "unified")
    && node.devices.some((device) => {
      const breakdown = memoryBreakdown.find((entry) => entry.deviceId === device.id);
      return breakdown !== undefined && breakdown.free < 0;
    })
  ));
  const offloadFeasible = memPolicy.offloadStrategy !== "none"
    && !hasUnifiedOverflow
    && hostMemoryBreakdown.free >= 0;
  const feasible = deviceFeasible || offloadFeasible;

  // --- Throughput estimation (roofline model) ---
  const throughput = estimateThroughput(firstDevice, model, pipeline, denseWeightsPerDevice + expertWeightsPerDevice);

  // --- Bottleneck identification ---
  const bottleneck = feasible
    ? identifyBottleneck(firstDevice, model, pipeline, topology)
    : "capacity";

  // --- Recommendations ---
  const recommendations: string[] = [];
  if (!deviceFeasible) {
    const largestOverflow = Math.max(
      ...memoryBreakdown.map((breakdown) => Math.max(0, -breakdown.free)),
    );
    const overflowGiB = (largestOverflow / (1024 ** 3)).toFixed(1);
    recommendations.push(`Largest device capacity overage is ${overflowGiB} GiB`);
    if (model.quantization.weights === "fp16") {
      recommendations.push("Try FP8 quantization to halve weight memory");
    }
    if (parallelism.tensorParallel < topology.nodes[0].devices.length) {
      recommendations.push(`Increase TP from ${parallelism.tensorParallel} to ${topology.nodes[0].devices.length}`);
    }
  }
  if (model.moe && hotExpertsPerDevice < model.moe.numExperts / parallelism.expertParallel) {
    const hitRate = hotExpertsPerDevice / (model.moe.numExperts / parallelism.expertParallel);
    recommendations.push(`Expert cache can hold ${hotExpertsPerDevice}/${model.moe.numExperts / parallelism.expertParallel} experts (${(hitRate * 100).toFixed(0)}% theoretical max hit rate)`);
  }
  if (bottleneck === "memory_bandwidth" && model.quantization.kvCache === "fp16") {
    recommendations.push("FP8 KV cache could reduce memory bandwidth pressure during decode");
  }
  if (!deviceFeasible && memPolicy.offloadStrategy !== "none" && !offloadFeasible) {
    recommendations.push(
      hasUnifiedOverflow
        ? "Unified memory has no separate host tier to absorb the overage"
        : "Host memory cannot hold the modeled offload requirement",
    );
  }

  return { feasible, memoryBreakdown, hostMemoryBreakdown, bottleneck, estimatedThroughput: throughput, recommendations };
}

// ============================================================
// Throughput Estimation
// ============================================================

function estimateThroughput(
  device: DeviceSpec,
  model: ModelProfile,
  pipeline: PipelineConfig,
  weightsOnDevice: number,
): ThroughputEstimate {
  const { batchSize, inputSeqLen, outputSeqLen } = pipeline;

  // --- Prefill: compute-bound ---
  // FLOPs per token ≈ 2 * total_params (forward pass)
  // For MoE: only active experts contribute
  let activeParams = model.totalParams;
  if (model.moe) {
    const denseParams = model.layers.reduce((s, l) => s + l.attentionBytes, 0) / bytesPerElement(model.quantization.weights);
    const activeExpertParams = model.moe.activeExpertsPerToken * model.moe.expertSize / bytesPerElement(model.quantization.weights);
    activeParams = denseParams + activeExpertParams * model.architecture.numLayers;
  }
  const flopsPerToken = 2 * activeParams;
  const prefillFlops = flopsPerToken * inputSeqLen * batchSize;

  const computeFlops = model.quantization.weights === "fp8" ? device.compute.fp8Flops : device.compute.fp16Flops;
  const mfu = 0.45; // typical model FLOPS utilization
  const prefillTimeSec = prefillFlops / (computeFlops * mfu);
  const prefillToksPerSec = (inputSeqLen * batchSize) / prefillTimeSec;

  // --- Decode: memory-bandwidth-bound ---
  // Each token reads all weights once
  const bytesPerDecodeStep = weightsOnDevice; // read all weights for one token
  const mbu = 0.85; // memory bandwidth utilization
  const decodeStepTimeSec = bytesPerDecodeStep / (device.memory.bandwidthBytesPerSec * mbu);
  const decodeToksPerSec = batchSize / decodeStepTimeSec;

  const timeToFirstTokenMs = prefillTimeSec * 1000;
  const interTokenLatencyMs = decodeStepTimeSec * 1000;

  return { prefillToksPerSec, decodeToksPerSec, timeToFirstTokenMs, interTokenLatencyMs };
}

// ============================================================
// Bottleneck Identification
// ============================================================

function identifyBottleneck(
  device: DeviceSpec,
  model: ModelProfile,
  pipeline: PipelineConfig,
  topology: HardwareTopology,
): StaticAnalysisResult["bottleneck"] {
  // Arithmetic intensity during decode
  const activeParams = model.moe
    ? model.moe.activeExpertsPerToken * model.moe.expertSize + model.layers[0].attentionBytes
    : model.layers[0].attentionBytes + model.layers[0].ffnBytes;

  // FLOPs per byte = 2 * batch / bytes_per_element (for GEMV in decode)
  const bpe = bytesPerElement(model.quantization.weights);
  const arithmeticIntensity = 2 * pipeline.batchSize / bpe;

  // Roofline ridge point
  const computeFlops = model.quantization.weights === "fp8" ? device.compute.fp8Flops : device.compute.fp16Flops;
  const ridgePoint = computeFlops / device.memory.bandwidthBytesPerSec;

  if (arithmeticIntensity < ridgePoint) {
    return "memory_bandwidth"; // Below ridge = memory-bound
  }

  // Check if interconnect is the bottleneck (multi-device communication)
  if (topology.nodes[0]?.devices.length > 1 && pipeline.parallelism.tensorParallel > 1) {
    const interDevice = topology.nodes[0].interDeviceLinks[0];
    if (interDevice) {
      // AllReduce volume per step ≈ 2 * hidden * batch * bytes * (tp-1)/tp
      const hiddenBytes = model.architecture.hiddenDim * bpe * pipeline.batchSize;
      const allReduceBytes = 2 * hiddenBytes * (pipeline.parallelism.tensorParallel - 1) / pipeline.parallelism.tensorParallel;
      const commTime = allReduceBytes / interDevice.bandwidthBytesPerSec;
      const computeTime = activeParams * 2 / computeFlops;
      if (commTime > computeTime * 0.5) {
        return "interconnect";
      }
    }
  }

  return "compute";
}
