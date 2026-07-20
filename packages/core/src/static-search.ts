import { canonicalJsonFingerprint } from "./result-artifact.js";
import { analyzeStatic } from "./static-analysis.js";
import type {
  HardwareTopology,
  MemoryPolicyConfig,
  ModelProfile,
  ParallelismConfig,
  QuantType,
  StaticAnalysisResult,
} from "./types.js";

export type StaticSearchObjective =
  | "decode_throughput"
  | "prefill_throughput"
  | "time_to_first_token"
  | "inter_token_latency"
  | "device_headroom";

export interface StaticSearchTopology {
  readonly id: string;
  readonly topology: HardwareTopology;
}

export interface StaticSearchSpace {
  readonly topologies: readonly StaticSearchTopology[];
  readonly kvCacheQuantizations: readonly QuantType[];
  readonly activationQuantizations: readonly QuantType[];
  readonly batchSizes: readonly number[];
  readonly inputSeqLens: readonly number[];
  readonly outputSeqLens: readonly number[];
  readonly tensorParallel: readonly number[];
  readonly pipelineParallel: readonly number[];
  readonly expertParallel: readonly number[];
  readonly dataParallel: readonly number[];
  readonly memoryPolicies: readonly MemoryPolicyConfig[];
}

export interface StaticSearchConstraints {
  readonly requireFeasible: boolean;
  readonly maximumDeviceUsedFraction?: number;
  readonly maximumTimeToFirstTokenMs?: number;
  readonly maximumInterTokenLatencyMs?: number;
}

export interface StaticSearchRequest {
  readonly objective: StaticSearchObjective;
  readonly topK: number;
  readonly maxCandidates: number;
  readonly constraints: StaticSearchConstraints;
  readonly space: StaticSearchSpace;
}

export interface StaticSearchCandidate {
  readonly rank: number;
  readonly candidateId: string;
  readonly topologyId: string;
  readonly kvCacheQuantization: QuantType;
  readonly activationQuantization: QuantType;
  readonly batchSize: number;
  readonly inputSeqLen: number;
  readonly outputSeqLen: number;
  readonly parallelism: ParallelismConfig;
  readonly memory: MemoryPolicyConfig;
  readonly feasible: boolean;
  readonly score: number;
  readonly deviceUsedFraction: number;
  readonly analysis: StaticAnalysisResult;
}

export interface StaticSearchResult {
  readonly objective: StaticSearchObjective;
  readonly exhaustive: true;
  readonly declaredCandidateCount: number;
  readonly evaluatedCandidateCount: number;
  readonly eligibleCandidateCount: number;
  readonly returnedCandidateCount: number;
  readonly rejectionCounts: Readonly<Record<string, number>>;
  readonly candidates: readonly StaticSearchCandidate[];
  readonly evidence: {
    readonly confidence: ModelProfile["provenance"]["evidence"];
    readonly source: string;
    readonly assumptions: readonly string[];
  };
}

export interface StaticSearchProgress {
  readonly completedCandidates: number;
  readonly totalCandidates: number;
}

interface CandidateConfig {
  readonly topology: StaticSearchTopology;
  readonly kvCacheQuantization: QuantType;
  readonly activationQuantization: QuantType;
  readonly batchSize: number;
  readonly inputSeqLen: number;
  readonly outputSeqLen: number;
  readonly parallelism: ParallelismConfig;
  readonly memory: MemoryPolicyConfig;
}

export function searchStaticConfigurations(
  model: ModelProfile,
  request: StaticSearchRequest,
  onProgress?: (progress: StaticSearchProgress) => void,
): StaticSearchResult {
  validateSearchRequest(request);
  const axes = request.space;
  const declaredCandidateCount = checkedProduct([
    axes.topologies.length,
    axes.kvCacheQuantizations.length,
    axes.activationQuantizations.length,
    axes.batchSizes.length,
    axes.inputSeqLens.length,
    axes.outputSeqLens.length,
    axes.tensorParallel.length,
    axes.pipelineParallel.length,
    axes.expertParallel.length,
    axes.dataParallel.length,
    axes.memoryPolicies.length,
  ]);
  if (declaredCandidateCount > request.maxCandidates) {
    throw new Error(
      `static search declares ${declaredCandidateCount} candidates, exceeding maxCandidates ${request.maxCandidates}`,
    );
  }
  onProgress?.({
    completedCandidates: 0,
    totalCandidates: declaredCandidateCount,
  });

  const rejectionCounts = new Map<string, number>();
  const accepted: Omit<StaticSearchCandidate, "rank">[] = [];
  let evaluatedCandidateCount = 0;
  let completedCandidateCount = 0;
  const progressInterval = Math.max(
    1,
    Math.floor(declaredCandidateCount / 20),
  );
  const reportCandidateProgress = () => {
    completedCandidateCount++;
    if (
      completedCandidateCount === declaredCandidateCount
      || completedCandidateCount % progressInterval === 0
    ) {
      onProgress?.({
        completedCandidates: completedCandidateCount,
        totalCandidates: declaredCandidateCount,
      });
    }
  };
  for (const candidate of enumerateCandidates(axes)) {
    const invalidReason = validateCandidate(model, candidate);
    if (invalidReason !== undefined) {
      increment(rejectionCounts, invalidReason);
      reportCandidateProgress();
      continue;
    }
    evaluatedCandidateCount++;
    const candidateModel = withRuntimeQuantization(
      model,
      candidate.kvCacheQuantization,
      candidate.activationQuantization,
    );
    const analysis = analyzeStatic(candidate.topology.topology, candidateModel, {
      batchSize: candidate.batchSize,
      inputSeqLen: candidate.inputSeqLen,
      outputSeqLen: candidate.outputSeqLen,
      parallelism: candidate.parallelism,
      memory: candidate.memory,
    });
    const deviceUsedFraction = maximumDeviceUsedFraction(analysis);
    const constraintReason = rejectedByConstraint(
      analysis,
      deviceUsedFraction,
      request.constraints,
    );
    if (constraintReason !== undefined) {
      increment(rejectionCounts, constraintReason);
      reportCandidateProgress();
      continue;
    }
    const identity = candidateIdentity(candidate);
    accepted.push({
      candidateId: canonicalJsonFingerprint(identity),
      topologyId: candidate.topology.id,
      kvCacheQuantization: candidate.kvCacheQuantization,
      activationQuantization: candidate.activationQuantization,
      batchSize: candidate.batchSize,
      inputSeqLen: candidate.inputSeqLen,
      outputSeqLen: candidate.outputSeqLen,
      parallelism: candidate.parallelism,
      memory: candidate.memory,
      feasible: analysis.feasible,
      score: objectiveScore(
        request.objective,
        analysis,
        deviceUsedFraction,
      ),
      deviceUsedFraction,
      analysis,
    });
    reportCandidateProgress();
  }
  accepted.sort((left, right) => (
    Number(right.feasible) - Number(left.feasible)
    || right.score - left.score
    || left.candidateId.localeCompare(right.candidateId)
  ));
  const candidates = accepted.slice(0, request.topK).map(
    (candidate, index): StaticSearchCandidate => ({
      rank: index + 1,
      ...candidate,
    }),
  );
  return {
    objective: request.objective,
    exhaustive: true,
    declaredCandidateCount,
    evaluatedCandidateCount,
    eligibleCandidateCount: accepted.length,
    returnedCandidateCount: candidates.length,
    rejectionCounts: Object.fromEntries(
      [...rejectionCounts.entries()].sort(([left], [right]) => (
        left.localeCompare(right)
      )),
    ),
    candidates,
    evidence: {
      confidence: model.provenance.evidence,
      source: model.provenance.source,
      assumptions: [
        ...model.provenance.assumptions,
        "Ranking exhaustively evaluates only the explicitly declared finite candidate space.",
        "Static throughput uses the analyzer's uncalibrated roofline assumptions.",
      ],
    },
  };
}

function* enumerateCandidates(
  space: StaticSearchSpace,
): Generator<CandidateConfig> {
  for (const topology of space.topologies) {
    for (const kvCacheQuantization of space.kvCacheQuantizations) {
      for (const activationQuantization of space.activationQuantizations) {
        for (const batchSize of space.batchSizes) {
          for (const inputSeqLen of space.inputSeqLens) {
            for (const outputSeqLen of space.outputSeqLens) {
              for (const tensorParallel of space.tensorParallel) {
                for (const pipelineParallel of space.pipelineParallel) {
                  for (const expertParallel of space.expertParallel) {
                    for (const dataParallel of space.dataParallel) {
                      for (const memory of space.memoryPolicies) {
                        yield {
                          topology,
                          kvCacheQuantization,
                          activationQuantization,
                          batchSize,
                          inputSeqLen,
                          outputSeqLen,
                          parallelism: {
                            tensorParallel,
                            pipelineParallel,
                            expertParallel,
                            dataParallel,
                          },
                          memory,
                        };
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

function validateSearchRequest(request: StaticSearchRequest): void {
  if (!Number.isSafeInteger(request.topK) || request.topK <= 0) {
    throw new Error("static search topK must be a positive safe integer");
  }
  if (!Number.isSafeInteger(request.maxCandidates) || request.maxCandidates <= 0) {
    throw new Error("static search maxCandidates must be a positive safe integer");
  }
  for (const [name, values] of Object.entries(request.space)) {
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`static search axis ${name} must not be empty`);
    }
  }
  assertUnique(request.space.topologies.map((entry) => entry.id), "topology IDs");
  for (const topology of request.space.topologies) {
    if (
      topology.topology.nodes.reduce(
        (sum, node) => sum + node.devices.length,
        0,
      ) === 0
    ) {
      throw new Error(`static search topology ${topology.id} has no devices`);
    }
  }
  for (const [name, values] of Object.entries({
    kvCacheQuantizations: request.space.kvCacheQuantizations,
    activationQuantizations: request.space.activationQuantizations,
    batchSizes: request.space.batchSizes,
    inputSeqLens: request.space.inputSeqLens,
    outputSeqLens: request.space.outputSeqLens,
    tensorParallel: request.space.tensorParallel,
    pipelineParallel: request.space.pipelineParallel,
    expertParallel: request.space.expertParallel,
    dataParallel: request.space.dataParallel,
  })) {
    assertUnique(values.map(String), `${name} values`);
  }
  assertUnique(
    request.space.memoryPolicies.map((policy) => (
      canonicalJsonFingerprint(policy)
    )),
    "memory policies",
  );
  for (const [name, value] of Object.entries({
    maximumDeviceUsedFraction:
      request.constraints.maximumDeviceUsedFraction,
    maximumTimeToFirstTokenMs:
      request.constraints.maximumTimeToFirstTokenMs,
    maximumInterTokenLatencyMs:
      request.constraints.maximumInterTokenLatencyMs,
  })) {
    if (
      value !== undefined
      && (!Number.isFinite(value) || value < 0)
    ) {
      throw new Error(`static search constraint ${name} must be finite and non-negative`);
    }
  }
  if (
    request.constraints.maximumDeviceUsedFraction !== undefined
    && request.constraints.maximumDeviceUsedFraction > 1
  ) {
    throw new Error(
      "static search constraint maximumDeviceUsedFraction must not exceed 1",
    );
  }
}

function validateCandidate(
  model: ModelProfile,
  candidate: CandidateConfig,
): string | undefined {
  for (const [name, value] of Object.entries({
    batchSize: candidate.batchSize,
    inputSeqLen: candidate.inputSeqLen,
    outputSeqLen: candidate.outputSeqLen,
    ...candidate.parallelism,
  })) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      return `invalid_${name}`;
    }
  }
  if (candidate.parallelism.pipelineParallel > model.architecture.numLayers) {
    return "pipeline_parallel_exceeds_layers";
  }
  if (model.moe === undefined && candidate.parallelism.expertParallel !== 1) {
    return "expert_parallel_requires_moe";
  }
  if (
    model.moe !== undefined
    && candidate.parallelism.expertParallel > model.moe.numExperts
  ) {
    return "expert_parallel_exceeds_experts";
  }
  const devices = candidate.topology.topology.nodes.reduce(
    (sum, node) => sum + node.devices.length,
    0,
  );
  const requiredDevices = Object.values(candidate.parallelism)
    .reduce((product, value) => product * value, 1);
  if (!Number.isSafeInteger(requiredDevices) || requiredDevices > devices) {
    return "parallelism_exceeds_devices";
  }
  return undefined;
}

function rejectedByConstraint(
  analysis: StaticAnalysisResult,
  deviceUsedFraction: number,
  constraints: StaticSearchConstraints,
): string | undefined {
  if (constraints.requireFeasible && !analysis.feasible) {
    return "infeasible";
  }
  if (
    constraints.maximumDeviceUsedFraction !== undefined
    && deviceUsedFraction > constraints.maximumDeviceUsedFraction
  ) {
    return "device_used_fraction";
  }
  if (
    constraints.maximumTimeToFirstTokenMs !== undefined
    && analysis.estimatedThroughput.timeToFirstTokenMs
      > constraints.maximumTimeToFirstTokenMs
  ) {
    return "time_to_first_token";
  }
  if (
    constraints.maximumInterTokenLatencyMs !== undefined
    && analysis.estimatedThroughput.interTokenLatencyMs
      > constraints.maximumInterTokenLatencyMs
  ) {
    return "inter_token_latency";
  }
  return undefined;
}

function objectiveScore(
  objective: StaticSearchObjective,
  analysis: StaticAnalysisResult,
  deviceUsedFraction: number,
): number {
  switch (objective) {
    case "decode_throughput":
      return analysis.estimatedThroughput.decodeToksPerSec;
    case "prefill_throughput":
      return analysis.estimatedThroughput.prefillToksPerSec;
    case "time_to_first_token":
      return -analysis.estimatedThroughput.timeToFirstTokenMs;
    case "inter_token_latency":
      return -analysis.estimatedThroughput.interTokenLatencyMs;
    case "device_headroom":
      return -deviceUsedFraction;
  }
}

function maximumDeviceUsedFraction(analysis: StaticAnalysisResult): number {
  return Math.max(...analysis.memoryBreakdown.map((memory) => (
    (memory.totalBytes - memory.free) / memory.totalBytes
  )));
}

function withRuntimeQuantization(
  model: ModelProfile,
  kvCache: QuantType,
  activations: QuantType,
): ModelProfile {
  const scale = bytesPerElement(kvCache)
    / bytesPerElement(model.quantization.kvCache);
  return {
    ...model,
    quantization: { ...model.quantization, kvCache, activations },
    layers: model.layers.map((layer) => ({
      ...layer,
      kvCachePerToken: layer.kvCachePerToken * scale,
    })),
  };
}

function candidateIdentity(candidate: CandidateConfig): unknown {
  return {
    topologyId: candidate.topology.id,
    kvCacheQuantization: candidate.kvCacheQuantization,
    activationQuantization: candidate.activationQuantization,
    batchSize: candidate.batchSize,
    inputSeqLen: candidate.inputSeqLen,
    outputSeqLen: candidate.outputSeqLen,
    parallelism: candidate.parallelism,
    memory: candidate.memory,
  };
}

function bytesPerElement(quant: QuantType): number {
  switch (quant) {
    case "fp32": return 4;
    case "fp16":
    case "bf16": return 2;
    case "fp8":
    case "int8": return 1;
    case "int4":
    case "nf4": return 0.5;
  }
}

function checkedProduct(values: readonly number[]): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new Error("static search candidate count exceeds safe integer range");
    }
  }
  return product;
}

function increment(counts: Map<string, number>, key: string): void {
  counts.set(key, (counts.get(key) ?? 0) + 1);
}

function assertUnique(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`static search ${label} must be unique`);
  }
}
