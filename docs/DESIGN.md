# Inference Simulator — Implementation Guide

A discrete-event simulator for LLM inference resource allocation and scheduling.
Models hardware topologies, ONNX model profiles, and memory management policies
to predict throughput, latency, memory flow, and bottlenecks — **without real hardware**.

## Repository Structure

```
inference-sim/
├── packages/
│   ├── core/              # Pure computation, zero deps, runs in Node + Browser
│   │   ├── src/
│   │   │   ├── index.ts           # Re-exports all public API
│   │   │   ├── types.ts           # All type definitions (DONE)
│   │   │   ├── presets.ts         # Hardware + topology presets (DONE)
│   │   │   ├── models.ts          # Model profile presets (DONE)
│   │   │   ├── static-analysis.ts # Phase 1: static memory analyzer (DONE)
│   │   │   ├── roofline.ts        # Performance model utilities
│   │   │   ├── expert-cache.ts    # Expert cache simulation (Phase 2)
│   │   │   ├── event-loop.ts      # Discrete-event engine (Phase 2)
│   │   │   ├── governor.ts        # Governor simulation (Phase 3)
│   │   │   └── trace.ts           # Event recording for visualization
│   │   └── tests/
│   │       └── static-analysis.test.ts  # (DONE, 4 passing)
│   │
│   ├── cli/               # Node.js command-line interface
│   │   ├── src/
│   │   │   ├── main.ts           # CLI entry point
│   │   │   ├── commands/
│   │   │   │   ├── analyze.ts    # `sim analyze` — static analysis
│   │   │   │   ├── simulate.ts   # `sim run` — full simulation
│   │   │   │   ├── compare.ts    # `sim compare` — multi-config comparison
│   │   │   │   └── list.ts       # `sim list` — show available presets
│   │   │   ├── config-loader.ts  # YAML config parsing
│   │   │   └── report.ts         # Terminal output formatting
│   │   └── package.json
│   │
│   └── web/               # React visualization dashboard
│       ├── src/
│       │   ├── App.tsx
│       │   ├── components/
│       │   │   ├── TopologyView.tsx
│       │   │   ├── MemoryTimeline.tsx
│       │   │   ├── ExpertHeatmap.tsx
│       │   │   ├── RooflinePlot.tsx
│       │   │   ├── BandwidthSankey.tsx
│       │   │   ├── ConfigPanel.tsx
│       │   │   └── MetricsPanel.tsx
│       │   ├── hooks/
│       │   │   └── useSimulation.ts
│       │   └── workers/
│       │       └── sim.worker.ts
│       └── package.json
│
├── examples/              # Example YAML configs
├── docs/
│   └── DESIGN.md          # This file
├── package.json           # Root workspace config
├── pnpm-workspace.yaml
└── tsconfig.base.json
```

## Current State (Phase 1 DONE)

The following is already implemented and tested:

- `packages/core/src/types.ts` — Full type system
- `packages/core/src/presets.ts` — GPU presets (H100, H200, A100, L40S, RTX 4090/5090, B200, M1-M4) + topology presets
- `packages/core/src/models.ts` — Model presets (Llama-3 8B/70B, Mixtral-8x22B, DeepSeek-V2, Qwen3-235B)
- `packages/core/src/static-analysis.ts` — Static memory breakdown + roofline throughput + bottleneck ID
- 4 passing vitest tests

---

## Phase 2: Expert Cache Simulation + Event Loop

### Goal

Token-by-token simulation that models expert cache dynamics for MoE models.
Captures warm-up, steady-state hit rates, eviction patterns, and prefetch effectiveness.

### File: `packages/core/src/expert-cache.ts`

```typescript
/**
 * Expert cache simulation.
 *
 * Models a tiered cache: hot (device VRAM) → warm (host RAM) → cold (not loaded).
 * Tracks per-expert access frequency, last-access time, and load/evict events.
 */

export interface ExpertCacheConfig {
  /** Max experts that fit in hot tier (device VRAM) */
  hotCapacity: number;
  /** Max experts in warm tier (host RAM). Infinity = unlimited. */
  warmCapacity: number;
  /** Eviction policy */
  policy: "lru" | "lfu" | "arc";
  /** Number of experts to prefetch ahead based on predicted routing */
  prefetchAhead: number;
}

export interface ExpertCacheState {
  hotTier: Map<number, ExpertEntry>;  // expertId → entry
  warmTier: Map<number, ExpertEntry>;
  stats: ExpertCacheStats;
}

export interface ExpertEntry {
  expertId: number;
  lastAccess: number;   // token index
  accessCount: number;
  sizeBytes: number;
}

export interface ExpertCacheStats {
  hits: number;
  misses: number;
  warmHits: number;    // miss in hot, hit in warm (cheaper load)
  coldMisses: number;  // miss in both hot and warm
  evictions: number;
  prefetchHits: number; // would-be misses caught by prefetch
}

export interface ExpertRouteResult {
  /** Experts already in hot tier — zero load latency */
  hotHits: number[];
  /** Experts loaded from warm tier */
  warmLoads: number[];
  /** Experts loaded from cold (disk/network) */
  coldLoads: number[];
  /** Total load latency for this token (max of all loads, since parallel) */
  loadLatencyNs: number;
  /** Experts evicted to make room */
  evicted: number[];
}

/**
 * Create initial cache state.
 */
export function createExpertCache(config: ExpertCacheConfig): ExpertCacheState;

/**
 * Simulate routing for one token's expert selection.
 *
 * @param state - Current cache state (mutated in place)
 * @param selectedExperts - Expert IDs selected by the router for this token
 * @param tokenIdx - Current token index (for LRU tracking)
 * @param interconnect - Bandwidth for load latency calculation
 * @returns Route result with hit/miss breakdown and load latency
 */
export function routeToken(
  state: ExpertCacheState,
  config: ExpertCacheConfig,
  selectedExperts: number[],
  tokenIdx: number,
  expertSizeBytes: number,
  hostToDeviceBandwidth: number,  // bytes/sec
  diskToHostBandwidth: number,    // bytes/sec (for cold loads)
): ExpertRouteResult;

/**
 * Advance prefetch queue based on predicted next-token routing.
 * Uses frequency-based prediction: most-accessed experts not currently in hot tier.
 */
export function advancePrefetch(
  state: ExpertCacheState,
  config: ExpertCacheConfig,
  expertSizeBytes: number,
): number[];  // expert IDs being prefetched
```

### File: `packages/core/src/expert-distribution.ts`

```typescript
/**
 * Expert activation distribution sampling.
 *
 * Given a distribution type, generates which experts are activated per token.
 * This drives the cache simulation.
 */

import type { ExpertDistribution } from "./types.js";

/**
 * Sample which experts are activated for one token.
 *
 * @param dist - Distribution configuration
 * @param numExperts - Total number of experts
 * @param activePerToken - How many experts are selected per token
 * @param rng - Random number generator (seeded for reproducibility)
 * @returns Array of selected expert IDs (length = activePerToken)
 */
export function sampleExperts(
  dist: ExpertDistribution,
  numExperts: number,
  activePerToken: number,
  rng: () => number,
): number[];

/**
 * Create a seeded PRNG for reproducible simulations.
 * Uses xoshiro128** algorithm.
 */
export function createRng(seed: number): () => number;
```

### File: `packages/core/src/event-loop.ts`

```typescript
/**
 * Discrete-event simulation engine.
 *
 * Simulates token-by-token inference, recording all events for later
 * visualization and analysis.
 *
 * Architecture:
 * - Processes tokens sequentially (prefill phase, then decode phase)
 * - For each token: route experts → check cache → load/evict → compute → record
 * - Produces a SimTrace with all events and a summary
 */

import type {
  HardwareTopology,
  ModelProfile,
  PipelineConfig,
  SimEvent,
  SimTrace,
  SimSummary,
} from "./types.js";
import type { ExpertCacheConfig, ExpertCacheState } from "./expert-cache.js";

export interface SimulationConfig {
  /** Total tokens to simulate (prefill + decode) */
  numTokens: number;
  /** Tokens to skip before collecting stats (cache warm-up) */
  warmupTokens: number;
  /** Random seed for reproducibility */
  seed: number;
  /** Record full event trace (large!) or just summary */
  fullTrace: boolean;
  /** Snapshot memory state every N tokens */
  snapshotInterval: number;
}

export interface SimulationInput {
  topology: HardwareTopology;
  model: ModelProfile;
  pipeline: PipelineConfig;
  simulation: SimulationConfig;
}

/**
 * Run the full simulation.
 *
 * Steps per decode token:
 * 1. For each MoE layer:
 *    a. Sample expert routing (from distribution)
 *    b. Check expert cache (hit/miss)
 *    c. Load missing experts (calculate latency)
 *    d. Evict if needed
 *    e. Compute attention (roofline model)
 *    f. Compute expert FFN (roofline model)
 *    g. AllReduce if TP > 1 (communication model)
 *    h. AllToAll if EP > 1 (communication model)
 * 2. Record events
 * 3. Advance to next token
 *
 * @returns SimTrace with all recorded events and computed summary
 */
export function runSimulation(input: SimulationInput): SimTrace;

/**
 * Lightweight version that only computes summary stats (no full trace).
 * Much faster for parameter sweeps.
 */
export function runSimulationSummary(input: SimulationInput): SimSummary;
```

### File: `packages/core/src/roofline.ts`

```typescript
/**
 * Roofline performance model.
 *
 * For any operation, performance is limited by either:
 * - Compute throughput (FLOPS)
 * - Memory bandwidth (bytes/sec)
 *
 * The "ridge point" is where compute = bandwidth. Below it = memory-bound.
 *
 * This module calculates per-operation latency using the roofline model.
 */

import type { DeviceSpec, QuantType } from "./types.js";

export interface OpProfile {
  name: string;
  flops: number;
  bytesRead: number;
  bytesWritten: number;
}

export interface EfficiencyModel {
  /** Model FLOPS Utilization (0.0-1.0). Typical: 0.3-0.6 for prefill, lower for decode */
  computeMFU: number;
  /** Memory Bandwidth Utilization (0.0-1.0). Typical: 0.7-0.9 */
  memoryMBU: number;
  /** Communication efficiency (0.0-1.0). Typical: 0.8-0.95 */
  commEfficiency: number;
}

/**
 * Default efficiency model based on operation phase.
 */
export function defaultEfficiency(phase: "prefill" | "decode"): EfficiencyModel;

/**
 * Calculate operation latency using roofline model.
 *
 * latency = max(compute_time, memory_time)
 * compute_time = flops / (device.peakFlops * mfu)
 * memory_time = (bytesRead + bytesWritten) / (device.bandwidth * mbu)
 *
 * @returns Latency in nanoseconds
 */
export function estimateLatencyNs(
  op: OpProfile,
  device: DeviceSpec,
  efficiency: EfficiencyModel,
  quantization: QuantType,
): number;

/**
 * Calculate arithmetic intensity (FLOPS / byte).
 * Used to determine if an op is compute-bound or memory-bound.
 */
export function arithmeticIntensity(op: OpProfile): number;

/**
 * Calculate the ridge point for a device (FLOPS / bandwidth).
 * Operations below this intensity are memory-bound.
 */
export function ridgePoint(device: DeviceSpec, quant: QuantType): number;

/**
 * Build OpProfile for attention in decode mode (single token, all KV).
 */
export function attentionDecodeOp(
  batchSize: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  seqLen: number,
  quant: QuantType,
): OpProfile;

/**
 * Build OpProfile for FFN/expert compute.
 */
export function ffnOp(
  batchSize: number,
  hiddenDim: number,
  intermediateSize: number,
  quant: QuantType,
): OpProfile;

/**
 * Build OpProfile for attention in prefill mode (full sequence).
 */
export function attentionPrefillOp(
  batchSize: number,
  seqLen: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  quant: QuantType,
): OpProfile;
```

### File: `packages/core/src/communication.ts`

```typescript
/**
 * Communication cost model.
 *
 * Models latency for collective operations (AllReduce, AllGather, AllToAll)
 * based on interconnect topology.
 */

import type { InterconnectSpec } from "./types.js";

export type CollectiveOp = "allreduce" | "allgather" | "reducescatter" | "alltoall";

/**
 * Estimate collective operation latency.
 *
 * Uses the ring algorithm model:
 *   allreduce: 2 * (n-1)/n * bytes / bandwidth + 2 * (n-1) * latency
 *   allgather: (n-1)/n * bytes / bandwidth + (n-1) * latency
 *   alltoall:  (n-1)/n * bytes / bandwidth + (n-1) * latency
 *
 * @param op - Collective operation type
 * @param messageBytes - Total message size in bytes
 * @param worldSize - Number of participants
 * @param interconnect - Link specification (bandwidth + latency)
 * @param efficiency - Communication efficiency factor (0.0-1.0)
 * @returns Latency in nanoseconds
 */
export function estimateCollectiveLatencyNs(
  op: CollectiveOp,
  messageBytes: number,
  worldSize: number,
  interconnect: InterconnectSpec,
  efficiency?: number,
): number;

/**
 * Estimate point-to-point transfer latency (e.g., expert load from host).
 *
 * time = bytes / bandwidth + latency
 */
export function estimateTransferLatencyNs(
  bytes: number,
  interconnect: InterconnectSpec,
): number;
```

---

## Phase 3: Governor Simulation

### Goal

Simulate the DeviceGovernor/HostGovernor resource management layer,
including the pressure protocol, to validate the design from MEMORY_ARCHITECTURE.md.

### File: `packages/core/src/governor.ts`

```typescript
/**
 * Governor simulation.
 *
 * Models the two-tier resource management system:
 * - DeviceGovernor: per-device VRAM budget management
 * - HostGovernor: per-node host RAM + pressure protocol
 *
 * The pressure protocol uses an epoch-based ticket system:
 * 1. Device needs memory → creates PressureTicket (brief lock)
 * 2. Host checks free pages → grants immediately or sends ReclaimNotice
 * 3. Victim devices evict pages (at their own pace)
 * 4. Host grants ticket when enough pages freed
 *
 * Key invariant: NO lock held across await/wait points.
 */

export interface DeviceGovernorState {
  deviceId: string;
  capacity: number;
  allocations: Map<string, Allocation>;
  budgets: {
    weights: { limit: number; used: number };
    kvCache: { limit: number; used: number };
    expertCache: { limit: number; used: number };
    activations: { limit: number; used: number };
  };
}

export interface HostGovernorState {
  capacity: number;
  free: number;
  allocations: Map<string, Allocation>;
  pendingTickets: PressureTicket[];
  epoch: number;
}

export interface Allocation {
  id: string;
  size: number;
  category: "weights" | "kv_cache" | "expert_cache" | "activations" | "staging";
  deviceId: string;
}

export interface PressureTicket {
  id: string;
  epoch: number;
  deviceId: string;
  bytesNeeded: number;
  createdAtNs: number;
  status: "pending" | "granted" | "cancelled" | "timeout";
}

export interface ReclaimNotice {
  targetDevice: string;
  bytesRequested: number;
  priority: "normal" | "urgent";
}

export interface GovernorEvent {
  kind: "allocate" | "free" | "pressure_request" | "reclaim_notice" |
        "reclaim_complete" | "pressure_grant" | "pressure_timeout";
  timestampNs: number;
  deviceId: string;
  bytes: number;
  details?: string;
}

/**
 * Create initial governor states for a topology.
 */
export function initGovernors(
  topology: import("./types.js").HardwareTopology,
  memoryPolicy: import("./types.js").MemoryPolicyConfig,
): { devices: DeviceGovernorState[]; host: HostGovernorState };

/**
 * Request device memory allocation. Returns whether it succeeded
 * and if pressure protocol was triggered.
 */
export function requestDeviceMemory(
  device: DeviceGovernorState,
  host: HostGovernorState,
  category: Allocation["category"],
  bytes: number,
  timestampNs: number,
): { granted: boolean; events: GovernorEvent[]; latencyNs: number };

/**
 * Simulate pressure protocol resolution.
 * Called when a device request cannot be fulfilled from free pages.
 *
 * @returns Events generated and total resolution latency
 */
export function resolvePressure(
  ticket: PressureTicket,
  host: HostGovernorState,
  allDevices: DeviceGovernorState[],
  timestampNs: number,
  hostToDeviceBandwidth: number,
): { events: GovernorEvent[]; resolutionLatencyNs: number };
```

---

## Phase 4: CLI Implementation

### File: `packages/cli/src/main.ts`

```typescript
/**
 * CLI entry point.
 *
 * Commands:
 *   sim analyze   — Static memory analysis (Phase 1)
 *   sim run       — Full simulation with event loop (Phase 2+3)
 *   sim compare   — Compare multiple configs side-by-side
 *   sim list      — Show available presets (hardware, models, topologies)
 *
 * Dependencies: commander, chalk, yaml
 */
```

### Command: `sim analyze`

```
Usage: sim analyze [options]

Options:
  --hardware <preset|file>   Hardware topology (preset name or YAML path)
  --model <preset|file>      Model profile (preset name or YAML path)
  --quant <weights:kv>       Quantization, e.g., "fp8:fp8" (default: "fp16:fp16")
  --batch <n>                Batch size (default: 1)
  --seq <input:output>       Sequence lengths, e.g., "4096:2048"
  --tp <n>                   Tensor parallelism degree
  --pp <n>                   Pipeline parallelism degree
  --ep <n>                   Expert parallelism degree
  --json                     Output as JSON instead of formatted table
```

Example output:
```
╭─────────────────────────────────────────────────────╮
│  Mixtral-8x22B (FP8) on DGX-H100 (8×H100 SXM)     │
╰─────────────────────────────────────────────────────╯

Parallelism: TP=4, PP=1, EP=2, DP=1

Memory Layout (per GPU):
  ├─ Weights (TP shard)     18.2 GiB  ████████░░░░░░░░
  ├─ KV Cache (max)         28.4 GiB  ████████████░░░░
  ├─ Expert Cache (EP)      24.0 GiB  ██████████░░░░░░
  ├─ Activations             3.8 GiB  ██░░░░░░░░░░░░░░
  └─ Free                    5.6 GiB  ███░░░░░░░░░░░░░
  Total: 80.0 GiB

Throughput (roofline estimate):
  Prefill:  3,420 tok/s  (compute-bound, MFU=0.45)
  Decode:     112 tok/s  (memory BW-bound, 92% HBM)
  TTFT:      38.4 ms
  ITL:        8.9 ms/tok

Bottleneck: HBM bandwidth during decode
Recommendations:
  • FP8 KV cache could free 14.2 GiB for more expert cache
  • Expert cache holds 100% of assigned experts (no offload needed)
```

### Command: `sim run`

```
Usage: sim run [options]

Options:
  --config <file>            YAML config file (includes all settings)
  --tokens <n>               Number of tokens to simulate (default: 1000)
  --warmup <n>               Warm-up tokens (default: 100)
  --seed <n>                 Random seed (default: 42)
  --trace <file>             Output trace JSON for visualization
  --json                     Output summary as JSON
```

### Command: `sim compare`

```
Usage: sim compare [options] <config1.yaml> <config2.yaml> [config3.yaml...]

Runs static analysis (or full sim) on multiple configs and shows side-by-side comparison.

Output: table with metrics per config (throughput, memory usage, bottleneck, etc.)
```

### Config YAML Format

```yaml
# Full simulation config
hardware:
  preset: "dgx-h100"
  # OR custom topology:
  # custom:
  #   nodes:
  #     - id: node0
  #       devices:
  #         - kind: gpu
  #           memory: { capacity: "80 GiB", bandwidth: "3.35 TB/s" }
  #           compute: { fp16: "990 TFLOPS", fp8: "1979 TFLOPS" }
  #       host_memory: { capacity: "2 TiB", bandwidth: "50 GB/s" }
  #       interconnect: { kind: nvlink, bandwidth: "900 GB/s" }

model:
  preset: "mixtral-8x22b"
  # OR custom:
  # custom:
  #   name: "My-MoE-Model"
  #   architecture: { kind: moe, num_layers: 56, hidden_dim: 6144, ... }
  #   moe: { num_experts: 8, active_per_token: 2, expert_size: "3.1 GB" }

quantization:
  weights: fp8
  kv_cache: fp8
  activations: fp16

pipeline:
  batch_size: 32
  input_seq_len: 4096
  output_seq_len: 2048
  parallelism:
    tensor_parallel: 4
    pipeline_parallel: 1
    expert_parallel: 2
    data_parallel: 1

memory:
  kv_cache_budget: 0.4        # fraction of VRAM
  expert_cache_budget: 0.3
  pinned_pool: 0.1
  offload: partial             # none | partial | full
  prefetch_ahead: 2
  pressure_threshold: 0.85

simulation:
  mode: full                   # static | full
  tokens: 1000
  warmup: 100
  seed: 42
  snapshot_interval: 50
  trace_output: trace.json
```

---

## Phase 5: Web Visualization

### Tech Stack

- **Framework:** React 19 + TypeScript
- **Build:** Vite
- **Charts:** D3.js (custom SVG for memory timeline, heatmap)
- **Topology diagram:** React Flow (interactive node graph)
- **State:** Zustand (lightweight store)
- **Simulation:** Web Worker (non-blocking UI)

### Components Specification

#### `ConfigPanel.tsx`
- Dropdowns for hardware preset, model preset, quantization
- Sliders for batch size, TP/PP/EP degrees, memory budgets
- "Run Simulation" button → dispatches to Web Worker
- Import/export YAML config

#### `MemoryTimeline.tsx`
- X-axis: token index (or time)
- Y-axis: memory bytes, stacked area chart
- Layers: weights (blue), KV cache (green), expert cache (orange), activations (purple), free (gray)
- Vertical markers for pressure events, eviction cascades
- Hover tooltip: exact bytes per category
- One chart per device (switchable tabs or small multiples)

#### `ExpertHeatmap.tsx`
- Grid: X = token index (time), Y = expert ID
- Color: hot tier (green), warm tier (yellow), cold/not loaded (gray), active-this-token (red border)
- Shows cache warm-up pattern (starts cold → stabilizes)
- Click expert → show access frequency, last access, eviction count

#### `RooflinePlot.tsx`
- Classic roofline: X = arithmetic intensity (FLOP/byte), Y = performance (FLOP/s)
- Ceiling lines: compute ceiling, memory bandwidth ceiling
- Plot points: attention-prefill, attention-decode, FFN-prefill, FFN-decode, expert-compute
- Ridge point marked
- Shows which ops are compute-bound vs memory-bound

#### `TopologyView.tsx`
- Interactive graph: nodes = devices, edges = interconnects
- Node size ∝ memory capacity, node fill = utilization %
- Edge thickness ∝ bandwidth, edge color = saturation during sim
- Click device → show memory breakdown pie chart
- Animated data flow during simulation playback

#### `BandwidthSankey.tsx`
- Sankey diagram: sources → interconnects → destinations
- Shows data movement volume over simulation
- Highlights bottleneck links (> 80% saturation)

#### `MetricsPanel.tsx`
- Summary cards: Prefill tok/s, Decode tok/s, TTFT, ITL
- Expert cache: hit rate, prefetch hit rate, cold miss rate
- Pressure protocol: events triggered, avg resolution time
- Comparison mode: side-by-side metrics for multiple configs

### Web Worker Protocol

```typescript
// Messages from main thread → worker
type WorkerInput =
  | { kind: "run"; input: SimulationInput }
  | { kind: "abort" };

// Messages from worker → main thread
type WorkerOutput =
  | { kind: "progress"; tokenIdx: number; totalTokens: number }
  | { kind: "snapshot"; data: DeviceMemoryBreakdown[] }
  | { kind: "complete"; trace: SimTrace }
  | { kind: "error"; message: string };
```

---

## Key Algorithms

### Roofline Model

```
For each operation:
  compute_time_ns = flops / (device.peakFlops * MFU) * 1e9
  memory_time_ns = total_bytes / (device.bandwidth * MBU) * 1e9
  latency_ns = max(compute_time_ns, memory_time_ns)

Decode (batch=1): arithmetic_intensity = 2/bytes_per_weight ≈ 1-2 → MEMORY BOUND
Prefill (batch=32, seq=4096): arithmetic_intensity = 2*batch*seq/bytes ≈ high → COMPUTE BOUND
```

### Expert Cache LRU

```
on access(expert_id):
  if expert_id in hot_tier:
    hot_tier.move_to_front(expert_id)
    return HIT
  if expert_id in warm_tier:
    warm_tier.remove(expert_id)
    if hot_tier.full():
      evicted = hot_tier.pop_back()
      warm_tier.push_front(evicted)  // demote to warm
    hot_tier.push_front(expert_id)
    return WARM_HIT (latency = host→device transfer)
  // Cold miss
  if hot_tier.full():
    evicted = hot_tier.pop_back()
    warm_tier.push_front(evicted)
  hot_tier.push_front(expert_id)
  return COLD_MISS (latency = disk→host→device transfer)
```

### Prefetch Strategy

```
on each token completion:
  // Predict next token's experts using frequency distribution
  candidates = top-K most-frequent experts NOT in hot_tier
  for expert in candidates[:prefetch_ahead]:
    if bandwidth_available:
      start_async_load(expert, warm→hot)
      mark_as_prefetching(expert)
```

### Zipf Distribution Sampling

```
For expert selection with Zipf distribution (parameter s):
  weight[i] = 1 / (i+1)^s  for i in 0..num_experts
  normalize weights to sum=1
  sample activePerToken experts WITHOUT replacement using weights
```

### Communication Model (Ring AllReduce)

```
allreduce_latency = 2 * (n-1) * alpha + 2 * (n-1)/n * bytes / beta
  where:
    n = world_size (number of participants)
    alpha = per-message latency (interconnect.latencyNs)
    beta = bandwidth (interconnect.bandwidthBytesPerSec * efficiency)
```

---

## Testing Strategy

### Unit Tests (vitest)

- `static-analysis.test.ts` — Memory breakdown correctness (DONE)
- `expert-cache.test.ts` — LRU/LFU behavior, hit rate calculation
- `roofline.test.ts` — Latency estimates match known values
- `communication.test.ts` — Collective latency formulas
- `event-loop.test.ts` — Full simulation produces valid trace
- `governor.test.ts` — Pressure protocol resolution

### Integration Tests

- Compare simulator output against published benchmarks:
  - Llama-3-70B on 8×A100: compare decode tok/s with vLLM published numbers
  - Mixtral-8x22B on DGX-H100: compare with InferSim estimates

### Property Tests

- Memory breakdown: sum of categories ≤ device capacity (when feasible)
- Expert cache: hits + misses = total accesses
- Governor: no allocation exceeds capacity at any timestep
- Trace: timestamps are monotonically increasing

---

## Implementation Notes

### Pure Computation Constraint

`@inference-sim/core` MUST have zero runtime dependencies and no I/O.
It must work identically in Node.js and browser (Web Worker).
All randomness uses the seeded PRNG, never `Math.random()`.

### Number Precision

Use regular JavaScript numbers (f64). For memory sizes, use bytes as integers.
For time, use nanoseconds as integers (safe up to ~104 days in Number.MAX_SAFE_INTEGER).

### Performance Target

Full simulation of 1000 tokens for a 128-expert MoE model should complete in < 100ms
on a modern CPU. This enables interactive parameter sweeps in the web UI.

### Latency Jitter (from llm-d-inference-sim learnings)

For more realistic traces, add Gaussian jitter to computed latencies:
- std_dev = base_latency * jitter_factor (default jitter_factor = 0.1)
- Clamp to ±70% of mean (never negative)
- Optional — disabled by default for deterministic testing

### Serialization

SimTrace should be JSON-serializable for:
1. Saving to disk (CLI `--trace` output)
2. Passing from Web Worker to main thread
3. Loading in visualization dashboard

---

## Future Extensions (not in v1)

- **ONNX model parsing:** Read .onnx file → auto-extract ModelProfile
- **FrozenPlan validation:** Read execution plan → simulate with real DAG dependencies
- **P/D Disaggregation:** Separate prefill/decode phases on different devices
- **Request-level simulation:** Multiple concurrent requests, batching dynamics
- **Auto-optimizer:** Given model + hardware, search for optimal parallelism config
- **Empirical calibration:** Profile real hardware, adjust MFU/MBU to match measurements
