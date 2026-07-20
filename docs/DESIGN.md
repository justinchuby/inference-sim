# Inference Simulator Design

A discrete-event simulator for LLM inference resource allocation and scheduling.
Models hardware topologies, ONNX model profiles, and memory management policies
to predict throughput, latency, memory flow, and bottlenecks — **without real hardware**.

## Goals

1. **Memory flow visualization** — where do weights/KV/activations live at each step?
2. **Expert cache dynamics** — hit rates, prefetch effectiveness, eviction patterns
3. **Governor behavior validation** — does the pressure protocol work as designed?
4. **Config comparison** — same model on different hardware, find optimal placement
5. **Bottleneck identification** — compute-bound vs memory-bound vs communication-bound

## Non-Goals (v1)

- Not a serving simulator (no request queuing, no SLO optimization)
- Not a training simulator
- Not kernel-accurate (uses roofline model, not cycle-level simulation)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI / Web UI                        │
├─────────────────────────────────────────────────────────┤
│                    SimEngine (core)                      │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Hardware    │  │    Model     │  │   Scheduler   │  │
│  │  Topology    │  │   Profile    │  │   Policies    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Event Loop (discrete time)              ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Trace / Report / Timeline               ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Package Structure (TypeScript monorepo)

```
packages/
  core/                 # Pure computation, no I/O, runs anywhere
    src/
      hardware.ts       # Hardware topology modeling
      model.ts          # Model profile (from ONNX or manual spec)
      scheduler.ts      # Governor simulation, placement policies
      event-loop.ts     # Discrete-event simulation engine
      roofline.ts       # Performance model (FLOPS, bandwidth)
      trace.ts          # Event recording for visualization
      types.ts          # Shared types
    index.ts

  cli/                  # Node.js CLI entry point
    src/
      main.ts           # CLI with YAML/JSON config input
      onnx-reader.ts    # Parse ONNX model for profile extraction
      report.ts         # Terminal output formatting

  web/                  # Browser visualization
    src/
      App.tsx
      components/
        TopologyView.tsx      # Hardware topology diagram
        MemoryTimeline.tsx    # Memory allocation over time
        ExpertHeatmap.tsx     # Expert placement + cache hit/miss
        BandwidthFlow.tsx     # Data movement Sankey diagram
        ConfigPanel.tsx       # Input hardware/model config
        MetricsPanel.tsx      # TTFT, TPOT, throughput, stalls
      hooks/
        useSimulation.ts     # Run sim in web worker
      workers/
        sim.worker.ts        # Core sim in web worker thread
```

---

## Core Data Model

### Hardware Topology

```typescript
interface HardwareTopology {
  nodes: NodeSpec[];
  interNodeLinks: InterconnectSpec[];  // e.g., InfiniBand, Ethernet
}

interface NodeSpec {
  id: string;
  devices: DeviceSpec[];
  hostMemory: MemorySpec;
  interDeviceLinks: InterconnectSpec[];  // NVLink, PCIe, etc.
}

interface DeviceSpec {
  id: string;
  kind: "gpu" | "npu" | "unified";  // unified = Apple Silicon
  memory: MemorySpec;
  compute: ComputeSpec;
}

interface MemorySpec {
  capacityBytes: number;
  bandwidthBytesPerSec: number;  // HBM bandwidth or DDR
  latencyNs: number;
}

interface ComputeSpec {
  fp16Flops: number;
  fp8Flops: number;
  int8Flops: number;
  // For roofline model
  arithmeticIntensityThreshold: number;  // FLOPS / byte
}

interface InterconnectSpec {
  endpoints: [string, string];  // device/node IDs
  bandwidthBytesPerSec: number;
  latencyNs: number;
  kind: "nvlink" | "pcie" | "infiniband" | "ethernet" | "thunderbolt" | "on-chip";
}
```

### Hardware Presets

```typescript
// Built-in hardware library
const PRESETS = {
  "h100-sxm": { memory: 80GiB, hbm: 3.35TB/s, fp16: 990TFLOPS, ... },
  "h200-sxm": { memory: 141GiB, hbm: 4.8TB/s, ... },
  "a100-80g": { memory: 80GiB, hbm: 2.0TB/s, fp16: 312TFLOPS, ... },
  "l40s":     { memory: 48GiB, hbm: 864GB/s, ... },
  "rtx-4090": { memory: 24GiB, hbm: 1.0TB/s, fp16: 165TFLOPS, ... },
  "m1-max":   { unified: 64GiB, bandwidth: 400GB/s, ... },
  "m4-max":   { unified: 128GiB, bandwidth: 546GB/s, ... },
  // ...
};

const TOPOLOGY_PRESETS = {
  "dgx-h100":   { nodes: 1, gpus: 8, nvlink: 900GB/s, ... },
  "2x-dgx-h100": { nodes: 2, gpus: 16, ib: 400Gb/s, ... },
  "4x-mac-studio-m4": { nodes: 4, thunderbolt: 40Gb/s, ... },
};
```

### Model Profile

```typescript
interface ModelProfile {
  name: string;
  architecture: ModelArchitecture;
  totalParams: number;
  quantization: Quantization;
  layers: LayerProfile[];
  // MoE-specific
  moe?: MoEProfile;
}

interface ModelArchitecture {
  kind: "dense" | "moe";
  numLayers: number;
  hiddenDim: number;
  numHeads: number;
  numKVHeads: number;  // GQA
  vocabSize: number;
  intermediateSize: number;
}

interface MoEProfile {
  numExperts: number;
  activeExpertsPerToken: number;
  expertSize: number;  // bytes per expert (post-quantization)
  // Activation distribution — critical for cache simulation
  activationDistribution: ExpertDistribution;
}

// How often each expert is activated (from profiling or synthetic)
type ExpertDistribution =
  | { kind: "uniform" }
  | { kind: "zipf"; s: number }  // power-law skew
  | { kind: "empirical"; frequencies: number[] }  // measured per-expert
  | { kind: "clustered"; hotExperts: number; hotFrequency: number };

interface LayerProfile {
  index: number;
  attentionBytes: number;    // weight size
  ffnBytes: number;          // weight size (dense) or shared expert size (MoE)
  kvCachePerToken: number;   // bytes per token per layer
}

interface Quantization {
  weights: "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4" | "nf4";
  kvCache: "fp16" | "fp8" | "int8";
  activations: "fp16" | "bf16" | "fp8";
}
```

### Pipeline Configuration

```typescript
interface PipelineConfig {
  batchSize: number;
  inputSeqLen: number;
  outputSeqLen: number;
  parallelism: ParallelismConfig;
  memory: MemoryPolicyConfig;
}

interface ParallelismConfig {
  tensorParallel: number;    // TP degree
  pipelineParallel: number;  // PP stages
  expertParallel: number;    // EP degree (MoE)
  dataParallel: number;      // DP replicas
}

interface MemoryPolicyConfig {
  // Governor config
  kvCacheBudgetFraction: number;      // 0.0-1.0, fraction of free VRAM for KV
  expertCacheBudgetFraction: number;  // 0.0-1.0, fraction for hot experts
  pinnedPoolFraction: number;         // non-evictable portion

  // Offload policy
  offloadStrategy: "none" | "partial" | "full";
  prefetchAhead: number;              // experts to prefetch ahead

  // Pressure protocol
  pressureThreshold: number;          // 0.0-1.0, when to trigger eviction
  reclaimBatchSize: number;           // pages to reclaim per pressure event
}
```

---

## Simulation Engine

### Event Loop

```typescript
type SimEvent =
  | { kind: "token_start"; tokenIdx: number; batchIdx: number }
  | { kind: "layer_start"; layerIdx: number }
  | { kind: "attention_compute"; layerIdx: number; durationNs: number }
  | { kind: "expert_route"; layerIdx: number; expertIds: number[] }
  | { kind: "expert_cache_hit"; expertId: number; device: string }
  | { kind: "expert_cache_miss"; expertId: number; loadFromDevice: string }
  | { kind: "expert_load_start"; expertId: number; bytes: number }
  | { kind: "expert_load_complete"; expertId: number; durationNs: number }
  | { kind: "expert_evict"; expertId: number; device: string }
  | { kind: "ffn_compute"; layerIdx: number; durationNs: number }
  | { kind: "allreduce_start"; bytes: number; group: string }
  | { kind: "allreduce_complete"; durationNs: number }
  | { kind: "all_to_all_start"; bytes: number }
  | { kind: "all_to_all_complete"; durationNs: number }
  | { kind: "kv_cache_allocate"; layerIdx: number; bytes: number }
  | { kind: "kv_cache_evict"; tokens: number; bytes: number }
  | { kind: "pressure_request"; device: string; bytesNeeded: number }
  | { kind: "pressure_grant"; device: string; bytesGranted: number }
  | { kind: "pressure_reclaim"; fromDevice: string; bytes: number }
  | { kind: "token_complete"; tokenIdx: number; latencyNs: number }
  | { kind: "memory_snapshot"; allocations: MemorySnapshot[] };

interface SimTrace {
  events: SimEvent[];
  timestamps: number[];  // nanosecond timeline
}
```

### Performance Model (Roofline)

```typescript
/**
 * Roofline model: min(compute_time, memory_time)
 *
 * For each operation:
 *   compute_time = FLOPS / device.peakFlops
 *   memory_time  = bytes_accessed / device.memoryBandwidth
 *   actual_time  = max(compute_time, memory_time) / efficiency
 *
 * Decode is memory-bound (low arithmetic intensity).
 * Prefill is compute-bound (high batch × seq_len).
 */
function estimateOpLatency(
  op: OpProfile,
  device: DeviceSpec,
  efficiency: EfficiencyModel
): number {
  const computeNs = op.flops / (device.compute.fp16Flops * efficiency.computeMFU);
  const memoryNs = op.bytesAccessed / (device.memory.bandwidthBytesPerSec * efficiency.memoryMBU);
  return Math.max(computeNs, memoryNs) * 1e9;
}

interface EfficiencyModel {
  computeMFU: number;   // Model FLOPS Utilization, typically 0.3-0.7
  memoryMBU: number;    // Memory Bandwidth Utilization, typically 0.7-0.9
  commEfficiency: number;  // Communication efficiency, typically 0.8-0.95
}
```

### Expert Cache Simulation

```typescript
interface ExpertCache {
  hotTier: Map<number, ExpertEntry>;    // On device VRAM
  warmTier: Map<number, ExpertEntry>;   // On host RAM
  coldTier: Set<number>;                // On disk / not loaded

  policy: "lru" | "lfu" | "frequency-aware";
  capacity: number;  // max experts in hot tier

  // Stats
  hits: number;
  misses: number;
  evictions: number;
  prefetchHits: number;
}

/**
 * Simulate expert routing for one token.
 * Returns: which experts are cache hits, which need loading.
 */
function routeToken(
  cache: ExpertCache,
  selectedExperts: number[],
  prefetchQueue: number[]
): ExpertRouteResult {
  const hits: number[] = [];
  const misses: number[] = [];

  for (const expertId of selectedExperts) {
    if (cache.hotTier.has(expertId)) {
      hits.push(expertId);
      cache.hits++;
    } else {
      misses.push(expertId);
      cache.misses++;
      // Load from warm/cold → hot, evict if needed
      loadExpert(cache, expertId);
    }
  }

  // Prefetch next likely experts (overlap with current compute)
  updatePrefetchQueue(cache, prefetchQueue);

  return { hits, misses, loadLatencyNs: calculateLoadTime(misses) };
}
```

### Governor Simulation

```typescript
interface SimDeviceGovernor {
  deviceId: string;
  totalCapacity: number;
  allocations: Map<string, SimAllocation>;  // PhysicalAllocationId → entry

  // Budget partitions
  weightsBudget: number;
  kvCacheBudget: number;
  expertCacheBudget: number;
  activationsBudget: number;

  used: { weights: number; kvCache: number; expertCache: number; activations: number };
}

interface SimHostGovernor {
  totalCapacity: number;
  allocations: Map<string, SimAllocation>;

  // Pressure protocol state
  pendingTickets: PressureTicket[];
  epoch: number;
}

/**
 * Simulate pressure protocol interaction.
 * Returns timeline of events (request → reclaim notices → grants).
 */
function simulatePressure(
  requestingDevice: SimDeviceGovernor,
  host: SimHostGovernor,
  allDevices: SimDeviceGovernor[],
  bytesNeeded: number
): PressureTimeline {
  // 1. Device creates ticket (brief lock)
  const ticket = createTicket(host, requestingDevice.deviceId, bytesNeeded);

  // 2. Host checks if free pages available
  if (host.freePages >= bytesNeeded) {
    return { events: [grant(ticket)], totalLatencyNs: HOST_LOCK_NS };
  }

  // 3. Send reclaim notices to other devices
  const victims = selectVictims(allDevices, requestingDevice.deviceId, bytesNeeded);
  const reclaimEvents = victims.map(v => simulateReclaim(v, host));

  // 4. After reclaim completes, grant
  const reclaimLatency = Math.max(...reclaimEvents.map(e => e.latencyNs));
  return {
    events: [...reclaimEvents, grant(ticket)],
    totalLatencyNs: reclaimLatency + HOST_LOCK_NS
  };
}
```

---

## Simulation Modes

### Mode 1: Static Analysis (instant)

No event loop. Pure math:
- Memory breakdown: where each component lives
- Feasibility check: does it fit?
- Optimal placement suggestion

```typescript
interface StaticAnalysisResult {
  feasible: boolean;
  memoryBreakdown: {
    device: string;
    weights: number;
    kvCache: number;
    expertCache: number;
    activations: number;
    free: number;
  }[];
  bottleneck: "compute" | "memory_bandwidth" | "interconnect" | "capacity";
  estimatedThroughput: { prefillToksPerSec: number; decodeToksPerSec: number };
}
```

### Mode 2: Steady-State Simulation (fast)

Simulate N tokens with the event loop. Captures:
- Expert cache warm-up and steady-state hit rate
- Average latency breakdown (compute vs load vs communication)
- Pressure protocol frequency

### Mode 3: Trace Simulation (detailed)

Full event trace for visualization. Every memory allocation, every cache miss,
every collective. Generates a timeline JSON that the web UI can render.

---

## Visualization (Web UI)

### 1. Memory Timeline

X-axis: time (or token index).
Y-axis: memory usage per device, stacked by category (weights/KV/experts/activations).
Shows eviction events, pressure events as markers.

### 2. Expert Heatmap

Grid: experts × time steps.
Color: hot (on device) / warm (on host) / cold (not loaded).
Overlay: which experts are active per token → shows cache hit/miss pattern.

### 3. Topology + Data Flow

Interactive diagram of hardware topology.
Animated data flow during simulation: show bytes moving over NVLink/PCIe/IB.
Bandwidth utilization color coding (green = low, red = saturated).

### 4. Roofline Plot

Classic roofline: arithmetic intensity vs performance.
Plot each operation (attention, FFN, expert, allreduce) to show compute vs memory bound.

### 5. Config Comparison

Side-by-side: same model on different hardware.
Or: same hardware with different parallelism configs.
Radar chart: throughput, latency, memory efficiency, cost.

---

## Input Formats

### YAML Config (CLI)

```yaml
hardware:
  preset: "dgx-h100"  # or custom spec below
  # custom:
  #   nodes: [...]

model:
  preset: "mixtral-8x22b"  # or custom
  quantization:
    weights: fp8
    kv_cache: fp8

pipeline:
  batch_size: 32
  input_seq_len: 4096
  output_seq_len: 2048
  parallelism:
    tensor_parallel: 4
    expert_parallel: 2

memory:
  kv_cache_budget: 0.4
  expert_cache_budget: 0.3
  offload: partial
  prefetch_ahead: 2

simulation:
  mode: steady_state
  num_tokens: 1000
  warmup_tokens: 100
```

### ONNX Model Input (CLI)

```bash
# Extract model profile directly from ONNX file
onnx-sim profile ./model.onnx --output model-profile.json

# Run simulation
onnx-sim run --hardware dgx-h100.yaml --model model-profile.json --pipeline config.yaml
```

---

## Output Formats

### CLI Report

```
╭─────────────────────────────────────────────────────╮
│  Simulation: Mixtral-8x22B (FP8) on 8×H100 SXM     │
╰─────────────────────────────────────────────────────╯

Memory Layout (per GPU after TP=4, EP=2):
  ┌────────────────────────────────────────┐
  │ Weights (TP shard)     │  18.2 GiB     │ ████████░░
  │ KV Cache (max)         │  28.4 GiB     │ ████████████░░
  │ Expert Cache (hot)     │  24.0 GiB     │ ███████████░░
  │ Activations            │   3.8 GiB     │ ██░░░░░░░░
  │ Free                   │   5.6 GiB     │ ███░░░░░░░
  └────────────────────────────────────────┘

Expert Cache (steady state after 100 tokens warmup):
  Hit rate:        91.2%
  Prefetch hits:   8.1%  (would-be misses caught by prefetch)
  Cold loads:      0.7%  (stall events)
  Avg load time:   142 μs (PCIe, 3.1 GB expert × 2)

Performance:
  Prefill (4096 tokens):   3,420 tok/s  [compute-bound]
  Decode:                    112 tok/s  [memory BW-bound]
  Expert stall overhead:     2.1% of decode

Pressure Protocol:
  Events triggered:  14 / 1000 tokens
  Avg resolution:    89 μs
  Max resolution:    340 μs (cascade reclaim)

Bottleneck: HBM bandwidth during decode (92% utilized)
Suggestion: FP8 KV cache → 50% KV reduction → more expert budget
```

### JSON Trace (for web visualization)

```json
{
  "metadata": { "model": "...", "hardware": "...", "config": "..." },
  "timeline": [
    { "t": 0, "event": "token_start", "token": 0 },
    { "t": 1200, "event": "attention_compute", "layer": 0, "duration": 1200 },
    { "t": 1200, "event": "expert_route", "layer": 0, "experts": [12, 45, 103, ...] },
    { "t": 1200, "event": "expert_cache_hit", "expert": 12 },
    { "t": 1200, "event": "expert_cache_miss", "expert": 103 },
    { "t": 1200, "event": "expert_load_start", "expert": 103, "from": "host" },
    { "t": 1342, "event": "expert_load_complete", "expert": 103, "duration": 142000 },
    ...
  ],
  "snapshots": [
    { "t": 0, "memory": { "gpu0": { "weights": ..., "kv": ..., "experts": ... } } },
    ...
  ],
  "summary": { "throughput": ..., "hitRate": ..., ... }
}
```

---

## Tech Stack

- **Runtime:** Node.js (CLI) + Browser (Web Worker for simulation)
- **Core:** Pure TypeScript, zero dependencies (runs in both environments)
- **CLI:** Commander + chalk (minimal)
- **Web:** React + D3.js (timeline/heatmap) + React Flow (topology diagram)
- **Build:** tsup (core/cli) + Vite (web)
- **Monorepo:** pnpm workspaces
- **ONNX parsing:** onnxruntime-node or protobuf.js (for model profile extraction)

---

## Phased Implementation

### Phase 1: Static Memory Calculator

- Hardware presets (H100, A100, M4 Max, etc.)
- Model presets (Llama-70B, Mixtral-8x22B, DeepSeek-V2, etc.)
- Memory breakdown calculation
- Feasibility check + bottleneck identification
- CLI output

### Phase 2: Expert Cache Simulation

- Event loop (token-by-token)
- Expert activation sampling (Zipf / empirical distribution)
- LRU/LFU cache simulation
- Hit rate, load latency, stall percentage
- Prefetch policy simulation

### Phase 3: Governor + Pressure Protocol

- DeviceGovernor / HostGovernor simulation
- Pressure ticket lifecycle
- Eviction cascade modeling
- Multi-device interaction

### Phase 4: Web Visualization

- Memory timeline chart
- Expert heatmap
- Topology diagram with data flow
- Interactive config editor
- Side-by-side comparison mode

### Phase 5: ONNX Integration

- Read ONNX model → auto-extract profile
- Read FrozenPlan → simulate execution
- Validate plan against hardware constraints

---

## Relationship to onnx-genai

This simulator validates the designs in:
- `docs/MEMORY_ARCHITECTURE.md` — Governor behavior, pressure protocol
- `docs/DISTRIBUTED_RUNTIME.md` — Communication costs, plan scheduling
- `specs/tla/` — The TLA+ specs prove correctness; the simulator proves performance

The simulator can also serve as a **planning tool** for onnx-genai itself:
given a model + hardware, output the optimal `RuntimeConfig` that onnx-genai
should use (parallelism degrees, cache budgets, prefetch depth).

---

## Open Questions

1. **ONNX profile extraction:** use onnxruntime-node (heavy) or just parse protobuf (lightweight)?
   → Suggest: protobuf.js for shape/type info, manual FLOPS calculation per op type
2. **Validation data:** benchmark against InferSim / real vLLM measurements?
3. **Repo name:** `onnx-inference-sim`? `nxrt-sim`? `inference-sim`?
4. **License:** MIT (matches onnx-genai)
