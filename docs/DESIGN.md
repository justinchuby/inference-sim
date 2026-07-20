# Inference Simulator Design

**Status:** executable design, Phase 1 complete; Phase 2 in progress
**Protocol contract:** onnx-genai memory/distributed contract revision 2

`inference-sim` is a deterministic discrete-event simulator for LLM inference
resource planning and protocol behavior. It has two distinct jobs:

1. determine whether a model/plan can fit and execute on a topology; and
2. explore timing, contention, scheduling, and failure behavior without owning
   the real hardware.

It is not a benchmark oracle. Absolute throughput claims require calibration
against released model artifacts, measured kernels, and measured transports.

## 1. Design Goals

- Reproduce a run exactly from its input, seed, and scheduler decision trace.
- Model resources as owned, contended timelines rather than adding independent
  latency estimates.
- Preserve onnx-genai identities and protocol state transitions closely enough
  to consume a future `FrozenPlan` and protocol trace.
- Detect impossible schedules, memory overcommit, ordering divergence, leaked
  allocations, and premature buffer reuse.
- Separate correctness conclusions from calibrated performance estimates.
- Keep `@inference-sim/core` deterministic, browser-compatible, and free of I/O.

## 2. Non-Goals

- Predict exact latency from peak FLOPS and bandwidth alone.
- Simulate CUDA kernels, NCCL, operating-system paging, or network packets
  instruction by instruction.
- Treat a sampled visualization trace as protocol-conformance evidence.
- Claim that simulator/TLA+ success proves a production implementation.
- Optimize placement before the simulator can explain and reproduce one plan.

## 3. Confidence Classes

Every result is labeled:

| Class | Meaning |
|---|---|
| `exact` | Derived from identities, byte extents, dependency/order rules, or conservation laws. |
| `bounded` | Exhaustively checked within an explicitly finite scenario. |
| `calibrated` | Estimated from measured hardware/model coefficients with provenance. |
| `heuristic` | Based on an uncalibrated approximation such as default MFU/MBU. |

Feasibility, overflow, ordering, and ownership results may be `exact`. Latency
and throughput are never `exact`.

## 4. Architecture

```text
Scenario input
  |
  v
Validation + normalization
  - stable IDs
  - checked integer extents
  - topology and parallelism constraints
  - evidence/provenance
  |
  +----------------------+
  |                      |
  v                      v
Static capacity plan     Deterministic event kernel
                         |
                         +-- request/DAG scheduler
                         +-- device and link resources
                         +-- memory governors
                         +-- communicator groups
                         +-- cache/offload policy
                         |
                         v
                  Protocol trace + metrics
                         |
                         +-- independent invariant/replay checks
                         +-- reports and visualization
```

The static analyzer and event simulator share validated input types, but they
do not share mutable state. Static feasibility is a necessary precondition, not
proof that a concurrent schedule completes.

## 5. Deterministic Event Kernel

The event kernel is the only owner of simulation time.

### 5.1 Ordering

Each scheduled event has:

```typescript
interface ScheduledEvent<E> {
  readonly id: number;
  readonly timestampNs: number;
  readonly sequence: number;
  readonly payload: E;
}
```

Events are ordered by `(timestampNs, sequence)`. `sequence` is a monotonic
insertion order, so same-time behavior is stable across engines and runs.
Wall-clock time and `Math.random()` are forbidden in core simulation.

An event handler may schedule more work at the current or future simulated
time. Scheduling in the past, unsafe integers, negative durations, duplicate
event IDs, and exceeding the configured event limit are fatal scenario errors.

### 5.2 Resource Contention

Compute devices, copy engines, host-memory channels, and interconnect links are
resources with availability timelines. An operation reserves a resource over
an interval:

```text
start = max(dependency_ready, resource_available)
finish = start + modeled_duration
resource_available = finish
```

Independent resources may overlap. Operations sharing a resource serialize
unless the resource declares explicit capacity/lane semantics. The simulator
must not model overlap by taking `max()` over durations after the fact.

### 5.3 Cancellation

Cancellation marks a queued event ID. A cancelled event is skipped when popped;
it never reuses an ID. Cancellation of a protocol operation is separate from
event-queue cancellation and must follow the modeled owner/state machine.

## 6. Stable Identity Model

All protocol events use stable values, never array positions or object identity:

- `TopologyEpoch`
- `GlobalDeviceId = (nodeId, localDeviceId)`
- immutable world `RankId`
- `GroupId` plus ordered-membership hash
- `ExecutionId` and plan-time `CommSequenceId`
- `PressureRequestId` and configuration generation
- process-unique `PhysicalAllocationId`
- simulator-local scheduled event ID and source sequence

An identity cannot be reused while any event, allocation, operation, or trace
entry can still refer to it.

## 7. Memory and Pressure Protocol

### 7.1 Physical Ledger

The authoritative ledger is keyed by `PhysicalAllocationId`. Views and aliases
do not create another physical charge.

At every transition:

```text
capacity
  = fixed_non_reclaimable
  + free
  + reclaimable
  + granted_unclaimed
  + claimed_live
  + other_explicit_classes
```

All byte arithmetic is finite, non-negative, integer, and checked before state
mutation. Unified-memory residency reclassifies one ledger entry; it does not
charge host and device copies simultaneously.

### 7.2 Pressure Ticket States

```text
Idle -> Pending -> Granted -> Claimed -> Completed
                  |           |
                  v           v
              Cancelled    release

Pending/Granted -> Failed(timeout)
Pending         -> Failed(reconfigure)
```

The exact allocation charge and `Pending -> Granted` commit atomically before a
wakeup is observable. Claim transfers that allocation once. Cancellation or
timeout racing a grant has one ledger-ordered winner and returns the exact
unclaimed allocation when it wins.

Reconfiguration increments a configuration generation and resolves every
prior-generation pending ticket. Reclaim notices are bounded notifications;
credited bytes become available only when reclaim completion linearizes.

### 7.3 Simulator Conformance

The governor emits a lossless contract-revisioned trace. An independent replay
checker reconstructs ticket/allocation state and evaluates invariants after
every event. It does not call the governor's transition functions.

Required campaigns include:

- multiple variable-sized tickets from one and multiple devices;
- exact-capacity and fixed-charge boundaries;
- grant versus claim, cancel, timeout, release, and reconfigure;
- notification saturation and requester-as-reclaim-victim; and
- checked-arithmetic and duplicate-identity failures.

## 8. Distributed Execution

### 8.1 Frozen Plan

The target input is one immutable DAG:

```typescript
interface PlanStep {
  id: number;
  participants: readonly RankId[];
  dependencies: readonly number[];
  reads: readonly PhysicalAllocationId[];
  writes: readonly PhysicalAllocationId[];
  operation: ComputeStep | TransferStep | CollectiveStep;
}
```

Each rank derives a local view. A step transitions exactly once:

```text
Pending -> InFlight -> Terminal
```

Enqueue is not terminal completion. Dependent steps become ready only after the
required rank-local device/transport fence.

### 8.2 Collective Ordering

Each communicator has frozen ordered membership and its own submit sequencer.
Collectives consume `(ExecutionId, CommSequenceId)` lexicographically within
that group. Different groups may interleave independently.

Coordinator admission/skip is monotonic. A skip before submission is observed
by every member. Failure after any member submits triggers abort; a rank cannot
silently omit the operation.

Completion is rank-local. Abort closes new submission before already-enqueued
operations quiesce and release allocation leases.

Collective steps declare both an immutable communicator group/sequence and the
transport links they reserve. Group ordering alone is not a bandwidth model;
collectives and point-to-point transfers contend on the same link lanes.

### 8.3 Buffer Ownership

Every operation registers complete read and write `PhysicalAllocationId` lease
sets before enqueue:

- read/read aliasing is legal;
- a writer conflicts with all other readers and writers;
- in-place operations hold the allocation exclusively;
- dropping an observation handle does not release backend ownership; and
- free/reuse occurs only after terminal success or abort quiescence.

## 9. Speculative Decoding

Speculative decoding is an execution protocol, not a scalar speedup factor.
The simulator models the proposer, target verifier, acceptance decision, cache
transaction, and committed output separately.

### 9.1 Supported Proposer Families

The scenario declares one of the onnx-genai proposer families:

| Family | State/cost that must be modeled |
|---|---|
| `prompt_lookup` | CPU lookup/search cost; no draft-model KV |
| `draft_model` | Separate model placement, compute, KV, and rewind |
| `mtp` | Target hidden seed, iterative sidecar compute, proposal-local or accepted-prefix KV, recurrent state |
| `eagle3` | Three target hidden taps, fused draft state, sidecar KV |
| `shared_kv` | Assistant compute plus read leases on declared target KV groups |
| `self_speculative` | Early-exit target layers and completion by remaining layers |

The simulator validates runtime eligibility such as proposal availability,
greedy/temperature-zero restrictions, grammar incompatibility, required target
hidden outputs, and declared KV lifetime. Unsupported combinations fail the
scenario; they do not silently fall back to target-only timing.

### 9.2 Verification Transaction

One linear speculative iteration is:

```text
target base step
  -> guaranteed target token and proposer seed
  -> capture composite checkpoint before speculative state mutation
  -> draft up to configured width
  -> one authoritative target verification forward
  -> accept longest matching prefix
  -> restore every target/proposer state stream to the accepted draft prefix
  -> ingest and commit the target-authoritative correction token on mismatch,
     or bonus token on full acceptance
```

The proposal width convention follows onnx-genai:

```text
proposal_width = 1 guaranteed target token + max additional draft tokens
```

Target verification is authoritative. The simulator never commits a draft
token solely because an acceptance-rate distribution sampled it as accepted.

### 9.3 Composite Checkpoint

The checkpoint is opaque to the engine-level transaction and bound to sequence,
generation, row mapping, and base logical length. It covers all configured
state groups:

- logical token sequence;
- dense/static/paged target KV;
- CSA compressed, index, and carry cursors plus bounded overwritten carry;
- proposer-local KV;
- MTP/EAGLE recurrent state;
- shared-KV lease/view state; and
- sampler/processor state when sampling speculation is explicitly supported.

Restore uses `checkpoint + accepted_prefix_offset`, not a naked target token
length. Rejected capacity tails may remain physically stale only when all
readers are validity-masked. Active carry/recurrent state needed by future
steps must be restored. A correction/bonus token is a new state write after
restore; increasing a cursor must never resurrect its stale speculative slot.

### 9.4 Acceptance Inputs

Acceptance behavior is configured using one of:

1. replayed token-level target/proposer traces;
2. empirical conditional acceptance by draft position and context bucket; or
3. an explicitly heuristic stochastic model with a fixed seed.

A single average acceptance rate is insufficient because accepted-prefix length
is a first-mismatch process. The stochastic fallback provides conditional
`P(match at position i | positions < i matched)`.

### 9.5 Timing and Resource Interaction

The simulator schedules proposer and target work on their actual assigned
devices and resources. Draft and target batches may differ. Shared-KV proposers
hold read leases on target allocations; separate drafts own their own KV.

Verification can use a multi-token target forward only when the selected
backend capability declares it. Rollback cost distinguishes:

- cursor-only logical restore;
- bounded device-to-device state snapshot restore;
- paged-table mutation;
- sidecar reset; and
- recomputation, which is never assumed free.

Distributed target verification emits normal plan communication steps and
therefore obeys per-group collective ordering and buffer leases.

### 9.6 Correctness Invariants and Metrics

After every iteration:

- committed output equals the target-authoritative path for the modeled
  acceptance rule;
- all state groups have the same committed logical prefix;
- rejected drafts own no live allocation or lease;
- checkpoint identity cannot be reused across generation/topology epochs; and
- logical/capacity counters remain conserved after restore.

Reported metrics include accepted-prefix histogram, acceptance by position,
correction/bonus counts, effective committed tokens per target forward,
proposer/verification/rollback time, rejected compute and communication,
target/proposer KV high-water marks, and speedup versus a target-only simulation
using the same resource model.

### 9.7 Expert Routing and Cache Tiers

MoE routing samples weighted experts without replacement using a fixed,
trace-replayable seed. A route cannot select the same expert twice for one
token, and the complete routed working set must fit the hot-cache byte
capacity.

Cold storage is the authoritative expert-weight backing store. Warm and hot
tiers are independent cache copies, so one expert may be present in both.
Loads retain the source copy and reserve exact target-tier bytes before
transfer completion. Resident plus reserved bytes may never exceed capacity.
Completed copies become visible only at their deterministic completion event.

Eviction is byte-capacity LRU with expert identity as the stable tie-breaker.
Experts selected by an in-flight route are protected until access. Prefetch is
asynchronous, deduplicates target copies, and fails atomically if the complete
request cannot be reserved; demand routing likewise cannot consume RNG or emit
a partial route when admission fails.

The trace records routes, prefetch requests, evictions, load start/completion,
source tier, and access stalls. Independent replay re-derives seeded routes,
LRU victims, latency, capacity, and route-to-access correspondence. Metrics
include hot/warm/cold outcomes, bytes moved, evictions, load counts, stall time,
and per-tier high-water bytes.

## 10. Device Configuration Coverage

Device configurations are composed from memory domains, compute devices,
access capabilities, and links. Presets are data; core simulation must not
branch on names such as `dgx-h100`.

### 10.1 Required Topology Families

| Family | Memory semantics | Required simulation behavior |
|---|---|---|
| CPU-only | Host RAM is the hot/warm domain | No DeviceGovernor or fake host-device copy |
| Single discrete GPU + CPU | Separate VRAM and host RAM | Ticketed offload, copy fence, source release after publish |
| Multi-GPU discrete | Per-GPU VRAM plus node-shared host RAM | One HostGovernor, multiple DeviceGovernors, inter-GPU groups |
| GPU + NPU | Separate accelerator domains plus pinned host DMA | Pinned/pageable classes; NPU requests cannot silently degrade |
| Unified memory | One coherent physical ledger | Residency reclassification without copy or double charge |
| Multi-node | Per-node topology plus inter-node links | Cluster execution IDs, communicator groups, topology epoch/failure |

The same composition also covers heterogeneous scenarios listed by onnx-genai:
CUDA+MLX overflow, NPU attention with GPU FFN, multi-vendor GPU via host/RDMA,
and multiple CPU EPs for deterministic testing.

### 10.2 Scenario Schema

```typescript
interface SimulationScenario {
  readonly memoryDomains: readonly MemoryDomainSpec[];
  readonly devices: readonly SimDeviceSpec[];
  readonly links: readonly SimLinkSpec[];
  readonly placements: readonly PartitionPlacement[];
  readonly transfers: readonly TransferRequirement[];
  readonly groups: readonly CommunicatorGroupSpec[];
  readonly workload: WorkloadSpec;
  readonly execution: ExecutionPolicy;
  readonly calibration: CalibrationSet;
}
```

Each memory domain declares physical capacity, host/device accessibility,
coherence, allocation classes, and governor owner. Each directed link declares
endpoints, bandwidth/latency curves, concurrency lanes, staging requirement,
and evidence provenance.

Validation proves:

- every placement references a capable device;
- every transfer has a direct or staged path;
- every communicator group has immutable ordered world ranks;
- parallelism degrees correspond to concrete participants;
- discrete copies use distinct physical identities until commit;
- unified aliases preserve one physical identity; and
- fixed workspaces, staging, checkpoint snapshots, KV, and speculative sidecars
  fit their owning domains.

## 11. Performance Model

### 11.1 Operation Cost

Roofline estimates provide a base duration:

```text
compute_time = FLOPs / measured_effective_FLOPs
memory_time  = bytes / measured_effective_bandwidth
base_time    = max(compute_time, memory_time) + launch_overhead
```

Collective cost depends on algorithm, participants, topology path, message
extent, contention, and calibrated efficiency. A single ring formula is only a
heuristic fallback.

### 11.2 Provenance

Every hardware/model coefficient carries:

- source or measurement artifact;
- measurement date and software stack;
- dtype, shape/batch regime, and algorithm;
- confidence class; and
- valid interpolation range.

Unproven product names and unreleased workloads are not performance evidence.
Preset values without provenance are clearly marked heuristic.

### 11.3 Validation

Calibration uses released model artifacts and compares distributions, not one
headline number:

- time to first token;
- inter-token latency percentiles;
- throughput versus batch/sequence;
- peak and steady-state memory;
- collective latency versus message size; and
- cache hit/miss and bytes moved by tier.

## 12. Static Analysis Requirements

The current static analyzer is useful scaffolding, but production-quality
feasibility must:

- validate every device rather than applying the first device's capacity;
- assign layers/experts/KV to concrete ranks instead of dividing totals by
  degrees independently;
- reject incompatible TP/PP/EP/DP group construction;
- include plan workspaces, staging pools, fragmentation reserve, fixed charges,
  and concurrent high-water marks;
- check host capacity and transport accessibility before declaring offload
  feasible; and
- report assumptions and confidence class beside each estimate.

`offloadStrategy !== "none"` never makes an over-capacity plan feasible by
itself.

## 13. Trace Model

Simulation trace and visualization trace are separate views:

- **Protocol trace:** lossless, stable identities, source sequence, exact state
  transitions; accepted by replay/invariant checkers.
- **Visualization trace:** may aggregate or sample for size; never used as
  conformance evidence.
- **Decision trace:** seed plus event/scheduler choices required to replay a
  failure.

JSON output uses an explicit schema and contract revision. Unknown revisions,
missing events, sequence gaps, and non-finite numbers are errors.

## 14. Simulation Modes and Product Surfaces

The historical designs identified useful product behavior that remains in
scope, but all modes use the same validated scenario and core semantics.

### 14.1 Modes

| Mode | Purpose | Output |
|---|---|---|
| Static | Immediate fit and placement screening | Capacity report, invalid constraints, heuristic bottleneck |
| Protocol | Deterministic bounded state/schedule campaign | Lossless protocol and decision traces, invariant report |
| Steady state | Warm-up plus long-run workload dynamics | Aggregated latency, utilization, cache and speculation metrics |
| Detailed trace | Debug one execution/request | Full resource, memory, DAG, communication, and checkpoint timeline |
| Compare/search | Evaluate scenarios with identical workload seeds | Ranked deltas with confidence/provenance, never an unexplained “optimal” answer |

Detailed mode is the source for visualization. Steady-state mode may aggregate
events only after protocol checks have consumed the lossless stream.

### 14.2 Inputs

CLI/browser inputs use versioned YAML or JSON and may reference:

- a built-in or custom device/topology composition;
- a manual, ONNX-extracted, or package-metadata model profile;
- an onnx-genai `FrozenPlan`/placement export when available;
- quantization and tensor wire formats;
- batch, sequence, TP/PP/EP/DP, cache, offload, and pressure policy;
- speculative proposer family, width, state groups, and acceptance source;
- simulation mode, seed, warm-up, event bound, and trace policy; and
- calibration datasets with provenance.

ONNX parsing extracts graph structure, shapes, dtypes, external-data extents,
operator profiles, and runtime metadata. It does not infer measured throughput.

Example:

```yaml
schema_version: 1
hardware:
  preset: dgx-h200
model:
  preset: deepseek-v2
  quantization: { weights: fp8, kv_cache: fp8, activations: fp16 }
pipeline:
  batch_size: 8
  input_seq_len: 4096
  output_seq_len: 1024
  parallelism: { tp: 4, pp: 1, ep: 2, dp: 1 }
speculative:
  family: mtp
  max_additional_tokens: 4
  kv_mode: proposal_local
  acceptance:
    kind: empirical_by_position
    conditional_match: [0.82, 0.71, 0.60, 0.50]
simulation:
  mode: detailed
  seed: 42
  max_events: 1000000
```

### 14.3 Outputs and Views

Machine output includes scenario hash, schema/contract revisions, confidence
labels, assumptions, protocol/decision traces, snapshots, and summary metrics.

Product views retained from the original design:

- memory timeline per physical domain and category;
- expert hot/warm/cold heatmap with load/evict/prefetch overlays;
- topology graph and bandwidth-flow/Sankey view;
- roofline plot with measured and heuristic ceilings distinguished;
- rank-local DAG/collective timeline;
- pressure ticket and allocation-ledger inspector;
- speculative proposal/verify/accept/rollback timeline and acceptance histogram;
- TTFT, ITL percentiles, throughput, stalls, wasted work, and utilization; and
- side-by-side configuration comparison.

The UI runs core simulation in a Web Worker. Progress and abort are worker
control messages; aborting the UI task does not fabricate a successful protocol
terminal state.

### 14.4 Package Boundaries

```text
packages/core   deterministic simulation, models, traces, checks; zero I/O
packages/cli    config/model/plan loading, reports, trace export
packages/web    worker orchestration and visualization
```

`core` has zero runtime dependencies and runs identically in Node.js and a
browser worker. CLI ONNX parsing may use a separate dependency and converts
into versioned core profile types.

## 15. Implementation Plan

### Phase 0: Static Scaffold

Current repository state:

- topology and model presets;
- static memory/throughput approximation;
- four static-analysis tests.

Known limitation: several presets and formulas are heuristic and lack
provenance.

### Phase 1: Deterministic Protocol Vertical Slice

Deliver:

- deterministic event queue;
- pressure governor with exact ledger/ticket transitions;
- contract-revisioned pressure trace;
- independent replay/invariant checker;
- deterministic race tests;
- corrected design and public exports;
- initial composite speculative transaction semantics with explicit rollback
  protection.

Exit criteria:

- same-time events execute in stable insertion order;
- past/invalid scheduling is rejected;
- every valid governor trace replays;
- deliberate trace mutation is rejected at the shortest prefix;
- capacity is conserved after every event; and
- baseline build/tests pass without network or hardware.

Status: complete. The initial slice had 21 tests across static analysis, event
ordering, pressure protocol/replay, trace mutation, and speculative
checkpoint/restore boundaries. The total suite has since grown to 74 tests.

### Phase 2: FrozenPlan, Communicator, and Topology Composition

- rank-local DAG scheduler;
- per-group submit sequencers across overlapping executions/groups;
- rank-local completion and failure state machine;
- read/write allocation lease registry; and
- independent collective/buffer replay checks;
- composable CPU, discrete, unified, heterogeneous, and multi-node memory
  domains; and
- capability/path validation for every placement and transfer.

Status: in progress. Implemented:

- versioned capability/memory-domain scenario schema;
- valid built-in scenarios for all six required topology families;
- exact reservation ledger with explicit shared-allocation aliases;
- directed direct/staged transfer-path validation;
- FrozenPlan DAG, rank/device, capability, topology-epoch, and physical-buffer
  validation;
- deterministic multi-lane compute/link/collective scheduling;
- communicator sequence validation and cross-execution per-group submission
  ownership;
- transport contention between collectives and point-to-point transfers; and
- independent plan-trace schedule/resource/lease replay;
- explicit pinned staging allocations sized for each staged transfer; and
- terminal success/failure/abort events with rank-local state and post-failure
  quiescence; and
- completion-driven FrozenPlan submission on the deterministic event kernel,
  with causal `submittedAtNs` trace evidence.

Remaining Phase 2 work:

- topology-epoch change and node/link failure injection;
- adversarial campaigns across overlapping executions and communicator groups.

### Phase 3: Speculative, Workload, and Cache Dynamics

- integrate the Phase 1 speculative transaction primitive with event resources,
  acceptance models, traces, and target-only differential checks;
- target-only baseline plus prompt-lookup, draft-model, MTP, EAGLE-3,
  shared-KV, and self-speculative execution;
- composite checkpoint/restore across target, CSA, paged KV, proposer, and
  recurrent state;
- seeded expert routing without replacement;
- byte-capacity expert tiers and asynchronous prefetch;
- KV growth and paging;
- device/link resource contention;
- calibrated roofline and collective models; and
- request batching and prefill/decode overlap.

Status: in progress. The first executable slice supports deterministic
token-trace replay and seeded conditional first-mismatch acceptance, drives the
composite checkpoint/restore transaction until an exact output budget is
committed, checks final logical-length parity against target-only execution,
and reports accepted-prefix, position acceptance, correction/bonus, rejected
draft, and committed-token-per-target-forward metrics. Paged target KV now
models exact byte capacity, stable non-reused physical page identities,
checkpoint-relative accepted-prefix restore, logical tail masking, allocation
and release metrics, and independent trace replay; the speculative workload
checks that its final KV length matches committed target state. Expert-cache
workloads now model seeded weighted routing without replacement, exact
hot/warm byte capacities and reservations, deterministic LRU eviction,
asynchronous initial prefetch, stalls, metrics, and independent replay.
Timing/resource plans, family-specific eligibility/state, adaptive prefetch
policy, request batching, and differential token-value traces remain.

### Phase 4: Product Surfaces

- CLI scenario validation, run, compare, and trace export;
- browser worker API with bounded progress/cancellation;
- topology, memory, protocol, and latency views; and
- calibration/import tooling.

The CLI and web UI consume core APIs; they do not own simulation semantics.

Status: in progress. The initial CLI lists presets, materializes built-in
scenarios and exact memory ledgers, validates scenario YAML/JSON, runs the
legacy static-analysis examples, and executes speculative workload configs.
It also executes exact-capacity expert-cache workload configs.
Compare/search, FrozenPlan file execution/export, ONNX import, progress/abort
control, and the browser worker/UI remain.

## 16. Testing and Delivery Gates

Every core change runs:

1. TypeScript strict build.
2. Unit and property-style tests with fixed seeds.
3. Protocol trace replay and invariant checks.
4. Determinism test: identical input produces byte-identical protocol and
   decision traces.
5. Negative test: one invalid transition/event is rejected.
6. Static checks for finite, safe integer byte/time inputs.
7. Differential target-only versus speculative committed-token/state parity.
8. Scenario matrix coverage for all six required topology families.

Protocol tests are never retried to green. A failing schedule must print its
seed and decision trace.

Nightly/extended validation adds larger schedules, cross-group execution,
backend calibration datasets, and differential comparison with onnx-genai
traces.

## 17. Feasibility Assessment

The project is feasible if developed in the order above. A browser-compatible
deterministic simulator can model protocol ownership and resource contention
well. It can also provide useful relative performance comparisons after
calibration.

The project is not feasible as a universal latency predictor based only on
model parameter counts and peak hardware specifications. Accuracy depends on
explicit plans, measured coefficients, topology contention, and evidence
provenance. Those limits are part of the output contract, not documentation
fine print.
