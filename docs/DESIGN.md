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

Fault observation is time-anchored. At the same nanosecond a fault is processed
before completion-driven submissions. A non-success terminal records the typed
device/link/epoch fault, exact observation time, and the complete ordered set
of unsubmitted step IDs. Replay rejects any step submitted at or after the
fault and any ready step omitted before it.

A communicator execution cancelled before its first submission is skipped
without poisoning the epoch. Partial submission poisons the epoch and
propagates abort through overlapping communicator queues. New execution
registration remains closed until surviving old-epoch executions drain and the
sequencer advances to a newer topology epoch.

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

| Family | onnx-genai status | Proposal prefix | State/cost modeled |
|---|---|---|---|
| `prompt_lookup` | Current | None | Host CPU lookup/search; no proposer KV |
| `draft_model` | Current | None | Separate model compute plus committed-prefix draft KV and rewind |
| `mtp` | Current | One guaranteed target token | One target hidden seed, iterative sidecar compute, proposal-local KV and recurrent state |
| `eagle3` | Current | One guaranteed target token | Three target hidden taps, fused recurrent state, proposal-local sidecar KV |
| `shared_kv` | Current | One guaranteed target token | One target hidden seed, assistant compute, borrowed target-KV read leases, no proposer KV |
| `self_speculative` | Design-only | None | Early-exit target layers and completion by remaining layers |

The simulator validates runtime eligibility such as proposal availability,
greedy/temperature-zero restrictions, grammar incompatibility, required target
hidden outputs, and declared KV lifetime. Unsupported combinations fail the
scenario; they do not silently fall back to target-only timing.

`self_speculative` is not a current `onnx_genai_engine::SpeculativeMode`.
Scenarios must opt into its design-only contract and declare a valid
`1 <= early_exit_layer < target_layer_count` split. Results retain that support
classification so projected behavior cannot be confused with released runtime
support.

The revisioned core family contract is the only source used by the CLI,
browser, transaction simulator, and topology compiler. State groups have one
of three lifetimes:

- `committed_prefix`: target or draft KV restored to checkpoint plus accepted
  offset;
- `proposal_local`: sidecar/recurrent state cleared after each verification;
- `borrowed`: a target-owned shared-KV lease with no proposer allocation.

### 9.2 Verification Transaction

For target-coupled MTP, EAGLE-3, and shared-KV, one linear speculative
iteration is:

```text
target base step
  -> guaranteed target token and proposer seed
  -> capture composite checkpoint before speculative state mutation
  -> draft up to the configured additional width
  -> one authoritative target verification forward
  -> accept longest matching prefix
  -> restore every target/proposer state stream to the accepted draft prefix
  -> ingest and commit the target-authoritative correction token on mismatch,
     a bonus token on full acceptance when output budget remains, or no extra
     token when a fully accepted proposal exactly fills the output tail
```

Proposal width is family-specific and follows onnx-genai:

```text
prompt_lookup / draft_model / self_speculative:
  proposal_width = additional_draft_tokens

mtp / eagle3 / shared_kv:
  proposal_width = 1 guaranteed_target_token + additional_draft_tokens
```

`max_additional_tokens`, replayed accepted-prefix counts, conditional
probabilities, and proposer-local sidecar capacity all use the
`additional_draft_tokens` coordinate. Target verification, target rollback,
paged-KV candidate state, and public proposal statistics use the full
`proposal_width` coordinate. The guaranteed prefix is target-authored and
cannot be rejected or charged as proposer compute.

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
A zero-token post-restore append is legal only for a non-empty, fully accepted
output tail.

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
resources. Prompt lookup uses a host CPU rank and `lookup` capability. A
separate draft model uses a declared draft-capable placement. MTP, EAGLE-3,
shared-KV, and self-speculative work is target-coupled and prefers a
draft-capable target placement. Family cost scales are explicit heuristic
coefficients and remain provenance-labeled; they are not throughput claims.
Draft and target batches may differ. Shared-KV proposers hold read leases on
target allocations; separate drafts own their own KV.

Verification can use a multi-token target forward only when the selected
backend capability declares it. Rollback cost distinguishes:

- cursor-only logical restore;
- bounded device-to-device state snapshot restore;
- paged-table mutation;
- sidecar reset; and
- recomputation, which is never assumed free.

Distributed target verification emits normal plan communication steps and
therefore obeys per-group collective ordering and buffer leases.

The heuristic topology cost separates one fixed device-kind invocation cost
from incremental per-token work. A multi-token verification therefore
amortizes forward launch/synchronization cost without making its additional
token work free. Both terms retain heuristic provenance until calibrated.

### 9.6 Continuous Serving Composition

Each decoding request owns its speculative transaction and acceptance stream.
Conditional acceptance is keyed by request, committed output coordinate, and
draft position so changing topology timing cannot silently assign another
request or token's random outcome. Replay acceptance is explicitly keyed by
request ID.

A serving decode batch records, per request:

- guaranteed-prefix, additional-draft, full-proposal, and accepted counts;
- target verification width;
- correction, bonus, accepted-tail, or target-only outcome; and
- committed output count.

Admission reserves the worst live target state for the batch, including
candidate drafts and the target verification row that may produce a
full-acceptance bonus token. Batch completion restores each request's composite
checkpoint, commits its accepted prefix plus a correction or budget-permitted
bonus, emits the committed burst at verification completion, and releases
terminal request state. A fully accepted output tail commits without inventing
a bonus. Independent serving replay re-derives acceptance, scheduling,
transient KV, and transaction results.

### 9.7 Correctness Invariants and Metrics

After every iteration:

- committed output equals the target-authoritative path for the modeled
  acceptance rule;
- all `committed_prefix` state groups have the same committed logical prefix;
- `proposal_local` state and borrowed leases return to zero at the iteration
  boundary;
- rejected drafts own no live allocation or lease;
- checkpoint identity cannot be reused across generation/topology epochs; and
- logical/capacity counters remain conserved after restore.

Reported metrics include accepted-prefix histogram, acceptance by position,
correction/bonus counts, effective committed tokens per target forward,
proposer/verification/rollback time, rejected compute and communication,
target/proposer KV high-water marks, and speedup versus a target-only simulation
using the same resource model.

### 9.8 Expert Routing and Cache Tiers

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

### 9.8 Continuous Serving and Chunked Prefill

Serving workloads declare request identity, arrival time, prompt/output token
counts, and priority. The scheduler is deterministic and decode-first:

1. all arrivals at the same nanosecond pass a dispatch barrier before
   selection;
2. active decode sequences reserve one token each in priority/arrival/id order;
3. remaining sequence and token slots admit chunked prefill;
4. every batch reserves transient KV growth before it starts; and
5. completed requests release their full live KV extent atomically.

The final prefill chunk emits the first output token from its logits without
adding that token to KV. Each later decode step processes the previous output
token, appends one KV position, and emits the next token. A request therefore
peaks at `prompt_tokens + output_tokens - 1` KV positions.

Arrivals may occur while a non-preemptive batch is running and become eligible
at the next dispatch. A lossless trace records arrivals, exact batch
membership, prefill slices, decode participants, duration, token source,
completion, and KV counters. Independent replay re-derives every scheduler
decision and rejects the first mutation.

Each dynamic batch compiles into the same topology-aware FrozenPlan path used
by other workloads. The current heuristic treats prefill and decode as linear
token work with the selected topology coefficients; calibration will replace
that fallback with shape-regime measurements. Metrics include TTFT and ITL
average/p50/p95, request latency, throughput, sequence/token batch utilization,
idle/service time, and KV high water.

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

### 11.4 Calibration Import Contract

Topology cost calibration uses a revisioned dataset rather than unversioned
coefficient overrides. A dataset records:

- evidence kind (`measured` or `synthetic`), source, software stack, model
  artifact, and measurement date for measured evidence;
- the exact scenario IDs to which the operator measurements may be applied;
- an explicit CPU, GPU, and NPU class label so a measurement from one product
  is not silently generalized to every device of the same kind;
- repeated duration observations for invocation, attention, FFN, draft, and
  lookup work;
- the work-item regime for every observation point;
- activation, collective, and cold-load model constants; and
- minimum sample count plus normalized-RMSE and P95-relative-error gates.

`work_items` is the unsharded token work presented to the topology model:

```text
work_items = token_width * batch_size * max(1, active_experts)
duration   = invocation_overhead + ns_per_work_item * work_items / shard_degree
```

Each device kind requires an invocation observation at zero work items and at
least two distinct positive work-item points for every compute capability.
Every point retains at least three repeated samples. Invocation overhead is the
median invocation duration. Capability slopes are deterministic least-squares
fits against that fixed overhead. The importer reports the fitted coefficient,
sample count, observed range, normalized RMSE, and P95 relative error for all
15 device/capability groups.

Import fails closed on incomplete coverage, duplicate identities, invalid
provenance, unsafe integer time values, non-positive fitted coefficients, or a
quality gate violation. The observed min/max work-item range is embedded in
the cost model; execution outside that interpolation range is rejected rather
than extrapolated under a `calibrated` label. A stable dataset fingerprint is
included for replay and result attribution. It identifies normalized dataset
content but is not a cryptographic integrity signature.

Synthetic datasets exercise the import path but remain `heuristic`. Measured
operator coefficients may make the cost model `calibrated`; an end-to-end
latency result is `calibrated` only when its device, memory-domain, and link
evidence is also calibrated. The weakest performance evidence determines the
result label; the current conservative aggregation includes all performance
evidence in the applicable scenario. Built-in topology presets remain
heuristic, so importing measured kernel timings alone does not turn their
latency rankings into hardware predictions.

The current fitted model is deliberately linear and device-kind scoped. It
does not yet fit roofline knees, sequence-length effects, collective curves,
cache-tier distributions, or per-product device overrides. Those require a
later calibration revision rather than backward-incompatible interpretation
of revision 1 data.

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
Fault traces additionally prove submission-prefix completeness, fault-first
same-time ordering, rank-local failure/abort state, and quiescence.

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
checkpoint/restore boundaries. The total suite has since grown beyond 100
tests.

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
  with causal `submittedAtNs` trace evidence; and
- topology-aware workload compilation into device compute, directed transfer,
  pipeline, and tensor-collective steps, followed by independent trace replay
  and resource-utilization accounting;
- time-anchored device, link, and topology-epoch fault injection with typed
  terminal evidence, submission closure, in-flight quiescence, and independent
  replay;
- deterministic fault campaigns over every device/link used by a plan plus an
  epoch-change case; and
- communicator abort propagation across overlapping groups, poisoned-epoch
  admission closure, and explicit epoch advance.

Remaining Phase 2 work:

- seeded large-scale campaigns that run multiple FrozenPlan executors
  concurrently through shared communicator sequencers;
- node-wide correlated failure and recovery/replan policy.

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

Status: in progress. The executable slice supports deterministic
accepted-prefix replay and seeded conditional first-mismatch acceptance, drives the
composite checkpoint/restore transaction until an exact output budget is
committed, checks final logical-length parity against target-only execution,
and reports accepted-prefix, position acceptance, correction/bonus/accepted-tail,
rejected draft, and committed-token-per-target-forward metrics. Paged target KV now
models exact byte capacity, stable non-reused physical page identities,
checkpoint-relative accepted-prefix restore, logical tail masking, allocation
and release metrics, and independent trace replay; the speculative workload
checks that its final KV length matches committed target state. Expert-cache
workloads now model seeded weighted routing without replacement, exact
hot/warm byte capacities and reservations, deterministic LRU eviction,
asynchronous initial prefetch, stalls, metrics, and independent replay.
Speculative and expert-cache logical traces now compile into FrozenPlan
resources across all six required topology families. Link duration uses each
declared directed link's latency and bandwidth; compute coefficients are
explicitly heuristic and provenance-tagged. All six proposer families use
revisioned family-specific eligibility, state lifetime, execution placement,
and cost contracts. A 6 proposer x 6 device-topology matrix executes and
replays each profile with target-only final-state differential checks.
Continuous serving now models arrival-time dispatch, decode-first batching,
chunked prefill, exact KV admission/release, TTFT/ITL, and per-batch topology
execution with independent replay. It composes all six proposer contracts with
per-request deterministic acceptance, composite checkpoint transactions,
candidate-state KV admission, multi-token burst emission, and a 6 proposer x 6
topology execution matrix. Revisioned calibration import now preserves repeated
operator observations, provenance, applicability, quality diagnostics, and
valid interpolation ranges, with fail-closed execution outside those ranges.
Backend trace collection, measured transport/collective calibration, adaptive
prefetch policy, and differential token-value traces remain; self-speculative
remains design-only.
The same serving workload can also execute across all six topology presets and
produce a deterministic latency ranking with per-run replay evidence.

### Phase 4: Product Surfaces

- CLI scenario validation, run, compare, and trace export;
- browser worker API with bounded progress/cancellation;
- topology, memory, protocol, and latency views; and
- calibration/import tooling.

The CLI and web UI consume core APIs; they do not own simulation semantics.

Status: in progress. The initial CLI lists presets, materializes built-in
scenarios and exact memory ledgers, validates scenario YAML/JSON, runs the
legacy static-analysis examples, and executes speculative workload configs.
It also executes exact-capacity expert-cache workload configs, compiles
target-only/speculative/expert-cache workloads onto a selected topology, and
compares one workload deterministically across all six presets.
The `serving` command executes arrival-driven target-only or speculative
continuous batching on a selected topology and reports request timing,
scheduler trace/replay, accepted/rejected draft work, batch work, and topology
operation summaries.
The `serving-compare` command runs that same request and acceptance
configuration across all six topology presets and reports a stable ranked
summary without duplicating six full traces in CLI output. The `calibrate`
command validates and fits revisioned YAML/JSON datasets; topology and serving
commands accept an optional calibration path and reject scenario or
work-item-range mismatches.
The `fault-campaign` command compiles that workload, executes and replays a
successful baseline, then injects deterministic mid-operation faults into each
used device/link and the next topology epoch.
The initial React browser workbench uses shadcn/Radix controls and Recharts,
runs bounded core simulations in a dedicated Worker, terminates that Worker on
cancel, lazy-loads visualization code, and presents topology selection,
continuous-serving/speculative-family/expert-cache controls, request TTFT/ITL,
heuristic modeled latency and throughput, memory, acceptance, cache, and resource
utilization charts, and recent request/iteration/route inspection. Serving
controls select target-only or any proposer family plus width and acceptance.
The browser can compare all six serving topologies with a ranked latency chart,
fastest-topology detail view, and comparison inspector. General configuration
search, FrozenPlan file execution/export, ONNX import, trace export, and richer
progress phases remain. Calibration YAML/JSON import shares the core parser and
fit contract with the CLI, enforces a 1 MiB input limit, refits in the Worker,
and reports the dataset fingerprint and fit diagnostics in the result view.

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
