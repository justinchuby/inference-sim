# Inference Simulator Design

**Status:** executable design, Phases 1-2 complete; Phases 3-4 in progress
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

Collective steps declare an immutable communicator group/sequence, algorithm,
and the directed transport links they reserve. Plan contract revision 4 carries
the algorithm into execution evidence, and replay rejects relabeling an
`all_reduce_ring` as `all_to_all_v` or vice versa. Group ordering alone is not
a bandwidth model; collectives and point-to-point transfers contend on the
same link lanes.

A node failure is a correlated fault derived from structured `nodeId`
membership, never from resource-name prefixes. Every participating rank whose
device belongs to the failed node becomes failed at observation time. Surviving
participants abort, new plan submission closes, and already submitted
device/transport work quiesces before terminalization. Replay independently
re-derives failed ranks from scenario membership.

### 8.3 Buffer Ownership

Every operation registers complete read and write `PhysicalAllocationId` lease
sets before enqueue:

- read/read aliasing is legal;
- a writer conflicts with all other readers and writers;
- in-place operations hold the allocation exclusively;
- dropping an observation handle does not release backend ownership; and
- free/reuse occurs only after terminal success or abort quiescence.

Topology compilation also inserts allocation hazard dependencies in source-plan
order. A read waits for the last writer; a write waits for the last writer and
all readers since that write. Read/read remains parallel. This RAW/WAR/WAW
closure is required even when otherwise independent owner transfers alias one
host staging allocation; resource-lane serialization alone does not establish
buffer lifetime.

### 8.4 Concurrent Plan Campaigns

Multiple valid `FrozenPlan` instances may be admitted into one deterministic
executor. Admission order is a contiguous sequence ordered first by arrival
time and then by an explicit tie-break. At an arrival boundary, completions and
their newly ready submissions are processed before the new admission batch.
Ready work is then selected by admission order and source-plan step order.

All executions share the scenario's actual compute lanes, directed link lanes,
collective lanes, communicator sequencers, and physical-allocation lease
registry. A repeated `PhysicalAllocationId` still names one physical
reservation: it is never silently namespaced by execution. Consequently,
read/read access may overlap while conflicting KV, workspace, checkpoint, or
sidecar access is lease-serialized even if the operations use otherwise
different devices or links. Modeling independent per-execution storage requires
the scenario to declare distinct physical allocations and charge their bytes.

The seeded concurrent campaign clones one plan's execution envelope but
intentionally retains its physical allocation identities. It is a stress test
for shared scheduler, communicator, resource, and lease behavior; it is not a
request-throughput shortcut and does not create free KV capacity. Continuous
request throughput remains the responsibility of the paged-KV serving model.

The concurrent trace records canonical admissions, a global operation
submission sequence with execution-local source sequences, and one terminal per
execution. Independent replay reconstructs dependency readiness, communicator
ownership release, resource-lane choice, lease-constrained start time, rank
completion, and terminal order. It rejects unexplained submission delay,
collective overtaking, resource reassignment, lease overlap, and terminal
mutation.

A concurrent request may bind selected steps to absolute not-before
timestamps. Such a step becomes submit-eligible only after both its release
time and all DAG dependencies. The release is an event, not an advance
reservation, so unconstrained work may still use the resource before that
time. Replay receives the same request constraint and rejects early or
unexplained-late submission.

Admissions, operations, terminals, and delayed step releases all consume the
scenario `maxEvents` budget; manual admission cannot bypass the runtime/trace
size guard.

A concurrent node fault requires every target execution to have been admitted
and still be active before the fault timestamp. The fault atomically closes
submission for the entire old epoch. Only the one shared global schedule prefix
submitted strictly before fault observation may quiesce; every participating
execution receives a node-derived failed/aborted rank terminal. Replay checks
the global prefix, rejects any operation submitted at or after the fault, and
rejects any dependency- and communicator-ready operation omitted before it.

### 8.5 Node Failover and Replan

Failover is an explicit two-epoch transaction. A node fault must interrupt the
old plan, and the complete old execution must quiesce before recovery
admission. Recovery requires a separately validated scenario and `FrozenPlan`
with a strictly newer matching topology epoch and a new execution identity.
The failover policy requires the failed node's devices and memory domains to be
absent from that recovery scenario; it never edits or resumes the old plan.

The handoff records the failed node and execution, old/new epochs, fault time,
old-execution quiescence, and recovery admission. Independent replay validates
both plan traces and reconstructs the handoff and total wall-clock completion.
The explicit failover runner handles one recovery plan. The concurrent
node-failure campaign independently establishes the cluster-wide old-epoch
quiescence coordinate from all active executions; a recovery controller may
admit a separately validated newer-epoch plan only after that coordinate.

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

1. revisioned token-level target/proposer traces;
2. empirical conditional acceptance by draft position and context bucket; or
3. an explicitly heuristic stochastic model with a fixed seed.

A single average acceptance rate is insufficient because accepted-prefix length
is a first-mismatch process. The stochastic fallback provides conditional
`P(match at position i | positions < i matched)`.

The revision-2 token trace binds the evidence to non-empty runtime, target
model, proposer, tokenizer, and generation-config fingerprints plus distinct
target-only and speculative run IDs. It records the exact prompt and
target-only output token IDs. Each iteration records:

```text
proposal_token_ids = the full family-specific proposal
target_token_ids   = only tokens actually selected by the target loop:
                     through the first mismatch, accepted proposal rows plus
                     the bonus row, or accepted proposal rows at the tail
```

The oracle computes the longest equal prefix. A mismatch commits that prefix
plus `target_token_ids[accepted]` as the correction. Target rows after the
first mismatch are counterfactual rejected-path evidence and are forbidden.
Full acceptance commits
the proposal and, only when output budget remains, the final target row as a
bonus. A fully accepted tail does not commit the unused final row. For
target-coupled families, proposal position zero must match target position zero
or the trace is structurally invalid.

The trace importer fails closed on unknown fields, unsupported revisions,
duplicate iteration IDs, missing or counterfactual target rows, over-wide or
over-budget proposals, rejected guaranteed prefixes, missing/trailing iterations, and
unbound provenance. It then independently replays the derived widths and
accepted counts through the composite state transaction and requires
token-decision/state-decision parity at every iteration. Token mismatch against
the target-only output is reported separately from malformed evidence.

Runtime evidence is captured as two independent revision-1 artifacts: one
target-only run and one speculative run. Both must complete by the controlled
`max_tokens` condition and bind identical runtime, model, tokenizer,
generation-config, prompt, and output-length coordinates. The speculative
artifact records each iteration's output offset, proposal, actually selected
target tokens, and runtime-claimed commits. Import independently derives every
commit, rejects provenance or terminal-count mismatches, compares the derived
stream with the speculative run output, and finally compares that output with
the target-only run. EOS and stop-sequence truncation are deliberately outside
revision 1 rather than being silently misclassified as accepted tails.

This proves deterministic selected-token and logical-state parity for the
captured configuration. It does not prove logit equality, sampling-distribution
equality, numerical kernel equivalence, or physical cache-byte equality.

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

Revision-4 cache execution separates route admission from route access.
`beginTokenRoute()` freezes the seeded expert decision, reserves any required
demand loads, and returns stable load IDs. The caller may retime those pending
loads from physical evidence before `completeTokenRoute()` advances cache time
and emits access. Only one route transaction may be active; prefetch and a
second route are rejected until it completes. The original `processToken()`
entry point composes both phases with configured logical latency for standalone
workloads.

Adaptive prefetch is a bounded warm-tier feedback policy, not a future-route
oracle. After configured token intervals, it ranks only previously completed
route observations by frequency, then recency, then expert ID. Candidates
below the minimum observation count, already warm, already pending to warm, or
outside the exact reservation budget are excluded. Selection is capped per
decision and never reserves hot-tier bytes. Each policy decision is recorded
before its adaptive prefetch request; replay reconstructs the observation
history and rejects changed, omitted, or relabeled decisions. This policy can
reduce future cold misses, but its copies, evictions, and bandwidth remain
fully charged in the cache protocol's latency and byte ledger.

Current scenario contract revision 4 retains revision 3's explicit
cold-storage domain on every preset node, local CPU endpoint, authoritative
expert-backing allocation, warm-cache allocation, directed storage-read link,
and device/unified hot-expert cache allocation on every FFN placement. Revision
4 changes physical routing from minimum hop count to a deterministic
message-size-aware minimum-duration path:

```text
declared_route_duration(bytes) =
  sum(link.latency_ns + ceil(bytes / link.bandwidth_bytes_per_sec * 1e9))
```

The route is directed, never revisits a memory domain, preserves the exact
ordered link IDs, and breaks equal-duration ties by that ordered link-ID tuple.
A transfer requiring pinned staging must traverse a pinned memory domain as a
true intermediate; the source, target, or a cycle through either endpoint
cannot satisfy the requirement. Invalid byte extents, unsafe durations, and
unreachable constraints fail closed. Because the selected route can change
with message size, route identity is part of the scenario contract rather than
a presentation detail.

Expert parallelism uses an explicit, replayable placement contract rather than
an expert-ID hash or an even division of aggregate work. The contract carries
the ordered expert universe and either `contiguous` or `round_robin`
assignment. FFN placements are ordered by their immutable communicator rank.
For expert index `i`, `N` experts, and `EP` ranks:

```text
contiguous owner  = floor(i * EP / N)
round_robin owner = i mod EP
```

The current expert-cache and serving adapters select `contiguous`, matching the
recommended initial onnx-genai static placement. Workload profiles, the CLI
`placement_strategy`, and the React workbench's shadcn/Radix selector may
explicitly request `round_robin`; all three feed the same core contract.
Routed profiles separately declare `expertTokenPlacement=round_robin`.
Assignment order is token-major, so token ordinal `t` originates on
communicator rank `t mod EP`; each selected expert's destination is resolved
through the expert-owner contract. This produces an explicit source-to-owner
byte matrix for dispatch and its transpose for gather. Owner-local matrix
entries remain dependency-visible but consume no physical link.
Missing experts, duplicate expert IDs, incomplete per-token top-K assignments,
duplicate assignments within one token, an FFN-placement count different from
`EP`, or ambiguous communicator membership are fatal input errors.

The expert-cache initialization trace is authoritative for each expert's byte
extent. Topology projection resolves route and load identities through that
table; it does not accept a second uniform-size scalar or infer one expert's
size from another. Variable-size experts therefore preserve their exact
warm/cold transfer bytes through core, CLI, and Web execution.

Expert-cache trace revision 5 makes tier ownership explicit. A standalone
cache without a topology has one implicit `default` partition per tier. A
topology-bound workload instead derives one hot partition per non-empty EP
owner and one warm partition per node that owns experts from the same
expert-placement contract used by
physical execution. `hot_capacity_bytes` is the capacity limit for each owner;
`warm_capacity_bytes` is the limit for each node. A partition is capped at the
total bytes of experts it can own, and the snapshot's top-level capacities are
the checked sums of those effective partitions. The snapshot also reports
resident, reserved, capacity, and LRU-ordered resident IDs per partition.

Admission, pending reservations, victim selection, initial residency, and
adaptive-prefetch budgeting are checked independently in each partition. An
owner's miss can evict only that owner's LRU expert. Trace eviction evidence
includes the partition identity, and replay rejects a changed owner before a
later load can hide the corruption. Every expert must appear exactly once in
each tier's partition table, partition capacities must sum to the declared
aggregate, and malformed or prepartitioned topology-serving input fails
closed. This prevents a global logical LRU from evicting state on an unrelated
GPU and allows independent device caches to use their real aggregate capacity.

Demand loads, background prefetch, and routed FFN compute resolve the same
expert to the same owner. Under `EP > 1`, the complete expert bytes move only
to that owner's hot cache and only owners receiving routed assignments emit
FFN compute. The AllToAllV gather reads only those active owner workspaces;
non-owning ranks participate in the collective protocol with zero routed work.
Under `EP == 1`, an explicitly tensor-sharded FFN may continue to divide an
expert load across its participating FFN placements.

Explicit routed profiles keep warm and cold traffic on separate physical
paths. Warm bytes move from the owner node's warm allocation; cold bytes move
from its backing allocation through the storage path. The heuristic
`coldLoadByteMultiplier` is not applied when those explicit routes exist,
because doing so would both double-charge the storage cost and mislabel cold
traffic as a warm-cache read.

The topology compiler extracts validated prefetch `load_start` events and emits
background FrozenPlan transfers after their causal route. These transfers
occupy storage link lanes, write the warm pool, may overlap subsequent compute
where allocation leases permit, and are independently replayed. Writes are
ordered per warm domain, while independent owner nodes may prefetch
concurrently. A single expert prefetch targets its owner's node; it is not
replicated or divided across all nodes. The projection preserves
expert-to-prefetch provenance, so a later warm demand cannot read the warm pool
until its producing storage copy has completed on that node. The current lease
model is allocation-granular rather than cache-slot/range-granular; unrelated
expert reads and writes in one warm allocation are therefore conservatively
serialized. Completed logical prefetches determine later warm-demand tiers,
while the physical projection charges and enforces the copy.

### 9.9 Continuous Serving and Chunked Prefill

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

Serving may opt into revision-3 stateful expert-cache composition. One cache
instance persists across all batches, and every target token in prefill or
verification receives one seeded top-K route. Draft-only proposer work does
not route through the target MoE unless a future proposer profile explicitly
declares that architecture. Per-batch routed FFN work uses the same AllToAllV
and owner-routed topology path as standalone expert-cache workloads.

The current bounded baseline serializes cache route decisions inside a batch,
while the resulting target work remains batched in one topology plan. Each
route is a two-phase transaction. `beginTokenRoute()` identifies new demand
loads without recording an access. Every new load compiles into its own
topology plan and is admitted to the same absolute-time streaming runtime:

- under expert parallelism, a cold load starts at the owning rank's node-local
  backing storage and reaches only that owner's hot-expert cache through the
  declared storage and device links;
- a warm load starts in the owning rank's node-local warm domain;
- under tensor-sharded `EP == 1` execution, the load may be divided across the
  participating FFN placements;
- CPU-only and unified-memory warm promotion is a same-domain state transition
  and therefore has no fictitious copy plan;
- all new loads from one route are admitted before any of their terminals are
  awaited, preserving resource and lease contention; and
- cold shards that reuse one node-local staging allocation are explicitly
  ordered.

The cache completion of each demand load is retimed from its physical owner
terminal, or from the maximum terminal when an `EP == 1` tensor shard has more
than one destination. `completeTokenRoute()` may record the expert access only
after those events make every required expert hot. The aggregate target plan
then carries routed AllToAllV/owner-FFN work, reads the same owner hot-expert
allocations written by demand terminals, but carries zero demand-load bytes.
It is admitted at cache readiness.

Physical capacity validation is owner-aware. For each EP rank, required hot
bytes are bounded by both its partition capacity and the total bytes of
experts that rank can own. Warm capacity is checked analogously per owner node.
This rejects an undersized owner even when aggregate capacity across unrelated
ranks is sufficient, while avoiding fictitious reservation of unused logical
capacity beyond the complete expert universe. Consequently there is one
authoritative timeline:

```text
route decision
  -> physical demand terminals
  -> cache access
  -> aggregate target-plan admission
  -> foreground terminal

batch_duration = foreground_terminal_time - scheduler_batch_start_time
```

`cacheConstraintNs` remains trace evidence for the interval from scheduler
batch start through serialized route readiness; it is not a second duration
estimate. Demand transfers appear only in their load plans, so the target plan
cannot charge them again. The next batch cannot observe cache state until the
current batch foreground completes. Serving replay reuses the already
materialized batch result and must match both batch work and scheduler start
time, while the full expert-cache trace is replayed independently from
initialization through final residency.

Topology results identify the final foreground workload step separately from
total plan drain. `foregroundDurationNs` is taken from that step's replayed
trace event; `backgroundDrainNs` is the remaining quiescence tail. Composed
serving admits demand plans at their route request time and each batch plan at
its cache-ready time in one streaming concurrent-plan runtime. It runs only
until that batch's foreground terminal before returning its service duration.
Resource lanes, physical allocation leases, and collective submit ordering
remain live for older background steps. After the final batch, the runtime
drains and the existing concurrent-plan verifier independently replays every
admission, operation, and terminal.

Serving request completion and resource observation are separate clocks.
`totalDurationNs` ends when the final request completes, while
`resourceObservationNs` extends through any replay-verified physical
background drain. Resource busy time, operation counts, and service time are
derived from the authoritative global operation trace, not by summing
isolated per-batch utilization. Utilization divides by
`resourceObservationNs * capacityLanes` and fails closed above one.

Revision-4 expert-cache traces permit a pending load completion to be retimed
without changing its reservation. In composed serving, each adaptive
warm load is first deferred so later routes in the same aggregate batch cannot
consume a provisional fixed-latency copy. Every step on its physical storage
path receives an absolute not-before constraint equal to the later of its
recorded policy decision time and containing-plan admission. The streaming
runtime releases the step only after both that time and its DAG dependencies
are satisfied; replay receives the same per-request constraints and rejects an
early submission. This prevents a fast
foreground topology from turning an aggregate-batch route into a future
oracle. Once every per-node transfer terminal is submitted, the load is
retimed to their latest completion. If that copy finishes while later routes
in the same aggregate batch are still being evaluated, cache visibility is
conservatively delayed to the current cache clock while
`physicalCompletesAtNs` preserves the actual earlier completion. Missing
terminals and starts before the logical policy decision are protocol errors.
Independent cache replay verifies the provisional load, physical completion,
visibility time, and every retime.

## 10. Device Configuration Coverage

Device configurations are composed from memory domains, compute devices,
access capabilities, and links. Presets are data; core simulation must not
branch on names such as `dgx-h100`.

Parallelism composition is explicit. `cartesian` means TP, PP, EP, and DP
degrees form independent rank axes and therefore multiply. The bounded
`overlap_by_capability` mode permits the same ordered rank group to execute TP
attention and EP routed FFN at different phases; it currently requires equal
TP/EP degree with PP=DP=1. Validation rejects treating that shared-rank overlap
as an independent Cartesian product. The overlap degree is not restricted to
two: compiler, execution, fault, and replay paths cover arbitrary communicator
sizes.

### 10.1 Required Topology Families

| Family | Memory semantics | Required simulation behavior |
|---|---|---|
| CPU-only | Host RAM is the hot/warm domain; storage is distinct | No DeviceGovernor or fake host-device copy |
| Single discrete GPU + CPU | Separate VRAM, host RAM, and storage | Ticketed offload, copy fence, source release after publish |
| Multi-GPU discrete | Per-GPU VRAM plus node-shared host/storage | One HostGovernor, multiple DeviceGovernors, inter-GPU groups |
| GPU + NPU | Separate accelerator domains plus pinned host DMA and storage | Pinned/pageable classes; NPU requests cannot silently degrade |
| Unified memory | One coherent compute ledger plus non-coherent storage | Residency reclassification without copy or double charge |
| Multi-node | Per-node storage/host/device topology plus inter-node links | Cluster execution IDs, communicator groups, topology epoch/failure |

The same composition also covers heterogeneous scenarios listed by onnx-genai:
CUDA+MLX overflow, NPU attention with GPU FFN, multi-vendor GPU via host/RDMA,
and multiple CPU EPs for deterministic testing.

The public parameterized multi-GPU builder materializes 2 through 64 ranks with
per-rank VRAM, placement allocations, PCIe links, DeviceGovernors, and a closed
bidirectional inter-GPU ring. It validates the completed storage/cache tiers
before returning. The browser intentionally exposes only 2, 4, and 8 ranks as
a bounded option set and rejects any other Worker input; the fixed two-rank
`multi-gpu` preset retains its original scenario and link identities for
calibration compatibility.

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
- Cartesian or capability-overlap parallelism semantics correspond to concrete
  participants;
- discrete copies use distinct physical identities until commit;
- unified aliases preserve one physical identity; and
- fixed workspaces, staging, checkpoint snapshots, KV, speculative sidecars,
  and warm/hot expert caches fit their owning domains.

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

For an uncalibrated `N`-rank communicator, immutable communicator order defines
the logical ring. `all_reduce_ring` executes `2(N-1)` neighbor phases with
`ceil(per_rank_extent/N)` bytes per rank and phase. Routed `all_to_all_v`
executes `N-1` pairwise-exchange phases using the exact round-robin
token-source to expert-owner byte matrix. Each non-zero matrix cell resolves
its own message-size-aware physical path; zero and owner-local cells reserve no
link. Gather transposes the dispatch matrix. An aggregate-only internal
fallback distributes bytes deterministically across remote pairs, remains
heuristic, and is not used by validated routed profiles.

Every logical edge resolves to a directed physical path. Per-phase duration is
the greater of the longest path duration and accumulated service on any shared
directed link. The FrozenPlan collective reserves the union of all phase links
for its full duration, which is conservative for contention with other
operations. A missing directed path fails compilation. Imported end-to-end
collective calibration replaces this duration fallback but not the declared
resource reservation. Calibration revision 3 supports all-reduce directly and
AllToAllV by binding each curve to a canonical traffic signature.

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
- repeated transfer and collective observations scoped to an exact scenario,
  ordered link path, participant count, algorithm, optional AllToAllV traffic
  signature, and message size;
- the work-item regime for every observation point;
- activation, collective, and cold-load model constants; and
- minimum sample count plus normalized-RMSE and P95-relative-error gates.

Dense TP and routed EP have different work coordinates:

```text
dense_work_items = token_width * batch_size
dense_duration   = invocation_overhead
                 + ns_per_work_item * dense_work_items / tensor_degree

owner_work_items(owner) = count(routed assignments owned by owner)
owner_ffn_duration       = invocation_overhead
                         + ns_per_work_item * owner_work_items(owner)
```

Routed FFN work is not divided by `EP` after routing. Each assignment is
charged exactly once to its owner, so route skew changes per-rank service time.
The current linear coefficient remains heuristic until observations cover the
owner-work regime.

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

Calibration dataset revision 3 groups communication observations by:

```text
(scenario_id, operation, ordered_link_ids, participant_count, algorithm,
 traffic_signature?)
```

Every group requires at least two distinct positive message-size points and at
least three repeated duration samples per point. Median duration must be
non-decreasing with message size. Execution uses deterministic piecewise-linear
interpolation between adjacent medians. It does not extrapolate and does not
fall back to declared bandwidth while an imported calibration is active. A
missing exact path, changed link order, participant-count change, algorithm
change, or out-of-range byte extent is an execution error. Point-to-point
curves name exactly one directed link and two endpoints. The current topology
compiler identifies dense tensor collectives as `all_reduce_ring`. Routed
expert units on an overlap-capable topology emit one tensor all-reduce followed
by `all_to_all_v` dispatch, owner-local FFN compute, and `all_to_all_v` gather.
AllToAllV observations require
`all_to_all_v_matrix_v1:<matrix-json>`. The matrix uses immutable communicator
rank order, includes local diagonal cells, contains non-negative safe integers,
and is divided by the GCD of all non-zero cells before serialization. This
normalizes proportional traffic shapes across message sizes while preserving
dispatch/gather direction, local fraction, and owner skew. Missing,
dimension-mismatched, malformed, or further-reducible signatures are rejected;
non-AllToAllV observations cannot declare one.
For every communicator size, all-reduce uses the ordered neighbor ring and
AllToAllV enumerates every pairwise phase offset; the plan reserves the stable
union of their directed physical paths. Every supported
algorithm/path/participant-count combination requires its own observations.
This arbitrary-rank phase model and EP-owner behavior is preserved in topology
cost-model revision 9. Revision 9 binds AllToAllV curves to the canonical
traffic signature generated from the actual token-source to expert-owner
matrix. Curve selection must match scenario, operation, ordered links,
participants, algorithm, signature, and byte range. Revision 8 cost models are
rejected.

Synthetic datasets exercise the import path but remain `heuristic`. Measured
operator coefficients may make the cost model `calibrated`; an end-to-end
latency result is `calibrated` only when every performance input it uses is
also calibrated. The weakest performance evidence determines the result label.
The current conservative aggregation includes the applicable cost model plus
scenario device, memory, and link evidence. Built-in topology presets remain
heuristic, so importing measured kernel timings alone does not turn their
latency rankings into hardware predictions.

When revision 3 exact-path communication curves are present, declared link
bandwidth and latency do not determine the selected path's final duration, but
they still determine which physical path is selected for the message size.
Their provenance therefore remains part of end-to-end confidence. The
calibration curve applies only after routing and must exactly match the
selected ordered link path, scenario, operation, participant count, algorithm,
and byte range. The simulator does not search for an alternate calibrated path
when the declared minimum-duration route lacks a curve; it fails closed. A
future calibration contract may model route selection directly, but revision 3
does not.

The current compute fit is deliberately linear and device-kind scoped.
Transport and collective timing is measured-curve scoped, but the model does
not yet fit compute roofline knees, sequence-length effects, cache-tier
distributions, or per-product device overrides. Those require a later
calibration revision rather than weakening revision 3 applicability.

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
Revision-2 `inference-sim/onnx-model` manifests bind the ONNX protobuf and each
referenced external-data file by SHA-256, retain canonical initializer names,
dtypes, dimensions, logical/storage extents, and sorted operator counts, and
normalize only explicitly published architecture fields. External paths must
remain inside the model package and every referenced range must fit the actual
sidecar. Sidecars are streamed for hashing rather than loaded into memory.
Profile readiness lists every missing architecture field; tensor-name pattern
matching is not accepted as architecture evidence. For MoE, readiness also
requires active expert count plus routed and shared expert bytes per layer;
revision 1's dense-only readiness rule is intentionally not accepted as
equivalent evidence.

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

Browser result artifact revision 1 packages the exact dashboard input,
wall-clock-independent summary, and complete mode-specific core evidence in one
JSON envelope. Its contract map names every schema/state-machine revision
needed to interpret the evidence. Canonical key ordering and separate input,
output, and envelope FNV-1a fingerprints over UTF-8 JSON make byte-stable
reproduction and tamper detection explicit. These fingerprints are not
cryptographic signatures, and validating one does not replace replay or
re-execution.
Non-finite numbers, non-JSON objects, cycles, excessive nesting, unknown
envelope fields, and fingerprint mismatches fail closed.
The contract map is execution-path scoped rather than a list of every feature
the dashboard supports. For example, target-only serving does not bind
speculative or expert-cache revisions. Import requires the exact current
applicable contract set, validates the embedded Dashboard configuration, then
re-executes in the Worker and compares input, output, and envelope fingerprints.
A valid old artifact whose output differs from the current implementation is
reported as a deterministic mismatch; a malformed, tampered, oversized, or
contract-incompatible artifact is rejected before execution.

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
- explicit per-node cold-storage domains, backing/warm allocations, and
  directed storage-read links; and
- capability/path validation for every placement and transfer.

Status: complete. Implemented:

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
- capability-overlap TP/EP plans with algorithm-labeled bidirectional
  all-reduce plus AllToAllV dispatch/gather, explicit contiguous/round-robin
  expert ownership, and route-skewed owner-local FFN work, without multiplying
  the same rank group into fictitious compute shards;
- time-anchored device, link, and topology-epoch fault injection with typed
  terminal evidence, submission closure, in-flight quiescence, and independent
  replay;
- deterministic fault campaigns over every device/link used by a plan plus an
  epoch-change case; and
- communicator abort propagation across overlapping groups, poisoned-epoch
  admission closure, and explicit epoch advance; and
- seeded multi-execution campaigns with arrival-ordered admission, shared
  device/link/collective lanes, shared physical-allocation leases, per-group
  communicator ownership, and independent global trace replay; and
- structured node-wide rank failure plus explicit quiesce-before-admit failover
  to a separately validated newer-epoch scenario and replanned execution; and
- atomic node-fault fanout across every active concurrent old-epoch execution,
  with one shared pre-fault schedule prefix, global quiescence, admission
  closure, and independent prefix/terminal replay.

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

Status: in progress. The executable slice supports deterministic revisioned
token-value traces, accepted-prefix replay, and seeded conditional
first-mismatch acceptance, drives the
composite checkpoint/restore transaction until an exact output budget is
committed, checks final logical-length parity against target-only execution,
and reports accepted-prefix, position acceptance, correction/bonus/accepted-tail,
rejected draft, and committed-token-per-target-forward metrics. Paged target KV now
models exact byte capacity, stable non-reused physical page identities,
checkpoint-relative accepted-prefix restore, logical tail masking, allocation
and release metrics, and independent trace replay; the speculative workload
checks that its final KV length matches committed target state. Expert-cache
workloads now model seeded weighted routing without replacement, exact
per-expert byte extents, hot/warm byte capacities and reservations,
deterministic partition-local LRU eviction,
asynchronous initial prefetch, history-driven adaptive warm prefetch, stalls,
metrics, and independent replay. Validated prefetch loads compile into
background storage transfers that overlap compute and serialize only within
the same warm domain. Topology-bound execution derives independent hot
partitions per EP owner and warm partitions per node, with aggregate and
per-partition replay evidence.
Speculative and expert-cache logical traces now compile into FrozenPlan
resources across all six required topology families. Physical routes minimize
declared directed-link latency plus message-size service time with stable
ordered-link tie-breaking and no domain revisits. Default link duration uses
those same declarations; imported revision 3 calibration instead supplies the
selected exact path's transfer or collective duration with fail-closed
identity and message-range checks. Link provenance remains relevant because
the declarations still select the path. Compute coefficients remain
provenance-tagged. Routed expert dispatch and gather use an explicit
round-robin token-source to owner traffic matrix, so network service reflects
expert skew and omits owner-local bytes. Calibration revision 3 canonicalizes
that matrix into the exact curve identity, enabling measured AllToAllV without
collapsing different skew shapes. Routed expert profiles
on multi-GPU and multi-node presets
execute TP attention, bidirectional all-reduce, AllToAllV dispatch, owner-local
FFN, and AllToAllV gather; demand loads and prefetches target the same explicit
owner, and dense profiles do not pay expert collectives. Arbitrary-rank custom
topologies use the same compiler, execution, fault, and replay path; a
four-GPU ring fixture covers dense TP, routed EP, MTP speculative verification,
every participating device, and every used collective link, while an eight-GPU
fixture guards scale independence. All six proposer families use
revisioned family-specific eligibility, state lifetime, execution placement,
and cost contracts. A 6 proposer x 6 device-topology matrix executes and
replays each profile with target-only final-state differential checks.
Continuous serving now models arrival-time dispatch, decode-first batching,
chunked prefill, exact KV admission/release, TTFT/ITL, and per-batch topology
execution with independent replay. Batch duration estimation receives the
authoritative scheduler start time, and replay must reproduce both batch work
and that time coordinate. This is the contract needed for future stateful
cache residency, eviction, and prefetch completion across batch boundaries.
Revision-3 composed serving preserves one expert-cache instance across
batches, routes every target token through the selected top-K experts, emits
routed AllToAllV/FFN plans, and independently replays scheduler, cache, and
global physical traces. Demand loads now compile into separately admitted
warm/cold topology plans; cache access and aggregate target-plan admission are
gated by their physical terminals rather than a second logical latency model.
Adaptive warm prefetch completion is retimed from shared physical
storage-transfer terminals and may remain in flight across batch boundaries.
It composes all six proposer
contracts with
per-request deterministic acceptance, composite checkpoint transactions,
candidate-state KV admission, multi-token burst emission, and a 6 proposer x 6
topology execution matrix. Revisioned calibration import now preserves repeated
operator and communication observations, provenance, applicability, quality
diagnostics, exact transport identity, and valid interpolation ranges, with
fail-closed execution outside those ranges.
Token-value traces now reconstruct correction, bonus, and accepted-tail values,
compare them with a bound target-only run, replay the same decisions through
composite state, and execute a 6 proposer x 6 topology matrix. The companion
onnx-genai integration branch now provides explicit opt-in iteration capture
and an atomic runtime-artifact writer; merging that producer and collecting
real hardware calibration datasets remain.
self-speculative remains design-only.
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
The `onnx-inspect` command decodes standard ONNX protobufs through the current
ONNX 1.20 schema and emits the shared revision-2 model manifest. Optional
onnx-genai fixture manifests, legacy `genai_config.json`, and portable
inference metadata are normalized without changing their evidence strength.
Malformed protobufs, sparse/segmented initializers, unsafe external paths,
truncated sidecars, duplicate identities, inconsistent totals, stale
revisions, and fingerprint mismatches fail closed.
The `onnx-static` command resolves a ready manifest into a `ModelProfile` and
runs the shared static analyzer. Initializer byte and element totals remain
exact inventory; dominant weight dtype selection, non-expert per-layer
distribution, dense attention/FFN splitting, and runtime KV/activation dtypes
are provenance-labeled assumptions. Built-in model presets carry heuristic
provenance too. MoE expert residency uses explicit per-layer units: PP assigns
layers, EP shards routed experts, and shared experts remain replicated across
EP ranks.
The React workbench accepts that same revision-2 JSON manifest rather than
decoding ONNX protobufs on the main thread. Import performs a bounded,
fingerprint-validating parse; the Worker repeats validation before profile
resolution and analysis. ONNX sessions have their own hardware, runtime dtype,
sequence, parallelism, and offload controls and do not masquerade as replayable
serving/speculative dashboard artifacts. The result exposes per-device
weights, expert residency, KV, activation, and free capacity alongside the
manifest inventory and profile assumptions.

Static configuration search is a deterministic Cartesian enumeration of only
the caller-declared finite axes. `maxCandidates` is checked before evaluation;
empty or duplicate axes, duplicate topology identities, non-finite
constraints, empty topologies, and unsafe candidate counts fail closed.
Structurally invalid TP/PP/EP/DP candidates and candidates excluded by fit,
headroom, TTFT, or ITL constraints are counted by reason. Eligible candidates
are ranked by the selected static objective with a stable candidate-identity
tie-break. Output includes declared, evaluated, eligible, returned, and
rejected counts. The ranking inherits the model profile's evidence and the
static analyzer's roofline assumptions; it describes the declared search space
and is not a global optimum claim.
All CLI commands that consume one scenario resolve a shared scenario-target
grammar: a fixed preset name or `multi-gpu-ring-N` for `N=2..64`. Scenario
materialization, workload execution, serving, runtime-capture verification,
fault campaigns, concurrent campaigns, and node operations use that resolver.
The two comparison commands deliberately retain the six fixed topology
families, keeping their ranking population stable.
The `speculative-trace` command verifies a revisioned YAML/JSON token trace and
optionally executes its derived workload on a scenario target. It exits zero on
parity, two on a well-formed token mismatch, and one on malformed evidence or
execution failure.
The `speculative-capture` command binds separate target-only and speculative
runtime artifacts, verifies the runtime's iteration-level commit claims, then
runs the same token, state, and optional topology checks.

The React workbench makes paired runtime capture import the primary evidence
path in the Spec view. The two selected artifacts are role-ordered by their
validated contents, not by filename or selection order. The browser requires
exactly one target-only and one speculative artifact, applies the same strict
revision/provenance/iteration binding as the CLI, and derives the revision-2
token trace only after that binding succeeds. Importing a preassembled token
trace remains a secondary debugging and compatibility path; it is not
equivalent evidence that the runtime independently emitted both sides.

Parsing and an immediate preview provide file-level feedback. The authoritative
run repeats token-value verification, composite state replay, and topology
execution in the Worker. Controls derived from imported evidence replace the
heuristic sliders for that run. Parity and first-mismatch results remain visible
alongside both runtime capture identities rather than being collapsed into a
generic error state. Failed replacement imports preserve the last valid
evidence and report the new failure; they never partially install one member of
a pair.

The `serving` command executes arrival-driven target-only or speculative
continuous batching on a selected topology and reports request timing,
scheduler trace/replay, accepted/rejected draft work, batch work, and topology
operation summaries. An optional revisioned `expert_cache` block composes
persistent routed-expert residency with the same serving and speculative
timeline and reports cache replay evidence.
The `serving-compare` command runs that same request and acceptance
configuration across all six topology presets and reports a stable ranked
summary without duplicating six full traces in CLI output. The `calibrate`
command validates and fits revisioned YAML/JSON datasets; topology and serving
commands accept an optional calibration path and reject scenario or
work-item-range mismatches.
The `fault-campaign` command compiles that workload, executes and replays a
successful baseline, then injects deterministic mid-operation faults into each
used device/link and the next topology epoch.
The `concurrent-campaign` command compiles one workload, generates seeded
arrival-ordered execution envelopes, runs them through shared resources and
communicator sequencers, and independently replays the global trace. Its output
explicitly reports that repeated physical allocation IDs remain shared and
lease-serialized.
The `node-failover` command injects a structured node fault into an old-epoch
plan, waits for terminal quiescence, then admits a separately compiled workload
on a failed-node-free preset at the next topology epoch. Both execution traces
and the handoff are independently replayed.
The `concurrent-node-failure` command applies one node fault to a seeded
multi-execution campaign, closes all old-epoch submission atomically, quiesces
only the already submitted global prefix, and independently replays every
failed execution terminal.
The `plan-export` command compiles one workload into a revision-1
`inference-sim/frozen-plan` artifact. The artifact is self-contained: it embeds
the exact validated scenario and FrozenPlan rather than retaining a preset
name that could resolve differently in a later build. It records the scenario
schema and plan contract revisions plus canonical fingerprints for the
scenario, plan, and complete unsigned envelope. `plan-run` verifies those
boundaries, rejects unknown plan fields, validates the embedded scenario-plan
pair, executes it, and independently replays the resulting trace. A producer
must regenerate the artifact when either contract revision changes; readers do
not silently migrate executable plans.
These FNV fingerprints are deterministic integrity and parity identifiers, not
cryptographic signatures or producer authentication. Deployment tooling must
establish artifact provenance separately when plans cross a trust boundary.
The initial React browser workbench uses shadcn/Radix controls and Recharts,
runs bounded core simulations in a dedicated Worker, terminates that Worker on
cancel, lazy-loads visualization code, and presents topology selection,
continuous-serving/speculative-family/expert-cache controls, request TTFT/ITL,
heuristic modeled latency and throughput, memory, acceptance, cache, and resource
utilization charts, and recent request/iteration/route inspection. Serving
controls select target-only or any proposer family plus width and acceptance,
and use a shadcn/Radix switch to compose the Experts-tab cache parameters into
that serving run.
Multi-GPU runs additionally select 2, 4, or 8 ranks and delegate topology
construction to the validated core builder; the UI does not synthesize
placements or links.
Completed browser runs export a revision-1 deterministic result artifact that
contains the selected configuration, summary, and complete speculative,
expert-cache, serving, or six-topology comparison evidence. The Worker creates
the artifact from the same execution shown on screen. Contract revisions and
canonical fingerprints are owned by the shared core artifact contract;
wall-clock Worker duration is UI-only and excluded from the envelope.
The header import action accepts a bounded revision-1 JSON artifact, restores
its calibration and token-trace inputs when present, re-executes the selected
mode in the Worker, and reports expected/current input, output, and envelope
fingerprints. Invalid envelopes and stale applicable contracts fail closed;
valid historical evidence may complete with an explicit artifact mismatch.
The adjacent FrozenPlan action accepts the self-contained JSON plan artifact
through a distinct Worker request. The Worker repeats strict parsing, executes
the embedded scenario-plan pair, independently replays the plan trace, and
returns bounded operation evidence plus rank terminals. Dashboard sliders are
not applied to an imported plan, and the result view identifies the artifact,
scenario, topology epoch, and all three integrity fingerprints.
The browser can compare all six serving topologies with a ranked latency chart,
fastest-topology detail view, and comparison inspector. Standalone FrozenPlan
file export is available in the CLI and shared core contract, and browser
execution/import is available in the React workbench. ONNX Analyze/Search modes
share the core resolver, analyzer, and bounded search contract. Search scope
controls expand visible finite topology, KV dtype, batch, parallelism, and
offload axes, display the declared count before execution, and report ranked
candidates plus complete rejection accounting. Artifact history management
and richer progress phases remain. Calibration YAML/JSON
import shares the core parser and fit contract with the CLI, enforces a 1 MiB
input limit, refits in the Worker, and reports the dataset fingerprint, compute
diagnostics, and transport-curve diagnostics in the result view.

New browser controls must extend the existing React and shadcn/Radix component
layer. Core placement, routing, replay, and timing semantics remain in
`@inference-sim/core`; the UI may select declared contracts but must not
reimplement owner assignment or simulation state.

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
