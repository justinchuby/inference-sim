# inference-sim

A deterministic simulator for LLM inference placement, memory protocols, and
resource scheduling.

The project is intentionally split between exact protocol checks and calibrated
performance estimates. It can prove ledger and state-transition properties; it
does not claim hardware-accurate latency without calibration data.

## Current Status

Phase 1 is complete in `@inference-sim/core`:

- heuristic static memory and roofline analysis;
- deterministic discrete-event scheduling;
- exact host-pressure ticket and allocation accounting;
- contract-revisioned pressure traces;
- independent trace replay with invariant checks;
- composite speculative checkpoint/restore transaction semantics.

Phase 2 is in progress:

- capability-based scenarios for all six required topology families;
- exact physical memory-domain ledgers and shared-allocation aliases;
- FrozenPlan DAG validation and deterministic resource scheduling;
- collective ordering, transport-link contention, and allocation leases; and
- independent plan-trace replay with rank-local success/failure/abort
  quiescence.

Phase 3 has an initial speculative workload slice:

- token-trace replay or seeded conditional first-mismatch acceptance;
- exact output-budget and target-only final-length parity;
- accepted-prefix, correction/bonus, rejection, and efficiency metrics; and
- composite target/proposer state rollback on every iteration;
- exact-capacity paged KV allocation with checkpoint-relative rollback,
  non-reused physical page identities, and independent trace replay; and
- speculative KV high-water, allocation, release, and final-reservation
  metrics.
- byte-capacity hot/warm expert caches with weighted routing without
  replacement, deterministic LRU eviction, asynchronous prefetch, and
  independent trace replay.

The initial CLI and browser dashboard are implemented. The dashboard runs core
simulation in a cancellable Web Worker and exposes topology selection,
speculative and expert-cache controls, memory/caching charts, and a recent
event inspector.

## Direction

The simulation model is being built around the onnx-genai memory and
distributed-runtime contracts. Planned scenarios cover:

- CPU-only;
- discrete GPU plus CPU;
- multi-GPU;
- GPU plus NPU;
- unified memory; and
- multi-node and heterogeneous execution.

Speculative decoding is a first-class workload model. It will cover
prompt-lookup, draft-model, MTP, EAGLE-3, shared-KV, and self-speculative
proposers with target-authoritative verification and composite checkpoint
restore.

## Development

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm test

pnpm sim presets
pnpm sim scenario gpu-npu
pnpm sim static examples/mixtral-dgx-h100.yaml
pnpm sim speculative examples/speculative-mtp.yaml
pnpm sim expert-cache examples/expert-cache.yaml
pnpm dev:web
```

See [docs/DESIGN.md](docs/DESIGN.md) for contracts, scope, confidence classes,
device semantics, speculative execution, and delivery gates.

## License

MIT
