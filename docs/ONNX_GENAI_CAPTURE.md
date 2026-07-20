# onnx-genai Runtime Capture

`inference-sim` can verify token and logical-state parity from two independent
`onnx-genai` runs:

1. a target-only baseline; and
2. the speculative configuration under test.

The producer is implemented on the `onnx-genai` branch
`feat/speculative-conformance-capture`:

- `96ad963` adds the explicit `Engine::generate_with_speculative_trace` API;
- `da333a9` adds the `onnx-genai capture` artifact writer.

The branch is intentionally separate from unrelated runtime work. Until it is
merged, build the producer from that branch and consume its JSON with
`inference-sim/main`.

## Evidence Coordinates

Both runs must use the same:

- `runtime_revision`;
- target `model_fingerprint`;
- `tokenizer_fingerprint`;
- `generation_config_fingerprint`;
- prompt; and
- exact max-token output length.

Run IDs must differ. The speculative run additionally binds its engine-reported
family and configured additional width plus a required proposer fingerprint.
For model-free prompt lookup, the proposer fingerprint should identify the
canonical proposer configuration rather than a nonexistent model artifact.

`generation_config_fingerprint` covers target token selection and termination
settings shared by the two runs. It must not include the speculative proposer
mode, which is bound separately. A content-addressed package manifest is
preferred for model and tokenizer fingerprints; directory names and mutable
tags are not sufficient evidence.

Revision 1 deliberately permits only greedy runs ending at `max_tokens`. The
producer disables EOS and stop sequences. Context-length completion, EOS, or a
stop-sequence finish is rejected instead of being mislabeled as an accepted
tail.

## Capture

Build the runtime producer:

```bash
cargo build -p onnx-genai --bin onnx-genai
```

Capture the target-only baseline:

```bash
target/debug/onnx-genai capture \
  --model /path/to/model \
  --output /tmp/target.json \
  --role target-only \
  --id target-run-001 \
  --runtime-revision <exact-commit> \
  --model-fingerprint <content-fingerprint> \
  --tokenizer-fingerprint <content-fingerprint> \
  --generation-config-fingerprint <greedy-max-token-config-fingerprint> \
  --max-new-tokens 32 \
  "prompt"
```

Capture a metadata-configured speculative run:

```bash
target/debug/onnx-genai capture \
  --model /path/to/model \
  --output /tmp/speculative.json \
  --role speculative \
  --id speculative-run-001 \
  --runtime-revision <same-exact-commit> \
  --model-fingerprint <same-content-fingerprint> \
  --tokenizer-fingerprint <same-content-fingerprint> \
  --generation-config-fingerprint <same-config-fingerprint> \
  --proposer-fingerprint <proposer-content-or-config-fingerprint> \
  --max-new-tokens 32 \
  "prompt"
```

For a prompt-lookup conformance run, add:

```text
--prompt-lookup-ngram 1 --prompt-lookup-max-tokens 4
```

The writer publishes JSON through a same-directory temporary file and rename.
The terminal record repeats output and iteration counts so a partial or
misassembled artifact fails closed.

## Verify

Bind, replay, and optionally execute the derived workload on a topology:

```bash
pnpm sim speculative-capture \
  /tmp/target.json \
  /tmp/speculative.json \
  single-gpu-cpu
```

The command independently:

- derives longest-prefix acceptance from proposal and target-selected tokens;
- rejects target rows after the first mismatch;
- verifies every runtime-claimed committed token and output offset;
- reconstructs the full speculative output;
- compares it with the target-only output;
- replays the same decisions through composite logical state; and
- executes the resulting workload on the selected topology.

Exit status `0` means token parity. Status `2` means the evidence is
well-formed but the outputs differ. Status `1` means malformed, incomplete,
misbound, or failed evidence and must not be interpreted as a model mismatch.

This proves selected-token and logical-state parity for the captured
configuration. It does not prove logit equality, sampling-distribution
equality, numerical kernel equivalence, or physical cache-byte equality.
