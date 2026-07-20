# Representative Model Validation

**Status:** architecture audit, 2026-07-20

This audit asks whether a selected model changes the executable simulation, not
merely whether its metadata can be displayed. A model family passes only when
its component invocations, weight capacity, placement constraints, and required
communication are bound to replayable workload evidence.

## Validation Matrix

| Representative family | Required behavior | Current result |
|---|---|---|
| Llama 3 dense decoder | Bind target weights and active attention/FFN work to decode execution and capacity | **Complete for the declared target model.** Kernel timing remains heuristic without calibration. |
| Gemma 4 / Qwen-VL composite | Run prompt-only vision/embedding or projector components, capacity-check their weights, place each component, and reserve declared dataflow transfers | **Complete for declared component scheduling.** Prompt work runs once before decode and before speculative proposal. Image-tile token expansion still requires request-level tile input. |
| Whisper encoder-decoder | Run the audio encoder once before the autoregressive decoder and preserve its dataflow dependency | **Complete for declared component scheduling.** Encoder work is not repeated for chunked prefill batches. |
| Audio codec / any-to-any composite | Execute every ordered single-pass stage without inventing a token decoder | **Complete for declared component scheduling.** The dashboard selects the one-shot Pipeline workload and reports invocation throughput rather than token decode throughput. |
| Autoregressive TTS plus vocoder | Run prompt components, decode code tokens, then invoke the final vocoder exactly once | **Complete for flat AR plus final-stage scheduling.** Nested autoregressive inner-loop details remain partial. |
| Mixtral / DeepSeek MoE | Derive expert count and top-k from the model, bind EP ownership and All-to-All traffic, and compose TP/EP according to the selected topology | **Partial.** Active expert weight traffic affects target timing, but the model does not configure the expert workload. |
| Target plus independent draft model | Capacity-check and place both models; derive proposer work from the imported draft profile; keep target verification separate | **Partial.** The family and target are bound, but draft execution still uses a family heuristic and ignores imported draft weights. |

These are deliberately adversarial checks. Parsing a valid
`inference_metadata.yaml`, rendering a pipeline graph, or producing generic
expert/speculative events does not satisfy the corresponding row.

## Executable Guards

The web test suite constructs small ONNX manifests for each architecture shape
and asserts the execution coverage embedded in `DashboardModelBinding`:

- dense single-target packages report `complete/full_model`;
- composite VLM packages schedule prompt components exactly once before decode;
- Whisper and TTS packages preserve prompt and final phase boundaries;
- pure any-to-any packages produce ordered component-tagged frozen-plan steps;
- cross-device dataflow reserves replayable transfer paths and
  `device_transfer: false` enforces colocation;
- independent draft packages disclose that proposer cost is not model-bound;
- representative MoE presets disclose that EP routing is not model-bound; and
- component phase gates and loop bounds survive metadata normalization and feed
  the pipeline scheduler.

Coverage is part of the deterministic dashboard input. Artifact replay cannot
silently upgrade a target-only simulation into a full-pipeline claim.

## Required Next Implementation

1. Make component placement an explicit user-editable mapping constrained by
   device capability, resource limits, and metadata preferences.
2. Bind model MoE dimensions to routed-expert workload generation. Treat manual
   expert-count/top-k changes as explicit overrides recorded in the artifact.
3. Bind an independent draft component to draft placement, memory, and
   architecture-derived work. Do not use the generic proposer coefficient when
   a complete draft profile is available.
4. Model iterative scheduler/CFG costs and nested autoregressive inner-loop
   execution before upgrading those coverage limitations.
5. Enforce metadata hardware requirements, including dtype support and minimum
   useful TP degree, during scenario validation.

The simulator now reports end-to-end declared component scheduling for the
complete rows above. It still does not claim model-bound expert-parallel,
imported-draft speculative, nested-AR, or iterative-scheduler throughput while
their machine-readable limitations remain present.
