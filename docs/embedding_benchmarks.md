# Embedding runtime benchmarks

Embedding timing is independent of the existing end-to-end launcher timing.
The latter includes installations, downloads, training and analysis; it remains
in the existing `wall_seconds` columns and is explicitly labeled
`launcher_timing_scope: end_to_end_commands_including_setup`.

## Definitions

The supported persistent-model batch path records:

- **Model loading:** import of the embedder module, local weight loading,
  model construction and device placement, or PyRosetta initialization. Loaded
  once per batch, only if at least one record needs computation. Python/torch
  startup and initial CUDA context setup precede this timer.
- **Computation/preprocessing:** per-record sequence handling, structure parsing,
  tokenization, embedding forward pass, metadata and CPU/dtype conversion.
  This includes transfers performed before writing the output payload.
- **Serialization:** directory creation, payload construction, `torch.save` and
  its save-log output. It measures OS-buffered writes, not durable `fsync`.
- **Total embedding seconds:** model loading plus computation and serialization
  for successful fresh records. CSV reading, cache validation, benchmark/report
  writing, process startup, dependency installation and downloads are excluded.
- **Seconds per structure:** total embedding seconds divided by fresh structures;
  model loading is therefore amortized over the fresh subset.
- **Residues per second:** fresh input biological residues divided by fresh
  computation/preprocessing seconds. It excludes loading and serialization.
  Unknown residues count; inserted separator/BOS/EOS tokens do not. Structural
  methods use `resolved_ag_seq` for a common input-work denominator, even if an
  individual parser ultimately emits fewer residues.

CUDA is synchronized at model-loading, record and serialization boundaries.
First-forward initialization is retained; no hidden warm-up is performed.
Models, pretrained weights, tensor dtypes, chain conventions, conditioning and
training settings are unchanged. One-hot uses the exact existing vocabulary
`|ACDEFGHIKLMNPQRSTVWY`; unknowns remain all-zero float32 rows.

`peak_vram_mib` is sampled NVIDIA compute-process memory for the batch process
and its children at 0.5-second intervals. It excludes unrelated jobs and can
miss brief peaks. `peak_torch_allocated_mib` and `peak_torch_reserved_mib` use
CUDA allocator high-water marks reset before model loading; they exclude
non-PyTorch allocations and CUDA context/driver overhead. CPU-only fresh batches
report zero process VRAM. Unavailable and entirely reused measurements are null.
These are observed peaks, not a proven minimum GPU capacity.

## Freshness and provenance

Each batch writes `embedding_benchmark.json` next to `.batch_complete.json` and
an immutable `benchmarks/<invocation-id>.json` archive. Every record is marked
`fresh`, `reused` or `failed`. Cached records have no compute/serialization times
and contribute no structures or residues to throughput. All-cached batches do
not load the model and have null throughput/timing fields. Mixed fresh/reused
batches describe only their fresh subset and are **not eligible** for the full
fresh comparison. Failed batches retain diagnostic measurements but do not
publish successful aggregate throughput.

A source/environment-definition fingerprint (`suite_id`) covers all embedders,
shared batch/timing code and local structure helpers. A `cohort_id` identifies
ordered selected rows and the residue source. A change to the implementation
during a batch invalidates its comparison. Each launcher sets a distinct
`MANGO_LAUNCH_ID`; the report does not publish historical batch measurements as
current fresh results when Snakemake skips an existing output.

The GPU CSV uses separate `embedding_*` fields. These repeat across split rows;
count each embedder only once when summarizing runtime. The JSON retains individual
records and groups eligible current measurements by `(suite_id, cohort_id)`.
Never combine different groups into one benchmark. Hardware, PyTorch/CUDA
versions and CPU thread count are also recorded; inspect these when comparing
runs on different hosts. Environment differences required by the embedders are
preserved. Existing pre-instrumentation outputs have no embedding benchmark.

## Running a fresh comparison

First finish the currently running pipeline. Warm/download model assets using
the normal `weights` targets. Measured batch work enforces offline loading:
missing assets cause a clear failure rather than folding downloads into model
loading time. Normal runs preserve per-record reuse.

For a deliberate repeat, pass the top-level Snakemake config option
`embedding_benchmark_force_fresh=true` and target `embed_antigen`, with
`execution.batch_embeddings` enabled by the GPU overlay. Keep each embedder in
its own Snakemake invocation/environment, as the launcher does. This option
recomputes antigen outputs without changing model or training configuration;
subsequent training may rerun because the embedding files were rewritten.
Use the same selected records and configuration for every method. For repeated
fresh measurements with the flag already enabled, force the selected batch
rule as well (`--forcerun batch_antigen_one_hot`, for example).

A standalone Snakemake embedding comparison can share an explicit unique
`MANGO_LAUNCH_ID` across its per-embedder invocations. Raw batch archives remain
available even without a launcher report. Prefer a separate benchmark output
root for standalone comparisons to avoid rewriting inputs to a live study.

Do not change implementation files or start GPU benchmarks beside a running
pipeline. Prepare changes separately, then apply them only after its launcher
and background prefetch processes exit. The old launcher timings remain valid
end-to-end observations; they are not fresh embedding benchmarks.
