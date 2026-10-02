# Controlled antigen-embedder comparison

The v3 model generates one variable antibody domain conditioned on antigen and
an H/L selector. Targets are SAbDab2 `VH_numerable_seq` and `VL_numerable_seq`;
full deposited chains are provenance only. Existing records must be standardized
again. Existing v2 checkpoints require retraining.

PRE scores an antigen-only pose. Every non-antigen residue is deleted before
energy evaluation, preserving antigen coordinates. Old full-complex PRE caches
are rejected by the batch model-name contract and by the model loader. Regenerate
PRE before training. Historical PRE timing/results describe the old implementation;
do not pool them with corrected benchmarks. Bound antigen conformations remain
an experimental assumption; this fix does not create unbound structures.

All seven decoders use four layers, eight heads, hidden width 488 and 2048
positions. This width minimizes the worst total-parameter deviation from full
IgLM among multiples of eight near its size, retaining the common decoder and
existing direct linear antigen projections. Full IgLM's public GPT2 configuration
has 12,889,600 parameters (four layers, eight heads, width 512, 512 positions,
vocabulary 33). MANGO totals range from 12,471,328 to 13,220,408, approximately
-3.25% to +2.57%. Count frozen embedder weights separately; they are not trainable
MANGO parameters. No dummy parameters or pretrained-weight changes are used.
Reference: https://github.com/Graylab/IgLM/blob/main/iglm/trained_models/IgLM/config.json

For a given seed, decoder and selector initialization are identical across
embedders. Projection initialization uses an isolated RNG state so its input
width cannot change subsequent dropout draws. Training row ordering, optimizer,
learning rate, accumulation setting and stopping policy remain common. Different
antigen lengths can still cause different dropout draws during actual execution.
Training records must have unique IDs and nonempty cluster IDs, and no cluster
may cross splits. Checkpoints record an ordered cohort fingerprint covering
splits, clusters, target sequences and antigen sequences/chains. Evaluation
requires the same fingerprint and fails if an embedding is missing.

Training epoch NLL, validation NLL, checkpoint selection and evaluation all use
sum of target negative log likelihood divided by target-token count. Count
residues plus EOS; exclude antigen, chain selector and BOS. Each optimizer step
uses the same token mean within its accumulation window, including a partial
final window. Per-iteration loss remains that example's token mean. Training
epoch loss observes changing weights and dropout; it is not a fixed-checkpoint
evaluation of the training split. Heavy/light breakdowns remain available in
evaluation alongside the pooled token mean.

These controls do not establish an IgLM performance baseline: IgLM also differs
in pretraining data, species conditioning and infilling objective. Before making
scientific superiority claims, run paired training seeds and a separately defined
IgLM generation/scoring protocol, plus an antigen-conditioning ablation. Use
held-out antigen clusters as resampling units. Smoke results remain plumbing
checks. No plot implementation was changed for these corrections.

MANGO likelihood figures display one pooled heavy-plus-light value per split
and embedder, with no separate chain panels. NLL is weighted by target-token
counts and perplexity is the exponential of pooled NLL, never a mean of chain
perplexities. Historical exports without chain breakdowns are explicitly labeled
legacy/unspecified and cannot be mixed with the new aggregate evaluation.
Generated-design analyses currently select heavy chains only; those outputs must
not be relabeled as aggregate H/L results.

Audited cache reuse records tensor checksums, antigen identities and implementation
hashes in `audited_cache.json`. Target-only VH/VL updates can reuse antigen
embeddings; antigen or tensor changes invalidate reuse. Such runs are reported
as reused, never as fresh embedding benchmarks.

Each training run writes `training_runtime.json`: wall time includes validation,
checkpointing and live-curve output, but excludes embedding, installation and
downloads. It records CUDA peak allocated/reserved memory plus sampled process
VRAM; the latter is an observed peak rather than a minimum hardware requirement.
