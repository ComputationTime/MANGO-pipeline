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

The scratch IgLM baseline below uses the same training cohort and autoregressive
objective as MANGO. The published pretrained IgLM also differs in pretraining
data, species conditioning and its infilling objective. Scientific superiority
claims require paired training seeds and a defined generation/scoring protocol,
plus antigen-conditioning ablations. Use held-out antigen clusters as resampling
units. Smoke results remain plumbing checks.

MANGO likelihood figures display one pooled heavy-plus-light value per split
and embedder, with no separate chain panels. NLL is weighted by target-token
counts and perplexity is the exponential of pooled NLL, never a mean of chain
perplexities. Historical exports without chain breakdowns are explicitly labeled
legacy/unspecified and cannot be mixed with the new aggregate evaluation.
Generated-design analyses currently select heavy chains only; those outputs must
not be relabeled as aggregate H/L results.

The constant and random antigen controls match each record's full frozen ESM2
650M shape `[L, 1280]`, preserving the decoder prefix length and projection width.
The constant tensor is all zero; the random tensor is deterministic Gaussian
noise keyed by record id and seed. Only the source shape is used, including the
number of inter-chain separator positions; no separator values, antigen embedding
values, or antibody data are copied. These controls retain antigen length and
therefore test the value of antigen features at a matched prefix size. They are
distinct from the historical single-token controls and require fresh training.
Mean-pooled ESM2 remains a separate single-token control.

`control_cross_attention_esm2` is an architecture control using the same correctly
paired frozen ESM2 650M tensors. Its architecture is
`antigen_chain_cross_attention_gpt2_v1`. Each GPT-2 layer attends to the projected
antigen memory through its own cross-attention module; the autoregressive input
contains only the H/L selector, BOS and previous target residues. The decoder's
sequence length is therefore `2 + target_length + EOS`, rather than
`antigen_length + 2 + target_length + EOS`. Antigen memory keeps all its tokens.
Training and generation use the same memory, and loss still covers only target
residues and EOS. No reference antibody is supplied as context.

The implementation uses [GPT-2's native cross-attention](https://github.com/huggingface/transformers/blob/v4.57.2/src/transformers/models/gpt2/modeling_gpt2.py).
Shared decoder, selector and projection weights start identically to the prefix
baseline for a given seed; only cross-attention weights are added. At width 488,
four layers add 3,822,016 trainable parameters, taking the ESM2 control from
13,095,480 to 16,917,496 parameters. This is not a parameter-matched comparison.
The architecture, source embedding tag and extra parameter count are recorded
separately so any observed NLL difference can be interpreted with that limitation.
The ordinary seven-representation study remains on the prefix architecture.

The focused `antigen-length-ablation` mode compares constant, random and
token-merged ESM2 prefixes at 1 token, ¼L, ½L, ¾L and L, using
`max(1, ceil(fraction * L))` per record.
L is the actual frozen ESM2 650M tensor length, including chain-break positions.
All use width 1280 and the prefix architecture, with the same decoder,
projection size, initialization, cohort, training seed, row order and stopping
policy. Fractional random tensors are exact prefixes of the same record/seed's
full noise tensor; the 1280-wide one-token control matches its first row.
The full-length controls retain their existing run IDs and can be reused.

The ESM2 variant compresses correctly paired frozen 650M embeddings after
inference. Each round divides tokens by alternating positions, matches source
tokens to destination tokens by cosine similarity, and merges the strongest
edges. Group averages are weighted by original token counts and groups are
ordered by their earliest source position. The matching idea follows
[ToMe](https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py);
this implementation uses final ESM2 embeddings as its similarity metric and
leaves ESM2 itself unchanged. Zero chain-break vectors participate with finite
zero cosine scores. No antibody fields participate. The one-token endpoint is
exactly mean pooling and reuses its existing control; L reuses ordinary ESM2.
Only ¼L, ½L and ¾L add new training runs.

Figure 9 has separate training/test NLL panels with all three curves
and a full-ESM2 reference. Evaluation uses the best validation checkpoint for
each model, never test data for selection. The x-axis is categorical because
fractional lengths vary by record; actual context length ranges and means are
exported with the plotted data. Cohort fingerprints, heavy/light token counts,
embedding widths, architecture and loss reduction must agree before plotting.
Actual context-count statistics must also agree across all three curves at
each level. The figure CSV identifies the merging algorithm and any reuse of
full-length shape-control counts for an older ESM2 reference evaluation.

The `iglm-baseline` launcher trains the full IgLM architecture from scratch and
adds its NLL to Figure 9. The architecture and 33-token vocabulary follow the
[official configuration](https://github.com/Graylab/IgLM/blob/main/iglm/trained_models/IgLM/config.json)
and [vocabulary](https://github.com/Graylab/IgLM/blob/main/iglm/trained_models/vocab.txt).
Its 12,889,600 parameters are initialized randomly; pretrained weights are never
loaded. Every training example is `[HEAVY/LIGHT], [HUMAN], residues, [SEP]`.
The species token is fixed across the complete mixed cohort and acts as a start
symbol, without supplying individual species labels. The H/L token selects the
requested chain. Only residues and EOS receive labels, using the common
target-token mean including EOS. No antigen or antibody prompt is supplied.

Training uses the same VH/VL targets, cluster splits, Adam settings, accumulation,
seed, epoch order, maximum epochs and validation patience as MANGO. The best
validation checkpoint is evaluated on train/validation/test; pooled heavy/light
counts and cohort fingerprints must match MANGO before plotting. Its full-chain
autoregressive objective provides an architecture baseline under the comparison
training protocol. The evaluation records that objective explicitly.

IgLM's NLL is independent of antigen context length, so Figure 9 shows one
horizontal dotted line per split. The CSV stores two distinct reference rows
with `model_family=IgLM`, `mode=iglm_scratch` and zero antigen context, alongside
the 30 original MANGO curve points. Run metadata records the full architecture,
parameter count, vocabulary, fixed species token, initialization and optimizer
settings. `antigen-length-ablation` also trains/reuses IgLM in its own invocation
before aggregating the figure.

Audited cache reuse records tensor checksums, antigen identities and implementation
hashes in `audited_cache.json`. Target-only VH/VL updates can reuse antigen
embeddings; antigen or tensor changes invalidate reuse. Such runs are reported
as reused, never as fresh embedding benchmarks.

Each training run writes `training_runtime.json`: wall time includes validation,
checkpointing and live-curve output, but excludes embedding, installation and
downloads. It records CUDA peak allocated/reserved memory plus sampled process
VRAM; the latter is an observed peak rather than a minimum hardware requirement.
