# Cloud GPU setup: from a new instance to MANGO results

This is the supported setup path for one Linux NVIDIA GPU. It does not require
Slurm, a system CUDA toolkit, or manually downloaded public model files.

## 1. Provision the instance

Recommended minimum:

- Linux x86_64 (Ubuntu 22.04 or 24.04 is a straightforward choice)
- one NVIDIA RTX A6000 or another CUDA GPU with at least 20 GiB VRAM
- 64 GiB system RAM
- 150 GiB free disk
- outbound HTTPS access during initial setup

The rule environments use CUDA 12.1, whose Linux driver minimum is 525.60.13.
Use a current NVIDIA-enabled cloud image and verify it before copying data:

```bash
nvidia-smi
```

The RTX A6000 has 48 GiB VRAM. You do **not** need to install the CUDA toolkit;
the workflow's Conda environments include the CUDA runtime.

## 2. Install the two host prerequisites

The host needs only Git and Conda/Miniforge. Check first:

```bash
git --version
conda --version
```

If Conda is missing, install
[Miniforge](https://github.com/conda-forge/miniforge) using its official Linux
installer, then open a new shell or source its `conda.sh`. Do not install
PyTorch, ESM, PyRosetta, Snakemake, or CUDA globally; MANGO isolates those
packages because several embedders have conflicting requirements.

## 3. Copy or clone the repository

If these changes are on a pushed Git branch, clone that branch. When copying
the working tree directly, omit generated environments and artifacts so the
cloud run starts clean:

```bash
rsync -av --exclude=.snakemake/ --exclude=artifacts/ \
  MANGO/ USER@HOST:/PATH/MANGO/
```

On the instance:

```bash
cd /PATH/MANGO
chmod +x run_gpu.sh install_mango.sh
```

The `MANGO` Git repository can remain nested inside the larger pipeline
workspace; nothing in the runner requires flattening or removing its `.git`.

## 4. Choose a run mode

| Mode | Embedders | Extra user action |
|---|---|---|
| `smoke` / `small` | one-hot, BioPython, ESM2, ESM-IF, ProteinMPNN; Rosetta interface QC after Boltz/Chai | PyRosetta academic/non-commercial eligibility |
| `study` | all seven implemented embedders, including ESM3 and PyRosetta PRE; Rosetta interface QC | accepted ESM3 terms, `HF_TOKEN`, and PyRosetta eligibility |
| `esm2-ablation` | ESM2 8M, 35M, and 150M plus the existing 650M evaluation | completed full-study `esm2` 650M `eval.json` |
| `antigen-controls` | shape-matched constant/random, mean-pooled, shuffled, and cross-attention ESM2 controls | completed ESM2 650M embeddings and `eval.json` |
| `antigen-length-ablation` | constant, random and token-merged ESM2 at 1 token, ¼L, ½L, ¾L and L | completed ESM2 650M embeddings and `eval.json` |
| `iglm-baseline` | full IgLM trained from scratch; reference on Figure 9 | completed context-length comparison |
| `abrank-transfer` | eight AbRank controls followed by matching SAbDab2 fine-tuning; separate figures | prepared, audited CATH-protected cohorts; see [AbRank transfer](abrank_transfer.md) |
| `asd-transfer` | eight ASD 2.0 controls followed by matching SAbDab2 fine-tuning; separate figures | downloaded ASD and audited family/RIOT pool; see [ASD transfer](asd_transfer.md) |
| `asd-loss-comparison` | ESM2 normal/cross-attention, CE/focal, with/without ASD pretraining: eight final models | same audited ASD/native cohorts; see [loss comparison](asd_loss_comparison.md) |
| `asd-prototype` | small HER2-free ASD cohort; same eight ESM2 models, one epoch per stage | existing audited ASD/native cohorts and cached ESM2; see [prototype](asd_prototype.md) |
| `asd-prototype-iglm` | native-only and ASD-pretrained IgLM controls; combined ten-model plot | completed `asd-prototype`; same cohorts and one epoch per stage |
| `antigen-diagnostics` | correct/wrong/removed/pooled antigen probes, CDR NLL and cluster uncertainty | completed prototype and matched IgLM; see [diagnostics](antigen_diagnostics.md) |
| `antigen-diagnostic-training` | eight small position/pooling/continuation/IgLM-adapter follow-ups | completed diagnostic probes and cached ESM2 |
| `asd-scaling` | ASD-only nested 2k/4k cohorts, IgLM and paired/shuffled cross-attention; optimization controls | audited ASD/native protection inputs and cached ESM2; see [scaling](asd_scaling.md) |
| `asd-scaling-8k` | add three one-epoch 8k models to the frozen 2k/4k comparison | completed `asd-scaling`; relaxes antigen frequency cap; see [scaling](asd_scaling.md#8000-record-extension) |
| `asd-scaling-16k` | add 16k IgLM/cross-attention controls and regular MANGO at 2k/4k/8k/16k | completed 8k comparison; dominant-antigen shuffle limit is disclosed; see [scaling](asd_scaling.md#16000-record-extension-and-regular-mango) |
| `asd-scaling-fixed-positions` | add fixed-position prefix MANGO at 2k/4k/8k/16k | completed 16k/regular comparison; same frozen cohorts and cached ESM2 |
| `*-esm3` | default set plus ESM3 | accept model terms and set `HF_TOKEN` |
| `*-pyrosetta` | default set plus PyRosetta PRE | academic/non-commercial eligibility |
| `*-all` | every implemented embedder above | both requirements |

AF-M is not included in `all`: it is still an implementation stub and requires
an MSA/database/container design. AF3 remains cluster-deferred. Boltz-2 and
Chai-1 run automatically after generation on a mode-specific bounded subset,
followed by fixed-backbone Rosetta interface quality-control scoring. Figure 2
and the minimal Rosetta paired plot are part of strict GPU completion.

## 5. Optional gated setup

### ESM3

Visit the [ESM3 model page](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1),
accept its terms, create a read token, and expose it only through the shell:

```bash
export HF_TOKEN=hf_...
```

Do not put the token in YAML, Git, shell history, or a job script committed to
the repository. On a managed platform, use its secret/environment-variable
facility.

### PyRosetta

PyRosetta PRE has no separate neural-network weight file. The Rosetta database
and score functions are distributed with the PyRosetta package. The isolated
environment installs the official optimized quarterly build automatically from
RosettaCommons. The current Linux/Python 3.11 wheel is roughly 1.7 GB, so its
first environment creation can look idle while that single file downloads.

Academic and non-commercial downloads are covered by the
[PyRosetta non-commercial licence](https://www.pyrosetta.org/downloads). Review
the terms and use the [Rosetta licence/download page](https://www.rosettacommons.org/software/license-and-download)
if your institution requires a signed licence or credentials. Commercial or
fee-for-service work requires separate permission. The workflow cannot accept
legal terms on your behalf.

The `pyrosetta_pre` antigen representation is still scientifically opt-in
because its SAbDab mmCIF author-chain mapping needs validation before a full
study. The default post-generation `InterfaceAnalyzer` step is a separate,
validated plumbing path over Boltz/Chai structures with an explicit H/L/antigen
chain contract.

## 6. Download-only check (optional but useful)

The normal smoke/study commands do this automatically. To verify network access
and model authorization without starting embedding or training:

```bash
./run_gpu.sh weights
```

Optional variants are:

```bash
./run_gpu.sh weights-esm3
./run_gpu.sh weights-pyrosetta
./run_gpu.sh weights-all
```

Downloads are cached under `artifacts/weights/`, `artifacts/cache/torch/`, and
`artifacts/cache/huggingface/`. Successful prefetch jobs write machine-readable
markers under `artifacts/weights/pretrained/`. Rerunning is safe and reuses
complete files.

## 7. Run the smoke test, then the study

Start with the five-record, one-epoch run:

```bash
./run_gpu.sh smoke
```

Only after it finishes, launch the full filtered SAbDab2 study:

```bash
./run_gpu.sh study
```

After the full 650M ESM2 evaluation exists, the focused size ablation reuses
that result and trains/evaluates only the three smaller frozen embedders:

```bash
./run_gpu.sh esm2-ablation
```

It writes `fig7_esm2_size_ablation.{png,csv}` under
`artifacts/analysis/figures/` and refuses to retrain the 650M control if its
expected evaluation artifact is absent.

Train and evaluate the antigen-conditioning controls with the same cohort and
MANGO hyperparameters as ESM2 650M:

```bash
./run_gpu.sh antigen-controls
```

The constant and random controls match the full `[L, 1280]` shape of each
record's cached ESM2 650M embedding, including its chain-break token count.
Constant replaces every value with zero; random replaces every value with
deterministic Gaussian noise keyed only by record id and seed. Both retain
antigen length through the prefix size, but neither uses ESM2 embedding values,
antigen sequence values, or antibody data. Missing or incompatible ESM2 shape
references fail rather than falling back to one token. The
mean-pooled ESM2 control averages each correctly paired frozen ESM2 tensor to
one token, isolating antigen information from the long-prefix burden. The
shuffled control assigns every record a different record's frozen ESM2 650M
tensor using a fixed-point-free permutation within each data split. This keeps
real ESM2 values and prefix-length distribution while breaking the correct
antibody-antigen pairing. The cross-attention control reads the original,
correctly paired ESM2 tensors as separate memory at every decoder layer. Its
decoder prefix contains only the learned H/L selector and BOS, so antigen tokens
do not consume decoder positions. Decoder width, depth, initialization of shared
parameters, targets, split, optimizer and stopping settings match the prefix
baseline. Cross-attention adds parameters; their count and architecture are
recorded in the model and evaluation metadata and figure data. Lower NLL is an
experimental outcome, not a consequence guaranteed by the shorter prefix.
The mode writes `fig8_antigen_controls.{png,csv}`
under `artifacts/analysis/figures/`.

These controls replace the earlier single-token constant/random configuration.
Their changed configuration creates new run IDs; existing single-token
checkpoints and control evaluations must not be treated as shape-matched results.
The cross-attention control also has its own architecture and run ID and requires
fresh training. It reuses ESM2 files directly, without copying or re-embedding them.

To compare context size for constant, random and token-merged ESM2 inputs:

```bash
./run_gpu.sh antigen-length-ablation
```

This trains/evaluates the constant and random controls at one token and at ¼,
½, ¾ and all of each record's ESM2 context length, plus ESM2 token merging at
the same lengths. Load `config/antigen_controls.yaml`
before `config/antigen_length_ablation.yaml`, then `config/iglm_baseline.yaml`
for equivalent manual Snakemake runs. Fractions use `max(1, ceil(fraction * L))`; L includes ESM2's chain-break
tokens. Width stays 1280 and the decoder, targets, cluster split, optimizer,
training order, seed and early-stopping settings stay common. Fractional random
contexts are nested prefixes of the same per-record full-length Gaussian tensor.
The configured one-token random vector also matches its first row. Constant
contexts contain only zeros. Fractional contexts retain antigen-length information
through their token count, without using antigen embedding values.

The ESM2 curve uses the correctly paired frozen ESM2 650M values. At each
fraction, deterministic cosine-similarity bipartite matching merges tokens
until exactly `max(1, ceil(fraction * L))` remain. Weighted averages carry the
number of original tokens in each group; output groups retain their earliest
source position. All source tokens, including chain-break positions, contribute.
Merging runs on CPU in the existing isolated tensor environment after the
ordinary ESM2 forward pass, so this reduces decoder context rather than ESM2
inference cost. Its one-token endpoint reuses `control_pooled_esm2` (the mean of
all source tokens); its L endpoint reuses the completed ESM2 evaluation. Only
the three intermediate ESM2 lengths require new models.

Full-length endpoints reuse the existing full-length control run IDs. Every
variant runs in its own Snakemake invocation, and failures do not prevent the
remaining lengths from being attempted. Reruns reuse completed outputs; older
evaluations are refreshed when needed to record context-length statistics.

The strict output is `artifacts/analysis/figures/fig9_antigen_context_length_ablation.png`,
with exact values in `fig9_antigen_context_length_ablation_data.csv` beside it.
Separate train and test panels show constant, random and token-merged ESM2 NLL
across the five context sizes, with full-length ESM2 as a horizontal reference. Values are pooled
heavy/light target-token NLL from each model's best validation checkpoint,
including EOS; training curves are not used as training-split evaluations.
The CSV records actual minimum/mean/maximum context token counts per split,
cohort fingerprints, run IDs, checkpoint epochs, example/token counts and
perplexities and the merging algorithm. Older reference evaluations may obtain
context counts from the matching full-length shape control, recorded in
`context_lengths_source`; the original reference NLL remains unchanged. A plot
is produced only when all fifteen compatible curve points exist (fourteen
control evaluations plus the reused ESM2 reference) and the IgLM evaluation
is complete. The CSV contains 30 MANGO rows plus two IgLM reference rows.

To add the scratch IgLM reference to an existing context-length comparison:

```bash
./run_gpu.sh iglm-baseline
```

This trains the published full IgLM GPT-2 architecture (four layers, eight heads,
width 512, 512 positions and vocabulary 33; 12,889,600 parameters) from random
initialization in the existing isolated model environment. It uses no public
pretrained weights. The full-chain autoregressive objective and the Adam learning
rate, accumulation, seed, epoch ordering, maximum epochs and validation patience
come from the same MANGO configuration. VH/VL targets, cluster-separated splits
and the ordered cohort fingerprint match the existing comparison.

Native IgLM H/L tokens select the chain. A fixed `[HUMAN]` token serves as the
second start token for every record, since the processed comparison cohort has
no antibody-species field. This convention adds no per-record species label and
does not filter the cohort by species. Residues and native `[SEP]` EOS contribute
to NLL; both control tokens are excluded. This uses full-chain autoregression
throughout training and evaluation. No infilling task or antibody prompt is used.

The run writes checkpoints, evaluation, live training curves, exact architecture,
vocabulary, initialization and training settings under
`artifacts/runs/iglm_scratch__<hash>/`. Figure 9 shows its training/test NLL as
horizontal dotted lines because IgLM has no antigen context. The existing MANGO
points are consumed from their completed evaluations. Rerun the same command to
reuse completed outputs; failures retain the workflow log and return nonzero.

Plain study includes both ESM3 and PyRosetta PRE. Validate both first with:

```bash
./run_gpu.sh smoke-esm3
./run_gpu.sh study-esm3

./run_gpu.sh smoke-pyrosetta
./run_gpu.sh study-pyrosetta

./run_gpu.sh smoke-all
./run_gpu.sh study
```

The suffixed study modes remain available for focused six- or seven-method
reruns, but `study-all` and plain `study` now select the same seven embedders.

Even `smoke` downloads the complete SAbDab2 archive once (about 876 MB
compressed and roughly 4.1 GB extracted), because Zenodo distributes it as one
archive. Processing then retains only five cluster-separated records.

The launcher performs these operations automatically:

1. validate `nvidia-smi` and run a real CUDA matrix multiplication;
2. create a pinned Snakemake driver under `.snakemake/`;
3. create isolated rule environments under `.snakemake/conda-gpu/`;
4. download and verify all active embedder weights;
5. fetch/reuse SAbDab2;
6. embed, train, evaluate, reconstruct, generate, score, and plot;
7. always write `artifacts/gpu/results_report.{json,csv}` with per-embedder
   status and available numbers;
8. write `artifacts/gpu/results_complete.json` only if every required output
   completed.

GPU jobs are serialized. Each embedder loads its pretrained model once per
batch, and an interrupted batch reuses valid per-record outputs on restart.
The launcher starts a separate Snakemake invocation for every embedder, so even
a failed Conda environment or weight setup cannot block the remaining methods.
Successful core runs retain their evaluation and prediction results; the final
report names the first missing stage and separately identifies missing analysis
metrics. The shell command returns nonzero after writing a partial report so
cloud automation still notices that the study was not fully complete.

While an embedder is training, its run directory contains a live learning
curve:

```text
artifacts/runs/<embedder>__<hash>/training_curve.csv
artifacts/runs/<embedder>__<hash>/training_curve.png
```

The CSV is flushed after every training iteration. The PNG is safely replaced
every `model.loss_plot_interval` iterations (250 by default) and after every
epoch-end validation pass. It shows raw and rolling-mean training NLL plus the
validation NLL points on a global training-iteration x-axis.

## 8. Weight ownership and automatic behavior

| Component | Weight behavior |
|---|---|
| one-hot | no weights |
| BioPython | no weights |
| AbLang2 context | downloaded from configured Zenodo URL into `artifacts/weights/` |
| ESM2 | preloaded automatically through fair-esm into the project Torch cache |
| ESM-IF1 | preloaded automatically through fair-esm into the project Torch cache |
| ProteinMPNN | downloaded directly from the official repository into `artifacts/weights/` |
| ESM3 | downloaded automatically into the project Hugging Face cache after authorization |
| PyRosetta PRE | no separate weights; database is inside the automatically installed package |
| MANGO model | trained from scratch by this workflow; checkpoints go under `artifacts/runs/` |

IgLM and AntiBERTy used for Figure 3 are installed in isolated analysis
environments and use their package-managed pretrained assets automatically.
ANARCI's germline database is installed with its pinned analysis environment.

## 9. Troubleshooting

- **`nvidia-smi` fails:** fix the cloud image/driver before running MANGO.
- **CUDA preflight fails:** inspect `artifacts/gpu/preflight.json` if present;
  confirm the instance exposes the GPU and has driver 525.60.13 or newer.
- **HTTP/timeout failure:** rerun the same command. Completed downloads and
  outputs are retained.
- **ESM3 401/403:** accept the model terms and verify `HF_TOKEN` is exported in
  the same shell that starts `run_gpu.sh`.
- **No space left:** enlarge the volume, keeping `.snakemake/`, `artifacts/`, and
  the repository on the large volume.
- **PyRosetta install/import failure:** first try `weights-pyrosetta`; then check
  the official download page and the Snakemake log for wheel availability for
  Linux/Python 3.11.
- **A scientific job fails:** rerun unchanged after correcting the cause.
  Snakemake resumes missing work. Logs are under `artifacts/logs/` and
  `.snakemake/log/`. Inspect `artifacts/gpu/results_report.csv` first for the
  successful methods' NLL/perplexity and `results_report.json` for failure stage
  and expected log paths.

Do not interpret smoke-run metrics scientifically. Its purpose is to prove the
complete software, weight, GPU, and artifact path before spending time on the
full study.
