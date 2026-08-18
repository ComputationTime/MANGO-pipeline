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
