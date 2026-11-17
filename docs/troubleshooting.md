# Troubleshooting

On a new machine run `./run_gpu.sh doctor` first: it checks the GPU, driver, disk, RAM, Conda, network and `HF_TOKEN` without changing anything.

Afterwards, start from `artifacts/gpu/results_report.json`: it names, per embedder,
the first missing stage and the log files to open (`artifacts/logs/`,
`.snakemake/log/`). Re-running the same `./run_gpu.sh <mode>` resumes.

| Symptom | Likely cause / fix |
|---|---|
| `nvidia-smi is missing` / preflight fails | Install an NVIDIA driver ≥ 525.60. The launcher fails fast rather than falling back to CPU (`MANGO_REQUIRE_CUDA=1`). |
| CUDA out of memory / card below 20 GB | Add `MANGO_EXECUTION_CONFIG=config/local_12gb.yaml` (validated on a 12 GB card). GPU rules share `gpu=1` and run serially. |
| ESM3 fails to download | Accept the model terms on Hugging Face and `export HF_TOKEN=...` (never commit it). Other embedders still run. |
| PyRosetta env fails | PyRosetta needs an eligible licence; the other embedders are unaffected. Validate PRE with `smoke-all` before trusting its results. |
| Conda solve error in one env | Environments are isolated per rule on purpose; fix that env's YAML under `workflow/envs/`, never install everything into one env. |
| Weight download corrupt or interrupted | Re-run; downloads are verified and reused. `./run_gpu.sh weights --all` fetches weights only. |
| Figures missing but runs finished | Cross-embedder figures are strict outputs: they need every requested method. Check the report for the failed branch. |
| Host `python -m unittest` fails with `No module named torch/pandas` | Expected: the host Python is deliberately bare. Run the tests inside a workflow Conda env (see `scripts/check.sh`). |

More detail: [cloud_gpu_setup.md](cloud_gpu_setup.md).
