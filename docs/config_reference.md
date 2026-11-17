# Configuration reference

`config/config.yaml` is the single source of truth; rules and scripts must not
hardcode paths, model sizes or split names. Mode overlays are layered on top by
`run_gpu.sh` (`gpu.yaml` always, then `smoke.yaml`/`small.yaml`, then the
embedder overlay such as `gpu_all.yaml`). Set `MANGO_EXECUTION_CONFIG` to add a
hardware overlay (`config/local_12gb.yaml` for a 12 GB card).

| Section | Purpose |
|---|---|
| `experiment` | Study name and global seed. |
| `execution` | Execution policy (`min_gpu_memory_gb`). `run_gpu.sh` sets `MANGO_REQUIRE_CUDA=1`, so there is no silent CPU fallback. |
| `dataset` | Which fetcher/standardizer pair to use (SAbDab2) and its filters. |
| `processing` | Cleaning and the group-aware split on `ab_ag_cluster`. |
| `embedding` | `seq_source` (`resolved` residues only, so all representations share a length, or `expected`) and the splits to embed. |
| `embedders` | One entry per antigen representation: `method`, `class`, `label`, plus method-specific options (e.g. ProteinMPNN `model`, `noise`). |
| `active_embedders` | Which embedders the DAG builds; overlays such as `gpu_all.yaml` extend it. |
| `model` | Fixed-position GPT-2 decoder (4 layers, 8 heads), epochs, lr, early-stop patience, loss-plot interval, train/val/predict splits. |
| `generation` | Chain to generate (`heavy`/`light`), target selection (one per held-out cluster), designs per target, sampling parameters. |
| `structure_prediction` | Boltz-2 / Chai-1 confidence settings and bounds on how many designs are folded. |
| `rosetta_interface` | Fixed-backbone Rosetta `InterfaceAnalyzer` QC (`enabled`, `scorefxn`). |
| `analysis` | Cohort selection, IgLM/AntiBERTy/AbLang2, germline scheme, developability, and the per-figure `plots` switches. |
| `weights` | Where model weights live and the public weight URLs. |

## Overlays

| File | Used for |
|---|---|
| `gpu.yaml` | Base single-GPU settings (always applied by `run_gpu.sh`). |
| `smoke.yaml`, `small.yaml` | Tiny plumbing run and bounded validation run. |
| `gpu_esm3.yaml`, `gpu_pyrosetta.yaml`, `gpu_all.yaml` | Add gated ESM3 / PyRosetta PRE / both to `active_embedders`. |
| `local_12gb.yaml` | Lowers `min_gpu_memory_gb` to 10 and sets generation / AntiBERTy batch sizes validated on a 12 GB RTX 4070. |
| `structure.yaml` | Dedicated Boltz/Chai structure-confidence rerun with five diffusion samples. |
| `esm2_ablation.yaml`, `antigen_controls.yaml`, `antigen_length_ablation.yaml`, `iglm_baseline.yaml` | Ablation/control modes of `run_gpu.sh`. |

Run identity: a trained run is `artifacts/runs/<embedder>__<hash>/`, where the
hash covers the dataset, processing, embedding and model configuration, so
changing those fields trains a new run instead of overwriting an old one.
