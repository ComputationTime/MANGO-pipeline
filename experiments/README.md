# experiments/

Side studies that were previously mixed into the main MANGO workflow. They are
**unsupported**: not run by `./run_gpu.sh`, not part of the paper's
`smoke`/`smoke-all`/`study` path, possibly stale. Their history before the
cleanup is on branch `archive/experiments` (tag `pre-cleanup`).

Layout mirrors the repository so existing paths keep working:

```text
experiments/run_experiment.sh   launcher (was the side-study modes of run_gpu.sh)
experiments/workflow/*.smk      study workflows (conda envs still live in ../workflow/envs)
experiments/scripts/            study scripts; they import shared modules from ../workflow/scripts
experiments/config/             study configs
experiments/docs/               per-study notes
experiments/tests/              per-study tests
experiments/download_*.sh, run_interface_attention.sh   data/launch helpers
```

Run from the repository root:

```bash
./experiments/run_experiment.sh <mode>     # see the header of the script for modes
python -m unittest discover -s experiments/tests   # inside a suitable conda env
```

| Study | Mode(s) | Notes |
|---|---|---|
| ASD transfer / prototype / loss comparison / scaling | `asd-transfer`, `asd-prototype[-iglm]`, `asd-loss-comparison`, `asd-scaling[-8k\|-16k\|-fixed-positions]` | `docs/asd_*.md` |
| AbRank transfer | `abrank-transfer` | `docs/abrank_transfer.md` |
| OpenAlphaSeq pairing / affinity | `openalphaseq-pairing`, `openalphaseq-affinity`, `binding-affinity` | reads `MANGO_BINDING_DATA` (now `_to_review/MANGO_BINDING_DATA`) |
| Affinity studies | `affinity-*`, `short-affinity-*`, `legacy-adapter-affinity` | |
| Mechanism / controls | `native-mechanism-*`, `controlled-antigen-*` | |
| Diagnostics | `antigen-diagnostics`, `antigen-diagnostic-training`, `epitope-diagnostics[-smoke]`, `interface-supervision[-smoke]` | |
| Homology / witness search | (run the smk directly) | `workflow/homology_backend.smk` |

Shared core helpers that these studies import stay in `workflow/scripts/`
(`model_common`, `chain_examples`, `device_common`, ...). Do not move them
without updating the studies.
