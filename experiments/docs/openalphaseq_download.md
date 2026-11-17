# Download OpenAlphaSeq

Download the complete official [aalphabio/open-alphaseq repository](https://huggingface.co/datasets/aalphabio/open-alphaseq):

```bash
./experiments/download_openalphaseq.sh
```

The launcher uses Python 3.9 or newer and its standard library. It requires
outbound HTTPS and leaves model/analysis environments unchanged. No account,
token, GPU, or package installation is needed. Set
`OPENALPHASEQ_DOWNLOAD_PYTHON=/path/to/python` to choose the interpreter.

The default output is `artifacts/data/openalphaseq/`. Downloads preserve the
publisher's directory layout, including all experiment Parquet files, data
cards, repository README, license and tutorial. The default reviewed revision is
`3fb6b28b3ee758a7dfeaca5f14949dc505e90fda`: 23 files totaling 44,676,842 bytes.
This snapshot contains nine experiment tables:

| Experiment | Raw rows |
|---|---:|
| YM_0005 | 163,882 |
| YM_0549 | 299,910 |
| YM_0693 | 134,580 |
| YM_0852 | 120,208 |
| YM_0985 | 148,128 |
| YM_0988 | 186,745 |
| YM_0989 | 94,604 |
| YM_0990 | 150,048 |
| YM_1068 | 407,803 |
| **Total** | **1,705,908** |

These counts were read from the downloaded Parquet footers. They count raw
experiment rows, including controls, missing affinities and replicate
observations. The downloader preserves every experiment, including HER2;
binding, domain standardization, HER2 exclusion and benchmark-overlap filtering
belong to a separate training-cohort preparation step.

Rerun the same command to verify and reuse completed files. A failed individual
transfer restarts from its beginning on the next attempt. Network failures are
retried; completed files are reused. Downloads are published atomically after
their byte counts and publisher checksums match. Large-file SHA-256 checksums
come from Hugging Face LFS metadata; ordinary files use their Git blob SHA-1.
Parquet files also require valid `PAR1` header/footer markers. Local SHA-256
digests and immutable upstream revision are recorded in `download_manifest.json`.

Plan a snapshot or verify local files offline:

```bash
./experiments/download_openalphaseq.sh --metadata-only --output artifacts/data/openalphaseq_plan
./experiments/download_openalphaseq.sh --verify-only
```

To fetch a later upstream release, explicitly select a new destination:

```bash
./experiments/download_openalphaseq.sh --revision main --output artifacts/data/openalphaseq_new
```

Each output directory remains pinned to its first resolved Git revision.
Changing that revision in place fails rather than mixing snapshots. Destinations
with unrelated existing files are rejected. All data and generated manifests
remain under ignored `artifacts/`; the downloader does not launch training.

Read tables in a separate analysis environment with a Parquet reader installed:

```python
import pandas as pd

table = pd.read_parquet("artifacts/data/openalphaseq/data/YM_0693/data.parquet")
```
