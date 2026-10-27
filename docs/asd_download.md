# Download ASD 2.0

From the repository root:

```bash
./download_asd.sh
```

The launcher creates a small isolated Python environment under
`artifacts/envs/asd_download/` and installs `gdown==6.4.2`. It requires Python
3.10 or newer with `venv` support and outbound HTTPS. It uses public Google Drive
links, with no Google login or API key. Existing model environments are untouched.
Set `ASD_DOWNLOAD_PYTHON=/path/to/python` to select the bootstrap interpreter.

The default destination is `artifacts/data/asd/asd_2.0/`. The
[official release](https://naturalantibody.com/asd/) is distributed as a Delta
table, approximately 2.45 GB of active data parts at the reviewed snapshot.
The script downloads the JSON transaction history and only the active Parquet
parts. The shared folder also contains obsolete parts, checksum sidecars, and
CSV exports; these are not needed to load the Delta table.

Inspect the snapshot and download plan first:

```bash
./download_asd.sh --metadata-only
```

Choose another destination if needed:

```bash
./download_asd.sh --output artifacts/data/asd/another_snapshot
```

Rerun the same command to resume. The downloader keeps partial transfers,
checks completed files, and atomically publishes each downloaded file. File
sizes must match the Delta log, Parquet headers and footers must be present,
and local SHA-256 hashes are saved in `download_manifest.json`. Later runs
verify these hashes before reusing files. The hashes record the local snapshot;
they are not publisher-provided checksums.

The manifest pins the Delta version selected on the first run. Use a new output
directory to fetch a later upstream snapshot. `complete` becomes `true` only
after every active part has passed verification. Metadata-only and failed runs
leave it `false`. Incomplete JSON histories, including checkpoint-only histories,
fail explicitly rather than guessing which files belong to the table.

The result can be opened by a Delta reader without Spark. For example, in a
separate analysis environment with `deltalake` installed:

```python
from deltalake import DeltaTable

table = DeltaTable("artifacts/data/asd/asd_2.0")
# Select columns or batches before materializing this large table in memory.
```

ASD is provided under CC BY-NC 4.0 for noncommercial research; see the
[publisher's access information](https://naturalantibody.com/asd/).
Downloading does not filter binders, remove benchmark overlap, or launch training.
