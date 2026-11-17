"""Snakemake adapter for the shared, frozen 650M ESM2 antigen store."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / 'workflow/scripts'))
from asd_transfer_embeddings import embed_esm2

with open(snakemake.log[0], 'w') as log:
    sys.stdout = sys.stderr = log
    embed_esm2(dict(snakemake.params.study), 'openalphaseq', snakemake.input.antigens, snakemake.output.marker)
