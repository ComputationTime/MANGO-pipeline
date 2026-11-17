"""Snakemake adapter for the shared, frozen 650M ESM2 antigen store."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / 'workflow/scripts'))
from asd_transfer_embeddings import embed_esm2

with open(snakemake.log[0], 'w') as log:
    sys.stdout = sys.stderr = log
    embed_esm2(dict(snakemake.params.study), 'openalphaseq', snakemake.input.antigens, snakemake.output.marker)
