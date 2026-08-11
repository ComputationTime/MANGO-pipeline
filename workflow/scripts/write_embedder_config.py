"""Write the one-per-embedder config JSON that sits beside the split folders.

Separate from the embedding scripts so the config exists even before any
structure has been embedded, and so it does not need the embedder's heavy
environment to regenerate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        out = smk.output[0]
        spec = dict(smk.params.spec)
        tag = getattr(smk.wildcards, "embedder", spec.get("method", "unknown"))
        ec.write_embedder_config(out, tag, spec, smk.params.seq_source)
        return

    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--spec", required=True, help="JSON embedder spec")
    p.add_argument("--seq-source", default="resolved")
    a = p.parse_args()
    ec.write_embedder_config(a.out, a.tag, json.loads(a.spec), a.seq_source)


if __name__ == "__main__":
    main()
