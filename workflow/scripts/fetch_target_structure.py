"""Download one therapeutic target structure from RCSB.

Downloads atomically (to a .partial first) so an interrupted transfer is never
mistaken for a complete file, and sanity-checks that what came back is actually
mmCIF rather than an HTML error page.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

USER_AGENT = "MANGO-target-downloader/1.0"
TIMEOUT = 120


def fetch(pdb: str, base_url: str, out: str) -> None:
    url = f"{base_url.rstrip('/')}/{pdb.upper()}.cif"
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.unlink(missing_ok=True)

    print(f"downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            with partial.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    # An RCSB miss returns a small HTML body with a 200 in some proxy setups.
    head = partial.open("rb").read(512).decode("utf-8", errors="replace")
    if "data_" not in head:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{url} did not return mmCIF (no 'data_' block in the first 512 bytes). "
            f"Check that {pdb!r} is a valid PDB id."
        )

    partial.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)", flush=True)


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        fetch(smk.wildcards.pdb, smk.params.base_url, smk.output.cif)
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", required=True)
    p.add_argument("--base-url", default="https://files.rcsb.org/download")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fetch(a.pdb, a.base_url, a.out)


if __name__ == "__main__":
    main()
