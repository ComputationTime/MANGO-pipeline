"""Fetch and stage the SAbDab2 ML dataset (splits + structures) from Zenodo.

The dataset is a single ``splits.tar.gz`` archive that unpacks to a
``splits_final/`` directory containing the ``.cif`` structures and the
``*_split.csv`` metadata/split tables.

The download is atomic and idempotent: if the structures are already present
the archive is not re-downloaded, and interrupted downloads are never mistaken
for complete ones.

This module runs in two ways:

* As a Snakemake ``script:`` -- it reads ``snakemake.params`` and touches the
  output marker when the data is ready.
* Standalone --  ``python fetch_sabdab2.py --artifact-root artifacts \\
  --version 0.1.0`` for local testing outside the workflow.
"""

import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ZENODO_URL = (
    "https://zenodo.org/api/records/20083995/files/splits.tar.gz/content"
)


def download_file(url: str, destination: Path, max_attempts: int = 6) -> None:
    """Download ``url`` to ``destination`` atomically, printing progress.

    Writes to a ``.partial`` sibling first and only renames into place on a
    fully successful transfer, so a killed process can never leave a truncated
    file that later looks valid.
    """
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading SAbDab2 archive from {url}", flush=True)
    for attempt in range(1, max_attempts + 1):
        downloaded = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "MANGO-SAbDab2-downloader/1.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"  resuming at {downloaded / 1e6:.0f} MB", flush=True)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                is_partial = response.status == 206
                if downloaded and not is_partial:
                    # A server may legally ignore Range. Restart safely rather
                    # than appending a full response to an existing prefix.
                    downloaded = 0
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[1])
                else:
                    total = downloaded + int(response.length or 0)
                chunk = 1024 * 1024  # 1 MiB
                next_mark = max(5, (downloaded * 100 // total // 5 + 1) * 5) if total else 5
                mode = "ab" if downloaded and is_partial else "wb"
                with partial.open(mode) as output:
                    while True:
                        block = response.read(chunk)
                        if not block:
                            break
                        output.write(block)
                        downloaded += len(block)
                        if total:
                            pct = downloaded * 100 // total
                            if pct >= next_mark:
                                print(
                                    f"  ... {pct:3d}%  "
                                    f"({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)",
                                    flush=True,
                                )
                                next_mark = pct - (pct % 5) + 5
                        else:
                            print(f"  ... {downloaded / 1e6:.0f} MB", flush=True)
            if total and downloaded != total:
                raise OSError(
                    f"incomplete response: received {downloaded} of {total} bytes"
                )
            partial.replace(destination)
            print(f"Download complete -> {destination}", flush=True)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == max_attempts:
                raise
            delay = min(30, 2 ** (attempt - 1))
            kept = partial.stat().st_size if partial.exists() else 0
            print(
                f"  transient download failure ({type(exc).__name__}); "
                f"kept {kept / 1e6:.0f} MB, retrying in {delay}s "
                f"[{attempt}/{max_attempts}]",
                flush=True,
            )
            time.sleep(delay)


def extract(archive: Path, destination: Path) -> None:
    """Safely extract ``archive`` into ``destination``.

    Rejects absolute paths, path traversal (``..``), and symlink/hardlink
    members so a malicious archive cannot write outside ``destination``.
    """
    destination = destination.resolve()

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"Unsafe path in SAbDab2 archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
        tar.extractall(destination)


def ensure_sabdab2(
    artifact_root: str | Path, version: str, url: str = DEFAULT_ZENODO_URL
) -> Path:
    """Ensure the SAbDab2 ``splits_final`` directory exists; return its path.

    Idempotent: a no-op if the structures are already staged.
    """
    dataset_root = Path(artifact_root) / "data" / f"sabdab2_v{version}"
    archive = dataset_root / "splits.tar.gz"
    structures = dataset_root / "splits_final"

    if structures.is_dir():
        print(f"SAbDab2 already present at {structures}", flush=True)
        return structures

    dataset_root.mkdir(parents=True, exist_ok=True)

    if not archive.is_file():
        download_file(url, archive)

    print(f"Extracting {archive.name} ...", flush=True)
    with tempfile.TemporaryDirectory(dir=dataset_root) as temp_dir:
        staging = Path(temp_dir)
        extract(archive, staging)

        candidates = list(staging.rglob("splits_final"))
        if not candidates:
            raise RuntimeError(
                "Archive did not contain a 'splits_final/' directory; "
                f"contents under {staging}: {[p.name for p in staging.iterdir()]}"
            )
        shutil.move(str(candidates[0]), str(structures))

    print(f"SAbDab2 staged at {structures}", flush=True)
    return structures


def main() -> None:
    # Snakemake execution: the injected `snakemake` object carries params/output.
    smk = globals().get("snakemake")
    if smk is not None:
        structures = ensure_sabdab2(
            artifact_root=smk.params.artifact_root,
            version=smk.params.version,
            url=smk.params.url,
        )
        # Sanity-check the expected split table is present before signalling done.
        split_csv = structures / smk.params.split_file
        if not split_csv.is_file():
            raise RuntimeError(f"Expected split file missing: {split_csv}")
        Path(smk.output.marker).write_text(f"{structures}\n", encoding="utf-8")
        return

    # Standalone execution.
    import argparse

    parser = argparse.ArgumentParser(description="Fetch the SAbDab2 ML dataset.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--url", default=DEFAULT_ZENODO_URL)
    args = parser.parse_args()
    ensure_sabdab2(args.artifact_root, args.version, args.url)


if __name__ == "__main__":
    sys.exit(main())
