"""Download and stage AbLang2 paired-model weights once for all embed jobs."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REQUIRED_FILES = ("model.pt", "hparams.json")


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe path in AbLang2 archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not allowed in AbLang2 archive: {member.name}")
        tf.extractall(destination)


def _find_payload(extracted: Path) -> Path:
    candidates = [path.parent for path in extracted.rglob("model.pt")]
    valid = [path for path in candidates if (path / "hparams.json").is_file()]
    if len(valid) != 1:
        raise RuntimeError(
            "expected exactly one directory containing model.pt and hparams.json "
            f"in the AbLang2 archive, found {len(valid)}"
        )
    return valid[0]


def fetch(url: str, weights_dir: str, marker: str) -> None:
    destination = Path(weights_dir)
    marker_path = Path(marker)
    existing = [destination / name for name in REQUIRED_FILES]
    if all(path.is_file() and path.stat().st_size > 0 for path in existing):
        try:
            metadata = json.loads(marker_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        if metadata.get("url") == url:
            print(f"AbLang2 weights already present in {destination}", flush=True)
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ablang2-", dir=destination.parent) as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / "weights.tar.gz"
        digest = hashlib.sha256()
        request = urllib.request.Request(url, headers={"User-Agent": "MANGO-Snakemake/1"})
        print(f"downloading AbLang2 weights from {url}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                digest.update(chunk)

        extracted = tmpdir / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        payload = _find_payload(extracted)

        destination.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            source = payload / name
            if source.stat().st_size == 0:
                raise RuntimeError(f"downloaded AbLang2 file is empty: {name}")
            shutil.copy2(source, destination / name)

    marker_path.write_text(
        json.dumps({"url": url, "sha256": digest.hexdigest()}, sort_keys=True) + "\n"
    )
    print(f"staged AbLang2 weights in {destination}", flush=True)


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        fetch(
            url=smk.params.url,
            weights_dir=smk.params.weights_dir,
            marker=smk.output.marker,
        )
        return

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    fetch(args.url, args.weights_dir, args.marker)


if __name__ == "__main__":
    main()
