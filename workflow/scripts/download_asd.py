"""Download the active ASD 2.0 Delta snapshot from its public Google Drive folder.

Run ./download_asd.sh to provision an isolated gdown environment automatically.
No Google account, API key, GPU, Spark, or model environment is required.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote


FOLDER_ID = "1iSF3BZdDu2jjxIz3sSz5wjJMbPsAkPDM"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "artifacts/data/asd/asd_2.0"
MANIFEST = "download_manifest.json"
LOG_NAME = re.compile(r"_delta_log/(\d{20})\.json")


def safe_path(value):
    """Reject absolute paths, encoded traversal, and platform-specific separators."""
    decoded = unquote(value)
    path = PurePosixPath(decoded)
    if (not decoded or path.is_absolute() or ".." in path.parts
            or "\\" in decoded or ":" in decoded or "\x00" in decoded):
        raise ValueError(f"Unsafe release path: {value!r}")
    return path.as_posix()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output, manifest):
    temporary = output / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / MANIFEST)


def validate_file(path, expected_size=None, expected_sha256=None):
    size = path.stat().st_size
    if not size or (expected_size is not None and size != expected_size):
        raise ValueError(f"Wrong file size for {path.name}: {size}, expected {expected_size}")
    if path.name.endswith((".parquet", ".parquet.download")):
        with path.open("rb") as handle:
            start = handle.read(4)
            handle.seek(-4, 2)
            end = handle.read(4)
        if start != b"PAR1" or end != b"PAR1" or size < 12:
            raise ValueError(f"Invalid Parquet header/footer: {path.name}")
    else:
        for line in path.read_text().splitlines():
            if line.strip():
                action = json.loads(line)
                if not isinstance(action, dict):
                    raise ValueError(f"Invalid Delta log action in {path.name}")
    checksum = sha256(path)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path.name}")
    return {"bytes": size, "sha256": checksum}


def fetch_file(gdown, output, entry, previous, retries, timeout, quiet):
    """Reuse verified files; let gdown resume partial transfers and handle Drive HTML."""
    path = output / entry["path"]
    if path.is_file():
        try:
            return validate_file(path, entry.get("size"), previous.get("sha256"))
        except (ValueError, OSError):
            print(f"Repairing invalid cached file: {entry['path']}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".download")
    result = gdown.download(
        id=entry["id"], output=str(staging), resume=True, quiet=quiet,
        use_cookies=False, timeout=timeout, retries=retries,
    )
    if result is None:
        raise RuntimeError(f"Google Drive did not return a file: {entry['path']}")
    try:
        info = validate_file(staging, entry.get("size"), previous.get("sha256"))
    except (ValueError, OSError):
        # A published staging file is complete according to gdown. Do not reuse
        # it on the next invocation if validation found HTML or truncation.
        staging.unlink(missing_ok=True)
        raise
    staging.replace(path)
    return info


def read_snapshot(output, log_paths):
    """Replay complete JSON history; obsolete or removed data parts stay excluded."""
    active = {}
    metadata = None
    protocol = None
    for name in log_paths:
        for line in (output / name).read_text().splitlines():
            if not line.strip():
                continue
            action = json.loads(line)
            if "protocol" in action:
                protocol = action["protocol"]
            if "metaData" in action:
                metadata = action["metaData"]
            if "remove" in action:
                active.pop(safe_path(action["remove"]["path"]), None)
            if "add" in action:
                addition = action["add"]
                if addition.get("deletionVector"):
                    raise ValueError("This downloader does not support Delta deletion vectors")
                name = safe_path(addition["path"])
                if not name.endswith(".parquet") or addition["size"] < 12:
                    raise ValueError(f"Invalid Delta data part: {name}")
                active[name] = addition
    if not protocol or protocol.get("minReaderVersion", 99) > 1:
        raise ValueError("Expected the ASD 2.0 Delta reader-version-1 format")
    if not metadata or metadata.get("format", {}).get("provider") != "parquet":
        raise ValueError("Missing Parquet table metadata")
    fields = {field["name"]: field for field in json.loads(metadata["schemaString"])["fields"]}
    chains = fields.get("chains", {}).get("type", {})
    if not isinstance(chains, dict):
        raise ValueError("Missing ASD 2.0 chains schema")
    chain_fields = {field["name"] for field in chains.get("fields", [])}
    if not {"antibody_chains", "target_chains"} <= chain_fields or not active:
        raise ValueError("Missing ASD 2.0 antibody/antigen chains or active data parts")
    return active, metadata


def download(output=DEFAULT_OUTPUT, metadata_only=False, retries=3, timeout=30,
             quiet=False, gdown_module=None):
    if gdown_module is None:
        try:
            import gdown as gdown_module
        except ImportError as error:
            raise RuntimeError("gdown is required; run ./download_asd.sh") from error
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and (previous.get("source_url") != FOLDER_URL
                     or previous.get("manifest_version") != 1):
        raise ValueError("Output contains a manifest from a different dataset or downloader")
    print("Discovering public ASD 2.0 release files...", flush=True)
    listing = gdown_module.download_folder(
        id=FOLDER_ID, output=str(output), skip_download=True, quiet=True,
        use_cookies=False, timeout=timeout,
    )
    entries = {}
    for item in listing or []:
        name = safe_path(item.path)
        if name in entries:
            raise ValueError(f"Duplicate path in public release: {name}")
        entries[name] = {"path": name, "id": item.id}
    versions = {int(match.group(1)): name for name in entries
                if (match := LOG_NAME.fullmatch(name))}
    if not versions:
        raise ValueError("No Delta JSON history found in the public release")
    version = previous.get("delta_version", max(versions))
    if not isinstance(version, int) or version < 0:
        raise ValueError("Invalid pinned Delta version")
    missing = sorted(set(range(version + 1)) - versions.keys())
    if missing:
        raise ValueError(f"Incomplete Delta JSON history (missing versions {missing}); "
                         "checkpoint-only histories are not supported")
    for local in (output / "_delta_log").glob("*.json"):
        if local.stem.isdigit() and int(local.stem) > version:
            raise ValueError("Output contains newer Delta logs than the pinned snapshot; "
                             "choose a separate output directory")
    manifest = {
        "manifest_version": 1, "dataset": "ASD", "release": "2.0",
        "source_url": FOLDER_URL, "delta_version": version,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False, "status": "downloading_metadata",
        "files": dict(previous.get("files", {})),
    }
    write_manifest(output, manifest)
    try:
        log_paths = [versions[number] for number in range(version + 1)]
        for name in log_paths:
            info = fetch_file(gdown_module, output, entries[name],
                              previous.get("files", {}).get(name, {}), retries, timeout, quiet)
            manifest["files"][name] = {**entries[name], **info}
            write_manifest(output, manifest)
        active, metadata = read_snapshot(output, log_paths)
        missing_parts = sorted(set(active) - entries.keys())
        if missing_parts:
            raise ValueError(f"Public folder is missing active data parts: {missing_parts}")
        parts = []
        rows = []
        for name, addition in sorted(active.items()):
            stats = json.loads(addition.get("stats") or "{}")
            rows.append(stats.get("numRecords"))
            part = {**entries[name], "size": addition["size"]}
            parts.append(part)
            old_info = previous.get("files", {}).get(name, {})
            saved_hash = ({"sha256": old_info["sha256"]}
                          if old_info.get("id") == part["id"] and old_info.get("sha256") else {})
            manifest["files"][name] = {**part, **saved_hash, "verified": False}
        manifest.update({
            "table_id": metadata["id"], "data_file_count": len(parts),
            "data_bytes": sum(part["size"] for part in parts),
            "rows_from_delta_statistics": sum(rows) if all(isinstance(n, int) for n in rows) else None,
            "status": "metadata_only" if metadata_only else "downloading_data",
        })
        write_manifest(output, manifest)
        print(f"Delta version {version}: {len(parts)} active parts, "
              f"{manifest['data_bytes'] / 1e9:.2f} GB, "
              f"{manifest['rows_from_delta_statistics']} rows in Delta statistics", flush=True)
        if metadata_only:
            return manifest
        for index, entry in enumerate(parts, 1):
            print(f"[{index}/{len(parts)}] {entry['path']}", flush=True)
            info = fetch_file(gdown_module, output, entry,
                              previous.get("files", {}).get(entry["path"], {}), retries, timeout, quiet)
            manifest["files"][entry["path"]] = {**entry, **info, "verified": True}
            write_manifest(output, manifest)
        manifest.update({"complete": True, "status": "complete",
                         "completed_utc": datetime.now(timezone.utc).isoformat()})
        write_manifest(output, manifest)
        print(f"ASD 2.0 download complete: {output}", flush=True)
        return manifest
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        write_manifest(output, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Delta table destination (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download transaction logs and write the plan without fetching data parts")
    parser.add_argument("--retries", type=int, default=3, help="Additional network attempts per file")
    parser.add_argument("--timeout", type=float, default=30, help="Network timeout in seconds")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file transfer progress bars")
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0:
        parser.error("--retries must be nonnegative and --timeout must be positive")
    try:
        download(**vars(args))
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ASD download failed: {error}\nRerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
