"""Download the complete, immutable public OpenAlphaSeq repository snapshot.

Uses only Python's standard library. Reruns verify and reuse completed files;
interrupted individual transfers restart through an atomic staging file.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
from urllib.error import URLError
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen


REPOSITORY = 'aalphabio/open-alphaseq'
HUB = 'https://huggingface.co'
DEFAULT_REVISION = '3fb6b28b3ee758a7dfeaca5f14949dc505e90fda'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'artifacts/data/openalphaseq'
MANIFEST = 'download_manifest.json'
HEADERS = {'User-Agent': 'MANGO-pipeline/OpenAlphaSeqDownloader', 'Accept-Encoding': 'identity'}


def safe_path(value):
    decoded = unquote(value)
    path = PurePosixPath(decoded)
    if (not decoded or decoded == '.' or path.is_absolute() or '..' in path.parts
            or '.git' in path.parts or '\\' in decoded or ':' in decoded or '\x00' in decoded
            or path.as_posix() != decoded or decoded in {MANIFEST, MANIFEST + '.tmp'}):
        raise ValueError(f'Unsafe repository path: {value!r}')
    return decoded


def local_path(output, name):
    path = output / safe_path(name)
    if path.is_symlink() or not path.resolve().is_relative_to(output):
        raise ValueError(f'Repository path escapes the destination: {name}')
    return path


def atomic_manifest(output, manifest):
    temporary = output / (MANIFEST + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary.replace(output / MANIFEST)


def repository_info(revision, timeout, retries, opener):
    url = f'{HUB}/api/datasets/{REPOSITORY}/revision/{quote(revision, safe="")}?blobs=true'
    for attempt in range(retries + 1):
        try:
            with opener(Request(url, headers=HEADERS), timeout=timeout) as response:
                return json.load(response)
        except (OSError, URLError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def file_plan(info):
    if not re.fullmatch(r'[0-9a-f]{40}', info.get('sha', '')):
        raise ValueError('Repository metadata has no immutable Git revision')
    plan = {}
    for item in info['siblings']:
        name = safe_path(item['rfilename'])
        if name in plan:
            raise ValueError(f'Duplicate repository path: {name}')
        size, blob = item.get('size'), item.get('blobId', '')
        digest = (item.get('lfs') or {}).get('sha256')
        if (not isinstance(size, int) or size < 0 or not re.fullmatch(r'[0-9a-f]{40}', blob)
                or (digest is not None and not re.fullmatch(r'[0-9a-f]{64}', digest))):
            raise ValueError(f'Missing publisher size/checksum for {name}')
        plan[name] = dict(path=name, expected_bytes=size, git_blob_sha1=blob, publisher_sha256=digest)
    if not {'README.md', 'LICENSE'} <= plan.keys() or not any(p.endswith('.parquet') for p in plan):
        raise ValueError('Expected OpenAlphaSeq data tables, README and LICENSE')
    return dict(sorted(plan.items()))


def validate_file(path, entry):
    size = path.stat().st_size
    if size != entry['expected_bytes']:
        raise ValueError(f'File size mismatch: {entry["path"]}')
    digest = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
            blob.update(chunk)
    if entry['publisher_sha256']:
        valid = digest.hexdigest() == entry['publisher_sha256']
    else:
        valid = blob.hexdigest() == entry['git_blob_sha1']
    if not valid:
        raise ValueError(f'Publisher checksum mismatch: {entry["path"]}')
    if entry['path'].endswith('.parquet'):
        if size < 12:
            raise ValueError(f'Truncated Parquet file: {entry["path"]}')
        with path.open('rb') as handle:
            first = handle.read(4)
            handle.seek(-4, 2)
            last = handle.read(4)
        if first != b'PAR1' or last != b'PAR1':
            raise ValueError(f'Invalid Parquet header/footer: {entry["path"]}')
    return dict(bytes=size, sha256=digest.hexdigest(), verified=True)


def fetch_file(output, entry, revision, timeout, retries, opener):
    path = local_path(output, entry['path'])
    if path.is_file():
        try:
            return validate_file(path, entry), False
        except (ValueError, OSError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = local_path(output, entry['path'] + '.download')
    url = f'{HUB}/datasets/{REPOSITORY}/resolve/{revision}/{quote(entry["path"], safe="/")}?download=true'
    for attempt in range(retries + 1):
        try:
            with opener(Request(url, headers=HEADERS), timeout=timeout) as response, staging.open('wb') as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    handle.write(chunk)
            checked = validate_file(staging, entry)
            staging.replace(path)
            return checked, True
        except (OSError, URLError, ValueError):
            staging.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def download(output=DEFAULT_OUTPUT, revision=None, metadata_only=False, verify_only=False,
             timeout=60, retries=3, quiet=False, opener=None):
    if timeout <= 0 or retries < 0 or (metadata_only and verify_only):
        raise ValueError('Use a positive timeout, nonnegative retries, and one download mode')
    opener = opener or urlopen
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if previous and (previous.get('repository') != REPOSITORY or previous.get('manifest_version') != 1):
        raise ValueError('Destination contains a different dataset/downloader manifest')
    if not previous and any(output.iterdir()):
        raise ValueError('Choose an empty destination or one containing this downloader\'s manifest')
    if verify_only:
        if not previous:
            raise ValueError('No pinned download manifest to verify')
        if revision is not None and revision != previous['revision']:
            raise ValueError('Offline verification requires the pinned revision')
        pinned = previous['revision']
        keys = ('path', 'expected_bytes', 'git_blob_sha1', 'publisher_sha256')
        plan = {name: {k: entry[k] for k in keys} for name, entry in previous['files'].items()}
    else:
        requested = revision or previous.get('revision') or DEFAULT_REVISION
        info = repository_info(requested, timeout, retries, opener)
        pinned, plan = info['sha'], file_plan(info)
        if re.fullmatch(r'[0-9a-f]{40}', requested) and pinned != requested:
            raise ValueError('Publisher returned a different Git revision')
        if previous and pinned != previous['revision']:
            raise ValueError('Revision differs from the pinned snapshot; choose a new output directory')
    experiments = sorted({PurePosixPath(p).parent.name for p in plan if p.endswith('.parquet')})
    manifest = dict(manifest_version=1, dataset='OpenAlphaSeq', repository=REPOSITORY,
        source_url=f'{HUB}/datasets/{REPOSITORY}', revision=pinned,
        started_utc=previous.get('started_utc', datetime.now(timezone.utc).isoformat()),
        last_attempt_utc=datetime.now(timezone.utc).isoformat(), complete=False,
        status='metadata_only' if metadata_only else 'verifying' if verify_only else 'downloading',
        file_count=len(plan), parquet_files=sum(p.endswith('.parquet') for p in plan),
        total_bytes=sum(e['expected_bytes'] for e in plan.values()), experiments=experiments,
        files={name: dict(entry, verified=False) for name, entry in plan.items()})
    atomic_manifest(output, manifest)
    if not quiet:
        print(f'OpenAlphaSeq {pinned}: {len(plan)} files, {manifest["total_bytes"] / 1e6:.2f} MB, '
              f'{len(experiments)} experiments', flush=True)
    if metadata_only:
        return manifest
    fetched = reused = 0
    try:
        for number, (name, entry) in enumerate(plan.items(), 1):
            if verify_only:
                checked, transferred = validate_file(local_path(output, name), entry), False
            else:
                checked, transferred = fetch_file(output, entry, pinned, timeout, retries, opener)
            fetched += int(transferred)
            reused += int(not transferred)
            manifest['files'][name].update(checked)
            atomic_manifest(output, manifest)
            if not quiet:
                print(f'[{number}/{len(plan)}] {"Downloaded" if transferred else "Verified"}: {name}', flush=True)
        manifest.update(complete=True, status='complete', downloaded_files=fetched, reused_files=reused,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_manifest(output, manifest)
        if not quiet:
            print(f'OpenAlphaSeq complete: {output} ({fetched} downloaded, {reused} reused)', flush=True)
        return manifest
    except Exception as error:
        manifest.update(complete=False, status='failed', error=str(error))
        atomic_manifest(output, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--revision', help=f'Git revision or branch (fresh download defaults to {DEFAULT_REVISION})')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--metadata-only', action='store_true', help='Pin the snapshot and save a download plan')
    mode.add_argument('--verify-only', action='store_true', help='Recheck every local file without network access')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        download(**vars(args))
    except (OSError, URLError, ValueError, KeyError) as error:
        print(f'OpenAlphaSeq download failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
