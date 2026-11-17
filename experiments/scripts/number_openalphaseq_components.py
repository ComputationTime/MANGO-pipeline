"""Cache full ANARCI IMGT domain assignments for every expression component."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time


def number(root, threads=6, batch_size=5000):
    from anarci import anarci
    root = Path(root)
    source = root / 'components.fasta'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    db = sqlite3.connect(root / 'domains.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
    db.execute('''CREATE TABLE IF NOT EXISTS domains
        (component_id TEXT PRIMARY KEY, sequence TEXT, status TEXT, chain TEXT,
         target TEXT, cdr3_indices TEXT, start INTEGER, end INTEGER)''')
    old = db.execute("SELECT value FROM metadata WHERE key='source_sha256'").fetchone()
    if old and old[0] != digest:
        raise ValueError('ANARCI inputs changed')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('source_sha256',?)", (digest,)); db.commit()
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    old_impl = db.execute("SELECT value FROM metadata WHERE key='implementation_sha256'").fetchone()
    if old_impl and old_impl[0] != implementation:
        raise ValueError('ANARCI implementation changed; use a new annotation cache')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('implementation_sha256',?)", (implementation,));db.commit()
    known = {r[0] for r in db.execute('SELECT component_id FROM domains')}
    lines = source.read_text().splitlines()
    items = [(lines[i][1:], lines[i+1]) for i in range(0,len(lines),2) if lines[i][1:] not in known]
    hmmer = shutil.which('hmmscan') or str(Path(sys.executable).parent / 'hmmscan')
    total = len(lines)//2; started = time.monotonic()
    for begin in range(0,len(items),batch_size):
        batch = items[begin:begin+batch_size]
        numberings, details, _ = anarci(batch, scheme='imgt', allow={'H','K','L'},
            hmmerpath=str(Path(hmmer).parent), ncpu=threads)
        rows = []
        for (key, sequence), domains, info in zip(batch, numberings, details):
            if not domains or len(domains) != 1:
                rows.append((key,sequence,'unassigned_or_multiple',None,None,None,None,None)); continue
            (numbering,start,end), detail = domains[0], info[0]
            entries = [(int(pos[0]), aa) for pos,aa in numbering if aa!='-']
            target = ''.join(aa for pos,aa in entries)
            if sequence[int(start):int(start)+len(target)] != target:
                raise ValueError('ANARCI domain is not an exact contiguous segment')
            chain = 'heavy' if detail['chain_type']=='H' else 'light'
            cdr3 = [i for i,(pos,aa) in enumerate(entries) if 105<=pos<=117]
            rows.append((key,sequence,'valid',chain,target,json.dumps(cdr3),int(start),int(end)))
        db.executemany('INSERT INTO domains VALUES (?,?,?,?,?,?,?,?)',rows);db.commit()
        done=len(known)+min(begin+batch_size,len(items))
        progress=dict(complete=False,annotated=done,total=total,source_sha256=digest,
                      wall_seconds=time.monotonic()-started)
        (root/'numbering_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(f'ANARCI {done:,}/{total:,}; {progress["wall_seconds"]:.1f}s',flush=True)
    status=dict(complete=True,total=total,source_sha256=digest,
                counts=dict(Counter(r[0] for r in db.execute('SELECT status FROM domains'))))
    db.close();(root/'numbering_complete.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--threads',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=5000)
    number(**vars(p.parse_args()))
