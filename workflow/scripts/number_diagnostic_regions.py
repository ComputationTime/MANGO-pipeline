"""Annotate target regions for analysis only; never supply them to the model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

# IMGT definitions: https://www.imgt.eu/IMGTrepertoire/Proteins/rules_Proteindisplay_V.pdf
def region(position):
    if 27 <= position <= 38:
        return 'cdr1'
    if 56 <= position <= 65:
        return 'cdr2'
    if 105 <= position <= 117:
        return 'cdr3'
    return 'framework'


def number(records, output, threads=2):
    from anarci import anarci
    with open(records) as handle:
        rows = list(csv.DictReader(handle))
    sequences = {}
    for row in rows:
        for chain, column in [('heavy', 'resolved_H_seq'), ('light', 'resolved_L_seq')]:
            seq = row[column]
            key = hashlib.sha256(seq.encode()).hexdigest()
            if key in sequences and sequences[key] != (seq, chain):
                raise ValueError('Same diagnostic target has conflicting chain labels')
            sequences[key] = (seq, chain)
    hmmscan = shutil.which('hmmscan') or str(Path(__import__('sys').executable).parent / 'hmmscan')
    result = {}
    items = sorted(sequences.items())
    for start in range(0, len(items), 100):
        batch = items[start:start + 100]
        numberings, details, _ = anarci([(key, seq) for key, (seq, _) in batch], scheme='imgt',
            allow={'H', 'K', 'L'}, hmmerpath=str(Path(hmmscan).parent), ncpu=threads)
        for (key, (seq, chain)), domains, annotations in zip(batch, numberings, details):
            labels = ['unassigned'] * len(seq)
            valid = []
            for domain, detail in zip(domains or [], annotations or []):
                if (chain == 'heavy') != (detail['chain_type'] == 'H'):
                    continue
                numbering, begin, end = domain
                numbered = [(int(pos[0]), aa) for pos, aa in numbering if aa != '-']
                numbered_seq = ''.join(aa for _, aa in numbered)
                if seq[int(begin):int(begin) + len(numbered_seq)] != numbered_seq:
                    raise ValueError('ANARCI numbered target does not align exactly to original sequence')
                valid.append((begin, numbered))
            if len(valid) == 1:
                begin, numbered = valid[0]
                labels[int(begin):int(begin) + len(numbered)] = [region(pos) for pos, _ in numbered]
            result[key] = dict(sequence=seq, chain=chain, labels=labels + ['eos'],
                               status='numbered' if len(valid) == 1 else 'unassigned')
        print(f'ANARCI diagnostic targets: {min(start + 100, len(items))}/{len(items)}', flush=True)
    value = dict(records_sha256=hashlib.sha256(Path(records).read_bytes()).hexdigest(),
                 scheme='imgt', region_definition='CDR1 27-38, CDR2 56-65, CDR3 105-117; inclusive; insertions retained',
                 source='https://www.imgt.eu/IMGTrepertoire/Proteins/rules_Proteindisplay_V.pdf',
                 status_counts=dict(Counter(v['status'] for v in result.values())), sequences=result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2)
    number(**vars(parser.parse_args()))
