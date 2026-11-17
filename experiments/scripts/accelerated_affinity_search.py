"""Validated GPU search for the versioned short-screen cohort audit."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import csv
from pathlib import Path
import subprocess

from affinity_study_common import digest, freeze, read_json
from prepare_affinity_study import fasta, search as cpu_search


def command(settings, directory, graph=False):
    args = [settings['mmseqs'], 'easy-search', str(directory/'query.fa'),
        str(directory/'target_padded'), str(directory/'hits.tsv'), str(directory/'tmp'),
        '--gpu', '1', '--min-seq-id', str(settings['antibody_identity']),
        '-c', str(settings['antibody_coverage']), '--cov-mode', '0', '-e', '0.001',
        '--threads', '4', '--max-seqs', '100000', '--alignment-mode', '3',
        '--format-output', 'query,target,fident,qcov,tcov,evalue']
    if not graph:
        args += ['--max-accept', '1']
    return args


def search(settings, root, name, queries, targets, antibody):
    if not antibody:
        return cpu_search(settings, root, name, queries, targets, antibody)
    validation = Path(settings['search_validation_root'])
    result = read_json(validation/'results_report.json')
    protocol = read_json(validation/'protocol.json')
    if not result['complete'] or not result['validated']:
        raise ValueError('GPU search backend has not passed validation')
    if digest(settings['mmseqs']) != protocol['binary_sha256']:
        raise ValueError('Search binary differs from the validated binary')
    if settings['antibody_identity'] != .9 or settings['antibody_coverage'] != .8:
        raise ValueError('Protection thresholds differ from backend validation')
    queries, targets = set(queries)-{''}, set(targets)-{''}
    directory = root/'searches'/name
    directory.mkdir(parents=True, exist_ok=True)
    fasta(directory/'query.fa', queries)
    fasta(directory/'target.fa', targets)
    graph = name == 'all_lineages'
    manifest = dict(query_sha256=digest(directory/'query.fa'),
        target_sha256=digest(directory/'target.fa'), graph=graph,
        validation_sha256=digest(validation/'results_report.json'),
        command=command(settings, directory, graph))
    freeze(directory/'search_manifest.json', manifest)
    if not queries or not targets:
        (directory/'hits.tsv').write_text('')
        return set()
    commands = [
        [settings['mmseqs'], 'createdb', str(directory/'target.fa'),
            str(directory/'target_db'), '--shuffle', '0'],
        [settings['mmseqs'], 'makepaddedseqdb', str(directory/'target_db'),
            str(directory/'target_padded')],
        command(settings, directory, graph)]
    with (directory/'search.log').open('w') as log:
        for argv in commands:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    found = set()
    with (directory/'hits.tsv').open() as handle:
        for row in csv.reader(handle, delimiter='\t'):
            if len(row) != 6 or float(row[2]) < .9 or min(float(row[3]), float(row[4])) < .8 or float(row[5]) > .001:
                raise ValueError('Search returned an alignment outside the frozen protection cutoffs')
            found.add(row[0])
    freeze(directory/'search_complete.json', dict(manifest_sha256=digest(directory/'search_manifest.json'),
        hits_sha256=digest(directory/'hits.tsv'), positive_queries=len(found)))
    return found


def main():
    import argparse
    import prepare_short_affinity as original
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['selection', 'audit'])
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    original.search = search
    (original.diagnostic_selection if args.job == 'selection' else original.audit)(Path(args.root))


if __name__ == '__main__':
    main()
