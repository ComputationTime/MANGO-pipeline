"""CPU-only verification of sparse candidate alignment against the full DB."""
from pathlib import Path
import csv
import shutil

from affinity_study_common import digest, freeze, read_json, write_json
from benchmark_homology_backend import fasta, call
from witness_homology_search import verify_candidates, checked_hits

ROOT=Path('artifacts/experiments/witness_revalidation_v1')
PRIOR=Path('artifacts/experiments/homology_backend_probe_v1')
BINARY='artifacts/envs/cath-split/bin/mmseqs'
TARGET=Path('artifacts/experiments/affinity_short_v3/searches/external_antibodies/target.fa')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(binary=BINARY,binary_sha256=digest(BINARY),
        input_sha256={str(p):digest(p) for p in (PRIOR/'queries.fa',PRIOR/'gpu.tsv',PRIOR/'cpu.tsv',TARGET)},
        implementation_sha256={str(Path(__file__)):digest(__file__),
            'workflow/scripts/witness_homology_search.py':digest('workflow/scripts/witness_homology_search.py')},
        filters=dict(identity=.9,both_coverage=.8,evalue=.001),
        acceptance='Recover every supplied certified reference pair after full-database realignment; reject every added shuffle-pair control; match full-database E-values within rounding.',
        sources=['https://github.com/soedinglab/MMseqs2/wiki#prefiltering-format',
            'https://raw.githubusercontent.com/soedinglab/MMseqs2/18-8cc5c/src/alignment/Alignment.cpp'],
        scope='CPU-only sparse-realignment validation; not end-to-end GPU performance or model training.')
    freeze(ROOT/'protocol.json',protocol)
    write_json(ROOT/'state.json',dict(status='running'))
    for name in protocol['implementation_sha256']:
        destination=ROOT/'frozen_source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(name,destination)
    with (PRIOR/'gpu.tsv').open() as handle:candidates=list(csv.reader(handle,delimiter='\t'))
    expected={(r[0],r[1]):r for r in candidates}
    targets=fasta(TARGET);negatives=[]
    for index,target in enumerate(sorted(targets)[:16]):
        negatives.append(['shuffle_'+str(index),target,'1','1','1','0'])
    with (ROOT/'candidates.tsv').open('w') as handle:
        csv.writer(handle,delimiter='\t',lineterminator='\n').writerows(candidates+negatives)
    # Execute this same file in a bounded child; the parent records exit/timeout.
    import sys
    stage=call([sys.executable,str(Path(__file__)),'--worker'],ROOT/'worker.log',180)
    result=dict(complete=stage['status']=='complete',stage=stage,validated=False,
        gpu_used=False,model_training_launched=False,scope=protocol['scope'])
    if result['complete']:
        rows=checked_hits(ROOT/'alignment/verified.tsv');actual={(r[0],r[1]):r for r in rows}
        missing=sorted(set(expected)-set(actual));bad=sorted({r[0] for r in negatives}&{r[0] for r in rows})
        # E-values were rounded in the TSV; compare within 0.5% relative error.
        discrepancies=[abs(float(actual[k][5])-float(v[5]))/max(float(v[5]),1e-300) for k,v in expected.items() if k in actual]
        max_error=max(discrepancies,default=0.)
        result.update(validated=not missing and not bad and max_error<=.005,
            reference_pairs=len(expected),recovered_reference_pairs=len(set(expected)&set(actual)),
            missing_reference_pairs=missing,accepted_shuffle_controls=bad,max_relative_evalue_error=max_error,
            alignment_sha256=digest(ROOT/'alignment/verified.tsv'))
    write_json(ROOT/'results_report.json',result)
    write_json(ROOT/'state.json',dict(status='complete' if result['complete'] else 'incomplete'))
    (ROOT/'findings.md').write_text('# Witness revalidation\n\n'+
        f'CPU sparse-realignment validation passed: {result["validated"]}.\n'+
        'Full-target database is used for final E-values. No GPU or model training was used.\n')


if __name__=='__main__':
    import sys
    if '--worker' in sys.argv:
        verify_candidates(dict(mmseqs=BINARY),ROOT/'alignment',PRIOR/'queries.fa',TARGET,ROOT/'candidates.tsv')
    else:main()
