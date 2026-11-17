"""Freeze finite measured pairs and only the chain/antigen comparisons they require."""
import argparse
import json
from pathlib import Path
import sqlite3

from finalize_openalphaseq_pairing import digest, finalize, write_json


def main(study_path):
    study = json.loads(Path(study_path).read_text())
    root = Path(study['experiment_root'])
    finalize(study)
    temporary = root / 'measurements.tmp.sqlite'
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.execute('ATTACH DATABASE ? AS raw', (str(root / 'raw.sqlite'),))
        db.execute('ATTACH DATABASE ? AS cohort', (str(root / 'cohort.sqlite'),))
        db.executescript('''
        CREATE TABLE pairs AS
        SELECT m.query_id,o.antigen_id,
            COUNT(*) AS all_count,SUM(o.affinity) AS all_sum,
            SUM(o.sufficient_replicates=1) AS reliable_count,
            SUM(CASE WHEN o.sufficient_replicates=1 THEN o.affinity ELSE 0 END) AS reliable_sum,
            SUM(o.sufficient_replicates=1 AND o.above_background=1) AS positive_count,
            SUM(CASE WHEN o.sufficient_replicates=1 AND o.above_background=1
                     THEN o.affinity ELSE 0 END) AS positive_sum
        FROM raw.observations o JOIN cohort.raw_map m ON o.antibody_id=m.raw_id
        WHERE o.affinity IS NOT NULL
        GROUP BY m.query_id,o.antigen_id;
        CREATE UNIQUE INDEX pairs_key ON pairs(query_id,antigen_id);
        CREATE TABLE required AS
        SELECT DISTINCT t.id AS target_id,p.antigen_id,t.chain,t.sequence,t.ntokens
        FROM pairs p JOIN cohort.queries q ON p.query_id=q.id
        JOIN cohort.targets t ON (t.id=q.heavy AND q.heavy_novel=1)
                              OR (t.id=q.light AND q.light_novel=1)
        WHERE t.seen_training=0;
        CREATE UNIQUE INDEX required_key ON required(antigen_id,chain,target_id);
        ''')
        counts = dict(measured_pairs=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
                      required_chain_antigen_scores=db.execute('SELECT COUNT(*) FROM required').fetchone()[0],
                      antigens=db.execute('SELECT COUNT(DISTINCT antigen_id) FROM required').fetchone()[0],
                      novel_chains=db.execute('SELECT COUNT(DISTINCT target_id) FROM required').fetchone()[0])
    temporary.replace(root / 'measurements.sqlite')
    write_json(root / 'measurements_complete.json', dict(complete=True, **counts,
        measurements_sha256=digest(root / 'measurements.sqlite'),
        cohort_sha256=digest(root / 'cohort.sqlite'),
        training_records_sha256=digest(study['training_records']),
        implementation_sha256=digest(__file__)))
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-path', default='artifacts/experiments/openalphaseq_affinity_full_v1/study.json')
    main(**vars(parser.parse_args()))
