"""Real cached examples for software checks; never scientific eligibility."""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[2] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import argparse
from pathlib import Path
import yaml
from affinity_study_common import digest,read_csv,write_csv,write_json


def main(config):
    cfg=yaml.safe_load(Path(config).read_text());root=Path(cfg['experiment_root'])
    source=Path('artifacts/experiments/affinity_improvement_smoke_v1/pool.csv')
    rows=read_csv(source)
    for r in rows:r.update(biological_group=r['ab_ag_cluster'],source_split=r['split'])
    write_csv(root/'records.csv',rows)
    write_json(root/'cohort_audit.json',dict(complete=True,feasible=True,smoke_only=True,
        reason='software_fixture_only_not_a_protected_scientific_cohort',source_sha256=digest(source),
        files_sha256={'records.csv':digest(root/'records.csv')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(**vars(p.parse_args()))
