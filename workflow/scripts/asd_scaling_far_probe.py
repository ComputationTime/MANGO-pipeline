"""ASD inference controls using donors from disjoint CATH families/components."""
import contextlib
import json
from pathlib import Path
import sys
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antigen_diagnostics import atomic_json, mango_token_nll, read_csv, sha256, wrong_antigen_map
from asd_scaling_job import validate
from chain_examples import example_chains
from device_common import get_device
from model_common import MangoModel, load_embedding


def probe(study, run):
    torch.set_num_threads(2)
    spec = study['runs'][run]
    _, location = validate(study, spec['size'])
    root = Path(study['experiment_root'])
    directory = root / 'runs' / run
    ready = json.loads((directory / 'train_complete.json').read_text())
    if ready['checkpoint_sha256'] != sha256(ready['checkpoint']):
        raise ValueError('ASD far-probe checkpoint changed')
    checkpoint = torch.load(ready['checkpoint'], map_location='cpu', weights_only=False)
    model = MangoModel.from_checkpoint(checkpoint).to(get_device('ASD different-family antigen probes')).eval()
    rows = read_csv(root / 'evaluation_records.csv')
    donors = [wrong_antigen_map(rows, 1000 + study['seed'] + i) for i in range(2)]
    with (directory / 'far_token_scores.jsonl.tmp').open('w') as handle, torch.no_grad():
        for row in rows:
            for i, mapping in enumerate(donors):
                donor = mapping[row['id']]
                if (set(row['cath_families'].split(';')) & set(donor['cath_families'].split(';')) or
                        row['split'] != donor['split'] or row['ab_ag_cluster'] == donor['ab_ag_cluster']):
                    raise ValueError('Far-probe donor shares a CATH family/group or crosses split')
                path = Path(location['embeddings']) / 'antigen/esm2' / donor['split'] / f'{donor["id"]}.pt'
                memory = load_embedding(str(path)).to(next(model.parameters()).device)
                if not model.uses_cross_attention and memory.shape[1] != len(row['resolved_ag_seq']):
                    memory = torch.nn.functional.interpolate(memory.transpose(1, 2), size=len(row['resolved_ag_seq']),
                        mode='linear', align_corners=False).transpose(1, 2)
                for chain in example_chains(row):
                    sequence = row['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']
                    losses = mango_token_nll(model, memory, chain, sequence)
                    handle.write(json.dumps(dict(model=run, variant=f'far_wrong_{i}', id=row['id'], split=row['split'],
                        chain=chain, cluster=row['ab_ag_cluster'], antigen_length=len(row['resolved_ag_seq']),
                        sequence=sequence, donor_id=donor['id'], token_nll=losses.cpu().tolist()), separators=(',', ':')) + '\n')
    (directory / 'far_token_scores.jsonl.tmp').replace(directory / 'far_token_scores.jsonl')
    atomic_json(directory / 'far_eval.json', dict(status='complete', checkpoint_sha256=ready['checkpoint_sha256'],
                heldout_sha256=sha256(root / 'evaluation_records.csv'),
                token_scores_sha256=sha256(directory / 'far_token_scores.jsonl'),
                protocol='two within-split donor draws; disjoint CATH families and ab_ag_cluster; nearest-length candidates; prefix models interpolate donors to preserve target positions'))


if __name__ == '__main__':
    smk = globals()['snakemake']
    Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            probe(dict(smk.params.study), str(smk.wildcards.run))
        except Exception:
            traceback.print_exc()
            raise
