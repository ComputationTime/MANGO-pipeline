"""Paired antigen ablations on audited pilot checkpoints; never changes them."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def balanced_records(rows, limit, seed):
    """Sample whole records for inspection within their existing train split."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['ab_ag_cluster']].append(row)
    rng = random.Random(seed)
    buckets = [list(groups[key]) for key in sorted(groups)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    selected = []
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
    return selected


def wrong_antigen_map(rows, seed):
    """Replace antigen identity within split, avoiding the same sequence/group.

    Each original antigen has one deterministic donor. Nearest-length candidates
    reduce length confounding. Prefix probes additionally resample donor memory
    to the original token count so antibody absolute positions stay identical.
    """
    selected, by_split = {}, defaultdict(dict)
    for row in rows:
        by_split[row['split']].setdefault(row['resolved_ag_seq'], row)
    rng = random.Random(seed)
    for split, antigens in sorted(by_split.items()):
        for sequence, row in sorted(antigens.items()):
            candidates = [other for seq, other in antigens.items()
                          if seq != sequence and other['ab_ag_cluster'] != row['ab_ag_cluster']]
            candidates.sort(key=lambda other: (abs(len(other['resolved_ag_seq']) - len(sequence)), other['id']))
            if not candidates:
                raise ValueError(f'No distinct-family antigen donor for {split}/{row["id"]}')
            selected[split, sequence] = rng.choice(candidates[:8])
    return {row['id']: selected[row['split'], row['resolved_ag_seq']] for row in rows}


@lru_cache(maxsize=4096)
def memory(path):
    from model_common import load_embedding
    return load_embedding(path)


def embedding_path(study, row, dataset='sabdab2'):
    return str(Path(study['mango_root']) / dataset / 'embeddings/antigen/esm2' /
               row['split'] / f'{row["id"]}.pt')


def mango_token_nll(model, h_ag, chain, sequence, mode='correct', rebase=False,
                    standard_causal_mask=False):
    """Expose exactly the production loss's residue+EOS logits for diagnosis."""
    import torch
    import torch.nn.functional as F
    target = model.target_token_ids(sequence).to(next(model.parameters()).device)
    if mode == 'no_context' and not model.uses_cross_attention:
        h_ag = h_ag[:, :0]
    prefix = model._prefix(h_ag, chain)
    inputs = torch.cat([prefix, model.lm.transformer.wte(target[:, 1:])], dim=1)
    model._check_length(inputs.shape[1])
    context = {} if mode == 'no_context' else model._antigen_context(h_ag)
    rebase = rebase or getattr(model, 'uses_fixed_positions', False)
    if rebase:
        if model.uses_cross_attention:
            raise ValueError('Relative target positions only apply to prefix models')
        positions = torch.cat([torch.arange(h_ag.shape[1], device=inputs.device),
                               torch.arange(inputs.shape[1] - h_ag.shape[1], device=inputs.device)])
        context['position_ids'] = positions.unsqueeze(0)
        # Transformers 4.57 infers packed sequences from discontinuous position
        # IDs if no attention mask is supplied. Explicitly mark one sequence so
        # its ordinary causal mask still lets every antibody token see antigen.
        standard_causal_mask = True
    if standard_causal_mask:
        context['attention_mask'] = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.long)
    logits = model.lm(inputs_embeds=inputs, use_cache=False, **context).logits
    scored = logits[:, prefix.shape[1] - 1:-1].reshape(-1, logits.shape[-1])
    return F.cross_entropy(scored, target[:, 1:].reshape(-1), reduction='none')


def iglm_token_nll(model, chain, sequence, h_ag=None):
    import torch.nn.functional as F
    ids, _ = model.encode(sequence, chain)
    context = {'encoder_hidden_states': model.antigen_projection(h_ag)} if h_ag is not None else {}
    logits = model.lm(input_ids=ids, use_cache=False, **context).logits
    return F.cross_entropy(logits[:, 1:-1].reshape(-1, logits.shape[-1]), ids[:, 2:].reshape(-1), reduction='none')


def load_probe(study, name, device):
    import torch
    import model_common as mc
    import iglm_baseline as iglm
    spec = study['probes'][name]
    if spec['family'] == 'iglm':
        directory = Path(study['iglm_root']) / spec['phase']
    else:
        directory = Path(study['mango_root']) / spec['phase'] / 'cross_entropy' / spec['architecture']
    path = directory / 'checkpoints/best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if spec['family'] == 'iglm':
        metadata = checkpoint['metadata']
        model_spec = dict(initialization='random_from_scratch', objective=metadata['objective'],
                          species_token=metadata['species_token'])
        model = iglm.IgLMScratch(model_spec, metadata['model_config'])
        model.lm.load_state_dict(checkpoint['lm'])
        model.lm.to(device).eval()
    else:
        metadata = checkpoint
        model = mc.MangoModel.from_checkpoint(checkpoint).to(device).eval()
    expected = mc.comparison_cohort(Path(study['cohort_dir']) / 'sabdab2_records.csv')
    if metadata['cohort_sha256'] != expected:
        raise ValueError('Probe checkpoint does not match the audited native cohort')
    audit = dict(checkpoint=str(path), checkpoint_sha256=sha256(path), cohort_sha256=expected,
                 decoder_config=model.lm.config.to_dict(),
                 trainable_parameters=sum(p.numel() for p in model.lm.parameters()),
                 official_eval=str(directory / 'eval.json'))
    return model, audit


def probe(study, name):
    import torch
    import torch.nn.functional as F
    from device_common import get_device
    from chain_examples import example_chains
    from model_common import comparison_cohort

    started = time.monotonic()
    torch.set_num_threads(2)
    device = get_device('antigen-conditioning diagnostics')
    root = Path(study['experiment_root']) / 'probes' / name
    root.mkdir(parents=True, exist_ok=True)
    records_path = Path(study['cohort_dir']) / 'sabdab2_records.csv'
    digest = comparison_cohort(records_path)
    all_rows = read_csv(records_path)
    split_audit = json.loads((Path(study['cohort_dir']) / 'split_audit.json').read_text())
    if sha256(records_path) != split_audit['sha256'][str(records_path)]:
        raise ValueError('Prepared native records differ from split audit')
    train = balanced_records([r for r in all_rows if r['split'] == 'train'],
                             study['train_probe_records'], study['seed'])
    rows = train + [r for r in all_rows if r['split'] != 'train']
    model, audit = load_probe(study, name, device)
    donors = [wrong_antigen_map(all_rows, 1000 + study['seed'] + i)
              for i in range(study['wrong_antigen_repeats'])]
    family = study['probes'][name]['family']
    variants = ['correct']
    if family == 'mango':
        variants += [f'wrong_{i}' for i in range(len(donors))] + ['no_context', 'pooled']
    donor_audit = {str(i): {r['id']: donors[i][r['id']]['id'] for r in rows} for i in range(len(donors))}
    atomic_json(root / 'inputs.json', dict(study=study, probe=name, source=audit,
                train_probe_ids=[r['id'] for r in train], wrong_antigen_donors=donor_audit,
                wrong_antigen_protocol='same split, distinct antigen sequence and ab_ag_cluster; nearest-length candidates; prefix length preserved by linear interpolation'))
    totals = defaultdict(lambda: [0., 0, 0])
    temporary = root / 'token_scores.jsonl.tmp'
    with temporary.open('w', buffering=1) as handle, torch.no_grad():
        for index, row in enumerate(rows):
            correct = memory(embedding_path(study, row)).to(device) if family == 'mango' else None
            for variant in variants:
                donor = None
                h_ag = correct
                if variant.startswith('wrong_'):
                    donor = donors[int(variant.split('_')[1])][row['id']]
                    h_ag = memory(embedding_path(study, donor)).to(device)
                    if not model.uses_cross_attention and h_ag.shape[1] != correct.shape[1]:
                        h_ag = F.interpolate(h_ag.transpose(1, 2), size=correct.shape[1],
                                             mode='linear', align_corners=False).transpose(1, 2)
                elif variant == 'pooled':
                    if model.uses_cross_attention:
                        h_ag = correct.mean(dim=1, keepdim=True)
                    else:
                        h_ag = correct.mean(dim=1, keepdim=True).expand_as(correct)
                for chain in example_chains(row):
                    sequence = row['resolved_H_seq' if chain == 'heavy' else 'resolved_L_seq']
                    loss = (iglm_token_nll(model, chain, sequence) if family == 'iglm' else
                            mango_token_nll(model, h_ag, chain, sequence, variant))
                    tokens = loss.cpu().tolist()
                    if len(tokens) != len(sequence) + 1 or not torch.isfinite(loss).all():
                        raise ValueError('Invalid per-token diagnostic loss')
                    value = dict(model=name, variant=variant, id=row['id'], split=row['split'],
                                 chain=chain, cluster=row['ab_ag_cluster'],
                                 antigen_length=len(row['resolved_ag_seq']), sequence=sequence,
                                 donor_id=donor['id'] if donor else None,
                                 donor_antigen_length=len(donor['resolved_ag_seq']) if donor else None,
                                 token_nll=tokens)
                    handle.write(json.dumps(value, separators=(',', ':')) + '\n')
                    tally = totals[row['split'], variant]
                    tally[0] += sum(tokens)
                    tally[1] += len(tokens)
                    tally[2] += 1
            if (index + 1) % 100 == 0:
                print(f'{name}: {index+1}/{len(rows)} records', flush=True)
    temporary.replace(root / 'token_scores.jsonl')
    summary = [dict(split=split, variant=variant, nll=total / tokens, n_tokens=tokens,
                    n_examples=examples) for (split, variant), (total, tokens, examples) in totals.items()]
    official = json.loads(Path(audit['official_eval']).read_text())
    for split in ('val', 'test'):
        measured = next(s for s in summary if s['split'] == split and s['variant'] == 'correct')
        expected = official['splits'][split]
        if (abs(measured['nll'] - expected['nll']) > 2e-6 or
                measured['n_tokens'] != expected['n_tokens'] or measured['n_examples'] != expected['n_examples']):
            raise ValueError('Per-token diagnostic scorer does not reproduce official NLL')
    atomic_json(root / 'complete.json', dict(status='complete', model=name, source=audit,
                cohort_sha256=digest, source_records_sha256=sha256(records_path),
                token_scores_sha256=sha256(root / 'token_scores.jsonl'),
                wall_seconds=time.monotonic() - started, summary=summary))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    smk = globals().get('snakemake')
    if smk is not None:
        import contextlib
        import traceback
        Path(smk.log[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(smk.log[0], 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                probe(dict(smk.params.study), str(smk.wildcards.model))
            except Exception:
                traceback.print_exc()
                raise
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/antigen_diagnostics.yaml')
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    import yaml
    probe(yaml.safe_load(Path(args.config).read_text()), args.model)


if __name__ == '__main__':
    main()
