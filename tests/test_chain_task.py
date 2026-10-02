"""Task contract checks across training, evaluation and downstream analysis."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'workflow/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / 'analysis'))
import train_mango
import evaluate_mango
import select_cohort
import structure_confidence_common


class ChainTaskTests(unittest.TestCase):
    def test_targets_expand_only_after_split_and_keep_chain_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / 'records.csv'
            pd.DataFrame([
                dict(id='train', split='train', resolved_H_seq='ACD', resolved_L_seq='EF'),
                dict(id='heldout', split='test', resolved_H_seq='GH', resolved_L_seq='IK'),
            ]).to_csv(records, index=False)
            rows = train_mango._rows(records, ['train'], ['heavy', 'light'])
            self.assertEqual([(r.id, r.chain, r.sequence) for r in rows],
                             [('train', 'heavy', 'ACD'), ('train', 'light', 'EF')])

    def test_evaluation_uses_selected_target_and_only_antigen_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            ag = Path(tmp) / 'antigen/one_hot/test/a.pt'
            ag.parent.mkdir(parents=True)
            torch.save({'embedding': torch.zeros(2, 21)}, ag)
            row = SimpleNamespace(id='a', split='test', resolved_H_seq='ACD', resolved_L_seq='EF')
            selected = []
            model = SimpleNamespace(
                target_token_ids=lambda seq: selected.append(seq) or torch.zeros(1, len(seq)+2, dtype=torch.long),
                n_target_tokens=lambda ids: ids.shape[1]-1,
                loss=lambda antigen, chain, ids: torch.tensor(2.0 if chain == 'light' else 3.0),
            )
            result = evaluate_mango._split_nll(model, [row], tmp, 'one_hot', 'light', 'cpu')
            self.assertEqual(selected, ['EF'])
            self.assertEqual(result, (2.0, 3, 1, 0))

    def test_light_designs_cannot_enter_heavy_only_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            designs = Path(tmp) / 'designs.csv'
            records = Path(tmp) / 'records.csv'
            pd.DataFrame([dict(embedder='one_hot', run_id='run', target_id='a',
                               design_index=0, sequence='ACD', status='ok', chain_type='light')]).to_csv(designs, index=False)
            pd.DataFrame([dict(id='a', resolved_L_seq='EF')]).to_csv(records, index=False)
            with self.assertRaisesRegex(ValueError, 'heavy-chain designs only'):
                select_cohort.select_cohort([designs], records, 'one_hot', 1, 0, Path(tmp)/'cohort.csv')
            with self.assertRaisesRegex(ValueError, 'heavy-chain designs only'):
                structure_confidence_common.select_designs(designs, 1, 0)


if __name__ == '__main__':
    unittest.main()
