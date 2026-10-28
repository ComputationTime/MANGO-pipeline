"""Fair-membership checks for the ten-model prototype comparison."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts/analysis'))
from plot_asd_loss_comparison import plot


class ASDWithIgLMPlotTests(unittest.TestCase):
    def fixtures(self, root):
        mango, iglm = [], []
        score = dict(nll=1., perplexity=2.718281828, n_examples=2, n_tokens=7,
                     by_chain={'heavy': dict(n_examples=1, n_tokens=4),
                               'light': dict(n_examples=1, n_tokens=3)})
        splits = {split: score for split in ('train', 'val', 'test_cath_assigned', 'test')}
        for phase in ('sabdab2_scratch', 'sabdab2_finetuned'):
            transferred = phase == 'sabdab2_finetuned'
            for objective in ('cross_entropy', 'focal'):
                for architecture in ('mango', 'cross_attention'):
                    path = root / phase / objective / architecture / 'eval.json'
                    path.parent.mkdir(parents=True)
                    payload = dict(experiment_phase=phase, objective=objective,
                                   architecture_label=architecture, cohort_sha256='same',
                                   evaluation_loss='cross_entropy', embedding_tag='esm2',
                                   initialization={'sha256': 'asd'} if transferred else None,
                                   splits=splits, run_id=path.parent.name, training_loss={'name': objective})
                    path.write_text(json.dumps(payload))
                    mango.append(path)
            path = root / 'iglm' / phase / 'eval.json'
            path.parent.mkdir(parents=True)
            payload = dict(experiment_phase=phase, cohort_sha256='same', model_family='IgLM',
                           antigen_conditioning=False, public_pretrained_weights_loaded=False,
                           antibody_prompt_used=False, target_chains=['heavy', 'light'],
                           training_loss={'name': 'cross_entropy'}, evaluation_loss='cross_entropy',
                           checkpoint_selection='validation_nll', loss_reduction='target_token_mean_including_eos',
                           transfer_initialization={'sha256': 'iglm_asd'} if transferred else None,
                           splits=splits, run_id='iglm_' + phase)
            path.write_text(json.dumps(payload))
            iglm.append(path)
        return mango, iglm

    def test_both_routes_added_and_original_eight_model_plot_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mango, iglm = self.fixtures(root)
            for paths, count in ((None, 32), (iglm, 40)):
                plot(mango, root / 'figure.png', root / 'table.csv', iglm_paths=paths)
                with (root / 'table.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), count)
                self.assertEqual(sum(r['architecture'] == 'iglm' for r in rows), count - 32)
                self.assertTrue((root / 'figure.png').is_file())

    def test_different_cohort_chain_counts_or_scoring_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mango, iglm = self.fixtures(root)
            original = iglm[0].read_text()
            for override in ('cohort', 'counts', 'chain_counts', 'loss', 'pretrained'):
                payload = json.loads(original)
                if override == 'cohort':
                    payload['cohort_sha256'] = 'different'
                elif override == 'counts':
                    payload['splits']['test']['n_tokens'] += 1
                elif override == 'chain_counts':
                    payload['splits']['test']['by_chain']['heavy']['n_tokens'] += 1
                elif override == 'loss':
                    payload['evaluation_loss'] = 'focal'
                else:
                    payload['public_pretrained_weights_loaded'] = True
                iglm[0].write_text(json.dumps(payload))
                with self.subTest(override=override), self.assertRaises(ValueError):
                    plot(mango, root / 'bad.png', root / 'bad.csv', iglm_paths=iglm)
            iglm[0].write_text(original)
            with self.assertRaisesRegex(ValueError, 'Exactly two'):
                plot(mango, root / 'bad.png', root / 'bad.csv', iglm_paths=[iglm[0], iglm[0]])


if __name__ == '__main__':
    unittest.main()
