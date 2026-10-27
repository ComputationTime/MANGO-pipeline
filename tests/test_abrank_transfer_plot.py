import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow'/'scripts'/'analysis'))
from plot_abrank_transfer import plot


class TransferPlotTests(unittest.TestCase):
    def test_primary_and_secondary_are_separate_and_cohort_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tags=['esm2','control_constant','control_random',
                'control_constant_l1','control_random_l1','control_pooled_esm2',
                'control_shuffled','control_cross_attention_esm2'];paths=[]
            for i,tag in enumerate(tags):
                payload=dict(embedder=tag,cohort_sha256='same',run_id=tag,architecture='prefix',
                             checkpoint_epoch=1,initialization={'sha256':'source'},splits={})
                for split,n in [('train',10),('test_cath_assigned',4),('test',6)]:
                    payload['splits'][split]=dict(nll=.5+i*.1,perplexity=2,n_examples=n,n_tokens=n*100)
                path=root/f'{tag}.json';path.write_text(json.dumps(payload));paths.append(path)
            plot(paths,'sabdab2',tags,{t:t for t in tags},root/'fine.png',root/'fine.csv',
                 {'secondary_only_test_records':1},dpi=60)
            import csv
            with (root/'fine.csv').open() as fh:
                rows=list(csv.DictReader(fh))
            self.assertEqual(len(rows),3*len(tags))
            self.assertEqual({r['split'] for r in rows},{'train','test_cath_assigned','test'})
            self.assertTrue((root/'fine.png').is_file())
            wrong=json.loads(paths[-1].read_text());wrong['cohort_sha256']='other'
            paths[-1].write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'different cohorts'):
                plot(paths,'sabdab2',tags,{t:t for t in tags},root/'bad.png',root/'bad.csv',{})
            self.assertFalse((root/'bad.png').exists())


if __name__=='__main__': unittest.main()
