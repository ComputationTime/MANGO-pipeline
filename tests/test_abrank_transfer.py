import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflow'/'scripts'))
from prepare_abrank_transfer import components, assign_splits, assert_disjoint, select_holdout


def row(identifier, family, heavy=None, light=None, source_group=None):
    return dict(id=identifier, cath_families=family,
                resolved_H_seq=heavy or 'heavy_'+identifier,
                resolved_L_seq=light or 'light_'+identifier,
                source_ab_ag_cluster=source_group or identifier,
                source_antigen_cluster='', ab_ag_cluster='', split='')


class SplitTests(unittest.TestCase):
    def test_multidomain_family_connections_are_transitive(self):
        rows=[row('a','1.1.1.1'),row('b','1.1.1.1;2.2.2.2'),row('c','2.2.2.2'),row('d','3.3.3.3')]
        self.assertEqual(sorted(len(g) for g in components(rows)),[1,3])

    def test_source_groups_and_exact_chains_do_not_cross_splits(self):
        rows=[row('a','1',heavy='shared'),row('b','2',heavy='shared'),
              row('c','3',source_group='same'),row('d','4',source_group='same')]
        self.assertEqual(sorted(len(g) for g in components(rows,True)),[2,2])

    def test_imbalanced_split_retains_training_family_diversity(self):
        rows=[row(str(i),'large') for i in range(100)]
        rows += [row(f'small_{i}',f'family_{i}') for i in range(40)]
        first=assign_splits(copy.deepcopy(rows),seed=0)
        self.assertEqual(first,assign_splits(copy.deepcopy(rows),seed=0))
        by_split={s:[r for r in first if r['split']==s] for s in ['train','val','test']}
        self.assertGreater(len({r['cath_families'] for r in by_split['train']}),20)
        self.assertTrue(all(by_split.values()))
        for a,b in [('train','test'),('train','val'),('test','val')]:
            assert_disjoint(by_split[a],by_split[b])

    def test_individual_target_overlap_fails_audit(self):
        left=row('a','1',light='shared');left['ab_ag_cluster']='a'
        right=row('b','2',light='shared');right['ab_ag_cluster']='b'
        with self.assertRaisesRegex(ValueError,'Contamination'):
            assert_disjoint([left],[right])

    def test_holdout_component_cap_is_respected(self):
        groups=[[row(f'{g}_{i}',str(g)) for i in range(g)] for g in range(1,9)]
        chosen,remaining=select_holdout(groups,10,seed=4,max_groups=2)
        self.assertLessEqual(len(chosen),2)
        self.assertEqual(sum(map(len,chosen)),10)
        self.assertEqual(len(chosen)+len(remaining),len(groups))


class TransferTests(unittest.TestCase):
    def test_loads_all_matching_weights_and_validates_task_and_control(self):
        import torch
        from train_mango import initialize_from_checkpoint
        class Model(torch.nn.Module):
            TASK='antigen+chain_type->chain'
            architecture='antigen_chain_gpt2_v3'
            d_ag=3
            target_chains=('heavy','light')
            def __init__(self):
                super().__init__()
                self.chain_embedding=torch.nn.Embedding(2,4)
                self.antigen_projection=torch.nn.Linear(3,4)
                self.lm=torch.nn.Linear(4,5)
        source=Model();target=Model()
        checkpoint=dict(architecture=source.architecture,task=source.TASK,d_ag=3,
                        target_contract='imgt_variable_vh_vl_v1',target_chains=['heavy','light'],
                        embedder='constant',embedding_tag='constant',cohort_sha256='source',
                        run_id='pretrained',epoch=2,optimizer={'should_not':'load'})
        for name in ['chain_embedding','antigen_projection','lm']:
            checkpoint[name]=getattr(source,name).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'best.pt';torch.save(checkpoint,path)
            provenance=initialize_from_checkpoint(target,path,'constant','constant')
            self.assertFalse(provenance['optimizer_loaded'])
            self.assertEqual(provenance['source_cohort_sha256'],'source')
            for a,b in zip(source.parameters(),target.parameters()):
                self.assertTrue(torch.equal(a,b))
            for key,value in [('embedder','random'),('d_ag',4),('architecture','cross'),
                              ('target_contract','full_chains'),('task','other')]:
                wrong=dict(checkpoint);wrong[key]=value;torch.save(wrong,path)
                with self.assertRaisesRegex(ValueError,key):
                    initialize_from_checkpoint(target,path,'constant','constant')


if __name__=='__main__':
    unittest.main()
