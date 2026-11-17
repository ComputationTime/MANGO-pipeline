"""Meaningful failure checks for the grouping and measurement provenance audit."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'workflow/scripts'))
from audit_affinity_bottleneck import canonical_candidate, edit_distance, exclusion_paths, graph_edges, parse_mutations, partition
from review_affinity_lineage_bridges import cdr3_labels, traverse_components


class BottleneckAuditTests(unittest.TestCase):
    def test_propagation_witness_keeps_typed_bridge(self):
        rows=[dict(id='a'),dict(id='b'),dict(id='c')]
        edges=[('row:a','cath:x','cath_family'),('row:b','cath:x','cath_family'),
               ('row:b','sequence:s','light_sequence'),('row:c','sequence:s','light_sequence')]
        self.assertEqual(partition(rows,edges),[('a','b','c')])
        self.assertEqual(partition(rows,edges,{'cath_family'}),[('a','b'),('c',)])
        paths={r['id']:r for r in exclusion_paths(edges,{'a'})}
        self.assertEqual(paths['c']['witness_id'],'a')
        self.assertEqual(paths['c']['edge_types'],'cath_family;cath_family;light_sequence;light_sequence')
        self.assertFalse(paths['c']['direct'])

    def test_source_groups_cannot_disappear_in_diagnostics(self):
        rows=[dict(id='a'),dict(id='b')]
        edges=[('row:a','source:1','source_cluster'),('row:b','source:1','source_cluster')]
        self.assertEqual(partition(rows,edges,{'source_cluster'}),[('a','b')])

    def test_alignment_below_frozen_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError,'threshold'):
            graph_edges([], [('a','b','.89','1','1','0')])

    def test_reciprocal_alignment_is_one_edge(self):
        edges=graph_edges([], [('a','b','.9','.8','.8','0'),('b','a','.9','.8','.8','0')])
        self.assertEqual(edges,[('sequence:a','sequence:b','near_antibody')])

    def candidate(self, rid='x', antigen='AG'):
        link=dict(affinity_molar=1e-9,pmid='12345',affinity_method='SPR',temperature='25')
        row=dict(id=rid,pdb_id='pdb_00001abc',heavy_chain='H',light_chain='L',
                 expected_heavy_seq='VH',expected_light_seq='VL',expected_ag_seq=antigen)
        source=dict(pdb='1abc',Hchain='H',Lchain='L',affinity='1e-9',pmid='12345',affinity_method='SPR',temperature='25')
        return link,row,source

    def test_crystal_copies_do_not_become_replicates(self):
        a=self.candidate('copy1');b=self.candidate('copy2')
        self.assertEqual(canonical_candidate(*a)[0],canonical_candidate(*b)[0])

    def test_missing_full_antigen_prevents_overcollapse(self):
        a=self.candidate('copy1','');b=self.candidate('copy2','')
        self.assertNotEqual(canonical_candidate(*a)[0],canonical_candidate(*b)[0])

    def test_wrong_chain_or_label_link_rejected(self):
        link,row,source=self.candidate();source['Hchain']='L'
        with self.assertRaisesRegex(ValueError,'identify'):
            canonical_candidate(link,row,source)
        link,row,source=self.candidate();source['affinity']='2e-9'
        with self.assertRaisesRegex(ValueError,'affinity changed'):
            canonical_candidate(link,row,source)

    def test_raw_mutation_chain_and_insertion_number_retained(self):
        self.assertEqual(parse_mutations('YH100AF;SL-2A'),[('Y','H','100A','F'),('S','L','-2','A')])
        with self.assertRaises(ValueError):
            parse_mutations('delH100')

    def test_cdr3_insertions_are_retained_in_distance(self):
        self.assertEqual(edit_distance('ARDYY','ARDDYY'),1)
        self.assertEqual(edit_distance('ARDYY','ARGYY'),1)

    def test_autoregressive_eos_is_removed_explicitly(self):
        self.assertEqual(cdr3_labels(dict(sequence='ACD',labels=['framework','cdr3','cdr3','eos'])),'CD')
        self.assertEqual(cdr3_labels(dict(sequence='ACD',labels=['framework','cdr3','cdr3'])),'CD')
        with self.assertRaisesRegex(ValueError,'align'):
            cdr3_labels(dict(sequence='ACD',labels=['framework','cdr3','cdr3','unknown']))

    def test_independent_bfs_counts_only_records(self):
        graph={'row:a':{'factor'},'factor':{'row:a','row:b'},'row:b':{'factor'},'row:c':set()}
        self.assertEqual(traverse_components(graph,{'row:a','row:b','row:c'}),[('a','b'),('c',)])


if __name__=='__main__':
    unittest.main()
