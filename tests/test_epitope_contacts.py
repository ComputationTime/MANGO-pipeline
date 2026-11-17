import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
from Bio.PDB import Structure,Model,Chain,Residue,Atom,MMCIFIO
from epitope_diagnostics_common import contact_annotations


class ContactMappingTests(unittest.TestCase):
    def test_variable_domain_author_numbering_hydrogens_and_chain_break(self):
        structure=Structure.Structure('test');model=Model.Model(0);structure.add(model)
        for name in ('A','B','C'):model.add(Chain.Chain(name))
        def residue(chain,name,number,x,insertion=' '):
            r=Residue.Residue((' ',number,insertion),name,' ')
            r.add(Atom.Atom('CA',np.array([x,0.,0.]),1.,1.,' ',' CA ',number,element='C'))
            model[chain].add(r);return r
        residue('A','ALA',11,0.);residue('A','CYS',18,10.)
        residue('B','ALA',100,0.) # Outside selected target: must not create a contact.
        residue('B','GLY',101,4.5,'A');y=residue('B','TYR',202,30.)
        y.add(Atom.Atom('H',np.array([10.,0.,0.]),1.,1.,' ',' H  ',300,element='H'))
        residue('C','ASP',5,2.)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'test.cif';writer=MMCIFIO();writer.set_structure(structure);writer.save(str(path))
            row=dict(id='test',pdb_path=str(path),heavy_chain='B',resolved_H_seq='GY',antigen_chains='A,C',resolved_ag_seq='AC,D')
            result=contact_annotations(row,[4.,5.,8.])
        self.assertEqual(result['embedding_length'],4)
        self.assertEqual(result['contacts']['5.0']['epitope'],[0,3])
        self.assertEqual(result['contacts']['5.0']['paratope'],[0])
        self.assertEqual(result['contacts']['4.0']['epitope'],[3])
        self.assertEqual(result['contacts']['8.0']['epitope'],[0,1,3])
        self.assertTrue(result['antigen_residue_mapping'][2]['chain_break'])


if __name__=='__main__':unittest.main()
