import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workflow/scripts'))
from Bio.PDB import Structure,Model,Chain,Residue,Atom,MMCIFIO
from interface_supervision_common import antigen_input,contact_labels


class AntigenInputSeparationTests(unittest.TestCase):
    def test_changing_antibody_coordinates_changes_only_labels(self):
        structure=Structure.Structure('test');model=Model.Model(0);structure.add(model)
        for chain in ('A','B'):model.add(Chain.Chain(chain))
        for chain,name,num,x in [('A','ALA',1,0.),('A','CYS',2,10.),('B','GLY',15,4.)]:
            residue=Residue.Residue((' ',num,' '),name,' ');residue.add(Atom.Atom('CA',np.array([x,0.,0.]),1.,1.,' ',' CA ',num,element='C'));model[chain].add(residue)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input.cif';writer=MMCIFIO();writer.set_structure(structure);writer.save(str(path))
            row=dict(pdb_path=str(path),antigen_chains='A',resolved_ag_seq='AC',heavy_chain='B',resolved_H_seq='G',light_chain='',resolved_L_seq='')
            before=antigen_input(row);labels=contact_labels(row)
            model['B'][(' ',15,' ')]['CA'].coord+=100;writer.set_structure(structure);writer.save(str(path))
            after=antigen_input(row);changed=contact_labels(row)
        np.testing.assert_array_equal(before['coords'],after['coords']);np.testing.assert_array_equal(before['keep'],after['keep'])
        self.assertEqual(int(labels['heavy'].sum()),1);self.assertEqual(int(changed['heavy'].sum()),0)


if __name__=='__main__':unittest.main()
