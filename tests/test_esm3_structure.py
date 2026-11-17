"""ESM3 must receive aligned antigen structure without any antibody atoms."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflow/scripts'))
from embed_antigen_esm3 import antigen_structure, esm3
from embed_batch import batch_embed, _can_reuse
from prefetch_embedder_weights import _esm3


def write_cif(path, antibody_offset=0, antigen_offset=0):
    structure = Structure.Structure('fixture')
    model = Model.Model(0)
    structure.add(model)
    for chain_id, residues, offset in [('Ag_long', ['ALA', 'CYS'], antigen_offset),
                                        ('B', ['ASP'], 20), ('H', ['TRP'], 1000 + antibody_offset)]:
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for position, name in enumerate(residues, 1):
            residue = Residue.Residue((' ', position, ' '), name, ' ')
            chain.add(residue)
            for atom_id, atom in enumerate(['N', 'CA', 'C', 'O']):
                coords = np.array([offset + position * 4 + atom_id, atom_id % 2, 0.], dtype=np.float32)
                residue.add(Atom.Atom(atom, coords, 1., 1., ' ', f'{atom:>4}', position * 4 + atom_id, element=atom[0]))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))


class ESM3StructureTests(unittest.TestCase):
    def test_weight_prefetch_keeps_lazy_structure_encoder_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / 'weights.pth').write_bytes(b'asset-orchestration-test')
            model = Mock()
            model.parameters.return_value = [torch.zeros(1)]
            loader = Mock(return_value=model)
            module = ModuleType('esm.models.esm3')
            module.ESM3 = SimpleNamespace(from_pretrained=loader)
            modules = {'esm': ModuleType('esm'), 'esm.models': ModuleType('esm.models'), 'esm.models.esm3': module}
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {'HF_HOME': str(cache)}):
                _esm3('esm3-sm-open-v1', 'sequence_structure')
            self.assertEqual(loader.call_args.kwargs['device'].type, 'cpu')
            model.get_structure_encoder.assert_called_once()

    def fixture(self, root):
        path = root / 'complex.cif'
        write_cif(path)
        return dict(id='example', split='train', antigen_chains='Ag_long,B',
                    resolved_ag_seq='AC,D', pdb_path=str(path))

    def runtime(self, calls, structure=True):
        def encode(protein):
            calls.append(protein)
            return SimpleNamespace(protein=protein, structure=torch.ones(len(protein.sequence) + 2) if structure else None,
                                   coordinates=protein.coordinates)
        def logits(encoded, config):
            size = len(encoded.protein.sequence) + 2
            return SimpleNamespace(embeddings=torch.arange(size * 3).reshape(1, size, 3).float())
        return SimpleNamespace(encode=encode, logits=logits), SimpleNamespace, None

    def test_author_chains_atom37_alignment_and_antibody_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            sequence, coords, chains = antigen_structure(row, 'resolved')
            self.assertEqual(sequence, 'AC|D')
            self.assertEqual(chains, ['Ag_long', 'B'])
            self.assertEqual(tuple(coords.shape), (4, 37, 3))
            self.assertTrue(torch.isnan(coords[2]).all())
            self.assertTrue(torch.isnan(coords[:, 3]).all())  # CB must stay masked.
            self.assertTrue(torch.isfinite(coords[[0, 1, 3]][:, [0, 1, 2, 4]]).all())
            write_cif(row['pdb_path'], antibody_offset=1000)
            changed = antigen_structure(row, 'resolved')[1]
            torch.testing.assert_close(coords, changed, rtol=0, atol=0, equal_nan=True)
            write_cif(row['pdb_path'], antigen_offset=10)
            self.assertFalse(torch.equal(coords[0, 0], antigen_structure(row, 'resolved')[1][0, 0]))

    def test_structure_track_is_supplied_and_sequence_only_baseline_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            calls = []
            output = root / 'structural.pt'
            esm3('', row['id'], output, 'resolved', 'esm3-sm-open-v1', 'esm3_structure',
                 row=row, runtime=self.runtime(calls), representation='sequence_structure')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].sequence, 'AC|D')
            self.assertEqual(tuple(calls[0].coordinates.shape), (4, 37, 3))
            payload = torch.load(output, weights_only=False)
            self.assertEqual(payload['representation'], 'sequence_structure')
            self.assertEqual(payload['conditioning_scope'], 'antigen_only')
            self.assertEqual(payload['structure_scope'], 'joint_antigen_chains')
            self.assertEqual(payload['shape'], [4, 3])
            self.assertTrue(payload['antigen_backbone_sha256'])
            calls.clear()
            esm3('', row['id'], root / 'sequence.pt', 'resolved', 'esm3-sm-open-v1', 'esm3',
                 row=row, runtime=self.runtime(calls))
            self.assertEqual([p.sequence for p in calls], ['AC', 'D'])
            self.assertTrue(all(p.coordinates is None for p in calls))

    def test_missing_misaligned_or_unencoded_structures_fail_without_sequence_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            for updates, message in [({'resolved_ag_seq': 'AA,D'}, 'mismatch'),
                                     ({'antigen_chains': 'Ag_long,missing'}, 'missing')]:
                with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, message):
                    antigen_structure(dict(row, **updates), 'resolved')
            with self.assertRaisesRegex(ValueError, 'requires resolved'):
                antigen_structure(row, 'expected')
            with self.assertRaises(FileNotFoundError):
                antigen_structure(dict(row, pdb_path=str(root / 'missing.cif')), 'resolved')
            with self.assertRaisesRegex(RuntimeError, 'structure track'):
                esm3('', row['id'], root / 'bad.pt', 'resolved', 'esm3-sm-open-v1', 'esm3_structure',
                     row=row, runtime=self.runtime([], structure=False), representation='sequence_structure')
            self.assertFalse((root / 'bad.pt').exists())

    def test_missing_oxygen_keeps_sequence_and_masks_only_the_absent_atom(self):
        from Bio.PDB import MMCIFParser
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            structure = MMCIFParser(QUIET=True, auth_chains=True).get_structure('fixture', row['pdb_path'])
            del structure[0]['Ag_long'][(' ', 2, ' ')]['O']
            writer = MMCIFIO()
            writer.set_structure(structure)
            writer.save(row['pdb_path'])
            sequence, coords, _ = antigen_structure(row, 'resolved')
            self.assertEqual(sequence, 'AC|D')
            self.assertTrue(torch.isnan(coords[1, 4]).all())
            self.assertTrue(torch.isfinite(coords[1, :3]).all())

    def test_batch_reuses_structure_outputs_and_rejects_sequence_only_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            records = root / 'records.csv'
            with records.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            marker = root / 'out/.batch_complete.json'
            kwargs = dict(records_csv=str(records), output_dir=str(root / 'out'), marker=str(marker),
                kind='antigen', tag='esm3_structure', method='esm3',
                spec=dict(model_name='esm3-sm-open-v1', representation='sequence_structure'),
                seq_source='resolved', splits=['train'],
                implementation=str(Path(__file__).resolve().parents[1] / 'workflow/scripts/embed_antigen_esm3.py'))
            with patch('embed_antigen_esm3.load_runtime', return_value=self.runtime([])) as load:
                batch_embed(**kwargs)
                self.assertEqual(load.call_count, 1)
                batch_embed(**kwargs)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(json.loads(marker.read_text())['reused'], 1)
            path = root / 'out/train/example.pt'
            self.assertFalse(_can_reuse(path, row, 'esm3_structure', 'esm3', 'esm3-sm-open-v1', [],
                                        representation='sequence'))


if __name__ == '__main__':
    unittest.main()
