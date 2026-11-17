import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'workflow/scripts'))
from accelerated_affinity_search import command, search
from run_short_affinity_accelerated import Controller
from affinity_study_common import write_json, digest


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(mmseqs='mmseqs', antibody_identity=.9, antibody_coverage=.8)

    def test_full_graph_never_stops_at_first_match(self):
        graph = command(self.settings, Path('search'), graph=True)
        self.assertNotIn('--max-accept', graph)
        self.assertEqual(graph[graph.index('--max-seqs')+1], '100000')

    def test_existence_search_can_stop_only_after_accepted_hit(self):
        args = command(self.settings, Path('search'))
        self.assertEqual(args[args.index('--max-accept')+1], '1')
        for flag, value in [('--gpu','1'),('--min-seq-id','0.9'),('-c','0.8'),('-e','0.001')]:
            self.assertEqual(args[args.index(flag)+1], value)

    def test_antigen_search_is_unchanged(self):
        with patch('accelerated_affinity_search.cpu_search', return_value={'hit'}) as cpu:
            self.assertEqual(search(self.settings, Path('root'), 'ag', ['a'], ['b'], False), {'hit'})
            cpu.assert_called_once_with(self.settings, Path('root'), 'ag', ['a'], ['b'], False)

    def test_failed_backend_cannot_be_used(self):
        settings = dict(self.settings, search_validation_root='validation')
        with patch('accelerated_affinity_search.read_json', side_effect=[dict(complete=False, validated=False), {}]):
            with self.assertRaises(ValueError):
                search(settings, Path('root'), 'ab', ['a'], ['b'], True)


class DiagnosticReuseTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory); prior = base/'prior'; prior.mkdir()
        binding = base/'binding'; binding.mkdir(); (binding/'input').write_text('protected input')
        controller = Controller.__new__(Controller)
        controller.root = base/'new'; controller.root.mkdir()
        controller.cfg = dict(diagnostic_reuse_root=str(prior), binding_root=str(binding),
            reference_root=str(base/'reference'), diagnostic_pairs_per_target=1000, seeds=[0], parity_tolerance=3e-6)
        write_json(prior/'protocol.json', dict(settings=controller.cfg))
        (prior/'diagnostic_selection.csv').write_text('fixed selection')
        selection_sha = digest(prior/'diagnostic_selection.csv')
        write_json(prior/'diagnostic_selection.json', dict(selection_sha256=selection_sha,
            files_sha256={'input':digest(binding/'input')}))
        (prior/'diagnostics').mkdir()
        for kind in ('base','iglm'):
            run=f'pilot_{kind}_s0'; checkpoint=base/'reference'/'runs'/run/'best.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_text('fixed checkpoint')
            scores=prior/'diagnostics'/f'{run}.csv'; scores.write_text('fixed scores')
            write_json(prior/'diagnostics'/f'{run}.complete.json', dict(status='complete',
                checkpoint_sha256=digest(checkpoint), scores_sha256=digest(scores),selection_sha256=selection_sha))
        return controller

    def test_reuse_copies_exact_bytes_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.fixture(directory); controller.reuse_diagnostics()
            self.assertTrue((controller.root/'diagnostic_reuse.json').exists())
            self.assertEqual(digest(controller.root/'diagnostics/pilot_base_s0.csv'),
                digest(Path(controller.cfg['diagnostic_reuse_root'])/'diagnostics/pilot_base_s0.csv'))

    def test_reuse_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            controller=self.fixture(directory)
            (Path(controller.cfg['reference_root'])/'runs/pilot_base_s0/best.pt').write_text('changed')
            with self.assertRaises(ValueError):controller.reuse_diagnostics()


if __name__ == '__main__':
    unittest.main()
