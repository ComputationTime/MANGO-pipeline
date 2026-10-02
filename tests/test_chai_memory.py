import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


PATH = Path(__file__).resolve().parents[1] / "workflow/scripts/analysis/chai_memory.py"
spec = importlib.util.spec_from_file_location("chai_memory", PATH)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class ChaiMemoryTests(unittest.TestCase):
    def test_defers_only_later_weights_and_preserves_calls(self):
        loads = []
        calls = []

        def loader(key, device):
            loads.append((key, device))

            def forward(*args, **kwargs):
                calls.append((args, kwargs))
                return args[0] + kwargs["offset"]

            return SimpleNamespace(forward=forward)

        chai = SimpleNamespace(load_exported=loader)
        with memory.defer_later_stage_weights(chai):
            chai.load_exported("trunk.pt", "cuda:0")
            later = chai.load_exported("diffusion_module.pt", "cuda:0")
            self.assertEqual(loads, [("trunk.pt", "cuda:0")])
            self.assertEqual(later.forward(7, offset=2), 9)
            self.assertEqual(later.forward(8, offset=3), 11)
            self.assertEqual(loads, [("trunk.pt", "cuda:0"),
                                     ("diffusion_module.pt", "cuda:0")])
            self.assertEqual(calls, [((7,), {"offset": 2}), ((8,), {"offset": 3})])
        self.assertIs(chai.load_exported, loader)

    def test_restores_loader_after_exception(self):
        loader = object()
        chai = SimpleNamespace(load_exported=loader)
        with self.assertRaises(RuntimeError):
            with memory.defer_later_stage_weights(chai):
                raise RuntimeError("prediction failed")
        self.assertIs(chai.load_exported, loader)


if __name__ == "__main__":
    unittest.main()
