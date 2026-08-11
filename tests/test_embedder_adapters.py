import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import embed_antigen_biopython


class ContributedEmbedderAdapterTests(unittest.TestCase):
    def test_biophysical_descriptor_is_repeated_per_residue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.csv"
            pd.DataFrame([{
                "id": "example",
                "antigen_chains": "A,B",
                "resolved_ag_seq": "ACD,EF",
            }]).to_csv(records, index=False)
            out = root / "embedding.pt"
            embed_antigen_biopython.biopython(
                str(records), "example", str(out), "resolved", "biopython"
            )
            payload = torch.load(out, map_location="cpu", weights_only=False)
            matrix = payload["embedding"]
            self.assertEqual(tuple(matrix.shape), (6, 11))  # 3 + separator + 2
            self.assertTrue(torch.equal(matrix[0], matrix[1]))
            self.assertTrue(torch.equal(matrix[1], matrix[2]))
            self.assertTrue(torch.equal(matrix[3], torch.zeros(11)))
            self.assertEqual(payload["axis"], "residue")


if __name__ == "__main__":
    unittest.main()
