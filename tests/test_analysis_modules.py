import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_cohort = load("select_cohort")
score_biophysical = load("score_biophysical")
score_germline = load("score_germline")
plot_nll = load("plot_nll")
plot_ab_likeness = load("plot_ab_likeness")
plot_germline = load("plot_germline")
plot_developability = load("plot_developability")


class AnalysisModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _csv(self, name, rows):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_cohort_selection_is_order_independent_and_shared_by_embedder(self):
        records = self._csv("records.csv", [
            {"id": "target-a", "resolved_L_seq": "DIQMTQ"},
        ])
        base = [
            {"run_id": "run", "target_id": "target-a", "design_index": i,
             "sequence": f"EVQLV{i}A", "status": "ok"}
            for i in range(10)
        ]
        selected = []
        for tag, rows in [("one_hot", base), ("esm2", list(reversed(base)))]:
            tagged = [dict(row, embedder=tag, run_id=f"run-{tag}") for row in rows]
            designs = self._csv(f"{tag}.csv", tagged)
            out = self.root / f"cohort-{tag}.csv"
            frame = select_cohort.select_cohort(
                [designs], records, tag, n_per_target=4, seed=17, out_csv=out
            )
            selected.append(frame["design_index"].tolist())
            self.assertTrue((frame["light_sequence"] == "DIQMTQ").all())
        self.assertEqual(selected[0], selected[1])

    def test_biophysical_metrics(self):
        gravy, charge = score_biophysical.score_sequence("ACDEFGHIKLMNPQRSTVWY", 7.4)
        self.assertIsInstance(gravy, float)
        self.assertIsInstance(charge, float)

    def test_levenshtein_and_vj_reference(self):
        self.assertEqual(score_germline.levenshtein("ABC", "ADC"), 1)
        self.assertEqual(score_germline.levenshtein("ABC", "ABCD"), 1)
        v = "A" * 100 + "-" * 28
        j = "-" * 100 + "B" * 28
        database = {"V": {"H": {"human": {"V1": v}}},
                    "J": {"H": {"human": {"J1": j}}}}
        self.assertEqual(
            score_germline.germline_reference(database, "human", "V1", "J1"),
            "A" * 100 + "B" * 28,
        )

    def test_all_plot_writers(self):
        embedders = ["one_hot", "esm2"]
        labels = {"one_hot": "One-hot", "esm2": "ESM-2"}
        evals = []
        for i, tag in enumerate(embedders):
            path = self.root / f"{tag}.json"
            path.write_text(json.dumps({
                "embedder": tag,
                "splits": {
                    "train": {"nll": 1.0 + i, "perplexity": 2.0,
                              "n_examples": 3, "n_tokens": 30},
                    "test": {"nll": 1.5 + i, "perplexity": 3.0,
                             "n_examples": 2, "n_tokens": 20},
                },
            }))
            evals.append(path)
        plot_nll.plot_nll(evals, embedders, labels, 80,
                          self.root / "fig1.png", self.root / "fig1.csv")

        inputs = {"iglm": [], "antiberty": [], "ablang2": []}
        germline_paths, bio_paths = [], []
        for i, tag in enumerate(embedders):
            base = [{"embedder": tag, "run_id": tag, "target_id": "t",
                     "design_index": j, "sequence": "EVQLV"} for j in range(3)]
            inputs["iglm"].append(self._csv(f"iglm-{tag}.csv", [
                dict(row, iglm_log_likelihood=-1.0 - i - j / 10)
                for j, row in enumerate(base)
            ]))
            inputs["antiberty"].append(self._csv(f"anti-{tag}.csv", [
                dict(row, antiberty_pseudo_log_likelihood=-2.0 - i - j / 10)
                for j, row in enumerate(base)
            ]))
            inputs["ablang2"].append(self._csv(f"ablang-{tag}.csv", [
                dict(row, ablang2_confidence=-0.5 - i - j / 10,
                     ablang2_mode="confidence", ablang2_status="ok")
                for j, row in enumerate(base)
            ]))
            germline_paths.append(self._csv(f"germline-{tag}.csv", [
                dict(row, germline_species="human", v_gene="IGHV1", v_identity=.9,
                     j_gene="IGHJ1", j_identity=.8, germline_reference_sequence="EVQLA",
                     ld_germline=3 + i + j, ld_germline_normalized=.1,
                     germline_status="ok") for j, row in enumerate(base)
            ]))
            bio_paths.append(self._csv(f"bio-{tag}.csv", [
                dict(row, gravy=-.2 + i / 10 + j / 20, charge_at_pH=1 + i + j,
                     charge_ph=7.4, metric_status="ok") for j, row in enumerate(base)
            ]))

        plot_ab_likeness.plot_ab_likeness(
            inputs, embedders, labels, 80, self.root / "fig3.png", self.root / "fig3.csv"
        )
        plot_germline.plot_germline(
            germline_paths, embedders, labels, 80,
            self.root / "fig4.png", self.root / "fig4.csv"
        )
        plot_developability.plot_developability(
            bio_paths, embedders, labels, 80,
            self.root / "fig5.png", self.root / "fig5.csv"
        )
        for figure in ("fig1.png", "fig3.png", "fig4.png", "fig5.png"):
            self.assertGreater((self.root / figure).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
