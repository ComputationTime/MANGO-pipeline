"""Check split isolation, chain mapping, and distribution denominators."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT / "workflow" / "scripts", _ROOT / "experiments" / "scripts", _ROOT / "experiments" / "scripts" / "analysis"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "experiments/scripts/analysis/plot_training_distributions.py"
SPEC = importlib.util.spec_from_file_location("training_distributions", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class TrainingDistributionTests(unittest.TestCase):
    def test_species_uses_germline_calls_and_preserves_locus(self):
        detail = dict(chain_type="K", species="rhesus", germlines={
            "v_gene": [("mouse", "IGKV5-43*01"), 0.96],
            "j_gene": [("mouse", "IGKJ2*01"), 1.0],
        })
        result = module.parse_assignment([detail], "light")
        self.assertEqual(result["germline_species"], "mouse")
        self.assertEqual(result["anarci_hmm_species"], "rhesus")
        self.assertEqual(result["v_family"], "IGKV5")
        self.assertEqual(result["j_family"], "IGKJ2")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(module.gene_family("IGLV3-1*01", "V"), "IGLV3")
        self.assertEqual(module.parse_assignment([detail], "heavy")["status"],
                         "no_matching_domain")

    def test_conflicting_species_and_missing_domains_remain_unassigned(self):
        detail = dict(chain_type="H", species="human", germlines={
            "v_gene": [("human", "IGHV3-23*01"), 0.98],
            "j_gene": [("mouse", "IGHJ4*01"), 1.0],
        })
        result = module.parse_assignment([detail], "heavy")
        self.assertEqual(result["germline_species"], "Unassigned")
        self.assertEqual(result["status"], "incomplete_assignment")
        self.assertEqual(module.parse_assignment(None, "heavy")["v_family"], "Unassigned")

    def test_only_processed_training_rows_and_no_cluster_leakage(self):
        rows = [dict(id="one", split="train", ab_ag_cluster="a"),
                dict(id="two", split="val", ab_ag_cluster="b"),
                dict(id="three", split="test", ab_ag_cluster="c")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            module.write_csv(path, rows)
            _, train = module.training_rows(path)
            self.assertEqual([row["id"] for row in train], ["one"])
            rows[1]["ab_ag_cluster"] = "a"
            module.write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "crosses"):
                module.training_rows(path)

    def test_antigen_maps_author_chain_and_source_organism(self):
        def entity(chain, name, organism, accession):
            return dict(rcsb_polymer_entity_container_identifiers=dict(
                entity_id="1", auth_asym_ids=[chain], uniprot_ids=[accession]),
                rcsb_polymer_entity=dict(pdbx_description=name),
                rcsb_entity_source_organism=[dict(ncbi_scientific_name=organism)])
        entry = dict(rcsb_id="TEST", polymer_entities=[
            entity("H", "Antibody", "Homo sapiens", "ANTIBODY"),
            entity("Y", "Lysozyme C", "Gallus gallus", "P00698"),
        ])
        result = module.antigen_annotation(entry, "Y")
        self.assertEqual(result["protein_key"], "UniProt:P00698")
        self.assertEqual(result["source_organism"], "Gallus gallus")
        with self.assertRaisesRegex(ValueError, "maps to 0"):
            module.antigen_annotation(entry, "C")
        # Local SAbDab chain labels can differ from deposited author labels.
        entry["polymer_entities"][1]["rcsb_polymer_entity_container_identifiers"]["entity_id"] = "2"
        result = module.antigen_annotation(entry, "renamed_chain", "2")
        self.assertEqual(result["protein_key"], "UniProt:P00698")
        missing = module.antigen_annotation(dict(metadata_status="entry_unavailable"), "A")
        self.assertEqual(missing["source_organism"], "Unknown")

    def test_trimmed_antigen_entity_sequence_verification(self):
        self.assertTrue(module.sequence_is_subsequence("ACDF", "AACDEF"))
        self.assertTrue(module.sequence_is_subsequence("AXDF", "AACDEF"))
        self.assertFalse(module.sequence_is_subsequence("WACDF", "AACDEF"))
        self.assertFalse(module.sequence_is_subsequence("", "AACDEF"))
        self.assertTrue(module.sequence_matches_entity("HHH" + "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"))
        self.assertFalse(module.sequence_matches_entity("HHHHHHHHHH" + "ACDEF", "ACDEF"))

    def test_counts_retain_repeats_and_include_unassigned_denominators(self):
        antibodies = [dict(target_chain="heavy", sequence="AAA", v_family="IGHV3",
                           j_family="IGHJ4", germline_species="human") for _ in range(2)]
        antibodies.append(dict(target_chain="heavy", sequence="XXX", v_family="Unassigned",
                               j_family="Unassigned", germline_species="Unassigned"))
        antigens = [dict(sequence="CCC", protein_label="Protein", source_organism="Organism")
                    for _ in range(2)]
        train = [dict(antigen_chains="Y"), dict(antigen_chains="Y,Z")]
        summary = module.summarize(antibodies, antigens, train)
        occurrence = [row for row in summary if row["entity"] == "heavy"
                      and row["unit"] == "training_chain_example" and row["plot"] == "V"]
        self.assertEqual(sum(row["count"] for row in occurrence), 3)
        self.assertAlmostEqual(sum(row["percent"] for row in occurrence), 100)
        unique = [row for row in summary if row["entity"] == "heavy"
                  and row["unit"] == "unique_sequence" and row["plot"] == "V"]
        self.assertEqual(sum(row["count"] for row in unique), 2)
        unique_antigens = [row for row in summary if row["entity"] == "antigen"
                           and row["unit"] == "unique_sequence" and row["plot"] == "protein"]
        self.assertEqual(sum(row["count"] for row in unique_antigens), 1)


if __name__ == "__main__":
    unittest.main()
