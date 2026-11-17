"""Validate ASD-specific chain masks, provenance, and antigen matching."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ANALYSIS = Path(__file__).resolve().parents[1] / "workflow/scripts/analysis"
sys.path.insert(0, str(ANALYSIS))
import plot_asd_training_distributions as module


class AsdTrainingDistributionTests(unittest.TestCase):
    def test_cohort_fingerprint_includes_target_masks(self):
        row = dict(id="one", split="train", ab_ag_cluster="group", resolved_H_seq="AAA",
                   resolved_L_seq="CCC", resolved_ag_seq="DDD", antigen_chains="AG",
                   target_contract="imgt_variable_vh_vl_v1", example_target_chains="heavy;light")
        before = module.cohort_digest([row])
        row["example_target_chains"] = "heavy"
        self.assertNotEqual(before, module.cohort_digest([row]))

    def test_antigen_sequence_match_excludes_antibody_entity(self):
        sequence = "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEQFKGLVLIAFSQYLQQCPFDEHVK"
        antibody = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC"
        def entity(identifier, seq):
            return dict(id=identifier, entity_poly=dict(pdbx_seq_one_letter_code_can=seq))
        target, decoy = entity("antigen", sequence), entity("antibody", antibody)
        result = module.matching_entities(sequence, dict(polymer_entities=[decoy, target]))
        self.assertEqual([row["id"] for row in result], ["antigen"])
        self.assertEqual(module.matching_entities(sequence, dict(polymer_entities=[decoy])), [])

    def test_engineered_construct_can_match_complete_antigen_fragment(self):
        reference = "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGEQFKGLVLIAFSQYLQQCPFDEHVK"
        entity = dict(entity_poly=dict(pdbx_seq_one_letter_code_can=reference))
        result = module.matching_entities("HHHHHH" + reference + "GGGGS", dict(polymer_entities=[entity]))
        self.assertEqual(result, [entity])

    def test_uniprot_species_is_independent_of_antibody_species(self):
        result = module.uniprot_annotation(dict(primaryAccession="P00001",
            organism=dict(scientificName="Gallus gallus"),
            proteinDescription=dict(recommendedName=dict(fullName=dict(value="Protein")))))
        self.assertEqual(result["protein_name"], "Protein")
        self.assertEqual(result["source_organism"], "Gallus gallus")

    def test_annotation_counts_only_selected_chain_examples(self):
        rows = [dict(id="one", resolved_H_seq="HEAVY", resolved_L_seq="LIGHT",
                     example_target_chains="heavy;light"),
                dict(id="two", resolved_H_seq="HEAVY2", resolved_L_seq="LIGHT",
                     example_target_chains="heavy")]
        def anarci(sequences, **kwargs):
            details = []
            for _, sequence in sequences:
                chain, locus = ("H", "H") if sequence.startswith("HEAVY") else ("K", "K")
                details.append([dict(chain_type=chain, species="mouse", germlines={
                    "v_gene": [("human", f"IG{locus}V1-1*01"), 1.0],
                    "j_gene": [("human", f"IG{locus}J1*01"), 1.0]})])
            return [None] * len(sequences), details, []
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"anarci": SimpleNamespace(anarci=anarci)}):
                module.common.annotate_antibodies(rows, Path(directory), 1)
            annotated = module.common.read_csv(Path(directory) / "training_antibodies.csv")
        self.assertEqual([(row["id"], row["target_chain"]) for row in annotated],
                         [("one", "heavy"), ("one", "light"), ("two", "heavy")])


if __name__ == "__main__":
    unittest.main()
