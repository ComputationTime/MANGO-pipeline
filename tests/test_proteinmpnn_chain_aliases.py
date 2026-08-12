from workflow.scripts.embed_antigen_proteinmpnn import alias_proteinmpnn_chains


def test_aliases_multichar_author_chain_ids():
    source = {
        "name": "example",
        "num_of_chains": 2,
        "seq": "AGST",
        "seq_chain_D1": "AG",
        "coords_chain_D1": {
            "N_chain_D1": [[0, 0, 0]],
            "CA_chain_D1": [[0, 0, 0]],
            "C_chain_D1": [[0, 0, 0]],
            "O_chain_D1": [[0, 0, 0]],
        },
        "seq_chain_E": "ST",
        "coords_chain_E": {
            "N_chain_E": [[1, 1, 1]],
            "CA_chain_E": [[1, 1, 1]],
            "C_chain_E": [[1, 1, 1]],
            "O_chain_E": [[1, 1, 1]],
        },
    }

    aliased = alias_proteinmpnn_chains(source, ["D1", "E"])

    assert aliased["seq_chain_A"] == "AG"
    assert aliased["seq_chain_B"] == "ST"
    assert "seq_chain_D1" not in aliased
    assert "CA_chain_A" in aliased["coords_chain_A"]
    assert "CA_chain_B" in aliased["coords_chain_B"]
    assert aliased["seq"] == "AGST"
    assert aliased["num_of_chains"] == 2
