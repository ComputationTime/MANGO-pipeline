"""Minimal mmCIF backbone extractor for the ProteinMPNN embedder.

The SAbDab2 dataset ships ``.cif`` files, but ProteinMPNN's bundled parser only
understands fixed-column PDB. Rather than round-trip through PDB (which caps
chain IDs at one character and residue numbers at 4 digits), this reads the cif
directly with Biopython and builds exactly the ``pdb_dict_list`` schema
ProteinMPNN's ``StructureDatasetPDB`` / ``tied_featurize_minimal`` expect:

    {
      "name": <str>,
      "num_of_chains": <int>,
      "seq": <concatenated one-letter seq>,
      "seq_chain_<C>":   <one-letter seq for chain C>,
      "coords_chain_<C>": {
          "N_chain_<C>":  [[x,y,z], ...],
          "CA_chain_<C>": [...], "C_chain_<C>": [...], "O_chain_<C>": [...],
      },
    }

Only residues that have all four backbone atoms (N, CA, C, O) are kept -- these
are the residues ProteinMPNN can actually featurize (it drops any residue with a
NaN backbone atom). Chain IDs are matched against the author chain identifiers
stored in the dataset's ``antigen_chains`` column.
"""

from __future__ import annotations

from Bio.PDB import MMCIFParser
from Bio.Data.PDBData import protein_letters_3to1_extended

BACKBONE = ("N", "CA", "C", "O")


def _one_letter(resname: str) -> str:
    return protein_letters_3to1_extended.get(resname.strip().upper(), "X")


def backbone_dict(cif_path: str, chains, name: str | None = None) -> dict:
    """Extract backbone coords for ``chains`` (author IDs) from ``cif_path``.

    Returns a single ProteinMPNN-style dict (see module docstring). Chains are
    emitted in the order given by ``chains``. Raises ValueError if a requested
    chain has no backbone-complete residues.
    """
    parser = MMCIFParser(QUIET=True, auth_chains=True)
    structure = parser.get_structure(name or "antigen", cif_path)
    model = next(structure.get_models())

    my_dict: dict = {}
    concat_seq = ""
    n_chains = 0

    for chain_id in chains:
        chain_id = chain_id.strip()
        if chain_id == "" or chain_id not in model:
            continue

        seq_chars = []
        coords = {atom: [] for atom in BACKBONE}
        for residue in model[chain_id]:
            # Skip hetero/water residues (het flag is non-blank).
            if residue.id[0].strip() != "":
                continue
            if not all(atom in residue for atom in BACKBONE):
                continue
            seq_chars.append(_one_letter(residue.resname))
            for atom in BACKBONE:
                coords[atom].append([float(c) for c in residue[atom].coord])

        if not seq_chars:
            continue

        seq = "".join(seq_chars)
        my_dict[f"seq_chain_{chain_id}"] = seq
        my_dict[f"coords_chain_{chain_id}"] = {
            f"{atom}_chain_{chain_id}": coords[atom] for atom in BACKBONE
        }
        concat_seq += seq
        n_chains += 1

    if n_chains == 0:
        raise ValueError(
            f"No backbone-complete residues for chains {list(chains)} in {cif_path}"
        )

    my_dict["name"] = name or "antigen"
    my_dict["num_of_chains"] = n_chains
    my_dict["seq"] = concat_seq
    return my_dict
