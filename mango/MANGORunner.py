"""High-level interface for generating antibody chains with a trained MANGO model.

MANGO's task is: given an antigen and a requested chain type, generate that
antibody chain. No antibody sequence is ever supplied as conditioning.
"""

from pathlib import Path

import torch

from mango.model.MANGO import MangoModel

ONE_HOT_VOCAB = "|ACDEFGHIKLMNPQRSTVWY"


def _chain_sequences(pdb_path, chains):
    """Ordered {chain_id: sequence} read from the PDB's resolved residues."""
    from Bio.PDB import PDBParser, PPBuilder

    structure = PDBParser(QUIET=True).get_structure("antigen", str(pdb_path))
    model = next(structure.get_models())
    builder = PPBuilder()
    sequences = {}
    for chain_id in chains:
        if chain_id not in model:
            raise KeyError(f"chain {chain_id!r} not found in {pdb_path}")
        sequences[chain_id] = "".join(
            str(peptide.get_sequence()) for peptide in builder.build_peptides(model[chain_id])
        )
    return sequences


def one_hot_embedding(sequences):
    """One-hot antigen embedding, shape (1, L, 21); chains joined by '|'."""
    full_seq = "|".join(sequences.values())
    matrix = torch.zeros((len(full_seq), len(ONE_HOT_VOCAB)), dtype=torch.float32)
    for i, aa in enumerate(full_seq):
        j = ONE_HOT_VOCAB.find(aa)
        if j >= 0:
            matrix[i, j] = 1.0
    return matrix.unsqueeze(0)


class MANGORunner:
    def __init__(self, checkpoint_path, ag_representation="one_hot", device=None):
        """Load a trained MANGO checkpoint.

        Args:
            checkpoint_path: Path to a checkpoint written by ``scripts/training/train_mango.py``.
            ag_representation: Antigen representation the checkpoint was trained
                with. ``one_hot`` can be computed directly from a PDB; every
                other representation needs a precomputed embedding
                (see ``scripts/embedding/embed_antigen_*.py``).
            device: torch device; defaults to CUDA when available.
        """
        self.ag_representation = ag_representation
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = MangoModel.from_checkpoint(checkpoint).to(self.device).eval()

    def embed_antigen(self, antigen_pdb_path, antigen_chains):
        """Embed antigen chains from a PDB file (``one_hot`` only)."""
        if str(self.ag_representation).lower() not in {"one_hot", "one-hot"}:
            raise NotImplementedError(
                f"Embedding from a PDB is only supported for one_hot; for "
                f"{self.ag_representation!r} pass antigen_embedding= produced by "
                "scripts/embedding/embed_antigen_*.py"
            )
        return one_hot_embedding(_chain_sequences(antigen_pdb_path, antigen_chains))

    def generate(
        self,
        antigen_pdb_path=None,
        antigen_chains=None,
        antigen_embedding=None,
        chain_to_generate="H",
        num_to_generate=10,
        top_p=1.0,
        temperature=1.0,
        max_new_tokens=130,
        batch_size=128,
    ):
        """Generate antibody chains conditioned on an antigen.

        Args:
            antigen_pdb_path: PDB of the antigen (with ``antigen_chains``).
            antigen_chains: Chain IDs to use as antigen, e.g. ``['A', 'B']``.
            antigen_embedding: Precomputed embedding (tensor or ``.pt`` path)
                used instead of the PDB.
            chain_to_generate: ``'H'`` (heavy) or ``'L'`` (light).
            num_to_generate: Number of sequences to sample.
            top_p: Nucleus-sampling cutoff.
            temperature: Sampling temperature.
            max_new_tokens: Maximum generated length per chain.
            batch_size: Sequences sampled per forward pass.

        Returns:
            A list of generated amino-acid sequences.
        """
        h_ag = self._antigen_tensor(antigen_pdb_path, antigen_chains, antigen_embedding)
        if h_ag.shape[-1] != self.model.d_ag:
            raise ValueError(
                f"Antigen embedding width {h_ag.shape[-1]} does not match the "
                f"checkpoint ({self.model.d_ag})"
            )
        chain = self.model.normalize_chain(chain_to_generate)
        h_ag = h_ag.to(self.device)

        sequences = []
        remaining = int(num_to_generate)
        while remaining > 0:
            current = min(int(batch_size), remaining)
            batch = self.model.generate_chain_batch(
                h_ag, chain, batch_size=current, max_new_tokens=max_new_tokens,
                do_sample=True, top_p=top_p, temperature=temperature,
            )
            sequences.extend(self.model.decode_chain(ids) for ids in batch)
            remaining -= current
        return sequences

    def _antigen_tensor(self, antigen_pdb_path, antigen_chains, antigen_embedding):
        if antigen_embedding is not None:
            if isinstance(antigen_embedding, (str, Path)):
                antigen_embedding = torch.load(
                    antigen_embedding, map_location="cpu", weights_only=False
                )["embedding"]
            return antigen_embedding.unsqueeze(0) if antigen_embedding.dim() == 2 else antigen_embedding
        if antigen_pdb_path is None or not antigen_chains:
            raise ValueError("Provide antigen_embedding, or antigen_pdb_path with antigen_chains")
        return self.embed_antigen(antigen_pdb_path, antigen_chains)
