"""PyRosetta per-residue energy (PRE) antigen embedding, H=1.

Ports mango.utils.Ag_structure_embeddings.Ag_embeddings.PyRosetta_PRE: scores
the pose with the full-atom score function and keeps the per-residue total
energy for the antigen chains only. PRE is a scalar per residue, so the antigen
"vocabulary" here is R^1 -- the cross-attention projects it up to d_model like
any other representation.

Output shape: [L, 1] over backbone-resolved antigen residues.

The second biophysically informed representation the grant contrasts against
learned embeddings.

STATUS: ported but UNTESTED -- PyRosetta is licence-gated and is not installed
in the development environment. Two things to confirm on first real run:
  1. pose_from_file accepts the dataset's mmCIF directly (else convert to PDB);
  2. pdb_info().chain() returns the AUTHOR chain ids that antigen_chains uses.
Both are asserted below, so a mismatch fails loudly rather than silently
embedding the wrong chains.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_common as ec


def _init_pyrosetta():
    try:
        import pyrosetta
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "PyRosetta is required for the pyrosetta_pre embedder. Install it in "
            "the embed_pyrosetta env (see README.md) or drop 'pyrosetta_pre' "
            "from active_embedders in config/config.yaml."
        ) from exc

    pyrosetta.init(
        "-mute all -ignore_unrecognized_res 1 -load_PDB_components false",
        silent=True,
    )
    return pyrosetta


def load_runtime(score_function: str):
    """Initialize PyRosetta and its score metric once for a records batch."""
    pyrosetta = _init_pyrosetta()
    score_fxn = pyrosetta.create_score_function(score_function)
    metric = (
        pyrosetta.rosetta.core.simple_metrics.per_residue_metrics
        .PerResidueEnergyMetric()
    )
    metric.set_scorefunction(score_fxn)
    return pyrosetta, metric


def pyrosetta_pre(
    records_csv: str,
    record_id: str,
    cif_path: str,
    out: str,
    score_function: str,
    normalize: str,
    add_chain_breaks: bool,
    tag: str,
    row: dict | None = None,
    runtime=None,
) -> None:
    row = row if row is not None else ec.load_row(records_csv, record_id)
    chains = [c for c in row["antigen_chains"].split(ec.SEP) if c]

    if runtime is None:
        pyrosetta, metric = load_runtime(score_function)
    else:
        pyrosetta, metric = runtime

    suffixes = {suffix.lower() for suffix in Path(cif_path).suffixes}
    if {".cif", ".mmcif"} & suffixes:
        pose = pyrosetta.Pose()
        importer = pyrosetta.rosetta.core.import_pose
        importer.pose_from_file(pose, str(cif_path), False, importer.FileType.CIF_file)
    else:
        pose = pyrosetta.pose_from_file(cif_path)
    info = pose.pdb_info()

    present = {info.chain(i) for i in range(1, pose.total_residue() + 1)}
    missing = [c for c in chains if c not in present]
    if missing:
        raise RuntimeError(
            f"{record_id}: antigen chain(s) {missing} not found in {cif_path}. "
            f"Chains present in the pose: {sorted(present)}. This usually means "
            f"PyRosetta renamed the author chain ids while reading the mmCIF."
        )

    pre = {int(i): float(v) for i, v in metric.calculate(pose).items()}
    by_chain = {
        chain: [pre[i] for i in range(1, pose.total_residue() + 1)
                if info.chain(i) == chain and i in pre]
        for chain in chains
    }
    if not any(by_chain.values()):
        raise RuntimeError(f"{record_id}: no antigen residues scored")
    raw = np.asarray([value for chain in chains for value in by_chain[chain]],
                     dtype=np.float32)
    raw_mean, raw_std = float(raw.mean()), float(raw.std())
    if normalize not in {"none", "zscore"}:
        raise ValueError("normalize must be 'none' or 'zscore'")
    rows = []
    for index, chain in enumerate(chains):
        values = np.asarray(by_chain[chain], dtype=np.float32)
        if normalize == "zscore":
            values = (values - raw_mean) / (raw_std if raw_std > 1e-8 else 1.0)
        rows.extend(values.tolist())
        if add_chain_breaks and index + 1 < len(chains):
            rows.append(0.0)
    mat = torch.tensor(rows, dtype=torch.float32).unsqueeze(-1)

    ec.save_embedding(
        out,
        mat,
        meta=ec.build_meta(
            embedder=tag,
            model_name=f"pyrosetta_{score_function}",
            matrix=mat,
            chains=chains,
            chain_separator_token=add_chain_breaks,
            id=record_id,
            seq_source="structure",
            normalize=normalize,
            raw_mean=raw_mean,
            raw_std=raw_std,
            contributed_module="mango-embedders",
        ),
    )


def main() -> None:
    smk = globals().get("snakemake")
    if smk is not None:
        spec = dict(smk.params.spec)
        pyrosetta_pre(
            records_csv=smk.input.records,
            record_id=smk.wildcards.instance,
            cif_path=smk.input.cif,
            out=smk.output[0],
            score_function=spec.get("score_function", "ref2015"),
            normalize=spec.get("normalize", "none"),
            add_chain_breaks=bool(spec.get("add_chain_breaks", True)),
            tag=smk.wildcards.embedder,
        )
        return

    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--id", required=True, dest="record_id")
    p.add_argument("--cif", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--score-function", default="ref2015")
    p.add_argument("--normalize", choices=["none", "zscore"], default="none")
    p.add_argument("--no-chain-breaks", action="store_true")
    p.add_argument("--tag", default="pyrosetta_pre")
    a = p.parse_args()
    pyrosetta_pre(a.records, a.record_id, a.cif, a.out, a.score_function,
                  a.normalize, not a.no_chain_breaks, a.tag)


if __name__ == "__main__":
    main()
