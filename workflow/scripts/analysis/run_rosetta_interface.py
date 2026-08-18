"""Score predicted antibody-antigen complexes without changing their coordinates."""

import sys
from pathlib import Path

import pandas as pd


COLUMNS = [
    "embedder", "run_id", "target_id", "design_index", "sequence",
    "predictor", "sample_index", "structure_path", "interface",
    "scorefxn", "dG_separated", "dSASA_int", "dG_dSASA_x100",
    "delta_unsat_hbonds", "hbonds_int", "nres_int", "status",
]


def _chain_groups(pose):
    """Return pose chain letters as antibody(first two) versus antigen(rest)."""
    info = pose.pdb_info()
    chains = []
    for chain_index in range(1, pose.num_chains() + 1):
        chain = info.chain(pose.conformation().chain_begin(chain_index))
        if chain not in chains:
            chains.append(chain)
    if len(chains) < 3:
        raise ValueError(
            f"expected heavy, light, and at least one antigen chain; found {chains}"
        )
    return "".join(chains[:2]), "".join(chains[2:])


def _metric(mover, name):
    value = getattr(mover, name)()
    return float(value)


def score_table(input_csv, output_csv, scorefxn_name="ref2015"):
    import pyrosetta
    from pyrosetta.rosetta.core.pose import DockingPartners
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pyrosetta.init(
        f"-mute all -constant_seed -ignore_unrecognized_res true "
        f"-score:weights {scorefxn_name}"
    )
    scorefxn = pyrosetta.create_score_function(scorefxn_name)
    source = pd.read_csv(input_csv, keep_default_na=False)
    source = source.loc[source["status"] == "ok"].copy()
    if source.empty:
        raise RuntimeError(f"{input_csv} contains no successful structures")
    # Diffusion draws are correlated technical replicates. Score only the
    # predictor's highest-confidence draw for each biological design.
    source["confidence_score"] = pd.to_numeric(
        source["confidence_score"], errors="coerce")
    keys = ["embedder", "target_id", "design_index", "predictor"]
    source = source.loc[source.groupby(keys)["confidence_score"].idxmax()].copy()

    rows = []
    for record in source.to_dict("records"):
        structure = Path(record["structure_path"])
        base = {column: record.get(column, "") for column in COLUMNS}
        base.update({"scorefxn": scorefxn_name, "status": "ok"})
        try:
            if not structure.is_file():
                raise FileNotFoundError(structure)
            pose = pyrosetta.pose_from_file(str(structure))
            antibody, antigen = _chain_groups(pose)
            interface = f"{antibody}_{antigen}"
            mover = InterfaceAnalyzerMover()
            mover.set_interface(
                DockingPartners.docking_partners_from_string(interface))
            mover.set_scorefunction(scorefxn)
            mover.set_pack_input(False)
            mover.set_pack_separated(False)
            mover.set_compute_packstat(False)
            mover.apply(pose)
            d_g = _metric(mover, "get_interface_dG")
            d_sasa = _metric(mover, "get_interface_delta_sasa")
            base.update({
                "interface": interface,
                "dG_separated": d_g,
                "dSASA_int": d_sasa,
                "dG_dSASA_x100": 100.0 * d_g / d_sasa if d_sasa else "",
                "delta_unsat_hbonds": _metric(
                    mover, "get_interface_delta_hbond_unsat"),
                "hbonds_int": pose.scores.get("hbonds_int", ""),
                "nres_int": _metric(mover, "get_num_interface_residues"),
            })
        except Exception as exc:
            base["status"] = f"error: {type(exc).__name__}: {exc}"
        rows.append(base)

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output, index=False)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise RuntimeError(
            f"Rosetta failed for {len(failures)}/{len(rows)} structures; "
            f"diagnostics: {output}"
        )


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("run_rosetta_interface.py must run through Snakemake")
    score_table(smk.input.scores, smk.output[0], smk.params.scorefxn)


if __name__ == "__main__":
    main()
