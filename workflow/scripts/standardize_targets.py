"""Describe the Aim 2 therapeutic targets as a records-shaped table.

Emits the same columns process_records.py produces, with split="target", so the
antigen embedder rules can consume these structures unchanged.

Only `antigen_chains` is required for MANGO generation. An optional
`light_chain` may be retained as a downstream complex-scoring partner, but it is
never embedded or supplied to MANGO. Missing antigen-chain annotations are a
hard error that lists the chains present in the structure.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import standardize_common as sc
from lib.mmcif import backbone_dict

SEP = sc.SEP

OUTPUT_COLUMNS = [
    "id", "pdb_path", "antigen_chains", "chains",
    "expected_heavy_seq", "expected_light_seq", "expected_ag_seq",
    "resolved_H_seq", "resolved_L_seq", "resolved_ag_seq",
    "split",
]


def _all_chains(cif_path: str) -> dict:
    """{chain_id: resolved sequence} for every backbone-complete chain."""
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("target", cif_path)
    model = next(iter(structure))
    ids = [chain.id for chain in model]
    d = backbone_dict(cif_path, ids)
    return {c: d.get(f"seq_chain_{c}", "") for c in ids}


def _describe(pdb: str, cif_path: str, found: dict) -> str:
    lines = [f"  {pdb}: chains present in {cif_path}"]
    for chain, seq in sorted(found.items()):
        preview = seq[:60] + ("..." if len(seq) > 60 else "")
        lines.append(f"    {chain!r:>6}  len={len(seq):<5} {preview}")
    return "\n".join(lines)


def standardize_targets(targets, structures_dir: str, split: str, out_csv: str):
    rows = []
    problems = []

    for entry in targets:
        pdb = entry["pdb"]
        cif_path = str(Path(structures_dir) / f"{pdb}.cif")
        found = _all_chains(cif_path)

        chains_cfg = entry.get("antigen_chains")
        light_cfg = entry.get("light_chain")
        if not chains_cfg:
            problems.append(
                f"  {pdb}: antigen_chains not set\n"
                + _describe(pdb, cif_path, found)
            )
            continue

        chains = [c.strip() for c in str(chains_cfg).split(SEP) if c.strip()]
        light = str(light_cfg or "").strip()
        missing = [c for c in chains + ([light] if light else []) if c not in found]
        if missing:
            problems.append(
                f"  {pdb}: configured chain(s) {missing} not in structure\n"
                + _describe(pdb, cif_path, found)
            )
            continue
        if light and light in chains:
            problems.append(
                f"  {pdb}: light_chain {light!r} is also listed as an antigen "
                "chain; they must be different chains\n"
                + _describe(pdb, cif_path, found)
            )
            continue

        rows.append(
            {
                "id": pdb,
                "pdb_path": cif_path,
                "antigen_chains": SEP.join(chains),
                "chains": SEP.join(([light] if light else []) + chains),
                "expected_heavy_seq": "",
                "expected_light_seq": "",
                "expected_ag_seq": "",
                # Target sequences are absent for external generation records.
                "resolved_H_seq": "",
                "resolved_L_seq": found[light] if light else "",
                "resolved_ag_seq": SEP.join(found[c] for c in chains),
                "split": split,
                "antibody_name": entry.get("antibody", ""),
                "target_name": entry.get("target", ""),
            }
        )

    if problems:
        raise ValueError(
            "generation.targets needs antigen_chains filled in "
            f"for {len(problems)} target(s) in config/config.yaml.\n"
            "antigen_chains must identify only the antigen. Optional light_chain "
            "metadata is reserved for downstream complex scoring and is never a "
            "MANGO input.\n\n"
            + "\n\n".join(problems)
        )

    out = pd.DataFrame(rows)[OUTPUT_COLUMNS + ["antibody_name", "target_name"]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} target records -> {out_csv}", flush=True)
    return out


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        standardize_targets(
            targets=[dict(t) for t in smk.params.targets],
            structures_dir=smk.params.structures_dir,
            split=smk.params.split,
            out_csv=smk.output.records,
        )
        return

    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True, help="JSON list of target entries")
    p.add_argument("--structures-dir", required=True)
    p.add_argument("--split", default="target")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    standardize_targets(json.loads(a.targets), a.structures_dir, a.split, a.out)


if __name__ == "__main__":
    main()
