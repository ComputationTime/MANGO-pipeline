"""Snapshot protected SAbDab2 tests and deduplicate antigen queries for CATH."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def clean_sequence(value):
    sequence = "".join(str(value).split()).upper()
    if not sequence or not set(sequence) <= AMINO_ACIDS:
        raise ValueError("Missing or invalid amino-acid sequence")
    return sequence


def sequence_id(sequence):
    return "ag_" + hashlib.sha256(sequence.encode()).hexdigest()[:24]


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError("An empty table needs explicit columns")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(abrank, sabdab, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(sabdab, newline="") as fh:
        native = list(csv.DictReader(fh))
    protected = [r for r in native if r["split"] == "test"]
    if not protected:
        raise ValueError("The protected SAbDab2 test cohort must be nonempty")
    write_csv(output / "protected_sabdab2_test.csv", protected)
    sequences = {}
    members = []
    invalid = []

    def add(value, dataset, record_id, split, chain, origin):
        try:
            sequence = clean_sequence(value)
        except ValueError:
            invalid.append(dict(dataset=dataset, record_id=record_id, chain=chain, origin=origin))
            return
        sid = sequence_id(sequence)
        if sid in sequences and sequences[sid] != sequence:
            raise AssertionError("Sequence hash collision")
        sequences[sid] = sequence
        members.append(dict(sequence_id=sid, dataset=dataset, record_id=record_id,
                            original_split=split, chain=chain, origin=origin,
                            length=len(sequence)))

    for row in native:
        chains = row["antigen_chains"].split(",")
        for column in ["resolved_ag_seq", "expected_ag_seq"]:
            parts = row[column].split(",")
            if len(chains) != len(parts):
                raise ValueError(f"{row['id']}: chain/sequence count differs")
            for chain, sequence in zip(chains, parts):
                add(sequence, "sabdab2", row["id"], row["split"], chain, column)
    with open(abrank, newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            add(row["Ag_seq"], "abrank", f"abrank_{index:07d}", "unassigned", "AG", "Ag_seq")
    with (output / "antigen_queries.fasta").open("w") as fh:
        for sid, sequence in sorted(sequences.items()):
            fh.write(f">{sid}\n{sequence}\n")
    write_csv(output / "antigen_members.csv", members)
    write_csv(output / "invalid_antigen_sequences.csv", invalid,
              ["dataset", "record_id", "chain", "origin"])
    manifest = {
        "abrank": {"path": str(abrank), "sha256": hashlib.sha256(Path(abrank).read_bytes()).hexdigest()},
        "sabdab2": {"path": str(sabdab), "sha256": hashlib.sha256(Path(sabdab).read_bytes()).hexdigest()},
        "protected_test_records": len(protected),
        "protected_test_clusters": len({r["ab_ag_cluster"] for r in protected}),
        "unique_antigen_queries": len(sequences),
        "invalid_antigen_entries": len(invalid),
        "protected_figures": {},
    }
    for name in ["fig8_antigen_controls.png", "fig8_antigen_controls_data.csv",
                 "fig9_antigen_context_length_ablation.png", "fig9_antigen_context_length_ablation_data.csv"]:
        path = Path("artifacts/analysis/figures") / name
        if path.exists():
            manifest["protected_figures"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abrank", required=True)
    parser.add_argument("--sabdab", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.abrank, args.sabdab, args.output)
