"""Describe the processed MANGO training set using Figure 6's visual grammar.

Run ``--stage antibodies`` and ``--stage antigens`` in analysis_anarci,
and ``--stage plot`` in analysis_plot. Intermediate data
are cached under the output directory; no training inputs are modified.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parents[3] / "workflow" / "scripts",):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import textwrap
import urllib.parse
import urllib.request


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def training_rows(path):
    rows = read_csv(path)
    seen = defaultdict(set)
    for row in rows:
        cluster = row["ab_ag_cluster"].strip()
        if not cluster:
            raise ValueError(f"Missing ab_ag_cluster for {row['id']}")
        seen[cluster].add(row["split"])
    if any(len(splits) > 1 for splits in seen.values()):
        raise ValueError("An ab_ag_cluster crosses dataset splits")
    train = [row for row in rows if row["split"] == "train"]
    if not train or len({row["id"] for row in train}) != len(train):
        raise ValueError("Training records must be nonempty and have unique IDs")
    return rows, train


def gene_family(gene, segment):
    match = re.match(rf"(IG[HKL]{segment}\d+)", gene.upper())
    return match.group(1) if match else "Unassigned"


def parse_assignment(details, target_chain):
    allowed = {"H"} if target_chain == "heavy" else {"K", "L"}
    domains = [detail for detail in (details or [])
               if detail.get("chain_type") in allowed]
    result = dict(anarci_chain_type="", anarci_hmm_species="",
                  germline_species="Unassigned", v_gene="", j_gene="",
                  v_family="Unassigned", j_family="Unassigned",
                  status="no_matching_domain")
    if len(domains) != 1:
        if domains:
            result["status"] = "multiple_matching_domains"
        return result
    detail = domains[0]
    result.update(anarci_chain_type=detail["chain_type"],
                  anarci_hmm_species=detail.get("species", ""))
    species = []
    for segment in ("v", "j"):
        call = detail.get("germlines", {}).get(f"{segment}_gene")
        if call and call[0] and len(call[0]) == 2:
            organism, gene = call[0]
            result[f"{segment}_gene"] = gene
            result[f"{segment}_family"] = gene_family(gene, segment.upper())
            species.append(organism)
    if len(species) == 2 and species[0] == species[1] and all(species):
        result["germline_species"] = species[0]
    result["status"] = (
        "ok" if result["germline_species"] != "Unassigned"
        and result["v_family"] != "Unassigned"
        and result["j_family"] != "Unassigned" else "incomplete_assignment"
    )
    return result


def annotate_antibodies(train, output, ncpu, cache_path=None):
    from anarci import anarci
    from chain_examples import example_chains

    cache_path = Path(cache_path) if cache_path else output / "anarci_sequence_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    examples = []
    for row in train:
        for chain, field in (("heavy", "resolved_H_seq"), ("light", "resolved_L_seq")):
            if chain not in example_chains(row):
                continue
            sequence = row[field]
            if not sequence:
                raise ValueError(f"{row['id']}: empty {chain} training target")
            key = chain + ":" + hashlib.sha256(sequence.encode()).hexdigest()
            examples.append(dict(id=row["id"], split="train", target_chain=chain,
                                 sequence=sequence, cache_key=key))
    pending = dict((row["cache_key"], row) for row in examples
                   if row["cache_key"] not in cache)

    def batch(items):
        try:
            _, details, _ = anarci(
                [(key, row["sequence"]) for key, row in items], scheme="imgt",
                output=False, allow={"H", "K", "L"}, assign_germline=True,
                allowed_species=None, ncpu=ncpu,
            )
            if len(details) != len(items):
                raise ValueError("ANARCI returned a different number of results")
        except Exception:
            # Isolate a problematic sequence without losing the rest of a batch.
            if len(items) == 1:
                raise
            midpoint = len(items) // 2
            batch(items[:midpoint])
            batch(items[midpoint:])
            return
        for (key, row), detail in zip(items, details):
            cache[key] = parse_assignment(detail, row["target_chain"])

    items = list(pending.items())
    print(f"ANARCI: {len(examples)} training targets; {len(items)} uncached unique sequences",
          flush=True)
    for start in range(0, len(items), 200):
        batch(items[start:start + 200])
        write_json(cache_path, cache)
        print(f"ANARCI: {min(start + 200, len(items))}/{len(items)} annotated", flush=True)
    annotated = [{**{k: v for k, v in row.items() if k != "cache_key"},
                  **cache[row["cache_key"]]} for row in examples]
    write_csv(output / "training_antibodies.csv", annotated)
    print("ANARCI status:", dict(Counter(row["status"] for row in annotated)), flush=True)


def fetch_entries(pdb_ids, output):
    path = output / "rcsb_entry_cache.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    pending = sorted(pdb_id for pdb_id in set(pdb_ids)
                     if pdb_id not in cache or any("entity_poly" not in entity
                         for entity in cache[pdb_id]["entry"]["polymer_entities"]))
    fields = """rcsb_id polymer_entities {
      entity_poly { pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { entity_id auth_asym_ids uniprot_ids }
      rcsb_polymer_entity { pdbx_description rcsb_polymer_name_combined { names } }
      rcsb_entity_source_organism { ncbi_scientific_name scientific_name ncbi_taxonomy_id }
    }"""
    for start in range(0, len(pending), 50):
        requested = pending[start:start + 50]
        query = "{ entries(entry_ids: " + json.dumps(requested) + ") { " + fields + " } }"
        url = "https://data.rcsb.org/graphql?" + urllib.parse.urlencode({"query": query})
        with urllib.request.urlopen(url, timeout=60) as handle:
            payload = json.load(handle)
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        entries = payload["data"]["entries"] or []
        retrieved = {entry["rcsb_id"]: entry for entry in entries if entry}
        missing = set(requested) - retrieved.keys()
        for pdb_id in sorted(missing):
            # Entries absent from the public archive remain explicit unknowns.
            # Retain their training examples in every distribution denominator.
            retrieved[pdb_id] = dict(rcsb_id=pdb_id, polymer_entities=[],
                                     metadata_status="entry_unavailable")
            print(f"RCSB: {pdb_id} unavailable; retaining as Unknown", flush=True)
        for pdb_id, entry in retrieved.items():
            cache[pdb_id] = dict(entry=entry, retrieved_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, cache)
        print(f"RCSB: {min(start + 50, len(pending))}/{len(pending)} new PDB entries cached", flush=True)
    return cache


def antigen_annotation(entry, chain, local_entity_id=None):
    if entry.get("metadata_status") == "entry_unavailable":
        return dict(entity_id="", protein_key="Unknown", protein_name="Unknown",
                    uniprot_ids="", source_organism="Unknown")
    entities = []
    for entity in entry["polymer_entities"]:
        ids = entity["rcsb_polymer_entity_container_identifiers"]
        if (ids["entity_id"] == local_entity_id if local_entity_id is not None
                else chain in (ids.get("auth_asym_ids") or [])):
            entities.append(entity)
    if len(entities) != 1:
        raise ValueError(f"{entry['rcsb_id']}: author chain {chain!r} maps to {len(entities)} entities")
    entity = entities[0]
    ids = entity["rcsb_polymer_entity_container_identifiers"]
    description = entity["rcsb_polymer_entity"]
    names = (description.get("rcsb_polymer_name_combined") or {}).get("names")
    name = "; ".join(names) if names else description.get("pdbx_description") or "Unknown"
    accessions = sorted(ids.get("uniprot_ids") or [])
    organisms = sorted({organism.get("ncbi_scientific_name") or organism.get("scientific_name")
                        for organism in entity.get("rcsb_entity_source_organism") or []}
                       - {None, ""})
    key = "UniProt:" + ";".join(accessions) if accessions else "PDB-name:" + " ".join(name.lower().split())
    return dict(entity_id=ids["entity_id"], protein_key=key, protein_name=name,
                uniprot_ids=";".join(accessions),
                source_organism="; ".join(organisms) if organisms else "Unknown")


def local_polymer_entities(path):
    """Read entity IDs before the atom table; SAbDab sometimes renames chains."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    lines = []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("_atom_site."):
                break
            lines.append(line)
    while lines and lines[-1].strip() in ("", "loop_", "#"):
        lines.pop()
    header = MMCIF2Dict(StringIO("".join(lines)))
    return {chain.strip(): (entity_id, sequence)
            for entity_id, strands, sequence in zip(
                header["_entity_poly.entity_id"], header["_entity_poly.pdbx_strand_id"],
                header["_entity_poly.pdbx_seq_one_letter_code"])
            for chain in strands.split(",") if chain.strip() not in ("", "?", ".")}


def sequence_is_subsequence(local, reference):
    """Allow structure trimming/missing residues and modified-residue X calls."""
    local = re.sub(r"\([^)]*\)", "X", local)
    local = "".join(local.split())
    reference = "".join(reference.split())
    if not local or not reference:
        return False
    position = 0
    for residue in local:
        while position < len(reference) and residue not in (reference[position], "X") and reference[position] != "X":
            position += 1
        if position == len(reference):
            return False
        position += 1
    return True


def sequence_matches_entity(local, reference):
    if sequence_is_subsequence(local, reference):
        return True
    # An archive revision can omit a terminal purification tag still present in
    # SAbDab. Require the full current entity sequence and at least 80% coverage.
    return (len(reference) >= 0.8 * len(local)
            and sequence_is_subsequence(reference, local))


def annotate_antigens(train, source_path, output):
    source = {row["INSTANCE"]: row for row in read_csv(source_path)}
    cache = fetch_entries([row["pdb_id"][-4:].upper() for row in train], output)
    annotations = []
    for row in train:
        chains = row["antigen_chains"].split(",")
        sequences = row["resolved_ag_seq"].split(",")
        clusters = source[row["id"]]["agclusters"].split("/")
        if not len(chains) == len(sequences) == len(clusters):
            raise ValueError(f"{row['id']}: antigen chains, sequences, and clusters differ in length")
        pdb_id = row["pdb_id"][-4:].upper()
        local_entities = local_polymer_entities(row["pdb_path"])
        for chain, sequence, cluster in zip(chains, sequences, clusters):
            local_entity_id, local_sequence = local_entities[chain]
            entry = cache[pdb_id]["entry"]
            annotation = antigen_annotation(entry, chain, local_entity_id)
            mapping = "local_entity_id_sequence_verified"
            if entry.get("metadata_status") == "entry_unavailable":
                mapping = "entry_unavailable"
            else:
                entity = next(entity for entity in entry["polymer_entities"]
                              if entity["rcsb_polymer_entity_container_identifiers"]["entity_id"] == local_entity_id)
                reference = entity["entity_poly"]["pdbx_seq_one_letter_code_can"]
                if not sequence_matches_entity(local_sequence, reference):
                    raise ValueError(f"{row['id']}: local antigen chain {chain} differs from RCSB entity {local_entity_id}")
            annotations.append(dict(id=row["id"], split="train", pdb_id=pdb_id,
                                    author_chain=chain, sequence=sequence,
                                    antigen_cluster=cluster, antigen_chain_count=len(chains),
                                    metadata_mapping=mapping,
                                    **annotation))
    # One label per accession set prevents synonymous PDB descriptions splitting counts.
    names = defaultdict(Counter)
    for row in annotations:
        names[row["protein_key"]][row["protein_name"]] += 1
    for row in annotations:
        row["protein_name"] = names[row["protein_key"]].most_common(1)[0][0]
        row["protein_label"] = row["protein_name"] + (
            f" [{row['uniprot_ids']}]" if row["uniprot_ids"] else ""
        )
    write_csv(output / "training_antigens.csv", annotations)
    print(f"Antigens: {len(annotations)} chain occurrences annotated", flush=True)


def summarize(antibodies, antigens, train):
    summary = []

    def add(rows, entity, unit, columns):
        for plot, field in columns:
            counts = Counter(row[field] or "Unassigned" for row in rows)
            for category, count in counts.most_common():
                summary.append(dict(entity=entity, unit=unit, plot=plot,
                                    category=category, count=count, denominator=len(rows),
                                    percent=100.0 * count / len(rows)))

    for chain in ("heavy", "light"):
        rows = [row for row in antibodies if row["target_chain"] == chain]
        if not rows:
            continue
        columns = [("V", "v_family"), ("J", "j_family"), ("species", "germline_species")]
        add(rows, chain, "training_chain_example", columns)
        unique = list({row["sequence"]: row for row in rows}.values())
        add(unique, chain, "unique_sequence", columns)
    columns = [("protein", "protein_label"), ("species", "source_organism")]
    add(antigens, "antigen", "antigen_chain_occurrence", columns)
    # Unique sequences can occur with different metadata. Use a combined label
    # rather than choosing an arbitrary annotation for that sequence.
    by_sequence = defaultdict(list)
    for row in antigens:
        by_sequence[row["sequence"]].append(row)
    unique = [{field: " | ".join(sorted({row[field] for row in rows}))
               for _, field in columns} for rows in by_sequence.values()]
    add(unique, "antigen", "unique_sequence", columns)
    add([dict(chain_count=str(len(row["antigen_chains"].split(",")))) for row in train],
        "antigen", "training_complex", [("chain_count", "chain_count")])
    return summary


def plot(records_path, train, output, title="MANGO training set", source_panel=False,
         antigen_notes=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import plot_common as pc

    antibodies = read_csv(output / "training_antibodies.csv")
    antigens = read_csv(output / "training_antigens.csv")
    expected_ids = {row["id"] for row in train}
    for rows in (antibodies, antigens):
        if {row["id"] for row in rows} != expected_ids or any(row["split"] != "train" for row in rows):
            raise ValueError("Annotations do not match the current training set")
    summary = summarize(antibodies, antigens, train)
    if source_panel:
        source_counts = Counter(" + ".join(sorted(set(row["source"].split(";")))) for row in train)
        summary.extend(dict(entity="dataset", unit="training_record", plot="source",
                            category=source, count=count, denominator=len(train),
                            percent=100.0 * count / len(train))
                       for source, count in source_counts.most_common())
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(25, 23),
                             gridspec_kw={"height_ratios": [1.7, 1, 1]})
    color = pc.PALETTE[0]
    for ax, dimension, panel_title in zip(axes[0, :2], ("protein", "species"),
                                    ("Antigen proteins", "Antigen source organisms")):
        counts = [row for row in summary if row["entity"] == "antigen"
                  and row["unit"] == "antigen_chain_occurrence" and row["plot"] == dimension]
        shown = counts[:10]
        other = sum(row["count"] for row in counts[10:])
        if other:
            shown = shown + [dict(category=f"Other ({len(counts) - 10} categories)", count=other)]
        labels = [textwrap.fill(row["category"], 38) for row in shown]
        bars = ax.barh(labels, [row["count"] for row in shown], color=color)
        ax.invert_yaxis()
        ax.set_xscale("log")
        pc.style(ax, panel_title, xlabel=("Antigen record occurrences (log scale)" if source_panel
                                         else "Antigen chain occurrences (log scale)"))
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for bar, row in zip(bars, shown):
            ax.annotate(f" {row['count']:,}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center", fontsize=10)
        ax.set_xlim(left=0.8, right=max(row["count"] for row in shown) * 2)
        ax.tick_params(axis="y", labelsize=10)
    ax = axes[0, 2]
    if source_panel:
        shown = source_counts.most_common(10)
        other = sum(count for _, count in source_counts.most_common()[10:])
        if other:
            shown.append((f"Other ({len(source_counts) - 10} source combinations)", other))
        bars = ax.barh([textwrap.fill(label, 33) for label, _ in shown],
                       [count for _, count in shown], color=color)
        pc.style(ax, "ASD data sources", xlabel="Training records (log scale)")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(left=0.8, right=max(count for _, count in shown) * 2)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for bar, (_, count) in zip(bars, shown):
            ax.annotate(f" {count:,}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center", fontsize=10)
    else:
        counts = Counter(len(row["antigen_chains"].split(",")) for row in train)
        bars = ax.bar([str(k) for k in sorted(counts)], [counts[k] for k in sorted(counts)], color=color)
        pc.style(ax, "Antigen chains per complex", xlabel="Number of antigen chains", ylabel="Training complexes")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        pc.label_bars(ax, bars, [counts[k] for k in sorted(counts)], fmt="{:,.0f}")
    for row_index, chain in enumerate(("heavy", "light"), start=1):
        for ax, dimension, xlabel in zip(axes[row_index], ("V", "J", "species"),
                                          ("V Gene Family", "J Gene Family", "Nearest-germline species (ANARCI)")):
            counts = [row for row in summary if row["entity"] == chain
                      and row["unit"] == "training_chain_example" and row["plot"] == dimension]
            bars = ax.bar([row["category"] for row in counts], [row["count"] for row in counts],
                          color=pc.PALETTE[0 if chain == "heavy" else 1])
            pc.style(ax, f"{chain.capitalize()} chain", xlabel=xlabel, ylabel="Training chain examples")
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)
            ax.tick_params(axis="x", rotation=90 if dimension == "V" else 55, labelsize=10)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
    record_label = "records" if source_panel else "complexes"
    fig.suptitle(f"{title}: {len(train):,} {record_label}; {len(antibodies):,} antibody chain examples", y=1.01)
    fig._mango_layout_rect = (0, 0.01, 1, 0.99)
    # The complete CSV contains every category; only the antigen panels group a tail.
    pc.save(fig, pd.DataFrame(summary), output / "training_distributions.png",
            output / "training_distributions_data.csv", 300)
    stats = dict(
        records_csv=str(records_path), records_sha256=hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
        split="train", training_complexes=len(train), training_antibody_chain_examples=len(antibodies),
        training_records=len(train), examples_by_chain=dict(Counter(row["target_chain"] for row in antibodies)),
        unique_heavy_sequences=len({row["sequence"] for row in antibodies if row["target_chain"] == "heavy"}),
        unique_light_sequences=len({row["sequence"] for row in antibodies if row["target_chain"] == "light"}),
        unique_paired_antibody_sequences=len({(row["resolved_H_seq"], row["resolved_L_seq"]) for row in train}),
        antigen_chain_occurrences=len(antigens), unique_antigen_chain_sequences=len({row["sequence"] for row in antigens}),
        unique_antigen_complex_sequences=len({row["resolved_ag_seq"] for row in train}),
        ab_ag_clusters=len({row["ab_ag_cluster"] for row in train}),
        antigen_clusters=len({row["antigen_cluster"] for row in antigens if row.get("antigen_cluster")}) or None,
        antigen_protein_categories=len({row["protein_key"] for row in antigens}),
        antigen_unknown_name_occurrences=sum(row["protein_name"] == "Unknown" for row in antigens),
        antigen_unknown_source_occurrences=sum(row["source_organism"] == "Unknown" for row in antigens),
        antigen_metadata_mapping=dict(Counter(row["metadata_mapping"] for row in antigens)),
        antibody_assignment_status=dict(Counter(row["status"] for row in antibodies)),
        notes=["Primary plots count occurrences, retaining repeated training records.",
               "Antigen proteins are grouped by UniProt accession set when available, otherwise by PDB name.",
               "Antigen source organisms come from RCSB polymer source metadata, not expression hosts.",
               "ANARCI nearest-germline species is a sequence assignment, not deposited donor species.",
               "Antigen panels display the ten largest categories and an explicitly aggregated Other bar.",
               "CSV percentages include unassigned/unknown categories; unique-sequence summaries are also included."],
    )
    if source_panel:
        stats.pop("training_complexes")
        stats["notes"][:3] = list(antigen_notes or [])
    write_json(output / "training_distributions_summary.json", stats)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="artifacts/data/sabdab2_v0.1.0/processed/records.csv")
    parser.add_argument("--source-split", default="artifacts/data/sabdab2_v0.1.0/splits_final/abag_split.csv")
    parser.add_argument("--out-dir", default="artifacts/analysis/training_distributions")
    parser.add_argument("--stage", choices=("antibodies", "antigens", "plot"), required=True)
    parser.add_argument("--ncpu", type=int, default=4)
    parser.add_argument("--anarci-cache", help="Share sequence annotations across related cohorts")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, train = training_rows(args.records)
    if args.stage == "antibodies":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        annotate_antibodies(train, output, args.ncpu, args.anarci_cache)
    elif args.stage == "antigens":
        annotate_antigens(train, args.source_split, output)
    else:
        plot(args.records, train, output)


if __name__ == "__main__":
    main()
