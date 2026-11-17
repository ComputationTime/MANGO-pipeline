"""Compute Figure 6-style distributions for the frozen ASD training cohorts.

Use --stage metadata in an existing PyArrow environment, --stage antibodies
in analysis_anarci, and --stage plot in analysis_plot. No model inputs change.
"""

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chain_examples import example_chains
import plot_training_distributions as common


ROOT = Path("artifacts/analysis/asd_training_distributions")
RAW = Path("artifacts/data/asd/asd_2.0")
COHORTS = {
    "pilot_2000": ("artifacts/experiments/asd_her2_free_prototype_v1/cohort/asd_records.csv",
                   "artifacts/experiments/asd_her2_free_prototype_v1/asd/cross_entropy/mango/model_config.json"),
    **{f"scaling_{size}": (
        f"artifacts/experiments/asd_scaling_fixed_positions_v1/n{size}/cohort/asd_records.csv",
        f"artifacts/experiments/asd_scaling_fixed_positions_v1/runs/n{size}_fixed/model_config.json")
       for size in (2000, 4000, 8000, 16000)},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_digest(rows):
    """Match model_common.comparison_cohort without importing a model stack."""
    fields = ("id", "split", "ab_ag_cluster", "resolved_H_seq", "resolved_L_seq",
              "resolved_ag_seq", "antigen_chains", "target_contract", "example_target_chains")
    return hashlib.sha256(json.dumps([[row.get(key, "") for key in fields] for row in rows],
                                    separators=(",", ":")).encode()).hexdigest()


def load_cohorts():
    cohorts = {}
    for name, (records, model) in COHORTS.items():
        rows, train = common.training_rows(records)
        audit = json.loads(Path(records).with_name("split_audit.json").read_text())
        if sha256(records) != audit["sha256"][records]:
            raise ValueError(f"Frozen ASD cohort changed: {records}")
        trained = json.loads(Path(model).read_text())
        if cohort_digest(rows) != trained["cohort_sha256"]:
            raise ValueError(f"Cohort does not match trained model: {model}")
        for row in train:
            example_chains(row)
            if row.get("has_buzz") == "True" or row["antigen_id"] in audit.get("excluded_her2_antigen_ids", []):
                raise ValueError(f"HER2/BUZZ exclusion differs from the audited cohort: {name}")
        cohorts[name] = train
    return cohorts


def extract_source_metadata(cohorts):
    import bisect
    import pyarrow as pa
    import pyarrow.parquet as pq

    needed = {}
    for train in cohorts.values():
        for row in train:
            identifier = int(row["source_row_id"])
            sequence = row["resolved_ag_seq"]
            if identifier in needed and needed[identifier] != sequence:
                raise ValueError("Source row has conflicting antigen sequences")
            needed[identifier] = sequence
    provenance = dict(raw_manifest_sha256=sha256(RAW / "download_manifest.json"),
                      cohorts={name: sha256(paths[0]) for name, paths in COHORTS.items()})
    output = ROOT / "selected_source_metadata.json"
    pinned = ROOT / "selected_source_metadata_provenance.json"
    if output.exists() and pinned.exists() and json.loads(pinned.read_text()) == provenance:
        return json.loads(output.read_text())
    manifest = json.loads((RAW / "download_manifest.json").read_text())
    identifiers, offset, metadata = sorted(needed), 0, {}
    for name in sorted(key for key in manifest["files"] if key.endswith(".parquet")):
        file = pq.ParquetFile(RAW / name)
        verified = False
        for group in range(file.num_row_groups):
            count = file.metadata.row_group(group).num_rows
            selected = identifiers[bisect.bisect_right(identifiers, offset):
                                   bisect.bisect_right(identifiers, offset + count)]
            if selected:
                if not verified:
                    if sha256(RAW / name) != manifest["files"][name]["sha256"]:
                        raise ValueError(f"Raw ASD file changed: {name}")
                    verified = True
                table = file.read_row_group(group, columns=["metadata", "chains.target_chains", "dataset"])
                rows = table.take(pa.array([identifier - offset - 1 for identifier in selected])).to_pylist()
                for identifier, row in zip(selected, rows):
                    if needed[identifier] not in dict(row["chains"]["target_chains"]).values():
                        raise ValueError(f"Source antigen does not match cohort: {identifier}")
                    metadata[str(identifier)] = {**row["metadata"], "dataset": row["dataset"]}
            offset += count
    if len(metadata) != len(needed) or offset != manifest["rows_from_delta_statistics"]:
        raise ValueError("ASD source row coverage or frozen snapshot size differs")
    common.write_json(output, metadata)
    common.write_json(pinned, provenance)
    print(f"Verified {len(metadata):,} selected source rows in the pinned ASD snapshot", flush=True)
    return metadata


def uniprot_annotation(payload):
    description = payload.get("proteinDescription") or {}
    name = (description.get("recommendedName") or {}).get("fullName", {}).get("value")
    if not name:
        submitted = description.get("submissionNames") or description.get("submittedName") or []
        name = submitted[0].get("fullName", {}).get("value") if submitted else None
    return dict(accession=payload["primaryAccession"], protein_name=name or "Unknown",
                source_organism=(payload.get("organism") or {}).get("scientificName") or "Unknown",
                retrieved_utc=datetime.now(timezone.utc).isoformat())


def fetch_uniprot(accessions):
    path = ROOT / "uniprot_metadata_cache.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    pending = sorted(set(accessions) - cache.keys())
    for start in range(0, len(pending), 50):
        requested = pending[start:start + 50]
        parameters = dict(query=" OR ".join("accession:" + accession for accession in requested),
                          format="json", fields="accession,protein_name,organism_name", size=500)
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(parameters)
        with urllib.request.urlopen(url, timeout=60) as handle:
            results = json.load(handle)["results"]
        for result in results:
            cache[result["primaryAccession"]] = uniprot_annotation(result)
        # Follow retired/secondary accession redirects individually rather than
        # assigning an unrelated result from a multi-accession response.
        for accession in set(requested) - cache.keys():
            url = f"https://rest.uniprot.org/uniprotkb/{accession}.json?" + urllib.parse.urlencode(
                dict(fields="accession,protein_name,organism_name"))
            try:
                with urllib.request.urlopen(url, timeout=60) as handle:
                    payload = json.load(handle)
                cache[accession] = uniprot_annotation(payload)
            except urllib.error.HTTPError as exc:
                if exc.code not in (404, 410):
                    raise
                cache[accession] = dict(accession=accession, protein_name="Unknown",
                                       source_organism="Unknown", status="unavailable")
        common.write_json(path, cache)
        print(f"UniProt: {min(start + 50, len(pending))}/{len(pending)} accession annotations", flush=True)
    return cache


def matching_entities(sequence, entry):
    """Identify the antigen in a complex by strong exact sequence matches."""
    matches = []
    words = {sequence[index:index + 20] for index in range(len(sequence) - 19)}
    for entity in entry.get("polymer_entities") or []:
        reference = "".join(entity["entity_poly"]["pdbx_seq_one_letter_code_can"].split())
        if not any(reference[index:index + 20] in words for index in range(len(reference) - 19)):
            continue
        blocks = difflib.SequenceMatcher(None, sequence, reference, autojunk=False).get_matching_blocks()
        coverage = sum(block.size for block in blocks) / min(len(sequence), len(reference))
        longest = max(block.size for block in blocks)
        if coverage >= 0.9 and longest >= min(40, 0.8 * min(len(sequence), len(reference))):
            matches.append((coverage, longest, entity))
    if not matches:
        return []
    best = max(item[0] for item in matches)
    return [entity for coverage, _, entity in matches if coverage >= best - 0.01]


def prepare_metadata(cohorts):
    metadata = extract_source_metadata(cohorts)
    accession_ids = {row["target_uniprot"].upper() for row in metadata.values() if row["target_uniprot"]}
    uniprot = fetch_uniprot(accession_ids)
    cache = ROOT / "rcsb_entry_cache.json"
    native_cache = Path("artifacts/analysis/training_distributions/rcsb_entry_cache.json")
    if not cache.exists() and native_cache.exists():
        shutil.copyfile(native_cache, cache)
    pdb_ids = {row["target_pdb"].upper() for row in metadata.values() if row["target_pdb"]}
    entries = common.fetch_entries(pdb_ids, ROOT)
    by_antigen = defaultdict(dict)
    sequences = {}
    for train in cohorts.values():
        for row in train:
            by_antigen[row["antigen_id"]][row["source_row_id"]] = metadata[row["source_row_id"]]
            sequences[row["antigen_id"]] = row["resolved_ag_seq"]
    annotations = {}
    for antigen, sources in by_antigen.items():
        accessions = {row["target_uniprot"].upper() for row in sources.values() if row["target_uniprot"]}
        raw_names = Counter(row["target_name"] for row in sources.values() if row["target_name"])
        names, organisms = set(), set()
        if accessions:
            for accession in accessions:
                annotation = uniprot[accession]
                names.add(annotation["protein_name"])
                organisms.add(annotation["source_organism"])
            method = "ASD_UniProt_accession"
        else:
            matched = []
            for pdb in {row["target_pdb"].upper() for row in sources.values() if row["target_pdb"]}:
                matched.extend(matching_entities(sequences[antigen], entries[pdb]["entry"]))
            for entity in matched:
                accessions.update(entity["rcsb_polymer_entity_container_identifiers"].get("uniprot_ids") or [])
                description = entity["rcsb_polymer_entity"]
                preferred = (description.get("rcsb_polymer_name_combined") or {}).get("names")
                names.add("; ".join(preferred) if preferred else description.get("pdbx_description") or "Unknown")
                organisms.update(source.get("ncbi_scientific_name") or source.get("scientific_name") or "Unknown"
                                 for source in entity.get("rcsb_entity_source_organism") or [])
            method = "RCSB_sequence_match" if matched else "ASD_name_only_or_unknown"
        names.discard("Unknown")
        organisms.discard("Unknown")
        name = "; ".join(sorted(names)) if names else (raw_names.most_common(1)[0][0] if raw_names else "Unknown")
        key = "UniProt:" + ";".join(sorted(accessions)) if accessions else "Name:" + name.lower()
        annotations[antigen] = dict(protein_key=key, protein_name=name,
            uniprot_ids=";".join(sorted(accessions)), source_organism="; ".join(sorted(organisms)) or "Unknown",
            metadata_mapping=method)
    # Use one protein label per accession set, retaining all exact antigen IDs.
    names = defaultdict(Counter)
    for antigen, annotation in annotations.items():
        names[annotation["protein_key"]][annotation["protein_name"]] += 1
    for annotation in annotations.values():
        annotation["protein_name"] = names[annotation["protein_key"]].most_common(1)[0][0]
        annotation["protein_label"] = annotation["protein_name"] + (
            " [" + annotation["uniprot_ids"] + "]" if annotation["uniprot_ids"] else "")
    common.write_json(ROOT / "antigen_metadata.json", annotations)
    for name, train in cohorts.items():
        output = ROOT / name
        output.mkdir(exist_ok=True)
        common.write_csv(output / "training_antigens.csv", [dict(
            id=row["id"], split="train", author_chain="AG", antigen_id=row["antigen_id"],
            sequence=row["resolved_ag_seq"], **annotations[row["antigen_id"]]) for row in train])
    print("Antigen metadata methods:", dict(Counter(row["metadata_mapping"] for row in annotations.values())), flush=True)


def annotate(cohorts, ncpu):
    cache = ROOT / "anarci_sequence_cache.json"
    if not cache.exists():
        source = Path("artifacts/analysis/training_distributions/anarci_sequence_cache.json")
        if source.exists():
            shutil.copyfile(source, cache)
    for name in reversed(cohorts):
        output = ROOT / name
        output.mkdir(exist_ok=True)
        print("Annotating", name, flush=True)
        common.annotate_antibodies(cohorts[name], output, ncpu, cache)


def plots(cohorts):
    import pandas as pd

    notes = ["Primary antigen counts retain each selected ASD record; antibody counts respect example_target_chains.",
             "Antigen identities use ASD UniProt accessions or sequence-matched RCSB entities; unavailable annotations remain Unknown.",
             "Antigen source organisms come from UniProt/RCSB biological-source metadata, not antibody germline species or expression hosts."]
    overview, combined = [], []
    for name, train in cohorts.items():
        output = ROOT / name
        label = "ASD pilot pretraining" if name.startswith("pilot") else "ASD scaling pretraining"
        common.plot(COHORTS[name][0], train, output, title=label, source_panel=True, antigen_notes=notes)
        stats_path = output / "training_distributions_summary.json"
        stats = json.loads(stats_path.read_text())
        stats.update(cohort=name, model_config=COHORTS[name][1], model_cohort_sha256=json.loads(
            Path(COHORTS[name][1]).read_text())["cohort_sha256"],
            target_chain_masks=dict(Counter(row["example_target_chains"] for row in train)))
        common.write_json(stats_path, stats)
        overview.append(dict(cohort=name, training_records=len(train),
            antigen_sequences=stats["unique_antigen_chain_sequences"],
            heavy_examples=stats["examples_by_chain"].get("heavy", 0),
            unique_heavy_sequences=stats["unique_heavy_sequences"],
            light_examples=stats["examples_by_chain"].get("light", 0),
            unique_light_sequences=stats["unique_light_sequences"],
            total_chain_examples=stats["training_antibody_chain_examples"],
            unknown_antigen_names=stats["antigen_unknown_name_occurrences"],
            unknown_antigen_organisms=stats["antigen_unknown_source_occurrences"]))
        rows = common.read_csv(output / "training_distributions_data.csv")
        for row in rows:
            if row["unit"] == "training_complex":
                row["unit"] = "training_record"
            combined.append(dict(cohort=name, **row))
        common.write_csv(output / "training_distributions_data.csv", rows)
    common.write_csv(ROOT / "cohort_summary.csv", overview)
    common.write_json(ROOT / "cohort_summary.json", overview)
    common.write_csv(ROOT / "all_cohorts_distribution_data.csv", combined)
    report = ["# ASD pretraining distributions", "",
        "The five cohorts below match the frozen cohort hashes recorded by trained models. Only training splits are counted. The largest scaling cohort is the main result; the pilot used a different antigen-length cap. The four scaling cohorts are nested, so their counts must not be added together.", "",
        "| Cohort | Records | Antigen sequences | Heavy examples | Unique heavy | Light examples | Unique light |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in overview:
        report.append("| [" + row["cohort"] + "](" + row["cohort"] + "/training_distributions.png) | " + " | ".join(f"{row[key]:,}" for key in
            ("training_records", "antigen_sequences", "heavy_examples", "unique_heavy_sequences", "light_examples", "unique_light_sequences")) + " |")
    report.extend(["", "Each cohort directory contains training_distributions.png, nine individual subplot PNGs, complete category counts/percentages, the summary JSON, and per-example annotations. The V/J/species panels use the same ANARCI settings as the native MANGO analysis: IMGT and unrestricted germline species. Antibody species means nearest germline, not donor species.", "",
        "ASD deduplicates each (antigen, requested chain, target sequence) after splitting. example_target_chains controls which heavy/light targets enter the distributions; the presence of a sequence column alone does not make it a training example. Antigen panels count selected records, preserving the native report's one-antigen-occurrence-per-record convention. ASD stores these antigens as a single sequence input (AG), including engineered constructs.", "",
        "Source annotations are recovered from the exact source_row_id in the pinned ASD Delta version 12 Parquet snapshot. Raw file hashes and the selected antigen sequences are verified. Protein/source-organism metadata use ASD's UniProt accession when present; otherwise they use matching polymer entities in the cited PDB entry. RCSB sequence matches require at least 90% matching residues relative to the shorter sequence and a continuous exact match of at least 40 residues (or 80% of the shorter sequence if shorter). Ambiguous matched source labels are retained together; no donor organism is inferred from antibody sequence. Unknown values remain in all denominators.", "",
        "RCSB and UniProt responses are cached locally with retrieval timestamps. Names are grouped by accession set where available, otherwise by name; antigen sequence counts use exact sequences. The antigen and dataset-source panels show ten leading categories and an explicit Other category; CSVs retain every category. A source combination means the training triple has observations from those sources, counted once. Exact-sequence and occurrence-weighted summaries are both provided.", "",
        "The prepared parent ASD pool has 598,978 training rows and 605,585 selected chain examples (598,902 heavy; 6,683 light), according to its frozen split audit. That full prepared pool has no trained model artifacts in asd_cath_transfer_v1; it is distinct from the smaller cohorts actually used above. The used cohorts are HER2/BUZZ-free.", "",
        "## Reproduce", "", "Run the metadata stage in the existing PyArrow environment, antibodies in analysis_anarci, and plot in analysis_plot:", "", "```bash",
        "python workflow/scripts/analysis/plot_asd_training_distributions.py --stage metadata",
        "python workflow/scripts/analysis/plot_asd_training_distributions.py --stage antibodies --ncpu 4",
        "python workflow/scripts/analysis/plot_asd_training_distributions.py --stage plot", "```", ""])
    (ROOT / "README.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("metadata", "antibodies", "plot"), required=True)
    parser.add_argument("--ncpu", type=int, default=4)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts()
    if args.stage == "metadata":
        prepare_metadata(cohorts)
    elif args.stage == "antibodies":
        annotate(cohorts, args.ncpu)
    else:
        plots(cohorts)


if __name__ == "__main__":
    main()
