"""Run Chai-1 on selected generated H + target L + antigen complexes."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_confidence_common as sc


def _fasta(chains):
    return "".join(f">protein|name={name}\n{seq}\n" for name, seq in chains)


def run(designs_csv, records_csv, target_id, n_designs, seed, samples,
        recycles, use_msa_server, cache_dir, raw_dir, output_csv,
        use_esm_embeddings=True):
    os.environ["CHAI_DOWNLOADS_DIR"] = str(Path(cache_dir).resolve())
    from chai_lab.chai1 import run_inference

    selected = sc.select_designs(designs_csv, n_designs, seed)
    antigen, light = sc.target_context(records_csv, target_id)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for design in selected.itertuples(index=False):
        name = f"design_{int(design.design_index):06d}"
        job = raw / name
        job.mkdir(parents=True, exist_ok=True)
        fasta = job / "input.fasta"
        fasta.write_text(_fasta(sc.complex_chains(design.sequence, light, antigen)))
        output = job / "prediction"
        output.mkdir(exist_ok=True)
        try:
            candidates = run_inference(
                fasta_file=fasta, output_dir=output,
                num_trunk_recycles=int(recycles), num_diffn_samples=int(samples),
                seed=int(seed) + int(design.design_index), device="cuda:0",
                use_esm_embeddings=bool(use_esm_embeddings),
                use_msa_server=bool(use_msa_server),
            )
            for sample_idx, (path, ranking) in enumerate(
                    zip(candidates.cif_paths, candidates.ranking_data)):
                scores_path = output / f"scores.model_idx_{sample_idx}.npz"
                with np.load(scores_path) as values:
                    ptm = float(np.asarray(values["ptm"]).squeeze())
                    iptm = float(np.asarray(values["iptm"]).squeeze())
                    clash = bool(np.asarray(values["has_inter_chain_clashes"]).squeeze())
                rows.append({
                    "embedder": design.embedder, "run_id": design.run_id,
                    "target_id": design.target_id, "design_index": int(design.design_index),
                    "sequence": design.sequence, "predictor": "chai",
                    "sample_index": sample_idx,
                    "confidence_score": float(ranking.aggregate_score.item()),
                    "ptm": ptm, "iptm": iptm,
                    "complex_plddt": float(candidates.plddt[sample_idx].mean()),
                    "mean_pae": float(candidates.pae[sample_idx].mean()),
                    "has_inter_chain_clashes": clash, "structure_path": str(path),
                    "status": "ok",
                })
        except Exception as exc:
            rows.append({
                "embedder": design.embedder, "run_id": design.run_id,
                "target_id": design.target_id, "design_index": int(design.design_index),
                "sequence": design.sequence, "predictor": "chai", "sample_index": "",
                "confidence_score": "", "ptm": "", "iptm": "",
                "complex_plddt": "", "mean_pae": "",
                "has_inter_chain_clashes": "", "structure_path": "",
                "status": f"error: {type(exc).__name__}: {exc}",
            })
    sc.write_rows(rows, output_csv)
    expected = {int(design.design_index) for design in selected.itertuples(index=False)}
    successful = {
        int(row["design_index"]) for row in rows if row["status"] == "ok"
    }
    if successful != expected:
        missing = sorted(expected - successful)
        raise RuntimeError(
            "Chai did not produce a valid structure for every selected design; "
            f"missing design indices {missing}; diagnostics: {output_csv}"
        )


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("run_chai_confidence.py must run through Snakemake")
    run(smk.input.designs, smk.input.records, smk.wildcards.instance,
        smk.params.n_designs, smk.params.seed, smk.params.samples,
        smk.params.recycles, smk.params.use_msa_server, smk.params.cache,
        smk.output.raw, smk.output.scores)


if __name__ == "__main__":
    main()
