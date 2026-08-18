"""Run Boltz on selected generated H + target L + antigen complexes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_confidence_common as sc


def _yaml(chains):
    lines = ["version: 1", "sequences:"]
    for chain_id, sequence in chains:
        lines += ["  - protein:", f"      id: {chain_id}",
                  f"      sequence: {sequence}", "      msa: empty"]
    return "\n".join(lines) + "\n"


def run(designs_csv, records_csv, target_id, n_designs, seed, samples,
        recycles, use_msa_server, cache_dir, raw_dir, output_csv):
    selected = sc.select_designs(designs_csv, n_designs, seed)
    antigen, light = sc.target_context(records_csv, target_id)
    raw = Path(raw_dir)
    inputs = raw / "inputs"
    outputs = raw / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in selected.itertuples(index=False):
        name = f"design_{int(row.design_index):06d}"
        (inputs / f"{name}.yaml").write_text(
            _yaml(sc.complex_chains(row.sequence, light, antigen)))
        jobs.append((name, row))

    cmd = ["boltz", "predict", str(inputs), "--out_dir", str(outputs),
           "--cache", str(cache_dir), "--accelerator", "gpu", "--devices", "1",
           "--diffusion_samples", str(int(samples)), "--recycling_steps", str(int(recycles)),
           "--output_format", "mmcif", "--write_full_pae"]
    if use_msa_server:
        # Input files deliberately say `msa: empty` by default. Remove that
        # declaration when opting into the external MSA service.
        for path in inputs.glob("*.yaml"):
            path.write_text(path.read_text().replace("      msa: empty\n", ""))
        cmd.append("--use_msa_server")
    env = dict(os.environ)
    env["BOLTZ_CACHE"] = str(Path(cache_dir).resolve())
    subprocess.run(cmd, check=True, env=env)

    rows = []
    for name, design in jobs:
        # Boltz 2.2 places batch results under
        # ``boltz_results_<input-directory>/predictions``. Older releases and
        # some wrappers write ``predictions`` directly below ``--out_dir``.
        # Accept both layouts so normalization is insulated from that wrapper
        # detail.
        candidates = [outputs / "predictions" / name]
        candidates.extend(sorted(outputs.glob(f"boltz_results_*/predictions/{name}")))
        pred = next((path for path in candidates if path.is_dir()), candidates[0])
        confidence_files = sorted(pred.glob(f"confidence_{name}_model_*.json"))
        for fallback_idx, confidence_path in enumerate(confidence_files):
            sample_idx = int(confidence_path.stem.rsplit("_", 1)[-1])
            data = json.loads(confidence_path.read_text())
            pae_path = pred / f"pae_{name}_model_{sample_idx}.npz"
            mean_pae = float("nan")
            if pae_path.is_file():
                with np.load(pae_path) as values:
                    key = "pae" if "pae" in values.files else values.files[0]
                    mean_pae = float(np.asarray(values[key]).mean())
            structure = pred / f"{name}_model_{sample_idx}.cif"
            rows.append({
                "embedder": design.embedder, "run_id": design.run_id,
                "target_id": design.target_id, "design_index": int(design.design_index),
                "sequence": design.sequence, "predictor": "boltz2",
                "sample_index": sample_idx if confidence_files else fallback_idx,
                "confidence_score": data.get("confidence_score"),
                "ptm": data.get("ptm"), "iptm": data.get("iptm"),
                "complex_plddt": data.get("complex_plddt"), "mean_pae": mean_pae,
                "has_inter_chain_clashes": "", "structure_path": str(structure),
                "status": "ok" if structure.is_file() else "missing_structure",
            })
    if not rows:
        raise RuntimeError("Boltz completed without any confidence JSON outputs")
    sc.write_rows(rows, output_csv)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("run_boltz_confidence.py must run through Snakemake")
    run(smk.input.designs, smk.input.records, smk.wildcards.instance,
        smk.params.n_designs, smk.params.seed, smk.params.samples,
        smk.params.recycles, smk.params.use_msa_server, smk.params.cache,
        smk.output.raw, smk.output.scores)


if __name__ == "__main__":
    main()
