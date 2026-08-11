"""Embed every selected record while loading a heavyweight model only once."""

import csv
import hashlib
import importlib
import json
from pathlib import Path

import torch


MODULES = {
    "one_hot": "embed_antigen_one_hot",
    "biopython": "embed_antigen_biopython",
    "pyrosetta_pre": "embed_antigen_pyrosetta_pre",
    "esm2": "embed_antigen_esm2",
    "esm3": "embed_antigen_esm3",
    "esmif": "embed_antigen_esmif",
    "proteinmpnn": "embed_antigen_proteinmpnn",
    "ablang2": "embed_antibody_ablang2",
}


def _signature(kind, tag, method, spec, seq_source, splits):
    value = {
        "kind": kind,
        "tag": tag,
        "method": method,
        "spec": spec,
        "seq_source": seq_source,
        "splits": list(splits),
        "format": 1,
    }
    blob = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _expected_model(method, spec):
    if method == "one_hot":
        return "one_hot"
    if method == "biopython":
        return "biopython_protparam"
    if method == "pyrosetta_pre":
        return f"pyrosetta_{spec['score_function']}"
    if method == "esm2":
        return f"esm2_{spec['size']}_UR50D"
    if method in {"esm3", "esmif"}:
        return spec["model_name"]
    if method == "proteinmpnn":
        return (
            f"proteinmpnn_{spec['model']}_v_48_{int(spec['noise']):03d}"
        )
    if method == "ablang2":
        return "ablang2-paired"
    raise ValueError(f"unsupported batch method {method!r}")


def _can_reuse(path, row, tag, method, expected_model, dependencies):
    target = Path(path)
    if not target.is_file():
        return False
    try:
        if any(Path(dep).stat().st_mtime > target.stat().st_mtime for dep in dependencies):
            return False
        payload = torch.load(target, map_location="cpu", weights_only=False)
        expected_tag = "ablang2" if method == "ablang2" else tag
        return (
            payload.get("id") == row["id"]
            and payload.get("embedder") == expected_tag
            and payload.get("model_name") == expected_model
            and isinstance(payload.get("shape"), list)
            and payload.get("embedding") is not None
        )
    except Exception:
        return False


def _runtime(module, method, spec, model_dir, weights):
    if method in {"one_hot", "biopython"}:
        return None
    if method == "pyrosetta_pre":
        return module.load_runtime(spec["score_function"])
    if method == "ablang2":
        return module.load_runtime(model_dir)
    if method == "esm2":
        return module.load_runtime(spec["size"])
    if method in {"esm3", "esmif"}:
        return module.load_runtime(spec["model_name"])
    if method == "proteinmpnn":
        return module.load_runtime(weights, spec["model"], int(spec["noise"]))
    raise ValueError(f"unsupported batch method {method!r}")


def _embed_one(module, method, records_csv, row, out, seq_source, tag, spec,
               runtime, model_dir, weights):
    record_id = row["id"]
    if method == "one_hot":
        return module.one_hot(records_csv, record_id, out, seq_source, tag, row=row)
    if method == "biopython":
        return module.biopython(records_csv, record_id, out, seq_source, tag, row=row)
    if method == "ablang2":
        return module.ablang2_embed(
            records_csv, record_id, out, seq_source, model_dir,
            row=row, runtime=runtime,
        )
    if method == "esm2":
        return module.esm2(
            records_csv, record_id, out, seq_source, spec["size"], tag,
            row=row, runtime=runtime,
        )
    if method == "esm3":
        return module.esm3(
            records_csv, record_id, out, seq_source, spec["model_name"], tag,
            row=row, runtime=runtime,
        )
    cif = row["pdb_path"]
    if not Path(cif).is_file():
        raise FileNotFoundError(f"{record_id}: structure does not exist: {cif}")
    if method == "esmif":
        return module.esmif(
            records_csv, record_id, cif, out, spec["model_name"], tag,
            row=row, runtime=runtime,
        )
    if method == "pyrosetta_pre":
        return module.pyrosetta_pre(
            records_csv, record_id, cif, out, spec["score_function"],
            spec.get("normalize", "none"),
            bool(spec.get("add_chain_breaks", True)), tag,
            row=row, runtime=runtime,
        )
    if method == "proteinmpnn":
        return module.proteinmpnn(
            records_csv, record_id, cif, out, weights, spec["model"],
            int(spec["noise"]), tag, row=row, runtime=runtime,
        )
    raise ValueError(f"unsupported batch method {method!r}")


def batch_embed(records_csv, output_dir, marker, kind, tag, method, spec,
                seq_source, splits, implementation, model_dir=None,
                weights=None):
    if method not in MODULES:
        raise ValueError(f"batching is not implemented for {method!r}")
    wanted = set(splits)
    with open(records_csv, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in wanted]
    if not rows:
        raise RuntimeError(f"no records found for splits {sorted(wanted)}")

    signature = _signature(kind, tag, method, spec, seq_source, splits)
    marker_path = Path(marker)
    old_signature = None
    if marker_path.is_file():
        try:
            old_signature = json.loads(marker_path.read_text()).get("signature")
        except Exception:
            pass
    signature_changed = old_signature is not None and old_signature != signature

    module = importlib.import_module(MODULES[method])
    expected_model = _expected_model(method, spec)
    runtime = None
    built = 0
    reused = 0
    for index, row in enumerate(rows, start=1):
        out = Path(output_dir) / row["split"] / f"{row['id']}.pt"
        dependencies = [records_csv, implementation, __file__]
        if method in {"esmif", "proteinmpnn", "pyrosetta_pre"}:
            dependencies.append(row["pdb_path"])
        if weights:
            dependencies.append(weights)
        if not signature_changed and _can_reuse(
            out, row, tag, method, expected_model, dependencies
        ):
            reused += 1
        else:
            if runtime is None:
                runtime = _runtime(module, method, spec, model_dir, weights)
            _embed_one(
                module, method, records_csv, row, str(out), seq_source, tag,
                spec, runtime, model_dir, weights,
            )
            built += 1
        if index % 100 == 0 or index == len(rows):
            print(
                f"[{tag}] {index}/{len(rows)} records "
                f"({built} built, {reused} reused)",
                flush=True,
            )

    result = {
        "status": "complete",
        "kind": kind,
        "tag": tag,
        "method": method,
        "model_name": expected_model,
        "records": len(rows),
        "built": built,
        "reused": reused,
        "splits": list(splits),
        "signature": signature,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[{tag}] batch complete -> {marker}", flush=True)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("embed_batch.py is intended to run through Snakemake")
    batch_embed(
        records_csv=smk.input.records,
        output_dir=smk.params.output_dir,
        marker=smk.output.marker,
        kind=smk.params.kind,
        tag=smk.params.tag,
        method=smk.params.method,
        spec=dict(smk.params.spec),
        seq_source=smk.params.seq_source,
        splits=list(smk.params.splits),
        implementation=smk.input.implementation,
        model_dir=getattr(smk.params, "model_dir", None),
        weights=getattr(smk.input, "weights", None),
    )


if __name__ == "__main__":
    main()
