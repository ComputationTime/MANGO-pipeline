"""Script-side helpers around the MANGO model (embedding loading, cohort checks).

The model itself lives in ``mango/model/MANGO.py``.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mango.model.MANGO import MangoModel  # noqa: E402,F401


# --- embedding loading -------------------------------------------------------
def load_embedding(path: str) -> "torch.Tensor":
    """Load a saved embedding as (1, L, H)."""
    from embedding_store import load_stored_embedding
    if not Path(path).is_file():
        return load_stored_embedding(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("model_name", "")).startswith("pyrosetta_") and payload.get("conditioning_scope") != "antigen_only":
        raise ValueError(f"Unsafe legacy PRE cache; recompute antigen-only embedding: {path}")
    return payload["embedding"].unsqueeze(0)


def comparison_cohort(records_csv):
    """Validate target/split contracts and fingerprint the ordered comparison data."""
    import csv
    import hashlib
    import json
    with open(records_csv) as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    clusters = {}
    for row in rows:
        if row.get("target_contract") != "imgt_variable_vh_vl_v1":
            raise ValueError("Re-standardize records: variable VH/VL target contract required")
        if row["id"] in seen:
            raise ValueError(f"Duplicate record: {row['id']}")
        seen.add(row["id"])
        cluster = row.get("ab_ag_cluster")
        if not cluster:
            raise ValueError("Comparison requires ab_ag_cluster for every record")
        if clusters.setdefault(cluster, row["split"]) != row["split"]:
            raise ValueError(f"Cluster crosses splits: {cluster}")
    fields = ("id", "split", "ab_ag_cluster", "resolved_H_seq", "resolved_L_seq", "resolved_ag_seq", "antigen_chains", "target_contract")
    if rows and 'example_target_chains' in rows[0]:
        from chain_examples import example_chains
        for row in rows:
            example_chains(row)
        fields += ('example_target_chains',)
    canonical = [[row.get(k, "") for k in fields] for row in rows]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
