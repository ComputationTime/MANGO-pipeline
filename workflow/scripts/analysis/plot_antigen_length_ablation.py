"""Train/test NLL versus constant/random/token-merged ESM2 context size."""

from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc


SPLITS = ("train", "test")
LEVELS = (None, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
LABELS = ("1 token", "¼L", "½L", "¾L", "L")
ARCHITECTURE = "antigen_chain_gpt2_v3"
MODES = ("constant", "random", "esm2_merged")
MERGING = "cosine_bipartite_weighted_v1"


def _level(spec):
    method = spec.get("method")
    if method == "esm2" and spec.get("size") == "t33_650M":
        return "esm2_merged", Fraction(1)
    if method == "antigen_pooled" and spec.get("pooling") == "mean_over_antigen_tokens_v1" and spec.get("output_tokens") == 1:
        return "esm2_merged", None
    if method == "antigen_merged":
        if spec.get("merging") != MERGING:
            raise ValueError("unsupported length-ablation token merging algorithm")
        mode = "esm2_merged"
    elif method == "antigen_control" and spec.get("mode") in {"constant", "random"}:
        mode = spec["mode"]
    else:
        raise ValueError("length ablation requires constant/random or ESM2 token-merging inputs")
    if spec.get("shape_mode", "fixed") == "fixed":
        if int(spec.get("tokens", 1)) != 1:
            raise ValueError("the fixed-length endpoint must contain exactly one token")
        return mode, None
    if spec.get("shape_mode") != "match_source":
        raise ValueError("unsupported length-ablation shape_mode")
    fraction = Fraction(str(spec.get("length_fraction", 1)))
    if fraction not in LEVELS[1:]:
        raise ValueError("length ablation fractions must be 1/4, 1/2, 3/4 or 1")
    return mode, fraction


def plot_antigen_length_ablation(eval_jsons, reference_eval, embedders, specs,
                                 dpi, out_figure, out_data, iglm_eval=None):
    if len(eval_jsons) != len(embedders) or len(set(embedders)) != len(embedders):
        raise ValueError("one evaluation is required for each unique ablation tag")
    with open(reference_eval) as handle:
        reference = json.load(handle)
    if reference.get("embedder") != "esm2" or int(reference.get("d_ag", 0)) != 1280:
        raise ValueError("the reference must be an ESM2 650M evaluation with width 1280")
    fingerprint = reference.get("cohort_sha256")
    if not fingerprint:
        raise ValueError("reference evaluation must record its cohort fingerprint")
    payloads = {}
    for path in eval_jsons:
        with open(path) as handle:
            payload = json.load(handle)
        tag = payload.get("embedder")
        if tag in payloads:
            raise ValueError(f"duplicate length-ablation evaluation for {tag!r}")
        payloads[tag] = payload
    if set(payloads) != set(embedders):
        raise ValueError("length-ablation evaluation tags do not match configuration")

    # Historical reference evaluations lack context statistics. Full-length
    # shape controls measure precisely the same ESM2 token counts in this cohort.
    full_controls = [payloads[tag] for tag in embedders
                     if _level(specs[tag]) in (("constant", Fraction(1)), ("random", Fraction(1)))]
    for split in SPLITS:
        if len(full_controls) == 2:
            first, second = [p.get("splits", {}).get(split, {}).get("antigen_context_tokens")
                             for p in full_controls]
            if first != second:
                raise ValueError("full-length controls disagree on ESM2 context lengths")

    seen = set()
    rows = []
    for tag in embedders:
        spec, payload = specs[tag], payloads[tag]
        mode, level = _level(spec)
        is_reference = tag == "esm2" and spec.get("method") == "esm2"
        if is_reference and payload != reference:
            raise ValueError("the ESM2 L endpoint must reuse the reference evaluation")
        if (mode, level) in seen:
            raise ValueError(f"duplicate {mode} context level {level}")
        seen.add((mode, level))
        if payload.get("cohort_sha256") != fingerprint:
            raise ValueError(f"{tag}: cohort fingerprint differs from ESM2")
        architecture = payload.get("architecture")
        prefix = payload.get("antigen_in_decoder_prefix")
        if is_reference and architecture is None:
            # Older reference files record architecture in the model contract.
            with open(Path(reference_eval).parent / "model_config.json") as handle:
                architecture = json.load(handle).get("architecture")
            prefix = architecture == ARCHITECTURE
        if architecture != ARCHITECTURE or prefix is not True:
            raise ValueError(f"{tag}: length controls require the common prefix architecture")
        if int(payload["d_ag"]) != 1280 or int(spec.get("width", 1280)) != 1280:
            raise ValueError(f"{tag}: embedding width must match ESM2 (1280)")
        if payload.get("loss_reduction") != reference.get("loss_reduction") or payload.get("loss_reduction") != "target_token_mean_including_eos":
            raise ValueError(f"{tag}: incompatible loss reduction")
        for split in SPLITS:
            values = payload.get("splits", {}).get(split)
            baseline = reference.get("splits", {}).get(split)
            if not values or not baseline:
                raise ValueError(f"{tag}: missing {split} evaluation")
            for key in ("n_examples", "n_tokens"):
                if int(values[key]) <= 0 or values[key] != baseline[key]:
                    raise ValueError(f"{tag}: {split} {key} differs from ESM2")
            if set(values.get("by_chain", {})) != {"heavy", "light"}:
                raise ValueError(f"{tag}: pooled heavy/light evaluation is required")
            for chain in ("heavy", "light"):
                for key in ("n_examples", "n_tokens"):
                    if values["by_chain"][chain][key] != baseline.get("by_chain", {}).get(chain, {}).get(key):
                        raise ValueError(f"{tag}: {split} {chain} cohort differs from ESM2")
            lengths = values.get("antigen_context_tokens")
            lengths_source = tag
            if is_reference and not lengths and len(full_controls) == 2:
                lengths = full_controls[0].get("splits", {}).get(split, {}).get("antigen_context_tokens")
                lengths_source = full_controls[0]["embedder"]
            if not lengths:
                raise ValueError(f"{tag}: re-evaluate to record actual antigen context lengths")
            if (
                lengths["n_records"] != values["by_chain"]["heavy"]["n_examples"]
                or not 1 <= lengths["min"] <= lengths["mean"] <= lengths["max"]
                or (level is None and (lengths["min"] != 1 or lengths["max"] != 1))
            ):
                raise ValueError(f"{tag}: inconsistent actual antigen context lengths")
            if not all(math.isfinite(float(value)) for value in (
                values["nll"], values["perplexity"], baseline["nll"], lengths["mean"]
            )):
                raise ValueError(f"{tag}: non-finite length-ablation metrics")
            rows.append({
                "embedder": tag, "mode": mode,
                "model_family": "MANGO",
                "context_order": LEVELS.index(level),
                "context_label": LABELS[LEVELS.index(level)],
                "length_fraction": float(level) if level is not None else None,
                "fixed_context_tokens": 1 if level is None else None,
                "context_tokens_min": lengths["min"],
                "context_tokens_mean": lengths["mean"],
                "context_tokens_max": lengths["max"],
                "context_lengths_source": lengths_source,
                "n_records": lengths["n_records"],
                "embedding_width": int(payload["d_ag"]),
                "architecture": architecture,
                "token_merging": ("identity" if is_reference else
                                  spec.get("merging", spec.get("pooling", "none"))),
                "cohort_sha256": fingerprint,
                "run_id": payload.get("run_id"),
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                "split": split, "nll": values["nll"],
                "perplexity": values["perplexity"],
                "n_examples": values["n_examples"], "n_tokens": values["n_tokens"],
                "esm2_reference_nll": baseline["nll"],
                "esm2_reference_run_id": reference.get("run_id"),
            })
    if seen != {(mode, level) for mode in MODES for level in LEVELS}:
        raise ValueError("all five context levels are required for constant, random and token-merged ESM2")

    # Each fractional ESM2 context must actually match the corresponding control
    # lengths, not merely advertise the same fraction in its configuration.
    for level in LEVELS:
        for split in SPLITS:
            group = [row for row in rows if row["context_order"] == LEVELS.index(level)
                     and row["split"] == split]
            for key in ("context_tokens_min", "context_tokens_mean", "context_tokens_max", "n_records"):
                if len({row[key] for row in group}) != 1:
                    raise ValueError(f"{split} {level}: context lengths differ between curves")

    if iglm_eval is not None:
        with open(iglm_eval) as handle:
            iglm = json.load(handle)
        if iglm.get("cohort_sha256") != fingerprint:
            raise ValueError("IgLM cohort fingerprint differs from MANGO")
        if (iglm.get("embedder") != "iglm_scratch"
                or iglm.get("architecture") != "iglm_gpt2_autoregressive_v1"
                or iglm.get("initialization") != "random_from_scratch"
                or iglm.get("pretrained_weights_loaded") is not False
                or iglm.get("antigen_conditioning") is not False
                or iglm.get("antibody_prompt_used") is not False
                or iglm.get("objective") != "full_chain_autoregressive_v1"
                or iglm.get("loss_reduction") != "target_token_mean_including_eos"
                or iglm.get("target_contract") != "imgt_variable_vh_vl_v1"):
            raise ValueError("incompatible scratch IgLM evaluation protocol")
        model_config = iglm.get("model_config", {})
        if (iglm.get("trainable_parameters") != 12889600
                or any(model_config.get(key) != value for key, value in
                       dict(n_layer=4, n_head=8, n_embd=512, n_positions=512, vocab_size=33).items())):
            raise ValueError("the IgLM reference must use the published full architecture")
        for split in SPLITS:
            values = iglm.get("splits", {}).get(split, {})
            baseline = reference["splits"][split]
            if any(values.get(key) != baseline[key] for key in ("n_examples", "n_tokens")):
                raise ValueError(f"IgLM {split} example/token counts differ from MANGO")
            if set(values.get("by_chain", {})) != {"heavy", "light"}:
                raise ValueError("IgLM requires pooled heavy/light evaluation")
            for chain in ("heavy", "light"):
                if any(values["by_chain"][chain].get(key) != baseline["by_chain"][chain][key]
                       for key in ("n_examples", "n_tokens")):
                    raise ValueError(f"IgLM {split} {chain} cohort differs from MANGO")
            if not all(math.isfinite(float(values.get(key, float("nan"))))
                       for key in ("nll", "perplexity")):
                raise ValueError("non-finite IgLM evaluation metrics")
            rows.append(dict(
                embedder="iglm_scratch", mode="iglm_scratch", model_family="IgLM",
                context_order=None, context_label="No antigen", length_fraction=None,
                fixed_context_tokens=0, context_tokens_min=0, context_tokens_mean=0,
                context_tokens_max=0, context_lengths_source="no_antigen",
                n_records=values["by_chain"]["heavy"]["n_examples"], embedding_width=0,
                architecture=iglm["architecture"], token_merging="none",
                initialization=iglm["initialization"], species_token=iglm.get("species_token"),
                objective=iglm["objective"], trainable_parameters=iglm["trainable_parameters"],
                cohort_sha256=fingerprint, run_id=iglm.get("run_id"),
                checkpoint_epoch=iglm.get("checkpoint_epoch"), split=split,
                nll=values["nll"], perplexity=values["perplexity"],
                n_examples=values["n_examples"], n_tokens=values["n_tokens"],
                esm2_reference_nll=baseline["nll"],
                esm2_reference_run_id=reference.get("run_id"),
            ))

    data = pd.DataFrame(rows).sort_values(["split", "mode", "context_order"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, split, title in zip(axes, SPLITS, ("Training split", "Test split")):
        for mode in MODES:
            points = data.loc[(data["split"] == split) & (data["mode"] == mode)].sort_values("context_order")
            color = pc.EMBEDDER_COLORS["esm2" if mode == "esm2_merged" else f"control_{mode}"]
            ax.plot(points["context_order"], points["nll"],
                    marker="s" if mode == "constant" else "o",
                    markersize=7 if mode == "constant" else 5,
                    markerfacecolor="white" if mode == "constant" else color,
                    color=color, label="ESM2 (token merging)" if mode == "esm2_merged" else mode.capitalize())
        ax.axhline(reference["splits"][split]["nll"], linestyle="--",
                   color=pc.EMBEDDER_COLORS["esm2"], alpha=0.5, label="ESM2 (full L)")
        if iglm_eval is not None:
            ax.axhline(iglm["splits"][split]["nll"], linestyle=":", linewidth=2,
                       color=pc.PALETTE[5], label="IgLM (scratch; no antigen)")
        ax.set_xticks(range(len(LEVELS)), LABELS)
        pc.style(ax, title, xlabel="Context size (L = per-record ESM2 length)",
                 ylabel="Pooled heavy/light NLL")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3 if iglm_eval else 4,
               bbox_to_anchor=(0.5, -0.06 if iglm_eval else -0.01))
    pc.save(fig, data, out_figure, out_data, dpi)


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("plot_antigen_length_ablation.py must run through Snakemake")
    plot_antigen_length_ablation(
        list(smk.input.evals), smk.input.reference, list(smk.params.embedders),
        {tag: dict(spec) for tag, spec in dict(smk.params.specs).items()},
        smk.params.dpi, smk.output.figure, smk.output.data,
        iglm_eval=smk.input.iglm[0] if smk.input.iglm else None,
    )


if __name__ == "__main__":
    main()
