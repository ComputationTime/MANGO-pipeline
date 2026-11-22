"""Score prepared AlphaSeq scFvs and store FLAb average perplexity.

IgLM, AntiBERTy, and AbLang2 return a mean log-likelihood. Each trained MANGO
model is a separate checkpoint: pass its name, for example ``esm2`` or
``one_hot``, plus that checkpoint. The plotted column is
``{model}_average_perplexity``.

IgLM and AntiBERTy import from the base environment. AbLang2 and MANGO import
from the MANGO environment.
"""

from pathlib import Path
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flab_common import apply_chain_nlls, log_likelihoods_to_nlls


PIPELINE_SCRIPTS = Path(__file__).resolve().parents[1]


def _limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["target", "heavy_sequence", "light_sequence"]
    ).reset_index(drop=True)
    if limit is None:
        return ordered
    if limit < 1:
        raise ValueError("--limit must be positive")
    return ordered.head(int(limit)).reset_index(drop=True)


def _unique(sequences) -> list[str]:
    return list(dict.fromkeys(sequences))


def _score_map(sequences, score_one, label: str) -> dict[str, float]:
    scores = {}
    total = len(sequences)
    for index, sequence in enumerate(sequences, start=1):
        try:
            scores[sequence] = float(score_one(sequence))
        except Exception as exc:
            scores[sequence] = float("nan")
            print(f"{label} failed on a {len(sequence)}-residue chain: {exc}", flush=True)
        if index == 1 or index % 25 == 0 or index == total:
            print(f"{label}: {index}/{total}", flush=True)
    return scores


def score_iglm(frame: pd.DataFrame) -> pd.DataFrame:
    from iglm import IgLM

    model = IgLM()

    def score_chain(chain_token):
        def score_one(sequence):
            return model.log_likelihood(sequence, chain_token, "[HUMAN]")
        return score_one

    heavy = log_likelihoods_to_nlls(
        _score_map(_unique(frame["heavy_sequence"]), score_chain("[HEAVY]"), "IgLM heavy")
    )
    light = log_likelihoods_to_nlls(
        _score_map(_unique(frame["light_sequence"]), score_chain("[LIGHT]"), "IgLM light")
    )
    return apply_chain_nlls(frame, "iglm", heavy, light)


def score_antiberty(frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    os.environ.setdefault("TQDM_DISABLE", "1")
    from antiberty import AntiBERTyRunner

    model = AntiBERTyRunner()

    def score_one(sequence):
        values = model.pseudo_log_likelihood([sequence], batch_size=int(batch_size))
        return float(values.detach().reshape(-1)[0].cpu())

    heavy = log_likelihoods_to_nlls(
        _score_map(_unique(frame["heavy_sequence"]), score_one, "AntiBERTy heavy")
    )
    light = log_likelihoods_to_nlls(
        _score_map(_unique(frame["light_sequence"]), score_one, "AntiBERTy light")
    )
    return apply_chain_nlls(frame, "antiberty", heavy, light)


def score_ablang2(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    import ablang2

    scripts = str(PIPELINE_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import device_common

    device = str(device_common.get_device("AbLang2 FLAb scoring"))
    model = ablang2.pretrained(model_to_use=model_name, random_init=False, ncpu=1, device=device)

    def score_many(sequences, chain_type: str) -> dict[str, float]:
        if chain_type == "heavy":
            inputs = [(sequence, "") for sequence in sequences]
        else:
            inputs = [("", sequence) for sequence in sequences]
        scores = {}
        chunk = 16
        for start in range(0, len(inputs), chunk):
            batch_inputs = inputs[start:start + chunk]
            values = model([list(item) for item in batch_inputs], mode="pseudo_log_likelihood")
            for item, value in zip(batch_inputs, values):
                sequence = item[0] if chain_type == "heavy" else item[1]
                scores[sequence] = float(value)
            done = min(start + chunk, len(inputs))
            print(f"AbLang2 {chain_type}: {done}/{len(inputs)}", flush=True)
        return log_likelihoods_to_nlls(scores)

    heavy = score_many(_unique(frame["heavy_sequence"]), "heavy")
    light = score_many(_unique(frame["light_sequence"]), "light")
    return apply_chain_nlls(frame, "ablang2", heavy, light)


def _parse_embeddings(items) -> dict[str, str]:
    embeddings = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"antigen embedding must be TARGET=PATH, got {item!r}")
        target, path = item.split("=", 1)
        embeddings[target] = path
    return embeddings


def score_trained_model(
    frame: pd.DataFrame,
    model_name: str,
    checkpoint: str,
    embeddings: dict[str, str],
) -> pd.DataFrame:
    """Teacher-forced perplexity from one already-trained MANGO checkpoint."""
    import torch

    scripts = str(PIPELINE_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import device_common
    import model_common

    missing = sorted(set(frame["target"]) - set(embeddings))
    if missing:
        raise ValueError(f"no antigen embedding for {missing}")
    device = device_common.get_device(f"MANGO FLAb scoring {model_name}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_common.MangoModel.from_checkpoint(payload).to(device).eval()
    antigen = {
        target: model_common.load_embedding(path).to(device)
        for target, path in embeddings.items()
    }
    cache: dict[tuple[str, str, str], float] = {}

    def chain_nll(target: str, chain: str, sequence: str) -> float:
        key = (target, chain, sequence)
        if key not in cache:
            if chain not in model.target_chains:
                raise ValueError(f"checkpoint was not trained for {chain} chains")
            target_ids = model.target_token_ids(sequence)
            with torch.no_grad():
                loss = model.loss(antigen[target], chain, target_ids)
            cache[key] = float(loss.item())
        return cache[key]

    pieces = []
    for target, group in frame.groupby("target", sort=False):
        heavy = {
            sequence: chain_nll(target, "heavy", sequence)
            for sequence in _unique(group["heavy_sequence"])
        }
        light = {
            sequence: chain_nll(target, "light", sequence)
            for sequence in _unique(group["light_sequence"])
        }
        print(
            f"{model_name} {target}: {len(heavy)} heavy and {len(light)} light chains",
            flush=True,
        )
        pieces.append(apply_chain_nlls(group, model_name, heavy, light))
    return pd.concat(pieces, ignore_index=True)


def score(frame, model, limit, ablang2_model, batch_size, checkpoint, embeddings):
    selected = _limit(frame, limit)
    if model == "iglm":
        return score_iglm(selected)
    if model == "antiberty":
        return score_antiberty(selected, batch_size)
    if model == "ablang2":
        return score_ablang2(selected, ablang2_model)
    if not checkpoint or not embeddings:
        raise ValueError(
            f"{model} is a trained MANGO model and requires --checkpoint and --antigen-embedding"
        )
    return score_trained_model(selected, model, checkpoint, _parse_embeddings(embeddings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="iglm, antiberty, ablang2, or the name of an already-trained MANGO model",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ablang2-model", default="ablang2-paired")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--antigen-embedding", action="append", default=[])
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.prepared], ignore_index=True)
    scored = score(
        frame,
        args.model,
        args.limit,
        args.ablang2_model,
        args.batch_size,
        args.checkpoint,
        args.antigen_embedding,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)
    ok = int(scored[f"{args.model}_status"].eq("ok").sum())
    print(f"{args.model}: scored {ok}/{len(scored)} antibodies -> {args.out}", flush=True)
    if ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
