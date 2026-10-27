"""Train/evaluate full IgLM from scratch as an antigen-free SAbDab2 baseline.

Architecture and vocabulary follow Graylab/IgLM's public full-model contract.
Full-chain autoregression, cohort, optimization and token reduction match the
MANGO comparison. No public pretrained checkpoint or antibody prompt is used.
Imports stay lightweight so Snakemake can compute the run identity.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARCHITECTURE = "iglm_gpt2_autoregressive_v1"
VOCAB = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         *"ACDEFGHIKLMNPQRSTVWY", "[CAMEL]", "[HUMAN]", "[MOUSE]",
         "[RABBIT]", "[RAT]", "[RHESUS]", "[HEAVY]", "[LIGHT]")
MODEL_CONFIG = dict(
    vocab_size=33, n_layer=4, n_head=8, n_embd=512, n_inner=2048,
    n_positions=512, n_ctx=512, activation_function="gelu_new",
    attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1,
    initializer_range=0.02, layer_norm_epsilon=1e-5,
    condition_on_chain=True, condition_on_species=True,
    bos_token_id=2, eos_token_id=3, pad_token_id=0,
)
SPECIES = frozenset(VOCAB[25:31])
LOSS_REDUCTION = "target_token_mean_including_eos"


def experiment_hash(config):
    model_cfg = dict(config["model"])
    for key in ("retrain", "loss_plot_interval"):
        model_cfg.pop(key, None)
    scientific = dict(dataset=config["dataset"], processing=config["processing"],
                      training=model_cfg, baseline=config["iglm_baseline"],
                      architecture=ARCHITECTURE, model_config=MODEL_CONFIG,
                      vocabulary=VOCAB, experiment_seed=config["experiment"]["seed"])
    return hashlib.sha1(json.dumps(scientific, sort_keys=True).encode()).hexdigest()[:8]


class IgLMScratch:
    """The full IgLM GPT-2, with native chain/species tokens and [SEP] EOS."""

    def __init__(self, spec, model_config=None):
        from transformers import GPT2Config, GPT2LMHeadModel

        if spec.get("initialization") != "random_from_scratch":
            raise ValueError("this baseline requires random_from_scratch initialization")
        if spec.get("objective") != "full_chain_autoregressive_v1":
            raise ValueError("this baseline requires full-chain autoregressive training")
        if spec.get("species_token") not in SPECIES:
            raise ValueError("a fixed native IgLM species token is required")
        self.spec = dict(spec)
        self.vocab = {token: index for index, token in enumerate(VOCAB)}
        self.model_config = dict(MODEL_CONFIG if model_config is None else model_config)
        # Deliberately instantiate the architecture; never call from_pretrained.
        self.lm = GPT2LMHeadModel(GPT2Config(**self.model_config))

    def encode(self, sequence, chain):
        import torch

        if chain not in {"heavy", "light"}:
            raise ValueError(f"unknown IgLM chain type {chain!r}")
        if not sequence or any(residue not in VOCAB[5:25] for residue in sequence):
            raise ValueError("IgLM requires a nonempty sequence of canonical amino acids")
        tokens = [f"[{chain.upper()}]", self.spec["species_token"], *sequence, "[SEP]"]
        if len(tokens) > self.model_config["n_positions"]:
            raise ValueError("IgLM target exceeds its positional capacity")
        ids = torch.tensor([[self.vocab[token] for token in tokens]], dtype=torch.long,
                           device=next(self.lm.parameters()).device)
        labels = ids.clone()
        labels[:, :2] = -100  # Only residues and EOS contribute to NLL.
        return ids, labels

    def loss(self, sequence, chain):
        ids, labels = self.encode(sequence, chain)
        return self.lm(input_ids=ids, labels=labels, use_cache=False).loss


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_checkpoint(path, payload):
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(records_csv, model_cfg, spec, run_dir, run_hash, global_state,
          model_config=None, initial_checkpoint=None):
    import os
    import time
    import torch
    from device_common import get_device
    from model_common import comparison_cohort
    from train_mango import _rows, normalize_gradients
    from training_curve import write_training_plot
    from training_order import epoch_training_rows
    from vram_monitor import VramMonitor

    fingerprint = comparison_cohort(records_csv)
    if set(model_cfg["train_splits"]) & set(model_cfg["val_splits"]):
        raise ValueError("training and validation splits overlap")
    chains = tuple(model_cfg["target_chains"])
    if set(chains) != {"heavy", "light"} or len(chains) != 2:
        raise ValueError("the IgLM comparison requires both heavy and light chains")
    train_rows = _rows(records_csv, model_cfg["train_splits"], chains)
    val_rows = _rows(records_csv, model_cfg["val_splits"], chains)
    if not train_rows or not val_rows:
        raise ValueError("nonempty training and validation cohorts are required")
    device = get_device("training IgLM from scratch")
    torch.manual_seed(int(model_cfg["seed"]))
    model = IgLMScratch(spec, model_config)
    transfer = None
    if initial_checkpoint:
        source = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        source_meta = source["metadata"]
        expected = dict(architecture=ARCHITECTURE, model_config=model.model_config,
                        vocabulary=list(VOCAB), objective=spec["objective"],
                        species_token=spec["species_token"], target_chains=list(chains),
                        target_contract="imgt_variable_vh_vl_v1", loss_reduction=LOSS_REDUCTION,
                        initialization="random_from_scratch", pretrained_weights_loaded=False,
                        antigen_conditioning=False, antibody_prompt_used=False)
        for key, value in expected.items():
            if source_meta.get(key) != value:
                raise ValueError(f"IgLM initialization checkpoint has incompatible {key}")
        if source_meta.get("transfer_initialization"):
            raise ValueError("IgLM pretraining must start from scratch")
        model.lm.load_state_dict(source["lm"], strict=True)
        transfer = dict(path=str(initial_checkpoint),
                        sha256=hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest(),
                        source_cohort_sha256=source_meta["cohort_sha256"],
                        source_epoch=source["epoch"], optimizer_loaded=False)
        del source
    model.lm.to(device)
    # Validate every target before spending GPU time on training.
    for row in train_rows + val_rows:
        model.encode(row.sequence, row.chain)
    optimizer = torch.optim.Adam(model.lm.parameters(), lr=float(model_cfg["lr"]))
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    metadata = dict(
        embedder="iglm_scratch", run_id=run_id, experiment_hash=run_hash,
        architecture=ARCHITECTURE, model_family="IgLM", task="chain_type->chain",
        cohort_sha256=fingerprint, target_contract="imgt_variable_vh_vl_v1",
        target_chains=list(chains), loss_reduction=LOSS_REDUCTION,
        initialization="asd_checkpoint" if transfer else "random_from_scratch",
        pretrained_weights_loaded=bool(transfer), public_pretrained_weights_loaded=False,
        transfer_initialization=transfer,
        training_loss={"name": "cross_entropy"}, checkpoint_selection="validation_nll",
        objective=spec["objective"], species_token=spec["species_token"],
        species_conditioning="fixed_token_not_record_species",
        antigen_conditioning=False, antibody_prompt_used=False,
        model_config=model.model_config, vocabulary=list(VOCAB),
        trainable_parameters=sum(p.numel() for p in model.lm.parameters()),
        training_settings=dict(model_cfg),
    )
    _write_json(run_dir / "model_config.json", metadata)
    _write_json(run_dir / "config.json", global_state)
    print(f"[{run_id}] {metadata['initialization']}; {metadata['trainable_parameters']} parameters; "
          f"train={len(train_rows)} val={len(val_rows)} device={device}", flush=True)
    history = []
    iteration = 0
    accumulation = max(1, int(model_cfg["grad_accum"]))
    interval = max(1, int(model_cfg.get("loss_plot_interval", 250)))
    monitor = VramMonitor(os.getpid())
    monitor.start()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started, status = time.monotonic(), "failed"
    try:
        with (run_dir / "training_curve.csv").open("w", newline="") as curve, (run_dir / "metrics.jsonl").open("w") as metrics:
            writer = csv.DictWriter(curve, fieldnames=["iteration", "epoch", "phase", "loss", "record_id", "chain_type"])
            writer.writeheader()

            def record(epoch, phase, loss, row=None):
                point = dict(iteration=iteration, epoch=epoch, phase=phase, loss=loss,
                             record_id=row.id if row else "", chain_type=row.chain if row else "")
                history.append(point)
                writer.writerow(point)
                curve.flush()

            def epoch_loss(rows, training, epoch):
                nonlocal iteration
                model.lm.train(training)
                total = total_tokens = window_tokens = 0
                optimizer.zero_grad()
                for index, row in enumerate(rows, start=1):
                    with torch.set_grad_enabled(training):
                        loss = model.loss(row.sequence, row.chain)
                    value, tokens = float(loss.item()), len(row.sequence) + 1
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite IgLM loss for {row.id}/{row.chain}")
                    if training:
                        (loss * tokens).backward()
                        window_tokens += tokens
                        if index % accumulation == 0 or index == len(rows):
                            normalize_gradients(model.lm.parameters(), window_tokens)
                            optimizer.step()
                            optimizer.zero_grad()
                            window_tokens = 0
                        iteration += 1
                        record(epoch, "train", value, row)
                        if iteration % interval == 0:
                            write_training_plot(history, run_dir / "training_curve.png", run_id)
                    total += value * tokens
                    total_tokens += tokens
                return total / total_tokens

            best, stale = float("inf"), 0
            for epoch in range(int(model_cfg["epochs"])):
                ordered = epoch_training_rows(train_rows, seed=int(model_cfg["seed"]),
                                              epoch=epoch, shuffle=bool(model_cfg.get("shuffle_train", True)))
                tr = epoch_loss(ordered, True, epoch)
                va = epoch_loss(val_rows, False, epoch)
                record(epoch, "train_epoch", tr)
                record(epoch, "validation", va)
                write_training_plot(history, run_dir / "training_curve.png", run_id)
                improved = va < best
                metrics.write(json.dumps(dict(epoch=epoch, iteration=iteration, train_loss=tr,
                                              val_loss=va, improved=improved, run_id=run_id,
                                              train_order_seed=int(model_cfg["seed"]) + epoch)) + "\n")
                metrics.flush()
                checkpoint = dict(metadata=metadata, lm=model.lm.state_dict(),
                                  optimizer=optimizer.state_dict(), epoch=epoch, val_loss=va)
                _save_checkpoint(run_dir / "checkpoints/latest.pt", checkpoint)
                if improved:
                    best, stale = va, 0
                    _save_checkpoint(run_dir / "checkpoints/best.pt", checkpoint)
                else:
                    stale += 1
                print(f"[{run_id}] epoch {epoch + 1}: train={tr:.4f} val={va:.4f}"
                      f"{' *' if improved else ''}", flush=True)
                if stale >= int(model_cfg["patience"]) and not improved:
                    break
        if not (run_dir / "checkpoints/best.pt").is_file():
            raise RuntimeError("no IgLM best checkpoint was written")
        status = "complete"
    finally:
        gpu = str(device).startswith("cuda")
        if gpu:
            torch.cuda.synchronize()
        _write_json(run_dir / "training_runtime.json", dict(
            status=status, embedder="iglm_scratch", wall_seconds=time.monotonic() - started,
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if gpu else 0,
            peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if gpu else 0,
            **monitor.finish()))


def evaluate(records_csv, checkpoint_path, output, splits=("train", "val", "test"),
             evaluation_subsets=None):
    import torch
    from device_common import get_device
    from model_common import comparison_cohort
    from train_mango import _rows

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if metadata["cohort_sha256"] != comparison_cohort(records_csv):
        raise ValueError("IgLM evaluation cohort differs from training checkpoint")
    if (metadata["architecture"] != ARCHITECTURE or
            metadata.get("public_pretrained_weights_loaded", metadata["pretrained_weights_loaded"])):
        raise ValueError("invalid scratch IgLM checkpoint contract")
    transferred = bool(metadata.get("transfer_initialization"))
    if (metadata["initialization"] != ("asd_checkpoint" if transferred else "random_from_scratch")
            or bool(metadata["pretrained_weights_loaded"]) != transferred):
        raise ValueError("inconsistent IgLM transfer initialization")
    spec = {key: metadata[key] for key in ("initialization", "objective", "species_token")}
    spec["initialization"] = "random_from_scratch"  # Construct before restoring local weights.
    model = IgLMScratch(spec, metadata["model_config"])
    model.lm.load_state_dict(checkpoint["lm"], strict=True)
    model.lm.to(get_device("evaluation IgLM from scratch")).eval()
    result = dict(metadata, checkpoint_epoch=checkpoint["epoch"],
                  checkpoint_val_loss=checkpoint["val_loss"], evaluation_loss="cross_entropy", splits={})
    planned = [(split, _rows(records_csv, [split], metadata["target_chains"])) for split in splits]
    for label, subset in (evaluation_subsets or {}).items():
        if label in splits:
            raise ValueError(f"Evaluation subset duplicates split: {label}")
        ids = set(subset["ids"])
        rows = [r for r in _rows(records_csv, [subset["source_split"]], metadata["target_chains"])
                if r.id in ids]
        if not ids or {r.id for r in rows} != ids:
            raise ValueError(f"Evaluation subset {label} has missing or unknown IDs")
        planned.append((label, rows))
    with torch.no_grad():
        for split, rows in planned:
            if not rows:
                raise ValueError(f"missing IgLM evaluation split {split}")
            by_chain = {}
            total = tokens = examples = 0
            for chain in metadata["target_chains"]:
                chain_total = chain_tokens = chain_examples = 0
                for row in rows:
                    if row.chain != chain:
                        continue
                    value = float(model.loss(row.sequence, row.chain).item())
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite IgLM evaluation for {row.id}")
                    n_tokens = len(row.sequence) + 1
                    chain_total += value * n_tokens
                    chain_tokens += n_tokens
                    chain_examples += 1
                if not chain_examples:
                    raise ValueError(f"missing {chain} examples in IgLM {split}")
                nll = chain_total / chain_tokens
                by_chain[chain] = dict(nll=nll, perplexity=math.exp(min(nll, 50)),
                                       n_tokens=chain_tokens, n_examples=chain_examples,
                                       n_skipped_missing_embeddings=0)
                total += chain_total
                tokens += chain_tokens
                examples += chain_examples
            nll = total / tokens
            result["splits"][split] = dict(nll=nll, perplexity=math.exp(min(nll, 50)),
                                            n_tokens=tokens, n_examples=examples,
                                            by_chain=by_chain, n_skipped_missing_embeddings=0)
            print(f"[IgLM] {split}: nll={nll:.4f} perplexity={math.exp(min(nll, 50)):.4f}", flush=True)
    _write_json(output, result)
    return result


def main():
    smk = globals().get("snakemake")
    if smk is None:
        raise RuntimeError("iglm_baseline.py must run through Snakemake")
    if smk.params.action == "train":
        train(smk.input.records, dict(smk.params.model_cfg), dict(smk.params.spec),
              smk.params.run_dir, smk.params.run_hash, dict(smk.params.global_state))
    else:
        evaluate(smk.input.records, smk.input.checkpoint, smk.output.evaluation,
                 list(smk.params.splits))


if __name__ == "__main__":
    main()
