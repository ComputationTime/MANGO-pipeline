"""Antigen + requested chain type -> antibody chain, using a causal GPT-2 head.

Prefix: projected antigen, learned H/L selector, BOS. Only target residues and
EOS receive labels. No antibody sequence enters the conditioning prefix.
"""

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]


def import_mango(dotted: str, mango_dir=None):
    """Import a mango.* submodule without executing the package __init__."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    mango_dir = Path(mango_dir) if mango_dir else REPO_ROOT / "mango"
    if "mango" not in sys.modules:
        pkg = types.ModuleType("mango")
        pkg.__path__ = [str(mango_dir)]
        pkg.__file__ = str(mango_dir / "__init__.py")  # mango_utils reads this
        sys.modules["mango"] = pkg
    if "mango.utils" not in sys.modules:
        upkg = types.ModuleType("mango.utils")
        upkg.__path__ = [str(mango_dir / "utils")]
        sys.modules["mango.utils"] = upkg
    return importlib.import_module(dotted)


class MangoModel(nn.Module):
    """Antigen-only conditioning with an explicit heavy/light selector."""

    ARCHITECTURE = "antigen_chain_gpt2_v3"
    TASK = "antigen+chain_type->chain"
    CHAIN_IDS = {"heavy": 0, "light": 1}

    @staticmethod
    def normalize_chain(chain):
        aliases = {"h": "heavy", "heavy": "heavy", "l": "light", "light": "light"}
        try:
            return aliases[str(chain).lower()]
        except KeyError:
            raise ValueError(f"Unknown chain {chain!r}; choose heavy/H or light/L") from None

    def __init__(self, d_ag: int, mango_dir=None, target_chains=("heavy", "light")):
        super().__init__()
        mu = import_mango("mango.utils.mango_utils", mango_dir)
        self.vocab = dict(mu.ablang_vocab)
        self.decode = {i: c for c, i in self.vocab.items()}
        import copy
        self.configs = copy.deepcopy(mu.MANGO_configs)
        self.configs.n_embd = 488
        self.bad_word_ids = list(mu.BAD_WORD_IDS)
        self.d_model = self.configs.hidden_size

        from transformers import GPT2LMHeadModel

        self.lm = GPT2LMHeadModel(self.configs)
        self.chain_embedding = nn.Embedding(2, self.d_model)
        # Projection width must not alter decoder initialization or dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.antigen_projection = nn.Linear(d_ag, self.d_model)
        self.target_chains = tuple(self.normalize_chain(c) for c in target_chains)
        if not self.target_chains or len(set(self.target_chains)) != len(self.target_chains):
            raise ValueError("target_chains must be nonempty and unique")
        self.d_ag = d_ag

    # --- tokenisation --------------------------------------------------------
    def target_token_ids(self, sequence: str) -> "torch.Tensor":
        """(1, n+2) decoder tokens for the target chain: BOS + residues + EOS.

        One token per residue -- MANGO's vocab is AbLang2's, so unknown symbols
        fall back to 'X' exactly as AbLang2 would tokenise them.
        """
        ids = (
            [self.vocab["<"]]
            + [self.vocab.get(c, self.vocab["X"]) for c in sequence]
            + [self.vocab[">"]]
        )
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def n_target_tokens(target_ids: "torch.Tensor") -> int:
        """How many tokens the loss is actually computed over: len(target) + 1 (EOS)."""
        return int(target_ids.shape[1]) - 1

    # --- forward paths -------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get("architecture") != cls.ARCHITECTURE:
            raise ValueError(
                "Incompatible MANGO checkpoint: antigen + chain conditioning requires "
                "fresh training; checkpoints from earlier conditioning architectures "
                "cannot be reused."
            )
        model = cls(d_ag=int(checkpoint["d_ag"]), target_chains=checkpoint["target_chains"])
        model.chain_embedding.load_state_dict(checkpoint["chain_embedding"])
        model.antigen_projection.load_state_dict(checkpoint["antigen_projection"])
        model.lm.load_state_dict(checkpoint["lm"])
        return model

    def _check_length(self, length):
        if length > self.configs.n_positions:
            raise ValueError(
                f"Antigen + chain selector + target sequence requires {length} positions, "
                f"exceeding GPT-2 capacity {self.configs.n_positions}. "
                "Reduce antigen length or generation length."
            )

    def _prefix(self, h_ag, chain):
        """Identical training/generation prefix; no antibody sequence argument."""
        chain = self.normalize_chain(chain)
        if chain not in self.target_chains:
            raise ValueError(f"Checkpoint was not trained for {chain} chains")
        device = next(self.parameters()).device
        antigen = self.antigen_projection(h_ag)
        start = self.lm.transformer.wte(
            torch.tensor([[self.vocab["<"]]], device=device)
        )
        selector = self.chain_embedding(torch.tensor([[self.CHAIN_IDS[chain]]], device=device))
        prefix = torch.cat([antigen, selector, start], dim=1)
        self._check_length(prefix.shape[1])
        return prefix

    def loss(self, h_ag, chain, target_ids):
        """Teacher-forced target-chain cross-entropy.

        Loss covers the target residues and the end token only: the context block
        is labelled -100, so the number here is a target-chain NLL regardless of
        how long the antigen happens to be.
        """
        device = next(self.parameters()).device
        target_ids = target_ids.to(device)
        prefix = self._prefix(h_ag, chain)  # (1, m+2, d_model), ends on '<'
        # target_ids[:, 1:] is target + EOS -- '<' is already in the prefix.
        tail = self.lm.transformer.wte(target_ids[:, 1:])  # (1, n+1, d_model)
        inputs = torch.cat([prefix, tail], dim=1)
        self._check_length(inputs.shape[1])

        labels = torch.full(
            (1, inputs.shape[1]), -100, dtype=torch.long, device=device
        )
        # GPT2 shifts internally: the logit at position t is scored against
        # labels[t+1]. Placing target+EOS immediately after the '<' position makes
        # BOS predict the first residue and the last residue predict EOS.
        labels[0, prefix.shape[1] :] = target_ids[0, 1:]
        return self.lm(inputs_embeds=inputs, labels=labels).loss

    @torch.no_grad()
    def generate_chain(self, h_ag, chain, max_new_tokens, do_sample, top_p,
                       temperature):
        """Sample a requested chain given the antigen and the requested chain type.

        The prompt is byte-identical to training's prefix, so sampling matches
        the trained conditional. The true requested chain is never supplied.
        """
        return self.generate_chain_batch(
            h_ag, chain, batch_size=1, max_new_tokens=max_new_tokens,
            do_sample=do_sample, top_p=top_p, temperature=temperature,
        )[0]

    @torch.no_grad()
    def generate_chain_batch(self, h_ag, chain, batch_size, max_new_tokens,
                             do_sample, top_p, temperature):
        """Sample several requested chains from one conditioning pair at once.

        All members of a generation batch share the same antigen/chain-type
        prefix. Expanding that prefix lets Hugging Face perform each decoding
        step for the whole batch in one GPU call, which is substantially faster
        than invoking ``generate`` once per design.
        """
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        seed = self._prefix(h_ag, chain).expand(
            int(batch_size), -1, -1
        ).contiguous()
        self._check_length(seed.shape[1] + int(max_new_tokens))
        out = self.lm.generate(
            inputs_embeds=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.vocab[">"],
            pad_token_id=self.vocab["-"],
            bad_words_ids=self.bad_word_ids,  # blocks specials incl. '|' -> residues only
        )
        return [row.tolist() for row in out]

    def decode_chain(self, ids) -> str:
        """Ids -> chain AA string, stopping at end/sep, dropping special tokens."""
        specials = {"<", "-", ">", "*", "X", "|"}
        chars = []
        for i in ids:
            c = self.decode.get(int(i), "")
            if c in (">", "|"):  # end of chain / heavy-light separator
                break
            if c and c not in specials:
                chars.append(c)
        return "".join(chars)


# --- embedding loading -------------------------------------------------------
def load_embedding(path: str) -> "torch.Tensor":
    """Load a saved embedding as (1, L, H)."""
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
    canonical = [[row.get(k, "") for k in fields] for row in rows]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()
