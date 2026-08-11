"""Shared model code for the MANGO train/predict rules.

THE TASK
--------
Given the ANTIGEN and the LIGHT chain, predict the HEAVY chain. That single
sentence fixes the whole input/output contract, and every path through this
module -- training, evaluation, reconstruction, de novo generation -- honours
it identically. The heavy chain is never part of the conditioning signal, so
there is nothing to leak.

Sequence layout fed to GPT2 (positions left to right)::

    [ ctx_0 ... ctx_{m-1} ] [ < ] [ h_1 ... h_n ] [ > ]
     antigen-conditioned    start   heavy chain    end
     light-chain context           (teacher-forced)

    labels:  -100 x (m+1)          h_1 ... h_n      >

* ``ctx`` is the precomputed antibody CONTEXT embedding -- AbLang2 run on
  ``'*|L'``, i.e. the light chain with the heavy slot masked -- cross-attended
  against the antigen embedding. Dim 480 = MANGO's hidden size.
* The heavy tokens enter through GPT2's own ``wte``, so teacher forcing is
  exact: the logit at each heavy position sees only earlier heavy positions.
* Labels are -100 across the whole context block, so loss is computed on the
  heavy chain and its end token and nothing else. Every reported NLL is
  therefore a heavy-chain NLL, comparable across antigen representations.

Generation uses the identical prefix (``ctx`` + ``<``) and lets GPT2 continue
autoregressively, so what is sampled is exactly what was trained.

MANGO's 26-symbol vocabulary equals the AbLang2 token ids, so labels come
straight from the sequence via ``ablang_vocab`` -- no AbLang2 at model time.

The cross-attention stack (mango.utils.cross_attention.TransformerStack) and the
vocab/config constants (mango.utils.mango_utils) are imported WITHOUT running
mango/__init__.py, so this stays free of esm/ablang2/pyrosetta.
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
    """Cross-attention fusion (light-chain context <- antigen) + GPT2 LM head.

    Conditioning is antigen + light chain; the output is the heavy chain.
    """

    def __init__(self, d_ag: int, n_heads: int, n_layers: int, mango_dir=None):
        super().__init__()
        xattn = import_mango("mango.utils.cross_attention", mango_dir)
        mu = import_mango("mango.utils.mango_utils", mango_dir)
        self.vocab = dict(mu.ablang_vocab)
        self.decode = {i: c for c, i in self.vocab.items()}
        self.configs = mu.MANGO_configs
        self.bad_word_ids = list(mu.BAD_WORD_IDS)
        self.d_model = self.configs.hidden_size

        from transformers import GPT2LMHeadModel

        self.cross_attn = xattn.TransformerStack(
            d_model=self.d_model, n_heads=n_heads, n_layers=n_layers, d_Ag_rep=d_ag
        )
        self.lm = GPT2LMHeadModel(self.configs)
        self.d_ag = d_ag

    # --- tokenisation --------------------------------------------------------
    def heavy_token_ids(self, heavy: str) -> "torch.Tensor":
        """(1, n+2) decoder tokens for the target heavy chain: '<' + H + '>'.

        One token per residue -- MANGO's vocab is AbLang2's, so unknown symbols
        fall back to 'X' exactly as AbLang2 would tokenise them.
        """
        ids = (
            [self.vocab["<"]]
            + [self.vocab.get(c, self.vocab["X"]) for c in heavy]
            + [self.vocab[">"]]
        )
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def n_target_tokens(heavy_ids: "torch.Tensor") -> int:
        """How many tokens the loss is actually computed over: len(H) + 1 (EOS)."""
        return int(heavy_ids.shape[1]) - 1

    # --- forward paths -------------------------------------------------------
    def fuse(self, x_ctx: "torch.Tensor", h_ag: "torch.Tensor") -> "torch.Tensor":
        """x_ctx:(1,m,480), h_ag:(1,L_ag,d_ag) -> antigen-conditioned (1,m,480)."""
        return self.cross_attn(x_ctx, h_ag)

    def _prefix(self, x_ctx, h_ag):
        """Antigen-conditioned light-chain context + the start token embedding."""
        device = next(self.parameters()).device
        ctx = self.fuse(x_ctx, h_ag)  # (1, m, 480)
        start = self.lm.transformer.wte(
            torch.tensor([[self.vocab["<"]]], device=device)
        )  # (1, 1, 480)
        return torch.cat([ctx, start], dim=1)  # (1, m+1, 480)

    def loss(self, x_ctx, h_ag, heavy_ids):
        """Teacher-forced heavy-chain cross-entropy.

        Loss covers the heavy residues and the end token only: the context block
        is labelled -100, so the number here is a heavy-chain NLL regardless of
        how long the antigen or light chain happen to be.
        """
        device = next(self.parameters()).device
        heavy_ids = heavy_ids.to(device)
        prefix = self._prefix(x_ctx, h_ag)  # (1, m+1, 480), ends on '<'
        # heavy_ids[:, 1:] is H + '>' -- '<' is already in the prefix.
        tail = self.lm.transformer.wte(heavy_ids[:, 1:])  # (1, n+1, 480)
        inputs = torch.cat([prefix, tail], dim=1)

        labels = torch.full(
            (1, inputs.shape[1]), -100, dtype=torch.long, device=device
        )
        # GPT2 shifts internally: the logit at position t is scored against
        # labels[t+1]. Placing H+'>' immediately after the '<' position makes
        # '<' predict h_1 and h_n predict '>'.
        labels[0, prefix.shape[1] :] = heavy_ids[0, 1:]
        return self.lm(inputs_embeds=inputs, labels=labels).loss

    @torch.no_grad()
    def generate_heavy(self, x_ctx, h_ag, max_new_tokens, do_sample, top_p,
                       temperature):
        """Sample a heavy chain given the antigen and the light-chain context.

        The prompt is byte-identical to training's prefix, so sampling matches
        the trained conditional. The true heavy chain is never supplied.
        """
        seed = self._prefix(x_ctx, h_ag)
        out = self.lm.generate(
            inputs_embeds=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=self.vocab[">"],
            pad_token_id=self.vocab["-"],
            bad_words_ids=self.bad_word_ids,  # blocks specials incl. '|' -> heavy only
        )
        return out[0].tolist()

    def decode_heavy(self, ids) -> str:
        """Ids -> heavy AA string, stopping at end/sep, dropping special tokens."""
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
    return payload["embedding"].unsqueeze(0)
