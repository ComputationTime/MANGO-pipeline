"""Assign nearest heavy V/J germlines with ANARCI and report Levenshtein distance."""

from pathlib import Path

import pandas as pd


def levenshtein(left, right):
    """Unit-cost Levenshtein distance using O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def germline_reference(database, species, v_gene, j_gene):
    """Construct the closest V/J reference over ANARCI's 128 IMGT match states."""
    v = database["V"]["H"][species][v_gene]
    j = database["J"]["H"][species][j_gene]
    if len(v) != 128 or len(j) != 128:
        raise ValueError("ANARCI germline alignments must have 128 IMGT match states")
    merged = [v_aa if v_aa != "-" else j_aa for v_aa, j_aa in zip(v, j)]
    return "".join(aa for aa in merged if aa != "-")


def _domain_sequence(numbered_domain):
    numbering = numbered_domain[0]
    return "".join(aa for _, aa in numbering if aa != "-")


def score_germline(cohort_csv, scheme, allowed_species, ncpu, out_csv):
    from anarci import anarci
    from anarci.germlines import all_germlines

    df = pd.read_csv(cohort_csv, dtype={"sequence": str}, keep_default_na=False)
    inputs = [(str(i), sequence) for i, sequence in enumerate(df["sequence"])]
    numbered, details, _ = anarci(
        inputs, scheme=scheme, output=False, allow={"H"}, assign_germline=True,
        allowed_species=None if allowed_species is None else list(allowed_species),
        ncpu=int(ncpu),
    )

    annotations = []
    for i, row in enumerate(df.itertuples(index=False)):
        result = {
            "anarci_scheme": scheme, "germline_species": "", "v_gene": "",
            "v_identity": float("nan"), "j_gene": "", "j_identity": float("nan"),
            "numbered_heavy_sequence": "", "germline_reference_sequence": "",
            "ld_germline": float("nan"), "ld_germline_normalized": float("nan"),
            "germline_status": "",
        }
        try:
            if numbered[i] is None or details[i] is None:
                raise ValueError("ANARCI found no heavy-chain domain")
            heavy_domains = [
                (domain, detail) for domain, detail in zip(numbered[i], details[i])
                if detail.get("chain_type") == "H"
            ]
            if not heavy_domains:
                raise ValueError("ANARCI found no heavy-chain domain")
            domain, detail = heavy_domains[0]
            genes = detail.get("germlines", {})
            v_assignment, v_identity = genes.get("v_gene", (None, None))
            j_assignment, j_identity = genes.get("j_gene", (None, None))
            if not v_assignment or not j_assignment:
                raise ValueError("ANARCI did not assign both V and J germlines")
            species, v_gene = v_assignment
            j_species, j_gene = j_assignment
            if species != j_species:
                raise ValueError(f"V/J germline species disagree: {species}/{j_species}")

            domain_sequence = _domain_sequence(domain)
            reference = germline_reference(all_germlines, species, v_gene, j_gene)
            distance = levenshtein(domain_sequence, reference)
            denominator = max(len(domain_sequence), len(reference))
            result.update(
                germline_species=species, v_gene=v_gene, v_identity=float(v_identity),
                j_gene=j_gene, j_identity=float(j_identity),
                numbered_heavy_sequence=domain_sequence,
                germline_reference_sequence=reference,
                ld_germline=int(distance),
                ld_germline_normalized=distance / denominator if denominator else float("nan"),
                germline_status="ok",
            )
        except Exception as exc:
            result["germline_status"] = f"error: {type(exc).__name__}: {exc}"
        annotations.append(result)

    out = pd.concat([df, pd.DataFrame(annotations)], axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    n_ok = int((out["germline_status"] == "ok").sum())
    print(f"ANARCI assigned {n_ok}/{len(out)} nearest heavy V/J references -> {out_csv}",
          flush=True)
    return out


def main():
    smk = globals().get("snakemake")
    if smk is not None:
        score_germline(smk.input.cohort, smk.params.scheme,
                       smk.params.allowed_species, smk.threads, smk.output.metrics)
        return
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--scheme", default="imgt")
    p.add_argument("--allowed-species", nargs="*")
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    score_germline(a.cohort, a.scheme, a.allowed_species or None, a.ncpu, a.out)


if __name__ == "__main__":
    main()
