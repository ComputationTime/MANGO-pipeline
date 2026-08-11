"""Likelihood inputs for handbook Figure 1 (train and cluster-held-out test)."""


rule analysis_likelihood:
    input:
        [eval_json(tag) for tag in ANALYSIS_EMBEDDERS],
