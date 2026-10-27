"""Focused, serial-GPU AbRank -> SAbDab2 conditioning-control experiment."""
import json
from pathlib import Path

configfile: "config/abrank_transfer.yaml"

ROOT = config["experiment_root"]
COHORT = config["cohort_dir"]
TAGS = config["controls"]
PHASES = ["abrank", "sabdab2"]

wildcard_constraints:
    phase="abrank|sabdab2",
    tag="|".join(TAGS)


def records(w):
    return f"{COHORT}/{w.phase}_records.csv"


def embedding_tag(tag):
    return "esm2" if tag == "control_cross_attention_esm2" else tag


def marker(w):
    return f"{ROOT}/{w.phase}/embeddings/antigen/{embedding_tag(w.tag)}/complete.json"


def source_evaluation(w):
    return [f"{ROOT}/{w.phase}/runs/esm2/eval.json"] if w.tag in ["control_pooled_esm2", "control_shuffled"] else []


rule all:
    input:
        f"{ROOT}/plots/abrank_controls.png",
        f"{ROOT}/plots/sabdab2_finetuned_controls.png",


rule frozen_esm2:
    input:
        records=records,
        audit=f"{COHORT}/split_audit.json",
        original=config["original_sabdab_records"],
        implementation="workflow/scripts/embed_antigen_esm2.py",
    output:
        marker=f"{ROOT}/{{phase}}/embeddings/antigen/esm2/complete.json",
    params:
        job="esm2",
        study=config,
    log:
        f"{ROOT}/logs/{{phase}}_embed_esm2.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=20000,
    conda:
        "envs/embed_esm2.yaml"
    script:
        "scripts/abrank_transfer_job.py"


rule control_embeddings:
    input:
        records=records,
        audit=f"{COHORT}/split_audit.json",
        source=f"{ROOT}/{{phase}}/embeddings/antigen/esm2/complete.json",
        evaluation=source_evaluation,
        implementation=["workflow/scripts/embed_antigen_control.py", "workflow/scripts/pool_antigen_embeddings.py", "workflow/scripts/shuffle_antigen_embeddings.py"],
    output:
        marker=f"{ROOT}/{{phase}}/embeddings/antigen/{{tag}}/complete.json",
    wildcard_constraints:
        tag="control_constant|control_random|control_constant_l1|control_random_l1|control_pooled_esm2|control_shuffled"
    params:
        job="controls",
        study=config,
    log:
        f"{ROOT}/logs/{{phase}}_embed_{{tag}}.log",
    threads: 2
    resources:
        mem_mb=8000,
    conda:
        "envs/model.yaml"
    script:
        "scripts/abrank_transfer_job.py"


rule train_control:
    input:
        records=records,
        audit=f"{COHORT}/split_audit.json",
        embeddings=marker,
        initialization=lambda w: [f"{ROOT}/abrank/runs/{w.tag}/checkpoints/best.pt"] if w.phase == "sabdab2" else [],
        implementation=["workflow/scripts/train_mango.py", "workflow/scripts/model_common.py", "workflow/scripts/training_order.py"],
    output:
        ckpt=f"{ROOT}/{{phase}}/runs/{{tag}}/checkpoints/best.pt",
        latest=f"{ROOT}/{{phase}}/runs/{{tag}}/checkpoints/latest.pt",
        model_config=f"{ROOT}/{{phase}}/runs/{{tag}}/model_config.json",
        metrics=f"{ROOT}/{{phase}}/runs/{{tag}}/metrics.jsonl",
        curve=f"{ROOT}/{{phase}}/runs/{{tag}}/training_curve.csv",
        plot=f"{ROOT}/{{phase}}/runs/{{tag}}/training_curve.png",
        run_config=f"{ROOT}/{{phase}}/runs/{{tag}}/config.json",
    params:
        job="train",
        study=config,
    log:
        f"{ROOT}/logs/{{phase}}_train_{{tag}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=18000,
    conda:
        "envs/model.yaml"
    script:
        "scripts/abrank_transfer_job.py"


rule evaluate_control:
    input:
        records=records,
        audit=f"{COHORT}/split_audit.json",
        coverage=f"{COHORT}/sabdab2_test_coverage.csv",
        embeddings=marker,
        ckpt=f"{ROOT}/{{phase}}/runs/{{tag}}/checkpoints/best.pt",
        model_config=f"{ROOT}/{{phase}}/runs/{{tag}}/model_config.json",
        implementation="workflow/scripts/evaluate_mango.py",
    output:
        evaluation=f"{ROOT}/{{phase}}/runs/{{tag}}/eval.json",
    params:
        job="evaluate",
        study=config,
    log:
        f"{ROOT}/logs/{{phase}}_eval_{{tag}}.log",
    threads: 2
    resources:
        gpu=1,
        mem_mb=14000,
    conda:
        "envs/model.yaml"
    script:
        "scripts/abrank_transfer_job.py"


rule plot_controls:
    input:
        evaluations=lambda w: [f"{ROOT}/{w.phase}/runs/{tag}/eval.json" for tag in TAGS],
        audit=f"{COHORT}/split_audit.json",
    output:
        figure=f"{ROOT}/plots/{{phase}}_controls.png",
        data=f"{ROOT}/plots/{{phase}}_controls.csv",
    wildcard_constraints:
        phase="abrank"
    params:
        phase="abrank",
        labels=config["labels"],
        tags=TAGS,
    log:
        f"{ROOT}/logs/{{phase}}_plot.log",
    conda:
        "envs/analysis_plot.yaml"
    script:
        "scripts/analysis/plot_abrank_transfer.py"


rule plot_finetuned_controls:
    input:
        evaluations=[f"{ROOT}/sabdab2/runs/{tag}/eval.json" for tag in TAGS],
        audit=f"{COHORT}/split_audit.json",
    output:
        figure=f"{ROOT}/plots/sabdab2_finetuned_controls.png",
        data=f"{ROOT}/plots/sabdab2_finetuned_controls.csv",
    params:
        phase="sabdab2",
        labels=config["labels"],
        tags=TAGS,
    log:
        f"{ROOT}/logs/sabdab2_plot.log",
    conda:
        "envs/analysis_plot.yaml"
    script:
        "scripts/analysis/plot_abrank_transfer.py"
