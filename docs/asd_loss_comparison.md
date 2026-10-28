# ESM2 loss and pretraining comparison

`./run_gpu.sh asd-loss-comparison` runs the requested eight final models:

| SAbDab2 training route | Standard cross-entropy | Focal loss |
|---|---|---|
| From scratch | MANGO prefix; cross-attention MANGO | MANGO prefix; cross-attention MANGO |
| ASD pretraining → SAbDab2 fine-tuning | MANGO prefix; cross-attention MANGO | MANGO prefix; cross-attention MANGO |

Every model uses frozen ESM2 650M antigen embeddings. Antibody variable-domain
sequences remain output targets only; a learned selector requests heavy/light
chains. Decoder, projection and selector weights start randomly for scratch
runs and ASD pretraining. Each fine-tuning run loads **all** matching weights
from its own architecture/loss ASD checkpoint and starts a fresh optimizer.
There are four ASD pretraining stages and eight final SAbDab2 stages, for twelve
training stages in total. Constant/random/pooled/shuffled arms are not part of
this comparison.

Earlier runs and the default `train_mango.py` behavior use ordinary
cross-entropy/NLL. Focal loss is explicitly enabled only for the new focal
arms. The token objective is

```text
mean(-(1 - p(correct next token))^gamma * log p(correct next token))
```

with gamma = 2, uniform alpha = 1, and no amino-acid class weights. It applies
only to target residues and EOS, with the same causal shift and antigen/selector
mask as standard MANGO. Its probability-dependent weight remains in autograd.
This is the multiclass token application of [Lin et al.'s focal loss](https://arxiv.org/abs/1708.02002),
not a binary binding classifier. Dataset family imbalance is still present;
focal modulation changes the contribution of easy tokens, not antigen sampling.
The focal arms use focal loss in **both** ASD pretraining and native fine-tuning.

All models are selected and early-stopped by **ordinary validation NLL**, not
focal objective values. All evaluation tables, perplexities and final plots use
ordinary target-token NLL, including EOS. Training metrics record both
`train_objective` / `val_objective` and `train_nll` / `val_nll`; the legacy
`train_loss` / `val_loss` fields retain the NLL values for checkpoint checking.
Live focal curves label their y axis as focal objective. Every iteration is
flushed to CSV; large-run plots sample one in 100 training iterations and refresh
every 1,000 iterations plus epoch ends.

The [audited ASD cohort](asd_transfer.md) is reused unchanged: 605,585 distinct
training chain examples, including BUZZ high/medium/low bins, plus independent
validation/test components. All eight final models use exactly the same native
cohort: 509 training, 56 validation, and all original 426 test records. The
362-record confidently CATH-assigned test subset remains primary; the full 426
remain secondary. This native training subset differs from the earlier full
SAbDab2/IgLM figure, so that figure alone cannot isolate a pretraining benefit.

The existing optimizer, learning rate 1e-4, seed 0, shuffled training order,
ten-epoch maximum, token-normalized gradients and patience of two validation
epochs are retained. All GPU jobs request `gpu=1`. The launcher runs separate
Snakemake invocations for each endpoint so one failed model does not prevent
later models from being attempted. It runs the four native scratch endpoints
first; then each matching ASD-pretrained/fine-tuned endpoint and ASD evaluation.
ASD has roughly 600,000 training examples, so this is a substantially longer
queue than the prior AbRank experiment.

Configuration is `config/asd_loss_comparison.yaml`. Outputs are under
`artifacts/experiments/asd_focal_comparison_v1/`:

- `sabdab2_scratch/<objective>/<architecture>/`: native-only checkpoints,
  metrics, live curves and evaluations.
- `asd/<objective>/<architecture>/`: the four pretraining stages.
- `sabdab2_finetuned/<objective>/<architecture>/`: matching transfer runs.
- `results_report.json` / `.csv`: available results, current training iteration
  and missing-stage logs, refreshed after each queued endpoint.
- `plots/sabdab2_loss_comparison.png` / `.csv`: the strict eight-model comparison
  after all final evaluations finish.
- `logs/`: individual stage logs; `launcher.log` captures the background queue
  when launched with redirected output.

Rerun the same command to reuse completed jobs and cached antigen tensors.
An interrupted unfinished training job restarts its training phase; it does
not load a partial optimizer checkpoint automatically. Changes to a pinned
configuration or cohort require a new experiment root.
