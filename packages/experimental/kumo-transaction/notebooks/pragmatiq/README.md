# pragmatiq: PRAGMA-like and Blueprint v2, slot by slot

A lab on one multi-source book for two recipes:
[`pragma_like`](../../recipes/pragma_like), Revolut PRAGMA's bidirectional
encoder, and [`blueprint_v2`](../../recipes/blueprint_v2), PRAGMA's event and
profile encoders under a causal history decoder. Written for a training
session on NVIDIA GPUs, it tours a pretrained Blueprint v2 model (00), then
takes it apart slot by slot, following the kit's building-block taxonomy
([ADR 0014](../../docs/architecture/0014-building-block-taxonomy.md)), and
branches at the architecture slot: 03a is the PRAGMA-like encoder, 03b the
Blueprint v2 decoder. Every notebook plots what it teaches. The recipes, not
these notebooks, are the canonical workflow.

Notebooks 00 to 04 are adapted from the GTC Berlin 2026 training lab's
Blueprint v2 track, which runs the same notebooks on SDK 2.0.0a1.

## What you need

- One NVIDIA GPU, and about 30 GB of host memory for the default 4,000-user
  sample (about 7.7 GB per 1,000 users). The recorded runs used one GB10
  (DGX Spark).
- Two downloads, fetched once into `~/.cache/tfm/lab` (or `TFM_LAB_CACHE`):
  - the pragmatiq book `synth_100k_gen2_m31_seed0` (1.8 GB), 100,000 synthetic
    users, generated with pragmatiq 1.1.0;
  - the pretrained lab checkpoint `blueprint_v2_lab_tuned` (45 MB):
    `model.pt`, the tokenizer it was trained with, and its `config.yaml`. It
    is final-sweep candidate D, seed 42: 3,000 pretraining steps on one GB10
    (about 42 minutes), fine-tuned at LoRA rank 16
    ([`blueprint_v2_final_sweep`](../../benchmarks/results/blueprint_v2_final_sweep.md),
    [`blueprint_v2_final_d.yaml`](../../benchmarks/ablations/blueprint_v2_final_d.yaml)).

```bash
uv sync --extra notebooks --extra baselines --extra umap
uv run jupyter lab notebooks/pragmatiq/
```

**TODO (hosting):** the download URLs in `tfm.runtime.lab` are placeholders
until the archives are hosted. Until then, point the notebooks at local
copies:

```bash
export TFM_PRAGMATIQ_ROOT=/path/to/synth_100k_gen2_m31_seed0
export TFM_V2_CHECKPOINT=/path/to/blueprint_v2_lab_tuned
```

`scripts/package_lab_assets.py` builds the checkpoint directory from a run,
and the archives to host (notebook 05). A checkpoint packaged by SDK 2.0.0a1,
such as the GTC Berlin lab's, loads unchanged: `fetch_lab_assets` migrates
its `config.yaml` (`tfm migrate-config`).

### Smoke mode

With `TFM_NOTEBOOK_SMOKE=1`, nothing is downloaded. Each notebook generates a
120-user pragmatiq-shaped book (`tfm.data.synthetic.lab_book`), uses an
untrained model of the same architecture, and runs on CPU. CI runs the series
this way (`make test-notebooks`). Every cell executes, but the numbers and
plots are meaningless. The first cell of each notebook prints which mode it
is in.

## Series

| Notebook | Slot | Teaches | Lab GPU time |
|---|---|---|---|
| [`00_blueprint_v2_tour.ipynb`](00_blueprint_v2_tour.ipynb) | All | v1 against v2 slot by slot, the book and five tasks, key-value tokens, the network, what pretraining taught, and the series' result: LoRA against raw features on five tasks | about 10 min |
| [`01_data_and_labels.ipynb`](01_data_and_labels.ipynb) | 0 Data | The multi-source book, temporal cutoffs and label windows, the spliced identity-theft task, five downstream tasks and their customer-disjoint splits | about 4 min |
| [`02_event_representation.ipynb`](02_event_representation.ipynb) | 1 Event representation | Key-value tokens, percentile buckets, hashed identifiers, calendar and TimeRoPE, the event-boundary rule | about 3 min |
| [`03a_pragma_encoder.ipynb`](03a_pragma_encoder.ipynb) | 2 Architecture, 3 Objective, 4 Readout | PRAGMA-like: a bidirectional history encoder, masked token / event / key modelling, the `[USR]` readout, against Blueprint v2 | under 1 min, CPU, generated book |
| [`03b_blueprint_v2_decoder.ipynb`](03b_blueprint_v2_decoder.ipynb) | 2 Architecture, 3 Objective | Blueprint v2: profile and event encoders under a causal decoder, attention masks, per-field loss against its floors, a short pretraining run, how the checkpoint was configured | about 5 min |
| [`04_readout_and_adaptation.ipynb`](04_readout_and_adaptation.ipynb) | 4 Readout and adaptation | The embedding and the embedding space, LoRA fine-tuning on one task step by step: early stopping, what the adapters train, rank and head | about 5 min |
| [`05_scaling_up.ipynb`](05_scaling_up.ipynb) | All | The configs, what more pretraining, a larger LoRA rank, and a longer context bought, reproducing and packaging the checkpoint, multi-GPU, swapping the dataset | seconds, no GPU |
| [`06_swap_one_slot.ipynb`](06_swap_one_slot.ipynb) | Any one | The [slot ablation](../../benchmarks/ablations/building_blocks): change one slot, score every variant on the same split | about a minute, CPU, generated book |

Times are one GB10 with the book on local disk and the default 4,000-user
sample. Each notebook runs on its own and prepares its own sample, and every
one samples the same 4,000 users with the same seed. 00 is the overview;
01→06 is the teaching order, and 03a and 03b can be read in either order.
03a and 06 always run on a generated book on CPU, in either mode. 01 to 04
end with exercises.

The notebooks run no hyperparameter search. The forecast and
transaction-fraud tasks (`LAB_TASKS`), the sample size (`LAB_USERS`, or
`n_users=` in `fetch_lab_assets`), the checkpoint's config (`LAB_CONFIG`),
and the one XGBoost setting every baseline shares (`LAB_XGBOOST_PARAMS`) live
in `tfm.runtime.lab`; `tfm.runtime.evaluate.fraud_task` builds `fraud_user`.
A larger sample or another checkpoint (`TFM_V2_CHECKPOINT`) changes no
notebook code.

## Related series

- [`blueprint_v1`](../blueprint_v1): the flat decoder on TabFormer, following
  Blueprint v1's five notebooks.

## Rules

- Reusable logic belongs in `src/tfm/`, not in these notebooks.
- Clear cell outputs before committing.
- A notebook's numbers come from one 4,000-user sample and one customer
  split. The recorded results are in
  [`blueprint_v2_final_sweep.md`](../../benchmarks/results/blueprint_v2_final_sweep.md)
  (the checkpoint, three seeds) and
  [`blueprint_v2_hybrid_gb10_tuning.md`](../../benchmarks/results/blueprint_v2_hybrid_gb10_tuning.md)
  (the search behind the recipe's config).
