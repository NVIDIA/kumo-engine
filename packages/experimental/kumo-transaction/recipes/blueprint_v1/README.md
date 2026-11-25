# Blueprint v1: TabFormer, flat tokens, Llama decoder, XGBoost

**Support tier:** `experimental`

## Goal

Rebuild the [NVIDIA Transaction Foundation Model Blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)
(v1) on this kit, as the first example for someone new to transaction
foundation models. Pretrain a Llama decoder on the public TabFormer dataset by
next-event prediction, embed each transaction together with the customer's
recent history, and test whether those embeddings improve an XGBoost fraud
classifier.

It follows v1's flow, field tokenization, and model, and fixes v1's defects
(Blueprint v1 guide). The two
that matter most:

- **v1 trained without shifting its labels.** Its data module set
  `labels = input_ids`, and the training loop compared each position's logits
  with the token at that same position, so the model learned to copy its
  input. Here `NextEvent` compares position *i* with the token at *i + 1*.
  Notebook 03 shows both losses side by side.
- **v1 embedded each transaction on its own.** Here each labelled transaction
  is embedded with that customer's last 128 transactions up to and including
  it. The v1-style single-transaction embedding is kept as a control, so the
  value of the history is measured rather than assumed.

Start with the notebooks,
[`notebooks/blueprint_v1/`](../../notebooks/blueprint_v1). They
follow v1's five notebooks and run on CPU, or on 24 TabFormer users when
`TFM_TABFORMER_ROOT` is set. This README covers the full-size runs. This recipe
does not download TabFormer or any weights.

| Config | What it runs | Hardware |
|---|---|---|
| `config.smoke.yaml` | Generated book, tiny eager decoder, logistic arms; what CI runs | CPU |
| `config.small.yaml` | 24 TabFormer users, small Llama; checks the pipeline on real data | CPU or GPU |
| `config.yaml` | v1's Llama block (25.9M parameters here) and schedule on all TabFormer users; XGBoost on the GPU | one GPU with 128 GB memory (DGX Spark) |
| `config.multi_gpu.yaml` | `config.yaml` sharded with `fsdp2` | several GPUs, `torchrun` |

### The GPU path

Each heavy stage has a GPU implementation and a CPU equivalent. `config.yaml`
pretrains and fits XGBoost on the GPU; cuDF hashing is opt-in because RAPIDS
is installed outside `uv.lock` and its hashes are not bit-identical to the CPU
tokenizer's, so it changes the vocabulary.

| Stage | GPU | CPU equivalent |
|---|---|---|
| Merchant hashing | `features.tokenizer: rapids_field_tokenizer` (cuDF `hash_values`, one call per batch) | `transaction_field_tokenizer` |
| Pretraining | HuggingFace Llama, bf16, `sdpa`; `fsdp2` for several GPUs | the eager decoder |
| Fraud classifier | `evaluation.tabular_baselines.device: cuda` (XGBoost `device="cuda"`) | `device: cpu` |

v1 also pretrained through NeMo AutoModel. This kit trains the same HuggingFace
Llama with its own loop and `fsdp2`, and has no NeMo dependency.

## Architecture

- **Data:** `tabformer` adapter, `split_protocol: precomputed` (v1's date split).
- **Tokens:** `transaction_field_tokenizer` with v1's field layout minus `CUST`:
  `AMT MERCH CAT MCC HOUR DOW MONTH CARD CHIP ZIP3 STATE`, then `<sep>`. That is
  12 tokens per event, so a 128-event history is 12 × 128 + 1 = 1,537 tokens.
  `CUST` is dropped because it is the same token at every position of a user's
  history: free to predict, and it only identifies the user.
  `features.fields` selects the layout. Merchants are hashed on CPU; the
  optional `rapids_field_tokenizer` hashes on GPU (install RAPIDS separately).
- **Model:** `TrxFM(Flat(Decoder(...)), position=RoPE(), objective=NextEvent(), readout=LastTokenPool())`.
  The history model is HuggingFace Llama (`model.backbone: huggingface`),
  which applies rotary positions inside attention, so the composition declares
  `position: rope`. `composition.attention` (`eager` or `sdpa`) sets Llama's
  attention implementation.
- **Pretraining:** one row per window of a user's history
  (`data.pretrain_windows.events`), cut per user only, before the train cutoff.
- **Probe:** `evaluation.history_window`. One row per labelled
  transaction: the user's last `data.sequence_length` events up to and
  including it, pooled at the last token. Beside it, a `single_event` row holds
  the transaction alone, as v1 did. Training and evaluation users are disjoint.
- **Baselines:** `evaluation.tabular_baselines`. Every XGBoost arm gets the same
  Optuna search on an inner split grouped by user; `exclude_keys: [user]` keeps
  the user id out of the raw features.

## Task and metrics

Transaction-level fraud: is this transaction fraudulent, given the history
before it? The label is TabFormer's own `Is Fraud?` flag, available at the
event time (`label_delay: 0h`, as in v1). The training side keeps every fraud
and samples `history_window.train_negatives_per_positive` negatives per fraud;
the evaluation side scores every test-period transaction of the evaluation
users, at the natural fraud rate (`history_window.max_rows: null`). Every
XGBoost arm uses one fixed setting (`tabular_baselines.xgboost_params`), so
arms differ only in their features.

The arms in `metrics.json` that compare with v1's notebook 05:

| Metric key | Features |
|---|---|
| `history_window_xgboost_raw` | The transaction's own fields, no history (v1's raw arm) |
| `history_window_xgboost_raw_embedding` | Raw plus the history embedding |
| `history_window_xgboost_raw_embedding:single_event` | Raw plus the single-transaction embedding (v1's combined arm) |
| `history_window_tabular_xgboost` | Hand-built aggregates of the same history: the stronger reference |

The headline is the PR-AUC of `history_window_xgboost_raw_embedding` over
`history_window_xgboost_raw`. Three seeds on every test row are recorded in
[`blueprint_v1_seeds.md`](../../benchmarks/results/blueprint_v1_seeds.md): the
single-transaction embedding lifts raw-feature XGBoost at every seed, the
history embedding does not consistently; see
[the roadmap](../../docs/reference/roadmap.md), Milestone 1.

## Data

[IBM TabFormer](https://github.com/IBM/TabFormer) is a public synthetic book of
about 24.4M card transactions by 2,000 users. Download `transactions.tgz` from
the IBM TabFormer repository's data link, then build the splits. This takes
about 30 seconds and 10 GB of memory, on CPU:

```bash
mkdir -p ~/data/TabFormer && mv transactions.tgz ~/data/TabFormer/
uv run tfm prepare-tabformer --root ~/data/TabFormer
export TFM_TABFORMER_ROOT=~/data/TabFormer
```

`prepare-tabformer` extracts `raw/card_transaction.v1.csv` and writes
`temporal_split/{train,val,test,val_eval,test_eval}.parquet` with v1's
notebook-01 rule: sort the book by date and cut where the cumulative row count
reaches 80% and 90%. Do **not** commit the data or a symlink to it. You can
set `data.path` in the config instead of the environment variable.

Sequence cutoffs are the last event time in each partition, not the YAML
`train_end`, because the split is by row count (temporal split rules). A user's history can
span all three partitions. When `max_entities` is below the number of users,
the adapter prefers users with fraud so the small configs have positives.

## Commands

CPU-safe smoke (synthetic, eager decoder, default Kumo Engine loader):

```bash
uv run tfm smoke --recipe blueprint_v1
```

24 TabFormer users (`config.small.yaml`):

```bash
uv sync --extra native --extra huggingface --extra baselines
export TFM_TABFORMER_ROOT=~/data/TabFormer
uv run tfm train    --recipe blueprint_v1 --config small
uv run tfm infer    --recipe blueprint_v1 --config small
uv run tfm evaluate --recipe blueprint_v1 --config small
uv run tfm report   --recipe blueprint_v1 --config small
```

Each command runs the stages before it, so `report` alone trains, embeds,
evaluates, and reports. Pass `--run-dir` to choose the output directory and
`--reuse-checkpoint` to re-evaluate a trained run.

Full size on one GPU (`config.yaml`):

```bash
uv run tfm report --recipe blueprint_v1 --device cuda \
  --run-dir outputs/blueprint_v1/gb10
```

The seed moves pretraining, the probe's user split, and the negative sample.
To report a range rather than one seed:

```bash
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v1/config.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v1/gb10_seeds \
  --summary benchmarks/results/tabformer_hf_decoder_gb10.json --device cuda
```

The Llama backbone needs Python headers for Triton; see
[hardware and runtimes](../../docs/reference/hardware-and-runtimes.md).

Several GPUs: `config.yaml` trains with `training.backend: ddp`, which is the
eager loop in one process. Launched under `torchrun`, every GPU holds a full
replica of the model and trains on its own minibatches; gradients are averaged
in backward. `training.batch_size` is per GPU, so the global batch grows with
the number of GPUs.

Without a token cache, each rank loads and tokenizes all of TabFormer itself:
about 9 minutes and 48 GB of host memory per rank before the first step. With
one, training reads its windows from disk and loads no event at all, so run it
as three jobs with `config.cached.yaml` (the same settings plus
`data.token_cache`):

```bash
# Once: tokenize the train split (about 75 s with 8 workers, 1.4 GB on disk).
uv run tfm cache-tokens --recipe blueprint_v1 --config cached --workers 8

# Train on N GPUs; each rank starts in seconds and holds about 3 GB of host memory.
uv run torchrun --nproc-per-node 2 -m tfm.cli.main train \
  --recipe blueprint_v1 --config cached --device cuda \
  --run-dir outputs/blueprint_v1/cached

# Embed, evaluate, and report on one process; the probes still load events.
uv run tfm report --recipe blueprint_v1 --config cached --reuse-checkpoint \
  --device cuda --run-dir outputs/blueprint_v1/cached
```

The full-size configs set `training.log_every: 50`: every 50 steps the first
process prints step, epoch, loss, learning rate, gradient norm, throughput, and
time left. Each job also prints its stages with the time since it started.

The cache holds the sample's train split, checked by its users, the TabFormer
files, `label_delay`, and the tokenizer's fields; a run refuses a cache built
for anything else. `tfm report` without `--reuse-checkpoint` also works on N
GPUs, but then every rank loads events first.

`config.multi_gpu.yaml` shards parameters across GPUs with `fsdp2` instead. It
requires CUDA, `model.backbone: huggingface`, and `training.precision: bf16`,
and raises rather than training eager. Each rank prepares data independently,
so host memory scales with the world size; rank 0 alone embeds, evaluates, and
reports.

```bash
uv run torchrun --nproc-per-node 8 -m tfm.cli.main report \
  --recipe blueprint_v1 --config multi_gpu --device cuda \
  --run-dir outputs/blueprint_v1/multi_gpu
```

### Environment note

The `huggingface`, `baselines`, and `umap` extras are locked in `uv.lock`.
`uv sync` removes every extra you don't list, so name them all in one command
(`uv sync --extra native --extra huggingface --extra baselines --extra umap`). RAPIDS is
installed with `uv pip install` and is not in `uv.lock`, so a plain `uv run`
re-syncs the environment and removes it; use `uv run --no-sync` once it is
installed. cuDF is CUDA-version specific and needs `numpy<2.5`; on CUDA 13:
`uv pip install --extra-index-url=https://pypi.nvidia.com cudf-cu13 "numpy<2.5"`.

## Artifacts

Written under the run directory (or the smoke temp dir):

- `resolved_config.yaml`
- `data_validation_report.json`
- `split_manifest.json`
- `tokenizer_manifest.json`
- `checkpoint_manifest.json` and `hf_checkpoint/`
- `train_log.json`: every training step's loss, learning rate, gradient norm,
  and token count, for plotting the loss curve
- `embeddings.npz` and `embeddings_manifest.json`: one embedding per customer
  history at each cutoff, with its label and its customer, row, and event ids
  (`<side>_entity_ids` and so on, row for row with `<side>`)
- `metrics.json` / `metrics.md`
- `model_card.md`
- `environment.json`

## Limitations

- Experimental. The metrics of `config.smoke.yaml` and `config.small.yaml` exercise
  the pipeline; they are too small to be evidence of anything.
- Not bit-compatible with v1: the vocabulary, merchant hashing, labels, and
  probe all differ, and v1 checkpoints do not load. The v1-replay tooling is on
  the `archive/v1-replay` branch.
- The numbers are not comparable to v1's published notebook numbers: v1 probed
  single transactions on a fraud-stratified sample with per-arm hand tuning.
- TabFormer is synthetic and has one channel and one task. Results on it are
  not combined with results on any other dataset
  ([datasets](../../docs/reference/datasets.md)).
- CPU merchant hashing is not bit-identical to RAPIDS `hash_values`.

## Loader selection

Kumo Engine's experimental PyTorch/DiskGraph loader is the default
(`data.loader: kumo_pytorch`). Install `--extra native` for sequences and
`--extra kumo` for graph models. The [worked loader guide](../../examples/kumo-engine/README.md#canonical-recipes)
provides setup, commands and temporal/schema limits. Set `data.loader: python`
in a copied configuration to use the reference loader. Both retain preparation
costs; see [memory and materialization tradeoffs](../../docs/migration/native-loader-default.md#materialization-memory-and-preparation).
The current runners still ingest bounded object corpora, so this default does
not establish out-of-core training or a universal speed/memory advantage.
