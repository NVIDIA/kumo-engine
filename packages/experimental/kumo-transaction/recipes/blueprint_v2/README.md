# Blueprint v2: PRAGMA encoders under a causal history decoder

**Support tier:** `experimental`

## Goal

Combine PRAGMA's per-event compression with Blueprint v1's causal next-event
training, and show lift over raw-feature XGBoost on several tasks of the same
users. Each event, and the profile, is encoded into one summary token; a
causal decoder reads the summary tokens in time order and predicts the next
event's field values. The embedding at the cutoff is, by construction, a
function of the profile and the events up to the cutoff only, and the same
decoder can generate future events.

Design and leakage contract:
Blueprint v2 guide.
Milestone status: [the roadmap](../../docs/reference/roadmap.md). The
bidirectional comparison on the same rows is [`pragma_like`](../pragma_like).

| Config | Hidden | Heads | Profile / event / decoder depth | Parameters | Hardware |
|---|---|---|---|---|---|
| `config.smoke.yaml` | 32 | 4 | 1 / 2 / 2 | tiny | CPU; what CI runs |
| `config.yaml` | 256 | 8 | 2 / 4 / 4 | 8.4M + embedding table | one GPU, token cache; about 24 min of training on a GB10 |
| `config.scale.yaml` | 512 | 8 | 3 / 16 / 12 | ~99M + embedding table | one or more GPUs (`ddp`) |

The untuned lab model and v1's flat decoder on the same rows are ablations,
under [`benchmarks/ablations`](../../benchmarks/ablations).

## Architecture

```text
profile tokens (as of cutoff) -> profile encoder (bidirectional) -> z_a
event j key/value tokens      -> event encoder (bidirectional)   -> z_e[j]
[z_a, z_e[0..N-1]] + calendar, TimeRoPE(elapsed time) -> causal decoder -> h
h at the slot after the last event -> embedding
```

- Tokens: `key_value_time`, fit on the train partition. `config.yaml` hashes
  merchant, device, and counterparty into 4,096 buckets per key.
- Granularity: `profile_event_decoder`; depths are `profile_depth`,
  `event_depth`, and `decoder.depth`.
- Position: TimeRoPE over each event's log-seconds before the sequence's last
  event, so attention depends on elapsed time, not on event indices.
- Objective: `next_event_fields` (slot `j` predicts every value of event `j`)
  plus PRAGMA's within-event MLM through `objective.auxiliary:
  [token_event_key_mlm]`. The losses are summed.
- Readout: `last_token`.

Composed through `tfm.experimental.compose_blueprint_v2`, not the core
`TrxFM` surface.

## Task

Self-supervised pretraining, then frozen-embedding probes on fraud and on
every forecast label the book carries (for pragmatiq: `churn_6m`,
`default_12m`, `ltv_positive`, `aml`). With `evaluation.history_window`,
one row per labelled card transaction is also scored with that transaction's
own history. Pretraining targets are event-level; probe labels are
sequence-level at the cutoff and never enter the objective.

Fraud is spliced identity theft (`data.fraud_labels: spliced_identity_theft`,
[dataset strategy](../../docs/reference/datasets.md)):
on the pragmatiq book, the book's own fraud can be identified from a single
event, so it cannot show what history adds. At 10,000 users the splice makes
about 1,390 victims per seed.

## Data

`config.smoke.yaml` generates a tiny pragmatiq-shaped fixture in the run
directory. Every other config reads a local pragmatiq book, which is not part
of this repository. Build a 100,000-user book from a
[dynamiq-ai/pragmatiq](https://github.com/dynamiq-ai/pragmatiq) v1.1.0
checkout (Generator v2, an agent-based simulator, not LLM-driven) with this
kit's world file:

```bash
pragmatiq synth generate --out data/synth_100k_gen2_m31_seed0 \
  --config <this repo>/configs/data/pragmatiq/world.yaml \
  --n-users 100000 --seed 0 --n-workers 16
```

The same seed produces byte-identical output for any worker count; it took
77 s on a DGX Spark and wrote 128.5M events, 2023-01-01 to 2025-08-01. The
world file is pragmatiq's default stretched to 31 months, so the forecast
labels are issued after the train cutoff; in the default 25-month world every
`default_12m` row is before it and is skipped (probe split rules).

Profiles are cut at each sequence's cutoff: lifelong milestones dated after it
are dropped, and the rest are timed from the cutoff. Static attributes carry
no date, which is safe for Generator v2 because it draws them before
simulation; other books need the same guarantee (Blueprint v2 guide).

Holding the book as Python objects costs about 7.7 GB per 1,000 users, so
probes sample `max_entities` users. `config.yaml` pretrains on every
128-event train-side window of all users from a token cache
([token-cache guide](../../docs/reference/config-schema.md)), built once with
`tfm cache-tokens`. A run refuses a cache built from other settings.

## Commands

```bash
uv run tfm smoke --recipe blueprint_v2                     # CPU, generated data

export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run tfm cache-tokens --recipe blueprint_v2 --workers 6 --profile  # optional: project it
uv run tfm cache-tokens --recipe blueprint_v2 --workers 6  # once, about 4 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run tfm report --recipe blueprint_v2 --device cuda
```

`report` trains, embeds, evaluates, and writes a model card; `train`,
`infer`, and `evaluate` stop earlier. On a GB10 the caching allocator
fragments on variable-length batches; expandable segments hold it near the
22 GB the run allocates.

`config.scale.yaml` on N GPUs (`training.backend: ddp`, already set):

```bash
uv run torchrun --nproc-per-node N -m tfm.cli.main report --recipe blueprint_v2 \
  --config scale --run-dir outputs/blueprint_v2/<run-id> --device cuda
```

Every rank holds a full replica and trains on its share of the batches;
`training.batch_size` is per rank. Pass `--run-dir` so every rank agrees on it.
Rank 0 alone embeds, evaluates, and reports.

## Baselines and metrics

Every task reports, in `metrics.json`:

| Metric key | Features |
|---|---|
| `<task>_xgboost_raw` | The raw row: last event plus profile |
| `<task>_xgboost_raw_embedding` | Raw plus the frozen embedding |
| `<task>_xgboost_raw_embedding:surprise` | Raw, the embedding, and the model's next-event surprise on the latest events |
| `<task>_tabular_xgboost` | Hand-built aggregates of the same history: the stronger reference |
| `lora_<task>` | A LoRA fine-tune of the encoder with a linear head |

The headline is `xgboost_raw_embedding` over `xgboost_raw`, per task. Every
XGBoost arm uses one fixed setting (`evaluation.tabular_baselines.xgboost_params`),
with early stopping choosing only the tree count. The settings were chosen by
a staged search on one GB10, recorded in
[`blueprint_v2_hybrid_gb10_tuning.md`](../../benchmarks/results/blueprint_v2_hybrid_gb10_tuning.md)
with the scripts that ran it. This config at three seeds, beside `pragma_like`
on the same rows, is recorded in
[`blueprint_v2_seeds.md`](../../benchmarks/results/blueprint_v2_seeds.md). The
results index is [`benchmarks/results`](../../benchmarks/results/README.md).

## Notebooks

The [pragmatiq series](../../notebooks/pragmatiq) walks this recipe slot by
slot. It reads the book and a pretrained `config.yaml` checkpoint from
`TFM_PRAGMATIQ_ROOT` and `TFM_V2_CHECKPOINT`, or downloads them with
`tfm.runtime.lab`; `TFM_NOTEBOOK_SMOKE=1` runs every cell on CPU on a
generated book, as CI does. `scripts/package_lab_assets.py` packages a run's
checkpoint and tokenizer for the series.

## Artifacts

Under `outputs/blueprint_v2/<run-id>/`: `resolved_config.yaml`,
`data_validation_report.json`, `split_manifest.json`, `tokenizer_manifest.json`,
`checkpoint_manifest.json`, `embeddings_manifest.json`, `probe_manifest.json`,
`metrics.json` / `metrics.md`, `finetune_manifest.json` (with fine-tuning),
`model_card.md`, and `environment.json`.

## Limitations

Experimental. pragmatiq books are synthetic, and results on them are never
combined with results on TabFormer
([datasets](../../docs/reference/datasets.md)). The pragmatiq loader keeps
every sampled user's history in memory (48 GB at 10,000 users), which caps the
probe sample on one machine. Trains under `eager` or `ddp`, not `fsdp2`; no
multi-GPU run is recorded. The objective's head is given the target event's
keys, so the loss is conditioned on which fields the event has. No production,
fraud, credit, or paper-fidelity claims.

## Loader selection

Kumo Engine's experimental PyTorch/DiskGraph loader is the default
(`data.loader: kumo_pytorch`). Install `--extra native` for sequences and
`--extra kumo` for graph models. The [worked loader guide](../../examples/kumo-engine/README.md#canonical-recipes)
provides setup, commands and temporal/schema limits. Set `data.loader: python`
in a copied configuration to use the reference loader. Both retain preparation
costs; see [memory and materialization tradeoffs](../../docs/migration/native-loader-default.md#preparation-and-memory-tradeoffs).
The native CLI streams source-table feature tensors and retains active/prefetched
model batches. Source validation, index preparation, targets and evaluation arrays
still have costs; see the guide's explicit capability and checkpoint limits.
