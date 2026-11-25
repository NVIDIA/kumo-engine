# PRAGMA-like: profile, event, and history encoders

**Support tier:** `experimental`

## Goal

A PRAGMA-style encoder on this kit: the architecture of Revolut's
[PRAGMA](https://arxiv.org/html/2604.08649v1), pretrained by masked modelling
on a pragmatiq book, with its `[USR]` embedding scored on several tasks of the
same users against raw-feature XGBoost.

It borrows the four-file parquet contract and the masking rates from
[dynamiq-ai/pragmatiq](https://github.com/dynamiq-ai/pragmatiq) and does not
vendor that repository. It is not paper-scale and not bit-exact. Its event and
profile encoders are the ones [`blueprint_v2`](../blueprint_v2) puts under a
causal decoder; the two recipes' `config.yaml` files share their rows, split,
tokens, and probes, so they compare the architectures on the same users.

Notebooks: [`notebooks/pragmatiq/`](../../notebooks/pragmatiq) (03a is this
encoder). Milestone status: [the roadmap](../../docs/reference/roadmap.md).

| Config | What it runs | Hardware |
|---|---|---|
| `config.smoke.yaml` | Generated pragmatiq fixture, tiny model; what CI runs | CPU |
| `config.yaml` | Matched to `blueprint_v2/config.yaml`: 10,000 probe users, spliced fraud, token cache | one GPU |
| `config.pragma_s.yaml` / `_m` / `_l` | dynamiq-ai/pragmatiq's `small` / `medium` / `large` presets | 1 GPU / 1 80 GB GPU / 2 80 GB GPUs |
| `config.synth_1m.yaml` | A user sample of the 1M-user book | CPU |

## Paper to kit map

| PRAGMA paper | This recipe |
|---|---|
| §2.2 Key-value-time tokenization | `KeyValueTimeTokenizer` (byte-level text or hashed keys, not BPE) |
| `8·ln(1+Δt/8)` + TimeRoPE | `TimeRoPE` / `time_encode` |
| Profile / event / history encoders | `ProfileHistory` + `ProfileHistoryModel` |
| Calendar hour/dow/dom MLP | `CalendarEmbedding` |
| Token / event / key MLM + `[UNK]` dropout | `TokenEventKeyMLM`, with `objective.mlm.label_smoothing` |
| Padding-free varlen batches | `granularity.varlen` + `attention: sdpa` (no flash-attn) |
| LoRA fine-tuning | `fraud_task.finetune` (`method: lora`), reported as `lora_<task>` beside the frozen probe |
| PRAGMA-S/M/L size ladder | `config.pragma_s/m/l.yaml` |
| Muon optimizer | `training.optimizer.name: muon` (used by `config.yaml`) |

## Architecture

- Tokens: `key_value_time`; each event is `key value` pairs plus its time.
- Granularity: `profile_history`. An event encoder per event, a profile encoder
  over the static profile, and a bidirectional history encoder over both.
- Position: TimeRoPE on log-seconds.
- Objective: token / event / key MLM. There is no next-event head, so the
  model does not generate events and has no next-event surprise arm.
- Readout: the history encoder's `[USR]` slot (`usr_token`).

Composed through [`tfm.experimental`](../../src/tfm/experimental.py)
(experimental composition guide), not the core `TrxFM` surface.

## Data

A pragmatiq book: `events`, `profiles`, `transfers`, and `labels/*` parquet,
mixed `transaction` and `app` sources. With no `data.path` and no
`TFM_PRAGMATIQ_ROOT`, the smoke config generates a tiny fixture; every other
config needs a local book:

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
```

`max_entities` samples users, so the full book is never loaded. `config.yaml`
replaces the book's own fraud labels with spliced identity theft (dataset strategy)
and reads pretraining windows from a token cache (token-cache guide) shared with
`blueprint_v2`. `default_12m` histories are rebuilt at each row's `eval_ts`,
so events after it are excluded. Probe train and evaluation users are disjoint
(probe split rules).

## Commands

```bash
uv run tfm smoke --recipe pragma_like                    # CPU, generated data

export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run tfm cache-tokens --recipe pragma_like --workers 6  # once; shared with blueprint_v2
uv run tfm report --recipe pragma_like --device cuda

uv run tfm train --recipe pragma_like --config pragma_s --device cuda
```

`report` trains, embeds, evaluates, and writes a model card. PRAGMA-L does not
fit on an 80 GB H100 in fp32 at 256 events per user; it trains in bf16 at 4
per rank under `training.backend: ddp`, so two GPUs under `torchrun` give a
global batch of 8.

## Baselines and metrics

Every task reports raw-feature XGBoost (`xgboost_raw`), raw features plus the
embedding (`xgboost_raw_embedding`), and the hand-built aggregate reference
(`tabular_xgboost`), with ROC-AUC, PR-AUC, precision@k, recall@k, and ECE.
With `fraud_task.finetune.enabled`, `lora_<task>` is reported beside the frozen
probe, never instead of it; `finetune_manifest.json` records the adapted
modules and the trainable fraction.

## Artifacts

Under `outputs/pragma_like/<run-id>/`: `resolved_config.yaml`,
`data_validation_report.json`, `split_manifest.json`, `tokenizer_manifest.json`,
`checkpoint_manifest.json`, `embeddings_manifest.json`, `probe_manifest.json`,
`metrics.json` / `metrics.md`, `finetune_manifest.json` (with fine-tuning),
`model_card.md`, and `environment.json`.

## Limitations

Experimental. Not bit-exact with dynamiq-ai/pragmatiq: byte-level or hashed
values instead of BPE, and varlen is row-level (padding inside a kept event is
still computed), not a fused kernel. No PRAGMA + LLM variant. pragmatiq books
are synthetic, so no production, fraud, credit, or paper-fidelity claims.

`config.yaml` at three seeds, beside `blueprint_v2` at a matched size on the
same rows, is recorded in
[`blueprint_v2_seeds.md`](../../benchmarks/results/blueprint_v2_seeds.md).
The S/M/L ladder has no recorded run.

## Loader selection

Kumo Engine's experimental PyTorch/DiskGraph loader is the default
(`data.loader: kumo_pytorch`). Install `--extra native` for sequences and
`--extra kumo` for graph models. The [worked loader guide](../../examples/kumo-engine/README.md#canonical-recipes)
provides setup, commands and temporal/schema limits. Set `data.loader: python`
in a copied configuration to use the reference loader. Both retain preparation
costs; see [memory and materialization tradeoffs](../../docs/migration/native-loader-default.md#materialization-memory-and-preparation).
The current runners still ingest bounded object corpora, so this default does
not establish out-of-core training or a universal speed/memory advantage.
