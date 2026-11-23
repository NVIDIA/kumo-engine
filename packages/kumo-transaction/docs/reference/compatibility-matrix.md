# Compatibility matrix

What is supported, at which tier, and what tests it. A capability is
`reference` only with a documented recipe, tests, synthetic smoke, and
recorded artifacts. This table does not claim production quality. Recorded
results are indexed in [`benchmarks/results`](../../benchmarks/results/README.md);
milestone status is in [roadmap.md](roadmap.md).

## Recipes

| Recipe | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| `blueprint_v1` (+ `notebooks/blueprint_v1`) | `experimental` | recipe smoke, notebooks under nbmake | CPU / CUDA | v1's flow on the SDK (ADR 0017): 11-field tokens, Llama decoder, per-transaction fraud with history |
| `pragma_like` | `experimental` | unit, recipe smoke | CPU / CUDA | PRAGMA encoder (ADR 0007); `config.yaml` matched to `blueprint_v2`'s, unit-tested |
| `blueprint_v2` (+ `notebooks/pragmatiq`) | `experimental` | unit, contract, recipe smoke, notebooks in smoke mode | CPU / CUDA | PRAGMA encoders under a causal decoder (ADR 0012); `eager` or `ddp` |
| Slot ablations (`benchmarks/ablations`) | `experimental` | recipe smoke (shared probe split asserted) | CPU | One slot swapped per variant, median [min, max] over `--seeds` |
| 2.0.0a1 config migration (`tfm migrate-config`) | `reference` | unit (every moved key, every refusal, the lab's checkpoint config) | CPU | Rewrites 2.0.0a1 keys; refuses settings 2.0.0a2 cannot run; comments are not kept |

## Data and tokens

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Synthetic data adapter | `reference` | unit, contract, smoke | CPU | Default CI path |
| TabFormer adapter + `tfm prepare-tabformer` | `experimental` | unit, recipe smoke | CPU | Splits matched v1's `temporal_split/` row for row (2026-09-26); not fetched by CI |
| pragmatiq parquet adapter | `reference` | unit | CPU | Four-file contract; the 12-field path drops non-transaction sources |
| Spliced identity-theft fraud | `experimental` | contract | CPU | `data.fraud_labels: spliced_identity_theft` (ADR 0015) |
| Profile milestones as of cutoff | `reference` | contract | CPU | Milestones after the cutoff are dropped; the rest are timed from the cutoff |
| SequenceTokenizer protocol + registry | `reference` | unit, contract | CPU | Layouts are checked against the granularity |
| `transaction_field_tokenizer` (`flat_fields`) | `reference` | unit, contract | CPU | v1's field layout; `features.fields` selects a subset in order |
| `rapids_field_tokenizer` | `experimental` extra | unit (skip without cudf) | GPU | Same layout; cuDF `hash_values`, one call per batch; hashes differ from the CPU tokenizer's |
| `key_value_time` tokenizer | `experimental` | unit, contract | CPU | Train-only fit; byte-level text or `features.hashed_keys` buckets; not bit-exact vs pragmatiq |
| `flat_key_value` / `event_fields` token schemes | `experimental` | contract | CPU / CUDA | Decouple the token scheme from the structure |
| Windowed pretraining corpus | `experimental` | unit | CPU / CUDA | `data.pretrain_windows.events`; evaluation still pools one embedding per entity |
| Key-value-time token cache (`tfm cache-tokens`) | `experimental` | contract (parity and leakage vs the object path) | CPU, multi-process | pragmatiq books only, spliced fraud only (ADR 0016) |
| Flat-field token cache (`tfm cache-tokens`) | `experimental` | contract (windows, minibatches, ids, labels, and raw features vs the object path; a pretraining-only setup that loads no events) | CPU build, multi-process | TabFormer's precomputed train split with `transaction_field_tokenizer` and `data.pretrain_windows`; `train` reads only the cache, `evaluate` and `report` still load events; all 58,000 TabFormer windows equal the object path's; no recorded training result (ADR 0016) |

## Model slots

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Flat causal decoder + next-event + last-token pool | `reference` | unit, contract, recipe smoke | CPU / CUDA | The core `TrxFM` surface |
| Configurable attention kernel | `reference` | unit, contract | CPU / CUDA | `EagerAttention` is the correctness baseline |
| Fused SDPA attention | `experimental` | contract | CPU / CUDA | `composition.attention: sdpa` |
| HuggingFace Llama backbone | `experimental` extra | unit, contract | CPU / CUDA | `model.backbone: huggingface`; requires `position: rope` |
| GRU backbone | `experimental` | unit | CPU / CUDA | `model.backbone: gru` on `flat`; not packable |
| Backbone registry (`tfm.api.register_backbone`) | `experimental` | contract | CPU / CUDA | Adds a `model.backbone` name for `flat`; unknown names are refused with the registered list |
| Flat bidirectional encoder + masked fields + last-event readout (`flat_encoder`) | `experimental` | contract (bidirectional, padding, masking, refusals), recipe smoke | CPU / CUDA | `tfm.experimental` (ADR 0019); `transaction_field_tokenizer` and `position: rope` only; no surprise arm |
| Profile + event + history encoders (`profile_history`) | `experimental` | unit, recipe smoke | CPU / CUDA | `tfm.experimental` (ADR 0007); event and profile encoders shared with `profile_event_decoder`; pre-sharing checkpoints still load |
| Profile + event encoders + causal decoder (`profile_event_decoder`) | `experimental` | unit, contract (causal invariance, anti-copy, padding) | CPU / CUDA | `tfm.experimental` (ADR 0012) |
| `NoPE` on `profile_event_decoder` | `experimental` | contract | CPU / CUDA | The profile encoder keeps TimeRoPE |
| Varlen event encoding | `experimental` | unit | CPU / CUDA | `granularity.varlen`; skips all-padding event rows, equals the padded path |
| Sequence packing | `experimental` | unit | CPU / CUDA | `training.packing`; eager backbone only |
| Token / event / key MLM + label smoothing | `experimental` | unit | CPU / CUDA | PRAGMA's masking rates |
| Auxiliary objectives | `experimental` | unit | CPU / CUDA | `objective.auxiliary` on `next_event_fields` |
| SupervisedHead objective (`Binary`) | `experimental` | unit, contract | CPU / CUDA | `fraud_task.supervised_head`; the frozen probe stays the baseline |
| Mean readout | `experimental` | unit | CPU / CUDA | `composition.readout: mean` |

## Training

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Backend registry + capability checks | `reference` | unit, contract | CPU | Unsupported combinations are refused before data preparation (ADR 0011) |
| Backend contract suite | `reference` | contract | CPU; GPU backends on a GPU host | Eager parity, anti-copy, masking, HF round trip, determinism |
| `ddp` | `experimental` | contract (2 Gloo ranks; 2 and 4 NCCL ranks) | CPU / CUDA, `torchrun` | `batch_size` per rank; no recorded multi-GPU training result |
| `fsdp2` | `experimental` extra | unit, contract on GPU, 2- and 8-GPU runs | CUDA, `torchrun` | HuggingFace backbone and bf16 only; writes a consolidated HF checkpoint |
| Muon + gradient clipping | `experimental` | contract | CPU / CUDA | `training.optimizer.name: muon`; eager and `ddp` only |
| Progress log (`training.log_every`, `train_log.json`) | `reference` | unit, recipe smoke | CPU / CUDA, `torchrun` | First rank prints its own minibatch loss; every run writes the per-step curve; `fsdp2` reports no gradient norm |
| LoRA and full fine-tuning | `experimental` | unit | CPU / CUDA | `fraud_task.finetune`; reported beside the frozen probe |
| Joint-fusion head (DCNv2 + PLR) | `experimental` | unit | CPU / CUDA | `fraud_task.finetune.head: joint_fusion` |
| PRAGMA-S/M/L size ladder | `experimental` | unit | GPU | Dimensions follow dynamiq-ai/pragmatiq; no recorded result |

## Evaluation

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| XGBoost arms `xgboost_raw` / `xgboost_raw_embedding` / `tabular_xgboost` | `experimental` extra | contract | CPU / CUDA | The headline comparison; `tabular_baselines.device: cuda` trains on the GPU |
| Logistic arms `raw` / `embedding` / `combined` | `reference` | unit, smoke | CPU | Logistic probes the CPU smoke configs score |
| History-window fraud probe | `experimental` | contract | CPU / CUDA | One row per labelled event with its own history, plus a single-event control |
| Next-event surprise (`tfm.evaluation.surprise`) | `experimental` | contract (prefix-only, row independence) | CPU / CUDA | `evaluation.surprise` on `next_event_fields` models |
| 2-D and 3-D projection (`project`, `project_2d`) | `reference` | unit | CPU | PCA, or UMAP via the `umap` extra; inspection only |
| Lab notebook assets (`tfm.runtime.lab`) | `experimental` | unit, nbmake in smoke mode | CUDA; CPU smoke | The final-sweep checkpoint (`LAB_CONFIG`); a 2.0.0a1 checkpoint's config is migrated on load; download URLs are placeholders until hosted |

## Removed

| Capability | Removed in | Replacement |
|---|---|---|
| `training.backend: nemo` | 2.0.0a2 | `fsdp2`, which trains the same HuggingFace Llama |
| Recipes `tabformer_hf_decoder`, `profile_history_kvt`, `blueprint_v2_hybrid`, `decoder_fraud_baseline`, `building_block_comparison` | 2.0.0a2 | `blueprint_v1`, `pragma_like`, `blueprint_v2`, `blueprint_v1` smoke, `benchmarks/ablations`; see [migration](../migration/2.0.0a1-to-2.0.0a2.md) |
| 2.0.0a1 config keys (`data.pretrain_window_events`, `training.optimizer.muon_learning_rate`, ...) | 2.0.0a2 | Their sub-sections; `tfm migrate-config` rewrites a 2.0.0a1 config ([migration](../migration/2.0.0a1-to-2.0.0a2.md)) |

Planned, not started: hierarchical, Mamba, ranking, and explainability
components.
