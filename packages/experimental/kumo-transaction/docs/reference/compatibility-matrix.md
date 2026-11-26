# Compatibility matrix

What is supported, at which tier, and what tests it. A capability is
`reference` only with a documented recipe, tests, synthetic smoke, and
recorded artifacts. This table does not claim production quality. Recorded
results are indexed in [`benchmarks/results`](../../benchmarks/results/README.md);
milestone status is in [roadmap.md](roadmap.md).

## Recipes

| Recipe | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| `blueprint_v1` (+ `notebooks/blueprint_v1`) | `experimental` | recipe smoke, notebooks under nbmake | CPU / CUDA | v1's flow on the SDK (Blueprint v1 guide): 11-field tokens, Llama decoder, per-transaction fraud with history |
| `pragma_like` | `experimental` | unit, recipe smoke | CPU / CUDA | PRAGMA encoder (experimental composition guide); `config.yaml` matched to `blueprint_v2`'s, unit-tested |
| `blueprint_v2` (+ `notebooks/pragmatiq`) | `experimental` | unit, contract, recipe smoke, notebooks in smoke mode | CPU / CUDA | PRAGMA encoders under a causal decoder (Blueprint v2 guide); `eager` or `ddp` |
| `gnn_encoder` | `experimental` | unit, contract, recipe smoke | CPU checked; CUDA short check | Direct Engine GNN; binary supervision or future-subgraph pretraining; generated data; logistic probes |
| `gnn_sequence` | `experimental` | unit, contract, recipe smoke | CPU and CUDA smoke | Selected frozen Engine GNN conditions Blueprint v2 through profile tokens; cutoff-level binary supervision |
| `relational_inputs` | `experimental` | contract, CPU recipe smoke | CPU | One bounded fixed-FK declaration for profile/event sequence, GNN and hybrid; explicit entity labels; native candidate retrieval by default; see [limits](../../recipes/relational_inputs/README.md#limitations) |
| Slot ablations (`benchmarks/ablations`) | `experimental` | recipe smoke (shared probe split asserted) | CPU | One slot swapped per variant, median [min, max] over `--seeds` |
| 2.0.0a1 config migration (`tfm migrate-config`) | `reference` | unit (every moved key, every refusal, the lab's checkpoint config) | CPU | Rewrites 2.0.0a1 keys; refuses settings 2.0.0a2 cannot run; comments are not kept |

## Data and tokens

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Synthetic data adapter | `reference` | unit, contract, smoke | CPU | Default CI path |
| TabFormer adapter + `tfm prepare-tabformer` | `experimental` | unit, recipe smoke | CPU | Splits matched v1's `temporal_split/` row for row (2026-09-26); not fetched by CI |
| pragmatiq parquet adapter | `reference` | unit | CPU | Four-file contract; the 12-field path drops non-transaction sources |
| Spliced identity-theft fraud | `experimental` | contract | CPU | `data.fraud_labels: spliced_identity_theft` (dataset strategy) |
| Profile milestones as of cutoff | `reference` | contract | CPU | Milestones after the cutoff are dropped; the rest are timed from the cutoff |
| SequenceTokenizer protocol + registry | `reference` | unit, contract | CPU | Layouts are checked against the granularity |
| `transaction_field_tokenizer` (`flat_fields`) | `reference` | unit, contract | CPU | v1's field layout; `features.fields` selects a subset in order |
| `rapids_field_tokenizer` | `experimental` extra | unit (skip without cudf) | GPU | Same layout; cuDF `hash_values`, one call per batch; hashes differ from the CPU tokenizer's |
| `key_value_time` tokenizer | `experimental` | unit, contract | CPU | Train-only fit; byte-level text or `features.hashed_keys` buckets; not bit-exact vs pragmatiq |
| `flat_key_value` / `event_fields` token schemes | `experimental` | contract | CPU / CUDA | Decouple the token scheme from the structure |
| Windowed pretraining corpus | `experimental` | unit | CPU / CUDA | `data.pretrain_windows.events`; evaluation still pools one embedding per entity |
| Key-value-time token cache (`tfm cache-tokens`, format `columnar`) | `experimental` | contract (parity and leakage vs the object path) | CPU, multi-process | A columnar source whose event ids take pragmatiq's `user:microseconds:index` form |
| Batches built on the GPU from a token cache (`DeviceBatches`, the default `data.batch_builder`) | `experimental` | contract (equal to the host path, alone and through `setup_run`; CUDA cases on a GPU host only) | CUDA | Default on CUDA; `data.token_cache_on_device: false` keeps the cache on the host. Refused when the cache arrays exceed half of free GPU memory. No recorded speedup |
| Flat-field token cache (`tfm cache-tokens`, format `flat_fields`) | `experimental` | contract (windows, minibatches, ids, labels, and raw features vs the object path; a pretraining-only setup that loads no events) | CPU build, multi-process | TabFormer's precomputed train split with `transaction_field_tokenizer` and `data.pretrain_windows`; `train` reads only the cache, `evaluate` and `report` still load events; all 58,000 TabFormer windows equal the object path's; no recorded training result |

## Model slots

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Flat causal decoder + next-event + last-token pool | `reference` | unit, contract, recipe smoke | CPU / CUDA | The core `TrxFM` surface |
| Configurable attention kernel | `reference` | unit, contract | CPU / CUDA | `EagerAttention` is the correctness baseline |
| Fused SDPA attention | `experimental` | contract | CPU / CUDA | `composition.attention: sdpa` |
| HuggingFace Llama backbone | `experimental` extra | unit, contract | CPU / CUDA | `model.backbone: huggingface`; requires `position: rope` |
| GRU backbone | `experimental` | unit | CPU / CUDA | `model.backbone: gru` on `flat`; not packable |
| Backbone registry (`tfm.api.register_backbone`) | `experimental` | contract | CPU / CUDA | Adds a `model.backbone` name for `flat`; unknown names are refused with the registered list |
| Flat bidirectional encoder + masked fields + last-event readout (`flat_encoder`) | `experimental` | contract (bidirectional, padding, masking, refusals), recipe smoke | CPU / CUDA | `tfm.experimental`; a tokenizer implementing `FieldVocabularyTokenizer` and `position: rope` only; no surprise arm |
| Profile + event + history encoders (`profile_history`) | `experimental` | unit, recipe smoke | CPU / CUDA | `tfm.experimental` (experimental composition guide); event and profile encoders shared with `profile_event_decoder`; pre-sharing checkpoints still load |
| Profile + event encoders + causal decoder (`profile_event_decoder`) | `experimental` | unit, contract (causal invariance, anti-copy, padding) | CPU / CUDA | `tfm.experimental` (Blueprint v2 guide) |
| `NoPE` on `profile_event_decoder` | `experimental` | contract | CPU / CUDA | The profile encoder keeps TimeRoPE |
| Varlen event encoding | `experimental` | unit | CPU / CUDA | `granularity.varlen`; skips all-padding event rows, equals the padded path |
| Sequence packing | `experimental` | unit | CPU / CUDA | `training.packing`; eager backbone only |
| Token / event / key MLM + label smoothing | `experimental` | unit | CPU / CUDA | PRAGMA's masking rates |
| Auxiliary objectives | `experimental` | unit | CPU / CUDA | `objective.auxiliary` on `next_event_fields` |
| SupervisedHead objective (`Binary`) | `experimental` | unit, contract | CPU / CUDA | `fraud_task.supervised_head`; the frozen probe stays the baseline |
| `GNNEncoder` / `GNNSequence` | `experimental` | contract (cutoffs, repeated roots, alignment, frozen state, checkpoint round trip) | CPU checked | `tfm.experimental`; `kumo` extra; shared same-cutoff training, disjoint validation/prediction; eager only |
| `FutureSubgraph` | `experimental` | unit, contract (oracle, masks, delay, observed-through coverage, gradients) | CPU checked | Separate future targets; window-end membership; train-only signed-log normalization |
| Engine/shared/disjoint cost controls | `experimental` | CPU harness test; short L4 check | CPU / CUDA | [Recorded L4 check](../../benchmarks/results/graph_backends_l4_cost.md); historical untrimmed run; no native-loader/full-quality claim |
| Kumo graph tensor adapter | `experimental` | contract (native output and invalid-binding fixtures) | CPU native integration | `from_kumo_graph`; bounded FK schema, whole-second clocks; future aggregates remain Python |
| Source-loader providers | `experimental` | registry/import-path validation, native two-thread engine and four CLI smokes | CPU | Engine/AnchorTable/model-tensor capability required; provider identity and cleanup checked; [extension guide](extending.md#source-loader-providers) |
| Kumo PyTorch recipe loaders | `experimental` | real source-table/native-tensor contracts; continuous iterator and recipe smokes | CPU | Default CLI source-native sequence, graph and hybrid streams; train-fitted native encoders, bounded input retention and explicit Python reference option; [guide](../../examples/kumo-engine/README.md#canonical-recipes) |
| Mean readout | `experimental` | unit | CPU / CUDA | `composition.readout: mean` |

## Training

| Capability | Tier | Tests | Runtime | Notes |
|---|---|---|---|---|
| Backend registry + capability checks | `reference` | unit, contract | CPU | Unsupported combinations are refused before data preparation (backend contracts) |
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
| Key-value-time token cache formats 1–3 | 2.0.0a3 | Format 4: rebuild with `tfm cache-tokens` ([migration](../migration/2.0.0a2-to-2.0.0a3.md)) |
| torch < 2.9 | 2.0.0a3 | torch ≥ 2.9, the first release with `torch.optim.Muon` |
| `EventFieldsTokenizer` and `FlatKeyValueTokenizer` as `KeyValueTimeTokenizer` subclasses, and their vocabulary attributes (`field_kind`, `cat_vocab`, `token_to_id`, `key_vocab`, `field_tokens`) and `state_dict`/`from_state` | 2.0.0a3 | `tokenizer.vocabulary` and `tokenizer.pairs`; refit, or use `key_value_time` for saved state ([migration](../migration/2.0.0a2-to-2.0.0a3.md)) |
| `KeyValueTimeStats` | 2.0.0a3 | `tfm.features.tokenizers.vocabulary.KeyValueStats` |
| `KeyValueTimeBatch.hour` / `dow` / `dom`, `TokenCache.hour` / `dow` / `dom`, `CachedKeyValueTimeTokenizer`, `CacheRequest(root=...)` | 2.0.0a3 | `batch.events[name]` / `batch.event_array(name)`, `cache.events[name]`, `CachedTokenizer`, `CacheRequest(source=...)` ([migration](../migration/2.0.0a2-to-2.0.0a3.md)) |

Planned, not started: hierarchical, Mamba, ranking, and explainability
components.


Graph recipes also support direct supervised binary GNN training and supervised
GNN warm-up before the hybrid, with optional FutureSubgraph variants. Both are
experimental and CPU contract/smoke checked. Engine receives linear projections
without an additional input ReLU and validated layer-wise hop counts by default.
SDPA is the sequence default; eager remains a tested correctness control.
The historical L4 cost record predates these forward changes and must not be
used as current trimmed-model throughput.


Graph execution validates CPU batches once, carries validation through device
copies, and caches FK cardinality checks. Contract tests cover invalid inputs,
tracked mutation rejection and Engine output/gradient parity. Graph batch
constructors require CPU tensors and the documented read-only ownership rules.


Public graph batch builders and device transfers support outer inference-mode
loops through local version-tracked no-grad construction. CPU/CUDA contracts
cover context restoration, frozen-encoder parity and mutation rejection.
Already-created inference input tensors require explicit normal copies.


Graph-v4 experiment reuse validates embedded metadata before installing weights
and derives sidecars from it. Binary graph recipes reject missing fraud sources
and unavailable owner coverage. FutureSubgraph helpers enforce target mutation
guards. Matched binary sequence controls share Blueprint v2 encoding and consume
history-only batches; historical FutureSubgraph-mode controls remain available.
Source provenance supports outside-checkout execution and explicit unknown Git
identity. These paths remain experimental, with focused regression and smoke
coverage.

Graph execution views (`.for_execution`) retain CPU validation metadata and
source guards, transfer model inputs/targets, and avoid repeated signature
reconstruction. Full `.to()` remains available. CPU/CUDA contract coverage checks
source/device mutations, inference contexts, serialization, tensor lifetime and
output/loss/gradient/buffer parity. This does not promote the graph support tier.


Graph sampling records `event_key_encoding: json-array-v1` as part of sample
identity. Explicit observation watermarks allow quiet tails while horizon and
availability bounds remain enforced. The support tier remains experimental.

Graph stages include label-free FutureSubgraph pretraining, managed downstream
reuse of a plain encoder state dictionary, and full supervised comparisons.
Public graph-context hooks and registered/importable architecture, history,
readout, head, adapter and tokenizer factories are covered by contract tests.
The shared RecipeConfig schema applies to all families. Distributed graph
training and columnar graph construction are not supported by this path.


Graph recipes honor `data.amount_field` in graph features, future targets and raw
probes, with its identity bound into encoder manifests. Graph history and controls
support explicit `nope` or `time_rope` (graph default). Pretrain-only graph stages
need no sequence tokenizer. Token caches are rejected for graph recipes.
