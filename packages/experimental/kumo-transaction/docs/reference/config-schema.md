# Config schema

See `src/tfm/config.py` (`RecipeConfig`) and
`recipes/blueprint_v1/config.smoke.yaml`. Top-level keys: `seed`, `data`,
`features`, `model` (including `composition`), `objective`, `training`,
`fraud_task`, `evaluation`, `runtime`.

## Every key is live

Every section rejects keys it does not define, so a typo fails at load time
rather than being ignored. Keys that used to exist but never changed a run now
raise an error that names the replacement:

| Removed key | Use instead |
| --- | --- |
| `model.architecture` | `model.composition.granularity.type` |
| `model.causal` | Nothing: causality follows from `granularity.type` |
| `model.pooling` | `model.composition.readout` |
| `model.composition.granularity.decoder.vocab_size` | Nothing: the fitted tokenizer sets it |
| `features.categorical_fields`, `numeric_fields`, `timestamp_features` | `features.tokenizer`: the tokenizer defines its fields |
| `data.entity_key`, `event_time_key`, `event_order_key` | Nothing: adapters map their own columns |
| `objective.name` | `model.composition.objective` |
| `fraud_task.lora` | `fraud_task.finetune` with `method: lora` |
| `evaluation.metrics`, `report_format` | Nothing: every run writes the full metric set |

A key that applies to only one granularity (for example `history_depth`,
which only `profile_history` uses) is an error if it's set to a non-default
value under another granularity.

## Composition: one choice per slot

`model.composition` holds one choice per building-block slot (see
[ADR 0014](../architecture/0014-building-block-taxonomy.md)). The runtime
builds exactly that composition and runs `validate_composition` before
training, so an invalid pairing fails with the rule it breaks.

| Key | Choices | Slot |
| --- | --- | --- |
| `features.tokenizer` | `transaction_field_tokenizer`, `flat_key_value`, `key_value_time`, `event_fields` | Event representation |
| `position` | `nope`, `rope` (HuggingFace backbone only), `time_rope` | Event representation |
| `granularity.type` | `flat`, `profile_history`, `profile_event_decoder` | Sequence architecture |
| `attention` | `eager`, `sdpa` | Sequence architecture (kernel only) |
| `objective` | `next_event`, `next_event_fields`, `token_event_key_mlm` | Objective |
| `readout` | `last_token`, `usr_token`, `mean` | Readout |

The token scheme and the sequence structure are independent. `flat_key_value`
feeds key-value tokens to the flat decoder, and `event_fields` feeds the 12
flat field tokens to the hierarchical encoders. The only constraint is that the
structure can find each event's boundary.

`objective.auxiliary` adds losses to `next_event_fields`, in the order listed:
`token_event_key_mlm` (PRAGMA's masked-token loss, configured by `objective.mlm`),
`past_reconstruction` (predict the previous `objective.past_reconstruction.horizon`
events' fields), and `subsequence_contrastive` (a CoLES-style InfoNCE loss
between two random crops of each history; see `objective.contrastive`).

## Probe arms and evaluation

`fraud_task.use_raw_features` and `fraud_task.use_tfm_embeddings` select the
frozen-probe arms: `raw`, `embedding`, and `combined` (which runs only when
both are on). Setting both to false is an error.

`evaluation.history_window.enabled` (default `false`) adds a second probe
that scores one row per labelled fraud event, each with its own history up to
that event; `max_rows` caps the rows per side, and `null` keeps every row. Its users are split on every
labelled event in the evaluation window. `train_negatives_per_positive` keeps
every train-side positive and samples that many negatives per positive; the
evaluation side stays a uniform sample. `evaluation.tabular_baselines.exclude_keys`
lists event or profile keys, such as a user id, that the tabular arms build no
feature from. `evaluation.tabular_baselines.xgboost_trials` runs an Optuna search
per XGBoost arm; `xgboost_params` instead gives every XGBoost arm one fixed
setting (keys `max_depth`, `learning_rate`, `min_child_weight`, `subsample`,
`colsample_bytree`, `reg_lambda`, `reg_alpha`, `scale_pos_weight`), with early
stopping choosing only the tree count. Set one of them; setting both, or an unknown
key, is an error. `evaluation.surprise` (default `false`) adds
`xgboost_raw_embedding:surprise` on every probe with a history: the embedding
joined to the model's next-event surprise. It needs the XGBoost arms and a model
with a next-event field head. `features.fields` selects and orders the fields of
`transaction_field_tokenizer`. `evaluation.precision_at_k` sets the `k` of
precision@k. `evaluation.inference_batch_size` and `evaluation.inference_precision`
(`fp32` or `bf16`) set how embeddings are extracted.

## Runtime and data

`runtime.device` accepts `cpu`, `cuda`, `cuda:N`, or `auto`. `auto` selects
CUDA when `torch.cuda.is_available()` is true, otherwise CPU. Requesting
`cuda` without a GPU is an error.

`data.adapter` is `synthetic` (default), `tabformer`, `pragmatiq`, another
registered adapter, or an adapter's import path, `"package.module:attribute"`. External
books use `data.path` or `TFM_TABFORMER_ROOT` / `TFM_PRAGMATIQ_ROOT`.
`data.max_entities` (default `n_entities`) caps sampled users. `data.split_protocol`
is `timestamp` or `precomputed` (TabFormer v1 parquet partitions).
`data.amount_field` (default `amount`) is the numeric field the raw features and
a token cache read as each event's amount; a run whose train side has no event
carrying it as a number is refused by the token path. Graph recipes use the
same source field for event features, future amount targets and raw probes;
missing values retain the graph's masked-target/presence-flag semantics.
Managed graph encoder manifests bind the source field identity.

Graph recipes default `model.composition.position` to `time_rope` and also
accept `nope`, consistently for the hybrid and sequence controls. Pretrain-only
graph stages do not fit or persist a sequence tokenizer. Graph recipes currently
reject `data.token_cache` during config validation.

`data.token_cache_on_device` (default `true`) holds a token cache on the GPU and builds
training batches there; `false` keeps it on the host, and a CPU run ignores it.
`data.batch_builder` (default `token_cache`, experimental) selects what turns a token
cache's rows into training batches: a name registered with
`tfm.data.token_cache.builders.register_batch_builder`, or the
`"package.module:attribute"` path of a `tfm.data.token_cache.builders.BatchBuilder`.

`model.backbone` is `eager` (default), `huggingface`, or `gru` (an
experimental recurrent baseline); it applies to `flat` only. `training.backend` is a
registered backend: `eager` (default), `ddp`, or `fsdp2`. `fraud_task.head` is `logistic_regression` (CI
default) or `xgboost` (optional extra). `fraud_task.supervised_head.enabled` (default
`false`) adds a `SupervisedHead` objective so `Binary` heads train end to end,
weighted by `loss_weight`; `probe_only` keeps that loss from reaching the
backbone. Experimental `ProfileHistory` stacks
are built via `tfm.experimental` / `profile_history_from_config`, not the core
`TrxFM` export.

## Model width

`model.composition.granularity.decoder.ffn_size` sets the feed-forward width for
both backbones and defaults to `4 * hidden_size`. `decoder.max_seq_len` is the
model context; it is raised to cover `data.max_tokens` but never lowered, so a
config may declare a context longer than the length it is collated or trained
at.

`model.huggingface` carries the Llama-specific settings the neutral decoder
contract does not name: `num_key_value_heads` (grouped-query attention,
defaulting to `num_heads // 4`), `rope_theta`, and `tie_word_embeddings`. They
exist so a published reference architecture's exact config can be reproduced;
each is ignored by the eager backbone.

## Training

`training.packing` (default `false`) concatenates tokenized sequences into rows
of `decoder.max_seq_len` tokens with a
block-diagonal causal mask, so nearly every position carries a next-event
target. It applies to training only; inference and evaluation keep the padded
one-row-per-entity collate that pooled embeddings require, and it cannot be
combined with `fraud_task.supervised_head` because a packed row has no single
sequence-level label. It requires a backbone that honors segment boundaries:
the HuggingFace backbone does not, and raises rather than let attention leak
across entities.

`data.pretrain_windows` changes what the training corpus contains. Left
unset, each entity contributes one sequence of its most recent
`data.sequence_length` events, which is what evaluation needs but discards the
rest of the book. Set, it tiles each entity's ordered history into
non-overlapping windows of `events` events and collates them to `max_tokens`
(default: `data.max_tokens`) so windows and probe sequences can have different
row widths. Windows still end at or before the split cutoff, and evaluation still
pools one embedding per entity, so no downstream contract changes.

`model.composition.attention` selects the kernel: `eager` (default, the
correctness baseline) or `sdpa` (PyTorch's fused kernel, which can dispatch to
FlashAttention on CUDA). An unrecognized name is an error.

`model.composition.granularity.varlen` applies to `profile_history` only. It
runs the event encoder padding-free by dropping all-padding event rows and
trimming the token width, which is worthwhile when histories are heavy-tailed.
Results match the padded path to floating-point tolerance.

`model.composition.granularity.time_features` applies to `profile_history` and
`profile_event_decoder`. It names the per-event arrays the calendar embedding reads,
each as a cycle: `{array: period}`, or `{array: {period, origin}}` when the array does
not start at 0. Each becomes the sine and cosine of 2π (value − origin) / period. The
default is `{hour: 24, dow: 7, dom: {period: 31, origin: 1}}`. Key order does not
matter: the embedding reads `hour`, `dow` and `dom` first, then the others by name, so a
config whose keys are sorted on save loads the same weights. A run refuses a token cache
that does not hold every array named, before the first step. Training batches from a
token cache carry, and on the GPU hold, only these arrays.

`objective.mlm` applies to the `token_event_key_mlm` objective only:
`p_token`, `p_event`, `p_key`, `p_unk`, and `label_smoothing`. The defaults are
PRAGMA's (15% / 10% / 10% masking, 10% `[UNK]`-as-dropout excluded from the
loss, 0.1 smoothing). Smoothing matters because value ids are bucketed, so
adjacent buckets are near-synonyms rather than distinct classes.

`training.optimizer` configures the optimizer and the schedule: `betas`,
`weight_decay`, `lr_schedule` (`constant` default, or `cosine`),
`warmup_steps` (linear), and `min_learning_rate` (the cosine floor). The
defaults reproduce a plain constant-rate AdamW.

- `name: muon` (experimental, eager and `ddp` only) trains the hidden
  matrices, the weights of every `nn.Linear` with both dimensions above 1,
  with `torch.optim.Muon` at `muon.learning_rate`. Embedding tables, norms,
  biases, learned slots, and one-output heads stay on AdamW at
  `training.learning_rate`, as PRAGMA trains. `muon.momentum` and
  `muon.adjust_lr` (`original` or `match_rms_adamw`) pass through. The
  schedule scales both groups by the same factor.
- `max_grad_norm` clips the global gradient norm (default `None`, no
  clipping). Runs record the median and maximum pre-clip norm and the mean
  loss of the last 100 steps in `train_stats.json`.

`training.log_every` (default `None`) prints a progress line every that many
steps and at the last step: epoch, loss, learning rate, gradient norm, token
throughput, and time left. Only the first rank prints, and its loss is its own
minibatch loss. Whatever it is set to, every run writes each step's loss,
learning rate, gradient norm, and token count to `train_log.json`.

`features.hashed_keys` (`key_value_time` only, experimental) encodes each
listed key as one token, a stable BLAKE2 hash of the value into
`features.hash_buckets` buckets, instead of the value's bytes. Each hashed key
has its own range of buckets, because the output layer is tied to the input
embedding and a shared bucket would have to stand for a merchant, a device,
and a counterparty at once. Use it for identity keys such as these, whose
repeats and novelty matter more than their spelling. Nothing is fitted, so an
unseen value still gets a bucket. Two values of one key that collide share a
token. The default (no hashed keys) leaves every token id, tokenizer state,
and cache identity unchanged.

## Fine-tuning

`fraud_task.finetune` (default disabled) applies to every stack. When enabled,
each label task gets a second, independent result. The pretrained model is
copied (the frozen baseline is never touched), then two choices decide the rest:

- `method: lora` freezes the copy and attaches low-rank adapters to the modules
  named in `targets` (`qkv` and `out` by default), sized by `rank`, `alpha`,
  and `dropout`. A target matches a module name's dotted suffix, so the
  transformer blocks' MLP projections are `ff.0` and `ff.2`. `method: full`
  unfreezes every weight.
- `head: linear` reads the embedding with a fresh `Binary` head.
  `head: joint_fusion` is nuFormer's head: a DCNv2 network over the embedding
  and the PLR-embedded raw features, trained end to end with the encoder.

Training runs for `max_steps` at `learning_rate` and `batch_size`. With
`validation_fraction` above 0 (default 0), that share of the training
entities is held out, whole, for early stopping. Every `eval_every` steps the
held-out log-loss is checked. The weights with the lowest loss are kept, and
training stops after `patience` checks without improving. The manifest records
`best_step`, `best_validation_loss`, and `best_validation_pr_auc`. Metrics
appear as `<arm>_<task>`, where the arm is `lora`, `full`, or
`<method>_joint_fusion`, alongside (never in place of) the frozen-embedding
probe. `finetune_manifest.json` records the adapted modules and the trainable
parameter fraction.

## Experimental graph recipe family

Every recipe loads through `tfm.config.RecipeConfig`. `recipe_family` selects
an open runner registry: `sequence` (default), `graph`, a user-registered name,
or an importable `package.module:function`. The CLI parses YAML once and
passes that config to the runner. See the committed
[`gnn_encoder` config](../../recipes/gnn_encoder/config.smoke.yaml) and
[`gnn_sequence` config](../../recipes/gnn_sequence/config.smoke.yaml).

`model.composition.granularity.type` selects a registered graph architecture or
an importable `GraphArchitecture`. Built-ins are `gnn_encoder` and
`gnn_sequence`. `hidden_size`, `depth`, `fanout`, `target_depth` and `trim`
configure the graph. `decoder`, `profile_depth`, `event_depth`, `varlen` and
`time_features` configure the sequence stack and its controls, exactly as in
Blueprint v2. `time_features` reads named `batch.events` arrays.

`history_model` and `task_head` optionally select registered/importable factories;
`model.composition.readout` selects a built-in or user readout spec.
`encoder_type` selects the graph-stage architecture for a graph-history model.
The default uses the direct Kumo Engine implementation, no substitute.
`GraphBuildContext` is the same public schema/encoder context passed to a
user model factory and to built-ins.

The graph-only composition objective matches `training.objective` (`binary`
or `future_subgraph`) except in downstream-only mode, which is binary. A
hybrid composition is always cutoff-level `binary`: a graph at history end
cannot safely condition prefix-level next-event objectives. Surprise is refused.
`training` configures the encoder stage; `sequence_training` configures the
hybrid and sequence controls. Their optimizer, precision and logging settings
are shared with other recipes. Eager unpacked, single-process execution is the
current supported graph backend. `attention: sdpa` is the graph default;
`eager` is the correctness control.

`data.adapter` uses the common adapter registry/import-path resolver. Adapters
can return timestamped `transfers` alongside transaction events. Graph inputs
remain allowlisted numeric features with cutoff-bounded availability.
`data.sequence_length` limits retained history and `max_tokens` limits tokens
per event. Source `observed_through` is an explicit coverage watermark, not
the last event timestamp. `pretrain_end`, `horizon_days` and `label_delay_days`
apply to future-target construction. `probe_cutoff` applies to supervised arms.
`training_graphs` selects `shared` or `disjoint`; validation and extraction stay
disjoint. Graph sampling is never stratified by fraud labels.

`runtime.stage` is `full`, `pretrain` or `downstream`.
`runtime.encoder_checkpoint` is required for downstream-only training.
Pretrain-only FutureSubgraph defaults `data.load_labels` to false; other graph
stages require it true. Explicit choices inconsistent with the selected stage
are rejected. The graph defaults set `stratify_by_fraud: false` and refuse
label-dependent sampling. The [migration guide](../migration/2.0.0a3-to-2.0.0a4.md)
lists stage artifacts and plain encoder reuse.

Checkpoint identity covers configuration (apart from device), data, declared
module types, schema, preprocessing, sampling and stage. Embedded metadata is
validated before weights are installed; JSON sidecars are derived from it.
Plain encoder manifests bind their bytes and input contract separately.
Source provenance may be unknown when Git metadata is unavailable.


Graph recipe defaults apply identically to YAML mappings and typed Python model
sections. An omitted position selects `time_rope`; an explicit `nope` is retained.
The event schema is validated before graph feature/target construction in every
stage. The configured amount field must have at least one finite value in
training-owner history by a graph-training cutoff; held-out owners or future target values do not establish
that presence. Missing amounts on individual events remain supported.

## Recipe loader selection

`data.loader` defaults to `kumo_pytorch` for every sequence composition and graph
recipe. `python` selects the previous reference loader explicitly. A configured
token cache retains its batch builder; packing retains the packer over native
histories. See the [migration guide](../migration/native-loader-default.md) for
source build requirements and graph schema/clock limitations.
