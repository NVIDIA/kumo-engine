# Glossary

## Data and labels

- **Entity** — whose history a sequence is: a user, account, or card.
- **Event** — one timestamped record in an entity's history: a transaction,
  an app event, a transfer. Contract: `TransactionEvent`.
- **Profile** — attributes of an entity that are not events (age band,
  segment, lifelong milestones), as known at the cutoff. Contract:
  `EntityProfile`.
- **Book** — a dataset of entities, events, profiles, and labels.
- **Adapter** — code that maps an external book onto the kit's contracts.
- **Cutoff** — time at which a sequence is frozen; later events are invisible.
- **Observation window** — how far back from the cutoff a sequence reaches.
- **Label delay** — duration after an event before its label may be used.
- **Label window** — the `(start, end]` period whose labels one side of a
  probe may count (probe split rules).
- **Prediction horizon** — how far after the cutoff a forecast label looks
  (for example 12 months for `default_12m`).
- **Event-level / entity-level / sequence-level label** — a label on one
  event, on an entity at a date, or on a whole sequence at its cutoff. They
  stay distinct in types and APIs.
- **Leakage** — any path by which information from after the cutoff, or from
  the label, reaches a model input.

## Model

- **Slot** — one of the five building-block choices: data, event
  representation, sequence architecture, objective, readout and adaptation
  (building-block contracts).
- **Tokenizer** — turns events into token ids. **Layout** — the batch shape a
  tokenizer produces: `flat_fields`, `flat_key_value`, or `key_value_time`.
- **Key-value tokens** — each field becomes a `(key, value)` pair, so events
  can have any set of fields (PRAGMA).
- **Granularity / structure** — how tokens are grouped: one flat stream
  (`Flat`), or a hierarchy that encodes each event and then the history
  (`ProfileHistory`, `ProfileEventDecoder`).
- **Backbone** — the network that reads the stream: an eager transformer,
  HuggingFace Llama, or a GRU.
- **Causal / bidirectional** — whether a position may attend only to earlier
  positions, or to all of them.
- **TimeRoPE** — rotary position encoding over continuous log-time between
  events. **NoPE** — no position encoding; order comes from the causal mask.
- **Objective** — what pretraining asks the model to predict. **Auxiliary
  objective** — an extra loss added to the primary one.
- **MLM** — masked-language modelling: hide tokens and predict them.
- **Readout / pooling** — which state (or mean of states) becomes the
  embedding.
- **Embedding** — the fixed-size vector a readout produces for one sequence.

## Evaluation and adaptation

- **Probe** — a simple classifier fitted on frozen embeddings to measure what
  they encode.
- **Arm** — one feature set and classifier scored on a task's rows, such as
  `xgboost_raw`, `xgboost_raw_embedding`, `tabular_xgboost`, or a fine-tuned
  arm. On CPU without the XGBoost extra: `raw`, `embedding`, `combined`.
- **Raw row** — the last event's own fields plus the profile, with nothing
  from earlier events: what `xgboost_raw` reads.
- **Headline** — per task, the lift of `xgboost_raw_embedding` over
  `xgboost_raw` on PR-AUC (evaluation protocol).
- **Lift** — `x / x_ref - 1` for a metric over a stated reference arm.
- **Gain recovered** — `(raw+emb - raw) / (tabular - raw)` on PR-AUC: the
  share of the hand-built history features' gain an embedding recovers.
- **Tabular baseline** — XGBoost (`tabular_xgboost`) or logistic regression on
  hand-built aggregates of the same history, with no encoder; the stand-in
  for a production model.
- **Spliced identity theft** — the pragmatiq fraud task: a donor's
  transactions copied into a victim's history on a theft day, so the label
  depends on the history and not on one event (dataset strategy).
- **Surprise** — the model's next-event cross-entropy on a row's latest
  events, each scored from its prefix; an extra readout, not an objective.
- **Entity-disjoint** — no entity appears on both the fit and the score side.
- **History-window probe** — one row per labelled event, embedding the
  history up to that event; its **single-event control** embeds the event
  alone.
- **LoRA** — low-rank adapters trained while the base weights stay frozen.
- **Joint fusion** — a head that combines the embedding with raw features and
  trains end to end with the encoder (nuFormer's DCNv2 over PLR embeddings).
- **Seed range** — min / median / max over several seeds; a single seed is
  not a result.
- **Record** — a summary and its JSON under `benchmarks/results/`, from a
  committed config; the only evidence behind a claim.

## Recipes and runs

- **Recipe** — one reference architecture: a directory of configs and a
  README, run with `tfm <stage> --recipe <slug>`. **Variant** — one of its
  configs, `config.<variant>.yaml`, chosen with `--config <variant>`.
- **Smoke config** — `config.smoke.yaml`: generated data on CPU, run in CI.
- **Ablation** — a benchmark config that changes one slot of a recipe, under
  `benchmarks/ablations/`.
- **Token cache** — key-value-time tokens of a whole book written to disk once
  and reused by every run on it (token-cache guide).
- **Backend** — the training loop: `eager`, `ddp`, or `fsdp2` (backend contracts).

## Support

- **Reference** — documented, tested, smoke-run capability with recorded artifacts.
- **Experimental** — runnable but unstable.
- **Deprecated** — still works, scheduled for removal; use the named replacement.
- **Planned** — not implemented.
- **TrxFM** — public composition API for a transaction foundation model.
