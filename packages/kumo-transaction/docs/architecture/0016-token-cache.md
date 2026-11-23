# ADR 0016 — Token caches for whole-book runs

## Status

Accepted; experimental. Implemented in `tfm.data.token_cache` and
`tfm.data.adapters.pragmatiq_columns`, built with `tfm cache-tokens`, and
selected with `data.token_cache`. `tests/contract/test_token_cache.py` holds
decisions 2–4. Builds on [ADR 0015](0015-dataset-strategy-and-spliced-fraud.md);
changes no rule of [ADR 0013](0013-fraud-probe-split-integrity.md). The
[2026-10 amendment](#amendment-2026-10-a-flat-field-cache-for-tabformer) adds a
flat-field cache for TabFormer (`tfm.data.token_cache.flat`,
`tests/contract/test_flat_token_cache.py`).

## Context

`prepare_data` builds one `TransactionEvent` per book row, fits the tokenizer
on those objects, and collates every sequence into one padded batch before
cutting minibatches. On the v2 book (`synth_100k_gen2_m31_seed0`: 100,000
users, 128.5M events, 2.47B fields) that costs about 7.7 GB of memory per
1,000 users. The recorded lab runs therefore sample 10,000 users. Pretraining
on every user needs about 770 GB this way, far beyond the 121 GB workstation
(and beyond a typical training node).

Compute is not the constraint: reading the book's columns with pyarrow takes
seconds and counting every field a minute. GPU dataframes (cuDF) were
considered because the v1 Blueprint used them for its preprocessing. They
would speed up the part that is already fast, add a CUDA-versioned dependency
outside the lockfile, and need the same exactness checks (ADR 0009). They
stay a possible later backend (decision 7).

## Decision

1. **A cache is the book, tokenized once.** An event's key-value-time tokens
   depend only on its own fields and the fitted vocabulary, so every event is
   encoded once and any sequence is a slice. The cache directory (format 1)
   holds:
   - per event, in `order_events` order within each user: `ts_us` (true
     microseconds), `id_us` (the microseconds in `event_id`), `hour`, `dow`,
     `dom` (UTC), `amount`, and `by_loader` (the position of the event whose
     `event_id` index is `i`);
   - per token: `key_ids`, `value_ids` (uint16), `positions` (uint8), with
     `token_offsets` per event, already cut to `max_event_tokens`;
   - `users.parquet` (each user's event range), the users' `profiles.parquet`,
     `tokenizer.json` (`KeyValueTimeTokenizer.state_dict()`), and
     `manifest.json`.
2. **The vocabulary is fit on the train side of every cached user, and on
   nothing else.** For each user, the fit sees the events `prepare_data`'s
   train-side sequence holds: `ts < train_end` (`assign_temporal_split`),
   inside `observation_window` before `train_end`, and the most recent
   `sequence_length` of them. Profile attributes and milestones dated at or
   before `train_end` are added as `collect_statistics` adds them. The builder
   computes these statistics from columns: counts, sample totals, numeric
   readings, and the order in which keys and values first appear, which
   decides vocabulary ids and ties between equally frequent values. It then
   calls `fit_statistics`, so the fitted state equals `fit` on the same
   sequences. Later events are encoded but never counted, and no label file
   is read.
3. **What changes compared with a run without a cache.** Without a cache, the
   tokenizer is fit on the `max_entities` sample and pretraining reads that
   sample. With `data.token_cache`, it is fit on every cached user's train
   side (decision 2), and pretraining reads every cached user from disk. The
   probes and baselines keep the sample, its splits, its label windows, and
   its splice (ADR 0015). The sample must lie inside the cache. The split
   manifest and every reported probe number mean what they mean without a
   cache; the pretraining corpus and the vocabulary are wider.
4. **Cached batches equal the object path's batches.** On a synthetic book
   with users that straddle row groups and blocks, rows out of time order,
   tied timestamps whose `event_id` indices sort as strings, an event exactly
   at `train_end`, normalising numeric strings, and profile milestones on both
   sides of the cutoff, the contract tests hold:
   - the fitted state equals `KeyValueTimeTokenizer.fit` on the train-side
     sequences, also for a user subset;
   - changing only events after `train_end` leaves the vocabulary unchanged;
   - minibatches equal `collate_batch` over the whole corpus, then
     `shuffled_rows`, then `minibatches`, tensor for tensor. This covers both
     one row per user and `pretrain_windows.events` windows. Padding is to the
     corpus-wide maximum, as that path pads;
   - `encode` of any probe sequence equals the object tokenizer's `encode`.
     The cached tokenizer looks events up by `event_id`. A spliced copy is
     looked up through `IdentityTheftSplice.copied_from`, because its tokens
     are its donor event's.
5. **A cache states what it was built from.** The manifest records the book
   (path, row count, size, row groups, manifest hash), the population, the fit
   specification, and the tokenizer settings. A run recomputes these from its
   config and refuses a cache that differs.
6. **What a cache cannot reproduce is an error.** `data.token_cache` requires
   `adapter: pragmatiq`, `tokenizer: key_value_time`, a timestamp split, and
   `fraud_labels: spliced_identity_theft`. The cache holds no labels, so it
   cannot carry the book's own fraud labels into pretraining rows. It also
   refuses a non-UTC `calendar_tz`. Packing
   and `fraud_task.supervised_head` are already refused for key-value-time stacks.
7. **CPU first.** The builder uses pyarrow and NumPy in worker processes, one
   block of whole users each. A cuDF builder may be added as an optional
   extra if it writes the same format and passes the same contract tests.

## Consequences

- The eager and DDP trainers, the two that train key-value-time stacks, index
  their batches instead of copying them into a list, so cached minibatches are
  collated as they are read. `shard_batches` returns a strided view.
- A result pretrained with a cache is a different protocol from the recorded
  10,000-user runs (decision 3). It needs its own record, and it does not
  share a table with them as if only the seed differed.
- Rebuilding is required whenever the book, `train_end`, `observation_window`,
  `sequence_length`, `max_tokens`, `categorical_keys`, or the tokenizer
  settings change.
- The object path stays the reference. Recipes without `data.token_cache`
  behave exactly as before.

## Alternatives rejected

- **Stream objects per epoch.** This keeps the 6 KB-per-event cost on every
  pass and spends most of an epoch rebuilding Python objects.
- **Fit on the sample, pretrain on the cache.** Users outside the sample would
  meet a vocabulary fit without them: merchants and devices seen only there
  would become `[UNK]`.
- **cuDF now.** See Context. The CPU builder is not the bottleneck, and parity
  would have to be shown twice.

## Amendment (2026-10): a flat-field cache for TabFormer

### Context

The TabFormer recipe pretrains `transaction_field_tokenizer` windows of every
user. `prepare_data` builds them from objects: on all of TabFormer (1,584 users,
19.5M train events) that took about 7 minutes and 48 GB of host memory before
training, measured on a GB10, plus about 2 minutes to collate the 58,000
windows. Under `torchrun` every rank repeats it, although each rank trains on
only its share and only rank 0 evaluates.

### Decision

1. **The same command, a second format.** `tfm cache-tokens` builds a
   `flat_fields` cache (format 1) when the config uses
   `transaction_field_tokenizer`; `data.token_cache` selects it. It holds, per
   train event of the sample and in `order_events` order within each user:
   `tokens` (uint16, one column per field), `ts_us`, `amount`, `fraud`,
   `fraud_available_us`, and `event_id`, plus `users.parquet` and
   `manifest.json` (`tfm.data.token_cache.flat`).
2. **Nothing is reimplemented.** The field tokenizer's vocabulary is fixed, so
   there is nothing to fit. Workers map each train row of the sample with the
   adapter's own `tabformer_row_event` and the tokenizer's own `event_tokens`,
   in the adapter's row order, so event ids and their tie-breaking are the
   object path's.
3. **The cache holds the sample, not the book.** Unlike decision 3 above, the
   pretraining corpus and probes see the same users: the `max_entities` sample
   of `precomputed_sample`. The manifest identifies the sample by its user set,
   so another seed that samples the same users reads the same cache.
4. **Windows equal the object path's.** `FlatTokenCache.train_windows` cuts
   `build_entity_windows`'s windows from the arrays at the precomputed train
   cutoff (the sample's last train event), and `CachedFlatBatches` collates them
   as `collate_batch`, `shuffled_rows`, and `minibatches` do: input ids, masks,
   labels, fraud labels, raw features, and ids. The contract tests hold this on
   a synthetic book with tied timestamps, string-sorted event ids, truncation,
   and labels on both sides of the label delay; on all of TabFormer every one of
   the 58,000 windows was checked once.
5. **Training without objects.** `setup_run(pretraining_only=True)`, which the
   `train` command passes, reads a flat cache through `prepare_pretraining` and
   loads no event on any rank. `infer`, `evaluate`, and `report` still call
   `prepare_data`, which builds the probes from objects and reads the
   pretraining windows from the cache.
6. **What a flat cache cannot serve is an error:** another adapter, a
   timestamp split, no `data.pretrain_windows`, or `training.packing`.

### Consequences

- A full-size run splits into jobs: `tfm cache-tokens` once, `tfm train`
  (multi-GPU under `torchrun`), then `tfm evaluate --reuse-checkpoint` on one
  process (`recipes/blueprint_v1/config.cached.yaml`). On all of TabFormer the cache builds in about 75 s with 8 workers
  (7.5 GB peak, 1.4 GB on disk); a training job reaches its first step in
  seconds and holds about 3 GB of host memory per rank.
- Rebuild the cache when the TabFormer files, the sample, `label_delay`, or
  `features.fields` change.
