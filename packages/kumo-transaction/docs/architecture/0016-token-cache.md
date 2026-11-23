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
   encoded once and any sequence is a slice. The cache directory (format 4)
   holds at most 256 part files, `parts/00000.bin`, `parts/00001.bin`, …; the
   build splits its blocks into that many runs of consecutive blocks, by block
   count alone, and one task encodes each run and appends its blocks to the
   run's file. Each part file opens with a 32-byte header: magic, format
   version, part index, and the manifest's `build_id` (a hash of the cache's
   identity and block size, so identical builds stay byte-identical). The
   reader checks it when it first maps or opens the file, so a part copied
   from another cache is refused, not read. A block holds whole users, and its
   arrays, each 64-byte aligned, are:
   - per event, in `order_events` order within each user: `ts_us` (true
     microseconds), `id_us` (the microseconds in `event_id`), `hour`, `dow`,
     `dom` (UTC), `amount` (the numeric field `data.amount_field` names, 0.0
     where an event has none; a build whose train side never carries it is
     refused), and `by_loader` (the position of the event whose `event_id`
     index is `i`);
   - per token: `key_ids`, `value_ids` (uint16), `positions` (uint8), with
     `token_offsets` per event (starting at 0 in each block), already cut to
     `max_event_tokens`.

   Beside the parts: `users.parquet` (each user's event range in the whole
   cache), the users' `profiles.parquet`, `tokenizer.json`
   (`KeyValueTimeTokenizer.state_dict()`), and `manifest.json`, whose
   `blocks` list gives each block's events, tokens, users, part and array
   byte offsets, and whose `arrays` gives each array's dtype. `TokenCache`
   reads each array as one `ShardedArray`, equal to the blocks concatenated
   in order. The file count is bounded, not proportional to the book, because
   a reader keeps every file memory-mapped only while the files fit a quarter
   of its open-file limit; past that it reopens files with `os.pread`, which can
   add repeated file opens on network storage. A reader refuses any other format
   and names the command that rebuilds it.
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
   sequences. The builder reaches the tokenizer only through the
   `ColumnarTokenizer` protocol (`tfm.contracts.tokenizers`), and the book only
   through the adapter's `ColumnarSource` (`tfm.contracts.sources`): it cuts each
   block to these train-side events, and `fit_block`, `merge` and `fit_stats`
   do the counting. Later events are encoded but never counted, and no label file
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
   specification, the tokenizer's class by import path, and its `settings()`
   (with the version of its state format). A run recomputes these from its
   config and refuses a cache that differs.
6. **What a cache cannot reproduce is an error.** `data.token_cache` requires
   an `Adapter` with a columnar source (`pragmatiq` is one; a user adapter is
   named by import path), a tokenizer that implements `ColumnarTokenizer` (by
   registered name or import path), and a timestamp split. Native and spliced
   fraud-label tasks are supported for downstream scoring. Cached pretraining
   rows contain no task labels; native source labels remain on supervised
   batches outside the cache. Before the fit pass, the build runs its first block through fit, merge, a state round
   trip and encode, in the parent process, so a tokenizer or source that cannot
   build fails there; `key_value_time` refuses a non-UTC `calendar_tz` at that
   point. A cache built by any `ColumnarTokenizer` is read back into batches
   through `CachedTokenizer`. Packing
   and `fraud_task.supervised_head` are already refused for key-value-time stacks.
7. **CPU first.** The builder uses pyarrow and NumPy in worker processes, one
   block of whole users each. A cuDF builder may be added as an optional
   extra if it writes the same format and passes the same contract tests.
8. **The build is projected on the node that runs it.** How long a build takes,
   how much memory each worker needs, and whether reading the book once (spilling
   each decoded block to scratch) would beat decoding it twice all depend on the
   node and the book. A table of defaults per machine would be wrong on the first
   node it did not list. `tfm cache-tokens --profile` instead builds one sampled
   block per worker at once (the largest, the first, the last, and evenly spaced
   ones) and times each phase. Each pass's time per block is fitted as a fixed
   cost plus a cost per row, and every planned block, at its own size, is handed
   in order to the first free worker, as the build's pool does; a short last
   block therefore neither costs a whole round nor sets the range. It changes no
   setting; a later build prints its time beside the projection.

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
