# Changelog

## Unreleased

- Bind native replay requests to owned snapshots and declared split views.
  Keep label-free replay unlabeled and validate overlap within each view,
  including precomputed partitions with equal cutoffs.

- Validate native replay labels and source watermarks, reject source replacement
  during normalization, preserve late-observed future terminals and empty fit
  views, and retain frozen-GNN probes and standalone downstream checkpoint reuse.
  Honor amount-field and sampling settings in native sequence preparation.

- Replace canonical native recipes' prepared-window/indexed-batch path with
  source-table DiskGraph indexes, train-fitted native encoders, continuous
  seed/epoch iterators and owned CPU tensor prefetch. Stream graph and hybrid
  batches instead of retaining split-wide graph lists. Preserve explicit Python
  and token-cache workflows; document native limits and checkpoint migration.

- Bind managed model identity to ordered features; retain known-negative labels
  and aligned unknown-label inference masks. Validate FK columns and split
  separation, constrain graph window exemptions and group shared future batches
  at clock boundaries. Include the combined logistic evaluation arm.

- Add a bounded adapter declaration and generated/replayed prediction anchors
  shared by sequence, graph and hybrid reference recipes.
- Preserve event/observation clocks, point-in-time history ownership, canonical
  recent fanout and explicit entity-level binary label identities.
- Add native one-hop candidate retrieval across arbitrary fixed-FK
  frontiers; unsupported projections fail early.

- Exclude standalone Transaction ADRs from the public source sync and replace their references with public guides. Use structural resource cleanup and loader-supplied provenance without changing graph traversal inheritance.

- Preserve the advanced native example’s completed-index reuse (`--index`) and print its index path; validate reuse with source tables offline. Cache native graph edge clocks once per sampler.

- Make the pinned experimental Kumo PyTorch/DiskGraph integration the default for sequence, graph and hybrid recipe loading. Keep explicit `data.loader: python` reference selection and document preparation/materialization tradeoffs without an unmeasured speed or memory claim; keep separate FutureSubgraph targets. Add native graph/hybrid wiring, installation/migration guidance and executable integration checks.

## 2.0.0a4

- Use the reference Muon update on CPU, preserving exact parity across CPU bf16 kernel implementations. CUDA retains grouped updates.

- Treat non-finite probe amounts as missing (zero contribution), so mixed finite/NaN graph data can be evaluated. Keep object and cached raw aggregates consistent without mutating stored values.

- Reject label-like text inputs through shared event validation. Require a finite graph-training amount while retaining per-event missing-value masks.

- Validate graph source events in tokenizer-free stages; reject label-like input fields and amount fields absent from training history. Preserve explicit position choices when applying graph defaults to typed Python configs.

- Harden encoder checksums, cache population identities and integer event arrays, optional profile reports, and ignored-row loss masking. Preserve non-row tensor metadata and implicit source tokens for custom columnar loaders.

- Add experimental Kumo Engine graph encoders and graph-conditioned sequence models, with direct binary supervision or optional future-subgraph pretraining. Graph context enters through public event-encoding/history-decoding hooks.
- Use the shared recipe schema and open architecture, runner, data-adapter, tokenizer, readout and context-history factories. User components use the same inputs and build context as built-ins.
- Honor graph recipe amount and position slots across model and probe paths; bind amount identity into encoder manifests. Pretrain-only graph runs skip sequence tokenizers, and graph token caches are rejected explicitly.
- Add explicit pretrain, downstream and full graph stages. Label-free pretraining exports a plain encoder state dictionary; managed downstream loading verifies its bound feature/schema manifest. Binary training and scoring retain source-label coverage checks.
- Validate graph batches on CPU, retain mutation records through lightweight device execution views, share receiver-first edge checks, and keep future-target loss masking on the device.
- Carry 2.0.0a3's configured time arrays, fused vocabulary losses and device-side training metrics into graph recipes. Document graph clocks, frozen state, source identity, stage artifacts and support limits.
- **Custom training batches:** declare `for_execution(device)` and `has_supervision`, or inherit their defaults from `tfm.contracts.training.TrainableBatch`. See the [migration guide](docs/migration/2.0.0a3-to-2.0.0a4.md).

## 2.0.0a3 — 2026-11-23

Bring-your-own tokenizer, data source and token-cache format; training batches held
on the GPU; and a training step with fewer host syncs and less memory (measured under
`tfm.kernels.linear_cross_entropy` below). Items marked
**Breaking since 2.0.0a2** are mapped in the
[2.0.0a2 to 2.0.0a3 migration guide](docs/migration/2.0.0a2-to-2.0.0a3.md). A 2.0.0a2
config loads unchanged; every token cache is rebuilt.

### Since 2.0.0a2

- Changed: token cache format 3. Each build task encodes a run of consecutive
  blocks and appends them to one file, `parts/00000.bin`, …, so no step
  concatenates the blocks. A cache has at most 256 files whatever its size, so
  the number of files a reader opens does not grow with the book. The manifest
  gives each block's part and the byte offset of each of its 11 arrays.
  `TokenCache` reads each array as one `ShardedArray` over the shared
  `CacheFiles`, and `DeviceBatches` copies them to the device one block at a
  time. A cache with no more files than a quarter of the process's open-file
  limit (256 at Linux's usual login-shell limit of 1,024) is memory-mapped;
  a larger one is read with `os.pread`, with a warning naming the `ulimit -n`
  that lets it be memory-mapped. Known limit: the reader uses `os.pread` and
  `resource`, so it runs on Linux and macOS, not Windows. Format 3 also
  covers the hashed-key buckets' move from BLAKE2b to MurmurHash3. A cache of format 1 or 2 is refused with the
  command that rebuilds it; see `docs/migration/2.0.0a2-to-2.0.0a3.md`.
- Changed: token cache format 4. Each part file opens with a 32-byte header
  (`part_header`: magic, format version, part index, build id), and the
  manifest records the `build_id`, a hash of the cache's identity and block
  size. The reader checks the header the first time it maps or opens a file, so
  a part file copied from another cache raises `SchemaError` instead of being
  read. `CacheFiles(paths, build_id)` takes the id. A format 3 cache is refused
  with the rebuild command. The `pread` reader no longer prints an ignored
  exception at interpreter exit.
- New: `tfm cache-tokens --profile` (`tfm.data.token_cache.profile`) projects a
  key-value-time cache build on the node that will run it, without building it.
  It plans the whole book and builds one sampled block per worker at once, each
  in a fresh process. It reports the projected build time with a range, the
  cache's size and part files, each worker's peak memory against available
  memory, the longest user id, and whether reading the book once (a spill to
  `--scratch`) would beat decoding it twice. The report goes to
  `<token_cache>.profile.json`; a later build with the same settings prints its
  time beside the projection. `BlockPlan` records `longest_user_id`. The
  projection fits each pass's time per block as a fixed cost plus a cost per row
  (`fit_block_seconds`) and schedules every planned block over the workers in
  order (`makespan`), after the workers' measured start-up. The scratch a spill
  needs counts each block's distinct values (their UTF-8 bytes, offsets and
  parsed numbers) besides 11 bytes per field and 16 per event.
- `tfm.data.token_cache.store.open_file_budget()` and
  `tfm.data.token_cache.build.copy_profiles()` are now public (formerly
  `_open_budget`, `_copy_profiles`).
- New: `data.amount_field` (default `amount`) names the numeric field read as
  each event's amount: the mean and maximum in the raw features
  (`numeric_features`, read by the baselines' raw arm and joint fusion) and a
  token cache's `amount`. A run or cache build whose train side has no event
  carrying it as a number raises `SchemaError`. Before, such a book read every
  amount as 0.0 and the baseline trained without it. `CacheRequest` and
  `FlatCacheRequest` take `amount_field`, which joins the cache's identity, so a
  cache built earlier is refused with the rebuild command.
  `TokenCache.raw_features` no longer loops over rows in Python; it sums as
  Python's `sum` does, so its values are unchanged.
- Fixed: a token cache honours `features.exclude_keys`. The cache build fit
  and encoded excluded keys, so a cached run trained on the fields the object
  path drops, such as an identity key. A cache built with excluded keys before
  this fix is refused with the rebuild command; other caches are unchanged.
- Changed: `TokenCache.user_ids` and `user_index` are methods. `user_ids(index)`
  gives the ids at `index` (all by default); `user_index(users)` gives each
  user's position, or -1. The cache holds its user ids as one sorted byte
  array, about as many bytes per user as the longest id, instead of a Python
  tuple and dict. The build records that length in the manifest
  (`user_id_bytes`) and warns when it is over four times the mean, which
  usually means a malformed `user_id`.
- New: `tfm.contracts.tokenizers.ColumnarTokenizer`, the protocol the token
  cache builds through: `fit_block`, `merge`, `fit_stats`, `encode_block`,
  `state_dict` and `from_state`, over an `EventBlock` (moved from
  `tfm.data.adapters.pragmatiq_columns`, which still exports it). `encode_block`
  returns an `EncodedBlock`: the tokens and the tokenizer's own per-event
  arrays, which the cache stores beside `ts_us`, `id_us`, `amount` and
  `by_loader` and records with their dtypes in the manifest. A block whose
  array names or dtypes differ from the first block's is refused.
  `KeyValueTimeTokenizer` implements it, with `KeyValueStats` as its
  statistics, and its cache is byte-identical to the one built before.
- **Breaking since 2.0.0a2:** a token cache is built from the adapter's
  `ColumnarSource` (`tfm.contracts.sources`), not from a pragmatiq root:
  `CacheRequest` takes `source` instead of `root`. A source plans blocks of
  whole users, reads a block as an `EventBlock`, writes the cache's
  `profiles.parquet` and gives the fingerprint that joins the cache's
  identity. An adapter supplies one with `register_adapter(name,
  columnar_source=...)`; `pragmatiq` registers `PragmatiqColumns`, and
  `tfm.data.adapters.registry.columnar_source` raises, naming that remedy,
  for an adapter without one. This change alone leaves a cache's bytes unchanged;
  the tokenizer identity below still rebuilds every cache.
- **Breaking since 2.0.0a2:** a token cache is built by any tokenizer that
  implements `ColumnarTokenizer`, which gains `vocab_size` and `settings()`.
  `CacheRequest.tokenizer` takes the unfitted tokenizer instead of a
  `KeyValueTimeConfig`. The cache's identity records the tokenizer's import path
  and `settings()` instead of `tokenizer_config`, so every existing cache is
  refused once, with the command that rebuilds it; the manifest drops
  `field_kind`, which `tokenizer.json` holds. `KeyValueTimeTokenizer.state_dict`
  records `version: 1`; `from_state` reads a state without one as version 1 and
  refuses any other.
- New: `features.tokenizer` and `data.adapter` take an import path,
  `"package.module:attribute"`, besides a registered name, imported on use, so a
  user class or adapter reaches `tfm cache-tokens` and its worker processes.
  `features.tokenizer_options` constructs the tokenizer with those keyword
  arguments; it is refused alongside the built-ins' `fields`,
  `categorical_keys`, `hashed_keys` or `exclude_keys`. A user adapter is an
  `Adapter(load, columnar_source)` (`tfm.data.adapters`). A registry refuses a
  second entry under a taken name. A cache takes any `data.fraud_labels`: it holds
  none, and probes read the book's. The cache finds an event by parsing its id, so a
  source's event ids take pragmatiq's `user:microseconds:index` form.
- New: `tfm cache-tokens` runs the first block through fit, merge, a JSON state
  round trip and encode, and pickles what the workers receive, before the fit
  pass; a broken source or tokenizer fails there, naming
  `tfm.testing.conformance.check_columnar_tokenizer`. That check asserts that
  a tokenizer's fitted state and tokens do not depend on block size or merge
  order, and that its settings and state survive JSON and pickle. The build
  refuses a uint64 per-event array, which torch cannot index, and a cache with no
  token. `DeviceBatches` widens uint16 and uint32 per-event arrays to int32 and
  int64, since torch indexes neither on a device.
- New (experimental): `tfm.features.tokenizers.ValueVocabulary` (with
  `VocabularyConfig`), the field kinds, value ids, numeric quantile bins and
  hashing that field tokenizers share, independent of token layout. It fits from
  sequences (`collect`, `fit`) or block by block (`block_statistics`, `merge`,
  `fit`) and encodes one value at a time or a whole block (`value_ids`,
  `encode_events`), with the same results either way. A user tokenizer can hold
  one and get the vectorised path. `tfm.features.tokenizers.KeyValuePairs`
  (experimental) is the key-value layout over it, and
  `tfm.features.tokenizers.key_value_time` exposes `encode_record` and
  `collate_records`; `KeyValueTimeTokenizer` is that layout plus time features.
- New: `tfm.contracts.tokenizers.FieldNamedTokenizer`. Surprise evaluation names
  its per-field columns by the tokenizer's `field_names()` instead of reading
  `key_vocab`, so any layout whose batches label tokens by field in `key_ids` gets
  surprise features. A key literally named `KEY_<x>` is now reported as `KEY_<x>`,
  not `<x>`.
- Changed: `TokenCache` opens a cache built by any `ColumnarTokenizer`, not only
  `key_value_time`. `cache.tokenizer(profiles, max_events=...)` rebuilds the
  tokenizer from the import path the manifest records and returns a
  `CachedTokenizer`, which holds it instead of subclassing it and reads each
  event's tokens and per-event arrays from the cache. A profile prefix comes from
  a tokenizer that implements the new `tfm.contracts.tokenizers.ProfilePrefixEncoder`.
  The cached run's `tokenizer_manifest.json` is the cache's record of the
  tokenizer (import path, settings, vocabulary size, per-event arrays); the
  fitted state stays in the cache's `tokenizer.json`.
- **Breaking since 2.0.0a2:** per-event arrays travel by name.
  `KeyValueTimeBatch.events` and `EncodedKVTRecord.events` replace the `hour`,
  `dow` and `dom` fields, `TokenCache.events` replaces `TokenCache.hour`, `.dow`
  and `.dom`, and a model reads an array with `batch.event_array(name)`. A model
  that reads per-event arrays declares them (`tfm.contracts.kvt.EventArrayReader`;
  the calendar's are `CalendarEmbedding.ARRAYS`), and a run refuses a token cache
  that lacks one before the first step, naming it. `CachedKeyValueTimeTokenizer` is
  now `CachedTokenizer`; `TokenCache.capped` takes `max_events` instead of a
  tokenizer. The cache's raw features take the last event's hour and weekday from
  its timestamp, with the same values.
- New: a tokenizer class builds itself from the recipe through a classmethod
  `from_config(features, data)` and the `feature_keys` it reads
  (`tfm.features.tokenizers.registry.ConfiguredTokenizer`). The built-ins use it, so
  `tokenizer_for` checks no class, and a user class named by import path reads
  recipe keys the same way. A set `features` key the class does not list is
  refused, naming the key and the keys the class reads.
- **Breaking since 2.0.0a2:** `event_fields` refuses `features.categorical_keys`,
  which it ignored; remove the key. Its 12 fields are fixed, so it reads no `features`
  key.
- New (experimental): `data.batch_builder` selects what turns a token cache's rows
  into training batches, by registered name (`register_batch_builder`) or import
  path of a `tfm.data.token_cache.builders.BatchBuilder`. The default, `token_cache`,
  is the previous behaviour; every source receives the run's placement. The run
  records the source and whether it was the default.
- Changed: `tfm.contracts.tokenizers.ProfileEncoder` (new in 2.0.0a2) is
  `ProfilePrefixEncoder`, which extends `UsesProfiles` (`set_profiles`); a run hands
  the book's profiles to any `UsesProfiles` tokenizer. `FieldVocabularyTokenizer`
  replaces the `flat_encoder` check for `transaction_field_tokenizer`. The lab loads
  a checkpoint's tokenizer with the configured class's `from_state`.
- New (experimental): a tokenizer class names its token cache's format in
  `token_cache_format`, by registered name (`register_token_cache_format`) or import
  path of a `tfm.data.token_cache.formats.TokenCacheFormat`; the default is
  `columnar`. A format owns the cache's config checks, build, profile, open,
  pretraining rows and batches, so no run code tests for a tokenizer or cache class.
  `transaction_field_tokenizer` uses the built-in `flat_fields` (format unchanged).
  `prepare_pretraining` reads any format that `trains_without_book`.
- Changed: `AlignmentLosses` is defined in `tfm.contracts.outputs` (exported from
  `tfm.contracts`), so the blueprint v2 layer that returns it no longer imports
  `tfm.training`. `tfm.training.diagnostics` still imports it.
- Changed: `cache_request` moved from `tfm.runtime.token_cache` to
  `tfm.data.token_cache.columnar`, and `cache_path` to `tfm.data.token_cache.formats`;
  `flat_cache_request` moved to `tfm.data.token_cache.flat`. `PreparedData` and
  `PretrainingData` carry the open cache's `cache_format`.
- Changed: `granularity.type: flat_encoder` accepts any tokenizer that implements
  `FieldVocabularyTokenizer`, and the pretraining-window width check applies to any
  tokenizer class with `event_width` (`FixedWidthTokenizer`), not to two names. A
  `flat_fields` cache refuses `data.batch_builder`. A `features` key counts as set when
  it differs from its default, so an empty list a class does not read is refused.
- New: `model.composition.granularity.time_features` sets the per-event arrays the
  `profile_history` and `profile_event_decoder` calendar embedding reads, each as a
  cycle (`{array: period}` or `{array: {period, origin}}`, `tfm.contracts.kvt.CyclicTime`).
  The default, hour 24, weekday 7 and day of month 31 from 1, gives the same features
  as before, so trained models are unchanged. The models' `event_arrays` follow the
  config; `CalendarEmbedding.ARRAYS` is removed.
- Changed: `check_columnar_tokenizer` raises `ConformanceError` (a `TfmError`) instead
  of `AssertionError`, so `python -O` keeps every check. It also checks that each
  per-event array holds one number per event in a dtype torch can copy, and builds a
  cache from the source: its device batches (CUDA when available) all have one shape
  but for the last batch's rows, and equal its host batches. Given `sequences`, the
  train-side sequences of `fit`, it checks that the tokenizer's per-sequence `collate`
  gives the cache's tokens (`EVENT_FIELDS`).
- Changed: without a token cache, a run whose model reads a per-event array the
  tokenizer does not emit is refused at setup, naming the array, not at the first step
  (`KeyValueTimeBatch.check_event_arrays`).
- Changed: `tfm cache-tokens`'s pre-build check also fits the first block's two halves
  of users, merged in either order, and refuses a tokenizer whose state then differs
  from the whole block's: its cache would depend on the worker count or
  `rows_per_block`.
- Changed: training batches from a token cache load and carry only the per-event arrays
  the model reads (`event_arrays`), on the host and on the GPU, so an array a
  tokenizer stores but no model reads costs no GPU memory and no gather. A
  `BatchBuilder` and `TokenCacheFormat.batches` take `event_arrays` (experimental, so no
  migration step); `TokenCache.collate`, `CachedBatches` and `DeviceBatches` take it
  too (default: every array). The run records the list in `packing_manifest.json`.
- Changed: `optimizer.name: muon` trains with `tfm.training.optim.GroupedMuon`, a
  `torch.optim.Muon` whose step batches the Newton-Schulz iterations over matrices of
  one shape and updates momentum and parameters with `_foreach` ops. Arguments, state
  and updates are `torch.optim.Muon`'s (bit-identical on CPU); optimizer state loads
  into either.
- Added: `tfm.kernels.linear_cross_entropy`, the cross-entropy of `hidden @ weight.T`
  computed in row chunks, so the `[rows, vocab]` logits never exist whole. It is the
  `torch.library` custom op `tfm::linear_cross_entropy`; on CUDA one Triton kernel
  computes each row's loss and gradient. `profile_event_decoder`'s next-event, MLM and
  past-reconstruction losses use it; their `next_crit` and `mlm_crit` modules are gone.
  A target of -100 (`IGNORE_INDEX`) adds no loss and no gradient, and the loss is the
  mean over the other rows, as with `F.cross_entropy`'s default `ignore_index`. The
  count of those rows stays on the device; if every target is -100, the loss and
  gradients are 0, not NaN. Any other target outside the vocabulary raises on CPU,
  and fails a device-side assert on CUDA after the call, without a host sync. Targets
  may be a non-contiguous view; each row reads its own. Measured on one H100 NVL in
  bf16: the next-event head's extra peak memory (114,161 rows, vocabulary 12,640)
  falls 13.62 → 0.75 GiB, and the recipe's peak 24.8 → 11.6 GiB (38.6 → 11.6 GiB at
  vocabulary 24,928). No step-time gain was measured; the kernel's gain is memory.
- Changed: `SdpaAttention` leaves cuDNN out of PyTorch's backend choice for sequences of
  at most 64 tokens. cuDNN builds a new execution plan whenever the row count
  changes, which the varlen event encoder does every batch.
- **Breaking since 2.0.0a2:** requires torch 2.9 or newer, the first release with
  `torch.optim.Muon`. `GroupedMuon` subclasses it, so `import tfm.training` needs it
  for every optimizer. A CI job runs the optimizer tests on torch 2.9.
- Changed: uv is pinned to 0.13.8 (`[tool.uv] required-version`, and in CI), which
  writes `uv.lock` at revision 5; another uv refuses to lock or sync the project.
- Changed: `KVTEventEncoder` runs up to 64 tokens' worth of events as one attention
  sequence, with a block-diagonal mask from `segment_ids`, so each event still attends
  only to its own tokens. `EncoderBlock.forward` takes `segment_ids`.
- Changed: `train_model` keeps each step's loss, gradient norm and token count on the
  device, so a step no longer waits for the GPU. `TrainingProgress.step` accepts 0-d
  tensors and reads them back together, in one transfer: when a line is due, every 100
  steps, and at the new `TrainingProgress.finish`. `real_tokens` returns a 0-d tensor.
- Changed: `train_model` and `EagerTrainer.fit` accept a `tfm.training.trainer.BatchSource`,
  any sized loader that yields a fresh pass over its batches on each iteration, as well
  as a sequence of batches. An iterable loader owns its batch order, so `shuffle` is
  refused for it.
- New: `examples/kumo-engine/` pretrains Blueprint v2 on batches that Kumo Engine's
  DiskGraph loader builds from relational tables while the model trains. New extras:
  `kumo` and `pragmatiq`.
- Fixed: `training.backend: ddp` refuses a loader that can only be iterated, such as
  a `tfm.training.trainer.BatchSource` that is not a sequence, with an
  `UnsupportedCombinationError` before the process group starts. It hands each rank
  every `world_size`-th batch by index, and such a loader used to fail at the first
  batch with a `TypeError`.
- Changed: `apply_mlm_mask` (the `token_event_key_mlm` objective) draws each row's key
  decisions on the device, a hash of the key id and a per-row seed, instead of a Python
  loop with one host sync per key; no tensor scales with the vocabulary. The masking
  rates are unchanged; a seeded run masks different tokens.
- Changed: the order of `granularity.time_features` keys no longer matters: the
  calendar reads `hour`, `dow` and `dom` first, then the others by name.
- **Breaking since 2.0.0a2:** `EventFieldsTokenizer` and `FlatKeyValueTokenizer`
  hold a `ValueVocabulary` and `KeyValuePairs` instead of subclassing
  `KeyValueTimeTokenizer`, so they no longer match `ColumnarTokenizer`. The
  vocabulary's fields (`field_kind`, `cat_vocab`, `token_to_id`, …) move from the
  tokenizers to `tokenizer.vocabulary`; `key_vocab` and `field_tokens` move to
  `tokenizer.pairs`. `KeyValueTimeStats` is now
  `tfm.features.tokenizers.vocabulary.KeyValueStats`, and
  `KeyValueTimeConfig.fit_value_cap` is now `KeyValueTimeConfig.vocabulary().fit_value_cap`.
  `EventFieldsTokenizer` and `FlatKeyValueTokenizer` no longer save or load fitted
  state (`state_dict`, `from_state`); a lab checkpoint naming either is refused with
  `UnsupportedCombinationError`.
- Changed: fitting a `ValueVocabulary` twice raises `UnsupportedCombinationError`
  instead of encoding ids past the vocabulary. Two fields that map to one value token
  (`a="b_c"` and `a_b="c"` both give `VAL_a_b_c`) raise `SchemaError` instead of
  sharing an id. A block fit counts every value that parses as a finite float as a
  numeric reading, whatever kind its source gives it, as the sequence fit does.
- Fixed: `build_model` takes the prepared tokenizer, so `flat_encoder` masks by a
  fitted tokenizer's field layout instead of one rebuilt from the config.
- Fixed: MLM masking keeps one supervised token when `[UNK]` dropout would drop
  every selected one; before, such a batch had no target.
- New: `tfm.data.token_cache.fit_statistics_block` exposes the token cache's
  fit pass: it cuts a block to its train-side events and returns the
  tokenizer's `fit_block` statistics as a `BlockStatistics`. Merging with the
  tokenizer's `merge` gives the same statistics for any order of blocks and
  any block size; the build merges each block as it finishes, with at most
  twice `workers` blocks in flight. Each block counts
  its own users' profiles, so the build never holds every profile. The cache's
  copy of `profiles.parquet` is written in row groups of 4,096 rows and each
  block reads only the groups whose `user_id` range can hold its users; a
  `profiles.parquet` not sorted by `user_id` is refused, naming the remedy.
- New: a pragmatiq `events.parquet` may be a directory of parquet files, read
  in string order of their paths (`part-10` before `part-9`). The token cache
  plans its blocks one row group at a time, so planning memory no longer grows
  with the book, and lists the files once: a block whose event file changed
  after planning is refused.
- Changed: the token cache build stops with `UnsupportedCombinationError` when
  one user has more events than `rows_per_block`, instead of building a larger
  block for that user.
- New: on CUDA, a key-value-time token cache is held on the GPU and training
  batches are built there (`DeviceBatches`), equal to the host path tensor for
  tensor. `data.token_cache_on_device: false` keeps the cache on the host.
- Changed: `features.hashed_keys` bucket with MurmurHash3 (x86, 32-bit, seed 0)
  instead of BLAKE2. A tokenizer state saved without a `hash`, such as a
  checkpoint's or token cache's from 2.0.0a2 or earlier, keeps BLAKE2 and encodes
  as before. A new fit puts values in different buckets than a BLAKE2 fit did.
- Fixed: a key-value-time token cache built from a source whose events omit
  `source` gives each event the implicit `source` token that `encode` gives it, so
  the cache and the object path hold the same tokens. `settings()` records
  `event_encoding: implicit-source-v1`, so a cache built before is refused with the
  rebuild command.
- Fixed: per-event arrays need an integer or boolean dtype. A cache build,
  `TokenCache` and `check_columnar_tokenizer` refuse a fractional array; before, it
  was cut to integers when batches were built.
- Fixed: `CacheRequest` and `FlatCacheRequest` hash a subset's sorted, distinct user
  ids as a JSON array, so ids that contain a newline no longer give two populations
  one identity.
- Fixed: on the CPU, `linear_cross_entropy` gives an ignored row zero loss and zero
  gradient when its logits overflow, as the CUDA kernel already did. Before, an
  overflow on an ignored row made the CPU loss NaN.
- Changed: surprise evaluation refuses a `FieldNamedTokenizer` that names a field
  `all`, the name of its aggregate columns (`last{window}/all`).
- Fixed: a token cache build refuses a categorical, numeric or text field whose name
  looks like a label (`is_fraud`, `label`, `target`, …), as `validate_event` does for
  event objects (`tfm.data.validation.schema.validate_event_block`). Before, a
  `ColumnarSource` passed it to the tokenizer as a model input.
- **Breaking since 2.0.0a2:** `validate_event` refuses a text field whose name looks
  like a label, and compares field names without surrounding spaces as well as without
  case (` Fraud ` is refused). Rename the field, or leave it out of the book.
- Fixed: a token cache build refuses a per-event array that does not hold one value
  per event, one in non-native byte order, and a uint64 array in either byte order;
  `TokenCache` and `check_columnar_tokenizer` apply the same dtype rule
  (`tfm.contracts.tokenizers.event_array_problem`). Before, a two-column array was
  read back onto the wrong events, and a byte-swapped array failed only on its way to
  torch.

## 2.0.0a2 — 2026-11-06

A rewrite of the v1 notebook example into a reference-architecture kit. Nothing
from `v1.0.0` (tag, and branch `v1`) is compatible: no checkpoint, config, or
metric carries over. The exact v1 replay and its errata are on branch
`archive/v1-replay`. This entry summarizes the kit against `v1.0.0`; the
commit history has the steps in between. Items marked **Breaking since
2.0.0a1** changed after the GTC Berlin lab's snapshot (below), and the
[2.0.0a1 to 2.0.0a2 migration guide](docs/migration/2.0.0a1-to-2.0.0a2.md)
maps each of them; `tfm migrate-config` rewrites a 2.0.0a1 config.

### Since 2.0.0a1

- New: `tfm migrate-config` rewrites a 2.0.0a1 config (such as a checkpoint's
  `config.yaml`) into this release's keys and checks that it loads; it
  refuses a setting this release cannot run. The GTC Berlin lab's Blueprint v2
  checkpoint config migrates to `benchmarks/ablations/blueprint_v2_final_d.yaml`.
- `recipes/blueprint_v1/config.cached.yaml` is `config.yaml` with
  `data.token_cache` (2.0.0a1's `tabformer_hf_decoder/config.gb10_cached.yaml`),
  and `config.yaml` trains with `training.backend: ddp`, as 2.0.0a1's
  `config.gb10.yaml` did.
- Flat configs whose pretraining windows are wider than their row, or whose
  rows are wider than the model's context, are refused at load.
- `notebooks/pragmatiq`: a new tour (00) and notebooks 01, 02, 03b, and 04
  adapted from the GTC Berlin lab's Blueprint v2 track. The series runs on the
  final-sweep checkpoint: `tfm.runtime.lab.LAB_CONFIG` is
  `benchmarks/ablations/blueprint_v2_final_d.yaml`, the checkpoint folder is
  `blueprint_v2_lab_tuned`, and `fetch_lab_assets` migrates a 2.0.0a1
  checkpoint's `config.yaml`. Notebook 05 tells the final sweep.
- `notebooks/blueprint_v1/04` measures the embedding space it plots
  (silhouette and distance ratios by customer, in the full space and in 3-D).

### Recipes and entry points

- Three reference architectures, each run with
  `tfm <stage> --recipe <slug> [--config <variant|path>]`:
  - `blueprint_v1`: NVIDIA's TFM Blueprint on TabFormer, with v1's label
    shift fixed and each transaction scored with the user's recent history
    (Blueprint v1 guide).
  - `pragma_like`: a PRAGMA-style profile, event, and history encoder stack
    (experimental composition guide).
  - `blueprint_v2`: PRAGMA's encoders under a causal history decoder (Blueprint v2 guide).
- `pragma_like/config.yaml` is matched to `blueprint_v2/config.yaml` on
  everything but the architecture, and a unit test keeps it that way.
- `train`, `infer`, `evaluate`, and `report` each run the stages before
  them. `smoke` runs `config.smoke.yaml` on CPU, and `--reuse-checkpoint`
  re-evaluates a trained run.
- Slot-by-slot ablations live in `benchmarks/ablations/`. Each recipe has one
  notebook series per dataset: `notebooks/blueprint_v1` and
  `notebooks/pragmatiq`.
- **Breaking since 2.0.0a1:** recipes `tabformer_hf_decoder`,
  `profile_history_kvt`, `blueprint_v2_hybrid`, `decoder_fraud_baseline`, and
  `building_block_comparison`, and their per-recipe scripts, are replaced as
  the migration guide lists.

### Composition and contracts

- `TrxFM(...)` with the core exports `Flat`, `Decoder`, `NoPE`,
  `EagerAttention`, `LastTokenPool`, `NextEvent`, and `Binary`. The
  hierarchical stacks are in `tfm.experimental`.
- Five building-block slots (building-block contracts). Validation refuses the pairings that
  would leak or cannot learn, before any data is prepared.
- Event, entity, and sequence labels are distinct types. Profiles and labels
  are cut at each row's cutoff, and the fraud probe is entity-disjoint, with
  per-side label windows (probe split rules).
- New `tfm.api.register_backbone` adds a `model.backbone` name for the flat
  stack. Built in: `eager`, `huggingface` (Llama, which requires
  `position: rope`), and `gru`.
- New experimental `flat_encoder` granularity (flat-encoder guide, `tfm.experimental`):
  `FlatEncoder`, a bidirectional encoder on `transaction_field_tokenizer`
  rows, trained by the new `masked_fields` objective (`MaskedFields`) and read
  out by `last_event_mean` (`LastEventMeanPool`). The field tokenizer reports
  its field id ranges as a `FieldVocabulary`.
- `pragma_like` and `blueprint_v2` share their event and profile encoders
  (`tfm.layers.kvt_encoders`). Older `pragma_like` checkpoints still load.
- `TrxFM.to_config` reports the declared position, the feed-forward width, and
  the head objectives.
- **Breaking since 2.0.0a1:** `tfm.data.collate.sequence_fraud_label` is
  now `sequence_label(..., label="fraud")`. `collate_batch` takes `label` and
  `amount_key`.
- Surprise scoring (`tfm.evaluation.surprise`) accepts any model that
  implements `NextEventScorer`, not just one class.

### Data and tokens

- Adapters for generated data, TabFormer (`tfm prepare-tabformer`), and
  pragmatiq books (dataset strategy). Each dataset has one role, and results from
  different datasets are never combined.
- Spliced identity-theft fraud on pragmatiq (`data.fraud_labels:
  spliced_identity_theft`), because a single event identifies the book's own
  fraud labels.
- Tokenizers:
  - `transaction_field_tokenizer`, with `features.fields`;
  - the `key_value_time` tokenizer, with `categorical_keys` and `hashed_keys`;
  - the `flat_key_value` and `event_fields` schemes;
  - `rapids_field_tokenizer`, an optional cuDF extra.
- `tfm cache-tokens` writes a pragmatiq book's tokens to disk once (token-cache guide).
- `features.exclude_keys` (`key_value_time` only) drops identity keys, such as
  TabFormer's `user`, from fitting and encoding. Tokenizer states and token
  caches without it load unchanged.

### Training

- Training backends `eager`, `ddp`, and `fsdp2` behind one contract suite
  (backend contracts). Under `torchrun`, `training.batch_size` is per rank, and only
  rank 0 evaluates and writes artifacts.
- `training.precision: bf16` and sequence packing are available, along with
  these optimizer options: Muon, cosine schedules, and gradient clipping.
- LoRA and full fine-tuning, with a linear or joint-fusion head
  (`fraud_task.finetune`).
- **Breaking since 2.0.0a1:** `training.backend: nemo` is removed; use
  `fsdp2`.

### Evaluation

- One headline per task (evaluation protocol): XGBoost on the raw row with and without
  the frozen embedding, with hand-built `tabular_xgboost` as the reference.
  The model card opens with this table. One constructor,
  `xgboost_classifier`, builds every XGBoost model, on CPU or with
  `device: cuda`.
- The logistic `raw` / `embedding` / `combined` stack runs only when the
  XGBoost arms are off, which is the CPU smoke path. `single_event_raw` is
  dropped, because it repeated the history row's raw arm.
- History-window fraud probe, with a single-event control. Next-event
  surprise is an extra readout (`evaluation.surprise`).
- `evaluation.history_window.max_rows: null` scores every evaluation-side
  row. `blueprint_v1/config.yaml` uses it and one fixed XGBoost setting, as
  `blueprint_v2` and `pragma_like` already did.
- **Breaking since 2.0.0a1:**
  `tfm.evaluation.fraud.baselines.evaluate_fraud_baselines` and its helpers
  are removed. Use `evaluate_binary_probes` or the runtime evaluator.

### Configuration

- Every key changes behavior or is rejected. A removed key fails with an error
  that names its replacement.
- **Breaking since 2.0.0a1:** optional knobs are grouped into sub-sections:
  `data.pretrain_windows`, `training.optimizer.muon`,
  `evaluation.history_window`, and `fraud_task.supervised_head`. Also, these
  keys are removed: `data.pretrain_window_stride`,
  `training.packing_seq_length`, and joint fusion's `cross_layers` and
  `plr_frequencies`.

### Records and documentation

- Records live under `benchmarks/results/`, with an index that maps the
  recipe names in older records to the current ones; superseded records are
  in `benchmarks/results/archive/`. The roadmap is a checklist per milestone.
- Three-seed records for both milestones: `blueprint_v1_seeds` on TabFormer,
  and `blueprint_v2_seeds` for `blueprint_v2` beside `pragma_like` on
  pragmatiq.
- `pragma_like_tabformer_seed42` scores PRAGMA-like's encoder beside v1's
  decoder on the same TabFormer rows
  (`benchmarks/ablations/pragma_like_tabformer.yaml`).
  `flat_encoder_tabformer_seed42` scores flat-encoder guide's flat BERT encoder on v1's
  own tokens and the same rows
  (`benchmarks/ablations/flat_encoder_tabformer.yaml`).
- `blueprint_v2_final_sweep` tries `blueprint_v2` at 1,500 or 3,000
  pretraining steps, by LoRA rank 8 or 16, at three seeds, against a selection
  rule fixed in advance. The recipe config is unchanged; the winner, 3,000
  steps with rank 16, is `benchmarks/ablations/blueprint_v2_final_d.yaml`,
  whose seed 42 is the GTC Berlin lab's checkpoint.
  `benchmarks/blueprint_v2_final_sweep.sh` reruns the sweep and
  `benchmarks/blueprint_v2_final_sweep.py` scores it.
- A documentation site (`make docs`) renders the concepts, the contract guides,
  the notebooks, the results, and the API reference.

## 2.0.0a1 — 2026-10-15

The SDK the GTC Berlin 2026 training lab (`x-fx-141-v1`, `main` at `944882d`)
vendors and installs at `/opt/tfm`: `v2` at `9ed45db`, the TabFormer token
cache, and the lab's edits below. Its full entry, against `v1.0.0`, is this
file at tag `v2.0.0a1`. Recipe names and config keys in this section are
2.0.0a1's.

### Changes from the GTC Berlin lab

- New: `training.log_every` (default `None`, silent) prints step, epoch, loss,
  learning rate, gradient norm, token throughput, and time left every that many
  steps, from the first rank only. Every run now writes its per-step curve to
  `train_log.json`, whatever `log_every` is. The eager, DDP, and FSDP2 trainers
  feed the same `tfm.training.progress.TrainingProgress`; FSDP2 does not clip,
  so it reports no gradient norm. The TabFormer GB10 configs set
  `log_every: 50`.
- New: recipe jobs print each stage (setup, train, infer, evaluate) with the
  time since the job started, from the first rank only.
- New: `tfm.evaluation.representation.project(vectors, dims=2|3)` projects
  embeddings to two or three dimensions; `project_2d` is its `dims=2` case.
- New: `TrxFM.to_config()` reports an optional backbone's own `describe()`
  (the HuggingFace Llama backbone reports its depth, heads, KV heads, and
  sizes) instead of the unbuilt `Decoder` spec, or `None` without one.
  `cost_from_composition` refuses a composition that does not run the eager
  `DecoderModule`.
- `embeddings_manifest.json` names the arrays of `embeddings.npz`.
- New, internal and untested beyond the cost guard: `tfm.api.resolve` (slot
  names to components) and `tfm.diagnose.cost` (parameter and FLOP estimates
  from `TrxFM.to_config()`).

### TabFormer token cache and data-parallel training

- New: `tfm cache-tokens` also builds a flat-field token cache of TabFormer's
  train split (`transaction_field_tokenizer`, precomputed splits,
  `data.pretrain_window_events`). With `data.token_cache` set, pretraining
  windows are read from disk, and `train.py` builds no event objects on any
  rank. On all of TabFormer the cache builds once in about 75 s (8 workers,
  7.5 GB peak) against about 9 min and 48 GB per run, per rank, for the object
  path. `evaluate.py` and `report.py` still load events for their probes. The
  cached windows equal the object path's tensor for tensor (token-cache guide).
- New: `setup_run(..., pretraining_only=True)` and
  `tfm.runtime.prepare_pretraining`; the `train` command uses them.
- Fix: `training.backend: ddp` failed on its second step with two or more
  ranks whenever the model carried a task head that pretraining does not train
  (such as `heads: fraud`). DDP now sees only the loss, so untrained heads are
  skipped.
- `tabformer_hf_decoder/config.gb10.yaml` trains with `training.backend: ddp`:
  unchanged on one GPU, data-parallel under `torchrun --nproc-per-node N`.
