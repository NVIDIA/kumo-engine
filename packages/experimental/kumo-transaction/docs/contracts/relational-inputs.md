# Shared relational inputs

The bounded relational adapter declaration supplies keyed tables, foreign-key
roles, explicit event and observation clocks, and feature allowlists. Composite
keys remain join metadata. This is an experimental bounded data contract. The
[worked recipe](../../recipes/relational_inputs/README.md) runs sequence, graph
and hybrid variants from the same declaration.

A task chooses a root table and named history paths. A path is an ordered list
of relationship roles; each step must connect to the preceding table. Either FK
direction is allowed. Repeated events reached through several paths appear once,
identified by terminal table and primary key. Histories sort by event time then
canonical event identity, and retain the latest configured number of events.

## Time and ownership

Every timed row has an event clock and an observation clock. Untimed rows must
be explicitly immutable. A history may include a terminal event only when the
root, every intermediate row, every traversed link and that terminal row were
observable by prediction cutoff. Recent-neighbor selection happens after these
availability checks. Observation-window bounds are inclusive.

Versioned ownership declares two half-open intervals: effective ownership
`[valid_from, valid_until)` and knowledge `[known_from, known_until)`. Open ends
are explicit; both interval starts must be datetimes. Default historical ownership is evaluated at the terminal event's
time using only relationship versions known at prediction cutoff. Explicit
`prediction_time` ownership evaluates effective intervals at prediction cutoff.
Overlapping effective and knowledge intervals for the same child and role are
ambiguous and rejected, including duplicate version facts. A fixed projection
must preserve the observation time of every row and link along its paths.
Unsupported projection rules fail with the affected path before training.

## Anchors and labels

Anchors bind sample ID, root key and input cutoff. Recipes may generate one per
visible root and configured cutoff, generate them from a label table's prediction
timestamps, or replay an explicit anchor table. Repeated roots are independent
samples. An unavailable root is an error for explicit anchors.

The relational recipe runner binds checkpoint identity to the complete ordered
anchor table, including prediction rows. Checkpoint reuse requires that same
table. Custom prediction anchors are supported only in a new run, which fits
again; loading trained weights against a different prediction-anchor table is
not supported by this runner. Schema, ordered feature allowlists and encoder
identity are checked independently and remain required.

Tables with `role="labels"` are excluded from input projections, including in
pretraining without a task label configuration. Task-table binary labels are
**entity-level** labels at the anchor cutoff. They
are not event fraud labels or the existing truncated-history sequence label.
Their horizon, delay, observation time and split fitting cutoff are explicit;
labels unavailable at the fitting cutoff cannot train or score a task. Label
tables never enter input graphs or histories. All feature kinds reject prohibited
label-like names before fitting any tokenizer, including aliased source columns.

Graph fanout is per frontier node and directed relationship role at each hop.
Both FK directions expand from the table types actually reached. Equal event
times break by canonical primary-key identity ascending. Shared batching requires
identical cutoffs and windows, while disjoint batching preserves independent
sample components. Hybrid views retain complete anchor identities and ordering.

Backend parity tests must compare actual node/edge identities and ordered history
identities, including six hops with fanout ten, delayed availability, ties,
multiple paths and repeated roots. Native support is not implied by the data
contract. Kumo Engine is the recipe default; Python remains explicitly selectable.
Backend capability and workload performance are separate concerns.

## Binary batch compatibility

GraphSupervisedBatch, GraphHistoryBatch and SequenceSupervisedBatch now carry
an explicit `target_level` (`sequence` by default for existing callers, or
`entity` for task-table labels). Binary models reject a mismatched level before
encoder execution. History-only projection preserves it. Existing sequence
fraud rules are unchanged. `SequencePredictionBatch` carries guarded unlabeled
history and anchors; it supports inference without manufacturing zero labels.

The graph projection currently supports fixed FKs and numeric allowlists with
finite-value masks. It rejects versioned topology and categorical/text graph
encoding explicitly; point-in-time history traversal supports versioned ownership.
The canonical native runner indexes validated source tables and streams encoded
tensors from native tasks. It does not implement the whole reference contract.
Whole-second clocks, fixed ownership and backend-expressible observation rules
are required. A capped canonical tie, independent history availability clock,
unsupported timed projection, versioned ownership rule or meaningful per-hop
lookback fails before training rather than changing the contract. See the
[native capability and migration guide](../migration/native-loader-default.md#supported-boundaries).
Legacy object-based candidate adapters remain available to direct callers;
they are not the source-native recipe implementation.

Structural tables reached along history paths are observation-window exempt:
they retain their original creation/availability clocks even when created before
the activity window. Terminal activity tables are windowed. GraphTable.windowed
is explicit and defaults to the legacy windowed behavior. GraphBatch records
`window_exempt_types`; it still checks every node/link against the upper cutoff.
The default empty set preserves existing lower-bound rejection. This lets an
old but observable intermediate entity connect recent activity without changing
its timestamp or declaring a timed row immutable.

Root allowlists project into the existing profile-token input, using only roots
observable at each requested cutoff. Tokenizer fitting sees training profiles
only. Non-finite numeric profile values are missing; root keys/clocks never
become attributes. Intermediate context attributes remain graph features.
Canonical identity ordering is lexicographic on the typed JSON encoding,
including composite keys; it is not numeric sorting of raw integer keys.

Declared FK columns must exist on every child row. An explicit `None` denotes
an optional link; omitting the column is a schema error. A version log replaces
the whole role and must cover every child with a non-null fixed binding.

A root/cutoff pair cannot belong to different splits, whether anchors come from
labels, configured cutoffs or replay. Repeated prediction instances within one
split and the same root at different cutoffs remain supported.

Graph window exemptions apply to the root identity and strict intermediate
tables reached by history paths. A history terminal remains windowed even if
another path also traverses its table; the root identity is always exempt from
the lower bound. Timed tables not reached by any history path are windowed.
Every row and link still obeys the upper availability cutoff.

Inference exports one label position per sample ID. Unknown labels are NaN and
`<split>_labels_valid` marks which entries are known; unknown never means zero.
Managed checkpoint and encoder manifests record ordered allowlists explicitly,
so same-width feature reordering cannot silently reuse incompatible weights.

Relational recipes reject nonempty `objective.auxiliary`: these stages apply
only their declared primary/binary objectives. Sequence/hybrid decoder widths
and head counts must be positive and produce an even integral head dimension;
configuration validation rejects incompatible dimensions before data preparation.


## File-backed adapters and native encoding

`TableSource` supplies the same keyed-table, relationship, clock and allowlist
metadata over local scalar Parquet files. Register its factory with the data
adapter's `table_source` hook. Native recipe runners call that hook instead of
loading an object corpus. A missing hook is an unsupported capability, not a
reason to fall back silently.

Source keys are normalized through a disk-backed identity registry, including
composite keys. All declared rows, keys, FK bindings and clocks are validated
before native feature fitting. Labels stay outside the model input schema.
Only whole-second UTC native clocks are currently accepted. A source declaration
can describe broader rules; the backend must reject rules it cannot express.

Native feature encoders fit selected training anchors, not all source rows
before a date. Static root attributes and separately dated profile facts have
distinct inputs. Trained weights bind source-content hashes, ordered schema,
replay anchors, native contract and fitted-encoding fingerprints. Embedded
checkpoint metadata is authoritative; sidecars cannot authorize different
weights or encodings.

Continuous native iterators own shuffling by seed and epoch. Temporal stages
keep separate iterator tails so shared graphs never combine different cutoffs.
A CPU producer adapts delivered tensors into owned batches and prefetches one
batch, closing before source/index teardown even after an error or early stop.
Model batching does not reconstruct events or tokenize Python histories.
FutureSubgraph target vectors and optional Python reference-baseline features
are separate from this historical model-input path.
