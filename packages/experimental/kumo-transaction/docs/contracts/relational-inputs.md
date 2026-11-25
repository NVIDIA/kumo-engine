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
are explicit. Default historical ownership is evaluated at the terminal event's
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
contract. Python remains the default; native capability and performance gates
are separate.

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
The native graph implementation builds at most two availability/window-filtered
snapshots, reopens their completed index for one-hop directional plans, and
retrieves all eligible candidates before canonical recent fanout. These physical
reads are larger than logical fanout where degree exceeds the cap. It is a
bounded correctness integration, not a full-book streaming or speed claim.

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
