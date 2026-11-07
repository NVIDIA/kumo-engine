# G-ROWS-001: Select related rows through each request's cutoff

## Rule

For a root entity and a single `Rows` relation, selection retains the
distinct destination rows whose relation key equals the root's relation key
and whose `at` time is at or before that sample's `as_of_time`.

A historical `TimeWindow` uses `start < time <= stop`, with bounds measured
in seconds from that sample's `as_of_time`. A missing start leaves history
unbounded below. The examples use `stop=0`, so their upper bound is the cutoff.

Matching is by the declared relation keys. The meaning does not change merely
because neither of those keys is a table's primary key. An engine that cannot
execute a relation refuses it clearly at `check()` rather than select different
rows. The DiskGraph engine executes a relation only when exactly one of its two
keys is its own table's primary key, and refuses any other.

## Boundaries

A row exactly at the cutoff is included; a future row is excluded. A row
exactly at a window's start is excluded. A sample with no eligible matching
rows has an empty selection; its count is zero. Each request uses its own cutoff, even
when requests for the same entity are passed together.

In the examples, user 1 has events at 2, 9, 10, and 12. Its counts at cutoffs 9,
10, and 12 are respectively 2, 3, and 4. User 2's only event is at 11, so its
count at 10 is zero. These answers are calculated from the listed rows, not
copied from an engine's output.

C-ROWS-001 also counts the window `(cutoff - 1, cutoff]`. User 1's recent
counts at 9, 10, and 12 are each one. At 11 its all-history count is three,
but its recent count is zero: the event at 10 lies exactly at the lower bound.

## Scope

This rule covers one relation, unique primary keys,
non-null integer-second times, explicit `TimeWindow(column="at", stop=0)`
with an optional start, and distinct rows. The golden examples observe this
selection through raw counts. The tests separately establish what
the DiskGraph engine supports. This does not claim that all of this package's
time behavior is supported.

Multi-hop availability, null clocks, clock overrides, arrival time, graph
sampling, and future targets are outside this rule. In particular, this does
not settle per-hop filtering of timed intermediate tables.
The window convention matches DiskGraph's temporal graph sampler and
training-table traversal. Sequences, aggregates, and length queries must use
the same boundaries.

## Golden examples

- [C-ROWS-001](golden/C-ROWS-001.json): a root-primary-key to event-foreign-key
  relation with all-history and bounded counts, including both equality edges.
- [C-ROWS-002](golden/C-ROWS-002.json): equal group keys, with neither relation
  key being its table's primary key, with the same intended counts.

The assertions for the DiskGraph engine live in
[expectations.json](../../../../conformance/expectations.json), executed by
[test_guarantees.py](../../../../tests/conformance/test_guarantees.py). The
golden files supply the answer; these ordinary tests require an answer or a
specific refusal. They never ask the engine to choose what counts as passing.
The DiskGraph engine's specific support or refusal is stated by those
expectations, not by the row-selection rule.
