# G-HISTORY-001: Event boundaries and aligned outputs

## Rule

Grouped preserves event sequences and their sample membership. Flatten makes
one token sequence per sample, optionally placing explicitly requested reserved
markers and applying its token budget. Packed and padded are storage choices;
they do not change the selected values or fitted IDs. Samples never become one
logical cross-sample sequence merely because their storage buffer is shared.

## Boundaries

Grouped packed output retains event-token and sample-event boundaries; flat
packed output retains sample-token boundaries. Empty histories retain their
sample boundary. Zero-token grouped events remain real events. Marker positions
count toward the budget and have no original field/event owner. Before/after
markers reserve their budget first; clipping keeps the declared first/last end,
preferring complete event segments and clipping a single oversized segment
rather than dropping all content. Separators attach to their preceding event
segment under the declared between/after-each policy.

EventsOf references the retained rows of one direct grouped EventSource input.
A grouped Columns follower reads those original rows in the same order, before
masking, without independent row filtering, skipping or limits. It may read
declared fields the parent does not encode. Input map order does not affect
resolution. Unknown/self/cyclic/chain references, flattened/non-event parents,
and unsupported follower arrangements fail before source reads.

OccurrencesOf similarly follows the exact retained positions of an entity
token input. Explicit typed occurrence fields repeat for its emitted text
pieces; static entity positions have invalid occurrence fields. Truncation and
occurrence ordering follow the parent, not a second independent selection.
The independent dated/text/static boundary example is
[C-FIELDS-002](../G-FIELDS-001/golden/C-FIELDS-002.json).

## Scope

Internal clocks select/order rows but are not automatic model outputs.
Sequence contains only layout membership, offsets and validity. Calendar and
anchor-relative age values require explicit SampleSpec calculations and named
column inputs; layout never adds them. Latest-event-relative coordinates are a
model adapter calculation, not a loader flag. EventsOf covers event rows, not a
general dependency language or arbitrary join. Per-hop cutoff rules are
unchanged.

## Golden examples

- [C-HISTORY-001](golden/C-HISTORY-001.json): grouped and flat packed tokens
  share values but retain different event/sample boundaries.
- [C-HISTORY-002](golden/C-HISTORY-002.json): an explicit timing input follows
  retained events and emits anchor ages, including an empty sample.
