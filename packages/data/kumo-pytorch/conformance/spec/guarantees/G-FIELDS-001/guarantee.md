# G-FIELDS-001: Ordered explicit field outputs

## Rule

Fields contains an ordered sequence of Encode declarations. Columns keeps
different fields in separate typed arrays; Tokens concatenates each row's
encoded pieces in declaration order. One input cannot encode the same resolved
sampled value twice, including overlapping qualified/unqualified aliases.
Different named inputs may encode the same source differently.

## Boundaries

Token rows require ID-producing encoders and one explicit final dictionary;
Raw integers are not implicitly token IDs. Mixed column types are never merged
into a common floating tensor. NULL validity differs from valid zero/false,
empty text, an event with zero pieces, and padding. Piece positions retain
their original within-value position after clipping. Optional field IDs are
parallel metadata and do not change values. With no field-ID request, no field
dictionary is inferred. Forbidden selected fields fail validation even through
an alias or named encoding. Repeated scalar occurrences retain their own
boundaries and order; scalar-only composite/calculation use refuses them.

## Scope

Grouping and storage follow G-HISTORY-001. Source-internal timestamps and
calculation prerequisites are not automatically emitted. Calendar, clock and
age values appear only through explicit ordinary field outputs. Graph node
attributes and candidates use the same scalar encoders with their supported
column layouts; this is not a promise of token graph/candidate output.

DatedEntityValue may declare additional typed occurrence fields, such as an
anchor-relative EventTimeDelta. OccurrencesOf references an entity token input
and returns explicitly selected Raw columns aligned to its retained positions,
including repeated text pieces. Static entity positions have no dated
occurrence and therefore have invalid occurrence-field values. No timestamp or
age is inferred merely because dated values were selected. The declared
occurrence order distinguishes field-major output from chronological dated
output; it never changes the field encoders' identity or fitted mappings.

## Golden examples

- [C-FIELDS-001](golden/C-FIELDS-001.json): ordered category and raw-number
  columns retain separate exact types and NULL presence.
- [C-FIELDS-002](golden/C-FIELDS-002.json): explicit dated occurrence ages
  repeat across text pieces, follow first-per-value and cutoff selection, and
  remain invalid for static profile and padded positions.
