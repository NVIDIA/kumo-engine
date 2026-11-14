# G-ENTITY-001: the schema primary key identifies every requested entity

## Rule

`TableSpec.primary_key` is the single definition of external entity identity. An
`EntitySpec` selects a table, not a second lookup column. Every sample request's
`entity_id` means that table's primary-key value.
Candidate IDs use the primary key of their own candidate table. Graph roots and
graph-feature fitting use the same identity as ordinary sample construction.

The ordinary declaration is `EntitySpec(table="users")`. Its output type name
defaults to the table name. An explicit type name, filters, and availability
rules remain optional keyword settings; none changes the lookup column.

## Boundaries

An alternate entity-key argument is not supported in Python, serialized entity
declarations, or the sample builder. It must be rejected, not ignored. There is
no fallback or migration that reinterprets an alternate-column request.

The schema must declare the entity table and a nonempty primary key included
in that table's columns. This is declaration validation, not proof that actual
rows have unique, non-null keys. Source-data key validation is separate work.
Non-entity tables may be keyless under [G-KEYLESS-001](../G-KEYLESS-001/guarantee.md).
An undeclared entity table fails before the engine opens its data.

Relationship traversal still uses the declared relation columns, which can be
foreign keys. Anchor-table ID mappings still select the input column holding
request IDs. These are not overrides of entity identity. Time cutoffs, filters,
availability rules, and DiskGraph's support limits are unaffected.

## Scope

Callers holding another identifier must translate it to the declared primary
key before submitting a request.

The golden below checks the primary-key rule on the DiskGraph engine through
the public API, including JSON saving/loading and dictionary permutations.
Focused tests in
[test_entity_identity.py](../../../../tests/test_entity_identity.py) also cover
candidate and graph identity, optimized DiskGraph output, rejected entity-key
overrides, optional entity settings, validation, and artifact contracts. The raw-value
golden checker does not by itself certify candidate or graph layouts.

## Golden examples

- [C-ENTITY-001](golden/C-ENTITY-001.json): `users` is keyed by `pk` and has
  rows `(pk=1, external=2)` and `(pk=2, external=1)`. Request 1 must return
  `external=2`; request 2 must return `external=1`. Crossed identifiers expose
  lookup against the wrong column. These answers follow directly from the
  schema and rows, not from engine output.
