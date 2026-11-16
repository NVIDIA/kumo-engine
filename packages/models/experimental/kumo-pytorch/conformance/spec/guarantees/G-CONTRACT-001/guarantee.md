# G-CONTRACT-001: named definitions are unordered; explicit sequences retain order

## Rule

For a valid `DataLoaderSpec`, saving with `to_json()` and loading with
`DataLoaderSpec.from_json()` preserves its declarations, values, types, and
explicit sequences. The contract remains valid, keeps both fingerprints, and
produces the same answers.

Named definitions are addressed by name. Rearranging JSON object entries does
not change validity, calculated answers, fitted mappings, or fingerprints.
Derived entity/event/candidate values resolve prerequisites by name within their own
group; declarations need not precede their uses. Unknown references and circular
dependencies fail validation before the engine opens.

Where positions matter, the API uses an explicit sequence. Arithmetic operands,
Encode field selections and composite components retain their declared order.
Changing those sequences is not an equivalent
contract. Composite components are never silently alphabetized.

## Boundaries

kumo-pytorch computes fingerprints from its own standardized representation of the
reconstructed contract: sorted object keys, fixed formatting, and unchanged array
positions. Named table/relation records are canonicalized by name, and supplied
vocabulary entry records by their explicit IDs; their record placement is not
an output position. DType parameters, references, encoding settings, destinations,
reserved roles and layouts participate in identity. Incoming JSON key placement
and indentation are irrelevant. Sorting
keys for hashing does not define calculation or token order. This is not a
promise to recognize every mathematically equivalent expression or every
alternative numeric/text representation.

Field order applies before token truncation. Entity tokens use the declared
occurrence order: `"fields"` keeps field declaration order and each field's
occurrence order; `"chronological"` places undated values first, then dated
values chronologically, with field rank breaking equal-time ties. Events retain
their selected time order and use explicit field order within each event.
Text-piece order is preserved.

This preserves contract instructions, not source-data contents or separately
learned state. It adds no cross-group calculation references or new DiskGraph
operations. A declared value with missing data still follows its existing missing
result policy; an undeclared reference is a validation error, not missing data.

## Scope

Every golden contract is saved and reloaded with the public methods before
engine checks. The shared loader independently compares values and explicit
sequences, checks full and input-only fingerprints, and returns the loaded
contract. Required engine assertions execute original, reversed-key, and
fixed-seed shuffled-key forms against the same hand-worked answer or refusal.
Arrays are never shuffled. The direct-field example below executes on the
DiskGraph engine. Typed declarations use contract version 3; unsupported
versions are rejected explicitly rather than interpreted using guessed types or
defaults.

Focused tests in [test_contract_serialization.py](../../../../tests/test_contract_serialization.py)
also cover multi-step entity/candidate calculations, invalid dependencies,
operand order, state reuse, JSON formatting, and `Artifact.save()` files.
[test_token_field_order.py](../../../../tests/test_token_field_order.py) covers
explicit token positions, truncation, and composite keys, including DiskGraph's
column execution. G-FIELDS-001 and G-HISTORY-001 add encoded layout assertions;
the raw-value golden alone does not certify those layouts.

## Golden examples

- [C-CONTRACT-001](golden/C-CONTRACT-001.json): user 1 has `first=2` and
  `second=4`, so the named results are `z_first=2` and `a_second=4`, regardless
  of their dictionary placement. JSON saving/loading and dictionary permutations
  preserve those answers and both fingerprints.
