# G-TYPE-001: Stored types and explicit encodings

## Rule

Every resolved table column has one scalar DType. During authoring, omitted
column types are inferred from Parquet metadata. Explicit types must match
that metadata. A table may mix explicit types and inferred entries; omitting
its columns discovers all source columns. `InMemoryRows`, the Python rows that
serving and small in-process examples supply, requires explicit types because
Python rows provide no authoritative type metadata. Field reads retain that
resolved type; Raw preserves it rather than converting values to fit a model.
Vocabulary accepts any supported scalar type, Text accepts string only, and
ZScore/Buckets accept signed or unsigned integers, floats, and decimals only.
Booleans, dates, timestamps, bytes, and numeric-looking strings are not numbers
for these compatibility checks. A calculated field uses its resolved result
type, not the type of its original source column.

## Boundaries

NULL remains absent with explicit validity; zero and false are real values.
Integer width and signedness, decimal precision/scale, and timestamp units are
part of the declaration. Stored timestamps are whole-second UTC instants;
finer physical timestamps round toward positive infinity to the next whole
second, including negative epochs. For example, -1 microsecond becomes second 0
and +1 microsecond becomes second 1. Physical type mismatches fail when binding metadata
is available, not by parsing or silently casting values. Unsupported raw tensor
types refuse model output before fetching the requested batch; inspectable
output may retain typed non-tensor columns. Array-valued cells are outside this
scalar contract.

## Scope

This rule covers the supported scalar adapter types, not every native DiskGraph
type. Declaration incompatibilities are checked without source-row reads;
physical metadata and actual invalid values can only be checked at binding or
execution respectively. Types are resolved before the finalized contract is
fingerprinted or saved. The saved contract contains concrete types; rebinding
it checks source types rather than inferring over its declarations. Encodings
remain explicit. The golden checks exact numerical and boolean buffers;
parameterized type/encoder tests cover the compatibility matrix and typed
fixture tests cover empty/all-NULL physical schemas. Authoring tests cover
metadata inference, explicit/inferred mixtures, and saved-contract type drift.

## Golden examples

- [C-TYPE-001](golden/C-TYPE-001.json): signed integers, unsigned integers,
  booleans, floating zero, and NULL retain their declared types and validity.
- [C-TYPE-002](golden/C-TYPE-002.json): fractional physical timestamps on both
  sides of the Unix epoch normalize to the same whole-second UTC values.
