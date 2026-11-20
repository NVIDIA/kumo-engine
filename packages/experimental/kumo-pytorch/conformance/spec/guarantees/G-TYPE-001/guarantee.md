# G-TYPE-001: Stored types and explicit encodings

## Rule

Every resolved table column has one scalar DType. During authoring, omitted
column types are inferred from Parquet metadata and must agree across files.
Explicit types require compatible physical types. A table may mix explicit
types and inferred entries. Omitting its columns discovers all source columns.
`InMemoryRows`, which serving and small in-process examples use, requires
explicit types because Python rows provide no authoritative type metadata.
Field reads and Raw encoding preserve the resolved type.
Vocabulary accepts any supported scalar type, Text accepts string only, and
ZScore/Buckets accept signed or unsigned integers, floats, and decimals only.
Booleans, dates, timestamps, bytes, and numeric-looking strings are not numbers
for these compatibility checks. A calculated field uses its resolved result
type, not the type of its original source column.

## Boundaries

NULL remains absent with explicit validity; zero and false are real values.
Integer width and signedness, decimal precision/scale, and timestamp units are
part of the declaration. For a declared `int64`, DiskGraph accepts narrower
signed integers and scale-zero `decimal128` or `decimal256` from Parquet.
Native ingest rejects decimal values outside the signed 64-bit range.
Under `Plan.materialize` on DiskGraph, Parquet anchor entity ids for an `int64`
primary key accept the same types, are read as `int64`, and fail on values
outside that range.
Stored timestamps are whole-second UTC instants. Finer physical timestamps
round toward positive infinity to the next whole second, including negative
epochs. For example, -1 microsecond becomes second 0 and +1 microsecond becomes
second 1. Incompatible physical types fail when binding metadata is available.
Model output refuses unsupported raw tensor types before fetching the batch.
Inspectable output may retain typed non-tensor columns. Array-valued cells
are outside this scalar contract.

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
