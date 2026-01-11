# diskgraph-rowpack

Rowpack is a fast, row-oriented compression codec for DiskGraph node payloads. It compresses each node independently while using a shared per-table context (built from a sample) to encode repeated values efficiently. Think of it as a compact, schema-aware alternative to messagepack where the decoder reuses column dictionaries and type hints from the context file.

## Design goals

- **Row-at-a-time:** each node encodes to a standalone binary blob.
- **Shared context:** small, per-table context file drives encoding/decoding.
- **Fast decode:** minimal branching, bit-packed tags, no per-row schema.
- **Graceful fallback:** unexpected values are tolerated without failure.
- **Deterministic schema:** decoding produces consistent types.

## Context file (what gets stored)

The context file is a binary format with:

- **Schema**: column order, names, and logical types (int/float/string/bytes/bool/timestamp/array).
- **Per-column strategy**:
  - Nullability and tag width (2-bit tags).
  - Integer strategy (fixed width vs varint; bit width).
  - Float strategy (float32 vs float64).
- String strategy (raw, dictionary, alphabet, bpe, lz4).
  - Bytes strategy (raw, fixed length).
  - Array strategy for float vectors (float16/float32/float64).
- **Dictionaries**:
  - String dictionaries for categorical columns (bounded size, configurable).
  - Hex encoding parameters (upper/lower case).
  - ISO timestamp parsing hints.
- **Compression knobs**:
  - Per-column thresholds used during sampling.
  - Max dictionary size budget.

The context is written once during ingestion (one per node table) and loaded by the index reader to decode node blobs.

## Row encoding format

Each row encodes columns in fixed schema order:

1. **Tag bitstream**: 2 bits per column.
   - `00`: null
   - `01`: primary encoding for the column
   - `10`: fallback encoding for the column (unexpected value)
2. **Boolean bitset** (for bool columns only).
3. **Column payloads**, in order:
   - Primary payloads for tags `01`
   - Fallback payloads for tags `10`

This layout keeps parsing simple: read tags, then parse each column according to tag state.

## Column strategies

## Supported Parquet types

Rowpack is driven by `source_metadata.json` and ingestion maps Parquet/Arrow types as:

- **Int**: `Int8`, `Int16`, `Int32`, `Int64`
- **UInt**: `UInt8`, `UInt16`, `UInt32`, `UInt64`
- **Float**: `Float32`, `Float64`
- **Bool**: `Boolean`
- **String**: `Utf8`
- **Bytes**: `Binary`
- **Date**: `Date32`
- **Timestamp**: `Timestamp` with `Second|Millisecond|Microsecond|Nanosecond` units (timezone accepted, normalized to UTC)
- **Decimal**: `Decimal128`, `Decimal256` (requires `precision` + `scale`)
- **Arrays**: `List(Float16)`, `List(Float32)`, and `List(Float64)`

Other Parquet types are not supported by rowpack at this time.

### Integers
- **Int8/UInt8**: always stored as a single byte (no varint).
- **Int16/UInt16**: stored as fixed 16-bit unless varint is smaller for samples.
- **Int32/Int64**: choose varint when it reduces average size.

Varint uses zigzag for signed values; fixed width uses little-endian.

### Floats
- Floats are stored as-is (float32 or float64).
- Fallback tag stores full float32/float64.

### Strings
Rowpack detects and chooses the most efficient strategy:

- **Dictionary**: for categorical columns; dictionary entries stored in context.
- **Alphabet**: for short strings with a small character set and high coverage (up to 64 distinct chars, and up to the lz4 long-string threshold, default 128 bytes); length is varint-prefixed.
- **BPE**: for strings with repeated substrings; uses a column dictionary of common byte sequences and encodes as token streams.
- **LZ4**: only used when sampled long strings compress well.
- **Raw**: length-prefixed bytes.

If a value does not match the inferred strategy (e.g., a non-ISO string in an ISO column), it is encoded with the fallback tag as raw bytes.

### Bytes
- Raw length-prefixed bytes.
- Optional fixed-length enforcement; mismatches fall back to raw.

### Arrays of floats
Encodes length as a varint, followed by contiguous float payloads.
Uses the logical type's float width (float16/float32/float64).

### Nulls and fallback
Nulls are stored only in the tag stream; no payload.
Fallback payloads are always self-describing (length + bytes, or full-width numeric).

## Sampling and strategy selection

The builder samples a configurable number of rows per table and:

- Computes cardinality for dictionary eligibility.
- Measures lz4 compression ratio for long strings.
- Trains a lightweight BPE dictionary for repeated substrings when it yields a size win (enabled for short-ish strings, default avg len >= 12 and below the lz4 threshold).
- Chooses integer varint vs fixed width based on average size.

Sample sizes and dictionary budgets are configurable via `SourceMetadata.rowpack`.

## Decoding outputs

Rowpack decodes into:

- **MessagePack**: preserves original scalar types (including timestamp ext).
- **JSON**: "reasonable" JSON renderings; timestamps formatted to RFC3339 and bytes encoded as base64.

## Error handling

Rowpack is resilient by design. Unsupported or unexpected values:
- are encoded with fallback tags when writing,
- decode to best-effort types when reading.

## Context inspection utility

Use the built-in inspector to print a `.ctx` file summary:

```bash
cargo run -p diskgraph-rowpack --bin rowpack_ctx_inspect -- /path/to/node_data.ctx
```

## BPE defaults

- `bpe_max_dict_entries`: 256
- `bpe_max_dict_bytes`: 1_000_000
- `bpe_max_samples`: 5_000

It only errors on structural corruption (invalid tags, truncated payloads).
### Timestamps
- Encoded as `(secs, nanos)` varints for full unit fidelity.
- Parquet timestamp units (s/ms/us/ns) are normalized at ingest; timezones are accepted and normalized to UTC on output.

### Decimals
- Stored as fixed-width two’s-complement bytes (16 for Decimal128, 32 for Decimal256) when lossless is required.
- By default, decimals use integer encoding when `scale == 0` and float encoding when `scale != 0` (lossy).
Decimal storage is always lossless.
