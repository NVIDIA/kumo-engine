# diskgraph-infer

`diskgraph-infer` scans CSV/Parquet tables and produces a `source_metadata.json` that `diskgraph-ingest` can consume. It infers column types, primary keys, foreign keys, edge tables, and optional edge sort keys using a mix of schema inspection and column-name heuristics.

## Inputs and layout

The input root can be a local directory or a cloud URL:

- Local: `/path/to/dataset`
- S3: `s3://bucket/prefix`
- GCS: `gs://bucket/prefix` (or `gcs://`)
- ADLS: `az://account/container/prefix`

Tables can be provided as:

- Single files: `Users.parquet`, `Orders.csv`
- Per-table directories: `Users/part-00000.parquet`, `Orders/part-00000.csv`

Files are grouped by table name (filename or top-level directory). Only `.csv` and `.parquet` files are considered.

## Output

Writes `source_metadata.json` with:

- `tables[]` + inferred `columns[]`, `format`, `paths`
- inferred `primary_key`, `create_time`
- inferred `relationships[]` (foreign keys)
- inferred `edge_table` specs when a table looks like a pure join table
- optional `edge_sort_keys` when timestamp/weight columns exist
- `num_shards` based on dataset size

The output schema is the same one documented in `diskgraph-ingest/README.md`.

## Inference behavior (high level)

- **Primary keys**: chooses key-like columns (`id`, `<table>_id`, UUIDs, IPs, etc.) with table-name and token matching heuristics. Ambiguous ties are left unset.
- **Create time**: prefers timestamp/date columns with strong name hints (`created`, `start`, etc.) and valid value ranges.
- **Foreign keys**: matches key-like columns to other tables' primary keys using name/token overlap and type compatibility. Skips ambiguous ties.
- **Join/edge tables**: if a table has exactly two FK columns and any remaining columns are time/weight fields, it is marked as an explicit edge table and its FK relationships are pruned from the implicit relationship list.
- **Edge sort keys**: if a destination (or edge) table has a timestamp and/or weight column, a sort key spec is emitted with a fixed timestamp encoding (int16 days since UTC midnight).
- **Shards**: `num_shards` is estimated from total dataset bytes and available system memory (Linux only); otherwise a 1 GiB target shard size is used.

## Schema inference details

### Parquet

Parquet schemas are read via Arrow. Supported types map directly to `ingested_dtype`:

- **Int/UInt**: `Int8/16/32/64`, `UInt8/16/32/64`
- **Float**: `Float32`, `Float64`
- **Bool**: `Boolean`
- **String/Bytes**: `Utf8`, `Binary`
- **Date**: `Date32`
- **Timestamp**: `Timestamp` (second/milli/micro/nano, timezone normalized to UTC)
- **Decimal**: `Decimal128`, `Decimal256`
- **Arrays**: `List(Float16|Float32|Float64)`

String columns sample up to `sample_rows` values to infer fixed-length strings and set `fixed_len` when all values are the same length.

### CSV

CSV parsing samples `csv_sample_rows` rows (default 2,000), and for remote sources reads up to `csv_sample_bytes` (default 256 KiB).

Heuristics for CSV columns:

- **bool**: values in `true/false/1/0/yes/no`
- **timestamp/date**: recognizes common formats (`YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, RFC3339 variants)
- **integer timestamps**: if the column name looks time-like and values fall into epoch ranges
- **uuid/uuid256/ip**: detected from string shapes
- **fixed_len strings**: when all sampled values have equal length

If none of the above match, columns default to `string` (or `float32`/`int64` when numeric parsing is consistent).

## CLI usage

```bash
cargo run -p diskgraph-infer --bin infer_source --release -- \
  --input /mnt/localssd/datasets/hnm \
  --output /mnt/localssd/datasets/hnm/source_metadata.json
```

Options:

- `--input`: root directory or cloud URL.
- `--output`: output path for `source_metadata.json`.
- `--validate_all`: validate schema consistency across all files in a table (slower).

Wrapper script:

```bash
./diskgraph-infer/scripts/infer_source.sh /mnt/localssd/datasets/hnm
```

## Warnings and diagnostics

When `--validate_all` is enabled, the tool will record schema mismatches and list ignored files. Notes are also emitted when foreign key candidates are skipped due to type incompatibility or ambiguity.

The CLI prints a compact summary of tables, relationships, edge tables, and any warnings.
