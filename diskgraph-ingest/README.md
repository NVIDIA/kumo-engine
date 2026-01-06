# diskgraph-ingest

`diskgraph-ingest` converts raw CSV/Parquet source tables into DiskGraph-ready outputs: `index_metadata.json`, node data shards (binary by default, parquet optional), edge shard files (binary by default, parquet optional), and rowpack context files for node payloads.

## Inputs

- A `source_metadata.json` file describing:
  - node tables and schemas
  - edge tables (explicit edges)
  - foreign-key / primary-key relationships (implicit edges)
  - optional primary keys, timestamps, and weights
  - optional rowpack configuration per table
- Source data files (CSV or Parquet), local or cloud paths.

## source_metadata.json

Top-level schema (all fields optional except `dataset_name` and `tables`):

```json
{
  "dataset_name": "example",
  "tables": [/* TableSpec */],
  "relationships": [/* RelationshipSpec */],
  "rowpack": { /* RowPackConfig */ },
  "edge_format": "binary"
}
```

### TableSpec

```json
{
  "name": "Users",
  "format": "parquet",
  "paths": ["s3://bucket/users.parquet"],
  "columns": [/* ColumnSpec */],
  "primary_key": "id",
  "create_time": "created_at",
  "weight": "weight",
  "edge_table": {
    "src_fk": { "column": "src_id", "ref_table": "Users", "ref_column": "id" },
    "dst_fk": { "column": "dst_id", "ref_table": "Orders", "ref_column": "id" }
  },
  "rowpack": { /* RowPackConfig */ }
}
```

Rules:
- `format`: `csv` or `parquet`.
- `paths`: one or more files, all same schema.
- `primary_key` is optional; when present it must reference a numeric or string/bytes column.
- `create_time` (optional): must be `date` or `timestamp`.
- `weight` (optional): must be numeric.
- `edge_table` (optional): marks the table as an explicit edge list. Both FKs must reference a primary key on the destination table.
- `rowpack` (optional): per-table overrides for rowpack configuration. When present, it is merged with the top-level `rowpack` (table values override dataset values).
- `edge_format` (optional): `binary` (default) or `parquet`. Controls the output format for both edge shards and node data shards.

### ColumnSpec

```json
{
  "name": "id",
  "dtype": "string",
  "fixed_len": 16,
  "key_mode": "uuid",
  "precision": 10,
  "scale": 2
}
```

Fields:
- `dtype` (required): `int8|int16|int32|int64|uint8|uint16|uint32|uint64|float32|float64|bool|string|bytes|date|timestamp|decimal128|decimal256|float32_array|float64_array`
- `fixed_len` (optional): only valid for `string`/`bytes`. Fixed length in bytes.
- `key_mode` (optional): `fixed` or `uuid` (string/bytes only). Default is variable-length.
- `precision`/`scale` (optional): required for decimal columns and rejected for non-decimals.

### RelationshipSpec (implicit edges)

```json
{
  "name": "user_order",
  "src_table": "Orders",
  "src_column": "user_id",
  "dst_table": "Users",
  "dst_column": "id"
}
```

Rules:
- Must reference existing tables/columns.
- At least one side must be the primary key of its table.
- A primary-key-to-primary-key relationship cannot be declared in both directions.

### Key modes and compatibility

String/bytes keys:
- `key_mode: uuid` supports common UUID string formats (hex with/without dashes, upper/lower, optional braces or `urn:uuid:`) and raw 16-byte values for bytes.
- `key_mode: fixed` (or `fixed_len`) enforces fixed byte length.
- default is variable-length.

Compatibility rules:
- `uuid` only matches `uuid`.
- `fixed` only matches `fixed` of the same length.
- variable-length string/bytes can match each other and can match numeric types.

Key matching behavior:
- string values are compared as raw UTF-8 bytes; bytes are compared as raw bytes (no normalization).
- numeric ↔ string/bytes joins parse or format base-10 bytes.
- float/decimal → string/bytes uses a base-10 string representation.
- parse failures or missing targets skip the edge and are counted.
- duplicate primary keys are skipped and counted.

## Supported Parquet types

Ingestion expects schemas to match `source_metadata.json` and supports:

- **Int**: `Int8`, `Int16`, `Int32`, `Int64`
- **UInt**: `UInt8`, `UInt16`, `UInt32`, `UInt64`
- **Float**: `Float32`, `Float64`
- **Bool**: `Boolean`
- **String**: `Utf8`
- **Bytes**: `Binary`
- **Date**: `Date32`
- **Timestamp**: `Timestamp` with `Second|Millisecond|Microsecond|Nanosecond` units (timezone accepted, normalized to UTC)
- **Decimal**: `Decimal128`, `Decimal256` (requires `precision` + `scale` in metadata)
- **Arrays**: `List(Float32)` and `List(Float64)` only

Other Parquet/Arrow types are rejected during ingest.

## Outputs

- `index_metadata.json` with:
  - node/edge tables and shard ranges
  - rowpack `node_data_compression` per node table
  - primary key map paths (when configured)
- Per-shard index files under `nodes/<table>/shard-<id>/`:
  - `edges.bin` (edge blobs + optional node data)
  - `offsets.bin` (edge blob offsets)
- Rowpack context files per node table (`node_data.ctx` by default)
- `ingest_stats.json` with ingest stats (key failures, duplicate keys, schema echo)

## Ingestion stages

1) Encode (per table)
   - Read each input file and sample rows to build the rowpack context (one context per table).
   - Sampling uses only the first input file for that table.
   - Parse primary keys and foreign keys; invalid keys are skipped and recorded in stats.
   - Rowpack-encode the table primary key into a temp pkey file per shard (or round-robin when no PK).
   - Rowpack-encode node payloads into temp node data files per shard (written under `_tmp/`).
   - When a table has multiple input files, temp outputs are split per-file and numbered within each shard.
   - Enforce node data size limits at write time; oversized payloads are written as empty and counted.
   - For each outgoing FK, write temp files containing the encoded FK key (and optional sort key).
   - For explicit edge tables, write temp edge files for forward and reverse directions.

2) PK map (per table, per shard)
   - Build on-disk pk maps that map encoded PKs to shard-local IDs.
   - Fixed-width keys become sorted (key,id) files for binary search.
   - Variable-width keys become hash maps.
   - Duplicate PKs are dropped and counted; up to 10 samples are recorded.

3) Final nodes (per table, per shard)
   - Convert temp node data into the node binary format (stored under `_tmp/`).

4) Final edges (per relation, per shard)
   - Use pk maps to translate temp edge/FK keys into local IDs.
   - Emit forward and reverse edge shard files in the binary format under `_tmp/`.
   - Ensure edges are sorted by src_id (per relation and direction).

5) Finalize
   - Build the final per-shard index (`edges.bin` + `offsets.bin`) from the temp node/edge shards.
   - Write `index_metadata.json` and `ingest_stats.json`.
   - Remove temporary files and directories.

## Usage (CLI)

```bash
cargo run -p diskgraph-ingest -- --source /path/to/source_dir --output /path/to/index_dir --dest local --mapping hash
```

Flags:
- `--source`: directory containing `source_metadata.json` or the path to `source_metadata.json` directly.
- `--output`: output directory for `index_metadata.json` and parquet files.
- `--dest`: must be `local` (index build requires local output).
- `--mapping`: `hash|sorted` (primary key map strategy).
- `--profile [base]`: enable CPU profiling and emit `*-profile.pb`/`*-profile.svg` (ingest only; explore no longer supports profiling).

Helper scripts (release mode):
- `scripts/ingest_local.sh`
- `scripts/ingest_s3.sh`
- `scripts/ingest_gcs.sh`
- `scripts/ingest_adls.sh`

## Rowpack configuration

Rowpack settings are optional and can appear at:
- top-level (`rowpack`): default for all node tables.
- per table (`tables[].rowpack`): override for that table only.

When both are provided, the effective config is a merge where table values override top-level values.

Rowpack is a row-oriented compression format used for node payloads. It samples columns to pick fast encodings (dictionary, alphabet, BPE, numeric, timestamp, etc.) and writes a small context file that the reader uses to decode quickly. You typically only need to tune it when you want smaller node payloads (larger dictionaries, stricter thresholds) or faster encoding (smaller samples, lower dictionary budgets). The defaults are usually good enough.

RowPackConfig fields (all optional):
- `sample_rows` (int): rows to sample when building the context (default 10,000)
- `max_dict_bytes_per_col` (int): max bytes per string dict (default 1,000,000)
- `max_dict_entries` (int): max dict entries per string dict (default 50,000)
- `bpe_max_dict_bytes` (int): max bytes for BPE dict (default 1,000,000)
- `bpe_max_dict_entries` (int): max BPE entries (default 256)
- `bpe_max_samples` (int): max samples for BPE training (default 5,000)
- `bpe_min_gain_ratio` (float): minimum compression gain to keep BPE (default 0.95)
- `bpe_min_avg_len` (int): min average token length for BPE (default 12)
- `string_dict_coverage` (float): min coverage to enable dict (default 0.7)
- `string_long_min_len` (int): min length for long-string compression (default 128)
- `string_long_min_ratio` (float): min ratio to keep long-string compression (default 0.85)
- `string_iso_threshold` (float): threshold to classify ISO timestamps (default 0.85)
- `enable_float16` (bool): allow float16 (default true)
- `force_float64` (bool): force float64 (default false)
- `decimal_allow_lossy` (bool): allow lossy decimal compression (default true)

The context file path is written into `index_metadata.json` under
`node_data_compression.context_path` and used by the index reader/explorer.
