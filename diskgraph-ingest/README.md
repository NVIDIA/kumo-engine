# diskgraph-ingest

`diskgraph-ingest` turns CSV/Parquet source tables plus `source_metadata.json` into a local DiskGraph index. It builds rowpack contexts, sharded primary key maps, and edge shards, then runs the index builder to produce the binary `edges.bin`/`offsets.bin` files under the output directory.

## Inputs

- A `source_metadata.json` file describing:
  - node tables and schemas
  - explicit edge tables (optional)
  - foreign-key / primary-key relationships (implicit edges)
  - optional primary keys, timestamps, weights
  - optional edge sort keys
  - optional rowpack configuration
- Source data files (CSV or Parquet). Paths may be local or `s3://`, `gs://`, `az://`.

## source_metadata.json

Top-level schema (all fields optional except `dataset_name` and `tables`):

```json
{
  "dataset_name": "example",
  "tables": [/* TableSpec */],
  "relationships": [/* RelationshipSpec */],
  "edge_sort_keys": { /* EdgeSortKeysSpec */ },
  "rowpack": { /* RowPackConfig */ },
  "num_shards": 8,
  "edge_format": "binary"
}
```

Notes:
- `num_shards` defaults to `1` when omitted.
- `edge_format` exists but only `binary` is accepted (parquet output is rejected).

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
- `primary_key` (optional): numeric or string/bytes column.
- `create_time` (optional): `date` or `timestamp` column.
- `weight` (optional): numeric column.
- `edge_table` (optional): marks the table as an explicit edge list. Both FKs must reference a primary key on the destination table.
- `rowpack` (optional): per-table overrides for rowpack configuration. When present, it is merged with the top-level `rowpack` (table values override dataset values).

### EdgeSortKeysSpec (explicit sort keys)

Sort keys are defined per edge relation and per direction. They can pull a timestamp
and/or weight from either the destination node table or an edge table (when the
relation comes from an edge table). Each component is encoded and packed into a
single sortable key.

```json
{
  "edge_sort_keys": {
    "relationships": {
      "order_user": {
        "forward": {
          "timestamp": {
            "table": "User",
            "column": "created_at",
            "encoding": {"kind": "int32", "anchor": 0, "step": 1}
          },
          "weight": {
            "table": "User",
            "expr": "weight",
            "encoding": {"dtype": "float32"}
          }
        },
        "reverse": null
      }
    },
    "edge_tables": {
      "OrderItem": {
        "src_to_dst": {
          "weight": {
            "table": "OrderItem",
            "expr": "weight",
            "encoding": {"dtype": "float32"}
          }
        },
        "dst_to_src": {
          "timestamp": {
            "table": "Order",
            "column": "created_at",
            "encoding": {"kind": "int16", "anchor": 0, "step": 1}
          }
        }
      }
    }
  }
}
```

Notes:
- `relationships` keys are the relationship name (or auto-generated name).
- `edge_tables` keys are the edge table name.
- `table` must be either the destination table or the edge table for that direction.
- `expr` is a math expression over numeric columns in `table`.
  Use backticks for column names with special characters: `` `weight-kg` * 2 ``.
- Encodings:
  - timestamp `kind`: `int16|int32`
  - weight `dtype`: `byte|uint16|uint32|float16|float32`

### ColumnSpec

```json
{
  "name": "id",
  "source_dtype": "string",
  "ingested_dtype": "string",
  "fixed_len": 16,
  "key_mode": "uuid",
  "precision": 10,
  "scale": 2
}
```

Fields:
- `source_dtype` / `ingested_dtype` (required): `int8|int16|int32|int64|uint8|uint16|uint32|uint64|float32|float64|bool|string|bytes|date|timestamp|decimal128|decimal256|float32_array|float64_array`
- `fixed_len` (optional): only valid for `string`/`bytes`.
- `key_mode` (optional): `fixed` or `uuid` (string/bytes only).
- `timestamp_format` (optional): format string for string timestamps.
- `timestamp_unit` (optional): `seconds|millis|micros|nanos` for integer timestamps.
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

## Outputs (local only)

`diskgraph-ingest` requires a local output directory and builds a DiskGraph index:

- `index_metadata.json`
- `ingest_stats.json`
- `nodes/<table>/shard-<id>/offsets.bin`
- `nodes/<table>/shard-<id>/edges.bin`
- `nodes/<table>/pk_data-<id>.bin`
- `nodes/<table>/pk_hash-<id>.bin`
- `nodes/<table>/node_data.ctx` (rowpack context)

Temporary files are written under `_tmp/` and removed after finalize. The final `index_metadata.json` intentionally clears temporary paths.

## Ingestion stages (high level)

1) **Rowpack contexts**
   - Sample rows per node table and write `node_data.ctx`.
2) **Temp shards**
   - Encode node payloads and FK/edge keys into consolidated shard files under `_tmp/`.
   - Files are appended concurrently across input files; per-file order is not preserved, but row alignment across node data/pkeys/fkeys is preserved.
3) **Primary key maps**
   - Build sharded PK hash + data files (`pk_hash-<id>.bin`, `pk_data-<id>.bin`).
4) **Edge mapping**
   - Resolve implicit and explicit edges, translate keys to IDs, and write unsorted edge parts under `_tmp/edges_map` (one part per shard).
   - Sorting is deferred to the index build stage.
5) **Index build**
   - Run the DiskGraph index builder to produce `edges.bin` and `offsets.bin`.
6) **Finalize**
   - Write `index_metadata.json` and `ingest_stats.json`, then delete `_tmp/`.

## Usage (CLI)

```bash
cargo run -p diskgraph-ingest --bin ingest --release -- \
  --source /path/to/source_dir_or_metadata \
  --output /path/to/index_dir \
  --dest local
```

Flags:
- `--source`: directory containing `source_metadata.json` or the path to it.
- `--output`: local output directory for the index.
- `--dest`: must be `local` (index build requires local output).
- `--profile [dir]`: enable CPU profiling and emit `*-profile.pb`/`*-profile.svg`.
- `-m, --mem-profile`: enable jemalloc heap dump watchdog (writes `heap-*.heap` when RSS anon exceeds 80%; requires the `mem-profile` Cargo feature and `_RJEM_MALLOC_CONF=prof:true`).

Helper script:
- `scripts/ingest_local.sh`

## Rowpack configuration

Rowpack settings are optional and can appear at:
- top-level (`rowpack`): default for all node tables.
- per table (`tables[].rowpack`): override for that table only.

Rowpack is a row-oriented compression format used for node payloads. It samples columns to pick fast encodings (dictionary, alphabet, BPE, lz4, etc.) and writes a small context file that the reader uses to decode quickly. Defaults are usually good enough.

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

The context file path is written into `index_metadata.json` under
`node_data_compression.context_path` and used by the index reader/explorer.
