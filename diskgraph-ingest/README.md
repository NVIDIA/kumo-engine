# diskgraph-ingest

`diskgraph-ingest` converts raw CSV/Parquet source tables into DiskGraph-ready outputs: `index_metadata.json`, node parquet files, edge parquet files, and rowpack context files for node payloads.

## Inputs

- A `source_metadata.json` file describing:
  - node tables and schemas
  - edge tables (explicit edges)
  - foreign-key / primary-key relationships (implicit edges)
  - optional primary keys, timestamps, and weights
  - optional rowpack configuration per table
- Source data files (CSV or Parquet), local or cloud paths.

## Outputs

- `index_metadata.json` with:
  - node/edge tables and shard ranges
  - rowpack `node_data_compression` per node table
  - primary key map paths (when configured)
- Node parquet files with rowpack-compressed node payloads
- Edge parquet files (bidirectional)
- Rowpack context files per node table (`node_data.ctx` by default)

## Usage (CLI)

```bash
cargo run -p diskgraph-ingest -- --source /path/to/source_dir --output /path/to/index_dir --dest local --mapping hash
```

Flags:
- `--source`: directory containing `source_metadata.json` or the path to `source_metadata.json` directly.
- `--output`: output directory for `index_metadata.json` and parquet files.
- `--dest`: `local|s3|gcs|adls` (controls destination backend).
- `--mapping`: `hash|sorted` (primary key map strategy).
- `--profile [base]`: enable CPU profiling and emit `*-profile.pb`/`*-profile.svg`.

Helper scripts (release mode):
- `scripts/ingest_local.sh`
- `scripts/ingest_s3.sh`
- `scripts/ingest_gcs.sh`
- `scripts/ingest_adls.sh`

## Rowpack configuration

Rowpack settings live in `source_metadata.json` per node table:
- sample size
- dictionary budget
- float precision policy
- long-string compression thresholds

The context file path is written into `index_metadata.json` under
`node_data_compression.context_path` and used by the index reader/explorer.

