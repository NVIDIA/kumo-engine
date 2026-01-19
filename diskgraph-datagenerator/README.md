# Diskgraph Data Generator

This module generates synthetic Parquet datasets from a JSON config. It writes one folder per table under the output root, with per-shard `part-XXXXX.parquet` files. The generator is JSON-driven, supports the full ingest type set (except decimal), and creates primary/foreign keys without holding all keys in memory.

## Build

```bash
cargo build -p diskgraph-datagenerator
```

## Run

```bash
cargo run -p diskgraph-datagenerator -- \
  --config diskgraph-datagenerator/configs/small_100m.json \
  --output /mnt/localssd/datasets/small_100m
```

Set `DISKGRAPH_DATA_ROOT=/path/to/data` to override the `/mnt/localssd` base when using wrapper scripts.

Or use the wrapper script:

```bash
diskgraph-datagenerator/scripts/generate_dataset.sh diskgraph-datagenerator/configs/small_100m.json
```

Cloud example (auto-detects AWS/GCP/Azure metadata, writes to `kumo-unit-test` by default):

```bash
diskgraph-datagenerator/scripts/cloud_generate_dataset.sh diskgraph-datagenerator/configs/small_100m.json
```

## JSON config overview

- `total_rows`: total rows across all tables.
- `seed`: global seed (optional).
- `batch_size`: rows per batch (optional).
- `row_group_size`: Parquet row group size (optional).
- `tables`: schema and generation rules per table.
- `relationships`: foreign key mappings for join tables.

Each table defines a row share (`row_fraction` or `row_percent`), shard count, and columns. Each column specifies type and a generator. Primary key columns are marked with `primary_key: true`. Foreign key columns are defined via the `relationships` list (no per-column generator needed).

Notes:
- UUID/UUID256 columns emit lowercase hex strings by default; IP columns emit string IPs. Generators can override formats (e.g., dashed UUIDs or raw bytes) via `gen` config.
- `key_encoding` controls how synthetic key bytes are encoded (decimal/hex/uint64-le) for string/bytes columns.

See `diskgraph-datagenerator/configs/` for examples.
