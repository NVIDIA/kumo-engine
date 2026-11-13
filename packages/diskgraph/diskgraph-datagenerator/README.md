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

For cloud storage, provide the output location directly:

```bash
cargo run -p diskgraph-datagenerator --release -- \
  --config diskgraph-datagenerator/configs/small_100m.json \
  --output s3://example-bucket/diskgraph-datagenerator/small_100m
```

The `cloud_generate_dataset.sh` wrapper requires `DISKGRAPH_CLOUD_BUCKET`, or
`DISKGRAPH_CLOUD_CONTAINER` for Azure. It has no default destination. Its existing
metadata detection can select GCS when a metadata request fails. Use the direct
command above to choose the destination explicitly.

## JSON config overview

- `total_rows`: total rows across all tables.
- `seed`: global seed (optional).
- `batch_size`: rows per batch (optional).
- `row_group_size`: Parquet row group size (optional).
- `parquet_compression`: `uncompressed` (default), `snappy`, or `zstd`.
- `tables`: schema and generation rules per table.
- `relationships`: foreign key mappings for join tables.

Each table defines a row share (`row_fraction` or `row_percent`), shard count, and columns. Each column specifies type and a generator. Primary key columns are marked with `primary_key: true`. Foreign key columns are defined via the `relationships` list (no per-column generator needed).

Notes:
- UUID/UUID256 columns emit lowercase hex strings by default; IP columns emit string IPs. Generators can override formats (e.g., dashed UUIDs or raw bytes) via `gen` config.
- `key_encoding` controls how synthetic key bytes are encoded (decimal/hex/uint64-le) for string/bytes columns.

## Organic graph degrees

The `organic` foreign-key distribution generates a repeatable broken power
law without keeping a degree array in memory. It is intended for large graph
tables where most entities have one or no edges and a small number of popular
entities have much larger adjacency lists.

```json
{
  "type": "organic",
  "minimum_degree": 1,
  "permutation_seed": 1102,
  "segments": [
    { "end_fraction": 0.001, "exponent": 0.24 },
    { "end_fraction": 0.01,  "exponent": 0.42 },
    { "end_fraction": 0.1,   "exponent": 0.58 },
    { "end_fraction": 0.5,   "exponent": 1.20 }
  ]
}
```

`minimum_degree` assigns that many rows to every referenced key. Remaining
rows follow the listed density curves by popularity rank; keys beyond the last
`end_fraction` receive only the minimum. The rank is then permuted into the
primary-key space. This prevents popular keys from being stored in one numeric
range while preserving the exact degree histogram. Relations that point to
the same entity type can share `permutation_seed` to make the same entities
popular across edge families. Their row order is shuffled independently, so
generation remains deterministic, stateless, and parallel by Parquet shard.

See `diskgraph-datagenerator/configs/` for examples.
