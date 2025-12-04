# DiskGraph Data Generator

This tool produces synthetic datasets in the format expected by diskgraph: a dataset root containing `metadata.json` plus Parquet shards for node and edge tables. Scenarios cover combinations of optional dest_sort_key presence, presorted shards, and single vs multi-table layouts.

## Building

```bash
cargo build -p diskgraph-datagenerator
```

## Usage

```bash
# Generate all scenarios with multi-shard nodes/edges (shared shard count)
cargo run -p diskgraph-datagenerator -- --output /tmp/dg-data --scenario coverage --nodes 10000 --edges 50000 --node-shards 8
```

Arguments:
- `--output PATH` (required): directory where dataset subdirectories are written.
- `--scenario {coverage,simple-no-optional,temporal-presorted-with-edge-id,temporal-weighted-unsorted,weighted-only,mixed-multitable}`: which scenario(s) to generate. `coverage` generates all.
- `--nodes N`: approximate nodes per node type (default: 10_000).
- `--edges N`: approximate edges per relation (default: 50_000).
- `--node-shards N`: number of Parquet shards per node type and per edge relation (default: 4; multi-shard by default).

## Scenarios and Edge Cases

- `simple_no_optional`: one node type, one relation, no dest_sort_key.
- `temporal_presorted`: dest_sort_key present (32-bit); shards sorted by (src_id, dest_sort_key desc).
- `temporal_weighted`: dest_sort_key present (16-bit); shards sorted by (src_id, dest_sort_key desc).
- `weighted_only`: dest_sort_key present (8-bit) used purely as weight.
- `mixed_multitable`: multiple node types and relations, mixing presence/absence of dest_sort_key; all shards sorted and aligned.
- `coverage`: generates all the above under the output directory.

## Output Layout

Each scenario creates a dataset directory under `--output`, for example `simple_no_optional/`:
- `metadata.json`: matches the schema in `diskgraph-core/src/metadata.rs`.
- `nodes/TYPE/part-00000.parquet ...`: sharded Parquet files per node type with an `id` column.
- `edges/RELATION/part-00000.parquet ...`: shards for each edge relation containing `src_id`, `dst_id`, optional `dest_sort_key` columns.
Metadata records the relative shard paths for both nodes and edges plus per-shard id ranges (node shard min/max id; edge shard source min/max id) to keep the dataset self-contained and routable.

Generation guarantees:
- Node shards are contiguous, sorted by `id`.
- Edge shards are sorted by `(src_id, dest_sort_key DESC when present)`.
- Shard counts for nodes and edges match, and edge shard `N` only contains `src_id` values from node shard `N` of the relation’s source type.

Parquet files are generated with ArrowWriter using plain-encoded `Int64` columns to keep ingestion predictable.
