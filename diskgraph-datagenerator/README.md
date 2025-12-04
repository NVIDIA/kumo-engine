# DiskGraph Data Generator

This tool produces synthetic datasets in the format expected by diskgraph: a dataset root containing `metadata.json` plus Parquet shards for node and edge tables. Scenarios cover combinations of optional fields (dest_sort_key, edge_id), presorted vs unsorted shards, and single vs multi-table layouts.

## Building

```bash
cargo build -p diskgraph-datagenerator
```

## Usage

```bash
cargo run -p diskgraph-datagenerator -- --output /tmp/dg-data --scenario coverage --nodes 10000 --edges 50000 --edge-shards 4
```

Arguments:
- `--output PATH` (required): directory where dataset subdirectories are written.
- `--scenario {coverage,simple-no-optional,temporal-presorted-with-edge-id,temporal-weighted-unsorted,weighted-only,mixed-multitable}`: which scenario(s) to generate. `coverage` generates all.
- `--nodes N`: approximate nodes per node type (default: 10_000).
- `--edges N`: approximate edges per relation (default: 50_000).
- `--edge-shards N`: number of Parquet shards per edge relation (default: 4).

## Scenarios and Edge Cases

- `simple_no_optional`: one node type, one relation, no dest_sort_key or edge_id.
- `temporal_presorted_with_edge_id`: dest_sort_key present (32-bit), edge_id present (64-bit), input shards presorted by (src_id, dest_sort_key desc).
- `temporal_weighted_unsorted`: dest_sort_key present (16-bit) with unsorted shards to test robustness of ingest sorting; edge_id present (32-bit).
- `weighted_only`: dest_sort_key present (8-bit) used purely as weight; no edge_id.
- `mixed_multitable`: multiple node types and relations, mixing presence/absence of dest_sort_key and edge_id; both presorted and unsorted relations included.
- `coverage`: generates all the above under the output directory.

## Output Layout

Each scenario creates a dataset directory under `--output`, for example `simple_no_optional/`:
- `metadata.json`: matches the schema in `diskgraph-core/src/metadata.rs`.
- `nodes/TYPE_nodes.parquet`: one Parquet file per node type with an `id` column.
- `edges/RELATION/part-00000.parquet ...`: shards for each edge relation containing `src_id`, `dst_id`, optional `dest_sort_key`, and optional `edge_id` columns.

Parquet files are generated with ArrowWriter using plain-encoded `Int64` columns to keep ingestion predictable.
