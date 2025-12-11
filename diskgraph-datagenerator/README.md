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
- Node shards contain contiguous ranges per shard, sorted by `id`; gaps may exist between shard ranges.
- Edge shards are sorted by `(src_id, dest_sort_key DESC when present)`.
- Shard counts for nodes and edges match, and edge shard `N` only contains `src_id` values from node shard `N` of the relation’s source type.

Parquet files are generated with ArrowWriter using plain-encoded `Int64` columns to keep ingestion predictable.

## Massive-scale generator (single shard)

For large throughput benchmarks (hundreds of millions of nodes / billions of edges), use the dedicated binary:

```bash
# Generates nodes.parquet + two edge files and metadata.json under /tmp/dg-massive/run-<timestamp>
OUT_ROOT=/tmp/dg-massive RUN_ID=run-$(date +%s) \
  cargo run -p diskgraph-datagenerator --bin massive --release -- \
    --output "/tmp/dg-massive/${RUN_ID}" \
    --nodes 500000000 \
    --edges-per-relation 500000000 \
    --max-dst-id 100000000000 \
    --edge-batch-size 1000000 \
    --node-batch-size 5000000 \
    --relation0-name follows \
    --relation1-name blocks \
    --relation1-sort-bits 16
```

The generator streams Parquet row groups in batches (configurable) to keep RAM usage to a few GB while emitting very large files. It writes:

- `nodes.parquet` with a single `id` column.
- `edges_<relation>.parquet` for two relations (names configurable), optional `dest_sort_key` when `--relation{N}-sort-bits` is non-zero.
- `metadata.json` describing the dataset.

Helper scripts live in `scripts/` to generate and upload large datasets:

- `scripts/generate_massive_local.sh` – local filesystem output.
- `scripts/generate_massive_s3.sh` – generate locally, then `aws s3 cp --recursive` to `s3://kumo-unit-test/diskgraph-datagenerator/...` (editable).
- `scripts/generate_massive_gcs.sh` – generate locally, then `gsutil -m cp` to `gs://kumo-unit-test/diskgraph-datagenerator/...` (editable).
- `scripts/generate_massive_adls.sh` – generate locally, then `azcopy copy` to an ADLS/Blob URL (editable).

Each script exposes tunable constants at the top (nodes, edges per relation, dest_sort_key bits, batch sizes, prefixes, etc.) and relies on ambient credentials for the respective cloud CLI.
