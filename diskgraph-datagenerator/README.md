# DiskGraph Data Generator

This tool produces synthetic datasets in the format expected by diskgraph: a dataset root containing `metadata.json` plus Parquet shards for edge tables. Node shard ranges are recorded in metadata. Scenarios cover combinations of optional dest_sort_key presence, presorted shards, and single vs multi-table layouts.

## Building

```bash
cargo build -p diskgraph-datagenerator
```

## Usage

```bash
# Generate all scenarios with multi-shard edges
cargo run -p diskgraph-datagenerator -- --output /tmp/dg-data --scenario coverage --nodes 10000 --edges 50000 --node-shards 8
```

Arguments:
- `--output PATH` (required): directory where dataset subdirectories are written.
- `--scenario {coverage,simple-no-optional,temporal-presorted-with-edge-id,temporal-weighted-unsorted,weighted-only,mixed-multitable}`: which scenario(s) to generate. `coverage` generates all.
- `--nodes N`: approximate nodes per node type (default: 10_000).
- `--edges N`: approximate edges per relation (default: 50_000).
- `--node-shards N`: number of node shard ranges (recorded in metadata) and edge shards per relation (default: 4; multi-shard by default).

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
- `edges/RELATION/part-00000.parquet ...`: shards for each edge relation containing `src_id`, `dst_id`, optional `dest_sort_key` columns.
Metadata records node shard ranges plus the relative edge shard paths and per-shard source id ranges (edge shard source min/max id) to keep the dataset self-contained and routable.

Generation guarantees:
- Node shards contain contiguous ranges per shard; gaps may exist between shard ranges.
- Edge shards are sorted by `(src_id, dest_sort_key DESC when present)`.
- Shard counts for nodes and edges match, and edge shard `N` only contains `src_id` values from node shard `N` of the relation’s source type.

Parquet files are generated with ArrowWriter using plain-encoded `Int64` columns to keep ingestion predictable.

## Massive-scale generator (streaming, single shard)

For throughput benchmarks at hundreds of millions of nodes / billions of edges, use the `massive` binary. It streams Parquet row groups directly to the destination (local filesystem, S3, GCS, ADLS) in batches so RAM stays within a few GB. Default multipart size for S3 is 64 MiB (`--s3-part-size`), and Parquet row groups default to 1,000,000 rows.

Example (direct to S3, full scale by default):

```bash
# Uses ambient AWS profile (auto-selected like the diskgraph-index cloud tests), streams without local staging
S3_PREFIX=s3://kumo-unit-test/diskgraph-datagenerator RUN_ID=run-$(date +%s) \
  scripts/generate_massive_s3.sh
```

CLI highlights (see `--help`):
- `--dest {local,s3,gcs,adls}` and `--output <path or URL>` control the sink.
- `--nodes`, `--edges-per-relation`, `--max-dst-id` set scale; defaults: 500M nodes, 500M edges per relation, dst ids sampled up to 1e11.
- `--edge-batch-size`, `--node-batch-size`, `--row-group-size` set streaming/buffering; defaults keep memory low.
- `--relation{0,1}-name` and `--relation{0,1}-sort-bits` toggle relation names and optional `dest_sort_key`.
- `--s3-part-size` (default 64 MiB), `--s3-region`, `--s3-endpoint` allow tuning multipart uploads when `--dest s3`.

Helper scripts (all default to the full-scale run; override with env vars shown in the scripts):
- `scripts/generate_massive_local.sh` — local filesystem output (OUT_ROOT, RUN_ID).
- `scripts/generate_massive_s3.sh` — direct streaming to S3; auto-detects an AWS profile if none is set, disables EC2 IMDS, and verifies `sts get-caller-identity`.
- `scripts/generate_massive_gcs.sh` — direct streaming to GCS; runs a quick `gsutil` access check before generating.
- `scripts/generate_massive_adls.sh` — direct streaming to Azure Blob/ADLS; honors the usual `AZURE_*` env vars.

Each script writes `edges_<relation>.parquet` for two relations, and `metadata.json` under a per-run prefix like `.../run-<timestamp>-<rand>`. Metadata uses the larger of `nodes-1` and `max_dst_id` so `dst_id` values are always in-range for indexing.
