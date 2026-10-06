# DiskGraph Dataset Specification

This document captures the required on-disk layout and `index_metadata.json` schema for datasets ingested by DiskGraph.

## Directory Layout

- Dataset root contains:
  - `index_metadata.json`
- `nodes/TABLE/shard-00000/offsets.bin` — delta-encoded offsets for node edge blobs.
- `nodes/TABLE/shard-00000/edges.bin` — packed edge blobs (per node, per relation).
- `nodes/TABLE/pk_data-00000.bin` / `nodes/TABLE/pk_hash-00000.bin` — optional primary key maps.
- `nodes/TABLE/node_data.ctx` — rowpack context file for that node table (when node data is enabled).
- Shard alignment: shard `N` only contains `src_id` values within the min/max range of node shard `N` for the relation’s source type. Node shard ranges are contiguous within a shard, but ranges for a given type may have gaps between shards.
- Sorting (binary format): edge parts are sorted by `(src_id, dest_sort_key desc when present)` before being packed.

Legacy parquet edge outputs (format `parquet`) may use:
- `edges/RELATION/part-00000.parquet ...`

## Column Names (fixed for parquet edge format)

- Edge Parquet schema (depending on optional fields):
  - Required: `src_id` (Int64), `dst_id` (Int64)
  - Optional: `dest_sort_key` (Int64) when present in relation metadata
Column names are fixed and must adhere to this spec.

## index_metadata.json Schema

```jsonc
{
  "dataset_name": "my_dataset",
  "node_types": [
    {
      "name": "User",
      "max_id": 123456,
      "shards": [
        { "min_id": 0, "max_id": 49999 },
        { "min_id": 50000, "max_id": 99999 }
      ],
      "node_data_compression": {
        "codec": "rowpack",
        "context_path": "nodes/User/node_data.ctx"
      }
    }
  ],
  "edge_relations": [
    {
      "name": "Follows",
      "src_type": "User",
      "dst_type": "User",
      "shards": [
        { "path": "nodes/User/shard-00000/edges.bin", "src_min_id": 0, "src_max_id": 49999 },
        { "path": "nodes/User/shard-00001/edges.bin", "src_min_id": 50000, "src_max_id": 99999 }
      ],
      "dest_sort_key_bits": null,
      "dest_sort_key": null,
      "format": "binary"
    }
  ]
}
```

Field notes:
- `dataset_name`: arbitrary identifier for this dataset.
- `node_types[*].name`: logical node type name.
- `node_types[*].max_id`: maximum node id observed for the type (may be greater than the total node count if there are gaps between shard ranges).
- `node_types[*].shards`: contiguous ID ranges for node shards; ranges may be non-contiguous across shards of the same type. Gaps between shards are allowed and their widths are dataset-defined (no fixed spacing requirement).
- `node_types[*].node_data_compression`: optional, describes node payload compression (rowpack context path).
- `node_types[*].node_data_paths`: optional list of node data input parts used during index build; final indices typically omit this field after embedding node data in `edges.bin`.
- `edge_relations[*].name`: logical relation name.
- `edge_relations[*].src_type` / `dst_type`: refer to node type names.
- `edge_relations[*].shards`: relative paths and source ID ranges for edge shards; shard count must align with the source node type shard count. For binary format, these paths point at the shard’s `edges.bin`.
- `edge_relations[*].dest_sort_key_bits`: multiple of 8 up to 64 (bit width) or null to indicate presence/width of `dest_sort_key`.
- `edge_relations[*].dest_sort_key`: optional metadata describing timestamp/weight encodings and original expressions.

## Validation Rules

- All IDs in Parquet files must fall within the shard ranges declared in metadata.
- Shards for a given node type must not overlap; gaps between shard ranges are allowed.
- Edge shard ranges must be a subset of the source node shard ranges.
- Sorting requirements must hold for fast lookup (binary format requires sorted edge parts before packing).

## Non-goals

- Compression/layout specifics are implementation-defined.
- No alternate column names are supported; schemas are fixed as above.
