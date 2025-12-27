# DiskGraph Dataset Specification

This document captures the required on-disk layout and `metadata.json` schema for datasets ingested by DiskGraph.

## Directory Layout

- Dataset root contains:
  - `metadata.json`
  - `nodes/TYPE/part-00000.parquet ...` — optional; node parquet shards are not required for indexing.
  - `edges/RELATION/part-00000.parquet ...` — sharded Parquet files per edge relation.
- Shard alignment: edge shard `N` only contains `src_id` values within the min/max range of node shard `N` for the relation’s source type. Node shard ranges are contiguous within a shard, but ranges for a given type may have gaps between shards.
- Sorting:
  - Node shards (when present): sorted by `id` ascending.
  - Edge shards: sorted by `(src_id, dest_sort_key desc when present)`.

## Column Names (fixed)

- Node Parquet schema (optional): `id` (Int64).
- Edge Parquet schema (depending on optional fields):
  - Required: `src_id` (Int64), `dst_id` (Int64)
  - Optional: `dest_sort_key` (Int64) when present in relation metadata
Column names are fixed and must adhere to this spec.

## metadata.json Schema

```jsonc
{
  "dataset_name": "my_dataset",
  "node_types": [
    {
      "name": "User",
      "max_id": 123456,
      "shards": [
        { "path": "nodes/user/part-00000.parquet", "min_id": 0, "max_id": 49999 },
        { "path": "nodes/user/part-00001.parquet", "min_id": 50000, "max_id": 99999 }
      ]
    }
  ],
  "edge_relations": [
    {
      "name": "Follows",
      "src_type": "User",
      "dst_type": "User",
      "shards": [
        { "path": "edges/follows/part-00000.parquet", "src_min_id": 0, "src_max_id": 49999 },
        { "path": "edges/follows/part-00001.parquet", "src_min_id": 50000, "src_max_id": 99999 }
      ],
      "dest_sort_key_bits": null
    }
  ]
}
```

Field notes:
- `dataset_name`: arbitrary identifier for this dataset.
- `node_types[*].name`: logical node type name.
- `node_types[*].max_id`: maximum node id observed for the type (may be greater than the total node count if there are gaps between shard ranges).
- `node_types[*].shards`: relative paths and contiguous ID ranges for node shards; ranges may be non-contiguous across shards of the same type. Gaps between shards are allowed and their widths are dataset-defined (no fixed spacing requirement). Paths may be omitted on disk (node parquet optional).
- `edge_relations[*].name`: logical relation name.
- `edge_relations[*].src_type` / `dst_type`: refer to node type names.
- `edge_relations[*].shards`: relative paths and source ID ranges for edge shards; shard count must align with the source node type shard count.
- `edge_relations[*].dest_sort_key_bits`: one of `4|8|16|32` (bit width) or null to indicate presence/width of `dest_sort_key`.

## Validation Rules

- All IDs in Parquet files must fall within the shard ranges declared in metadata.
- Shards for a given node type must not overlap; gaps between shard ranges are allowed.
- Edge shard ranges must be a subset of the source node shard ranges.
- Sorting requirements must hold for fast lookup.

## Non-goals

- Compression/layout specifics are implementation-defined.
- No alternate column names are supported; schemas are fixed as above.
