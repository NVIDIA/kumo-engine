use crate::errors::{DiskgraphError, Result};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// Optional compression metadata for node data blobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDataCompression {
    pub codec: String,
    pub context_path: String,
}

/// Optional primary key mapping metadata for node tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaryKeyMap {
    pub column: String,
    pub dtype: String,
    pub mapping_path: String,
    pub mapping_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_mode: Option<String>,
}

/// Per-shard node metadata to route lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeShardMetadata {
    pub min_id: u64,
    pub max_id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_data_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_data_paths: Option<Vec<String>>,
}

/// Node type definition pulled from index_metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTypeMetadata {
    pub name: String,
    /// Maximum node id observed for the type.
    pub max_id: u64,
    #[serde(default)]
    pub has_node_data: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_data_compression: Option<NodeDataCompression>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_key_map: Option<PrimaryKeyMap>,
    /// Contiguous ID ranges for node shards.
    pub shards: Vec<NodeShardMetadata>,
}

/// Per-shard edge metadata to route source id lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeShardMetadata {
    pub path: String,
    pub src_min_id: u64,
    pub src_max_id: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeStorageFormat {
    Parquet,
    Binary,
}

fn default_edge_storage_format() -> EdgeStorageFormat {
    EdgeStorageFormat::Binary
}

/// Edge relation definition pulled from index_metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRelationMetadata {
    pub name: String,
    /// Source node type name.
    pub src_type: String,
    /// Destination node type name.
    pub dst_type: String,
    /// Relative paths to the edge shards from the dataset root.
    pub shards: Vec<EdgeShardMetadata>,
    /// Optional dest_sort_key configuration (bit width when present).
    pub dest_sort_key_bits: Option<u8>,
    #[serde(default = "default_edge_storage_format")]
    pub format: EdgeStorageFormat,
}

/// Root metadata payload found in index_metadata.json.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub node_types: Vec<NodeTypeMetadata>,
    pub edge_relations: Vec<EdgeRelationMetadata>,
}

impl DatasetMetadata {
    /// Heuristic size estimation that uses the bit widths of ids and optional
    /// edge payloads. The output must be conservative (over-estimate) to keep
    /// admission control safe.
    pub fn estimate_ondisk_bytes(&self) -> Result<u64> {
        let mut bytes: u64 = 0;
        // Nodes: ids only, packed densely with minimal overhead.
        for node in &self.node_types {
            let count_from_shards: u64 = node
                .shards
                .iter()
                .map(|s| {
                    if s.max_id >= s.min_id {
                        s.max_id - s.min_id + 1
                    } else {
                        0
                    }
                })
                .sum();
            let count = count_from_shards.max(node.max_id + 1);
            let per_node = 8u64;
            bytes = bytes
                .checked_add(count.saturating_mul(per_node))
                .ok_or_else(|| DiskgraphError::InvalidMetadata("node size overflow".into()))?;
        }
        // Edges: id + dest_sort_key + layout overhead (~16 bytes).
        for edge in &self.edge_relations {
            // We do not know the exact edge count from metadata. Use a conservative
            // heuristic: assume a fanout per source id (default 16) across the
            // covered src_id ranges. This intentionally over-estimates to keep
            // admission control safe.
            let src_span: u64 = edge
                .shards
                .iter()
                .map(|s| {
                    if s.src_max_id >= s.src_min_id {
                        s.src_max_id - s.src_min_id + 1
                    } else {
                        0
                    }
                })
                .sum();
            let assumed_fanout = 16u64;
            let count = src_span.saturating_mul(assumed_fanout).max(src_span);
            let mut per_edge = 8u64; // src_id + dst_id minimum footprint.
            if let Some(bits) = edge.dest_sort_key_bits {
                per_edge = per_edge.saturating_add((bits as u64).saturating_div(8).max(1));
            }
            // Allow for encoding overhead and indexing structures.
            per_edge = per_edge.saturating_add(16);
            bytes = bytes
                .checked_add(count.saturating_mul(per_edge))
                .ok_or_else(|| DiskgraphError::InvalidMetadata("edge size overflow".into()))?;
        }
        // Inflate by 20% to account for alignment, compression dictionaries, and
        // small files on disk.
        let inflated = bytes
            .saturating_mul(12)
            .checked_div(10)
            .ok_or_else(|| DiskgraphError::InvalidMetadata("size inflation overflow".into()))?;
        Ok(inflated)
    }
}
