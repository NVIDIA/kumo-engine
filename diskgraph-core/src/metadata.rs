use crate::errors::{DiskgraphError, Result};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

/// Bit width enum for ids and optional edge attributes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitWidth {
    Bits32,
    Bits64,
    Bits16,
    Bits8,
    Bits4,
}

impl BitWidth {
    pub fn as_bits(self) -> u8 {
        match self {
            BitWidth::Bits32 => 32,
            BitWidth::Bits64 => 64,
            BitWidth::Bits16 => 16,
            BitWidth::Bits8 => 8,
            BitWidth::Bits4 => 4,
        }
    }
}

/// Node type definition pulled from metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTypeMetadata {
    pub name: String,
    /// Maximum node id observed for the type.
    pub max_id: u64,
    /// Expected cardinality. Used for sizing the on-disk layout.
    pub estimated_count: Option<u64>,
    /// Input ID width (32 or 64 bits).
    pub id_width: BitWidth,
}

/// Edge relation definition pulled from metadata.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRelationMetadata {
    pub name: String,
    /// Source node type name.
    pub src_type: String,
    /// Destination node type name.
    pub dst_type: String,
    /// Relative paths to the Parquet shards from the dataset root.
    pub shards: Vec<String>,
    /// Optional dest_sort_key configuration.
    pub dest_sort_key_bits: Option<BitWidth>,
    /// Optional edge_id width.
    pub edge_id_bits: Option<BitWidth>,
    /// Whether input shards are pre-sorted by (src_id, dest_sort_key desc).
    pub presorted_by_src_and_sort_key: bool,
    /// Expected edge count for sizing admission control.
    pub estimated_count: Option<u64>,
}

/// Root metadata payload found in metadata.json.
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub dataset_name: String,
    pub node_types: Vec<NodeTypeMetadata>,
    pub edge_relations: Vec<EdgeRelationMetadata>,
    /// Optional hint about the logical timestamp column name used to derive
    /// dest_sort_key when present.
    pub dest_sort_key_name: Option<String>,
    /// Optional hint about the edge identifier column name.
    pub edge_id_name: Option<String>,
    /// Raw size of the source Parquet payload on object storage or disk.
    #[serde_as(as = "Option<DisplayFromStr>")]
    pub source_parquet_bytes: Option<u64>,
}

impl DatasetMetadata {
    /// Heuristic size estimation that uses the bit widths of ids and optional
    /// edge payloads. The output must be conservative (over-estimate) to keep
    /// admission control safe.
    pub fn estimate_ondisk_bytes(&self) -> Result<u64> {
        let mut bytes: u64 = 0;
        // Nodes: ids only, packed densely with minimal overhead.
        for node in &self.node_types {
            let count = node.estimated_count.unwrap_or(node.max_id + 1);
            let per_node = (node.id_width.as_bits() as u64).saturating_div(8).max(4);
            bytes = bytes
                .checked_add(count.saturating_mul(per_node))
                .ok_or_else(|| DiskgraphError::InvalidMetadata("node size overflow".into()))?;
        }
        // Edges: id + dest_sort_key + edge_id + layout overhead (~16 bytes).
        for edge in &self.edge_relations {
            let count = edge.estimated_count.unwrap_or(0);
            let mut per_edge = 8u64; // src_id + dst_id minimum footprint.
            if let Some(bits) = edge.dest_sort_key_bits {
                per_edge = per_edge.saturating_add((bits.as_bits() as u64).saturating_div(8).max(1));
            }
            if let Some(bits) = edge.edge_id_bits {
                per_edge = per_edge.saturating_add((bits.as_bits() as u64).saturating_div(8).max(4));
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
