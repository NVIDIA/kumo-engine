use serde::{Deserialize, Serialize};

/// Logical layout version for on-disk shard artifacts.
/// v3: compact count varints (1–2 bytes, capped at 10k per relation per node).
/// v2: zero-edge nodes write no counts or data; offsets may repeat (delta 0).
pub const LAYOUT_VERSION: u32 = 3;

/// Filename conventions for artifacts emitted per shard.
pub const OFFSETS_FILENAME: &str = "offsets.bin";
pub const EDGES_FILENAME: &str = "edges.bin";

/// Number of offsets to delta-pack in a chunk.
pub const DELTA_CHUNK: usize = 64;

/// Header persisted at the start of the offsets file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OffsetFileHeader {
    pub version: u32,
    pub nodes: u64,
    pub total_offsets: u64,
    pub delta_chunk: u16,
    /// Offsets refer to byte positions in the edge blob.
    pub edge_blob_word_size: u16,
}

impl OffsetFileHeader {
    pub fn new(nodes: u64, total_offsets: u64) -> Self {
        Self {
            version: LAYOUT_VERSION,
            nodes,
            total_offsets,
            delta_chunk: DELTA_CHUNK as u16,
            edge_blob_word_size: 1,
        }
    }
}

/// Metadata captured per node while building the shard.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeEdgeStats {
    pub edge_count: u32,
    pub byte_len: u32,
}

/// Relation-specific layout information used to decode packed edges.
#[derive(Debug, Clone)]
pub struct RelationLayout {
    pub relation_id: u16,
    pub dst_bits: u8,
    pub dest_sort_key_bits: Option<u8>,
    pub entry_bytes: usize,
    /// Precomputed masks to avoid per-edge recomputation.
    pub dst_mask: u64,
    pub dest_sort_key_mask: Option<u64>,
}

/// A view over an edge list stored in the edge data file. This is a lightweight
/// pointer into an mmapped region; decoding uses the on-disk layout rather than
/// copying data out.
pub struct EdgeListView<'a> {
    pub raw: &'a [u8],
    pub layouts: std::sync::Arc<Vec<RelationLayout>>,
}

impl<'a> EdgeListView<'a> {
    /// Returns the relation segments for this node. Each entry contains the
    /// edge count and raw slice for that relation's edges.
    pub fn relation_segments(&self) -> Result<Vec<RelationSegment<'a>>, std::io::Error> {
        if self.raw.is_empty() {
            // Node has no edges; return empty segments for each relation.
            return Ok(self
                .layouts
                .iter()
                .map(|layout| RelationSegment {
                    layout: layout.clone(),
                    count: 0,
                    data: &[],
                })
                .collect());
        }
        let mut segments = Vec::with_capacity(self.layouts.len());
        // First read all counts up front (prefix) to allow quick validation.
        let mut cursor = 0usize;
        let mut counts = Vec::with_capacity(self.layouts.len());
        for layout in self.layouts.iter() {
            let (count, consumed) = decode_varint(&self.raw[cursor..])?;
            cursor = cursor
                .checked_add(consumed)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            counts.push((layout.clone(), count));
        }
        // Now slice the data segments in the same order.
        for (layout, count) in counts {
            let bytes = count
                .checked_mul(layout.entry_bytes as u32)
                .ok_or(std::io::ErrorKind::InvalidData)? as usize;
            let end = cursor
                .checked_add(bytes)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if end > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let data = &self.raw[cursor..end];
            segments.push(RelationSegment {
                layout,
                count,
                data,
            });
            cursor = end;
        }
        Ok(segments)
    }
}

/// A relation-specific slice of edges for a node.
pub struct RelationSegment<'a> {
    pub layout: RelationLayout,
    pub count: u32,
    pub data: &'a [u8],
}

fn decode_varint(buf: &[u8]) -> Result<(u32, usize), std::io::Error> {
    if buf.is_empty() {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    let first = buf[0];
    let tag = first >> 6;
    // Fast path: 1-byte count (0..=63).
    if tag == 0 {
        return Ok(((first & 0x3F) as u32, 1));
    }
    // Fast path: 2-byte count (64..=0x3FFF).
    if tag == 1 {
        if buf.len() < 2 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let hi = (first & 0x3F) as u32;
        let lo = buf[1] as u32;
        return Ok(((hi << 8) | lo, 2));
    }
    Err(std::io::ErrorKind::InvalidData.into())
}
