use crate::pkmap::Key;
use diskgraph_core::metadata::PrimaryKeyMap;
use diskgraph_rowpack::{RowPackDecodeScratch, RowPackDecoder, ValueRef};
use serde::{Deserialize, Serialize};

/// Logical layout version for on-disk shard artifacts.
/// v4: offset deltas are 16/24/32-bit; node_data_len width stored in header.
/// v3: compact count varints (1–2 bytes, capped at 10k per relation per node).
/// v2: zero-edge nodes write no counts or data; offsets may repeat (delta 0).
pub const LAYOUT_VERSION: u32 = 4;

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
    /// Number of bytes per delta in offsets.bin (2, 3, or 4).
    pub delta_bytes: u8,
    /// Number of bytes used to store node_data_len in edges.bin (2, 3, or 4).
    pub node_data_len_bytes: u8,
    /// Total edges encoded across all relations (optional; may be zero if not set).
    pub edge_count: u64,
}

impl OffsetFileHeader {
    pub fn new(nodes: u64, total_offsets: u64) -> Self {
        Self {
            version: LAYOUT_VERSION,
            nodes,
            total_offsets,
            delta_chunk: DELTA_CHUNK as u16,
            edge_blob_word_size: 1,
            delta_bytes: 2,
            node_data_len_bytes: 2,
            edge_count: 0,
        }
    }
}

/// Metadata captured per node while building the shard.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NodeEdgeStats {
    pub edge_count: u64,
    pub byte_len: u64,
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
    pub has_node_data: bool,
    pub node_data_len_bytes: u8,
}

impl<'a> EdgeListView<'a> {
    /// Returns the relation segments for this node. Each entry contains the
    /// edge count and raw slice for that relation's edges.
    pub fn relation_segments(&self) -> Result<Vec<RelationSegment<'a>>, std::io::Error> {
        Ok(self.parse()?.0)
    }

    pub fn node_data(&self) -> Result<Option<&'a [u8]>, std::io::Error> {
        Ok(self.parse()?.1)
    }

    pub fn pk_matches(
        &self,
        pk: &PrimaryKeyMap,
        decoder: &RowPackDecoder,
        key: &Key,
    ) -> Result<Option<bool>, std::io::Error> {
        let raw = match self.node_data()? {
            Some(raw) => raw,
            None => return Ok(None),
        };
        let ctx = decoder.context();
        let col_idx = match pk.pk_column_index.and_then(|idx| {
            ctx.columns
                .get(idx)
                .filter(|col| col.name == pk.column)
                .map(|_| idx)
        }) {
            Some(idx) => idx,
            None => match ctx.columns.iter().position(|col| col.name == pk.column) {
                Some(idx) => idx,
                None => return Ok(None),
            },
        };
        let mut scratch = RowPackDecodeScratch::default();
        let value = match decoder.decode_column_value(raw, col_idx, &mut scratch) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        Ok(Some(pk_value_matches(key, &value)))
    }

    fn parse(&self) -> Result<(Vec<RelationSegment<'a>>, Option<&'a [u8]>), std::io::Error> {
        if self.raw.is_empty() {
            if self.has_node_data {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            // Node has no edges; return empty segments for each relation.
            let segments = self
                .layouts
                .iter()
                .map(|layout| RelationSegment {
                    layout: layout.clone(),
                    count: 0,
                    data: &[],
                })
                .collect();
            return Ok((segments, None));
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
        let node_data_len = if self.has_node_data {
            let len_bytes = self.node_data_len_bytes as usize;
            if len_bytes == 0 || len_bytes > 4 {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            if cursor + len_bytes > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let len = read_len_le(&self.raw[cursor..cursor + len_bytes]);
            cursor += len_bytes;
            Some(len)
        } else {
            None
        };
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
        let node_data = if let Some(len) = node_data_len {
            let end = cursor
                .checked_add(len)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if end > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            if end != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            Some(&self.raw[cursor..end])
        } else {
            if cursor != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            None
        };
        Ok((segments, node_data))
    }
}

fn pk_value_matches(key: &Key, value: &ValueRef<'_>) -> bool {
    match (key, value) {
        (Key::Int(k), ValueRef::Int64(v)) => *v >= 0 && *k == *v as u64,
        (Key::Int(k), ValueRef::Int32(v)) => *v >= 0 && *k == *v as u64,
        (Key::Int(k), ValueRef::UInt64(v)) => *k == *v,
        (Key::Int(k), ValueRef::UInt32(v)) => *k == *v as u64,
        (Key::Bytes(k), ValueRef::Bytes(v)) => k.as_slice() == *v,
        (Key::Bytes(k), ValueRef::String(v)) => {
            let bytes = v.as_bytes();
            let trimmed = trim_trailing_nuls(bytes);
            k.as_slice() == trimmed
        }
        (Key::Uuid(k), ValueRef::Uuid(v)) => k == v,
        (Key::Uuid256(k), ValueRef::Uuid256(v)) => k == v,
        (Key::Ip(k), ValueRef::Ip(v)) => k == v,
        _ => false,
    }
}

fn trim_trailing_nuls(bytes: &[u8]) -> &[u8] {
    match bytes.iter().rposition(|b| *b != 0) {
        Some(idx) => &bytes[..=idx],
        None => &[],
    }
}

/// A relation-specific slice of edges for a node.
pub struct RelationSegment<'a> {
    pub layout: RelationLayout,
    pub count: u32,
    pub data: &'a [u8],
}

fn read_len_le(buf: &[u8]) -> usize {
    match buf.len() {
        2 => u16::from_le_bytes([buf[0], buf[1]]) as usize,
        3 => (buf[0] as u32 | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16)) as usize,
        4 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize,
        _ => 0,
    }
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
    // Extended counts: 3 or 4 bytes.
    if tag == 2 {
        if buf.len() < 3 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let b0 = (first & 0x3F) as u32;
        let b1 = buf[1] as u32;
        let b2 = buf[2] as u32;
        return Ok(((b0 << 16) | (b1 << 8) | b2, 3));
    }
    if tag == 3 {
        if buf.len() < 4 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let b0 = (first & 0x3F) as u32;
        let b1 = buf[1] as u32;
        let b2 = buf[2] as u32;
        let b3 = buf[3] as u32;
        return Ok(((b0 << 24) | (b1 << 16) | (b2 << 8) | b3, 4));
    }
    Err(std::io::ErrorKind::InvalidData.into())
}
