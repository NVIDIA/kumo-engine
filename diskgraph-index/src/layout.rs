use serde::{Deserialize, Serialize};

/// Logical layout version for on-disk shard artifacts.
pub const LAYOUT_VERSION: u32 = 1;

/// Filename conventions for artifacts emitted per shard.
pub const OFFSETS_FILENAME: &str = "offsets.bin";
pub const EDGES_FILENAME: &str = "edges.bin";

/// Number of offsets to delta-pack in a chunk.
///
/// The offsets file stores:
/// - a header with version + counts
/// - a table of fixed-size chunks; each chunk stores the starting absolute
///   offset followed by `DELTA_CHUNK` u16 deltas to reconstruct the next
///   offsets. This balances size and decode cost.
pub const DELTA_CHUNK: usize = 64;

/// Header persisted at the start of the offsets file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OffsetFileHeader {
    pub version: u32,
    pub nodes: u64,
    pub total_offsets: u64,
    pub delta_chunk: u16,
    pub edge_blob_word_size: u16,
}

impl OffsetFileHeader {
    pub fn new(nodes: u64, total_offsets: u64) -> Self {
        Self {
            version: LAYOUT_VERSION,
            nodes,
            total_offsets,
            delta_chunk: DELTA_CHUNK as u16,
            // Edge blobs are expected to be 8-byte aligned and offsets point to
            // 8-byte words to keep metadata small.
            edge_blob_word_size: 8,
        }
    }
}

/// Metadata captured per node while building the shard.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeEdgeStats {
    pub edge_count: u32,
    pub byte_len: u32,
}

/// Logical view of an edge payload. The concrete encoding is left open; this
/// enum documents required fields and their intended bit-packing.
#[derive(Debug, Clone, Copy)]
pub enum EncodedEdgeFields {
    /// Only src/dst ids are present.
    Basic { dst: u64 },
    /// dest_sort_key is present.
    WithSortKey { dst: u64, dest_sort_key: u64 },
}

/// A view over an edge list stored in the edge data file. This is a lightweight
/// pointer into an mmapped region; decoding uses the on-disk layout rather than
/// copying data out.
pub struct EdgeListView<'a> {
    pub raw: &'a [u8],
    pub edge_count: u32,
    pub dest_sort_key_bits: Option<u8>,
}

impl<'a> EdgeListView<'a> {
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }
}
