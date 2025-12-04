use crate::layout::{EdgeListView, OffsetFileHeader, OFFSETS_FILENAME, EDGES_FILENAME};
use anyhow::Result;
use diskgraph_core::errors::DiskgraphError;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Memory-mapped shard reader. Provides zero-copy edge list access by decoding
/// offsets and slicing into the edge data file.
pub struct ShardReader {
    _offsets: Mmap,
    edges: Mmap,
    header: OffsetFileHeader,
}

impl ShardReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let offsets_path = dir.join(OFFSETS_FILENAME);
        let edges_path = dir.join(EDGES_FILENAME);
        let offsets_file = File::open(&offsets_path)?;
        let edges_file = File::open(&edges_path)?;
        let offsets = unsafe { Mmap::map(&offsets_file)? };
        let edges = unsafe { Mmap::map(&edges_file)? };
        let header: OffsetFileHeader = bincode::deserialize(&offsets[..std::mem::size_of::<OffsetFileHeader>()])
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?;
        Ok(Self {
            _offsets: offsets,
            edges,
            header,
        })
    }

    pub fn header(&self) -> &OffsetFileHeader {
        &self.header
    }

    /// Lookup a node's edge list by index within the shard. This is a stub that
    /// returns an empty view until the offset decoding is implemented.
    pub fn edges_for_node(&self, _node_idx: u64, dest_sort_key_bits: Option<u8>) -> Result<EdgeListView<'_>> {
        Ok(EdgeListView {
            raw: &self.edges[..0],
            edge_count: 0,
            dest_sort_key_bits,
        })
    }
}
