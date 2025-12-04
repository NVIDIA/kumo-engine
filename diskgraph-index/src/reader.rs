use crate::layout::{EdgeListView, OffsetFileHeader, RelationLayout, DELTA_CHUNK, OFFSETS_FILENAME, EDGES_FILENAME};
use anyhow::Result;
use diskgraph_core::errors::DiskgraphError;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
struct OffsetChunk {
    base_idx: u64,
    count: u16,
    data_offset: usize,
}

/// Memory-mapped shard reader. Provides zero-copy edge list access by decoding
/// offsets and slicing into the edge data file.
pub struct ShardReader {
    offsets_mmap: Mmap,
    edges: Mmap,
    header: OffsetFileHeader,
    chunks: Vec<OffsetChunk>,
    layouts: Arc<Vec<RelationLayout>>,
}

impl ShardReader {
    pub fn open(
        dir: &Path,
        layouts: Arc<Vec<RelationLayout>>,
    ) -> Result<Self> {
        let offsets_path = dir.join(OFFSETS_FILENAME);
        let edges_path = dir.join(EDGES_FILENAME);
        let offsets_file = File::open(&offsets_path)?;
        let edges_file = File::open(&edges_path)?;
        let offsets = unsafe { Mmap::map(&offsets_file)? };
        let edges = unsafe { Mmap::map(&edges_file)? };
        let header: OffsetFileHeader = bincode::deserialize(&offsets[..std::mem::size_of::<OffsetFileHeader>()])
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?;
        let chunks = parse_offset_chunks(&offsets, header.total_offsets)?;
        Ok(Self {
            offsets_mmap: offsets,
            edges,
            header,
            chunks,
            layouts,
        })
    }

    pub fn header(&self) -> &OffsetFileHeader {
        &self.header
    }

    /// Lookup a node's edge list by index within the shard.
    pub fn edges_for_node(&self, node_idx: u64) -> Result<EdgeListView<'_>> {
        if node_idx + 1 >= self.header.total_offsets {
            return Err(DiskgraphError::InvalidMetadata("node index out of range".into()).into());
        }
        let start = decode_offset(&self.offsets_mmap, &self.chunks, node_idx)?;
        let end = decode_offset(&self.offsets_mmap, &self.chunks, node_idx + 1)?;
        let start_byte = start as usize;
        let end_byte = end as usize;
        if end_byte > self.edges.len() || start_byte > end_byte {
            return Err(DiskgraphError::InvalidMetadata("offset points outside edge blob".into()).into());
        }
        let raw = &self.edges[start_byte..end_byte];
        Ok(EdgeListView {
            raw,
            relation_bits: 0, // relation id not stored per entry
            entry_bytes: 0,   // per relation entry sizes used via layouts
            layouts: self.layouts.clone(),
        })
    }
}

fn parse_offset_chunks(offsets: &Mmap, total_offsets: u64) -> Result<Vec<OffsetChunk>> {
    let mut chunks = Vec::new();
    let mut cursor = std::mem::size_of::<OffsetFileHeader>();
    let mut base_idx = 0u64;
    while (cursor as u64) < offsets.len() as u64 && base_idx < total_offsets {
        if cursor + 2 + 8 > offsets.len() {
            break;
        }
        let count = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
        let base_offset_bytes = &offsets[cursor + 2..cursor + 10];
        let _base = u64::from_le_bytes(base_offset_bytes.try_into().unwrap());
        chunks.push(OffsetChunk {
            base_idx,
            count,
            data_offset: cursor,
        });
        let chunk_bytes = 2 + 8 + (count.saturating_sub(1) as usize) * 2;
        cursor += chunk_bytes;
        base_idx += count as u64;
    }
    Ok(chunks)
}

fn decode_offset(offsets: &Mmap, chunks: &[OffsetChunk], idx: u64) -> Result<u64> {
    let chunk_idx = idx as usize / DELTA_CHUNK;
    let chunk = chunks
        .get(chunk_idx)
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset chunk missing".into()))?;
    if idx < chunk.base_idx || idx >= chunk.base_idx + chunk.count as u64 {
        return Err(DiskgraphError::InvalidMetadata("offset chunk range mismatch".into()).into());
    }
    let within = idx - chunk.base_idx;
    let mut cursor = chunk.data_offset;
    let count = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
    if count != chunk.count {
        return Err(DiskgraphError::InvalidMetadata("offset chunk count mismatch".into()).into());
    }
    cursor += 2;
    let base_bytes: [u8; 8] = offsets[cursor..cursor + 8]
        .try_into()
        .map_err(|_| DiskgraphError::InvalidMetadata("offset base slice error".into()))?;
    let base = u64::from_le_bytes(base_bytes);
    if within == 0 {
        return Ok(base);
    }
    cursor += 8;
    let mut acc = base;
    for _ in 0..within {
        if cursor + 2 > offsets.len() {
            return Err(DiskgraphError::InvalidMetadata("offset delta out of bounds".into()).into());
        }
        let delta = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
        acc = acc
            .checked_add(delta as u64)
            .ok_or_else(|| DiskgraphError::InvalidMetadata("offset delta overflow".into()))?;
        cursor += 2;
    }
    Ok(acc)
}
