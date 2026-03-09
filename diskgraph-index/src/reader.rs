use crate::layout::{
    EdgeListView, OffsetFileHeader, RelationLayout, EDGES_FILENAME, OFFSETS_FILENAME,
};
use anyhow::Result;
use diskgraph_core::errors::DiskgraphError;
use libc::madvise;
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
    edge_count: u64,
    has_node_data: bool,
}

impl ShardReader {
    pub fn open(
        dir: &Path,
        layouts: Arc<Vec<RelationLayout>>,
        has_node_data: bool,
    ) -> Result<Self> {
        let offsets_path = dir.join(OFFSETS_FILENAME);
        let edges_path = dir.join(EDGES_FILENAME);
        let offsets_file = File::open(&offsets_path)?;
        let edges_file = File::open(&edges_path)?;
        let offsets = unsafe { Mmap::map(&offsets_file)? };
        let edges = unsafe { Mmap::map(&edges_file)? };
        advise_random(&edges);
        let header_len = bincode::serialized_size(&OffsetFileHeader::new(0, 0))
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?
            as usize;
        if offsets.len() < header_len {
            return Err(DiskgraphError::InvalidMetadata("invalid offsets header".into()).into());
        }
        let header: OffsetFileHeader = bincode::deserialize(&offsets[..header_len])
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?;
        if !matches!(header.delta_bytes, 2..=4) {
            return Err(
                DiskgraphError::InvalidMetadata("offset delta width invalid".into()).into(),
            );
        }
        if !matches!(header.node_data_len_bytes, 2..=4) {
            return Err(
                DiskgraphError::InvalidMetadata("node data len width invalid".into()).into(),
            );
        }
        let chunks = parse_offset_chunks(
            &offsets,
            header.total_offsets,
            header.delta_bytes,
            header_len,
        )?;
        let edge_count = header.edge_count;
        tracing::info!(
            dir = %dir.display(),
            edge_count,
            total_offsets = header.total_offsets,
            delta_bytes = header.delta_bytes,
            node_data_len_bytes = header.node_data_len_bytes,
            "Shard reader opened"
        );
        Ok(Self {
            offsets_mmap: offsets,
            edges,
            header,
            chunks,
            layouts,
            edge_count,
            has_node_data,
        })
    }

    pub fn header(&self) -> &OffsetFileHeader {
        &self.header
    }

    pub fn edge_count(&self) -> u64 {
        self.edge_count
    }

    pub fn offsets_len(&self) -> u64 {
        self.offsets_mmap.len() as u64
    }

    pub fn edges_len(&self) -> u64 {
        self.edges.len() as u64
    }

    /// Lookup a node's edge list by index within the shard.
    pub fn edges_for_node(&self, node_idx: u64) -> Result<EdgeListView<'_>> {
        if node_idx + 1 >= self.header.total_offsets {
            return Err(DiskgraphError::InvalidMetadata("node index out of range".into()).into());
        }
        let start = decode_offset(
            &self.offsets_mmap,
            &self.chunks,
            node_idx,
            self.header.delta_bytes,
        )?;
        let end = decode_offset(
            &self.offsets_mmap,
            &self.chunks,
            node_idx + 1,
            self.header.delta_bytes,
        )?;
        let start_byte = start as usize;
        let end_byte = end as usize;
        if end_byte > self.edges.len() || start_byte > end_byte {
            return Err(
                DiskgraphError::InvalidMetadata("offset points outside edge blob".into()).into(),
            );
        }
        let raw = &self.edges[start_byte..end_byte];
        Ok(EdgeListView {
            raw,
            layouts: self.layouts.clone(),
            has_node_data: self.has_node_data,
            node_data_len_bytes: self.header.node_data_len_bytes,
        })
    }
}

fn advise_random(map: &Mmap) {
    let ret = unsafe {
        madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_RANDOM,
        )
    };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}

fn parse_offset_chunks(
    offsets: &Mmap,
    total_offsets: u64,
    delta_bytes: u8,
    header_len: usize,
) -> Result<Vec<OffsetChunk>> {
    let mut chunks = Vec::new();
    let mut cursor = header_len;
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
        let chunk_bytes = 2 + 8 + (count.saturating_sub(1) as usize) * delta_bytes as usize;
        cursor += chunk_bytes;
        base_idx += count as u64;
    }
    Ok(chunks)
}

fn decode_offset(offsets: &Mmap, chunks: &[OffsetChunk], idx: u64, delta_bytes: u8) -> Result<u64> {
    let chunk = find_chunk(chunks, idx)?;
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
    match delta_bytes {
        2 => {
            for _ in 0..within {
                if cursor + 2 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
                acc = acc.checked_add(delta as u64).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 2;
            }
        }
        3 => {
            let mut remaining = within;
            while remaining >= 4 {
                if cursor + 12 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                for _ in 0..4 {
                    let delta = offsets[cursor] as u64
                        | ((offsets[cursor + 1] as u64) << 8)
                        | ((offsets[cursor + 2] as u64) << 16);
                    acc = acc.checked_add(delta).ok_or_else(|| {
                        DiskgraphError::InvalidMetadata("offset delta overflow".into())
                    })?;
                    cursor += 3;
                }
                remaining -= 4;
            }
            for _ in 0..remaining {
                if cursor + 3 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = offsets[cursor] as u64
                    | ((offsets[cursor + 1] as u64) << 8)
                    | ((offsets[cursor + 2] as u64) << 16);
                acc = acc.checked_add(delta).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 3;
            }
        }
        4 => {
            for _ in 0..within {
                if cursor + 4 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = u32::from_le_bytes([
                    offsets[cursor],
                    offsets[cursor + 1],
                    offsets[cursor + 2],
                    offsets[cursor + 3],
                ]);
                acc = acc.checked_add(delta as u64).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 4;
            }
        }
        _ => {
            return Err(
                DiskgraphError::InvalidMetadata("offset delta width invalid".into()).into(),
            );
        }
    }
    Ok(acc)
}

fn find_chunk(chunks: &[OffsetChunk], idx: u64) -> Result<&OffsetChunk> {
    if chunks.is_empty() {
        return Err(DiskgraphError::InvalidMetadata("no offset chunks".into()).into());
    }
    let chunk_idx = (idx / crate::layout::DELTA_CHUNK as u64) as usize;
    let chunk = chunks
        .get(chunk_idx)
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset chunk missing".into()))?;
    // Validate the chunk really covers this index.
    if idx < chunk.base_idx {
        return Err(DiskgraphError::InvalidMetadata("offset chunk range mismatch".into()).into());
    }
    if idx >= chunk.base_idx + chunk.count as u64 {
        return Err(DiskgraphError::InvalidMetadata("offset chunk range mismatch".into()).into());
    }
    Ok(chunk)
}
