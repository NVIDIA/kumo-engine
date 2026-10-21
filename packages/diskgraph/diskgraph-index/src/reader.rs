// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::layout::{
    EdgeListView, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use anyhow::Result;
use diskgraph_core::errors::DiskgraphError;
use libc::madvise;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Memory-mapped shard reader. Provides zero-copy edge list access by decoding
/// offsets and slicing into the edge data file.
pub struct ShardReader {
    offsets_mmap: Mmap,
    edges: Mmap,
    header: OffsetFileHeader,
    header_len: usize,
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
            header_len,
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
            self.header.delta_bytes,
            self.header_len,
            node_idx,
        )?;
        let end = decode_offset(
            &self.offsets_mmap,
            self.header.delta_bytes,
            self.header_len,
            node_idx + 1,
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

pub(crate) fn decode_offset(
    offsets: &[u8],
    delta_bytes: u8,
    header_len: usize,
    idx: u64,
) -> Result<u64> {
    let (mut cursor, within) = locate_chunk(offsets, delta_bytes, header_len, idx)?;
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

/// Chunks are fixed-size (`DELTA_CHUNK` offsets, last one partial), so a
/// chunk's byte position follows from its index without a directory.
fn locate_chunk(
    offsets: &[u8],
    delta_bytes: u8,
    header_len: usize,
    idx: u64,
) -> Result<(usize, u64)> {
    let chunk = DELTA_CHUNK as u64;
    let chunk_idx = idx / chunk;
    let stride = 10 + (chunk - 1) * delta_bytes as u64;
    let cursor = header_len as u64 + chunk_idx * stride;
    if cursor + 10 > offsets.len() as u64 {
        return Err(DiskgraphError::InvalidMetadata("offset chunk missing".into()).into());
    }
    let cursor = cursor as usize;
    let count = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]) as u64;
    let within = idx % chunk;
    if within >= count {
        return Err(DiskgraphError::InvalidMetadata("offset chunk range mismatch".into()).into());
    }
    Ok((cursor, within))
}
