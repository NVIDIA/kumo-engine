use crate::layout::{EdgeListView, NodeEdgeStats, OffsetFileHeader, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME};
use crate::streaming::{EdgeShardInput, NodeShardInput};
use anyhow::Result;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum IndexBuildError {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Parameters for building a single shard index.
#[derive(Debug, Clone)]
pub struct ShardBuildParams {
    pub output_dir: PathBuf,
    pub node_meta: NodeTypeMetadata,
    pub relation_meta: EdgeRelationMetadata,
    /// Shard index we are building (used for selecting aligned edge shards).
    pub shard_index: usize,
}

/// Consumes a node shard and its aligned edge shards and emits two files:
/// - offsets.bin: delta-encoded offsets into edges.bin
/// - edges.bin: edge payload blobs, 8-byte aligned
///
/// The actual encoding of edge payloads is stubbed until the compression
/// strategy is finalized.
pub struct ShardBuilder {
    params: ShardBuildParams,
}

impl ShardBuilder {
    pub fn new(params: ShardBuildParams) -> Self {
        Self { params }
    }

    /// Build the shard index. The incoming node shard and edge shards are
    /// currently assumed to be local Parquet files (see `ParquetSource`), but
    /// the streaming layer can be extended to other protocols without changing
    /// this signature.
    pub fn build(
        &self,
        _node_shard: &NodeShardInput,
        _edge_shards: &[EdgeShardInput],
    ) -> Result<ShardBuildOutput> {
        fs::create_dir_all(&self.params.output_dir)?;
        let offsets_path = self.params.output_dir.join(OFFSETS_FILENAME);
        let edges_path = self.params.output_dir.join(EDGES_FILENAME);

        // TODO: stream Parquet rows, group by src_id, encode offsets + edge blobs.
        // For now, write minimal headers to keep the files well-formed.
        self.write_placeholder_offsets(&offsets_path)?;
        self.write_placeholder_edges(&edges_path)?;

        info!(
            "built shard {} for node type {} relation {}",
            self.params.shard_index, self.params.node_meta.name, self.params.relation_meta.name
        );
        Ok(ShardBuildOutput {
            offsets_path,
            edges_path,
            stats: NodeEdgeStats::default(),
        })
    }

    fn write_placeholder_offsets(&self, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        let header = OffsetFileHeader::new(0, 0);
        let bytes = bincode::serialize(&header).map_err(|_| IndexBuildError::NotImplemented("offset encoding"))?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn write_placeholder_edges(&self, path: &Path) -> Result<()> {
        File::create(path)?;
        Ok(())
    }
}

/// Output of a shard build, including basic statistics.
pub struct ShardBuildOutput {
    pub offsets_path: PathBuf,
    pub edges_path: PathBuf,
    pub stats: NodeEdgeStats,
}

/// Helper for delta-encoding offsets in fixed-size chunks. Not yet wired into
/// the builder; provided to document the intended layout.
pub fn delta_encode_offsets(offsets: &[u64]) -> Result<Vec<u8>> {
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    // Chunk into DELTA_CHUNK groups; each group stores base + u16 deltas.
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < offsets.len() {
        let chunk_end = (idx + DELTA_CHUNK).min(offsets.len());
        let base = offsets[idx];
        out.extend_from_slice(&base.to_le_bytes());
        let mut prev = base;
        for &off in &offsets[idx + 1..chunk_end] {
            let delta = off
                .checked_sub(prev)
                .ok_or(IndexBuildError::NotImplemented("offsets must be non-decreasing"))?;
            let delta_u16: u16 = delta.try_into().map_err(|_| IndexBuildError::NotImplemented("delta too large for u16"))?;
            out.extend_from_slice(&delta_u16.to_le_bytes());
            prev = off;
        }
        idx = chunk_end;
    }
    Ok(out)
}

/// Placeholder for edge blob encoding. The final implementation will pack
/// destination ids and optional dest_sort_key into a tight format.
pub fn encode_edge_blob(_edges: &[EdgeListView<'_>]) -> Result<Vec<u8>> {
    Err(IndexBuildError::NotImplemented("edge blob encoding not implemented"))?
}
