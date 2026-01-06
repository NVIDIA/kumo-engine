use crate::layout::{
    NodeEdgeStats, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use crate::streaming::{EdgeShardInput, ParquetSource};
use crate::truncate::truncate_counts;
use anyhow::Result;
use arrow_array::{BinaryArray, Int64Array};
use arrow_schema::{DataType, SchemaRef};
use bytes::Bytes;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use futures::{future::BoxFuture, StreamExt};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use serde_json;
use serde::Serialize;
use std::sync::mpsc::{self, Receiver};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::cell::RefCell;
use thiserror::Error;
use tokio::runtime::Runtime;
use tracing::info;

/// Hard cap on edges per relation for a single node to keep varint encoding small.
/// Compact varint supports up to 14 bits (0x3FFF). We cap relation edge lists
/// at that limit to guarantee counts fit the encoding without overflow.
const MAX_RELATION_EDGE_COUNT: usize = 0x3FFF;
const MAX_NODE_DATA_LEN: usize = 60 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildStage {
    BuildIndex,
}

#[derive(Clone, Debug)]
pub struct BuildProgress {
    pub stage: BuildStage,
    pub current: u64,
    pub total: Option<u64>,
}

pub trait ProgressSink {
    fn report(&self, update: BuildProgress);
}

pub struct ProgressReporter<F: FnMut(BuildProgress)> {
    inner: RefCell<F>,
}

impl<F: FnMut(BuildProgress)> ProgressReporter<F> {
    pub fn new(inner: F) -> Self {
        Self {
            inner: RefCell::new(inner),
        }
    }
}

impl<F: FnMut(BuildProgress)> ProgressSink for ProgressReporter<F> {
    fn report(&self, update: BuildProgress) {
        (self.inner.borrow_mut())(update);
    }
}

type PrefetchBatch =
    std::result::Result<Option<arrow_array::RecordBatch>, parquet::errors::ParquetError>;

#[derive(Debug, Error)]
pub enum IndexBuildError {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("offset delta exceeds u16 max")]
    OffsetDeltaOverflow,
    #[error("edge or byte count exceeds supported limit")]
    CountOverflow,
    #[error("value out of range for {0}")]
    ValueOutOfRange(&'static str),
    #[error("edges remain after exhausting node ids (misaligned shards)")]
    DanglingEdges,
    #[error("input edges not sorted by src_id")]
    EdgeOrderingViolation,
    #[error("invalid column type for {0}")]
    InvalidColumn(&'static str),
    #[error("unknown relation in shard input")]
    UnknownRelation,
    #[error("shard index mismatch (expected {node}, got {edge})")]
    ShardIndexMismatch { node: usize, edge: usize },
    #[error("node shard index {index} missing from metadata")]
    NodeShardMissing { index: usize },
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("build aborted")]
    Aborted,
}
fn localize_edge_inputs(
    edge_shards: &[EdgeShardInput],
    node_data: Option<&[ParquetSource]>,
) -> Result<(Vec<EdgeShardInput>, Option<Vec<ParquetSource>>)> {
    Ok((edge_shards.to_vec(), node_data.map(|paths| paths.to_vec())))
}

/// Parameters for building a single shard index.
#[derive(Debug, Clone)]
pub struct ShardBuildParams {
    pub output_dir: PathBuf,
    pub node_meta: NodeTypeMetadata,
    /// All node type metadata, used to derive destination id bit widths per relation.
    pub node_types: Vec<NodeTypeMetadata>,
    pub relation_metas: Vec<EdgeRelationMetadata>,
    /// Shard index we are building (used for selecting aligned edge shards).
    pub shard_index: usize,
    /// Optional node data Parquet source for this shard.
    pub node_data_paths: Option<Vec<ParquetSource>>,
}

/// Consumes node shard metadata and all aligned edge shards (across relations) and emits two files:
/// - offsets.bin: delta-encoded byte offsets into edges.bin
/// - edges.bin: packed edge payload blobs (relation id + dst id + optional dest_sort_key)
///
/// Entry layout:
/// [relation_id bits][dst_id bits (per relation)][dest_sort_key bits (optional per relation)]
pub struct ShardBuilder {
    params: ShardBuildParams,
}

impl ShardBuilder {
    pub fn new(params: ShardBuildParams) -> Self {
        Self { params }
    }

    pub fn build(&self, edge_shards: &[EdgeShardInput]) -> Result<ShardBuildOutput> {
        self.build_with_progress(edge_shards, None, None, None)
    }

    pub fn build_with_progress(
        &self,
        edge_shards: &[EdgeShardInput],
        total_nodes: Option<u64>,
        progress: Option<&dyn ProgressSink>,
        should_abort: Option<&dyn Fn() -> bool>,
    ) -> Result<ShardBuildOutput> {
        for e in edge_shards {
            if e.source_shard_index != self.params.shard_index {
                return Err(IndexBuildError::ShardIndexMismatch {
                    node: self.params.shard_index,
                    edge: e.source_shard_index,
                }
                .into());
            }
        }
        fs::create_dir_all(&self.params.output_dir)?;
        let offsets_path = self.params.output_dir.join(OFFSETS_FILENAME);
        let edges_path = self.params.output_dir.join(EDGES_FILENAME);

        let rt = Arc::new(Runtime::new()?);
        let (edge_inputs, localized_node_data) =
            localize_edge_inputs(edge_shards, self.params.node_data_paths.as_deref())?;

        let layouts = Arc::new(compute_relation_layouts(
            &self.params.relation_metas,
            &self.params.node_types,
        )?);

        let mut node_stream =
            NodeStream::try_new(&self.params.node_meta, self.params.shard_index)?;
        let mut edge_stream =
            MergedEdgeStream::try_new(rt.clone(), &edge_inputs, &self.params.relation_metas)?;
        let expected_nodes = shard_node_count(&self.params.node_meta, self.params.shard_index)?;
        if !self.params.node_meta.has_node_data && self.params.node_data_paths.is_some() {
            return Err(IndexBuildError::NotImplemented(
                "node data provided but metadata disables it",
            )
            .into());
        }
        let mut node_data_stream = if self.params.node_meta.has_node_data {
            let sources = localized_node_data
                .as_ref()
                .ok_or(IndexBuildError::NotImplemented("node data missing for shard"))?;
            Some(NodeDataStream::try_new(
                rt.clone(),
                sources,
                expected_nodes,
            )?)
        } else {
            None
        };
        let mut node_data_buf: Vec<u8> = Vec::new();

        let mut offsets_writer = OffsetChunkWriter::try_new(&offsets_path)?;
        let mut edges_writer = EdgeWriter::try_new(&edges_path, layouts.clone())?;
        let mut node_buf: Vec<u8> = Vec::new();

        let mut nodes_seen = 0u64;
        let mut build_stats = BuildStats {
            truncated_edges_per_relation: vec![0; self.params.relation_metas.len()],
            ..Default::default()
        };
        let mut prev_offset: u64 = 0;
        let mut last_progress_at: u64 = 0;
        // Seed offsets with the start of the first node (0) to make start/end
        // lookups simpler on the read path.
        offsets_writer.push(edges_writer.byte_offset())?;
        // Reuse allocations for buckets and counts to avoid per-node Vec allocations.
        let rel_count = self.params.relation_metas.len();
        let mut buckets: Vec<Vec<EdgeRec>> = vec![Vec::new(); rel_count];
        let mut counts: Vec<usize> = vec![0; rel_count];
        let mut last_counts: Vec<usize> = vec![0; rel_count];
        let mut original_counts: Vec<usize> = vec![0; rel_count];
        let mut lens: Vec<usize> = vec![0; rel_count];
        let result = (|| {
            if let Some(progress) = progress {
                progress.report(BuildProgress {
                    stage: BuildStage::BuildIndex,
                    current: 0,
                    total: total_nodes,
                });
            }
            while let Some(node_id) = node_stream.next_id()? {
            if let Some(should_abort) = should_abort {
                if should_abort() {
                    return Err(IndexBuildError::Aborted.into());
                }
            }
            nodes_seen += 1;
            build_stats.nodes_ingested += 1;
            if let Some(progress) = progress {
                if nodes_seen - last_progress_at >= 100_000 {
                    progress.report(BuildProgress {
                        stage: BuildStage::BuildIndex,
                        current: nodes_seen,
                        total: total_nodes,
                    });
                    last_progress_at = nodes_seen;
                }
            }
            let start_offset = edges_writer.byte_offset();
            let mut node_data_len: Option<usize> = None;
            if let Some(stream) = node_data_stream.as_mut() {
                let len = stream.next_into(&mut node_data_buf)?;
                if len > MAX_NODE_DATA_LEN {
                    node_data_buf.clear();
                    build_stats.node_data_trimmed =
                        build_stats.node_data_trimmed.saturating_add(1);
                } else {
                    node_data_len = Some(len);
                }
            }
            // Collect edges for this node by relation.
            for (i, bucket) in buckets.iter_mut().enumerate() {
                bucket.clear();
                if last_counts[i] > bucket.capacity() {
                    bucket.reserve_exact(last_counts[i] - bucket.capacity());
                }
            }
            let mut truncated_node = false;
            edge_stream.drain_for_node(node_id, &mut buckets)?;
            // Compute lengths first to avoid borrowing conflicts, then apply truncation.
            let mut any_truncated = false;
            let mut total_edges: usize = 0;
            for i in 0..rel_count {
                let len = buckets[i].len();
                lens[i] = len;
                counts[i] = len;
                total_edges = total_edges.saturating_add(len);
                if len > MAX_RELATION_EDGE_COUNT {
                    buckets[i].truncate(MAX_RELATION_EDGE_COUNT);
                    counts[i] = MAX_RELATION_EDGE_COUNT;
                    total_edges = total_edges.saturating_sub(len - MAX_RELATION_EDGE_COUNT);
                    truncated_node = true;
                    any_truncated = true;
                    build_stats.capped_edge_lists = build_stats.capped_edge_lists.saturating_add(1);
                }
            }
            if any_truncated {
                original_counts.copy_from_slice(&lens);
            }
            if total_edges == 0 && node_data_len.is_none() {
                // No edges for this node across all relations; avoid writing zero counts.
                offsets_writer.push(start_offset)?;
                prev_offset = start_offset;
                continue;
            }
            let extra_bytes = node_data_len
                .map(|len| 2u64.saturating_add(len as u64))
                .unwrap_or(0);
            let node_bytes = EdgeWriter::node_length_bytes_fast(&layouts, &counts)
                .saturating_add(extra_bytes);

            // Determine if we need to truncate to stay within u16 delta budget.
            let base_span = start_offset
                .checked_sub(prev_offset)
                .ok_or(IndexBuildError::CountOverflow)?;
            let budget = (u16::MAX as u64)
                .checked_sub(base_span)
                .ok_or(IndexBuildError::OffsetDeltaOverflow)?;
            if node_bytes > budget {
                if !any_truncated {
                    original_counts.copy_from_slice(&lens);
                    any_truncated = true;
                }
                truncate_counts(
                    &mut buckets,
                    &mut counts,
                    &layouts,
                    start_offset,
                    prev_offset,
                    extra_bytes,
                )?;
                truncated_node = true;
            }

            if truncated_node {
                build_stats.truncated_nodes = build_stats.truncated_nodes.saturating_add(1);
            }
            // Write all counts + edge data for this node in a single buffer.
            for (rel_idx, target_len) in counts.iter().enumerate() {
                let truncated = if any_truncated {
                    original_counts[rel_idx].saturating_sub(*target_len)
                } else {
                    0
                };
                if truncated > 0 {
                    build_stats.truncated_edges_total += truncated as u64;
                    build_stats.truncated_edges_per_relation[rel_idx] = build_stats
                        .truncated_edges_per_relation[rel_idx]
                        .saturating_add(truncated as u64);
                }
            }
            let node_data_slice = node_data_len.map(|_| node_data_buf.as_slice());
            edges_writer.write_node(
                &counts,
                &buckets,
                total_edges,
                &mut node_buf,
                node_data_slice,
            )?;
            build_stats.edges_ingested =
                build_stats.edges_ingested.saturating_add(total_edges as u64);
            offsets_writer.push(edges_writer.byte_offset())?;
            prev_offset = edges_writer.byte_offset();
            last_counts.copy_from_slice(&counts);
        }

        if let Some(progress) = progress {
            progress.report(BuildProgress {
                stage: BuildStage::BuildIndex,
                current: nodes_seen,
                total: total_nodes.or(Some(nodes_seen)),
            });
        }

        if edge_stream.peek_any().is_some() {
            return Err(IndexBuildError::DanglingEdges.into());
        }

        offsets_writer.finish(nodes_seen, edges_writer.edge_count)?;
        let stats = NodeEdgeStats {
            edge_count: edges_writer.edge_count,
            byte_len: edges_writer.bytes_written,
        };
        edges_writer.finish()?;
        let report = BuildReport {
            node_edge_stats: stats,
            build_stats: build_stats.clone(),
        };
        let report_path = self.params.output_dir.join("build_stats.json");
        fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;

        info!(
            "built shard {} for node type {} (relations: {}, nodes={}, edges={}, bytes={})",
            self.params.shard_index,
            self.params.node_meta.name,
            self.params.relation_metas.len(),
            nodes_seen,
            stats.edge_count,
            stats.byte_len
        );

            Ok(ShardBuildOutput {
                offsets_path,
                edges_path,
                stats,
                layouts,
                build_stats,
            })
        })();
        result
    }
}

/// Output of a shard build, including basic statistics.
pub struct ShardBuildOutput {
    pub offsets_path: PathBuf,
    pub edges_path: PathBuf,
    pub stats: NodeEdgeStats,
    pub build_stats: BuildStats,
    pub layouts: Arc<Vec<RelationLayout>>,
}

/// Counters gathered during build. These are non-fatal events that help
/// surface warning conditions to callers.
#[derive(Debug, Default, Clone, Serialize)]
pub struct BuildStats {
    pub nodes_ingested: u64,
    pub edges_ingested: u64,
    pub truncated_nodes: u64,
    pub truncated_edges_total: u64,
    pub truncated_edges_per_relation: Vec<u64>,
    /// Number of per-relation edge lists that were capped at MAX_RELATION_EDGE_COUNT.
    pub capped_edge_lists: u64,
    /// Number of nodes whose payload exceeded MAX_NODE_DATA_LEN and was dropped.
    pub node_data_trimmed: u64,
}

#[derive(Serialize)]
struct BuildReport {
    node_edge_stats: NodeEdgeStats,
    build_stats: BuildStats,
}

pub fn compute_relation_layouts(
    relation_metas: &[EdgeRelationMetadata],
    node_types: &[NodeTypeMetadata],
) -> Result<Vec<RelationLayout>> {
    let mut out = Vec::new();
    for (i, rel) in relation_metas.iter().enumerate() {
        let dst_bits = node_types
            .iter()
            .find(|n| n.name == rel.dst_type)
            .map(|n| bits_for_id(n.max_id))
            .unwrap_or(64);
        let dst_mask = if dst_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << dst_bits) - 1
        };
        let sort_bits = rel.dest_sort_key_bits;
        let dest_sort_key_mask = sort_bits.map(|bits| {
            if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            }
        });
        out.push(RelationLayout {
            relation_id: i as u16,
            dst_bits,
            dest_sort_key_bits: rel.dest_sort_key_bits,
            entry_bytes: ((dst_bits as usize + rel.dest_sort_key_bits.unwrap_or(0) as usize + 7)
                / 8)
            .max(1),
            dst_mask,
            dest_sort_key_mask,
        });
    }
    Ok(out)
}

fn bits_for_id(max_id: u64) -> u8 {
    if max_id == 0 {
        1
    } else {
        64 - max_id.leading_zeros() as u8
    }
}

/// Delta-encode offsets in fixed-size chunks. Each chunk is stored as:
/// [u16 count][u64 base][(count-1) * u16 deltas].
fn delta_encode_offsets_into(offsets: &[u64], out: &mut Vec<u8>) -> Result<()> {
    if offsets.is_empty() {
        return Ok(());
    }
    let mut idx = 0;
    while idx < offsets.len() {
        let chunk_end = (idx + DELTA_CHUNK).min(offsets.len());
        let base = offsets[idx];
        let count = (chunk_end - idx) as u16;
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&base.to_le_bytes());
        let mut prev = base;
        for &off in &offsets[idx + 1..chunk_end] {
            let delta = off
                .checked_sub(prev)
                .ok_or(IndexBuildError::EdgeOrderingViolation)?;
            let delta_u16: u16 = delta
                .try_into()
                .map_err(|_| IndexBuildError::OffsetDeltaOverflow)?;
            out.extend_from_slice(&delta_u16.to_le_bytes());
            prev = off;
        }
        idx = chunk_end;
    }
    Ok(())
}

struct OffsetChunkWriter {
    pub(crate) writer: BufWriter<File>,
    chunk_buf: Vec<u64>,
    total_offsets: u64,
}

impl OffsetChunkWriter {
    fn try_new(path: &Path) -> Result<Self> {
        // Use a larger buffer to reduce flushes on many small writes.
        let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        let header_len =
            bincode::serialized_size(&OffsetFileHeader::new(0, 0)).unwrap_or(32) as usize;
        writer.write_all(&vec![0u8; header_len])?;
        Ok(Self {
            writer,
            chunk_buf: Vec::with_capacity(DELTA_CHUNK),
            total_offsets: 0,
        })
    }

    fn push(&mut self, offset: u64) -> Result<()> {
        self.chunk_buf.push(offset);
        self.total_offsets += 1;
        if self.chunk_buf.len() == DELTA_CHUNK {
            self.flush_chunk(Vec::new())?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self, mut scratch: Vec<u8>) -> Result<()> {
        if self.chunk_buf.is_empty() {
            return Ok(());
        }
        scratch.clear();
        delta_encode_offsets_into(&self.chunk_buf, &mut scratch)?;
        self.writer.write_all(&scratch)?;
        self.chunk_buf.clear();
        Ok(())
    }

    fn finish(mut self, node_count: u64, edge_count: u64) -> Result<u64> {
        self.flush_chunk(Vec::new())?;
        self.writer.flush()?;
        let mut header = OffsetFileHeader::new(node_count, self.total_offsets);
        header.edge_count = edge_count;
        let header_bytes = bincode::serialize(&header)
            .map_err(|_| IndexBuildError::NotImplemented("offset encoding"))?;
        let mut file = self.writer.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header_bytes)?;
        Ok(self.total_offsets)
    }
}

pub(crate) struct EdgeWriter {
    writer: BufWriter<File>,
    bytes_written: u64,
    edge_count: u64,
    layouts: Arc<Vec<RelationLayout>>,
}

impl EdgeWriter {
    fn try_new(path: &Path, layouts: Arc<Vec<RelationLayout>>) -> Result<Self> {
        // Larger buffer to reduce syscalls while writing large edge blobs.
        let writer = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        Ok(Self {
            writer,
            bytes_written: 0,
            edge_count: 0,
            layouts,
        })
    }

    fn write_node(
        &mut self,
        counts: &[usize],
        buckets: &[Vec<EdgeRec>],
        total_edges: usize,
        scratch: &mut Vec<u8>,
        node_data: Option<&[u8]>,
    ) -> Result<()> {
        scratch.clear();
        let edge_bytes = EdgeWriter::node_length_bytes_fast(&self.layouts, counts) as usize;
        let node_bytes = if let Some(node_data) = node_data {
            edge_bytes
                .checked_add(2)
                .and_then(|v| v.checked_add(node_data.len()))
                .ok_or(IndexBuildError::CountOverflow)?
        } else {
            edge_bytes
        };
        if node_bytes > scratch.capacity() {
            scratch.reserve(node_bytes - scratch.capacity());
        }
        for &count in counts.iter() {
            debug_assert!(count <= MAX_RELATION_EDGE_COUNT);
            if count <= 63 {
                scratch.push((count & 0x3F) as u8);
            } else {
                let b0 = 0x40 | (((count >> 8) & 0x3F) as u8);
                let b1 = (count & 0xFF) as u8;
                scratch.push(b0);
                scratch.push(b1);
            }
        }
        if let Some(node_data) = node_data {
            let len: u16 = node_data
                .len()
                .try_into()
                .map_err(|_| IndexBuildError::ValueOutOfRange("node_data_len"))?;
            scratch.extend_from_slice(&len.to_le_bytes());
        }
        for (rel_idx, (layout, rel_edges)) in
            self.layouts.iter().zip(buckets.iter()).enumerate()
        {
            let target_len = counts[rel_idx];
            for edge in rel_edges.iter().take(target_len) {
                if edge.dst > layout.dst_mask {
                    return Err(IndexBuildError::ValueOutOfRange("dst_id").into());
                }
                debug_assert!(
                    (layout.dst_bits as usize
                        + layout.dest_sort_key_bits.unwrap_or(0) as usize)
                        <= layout.entry_bytes * 8
                );
                let mut acc: u128 = 0;
                let mut bit_cursor = 0usize;
                let dst_val = edge.dst & layout.dst_mask;
                acc |= (dst_val as u128) << bit_cursor;
                bit_cursor += layout.dst_bits as usize;
                if layout.dest_sort_key_bits.is_some() {
                    let mask = layout.dest_sort_key_mask.unwrap_or(u64::MAX);
                    let raw_sort = edge.dest_sort_key.unwrap_or(0);
                    if raw_sort > mask {
                        return Err(IndexBuildError::ValueOutOfRange("dest_sort_key").into());
                    }
                    let sort_val = edge.dest_sort_key.unwrap_or(0) & mask;
                    acc |= (sort_val as u128) << bit_cursor;
                }
                let bytes_needed = layout.entry_bytes.min(16);
                let bytes = acc.to_le_bytes();
                scratch.extend_from_slice(&bytes[..bytes_needed]);
            }
        }
        if let Some(node_data) = node_data {
            scratch.extend_from_slice(node_data);
        }
        debug_assert!(scratch.len() == node_bytes);
        self.writer.write_all(scratch)?;
        self.bytes_written = self
            .bytes_written
            .checked_add(scratch.len() as u64)
            .ok_or(IndexBuildError::CountOverflow)?;
        self.edge_count = self
            .edge_count
            .checked_add(total_edges as u64)
            .ok_or(IndexBuildError::CountOverflow)?;
        Ok(())
    }

    /// Compute total byte length for a node given relation counts.
    pub(crate) fn node_length_bytes(
        layouts: &[RelationLayout],
        counts: &[usize],
    ) -> std::result::Result<u64, IndexBuildError> {
        let mut len: u64 = 0;
        for (layout, &count) in layouts.iter().zip(counts.iter()) {
            let count_len = if count <= 63 { 1 } else { 2 };
            len = len
                .checked_add(count_len)
                .ok_or(IndexBuildError::CountOverflow)?;
            len = len
                .checked_add(
                    (count as u64)
                        .checked_mul(layout.entry_bytes as u64)
                        .ok_or(IndexBuildError::CountOverflow)?,
                )
                .ok_or(IndexBuildError::CountOverflow)?;
        }
        Ok(len)
    }

    #[inline]
    pub(crate) fn node_length_bytes_fast(layouts: &[RelationLayout], counts: &[usize]) -> u64 {
        let mut len: u64 = 0;
        for (layout, &count) in layouts.iter().zip(counts.iter()) {
            let count_len = if count <= 63 { 1 } else { 2 };
            len = len
                .saturating_add(count_len)
                .saturating_add((count as u64).saturating_mul(layout.entry_bytes as u64));
        }
        len
    }

    fn byte_offset(&self) -> u64 {
        self.bytes_written
    }

    fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EdgeRec {
    src: u64,
    dst: u64,
    dest_sort_key: Option<u64>,
    relation_id: u16,
}

struct NodeStream {
    next_id: u64,
    max_id: u64,
    done: bool,
}

impl NodeStream {
    fn try_new(node_meta: &NodeTypeMetadata, shard_index: usize) -> Result<Self> {
        let shard = node_meta
            .shards
            .get(shard_index)
            .ok_or(IndexBuildError::NodeShardMissing { index: shard_index })?;
        Ok(Self {
            next_id: shard.min_id,
            max_id: shard.max_id,
            done: false,
        })
    }

    fn next_id(&mut self) -> Result<Option<u64>> {
        if self.done {
            return Ok(None);
        }
        if self.next_id > self.max_id {
            self.done = true;
            return Ok(None);
        }
        let val = self.next_id;
        if self.next_id == self.max_id {
            self.done = true;
        } else {
            self.next_id += 1;
        }
        Ok(Some(val))
    }
}

pub(crate) fn shard_node_count(node_meta: &NodeTypeMetadata, shard_index: usize) -> Result<u64> {
    let shard = node_meta
        .shards
        .get(shard_index)
        .ok_or(IndexBuildError::NodeShardMissing { index: shard_index })?;
    if shard.max_id < shard.min_id {
        return Ok(0);
    }
    Ok(shard.max_id - shard.min_id + 1)
}

enum EdgeSubStream {
    Parquet(ParquetEdgeSubStream),
    Binary(BinaryEdgeSubStream),
}

struct ParquetEdgeSubStream {
    _schema: SchemaRef,
    batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    batch_len: usize,
    dest_sort_key_bits: Option<u8>,
    relation_id: u16,
    src_idx: usize,
    dst_idx: usize,
    sort_idx: Option<usize>,
    current_src: Option<Int64Array>,
    current_dst: Option<Int64Array>,
    current_sort: Option<Int64Array>,
    rx: Receiver<PrefetchBatch>,
}

struct BinaryEdgeRecord {
    src_id: u64,
    dst_id: u64,
    sort_key: Option<u64>,
}

struct BinaryEdgeSubStream {
    reader: std::io::BufReader<std::fs::File>,
    remaining: u64,
    relation_id: u16,
    with_sort: bool,
    src_width: u8,
    dst_width: u8,
    sort_width: u8,
    next_record: Option<BinaryEdgeRecord>,
}

impl EdgeSubStream {
    fn try_new(
        rt: Arc<Runtime>,
        input: &EdgeShardInput,
        relation_id: u16,
        dest_sort_key_bits: Option<u8>,
    ) -> Result<Self> {
        match input.format {
            diskgraph_core::metadata::EdgeStorageFormat::Parquet => {
                let reader = object_reader_from_source(&input.parquet, &rt)?;
                let (tx, rx) = mpsc::sync_channel(32);
                let schema = rt.block_on(async move {
                    let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
                    builder = builder.with_batch_size(256_000);
                    let stream = builder.build()?;
                    let schema = stream.schema().clone();
                    tokio::spawn(async move {
                        futures::pin_mut!(stream);
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(batch) => {
                                    if tx.send(Ok(Some(batch))).is_err() {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(e));
                                    return;
                                }
                            }
                        }
                        let _ = tx.send(Ok(None));
                    });
                    Ok::<_, parquet::errors::ParquetError>(schema)
                })?;
                Ok(EdgeSubStream::Parquet(ParquetEdgeSubStream {
                    _schema: schema,
                    batch: None,
                    idx_in_batch: 0,
                    batch_len: 0,
                    dest_sort_key_bits,
                    relation_id,
                    src_idx: 0,
                    dst_idx: 0,
                    sort_idx: None,
                    current_src: None,
                    current_dst: None,
                    current_sort: None,
                    rx,
                }))
            }
            diskgraph_core::metadata::EdgeStorageFormat::Binary => {
                let local_path = match &input.parquet {
                    ParquetSource::LocalFile(path) => path.clone(),
                };
                let file = std::fs::File::open(&local_path)?;
                let file_len = file.metadata()?.len();
                if file_len < 8 {
                    return Err(IndexBuildError::NotImplemented(
                        "binary edge file too small",
                    )
                    .into());
                }
                let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
                let mut header = [0u8; 12];
                reader.read_exact(&mut header[..8])?;
                if header[0..4] != *b"DGED" {
                    return Err(IndexBuildError::NotImplemented(
                        "invalid binary edge magic",
                    )
                    .into());
                }
                let version = header[4];
                if version != 1 && version != 2 {
                    return Err(IndexBuildError::NotImplemented(
                        "unsupported binary edge version",
                    )
                    .into());
                }
                let with_sort = (header[5] & 1) != 0;
                let (src_width, dst_width, sort_width, header_len) = if version == 1 {
                    let sort_width = if with_sort { 4 } else { 0 };
                    (8u8, 8u8, sort_width, 8u64)
                } else {
                    reader.read_exact(&mut header[8..12])?;
                    let src_width = header[6];
                    let dst_width = header[7];
                    let sort_width = header[8];
                    if src_width == 0 || dst_width == 0 || src_width > 8 || dst_width > 8 {
                        return Err(IndexBuildError::NotImplemented(
                            "invalid binary edge width",
                        )
                        .into());
                    }
                    if with_sort && sort_width == 0 {
                        return Err(IndexBuildError::NotImplemented(
                            "binary edge missing dest_sort_key width",
                        )
                        .into());
                    }
                    if sort_width > 8 {
                        return Err(IndexBuildError::NotImplemented(
                            "binary edge sort key width too large",
                        )
                        .into());
                    }
                    (src_width, dst_width, sort_width, 12u64)
                };
                if dest_sort_key_bits.is_some() && !with_sort {
                    return Err(IndexBuildError::NotImplemented(
                        "binary edge missing dest_sort_key",
                    )
                    .into());
                }
                let record_size =
                    src_width as u64 + dst_width as u64 + sort_width as u64;
                let payload = file_len - header_len;
                if payload % record_size != 0 {
                    return Err(IndexBuildError::NotImplemented(
                        "binary edge file size mismatch",
                    )
                    .into());
                }
                let remaining = payload / record_size;
                Ok(EdgeSubStream::Binary(BinaryEdgeSubStream {
                    reader,
                    remaining,
                    relation_id,
                    with_sort,
                    src_width,
                    dst_width,
                    sort_width,
                    next_record: None,
                }))
            }
        }
    }

    fn peek_src(&mut self) -> Result<Option<u64>> {
        match self {
            EdgeSubStream::Parquet(inner) => inner.peek_src(),
            EdgeSubStream::Binary(inner) => inner.peek_src(),
        }
    }

    fn drain_edges(
        &mut self,
        node_id: u64,
        buckets: &mut [Vec<EdgeRec>],
    ) -> Result<Option<u64>> {
        match self {
            EdgeSubStream::Parquet(inner) => inner.drain_edges(node_id, buckets),
            EdgeSubStream::Binary(inner) => inner.drain_edges(node_id, buckets),
        }
    }
}

impl ParquetEdgeSubStream {
    fn peek_src(&mut self) -> Result<Option<u64>> {
        loop {
            if let Some(_) = &self.batch {
                if self.idx_in_batch < self.batch_len {
                    let src_col = self
                        .current_src
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("src_id"))?;
                    return Ok(Some(src_col.value(self.idx_in_batch) as u64));
                }
            }
            match self.rx.recv() {
                Ok(Ok(Some(batch))) => {
                    if self.batch.is_none() {
                        self.src_idx = batch
                            .schema()
                            .index_of("src_id")
                            .map_err(|_| IndexBuildError::InvalidColumn("src_id"))?;
                        self.dst_idx = batch
                            .schema()
                            .index_of("dst_id")
                            .map_err(|_| IndexBuildError::InvalidColumn("dst_id"))?;
                        self.sort_idx = if self.dest_sort_key_bits.is_some() {
                            Some(
                                batch
                                    .schema()
                                    .index_of("dest_sort_key")
                                    .map_err(|_| IndexBuildError::InvalidColumn("dest_sort_key"))?,
                            )
                        } else {
                            None
                        };
                    }
                    self.current_src = Some(
                        batch
                            .column(self.src_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or(IndexBuildError::InvalidColumn("src_id"))?
                            .clone(),
                    );
                    self.current_dst = Some(
                        batch
                            .column(self.dst_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or(IndexBuildError::InvalidColumn("dst_id"))?
                            .clone(),
                    );
                    self.current_sort = self.sort_idx.and_then(|idx| {
                        batch
                            .column(idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .cloned()
                    });
                    self.batch = Some(batch);
                    self.idx_in_batch = 0;
                    self.batch_len = self.batch.as_ref().map(|b| b.num_rows()).unwrap_or(0);
                }
                Ok(Ok(None)) => return Ok(None),
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(IndexBuildError::Parquet(
                        parquet::errors::ParquetError::General(
                            "prefetch channel closed".to_string(),
                        ),
                    )
                    .into())
                }
            }
        }
    }

    fn drain_edges(
        &mut self,
        node_id: u64,
        buckets: &mut [Vec<EdgeRec>],
    ) -> Result<Option<u64>> {
        loop {
            if let Some(_) = &self.batch {
                if self.idx_in_batch < self.batch_len {
                    let src_col = self
                        .current_src
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("src_id"))?;
                    let dst_col = self
                        .current_dst
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("dst_id"))?;
                    let sort_col = self.current_sort.as_ref();
                    while self.idx_in_batch < self.batch_len {
                        let row = self.idx_in_batch;
                        let src_val = src_col.value(row) as u64;
                        if src_val != node_id {
                            return Ok(Some(src_val));
                        }
                        let sort_val = sort_col.map(|c| c.value(row) as u64);
                        buckets[self.relation_id as usize].push(EdgeRec {
                            src: src_val,
                            dst: dst_col.value(row) as u64,
                            dest_sort_key: sort_val,
                            relation_id: self.relation_id,
                        });
                        self.idx_in_batch += 1;
                    }
                }
            }
            match self.rx.recv() {
                Ok(Ok(Some(batch))) => {
                    if self.batch.is_none() {
                        self.src_idx = batch
                            .schema()
                            .index_of("src_id")
                            .map_err(|_| IndexBuildError::InvalidColumn("src_id"))?;
                        self.dst_idx = batch
                            .schema()
                            .index_of("dst_id")
                            .map_err(|_| IndexBuildError::InvalidColumn("dst_id"))?;
                        self.sort_idx = if self.dest_sort_key_bits.is_some() {
                            Some(
                                batch
                                    .schema()
                                    .index_of("dest_sort_key")
                                    .map_err(|_| IndexBuildError::InvalidColumn("dest_sort_key"))?,
                            )
                        } else {
                            None
                        };
                    }
                    self.current_src = Some(
                        batch
                            .column(self.src_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or(IndexBuildError::InvalidColumn("src_id"))?
                            .clone(),
                    );
                    self.current_dst = Some(
                        batch
                            .column(self.dst_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or(IndexBuildError::InvalidColumn("dst_id"))?
                            .clone(),
                    );
                    self.current_sort = self.sort_idx.and_then(|idx| {
                        batch
                            .column(idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .cloned()
                    });
                    self.batch = Some(batch);
                    self.idx_in_batch = 0;
                    self.batch_len = self.batch.as_ref().map(|b| b.num_rows()).unwrap_or(0);
                }
                Ok(Ok(None)) => return Ok(None),
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(IndexBuildError::Parquet(
                        parquet::errors::ParquetError::General(
                            "prefetch channel closed".to_string(),
                        ),
                    )
                    .into())
                }
            }
        }
    }
}

impl BinaryEdgeSubStream {
    fn peek_src(&mut self) -> Result<Option<u64>> {
        if self.next_record.is_none() {
            self.next_record = self.read_next_record()?;
        }
        Ok(self.next_record.as_ref().map(|r| r.src_id))
    }

    fn drain_edges(
        &mut self,
        node_id: u64,
        buckets: &mut [Vec<EdgeRec>],
    ) -> Result<Option<u64>> {
        loop {
            if self.next_record.is_none() {
                self.next_record = self.read_next_record()?;
            }
            let Some(record) = self.next_record.take() else {
                return Ok(None);
            };
            if record.src_id != node_id {
                self.next_record = Some(record);
                return Ok(self.next_record.as_ref().map(|r| r.src_id));
            }
            buckets[self.relation_id as usize].push(EdgeRec {
                src: record.src_id,
                dst: record.dst_id,
                dest_sort_key: record.sort_key,
                relation_id: self.relation_id,
            });
        }
    }

    fn read_next_record(&mut self) -> Result<Option<BinaryEdgeRecord>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let src_id = read_u64_width(&mut self.reader, self.src_width)?;
        let dst_id = read_u64_width(&mut self.reader, self.dst_width)?;
        let sort_key = if self.with_sort {
            Some(read_u64_width(&mut self.reader, self.sort_width)?)
        } else {
            None
        };
        self.remaining = self.remaining.saturating_sub(1);
        Ok(Some(BinaryEdgeRecord {
            src_id,
            dst_id,
            sort_key,
        }))
    }
}

fn read_u64_width(reader: &mut std::io::BufReader<std::fs::File>, width: u8) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf[..width as usize])?;
    Ok(u64::from_le_bytes(buf))
}

enum NodeDataStream {
    Parquet(ParquetNodeDataStream),
    MultiBinary(MultiBinaryNodeDataStream),
}

struct ParquetNodeDataStream {
    batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    batch_len: usize,
    data_idx: usize,
    current_data: Option<BinaryArray>,
    rx: Receiver<PrefetchBatch>,
    expected_rows: u64,
    seen_rows: u64,
}

struct MultiBinaryNodeDataStream {
    files: Vec<PathBuf>,
    current_idx: usize,
    reader: Option<std::io::BufReader<std::fs::File>>,
    expected_rows: u64,
    seen_rows: u64,
}

impl NodeDataStream {
    fn try_new(rt: Arc<Runtime>, sources: &[ParquetSource], expected_rows: u64) -> Result<Self> {
        if sources.is_empty() {
            return Err(IndexBuildError::NotImplemented("node data missing for shard").into());
        }
        let paths: Vec<PathBuf> = sources
            .iter()
            .map(|source| match source {
                ParquetSource::LocalFile(path) => path.clone(),
            })
            .collect();
        if paths.len() == 1 && paths[0].extension().and_then(|e| e.to_str()) != Some("bin") {
            let file = std::fs::File::open(&paths[0])?;
            let reader = SerializedFileReader::new(file)?;
            let rows = reader.metadata().file_metadata().num_rows() as u64;
            if rows != expected_rows {
                return Err(IndexBuildError::NotImplemented(
                    "node data row count mismatch",
                )
                .into());
            }
            let reader = object_reader_from_source(&sources[0], &rt)?;
            let (tx, rx) = mpsc::sync_channel(32);
            let data_idx = rt.block_on(async move {
                let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
                builder = builder.with_batch_size(256_000);
                let stream = builder.build()?;
                let schema = stream.schema().clone();
                if schema.fields().len() != 1 {
                    return Err(parquet::errors::ParquetError::General(
                        "node data parquet must have exactly one column".into(),
                    ));
                }
                let idx = schema.index_of("data").map_err(|_| {
                    parquet::errors::ParquetError::General("missing data column".into())
                })?;
                let field = schema.field(idx);
                if !matches!(field.data_type(), DataType::Binary) {
                    return Err(parquet::errors::ParquetError::General(
                        "node data column must be binary".into(),
                    ));
                }
                tokio::spawn(async move {
                    futures::pin_mut!(stream);
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(batch) => {
                                if tx.send(Ok(Some(batch))).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(e));
                                return;
                            }
                        }
                    }
                    let _ = tx.send(Ok(None));
                });
                Ok::<_, parquet::errors::ParquetError>(idx)
            })?;
            return Ok(NodeDataStream::Parquet(ParquetNodeDataStream {
                batch: None,
                idx_in_batch: 0,
                batch_len: 0,
                data_idx,
                current_data: None,
                rx,
                expected_rows,
                seen_rows: 0,
            }));
        }
        for path in &paths {
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                return Err(IndexBuildError::NotImplemented(
                    "node data parts must be .bin files",
                )
                .into());
            }
        }
        Ok(NodeDataStream::MultiBinary(MultiBinaryNodeDataStream {
            files: paths,
            current_idx: 0,
            reader: None,
            expected_rows,
            seen_rows: 0,
        }))
    }
    fn next_into(&mut self, out: &mut Vec<u8>) -> Result<usize> {
        match self {
            NodeDataStream::Parquet(inner) => inner.next_into(out),
            NodeDataStream::MultiBinary(inner) => inner.next_into(out),
        }
    }
}

impl ParquetNodeDataStream {
    fn next_into(&mut self, out: &mut Vec<u8>) -> Result<usize> {
        loop {
            if let Some(_) = &self.batch {
                if self.idx_in_batch < self.batch_len {
                    let data_col = self
                        .current_data
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("data"))?;
                    let bytes = data_col.value(self.idx_in_batch);
                    out.clear();
                    out.extend_from_slice(bytes);
                    self.idx_in_batch += 1;
                    self.seen_rows += 1;
                    return Ok(bytes.len());
                }
            }
            match self.rx.recv() {
                Ok(Ok(Some(batch))) => {
                    let batch_len = batch.num_rows();
                    if self.seen_rows + batch_len as u64 > self.expected_rows {
                        return Err(IndexBuildError::NotImplemented(
                            "node data row count exceeds shard range",
                        )
                        .into());
                    }
                    self.current_data = Some(
                        batch
                            .column(self.data_idx)
                            .as_any()
                            .downcast_ref::<BinaryArray>()
                            .ok_or(IndexBuildError::InvalidColumn("data"))?
                            .clone(),
                    );
                    self.batch = Some(batch);
                    self.idx_in_batch = 0;
                    self.batch_len = batch_len;
                }
                Ok(Ok(None)) => {
                    return Err(IndexBuildError::NotImplemented(
                        "node data row count shorter than shard range",
                    )
                    .into())
                }
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    return Err(IndexBuildError::Parquet(
                        parquet::errors::ParquetError::General(
                            "node data prefetch channel closed".to_string(),
                        ),
                    )
                    .into())
                }
            }
        }
    }
}

impl MultiBinaryNodeDataStream {
    fn next_into(&mut self, out: &mut Vec<u8>) -> Result<usize> {
        if self.seen_rows >= self.expected_rows {
            return Err(IndexBuildError::NotImplemented(
                "node data row count exceeds shard range",
            )
            .into());
        }
        loop {
            if self.reader.is_none() {
                if self.current_idx >= self.files.len() {
                    return Err(IndexBuildError::NotImplemented(
                        "node data row count shorter than shard range",
                    )
                    .into());
                }
                let path = &self.files[self.current_idx];
                let file = std::fs::File::open(path)?;
                self.reader = Some(std::io::BufReader::with_capacity(1024 * 1024, file));
            }
            let reader = self
                .reader
                .as_mut()
                .ok_or(IndexBuildError::NotImplemented("node data reader missing"))?;
            match read_u32_len(reader) {
                Ok(Some(len)) => {
                    out.clear();
                    if len > 0 {
                        out.resize(len as usize, 0u8);
                        if let Err(_) = reader.read_exact(out.as_mut_slice()) {
                            return Err(IndexBuildError::NotImplemented(
                                "node data row truncated",
                            )
                            .into());
                        }
                    }
                    self.seen_rows += 1;
                    return Ok(len as usize);
                }
                Ok(None) => {
                    self.reader = None;
                    self.current_idx += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

fn read_u32_len(reader: &mut std::io::BufReader<std::fs::File>) -> Result<Option<u32>> {
    let mut buf = [0u8; 4];
    let mut read = 0usize;
    while read < buf.len() {
        let n = reader.read(&mut buf[read..])?;
        if n == 0 {
            if read == 0 {
                return Ok(None);
            }
            return Err(IndexBuildError::NotImplemented("node data length truncated").into());
        }
        read += n;
    }
    Ok(Some(u32::from_le_bytes(buf)))
}

struct MergedEdgeStream {
    streams: Vec<EdgeSubStream>,
    head_srcs: Vec<Option<u64>>,
}

impl MergedEdgeStream {
    fn try_new(
        rt: Arc<Runtime>,
        inputs: &[EdgeShardInput],
        relation_metas: &[EdgeRelationMetadata],
    ) -> Result<Self> {
        let mut rel_lookup = std::collections::HashMap::new();
        for (idx, meta) in relation_metas.iter().enumerate() {
            rel_lookup.insert(meta.name.clone(), (idx as u16, meta.dest_sort_key_bits));
        }
        let mut streams = Vec::new();
        let mut head_srcs = Vec::new();
        for (_idx, input) in inputs.iter().enumerate() {
            let (relation_id, sort_bits) = rel_lookup
                .get(&input.relation)
                .ok_or(IndexBuildError::UnknownRelation)?;
            let stream = EdgeSubStream::try_new(rt.clone(), input, *relation_id, *sort_bits)?;
            streams.push(stream);
            head_srcs.push(streams.last_mut().unwrap().peek_src()?);
        }
        Ok(Self {
            streams,
            head_srcs,
        })
    }

    fn peek_any(&self) -> Option<u64> {
        self.head_srcs.iter().copied().flatten().next()
    }

    fn drain_for_node(&mut self, node_id: u64, buckets: &mut [Vec<EdgeRec>]) -> Result<()> {
        for idx in 0..self.streams.len() {
            let Some(head_src) = self.head_srcs[idx] else { continue };
            if head_src != node_id {
                continue;
            }
            let next_src = self.streams[idx].drain_edges(node_id, buckets)?;
            if let Some(next_src) = next_src {
                if next_src < head_src {
                    return Err(IndexBuildError::EdgeOrderingViolation.into());
                }
            }
            self.head_srcs[idx] = next_src;
        }
        Ok(())
    }
}

fn object_reader_from_source(source: &ParquetSource, _rt: &Runtime) -> Result<AnyParquetReader> {
    let path = match source {
        ParquetSource::LocalFile(path) => path,
    };
    Ok(AnyParquetReader::Local(
        LocalParquetObjectReader::new(LocalAsyncFileReader { path: path.clone() }),
    ))
}

enum AnyParquetReader {
    Local(LocalParquetObjectReader),
}

impl AsyncFileReader for AnyParquetReader {
    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_metadata(options),
        }
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_byte_ranges(ranges),
        }
    }

    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        match self {
            AnyParquetReader::Local(r) => r.get_bytes(range),
        }
    }
}

struct LocalParquetObjectReader {
    inner: LocalAsyncFileReader,
}

impl LocalParquetObjectReader {
    fn new(inner: LocalAsyncFileReader) -> Self {
        Self { inner }
    }
}

impl AsyncFileReader for LocalParquetObjectReader {
    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
        let path = self.inner.path.clone();
        Box::pin(async move {
            let meta = tokio::task::spawn_blocking(move || {
                let file = std::fs::File::open(path)?;
                let reader = SerializedFileReader::new(file)?;
                Ok::<_, parquet::errors::ParquetError>(reader.metadata().clone())
            })
            .await
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))??;
            Ok(std::sync::Arc::new(meta))
        })
    }

    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let path = self.inner.path.clone();
        Box::pin(async move {
            let buf = tokio::task::spawn_blocking(move || {
                let mut f = std::fs::File::open(&path)?;
                use std::io::{Read, Seek, SeekFrom};
                f.seek(SeekFrom::Start(range.start))?;
                let len: usize = (range.end - range.start)
                    .try_into()
                    .map_err(|_| parquet::errors::ParquetError::General("range too large".into()))?;
                let mut buf = vec![0u8; len];
                f.read_exact(&mut buf)?;
                Ok::<_, parquet::errors::ParquetError>(buf)
            })
            .await
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))??;
            Ok(Bytes::from(buf))
        })
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let path = self.inner.path.clone();
        Box::pin(async move {
            let bufs = tokio::task::spawn_blocking(move || {
                let mut out = Vec::with_capacity(ranges.len());
                for range in ranges {
                    let mut f = std::fs::File::open(&path)?;
                    use std::io::{Read, Seek, SeekFrom};
                    f.seek(SeekFrom::Start(range.start))?;
                    let len: usize = (range.end - range.start).try_into().map_err(|_| {
                        parquet::errors::ParquetError::General("range too large".into())
                    })?;
                    let mut buf = vec![0u8; len];
                    f.read_exact(&mut buf)?;
                    out.push(Bytes::from(buf));
                }
                Ok::<_, parquet::errors::ParquetError>(out)
            })
            .await
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))??;
            Ok(bufs)
        })
    }
}

struct LocalAsyncFileReader {
    path: PathBuf,
}
