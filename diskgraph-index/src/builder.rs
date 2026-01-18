use crate::layout::{
    NodeEdgeStats, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use crate::streaming::EdgeShardInput;
use crate::truncate::truncate_counts;
use anyhow::Result;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use lz4_flex::frame::FrameDecoder;
use serde_json;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::cell::RefCell;
use thiserror::Error;

const MAX_NODE_DATA_LEN: usize = 64 * 1024 * 1024;
const OFFSET_OVERHEAD_BYTES: u64 = 128;
const MAX_VARINT_COUNT: usize = 0x3FFF_FFFF;

#[cfg(target_os = "linux")]
fn advise_willneed(file: &File) {
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_willneed(_file: &File) {}

fn count_len(count: usize) -> u64 {
    if count <= 63 {
        1
    } else if count <= 0x3FFF {
        2
    } else if count <= 0x3F_FFFF {
        3
    } else {
        4
    }
}

fn max_delta_for_bytes(delta_bytes: u8) -> u64 {
    match delta_bytes {
        2 => u16::MAX as u64,
        3 => 0xFF_FF_FF,
        4 => u32::MAX as u64,
        _ => 0,
    }
}

fn write_len_le(out: &mut Vec<u8>, len: u32, len_bytes: u8) -> Result<()> {
    match len_bytes {
        2 => {
            let len_u16 = u16::try_from(len)
                .map_err(|_| IndexBuildError::ValueOutOfRange("node_data_len"))?;
            out.extend_from_slice(&len_u16.to_le_bytes());
        }
        3 => {
            if len > 0xFF_FF_FF {
                return Err(IndexBuildError::ValueOutOfRange("node_data_len").into());
            }
            out.push((len & 0xFF) as u8);
            out.push(((len >> 8) & 0xFF) as u8);
            out.push(((len >> 16) & 0xFF) as u8);
        }
        4 => {
            out.extend_from_slice(&len.to_le_bytes());
        }
        _ => return Err(IndexBuildError::NotImplemented("node data len width").into()),
    }
    Ok(())
}

fn estimate_max_node_bytes(
    layouts: &[RelationLayout],
    max_counts: &[usize],
    node_data_max_len: u32,
    node_data_len_bytes: u8,
    has_node_data: bool,
) -> u64 {
    let mut total = OFFSET_OVERHEAD_BYTES;
    for (layout, &count) in layouts.iter().zip(max_counts.iter()) {
        total = total
            .saturating_add(count_len(count))
            .saturating_add((count as u64).saturating_mul(layout.entry_bytes as u64));
    }
    if has_node_data && node_data_max_len > 0 {
        total = total
            .saturating_add(node_data_len_bytes as u64)
            .saturating_add(node_data_max_len as u64);
    }
    total
}

fn select_delta_bytes(
    layouts: &[RelationLayout],
    max_counts: &[usize],
    node_data_max_len: u32,
    has_node_data: bool,
) -> Result<u8> {
    for delta_bytes in [2u8, 3u8, 4u8] {
        let required = estimate_max_node_bytes(
            layouts,
            max_counts,
            node_data_max_len,
            delta_bytes,
            has_node_data,
        );
        if required <= max_delta_for_bytes(delta_bytes) {
            return Ok(delta_bytes);
        }
    }
    Err(IndexBuildError::OffsetDeltaOverflow.into())
}

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
    #[error("build aborted")]
    Aborted,
}

#[derive(Debug)]
pub struct MemoryBudget {
    total: u64,
    available: Mutex<u64>,
    cvar: Condvar,
}

pub struct MemoryBudgetGuard {
    budget: Arc<MemoryBudget>,
    bytes: u64,
}

impl MemoryBudget {
    pub fn new(total: u64) -> Self {
        Self {
            total,
            available: Mutex::new(total),
            cvar: Condvar::new(),
        }
    }

    pub fn acquire(self: &Arc<Self>, bytes: u64) -> Result<MemoryBudgetGuard> {
        if bytes == 0 {
            return Ok(MemoryBudgetGuard {
                budget: Arc::clone(self),
                bytes,
            });
        }
        if bytes > self.total {
            return Err(IndexBuildError::NotImplemented(
                "memory budget too small for shard",
            )
            .into());
        }
        let mut available = self.available.lock().unwrap();
        while *available < bytes {
            available = self.cvar.wait(available).unwrap();
        }
        *available = available.saturating_sub(bytes);
        Ok(MemoryBudgetGuard {
            budget: Arc::clone(self),
            bytes,
        })
    }

    fn release(&self, bytes: u64) {
        let mut available = self.available.lock().unwrap();
        *available = available.saturating_add(bytes).min(self.total);
        self.cvar.notify_all();
    }
}

impl Drop for MemoryBudgetGuard {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.budget.release(self.bytes);
        }
    }
}

impl MemoryBudgetGuard {
    fn release_bytes(&mut self, bytes: u64) {
        let released = bytes.min(self.bytes);
        if released == 0 {
            return;
        }
        self.bytes = self.bytes.saturating_sub(released);
        self.budget.release(released);
    }
}
fn localize_edge_inputs(
    edge_shards: &[EdgeShardInput],
    node_data: Option<&[PathBuf]>,
) -> Result<(Vec<EdgeShardInput>, Option<Vec<PathBuf>>)> {
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
    /// Optional node data sources for this shard.
    pub node_data_paths: Option<Vec<PathBuf>>,
    /// Maximum node data payload length observed for this shard (bytes).
    pub node_data_max_len: u32,
    pub mem_budget: Option<Arc<MemoryBudget>>,
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

        let (edge_inputs, localized_node_data) =
            localize_edge_inputs(edge_shards, self.params.node_data_paths.as_deref())?;

        let record_bytes = estimate_edge_record_bytes(&edge_inputs)?;
        let mut budget_guard = match &self.params.mem_budget {
            Some(budget) => Some(budget.acquire(record_bytes.saturating_mul(2))?),
            None => None,
        };

        let layouts = Arc::new(compute_relation_layouts(
            &self.params.relation_metas,
            &self.params.node_types,
        )?);

        let mut node_stream =
            NodeStream::try_new(&self.params.node_meta, self.params.shard_index)?;
        let mut edge_stream =
            MergedEdgeStream::try_new(&edge_inputs, &self.params.relation_metas)?;
        if let Some(guard) = &mut budget_guard {
            guard.release_bytes(record_bytes);
        }
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
                sources,
                expected_nodes,
            )?)
        } else {
            None
        };
        let mut node_data_buf: Vec<u8> = Vec::new();
        let rel_count = self.params.relation_metas.len();
        let max_counts = edge_stream.max_edges_per_relation(rel_count);
        let node_data_max_len = self.params.node_data_max_len.min(MAX_NODE_DATA_LEN as u32);
        let delta_bytes = select_delta_bytes(
            &layouts,
            &max_counts,
            node_data_max_len,
            self.params.node_meta.has_node_data,
        )?;
        let node_data_len_bytes = delta_bytes;
        let max_delta = max_delta_for_bytes(delta_bytes);
        let counts_min_bytes = rel_count as u64;
        let mut max_edge_counts = Vec::with_capacity(rel_count);
        for layout in layouts.iter() {
            let mut budget = max_delta.saturating_sub(OFFSET_OVERHEAD_BYTES);
            if self.params.node_meta.has_node_data && node_data_max_len > 0 {
                budget = budget
                    .saturating_sub(node_data_max_len as u64)
                    .saturating_sub(node_data_len_bytes as u64);
            }
            budget = budget.saturating_sub(counts_min_bytes);
            let cap = if layout.entry_bytes == 0 {
                0
            } else {
                (budget / layout.entry_bytes as u64) as usize
            };
            max_edge_counts.push(cap.min(MAX_VARINT_COUNT));
        }

        let mut offsets_writer =
            OffsetChunkWriter::try_new(&offsets_path, delta_bytes, node_data_len_bytes)?;
        let mut edges_writer =
            EdgeWriter::try_new(&edges_path, layouts.clone(), node_data_len_bytes)?;
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
                let cap = max_edge_counts.get(i).copied().unwrap_or(0);
                if cap > 0 && len > cap {
                    buckets[i].truncate(cap);
                    counts[i] = cap;
                    total_edges = total_edges.saturating_sub(len - cap);
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
                .map(|len| node_data_len_bytes as u64 + len as u64)
                .unwrap_or(0);
            let node_bytes = EdgeWriter::node_length_bytes_fast(&layouts, &counts)
                .saturating_add(extra_bytes);

            // Determine if we need to truncate to stay within u16 delta budget.
            let base_span = start_offset
                .checked_sub(prev_offset)
                .ok_or(IndexBuildError::CountOverflow)?;
            let budget = max_delta
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
                    max_delta,
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

        Ok(ShardBuildOutput {
            offsets_path,
            edges_path,
            stats,
            layouts,
            build_stats,
            wall_ms: 0,
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
    pub wall_ms: u64,
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
    /// Number of per-relation edge lists that were capped by delta budget.
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
/// [u16 count][u64 base][(count-1) * delta_bytes].
fn delta_encode_offsets_into(
    offsets: &[u64],
    out: &mut Vec<u8>,
    delta_bytes: u8,
) -> Result<()> {
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
            match delta_bytes {
                2 => {
                    let delta_u16: u16 = delta
                        .try_into()
                        .map_err(|_| IndexBuildError::OffsetDeltaOverflow)?;
                    out.extend_from_slice(&delta_u16.to_le_bytes());
                }
                3 => {
                    let delta_u32: u32 = delta
                        .try_into()
                        .map_err(|_| IndexBuildError::OffsetDeltaOverflow)?;
                    if delta_u32 > 0xFF_FF_FF {
                        return Err(IndexBuildError::OffsetDeltaOverflow.into());
                    }
                    out.push((delta_u32 & 0xFF) as u8);
                    out.push(((delta_u32 >> 8) & 0xFF) as u8);
                    out.push(((delta_u32 >> 16) & 0xFF) as u8);
                }
                4 => {
                    let delta_u32: u32 = delta
                        .try_into()
                        .map_err(|_| IndexBuildError::OffsetDeltaOverflow)?;
                    out.extend_from_slice(&delta_u32.to_le_bytes());
                }
                _ => return Err(IndexBuildError::NotImplemented("offset delta width").into()),
            }
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
    delta_bytes: u8,
    node_data_len_bytes: u8,
}

impl OffsetChunkWriter {
    fn try_new(path: &Path, delta_bytes: u8, node_data_len_bytes: u8) -> Result<Self> {
        // Use a larger buffer to reduce flushes on many small writes.
        let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        let header_len =
            bincode::serialized_size(&OffsetFileHeader::new(0, 0)).unwrap_or(32) as usize;
        writer.write_all(&vec![0u8; header_len])?;
        Ok(Self {
            writer,
            chunk_buf: Vec::with_capacity(DELTA_CHUNK),
            total_offsets: 0,
            delta_bytes,
            node_data_len_bytes,
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
        delta_encode_offsets_into(&self.chunk_buf, &mut scratch, self.delta_bytes)?;
        self.writer.write_all(&scratch)?;
        self.chunk_buf.clear();
        Ok(())
    }

    fn finish(mut self, node_count: u64, edge_count: u64) -> Result<u64> {
        self.flush_chunk(Vec::new())?;
        self.writer.flush()?;
        let mut header = OffsetFileHeader::new(node_count, self.total_offsets);
        header.delta_bytes = self.delta_bytes;
        header.node_data_len_bytes = self.node_data_len_bytes;
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
    node_data_len_bytes: u8,
}

impl EdgeWriter {
    fn try_new(
        path: &Path,
        layouts: Arc<Vec<RelationLayout>>,
        node_data_len_bytes: u8,
    ) -> Result<Self> {
        // Larger buffer to reduce syscalls while writing large edge blobs.
        let writer = BufWriter::with_capacity(1024 * 1024, File::create(path)?);
        Ok(Self {
            writer,
            bytes_written: 0,
            edge_count: 0,
            layouts,
            node_data_len_bytes,
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
                .checked_add(self.node_data_len_bytes as usize)
                .and_then(|v| v.checked_add(node_data.len()))
                .ok_or(IndexBuildError::CountOverflow)?
        } else {
            edge_bytes
        };
        if node_bytes > scratch.capacity() {
            scratch.reserve(node_bytes - scratch.capacity());
        }
        for &count in counts.iter() {
            if count <= 63 {
                scratch.push((count & 0x3F) as u8);
            } else if count <= 0x3FFF {
                let b0 = 0x40 | (((count >> 8) & 0x3F) as u8);
                let b1 = (count & 0xFF) as u8;
                scratch.push(b0);
                scratch.push(b1);
            } else if count <= 0x3F_FFFF {
                let b0 = 0x80 | (((count >> 16) & 0x3F) as u8);
                let b1 = ((count >> 8) & 0xFF) as u8;
                let b2 = (count & 0xFF) as u8;
                scratch.push(b0);
                scratch.push(b1);
                scratch.push(b2);
            } else if count <= MAX_VARINT_COUNT {
                let b0 = 0xC0 | (((count >> 24) & 0x3F) as u8);
                let b1 = ((count >> 16) & 0xFF) as u8;
                let b2 = ((count >> 8) & 0xFF) as u8;
                let b3 = (count & 0xFF) as u8;
                scratch.push(b0);
                scratch.push(b1);
                scratch.push(b2);
                scratch.push(b3);
            } else {
                return Err(IndexBuildError::ValueOutOfRange("edge_count").into());
            }
        }
        if let Some(node_data) = node_data {
            let len = node_data
                .len()
                .try_into()
                .map_err(|_| IndexBuildError::ValueOutOfRange("node_data_len"))?;
            write_len_le(scratch, len, self.node_data_len_bytes)?;
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
            let count_len = count_len(count);
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
            let count_len = count_len(count);
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
    Memory(MemoryEdgeSubStream),
}

#[derive(Clone, Copy)]
struct EdgeRecord {
    src_id: u64,
    dst_id: u64,
    sort_key: u64,
}

struct MemoryEdgeSubStream {
    records: Vec<EdgeRecord>,
    index: usize,
    relation_id: u16,
    sort_width: u8,
    max_edges_per_src: usize,
}

impl EdgeSubStream {
    fn try_new(
        input: &EdgeShardInput,
        relation_id: u16,
        dest_sort_key_bits: Option<u8>,
    ) -> Result<Self> {
        let (sort_width, mut records) = read_edge_parts(&input.paths, dest_sort_key_bits)?;
        radix_sort_edge_records(&mut records);
        if sort_width > 0 {
            sort_edge_ranges_by_sort_key(&mut records);
        }
        let max_edges_per_src = max_edges_per_src(&records);
        Ok(EdgeSubStream::Memory(MemoryEdgeSubStream {
            records,
            index: 0,
            relation_id,
            sort_width,
            max_edges_per_src,
        }))
    }

    fn peek_src(&mut self) -> Result<Option<u64>> {
        match self {
            EdgeSubStream::Memory(inner) => inner.peek_src(),
        }
    }

    fn drain_edges(
        &mut self,
        node_id: u64,
        buckets: &mut [Vec<EdgeRec>],
    ) -> Result<Option<u64>> {
        match self {
            EdgeSubStream::Memory(inner) => inner.drain_edges(node_id, buckets),
        }
    }

    fn max_edges_per_src(&self) -> usize {
        match self {
            EdgeSubStream::Memory(inner) => inner.max_edges_per_src,
        }
    }

    fn relation_id(&self) -> u16 {
        match self {
            EdgeSubStream::Memory(inner) => inner.relation_id,
        }
    }
}

impl MemoryEdgeSubStream {
    fn peek_src(&mut self) -> Result<Option<u64>> {
        Ok(self.records.get(self.index).map(|r| r.src_id))
    }

    fn drain_edges(
        &mut self,
        node_id: u64,
        buckets: &mut [Vec<EdgeRec>],
    ) -> Result<Option<u64>> {
        while let Some(record) = self.records.get(self.index) {
            if record.src_id != node_id {
                break;
            }
            buckets[self.relation_id as usize].push(EdgeRec {
                src: record.src_id,
                dst: record.dst_id,
                dest_sort_key: if self.sort_width > 0 {
                    Some(record.sort_key)
                } else {
                    None
                },
                relation_id: self.relation_id,
            });
            self.index += 1;
        }
        Ok(self.records.get(self.index).map(|r| r.src_id))
    }
}

fn max_edges_per_src(records: &[EdgeRecord]) -> usize {
    let mut max_len = 0usize;
    let mut cur = 0usize;
    let mut last = None;
    for rec in records {
        if last == Some(rec.src_id) {
            cur += 1;
        } else {
            max_len = max_len.max(cur);
            cur = 1;
            last = Some(rec.src_id);
        }
    }
    max_len.max(cur)
}

fn read_u64_width_slice(buf: &[u8], width: u8) -> u64 {
    let mut tmp = [0u8; 8];
    tmp[..width as usize].copy_from_slice(buf);
    u64::from_le_bytes(tmp)
}

fn read_edge_parts(
    paths: &[PathBuf],
    dest_sort_key_bits: Option<u8>,
) -> Result<(u8, Vec<EdgeRecord>)> {
    let mut records = Vec::new();
    if paths.is_empty() {
        let sort_width = if dest_sort_key_bits.is_some() { 1 } else { 0 };
        return Ok((sort_width, records));
    }
    let mut spec: Option<(u8, u8, u8)> = None;
    for path in paths {
        let file = std::fs::File::open(path)?;
        advise_willneed(&file);
        let file_len = file.metadata()?.len();
        if file_len < 3 {
            return Err(IndexBuildError::NotImplemented("binary edge file too small").into());
        }
        let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
        let mut header = [0u8; 3];
        reader.read_exact(&mut header)?;
        let src_width = header[0];
        let dst_width = header[1];
        let sort_width = header[2];
        if src_width == 0 || dst_width == 0 || src_width > 8 || dst_width > 8 {
            return Err(IndexBuildError::NotImplemented("invalid binary edge width").into());
        }
        if sort_width > 8 {
            return Err(IndexBuildError::NotImplemented(
                "binary edge sort key width too large",
            )
            .into());
        }
        if dest_sort_key_bits.is_some() && sort_width == 0 {
            return Err(IndexBuildError::NotImplemented(
                "binary edge missing dest_sort_key",
            )
            .into());
        }
        if let Some((s_src, s_dst, s_sort)) = spec {
            if s_src != src_width || s_dst != dst_width || s_sort != sort_width {
                return Err(IndexBuildError::NotImplemented(
                    "binary edge header mismatch",
                )
                .into());
            }
        } else {
            spec = Some((src_width, dst_width, sort_width));
        }
        let record_size = src_width as u64 + dst_width as u64 + sort_width as u64;
        let payload = file_len - 3;
        if payload % record_size != 0 {
            return Err(IndexBuildError::NotImplemented(
                "binary edge file size mismatch",
            )
            .into());
        }
        let remaining = payload / record_size;
        let reserve = usize::try_from(remaining).unwrap_or(usize::MAX);
        records.reserve(reserve);
        let mut buf = [0u8; 24];
        for _ in 0..remaining {
            let record_size = record_size as usize;
            reader.read_exact(&mut buf[..record_size])?;
            let src_id = read_u64_width_slice(&buf[..src_width as usize], src_width);
            let dst_off = src_width as usize;
            let dst_id = read_u64_width_slice(
                &buf[dst_off..dst_off + dst_width as usize],
                dst_width,
            );
            let sort_key = if sort_width > 0 {
                let sort_off = dst_off + dst_width as usize;
                read_u64_width_slice(
                    &buf[sort_off..sort_off + sort_width as usize],
                    sort_width,
                )
            } else {
                0
            };
            records.push(EdgeRecord {
                src_id,
                dst_id,
                sort_key,
            });
        }
    }
    let sort_width = spec.map(|(_, _, sort)| sort).unwrap_or(0);
    Ok((sort_width, records))
}

pub fn estimate_edge_record_bytes(edge_inputs: &[EdgeShardInput]) -> Result<u64> {
    let mut total_records = 0u64;
    for input in edge_inputs {
        for path in &input.paths {
            total_records = total_records.saturating_add(edge_record_count(path)?);
        }
    }
    Ok(total_records.saturating_mul(std::mem::size_of::<EdgeRecord>() as u64))
}

pub fn estimate_edge_sort_bytes(edge_inputs: &[EdgeShardInput]) -> Result<u64> {
    Ok(estimate_edge_record_bytes(edge_inputs)?.saturating_mul(2))
}

fn edge_record_count(path: &Path) -> Result<u64> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len < 3 {
        return Err(IndexBuildError::NotImplemented("binary edge file too small").into());
    }
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut header = [0u8; 3];
    reader.read_exact(&mut header)?;
    let src_width = header[0];
    let dst_width = header[1];
    let sort_width = header[2];
    if src_width == 0 || dst_width == 0 || src_width > 8 || dst_width > 8 {
        return Err(IndexBuildError::NotImplemented("invalid binary edge width").into());
    }
    if sort_width > 8 {
        return Err(IndexBuildError::NotImplemented(
            "binary edge sort key width too large",
        )
        .into());
    }
    let record_size = src_width as u64 + dst_width as u64 + sort_width as u64;
    let payload = file_len - 3;
    if payload % record_size != 0 {
        return Err(IndexBuildError::NotImplemented(
            "binary edge file size mismatch",
        )
        .into());
    }
    Ok(payload / record_size)
}

fn sort_edge_ranges_by_sort_key(records: &mut [EdgeRecord]) {
    if records.len() <= 1 {
        return;
    }
    let mut start = 0usize;
    while start < records.len() {
        let src_id = records[start].src_id;
        let mut end = start + 1;
        while end < records.len() && records[end].src_id == src_id {
            end += 1;
        }
        if end - start > 1 {
            records[start..end].sort_unstable_by(|a, b| b.sort_key.cmp(&a.sort_key));
        }
        start = end;
    }
}

fn radix_sort_edge_records(records: &mut [EdgeRecord]) {
    if records.len() <= 1 {
        return;
    }
    let mut or_src = 0u64;
    for rec in records.iter() {
        or_src |= rec.src_id;
    }
    let mut passes = ((64 - or_src.leading_zeros() + 7) / 8) as usize;
    passes = passes.max(1).min(8);
    if passes == 0 {
        return;
    }
    let mut tmp = vec![
        EdgeRecord {
            src_id: 0,
            dst_id: 0,
            sort_key: 0,
        };
        records.len()
    ];
    let mut swapped = false;
    {
        let mut from: &mut [EdgeRecord] = records;
        let mut to: &mut [EdgeRecord] = tmp.as_mut_slice();
        for pass in 0..passes {
            let shift = pass * 8;
            let mut counts = [0usize; 256];
            for rec in from.iter() {
                let byte = ((rec.src_id >> shift) & 0xFF) as usize;
                counts[byte] += 1;
            }
            let mut sum = 0usize;
            for count in counts.iter_mut() {
                let current = *count;
                *count = sum;
                sum = sum.saturating_add(current);
            }
            for rec in from.iter() {
                let byte = ((rec.src_id >> shift) & 0xFF) as usize;
                let idx = counts[byte];
                to[idx] = *rec;
                counts[byte] = idx + 1;
            }
            std::mem::swap(&mut from, &mut to);
            swapped = !swapped;
        }
        if swapped {
            to.copy_from_slice(from);
        }
    }
}

enum NodeDataStream {
    MultiBinary(MultiBinaryNodeDataStream),
}

struct MultiBinaryNodeDataStream {
    files: Vec<PathBuf>,
    current_idx: usize,
    reader: Option<FrameDecoder<std::fs::File>>,
    expected_rows: u64,
    seen_rows: u64,
}

impl NodeDataStream {
    fn try_new(sources: &[PathBuf], expected_rows: u64) -> Result<Self> {
        if sources.is_empty() {
            return Err(IndexBuildError::NotImplemented("node data missing for shard").into());
        }
        for path in sources {
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                return Err(IndexBuildError::NotImplemented(
                    "node data parts must be .bin files",
                )
                .into());
            }
        }
        Ok(NodeDataStream::MultiBinary(MultiBinaryNodeDataStream {
            files: sources.to_vec(),
            current_idx: 0,
            reader: None,
            expected_rows,
            seen_rows: 0,
        }))
    }
    fn next_into(&mut self, out: &mut Vec<u8>) -> Result<usize> {
        match self {
            NodeDataStream::MultiBinary(inner) => inner.next_into(out),
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
                advise_willneed(&file);
                self.reader = Some(FrameDecoder::new(file));
            }
            let reader = self
                .reader
                .as_mut()
                .ok_or(IndexBuildError::NotImplemented("node data reader missing"))?;
            match read_u32_len(reader) {
                Ok(Some(len)) => {
                    out.clear();
                    if len > 0 {
                        out.reserve(len as usize);
                        // SAFETY: read_exact fills the full buffer; clear on error avoids exposing uninit bytes.
                        unsafe {
                            out.set_len(len as usize);
                        }
                        if let Err(_) = reader.read_exact(out.as_mut_slice()) {
                            out.clear();
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

fn read_u32_len(reader: &mut impl Read) -> Result<Option<u32>> {
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
            let stream = EdgeSubStream::try_new(input, *relation_id, *sort_bits)?;
            streams.push(stream);
            head_srcs.push(streams.last_mut().unwrap().peek_src()?);
        }
        Ok(Self {
            streams,
            head_srcs,
        })
    }

    fn max_edges_per_relation(&self, rel_count: usize) -> Vec<usize> {
        let mut maxes = vec![0usize; rel_count];
        for stream in &self.streams {
            let rel_idx = stream.relation_id() as usize;
            if let Some(entry) = maxes.get_mut(rel_idx) {
                *entry = (*entry).max(stream.max_edges_per_src());
            }
        }
        maxes
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
