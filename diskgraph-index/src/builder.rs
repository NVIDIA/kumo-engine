use crate::layout::{
    NodeEdgeStats, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use crate::streaming::{EdgeShardInput, ParquetSource};
use crate::truncate::truncate_counts;
use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::SchemaRef;
use bytes::Bytes;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use futures::{future::BoxFuture, StreamExt};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader as StoreParquetObjectReader,
    ParquetRecordBatchStreamBuilder,
};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
use serde_json;
use serde::Serialize;
use diskgraph_fs::{build_s3_client, S3Config as FsS3Config};
use std::sync::mpsc::{self, Receiver};
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::runtime::Runtime;
use tracing::info;
use object_store::chunked::ChunkedStore;

/// Hard cap on edges per relation for a single node to keep varint encoding small.
/// Compact varint supports up to 14 bits (0x3FFF). We cap relation edge lists
/// at that limit to guarantee counts fit the encoding without overflow.
const MAX_RELATION_EDGE_COUNT: usize = 0x3FFF;

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
    #[error("object store error: {0}")]
    ObjectStore(String),
    #[error("build aborted")]
    Aborted,
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
        mut on_progress: Option<&mut dyn FnMut(u64, u64)>,
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

        let layouts = Arc::new(compute_relation_layouts(
            &self.params.relation_metas,
            &self.params.node_types,
        )?);

        let mut node_stream =
            NodeStream::try_new(&self.params.node_meta, self.params.shard_index)?;
        let mut edge_stream =
            MergedEdgeStream::try_new(rt.clone(), edge_shards, &self.params.relation_metas)?;

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
        while let Some(node_id) = node_stream.next_id()? {
            if let Some(should_abort) = should_abort {
                if should_abort() {
                    return Err(IndexBuildError::Aborted.into());
                }
            }
            nodes_seen += 1;
            build_stats.nodes_ingested += 1;
            if let Some(cb) = on_progress.as_deref_mut() {
                let total = total_nodes.unwrap_or(0);
                if nodes_seen - last_progress_at >= 10_000 {
                    cb(nodes_seen, total);
                    last_progress_at = nodes_seen;
                }
            }
            let start_offset = edges_writer.byte_offset();
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
            if total_edges == 0 {
                // No edges for this node across all relations; avoid writing zero counts.
                offsets_writer.push(start_offset)?;
                prev_offset = start_offset;
                continue;
            }
            let node_bytes = EdgeWriter::node_length_bytes_fast(&layouts, &counts);

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
            edges_writer.write_node(&counts, &buckets, total_edges, &mut node_buf)?;
            build_stats.edges_ingested =
                build_stats.edges_ingested.saturating_add(total_edges as u64);
            offsets_writer.push(edges_writer.byte_offset())?;
            prev_offset = edges_writer.byte_offset();
            last_counts.copy_from_slice(&counts);
        }

        if let Some(cb) = on_progress.as_deref_mut() {
            cb(nodes_seen, total_nodes.unwrap_or(nodes_seen));
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
    ) -> Result<()> {
        scratch.clear();
        let node_bytes = EdgeWriter::node_length_bytes_fast(&self.layouts, counts) as usize;
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

struct EdgeSubStream {
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
    last_src: Option<u64>,
    rx: Receiver<PrefetchBatch>,
}

impl EdgeSubStream {
    fn try_new(
        rt: Arc<Runtime>,
        input: &EdgeShardInput,
        relation_id: u16,
        dest_sort_key_bits: Option<u8>,
    ) -> Result<Self> {
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
        Ok(Self {
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
            last_src: None,
            rx,
        })
    }

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
                        if let Some(last_src) = self.last_src {
                            if src_val < last_src {
                                return Err(IndexBuildError::EdgeOrderingViolation.into());
                            }
                        }
                        if src_val != node_id {
                            return Ok(Some(src_val));
                        }
                        let sort_val = sort_col.map(|c| c.value(row) as u64);
                        self.last_src = Some(src_val);
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
            if head_src < node_id {
                return Err(IndexBuildError::EdgeOrderingViolation.into());
            }
            if head_src != node_id {
                continue;
            }
            let next_src = self.streams[idx].drain_edges(node_id, buckets)?;
            self.head_srcs[idx] = next_src;
        }
        Ok(())
    }
}

// helper retained for potential future use; keep to avoid dead-code lint in case cfg changes.
#[allow(dead_code)]
fn as_i64_array<'a>(
    array: &'a ArrayRef,
    _schema: &SchemaRef,
    name: &'static str,
) -> Result<&'a Int64Array> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| IndexBuildError::InvalidColumn(name))
        .map_err(Into::into)
}

fn object_reader_from_source(source: &ParquetSource, rt: &Runtime) -> Result<AnyParquetReader> {
    match source {
        ParquetSource::LocalFile(path) => Ok(AnyParquetReader::Local(
            LocalParquetObjectReader::new(LocalAsyncFileReader { path: path.clone() }),
        )),
        ParquetSource::S3(cfg) => {
            let allow_http = cfg
                .endpoint
                .as_deref()
                .map(|e| e.starts_with("http://"))
                .unwrap_or(false);
            let s3 = build_s3_client(FsS3Config {
                bucket: cfg.bucket.clone(),
                region: cfg.region.clone(),
                endpoint: cfg.endpoint.clone(),
                allow_http: Some(allow_http),
                virtual_hosted: cfg.virtual_hosted_style_request,
                profile: None,
                access_key_id: cfg.access_key_id.clone(),
                secret_access_key: cfg.secret_access_key.clone(),
                session_token: cfg.session_token.clone(),
            })
            .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let store: Arc<dyn ObjectStore> =
                Arc::new(ChunkedStore::new(s3.store, 32 * 1024 * 1024));
            let location = ObjectPath::from(cfg.key.clone());
            let reader = StoreParquetObjectReader::new(store, location);
            Ok(AnyParquetReader::Store(reader))
        }
        ParquetSource::Gcs(cfg) => {
            // For emulator endpoints (http), fake-gcs-server speaks JSON API instead of the XML API the
            // object_store client uses. Fetch the bytes directly via the JSON download API and read locally.
            if let Some(endpoint) = &cfg.endpoint {
                if endpoint.starts_with("http") {
                    let base = endpoint.trim_end_matches('/');
                    let key_enc =
                        utf8_percent_encode(&cfg.key, NON_ALPHANUMERIC).to_string();
                    let url = format!(
                        "{}/download/storage/v1/b/{}/o/{}?alt=media",
                        base, cfg.bucket, key_enc
                    );
                    let client = reqwest::Client::builder()
                        .danger_accept_invalid_certs(true)
                        .build()
                        .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
                    let bytes = rt.block_on(async {
                        let resp = client
                            .get(&url)
                            .send()
                            .await
                            .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?
                            .error_for_status()
                            .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
                        resp.bytes()
                            .await
                            .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))
                    })?;
                    let tmp_name = format!(
                        "diskgraph_gcs_{}_{}_{}.parquet",
                        cfg.bucket,
                        cfg.key.replace('/', "_"),
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos()
                    );
                    let tmp_path = std::env::temp_dir().join(tmp_name);
                    fs::write(&tmp_path, &bytes)?;
                    let reader =
                        LocalParquetObjectReader::new(LocalAsyncFileReader { path: tmp_path });
                    return Ok(AnyParquetReader::Local(reader));
                }
            }

            let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(&cfg.bucket);
            if let Some(sa_path) = &cfg.service_account_path {
                builder = builder.with_service_account_path(sa_path);
            } else if let Some(sa_key) = &cfg.service_account_key {
                builder = builder.with_service_account_key(sa_key);
            } else if let Some(app_creds) = &cfg.application_credentials_path {
                builder = builder.with_application_credentials(app_creds);
            } else if let Some(endpoint) = &cfg.endpoint {
                let sa = format!(
                    r#"{{"private_key":"private_key","private_key_id":"fake","client_email":"fake@example.com","disable_oauth":true,"gcs_base_url":"{endpoint}"}}"#
                );
                builder = builder.with_service_account_key(sa);
            }
            let store = builder
                .build()
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let store: Arc<dyn ObjectStore> =
                Arc::new(ChunkedStore::new(Arc::new(store), 32 * 1024 * 1024));
            let location = ObjectPath::from(cfg.key.clone());
            let reader = StoreParquetObjectReader::new(store, location);
            Ok(AnyParquetReader::Store(reader))
        }
        ParquetSource::Adls(cfg) => {
            let mut builder = MicrosoftAzureBuilder::new()
                .with_account(&cfg.account)
                .with_container_name(&cfg.container);
            if let Some(endpoint) = &cfg.endpoint {
                builder = builder.with_endpoint(endpoint.to_string());
            }
            if let Some(access_key) = &cfg.access_key {
                builder = builder.with_access_key(access_key);
            }
            if let Some(sas) = &cfg.sas_key {
                builder = builder.with_config(AzureConfigKey::SasKey, sas);
            }
            if let Some(bearer) = &cfg.bearer_token {
                builder = builder.with_bearer_token_authorization(bearer);
            }
            if let Some(client_id) = &cfg.client_id {
                builder = builder.with_client_id(client_id);
            }
            if let Some(client_secret) = &cfg.client_secret {
                builder = builder.with_client_secret(client_secret);
            }
            if let Some(tenant_id) = &cfg.tenant_id {
                builder = builder.with_tenant_id(tenant_id);
            }
            if let Some(use_emulator) = cfg.use_emulator {
                builder = builder.with_use_emulator(use_emulator);
            }
            let store = builder
                .build()
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let store: Arc<dyn ObjectStore> =
                Arc::new(ChunkedStore::new(Arc::new(store), 32 * 1024 * 1024));
            let location = ObjectPath::from(cfg.key.clone());
            let reader = StoreParquetObjectReader::new(store, location);
            Ok(AnyParquetReader::Store(reader))
        }
    }
}

enum AnyParquetReader {
    Local(LocalParquetObjectReader),
    Store(StoreParquetObjectReader),
}

impl AsyncFileReader for AnyParquetReader {
    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_metadata(options),
            AnyParquetReader::Store(r) => r.get_metadata(options),
        }
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_byte_ranges(ranges),
            AnyParquetReader::Store(r) => r.get_byte_ranges(ranges),
        }
    }

    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        match self {
            AnyParquetReader::Local(r) => r.get_bytes(range),
            AnyParquetReader::Store(r) => r.get_bytes(range),
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
