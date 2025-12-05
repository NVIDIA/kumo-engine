use crate::layout::{
    NodeEdgeStats, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use crate::streaming::{EdgeShardInput, NodeShardInput, ParquetSource};
use crate::truncate::{truncate_counts, varint_len};
use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::SchemaRef;
use bytes::Bytes;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use futures::{future::BoxFuture, stream::BoxStream, StreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader as StoreParquetObjectReader,
    ParquetRecordBatchStreamBuilder,
};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::runtime::Runtime;
use tracing::info;

/// Hard cap on edges per relation for a single node to keep varint encoding small.
/// Compact varint supports up to 14 bits (0x3FFF). We cap relation edge lists
/// at that limit to guarantee counts fit the encoding without overflow.
const MAX_RELATION_EDGE_COUNT: usize = 0x3FFF;

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
    #[error("input edges not sorted by src_id then dest_sort_key desc")]
    EdgeOrderingViolation,
    #[error("invalid column type for {0}")]
    InvalidColumn(&'static str),
    #[error("unknown relation in shard input")]
    UnknownRelation,
    #[error("shard index mismatch between node shard ({node}) and edge shard ({edge})")]
    ShardIndexMismatch { node: usize, edge: usize },
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("object store error: {0}")]
    ObjectStore(String),
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

/// Consumes a node shard and all aligned edge shards (across relations) and emits two files:
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

    pub fn build(
        &self,
        node_shard: &NodeShardInput,
        edge_shards: &[EdgeShardInput],
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

        let mut node_stream = NodeStream::try_new(rt.clone(), node_shard)?;
        let mut edge_stream =
            MergedEdgeStream::try_new(rt.clone(), edge_shards, &self.params.relation_metas)?;

        let mut offsets_writer = OffsetChunkWriter::try_new(&offsets_path)?;
        let mut edges_writer = EdgeWriter::try_new(&edges_path, layouts.clone())?;

        let mut nodes_seen = 0u64;
        let mut last_edge: Option<EdgeRec> = None;
        let mut build_stats = BuildStats {
            truncated_edges_per_relation: vec![0; self.params.relation_metas.len()],
            ..Default::default()
        };
        let mut prev_offset: u64 = 0;
        // Seed offsets with the start of the first node (0) to make start/end
        // lookups simpler on the read path.
        offsets_writer.push(edges_writer.byte_offset())?;
        // Reuse allocations for buckets and counts to avoid per-node Vec allocations.
        let mut buckets: Vec<Vec<EdgeRec>> = vec![Vec::new(); self.params.relation_metas.len()];
        let mut counts: Vec<usize> = vec![0; self.params.relation_metas.len()];
        while let Some(node_id) = node_stream.next_id()? {
            nodes_seen += 1;
            build_stats.nodes_ingested += 1;
            let start_offset = edges_writer.byte_offset();
            // Collect edges for this node by relation.
            for bucket in buckets.iter_mut() {
                bucket.clear();
            }
            let mut truncated_node = false;
            while let Some(edge) = edge_stream.peek_order_checked(last_edge.as_ref())? {
                if edge.src > node_id {
                    break;
                }
                let edge = edge_stream.pop().unwrap();
                last_edge = Some(edge);
                buckets[edge.relation_id as usize].push(edge);
            }
            let mut original_counts: Option<Vec<usize>> = None;
            // Compute lengths first to avoid borrowing conflicts, then apply truncation.
            let lens: Vec<usize> = buckets.iter().map(|b| b.len()).collect();
            for (i, len) in lens.iter().copied().enumerate() {
                counts[i] = len;
                if len > MAX_RELATION_EDGE_COUNT {
                    if original_counts.is_none() {
                        original_counts = Some(counts.clone());
                    }
                    buckets[i].truncate(MAX_RELATION_EDGE_COUNT);
                    counts[i] = MAX_RELATION_EDGE_COUNT;
                    truncated_node = true;
                    build_stats.capped_edge_lists = build_stats.capped_edge_lists.saturating_add(1);
                }
            }
            let original_counts = original_counts.unwrap_or_else(|| counts.clone());
            let total_edges: usize = counts.iter().sum();
            if total_edges == 0 {
                // No edges for this node across all relations; avoid writing zero counts.
                offsets_writer.push(start_offset)?;
                prev_offset = start_offset;
                continue;
            }
            let node_bytes = EdgeWriter::node_length_bytes(&layouts, &counts)?;

            // Determine if we need to truncate to stay within u16 delta budget.
            let base_span = start_offset
                .checked_sub(prev_offset)
                .ok_or(IndexBuildError::CountOverflow)?;
            let budget = (u16::MAX as u64)
                .checked_sub(base_span)
                .ok_or(IndexBuildError::OffsetDeltaOverflow)?;
            if node_bytes > budget {
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
            // Write all counts first (prefix), then edge data per relation.
            for (rel_idx, target_len) in counts.iter().enumerate() {
                let truncated = original_counts[rel_idx].saturating_sub(*target_len);
                if truncated > 0 {
                    build_stats.truncated_edges_total += truncated as u64;
                    build_stats.truncated_edges_per_relation[rel_idx] = build_stats
                        .truncated_edges_per_relation[rel_idx]
                        .saturating_add(truncated as u64);
                }
                edges_writer.write_count(*target_len as u32)?;
            }
            for (rel_idx, rel_edges) in buckets.iter_mut().enumerate() {
                let target_len = counts[rel_idx];
                if rel_edges.len() > target_len {
                    rel_edges.truncate(target_len);
                }
                for edge in rel_edges.iter().take(target_len) {
                    edges_writer.write_edge(edge)?;
                }
                build_stats.edges_ingested =
                    build_stats.edges_ingested.saturating_add(target_len as u64);
            }
            offsets_writer.push(edges_writer.byte_offset())?;
            prev_offset = edges_writer.byte_offset();
        }

        if edge_stream.peek().is_some() {
            return Err(IndexBuildError::DanglingEdges.into());
        }

        offsets_writer.finish(nodes_seen)?;
        let stats = NodeEdgeStats {
            edge_count: edges_writer
                .edge_count
                .try_into()
                .map_err(|_| IndexBuildError::CountOverflow)?,
            byte_len: edges_writer
                .bytes_written
                .try_into()
                .map_err(|_| IndexBuildError::CountOverflow)?,
        };
        edges_writer.finish()?;

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
#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub nodes_ingested: u64,
    pub edges_ingested: u64,
    pub truncated_nodes: u64,
    pub truncated_edges_total: u64,
    pub truncated_edges_per_relation: Vec<u64>,
    /// Number of per-relation edge lists that were capped at MAX_RELATION_EDGE_COUNT.
    pub capped_edge_lists: u64,
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
        let mut writer = BufWriter::with_capacity(256 * 1024, File::create(path)?);
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

    fn finish(mut self, node_count: u64) -> Result<u64> {
        self.flush_chunk(Vec::new())?;
        self.writer.flush()?;
        let header = OffsetFileHeader::new(node_count, self.total_offsets);
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
        let writer = BufWriter::with_capacity(256 * 1024, File::create(path)?);
        Ok(Self {
            writer,
            bytes_written: 0,
            edge_count: 0,
            layouts,
        })
    }

    fn write_count(&mut self, count: u32) -> Result<()> {
        let written = write_varint(&mut self.writer, count)?;
        self.bytes_written = self
            .bytes_written
            .checked_add(written as u64)
            .ok_or(IndexBuildError::CountOverflow)?;
        Ok(())
    }

    fn write_edge(&mut self, edge: &EdgeRec) -> Result<()> {
        let layout = self
            .layouts
            .get(edge.relation_id as usize)
            .ok_or(IndexBuildError::UnknownRelation)?;
        if edge.dst > layout.dst_mask {
            return Err(IndexBuildError::ValueOutOfRange("dst_id").into());
        }
        let bits_total = layout.dst_bits as usize + layout.dest_sort_key_bits.unwrap_or(0) as usize;
        if bits_total > layout.entry_bytes * 8 {
            return Err(IndexBuildError::NotImplemented("entry size too small").into());
        }
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

        let mut buf = [0u8; 16];
        let bytes_needed = layout.entry_bytes.min(16);
        buf[..bytes_needed].copy_from_slice(&acc.to_le_bytes()[..bytes_needed]);
        self.writer.write_all(&buf[..bytes_needed])?;

        self.bytes_written = self
            .bytes_written
            .checked_add(layout.entry_bytes as u64)
            .ok_or(IndexBuildError::CountOverflow)?;
        self.edge_count = self
            .edge_count
            .checked_add(1)
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
            let count_u32: u32 = count
                .try_into()
                .map_err(|_| IndexBuildError::CountOverflow)?;
            let count_len: u64 = varint_len(count_u32)?
                .try_into()
                .map_err(|_| IndexBuildError::CountOverflow)?;
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

    fn byte_offset(&self) -> u64 {
        self.bytes_written
    }

    fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

fn write_varint<W: Write>(writer: &mut W, val: u32) -> Result<usize> {
    if val <= 63 {
        writer.write_all(&[(val & 0x3F) as u8])?;
        return Ok(1);
    }
    if val <= 0x3FFF {
        let b0 = 0x40 | (((val >> 8) & 0x3F) as u8);
        let b1 = (val & 0xFF) as u8;
        writer.write_all(&[b0, b1])?;
        return Ok(2);
    }
    Err(IndexBuildError::CountOverflow.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EdgeRec {
    src: u64,
    dst: u64,
    dest_sort_key: Option<u64>,
    relation_id: u16,
}

struct NodeStream {
    stream: BoxStream<'static, parquet::errors::Result<arrow_array::RecordBatch>>,
    _schema: SchemaRef,
    current_batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    id_col_idx: usize,
    rt: Arc<Runtime>,
}

impl NodeStream {
    fn try_new(rt: Arc<Runtime>, node_shard: &NodeShardInput) -> Result<Self> {
        let reader = object_reader_from_source(&node_shard.parquet, &rt)?;
        let (stream, schema) = rt.block_on(async move {
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
            builder = builder.with_batch_size(8192);
            let stream = builder.build()?;
            let schema = stream.schema().clone();
            Ok::<_, parquet::errors::ParquetError>((stream, schema))
        })?;
        Ok(Self {
            stream: Box::pin(stream),
            _schema: schema,
            current_batch: None,
            idx_in_batch: 0,
            id_col_idx: 0, // initialized when first batch arrives
            rt,
        })
    }

    fn next_id(&mut self) -> Result<Option<u64>> {
        loop {
            if let Some(batch) = &self.current_batch {
                if self.idx_in_batch < batch.num_rows() {
                    let id_col = batch
                        .column(self.id_col_idx)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("id"))?;
                    let val = id_col.value(self.idx_in_batch) as u64;
                    self.idx_in_batch += 1;
                    return Ok(Some(val));
                }
            }
            match self.rt.block_on(self.stream.next()) {
                Some(Ok(batch)) => {
                    self.id_col_idx = batch
                        .schema()
                        .index_of("id")
                        .map_err(|_| IndexBuildError::InvalidColumn("id"))?;
                    self.current_batch = Some(batch);
                    self.idx_in_batch = 0;
                }
                Some(Err(e)) => return Err(e.into()),
                None => return Ok(None),
            }
        }
    }
}

struct EdgeSubStream {
    stream: BoxStream<'static, parquet::errors::Result<arrow_array::RecordBatch>>,
    _schema: SchemaRef,
    batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    dest_sort_key_bits: Option<u8>,
    relation_id: u16,
    src_idx: usize,
    dst_idx: usize,
    sort_idx: Option<usize>,
    current_src: Option<Int64Array>,
    current_dst: Option<Int64Array>,
    current_sort: Option<Int64Array>,
    rt: Arc<Runtime>,
}

impl EdgeSubStream {
    fn try_new(
        rt: Arc<Runtime>,
        input: &EdgeShardInput,
        relation_id: u16,
        dest_sort_key_bits: Option<u8>,
    ) -> Result<Self> {
        let reader = object_reader_from_source(&input.parquet, &rt)?;
        let (stream, schema) = rt.block_on(async move {
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
            builder = builder.with_batch_size(8192);
            let stream = builder.build()?;
            let schema = stream.schema().clone();
            Ok::<_, parquet::errors::ParquetError>((stream, schema))
        })?;
        Ok(Self {
            stream: Box::pin(stream),
            _schema: schema,
            batch: None,
            idx_in_batch: 0,
            dest_sort_key_bits,
            relation_id,
            src_idx: 0,
            dst_idx: 0,
            sort_idx: None,
            current_src: None,
            current_dst: None,
            current_sort: None,
            rt,
        })
    }

    fn next_rec(&mut self) -> Result<Option<EdgeRec>> {
        loop {
            if let Some(batch) = &self.batch {
                if self.idx_in_batch < batch.num_rows() {
                    let src_col = self
                        .current_src
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("src_id"))?;
                    let dst_col = self
                        .current_dst
                        .as_ref()
                        .ok_or_else(|| IndexBuildError::InvalidColumn("dst_id"))?;
                    let sort_col = self.current_sort.as_ref();
                    let row = self.idx_in_batch;
                    self.idx_in_batch += 1;
                    let sort_val = sort_col.map(|c| c.value(row) as u64);
                    return Ok(Some(EdgeRec {
                        src: src_col.value(row) as u64,
                        dst: dst_col.value(row) as u64,
                        dest_sort_key: sort_val,
                        relation_id: self.relation_id,
                    }));
                }
            }
            match self.rt.block_on(self.stream.next()) {
                Some(Ok(batch)) => {
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
                }
                Some(Err(e)) => return Err(e.into()),
                None => return Ok(None),
            }
        }
    }
}

#[derive(Eq)]
struct HeapItem {
    rec: EdgeRec,
    stream_idx: usize,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.rec.src.cmp(&other.rec.src) {
            Ordering::Equal => {
                let a = self.rec.dest_sort_key.unwrap_or(0);
                let b = other.rec.dest_sort_key.unwrap_or(0);
                b.cmp(&a) // descending dest_sort_key
            }
            ord => ord.reverse(), // min-heap behavior using BinaryHeap
        }
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.rec.src == other.rec.src
            && self.rec.dest_sort_key == other.rec.dest_sort_key
            && self.rec.relation_id == other.rec.relation_id
            && self.stream_idx == other.stream_idx
    }
}

struct MergedEdgeStream {
    streams: Vec<EdgeSubStream>,
    heap: BinaryHeap<HeapItem>,
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
        let mut heap = BinaryHeap::new();
        for (idx, input) in inputs.iter().enumerate() {
            let (relation_id, sort_bits) = rel_lookup
                .get(&input.relation)
                .ok_or(IndexBuildError::UnknownRelation)?;
            let mut stream = EdgeSubStream::try_new(rt.clone(), input, *relation_id, *sort_bits)?;
            if let Some(rec) = stream.next_rec()? {
                heap.push(HeapItem {
                    rec,
                    stream_idx: idx,
                });
            }
            streams.push(stream);
        }
        Ok(Self { streams, heap })
    }

    fn peek(&self) -> Option<&EdgeRec> {
        self.heap.peek().map(|h| &h.rec)
    }

    fn pop(&mut self) -> Option<EdgeRec> {
        if let Some(item) = self.heap.pop() {
            let rec = item.rec;
            if let Some(next_rec) = self.streams[item.stream_idx].next_rec().ok().flatten() {
                self.heap.push(HeapItem {
                    rec: next_rec,
                    stream_idx: item.stream_idx,
                });
            }
            Some(rec)
        } else {
            None
        }
    }

    fn peek_order_checked(&mut self, last: Option<&EdgeRec>) -> Result<Option<&EdgeRec>> {
        if let Some(current) = self.peek() {
            if let Some(prev) = last {
                if current.src < prev.src
                    || (current.src == prev.src
                        && current.dest_sort_key.unwrap_or(u64::MAX)
                            > prev.dest_sort_key.unwrap_or(u64::MAX))
                {
                    return Err(IndexBuildError::EdgeOrderingViolation.into());
                }
            }
        }
        Ok(self.peek())
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
            let mut builder = AmazonS3Builder::new()
                .with_bucket_name(&cfg.bucket)
                .with_virtual_hosted_style_request(
                    cfg.virtual_hosted_style_request.unwrap_or(false),
                );
            if let Some(region) = &cfg.region {
                builder = builder.with_region(region);
            }
            if let Some(endpoint) = &cfg.endpoint {
                builder = builder.with_endpoint(endpoint);
            }
            if let Some(ak) = &cfg.access_key_id {
                builder = builder.with_access_key_id(ak);
            }
            if let Some(sk) = &cfg.secret_access_key {
                builder = builder.with_secret_access_key(sk);
            }
            if let Some(token) = &cfg.session_token {
                builder = builder.with_token(token);
            }
            let store = builder
                .build()
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let store: Arc<dyn ObjectStore> = Arc::new(store);
            let location = ObjectPath::from(cfg.key.clone());
            let meta = rt
                .block_on(store.head(&location))
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let reader = StoreParquetObjectReader::new(store, meta);
            Ok(AnyParquetReader::Store(reader))
        }
        ParquetSource::Gcs(cfg) => {
            let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(&cfg.bucket);
            if let Some(endpoint) = &cfg.endpoint {
                builder = builder.with_url(endpoint);
            }
            if let Some(sa_path) = &cfg.service_account_path {
                builder = builder.with_service_account_path(sa_path);
            }
            if let Some(sa_key) = &cfg.service_account_key {
                builder = builder.with_service_account_key(sa_key);
            }
            if let Some(app_creds) = &cfg.application_credentials_path {
                builder = builder.with_application_credentials(app_creds);
            }
            let store = builder
                .build()
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let store: Arc<dyn ObjectStore> = Arc::new(store);
            let location = ObjectPath::from(cfg.key.clone());
            let meta = rt
                .block_on(store.head(&location))
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let reader = StoreParquetObjectReader::new(store, meta);
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
            let store: Arc<dyn ObjectStore> = Arc::new(store);
            let location = ObjectPath::from(cfg.key.clone());
            let meta = rt
                .block_on(store.head(&location))
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let reader = StoreParquetObjectReader::new(store, meta);
            Ok(AnyParquetReader::Store(reader))
        }
        ParquetSource::InMemory { store, location } => {
            let meta = rt
                .block_on(store.head(location))
                .map_err(|e| IndexBuildError::ObjectStore(e.to_string()))?;
            let reader = StoreParquetObjectReader::new(store.clone(), meta);
            Ok(AnyParquetReader::Store(reader))
        }
    }
}

enum AnyParquetReader {
    Local(LocalParquetObjectReader),
    Store(StoreParquetObjectReader),
}

impl AsyncFileReader for AnyParquetReader {
    fn get_metadata(
        &mut self,
    ) -> BoxFuture<'_, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_metadata(),
            AnyParquetReader::Store(r) => r.get_metadata(),
        }
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<usize>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        match self {
            AnyParquetReader::Local(r) => r.get_byte_ranges(ranges),
            AnyParquetReader::Store(r) => r.get_byte_ranges(ranges),
        }
    }

    fn get_bytes(
        &mut self,
        range: std::ops::Range<usize>,
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
    fn get_metadata(
        &mut self,
    ) -> BoxFuture<'_, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
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
        range: std::ops::Range<usize>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let path = self.inner.path.clone();
        Box::pin(async move {
            let buf = tokio::task::spawn_blocking(move || {
                let mut f = std::fs::File::open(&path)?;
                use std::io::{Read, Seek, SeekFrom};
                f.seek(SeekFrom::Start(range.start as u64))?;
                let mut buf = vec![0u8; range.end - range.start];
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
        ranges: Vec<std::ops::Range<usize>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let path = self.inner.path.clone();
        Box::pin(async move {
            let bufs = tokio::task::spawn_blocking(move || {
                let mut out = Vec::with_capacity(ranges.len());
                for range in ranges {
                    let mut f = std::fs::File::open(&path)?;
                    use std::io::{Read, Seek, SeekFrom};
                    f.seek(SeekFrom::Start(range.start as u64))?;
                    let mut buf = vec![0u8; range.end - range.start];
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
