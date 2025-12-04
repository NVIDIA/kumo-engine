use crate::layout::{NodeEdgeStats, OffsetFileHeader, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME};
use crate::streaming::{EdgeShardInput, NodeShardInput, ParquetSource};
use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::SchemaRef;
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture, stream::BoxStream};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::reader::FileReader;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::runtime::Runtime;
use tracing::info;

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
    #[error("edges remain after exhausting node ids (misaligned shards)")]
    DanglingEdges,
    #[error("input edges not sorted by src_id then dest_sort_key desc")]
    EdgeOrderingViolation,
    #[error("invalid column type for {0}")]
    InvalidColumn(&'static str),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
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
/// Edge encoding uses fixed 8-byte words. When dest_sort_key is present, each
/// edge is 16 bytes (dst_id, dest_sort_key). Otherwise, 8 bytes (dst_id).
pub struct ShardBuilder {
    params: ShardBuildParams,
}

impl ShardBuilder {
    pub fn new(params: ShardBuildParams) -> Self {
        Self { params }
    }

    /// Build the shard index from a node shard and its aligned edge shards.
    pub fn build(
        &self,
        node_shard: &NodeShardInput,
        edge_shards: &[EdgeShardInput],
    ) -> Result<ShardBuildOutput> {
        fs::create_dir_all(&self.params.output_dir)?;
        let offsets_path = self.params.output_dir.join(OFFSETS_FILENAME);
        let edges_path = self.params.output_dir.join(EDGES_FILENAME);

        let rt = Arc::new(Runtime::new()?);
        let mut node_stream = NodeStream::try_new(rt.clone(), node_shard)?;
        let mut edge_stream = MergedEdgeStream::try_new(rt.clone(), edge_shards, self.params.relation_meta.dest_sort_key_bits)?;

        let mut offsets_writer = OffsetChunkWriter::try_new(&offsets_path)?;
        let mut edges_writer = EdgeWriter::try_new(&edges_path, self.params.relation_meta.dest_sort_key_bits)?;

        let mut nodes_seen = 0u64;
        let mut last_edge: Option<EdgeRec> = None;
        while let Some(node_id) = node_stream.next_id()? {
            nodes_seen += 1;
            offsets_writer.push(edges_writer.word_offset())?;
            while let Some(edge) = edge_stream.peek_order_checked(last_edge.as_ref())? {
                if edge.src > node_id {
                    break;
                }
                let edge = edge_stream.pop().unwrap();
                last_edge = Some(edge);
                edges_writer.write_edge(&edge)?;
            }
        }
        // If edges remain for nodes we never saw, fail fast to avoid a corrupted index.
        if edge_stream.peek().is_some() {
            return Err(IndexBuildError::DanglingEdges.into());
        }
        // Sentinel offset
        offsets_writer.push(edges_writer.word_offset())?;
        let _total_offsets = offsets_writer.finish(nodes_seen)?;
        let bytes_written = edges_writer.bytes_written;
        let edge_count = edges_writer.edge_count;
        edges_writer.finish()?;

        info!(
            "built shard {} for node type {} relation {} (nodes={}, edges={}, bytes={})",
            self.params.shard_index,
            self.params.node_meta.name,
            self.params.relation_meta.name,
            nodes_seen,
            edge_count,
            bytes_written
        );

        Ok(ShardBuildOutput {
            offsets_path,
            edges_path,
            stats: NodeEdgeStats {
                edge_count: edge_count.try_into().map_err(|_| IndexBuildError::CountOverflow)?,
                byte_len: bytes_written.try_into().map_err(|_| IndexBuildError::CountOverflow)?,
            },
        })
    }
}

/// Output of a shard build, including basic statistics.
pub struct ShardBuildOutput {
    pub offsets_path: PathBuf,
    pub edges_path: PathBuf,
    pub stats: NodeEdgeStats,
}

/// Delta-encode offsets in fixed-size chunks. Each chunk is stored as:
/// [u16 count][u64 base][(count-1) * u16 deltas].
fn delta_encode_offsets(offsets: &[u64]) -> Result<Vec<u8>> {
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
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
                .ok_or(IndexBuildError::NotImplemented("offsets must be non-decreasing"))?;
            let delta_u16: u16 = delta.try_into().map_err(|_| IndexBuildError::OffsetDeltaOverflow)?;
            out.extend_from_slice(&delta_u16.to_le_bytes());
            prev = off;
        }
        idx = chunk_end;
    }
    Ok(out)
}

struct OffsetChunkWriter {
    writer: BufWriter<File>,
    chunk_buf: Vec<u64>,
    total_offsets: u64,
}

impl OffsetChunkWriter {
    fn try_new(path: &Path) -> Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        let header_len = bincode::serialized_size(&OffsetFileHeader::new(0, 0)).unwrap_or(32) as usize;
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
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> Result<()> {
        if self.chunk_buf.is_empty() {
            return Ok(());
        }
        let encoded = delta_encode_offsets(&self.chunk_buf)?;
        self.writer.write_all(&encoded)?;
        self.chunk_buf.clear();
        Ok(())
    }

    fn finish(mut self, node_count: u64) -> Result<u64> {
        self.flush_chunk()?;
        self.writer.flush()?;
        let header = OffsetFileHeader::new(node_count, self.total_offsets);
        let header_bytes = bincode::serialize(&header).map_err(|_| IndexBuildError::NotImplemented("offset encoding"))?;
        let mut file = self.writer.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header_bytes)?;
        Ok(self.total_offsets)
    }
}

struct EdgeWriter {
    writer: BufWriter<File>,
    bytes_written: u64,
    edge_count: u64,
    entry_words: u64,
}

impl EdgeWriter {
    fn try_new(path: &Path, dest_sort_key_bits: Option<u8>) -> Result<Self> {
        let writer = BufWriter::new(File::create(path)?);
        let entry_words = if dest_sort_key_bits.is_some() { 2 } else { 1 };
        Ok(Self {
            writer,
            bytes_written: 0,
            edge_count: 0,
            entry_words,
        })
    }

    fn write_edge(&mut self, edge: &EdgeRec) -> Result<()> {
        self.writer.write_all(&edge.dst.to_le_bytes())?;
        if let Some(sort) = edge.dest_sort_key {
            self.writer.write_all(&sort.to_le_bytes())?;
        }
        self.bytes_written += self.entry_words * 8;
        self.edge_count += 1;
        Ok(())
    }

    fn word_offset(&self) -> u64 {
        self.bytes_written / 8
    }

    fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EdgeRec {
    src: u64,
    dst: u64,
    dest_sort_key: Option<u64>,
}

struct NodeStream {
    stream: BoxStream<'static, parquet::errors::Result<arrow_array::RecordBatch>>,
    schema: SchemaRef,
    current_batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    rt: Arc<Runtime>,
}

impl NodeStream {
    fn try_new(rt: Arc<Runtime>, node_shard: &NodeShardInput) -> Result<Self> {
        let reader = object_reader_from_local(&node_shard.parquet)?;
        let (stream, schema) = rt.block_on(async move {
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
            builder = builder.with_batch_size(8192);
            let stream = builder.build()?;
            let schema = stream.schema().clone();
            Ok::<_, parquet::errors::ParquetError>((stream, schema))
        })?;
        Ok(Self {
            stream: Box::pin(stream),
            schema,
            current_batch: None,
            idx_in_batch: 0,
            rt,
        })
    }

    fn next_id(&mut self) -> Result<Option<u64>> {
        loop {
            if let Some(batch) = &self.current_batch {
                if self.idx_in_batch < batch.num_rows() {
                    let id_array = batch
                        .column_by_name("id")
                        .ok_or_else(|| IndexBuildError::NotImplemented("id column missing"))?;
                    let id_col: &Int64Array = as_i64_array(id_array, &self.schema, "id")?;
                    let val = id_col.value(self.idx_in_batch) as u64;
                    self.idx_in_batch += 1;
            return Ok(Some(val));
                }
            }
            match self.rt.block_on(self.stream.next()) {
                Some(Ok(batch)) => {
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
    schema: SchemaRef,
    batch: Option<arrow_array::RecordBatch>,
    idx_in_batch: usize,
    dest_sort_key_bits: Option<u8>,
    rt: Arc<Runtime>,
}

impl EdgeSubStream {
    fn try_new(rt: Arc<Runtime>, input: &EdgeShardInput, dest_sort_key_bits: Option<u8>) -> Result<Self> {
        let reader = object_reader_from_local(&input.parquet)?;
        let (stream, schema) = rt.block_on(async move {
            let mut builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
            builder = builder.with_batch_size(8192);
            let stream = builder.build()?;
            let schema = stream.schema().clone();
            Ok::<_, parquet::errors::ParquetError>((stream, schema))
        })?;
        Ok(Self {
            stream: Box::pin(stream),
            schema,
            batch: None,
            idx_in_batch: 0,
            dest_sort_key_bits,
            rt,
        })
    }

    fn next_rec(&mut self) -> Result<Option<EdgeRec>> {
        loop {
            if let Some(batch) = &self.batch {
                if self.idx_in_batch < batch.num_rows() {
                    let src_ref = batch
                        .column_by_name("src_id")
                        .ok_or_else(|| IndexBuildError::NotImplemented("src_id column missing"))?;
                    let dst_ref = batch
                        .column_by_name("dst_id")
                        .ok_or_else(|| IndexBuildError::NotImplemented("dst_id column missing"))?;
                    let src_col: &Int64Array = as_i64_array(src_ref, &self.schema, "src_id")?;
                    let dst_col: &Int64Array = as_i64_array(dst_ref, &self.schema, "dst_id")?;
                    let sort_col = self
                        .dest_sort_key_bits
                        .and_then(|_| batch.column_by_name("dest_sort_key"))
                        .map(|col| as_i64_array(col, &self.schema, "dest_sort_key"))
                        .transpose()?;
                    let row = self.idx_in_batch;
                    self.idx_in_batch += 1;
                    let sort_val = sort_col.map(|c| c.value(row) as u64);
                    return Ok(Some(EdgeRec {
                        src: src_col.value(row) as u64,
                        dst: dst_col.value(row) as u64,
                        dest_sort_key: sort_val,
                    }));
                }
            }
            match self.rt.block_on(self.stream.next()) {
                Some(Ok(batch)) => {
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
        self.rec.src == other.rec.src && self.rec.dest_sort_key == other.rec.dest_sort_key && self.stream_idx == other.stream_idx
    }
}

struct MergedEdgeStream {
    streams: Vec<EdgeSubStream>,
    heap: BinaryHeap<HeapItem>,
}

impl MergedEdgeStream {
    fn try_new(rt: Arc<Runtime>, inputs: &[EdgeShardInput], dest_sort_key_bits: Option<u8>) -> Result<Self> {
        let mut streams = Vec::new();
        let mut heap = BinaryHeap::new();
        for (idx, input) in inputs.iter().enumerate() {
            let mut stream = EdgeSubStream::try_new(rt.clone(), input, dest_sort_key_bits)?;
            if let Some(rec) = stream.next_rec()? {
                heap.push(HeapItem { rec, stream_idx: idx });
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
            // Refill from the same stream.
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
                        && current
                            .dest_sort_key
                            .unwrap_or(u64::MAX)
                            > prev.dest_sort_key.unwrap_or(u64::MAX))
                {
                    return Err(IndexBuildError::EdgeOrderingViolation.into());
                }
            }
        }
        Ok(self.peek())
    }
}

fn as_i64_array<'a>(array: &'a ArrayRef, _schema: &SchemaRef, name: &str) -> Result<&'a Int64Array> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| IndexBuildError::InvalidColumn(match name {
            "id" => "id",
            "src_id" => "src_id",
            "dst_id" => "dst_id",
            "dest_sort_key" => "dest_sort_key",
            _ => "unknown",
        })
        .into())
}

fn object_reader_from_local(source: &ParquetSource) -> Result<ParquetObjectReader> {
    let path = match source {
        ParquetSource::LocalFile(p) => p.clone(),
    };
    Ok(ParquetObjectReader::new(LocalAsyncFileReader { path }))
}

struct ParquetObjectReader {
    inner: LocalAsyncFileReader,
}

impl ParquetObjectReader {
    fn new(inner: LocalAsyncFileReader) -> Self {
        Self { inner }
    }
}

impl AsyncFileReader for ParquetObjectReader {
    fn get_metadata(&mut self) -> BoxFuture<'_, parquet::errors::Result<std::sync::Arc<ParquetMetaData>>> {
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

    fn get_bytes(&mut self, range: std::ops::Range<usize>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
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

    fn get_byte_ranges(&mut self, ranges: Vec<std::ops::Range<usize>>) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
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
