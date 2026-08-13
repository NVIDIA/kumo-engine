//! Narrow Databricks control protocol and Cloud Fetch decode primitives.
//!
//! Python owns authentication, query polling, and raw result-page metadata.
//! This module deliberately keeps signed URLs opaque and owns validation and
//! bulk Arrow decoding on the Rust side.

use crate::graph_config::DataSource;
use crate::source_memory::{ByteBudget, BytePermit};
use anyhow::{anyhow, bail, Context, Result};
use arrow_array::RecordBatch;
use arrow_buffer::Buffer;
use arrow_data::ArrayData;
use arrow_ipc::reader::StreamReader;
use arrow_schema::SchemaRef;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use lz4_flex::frame::FrameDecoder;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::ffi::OsStr;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Cursor, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_HELPER_LINE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SignedUrl(String);

impl SignedUrl {
    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SignedUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted signed URL>")
    }
}

#[derive(Clone, Deserialize)]
struct CloudFetchLink {
    url: SignedUrl,
    start_row_offset: u64,
    row_count: u64,
    /// Exact decompressed Arrow IPC object size advertised by Databricks.
    decoded_bytes: u64,
    #[serde(default)]
    headers: VecOrMapHeaders,
}

impl fmt::Debug for CloudFetchLink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CloudFetchLink")
            .field("url", &self.url)
            .field("start_row_offset", &self.start_row_offset)
            .field("row_count", &self.row_count)
            .field("decoded_bytes", &self.decoded_bytes)
            .field("headers", &"<redacted>")
            .finish()
    }
}

/// Serde-friendly header map with a deliberately non-revealing Debug view.
#[derive(Clone, Default, Deserialize)]
#[serde(transparent)]
struct VecOrMapHeaders(std::collections::BTreeMap<String, String>);

impl VecOrMapHeaders {
    fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
    }
}

#[derive(Clone, Copy, Debug)]
struct DownloadLimits {
    max_wire_bytes: u64,
    connect_timeout: Duration,
    read_timeout: Duration,
    /// Test-only escape hatch for a loopback fake object server.
    allow_http: bool,
}

impl Default for DownloadLimits {
    fn default() -> Self {
        Self {
            max_wire_bytes: 256 * 1024 * 1024,
            connect_timeout: Duration::from_secs(15),
            read_timeout: Duration::from_secs(300),
            allow_http: false,
        }
    }
}

fn cloud_fetch_http_agent(limits: DownloadLimits) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(0)
        .timeout_connect(limits.connect_timeout)
        .timeout_read(limits.read_timeout)
        .build()
}

fn is_forbidden_forwarded_header(name: &str) -> bool {
    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("cookie")
        || name.eq_ignore_ascii_case("host")
        || name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("content-length")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.to_ascii_lowercase().starts_with("x-databricks-")
}

impl fmt::Debug for VecOrMapHeaders {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted headers>")
    }
}

#[derive(Clone, Deserialize)]
struct InlineArrowBatch {
    data: String,
    row_count: u64,
}

impl InlineArrowBatch {
    fn decode(&self) -> Result<Vec<u8>> {
        BASE64
            .decode(&self.data)
            .map_err(|_| anyhow!("Databricks returned invalid inline Arrow data"))
    }
}

impl fmt::Debug for InlineArrowBatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InlineArrowBatch")
            .field("data", &"<redacted>")
            .field("row_count", &self.row_count)
            .finish()
    }
}

#[derive(Deserialize)]
struct ResultPage {
    query_id: String,
    page_id: u64,
    start_row_offset: u64,
    has_more_rows: bool,
    lz4_compressed: bool,
    #[serde(default)]
    arrow_schema: Option<String>,
    #[serde(default)]
    links: Vec<CloudFetchLink>,
    #[serde(default)]
    inline_batches: Vec<InlineArrowBatch>,
}

impl fmt::Debug for ResultPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResultPage")
            .field("query_id", &self.query_id)
            .field("page_id", &self.page_id)
            .field("start_row_offset", &self.start_row_offset)
            .field("has_more_rows", &self.has_more_rows)
            .field("lz4_compressed", &self.lz4_compressed)
            .field("arrow_schema", &"<redacted>")
            .field("links", &self.links)
            .field("inline_batches", &self.inline_batches)
            .finish()
    }
}

/// Stateful validation for one query's serialized result pages.
struct PageSequence {
    query_id: String,
    next_page_id: u64,
    next_row_offset: u64,
    finished: bool,
}

impl PageSequence {
    fn new(query_id: String) -> Result<Self> {
        if query_id.trim().is_empty() {
            bail!("Databricks returned an empty query identifier");
        }
        Ok(Self {
            query_id,
            next_page_id: 0,
            next_row_offset: 0,
            finished: false,
        })
    }

    fn validate(&mut self, page: &ResultPage) -> Result<()> {
        if self.finished {
            bail!("Databricks returned a page after the result ended");
        }
        if page.query_id != self.query_id || page.page_id != self.next_page_id {
            bail!("Databricks helper returned an unexpected result page");
        }
        if page.start_row_offset != self.next_row_offset {
            bail!("Databricks result page has a gap or overlap");
        }
        if !page.links.is_empty() && !page.inline_batches.is_empty() {
            bail!("Databricks mixed Cloud Fetch and inline Arrow results");
        }

        let page_start = self.next_row_offset;
        for link in &page.links {
            if link.row_count == 0 {
                continue;
            }
            if link.start_row_offset != self.next_row_offset {
                bail!("Databricks Cloud Fetch links have a gap or overlap");
            }
            if link.decoded_bytes == 0 {
                bail!("Databricks returned a nonempty link with no decoded bytes");
            }
            self.next_row_offset = self
                .next_row_offset
                .checked_add(link.row_count)
                .ok_or_else(|| anyhow!("Databricks result row offset overflowed"))?;
        }
        for batch in &page.inline_batches {
            if batch.row_count == 0 {
                continue;
            }
            self.next_row_offset = self
                .next_row_offset
                .checked_add(batch.row_count)
                .ok_or_else(|| anyhow!("Databricks result row offset overflowed"))?;
        }
        if page.has_more_rows && self.next_row_offset == page_start {
            bail!("Databricks result pagination made no progress");
        }

        self.next_page_id = self
            .next_page_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("Databricks result page identifier overflowed"))?;
        self.finished = !page.has_more_rows;
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn finish(&self) -> Result<u64> {
        if !self.finished {
            bail!("Databricks result ended before its final page");
        }
        Ok(self.next_row_offset)
    }
}

#[derive(Serialize)]
struct HelperRequest<'a> {
    id: u64,
    op: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'a DataSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config_file: Option<&'a Path>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_links: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancel: Option<bool>,
}

#[derive(Deserialize)]
struct HelperResponse<T> {
    id: u64,
    ok: bool,
    result: Option<T>,
}

#[derive(Deserialize)]
pub struct HelperHello {
    pub protocol: u64,
    pub connector: String,
    pub thrift: String,
}

#[derive(Deserialize)]
struct OpenQuery {
    query_id: String,
}

#[derive(Deserialize)]
struct PreparedSchema {
    arrow_schema: String,
}

/// One serialized helper subprocess. The child inherits the ingest process
/// group so an external retry supervisor can terminate both together.
pub struct DatabricksHelper {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
    next_request_id: u64,
}

impl DatabricksHelper {
    pub fn spawn() -> Result<Self> {
        let python = std::env::var_os("DISKGRAPH_PYTHON").unwrap_or_else(|| "python3".into());
        Self::spawn_with_python(python)
    }

    pub fn spawn_with_python(python: impl AsRef<OsStr>) -> Result<Self> {
        let mut command = Command::new(python);
        // The helper installs PR_SET_PDEATHSIG itself. Pass the expected
        // parent explicitly so it can distinguish a real orphan reparented to
        // PID 1 from a healthy container whose ingest process is PID 1.
        command.env("DISKGRAPH_PARENT_PID", std::process::id().to_string());
        if let Some(script) = std::env::var_os("DISKGRAPH_DATABRICKS_HELPER_SCRIPT") {
            command.arg(script);
        } else {
            command.args(["-m", "diskgraph.databricks_helper"]);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to start the Databricks control helper")?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Databricks helper stdin was unavailable"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Databricks helper stdout was unavailable"))?;
        Ok(Self {
            child,
            input: BufWriter::new(input),
            output: BufReader::new(output),
            next_request_id: 1,
        })
    }

    pub fn hello(&mut self) -> Result<HelperHello> {
        self.request("hello", None, None, None, None)
    }

    fn open(
        &mut self,
        source: &DataSource,
        config_file: Option<&Path>,
    ) -> Result<OpenQuery> {
        self.request("open", Some(source), config_file, None, None)
    }

    pub(crate) fn schema(
        &mut self,
        source: &DataSource,
        config_file: Option<&Path>,
    ) -> Result<SchemaRef> {
        let prepared: PreparedSchema =
            self.request("schema", Some(source), config_file, None, None)?;
        decode_arrow_schema(&prepared.arrow_schema)
    }

    fn next_page(
        &mut self,
        max_rows: u64,
        max_bytes: u64,
        max_links: u64,
    ) -> Result<ResultPage> {
        self.request(
            "next",
            None,
            None,
            Some((max_rows, max_bytes, max_links)),
            None,
        )
    }

    fn close(&mut self, cancel: bool) -> Result<()> {
        self.request::<serde_json::Value>("close", None, None, None, Some(cancel))?;
        Ok(())
    }

    fn request<T: DeserializeOwned>(
        &mut self,
        op: &str,
        source: Option<&DataSource>,
        config_file: Option<&Path>,
        budgets: Option<(u64, u64, u64)>,
        cancel: Option<bool>,
    ) -> Result<T> {
        let id = self.next_request_id;
        let request = HelperRequest {
            id,
            op,
            source,
            config_file,
            max_rows: budgets.map(|value| value.0),
            max_bytes: budgets.map(|value| value.1),
            max_links: budgets.map(|value| value.2),
            cancel,
        };
        serde_json::to_writer(&mut self.input, &request)
            .context("failed to encode a Databricks helper request")?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;

        let line = read_line_limited(&mut self.output, MAX_HELPER_LINE_BYTES)?;
        let response: HelperResponse<T> = serde_json::from_slice(&line)
            .map_err(|_| anyhow!("Databricks helper returned an invalid response"))?;
        if response.id != id {
            bail!("Databricks helper returned an unexpected response identifier");
        }
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("Databricks helper request identifier overflowed"))?;
        if !response.ok {
            bail!("Databricks helper operation failed");
        }
        response
            .result
            .ok_or_else(|| anyhow!("Databricks helper omitted its response payload"))
    }
}

fn decode_arrow_schema(encoded: &str) -> Result<SchemaRef> {
    let decoded = BASE64
        .decode(encoded)
        .map_err(|_| anyhow!("Databricks returned an invalid prepared Arrow schema"))?;
    let reader = StreamReader::try_new(Cursor::new(decoded), None)
        .map_err(|_| anyhow!("Databricks returned an invalid prepared Arrow schema"))?;
    Ok(reader.schema())
}

impl Drop for DatabricksHelper {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_line_limited(reader: &mut impl BufRead, max_bytes: usize) -> Result<Vec<u8>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            bail!("Databricks helper exited unexpectedly");
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(take) > max_bytes {
            bail!("Databricks helper response exceeded its size limit");
        }
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            return Ok(line);
        }
    }
}

struct CountingReader<R> {
    inner: R,
    bytes_read: u64,
}

impl<R> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            bytes_read: 0,
        }
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.bytes_read = self.bytes_read.saturating_add(count as u64);
        Ok(count)
    }
}

/// Turns the intermediate zero-length reads emitted at concatenated LZ4
/// frame boundaries into one seamless decoded byte stream.
struct ConcatenatedFrameDecoder<R: Read> {
    decoder: FrameDecoder<CountingReader<R>>,
    eof: bool,
}

impl<R: Read> ConcatenatedFrameDecoder<R> {
    fn new(reader: R) -> Self {
        Self {
            decoder: FrameDecoder::new(CountingReader::new(reader)),
            eof: false,
        }
    }
}

impl<R: Read> Read for ConcatenatedFrameDecoder<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.eof || buffer.is_empty() {
            return Ok(0);
        }
        loop {
            let before = self.decoder.get_ref().bytes_read;
            let count = self.decoder.read(buffer)?;
            if count != 0 {
                return Ok(count);
            }
            let after = self.decoder.get_ref().bytes_read;
            if after == before {
                self.eof = true;
                return Ok(0);
            }
        }
    }
}

#[derive(Debug)]
struct DecodedChunk {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    decoded_bytes: u64,
}

impl DecodedChunk {
    fn emitted_rows(&self) -> Result<u64> {
        self.batches.iter().try_fold(0u64, |rows, batch| {
            rows.checked_add(batch.num_rows() as u64)
                .ok_or_else(|| anyhow!("Databricks emitted row count overflowed"))
        })
    }
}

/// Decode the connector's bounded inline Arrow variant. The helper supplies
/// the schema message separately, matching connector 4.3.0's own assembly.
fn decode_inline_arrow_page(page: &ResultPage, max_decoded_bytes: u64) -> Result<DecodedChunk> {
    let arrow_schema = page
        .arrow_schema
        .as_deref()
        .ok_or_else(|| anyhow!("Databricks returned no inline Arrow schema"))?;
    let mut decoded = BASE64
        .decode(arrow_schema)
        .map_err(|_| anyhow!("Databricks returned an invalid inline Arrow schema"))?;
    let mut advertised_rows = 0u64;
    for batch in &page.inline_batches {
        advertised_rows = advertised_rows
            .checked_add(batch.row_count)
            .ok_or_else(|| anyhow!("Databricks inline row count overflowed"))?;
        let encoded = batch.decode()?;
        if page.lz4_compressed {
            ConcatenatedFrameDecoder::new(Cursor::new(encoded))
                .take(max_decoded_bytes.saturating_add(1))
                .read_to_end(&mut decoded)
                .map_err(|_| anyhow!("Databricks returned invalid inline LZ4 data"))?;
        } else {
            decoded.extend_from_slice(&encoded);
        }
        if decoded.len() as u64 > max_decoded_bytes {
            bail!("Databricks inline Arrow result exceeds the decoded size limit");
        }
    }
    decode_arrow_stream(decoded, advertised_rows, max_decoded_bytes)
}

fn decode_arrow_stream(
    decoded: Vec<u8>,
    advertised_rows: u64,
    max_decoded_bytes: u64,
) -> Result<DecodedChunk> {
    let decoded_len = decoded.len() as u64;
    validate_ipc_envelope(&decoded, max_decoded_bytes)?;
    let mut reader = StreamReader::try_new(Cursor::new(decoded), None)
        .map_err(|_| anyhow!("Databricks returned invalid Arrow IPC data"))?;
    let schema = reader.schema();
    let mut batches = Vec::new();
    let mut decoded_rows = 0u64;
    let mut rows_to_emit = advertised_rows;
    for batch in &mut reader {
        let batch = batch.map_err(|_| anyhow!("Databricks returned invalid Arrow IPC data"))?;
        decoded_rows = decoded_rows
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| anyhow!("Databricks decoded row count overflowed"))?;
        if rows_to_emit == 0 {
            continue;
        }
        let emit_rows = rows_to_emit.min(batch.num_rows() as u64) as usize;
        if emit_rows != 0 {
            batches.push(if emit_rows == batch.num_rows() {
                batch
            } else {
                batch.slice(0, emit_rows)
            });
            rows_to_emit -= emit_rows as u64;
        }
    }
    if decoded_rows < advertised_rows {
        bail!("Databricks Cloud Fetch object contained fewer rows than advertised");
    }
    Ok(DecodedChunk {
        batches,
        schema,
        decoded_bytes: decoded_len,
    })
}

struct BoundedCountingReader<R> {
    inner: R,
    count: Arc<AtomicU64>,
    maximum: u64,
    label: &'static str,
}

impl<R> BoundedCountingReader<R> {
    fn new(inner: R, count: Arc<AtomicU64>, maximum: u64, label: &'static str) -> Self {
        Self {
            inner,
            count,
            maximum,
            label,
        }
    }
}

impl<R: Read> Read for BoundedCountingReader<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        let observed = self.count.load(Ordering::Relaxed);
        if observed >= self.maximum {
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(io::Error::other(format!(
                    "Databricks Cloud Fetch {} exceeds its limit",
                    self.label
                ))),
            };
        }
        let remaining = usize::try_from(self.maximum - observed).unwrap_or(usize::MAX);
        let read_len = output.len().min(remaining);
        let count = self.inner.read(&mut output[..read_len])?;
        self.count.fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
}

enum LinkEvent {
    Schema(SchemaRef),
    Batch(DatabricksBatch),
    Finished {
        emitted_rows: u64,
        decoded_bytes: u64,
        wire_bytes: u64,
        peak_retained_batch_bytes: u64,
    },
}

struct DownloadEvent {
    download_id: u64,
    event: Result<LinkEvent>,
}

struct DownloadEventSender {
    download_id: u64,
    sender: std::sync::mpsc::Sender<DownloadEvent>,
}

impl DownloadEventSender {
    fn send(
        &self,
        event: Result<LinkEvent>,
    ) -> std::result::Result<(), std::sync::mpsc::SendError<DownloadEvent>> {
        self.sender.send(DownloadEvent {
            download_id: self.download_id,
            event,
        })
    }
}

pub(crate) struct DatabricksBatchLease {
    _permit: Arc<BytePermit>,
    consumer_ack: Option<std::sync::mpsc::Sender<()>>,
}

impl Drop for DatabricksBatchLease {
    fn drop(&mut self) {
        if let Some(ack) = self.consumer_ack.take() {
            let _ = ack.send(());
        }
    }
}

/// One decoded Arrow batch whose source-memory reservation remains held until
/// the batch has finished passing through Encode.
pub struct DatabricksBatch {
    batch: RecordBatch,
    lease: DatabricksBatchLease,
}

impl DatabricksBatch {
    pub(crate) fn into_parts(self) -> (RecordBatch, DatabricksBatchLease) {
        (self.batch, self.lease)
    }

    #[cfg(test)]
    fn acknowledge(&mut self) {
        if let Some(ack) = self.lease.consumer_ack.take() {
            let _ = ack.send(());
        }
    }
}

impl std::ops::Deref for DatabricksBatch {
    type Target = RecordBatch;

    fn deref(&self) -> &Self::Target {
        &self.batch
    }
}

fn retained_link_bytes(decoded_bytes: u64) -> Result<u64> {
    const MIN_MEMORY_OVERHEAD_BYTES: u64 = 4 * 1024 * 1024;
    // Databricks reports the serialized IPC length. Materializing nested and
    // variable-width arrays has been observed to require 3x that length before
    // accounting for the serialized input and other small fixed allocations.
    // Reserve the larger of 4x IPC or IPC plus a small-object floor. Only one
    // decoded Arrow batch per link may be outstanding, so the source budget
    // continues to bound aggregate memory while permitting one oversized link.
    let expanded_bytes = decoded_bytes
        .checked_mul(4)
        .ok_or_else(|| anyhow!("Databricks chunk memory size overflowed"))?;
    let bytes_with_floor = decoded_bytes
        .checked_add(MIN_MEMORY_OVERHEAD_BYTES)
        .ok_or_else(|| anyhow!("Databricks chunk memory size overflowed"))?;
    Ok(expanded_bytes.max(bytes_with_floor))
}

/// The heap a decoded Arrow batch pins, counting each distinct allocation once.
///
/// Not `RecordBatch::get_array_memory_size`, which sums `Buffer::capacity` per
/// buffer. `arrow_ipc`'s reader slices every column buffer of a decoded batch
/// out of one message body, so that sum counts the body once per buffer.
fn pinned_batch_bytes(batch: &RecordBatch) -> u64 {
    fn visit(data: &ArrayData, seen: &mut HashSet<(usize, usize)>) {
        let mut record = |buffer: &Buffer| {
            // `as_ptr` points into the slice; back it out to the allocation base.
            let base = buffer.as_ptr() as usize - buffer.ptr_offset();
            seen.insert((base, buffer.capacity()));
        };
        for buffer in data.buffers() {
            record(buffer);
        }
        if let Some(nulls) = data.nulls() {
            record(nulls.buffer());
        }
        for child in data.child_data() {
            visit(child, seen);
        }
    }

    let mut seen = HashSet::new();
    for column in batch.columns() {
        visit(&column.to_data(), &mut seen);
    }
    seen.iter()
        .map(|(_, capacity)| *capacity as u64)
        .fold(0u64, u64::saturating_add)
}

fn stream_cloud_fetch_link(
    agent: &ureq::Agent,
    link: &CloudFetchLink,
    compressed: bool,
    options: QueryStreamOptions,
    permit: Arc<BytePermit>,
    retained_bytes: u64,
    sender: &DownloadEventSender,
) -> Result<LinkEvent> {
    let limits = options.download;
    if link.decoded_bytes > options.max_decoded_chunk_bytes {
        bail!("Databricks Cloud Fetch object exceeds the decoded size limit");
    }
    let parsed = url::Url::parse(link.url.expose())
        .map_err(|_| anyhow!("Databricks returned an invalid signed object URL"))?;
    let valid_scheme = parsed.scheme() == "https"
        || (limits.allow_http
            && parsed.scheme() == "http"
            && parsed
                .host_str()
                .is_some_and(|host| host == "localhost" || host == "127.0.0.1"));
    if !valid_scheme || parsed.host_str().is_none() {
        bail!("Databricks signed object URL is not allowed");
    }

    let mut request = agent
        .get(link.url.expose())
        .set("Accept", "application/octet-stream");
    for (name, value) in link.headers.iter() {
        if !is_forbidden_forwarded_header(name) {
            request = request.set(name, value);
        }
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(status, _)) => {
            bail!("Databricks Cloud Fetch object returned HTTP {status}")
        }
        Err(ureq::Error::Transport(_)) => {
            bail!("Databricks Cloud Fetch object download failed")
        }
    };
    let content_length = response
        .header("Content-Length")
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| anyhow!("Databricks Cloud Fetch object has invalid Content-Length"))
        })
        .transpose()?;
    if content_length.is_some_and(|length| length > limits.max_wire_bytes) {
        bail!("Databricks Cloud Fetch object exceeds the wire size limit");
    }

    let wire_count = Arc::new(AtomicU64::new(0));
    let raw: Box<dyn Read + Send + Sync> = response.into_reader();
    let wire_reader = BoundedCountingReader::new(
        raw,
        Arc::clone(&wire_count),
        limits.max_wire_bytes,
        "wire body",
    );
    let decompressed: Box<dyn Read + Send> = if compressed {
        Box::new(ConcatenatedFrameDecoder::new(wire_reader))
    } else {
        Box::new(wire_reader)
    };
    let decoded_count = Arc::new(AtomicU64::new(0));
    let decoded_reader = BoundedCountingReader::new(
        decompressed,
        Arc::clone(&decoded_count),
        // The permit for this worker is based on the server-advertised size.
        // Enforce that exact bound while bytes are decoded, rather than only
        // comparing after Arrow has allocated batches from an oversized body.
        link.decoded_bytes,
        "decoded IPC stream",
    );
    let mut reader = StreamReader::try_new(decoded_reader, None)
        .map_err(|_| anyhow!("Databricks returned invalid Arrow IPC data"))?;
    sender
        .send(Ok(LinkEvent::Schema(reader.schema())))
        .map_err(|_| anyhow!("Databricks Encode consumer stopped"))?;

    let mut decoded_rows = 0u64;
    let mut retained_batch_bytes = 0u64;
    let mut peak_retained_batch_bytes = 0u64;
    let mut retained_batch_acks: VecDeque<(u64, std::sync::mpsc::Receiver<()>)> = VecDeque::new();
    let mut emitted_rows = 0u64;
    let mut rows_to_emit = link.row_count;
    loop {
        // Do not ask Arrow to allocate the next batch while the prior batch is
        // still retained by an Encode worker. This keeps decode-time and
        // queued Arrow memory inside the link's charged reservation.
        while let Some((released_bytes, consumer_ack)) = retained_batch_acks.pop_front() {
            consumer_ack
                .recv()
                .map_err(|_| anyhow!("Databricks Encode consumer stopped"))?;
            retained_batch_bytes = retained_batch_bytes.saturating_sub(released_bytes);
        }
        debug_assert_eq!(retained_batch_bytes, 0);
        let Some(batch) = reader.next() else {
            break;
        };
        let batch = batch.map_err(|_| anyhow!("Databricks returned invalid Arrow IPC data"))?;
        let batch_bytes = pinned_batch_bytes(&batch);
        if batch_bytes > permit.max_input_chunk_bytes() {
            bail!("Databricks Arrow batch exceeds the shared single-input memory limit");
        }
        if batch_bytes > retained_bytes {
            bail!(
                "Databricks Arrow batch exceeds its reserved input memory \
                 (batch {batch_bytes} bytes, reservation {retained_bytes} bytes, \
                 decoded IPC {} bytes)",
                link.decoded_bytes,
            );
        }
        decoded_rows = decoded_rows
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| anyhow!("Databricks decoded row count overflowed"))?;
        if rows_to_emit == 0 {
            continue;
        }
        let emit_rows = rows_to_emit.min(batch.num_rows() as u64) as usize;
        if emit_rows != 0 {
            let batch = if emit_rows == batch.num_rows() {
                batch
            } else {
                batch.slice(0, emit_rows)
            };
            let (consumer_ack, wait_for_consumer) = std::sync::mpsc::channel();
            sender
                .send(Ok(LinkEvent::Batch(DatabricksBatch {
                    batch,
                    lease: DatabricksBatchLease {
                        _permit: Arc::clone(&permit),
                        consumer_ack: Some(consumer_ack),
                    },
                })))
                .map_err(|_| anyhow!("Databricks Encode consumer stopped"))?;
            retained_batch_bytes = retained_batch_bytes
                .checked_add(batch_bytes)
                .ok_or_else(|| anyhow!("Databricks decoded Arrow memory size overflowed"))?;
            peak_retained_batch_bytes = peak_retained_batch_bytes.max(retained_batch_bytes);
            retained_batch_acks.push_back((batch_bytes, wait_for_consumer));
            if options.pause_download_while_batch_active {
                let (released_bytes, consumer_ack) = retained_batch_acks.pop_front().unwrap();
                consumer_ack
                    .recv()
                    .map_err(|_| anyhow!("Databricks Encode consumer stopped"))?;
                retained_batch_bytes = retained_batch_bytes.saturating_sub(released_bytes);
            }
            emitted_rows = emitted_rows.saturating_add(emit_rows as u64);
            rows_to_emit -= emit_rows as u64;
        }
    }
    // Arrow stops as soon as it observes the IPC end marker. Drain the
    // decoder so the final LZ4 frame checksum/footer and HTTP body are still
    // validated and included in the wire-length check.
    io::copy(reader.get_mut(), &mut io::sink())
        .map_err(|_| anyhow!("Databricks returned invalid Arrow IPC data"))?;
    if decoded_rows < link.row_count {
        bail!("Databricks Cloud Fetch object contained fewer rows than advertised");
    }
    let decoded_bytes = decoded_count.load(Ordering::Relaxed);
    if decoded_bytes != link.decoded_bytes {
        bail!("Databricks Cloud Fetch object has an unexpected decoded size");
    }
    let wire_bytes = wire_count.load(Ordering::Relaxed);
    if content_length.is_some_and(|length| length != wire_bytes) {
        bail!("Databricks Cloud Fetch object body length did not match Content-Length");
    }
    Ok(LinkEvent::Finished {
        emitted_rows,
        decoded_bytes,
        wire_bytes,
        peak_retained_batch_bytes,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct QueryStreamOptions {
    pub(crate) max_page_rows: u64,
    pub(crate) max_page_bytes: u64,
    pub(crate) max_page_links: u64,
    pub(crate) max_concurrent_downloads: usize,
    pub(crate) max_decoded_chunk_bytes: u64,
    pub(crate) memory_budget_bytes: u64,
    download: DownloadLimits,
    /// Service-time benchmark mode: after yielding a batch, pause its
    /// download/decode worker until the consumer requests the next batch.
    pub(crate) pause_download_while_batch_active: bool,
}

const DOWNLOADS_PER_SOURCE_LOADER: usize = 4;
// Cloud Fetch currently uses one blocking OS thread per active object. Keep
// that model bounded even on hosts with hundreds of cores. At the cap, each
// stream needs about 200 Mbps to fill a 100 Gbps link.
const MAX_CONCURRENT_DOWNLOADS: usize = 512;

pub(crate) fn download_parallelism(source_loaders: usize) -> usize {
    source_loaders
        .max(1)
        .saturating_mul(DOWNLOADS_PER_SOURCE_LOADER)
        .min(MAX_CONCURRENT_DOWNLOADS)
}

pub fn default_download_parallelism() -> usize {
    download_parallelism(crate::default_worker_threads())
}

impl QueryStreamOptions {
    #[doc(hidden)]
    pub fn allow_loopback_http_for_benchmark(&mut self) {
        self.download.allow_http = true;
    }

    pub(crate) fn pause_download_while_batch_active(&mut self) {
        self.pause_download_while_batch_active = true;
    }
}

impl Default for QueryStreamOptions {
    fn default() -> Self {
        const MAX_PAGE_BYTES: u64 = 128 * 1024 * 1024;
        Self {
            // Databricks Cloud Fetch honors maxBytes but can return more than
            // TFetchResults.maxRows (observed with connector 4.3.0). Keep a
            // row sanity bound derived from the densest supported Arrow
            // representation: one bit-packed Boolean value per row.
            max_page_rows: MAX_PAGE_BYTES * 8,
            max_page_bytes: MAX_PAGE_BYTES,
            max_page_links: 8,
            // Blocking object reads spend most of their time waiting on the
            // network. Give each cgroup-aware source/decode lane several I/O
            // streams; the shared byte budget independently limits how many
            // can retain decoded data at once.
            max_concurrent_downloads: default_download_parallelism(),
            max_decoded_chunk_bytes: 256 * 1024 * 1024,
            memory_budget_bytes: 1024 * 1024 * 1024,
            download: DownloadLimits::default(),
            pause_download_while_batch_active: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    pub rows: u64,
    pub wire_bytes: u64,
    pub decoded_ipc_bytes: u64,
    pub query_startup_nanos: u64,
}

struct PendingDownload {
    handle: JoinHandle<()>,
    reserved_bytes: u64,
}

struct DownloadSlotState {
    active: usize,
    next_ticket: u64,
    next_to_admit: u64,
    cancelled: BTreeSet<u64>,
}

struct DownloadSlots {
    maximum: usize,
    state: Mutex<DownloadSlotState>,
    changed: Condvar,
}

impl DownloadSlots {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            maximum: slots.max(1),
            state: Mutex::new(DownloadSlotState {
                active: 0,
                next_ticket: 0,
                next_to_admit: 0,
                cancelled: BTreeSet::new(),
            }),
            changed: Condvar::new(),
        })
    }

    fn issue_ticket(&self) -> Result<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Databricks download concurrency lock was poisoned"))?;
        let ticket = state.next_ticket;
        state.next_ticket = state
            .next_ticket
            .checked_add(1)
            .ok_or_else(|| anyhow!("Databricks download ticket overflowed"))?;
        Ok(ticket)
    }

    fn advance_cancelled(state: &mut DownloadSlotState) {
        while state.cancelled.remove(&state.next_to_admit) {
            state.next_to_admit = state.next_to_admit.saturating_add(1);
        }
    }

    fn cancel(&self, ticket: u64) {
        if let Ok(mut state) = self.state.lock() {
            if ticket >= state.next_to_admit {
                state.cancelled.insert(ticket);
                Self::advance_cancelled(&mut state);
                self.changed.notify_all();
            }
        }
    }

    fn acquire(self: &Arc<Self>, ticket: u64) -> Result<DownloadSlot> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Databricks download concurrency lock was poisoned"))?;
        loop {
            Self::advance_cancelled(&mut state);
            if ticket == state.next_to_admit && state.active < self.maximum {
                state.active += 1;
                state.next_to_admit = state
                    .next_to_admit
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Databricks download ticket overflowed"))?;
                Self::advance_cancelled(&mut state);
                self.changed.notify_all();
                break;
            }
            state = self
                .changed
                .wait(state)
                .map_err(|_| anyhow!("Databricks download concurrency lock was poisoned"))?;
        }
        Ok(DownloadSlot {
            slots: Arc::clone(self),
        })
    }
}

struct DownloadSlot {
    slots: Arc<DownloadSlots>,
}

impl Drop for DownloadSlot {
    fn drop(&mut self) {
        if let Ok(mut state) = self.slots.state.lock() {
            state.active = state.active.saturating_sub(1);
            self.slots.changed.notify_all();
        }
    }
}

/// Demand-driven, bounded query stream. Link-page lookahead scales with the
/// download pool and byte budget; batches enter Encode as downloads become
/// ready.
pub struct DatabricksQueryStream<'a> {
    helper: &'a mut DatabricksHelper,
    sequence: PageSequence,
    options: QueryStreamOptions,
    agent: ureq::Agent,
    download_slots: Arc<DownloadSlots>,
    pending_downloads: HashMap<u64, PendingDownload>,
    download_event_sender: std::sync::mpsc::Sender<DownloadEvent>,
    download_event_receiver: std::sync::mpsc::Receiver<DownloadEvent>,
    next_download_id: u64,
    pending_reserved_bytes: u64,
    pending_inline_pages: VecDeque<ResultPage>,
    pending_batches: VecDeque<DatabricksBatch>,
    budget: Arc<ByteBudget>,
    expected_schema: Option<SchemaRef>,
    lz4_compressed: Option<bool>,
    requested_page: bool,
    metrics: QueryMetrics,
    closed: bool,
}

impl<'a> DatabricksQueryStream<'a> {
    fn validate_options(options: QueryStreamOptions, require_memory_budget: bool) -> Result<()> {
        if options.max_page_rows == 0
            || options.max_page_bytes == 0
            || options.max_page_links == 0
            || options.max_concurrent_downloads == 0
            || options.max_decoded_chunk_bytes == 0
            || (require_memory_budget && options.memory_budget_bytes == 0)
        {
            bail!("Databricks query stream limits must be positive");
        }
        if options.max_concurrent_downloads > MAX_CONCURRENT_DOWNLOADS {
            bail!("Databricks download concurrency exceeds the blocking worker cap");
        }
        Ok(())
    }

    fn open_query(
        helper: &mut DatabricksHelper,
        source: &DataSource,
        config_file: Option<&Path>,
    ) -> Result<(OpenQuery, u64)> {
        let query_start = Instant::now();
        let opened = helper.open(source, config_file)?;
        let query_startup_nanos = query_start.elapsed().as_nanos() as u64;
        Ok((opened, query_startup_nanos))
    }

    pub fn open(
        helper: &'a mut DatabricksHelper,
        source: &DataSource,
        config_file: Option<&Path>,
        options: QueryStreamOptions,
    ) -> Result<Self> {
        Self::validate_options(options, true)?;
        let budget = ByteBudget::new(options.memory_budget_bytes)?;
        let (opened, query_startup_nanos) = Self::open_query(helper, source, config_file)?;
        Self::open_with_budget(
            helper,
            source,
            options,
            budget,
            query_startup_nanos,
            opened,
        )
    }

    pub(crate) fn open_with_shared_budget(
        helper: &'a mut DatabricksHelper,
        source: &DataSource,
        config_file: Option<&Path>,
        options: QueryStreamOptions,
        budget: Arc<ByteBudget>,
    ) -> Result<Self> {
        Self::validate_options(options, false)?;
        let (opened, query_startup_nanos) = Self::open_query(helper, source, config_file)?;
        Self::open_with_budget(
            helper,
            source,
            options,
            budget,
            query_startup_nanos,
            opened,
        )
    }

    fn open_with_budget(
        helper: &'a mut DatabricksHelper,
        source: &DataSource,
        options: QueryStreamOptions,
        budget: Arc<ByteBudget>,
        query_startup_nanos: u64,
        opened: OpenQuery,
    ) -> Result<Self> {
        if !matches!(source, DataSource::Databricks { .. }) {
            bail!("Databricks query stream requires a Databricks source");
        }
        let (download_event_sender, download_event_receiver) =
            std::sync::mpsc::channel();
        Ok(Self {
            helper,
            sequence: PageSequence::new(opened.query_id)?,
            options,
            agent: cloud_fetch_http_agent(options.download),
            download_slots: DownloadSlots::new(options.max_concurrent_downloads),
            pending_downloads: HashMap::new(),
            download_event_sender,
            download_event_receiver,
            next_download_id: 0,
            pending_reserved_bytes: 0,
            pending_inline_pages: VecDeque::new(),
            pending_batches: VecDeque::new(),
            budget,
            expected_schema: None,
            lz4_compressed: None,
            requested_page: false,
            metrics: QueryMetrics {
                query_startup_nanos,
                ..QueryMetrics::default()
            },
            closed: false,
        })
    }

    fn accept_chunk(
        &mut self,
        chunk: DecodedChunk,
        permit: Arc<BytePermit>,
    ) -> Result<()> {
        if let Some(expected) = &self.expected_schema {
            if expected.as_ref() != chunk.schema.as_ref() {
                bail!("Databricks changed Arrow schema between result chunks");
            }
        } else {
            self.expected_schema = Some(chunk.schema);
        }
        self.pending_batches
            .extend(chunk.batches.into_iter().map(|batch| DatabricksBatch {
                batch,
                lease: DatabricksBatchLease {
                    _permit: Arc::clone(&permit),
                    consumer_ack: None,
                },
            }));
        Ok(())
    }

    pub fn metrics(&self) -> QueryMetrics {
        self.metrics
    }

    fn record_decoded(&mut self, rows: u64, decoded_bytes: u64) -> Result<()> {
        let total_rows = self
            .metrics
            .rows
            .checked_add(rows)
            .ok_or_else(|| anyhow!("Databricks decoded row count overflowed"))?;
        let total_bytes = self
            .metrics
            .decoded_ipc_bytes
            .checked_add(decoded_bytes)
            .ok_or_else(|| anyhow!("Databricks decoded byte count overflowed"))?;
        self.metrics.rows = total_rows;
        self.metrics.decoded_ipc_bytes = total_bytes;
        Ok(())
    }

    fn spawn_link_page(&mut self, links: Vec<CloudFetchLink>, compressed: bool) -> Result<()> {
        if links.len() as u64 > self.options.max_page_links {
            bail!("Databricks returned too many Cloud Fetch links in one page");
        }
        let links = links
            .into_iter()
            .filter(|link| link.row_count > 0)
            .collect::<Vec<_>>();
        if links.is_empty() {
            return Ok(());
        }
        for link in links {
            let download_id = self.next_download_id;
            self.next_download_id = self
                .next_download_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("Databricks download id overflowed"))?;
            let agent = self.agent.clone();
            let options = self.options;
            let budget = Arc::clone(&self.budget);
            let download_slots = Arc::clone(&self.download_slots);
            // The response body and decoded IPC are streamed. Every emitted
            // batch retains this permit through the Encode worker handoff.
            let retained_bytes = retained_link_bytes(link.decoded_bytes)?;
            let budget_ticket = budget.issue_ticket(retained_bytes)?;
            let download_ticket = download_slots.issue_ticket()?;
            // A link reserves its IPC size and Arrow overhead before download.
            // Per-batch acknowledgements stop the producer when that window is
            // full. A later link can still download and decode a batch while an
            // earlier link is consumed.
            let sender = DownloadEventSender {
                download_id,
                sender: self.download_event_sender.clone(),
            };
            let reserved_bytes = retained_bytes;
            let handle = std::thread::Builder::new()
                .name("databricks-cloud-fetch".to_string())
                .spawn(move || {
                    let permit = budget
                        .acquire_ticket(budget_ticket, retained_bytes)
                        .map(Arc::new);
                    let result = match permit {
                        Ok(permit) => download_slots.acquire(download_ticket).and_then(|_slot| {
                            stream_cloud_fetch_link(
                                &agent,
                                &link,
                                compressed,
                                options,
                                permit,
                                retained_bytes,
                                &sender,
                            )
                        }),
                        Err(error) => {
                            download_slots.cancel(download_ticket);
                            Err(error)
                        }
                    };
                    let _ = sender.send(result);
                });
            let handle = match handle {
                Ok(handle) => handle,
                Err(_) => {
                    self.download_slots.cancel(download_ticket);
                    return Err(anyhow!("failed to start a Databricks download worker"));
                }
            };
            self.pending_reserved_bytes =
                self.pending_reserved_bytes
                    .checked_add(reserved_bytes)
                    .ok_or_else(|| anyhow!("Databricks pending byte count overflowed"))?;
            self.pending_downloads.insert(
                download_id,
                PendingDownload {
                    handle,
                    reserved_bytes,
                },
            );
        }
        Ok(())
    }

    fn request_page(&mut self) -> Result<bool> {
        let page = self.helper.next_page(
            self.options.max_page_rows,
            self.options.max_page_bytes,
            self.options.max_page_links,
        )?;
        self.sequence.validate(&page)?;
        let page_rows = page
            .links
            .iter()
            .map(|link| link.row_count)
            .chain(page.inline_batches.iter().map(|batch| batch.row_count))
            .try_fold(0u64, |total, rows| total.checked_add(rows))
            .ok_or_else(|| anyhow!("Databricks result page row count overflowed"))?;
        if page_rows > self.options.max_page_rows {
            bail!("Databricks result page exceeds the requested row limit");
        }
        let link_bytes = page
            .links
            .iter()
            .map(|link| link.decoded_bytes)
            .try_fold(0u64, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| anyhow!("Databricks result page byte count overflowed"))?;
        if link_bytes > self.options.max_page_bytes {
            bail!("Databricks result page exceeds the requested byte limit");
        }
        if self
            .lz4_compressed
            .is_some_and(|compressed| compressed != page.lz4_compressed)
        {
            bail!("Databricks changed compression between pages");
        }
        self.lz4_compressed = Some(page.lz4_compressed);
        self.requested_page = true;
        if page.links.is_empty() && !page.inline_batches.is_empty() {
            self.pending_inline_pages.push_back(page);
            return Ok(false);
        }
        let had_links = !page.links.is_empty();
        self.spawn_link_page(page.links, page.lz4_compressed)?;
        Ok(had_links)
    }

    fn accept_inline_page(&mut self, page: &ResultPage) -> Result<()> {
        let retained_bytes = self
            .options
            .max_decoded_chunk_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(self.options.max_page_bytes))
            .ok_or_else(|| anyhow!("Databricks inline memory size overflowed"))?;
        let permit = self.budget.acquire_input_chunk(retained_bytes)?;
        let chunk = decode_inline_arrow_page(page, self.options.max_decoded_chunk_bytes)?;
        self.record_decoded(chunk.emitted_rows()?, chunk.decoded_bytes)?;
        self.accept_chunk(chunk, Arc::new(permit))?;
        Ok(())
    }

    fn refill_page_lookahead(&mut self) -> Result<()> {
        // Keep one page more than the worker count ready, but never retain
        // signed links representing more than the shared source byte budget
        // plus the final bounded page. This fills small laptop and large-VM
        // worker pools without turning result pagination into an unbounded URL
        // queue or weakening the memory limit.
        let link_window = self
            .options
            .max_concurrent_downloads
            .saturating_add(self.options.max_page_links as usize);
        while self.pending_downloads.len() < link_window
            && self.pending_reserved_bytes < self.budget.normal_limit_bytes()
            && (!self.requested_page || !self.sequence.is_finished())
            && self.pending_batches.is_empty()
        {
            if !self.request_page()? {
                break;
            }
        }
        Ok(())
    }

    fn join_downloads(&mut self) {
        self.pending_batches.clear();
        let (replacement_sender, replacement_receiver) =
            std::sync::mpsc::channel();
        let old_sender = std::mem::replace(&mut self.download_event_sender, replacement_sender);
        let old_receiver =
            std::mem::replace(&mut self.download_event_receiver, replacement_receiver);
        drop(old_sender);
        drop(old_receiver);
        let downloads = self
            .pending_downloads
            .drain()
            .map(|(_, download)| download)
            .collect::<Vec<_>>();
        for download in downloads {
            let _ = download.handle.join();
        }
        self.pending_reserved_bytes = 0;
        self.pending_inline_pages.clear();
    }

    fn fail(&mut self, error: anyhow::Error) -> Option<Result<DatabricksBatch>> {
        self.closed = true;
        let _ = self.helper.close(true);
        self.join_downloads();
        Some(Err(error))
    }

    fn finish(&mut self) -> Option<Result<DatabricksBatch>> {
        let expected_rows = match self.sequence.finish() {
            Ok(rows) => rows,
            Err(error) => return self.fail(error),
        };
        if expected_rows != self.metrics.rows {
            return self.fail(anyhow!(
                "Databricks decoded row count did not match pagination"
            ));
        }
        self.closed = true;
        if let Err(error) = self.helper.close(false) {
            return Some(Err(error));
        }
        None
    }
}

impl Iterator for DatabricksQueryStream<'_> {
    type Item = Result<DatabricksBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.closed {
            return None;
        }
        loop {
            if let Some(batch) = self.pending_batches.pop_front() {
                return Some(Ok(batch));
            }
            if !self.pending_downloads.is_empty() {
                let event = loop {
                    match self
                        .download_event_receiver
                        .recv_timeout(Duration::from_secs(1))
                    {
                        Ok(event) => break event,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if self
                                .pending_downloads
                                .values()
                                .any(|download| download.handle.is_finished())
                            {
                                return self.fail(anyhow!(
                                    "Databricks download worker exited without completing its stream"
                                ));
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return self.fail(anyhow!(
                                "Databricks download worker event channel disconnected"
                            ));
                        }
                    }
                };
                let download_id = event.download_id;
                let event = match event.event {
                    Ok(event) => event,
                    Err(error) => return self.fail(error),
                };
                match event {
                    LinkEvent::Schema(schema) => {
                        if let Some(expected) = &self.expected_schema {
                            if expected.as_ref() != schema.as_ref() {
                                return self.fail(anyhow!(
                                    "Databricks changed Arrow schema between result chunks"
                                ));
                            }
                        } else {
                            self.expected_schema = Some(schema);
                        }
                    }
                    LinkEvent::Batch(batch) => {
                        return Some(Ok(batch));
                    }
                    LinkEvent::Finished {
                        emitted_rows,
                        decoded_bytes,
                        wire_bytes,
                        peak_retained_batch_bytes,
                    } => {
                        if let Err(error) = self.record_decoded(emitted_rows, decoded_bytes) {
                            return self.fail(error);
                        }
                        self.metrics.wire_bytes =
                            self.metrics.wire_bytes.saturating_add(wire_bytes);
                        let Some(download) = self.pending_downloads.remove(&download_id) else {
                            return self
                                .fail(anyhow!("Databricks download completed more than once"));
                        };
                        debug_assert!(peak_retained_batch_bytes <= download.reserved_bytes);
                        self.pending_reserved_bytes = self
                            .pending_reserved_bytes
                            .saturating_sub(download.reserved_bytes);
                        if download.handle.join().is_err() {
                            return self.fail(anyhow!("Databricks download worker panicked"));
                        }
                        if let Err(error) = self.refill_page_lookahead() {
                            return self.fail(error);
                        }
                    }
                }
                continue;
            }
            if let Some(page) = self.pending_inline_pages.pop_front() {
                if let Err(error) = self.accept_inline_page(&page) {
                    return self.fail(error);
                }
                continue;
            }
            if self.requested_page && self.sequence.is_finished() {
                return self.finish();
            }
            if let Err(error) = self.refill_page_lookahead() {
                return self.fail(error);
            }
        }
    }
}

impl Drop for DatabricksQueryStream<'_> {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.helper.close(true);
            self.closed = true;
        }
        self.join_downloads();
    }
}

fn validate_ipc_envelope(bytes: &[u8], max_message_bytes: u64) -> Result<()> {
    const CONTINUATION: u32 = 0xffff_ffff;
    let mut offset = 0usize;
    while offset < bytes.len() {
        let prefix = bytes
            .get(offset..offset + 4)
            .ok_or_else(|| anyhow!("Databricks returned truncated Arrow IPC metadata"))?;
        offset += 4;
        let mut metadata_len = u32::from_le_bytes(prefix.try_into().unwrap());
        if metadata_len == CONTINUATION {
            let prefix = bytes
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow!("Databricks returned truncated Arrow IPC metadata"))?;
            offset += 4;
            metadata_len = u32::from_le_bytes(prefix.try_into().unwrap());
        }
        if metadata_len == 0 {
            if offset != bytes.len() {
                bail!("Databricks returned trailing bytes after the Arrow stream");
            }
            return Ok(());
        }
        if u64::from(metadata_len) > max_message_bytes {
            bail!("Databricks Arrow IPC metadata exceeds the message limit");
        }
        let metadata_end = offset
            .checked_add(metadata_len as usize)
            .ok_or_else(|| anyhow!("Databricks Arrow IPC metadata length overflowed"))?;
        let metadata = bytes
            .get(offset..metadata_end)
            .ok_or_else(|| anyhow!("Databricks returned truncated Arrow IPC metadata"))?;
        let message = arrow_ipc::root_as_message(metadata)
            .map_err(|_| anyhow!("Databricks returned invalid Arrow IPC metadata"))?;
        let body_len = u64::try_from(message.bodyLength())
            .map_err(|_| anyhow!("Databricks returned a negative Arrow IPC body length"))?;
        if body_len > max_message_bytes {
            bail!("Databricks Arrow IPC body exceeds the message limit");
        }
        offset = metadata_end
            .checked_add(
                usize::try_from(body_len)
                    .map_err(|_| anyhow!("Databricks Arrow IPC body does not fit this platform"))?,
            )
            .ok_or_else(|| anyhow!("Databricks Arrow IPC body length overflowed"))?;
        if offset > bytes.len() {
            bail!("Databricks returned a truncated Arrow IPC body");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_ipc::writer::StreamWriter;
    use arrow_schema::{DataType, Field, Schema};
    use lz4_flex::frame::FrameEncoder;
    use std::sync::Arc;
    use std::{net::TcpListener, thread};

    static HELPER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn download_parallelism_scales_from_laptops_to_large_vms() {
        assert_eq!(download_parallelism(1), 4);
        assert_eq!(download_parallelism(11), 44);
        assert_eq!(download_parallelism(128), MAX_CONCURRENT_DOWNLOADS);
        assert_eq!(download_parallelism(256), MAX_CONCURRENT_DOWNLOADS);
        assert_eq!(download_parallelism(usize::MAX), MAX_CONCURRENT_DOWNLOADS);
        assert_eq!(
            QueryStreamOptions::default().max_concurrent_downloads,
            default_download_parallelism()
        );
        assert_eq!(
            QueryStreamOptions::default().download.read_timeout,
            Duration::from_secs(300)
        );

        let options = QueryStreamOptions {
            max_concurrent_downloads: MAX_CONCURRENT_DOWNLOADS + 1,
            ..QueryStreamOptions::default()
        };
        assert!(DatabricksQueryStream::validate_options(options, true).is_err());
    }

    #[test]
    fn link_reservation_backpressures_many_small_wide_batches() {
        let schema = Arc::new(Schema::new(
            (0..128)
                .map(|index| Field::new(format!("c{index}"), DataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        let arrays = (0..schema.fields().len())
            .map(|_| Arc::new(Int64Array::from(vec![1])) as ArrayRef)
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        let mut encoded = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut encoded, &schema).unwrap();
            for _ in 0..256 {
                writer.write(&batch).unwrap();
            }
            writer.finish().unwrap();
        }
        let decoded_bytes = encoded.len() as u64;
        let retained_bytes = retained_link_bytes(decoded_bytes).unwrap();
        let (result, events, _) = run_streaming_link(
            encoded,
            decoded_bytes,
            256,
            false,
            VecOrMapHeaders::default(),
        );
        let peak_retained_batch_bytes = match result.unwrap() {
            LinkEvent::Finished {
                peak_retained_batch_bytes,
                ..
            } => peak_retained_batch_bytes,
            _ => panic!("link did not finish"),
        };
        assert!(peak_retained_batch_bytes <= retained_bytes);
        let batches = events
            .into_iter()
            .filter_map(|event| match event.unwrap() {
                LinkEvent::Batch(batch) => Some(pinned_batch_bytes(&batch)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let flowed = batches.iter().copied().fold(0u64, u64::saturating_add);
        assert_eq!(batches.len(), 256);
        assert!(
            flowed > peak_retained_batch_bytes * 8,
            "link did not stream: {flowed} bytes flowed against a {peak_retained_batch_bytes} high water"
        );
    }

    /// A string column carries two buffers, three when nullable, against one
    /// for a non-null numeric, so text is where per-buffer accounting diverges
    /// most from the memory a batch holds.
    #[test]
    fn string_columns_are_charged_their_real_memory_not_their_buffer_count() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("colour", DataType::Utf8, false),
            Field::new("code", DataType::Int64, false),
        ]));
        // Large enough that the reservation is the 4x multiple, not the 4 MiB floor.
        let rows = 32_768;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| format!("article-{row:012}")),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| format!("product name {row:020}")),
                )) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| format!("colour {}", row % 64)),
                )) as ArrayRef,
                Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef,
            ],
        )
        .unwrap();

        let mut encoded = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut encoded, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let decoded_bytes = encoded.len() as u64;
        let retained_bytes = retained_link_bytes(decoded_bytes).unwrap();

        let (result, events, _) = run_streaming_link(
            encoded,
            decoded_bytes,
            rows as u64,
            false,
            VecOrMapHeaders::default(),
        );
        let peak = match result.expect("link failed") {
            LinkEvent::Finished {
                peak_retained_batch_bytes,
                ..
            } => peak_retained_batch_bytes,
            _ => panic!("link did not finish"),
        };
        assert!(
            peak <= retained_bytes,
            "charged {peak} against a {retained_bytes} reservation"
        );

        let decoded = events
            .into_iter()
            .filter_map(|event| match event.unwrap() {
                LinkEvent::Batch(batch) => Some(batch),
                _ => None,
            })
            .next()
            .expect("no batch was emitted");

        let charged = pinned_batch_bytes(&decoded);
        assert!(
            charged <= decoded_bytes,
            "charged {charged} bytes for {decoded_bytes} bytes of IPC"
        );

        // Guard against charging the reservation with `get_array_memory_size`,
        // which scores this same batch past the reservation.
        let inflated = decoded.get_array_memory_size() as u64;
        assert!(
            inflated > retained_bytes,
            "test data no longer exercises the inflation: {inflated} <= {retained_bytes}"
        );
        assert!(
            inflated >= charged * 4,
            "expected get_array_memory_size to multiply the real size: {inflated} vs {charged}"
        );
    }

    #[test]
    fn link_reservation_covers_nested_arrow_expansion() {
        let decoded_bytes = 9 * 1024 * 1024;
        assert_eq!(
            retained_link_bytes(decoded_bytes).unwrap(),
            4 * decoded_bytes
        );

        let small_decoded_bytes = 1024;
        assert_eq!(
            retained_link_bytes(small_decoded_bytes).unwrap(),
            small_decoded_bytes + 4 * 1024 * 1024
        );
    }

    fn ipc(rows: usize) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let values = Int64Array::from_iter_values((0..rows).map(|value| value as i64));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    fn ipc_value(value: i64) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![value]))],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    fn lz4_frame(bytes: &[u8]) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut writer = FrameEncoder::new(&mut encoded);
        writer.write_all(bytes).unwrap();
        writer.finish().unwrap();
        encoded
    }

    fn run_streaming_link(
        body: Vec<u8>,
        decoded_bytes: u64,
        advertised_rows: u64,
        compressed: bool,
        headers: VecOrMapHeaders,
    ) -> (Result<LinkEvent>, Vec<Result<LinkEvent>>, String) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let body_len = body.len();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]).to_ascii_lowercase();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            // A validation failure may close the response before the test
            // server has written the entire synthetic body.
            let _ = stream.write_all(&body);
            request
        });
        let link = CloudFetchLink {
            url: SignedUrl(format!("http://{address}/object?secret=value")),
            start_row_offset: 0,
            row_count: advertised_rows,
            decoded_bytes,
            headers,
        };
        let limits = DownloadLimits {
            max_wire_bytes: body_len as u64 + 1,
            allow_http: true,
            ..DownloadLimits::default()
        };
        let budget = ByteBudget::new(64 * 1024 * 1024).unwrap();
        let retained_bytes = retained_link_bytes(decoded_bytes).unwrap();
        let permit = Arc::new(budget.acquire(retained_bytes).unwrap());
        let (sender, receiver) = std::sync::mpsc::channel();
        let sender = DownloadEventSender {
            download_id: 0,
            sender,
        };
        let worker = thread::spawn(move || {
            stream_cloud_fetch_link(
                &cloud_fetch_http_agent(limits),
                &link,
                compressed,
                QueryStreamOptions {
                    max_decoded_chunk_bytes: 32 * 1024 * 1024,
                    download: limits,
                    ..QueryStreamOptions::default()
                },
                permit,
                retained_bytes,
                &sender,
            )
        });
        let events = receiver
            .into_iter()
            .map(|mut event| {
                if let Ok(LinkEvent::Batch(batch)) = &mut event.event {
                    batch.acknowledge();
                }
                event.event
            })
            .collect();
        (worker.join().unwrap(), events, server.join().unwrap())
    }

    #[test]
    fn link_download_waits_until_encode_releases_batch() {
        let body = ipc_value(7);
        let body_len = body.len();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        });
        let link = CloudFetchLink {
            url: SignedUrl(format!("http://{address}/object")),
            start_row_offset: 0,
            row_count: 1,
            decoded_bytes: body_len as u64,
            headers: VecOrMapHeaders::default(),
        };
        let limits = DownloadLimits {
            max_wire_bytes: body_len as u64 + 1,
            allow_http: true,
            ..DownloadLimits::default()
        };
        let budget = ByteBudget::new(64 * 1024 * 1024).unwrap();
        let retained_bytes = retained_link_bytes(body_len as u64).unwrap();
        let permit = Arc::new(budget.acquire(retained_bytes).unwrap());
        let (event_sender, event_receiver) = std::sync::mpsc::channel();
        let event_sender = DownloadEventSender {
            download_id: 0,
            sender: event_sender,
        };
        let (done_sender, done_receiver) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            let result = stream_cloud_fetch_link(
                &cloud_fetch_http_agent(limits),
                &link,
                false,
                QueryStreamOptions {
                    download: limits,
                    ..QueryStreamOptions::default()
                },
                permit,
                retained_bytes,
                &event_sender,
            );
            done_sender.send(result).unwrap();
        });

        assert!(matches!(
            event_receiver.recv().unwrap().event.unwrap(),
            LinkEvent::Schema(_)
        ));
        let batch = match event_receiver.recv().unwrap().event.unwrap() {
            LinkEvent::Batch(batch) => batch,
            _ => panic!("Cloud Fetch link did not download its first batch"),
        };
        assert!(matches!(
            done_receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        drop(batch);
        let result = done_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap_or_else(|_| panic!("Cloud Fetch link did not finish after consumption"));
        assert!(matches!(result.unwrap(), LinkEvent::Finished { .. }));
        worker.join().unwrap();
        server.join().unwrap();
    }

    fn schema_message_len(stream: &[u8]) -> usize {
        let mut offset = 0usize;
        let mut length = u32::from_le_bytes(stream[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if length == 0xffff_ffff {
            length = u32::from_le_bytes(stream[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }
        let metadata_end = offset + length as usize;
        let message = arrow_ipc::root_as_message(&stream[offset..metadata_end]).unwrap();
        metadata_end + message.bodyLength() as usize
    }

    fn write_fake_inline_helper(rows: usize) -> tempfile::TempDir {
        let stream_bytes = ipc(rows);
        let schema_end = schema_message_len(&stream_bytes);
        let schema = BASE64.encode(&stream_bytes[..schema_end]);
        let batch = BASE64.encode(&stream_bytes[schema_end..]);
        let script = r#"
import json
import sys

expected_id = 1
page_id = 0
for line in sys.stdin:
    request = json.loads(line)
    assert request["id"] == expected_id
    expected_id += 1
    op = request["op"]
    if op == "hello":
        result = {"protocol": 1, "connector": "4.3.0", "thrift": "0.22.0"}
    elif op == "schema":
        result = {"arrow_schema": "__SCHEMA__"}
    elif op == "open":
        result = {"query_id": "query-1"}
    elif op == "next":
        result = {
            "query_id": "query-1",
            "page_id": page_id,
            "start_row_offset": 0,
            "has_more_rows": False,
            "lz4_compressed": False,
            "arrow_schema": "__SCHEMA__",
            "links": [],
            "inline_batches": [{"data": "__BATCH__", "row_count": __ROWS__}],
        }
        page_id += 1
    elif op == "close":
        result = {}
    else:
        raise AssertionError(op)
    sys.stdout.write(json.dumps({"id": request["id"], "ok": True, "result": result}) + "\n")
    sys.stdout.flush()
"#
        .replace("__SCHEMA__", &schema)
        .replace("__BATCH__", &batch)
        .replace("__ROWS__", &rows.to_string());
        let directory = tempfile::tempdir().unwrap();
        let script_path = directory.path().join("fake_helper.py");
        std::fs::write(&script_path, script).unwrap();
        std::env::set_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT", &script_path);
        directory
    }

    fn fake_graph_config(source: DataSource, dataset_name: &str) -> crate::GraphConfig {
        use crate::graph_config::{ColumnSpec, IngestedType, TableSpec};

        crate::GraphConfig {
            dataset_name: dataset_name.into(),
            tables: vec![TableSpec {
                name: "customers".into(),
                columns: vec![ColumnSpec {
                    name: "id".into(),
                    source_dtype: None,
                    ingested_dtype: IngestedType::Int64,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                    multicategorical_delimiter: None,
                }],
                primary_key: Some("id".into()),
                create_time: None,
                end_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
                source,
                source_size: None,
                fkeys: Vec::new(),
                data_version: None,
                kumo_connector_id: None,
                kumo_source_table_name: None,
                group_by_timestamp: false,
            }],
            rowpack: None,
            num_shards: Some(1),
            edge_format: None,
            coercion_mode: Default::default(),
        }
    }

    #[test]
    fn signed_url_and_headers_are_redacted_from_debug() {
        let link = CloudFetchLink {
            url: SignedUrl("https://signed.example/?secret=value".into()),
            start_row_offset: 0,
            row_count: 1,
            decoded_bytes: 10,
            headers: VecOrMapHeaders([("x-secret".into(), "value".into())].into_iter().collect()),
        };
        let debug = format!("{link:?}");
        assert!(!debug.contains("signed.example"));
        assert!(!debug.contains("value"));
    }

    #[test]
    fn concatenated_lz4_frames_form_one_arrow_stream() {
        let plain = ipc(3);
        let split = plain.len() / 2;
        let mut object = lz4_frame(&plain[..split]);
        object.extend(lz4_frame(&plain[split..]));

        let (finished, events, _) = run_streaming_link(
            object,
            plain.len() as u64,
            3,
            true,
            VecOrMapHeaders::default(),
        );
        assert!(matches!(
            finished.unwrap(),
            LinkEvent::Finished {
                emitted_rows: 3,
                ..
            }
        ));
        assert_eq!(
            events
                .into_iter()
                .filter_map(|event| match event.unwrap() {
                    LinkEvent::Batch(batch) => Some(batch.num_rows()),
                    _ => None,
                })
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn valid_overread_is_truncated_before_consumer() {
        let plain = ipc(5);
        let (finished, events, _) = run_streaming_link(
            plain.clone(),
            plain.len() as u64,
            3,
            false,
            VecOrMapHeaders::default(),
        );
        assert!(matches!(
            finished.unwrap(),
            LinkEvent::Finished {
                emitted_rows: 3,
                ..
            }
        ));
        assert_eq!(
            events
                .into_iter()
                .filter_map(|event| match event.unwrap() {
                    LinkEvent::Batch(batch) => Some(batch.num_rows()),
                    _ => None,
                })
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn underread_is_rejected() {
        let plain = ipc(2);
        let (result, _, _) = run_streaming_link(
            plain.clone(),
            plain.len() as u64,
            3,
            false,
            VecOrMapHeaders::default(),
        );
        let error = match result {
            Ok(_) => panic!("under-read unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("fewer rows"));
    }

    #[test]
    fn decoded_body_cannot_exceed_its_advertised_budget() {
        let plain = ipc(3);
        let advertised = plain.len() as u64 - 1;
        let (result, _, _) =
            run_streaming_link(plain, advertised, 3, false, VecOrMapHeaders::default());
        let error = match result {
            Ok(_) => panic!("oversized decoded body unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("invalid Arrow IPC data")
                || error.to_string().contains("exceeds its limit")
        );
    }

    #[test]
    fn page_sequence_rejects_link_gaps() {
        let mut sequence = PageSequence::new("query".into()).unwrap();
        let page = ResultPage {
            query_id: "query".into(),
            page_id: 0,
            start_row_offset: 0,
            has_more_rows: false,
            lz4_compressed: true,
            arrow_schema: None,
            links: vec![CloudFetchLink {
                url: SignedUrl("https://signed.example".into()),
                start_row_offset: 1,
                row_count: 1,
                decoded_bytes: 1,
                headers: VecOrMapHeaders::default(),
            }],
            inline_batches: Vec::new(),
        };
        assert!(sequence.validate(&page).is_err());
    }

    #[test]
    fn object_download_forwards_only_safe_link_headers() {
        let body = ipc(1);
        let (result, _, request) = run_streaming_link(
            body.clone(),
            body.len() as u64,
            1,
            false,
            VecOrMapHeaders(
                [
                    ("Authorization".into(), "Bearer database-secret".into()),
                    ("Cookie".into(), "session=database-secret".into()),
                    ("x-ms-encryption-key".into(), "opaque".into()),
                ]
                .into_iter()
                .collect(),
            ),
        );
        result.unwrap();
        assert!(request.contains("x-ms-encryption-key: opaque"));
        assert!(!request.contains("authorization:"));
        assert!(!request.contains("cookie:"));
    }

    #[test]
    fn query_stream_bounds_page_lookahead_and_delivers_links_as_they_finish() {
        let _guard = HELPER_ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let bodies = (0..4).map(ipc_value).collect::<Vec<_>>();
        let decoded_sizes = bodies.iter().map(Vec::len).collect::<Vec<_>>();
        let server = thread::spawn(move || {
            // Do not finish any current-page response until both links from
            // the lookahead page have started. This deadlocks if pagination
            // waits for page zero to complete instead of overlapping it.
            let mut requests = Vec::new();
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 1024];
                let count = stream.read(&mut request).unwrap();
                let first_line = String::from_utf8_lossy(&request[..count])
                    .lines()
                    .next()
                    .unwrap()
                    .to_string();
                let index = first_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .trim_start_matches('/')
                    .parse::<usize>()
                    .unwrap();
                requests.push((index, stream));
            }
            requests.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
            for (index, mut stream) in requests {
                let body = &bodies[index];
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
                thread::sleep(Duration::from_millis(25));
            }
        });

        let page = |page_id: usize| {
            let start = page_id * 2;
            serde_json::json!({
                "query_id": "query-1",
                "page_id": page_id,
                "start_row_offset": start,
                "has_more_rows": page_id == 0,
                "lz4_compressed": false,
                "links": [
                    {
                        "url": format!("http://{address}/{start}"),
                        "start_row_offset": start,
                        "row_count": 1,
                        "decoded_bytes": decoded_sizes[start],
                        "headers": {}
                    },
                    {
                        "url": format!("http://{address}/{}", start + 1),
                        "start_row_offset": start + 1,
                        "row_count": 1,
                        "decoded_bytes": decoded_sizes[start + 1],
                        "headers": {}
                    }
                ],
                "inline_batches": []
            })
        };
        let pages = serde_json::to_string(&vec![page(0), page(1)]).unwrap();
        let script = r#"
import json
import sys

pages = json.loads(r'''__PAGES__''')
expected_id = 1
page_id = 0
for line in sys.stdin:
    request = json.loads(line)
    assert request["id"] == expected_id
    expected_id += 1
    op = request["op"]
    if op == "hello":
        result = {"protocol": 1, "connector": "4.3.0", "thrift": "0.22.0"}
    elif op == "open":
        result = {"query_id": "query-1"}
    elif op == "next":
        result = pages[page_id]
        page_id += 1
    elif op == "close":
        result = {}
    else:
        raise AssertionError(op)
    sys.stdout.write(json.dumps({"id": request["id"], "ok": True, "result": result}) + "\n")
    sys.stdout.flush()
"#
        .replace("__PAGES__", &pages);
        let directory = tempfile::tempdir().unwrap();
        let script_path = directory.path().join("fake_cloud_helper.py");
        std::fs::write(&script_path, script).unwrap();
        std::env::set_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT", &script_path);

        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id FROM source".into(),
            auth_profile: None,
        };
        let mut helper = DatabricksHelper::spawn().unwrap();
        helper.hello().unwrap();
        let options = QueryStreamOptions {
            memory_budget_bytes: 64 * 1024 * 1024,
            download: DownloadLimits {
                max_wire_bytes: 1024 * 1024,
                allow_http: true,
                ..DownloadLimits::default()
            },
            ..QueryStreamOptions::default()
        };
        let mut query = DatabricksQueryStream::open(&mut helper, &source, None, options).unwrap();
        let values = query
            .by_ref()
            .map(|batch| {
                let batch = batch.unwrap();
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>();
        let metrics = query.metrics();
        drop(query);
        server.join().unwrap();
        std::env::remove_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT");
        assert_eq!(values, vec![3, 2, 1, 0]);
        assert_eq!(metrics.rows, 4);
        assert_eq!(
            metrics.wire_bytes,
            decoded_sizes.iter().sum::<usize>() as u64
        );
    }

    #[test]
    fn byte_budget_tracks_retained_high_water() {
        let budget = ByteBudget::new(10).unwrap();
        let first = budget.acquire(4).unwrap();
        let second = budget.acquire(6).unwrap();
        assert_eq!(budget.high_water_bytes(), 10);
        drop(first);
        drop(second);
        assert!(budget.acquire(11).is_err());
    }

    #[test]
    fn byte_budget_admits_downloads_in_source_order() {
        let budget = ByteBudget::new(10).unwrap();
        let first_ticket = budget.issue_ticket(6).unwrap();
        let second_ticket = budget.issue_ticket(6).unwrap();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let (started_tx, started_rx) = std::sync::mpsc::channel();

        let second_budget = Arc::clone(&budget);
        let second_tx = result_tx.clone();
        let second = thread::spawn(move || {
            started_tx.send(()).unwrap();
            let permit = second_budget.acquire_ticket(second_ticket, 6).unwrap();
            second_tx.send((2, permit)).unwrap();
        });
        started_rx.recv().unwrap();

        let first_budget = Arc::clone(&budget);
        let first = thread::spawn(move || {
            let permit = first_budget.acquire_ticket(first_ticket, 6).unwrap();
            result_tx.send((1, permit)).unwrap();
        });

        let (id, first_permit) = result_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(id, 1);
        drop(first_permit);
        let (id, second_permit) = result_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(id, 2);
        drop(second_permit);
        first.join().unwrap();
        second.join().unwrap();
    }

    #[test]
    fn download_slots_admit_waiters_in_source_order() {
        let slots = DownloadSlots::new(1);
        let first_ticket = slots.issue_ticket().unwrap();
        let second_ticket = slots.issue_ticket().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();

        let second_slots = Arc::clone(&slots);
        let second_tx = tx.clone();
        let second = thread::spawn(move || {
            let permit = second_slots.acquire(second_ticket).unwrap();
            second_tx.send((2, permit)).unwrap();
        });
        let first_slots = Arc::clone(&slots);
        let first = thread::spawn(move || {
            let permit = first_slots.acquire(first_ticket).unwrap();
            tx.send((1, permit)).unwrap();
        });

        let (id, first_permit) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(id, 1);
        drop(first_permit);
        let (id, second_permit) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(id, 2);
        drop(second_permit);
        first.join().unwrap();
        second.join().unwrap();
    }

    #[test]
    fn helper_and_inline_query_stream_are_serialized_end_to_end() {
        let _guard = HELPER_ENV_LOCK.lock().unwrap();
        let _directory = write_fake_inline_helper(3);

        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id FROM values(1), (2), (3) AS t(id)".into(),
            auth_profile: None,
        };
        let mut helper = DatabricksHelper::spawn().unwrap();
        assert_eq!(helper.hello().unwrap().protocol, 1);
        let mut query =
            DatabricksQueryStream::open(&mut helper, &source, None, QueryStreamOptions::default())
                .unwrap();
        let rows: usize = query.by_ref().map(|batch| batch.unwrap().num_rows()).sum();
        assert_eq!(rows, 3);
        drop(query);
        std::env::remove_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT");
    }

    #[test]
    fn zero_row_query_completes_without_batches() {
        let _guard = HELPER_ENV_LOCK.lock().unwrap();
        let _directory = write_fake_inline_helper(0);
        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id WHERE false".into(),
            auth_profile: None,
        };
        let mut helper = DatabricksHelper::spawn().unwrap();
        helper.hello().unwrap();
        let query =
            DatabricksQueryStream::open(&mut helper, &source, None, QueryStreamOptions::default())
                .unwrap();
        assert!(query.collect::<Result<Vec<_>>>().unwrap().is_empty());
        std::env::remove_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT");
    }

    #[test]
    fn fake_databricks_source_builds_a_graph_in_one_query() {
        use crate::{ingest_with_stats_from_source, Dest, IngestOptions};

        let _guard = HELPER_ENV_LOCK.lock().unwrap();
        let _directory = write_fake_inline_helper(3);
        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id FROM values(1), (2), (3) AS t(id)".into(),
            auth_profile: Some("standalone".into()),
        };
        let config = fake_graph_config(source, "fake_databricks");
        let output = tempfile::tempdir().unwrap();
        let (metadata, stats) = ingest_with_stats_from_source(
            config,
            None,
            output.path().to_str().unwrap(),
            Dest::Local,
            IngestOptions {
                parallel: false,
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap();
        assert_eq!(metadata.node_types[0].max_id, 2);
        assert_eq!(stats.total_nodes, 3);
        let persisted =
            std::fs::read_to_string(output.path().join("diskgraph_config.json")).unwrap();
        assert!(persisted.contains(r#""auth_profile": "standalone""#));
        let report = std::fs::read_to_string(output.path().join("ingest_stats.json")).unwrap();
        assert!(report.contains(r#""auth_profile": "standalone""#));
        std::env::remove_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT");
    }

    #[test]
    fn resume_after_encode_does_not_initialize_credentials_or_helper() {
        use crate::{ingest_with_stats_from_source, Dest, IngestOptions, IngestPhase};

        let _guard = HELPER_ENV_LOCK.lock().unwrap();
        let _directory = write_fake_inline_helper(3);
        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id FROM values(1), (2), (3) AS t(id)".into(),
            auth_profile: Some("missing-after-encode".into()),
        };
        let config = fake_graph_config(source, "fake_databricks_resume");
        let output = tempfile::tempdir().unwrap();
        let root = output.path().join("snapshot");
        std::fs::create_dir_all(&root).unwrap();
        let target =
            Arc::new(crate::io::OutputTarget::parse(root.to_str().unwrap(), Dest::Local).unwrap());
        target.prepare().unwrap();
        let marker_identity = serde_json::to_vec(&config).unwrap();
        let mut context = crate::IngestContext::new(
            config.clone(),
            target,
            IngestOptions {
                parallel: false,
                ..Default::default()
            },
            root.to_str().unwrap(),
            std::collections::HashMap::new(),
        )
        .unwrap();
        context.prepare_input_schemas().unwrap();
        crate::run_phase_with_marker_payload(
            &root,
            "encode",
            false,
            None,
            &marker_identity,
            || context.build_temp_shards(),
        )
        .unwrap();
        drop(context);

        std::env::remove_var("DISKGRAPH_DATABRICKS_HELPER_SCRIPT");
        let previous_python = std::env::var_os("DISKGRAPH_PYTHON");
        std::env::set_var("DISKGRAPH_PYTHON", "/definitely/missing/python");
        let (metadata, _) = ingest_with_stats_from_source(
            config,
            None,
            root.to_str().unwrap(),
            Dest::Local,
            IngestOptions {
                parallel: false,
                resume_from_phase: Some(IngestPhase::PkMap),
                ..Default::default()
            },
            None,
            None,
        )
        .unwrap();
        assert_eq!(metadata.node_types[0].max_id, 2);
        if let Some(previous_python) = previous_python {
            std::env::set_var("DISKGRAPH_PYTHON", previous_python);
        } else {
            std::env::remove_var("DISKGRAPH_PYTHON");
        }
    }
}
