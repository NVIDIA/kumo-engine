// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures_util::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, ObjectStore, ObjectStoreExt, PutPayload};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::task::JoinSet;
use tracing::{debug, error, info, instrument, warn};

#[derive(Clone, Debug)]
pub struct CloudTransferConfig {
    pub part_size: usize,
    pub download_part_size: usize,
    pub max_queue: usize,
    pub max_files_in_flight: usize,
    pub max_part_requests: usize,
    pub max_file_retries: usize,
    pub buffer_count: usize,
    pub max_bytes: u64,
}

impl CloudTransferConfig {
    pub fn from_env() -> Self {
        let cpu_default = diskgraph_limits::effective_parallelism();
        let max_queue = env_usize("DISKGRAPH_CLOUD_QUEUE_MAX", cpu_default.max(1) / 8 + 1);
        let max_files_in_flight = env_usize("DISKGRAPH_CLOUD_FILES_IN_FLIGHT", cpu_default.max(1));
        let max_part_requests = env_usize("DISKGRAPH_CLOUD_MAX_REQUESTS", cpu_default.max(1));
        let part_size = env_usize("DISKGRAPH_CLOUD_PART_SIZE_MB", 64) * 1024 * 1024;
        let download_part_size = env_usize("DISKGRAPH_CLOUD_DOWNLOAD_PART_MB", 0).max(part_size);
        let buffer_count = env_usize(
            "DISKGRAPH_CLOUD_FILE_BUFFER_COUNT",
            cpu_default.saturating_mul(2).max(1),
        );
        let max_file_retries = env_usize("DISKGRAPH_CLOUD_FILE_RETRY_MAX", 2);
        let max_bytes = env_u64("DISKGRAPH_CLOUD_MAX_BYTES", default_cloud_max_bytes());
        Self {
            part_size,
            download_part_size,
            max_queue: max_queue.max(1),
            max_files_in_flight: max_files_in_flight.max(1),
            max_part_requests: max_part_requests.max(1),
            max_file_retries,
            buffer_count: buffer_count.max(1),
            max_bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CloudObjectCondition {
    pub size: u64,
    pub e_tag: Option<String>,
    pub version: Option<String>,
}

impl CloudObjectCondition {
    /// Return the known size only when GETs are pinned to an immutable object.
    pub fn pinned_size(&self) -> Option<u64> {
        (self.e_tag.is_some() || self.version.is_some()).then_some(self.size)
    }

    fn get_options(&self) -> GetOptions {
        GetOptions::new()
            .with_if_match(self.e_tag.clone())
            .with_version(self.version.clone())
    }

    fn validate_meta(&self, meta: &object_store::ObjectMeta) -> Result<()> {
        if meta.size != self.size {
            return Err(anyhow!(
                "source object changed size: expected {}, observed {}",
                self.size,
                meta.size
            ));
        }
        if self
            .e_tag
            .as_ref()
            .is_some_and(|expected| meta.e_tag.as_ref() != Some(expected))
        {
            return Err(anyhow!("source object changed ETag"));
        }
        if self
            .version
            .as_ref()
            .is_some_and(|expected| meta.version.as_ref() != Some(expected))
        {
            return Err(anyhow!("source object changed version"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CloudTransferStatsHandle {
    inner: Arc<CloudTransferStatsInner>,
}

#[derive(Clone, Debug)]
pub struct CloudTransferStatsSnapshot {
    pub total_buffers: usize,
    pub available_buffers: usize,
    pub queued_jobs: usize,
    pub inflight_files: usize,
    pub inflight_parts: usize,
    pub parts_started: u64,
    pub parts_errors: u64,
    pub budget_max_bytes: u64,
    pub budget_used_bytes: u64,
    pub last_error: Option<String>,
}

#[derive(Debug)]
pub struct CloudTransferTicket {
    result: Option<Result<()>>,
    receiver: Option<oneshot::Receiver<Result<()>>>,
}

impl CloudTransferTicket {
    pub fn ready(result: Result<()>) -> Self {
        Self {
            result: Some(result),
            receiver: None,
        }
    }

    pub fn wait(mut self) -> Result<()> {
        if let Some(result) = self.result.take() {
            return result;
        }
        let receiver = self.receiver.take().expect("cloud transfer ticket");
        receiver
            .blocking_recv()
            .map_err(|_| anyhow!("cloud transfer stopped"))?
    }
}

#[derive(Debug)]
pub struct CloudTransferDownloadTicket {
    receiver: Option<oneshot::Receiver<Result<PooledBuffer>>>,
}

impl CloudTransferDownloadTicket {
    pub fn wait(mut self) -> Result<PooledBuffer> {
        let receiver = self
            .receiver
            .take()
            .expect("cloud transfer download ticket");
        receiver
            .blocking_recv()
            .map_err(|_| anyhow!("cloud transfer stopped"))?
    }
}

#[derive(Debug)]
pub struct CloudTransferBytesTicket {
    ticket: CloudTransferDownloadTicket,
}

impl CloudTransferBytesTicket {
    pub fn wait(self) -> Result<PooledBytes> {
        let buffer = self.ticket.wait()?;
        Ok(buffer.into_pooled_bytes())
    }
}

#[derive(Debug)]
struct CloudTransferRangeTicket {
    receiver: Option<oneshot::Receiver<Result<PooledBuffer>>>,
}

impl CloudTransferRangeTicket {
    fn wait(mut self) -> Result<PooledBytes> {
        let receiver = self.receiver.take().expect("cloud transfer range ticket");
        let buffer = receiver
            .blocking_recv()
            .map_err(|_| anyhow!("cloud transfer stopped"))??;
        Ok(buffer.into_pooled_bytes())
    }
}

#[derive(Debug)]
struct CloudTransferHeadTicket {
    receiver: Option<oneshot::Receiver<Result<object_store::ObjectMeta>>>,
}

impl CloudTransferHeadTicket {
    fn wait(mut self) -> Result<object_store::ObjectMeta> {
        let receiver = self.receiver.take().expect("cloud transfer head ticket");
        receiver
            .blocking_recv()
            .map_err(|_| anyhow!("cloud transfer stopped"))?
    }
}

#[derive(Debug)]
struct CloudTransferStatsInner {
    total_buffers: usize,
    available_buffers: AtomicUsize,
    queued_jobs: AtomicUsize,
    inflight_files: AtomicUsize,
    inflight_parts: AtomicUsize,
    parts_started: AtomicU64,
    parts_errors: AtomicU64,
    budget: Arc<ByteBudget>,
    last_error: Mutex<Option<String>>,
}

impl CloudTransferStatsHandle {
    pub fn snapshot(&self) -> CloudTransferStatsSnapshot {
        let last_error = self
            .inner
            .last_error
            .lock()
            .ok()
            .and_then(|value| value.clone());
        CloudTransferStatsSnapshot {
            total_buffers: self.inner.total_buffers,
            available_buffers: self.inner.available_buffers.load(Ordering::Relaxed),
            queued_jobs: self.inner.queued_jobs.load(Ordering::Relaxed),
            inflight_files: self.inner.inflight_files.load(Ordering::Relaxed),
            inflight_parts: self.inner.inflight_parts.load(Ordering::Relaxed),
            parts_started: self.inner.parts_started.load(Ordering::Relaxed),
            parts_errors: self.inner.parts_errors.load(Ordering::Relaxed),
            budget_max_bytes: self.inner.budget.max_bytes(),
            budget_used_bytes: self.inner.budget.used_bytes(),
            last_error,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ByteBudget {
    max_bytes: u64,
    used_bytes: AtomicU64,
}

impl ByteBudget {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            used_bytes: AtomicU64::new(0),
        }
    }

    fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    fn reserve(&self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        loop {
            let used = self.used_bytes.load(Ordering::Relaxed);
            let next = used.saturating_add(bytes);
            if next > self.max_bytes {
                return Err(anyhow!(
                    "cloud transfer budget exceeded: used {} + {} > {}",
                    used,
                    bytes,
                    self.max_bytes
                ));
            }
            if self
                .used_bytes
                .compare_exchange(used, next, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.used_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }
}

#[derive(Clone, Debug)]
pub struct BufferPool {
    inner: Arc<BufferPoolInner>,
}

#[derive(Debug)]
struct BufferPoolInner {
    buffers: Mutex<Vec<Vec<u8>>>,
    available: Condvar,
    stats: Arc<CloudTransferStatsInner>,
    budget: Arc<ByteBudget>,
}

impl BufferPool {
    pub(crate) fn new(buffer_count: usize, budget: Arc<ByteBudget>) -> Self {
        let mut buffers = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            buffers.push(Vec::new());
        }
        let stats = Arc::new(CloudTransferStatsInner {
            total_buffers: buffer_count,
            available_buffers: AtomicUsize::new(buffer_count),
            queued_jobs: AtomicUsize::new(0),
            inflight_files: AtomicUsize::new(0),
            inflight_parts: AtomicUsize::new(0),
            parts_started: AtomicU64::new(0),
            parts_errors: AtomicU64::new(0),
            budget: Arc::clone(&budget),
            last_error: Mutex::new(None),
        });
        Self {
            inner: Arc::new(BufferPoolInner {
                buffers: Mutex::new(buffers),
                available: Condvar::new(),
                stats,
                budget,
            }),
        }
    }

    pub fn acquire(&self) -> PooledBuffer {
        let buffer = self.acquire_raw();
        self.wrap_accounted(buffer)
    }

    pub fn acquire_raw(&self) -> Vec<u8> {
        let mut guard = self.inner.buffers.lock().expect("cloud buffer pool lock");
        loop {
            if let Some(buffer) = guard.pop() {
                self.inner
                    .stats
                    .available_buffers
                    .fetch_sub(1, Ordering::Relaxed);
                return buffer;
            }
            guard = self
                .inner
                .available
                .wait(guard)
                .expect("cloud buffer pool wait");
        }
    }

    fn reserve_with_idle_reclaim(&self, bytes: u64) -> Result<()> {
        if self.inner.budget.reserve(bytes).is_ok() {
            return Ok(());
        }
        let mut guard = self.inner.buffers.lock().expect("cloud buffer pool lock");
        let reclaimed = guard
            .iter_mut()
            .map(|buffer| {
                let capacity = buffer.capacity() as u64;
                *buffer = Vec::new();
                capacity
            })
            .sum();
        self.inner.budget.release(reclaimed);
        drop(guard);
        self.inner.budget.reserve(bytes)
    }

    pub fn acquire_raw_sized(&self, len: usize) -> Result<Vec<u8>> {
        let mut buffer = self.acquire_raw();
        let old_cap = buffer.capacity();
        let mut reserved = 0usize;
        if len > old_cap {
            reserved = len - old_cap;
            if let Err(err) = self.reserve_with_idle_reclaim(reserved as u64) {
                self.release(buffer);
                return Err(err);
            }
            // Avoid retaining Vec's amortized 2x growth in an idle pool slot.
            buffer.reserve_exact(len);
        }
        let new_cap = buffer.capacity();
        if new_cap > old_cap + reserved {
            let extra = new_cap - (old_cap + reserved);
            if let Err(err) = self.reserve_with_idle_reclaim(extra as u64) {
                // Discard the grown allocation and undo its accounting.
                self.inner
                    .budget
                    .release(old_cap.saturating_add(reserved) as u64);
                drop(buffer);
                self.release(Vec::new());
                return Err(err);
            }
        } else if new_cap < old_cap + reserved {
            let give_back = old_cap + reserved - new_cap;
            self.inner.budget.release(give_back as u64);
        }
        buffer.resize(len, 0);
        Ok(buffer)
    }

    fn release(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        let mut guard = self.inner.buffers.lock().expect("cloud buffer pool lock");
        guard.push(buffer);
        self.inner
            .stats
            .available_buffers
            .fetch_add(1, Ordering::Relaxed);
        self.inner.available.notify_one();
    }

    pub fn wrap(&self, buffer: Vec<u8>) -> PooledBuffer {
        self.wrap_unaccounted(buffer)
    }

    fn wrap_accounted(&self, buffer: Vec<u8>) -> PooledBuffer {
        self.wrap_with_budget(buffer, false)
    }

    fn wrap_unaccounted(&self, buffer: Vec<u8>) -> PooledBuffer {
        self.wrap_with_budget(buffer, true)
    }

    fn wrap_with_budget(&self, buffer: Vec<u8>, reserve_capacity: bool) -> PooledBuffer {
        let capacity = buffer.capacity();
        if reserve_capacity && capacity > 0 {
            if let Err(err) = self.inner.budget.reserve(capacity as u64) {
                panic!("cloud transfer budget exceeded: {err}");
            }
        }
        PooledBuffer {
            cursor: Cursor::new(buffer),
            pool: Some(self.clone()),
            budget: Some(Arc::clone(&self.inner.budget)),
            tracked_capacity: capacity,
        }
    }
}

#[derive(Debug)]
pub struct PooledBuffer {
    cursor: Cursor<Vec<u8>>,
    pool: Option<BufferPool>,
    budget: Option<Arc<ByteBudget>>,
    tracked_capacity: usize,
}

impl PooledBuffer {
    pub fn into_inner(mut self) -> Vec<u8> {
        self.pool = None;
        self.budget = None;
        let cursor = std::mem::replace(&mut self.cursor, Cursor::new(Vec::new()));
        cursor.into_inner()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.cursor.get_ref().as_slice()
    }

    pub fn resize(&mut self, len: usize) -> Result<()> {
        let old_cap = self.tracked_capacity;
        let mut reserved = 0usize;
        if len > old_cap {
            reserved = len - old_cap;
            if let Some(budget) = &self.budget {
                budget.reserve(reserved as u64)?;
            }
        }
        self.cursor.get_mut().resize(len, 0);
        let new_cap = self.cursor.get_ref().capacity();
        if let Some(budget) = &self.budget {
            if new_cap > old_cap + reserved {
                let extra = new_cap - (old_cap + reserved);
                budget.reserve(extra as u64)?;
            } else if new_cap < old_cap + reserved {
                let give_back = old_cap + reserved - new_cap;
                budget.release(give_back as u64);
            }
        }
        self.tracked_capacity = new_cap;
        Ok(())
    }

    pub fn into_pooled_bytes(mut self) -> PooledBytes {
        let pool = self.pool.take();
        self.budget = None;
        let buffer = std::mem::take(self.cursor.get_mut());
        PooledBytes::new(buffer, pool)
    }
}

impl Write for PooledBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let old_cap = self.tracked_capacity;
        let pos = self.cursor.position() as usize;
        let cur_len = self.cursor.get_ref().len();
        let desired_len = cur_len.max(pos.saturating_add(buf.len()));
        let mut reserved = 0usize;
        if desired_len > old_cap {
            reserved = desired_len - old_cap;
            if let Some(budget) = &self.budget {
                budget
                    .reserve(reserved as u64)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
            }
        }
        let written = self.cursor.write(buf)?;
        let new_cap = self.cursor.get_ref().capacity();
        if let Some(budget) = &self.budget {
            if new_cap > old_cap + reserved {
                let extra = new_cap - (old_cap + reserved);
                budget
                    .reserve(extra as u64)
                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
            } else if new_cap < old_cap + reserved {
                let give_back = old_cap + reserved - new_cap;
                budget.release(give_back as u64);
            }
        }
        self.tracked_capacity = new_cap;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.cursor.flush()
    }
}

impl Seek for PooledBuffer {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let buffer = std::mem::take(self.cursor.get_mut());
            pool.release(buffer);
        }
    }
}

#[derive(Debug)]
pub struct PooledBytes {
    bytes: Bytes,
}

impl PooledBytes {
    fn new(buffer: Vec<u8>, pool: Option<BufferPool>) -> Self {
        let bytes = match pool {
            Some(pool) => Bytes::from_owner(ReturnToPool {
                buffer: Some(buffer),
                pool,
            }),
            None => Bytes::from(buffer),
        };
        Self { bytes }
    }

    pub fn bytes(&self) -> Bytes {
        self.bytes.clone()
    }
}

struct ReturnToPool {
    buffer: Option<Vec<u8>>,
    pool: BufferPool,
}

impl AsRef<[u8]> for ReturnToPool {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_deref().unwrap_or_default()
    }
}

impl Drop for ReturnToPool {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

#[derive(Clone)]
pub struct CloudTransfer {
    inner: Arc<CloudTransferInner>,
}

struct CloudTransferInner {
    store: Arc<dyn ObjectStore>,
    sender: mpsc::Sender<TransferJob>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
}

impl fmt::Debug for CloudTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudTransfer").finish()
    }
}

impl CloudTransfer {
    pub fn new(store: Arc<dyn ObjectStore>, config: CloudTransferConfig) -> Self {
        let budget = Arc::new(ByteBudget::new(config.max_bytes));
        let pool = BufferPool::new(config.buffer_count, Arc::clone(&budget));
        let stats = Arc::clone(&pool.inner.stats);
        let (sender, receiver) = mpsc::channel(config.max_queue);
        let store_clone = Arc::clone(&store);
        let pool_clone = pool.clone();
        let stats_clone = Arc::clone(&stats);
        thread::spawn(move || {
            let rt = Runtime::new().expect("cloud transfer runtime");
            let _ = rt.block_on(upload_loop(
                store,
                receiver,
                pool_clone,
                stats_clone,
                config,
            ));
        });
        Self {
            inner: Arc::new(CloudTransferInner {
                store: store_clone,
                sender,
                pool,
                stats,
            }),
        }
    }

    pub fn acquire_buffer(&self) -> PooledBuffer {
        self.inner.pool.acquire()
    }

    pub fn submit(&self, path: ObjectPath, buffer: Vec<u8>) -> Result<CloudTransferTicket> {
        self.submit_upload(path, buffer)
    }

    pub fn submit_upload(&self, path: ObjectPath, buffer: Vec<u8>) -> Result<CloudTransferTicket> {
        let path_str = path.to_string();
        let bytes = buffer.len();
        let (tx, rx) = oneshot::channel();
        let job = UploadJob {
            path,
            buffer,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if let Err(err) = self.inner.sender.blocking_send(TransferJob::Upload(job)) {
            let job = err.0.into_upload();
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            self.inner.pool.release(job.buffer);
            warn!(path = %path_str, "Upload queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        info!(path = %path_str, bytes, "Upload job queued");
        Ok(CloudTransferTicket {
            result: None,
            receiver: Some(rx),
        })
    }

    pub fn upload_bytes(&self, path: ObjectPath, buffer: Vec<u8>) -> Result<()> {
        self.submit_upload(path, buffer)?.wait()
    }

    /// Upload a local file via multipart, reading parts on demand from
    /// disk so peak memory is bounded by `part_size × inflight_parts`
    /// rather than by the file size.
    ///
    /// Functionally equivalent to `submit_upload(path, fs::read(src)?)`,
    /// but never holds the full file in memory: each part is read into
    /// a fresh `Vec<u8>` of `part_size` bytes via `pread` (positioned
    /// read, concurrent-safe across part tasks), submitted as a
    /// `put_part` payload, and dropped as the part completes. Retries
    /// re-read from disk — on transient upload failure we don't keep
    /// the file's bytes resident.
    ///
    /// Use this for any callers that have already produced the file on
    /// local disk (`OutputSink::Local` post-flush, the post-build
    /// reconciliation sweep, etc.). For files constructed in memory
    /// and not yet on disk, prefer [`Self::submit_upload`] — paying for
    /// a write+re-read cycle just to hand the bytes back to the upload
    /// queue is silly.
    pub fn submit_upload_from_path(
        &self,
        path: ObjectPath,
        src_path: PathBuf,
    ) -> Result<CloudTransferTicket> {
        let path_str = path.to_string();
        let (tx, rx) = oneshot::channel();
        let job = UploadFromPathJob {
            path,
            src_path,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if let Err(err) = self
            .inner
            .sender
            .blocking_send(TransferJob::UploadFromPath(job))
        {
            // No buffer to release back to the pool — the job carries
            // a path, not bytes. Just account for the dropped queue
            // slot and surface the error.
            let _ = err;
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            warn!(path = %path_str, "Upload-from-path queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        info!(path = %path_str, "Upload-from-path job queued");
        Ok(CloudTransferTicket {
            result: None,
            receiver: Some(rx),
        })
    }

    pub fn submit_download(&self, path: ObjectPath) -> Result<CloudTransferDownloadTicket> {
        self.submit_download_condition(path, None)
    }

    pub fn submit_download_condition(
        &self,
        path: ObjectPath,
        condition: Option<CloudObjectCondition>,
    ) -> Result<CloudTransferDownloadTicket> {
        let path_str = path.to_string();
        let (tx, rx) = oneshot::channel();
        let job = DownloadJob {
            path,
            condition,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if self
            .inner
            .sender
            .blocking_send(TransferJob::Download(job))
            .is_err()
        {
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            warn!(path = %path_str, "Download queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        info!(path = %path_str, "Download job queued");
        Ok(CloudTransferDownloadTicket { receiver: Some(rx) })
    }

    pub fn submit_download_bytes(&self, path: ObjectPath) -> Result<CloudTransferBytesTicket> {
        self.submit_download_bytes_condition(path, None)
    }

    pub fn submit_download_bytes_condition(
        &self,
        path: ObjectPath,
        condition: Option<CloudObjectCondition>,
    ) -> Result<CloudTransferBytesTicket> {
        Ok(CloudTransferBytesTicket {
            ticket: self.submit_download_condition(path, condition)?,
        })
    }

    /// Queue a bounded object range through the same concurrency, retry, and
    /// buffer-budget machinery used by complete-object downloads.
    fn submit_range(
        &self,
        path: ObjectPath,
        range: std::ops::Range<u64>,
        condition: Option<CloudObjectCondition>,
    ) -> Result<CloudTransferRangeTicket> {
        if range.start > range.end {
            return Err(anyhow!("invalid object byte range"));
        }
        let path_str = path.to_string();
        let (tx, rx) = oneshot::channel();
        let job = RangeJob {
            path,
            range,
            condition,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if self
            .inner
            .sender
            .blocking_send(TransferJob::Range(job))
            .is_err()
        {
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            warn!(path = %path_str, "Range queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        Ok(CloudTransferRangeTicket { receiver: Some(rx) })
    }

    pub fn get_range_blocking(
        &self,
        path: ObjectPath,
        range: std::ops::Range<u64>,
    ) -> Result<PooledBytes> {
        self.get_range_blocking_condition(path, range, None)
    }

    pub fn get_range_blocking_condition(
        &self,
        path: ObjectPath,
        range: std::ops::Range<u64>,
        condition: Option<CloudObjectCondition>,
    ) -> Result<PooledBytes> {
        self.submit_range(path, range, condition)?.wait()
    }

    /// Queue an object metadata request so synchronous readers reuse the
    /// transfer runtime and its retry policy.
    fn submit_head(&self, path: ObjectPath) -> Result<CloudTransferHeadTicket> {
        let path_str = path.to_string();
        let (tx, rx) = oneshot::channel();
        let job = HeadJob {
            path,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if self
            .inner
            .sender
            .blocking_send(TransferJob::Head(job))
            .is_err()
        {
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            warn!(path = %path_str, "Head queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        Ok(CloudTransferHeadTicket { receiver: Some(rx) })
    }

    pub fn head_blocking(&self, path: ObjectPath) -> Result<object_store::ObjectMeta> {
        self.submit_head(path)?.wait()
    }

    /// Delete an object through the transfer runtime and retry policy. A
    /// missing object is already in the requested state and succeeds.
    fn submit_delete(&self, path: ObjectPath) -> Result<CloudTransferTicket> {
        let path_str = path.to_string();
        let (tx, rx) = oneshot::channel();
        let job = DeleteJob {
            path,
            attempt: 0,
            response: tx,
        };
        self.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
        if self
            .inner
            .sender
            .blocking_send(TransferJob::Delete(job))
            .is_err()
        {
            self.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
            warn!(path = %path_str, "Delete queue rejected: transfer stopped");
            return Err(anyhow!("cloud transfer stopped"));
        }
        Ok(CloudTransferTicket {
            result: None,
            receiver: Some(rx),
        })
    }

    pub fn delete_blocking(&self, path: ObjectPath) -> Result<()> {
        self.submit_delete(path)?.wait()
    }

    pub fn stats_handle(&self) -> CloudTransferStatsHandle {
        CloudTransferStatsHandle {
            inner: Arc::clone(&self.inner.stats),
        }
    }

    #[instrument(level = "debug", skip(self), fields(prefix = %prefix))]
    pub async fn list_prefixes(&self, prefix: &ObjectPath) -> Result<Vec<ObjectPath>> {
        debug!(prefix = %prefix, "Listing prefixes");
        let result = self.inner.store.list_with_delimiter(Some(prefix)).await?;
        info!(prefix = %prefix, count = result.common_prefixes.len(), "Listed prefixes");
        Ok(result.common_prefixes)
    }

    #[instrument(level = "debug", skip(self), fields(prefix = %prefix))]
    pub async fn list_objects(&self, prefix: &ObjectPath) -> Result<Vec<object_store::ObjectMeta>> {
        debug!(prefix = %prefix, "Listing objects");
        let stream = self.inner.store.list(Some(prefix));
        let objects: Vec<object_store::ObjectMeta> = stream.try_collect().await?;
        info!(prefix = %prefix, count = objects.len(), "Listed objects");
        Ok(objects)
    }

    #[instrument(level = "debug", skip(self), fields(path = %path))]
    pub async fn head(&self, path: &ObjectPath) -> Result<object_store::ObjectMeta> {
        debug!(path = %path, "Reading object metadata");
        let meta = self.inner.store.head(path).await?;
        info!(path = %path, size = meta.size, "Read object metadata");
        Ok(meta)
    }

    #[instrument(level = "debug", skip(self), fields(path = %path))]
    pub async fn get_bytes(&self, path: &ObjectPath) -> Result<Bytes> {
        debug!(path = %path, "Downloading object bytes");
        let result = self.inner.store.get(path).await?;
        let bytes = result.bytes().await?;
        info!(path = %path, bytes = bytes.len(), "Downloaded object bytes");
        Ok(bytes)
    }

    #[instrument(level = "debug", skip(self), fields(path = %path))]
    pub async fn get_string(&self, path: &ObjectPath) -> Result<String> {
        let bytes = self.get_bytes(path).await?;
        let text = String::from_utf8(bytes.to_vec())?;
        Ok(text)
    }
}

enum TransferJob {
    Upload(UploadJob),
    UploadFromPath(UploadFromPathJob),
    Download(DownloadJob),
    Range(RangeJob),
    Head(HeadJob),
    Delete(DeleteJob),
}

impl TransferJob {
    fn into_upload(self) -> UploadJob {
        match self {
            TransferJob::Upload(job) => job,
            TransferJob::UploadFromPath(_)
            | TransferJob::Download(_)
            | TransferJob::Range(_)
            | TransferJob::Head(_)
            | TransferJob::Delete(_) => {
                panic!("unexpected non-in-memory-upload job")
            }
        }
    }
}

struct UploadJob {
    path: ObjectPath,
    buffer: Vec<u8>,
    attempt: usize,
    response: oneshot::Sender<Result<()>>,
}

/// Variant of [`UploadJob`] that streams bytes from a local file
/// instead of holding them in memory. Each part is read on demand from
/// `src_path`; retries re-read.
struct UploadFromPathJob {
    path: ObjectPath,
    src_path: PathBuf,
    attempt: usize,
    response: oneshot::Sender<Result<()>>,
}

struct DownloadJob {
    path: ObjectPath,
    condition: Option<CloudObjectCondition>,
    attempt: usize,
    response: oneshot::Sender<Result<PooledBuffer>>,
}

struct RangeJob {
    path: ObjectPath,
    range: std::ops::Range<u64>,
    condition: Option<CloudObjectCondition>,
    attempt: usize,
    response: oneshot::Sender<Result<PooledBuffer>>,
}

struct HeadJob {
    path: ObjectPath,
    attempt: usize,
    response: oneshot::Sender<Result<object_store::ObjectMeta>>,
}

struct DeleteJob {
    path: ObjectPath,
    attempt: usize,
    response: oneshot::Sender<Result<()>>,
}

enum TransferResult {
    Upload {
        job: UploadJob,
        result: Result<(), anyhow::Error>,
    },
    UploadFromPath {
        job: UploadFromPathJob,
        result: Result<(), anyhow::Error>,
    },
    Download {
        job: DownloadJob,
        result: Result<PooledBuffer, anyhow::Error>,
    },
    Range {
        job: RangeJob,
        result: Result<PooledBuffer, anyhow::Error>,
    },
    Head {
        job: HeadJob,
        result: Result<object_store::ObjectMeta, anyhow::Error>,
    },
    Delete {
        job: DeleteJob,
        result: Result<(), anyhow::Error>,
    },
}

struct SharedSlice {
    data: Arc<Vec<u8>>,
    start: usize,
    end: usize,
}

impl AsRef<[u8]> for SharedSlice {
    fn as_ref(&self) -> &[u8] {
        &self.data[self.start..self.end]
    }
}

async fn upload_loop(
    store: Arc<dyn ObjectStore>,
    mut receiver: mpsc::Receiver<TransferJob>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    config: CloudTransferConfig,
) -> Result<()> {
    let part_sema = Arc::new(Semaphore::new(config.max_part_requests));
    let mut inflight: JoinSet<TransferResult> = JoinSet::new();
    let mut retry_queue: VecDeque<TransferJob> = VecDeque::new();
    let mut receiver_closed = false;

    loop {
        while inflight.len() < config.max_files_in_flight {
            if let Some(job) = retry_queue.pop_front() {
                stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
                inflight.spawn(handle_transfer_job(
                    Arc::clone(&store),
                    Arc::clone(&part_sema),
                    pool.clone(),
                    Arc::clone(&stats),
                    config.part_size,
                    config.download_part_size,
                    job,
                ));
                stats.inflight_files.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            match receiver.try_recv() {
                Ok(job) => {
                    stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
                    inflight.spawn(handle_transfer_job(
                        Arc::clone(&store),
                        Arc::clone(&part_sema),
                        pool.clone(),
                        Arc::clone(&stats),
                        config.part_size,
                        config.download_part_size,
                        job,
                    ));
                    stats.inflight_files.fetch_add(1, Ordering::Relaxed);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    receiver_closed = true;
                    break;
                }
            }
        }

        if receiver_closed && inflight.is_empty() && retry_queue.is_empty() {
            break;
        }

        tokio::select! {
            Some(job) = receiver.recv(), if inflight.len() < config.max_files_in_flight => {
                stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
                inflight.spawn(handle_transfer_job(
                    Arc::clone(&store),
                    Arc::clone(&part_sema),
                    pool.clone(),
                    Arc::clone(&stats),
                    config.part_size,
                    config.download_part_size,
                    job,
                ));
                stats.inflight_files.fetch_add(1, Ordering::Relaxed);
            }
            Some(result) = inflight.join_next() => {
                stats.inflight_files.fetch_sub(1, Ordering::Relaxed);
                let result = result.unwrap_or_else(|err| {
                    TransferResult::Upload {
                        job: UploadJob {
                            path: ObjectPath::from(""),
                            buffer: Vec::new(),
                            attempt: 0,
                            response: oneshot::channel().0,
                        },
                        result: Err(anyhow!("transfer task failed: {err}")),
                    }
                });
                match result {
                    TransferResult::Upload { mut job, result } => match result {
                        Ok(()) => {
                            info!(path = %job.path, attempt = job.attempt, "Upload completed");
                            pool.release(job.buffer);
                            let _ = job.response.send(Ok(()));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, attempt = job.attempt, error = %err, "Upload failed after retries");
                                pool.release(job.buffer);
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, attempt = job.attempt, error = %err, "Upload failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Upload(job));
                            }
                        }
                    },
                    TransferResult::UploadFromPath { mut job, result } => match result {
                        Ok(()) => {
                            info!(path = %job.path, attempt = job.attempt, src = %job.src_path.display(), "Upload (from path) completed");
                            let _ = job.response.send(Ok(()));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, attempt = job.attempt, src = %job.src_path.display(), error = %err, "Upload (from path) failed after retries");
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, attempt = job.attempt, src = %job.src_path.display(), error = %err, "Upload (from path) failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::UploadFromPath(job));
                            }
                        }
                    },
                    TransferResult::Download { mut job, result } => match result {
                        Ok(buffer) => {
                            info!(path = %job.path, attempt = job.attempt, bytes = buffer.as_slice().len(), "Download completed");
                            let _ = job.response.send(Ok(buffer));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, attempt = job.attempt, error = %err, "Download failed after retries");
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, attempt = job.attempt, error = %err, "Download failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Download(job));
                            }
                        }
                    },
                    TransferResult::Range { mut job, result } => match result {
                        Ok(buffer) => {
                            info!(path = %job.path, start = job.range.start, end = job.range.end, attempt = job.attempt, "Range download completed");
                            let _ = job.response.send(Ok(buffer));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, start = job.range.start, end = job.range.end, attempt = job.attempt, error = %err, "Range download failed after retries");
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, start = job.range.start, end = job.range.end, attempt = job.attempt, error = %err, "Range download failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Range(job));
                            }
                        }
                    },
                    TransferResult::Head { mut job, result } => match result {
                        Ok(meta) => {
                            info!(path = %job.path, size = meta.size, attempt = job.attempt, "Head completed");
                            let _ = job.response.send(Ok(meta));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, attempt = job.attempt, error = %err, "Head failed after retries");
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, attempt = job.attempt, error = %err, "Head failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Head(job));
                            }
                        }
                    },
                    TransferResult::Delete { mut job, result } => match result {
                        Ok(()) => {
                            info!(path = %job.path, attempt = job.attempt, "Delete completed");
                            let _ = job.response.send(Ok(()));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                error!(path = %job.path, attempt = job.attempt, error = %err, "Delete failed after retries");
                                let _ = job.response.send(Err(err));
                            } else {
                                warn!(path = %job.path, attempt = job.attempt, error = %err, "Delete failed; re-queueing retry");
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Delete(job));
                            }
                        }
                    },
                }
            }
            else => {
                tokio::task::yield_now().await;
            }
        }
    }
    Ok(())
}

async fn handle_transfer_job(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    download_part_size: usize,
    job: TransferJob,
) -> TransferResult {
    match job {
        TransferJob::Upload(job) => {
            let UploadJob {
                path,
                buffer,
                attempt,
                response,
            } = job;
            let arc = Arc::new(buffer);
            let result = upload_file_once(
                Arc::clone(&store),
                Arc::clone(&part_sema),
                Arc::clone(&stats),
                part_size,
                &path,
                Arc::clone(&arc),
            )
            .await;
            let buffer = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
            TransferResult::Upload {
                job: UploadJob {
                    path,
                    buffer,
                    attempt,
                    response,
                },
                result,
            }
        }
        TransferJob::UploadFromPath(job) => {
            let UploadFromPathJob {
                path,
                src_path,
                attempt,
                response,
            } = job;
            let result = upload_file_once_from_path(
                Arc::clone(&store),
                Arc::clone(&part_sema),
                Arc::clone(&stats),
                part_size,
                &path,
                &src_path,
            )
            .await;
            TransferResult::UploadFromPath {
                job: UploadFromPathJob {
                    path,
                    src_path,
                    attempt,
                    response,
                },
                result,
            }
        }
        TransferJob::Download(job) => {
            let DownloadJob {
                path,
                condition,
                attempt,
                response,
            } = job;
            let result = download_file_once(
                Arc::clone(&store),
                Arc::clone(&part_sema),
                pool,
                Arc::clone(&stats),
                download_part_size,
                &path,
                condition.clone(),
            )
            .await;
            TransferResult::Download {
                job: DownloadJob {
                    path,
                    condition,
                    attempt,
                    response,
                },
                result,
            }
        }
        TransferJob::Range(job) => {
            let RangeJob {
                path,
                range,
                condition,
                attempt,
                response,
            } = job;
            let result = download_range_once(
                Arc::clone(&store),
                Arc::clone(&part_sema),
                pool,
                Arc::clone(&stats),
                &path,
                range.clone(),
                condition.clone(),
            )
            .await;
            TransferResult::Range {
                job: RangeJob {
                    path,
                    range,
                    condition,
                    attempt,
                    response,
                },
                result,
            }
        }
        TransferJob::Head(job) => {
            let HeadJob {
                path,
                attempt,
                response,
            } = job;
            stats.parts_started.fetch_add(1, Ordering::Relaxed);
            stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
            let permit = part_sema.acquire_owned().await;
            let result = match permit {
                Ok(_permit) => store
                    .head(&path)
                    .await
                    .map_err(|err| anyhow!("head failed for {}: {}", path, err)),
                Err(_) => Err(anyhow!("head semaphore closed for {}", path)),
            };
            stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
            TransferResult::Head {
                job: HeadJob {
                    path,
                    attempt,
                    response,
                },
                result,
            }
        }
        TransferJob::Delete(job) => {
            let DeleteJob {
                path,
                attempt,
                response,
            } = job;
            stats.parts_started.fetch_add(1, Ordering::Relaxed);
            stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
            let permit = part_sema.acquire_owned().await;
            let result = match permit {
                Ok(_permit) => match store.delete(&path).await {
                    Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
                    Err(err) => Err(anyhow!("delete failed for {}: {}", path, err)),
                },
                Err(_) => Err(anyhow!("delete semaphore closed for {}", path)),
            };
            stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
            TransferResult::Delete {
                job: DeleteJob {
                    path,
                    attempt,
                    response,
                },
                result,
            }
        }
    }
}

async fn get_exact_range_once(
    store: &Arc<dyn ObjectStore>,
    path: &ObjectPath,
    range: std::ops::Range<u64>,
    condition: Option<&CloudObjectCondition>,
) -> Result<Bytes> {
    let expected = usize::try_from(range.end - range.start)
        .map_err(|_| anyhow!("download range length exceeds addressable space"))?;
    let options = condition
        .map(CloudObjectCondition::get_options)
        .unwrap_or_default()
        .with_range(Some(range));
    let result = store.get_opts(path, options).await?;
    if let Some(condition) = condition {
        condition.validate_meta(&result.meta)?;
    }
    let bytes = result.bytes().await?;
    let actual = bytes.len();
    (actual == expected).then_some(bytes).ok_or_else(|| {
        anyhow!(
            "download range returned {} bytes, expected {}",
            actual,
            expected
        )
    })
}

async fn download_range_once(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    path: &ObjectPath,
    range: std::ops::Range<u64>,
    condition: Option<CloudObjectCondition>,
) -> Result<PooledBuffer> {
    stats.parts_started.fetch_add(1, Ordering::Relaxed);
    stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
    let permit = part_sema
        .acquire_owned()
        .await
        .map_err(|_| anyhow!("range semaphore closed for {}", path));
    let bytes = match permit {
        Ok(_permit) => get_exact_range_once(&store, path, range, condition.as_ref())
            .await
            .map_err(|error| {
                let message = format!("range download failed for {path}: {error}");
                error.context(message)
            }),
        Err(err) => Err(err),
    };
    stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
    let bytes = bytes?;
    let mut buffer = pool.acquire_raw_sized(bytes.len())?;
    buffer.copy_from_slice(&bytes);
    Ok(pool.wrap_accounted(buffer))
}

async fn download_file_once(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    path: &ObjectPath,
    condition: Option<CloudObjectCondition>,
) -> Result<PooledBuffer> {
    let object_size = match condition
        .as_ref()
        .and_then(CloudObjectCondition::pinned_size)
    {
        Some(size) => size,
        None => {
            let meta = store
                .head(path)
                .await
                .map_err(|err| anyhow!("download head failed for {}: {}", path, err))?;
            if let Some(condition) = &condition {
                condition.validate_meta(&meta)?;
            }
            meta.size
        }
    };
    let size = usize::try_from(object_size).map_err(|_| {
        anyhow!(
            "download size {} exceeds addressable space for {}",
            object_size,
            path
        )
    })?;
    let buffer = pool.acquire_raw_sized(size)?;
    if size == 0 {
        return Ok(pool.wrap_accounted(buffer));
    }
    let buffer = Arc::new(tokio::sync::Mutex::new(buffer));
    let mut tasks: JoinSet<Result<()>> = JoinSet::new();
    let mut offset = 0u64;
    let size_u64 = object_size;

    while offset < size_u64 {
        let end = std::cmp::min(offset + part_size as u64, size_u64);
        let path = path.clone();
        let store = Arc::clone(&store);
        let buffer = Arc::clone(&buffer);
        let part_sema = Arc::clone(&part_sema);
        let condition = condition.clone();
        let start_usize = offset as usize;
        stats.parts_started.fetch_add(1, Ordering::Relaxed);
        stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
        tasks.spawn(async move {
            let _permit = part_sema
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("download semaphore closed"))?;
            let bytes =
                get_exact_range_once(&store, &path, offset..end, condition.as_ref()).await?;
            let mut guard = buffer.lock().await;
            let dest = &mut guard[start_usize..start_usize + bytes.len()];
            dest.copy_from_slice(&bytes);
            Ok(())
        });
        offset = end;
    }

    let mut failed = None;
    while let Some(joined) = tasks.join_next().await {
        stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                failed = Some(anyhow!("download failed for {}: {}", path, err));
                stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                tasks.abort_all();
            }
            Err(err) => {
                failed = Some(anyhow!("download task failed for {}: {}", path, err));
                stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                tasks.abort_all();
            }
        }
    }

    let buffer = match Arc::try_unwrap(buffer) {
        Ok(buffer) => buffer.into_inner(),
        Err(buffer) => {
            let mut guard = buffer.lock().await;
            let mut vec = Vec::new();
            std::mem::swap(&mut vec, &mut *guard);
            vec
        }
    };

    if let Some(err) = failed {
        pool.release(buffer);
        return Err(err);
    }

    Ok(pool.wrap_accounted(buffer))
}

async fn upload_file_once(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    path: &ObjectPath,
    buffer: Arc<Vec<u8>>,
) -> Result<()> {
    let mut upload = store
        .put_multipart(path)
        .await
        .map_err(|err| anyhow!("upload failed for {}: {}", path, err))?;
    let len = buffer.len();
    let mut offset = 0usize;
    let mut tasks: JoinSet<object_store::Result<()>> = JoinSet::new();

    while offset < len {
        let end = std::cmp::min(offset + part_size, len);
        let slice = SharedSlice {
            data: Arc::clone(&buffer),
            start: offset,
            end,
        };
        let payload: PutPayload = Bytes::from_owner(slice).into();
        let fut = upload.put_part(payload);
        let part_sema = Arc::clone(&part_sema);
        stats.parts_started.fetch_add(1, Ordering::Relaxed);
        stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
        tasks.spawn(async move {
            let _permit =
                part_sema
                    .acquire_owned()
                    .await
                    .map_err(|_| object_store::Error::Generic {
                        store: "multipart",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::Other,
                            "upload semaphore closed",
                        )),
                    })?;
            fut.await
        });
        offset = end;
    }

    let mut failed = None;
    while let Some(joined) = tasks.join_next().await {
        stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                failed = Some(anyhow!("upload failed for {}: {}", path, err));
                tasks.abort_all();
                break;
            }
            Err(err) => {
                failed = Some(anyhow!("upload task failed for {}: {}", path, err));
                tasks.abort_all();
                break;
            }
        }
    }

    if let Some(err) = failed {
        return Err(err);
    }

    upload
        .complete()
        .await
        .map(|_| ())
        .map_err(|err| anyhow!("upload complete failed for {}: {}", path, err))
}

/// Path-sourced sibling of [`upload_file_once`]. Opens `src_path`,
/// reads each `part_size`-sized slice on demand inside a blocking task
/// (positioned `pread` — concurrent-safe), feeds the slice to
/// `put_part`, drops the buffer when the part task completes. Total
/// in-flight bytes for this file are bounded by
/// `part_size × inflight_parts` regardless of file size.
///
/// Loop shape mirrors [`upload_file_once`]: the per-part `put_part`
/// call returns a `'static` future once invoked (object_store captures
/// the payload + internal multipart state into the returned future),
/// so the main loop can keep using the `&mut MultipartUpload` to
/// initiate subsequent parts while previously-spawned ones upload in
/// parallel. The only difference is where the part bytes come from:
/// here we `pread` from disk inside `spawn_blocking` instead of
/// slicing an in-memory `Arc<Vec<u8>>`.
async fn upload_file_once_from_path(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    path: &ObjectPath,
    src_path: &Path,
) -> Result<()> {
    // `std::fs::File` so we can use the Unix `pread` API
    // (`read_exact_at`). Concurrent positioned reads on the same fd
    // are safe — each `pread` carries its own offset and does not
    // touch a shared cursor.
    let file = std::fs::File::open(src_path)
        .map_err(|err| anyhow!("open {} for upload: {}", src_path.display(), err))?;
    let file_size = file
        .metadata()
        .map_err(|err| anyhow!("stat {} for upload: {}", src_path.display(), err))?
        .len();
    let file = Arc::new(file);

    let mut upload = store
        .put_multipart(path)
        .await
        .map_err(|err| anyhow!("upload (from path) failed for {}: {}", path, err))?;

    let mut offset = 0u64;
    let mut tasks: JoinSet<object_store::Result<()>> = JoinSet::new();

    while offset < file_size {
        let end = std::cmp::min(offset + part_size as u64, file_size);
        let part_len = (end - offset) as usize;
        let part_offset = offset;

        // Read the part bytes BEFORE calling `put_part` — `put_part`
        // requires the payload up front. `spawn_blocking` keeps the
        // pread off the runtime; the just-written file's pages are hot
        // in the page cache so the cost is dominated by memcpy from
        // the cache. Sequential reads in the loop don't cost
        // throughput because the bottleneck is network upload, which
        // happens in parallel via the spawned put_part tasks below.
        let read_file = Arc::clone(&file);
        let buf = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<u8>> {
            let mut buf = vec![0u8; part_len];
            read_exact_at(&read_file, &mut buf, part_offset)?;
            Ok(buf)
        })
        .await
        .map_err(|err| {
            anyhow!(
                "pread join failed for {} @ {}: {}",
                src_path.display(),
                part_offset,
                err
            )
        })?
        .map_err(|err| {
            anyhow!(
                "pread failed for {} @ {}: {}",
                src_path.display(),
                part_offset,
                err
            )
        })?;

        let payload: PutPayload = Bytes::from(buf).into();
        let fut = upload.put_part(payload);
        let part_sema = Arc::clone(&part_sema);
        stats.parts_started.fetch_add(1, Ordering::Relaxed);
        stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
        tasks.spawn(async move {
            let _permit =
                part_sema
                    .acquire_owned()
                    .await
                    .map_err(|_| object_store::Error::Generic {
                        store: "multipart",
                        source: Box::new(io::Error::new(
                            io::ErrorKind::Other,
                            "upload semaphore closed",
                        )),
                    })?;
            fut.await
        });
        offset = end;
    }

    let mut failed = None;
    while let Some(joined) = tasks.join_next().await {
        stats.inflight_parts.fetch_sub(1, Ordering::Relaxed);
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                failed = Some(anyhow!("upload (from path) failed for {}: {}", path, err));
                stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                tasks.abort_all();
                break;
            }
            Err(err) => {
                failed = Some(anyhow!(
                    "upload (from path) task failed for {}: {}",
                    path,
                    err
                ));
                stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                tasks.abort_all();
                break;
            }
        }
    }

    if let Some(err) = failed {
        return Err(err);
    }

    upload
        .complete()
        .await
        .map(|_| ())
        .map_err(|err| anyhow!("upload (from path) complete failed for {}: {}", path, err))
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, bytes: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(bytes, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        match file.seek_read(bytes, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                offset += read as u64;
                bytes = &mut bytes[read..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn default_cloud_max_bytes() -> u64 {
    let total = diskgraph_limits::total_memory_bytes().unwrap_or(8 * 1024 * 1024 * 1024);
    total / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Once;

    #[test]
    fn positioned_read_requires_full_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("positioned.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut bytes = [0; 4];
        read_exact_at(&file, &mut bytes, 3).unwrap();
        assert_eq!(&bytes, b"3456");
        assert_eq!(
            read_exact_at(&file, &mut bytes, 8).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        read_exact_at(&file, &mut [], 100).unwrap();
        drop(file);
    }

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter =
                std::env::var("RUST_LOG").unwrap_or_else(|_| "diskgraph_fs=debug".to_string());
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
                .with_test_writer()
                .try_init();
        });
    }

    fn test_config(part_size: usize, max_file_retries: usize) -> CloudTransferConfig {
        CloudTransferConfig {
            part_size,
            download_part_size: part_size,
            max_queue: 4,
            max_files_in_flight: 2,
            max_part_requests: 2,
            max_file_retries,
            buffer_count: 2,
            max_bytes: 1024 * 1024,
        }
    }

    #[test]
    fn manifest_size_is_trusted_only_with_immutable_identity() {
        let mut condition = CloudObjectCondition {
            size: 123,
            e_tag: None,
            version: None,
        };
        assert_eq!(condition.pinned_size(), None);
        condition.e_tag = Some("etag".into());
        assert_eq!(condition.pinned_size(), Some(123));
        condition.e_tag = None;
        condition.version = Some("version".into());
        assert_eq!(condition.pinned_size(), Some(123));
    }

    #[test]
    fn pooled_file_buffer_budget_handles_growth_and_fragmentation() -> Result<()> {
        let budget = Arc::new(ByteBudget::new(150));
        let pool = BufferPool::new(1, Arc::clone(&budget));
        let first = pool.acquire_raw_sized(100)?;
        pool.release(first);
        let second = pool.acquire_raw_sized(101)?;
        assert_eq!(second.len(), 101);
        assert!(budget.used_bytes() <= 150);
        pool.release(second);

        let budget = Arc::new(ByteBudget::new(100));
        let pool = BufferPool::new(3, Arc::clone(&budget));
        let large = pool.acquire_raw_sized(60)?;
        let medium = pool.acquire_raw_sized(30)?;
        let small = pool.acquire_raw_sized(10)?;
        pool.release(large);
        pool.release(medium);
        pool.release(small);
        let consolidated = pool.acquire_raw_sized(90)?;
        assert_eq!(consolidated.len(), 90);
        assert!(budget.used_bytes() <= budget.max_bytes());
        pool.release(consolidated);
        Ok(())
    }

    #[test]
    fn cloud_transfer_writes_object() -> Result<()> {
        init_tracing();
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let transfer = CloudTransfer::new(store.clone(), test_config(5, 1));
        let mut buffer = transfer.acquire_buffer();
        buffer.write_all(b"hello world")?;
        let path = ObjectPath::from("test/part.parquet");
        transfer.submit(path.clone(), buffer.into_inner())?.wait()?;
        let rt = Runtime::new()?;
        let data = rt
            .block_on(async { store.get(&path).await?.bytes().await })
            .map_err(|e| anyhow!(e.to_string()))?;
        assert_eq!(data.as_ref(), b"hello world");
        Ok(())
    }

    #[test]
    fn cloud_transfer_reads_object() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let transfer = CloudTransfer::new(store.clone(), test_config(5, 1));
        let path = ObjectPath::from("test/read.parquet");
        let rt = Runtime::new()?;
        rt.block_on(async {
            store
                .put(&path, Bytes::from_static(b"read-back").into())
                .await
        })?;
        let buffer = transfer.submit_download(path)?.wait()?;
        assert_eq!(buffer.as_slice(), b"read-back");
        Ok(())
    }

    #[test]
    fn manifest_condition_rejects_mutated_object() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let transfer = CloudTransfer::new(store.clone(), test_config(3, 0));
        let path = ObjectPath::from("test/condition.parquet");
        let rt = Runtime::new()?;
        rt.block_on(async {
            store
                .put(&path, Bytes::from_static(b"original").into())
                .await
        })?;
        let original = rt.block_on(store.head(&path))?;
        let condition = CloudObjectCondition {
            size: original.size,
            e_tag: original.e_tag.clone(),
            version: original.version.clone(),
        };

        let wrong_size = CloudObjectCondition {
            size: original.size - 1,
            ..condition.clone()
        };
        assert!(transfer
            .submit_download_bytes_condition(path.clone(), Some(wrong_size.clone()))?
            .wait()
            .is_err());
        let err = transfer
            .get_range_blocking_condition(path.clone(), 1..5, Some(wrong_size))
            .unwrap_err();
        assert!(err.to_string().contains("changed size"), "{err:#}");

        let complete = transfer
            .submit_download_bytes_condition(path.clone(), Some(condition.clone()))?
            .wait()?;
        assert_eq!(complete.bytes().as_ref(), b"original");
        drop(complete);
        let range =
            transfer.get_range_blocking_condition(path.clone(), 1..5, Some(condition.clone()))?;
        assert_eq!(range.bytes().as_ref(), b"rigi");
        drop(range);

        rt.block_on(async {
            store
                .put(&path, Bytes::from_static(b"changed!").into())
                .await
        })?;
        assert!(transfer
            .submit_download_bytes_condition(path.clone(), Some(condition.clone()))?
            .wait()
            .is_err());
        assert!(transfer
            .get_range_blocking_condition(path.clone(), 0..3, Some(condition))
            .is_err());
        Ok(())
    }

    #[test]
    fn cloud_transfer_delete_is_idempotent() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let transfer = CloudTransfer::new(store.clone(), CloudTransferConfig::from_env());
        let path = ObjectPath::from("snapshot/_SUCCESS");
        let rt = Runtime::new()?;
        rt.block_on(async {
            store
                .put(&path, Bytes::from_static(b"published").into())
                .await
        })?;
        transfer.delete_blocking(path.clone())?;
        transfer.delete_blocking(path.clone())?;
        let error = rt.block_on(store.head(&path)).unwrap_err();
        assert!(matches!(error, object_store::Error::NotFound { .. }));
        Ok(())
    }

    #[test]
    fn cloud_transfer_lists_prefixes() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let transfer = CloudTransfer::new(store.clone(), test_config(5, 1));
        let rt = Runtime::new()?;
        rt.block_on(async {
            store
                .put(
                    &ObjectPath::from("root/a/file.txt"),
                    Bytes::from_static(b"one").into(),
                )
                .await?;
            store
                .put(
                    &ObjectPath::from("root/b/file.txt"),
                    Bytes::from_static(b"two").into(),
                )
                .await?;
            store
                .put(
                    &ObjectPath::from("root/c/sub/file.txt"),
                    Bytes::from_static(b"three").into(),
                )
                .await?;
            let prefixes = transfer.list_prefixes(&ObjectPath::from("root/")).await?;
            let mut prefix_strings: Vec<String> =
                prefixes.into_iter().map(|p| p.to_string()).collect();
            prefix_strings.sort();
            assert_eq!(
                prefix_strings,
                vec![
                    "root/a".to_string(),
                    "root/b".to_string(),
                    "root/c".to_string()
                ]
            );
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(())
    }
}
