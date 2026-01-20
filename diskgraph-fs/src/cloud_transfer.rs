use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures_util::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::task::JoinSet;

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
        let cpu_default = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        let max_queue = env_usize("DISKGRAPH_CLOUD_QUEUE_MAX", cpu_default.max(1) / 8 + 1);
        let max_files_in_flight = env_usize("DISKGRAPH_CLOUD_FILES_IN_FLIGHT", cpu_default.max(1));
        let max_part_requests = env_usize("DISKGRAPH_CLOUD_MAX_REQUESTS", cpu_default.max(1));
        let part_size = env_usize("DISKGRAPH_CLOUD_PART_SIZE_MB", 64) * 1024 * 1024;
        let download_part_size =
            env_usize("DISKGRAPH_CLOUD_DOWNLOAD_PART_MB", 0).max(part_size);
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
            available_buffers: self
                .inner
                .available_buffers
                .load(Ordering::Relaxed),
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
            guard = self.inner.available.wait(guard).expect("cloud buffer pool wait");
        }
    }

    pub fn acquire_raw_sized(&self, len: usize) -> Result<Vec<u8>> {
        let mut buffer = self.acquire_raw();
        let old_cap = buffer.capacity();
        let mut reserved = 0usize;
        if len > old_cap {
            reserved = len - old_cap;
            if let Err(err) = self.inner.budget.reserve(reserved as u64) {
                self.release(buffer);
                return Err(err);
            }
        }
        buffer.resize(len, 0);
        let new_cap = buffer.capacity();
        if new_cap > old_cap + reserved {
            let extra = new_cap - (old_cap + reserved);
            if let Err(err) = self.inner.budget.reserve(extra as u64) {
                self.release(buffer);
                return Err(err);
            }
        } else if new_cap < old_cap + reserved {
            let give_back = old_cap + reserved - new_cap;
            self.inner.budget.release(give_back as u64);
        }
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
    owner: Arc<Vec<u8>>,
    pool: Option<BufferPool>,
}

impl PooledBytes {
    fn new(buffer: Vec<u8>, pool: Option<BufferPool>) -> Self {
        let owner = Arc::new(buffer);
        let len = owner.len();
        let bytes = Bytes::from_owner(SharedSlice {
            data: Arc::clone(&owner),
            start: 0,
            end: len,
        });
        Self { bytes, owner, pool }
    }

    pub fn bytes(&self) -> Bytes {
        self.bytes.clone()
    }
}

impl Drop for PooledBytes {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let bytes = std::mem::take(&mut self.bytes);
            drop(bytes);
            match Arc::try_unwrap(std::mem::take(&mut self.owner)) {
                Ok(buffer) => pool.release(buffer),
                Err(owner) => pool.release((*owner).clone()),
            }
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
        let (tx, rx) = oneshot::channel();
        let job = UploadJob {
            path,
            buffer,
            attempt: 0,
            response: tx,
        };
        self.inner
            .stats
            .queued_jobs
            .fetch_add(1, Ordering::Relaxed);
        if let Err(err) = self.inner.sender.blocking_send(TransferJob::Upload(job)) {
            let job = err.0.into_upload();
            self.inner
                .stats
                .queued_jobs
                .fetch_sub(1, Ordering::Relaxed);
            self.inner.pool.release(job.buffer);
            return Err(anyhow!("cloud transfer stopped"));
        }
        Ok(CloudTransferTicket {
            result: None,
            receiver: Some(rx),
        })
    }

    pub fn submit_download(&self, path: ObjectPath) -> Result<CloudTransferDownloadTicket> {
        let (tx, rx) = oneshot::channel();
        let job = DownloadJob {
            path,
            attempt: 0,
            response: tx,
        };
        self.inner
            .stats
            .queued_jobs
            .fetch_add(1, Ordering::Relaxed);
        if let Err(_) = self.inner.sender.blocking_send(TransferJob::Download(job)) {
            self.inner
                .stats
                .queued_jobs
                .fetch_sub(1, Ordering::Relaxed);
            return Err(anyhow!("cloud transfer stopped"));
        }
        Ok(CloudTransferDownloadTicket { receiver: Some(rx) })
    }

    pub fn submit_download_bytes(&self, path: ObjectPath) -> Result<CloudTransferBytesTicket> {
        Ok(CloudTransferBytesTicket {
            ticket: self.submit_download(path)?,
        })
    }

    pub fn stats_handle(&self) -> CloudTransferStatsHandle {
        CloudTransferStatsHandle {
            inner: Arc::clone(&self.inner.stats),
        }
    }

    pub async fn list_prefixes(&self, prefix: &ObjectPath) -> Result<Vec<ObjectPath>> {
        let result = self
            .inner
            .store
            .list_with_delimiter(Some(prefix))
            .await?;
        Ok(result.common_prefixes)
    }

    pub async fn list_objects(&self, prefix: &ObjectPath) -> Result<Vec<object_store::ObjectMeta>> {
        let stream = self.inner.store.list(Some(prefix));
        let objects = stream.try_collect().await?;
        Ok(objects)
    }

    pub async fn head(&self, path: &ObjectPath) -> Result<object_store::ObjectMeta> {
        Ok(self.inner.store.head(path).await?)
    }

    pub async fn get_bytes(&self, path: &ObjectPath) -> Result<Bytes> {
        let result = self.inner.store.get(path).await?;
        Ok(result.bytes().await?)
    }

    pub async fn get_string(&self, path: &ObjectPath) -> Result<String> {
        let bytes = self.get_bytes(path).await?;
        let text = String::from_utf8(bytes.to_vec())?;
        Ok(text)
    }
}

enum TransferJob {
    Upload(UploadJob),
    Download(DownloadJob),
}

impl TransferJob {
    fn into_upload(self) -> UploadJob {
        match self {
            TransferJob::Upload(job) => job,
            TransferJob::Download(_) => panic!("unexpected download job"),
        }
    }
}

struct UploadJob {
    path: ObjectPath,
    buffer: Vec<u8>,
    attempt: usize,
    response: oneshot::Sender<Result<()>>,
}

struct DownloadJob {
    path: ObjectPath,
    attempt: usize,
    response: oneshot::Sender<Result<PooledBuffer>>,
}

enum TransferResult {
    Upload { job: UploadJob, result: Result<(), anyhow::Error> },
    Download { job: DownloadJob, result: Result<PooledBuffer, anyhow::Error> },
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
                                pool.release(job.buffer);
                                let _ = job.response.send(Err(err));
                            } else {
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Upload(job));
                            }
                        }
                    },
                    TransferResult::Download { mut job, result } => match result {
                        Ok(buffer) => {
                            let _ = job.response.send(Ok(buffer));
                        }
                        Err(err) => {
                            if let Ok(mut guard) = stats.last_error.lock() {
                                *guard = Some(err.to_string());
                            }
                            stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                            job.attempt += 1;
                            if job.attempt > config.max_file_retries {
                                let _ = job.response.send(Err(err));
                            } else {
                                stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                                retry_queue.push_back(TransferJob::Download(job));
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
        TransferJob::Download(job) => {
            let DownloadJob {
                path,
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
            )
            .await;
            TransferResult::Download {
                job: DownloadJob {
                    path,
                    attempt,
                    response,
                },
                result,
            }
        }
    }
}

async fn download_file_once(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    path: &ObjectPath,
) -> Result<PooledBuffer> {
    let meta = store
        .head(path)
        .await
        .map_err(|err| anyhow!("download head failed for {}: {}", path, err))?;
    let size = usize::try_from(meta.size).map_err(|_| {
        anyhow!(
            "download size {} exceeds addressable space for {}",
            meta.size,
            path
        )
    })?;
    let buffer = pool.acquire_raw_sized(size)?;
    if size == 0 {
        return Ok(pool.wrap_accounted(buffer));
    }
    let buffer = Arc::new(tokio::sync::Mutex::new(buffer));
    let mut tasks: JoinSet<object_store::Result<()>> = JoinSet::new();
    let mut offset = 0u64;
    let size_u64 = size as u64;

    while offset < size_u64 {
        let end = std::cmp::min(offset + part_size as u64, size_u64);
        let path = path.clone();
        let store = Arc::clone(&store);
        let buffer = Arc::clone(&buffer);
        let part_sema = Arc::clone(&part_sema);
        let start_usize = offset as usize;
        stats.parts_started.fetch_add(1, Ordering::Relaxed);
        stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
        tasks.spawn(async move {
            let _permit = part_sema.acquire_owned().await.map_err(|_| {
                object_store::Error::Generic {
                    store: "multipart",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::Other,
                        "download semaphore closed",
                    )),
                }
            })?;
            let bytes = store.get_range(&path, offset..end).await?;
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
            let _permit = part_sema.acquire_owned().await.map_err(|_| {
                object_store::Error::Generic {
                    store: "multipart",
                    source: Box::new(io::Error::new(
                        io::ErrorKind::Other,
                        "upload semaphore closed",
                    )),
                }
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
    let total = host_mem_bytes().unwrap_or(8 * 1024 * 1024 * 1024);
    total.saturating_mul(75) / 100
}

#[cfg(target_os = "linux")]
fn host_mem_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemTotal:") {
            let mut parts = line.split_whitespace();
            let _label = parts.next()?;
            let value = parts.next()?.parse::<u64>().ok()?;
            return Some(value.saturating_mul(1024));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn host_mem_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn cloud_transfer_writes_object() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let config = CloudTransferConfig {
            part_size: 5,
            download_part_size: 5,
            max_queue: 4,
            max_files_in_flight: 2,
            max_part_requests: 2,
            max_file_retries: 1,
            buffer_count: 2,
            max_bytes: 1024 * 1024,
        };
        let transfer = CloudTransfer::new(store.clone(), config);
        let mut buffer = transfer.acquire_buffer();
        buffer.write_all(b"hello world")?;
        let path = ObjectPath::from("test/part.parquet");
        transfer
            .submit(path.clone(), buffer.into_inner())?
            .wait()?;
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
        let config = CloudTransferConfig {
            part_size: 5,
            download_part_size: 5,
            max_queue: 4,
            max_files_in_flight: 2,
            max_part_requests: 2,
            max_file_retries: 1,
            buffer_count: 2,
            max_bytes: 1024 * 1024,
        };
        let transfer = CloudTransfer::new(store.clone(), config);
        let path = ObjectPath::from("test/read.parquet");
        let rt = Runtime::new()?;
        rt.block_on(async { store.put(&path, Bytes::from_static(b"read-back").into()).await })?;
        let buffer = transfer.submit_download(path)?.wait()?;
        assert_eq!(buffer.as_slice(), b"read-back");
        Ok(())
    }

    #[test]
    fn cloud_transfer_lists_prefixes() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let config = CloudTransferConfig {
            part_size: 5,
            download_part_size: 5,
            max_queue: 4,
            max_files_in_flight: 2,
            max_part_requests: 2,
            max_file_retries: 1,
            buffer_count: 2,
            max_bytes: 1024 * 1024,
        };
        let transfer = CloudTransfer::new(store.clone(), config);
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
            let prefixes = transfer
                .list_prefixes(&ObjectPath::from("root/"))
                .await?;
            let mut prefix_strings: Vec<String> =
                prefixes.into_iter().map(|p| p.to_string()).collect();
            prefix_strings.sort();
            assert_eq!(
                prefix_strings,
                vec!["root/a".to_string(), "root/b".to_string(), "root/c".to_string()]
            );
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(())
    }
}
