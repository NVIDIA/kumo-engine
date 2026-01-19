use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, PutPayload};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio::task::JoinSet;

#[derive(Clone, Debug)]
pub struct CloudTransferConfig {
    pub part_size: usize,
    pub max_queue: usize,
    pub max_files_in_flight: usize,
    pub max_part_requests: usize,
    pub max_file_retries: usize,
    pub buffer_count: usize,
}

impl CloudTransferConfig {
    pub fn from_env() -> Self {
        let cpu_default = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        let max_queue = env_usize("DISKGRAPH_CLOUD_QUEUE_MAX", cpu_default.max(1) / 8 + 1);
        let max_files_in_flight =
            env_usize("DISKGRAPH_CLOUD_FILES_IN_FLIGHT", cpu_default.max(1) / 4 + 1);
        let max_part_requests = env_usize("DISKGRAPH_CLOUD_MAX_REQUESTS", cpu_default.max(1));
        let part_size = env_usize("DISKGRAPH_CLOUD_PART_SIZE_MB", 64) * 1024 * 1024;
        let buffer_count = env_usize(
            "DISKGRAPH_CLOUD_FILE_BUFFER_COUNT",
            cpu_default.saturating_mul(2).max(1),
        );
        let max_file_retries = env_usize("DISKGRAPH_CLOUD_FILE_RETRY_MAX", 2);
        Self {
            part_size,
            max_queue: max_queue.max(1),
            max_files_in_flight: max_files_in_flight.max(1),
            max_part_requests: max_part_requests.max(1),
            max_file_retries,
            buffer_count: buffer_count.max(1),
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
    pub last_error: Option<String>,
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
            last_error,
        }
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
}

impl BufferPool {
    pub fn new(buffer_count: usize) -> Self {
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
            last_error: Mutex::new(None),
        });
        Self {
            inner: Arc::new(BufferPoolInner {
                buffers: Mutex::new(buffers),
                available: Condvar::new(),
                stats,
            }),
        }
    }

    pub fn acquire(&self) -> PooledBuffer {
        let mut guard = self.inner.buffers.lock().expect("cloud buffer pool lock");
        loop {
            if let Some(buffer) = guard.pop() {
                self.inner
                    .stats
                    .available_buffers
                    .fetch_sub(1, Ordering::Relaxed);
                return PooledBuffer {
                    cursor: Cursor::new(buffer),
                };
            }
            guard = self.inner.available.wait(guard).expect("cloud buffer pool wait");
        }
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
}

#[derive(Debug)]
pub struct PooledBuffer {
    cursor: Cursor<Vec<u8>>,
}

impl PooledBuffer {
    pub fn into_inner(self) -> Vec<u8> {
        self.cursor.into_inner()
    }
}

impl Write for PooledBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.cursor.write(buf)
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

#[derive(Clone)]
pub struct CloudTransfer {
    inner: Arc<CloudTransferInner>,
}

struct CloudTransferInner {
    sender: mpsc::Sender<UploadJob>,
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
        let pool = BufferPool::new(config.buffer_count);
        let stats = Arc::clone(&pool.inner.stats);
        let (sender, receiver) = mpsc::channel(config.max_queue);
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
            inner: Arc::new(CloudTransferInner { sender, pool, stats }),
        }
    }

    pub fn acquire_buffer(&self) -> PooledBuffer {
        self.inner.pool.acquire()
    }

    pub fn submit(&self, path: ObjectPath, buffer: Vec<u8>) -> Result<()> {
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
        if let Err(err) = self.inner.sender.blocking_send(job) {
            let job = err.0;
            self.inner
                .stats
                .queued_jobs
                .fetch_sub(1, Ordering::Relaxed);
            self.inner.pool.release(job.buffer);
            return Err(anyhow!("cloud transfer stopped"));
        }
        match rx.blocking_recv() {
            Ok(result) => result,
            Err(_) => Err(anyhow!("cloud transfer stopped")),
        }
    }

    pub fn stats_handle(&self) -> CloudTransferStatsHandle {
        CloudTransferStatsHandle {
            inner: Arc::clone(&self.inner.stats),
        }
    }
}

struct UploadJob {
    path: ObjectPath,
    buffer: Vec<u8>,
    attempt: usize,
    response: oneshot::Sender<Result<()>>,
}

struct UploadResult {
    job: UploadJob,
    result: Result<(), anyhow::Error>,
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
    mut receiver: mpsc::Receiver<UploadJob>,
    pool: BufferPool,
    stats: Arc<CloudTransferStatsInner>,
    config: CloudTransferConfig,
) -> Result<()> {
    let part_sema = Arc::new(Semaphore::new(config.max_part_requests));
    let mut inflight: JoinSet<UploadResult> = JoinSet::new();
    let mut retry_queue: VecDeque<UploadJob> = VecDeque::new();
    let mut receiver_closed = false;

    loop {
        while inflight.len() < config.max_files_in_flight {
            if let Some(job) = retry_queue.pop_front() {
                inflight.spawn(handle_job(
                    Arc::clone(&store),
                    Arc::clone(&part_sema),
                    Arc::clone(&stats),
                    config.part_size,
                    job,
                ));
                stats.inflight_files.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            match receiver.try_recv() {
                Ok(job) => {
                    stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
                    inflight.spawn(handle_job(
                        Arc::clone(&store),
                        Arc::clone(&part_sema),
                        Arc::clone(&stats),
                        config.part_size,
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
                inflight.spawn(handle_job(
                    Arc::clone(&store),
                    Arc::clone(&part_sema),
                    Arc::clone(&stats),
                    config.part_size,
                    job,
                ));
                stats.inflight_files.fetch_add(1, Ordering::Relaxed);
            }
            Some(result) = inflight.join_next() => {
                let UploadResult { mut job, result } = result.unwrap_or_else(|err| {
                    UploadResult {
                        job: UploadJob {
                            path: ObjectPath::from(""),
                            buffer: Vec::new(),
                            attempt: 0,
                            response: oneshot::channel().0,
                        },
                        result: Err(anyhow!("upload task failed: {err}")),
                    }
                });
                stats.inflight_files.fetch_sub(1, Ordering::Relaxed);
                match result {
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
                            retry_queue.push_back(job);
                        }
                    }
                }
            }
            else => {
                tokio::task::yield_now().await;
            }
        }
    }
    Ok(())
}

async fn handle_job(
    store: Arc<dyn ObjectStore>,
    part_sema: Arc<Semaphore>,
    stats: Arc<CloudTransferStatsInner>,
    part_size: usize,
    job: UploadJob,
) -> UploadResult {
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
    UploadResult {
        job: UploadJob {
            path,
            buffer,
            attempt,
            response,
        },
        result,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn cloud_transfer_writes_object() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let config = CloudTransferConfig {
            part_size: 5,
            max_queue: 4,
            max_files_in_flight: 2,
            max_part_requests: 2,
            max_file_retries: 1,
            buffer_count: 2,
        };
        let transfer = CloudTransfer::new(store.clone(), config);
        let mut buffer = transfer.acquire_buffer();
        buffer.write_all(b"hello world")?;
        let path = ObjectPath::from("test/part.parquet");
        transfer.submit(path.clone(), buffer.into_inner())?;
        let rt = Runtime::new()?;
        let data = rt
            .block_on(async { store.get(&path).await?.bytes().await })
            .map_err(|e| anyhow!(e.to_string()))?;
        assert_eq!(data.as_ref(), b"hello world");
        Ok(())
    }
}
