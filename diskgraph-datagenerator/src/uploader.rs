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
pub struct UploadStatsHandle {
    inner: Arc<UploadStats>,
}

#[derive(Debug)]
struct UploadStats {
    total_buffers: usize,
    available_buffers: AtomicUsize,
    queued_jobs: AtomicUsize,
    inflight_parts: AtomicUsize,
    parts_started: AtomicU64,
    parts_errors: AtomicU64,
    last_error: Mutex<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct UploadStatsSnapshot {
    pub total_buffers: usize,
    pub available_buffers: usize,
    pub queued_jobs: usize,
    pub inflight_parts: usize,
    pub parts_started: u64,
    pub parts_errors: u64,
    pub last_error: Option<String>,
}

impl UploadStatsHandle {
    pub fn snapshot(&self) -> UploadStatsSnapshot {
        let last_error = self
            .inner
            .last_error
            .lock()
            .ok()
            .and_then(|value| value.clone());
        UploadStatsSnapshot {
            total_buffers: self.inner.total_buffers,
            available_buffers: self
                .inner
                .available_buffers
                .load(Ordering::Relaxed),
            queued_jobs: self.inner.queued_jobs.load(Ordering::Relaxed),
            inflight_parts: self.inner.inflight_parts.load(Ordering::Relaxed),
            parts_started: self.inner.parts_started.load(Ordering::Relaxed),
            parts_errors: self.inner.parts_errors.load(Ordering::Relaxed),
            last_error,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileBufferPool {
    inner: Arc<FileBufferPoolInner>,
}

#[derive(Debug)]
struct FileBufferPoolInner {
    buffers: Mutex<Vec<Vec<u8>>>,
    available: Condvar,
    stats: Arc<UploadStats>,
}

impl FileBufferPool {
    pub fn new(buffer_count: usize) -> Self {
        let mut buffers = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            buffers.push(Vec::new());
        }
        let stats = Arc::new(UploadStats {
            total_buffers: buffer_count,
            available_buffers: AtomicUsize::new(buffer_count),
            queued_jobs: AtomicUsize::new(0),
            inflight_parts: AtomicUsize::new(0),
            parts_started: AtomicU64::new(0),
            parts_errors: AtomicU64::new(0),
            last_error: Mutex::new(None),
        });
        Self {
            inner: Arc::new(FileBufferPoolInner {
                buffers: Mutex::new(buffers),
                available: Condvar::new(),
                stats,
            }),
        }
    }

    pub fn acquire(&self) -> FileBuffer {
        let mut guard = self.inner.buffers.lock().expect("file buffer pool lock");
        loop {
            if let Some(buffer) = guard.pop() {
                self.inner
                    .stats
                    .available_buffers
                    .fetch_sub(1, Ordering::Relaxed);
                return FileBuffer {
                    cursor: Cursor::new(buffer),
                };
            }
            guard = self.inner.available.wait(guard).expect("file buffer pool wait");
        }
    }

    fn release(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        let mut guard = self.inner.buffers.lock().expect("file buffer pool lock");
        guard.push(buffer);
        self.inner
            .stats
            .available_buffers
            .fetch_add(1, Ordering::Relaxed);
        self.inner.available.notify_one();
    }

    pub fn release_buffer(&self, buffer: Vec<u8>) {
        self.release(buffer);
    }
}

#[derive(Debug)]
pub struct FileBuffer {
    cursor: Cursor<Vec<u8>>,
}

impl FileBuffer {
    pub fn into_inner(self) -> Vec<u8> {
        self.cursor.into_inner()
    }
}

impl Write for FileBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.cursor.flush()
    }
}

impl Seek for FileBuffer {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

#[derive(Debug)]
pub struct CloudSink {
    uploader: UploadExecutor,
    path: ObjectPath,
    buffer: FileBuffer,
}

impl CloudSink {
    pub fn new(uploader: UploadExecutor, path: ObjectPath, buffer: FileBuffer) -> Self {
        Self {
            uploader,
            path,
            buffer,
        }
    }

    pub fn finish(self) -> Result<()> {
        let buffer = self.buffer.into_inner();
        self.uploader.submit(self.path, buffer)
    }
}

impl Write for CloudSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buffer.flush()
    }
}

impl Seek for CloudSink {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.buffer.seek(pos)
    }
}

#[derive(Clone)]
pub struct UploadExecutor {
    inner: Arc<UploadExecutorInner>,
}

impl std::fmt::Debug for UploadExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadExecutor").finish()
    }
}

struct UploadExecutorInner {
    sender: mpsc::UnboundedSender<UploadRequest>,
    pool: FileBufferPool,
    stats: Arc<UploadStats>,
}

struct UploadRequest {
    path: ObjectPath,
    buffer: Vec<u8>,
    response: oneshot::Sender<Result<()>>,
}

#[derive(Clone)]
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

impl UploadExecutor {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        pool: FileBufferPool,
        part_size: usize,
        max_connections: usize,
        max_retries: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let stats = Arc::clone(&pool.inner.stats);
        let pool_clone = pool.clone();
        let _join = thread::spawn(move || {
            let rt = Runtime::new()?;
            rt.block_on(upload_loop(
                store,
                receiver,
                pool_clone,
                part_size,
                max_connections.max(1),
                max_retries,
            ))
        });
        let inner = Arc::new(UploadExecutorInner { sender, pool, stats });
        Self { inner }
    }

    pub fn submit(&self, path: ObjectPath, buffer: Vec<u8>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        let request = UploadRequest {
            path,
            buffer,
            response: tx,
        };
        if let Err(err) = self.inner.sender.send(request) {
            self.inner.pool.release_buffer(err.0.buffer);
            return Err(anyhow!("upload executor stopped"));
        }
        rx.blocking_recv()
            .map_err(|_| anyhow!("upload executor stopped"))?
    }

    pub fn stats_handle(&self) -> UploadStatsHandle {
        UploadStatsHandle {
            inner: Arc::clone(&self.inner.stats),
        }
    }
}

async fn upload_loop(
    store: Arc<dyn ObjectStore>,
    mut receiver: mpsc::UnboundedReceiver<UploadRequest>,
    pool: FileBufferPool,
    part_size: usize,
    max_connections: usize,
    max_retries: usize,
) -> Result<()> {
    let semaphore = Arc::new(Semaphore::new(max_connections));
    let mut sessions = JoinSet::new();
    loop {
        tokio::select! {
            Some(request) = receiver.recv() => {
                pool.inner.stats.queued_jobs.fetch_add(1, Ordering::Relaxed);
                let store = Arc::clone(&store);
                let semaphore = Arc::clone(&semaphore);
                let pool = pool.clone();
                sessions.spawn(async move {
                    handle_request(
                        store,
                        semaphore,
                        pool,
                        part_size,
                        max_retries,
                        request,
                    )
                    .await;
                });
            }
            Some(_) = sessions.join_next() => {}
            else => break,
        }
    }
    Ok(())
}

async fn handle_request(
    store: Arc<dyn ObjectStore>,
    semaphore: Arc<Semaphore>,
    pool: FileBufferPool,
    part_size: usize,
    max_retries: usize,
    mut request: UploadRequest,
) {
    pool.inner.stats.queued_jobs.fetch_sub(1, Ordering::Relaxed);
    let mut attempt = 0usize;
    loop {
        let outcome = upload_file_once(
            Arc::clone(&store),
            Arc::clone(&semaphore),
            part_size,
            &request.path,
            request.buffer,
            &pool.inner.stats,
        )
        .await;
        match outcome {
            UploadOutcome::Ok(buffer) => {
                pool.release(buffer);
                let _ = request.response.send(Ok(()));
                return;
            }
            UploadOutcome::Err(buffer, err) => {
                attempt += 1;
                if let Ok(mut guard) = pool.inner.stats.last_error.lock() {
                    *guard = Some(err.to_string());
                }
                pool.inner.stats.parts_errors.fetch_add(1, Ordering::Relaxed);
                if attempt > max_retries {
                    pool.release(buffer);
                    let _ = request.response.send(Err(err));
                    return;
                }
                request.buffer = buffer;
            }
        }
    }
}

enum UploadOutcome {
    Ok(Vec<u8>),
    Err(Vec<u8>, anyhow::Error),
}

async fn upload_file_once(
    store: Arc<dyn ObjectStore>,
    semaphore: Arc<Semaphore>,
    part_size: usize,
    path: &ObjectPath,
    buffer: Vec<u8>,
    stats: &UploadStats,
) -> UploadOutcome {
    let arc = Arc::new(buffer);
    let mut upload = match store.put_multipart(path).await {
        Ok(upload) => upload,
        Err(err) => {
            let buffer = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
            return UploadOutcome::Err(
                buffer,
                anyhow!("upload failed for {}: {}", path, err),
            );
        }
    };
    let len = arc.len();
    let mut offset = 0usize;
    let mut tasks: JoinSet<object_store::Result<()>> = JoinSet::new();
    while offset < len {
        let end = std::cmp::min(offset + part_size, len);
        let slice = SharedSlice {
            data: Arc::clone(&arc),
            start: offset,
            end,
        };
        let payload: PutPayload = Bytes::from_owner(slice).into();
        let fut = upload.put_part(payload);
        let semaphore = Arc::clone(&semaphore);
        stats.parts_started.fetch_add(1, Ordering::Relaxed);
        stats.inflight_parts.fetch_add(1, Ordering::Relaxed);
        tasks.spawn(async move {
            let _permit = semaphore.acquire_owned().await.map_err(|_| {
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
    drop(tasks);
    if let Some(err) = failed {
        let buffer = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
        return UploadOutcome::Err(buffer, err);
    }

    if let Err(err) = upload.complete().await {
        let buffer = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
        return UploadOutcome::Err(
            buffer,
            anyhow!("upload complete failed for {}: {}", path, err),
        );
    }

    let buffer = Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone());
    UploadOutcome::Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn upload_executor_writes_object() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let pool = FileBufferPool::new(2);
        let uploader = UploadExecutor::new(store.clone(), pool.clone(), 5, 2, 1);
        let mut buffer = pool.acquire();
        buffer.write_all(b"hello world")?;
        let path = ObjectPath::from("test/part.parquet");
        uploader.submit(path.clone(), buffer.into_inner())?;
        let rt = Runtime::new()?;
        let data = rt
            .block_on(async { store.get(&path).await?.bytes().await })
            .map_err(|e| anyhow!(e.to_string()))?;
        assert_eq!(data.as_ref(), b"hello world");
        Ok(())
    }
}
