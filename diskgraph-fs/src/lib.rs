use std::io::{self, SeekFrom, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Result};
use aws_config::{self, BehaviorVersion};
use aws_credential_types::provider::ProvideCredentials;
use aws_types::region::Region;
use bytes::Bytes;
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjectPath;
use object_store::ClientOptions;
use object_store::WriteMultipart;
use object_store::{ObjectStore, ObjectStoreExt};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};

/// Parse a bool environment flag (e.g., "1", "true").
fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[derive(Clone, Debug)]
pub struct S3Config {
    pub bucket: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub allow_http: Option<bool>,
    pub virtual_hosted: Option<bool>,
    pub profile: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedS3 {
    pub bucket: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub allow_http: bool,
    pub virtual_hosted: bool,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct S3Client {
    pub store: Arc<dyn ObjectStore>,
    pub resolved: ResolvedS3,
}

pub mod cloud_transfer;
pub use cloud_transfer::{
    BufferPool, CloudTransfer, CloudTransferBytesTicket, CloudTransferConfig,
    CloudTransferDownloadTicket, CloudTransferStatsHandle, CloudTransferStatsSnapshot,
    CloudTransferTicket, PooledBuffer, PooledBytes,
};

/// Build `ClientOptions` with a connect timeout for all cloud object stores.
///
/// Sets a **connect timeout** (TCP handshake + TLS setup) to detect
/// unreachable endpoints quickly.  The per-request timeout retains its
/// `object_store` default (~30 s) which applies to individual HTTP
/// requests, not to multi-part data transfers.
pub fn cloud_client_options() -> ClientOptions {
    ClientOptions::new().with_connect_timeout(Duration::from_secs(60))
}

pub fn build_s3_client(cfg: S3Config) -> Result<S3Client> {
    let allow_http = cfg
        .allow_http
        .or_else(|| env_flag("S3_ALLOW_HTTP"))
        .or_else(|| env_flag("AWS_ALLOW_HTTP"))
        .unwrap_or(false);
    let endpoint = cfg
        .endpoint
        .clone()
        .or_else(|| std::env::var("S3_ENDPOINT").ok());
    let profile = cfg
        .profile
        .clone()
        .or_else(|| std::env::var("S3_PROFILE").ok())
        .or_else(|| std::env::var("AWS_PROFILE").ok());
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(p) = profile.clone() {
        loader = loader.profile_name(p);
    }
    if let Some(region) = cfg
        .region
        .clone()
        .or_else(|| std::env::var("S3_REGION").ok())
    {
        loader = loader.region(Region::new(region));
    }
    let rt = Runtime::new()?;
    let aws_conf = rt.block_on(loader.load());
    let creds_provider = aws_conf.credentials_provider().ok_or_else(|| {
        anyhow!("No AWS credentials provider found; ensure AWS_PROFILE/SSO is set")
    })?;
    let creds = rt.block_on(async {
        creds_provider
            .provide_credentials()
            .await
            .map_err(|e| anyhow!("Failed to load AWS credentials: {e}"))
    })?;

    let region = cfg
        .region
        .clone()
        .or_else(|| std::env::var("S3_REGION").ok())
        .or_else(|| aws_conf.region().map(|r| r.as_ref().to_string()))
        .unwrap_or_else(|| "us-east-1".to_string());
    let virtual_hosted = cfg
        .virtual_hosted
        .or_else(|| env_flag("S3_VIRTUAL_HOSTED"))
        .unwrap_or_else(|| {
            endpoint
                .as_ref()
                .map_or(true, |ep| ep.contains("amazonaws.com"))
        });
    let access_key_id = cfg
        .access_key_id
        .clone()
        .or_else(|| std::env::var("S3_ACCESS_KEY").ok())
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .unwrap_or_else(|| creds.access_key_id().to_string());
    let secret_access_key = cfg
        .secret_access_key
        .clone()
        .or_else(|| std::env::var("S3_SECRET_KEY").ok())
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .unwrap_or_else(|| creds.secret_access_key().to_string());
    let session_token = cfg
        .session_token
        .clone()
        .or_else(|| std::env::var("S3_SESSION_TOKEN").ok())
        .or_else(|| creds.session_token().map(|t| t.to_string()));

    // with_client_options replaces the entire ClientOptions struct,
    // so it must come before with_allow_http (which writes into it).
    let mut builder = AmazonS3Builder::new()
        .with_client_options(cloud_client_options())
        .with_bucket_name(&cfg.bucket)
        .with_region(&region)
        .with_virtual_hosted_style_request(virtual_hosted)
        .with_allow_http(allow_http)
        .with_access_key_id(access_key_id.clone())
        .with_secret_access_key(secret_access_key.clone());
    if let Some(ep) = &endpoint {
        builder = builder.with_endpoint(ep);
    }
    if let Some(token) = &session_token {
        builder = builder.with_token(token);
    }
    let store = Arc::new(builder.build()?);
    Ok(S3Client {
        store,
        resolved: ResolvedS3 {
            bucket: cfg.bucket,
            region,
            endpoint,
            allow_http,
            virtual_hosted,
            access_key_id,
            secret_access_key,
            session_token,
        },
    })
}

pub async fn build_s3_client_async(cfg: S3Config) -> Result<S3Client> {
    let allow_http = cfg
        .allow_http
        .or_else(|| env_flag("S3_ALLOW_HTTP"))
        .or_else(|| env_flag("AWS_ALLOW_HTTP"))
        .unwrap_or(false);
    let endpoint = cfg
        .endpoint
        .clone()
        .or_else(|| std::env::var("S3_ENDPOINT").ok());
    let profile = cfg
        .profile
        .clone()
        .or_else(|| std::env::var("S3_PROFILE").ok())
        .or_else(|| std::env::var("AWS_PROFILE").ok());
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(p) = profile.clone() {
        loader = loader.profile_name(p);
    }
    if let Some(region) = cfg
        .region
        .clone()
        .or_else(|| std::env::var("S3_REGION").ok())
    {
        loader = loader.region(Region::new(region));
    }
    let aws_conf = loader.load().await;
    let creds_provider = aws_conf.credentials_provider().ok_or_else(|| {
        anyhow!("No AWS credentials provider found; ensure AWS_PROFILE/SSO is set")
    })?;
    let creds = creds_provider
        .provide_credentials()
        .await
        .map_err(|e| anyhow!("Failed to load AWS credentials: {e}"))?;

    let region = cfg
        .region
        .clone()
        .or_else(|| std::env::var("S3_REGION").ok())
        .or_else(|| aws_conf.region().map(|r| r.as_ref().to_string()))
        .unwrap_or_else(|| "us-east-1".to_string());
    let virtual_hosted = cfg
        .virtual_hosted
        .or_else(|| env_flag("S3_VIRTUAL_HOSTED"))
        .unwrap_or_else(|| {
            endpoint
                .as_ref()
                .map_or(true, |ep| ep.contains("amazonaws.com"))
        });
    let access_key_id = cfg
        .access_key_id
        .clone()
        .or_else(|| std::env::var("S3_ACCESS_KEY").ok())
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .unwrap_or_else(|| creds.access_key_id().to_string());
    let secret_access_key = cfg
        .secret_access_key
        .clone()
        .or_else(|| std::env::var("S3_SECRET_KEY").ok())
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .unwrap_or_else(|| creds.secret_access_key().to_string());
    let session_token = cfg
        .session_token
        .clone()
        .or_else(|| std::env::var("S3_SESSION_TOKEN").ok())
        .or_else(|| creds.session_token().map(|t| t.to_string()));

    let mut builder = AmazonS3Builder::new()
        .with_client_options(cloud_client_options())
        .with_bucket_name(&cfg.bucket)
        .with_region(&region)
        .with_virtual_hosted_style_request(virtual_hosted)
        .with_allow_http(allow_http)
        .with_access_key_id(access_key_id.clone())
        .with_secret_access_key(secret_access_key.clone());
    if let Some(ep) = &endpoint {
        builder = builder.with_endpoint(ep);
    }
    if let Some(token) = &session_token {
        builder = builder.with_token(token);
    }
    let store = Arc::new(builder.build()?);
    Ok(S3Client {
        store,
        resolved: ResolvedS3 {
            bucket: cfg.bucket,
            region,
            endpoint,
            allow_http,
            virtual_hosted,
            access_key_id,
            secret_access_key,
            session_token,
        },
    })
}

#[derive(Clone, Debug)]
pub struct GcsConfig {
    pub bucket: String,
}

#[derive(Clone, Debug)]
pub struct GcsClient {
    pub store: Arc<dyn ObjectStore>,
    pub bucket: String,
}

pub fn build_gcs_client(cfg: GcsConfig) -> Result<GcsClient> {
    // with_client_options replaces the entire ClientOptions struct.
    // Calling it after from_env() wipes any env-sourced client
    // settings (e.g. GOOGLE_ALLOW_HTTP), but those are rarely used.
    let builder = GoogleCloudStorageBuilder::from_env()
        .with_client_options(cloud_client_options())
        .with_bucket_name(&cfg.bucket);
    Ok(GcsClient {
        store: Arc::new(builder.build()?),
        bucket: cfg.bucket,
    })
}

#[derive(Clone, Debug)]
pub struct AdlsConfig {
    pub account: String,
    pub container: String,
    pub endpoint: Option<String>,
    pub allow_http: Option<bool>,
    pub access_key: Option<String>,
    pub sas_key: Option<String>,
    pub bearer_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub tenant_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedAdls {
    pub account: String,
    pub container: String,
    pub endpoint: Option<String>,
    pub allow_http: bool,
    pub access_key: Option<String>,
    pub sas_key: Option<String>,
    pub bearer_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub tenant_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AdlsClient {
    pub store: Arc<dyn ObjectStore>,
    pub resolved: ResolvedAdls,
}

pub fn build_adls_client(cfg: AdlsConfig) -> Result<AdlsClient> {
    let allow_http = cfg
        .allow_http
        .or_else(|| env_flag("AZURE_ALLOW_HTTP"))
        .unwrap_or(false);
    // with_client_options replaces the entire ClientOptions struct,
    // so it must come before with_allow_http.
    let mut builder = MicrosoftAzureBuilder::new()
        .with_client_options(cloud_client_options())
        .with_account(&cfg.account)
        .with_container_name(&cfg.container)
        .with_allow_http(allow_http);
    if let Some(ep) = cfg
        .endpoint
        .clone()
        .or_else(|| std::env::var("AZURE_ENDPOINT").ok())
    {
        builder = builder.with_endpoint(ep);
    }
    if let Some(access_key) = cfg
        .access_key
        .clone()
        .or_else(|| std::env::var("AZURE_ACCESS_KEY").ok())
    {
        builder = builder.with_access_key(access_key);
    }
    if let Some(sas) = cfg
        .sas_key
        .clone()
        .or_else(|| std::env::var("AZURE_SAS_KEY").ok())
    {
        builder = builder.with_config(object_store::azure::AzureConfigKey::SasKey, sas);
    }
    if let Some(bearer) = cfg
        .bearer_token
        .clone()
        .or_else(|| std::env::var("AZURE_BEARER_TOKEN").ok())
    {
        builder = builder.with_bearer_token_authorization(bearer);
    }
    if let Some(client_id) = cfg
        .client_id
        .clone()
        .or_else(|| std::env::var("AZURE_CLIENT_ID").ok())
    {
        builder = builder.with_client_id(client_id);
    }
    if let Some(client_secret) = cfg
        .client_secret
        .clone()
        .or_else(|| std::env::var("AZURE_CLIENT_SECRET").ok())
    {
        builder = builder.with_client_secret(client_secret);
    }
    if let Some(tenant_id) = cfg
        .tenant_id
        .clone()
        .or_else(|| std::env::var("AZURE_TENANT_ID").ok())
    {
        builder = builder.with_tenant_id(tenant_id);
    }
    Ok(AdlsClient {
        store: Arc::new(builder.build()?),
        resolved: ResolvedAdls {
            account: cfg.account,
            container: cfg.container,
            endpoint: cfg.endpoint,
            allow_http,
            access_key: cfg.access_key,
            sas_key: cfg.sas_key,
            bearer_token: cfg.bearer_token,
            client_id: cfg.client_id,
            client_secret: cfg.client_secret,
            tenant_id: cfg.tenant_id,
        },
    })
}

#[derive(Clone, Debug)]
pub struct UploadBufferPool {
    inner: Arc<UploadBufferPoolInner>,
}

#[derive(Debug)]
struct UploadBufferPoolInner {
    buffers: Mutex<Vec<Vec<u8>>>,
    available: Condvar,
    buffer_size: usize,
}

impl UploadBufferPool {
    pub fn new(buffer_size: usize, buffer_count: usize) -> Self {
        let mut buffers = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            buffers.push(Vec::with_capacity(buffer_size));
        }
        Self {
            inner: Arc::new(UploadBufferPoolInner {
                buffers: Mutex::new(buffers),
                available: Condvar::new(),
                buffer_size,
            }),
        }
    }

    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    fn acquire(&self) -> Vec<u8> {
        let mut guard = self.inner.buffers.lock().expect("upload buffer pool lock");
        loop {
            if let Some(mut buffer) = guard.pop() {
                buffer.clear();
                return buffer;
            }
            guard = self
                .inner
                .available
                .wait(guard)
                .expect("upload buffer pool wait");
        }
    }

    fn release(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        let mut guard = self.inner.buffers.lock().expect("upload buffer pool lock");
        guard.push(buffer);
        self.inner.available.notify_one();
    }
}

#[derive(Debug)]
struct PoolBuffer {
    data: Vec<u8>,
    len: usize,
    pool: UploadBufferPool,
}

impl AsRef<[u8]> for PoolBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

impl Drop for PoolBuffer {
    fn drop(&mut self) {
        let buffer = std::mem::take(&mut self.data);
        self.pool.release(buffer);
    }
}

#[derive(Debug)]
enum UploadCommand {
    Part(PoolBuffer),
    Finish(oneshot::Sender<Result<()>>),
    Abort,
}

#[derive(Debug)]
pub struct MultipartUploadWriter {
    sender: mpsc::Sender<UploadCommand>,
    pool: UploadBufferPool,
    buffer: Vec<u8>,
    part_size: usize,
    position: u64,
    join: Option<thread::JoinHandle<Result<()>>>,
    error: Arc<Mutex<Option<anyhow::Error>>>,
}

impl MultipartUploadWriter {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        pool: UploadBufferPool,
        max_concurrency: usize,
    ) -> Result<Self> {
        let part_size = pool.buffer_size();
        let (sender, receiver) = mpsc::channel(max_concurrency.max(1) * 2);
        let error = Arc::new(Mutex::new(None));
        let error_clone = Arc::clone(&error);
        let join = thread::spawn(move || {
            let rt = Runtime::new()?;
            rt.block_on(run_upload_loop(
                store,
                path,
                receiver,
                part_size,
                max_concurrency.max(1),
                error_clone,
            ))
        });
        Ok(Self {
            sender,
            pool: pool.clone(),
            buffer: pool.acquire(),
            part_size,
            position: 0,
            join: Some(join),
            error,
        })
    }

    pub fn finish(mut self) -> Result<()> {
        self.flush_buffer(true)?;
        let (tx, rx) = oneshot::channel();
        self.sender
            .blocking_send(UploadCommand::Finish(tx))
            .map_err(|_| anyhow!("multipart upload worker stopped"))?;
        let result = rx
            .blocking_recv()
            .map_err(|_| anyhow!("multipart upload worker dropped"))?;
        let join = self.join.take().expect("upload join handle");
        let join_result = join
            .join()
            .map_err(|_| anyhow!("multipart upload thread panicked"))?;
        result?;
        join_result
    }

    fn flush_buffer(&mut self, final_flush: bool) -> Result<()> {
        if self.buffer.is_empty() && !final_flush {
            return Ok(());
        }
        if self.buffer.is_empty() && final_flush {
            let buffer = std::mem::take(&mut self.buffer);
            if buffer.capacity() >= self.part_size && self.part_size > 0 {
                self.pool.release(buffer);
            }
            return Ok(());
        }
        let data = std::mem::take(&mut self.buffer);
        let len = data.len();
        let part = PoolBuffer {
            data,
            len,
            pool: self.pool.clone(),
        };
        self.sender
            .blocking_send(UploadCommand::Part(part))
            .map_err(|_| anyhow!("multipart upload worker stopped"))?;
        if !final_flush {
            self.buffer = Vec::new();
        }
        Ok(())
    }

    fn check_error(&self) -> io::Result<()> {
        if let Some(err) = self.error.lock().expect("upload error lock").as_ref() {
            return Err(io::Error::new(io::ErrorKind::Other, err.to_string()));
        }
        Ok(())
    }
}

impl Write for MultipartUploadWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.check_error()?;
        if self.buffer.capacity() == 0 {
            self.buffer = self.pool.acquire();
        }
        let mut offset = 0;
        while offset < buf.len() {
            let remaining = self.part_size - self.buffer.len();
            let to_write = remaining.min(buf.len() - offset);
            self.buffer
                .extend_from_slice(&buf[offset..offset + to_write]);
            offset += to_write;
            self.position += to_write as u64;
            if self.buffer.len() == self.part_size {
                self.flush_buffer(false)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.check_error()
    }
}

impl io::Seek for MultipartUploadWriter {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Current(0) => Ok(self.position),
            SeekFrom::End(0) => Ok(self.position),
            SeekFrom::Start(offset) if offset == self.position => Ok(self.position),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "multipart upload writer only supports no-op seeks",
            )),
        }
    }
}

impl Drop for MultipartUploadWriter {
    fn drop(&mut self) {
        if self.join.is_none() {
            return;
        }
        let sender = self.sender.clone();
        let _ = sender.try_send(UploadCommand::Abort);
        if self.buffer.capacity() >= self.part_size && self.part_size > 0 {
            let buffer = std::mem::take(&mut self.buffer);
            self.pool.release(buffer);
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

async fn run_upload_loop(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    mut receiver: mpsc::Receiver<UploadCommand>,
    part_size: usize,
    max_concurrency: usize,
    error: Arc<Mutex<Option<anyhow::Error>>>,
) -> Result<()> {
    let upload = store.put_multipart(&path).await?;
    let mut writer = WriteMultipart::new_with_chunk_size(upload, part_size);
    while let Some(command) = receiver.recv().await {
        match command {
            UploadCommand::Part(part) => {
                writer.wait_for_capacity(max_concurrency).await?;
                let payload = Bytes::from_owner(part);
                writer.put(payload);
            }
            UploadCommand::Finish(done) => {
                let result: Result<()> = writer
                    .finish()
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow!(e.to_string()));
                if let Err(err) = &result {
                    if let Ok(mut guard) = error.lock() {
                        *guard = Some(anyhow!(err.to_string()));
                    }
                }
                let _ = done.send(result);
                return Ok(());
            }
            UploadCommand::Abort => {
                let _ = writer.abort().await;
                return Ok(());
            }
        }
    }
    let _ = writer.abort().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn multipart_upload_writer_reuses_buffers() -> Result<()> {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test/part.parquet");
        let pool = UploadBufferPool::new(8, 2);
        let writer = MultipartUploadWriter::new(store.clone(), path.clone(), pool.clone(), 2)?;
        let mut writer = writer;
        writer.write_all(b"abcdef")?;
        writer.write_all(b"ghijklmnop")?;
        writer.finish()?;
        let rt = Runtime::new()?;
        let data = rt
            .block_on(async { store.get(&path).await?.bytes().await })
            .map_err(|e| anyhow!(e.to_string()))?;
        assert_eq!(data.as_ref(), b"abcdefghijklmnop");
        Ok(())
    }
}
