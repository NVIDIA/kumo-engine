use std::fs::File;
use std::io::{self, Seek, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, CloudTransfer,
    CloudTransferConfig, CloudTransferStatsHandle, CloudTransferTicket, GcsConfig, PooledBuffer,
    S3Config,
};
use object_store::path::Path as ObjectPath;

#[derive(Clone, Debug)]
pub enum OutputTarget {
    Local { root: PathBuf },
    S3 {
        prefix: String,
        transfer: CloudTransfer,
    },
    Gcs {
        prefix: String,
        transfer: CloudTransfer,
    },
    Adls {
        prefix: String,
        transfer: CloudTransfer,
    },
}

impl OutputTarget {
    pub fn parse(output: &str, _workers: usize) -> Result<Self> {
        let config = CloudTransferConfig::from_env();
        if output.starts_with("s3://") {
            let (bucket, prefix) = parse_s3_url(output)?;
            let s3 = build_s3_client(S3Config {
                bucket: bucket.clone(),
                region: None,
                endpoint: None,
                allow_http: None,
                virtual_hosted: None,
                profile: std::env::var("S3_PROFILE").ok(),
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
            })?;
            let transfer = CloudTransfer::new(s3.store, config);
            Ok(OutputTarget::S3 { prefix, transfer })
        } else if output.starts_with("gs://") {
            let (bucket, prefix) = parse_gs_url(output)?;
            let gcs = build_gcs_client(GcsConfig { bucket })?;
            let transfer = CloudTransfer::new(gcs.store, config);
            Ok(OutputTarget::Gcs { prefix, transfer })
        } else if output.starts_with("az://") {
            let (account, container, prefix) = parse_az_url(output)?;
            let adls = build_adls_client(AdlsConfig {
                account,
                container,
                endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                allow_http: None,
                access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
            })?;
            let transfer = CloudTransfer::new(adls.store, config);
            Ok(OutputTarget::Adls { prefix, transfer })
        } else {
            Ok(OutputTarget::Local {
                root: PathBuf::from(output),
            })
        }
    }

    pub fn prepare(&self, tables: &[String]) -> Result<()> {
        if let OutputTarget::Local { root } = self {
            std::fs::create_dir_all(root)?;
            for table in tables {
                std::fs::create_dir_all(root.join(table))?;
            }
        }
        Ok(())
    }

    pub fn parquet_sink(&self, table: &str, shard_idx: usize) -> Result<ParquetSink> {
        let filename = format!("part-{:05}.parquet", shard_idx);
        match self {
            OutputTarget::Local { root } => {
                let path = root.join(table).join(filename);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let file =
                    File::create(&path).with_context(|| format!("create {}", path.display()))?;
                Ok(ParquetSink::Local(file))
            }
            OutputTarget::S3 { prefix, transfer }
            | OutputTarget::Gcs { prefix, transfer }
            | OutputTarget::Adls { prefix, transfer } => {
                let key = join_prefix(prefix, &format!("{}/{}", table, filename));
                let buffer = transfer.acquire_buffer();
                Ok(ParquetSink::Cloud(CloudSink::new(
                    transfer.clone(),
                    ObjectPath::from(key),
                    buffer,
                )))
            }
        }
    }

    pub fn stats_handle(&self) -> Option<CloudTransferStatsHandle> {
        match self {
            OutputTarget::Local { .. } => None,
            OutputTarget::S3 { transfer, .. }
            | OutputTarget::Gcs { transfer, .. }
            | OutputTarget::Adls { transfer, .. } => Some(transfer.stats_handle()),
        }
    }
}

#[derive(Debug)]
pub enum ParquetSink {
    Local(File),
    Cloud(CloudSink),
}

impl ParquetSink {
    pub fn finish(self) -> Result<CloudTransferTicket> {
        match self {
            ParquetSink::Local(_) => Ok(CloudTransferTicket::ready(Ok(()))),
            ParquetSink::Cloud(writer) => writer.finish(),
        }
    }
}

impl Write for ParquetSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            ParquetSink::Local(file) => file.write(buf),
            ParquetSink::Cloud(writer) => writer.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            ParquetSink::Local(file) => file.flush(),
            ParquetSink::Cloud(writer) => writer.flush(),
        }
    }
}

impl Seek for ParquetSink {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        match self {
            ParquetSink::Local(file) => file.seek(pos),
            ParquetSink::Cloud(writer) => writer.seek(pos),
        }
    }
}

#[derive(Debug)]
pub struct CloudSink {
    transfer: CloudTransfer,
    path: ObjectPath,
    buffer: PooledBuffer,
}

impl CloudSink {
    fn new(transfer: CloudTransfer, path: ObjectPath, buffer: PooledBuffer) -> Self {
        Self {
            transfer,
            path,
            buffer,
        }
    }

    fn finish(self) -> Result<CloudTransferTicket> {
        let buffer = self.buffer.into_inner();
        self.transfer.submit(self.path, buffer)
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
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.buffer.seek(pos)
    }
}

fn parse_s3_url(url: &str) -> Result<(String, String)> {
    let url = url.strip_prefix("s3://").ok_or_else(|| anyhow!("bad s3 url"))?;
    let mut parts = url.splitn(2, '/');
    let bucket = parts.next().unwrap_or_default();
    let prefix = parts.next().unwrap_or_default();
    if bucket.is_empty() {
        return Err(anyhow!("s3 url must include bucket"));
    }
    Ok((bucket.to_string(), prefix.to_string()))
}

fn parse_gs_url(url: &str) -> Result<(String, String)> {
    let url = url.strip_prefix("gs://").ok_or_else(|| anyhow!("bad gs url"))?;
    let mut parts = url.splitn(2, '/');
    let bucket = parts.next().unwrap_or_default();
    let prefix = parts.next().unwrap_or_default();
    if bucket.is_empty() {
        return Err(anyhow!("gs url must include bucket"));
    }
    Ok((bucket.to_string(), prefix.to_string()))
}

fn parse_az_url(url: &str) -> Result<(String, String, String)> {
    let url = url.strip_prefix("az://").ok_or_else(|| anyhow!("bad az url"))?;
    let mut parts = url.splitn(3, '/');
    let account = parts.next().unwrap_or_default();
    let container = parts.next().unwrap_or_default();
    let prefix = parts.next().unwrap_or_default();
    if account.is_empty() || container.is_empty() {
        return Err(anyhow!("az url must include account/container"));
    }
    Ok((account.to_string(), container.to_string(), prefix.to_string()))
}

fn join_prefix(prefix: &str, path: &str) -> String {
    if prefix.is_empty() {
        return path.to_string();
    }
    format!("{}/{}", prefix.trim_end_matches('/'), path)
}
