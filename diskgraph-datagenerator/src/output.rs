use std::fs::File;
use std::io::{self, Seek, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig,
    MultipartUploadWriter, S3Config, UploadBufferPool,
};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

#[derive(Clone, Debug)]
pub enum OutputTarget {
    Local { root: PathBuf },
    S3 {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        pool: UploadBufferPool,
        max_concurrency: usize,
    },
    Gcs {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        pool: UploadBufferPool,
        max_concurrency: usize,
    },
    Adls {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        pool: UploadBufferPool,
        max_concurrency: usize,
    },
}

impl OutputTarget {
    pub fn parse(output: &str, workers: usize) -> Result<Self> {
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
            let pool = UploadBufferPool::new(64 * 1024 * 1024, pool_buffer_count(workers, 8));
            Ok(OutputTarget::S3 {
                prefix,
                store: s3.store,
                pool,
                max_concurrency: 8,
            })
        } else if output.starts_with("gs://") {
            let (bucket, prefix) = parse_gs_url(output)?;
            let gcs = build_gcs_client(GcsConfig { bucket })?;
            let pool = UploadBufferPool::new(64 * 1024 * 1024, pool_buffer_count(workers, 8));
            Ok(OutputTarget::Gcs {
                prefix,
                store: gcs.store,
                pool,
                max_concurrency: 8,
            })
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
            let pool = UploadBufferPool::new(64 * 1024 * 1024, pool_buffer_count(workers, 8));
            Ok(OutputTarget::Adls {
                prefix,
                store: adls.store,
                pool,
                max_concurrency: 8,
            })
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
            OutputTarget::S3 {
                prefix,
                store,
                pool,
                max_concurrency,
            }
            | OutputTarget::Gcs {
                prefix,
                store,
                pool,
                max_concurrency,
            }
            | OutputTarget::Adls {
                prefix,
                store,
                pool,
                max_concurrency,
            } => {
                let key = join_prefix(prefix, &format!("{}/{}", table, filename));
                let writer = MultipartUploadWriter::new(
                    Arc::clone(store),
                    ObjectPath::from(key),
                    pool.clone(),
                    *max_concurrency,
                )?;
                Ok(ParquetSink::Cloud(writer))
            }
        }
    }
}

#[derive(Debug)]
pub enum ParquetSink {
    Local(File),
    Cloud(MultipartUploadWriter),
}

impl ParquetSink {
    pub fn finish(self) -> Result<()> {
        match self {
            ParquetSink::Local(_) => Ok(()),
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

fn join_prefix(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), name)
    }
}

fn pool_buffer_count(workers: usize, max_concurrency: usize) -> usize {
    let min_buffers = max_concurrency.saturating_mul(2);
    let mut buffers = max_concurrency.saturating_mul(workers.max(1));
    if buffers < min_buffers {
        buffers = min_buffers;
    }
    if buffers > 64 {
        buffers = 64;
    }
    buffers
}

fn parse_s3_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid s3 url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in s3 url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_gs_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid gs url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in gs url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_az_url(raw: &str) -> Result<(String, String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid az url {raw}: {e}"))?;
    let account = url
        .host_str()
        .ok_or_else(|| anyhow!("missing account in az url"))?
        .to_string();
    let mut segments = url
        .path()
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty());
    let container = segments
        .next()
        .ok_or_else(|| anyhow!("missing container in az url"))?
        .to_string();
    let key = segments.collect::<Vec<_>>().join("/");
    Ok((account, container, key))
}
