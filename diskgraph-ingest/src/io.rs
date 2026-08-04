use crate::consts::TEMP_DIR_NAME;
use crate::errors::ConvertError;
use crate::graph_config::{
    parse_cloud_source, AzureCloud, CloudSource, DataSource, GraphConfig, ObjectStoreAuthority,
    ObjectStoreObject, TableSpec,
};
use crate::snapshot_mirror::{ManifestAccumulator, SnapshotMirror};

/// True when a snapshot-relative path is under the per-ingest temp tree
/// (`_tmp/...`). These are intermediate files that ingest deletes via
/// `cleanup_temp` before returning, so they must NOT appear in
/// `_MANIFEST.json` (would cause `verify_snapshot_integrity` to flag them
/// as missing post-cleanup) AND must NOT be uploaded to the dataplane
/// mirror (wasted bytes for files we're about to delete locally).
///
/// The check is rel-path based — the per-shard parallel write paths
/// already construct rel paths like `_tmp/nodes/<table>/shard-XX/...`.
fn is_temp_rel_path(rel: &str) -> bool {
    rel == TEMP_DIR_NAME || rel.starts_with(&format!("{}/", TEMP_DIR_NAME))
}

/// Recursive directory walker used by
/// [`OutputTarget::reconcile_unrecorded_files`]. For every regular file
/// under `dir`, computes its rel-path relative to `root`, skips temp
/// files and anything already in `known`, and emits the (rel_path,
/// absolute_path, size) tuple. Bytes are NOT read here — the mirror
/// submit streams them from disk directly so memory stays bounded.
fn walk_local_files(
    root: &Path,
    dir: &Path,
    known: &std::collections::HashSet<String>,
    out: &mut Vec<(String, PathBuf, u64)>,
) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            walk_local_files(root, &path, known, out)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .with_context(|| format!("strip_prefix {}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        if is_temp_rel_path(&rel) || known.contains(&rel) {
            continue;
        }
        let size = entry
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        out.push((rel, path, size));
    }
    Ok(())
}

use anyhow::{anyhow, ensure, Context, Result};
use bytes::Bytes;
use diskgraph_fs::{
    build_adls_client, build_adls_client_with_authority, build_gcs_client,
    build_gcs_client_with_base_url, build_s3_client, build_s3_client_without_env_endpoint,
    AdlsConfig, CloudTransfer, CloudTransferConfig, GcsConfig, PooledBytes, S3Config,
};
use object_store::{path::Path as ObjectPath, ObjectStore, ObjectStoreExt, WriteMultipart};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

#[derive(Clone, Copy, Debug)]
pub enum Dest {
    Local,
    S3,
    Gcs,
    Adls,
}

pub fn parse_s3_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid s3 url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in s3 url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

pub fn parse_gs_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw).map_err(|e| anyhow!("invalid gs url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow!("missing bucket in gs url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

pub fn parse_az_url(raw: &str) -> Result<(String, String, String)> {
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

pub(crate) fn build_input_transfer_for_table(table: &TableSpec) -> Result<Option<CloudTransfer>> {
    match &table.source {
        DataSource::ObjectStoreParquet { authority, .. } => {
            let first = table.validate_object_store_manifest()?;
            Ok(Some(build_transfer_for_object_store_source(
                &first.path,
                authority.as_ref(),
            )?))
        }
        DataSource::Parquet { paths } => build_input_transfer_for_cloud_paths(table, paths),
        DataSource::Csv { paths } => build_input_transfer_for_cloud_paths(table, paths),
        DataSource::Snowflake { .. }
        | DataSource::Databricks { .. }
        | DataSource::BigQuery { .. } => Ok(None),
    }
}

/// Compatibility wrapper for callers that still request one graph-level transfer.
pub fn build_input_transfer(source: &GraphConfig) -> Result<Option<CloudTransfer>> {
    for table in &source.tables {
        if let Some(transfer) = build_input_transfer_for_table(table)? {
            return Ok(Some(transfer));
        }
    }
    Ok(None)
}

fn build_input_transfer_for_cloud_paths(
    table: &TableSpec,
    paths: &[String],
) -> Result<Option<CloudTransfer>> {
    let mut cloud_paths = paths.iter().filter(|path| is_cloud_path(path));
    let Some(first) = cloud_paths.next() else {
        return Ok(None);
    };
    let first_authority =
        parse_cloud_source(first)?.ok_or_else(|| anyhow!("invalid cloud URL {first}"))?;
    for path in cloud_paths {
        ensure!(
            parse_cloud_source(path)? == Some(first_authority.clone()),
            "table {} object paths must share one provider authority",
            table.name
        );
    }
    Ok(Some(build_transfer_for_cloud_path(first)?))
}

pub fn build_transfer_for_cloud_path(path: &str) -> Result<CloudTransfer> {
    let store = build_store_for_cloud_path(path)?;
    Ok(CloudTransfer::new(store, CloudTransferConfig::from_env()))
}

fn build_transfer_for_object_store_source(
    path: &str,
    authority: Option<&ObjectStoreAuthority>,
) -> Result<CloudTransfer> {
    let store = build_store_for_cloud_path_with_authority(path, authority)?;
    Ok(CloudTransfer::new(store, CloudTransferConfig::from_env()))
}

/// Build an [`ObjectStore`] for any path — cloud (`s3://`, `gs://`, `az://`)
/// or local filesystem. The returned store is rooted at the path itself for
/// local destinations and at the bucket/container for cloud destinations;
/// in both cases [`object_path_for_path`] returns the appropriate dest
/// [`ObjectPath`] to compose with file basenames at submit time.
///
/// Used by the dataplane snapshot mirror, which is uniform across cloud and
/// local backends so the same code path covers production (S3) and
/// singlehost / dev / CI (local EBS or tmpfs).
///
/// **Supported destinations:**
/// - `s3://bucket/key` — AWS S3
/// - `gs://bucket/key` — Google Cloud Storage
/// - `az://account/container/key` — Azure Data Lake Storage Gen2
/// - any path without a `://` scheme — local filesystem (treated as
///   absolute or cwd-relative)
///
/// **Explicitly NOT supported (rejected at this boundary):**
/// - `dbfs://...` / `dbfs:/...` — Databricks File System. The
///   `object_store` crate has no DBFS backend; we'd need either a custom
///   `ObjectStore` impl over the DBFS REST API or for the caller to
///   resolve `dbfs://` to the underlying storage URL before reaching
///   Rust. Out of scope for this PR.
/// - `/Volumes/...` / Unity Catalog volume references — same situation;
///   no `object_store` adapter exists.
/// - `@db.schema.stage/...` / Snowflake internal stages — accessed via
///   SQL `PUT`/`GET`, not a REST/object-store API. Out of scope.
/// - any other scheme (`file://`, `hdfs://`, etc.) — rejected loudly so
///   typos don't silently land in the local-fs branch.
pub fn build_object_store_for_path(path: &str) -> Result<Arc<dyn ObjectStore>> {
    if is_cloud_path(path) {
        return build_store_for_cloud_path(path);
    }
    if let Some(reason) = unsupported_path_reason(path) {
        return Err(anyhow!(
            "unsupported dataplane path {path:?}: {reason}. Supported: \
             s3://, gs://, az://, or a local filesystem path."
        ));
    }
    // Local filesystem branch: the store is rooted at `path`, so submit
    // ObjectPaths are relative to `path`. Ensure the dir exists first —
    // `LocalFileSystem::new_with_prefix` requires it; cloud buckets are
    // pre-existing so this branch is the only one that needs to create.
    fs::create_dir_all(path).with_context(|| format!("create local prefix {path}"))?;
    let store = object_store::local::LocalFileSystem::new_with_prefix(path)
        .with_context(|| format!("build LocalFileSystem at {path}"))?;
    Ok(Arc::new(store))
}

/// Identify dataplane paths that look like a deliberate scheme we don't
/// support, so they fail loudly instead of slipping into the local-fs
/// branch (which would `create_dir_all("dbfs:")` — silently create a
/// literal `dbfs:` directory in the cwd and write the snapshot there).
///
/// Returns `Some(reason)` for known-unsupported shapes; `None` for
/// anything that looks like a legitimate local path. The check is
/// conservative — if the caller has a path that begins with a custom
/// scheme not enumerated here, we still reject it via the
/// generic-`://` arm rather than silently treating it as local.
fn unsupported_path_reason(path: &str) -> Option<&'static str> {
    let lower = path.to_ascii_lowercase();
    if lower.starts_with("dbfs://") || lower.starts_with("dbfs:/") {
        return Some(
            "DBFS / Unity Catalog volumes are not supported by the Rust \
             dataplane mirror; resolve to the underlying S3/ADLS URL on \
             the Python side or skip the mirror for Databricks deployments",
        );
    }
    if lower.starts_with("snowflake://") || path.starts_with('@') {
        return Some(
            "Snowflake internal stages are not supported by the Rust \
             dataplane mirror (no `object_store` impl over PUT/GET SQL); \
             out of scope for this PR",
        );
    }
    if lower.starts_with("file://") {
        return Some("use a bare local filesystem path instead of `file://...`");
    }
    if lower.starts_with("hdfs://") {
        return Some("HDFS is not a supported dataplane backend");
    }
    // Generic catch-all: any path with a scheme separator that isn't
    // one of the supported cloud schemes is rejected. Path absolutism
    // on Unix means `/foo/bar` doesn't trigger this; relative paths
    // also don't.
    if let Some((scheme, _)) = path.split_once("://") {
        if !scheme.is_empty() && !is_cloud_scheme(scheme) {
            return Some(
                "unrecognized URL scheme (only s3 / gs / az and local \
                 filesystem paths are supported)",
            );
        }
    }
    None
}

fn is_cloud_scheme(scheme: &str) -> bool {
    matches!(scheme.to_ascii_lowercase().as_str(), "s3" | "gs" | "az")
}

/// Companion to [`build_object_store_for_path`]. Returns the dest
/// [`ObjectPath`] (i.e., the path prefix to compose with per-file basenames
/// when submitting uploads/downloads) for the given path. For local paths,
/// the dest is `""` because the [`ObjectStore`] itself is rooted at the
/// local dir; for cloud paths, the dest is the path-within-bucket/container.
pub fn object_path_for_path(path: &str) -> Result<ObjectPath> {
    if is_cloud_path(path) {
        return object_path_from_cloud_url(path);
    }
    Ok(ObjectPath::from(""))
}

pub fn build_store_for_cloud_path(path: &str) -> Result<Arc<dyn ObjectStore>> {
    build_store_for_cloud_path_with_authority(path, None)
}

fn source_s3_settings(
    bucket: &str,
    authority: Option<&ObjectStoreAuthority>,
    worker_region: Option<String>,
) -> Result<(Option<String>, Option<bool>, bool)> {
    match authority {
        Some(ObjectStoreAuthority::S3 { region }) => Ok((
            Some(region.clone()),
            // AWS wildcard certificates do not cover dotted bucket labels.
            Some(!bucket.contains('.')),
            false,
        )),
        None => Ok((worker_region, None, true)),
        Some(_) => Err(anyhow!("S3 path has non-S3 source authority")),
    }
}

const GCS_PUBLIC_BASE_URL: &str = "https://storage.googleapis.com";

fn source_adls_settings(
    account: &str,
    authority: Option<&ObjectStoreAuthority>,
    worker_endpoint: Option<String>,
) -> Result<(Option<String>, Option<String>)> {
    match authority {
        Some(ObjectStoreAuthority::Adls {
            cloud: AzureCloud::Public,
        }) => Ok((
            Some(format!("https://{account}.blob.core.windows.net")),
            Some(object_store::azure::authority_hosts::AZURE_PUBLIC_CLOUD.to_string()),
        )),
        None => Ok((worker_endpoint, None)),
        Some(ObjectStoreAuthority::Adls { cloud }) => Err(anyhow!(
            "ADLS source cloud {cloud:?} is not supported by the current object-store client"
        )),
        Some(_) => Err(anyhow!("ADLS path has non-ADLS source authority")),
    }
}

fn build_store_for_cloud_path_with_authority(
    path: &str,
    authority: Option<&ObjectStoreAuthority>,
) -> Result<Arc<dyn ObjectStore>> {
    if path.starts_with("s3://") {
        let (bucket, _key) = parse_s3_url(path)?;
        let (region, virtual_hosted, inherit_worker_endpoint) =
            source_s3_settings(&bucket, authority, std::env::var("S3_REGION").ok())?;
        let config = S3Config {
            bucket: bucket.clone(),
            region,
            endpoint: None,
            allow_http: None,
            virtual_hosted,
            profile: std::env::var("S3_PROFILE")
                .or_else(|_| std::env::var("AWS_PROFILE"))
                .ok(),
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
        };
        let s3 = if inherit_worker_endpoint {
            build_s3_client(config)?
        } else {
            build_s3_client_without_env_endpoint(config)?
        };
        Ok(s3.store)
    } else if path.starts_with("gs://") {
        let base_url = match authority {
            Some(ObjectStoreAuthority::Gcs) => Some(GCS_PUBLIC_BASE_URL),
            None => None,
            Some(_) => return Err(anyhow!("GCS path has non-GCS source authority")),
        };
        let (bucket, _key) = parse_gs_url(path)?;
        let gcs = build_gcs_client_with_base_url(
            GcsConfig {
                bucket: bucket.clone(),
            },
            base_url,
        )?;
        Ok(gcs.store)
    } else if path.starts_with("az://") {
        let (account, container, _key) = parse_az_url(path)?;
        let (endpoint, authority_host) =
            source_adls_settings(&account, authority, std::env::var("AZURE_ENDPOINT").ok())?;
        let adls = build_adls_client_with_authority(
            AdlsConfig {
                account: account.clone(),
                container: container.clone(),
                endpoint,
                allow_http: None,
                access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
            },
            authority_host,
        )?;
        Ok(adls.store)
    } else {
        Err(anyhow!("unsupported source path {}", path))
    }
}

pub fn object_path_from_cloud_url(path: &str) -> Result<ObjectPath> {
    if path.starts_with("s3://") {
        let (_bucket, key) = parse_s3_url(path)?;
        Ok(ObjectPath::from(key))
    } else if path.starts_with("gs://") {
        let (_bucket, key) = parse_gs_url(path)?;
        Ok(ObjectPath::from(key))
    } else if path.starts_with("az://") {
        let (_account, _container, key) = parse_az_url(path)?;
        Ok(ObjectPath::from(key))
    } else {
        Err(anyhow!("unsupported source path {}", path))
    }
}

pub fn download_cloud_bytes(transfer: &CloudTransfer, path: &str) -> Result<PooledBytes> {
    download_cloud_bytes_condition(transfer, path, None)
}

pub fn download_cloud_bytes_condition(
    transfer: &CloudTransfer,
    path: &str,
    condition: Option<diskgraph_fs::CloudObjectCondition>,
) -> Result<PooledBytes> {
    transfer
        .submit_download_bytes_condition(object_path_from_cloud_url(path)?, condition)?
        .wait()
}

pub fn resolve_local_paths(source: &mut GraphConfig, base_dir: &Path) {
    for table in &mut source.tables {
        let paths = match &mut table.source {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => paths,
            _ => continue,
        };
        for path in paths {
            if is_cloud_path(path) {
                continue;
            }
            let path_buf = PathBuf::from(path.as_str());
            if path_buf.is_absolute() {
                continue;
            }
            let joined = base_dir.join(path_buf);
            *path = joined.to_string_lossy().to_string();
        }
    }
}

pub fn is_cloud_path(path: &str) -> bool {
    path.starts_with("s3://") || path.starts_with("gs://") || path.starts_with("az://")
}

/// Expand a cloud directory prefix to individual data file URLs.
///
/// Lists all objects under the prefix and returns sorted full cloud URLs for
/// every `.parquet` or `.csv` file found.  Returns an error if both types are
/// present in the same directory.
///
/// The list operation is bounded by a 10-minute timeout so that an
/// unreachable cloud endpoint cannot hang the ingest process indefinitely.
fn expand_cloud_data_paths(dir_url: &str) -> Result<(DataSource, u64)> {
    let transfer = build_transfer_for_cloud_path(dir_url)?;
    let prefix = object_path_from_cloud_url(dir_url)?;
    let list_timeout = Duration::from_secs(600);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let objects = rt.block_on(async {
        tokio::time::timeout(list_timeout, transfer.list_objects(&prefix))
            .await
            .map_err(|_| {
                anyhow!(
                    "Timed out listing data files at {} (exceeded {}s). \
                     Check that the cloud path is reachable and credentials \
                     are available.",
                    dir_url,
                    list_timeout.as_secs(),
                )
            })?
    })?;

    let cloud_root = match parse_cloud_source(dir_url)? {
        Some(CloudSource::S3 { bucket }) => format!("s3://{bucket}"),
        Some(CloudSource::Gcs { bucket }) => format!("gs://{bucket}"),
        Some(CloudSource::Adls { account, container }) => {
            format!("az://{account}/{container}")
        }
        None => return Err(anyhow!("invalid cloud source URL {dir_url}")),
    };

    let mut parquet_objects = Vec::new();
    let mut csv_files: Vec<String> = Vec::new();
    let mut total_bytes: u64 = 0;
    for object in objects {
        let loc = object.location.as_ref();
        let full = format!("{cloud_root}/{loc}");
        if loc.ends_with(".parquet") {
            ensure!(
                object.e_tag.is_some() || object.version.is_some(),
                "cloud Parquet object {full} has no immutable identity"
            );
            total_bytes = total_bytes.saturating_add(object.size);
            parquet_objects.push(ObjectStoreObject {
                path: full,
                size: object.size,
                e_tag: object.e_tag,
                version: object.version,
            });
        } else if loc.ends_with(".csv") {
            csv_files.push(full);
            total_bytes = total_bytes.saturating_add(object.size);
        }
    }
    if !parquet_objects.is_empty() && !csv_files.is_empty() {
        return Err(anyhow!(
            "Cloud path {} contains both .parquet and .csv files; must be one or the other",
            dir_url
        ));
    }
    if !parquet_objects.is_empty() {
        parquet_objects.sort_unstable_by(|left, right| left.path.cmp(&right.path));
        Ok((
            DataSource::ObjectStoreParquet {
                objects: parquet_objects,
                auth_profile: "ambient".to_string(),
                authority: None,
            },
            total_bytes,
        ))
    } else {
        csv_files.sort();
        Ok((DataSource::Csv { paths: csv_files }, total_bytes))
    }
}

fn expand_local_data_paths(dir_path: &str) -> Result<(DataSource, u64)> {
    let path = Path::new(dir_path.trim_end_matches('/'));
    if !path.is_dir() {
        // Already a concrete file path. Use the slash-trimmed path so that a
        // trailing slash appended by build_table_paths doesn't end up in the
        // returned path and break downstream opens.
        let path_str = path.to_string_lossy().into_owned();
        let size = fs::metadata(&path_str).map(|m| m.len()).unwrap_or(0);
        return Ok(if path_str.ends_with(".csv") {
            (
                DataSource::Csv {
                    paths: vec![path_str],
                },
                size,
            )
        } else {
            (
                DataSource::Parquet {
                    paths: vec![path_str],
                },
                size,
            )
        });
    }
    let mut parquet_files: Vec<String> = Vec::new();
    let mut csv_files: Vec<String> = Vec::new();
    let mut total_bytes: u64 = 0;
    let mut files = Vec::new();
    walk_local_files(path, path, &std::collections::HashSet::new(), &mut files)?;
    for (_, file_path, size) in files {
        let file_path = file_path.to_string_lossy().into_owned();
        if file_path.ends_with(".parquet") {
            total_bytes += size;
            parquet_files.push(file_path);
        } else if file_path.ends_with(".csv") {
            total_bytes += size;
            csv_files.push(file_path);
        }
    }
    if !parquet_files.is_empty() && !csv_files.is_empty() {
        return Err(anyhow!(
            "Directory {} contains both .parquet and .csv files; must be one or the other",
            dir_path
        ));
    }
    if !parquet_files.is_empty() {
        parquet_files.sort();
        Ok((
            DataSource::Parquet {
                paths: parquet_files,
            },
            total_bytes,
        ))
    } else {
        csv_files.sort();
        Ok((DataSource::Csv { paths: csv_files }, total_bytes))
    }
}

/// Expand a list of directory paths into a `DataSource` with individual file
/// paths.  All directories must contain only `.parquet` or only `.csv` files
/// (not a mix).  Concrete file paths are passed through unchanged.
///
/// I/O and transport failures → `ConvertError::SourcePathsSystem`.
/// Missing-file and mixed-type problems → `ConvertError::SourcePaths`.
pub(crate) fn expand_source_paths(
    dir_paths: Vec<String>,
    table_name: &str,
) -> Result<(DataSource, u64), ConvertError> {
    let mut merged: Option<DataSource> = None;
    let mut total_bytes: u64 = 0;
    for dir_path in &dir_paths {
        let (source, bytes) = if is_cloud_path(dir_path) {
            expand_cloud_data_paths(dir_path)
        } else {
            expand_local_data_paths(dir_path)
        }
        .map_err(|e| ConvertError::SourcePathsIO {
            table: table_name.to_string(),
            message: e.to_string(),
        })?;

        let is_empty = match &source {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => paths.is_empty(),
            DataSource::ObjectStoreParquet { objects, .. } => objects.is_empty(),
            _ => true,
        };
        if is_empty {
            return Err(ConvertError::SourcePaths {
                table: table_name.to_string(),
                message: format!("no .parquet or .csv files found in {dir_path}"),
            });
        }
        total_bytes += bytes;
        merged = Some(match (merged, source) {
            (None, s) => s,
            (Some(DataSource::Parquet { mut paths }), DataSource::Parquet { paths: new }) => {
                paths.extend(new);
                DataSource::Parquet { paths }
            }
            (
                Some(DataSource::ObjectStoreParquet {
                    mut objects,
                    auth_profile,
                    authority,
                }),
                DataSource::ObjectStoreParquet {
                    objects: new,
                    auth_profile: new_auth_profile,
                    authority: new_authority,
                },
            ) if auth_profile == new_auth_profile && authority == new_authority => {
                objects.extend(new);
                DataSource::ObjectStoreParquet {
                    objects,
                    auth_profile,
                    authority,
                }
            }
            (Some(DataSource::Csv { mut paths }), DataSource::Csv { paths: new }) => {
                paths.extend(new);
                DataSource::Csv { paths }
            }
            _ => {
                return Err(ConvertError::SourcePaths {
                    table: table_name.to_string(),
                    message: "mixed .parquet and .csv inputs across source directories".to_string(),
                })
            }
        });
    }
    Ok((
        merged.unwrap_or(DataSource::Parquet { paths: vec![] }),
        total_bytes,
    ))
}

pub fn require_local_output(target: &OutputTarget) -> Result<&Path> {
    match target {
        OutputTarget::Local { root, .. } => Ok(root.as_path()),
        _ => Err(anyhow!("sharded ingest requires a local output path")),
    }
}

#[derive(Debug)]
pub enum OutputTarget {
    Local {
        root: PathBuf,
        /// Optional secondary write channel. When set, every output file
        /// ingest writes through this target is also `submit_upload`'d to
        /// the dataplane via [`SnapshotMirror`]. End-of-ingest finalize
        /// (`snapshot_mirror::finalize_snapshot`) drains pending uploads
        /// and orders ``_MANIFEST.json`` then ``_SUCCESS`` last.
        mirror: Option<Arc<SnapshotMirror>>,
        /// Manifest accumulator. Always populated (used both with and
        /// without a mirror): every output file goes here so
        /// ``_MANIFEST.json`` records the file list + sizes for
        /// post-ingest integrity verification.
        manifest: Arc<ManifestAccumulator>,
    },
    S3 {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Gcs {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Adls {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
}

impl OutputTarget {
    pub fn parse(output: &str, dest: Dest) -> Result<Self> {
        Self::parse_with_dataplane(output, dest, None)
    }

    /// Parse the output path, optionally attaching a dataplane mirror to
    /// the local target. The mirror is only meaningful for local targets
    /// (the ingest write loop only operates on
    /// [`OutputTarget::Local`]); cloud targets ignore `dataplane_dir`.
    pub fn parse_with_dataplane(
        output: &str,
        dest: Dest,
        dataplane_dir: Option<&str>,
    ) -> Result<Self> {
        match dest {
            Dest::Local => {
                let mirror = match dataplane_dir {
                    Some(dir) => {
                        Some(Arc::new(SnapshotMirror::new(dir).with_context(|| {
                            format!("attach dataplane mirror at {dir}")
                        })?))
                    }
                    None => None,
                };
                Ok(OutputTarget::Local {
                    root: PathBuf::from(output),
                    mirror,
                    manifest: Arc::new(ManifestAccumulator::new()),
                })
            }
            Dest::S3 => {
                let (bucket, prefix) = parse_s3_url(output)?;
                let s3 = build_s3_client(S3Config {
                    bucket: bucket.clone(),
                    region: None,
                    endpoint: None,
                    allow_http: None,
                    virtual_hosted: None,
                    profile: std::env::var("S3_PROFILE")
                        .or_else(|_| std::env::var("AWS_PROFILE"))
                        .ok(),
                    access_key_id: None,
                    secret_access_key: None,
                    session_token: None,
                })?;
                Ok(OutputTarget::S3 {
                    prefix,
                    store: s3.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Gcs => {
                let (bucket, prefix) = parse_gs_url(output)?;
                let gcs = build_gcs_client(GcsConfig { bucket })?;
                Ok(OutputTarget::Gcs {
                    prefix,
                    store: gcs.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Adls => {
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
                Ok(OutputTarget::Adls {
                    prefix,
                    store: adls.store,
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
        }
    }

    pub fn prepare(&self) -> Result<()> {
        if let OutputTarget::Local { root, .. } = self {
            fs::create_dir_all(root)?;
        }
        Ok(())
    }

    pub fn join(&self, name: &str) -> PathBuf {
        match self {
            OutputTarget::Local { root, .. } => root.join(name),
            OutputTarget::S3 { prefix, .. }
            | OutputTarget::Gcs { prefix, .. }
            | OutputTarget::Adls { prefix, .. } => {
                if prefix.is_empty() {
                    PathBuf::from(name)
                } else {
                    PathBuf::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                }
            }
        }
    }

    /// Borrow the manifest accumulator on a local target. Cloud targets
    /// don't carry one (the dataplane mirror only attaches to local
    /// ingest output, which is the only `OutputTarget` variant the
    /// ingest write loop accepts in production).
    pub fn manifest(&self) -> Option<&Arc<ManifestAccumulator>> {
        match self {
            OutputTarget::Local { manifest, .. } => Some(manifest),
            _ => None,
        }
    }

    /// Borrow the dataplane mirror handle, if attached.
    pub fn mirror(&self) -> Option<&Arc<SnapshotMirror>> {
        match self {
            OutputTarget::Local { mirror, .. } => mirror.as_ref(),
            _ => None,
        }
    }

    /// Invalidate publication before any retry/fresh ingest mutates snapshot
    /// data. Local-first preserves the L1 => L2 implication: if remote deletion
    /// fails, the old remote snapshot remains coherent but L1 is no longer
    /// advertised as complete.
    pub(crate) fn invalidate_publication(&self) -> Result<()> {
        let OutputTarget::Local { root, mirror, .. } = self else {
            return Ok(());
        };
        let local_marker = root.join(crate::success_marker::SUCCESS_MARKER_FILENAME);
        match fs::remove_file(&local_marker) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| format!("remove {}", local_marker.display()))
            }
        }
        if let Some(mirror) = mirror {
            mirror.invalidate_success_marker()?;
        }
        Ok(())
    }

    /// Walk the local output root and pull any not-yet-recorded files
    /// into the manifest accumulator AND submit them to the dataplane
    /// mirror.
    ///
    /// This exists because the index builder
    /// (`diskgraph-index::builder::ShardJob::run`) writes per-shard
    /// `offsets.bin`, `edges.bin`, and `build_stats.json` directly via
    /// `File::create` — it lives in a sibling crate that has no
    /// dependency on `OutputTarget`, so it cannot route through
    /// [`Self::write_bytes`] / [`Self::sink`]. Without this sweep, those
    /// files exist locally after `build_index` but are invisible to
    /// `_MANIFEST.json` and never submitted to the mirror, breaking the
    /// L1 ⇒ L2 invariant: a successful ingest produces a snapshot that
    /// passes `verify_snapshot_integrity` (manifest doesn't list the
    /// files), is openable from L1 (files are on local disk), but is
    /// NOT openable from L2 after restore (mirror is missing the
    /// runtime-required shard files). This is the exact failure
    /// singlehost hit on AMI stop/start: NVMe wiped, restore from L2
    /// pulled an incomplete tree, gen_tt errored opening the first
    /// shard.
    ///
    /// Call once, after `build_index` and `finalize_metadata`, before
    /// `snapshot_mirror::finalize_snapshot`. Mirror submissions are
    /// drained inside `finalize_snapshot::wait_all_pending` before the
    /// `_MANIFEST.json`/`_SUCCESS` writes, preserving `_SUCCESS`-last
    /// atomicity on both backends.
    ///
    /// Files under `_tmp/` (deleted by `cleanup_temp`) are skipped so
    /// they cannot enter the manifest. No-op for non-`Local` targets.
    pub fn reconcile_unrecorded_files(&self) -> Result<()> {
        let OutputTarget::Local {
            root,
            mirror,
            manifest,
        } = self
        else {
            return Ok(());
        };
        let known = manifest.known_paths();
        let mut to_submit: Vec<(String, PathBuf, u64)> = Vec::new();
        // Walk recursively. We collect (rel_path, local_path, size)
        // tuples first then mutate, so a partial walk failure doesn't
        // leave the manifest half-updated. Sizes come from the file's
        // metadata — we don't need to read the bytes here; the mirror
        // submit streams from disk.
        walk_local_files(root, root, &known, &mut to_submit)?;
        for (rel, local_path, size) in to_submit {
            manifest.record(&rel, size);
            if let Some(m) = mirror {
                m.submit_from_path(&rel, &local_path)
                    .with_context(|| format!("dataplane mirror submit (reconciled) {rel}"))?;
            }
        }
        Ok(())
    }

    pub fn write_bytes(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        match self {
            OutputTarget::Local {
                root,
                mirror,
                manifest,
            } => {
                let path = root.join(name);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let len = bytes.len() as u64;
                // Local-first: durable on SSD before we offer any signal
                // (mirror submit, manifest record) that the file is present.
                fs::write(&path, &bytes).with_context(|| format!("write {}", path.display()))?;
                // ``_tmp/*`` files are deleted by ``cleanup_temp`` before
                // ingest returns, so they must not enter the manifest or
                // the dataplane mirror — otherwise verify_snapshot_integrity
                // (run post-ingest) would flag them as missing locally.
                if !is_temp_rel_path(name) {
                    manifest.record(name, len);
                    if let Some(m) = mirror {
                        // Upload from the durable local file so async mirror
                        // jobs cannot retain an unbounded series of serialized
                        // metadata buffers while source output is finalized.
                        m.submit_from_path(name, &path)
                            .with_context(|| format!("dataplane mirror submit {}", name))?;
                    }
                }
            }
            OutputTarget::S3 {
                store, rt, prefix, ..
            }
            | OutputTarget::Gcs {
                store, rt, prefix, ..
            }
            | OutputTarget::Adls {
                store, rt, prefix, ..
            } => {
                let obj = if prefix.is_empty() {
                    ObjectPath::from(name)
                } else {
                    ObjectPath::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                };
                rt.block_on(store.put(&obj, Bytes::from(bytes).into()))
                    .map_err(anyhow::Error::from)?;
            }
        }
        Ok(())
    }

    pub fn sink(&self, path: &Path) -> Result<OutputSink> {
        match self {
            OutputTarget::Local {
                root,
                mirror,
                manifest,
            } => {
                let file = fs::File::create(path)?;
                // For tee'd sinks we need the per-snapshot relative path so
                // the upload destination + manifest entry compose correctly.
                // Callers historically pass an absolute (root-prefixed)
                // path; recover the rel by stripping `root`.
                let rel_path = path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"));
                let tee = match rel_path {
                    // ``_tmp/*`` sinks bypass tee entirely — those files are
                    // deleted by ``cleanup_temp`` before ingest returns, so
                    // including them in the manifest or mirroring them to
                    // the dataplane would either fail verification (post-
                    // cleanup) or waste upload bandwidth.
                    Ok(rel) if is_temp_rel_path(&rel) => None,
                    Ok(rel) => Some(LocalSinkTee {
                        rel_path: rel,
                        // The local file IS the upload source: after
                        // `OutputSink::finish` flushes, `finalize`
                        // hands this path to
                        // `SnapshotMirror::submit_from_path` which
                        // reads chunks on demand. No in-memory
                        // duplication of the written bytes.
                        local_path: path.to_path_buf(),
                        bytes_written: 0,
                        mirror: mirror.clone(),
                        manifest: Arc::clone(manifest),
                    }),
                    // Sink path outside `root` — should not happen for the
                    // `target.join()` -> `target.sink()` pattern callers
                    // use, but stay safe and skip tee instead of panicking.
                    Err(_) => None,
                };
                Ok(OutputSink::Local { file, tee })
            }
            OutputTarget::S3 {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Gcs {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Adls {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            } => {
                let obj = ObjectPath::from(path.to_string_lossy().to_string());
                let upload = rt.block_on(store.put_multipart(&obj)).map_err(to_io_err)?;
                Ok(OutputSink::Multipart(MultipartSink {
                    writer: Some(WriteMultipart::new_with_chunk_size(upload, *part_size)),
                    rt: rt.clone(),
                    max_concurrency: *part_concurrency,
                    handle: rt.handle().clone(),
                }))
            }
        }
    }
}

pub enum OutputSink {
    /// Local-filesystem sink, with optional `tee` channel that records a
    /// manifest entry on `finish` and (when a dataplane mirror is
    /// configured) submits the file's contents for upload after the local
    /// flush completes.
    Local {
        file: fs::File,
        tee: Option<LocalSinkTee>,
    },
    Multipart(MultipartSink),
}

/// Per-sink tee state. See [`OutputTarget::sink`] for construction.
///
/// Does not buffer written bytes. The local file IS the buffer: after
/// [`OutputSink::finish`] flushes it, [`Self::finalize`] hands the
/// just-flushed file path to [`SnapshotMirror::submit_from_path`],
/// which streams chunks from disk to the upload backend bounded by
/// `part_size × inflight_parts` rather than file size. `bytes_written`
/// is tracked because the manifest needs the size — we record it from
/// the running counter rather than re-stating the file (the running
/// counter is already exact and avoids a syscall).
pub struct LocalSinkTee {
    rel_path: String,
    /// Absolute path to the local file. The mirror submit reads from
    /// here after the file is flushed.
    local_path: PathBuf,
    bytes_written: u64,
    mirror: Option<Arc<SnapshotMirror>>,
    manifest: Arc<ManifestAccumulator>,
}

impl LocalSinkTee {
    fn record_write(&mut self, data: &[u8]) {
        self.bytes_written += data.len() as u64;
    }

    /// Called after the local file has been flushed: records the manifest
    /// entry and (if mirror) hands the file off for streaming upload.
    /// Submit failures bubble up so ingest fails the activity rather than
    /// silently producing a half-mirrored snapshot.
    fn finalize(self) -> Result<()> {
        let LocalSinkTee {
            rel_path,
            local_path,
            bytes_written,
            mirror,
            manifest,
        } = self;
        manifest.record(&rel_path, bytes_written);
        if let Some(m) = mirror {
            m.submit_from_path(&rel_path, &local_path)
                .with_context(|| format!("dataplane mirror submit {}", rel_path))?;
        }
        Ok(())
    }
}

impl OutputSink {
    pub fn finish(self) -> io::Result<()> {
        match self {
            OutputSink::Local { mut file, tee } => {
                file.flush()?;
                if let Some(tee) = tee {
                    tee.finalize().map_err(io::Error::other)?;
                }
                Ok(())
            }
            OutputSink::Multipart(sink) => sink.finish(),
        }
    }
}

impl Write for OutputSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            OutputSink::Local { file, tee } => {
                let n = file.write(data)?;
                if let Some(tee) = tee {
                    // Tee tracks exactly the bytes accepted by the file —
                    // partial-write semantics propagate to the buffer.
                    tee.record_write(&data[..n]);
                }
                Ok(n)
            }
            OutputSink::Multipart(sink) => sink.write(data),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputSink::Local { file, .. } => file.flush(),
            OutputSink::Multipart(sink) => sink.flush(),
        }
    }
}

pub struct MultipartSink {
    writer: Option<WriteMultipart>,
    rt: Arc<Runtime>,
    max_concurrency: usize,
    handle: tokio::runtime::Handle,
}

impl MultipartSink {
    fn finish(mut self) -> io::Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        let mut writer = match self.writer.take() {
            Some(writer) => writer,
            None => return Ok(()),
        };
        let rt = self.rt.clone();
        Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        self.rt.block_on(writer.finish()).map_err(to_io_err)?;
        Ok(())
    }

    fn retry_with_backoff<T>(
        mut op: impl FnMut() -> Result<T, object_store::Error>,
    ) -> io::Result<T> {
        const RETRIES: usize = 3;
        const BASE_DELAY_MS: u64 = 250;
        let mut last_err: Option<object_store::Error> = None;
        for attempt in 0..=RETRIES {
            match op() {
                Ok(val) => return Ok(val),
                Err(err) => {
                    last_err = Some(err);
                    if attempt == RETRIES {
                        break;
                    }
                    let delay = BASE_DELAY_MS.saturating_mul(1 << attempt);
                    std::thread::sleep(Duration::from_millis(delay));
                }
            }
        }
        if let Some(err) = last_err {
            Err(to_io_err(err))
        } else {
            Err(io::Error::other("unknown multipart error"))
        }
    }
}

impl Write for MultipartSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            let max_concurrency = self.max_concurrency;
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(max_concurrency)))?;
            let _guard = self.handle.enter();
            writer.write(data);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        }
        Ok(())
    }
}

impl Drop for MultipartSink {
    fn drop(&mut self) {
        if let Err(err) = self.finish_inner() {
            eprintln!("MultipartSink finish error on drop: {err}");
        }
    }
}

fn to_io_err(err: object_store::Error) -> io::Error {
    io::Error::other(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskgraph_fs::CloudTransfer;

    #[test]
    fn source_authority_overrides_worker_routing_defaults() {
        assert_eq!(
            source_s3_settings("legacy-bucket", None, Some("us-east-1".to_string()),).unwrap(),
            (Some("us-east-1".to_string()), None, true),
            "legacy sources must retain worker routing"
        );
        let s3 = ObjectStoreAuthority::S3 {
            region: "us-west-2".to_string(),
        };
        assert_eq!(
            source_s3_settings("customer-bucket", Some(&s3), Some("us-east-1".to_string()),)
                .unwrap(),
            (Some("us-west-2".to_string()), Some(true), false),
        );
        assert_eq!(
            source_s3_settings("customer.bucket", Some(&s3), None,)
                .unwrap()
                .1,
            Some(false)
        );
        assert!(
            source_s3_settings("customer-bucket", Some(&ObjectStoreAuthority::Gcs), None,).is_err()
        );

        let adls = ObjectStoreAuthority::Adls {
            cloud: AzureCloud::Public,
        };
        assert_eq!(
            source_adls_settings(
                "customer",
                Some(&adls),
                Some("https://wrong.invalid".to_string()),
            )
            .unwrap(),
            (
                Some("https://customer.blob.core.windows.net".to_string()),
                Some("https://login.microsoftonline.com".to_string()),
            )
        );
        assert_eq!(
            source_adls_settings(
                "customer",
                None,
                Some("https://worker.invalid".to_string()),
            )
            .unwrap(),
            (Some("https://worker.invalid".to_string()), None),
            "legacy sources must retain worker routing"
        );
    }

    /// Locked-down dataplane scheme matrix: anything that isn't `s3://`,
    /// `gs://`, `az://`, or a bare local path must be rejected at the
    /// boundary with a message that names the scheme. Pre-this-check,
    /// `dbfs://...` would silently fall through to the local-fs branch
    /// and `create_dir_all("dbfs:")` — silently writing the snapshot to
    /// a literal `dbfs:` directory in the cwd. Any future regression
    /// along that fault line trips this test immediately.
    ///
    /// Note: `/Volumes/...` (Databricks Unity Catalog volumes) is a
    /// legitimate POSIX absolute path shape and we cannot reliably
    /// disambiguate it from a real local mount without runtime context.
    /// Treated as local on purpose; documented in
    /// `build_object_store_accepts_legitimate_local_paths`.
    #[test]
    fn build_object_store_rejects_unsupported_schemes() {
        let cases: &[(&str, &str)] = &[
            ("dbfs://path/to/snapshot", "DBFS"),
            ("dbfs:/path/to/snapshot", "DBFS"),
            ("DBFS://uppercase", "DBFS"),
            ("snowflake://stage/prefix", "Snowflake"),
            ("@db.schema.stage/path", "Snowflake"),
            ("file:///abs/path", "file://"),
            ("hdfs://nn:8020/path", "HDFS"),
            ("ftp://server/path", "URL scheme"),
        ];
        for (path, expected_substr) in cases {
            let err = build_object_store_for_path(path)
                .expect_err(&format!("expected rejection for {path}"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains(expected_substr),
                "rejection message for {path:?} should mention {expected_substr:?}; got: {msg}",
            );
        }
    }

    /// Companion to the rejection test: paths that DO look like
    /// legitimate local filesystem destinations must continue to work,
    /// including ones that incidentally have suspicious names. Uses
    /// `tempdir` so we don't pollute cwd.
    #[test]
    fn build_object_store_accepts_legitimate_local_paths() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        // Plain absolute path
        let abs = tmp.path().join("plain");
        let _ = build_object_store_for_path(abs.to_str().unwrap())?;
        // Path with a `:` that ISN'T a scheme separator (no `://`)
        let weird = tmp.path().join("name_with:colon");
        let _ = build_object_store_for_path(weird.to_str().unwrap())?;
        // POSIX `/Volumes/...` is technically also a UC Volume path on
        // Databricks but we can't disambiguate from a real local mount;
        // treated as local on purpose. Document via test.
        let volumes_like = tmp.path().join("Volumes/catalog/schema/vol");
        let _ = build_object_store_for_path(volumes_like.to_str().unwrap())?;
        Ok(())
    }

    /// Round-trip an upload + download against a local-filesystem
    /// [`ObjectStore`]. Same `CloudTransfer` API path as the cloud
    /// branch, exercising that the dataplane mirror works uniformly
    /// across backends — singlehost / dev / CI runs against this path.
    #[test]
    fn local_object_store_round_trip() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let prefix = tmp.path().to_str().expect("tmp path utf8");
        let store = build_object_store_for_path(prefix)?;
        let dest_dir = object_path_for_path(prefix)?;
        assert_eq!(dest_dir.as_ref(), "", "local dest_dir should be empty");

        let transfer = CloudTransfer::new(store, CloudTransferConfig::from_env());
        let rel = ObjectPath::from("nested/file.bin");
        let payload: &[u8] = b"hello dataplane mirror";
        let mut buf = transfer.acquire_buffer();
        buf.write_all(payload)?;
        transfer
            .submit_upload(rel.clone(), buf.into_inner())?
            .wait()?;

        // Verify on-disk side: file landed at <prefix>/<rel>.
        let on_disk = tmp.path().join("nested").join("file.bin");
        assert_eq!(std::fs::read(&on_disk)?, payload);

        // And round-trips back through submit_download.
        let pooled = transfer.submit_download(rel)?.wait()?;
        assert_eq!(pooled.as_slice(), payload);
        Ok(())
    }

    #[test]
    fn local_object_store_creates_missing_prefix() -> Result<()> {
        // The store builder is responsible for `mkdir -p` on the prefix —
        // singlehost ingest may target a fresh dataplane dir on first run.
        let tmp = tempfile::tempdir()?;
        let nested = tmp.path().join("level-a").join("level-b");
        let prefix = nested.to_str().expect("tmp path utf8");
        assert!(!nested.exists(), "precondition: nested dir absent");
        let _store = build_object_store_for_path(prefix)?;
        assert!(nested.is_dir(), "builder must create the prefix dir");
        Ok(())
    }

    /// End-to-end integration of the dataplane mirror through
    /// [`OutputTarget`]: every output `write_bytes` and every tee'd
    /// sink produces both a local file and an upload, end-of-ingest
    /// finalize writes `_MANIFEST.json` + `_SUCCESS` last on both
    /// sides, and `verify_snapshot_integrity` passes against the
    /// resulting local snapshot.
    #[test]
    fn output_target_mirror_round_trip() -> Result<()> {
        use crate::snapshot_mirror::{
            finalize_snapshot, verify_snapshot_integrity, MANIFEST_FILENAME,
        };
        use crate::success_marker::SUCCESS_MARKER_FILENAME;
        use std::io::Write as _;

        let local_root = tempfile::tempdir()?;
        let dataplane_root = tempfile::tempdir()?;
        let target = std::sync::Arc::new(OutputTarget::parse_with_dataplane(
            local_root.path().to_str().unwrap(),
            Dest::Local,
            Some(dataplane_root.path().to_str().unwrap()),
        )?);

        // Simulate the two write paths ingest uses: small JSON via
        // write_bytes, large streaming via sink.
        target.write_bytes("ingest_stats.json", b"{\"stat\":1}".to_vec())?;
        target.write_bytes(
            "kumo_col_stats/col_stats_full/customers.json",
            b"{\"col\":\"...\"}".to_vec(),
        )?;
        let sink_rel = "binary_index.bin";
        let sink_path = target.join(sink_rel);
        if let Some(parent) = sink_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut sink = target.sink(&sink_path)?;
        sink.write_all(&[7u8; 1024])?;
        sink.write_all(&[8u8; 256])?;
        sink.finish()?;

        // Finalize: drain pending uploads, write manifest + marker.
        let manifest = target.manifest().expect("local target carries manifest");
        let mirror = target.mirror();
        let result = finalize_snapshot(
            local_root.path(),
            mirror.map(|m| m.as_ref()),
            manifest,
            "{\"ok\":true}",
        )?;

        // Manifest sees all three writes (sink contributes one entry of
        // size 1024+256).
        assert_eq!(result.manifest.files.len(), 3);
        let by_path: std::collections::HashMap<_, _> = result
            .manifest
            .files
            .iter()
            .map(|e| (e.rel_path.clone(), e.size_bytes))
            .collect();
        assert_eq!(by_path.get("ingest_stats.json").copied(), Some(10));
        assert_eq!(by_path.get(sink_rel).copied(), Some(1280));

        // Dataplane mirror has every file, manifest, and marker.
        for rel in [
            "ingest_stats.json",
            "kumo_col_stats/col_stats_full/customers.json",
            sink_rel,
            MANIFEST_FILENAME,
            SUCCESS_MARKER_FILENAME,
        ] {
            let dp = dataplane_root.path().join(rel);
            assert!(dp.exists(), "{} missing on dataplane", rel);
        }

        // Local snapshot passes integrity verification.
        verify_snapshot_integrity(local_root.path())
            .map_err(|e| anyhow::anyhow!("verify failed: {e}"))?;
        target.invalidate_publication()?;
        assert!(!local_root.path().join(SUCCESS_MARKER_FILENAME).exists());
        assert!(!dataplane_root.path().join(SUCCESS_MARKER_FILENAME).exists());
        assert!(local_root.path().join(sink_rel).exists());
        assert!(dataplane_root.path().join(sink_rel).exists());
        target.invalidate_publication()?;
        Ok(())
    }

    #[test]
    fn temp_path_filter_recognizes_temp_tree() {
        // `_tmp` itself and anything under it.
        assert!(is_temp_rel_path("_tmp"));
        assert!(is_temp_rel_path("_tmp/foo.bin"));
        assert!(is_temp_rel_path("_tmp/nodes/users/shard-00000/pkey.bin"));
        // Look-alikes must NOT be filtered.
        assert!(!is_temp_rel_path("_tmpfile.json"));
        assert!(!is_temp_rel_path("not_tmp/foo"));
        assert!(!is_temp_rel_path("nodes/_tmp_in_name.bin"));
        assert!(!is_temp_rel_path(""));
        assert!(!is_temp_rel_path("ingest_stats.json"));
    }

    /// Mirror integration regression: `_tmp/*` writes flow through
    /// `OutputTarget` (because `target.write_bytes` and `target.sink` are
    /// the only chokepoints) but must not be recorded in the manifest or
    /// uploaded — the temp tree is cleaned up by `cleanup_temp` before
    /// ingest returns. Locks the filter on both code paths.
    #[test]
    fn output_target_skips_temp_paths_from_manifest_and_mirror() -> Result<()> {
        use std::io::Write as _;

        let local_root = tempfile::tempdir()?;
        let dataplane_root = tempfile::tempdir()?;
        let target = std::sync::Arc::new(OutputTarget::parse_with_dataplane(
            local_root.path().to_str().unwrap(),
            Dest::Local,
            Some(dataplane_root.path().to_str().unwrap()),
        )?);

        // write_bytes path: temp + non-temp.
        target.write_bytes("_tmp/scratch.json", b"abc".to_vec())?;
        target.write_bytes("col_stats.json", b"{}".to_vec())?;
        // sink path: temp + non-temp.
        for rel in ["_tmp/nodes/users/pkey.bin", "nodes/users/real.bin"] {
            let path = target.join(rel);
            if let Some(p) = path.parent() {
                fs::create_dir_all(p)?;
            }
            let mut sink = target.sink(&path)?;
            sink.write_all(&[7u8; 64])?;
            sink.finish()?;
        }

        // Manifest captured non-temp only.
        let manifest_paths: std::collections::BTreeSet<String> = target
            .manifest()
            .unwrap()
            .take()
            .into_iter()
            .map(|e| e.rel_path)
            .collect();
        assert!(manifest_paths.contains("col_stats.json"));
        assert!(manifest_paths.contains("nodes/users/real.bin"));
        assert!(!manifest_paths.iter().any(|p| p.starts_with("_tmp")));

        // Mirror has the non-temp uploads waiting; submit a sentinel and
        // drain them all to confirm no temp upload is in flight.
        if let Some(m) = target.mirror() {
            m.wait_all_pending()?;
        }
        // Files on dataplane: only non-temp.
        assert!(dataplane_root.path().join("col_stats.json").exists());
        assert!(dataplane_root.path().join("nodes/users/real.bin").exists());
        assert!(!dataplane_root.path().join("_tmp").exists());
        assert!(!dataplane_root
            .path()
            .join("_tmp/nodes/users/pkey.bin")
            .exists());

        // Local always has every write — temp or not — because temp files
        // are real ingest intermediates that get cleaned up later.
        assert!(local_root.path().join("_tmp/scratch.json").exists());
        assert!(local_root.path().join("_tmp/nodes/users/pkey.bin").exists());
        assert!(local_root.path().join("col_stats.json").exists());
        assert!(local_root.path().join("nodes/users/real.bin").exists());
        Ok(())
    }

    /// Without a mirror, `OutputTarget` still records the manifest so
    /// `verify_snapshot_integrity` can run on local-only ingests.
    #[test]
    fn output_target_local_only_finalize_passes_verify() -> Result<()> {
        use crate::snapshot_mirror::{finalize_snapshot, verify_snapshot_integrity};

        let local_root = tempfile::tempdir()?;
        let target = std::sync::Arc::new(OutputTarget::parse(
            local_root.path().to_str().unwrap(),
            Dest::Local,
        )?);
        target.write_bytes("a.json", b"hello".to_vec())?;
        let manifest = target.manifest().unwrap();
        finalize_snapshot(local_root.path(), None, manifest, "{}")?;
        verify_snapshot_integrity(local_root.path())
            .map_err(|e| anyhow::anyhow!("verify failed: {e}"))?;
        Ok(())
    }

    /// Regression: deeply-nested sink paths (the shape ingest uses for
    /// every per-shard output, e.g.
    /// `nodes/<table>/pk_hash-NNNNN.bin`) must produce a complete local
    /// file AND a manifest entry — not just the dataplane upload. This
    /// is the post-condition Block E hit when the local snapshot was
    /// missing 31 `nodes/*` files after a "successful" atomic-marker
    /// ingest. Asserts both the local bytes are durable AND
    /// `verify_snapshot_integrity` accepts the local root.
    #[test]
    fn output_target_nested_sink_lands_locally_with_mirror() -> Result<()> {
        use crate::snapshot_mirror::{finalize_snapshot, verify_snapshot_integrity};
        use std::io::Write as _;

        let local_root = tempfile::tempdir()?;
        let dataplane_root = tempfile::tempdir()?;
        let target = std::sync::Arc::new(OutputTarget::parse_with_dataplane(
            local_root.path().to_str().unwrap(),
            Dest::Local,
            Some(dataplane_root.path().to_str().unwrap()),
        )?);

        // Mirrors the shape `write_pk_hash_and_data` produces.
        let nested_rels = [
            "nodes/customers/pk_hash-00000.bin",
            "nodes/customers/pk_hash-00000.meta.json",
            "nodes/customers/pk_data-00000.bin",
            "nodes/customers/node_data.ctx",
            "nodes/orders/pk_hash-00001.bin",
        ];
        let payload: Vec<u8> = (0u8..=200).cycle().take(2048).collect();
        for rel in &nested_rels {
            let path = target.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            // Every code path a real ingest uses goes through `target.sink`
            // (via BufWriter) for the binary files and `target.write_bytes`
            // for the small JSON ones. Cover both shapes here.
            if rel.ends_with(".meta.json") {
                target.write_bytes(rel, payload.clone())?;
            } else {
                let mut sink = target.sink(&path)?;
                sink.write_all(&payload)?;
                sink.finish()?;
            }
        }

        // Finalize, then verify.
        let manifest = target.manifest().expect("local target carries manifest");
        let mirror = target.mirror();
        finalize_snapshot(
            local_root.path(),
            mirror.map(|m| m.as_ref()),
            manifest,
            "{\"ok\":true}",
        )?;

        // Every nested file must be present LOCALLY (the bug: they were
        // only on the dataplane) AND have the expected size.
        for rel in &nested_rels {
            let local = local_root.path().join(rel);
            let dp = dataplane_root.path().join(rel);
            let local_meta = fs::metadata(&local)
                .with_context(|| format!("local file missing: {}", local.display()))?;
            let dp_meta = fs::metadata(&dp)
                .with_context(|| format!("dataplane file missing: {}", dp.display()))?;
            assert_eq!(
                local_meta.len(),
                payload.len() as u64,
                "local file size wrong for {rel}"
            );
            assert_eq!(
                dp_meta.len(),
                payload.len() as u64,
                "dataplane file size wrong for {rel}"
            );
        }

        // The end-to-end integrity invariant: caller can verify the
        // local root immediately, with no extra copy from the dataplane.
        verify_snapshot_integrity(local_root.path())
            .map_err(|e| anyhow::anyhow!("verify against local root failed: {e}"))?;
        // Same invariant holds on the dataplane side independently.
        verify_snapshot_integrity(dataplane_root.path())
            .map_err(|e| anyhow::anyhow!("verify against dataplane failed: {e}"))?;
        Ok(())
    }

    use tempfile::tempdir;

    fn touch(dir: &std::path::Path, name: &str) {
        fs::write(dir.join(name), b"").unwrap();
    }

    fn paths_from(source: (DataSource, u64)) -> Vec<String> {
        match source.0 {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => paths,
            _ => vec![],
        }
    }

    #[test]
    fn test_expand_local_parquet_dir() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "a.parquet");
        let partition = dir.path().join("year=2025");
        fs::create_dir_all(&partition).unwrap();
        touch(&partition, "b.parquet");
        touch(dir.path(), "ignore.txt");
        let result = expand_local_data_paths(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(result.0, DataSource::Parquet { .. }));
        let files = paths_from(result);
        assert_eq!(files.len(), 2);
        assert!(files[0].ends_with("a.parquet"));
        assert!(files[1].ends_with("b.parquet"));
    }

    #[test]
    fn test_expand_local_csv_dir() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "b.csv");
        touch(dir.path(), "a.csv");
        let result = expand_local_data_paths(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(result.0, DataSource::Csv { .. }));
        let files = paths_from(result);
        assert_eq!(files.len(), 2);
        assert!(files[0].ends_with("a.csv"));
        assert!(files[1].ends_with("b.csv"));
    }

    #[test]
    fn test_expand_local_mixed_dir_errors() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "a.parquet");
        let partition = dir.path().join("year=2025");
        fs::create_dir(&partition).unwrap();
        touch(&partition, "b.csv");
        let err = expand_local_data_paths(dir.path().to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("both .parquet and .csv"));
    }

    #[test]
    fn test_expand_local_empty_dir() {
        let dir = tempdir().unwrap();
        let result = expand_local_data_paths(dir.path().to_str().unwrap()).unwrap();
        assert!(paths_from(result).is_empty());
    }

    #[test]
    fn test_expand_local_concrete_file_passthrough() {
        let dir = tempdir().unwrap();
        touch(dir.path(), "data.parquet");
        let file_path = dir
            .path()
            .join("data.parquet")
            .to_string_lossy()
            .into_owned();
        let result = expand_local_data_paths(&file_path).unwrap();
        assert!(matches!(result.0, DataSource::Parquet { .. }));
        assert_eq!(paths_from(result), vec![file_path]);
    }
}
