use anyhow::{anyhow, bail, Context, Result};
use object_store::path::Path as ObjectPath;
use object_store::ObjectMeta;
use diskgraph_fs::{build_s3_client_async, CloudTransfer, CloudTransferConfig, S3Config};
use futures_util::stream::{self, StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::VecDeque;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use url::Url;

const DEFAULT_BUCKET: &str = "kumo-prod-dataplane-bucket";
const DOORDASH_BUCKET: &str = "kumo-doordash-cache";

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub out_root: PathBuf,
    pub dry_run: bool,
    pub clean: bool,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            out_root: PathBuf::from("/mnt/localssd/datasets"),
            dry_run: false,
            clean: true,
        }
    }
}

pub async fn download_job(job_spec: &str, options: DownloadOptions) -> Result<PathBuf> {
    let (customer_name, job_id) = parse_job_spec(job_spec)?;
    validate_token("customer_name", &customer_name)?;
    validate_token("job_id", &job_id)?;

    let out_dir = options
        .out_root
        .join(format!("{}_{}", customer_name, job_id));

    if !options.clean && out_dir.exists() {
        println!("Using existing data at {} (skipping S3 check)", out_dir.display());
        return Ok(out_dir);
    }

    let bucket = bucket_for_customer(&customer_name);
    let customer_path = customer_for_paths(&customer_name);
    let s3 = build_s3_client_async(S3Config {
        bucket: bucket.to_string(),
        region: None,
        endpoint: None,
        allow_http: None,
        virtual_hosted: None,
        profile: std::env::var("S3_PROFILE").ok(),
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
    })
    .await?;
    let transfer_config = CloudTransferConfig::from_env();
    let transfer = CloudTransfer::new(s3.store, transfer_config.clone());

    let graphs_prefix_key = format!("{}/prod/jobs/{}/graphs/", customer_path, job_id);
    let graphs_prefix_s3 = format!("s3://{}/{}", bucket, graphs_prefix_key);
    let progress = ProgressTracker::new();
    let progress_task = spawn_progress(progress.clone());
    progress.log("Listing graphs".to_string());

    let mut graph_prefixes = list_prefixes(&transfer, &graphs_prefix_key)
        .await
        .context("list graphs")?;
    if graph_prefixes.is_empty() {
        bail!("no entries found under {}", graphs_prefix_s3);
    }

    let mut table_prefixes = Vec::new();
    let mut graphs_prefix_key = graphs_prefix_key;
    let first_prefix = graph_prefixes.remove(0);
    if attributes_exist(&transfer, &(graphs_prefix_key.clone() + &first_prefix)).await? {
        table_prefixes.push(first_prefix);
        table_prefixes.extend(graph_prefixes);
    } else {
        if !graph_prefixes.is_empty() {
            bail!(
                "multiple graph directories under {}; please use a specific job",
                graphs_prefix_s3
            );
        }
        graphs_prefix_key = format!("{}{}", graphs_prefix_key, first_prefix);
        let graph_s3 = format!("s3://{}/{}", bucket, graphs_prefix_key);
        progress.log("Listing tables".to_string());
        table_prefixes = list_prefixes(&transfer, &graphs_prefix_key)
            .await
            .context("list tables")?;
        if table_prefixes.is_empty() {
            bail!("no table directories found under {}", graph_s3);
        }
    }

    let out_dir = options
        .out_root
        .join(format!("{}_{}", customer_name, job_id));
    prepare_output_dir(&out_dir, &options.out_root, options.clean).await?;

    // Attempt to download config.json if it exists at graphs_prefix_key
    let config_key = format!("{}config.json", graphs_prefix_key);
    match transfer
        .get_bytes(&ObjectPath::from(config_key.as_str()))
        .await
    {
        Ok(bytes) => {
            progress.log("Downloading config.json".to_string());
            let config_out = out_dir.join("config.json");
            let mut file = tokio::fs::File::create(config_out).await?;
            file.write_all(bytes.as_ref()).await?;
            file.flush().await?;
        }
        Err(err) => {
            if !is_not_found(&err) {
                eprintln!("Warning: failed to check/download config.json: {:?}", err);
            }
        }
    }

    let mut download_items = Vec::new();
    for table_prefix in table_prefixes {
        let attrs_key = format!("{}{}attributes.json", graphs_prefix_key, table_prefix);
        let attrs_s3 = format!("s3://{}/{}", bucket, attrs_key);
        progress.log("Reading metadata".to_string());
        let attrs = get_object_string(&transfer, &attrs_key).await?;
        let table_meta = parse_table_metadata(&attrs)
            .with_context(|| format!("parse metadata from {}", attrs_s3))?;
        validate_table_name(&table_meta.table_name)?;

        let raw_prefix_key = format!(
            "{}/prod/data-persist-dir/ingest/fpv3.{}.{}/{}/",
            customer_path, table_meta.data_source, table_meta.source_name, table_meta.data_version
        );
        let raw_prefix_s3 = format!("s3://{}/{}", bucket, raw_prefix_key);

        let parquet_objects = list_parquet_objects(&transfer, &raw_prefix_key).await?;
        if parquet_objects.is_empty() {
            bail!("raw ingest data not found (likely GC'd): {}", raw_prefix_s3);
        }

        let table_out = out_dir.join(&table_meta.table_name);
        tokio::fs::create_dir_all(&table_out).await?;

        // Save attributes.json for validation/metadata purposes
        let attributes_out = table_out.join("attributes.json");
        tokio::fs::write(&attributes_out, &attrs).await?;

        for obj in parquet_objects {
            let key = obj.location.to_string();
            let size = obj.size as u64;
            let relative = key
                .strip_prefix(&raw_prefix_key)
                .unwrap_or(&key)
                .trim_start_matches('/');
            let out_path = table_out.join(relative);
            download_items.push(DownloadItem {
                key,
                size,
                out_path,
            });
        }
    }

    progress.set_totals(
        download_items.len() as u64,
        download_items.iter().map(|item| item.size).sum(),
    );
    progress.log("Downloading".to_string());

    if options.dry_run {
        progress.finish();
        let _ = progress_task.await;
        return Ok(out_dir);
    }

    let transfer_ref = transfer.clone();
    let max_requests = std::thread::available_parallelism()
        .map(|v| v.get() * 2)
        .unwrap_or(2);

    stream::iter(download_items)
        .map(Ok::<_, anyhow::Error>)
        .try_for_each_concurrent(max_requests, |item| {
            let transfer = transfer_ref.clone();
            let progress = progress.clone();
            async move {
                download_path(
                    item,
                    transfer,
                    &progress,
                )
                .await
            }
        })
        .await?;

    progress.finish();
    let _ = progress_task.await;
    if !options.dry_run {
        println!("Next steps:");
        println!(
            "  ./diskgraph-infer/scripts/infer_config.sh \"{}\"",
            out_dir.display()
        );
    }
    Ok(out_dir)
}

pub fn parse_job_spec(input: &str) -> Result<(String, String)> {
    if input.starts_with("http") || input.starts_with("https") {
        parse_job_url(input)
    } else {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() == 2 {
            Ok((parts[0].to_string(), parts[1].to_string()))
        } else {
            Err(anyhow!("invalid job spec '{}': expected 'customer job_id' or url", input))
        }
    }
}

fn parse_job_url(input: &str) -> Result<(String, String)> {
    let url = Url::parse(input).context("invalid job_url")?;
    let host = url.host_str().unwrap_or_default();
    let customer = host.split('.').next().unwrap_or_default().to_string();
    if customer.is_empty() {
        bail!("unable to parse customer from job_url");
    }
    let path = url.path();
    let job_id = extract_job_id(path)
        .ok_or_else(|| anyhow!("unable to parse job id from job_url"))?;
    Ok((customer, job_id))
}

fn bucket_for_customer(customer: &str) -> &'static str {
    if customer.eq_ignore_ascii_case("doordash") {
        DOORDASH_BUCKET
    } else {
        DEFAULT_BUCKET
    }
}

fn customer_for_paths(customer: &str) -> String {
    if customer.eq_ignore_ascii_case("doordash") {
        "kumo".to_string()
    } else {
        customer.to_string()
    }
}

fn extract_job_id(path: &str) -> Option<String> {
    for needle in ["trainingjob-", "bp-job-"] {
        if let Some(idx) = path.find(needle) {
            let rest = &path[idx..];
            let end = rest
                .find('/')
                .or_else(|| rest.find('?'))
                .unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }
    }
    None
}

async fn list_prefixes(transfer: &CloudTransfer, base_prefix: &str) -> Result<Vec<String>> {
    let prefix = ObjectPath::from(base_prefix);
    let mut prefixes = Vec::new();
    for prefix in transfer.list_prefixes(&prefix).await? {
        let full = prefix.to_string();
        let rel = full.strip_prefix(base_prefix).unwrap_or(&full);
        let mut rel = rel.to_string();
        if !rel.is_empty() && !rel.ends_with('/') {
            rel.push('/');
        }
        prefixes.push(rel);
    }
    Ok(prefixes)
}

async fn attributes_exist(transfer: &CloudTransfer, prefix: &str) -> Result<bool> {
    let key = format!("{}attributes.json", prefix);
    match transfer.head(&ObjectPath::from(key)).await {
        Ok(_) => Ok(true),
        Err(err) => {
            if is_not_found(&err) {
                Ok(false)
            } else {
                Err(err)
            }
        }
    }
}

async fn get_object_string(transfer: &CloudTransfer, key: &str) -> Result<String> {
    transfer.get_string(&ObjectPath::from(key)).await
}

#[derive(Debug)]
struct TableMetadata {
    table_name: String,
    source_name: String,
    data_source: String,
    data_version: String,
}

fn parse_table_metadata(input: &str) -> Result<TableMetadata> {
    let value: serde_json::Value = serde_json::from_str(input)?;
    let meta = value
        .get("table_metadata_config")
        .ok_or_else(|| anyhow!("missing table_metadata_config"))?;
    Ok(TableMetadata {
        table_name: get_string(meta, "table_name")?,
        source_name: get_string(meta, "source_name")?,
        data_source: get_string(meta, "data_source")?,
        data_version: get_string(meta, "data_version")?,
    })
}

fn get_string(map: &serde_json::Value, key: &str) -> Result<String> {
    let value = map
        .get(key)
        .ok_or_else(|| anyhow!("missing {}", key))?;
    match value {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        _ => Err(anyhow!("invalid {} type", key)),
    }
}

fn validate_table_name(name: &str) -> Result<()> {
    if name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(anyhow!("unsupported table_name '{}'", name))
    }
}

fn validate_token(label: &str, value: &str) -> Result<()> {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
    {
        Ok(())
    } else {
        Err(anyhow!("invalid {} '{}'", label, value))
    }
}

fn is_not_found(err: &anyhow::Error) -> bool {
    err.downcast_ref::<object_store::Error>()
        .map(|inner| matches!(inner, object_store::Error::NotFound { .. }))
        .unwrap_or(false)
}

async fn list_parquet_objects(
    transfer: &CloudTransfer,
    prefix: &str,
) -> Result<Vec<ObjectMeta>> {
    let mut objects = Vec::new();
    let prefix = ObjectPath::from(prefix);
    for obj in transfer.list_objects(&prefix).await? {
        if obj.location.as_ref().ends_with(".parquet") {
            objects.push(obj);
        }
    }
    Ok(objects)
}

struct DownloadItem {
    key: String,
    size: u64,
    out_path: PathBuf,
}

async fn download_path(
    item: DownloadItem,
    transfer: CloudTransfer,
    progress: &std::sync::Arc<ProgressTracker>,
) -> Result<()> {
    let key = item.key;
    let size = item.size;
    let out_path = item.out_path;
    if let Some(parent) = out_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if let Ok(meta) = tokio::fs::metadata(&out_path).await {
        if meta.len() == size {
            progress.add_bytes(size);
            progress.add_file_done();
            return Ok(())
        }
    }

    let path = ObjectPath::from(key.clone());
    let buffer = tokio::task::spawn_blocking(move || transfer.submit_download(path)?.wait())
        .await
        .map_err(|err| anyhow!("download task panicked: {}", err))??;
    let mut file = tokio::fs::File::create(&out_path).await?;
    file.write_all(buffer.as_slice()).await?;
    file.flush().await?;
    progress.add_bytes(size);
    progress.add_file_done();
    Ok(())
}

async fn prepare_output_dir(out_dir: &Path, out_root: &Path, clean: bool) -> Result<()> {
    if !out_dir.starts_with(out_root) {
        bail!("refusing to clear output outside of {}", out_root.display());
    }
    if out_dir == out_root {
        bail!("refusing to clear output root {}", out_root.display());
    }
    if let Some(name) = out_dir.file_name() {
        if name.is_empty() {
            bail!("refusing to clear output dir with empty name");
        }
    } else {
        bail!("refusing to clear output dir without name");
    }

    if clean {
        match tokio::fs::metadata(out_dir).await {
            Ok(meta) => {
                if meta.is_dir() {
                    tokio::fs::remove_dir_all(out_dir).await?;
                } else {
                    bail!(
                        "output path exists and is not a directory: {}",
                        out_dir.display()
                    );
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {} // Ignore not found error
            Err(err) => return Err(err.into()),
        }
    }

    tokio::fs::create_dir_all(out_dir).await?;
    Ok(())
}

struct ProgressTracker {
    bytes_done: std::sync::atomic::AtomicU64,
    bytes_total: std::sync::atomic::AtomicU64,
    files_done: std::sync::atomic::AtomicU64,
    files_total: std::sync::atomic::AtomicU64,
    status: std::sync::Mutex<String>,
    done: std::sync::atomic::AtomicBool,
    bar: Option<ProgressBar>,
}

impl ProgressTracker {
    fn new() -> std::sync::Arc<Self> {
        let bar = if std::io::stderr().is_terminal() {
            let pb = ProgressBar::new(0);
            pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(4));
            pb.set_style(spinner_style());
            pb.enable_steady_tick(std::time::Duration::from_millis(120));
            Some(pb)
        } else {
            None
        };
        std::sync::Arc::new(Self {
            bytes_done: std::sync::atomic::AtomicU64::new(0),
            bytes_total: std::sync::atomic::AtomicU64::new(0),
            files_done: std::sync::atomic::AtomicU64::new(0),
            files_total: std::sync::atomic::AtomicU64::new(0),
            status: std::sync::Mutex::new(String::new()),
            done: std::sync::atomic::AtomicBool::new(false),
            bar,
        })
    }

    fn set_totals(&self, files: u64, bytes: u64) {
        self.bytes_total
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
        self.files_total
            .store(files, std::sync::atomic::Ordering::Relaxed);
    }

    fn add_bytes(&self, bytes: u64) {
        self.bytes_done
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    fn add_file_done(&self) {
        self.files_done
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn finish(&self) {
        let total_bytes = self.bytes_total.load(std::sync::atomic::Ordering::Relaxed);
        let total_files = self.files_total.load(std::sync::atomic::Ordering::Relaxed);
        self.bytes_done
            .store(total_bytes, std::sync::atomic::Ordering::Relaxed);
        self.files_done
            .store(total_files, std::sync::atomic::Ordering::Relaxed);
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn log(&self, message: String) {
        if let Ok(mut guard) = self.status.lock() {
            *guard = message;
        }
    }
}

fn spawn_progress(progress: std::sync::Arc<ProgressTracker>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(pb) = progress.bar.clone() else {
            return;
        };
        let mut ticker = tokio::time::interval(std::time::Duration::from_millis(250));
        let mut last_tick = std::time::Instant::now();
        let mut is_spinner = true;
        let window = std::time::Duration::from_secs(5);
        let mut samples: VecDeque<(std::time::Instant, u64)> = VecDeque::new();
        loop {
            ticker.tick().await;
            let now = std::time::Instant::now();
            let bytes_done = progress
                .bytes_done
                .load(std::sync::atomic::Ordering::Relaxed);
            let bytes_total = progress
                .bytes_total
                .load(std::sync::atomic::Ordering::Relaxed);
            let files_done = progress
                .files_done
                .load(std::sync::atomic::Ordering::Relaxed);
            let files_total = progress
                .files_total
                .load(std::sync::atomic::Ordering::Relaxed);

            if bytes_total > 0 && is_spinner {
                pb.set_style(bar_style());
                pb.set_length(bytes_total);
                is_spinner = false;
            }
            if !is_spinner {
                pb.set_length(bytes_total);
                pb.set_position(bytes_done);
                pb.set_prefix(format!("{}/{}", files_done, files_total));
            }
            let elapsed = now.duration_since(last_tick);
            if elapsed.as_millis() > 0 {
                samples.push_back((now, bytes_done));
                while let Some((t, _)) = samples.front() {
                    if now.duration_since(*t) > window {
                        samples.pop_front();
                    } else {
                        break;
                    }
                }
                let mbps = if samples.len() >= 2 {
                    let (t0, b0) = samples.front().copied().unwrap();
                    let (t1, b1) = samples.back().copied().unwrap();
                    let secs = (t1 - t0).as_secs_f64();
                    if secs > 0.0 {
                        ((b1.saturating_sub(b0)) as f64 / secs) / (1024.0 * 1024.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let status = progress
                    .status
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_default();
                let status = truncate_status(&status, 60);
                if status.is_empty() {
                    pb.set_message(format!("{:.1} MB/s", mbps));
                } else {
                    pb.set_message(format!("{} | {:.1} MB/s", status, mbps));
                }
                last_tick = now;
            }

            if progress.done.load(std::sync::atomic::Ordering::Relaxed)
                && bytes_done >= bytes_total
                && files_done >= files_total
            {
                pb.finish_and_clear();
                break;
            }
        }
    })
}

fn truncate_status(status: &str, max_len: usize) -> String {
    if status.len() <= max_len {
        return status.to_string();
    }
    let mut out = String::with_capacity(max_len);
    for (i, ch) in status.chars().enumerate() {
        if i + 1 >= max_len {
            break;
        }
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:30.cyan/blue} {bytes}/{total_bytes} {prefix} ({eta})")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .progress_chars("|/-")
}
