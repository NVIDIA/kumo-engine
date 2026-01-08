use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::types::Object;
use aws_sdk_s3::{Client, Config};
use clap::Parser;
use futures_util::stream::{self, StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;
use std::collections::VecDeque;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use url::Url;

const BUCKET: &str = "kumo-prod-dataplane-bucket";

#[derive(Parser, Debug)]
#[command(about = "Download ingested graph parquet files for a training or bp job.")]
struct Args {
    /// Inputs: <customer_name> <job_id> or <job_url>.
    #[arg(value_name = "INPUT")]
    inputs: Vec<String>,
    /// Output root directory.
    #[arg(long, default_value = "/mnt/localssd/datasets")]
    out_root: PathBuf,
    /// Max in-flight S3 requests (objects + parts).
    #[arg(long, default_value_t = 64)]
    max_requests: usize,
    /// Part size in MB for multipart downloads.
    #[arg(long, default_value_t = 64)]
    part_size_mb: u64,
    /// Multipart threshold in MB (objects smaller than this are single-part).
    #[arg(long, default_value_t = 128)]
    multipart_threshold_mb: u64,
    /// Max concurrent parts per object.
    #[arg(long, default_value_t = 8)]
    part_concurrency: usize,
    /// Skip downloading parquet files.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;

    let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let s3_config = Config::new(&config);
    let client = Client::from_conf(s3_config);

    let graphs_prefix_key = format!(
        "{}/prod/jobs/{}/graphs/",
        args.customer_name, args.job_id
    );
    let graphs_prefix_s3 = format!("s3://{}/{}", BUCKET, graphs_prefix_key);
    let progress = ProgressTracker::new();
    let progress_task = spawn_progress(progress.clone());
    progress.log("Listing graphs".to_string());

    let mut graph_prefixes = list_prefixes(&client, &graphs_prefix_key)
        .await
        .context("list graphs")?;
    if graph_prefixes.is_empty() {
        bail!("no entries found under {}", graphs_prefix_s3);
    }

    let mut table_prefixes = Vec::new();
    let mut graphs_prefix_key = graphs_prefix_key;
    let first_prefix = graph_prefixes.remove(0);
    if attributes_exist(&client, &(graphs_prefix_key.clone() + &first_prefix)).await? {
        table_prefixes.push(first_prefix);
        table_prefixes.extend(graph_prefixes);
    } else {
        if !graph_prefixes.is_empty() {
            bail!("multiple graph directories under {}; please use a specific job", graphs_prefix_s3);
        }
        graphs_prefix_key = format!("{}{}", graphs_prefix_key, first_prefix);
        let graph_s3 = format!("s3://{}/{}", BUCKET, graphs_prefix_key);
        progress.log("Listing tables".to_string());
        table_prefixes = list_prefixes(&client, &graphs_prefix_key)
            .await
            .context("list tables")?;
        if table_prefixes.is_empty() {
            bail!("no table directories found under {}", graph_s3);
        }
    }

    let out_dir = args
        .out_root
        .join(format!("{}_{}", args.customer_name, args.job_id));
    prepare_output_dir(&out_dir, &args.out_root).await?;

    let request_sema = std::sync::Arc::new(Semaphore::new(args.max_requests));
    let mut download_items = Vec::new();
    for table_prefix in table_prefixes {
        let attrs_key = format!("{}{}attributes.json", graphs_prefix_key, table_prefix);
        let attrs_s3 = format!("s3://{}/{}", BUCKET, attrs_key);
        progress.log("Reading metadata".to_string());
        let attrs = get_object_string(&client, &attrs_key).await?;
        let table_meta = parse_table_metadata(&attrs)
            .with_context(|| format!("parse metadata from {}", attrs_s3))?;
        validate_table_name(&table_meta.table_name)?;

        let raw_prefix_key = format!(
            "{}/prod/data-persist-dir/ingest/fpv3.{}.{}" ,
            args.customer_name, table_meta.data_source, table_meta.source_name
        );
        let raw_prefix_key = format!("{}/{}/", raw_prefix_key, table_meta.data_version);
        let raw_prefix_s3 = format!("s3://{}/{}", BUCKET, raw_prefix_key);

        let parquet_objects = list_parquet_objects(&client, &raw_prefix_key).await?;
        if parquet_objects.is_empty() {
            bail!("raw ingest data not found (likely GC'd): {}", raw_prefix_s3);
        }

        let table_out = out_dir.join(&table_meta.table_name);
        tokio::fs::create_dir_all(&table_out).await?;
        for obj in parquet_objects {
            let Some(key) = obj.key() else {
                continue;
            };
            let size = obj.size().unwrap_or(0) as u64;
            let relative = key
                .strip_prefix(&raw_prefix_key)
                .unwrap_or(key)
                .trim_start_matches('/');
            let out_path = table_out.join(relative);
            download_items.push(DownloadItem {
                key: key.to_string(),
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

    if args.dry_run {
        progress.finish();
        let _ = progress_task.await;
        println!("Done. Data at {}", out_dir.display());
        println!(
            "Next: ./diskgraph-infer/scripts/infer_source.sh {}",
            out_dir.display()
        );
        return Ok(());
    }

    let part_size = args.part_size_mb * 1024 * 1024;
    let threshold = args.multipart_threshold_mb * 1024 * 1024;
    let client_ref = client.clone();
    let request_sema = request_sema.clone();
    stream::iter(download_items)
        .map(Ok::<_, anyhow::Error>)
        .try_for_each_concurrent(args.max_requests, |item| {
            let client = client_ref.clone();
            let request_sema = request_sema.clone();
            let progress = progress.clone();
            async move {
                download_path(
                    &client,
                    item,
                    part_size,
                    threshold,
                    args.part_concurrency,
                    &request_sema,
                    &progress,
                )
                .await
            }
        })
        .await?;

    progress.finish();
    let _ = progress_task.await;
    println!("Done. Data at {}", out_dir.display());
    println!(
        "Next: ./diskgraph-infer/scripts/infer_source.sh {}",
        out_dir.display()
    );
    Ok(())
}

fn parse_args() -> Result<ResolvedArgs> {
    let raw = Args::parse();
    let (customer_name, job_id) = match raw.inputs.as_slice() {
        [job_url] => parse_job_url(job_url)?,
        [customer, job] => (customer.to_string(), job.to_string()),
        _ => {
            return Err(anyhow!(
                "usage: download-ingested-graph <customer_name> <job_id> | <job_url>"
            ))
        }
    };

    validate_token("customer_name", &customer_name)?;
    validate_token("job_id", &job_id)?;

    Ok(ResolvedArgs {
        customer_name,
        job_id,
        out_root: raw.out_root,
        max_requests: raw.max_requests,
        part_size_mb: raw.part_size_mb,
        multipart_threshold_mb: raw.multipart_threshold_mb,
        part_concurrency: raw.part_concurrency,
        dry_run: raw.dry_run,
    })
}

#[derive(Debug)]
struct ResolvedArgs {
    customer_name: String,
    job_id: String,
    out_root: PathBuf,
    max_requests: usize,
    part_size_mb: u64,
    multipart_threshold_mb: u64,
    part_concurrency: usize,
    dry_run: bool,
}

fn parse_job_url(input: &str) -> Result<(String, String)> {
    let url = Url::parse(input).context("invalid job_url")?;
    let host = url.host_str().unwrap_or_default();
    let customer = host.split('.').next().unwrap_or_default().to_string();
    if customer.is_empty() {
        bail!("unable to parse customer from job_url");
    }
    let path = url.path();
    let job_id = extract_job_id(path).ok_or_else(|| anyhow!("unable to parse job id from job_url"))?;
    Ok((customer, job_id))
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

async fn list_prefixes(client: &Client, base_prefix: &str) -> Result<Vec<String>> {
    let mut prefixes = Vec::new();
    let mut token = None;
    loop {
        let mut req = client
            .list_objects_v2()
            .bucket(BUCKET)
            .prefix(base_prefix)
            .delimiter("/");
        if let Some(t) = token {
            req = req.continuation_token(t);
        }
        let resp = req.send().await?;
        for p in resp.common_prefixes() {
            if let Some(prefix) = p.prefix() {
                let rel = prefix.strip_prefix(base_prefix).unwrap_or(prefix);
                prefixes.push(rel.to_string());
            }
        }
        token = resp.next_continuation_token().map(|s| s.to_string());
        if token.is_none() {
            break;
        }
    }
    Ok(prefixes)
}

async fn attributes_exist(client: &Client, prefix: &str) -> Result<bool> {
    let key = format!("{}attributes.json", prefix);
    let resp = client.head_object().bucket(BUCKET).key(key).send().await;
    match resp {
        Ok(_) => Ok(true),
        Err(aws_sdk_s3::error::SdkError::ServiceError(service_err)) => {
            if service_err.err().is_not_found() {
                Ok(false)
            } else {
                Err(anyhow!(service_err.err().to_string()))
            }
        }
        Err(err) => Err(err.into()),
    }
}

async fn get_object_string(client: &Client, key: &str) -> Result<String> {
    let resp = client
        .get_object()
        .bucket(BUCKET)
        .key(key)
        .send()
        .await?;
    let bytes = resp.body.collect().await?.into_bytes();
    Ok(String::from_utf8(bytes.to_vec())?)
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

async fn list_parquet_objects(client: &Client, prefix: &str) -> Result<Vec<Object>> {
    let mut objects = Vec::new();
    let mut token = None;
    loop {
        let mut req = client.list_objects_v2().bucket(BUCKET).prefix(prefix);
        if let Some(t) = token {
            req = req.continuation_token(t);
        }
        let resp = req.send().await?;
        for obj in resp.contents() {
            if let Some(key) = obj.key() {
                if key.ends_with(".parquet") {
                    objects.push(obj.clone());
                }
            }
        }
        token = resp.next_continuation_token().map(|s| s.to_string());
        if token.is_none() {
            break;
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
    client: &Client,
    item: DownloadItem,
    part_size: u64,
    threshold: u64,
    part_concurrency: usize,
    request_sema: &Semaphore,
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
            return Ok(());
        }
    }

    if size >= threshold {
        let parts = build_ranges(size, part_size);
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&out_path)
            .await?;
        file.set_len(size).await?;
        drop(file);

        let key_ref = key.as_str();
        stream::iter(parts)
            .map(Ok::<_, anyhow::Error>)
            .try_for_each_concurrent(part_concurrency, |(start, end)| {
                let client = client.clone();
                let out_path = out_path.clone();
                let progress = progress.clone();
                async move {
                    download_range(&client, key_ref, &out_path, start, end, request_sema, &progress)
                        .await
                }
            })
            .await?;
    } else {
        download_full(client, &key, &out_path, request_sema).await?;
        progress.add_bytes(size);
    }
    progress.add_file_done();
    Ok(())
}

async fn download_full(
    client: &Client,
    key: &str,
    out_path: &Path,
    request_sema: &Semaphore,
) -> Result<()> {
    let _permit = request_sema.acquire().await?;
    let resp = client.get_object().bucket(BUCKET).key(key).send().await?;
    let mut reader = resp.body.into_async_read();
    let mut file = tokio::fs::File::create(out_path).await?;
    tokio::io::copy(&mut reader, &mut file).await?;
    file.flush().await?;
    Ok(())
}

async fn download_range(
    client: &Client,
    key: &str,
    out_path: &Path,
    start: u64,
    end: u64,
    request_sema: &Semaphore,
    progress: &std::sync::Arc<ProgressTracker>,
) -> Result<()> {
    let _permit = request_sema.acquire().await?;
    let range = format!("bytes={}-{}", start, end);
    let resp = client
        .get_object()
        .bucket(BUCKET)
        .key(key)
        .range(range)
        .send()
        .await?;
    let mut reader = resp.body.into_async_read();
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(out_path)
        .await?;
    file.seek(std::io::SeekFrom::Start(start)).await?;
    tokio::io::copy(&mut reader, &mut file).await?;
    file.flush().await?;
    progress.add_bytes((end - start) + 1);
    Ok(())
}

fn build_ranges(size: u64, part_size: u64) -> Vec<(u64, u64)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < size {
        let mut end = start + part_size - 1;
        if end >= size {
            end = size - 1;
        }
        ranges.push((start, end));
        start = end + 1;
    }
    ranges
}

async fn prepare_output_dir(out_dir: &Path, out_root: &Path) -> Result<()> {
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

    match tokio::fs::metadata(out_dir).await {
        Ok(meta) => {
            if meta.is_dir() {
                tokio::fs::remove_dir_all(out_dir).await?;
            } else {
                bail!("output path exists and is not a directory: {}", out_dir.display());
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
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
        self.done
            .store(true, std::sync::atomic::Ordering::Relaxed);
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
                pb.set_prefix(format!("{}/{} files", files_done, files_total));
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

            if progress
                .done
                .load(std::sync::atomic::Ordering::Relaxed)
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
        .progress_chars("|/-\\")
}
