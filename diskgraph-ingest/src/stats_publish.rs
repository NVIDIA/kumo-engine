use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use chrono::{SecondsFormat, Utc};
use futures::{stream, StreamExt};
use serde::Serialize;

use crate::graph_config::GraphConfig;
use crate::io::{
    build_store_for_cloud_path, is_cloud_path, parse_az_url, parse_gs_url, parse_s3_url, Dest,
    OutputTarget,
};
use crate::stats::{IngestReport, IngestStats};
use object_store::{GetResult, ObjectStoreExt};

#[derive(Clone, Debug)]
pub struct PublishConfig {
    pub source: String,
    pub output: String,
    pub dataset: String,
    pub run_id: String,
    pub stats_dest: Option<String>,
    pub stats_required: bool,
    pub profile_dir: Option<PathBuf>,
    pub dest: Dest,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub dataset: String,
    pub source_path: String,
    pub output_path: String,
    pub finished_at: String,
    pub user: String,
    pub git_sha: Option<String>,
    pub git_branch: Option<String>,
    pub git_dirty: Option<bool>,
    pub git_error: Option<String>,
    pub cloud: Option<String>,
    pub instance_type: Option<String>,
    pub cpu_logical: Option<u64>,
    pub cpu_physical: Option<u64>,
    pub memory_bytes_total: Option<u64>,
    pub disk_bytes_total: Option<u64>,
    pub disk_bytes_available: Option<u64>,
    pub wall_time_ms: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub total_bytes: u64,
    pub throughput_bytes_per_sec: u64,
    pub stage_timings_ms: StageTimings,
    pub normalized: NormalizedMetrics,
    pub peak_rss_anon_bytes: Option<u64>,
    pub peak_context_rss_anon_bytes: Option<u64>,
    pub peak_encode_rss_anon_bytes: Option<u64>,
    pub peak_pk_map_rss_anon_bytes: Option<u64>,
    pub peak_edges_map_rss_anon_bytes: Option<u64>,
    pub peak_index_rss_anon_bytes: Option<u64>,
    pub peak_finalize_rss_anon_bytes: Option<u64>,
    pub num_shards: Option<usize>,
    pub profiles_present: bool,
    pub build_stats_present: bool,
}

#[derive(Clone, Debug)]
pub struct PublishOutcome {
    pub summary: RunSummary,
    pub stats_root: String,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct StageTimings {
    pub context: u64,
    pub encode: u64,
    pub pk_map: u64,
    pub edges_map: u64,
    pub index: u64,
    pub finalize: u64,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct NormalizedMetrics {
    pub throughput_bytes_per_sec_per_cpu: Option<f64>,
    pub wall_time_ms_per_cpu: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CloudKind {
    Aws,
    Gcp,
    Adls,
}

pub fn publish_stats(stats: &IngestStats, config: PublishConfig) -> Result<Option<PublishOutcome>> {
    let stats_root = resolve_stats_root(&config)?;
    let Some(stats_root) = stats_root else {
        return Ok(None);
    };

    let target = stats_target_from_root(&stats_root)?;

    let ingest_stats = read_output_file(&config.output, "ingest_stats.json").ok();
    let build_stats = read_output_file(&config.output, "build_stats.json").ok();

    let profiles_present = profile_files(&config.profile_dir)
        .map(|files| !files.is_empty())
        .unwrap_or(false);

    let data_source_path =
        data_source_path_from_report(&ingest_stats).unwrap_or_else(|| config.source.clone());
    let summary = build_run_summary(
        stats,
        &config,
        &data_source_path,
        profiles_present,
        build_stats.is_some(),
    )?;
    let instance_dir = instance_dir_name(summary.instance_type.as_deref(), &summary.source_path);
    let run_prefix = format!("{}/{}/{}", config.dataset, instance_dir, config.run_id);

    target.write_bytes(
        &format!("{}/run_summary.json", run_prefix),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    if let Some(bytes) = ingest_stats {
        target.write_bytes(&format!("{}/ingest_stats.json", run_prefix), bytes)?;
    }
    if let Some(bytes) = build_stats {
        target.write_bytes(&format!("{}/build_stats.json", run_prefix), bytes)?;
    }
    if let Some(files) = profile_files(&config.profile_dir) {
        for path in files {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| anyhow!("invalid profile filename {}", path.display()))?;
            let bytes =
                std::fs::read(&path).with_context(|| format!("read profile {}", path.display()))?;
            target.write_bytes(&format!("{}/profiles/{}", run_prefix, name), bytes)?;
        }
    }
    Ok(Some(PublishOutcome {
        summary,
        stats_root,
    }))
}

pub fn default_run_id(user: &str, git_sha_prefix: &str) -> String {
    let ts = Utc::now().format("%Y-%m-%dT%H-%M-%SZ");
    format!("{ts}-{user}-{git_sha_prefix}")
}

pub fn resolve_stats_root(config: &PublishConfig) -> Result<Option<String>> {
    if let Some(dest) = config.stats_dest.as_ref() {
        return Ok(Some(dest.clone()));
    }
    if let Ok(dest) = std::env::var("DISKGRAPH_STATS_DEST") {
        if !dest.trim().is_empty() {
            return Ok(Some(dest));
        }
    }
    let cloud = detect_cloud_kind(&config.source, config.dest)?;
    if let Some(cloud) = cloud {
        let root = match cloud {
            CloudKind::Aws => "s3://kumo-unit-test/diskgraph-profiles".to_string(),
            CloudKind::Gcp => "gs://kumo-unit-test/diskgraph-profiles".to_string(),
            CloudKind::Adls => {
                let account = std::env::var("AZURE_STORAGE_ACCOUNT")
                    .or_else(|_| std::env::var("AZURE_ACCOUNT"))
                    .unwrap_or_else(|_| "account".to_string());
                let container = std::env::var("AZURE_STORAGE_CONTAINER")
                    .or_else(|_| std::env::var("AZURE_CONTAINER"))
                    .unwrap_or_else(|_| "container".to_string());
                format!("az://{account}/{container}/diskgraph-profiles")
            }
        };
        return Ok(Some(root));
    }
    Ok(None)
}

#[derive(Clone, Debug)]
pub struct PerfBaseline {
    pub wall_time_ms: u64,
    pub context_wall_ms: u64,
    pub encode_wall_ms: u64,
    pub pk_map_wall_ms: u64,
    pub edges_map_wall_ms: u64,
    pub index_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub git_sha: Option<String>,
}

pub fn fetch_perf_baseline(
    stats_root: &str,
    dataset: &str,
    instance_type: Option<&str>,
    source_path: &str,
    exclude_run_id: &str,
    limit: usize,
) -> Result<Option<PerfBaseline>> {
    if !stats_root.starts_with("s3://")
        && !stats_root.starts_with("gs://")
        && !stats_root.starts_with("az://")
    {
        return Ok(None);
    }
    let store = build_store_for_cloud_path(stats_root)?;
    let prefix = cloud_prefix(stats_root)?;
    let instance_dir = instance_dir_name(instance_type, source_path);
    let runs_prefix = object_path_join(&prefix, &format!("{}/{}", dataset, instance_dir));
    let rt = tokio::runtime::Runtime::new()?;
    let mut stream = store.list(Some(&runs_prefix));
    let summaries = rt.block_on(async {
        let mut locations = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item?;
            if let Some(name) = meta.location.filename() {
                if name == "run_summary.json" {
                    locations.push(meta.location);
                }
            }
        }
        let fetches = stream::iter(locations.into_iter().map(|location| {
            let store = Arc::clone(&store);
            async move {
                let get: GetResult = store.get(&location).await?;
                let data = get.bytes().await?;
                Ok::<_, object_store::Error>(data.to_vec())
            }
        }))
        .buffer_unordered(8);

        let mut collected = Vec::new();
        futures::pin_mut!(fetches);
        while let Some(result) = fetches.next().await {
            let data = result?;
            if let Ok(summary) = serde_json::from_slice::<RunSummary>(&data) {
                if summary.run_id != exclude_run_id {
                    collected.push(summary);
                }
            }
        }
        Ok::<_, object_store::Error>(collected)
    })?;
    if summaries.is_empty() {
        return Ok(None);
    }
    let mut summaries = summaries;
    summaries.sort_by(|a, b| b.finished_at.cmp(&a.finished_at));
    if summaries.len() > limit {
        summaries.truncate(limit);
    }
    let mut baseline: Option<PerfBaseline> = None;
    for summary in summaries {
        let candidate = PerfBaseline {
            wall_time_ms: summary.wall_time_ms,
            context_wall_ms: summary.stage_timings_ms.context,
            encode_wall_ms: summary.stage_timings_ms.encode,
            pk_map_wall_ms: summary.stage_timings_ms.pk_map,
            edges_map_wall_ms: summary.stage_timings_ms.edges_map,
            index_wall_ms: summary.stage_timings_ms.index,
            finalize_wall_ms: summary.stage_timings_ms.finalize,
            git_sha: summary.git_sha.clone(),
        };
        baseline = Some(match baseline {
            Some(prev) => {
                let mut merged = PerfBaseline {
                    wall_time_ms: prev.wall_time_ms.min(candidate.wall_time_ms),
                    context_wall_ms: prev.context_wall_ms.min(candidate.context_wall_ms),
                    encode_wall_ms: prev.encode_wall_ms.min(candidate.encode_wall_ms),
                    pk_map_wall_ms: prev.pk_map_wall_ms.min(candidate.pk_map_wall_ms),
                    edges_map_wall_ms: prev.edges_map_wall_ms.min(candidate.edges_map_wall_ms),
                    index_wall_ms: prev.index_wall_ms.min(candidate.index_wall_ms),
                    finalize_wall_ms: prev.finalize_wall_ms.min(candidate.finalize_wall_ms),
                    git_sha: prev.git_sha.clone(),
                };
                if candidate.wall_time_ms < prev.wall_time_ms {
                    merged.git_sha = candidate.git_sha.clone();
                }
                merged
            }
            None => candidate,
        });
    }
    Ok(baseline)
}

fn detect_cloud_kind(source: &str, dest: Dest) -> Result<Option<CloudKind>> {
    if let Ok(env) = std::env::var("DISKGRAPH_CLOUD") {
        let cloud = match env.to_lowercase().as_str() {
            "aws" | "s3" => Some(CloudKind::Aws),
            "gcp" | "gcs" => Some(CloudKind::Gcp),
            "adls" | "azure" | "az" => Some(CloudKind::Adls),
            _ => None,
        };
        if cloud.is_some() {
            return Ok(cloud);
        }
    }
    match dest {
        Dest::S3 => return Ok(Some(CloudKind::Aws)),
        Dest::Gcs => return Ok(Some(CloudKind::Gcp)),
        Dest::Adls => return Ok(Some(CloudKind::Adls)),
        Dest::Local => {}
    }
    if source.starts_with("s3://") {
        return Ok(Some(CloudKind::Aws));
    }
    if source.starts_with("gs://") {
        return Ok(Some(CloudKind::Gcp));
    }
    if source.starts_with("az://") {
        return Ok(Some(CloudKind::Adls));
    }
    if let Some(cloud) = detect_cloud_from_imds() {
        return Ok(Some(cloud));
    }
    if std::env::var("AWS_REGION").is_ok() || std::env::var("AWS_DEFAULT_REGION").is_ok() {
        return Ok(Some(CloudKind::Aws));
    }
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_ok()
        || std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok()
    {
        return Ok(Some(CloudKind::Gcp));
    }
    if std::env::var("AZURE_TENANT_ID").is_ok() || std::env::var("AZURE_STORAGE_ACCOUNT").is_ok() {
        return Ok(Some(CloudKind::Adls));
    }
    Ok(None)
}

fn detect_cloud_from_imds() -> Option<CloudKind> {
    if fetch_aws_instance_type().is_some() {
        return Some(CloudKind::Aws);
    }
    if fetch_gcp_instance_type().is_some() {
        return Some(CloudKind::Gcp);
    }
    if fetch_azure_instance_type().is_some() {
        return Some(CloudKind::Adls);
    }
    None
}

fn stats_target_from_root(root: &str) -> Result<OutputTarget> {
    if root.starts_with("s3://") {
        Ok(OutputTarget::parse(root, Dest::S3)?)
    } else if root.starts_with("gs://") {
        Ok(OutputTarget::parse(root, Dest::Gcs)?)
    } else if root.starts_with("az://") {
        Ok(OutputTarget::parse(root, Dest::Adls)?)
    } else {
        Ok(OutputTarget::parse(root, Dest::Local)?)
    }
}

fn read_output_file(output: &str, name: &str) -> Result<Vec<u8>> {
    if is_cloud_path(output) {
        let store = build_store_for_cloud_path(output)?;
        let prefix = cloud_prefix(output)?;
        let path = object_path_join(&prefix, name);
        let rt = tokio::runtime::Runtime::new()?;
        let bytes = rt.block_on(async {
            let result: GetResult = store.get(&path).await?;
            let data = result.bytes().await?;
            Ok::<_, object_store::Error>(data.to_vec())
        })?;
        Ok(bytes)
    } else {
        let path = Path::new(output).join(name);
        Ok(std::fs::read(&path).with_context(|| format!("read {}", path.display()))?)
    }
}

fn build_run_summary(
    stats: &IngestStats,
    config: &PublishConfig,
    data_source_path: &str,
    profiles_present: bool,
    build_stats_present: bool,
) -> Result<RunSummary> {
    let total_bytes = stats
        .perf
        .bytes_read
        .saturating_add(stats.perf.bytes_written);
    let wall_secs = (stats.perf.wall_time_ms as f64) / 1000.0;
    let throughput_bytes_per_sec = if wall_secs > 0.0 {
        (total_bytes as f64 / wall_secs) as u64
    } else {
        0
    };

    let machine = machine_stats(&config.output);
    let git = git_info();
    let (cloud, instance_type) = instance_info(data_source_path, config.dest)?;

    let (norm_throughput, norm_wall) = if let Some(cpu) = machine.cpu_logical {
        if cpu > 0 {
            (
                Some(throughput_bytes_per_sec as f64 / cpu as f64),
                Some(stats.perf.wall_time_ms as f64 / cpu as f64),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(RunSummary {
        run_id: config.run_id.clone(),
        dataset: config.dataset.clone(),
        source_path: data_source_path.to_string(),
        output_path: config.output.clone(),
        finished_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        user: current_user(),
        git_sha: git.sha,
        git_branch: git.branch,
        git_dirty: git.dirty,
        git_error: git.error,
        cloud,
        instance_type,
        cpu_logical: machine.cpu_logical,
        cpu_physical: machine.cpu_physical,
        memory_bytes_total: machine.memory_bytes_total,
        disk_bytes_total: machine.disk_bytes_total,
        disk_bytes_available: machine.disk_bytes_available,
        wall_time_ms: stats.perf.wall_time_ms,
        bytes_read: stats.perf.bytes_read,
        bytes_written: stats.perf.bytes_written,
        total_bytes,
        throughput_bytes_per_sec,
        stage_timings_ms: StageTimings {
            context: stats.perf.context_wall_ms,
            encode: stats.perf.encode_wall_ms,
            pk_map: stats.perf.pk_map_wall_ms,
            edges_map: stats.perf.edges_map_wall_ms,
            index: stats.perf.index_wall_ms,
            finalize: stats.perf.finalize_wall_ms,
        },
        normalized: NormalizedMetrics {
            throughput_bytes_per_sec_per_cpu: norm_throughput,
            wall_time_ms_per_cpu: norm_wall,
        },
        peak_rss_anon_bytes: stats.perf.peak_rss_anon_bytes,
        peak_context_rss_anon_bytes: stats.perf.peak_context_rss_anon_bytes,
        peak_encode_rss_anon_bytes: stats.perf.peak_encode_rss_anon_bytes,
        peak_pk_map_rss_anon_bytes: stats.perf.peak_pk_map_rss_anon_bytes,
        peak_edges_map_rss_anon_bytes: stats.perf.peak_edges_map_rss_anon_bytes,
        peak_index_rss_anon_bytes: stats.perf.peak_index_rss_anon_bytes,
        peak_finalize_rss_anon_bytes: stats.perf.peak_finalize_rss_anon_bytes,
        num_shards: stats.num_shards,
        profiles_present,
        build_stats_present,
    })
}

fn data_source_path_from_report(ingest_stats: &Option<Vec<u8>>) -> Option<String> {
    let bytes = ingest_stats.as_ref()?;
    let report: IngestReport = serde_json::from_slice(bytes).ok()?;
    data_source_path_from_config(&report.source)
}

fn data_source_path_from_config(source: &GraphConfig) -> Option<String> {
    let mut first_local: Option<String> = None;
    for table in &source.tables {
        for path in table.source_paths() {
            if is_cloud_path(path) {
                return Some(path.to_string());
            }
            if first_local.is_none() {
                first_local = Some(path.to_string());
            }
        }
    }
    first_local
}

fn current_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

struct GitInfo {
    sha: Option<String>,
    branch: Option<String>,
    dirty: Option<bool>,
    error: Option<String>,
}

fn git_info() -> GitInfo {
    let sha = run_git(&["rev-parse", "HEAD"]);
    let branch = run_git(&["rev-parse", "--abbrev-ref", "HEAD"]);
    let dirty = run_git(&["status", "--porcelain"]).map(|s| !s.trim().is_empty());
    if sha.is_err() || branch.is_err() || dirty.is_err() {
        let error = sha
            .as_ref()
            .err()
            .or_else(|| branch.as_ref().err())
            .or_else(|| dirty.as_ref().err())
            .map(|e| e.to_string());
        return GitInfo {
            sha: sha.ok(),
            branch: branch.ok(),
            dirty: dirty.ok(),
            error,
        };
    }
    GitInfo {
        sha: sha.ok(),
        branch: branch.ok(),
        dirty: dirty.ok(),
        error: None,
    }
}

fn run_git(args: &[&str]) -> Result<String> {
    let output = Command::new("git").args(args).output().context("run git")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("git {} failed: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

struct MachineStats {
    cpu_logical: Option<u64>,
    cpu_physical: Option<u64>,
    memory_bytes_total: Option<u64>,
    disk_bytes_total: Option<u64>,
    disk_bytes_available: Option<u64>,
}

fn machine_stats(output: &str) -> MachineStats {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.refresh_cpu();

    let cpu_logical = system.cpus().len() as u64;
    let cpu_physical = system.physical_core_count().map(|v| v as u64);

    let mut memory_bytes_total = system.total_memory();
    if memory_bytes_total < (4 * 1024 * 1024 * 1024) {
        memory_bytes_total = memory_bytes_total.saturating_mul(1024);
    }

    let (disk_bytes_total, disk_bytes_available) = if !is_cloud_path(output) {
        let output_path = Path::new(output);
        disk_stats_for_path(output_path)
    } else {
        (None, None)
    };

    MachineStats {
        cpu_logical: Some(cpu_logical).filter(|v| *v > 0),
        cpu_physical,
        memory_bytes_total: Some(memory_bytes_total).filter(|v| *v > 0),
        disk_bytes_total,
        disk_bytes_available,
    }
}

fn disk_stats_for_path(path: &Path) -> (Option<u64>, Option<u64>) {
    let mut disks = sysinfo::Disks::new();
    disks.refresh_list();
    let mut best: Option<(usize, u64, u64)> = None;
    for disk in disks.iter() {
        let mount = disk.mount_point();
        if let Ok(stripped) = path.strip_prefix(mount) {
            let score = mount.as_os_str().len();
            let total = disk.total_space();
            let avail = disk.available_space();
            if best
                .map(|(best_score, _, _)| score > best_score)
                .unwrap_or(true)
            {
                best = Some((score, total, avail));
            }
            let _ = stripped;
        }
    }
    if let Some((_, total, avail)) = best {
        (Some(total), Some(avail))
    } else {
        (None, None)
    }
}

fn instance_info(source: &str, dest: Dest) -> Result<(Option<String>, Option<String>)> {
    let cloud = detect_cloud_kind(source, dest)?;
    let info = if let Some(cloud) = cloud {
        (Some(cloud), fetch_instance_type(cloud))
    } else {
        match fetch_any_instance_type() {
            Some((cloud, itype)) => (Some(cloud), Some(itype)),
            None => (None, None),
        }
    };
    let cloud_name = info.0.map(|c| match c {
        CloudKind::Aws => "aws".to_string(),
        CloudKind::Gcp => "gcp".to_string(),
        CloudKind::Adls => "adls".to_string(),
    });
    Ok((cloud_name, info.1))
}

fn fetch_any_instance_type() -> Option<(CloudKind, String)> {
    fetch_instance_type(CloudKind::Aws)
        .map(|t| (CloudKind::Aws, t))
        .or_else(|| fetch_instance_type(CloudKind::Gcp).map(|t| (CloudKind::Gcp, t)))
        .or_else(|| fetch_instance_type(CloudKind::Adls).map(|t| (CloudKind::Adls, t)))
}

fn fetch_instance_type(cloud: CloudKind) -> Option<String> {
    match cloud {
        CloudKind::Aws => fetch_aws_instance_type(),
        CloudKind::Gcp => fetch_gcp_instance_type(),
        CloudKind::Adls => fetch_azure_instance_type(),
    }
}

fn fetch_aws_instance_type() -> Option<String> {
    let token = ureq::put("http://169.254.169.254/latest/api/token")
        .timeout(Duration::from_millis(800))
        .set("X-aws-ec2-metadata-token-ttl-seconds", "300")
        .call()
        .ok()?
        .into_string()
        .ok()?;
    let resp = ureq::get("http://169.254.169.254/latest/meta-data/instance-type")
        .timeout(Duration::from_millis(800))
        .set("X-aws-ec2-metadata-token", &token)
        .call()
        .ok()?;
    resp.into_string().ok().map(|s| s.trim().to_string())
}

fn fetch_gcp_instance_type() -> Option<String> {
    let resp =
        ureq::get("http://metadata.google.internal/computeMetadata/v1/instance/machine-type")
            .timeout(Duration::from_millis(800))
            .set("Metadata-Flavor", "Google")
            .call()
            .ok()?;
    let raw = resp.into_string().ok()?;
    Some(
        raw.trim()
            .rsplit('/')
            .next()
            .unwrap_or(raw.trim())
            .to_string(),
    )
}

fn fetch_azure_instance_type() -> Option<String> {
    let resp =
        ureq::get("http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01")
            .timeout(Duration::from_millis(800))
            .set("Metadata", "true")
            .call()
            .ok()?;
    resp.into_string().ok().map(|s| s.trim().to_string())
}

fn profile_files(profile_dir: &Option<PathBuf>) -> Option<Vec<PathBuf>> {
    let dir = profile_dir.as_ref()?;
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if name.ends_with("-profile.pb")
            || name.ends_with("-profile.svg")
            || (name.starts_with("heap-") && name.ends_with(".heap"))
        {
            files.push(path);
        }
    }
    Some(files)
}

fn instance_dir_name(instance_type: Option<&str>, source_path: &str) -> String {
    let raw = instance_type.unwrap_or("unknown");
    let suffix = if is_cloud_path(source_path) {
        "remote"
    } else {
        "local"
    };
    let mut name: String = raw
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ' ' | ':' => '_',
            _ => ch,
        })
        .collect();
    name.push('-');
    name.push_str(suffix);
    name
}

fn cloud_prefix(path: &str) -> Result<String> {
    if path.starts_with("s3://") {
        let (_bucket, prefix) = parse_s3_url(path)?;
        Ok(prefix)
    } else if path.starts_with("gs://") {
        let (_bucket, prefix) = parse_gs_url(path)?;
        Ok(prefix)
    } else if path.starts_with("az://") {
        let (_account, _container, prefix) = parse_az_url(path)?;
        Ok(prefix)
    } else {
        Err(anyhow!("unsupported cloud path {path}"))
    }
}

fn object_path_join(prefix: &str, name: &str) -> object_store::path::Path {
    if prefix.is_empty() {
        object_store::path::Path::from(name)
    } else {
        object_store::path::Path::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
    }
}
