use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{
    build_transfer_for_cloud_path, ingest_with_stats, ingest_with_stats_from_source,
    object_path_from_cloud_url, Dest, IngestOptions, GraphConfig,
};
use diskgraph_ingest::progress::{build_progress, format_bytes, format_duration};
use diskgraph_ingest::stats_publish::{
    default_run_id, fetch_perf_baseline, publish_stats, PublishConfig, PublishOutcome,
};
use std::fs;

#[cfg(all(feature = "mem-profile", target_os = "linux"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about = "Ingest raw CSV/Parquet into diskgraph index format.")]
struct Args {
    /// Path or URL to diskgraph_config.json, or a directory containing it.
    #[arg(long)]
    source: PathBuf,
    /// Output destination root (path or URL).
    #[arg(long)]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: DestArg,
    /// Write pprof profiles per phase to this directory.
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = ".")]
    profile: Option<PathBuf>,
    /// Enable the shared worker pool.
    #[arg(long, default_value_t = true)]
    parallel: bool,
    /// Enable jemalloc heap dump watchdog (80% RSS anon threshold).
    #[arg(short = 'm', long = "mem-profile")]
    mem_profile: bool,
    /// Max bytes of index-build edge sort memory allowed across workers.
    #[arg(long, value_name = "BYTES")]
    index_mem_budget_bytes: Option<u64>,
    /// Stats destination root (s3://, gs://, az://, or local path).
    #[arg(long, value_name = "URL")]
    stats_dest: Option<String>,
    /// Dataset name for stats publishing (defaults from source).
    #[arg(long, value_name = "NAME")]
    stats_dataset: Option<String>,
    /// Run id for stats publishing (defaults to timestamp-host-pid).
    #[arg(long, value_name = "ID")]
    stats_run_id: Option<String>,
    /// Fail the ingest if stats publishing fails.
    #[arg(long, default_value_t = false)]
    stats_required: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DestArg {
    Local,
    S3,
    Gcs,
    Adls,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,pprof=warn")
        .init();
    let args = Args::parse();
    let source_str = args.source.to_string_lossy();
    let source_path = if args.source.is_dir() {
        args.source.join("diskgraph_config.json")
    } else {
        args.source.clone()
    };
    if !is_cloud_url(&source_str) && !source_path.is_file() {
        anyhow::bail!(
            "diskgraph config not found at {}",
            source_path.display()
        );
    }
    if args.profile.is_some() && cfg!(debug_assertions) && cfg!(target_arch = "aarch64") {
        anyhow::bail!(
            "pprof profiling in debug builds can panic on aarch64; rerun with `--release`"
        );
    }
    if let Some(profile_dir) = args.profile.as_ref() {
        fs::create_dir_all(profile_dir)?;
    }
    let mem_watchdog: Option<mem_watchdog::MemWatchdog> = if args.mem_profile {
        #[cfg(feature = "mem-profile")]
        {
            let dir = args
                .profile
                .clone()
                .unwrap_or_else(|| PathBuf::from("."));
            Some(mem_watchdog::start(dir))
        }
        #[cfg(not(feature = "mem-profile"))]
        {
            tracing::warn!("--mem-profile requires the cargo feature 'mem-profile'");
            None
        }
    } else {
        None
    };
    let options = IngestOptions {
        profile_dir: args.profile.clone(),
        progress: build_progress(),
        parallel: args.parallel,
        index_mem_budget_bytes: args.index_mem_budget_bytes,
    };
    let dest = match args.dest {
        DestArg::Local => Dest::Local,
        DestArg::S3 => Dest::S3,
        DestArg::Gcs => Dest::Gcs,
        DestArg::Adls => Dest::Adls,
    };
    let (_metadata, stats) = if is_cloud_url(&source_str) {
        let config_url = ensure_config_url(&source_str);
        let metadata_bytes = download_cloud_config_bytes(&config_url)?;
        let source: GraphConfig = serde_json::from_slice(&metadata_bytes)?;
        ingest_with_stats_from_source(source, None, &args.output, dest, options)?
    } else {
        ingest_with_stats(&source_path, &args.output, dest, options)?
    };
    drop(mem_watchdog);
    let publish_outcome = match maybe_publish_stats(&args, &stats, &source_str, dest) {
        Ok(outcome) => outcome,
        Err(err) => {
            if args.stats_required {
                return Err(err);
            }
            tracing::warn!("stats publish failed: {err}");
            None
        }
    };
    println!("Output: {}", stats.output);
    println!(
        "Node tables: {} ({} nodes)",
        stats.node_tables, stats.total_nodes
    );
    println!(
        "Edge relations: {} ({} edges)",
        stats.edge_relations, stats.total_edges
    );
    if !stats.node_counts.is_empty() {
        println!("Nodes per table:");
        let mut entries: Vec<_> = stats.node_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
        }
    }
    if !stats.edge_counts.is_empty() {
        println!("Edges per relation:");
        let mut entries: Vec<_> = stats.edge_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
        }
    }
    if stats.pk_duplicate_total > 0 {
        println!("Duplicate primary keys:");
        println!("  Total skipped: {}", stats.pk_duplicate_total);
        let mut totals: std::collections::HashMap<&str, u64> = std::collections::HashMap::new();
        for (name, count) in &stats.pk_duplicate_counts {
            let table = name.split(':').next().unwrap_or(name.as_str());
            let entry = totals.entry(table).or_insert(0);
            *entry = entry.saturating_add(*count);
        }
        let mut samples_by_table: std::collections::HashMap<&str, Vec<String>> =
            std::collections::HashMap::new();
        for (name, samples) in &stats.pk_duplicate_samples {
            let table = name.split(':').next().unwrap_or(name.as_str());
            let entry = samples_by_table.entry(table).or_default();
            for sample in samples {
                if entry.len() >= 10 {
                    break;
                }
                entry.push(sample.clone());
            }
        }
        let mut entries: Vec<_> = totals.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
            if let Some(samples) = samples_by_table.get(name) {
                if !samples.is_empty() {
                    println!("    Samples: {}", samples.join(", "));
                }
            }
        }
    }
    if stats.key_parse_failures > 0 || stats.key_lookup_failures > 0 {
        println!("Key matching failures:");
        println!("  Parse failures: {}", stats.key_parse_failures);
        println!("  Lookup misses: {}", stats.key_lookup_failures);
        if !stats.key_failures_by_context.is_empty() {
            println!("  By context:");
            let mut entries: Vec<_> = stats.key_failures_by_context.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (name, counts) in entries {
                if counts.parse_failures == 0 && counts.lookup_failures == 0 {
                    continue;
                }
                println!(
                    "    {}: parse {}, lookup {}",
                    name, counts.parse_failures, counts.lookup_failures
                );
            }
        }
        if !stats.key_failure_samples.is_empty() {
            println!("  Examples:");
            for sample in &stats.key_failure_samples {
                println!("    {}", sample);
            }
        }
    }
    if let Some(trimmed) = stats.trimmed_strings {
        println!("Trimmed strings (>32KB): {}", trimmed);
    }
    let total_bytes = stats.perf.bytes_read.saturating_add(stats.perf.bytes_written);
    let wall_secs = (stats.perf.wall_time_ms as f64) / 1000.0;
    let bytes_per_sec = if wall_secs > 0.0 {
        (total_bytes as f64) / wall_secs
    } else {
        0.0
    };
    let baseline = publish_outcome
        .as_ref()
        .and_then(|outcome| {
            fetch_perf_baseline(
                &outcome.stats_root,
                &outcome.summary.dataset,
                outcome.summary.instance_type.as_deref(),
                &outcome.summary.source_path,
                &outcome.summary.run_id,
                30,
            )
            .ok()
            .flatten()
        });
    println!("Performance:");
    if let (Some(baseline), Some(outcome)) = (baseline.as_ref(), publish_outcome.as_ref()) {
        let instance = outcome
            .summary
            .instance_type
            .as_deref()
            .unwrap_or("unknown");
        let git = baseline
            .git_sha
            .as_ref()
            .map(|s| s.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "nogit".to_string());
        println!(
            "  Baseline: min of last 30 runs for {} on {} (git {})",
            outcome.summary.dataset, instance, git
        );
    }
    println!(
        "  Wall time: {}{}",
        format_duration(stats.perf.wall_time_ms),
        format_delta(stats.perf.wall_time_ms, baseline.as_ref().map(|b| b.wall_time_ms))
    );
    println!(
        "  Bytes read: {}",
        format_bytes(stats.perf.bytes_read)
    );
    println!(
        "  Bytes written: {}",
        format_bytes(stats.perf.bytes_written)
    );
    println!("  Total bytes: {}", format_bytes(total_bytes));
    println!("  Throughput: {}/s", format_bytes(bytes_per_sec as u64));
    if stats.perf.temp_bytes_total > 0 {
        println!("  Temp bytes (edges): {}", format_bytes(stats.perf.temp_bytes_total));
        if stats.perf.temp_fkeys_bytes > 0 {
            println!(
                "    fkeys: {}",
                format_bytes(stats.perf.temp_fkeys_bytes)
            );
        }
        if stats.perf.temp_edges_src_bytes > 0 {
            println!(
                "    edges_src: {}",
                format_bytes(stats.perf.temp_edges_src_bytes)
            );
        }
        if stats.perf.temp_edges_map_bytes > 0 {
            println!(
                "    edges_map: {}",
                format_bytes(stats.perf.temp_edges_map_bytes)
            );
        }
    }
    if stats.perf.temp_node_data_bytes > 0 {
        println!(
            "  Temp bytes (nodes): {}",
            format_bytes(stats.perf.temp_node_data_bytes)
        );
    }
    println!("  Stage timings (wall):");
    println!(
        "    context: {}{}",
        format_duration(stats.perf.context_wall_ms),
        format_delta(
            stats.perf.context_wall_ms,
            baseline.as_ref().map(|b| b.context_wall_ms)
        )
    );
    println!(
        "    encode: {}{}",
        format_duration(stats.perf.encode_wall_ms),
        format_delta(
            stats.perf.encode_wall_ms,
            baseline.as_ref().map(|b| b.encode_wall_ms)
        )
    );
    println!(
        "    pk_map: {}{}",
        format_duration(stats.perf.pk_map_wall_ms),
        format_delta(
            stats.perf.pk_map_wall_ms,
            baseline.as_ref().map(|b| b.pk_map_wall_ms)
        )
    );
    println!(
        "    edges_map: {}{}",
        format_duration(stats.perf.edges_map_wall_ms),
        format_delta(
            stats.perf.edges_map_wall_ms,
            baseline.as_ref().map(|b| b.edges_map_wall_ms)
        )
    );
    println!(
        "    index: {}{}",
        format_duration(stats.perf.index_wall_ms),
        format_delta(
            stats.perf.index_wall_ms,
            baseline.as_ref().map(|b| b.index_wall_ms)
        )
    );
    println!(
        "    finalize: {}{}",
        format_duration(stats.perf.finalize_wall_ms),
        format_delta(
            stats.perf.finalize_wall_ms,
            baseline.as_ref().map(|b| b.finalize_wall_ms)
        )
    );
    println!(
        "Next: {}",
        explore_command_hint(&args.dest, &stats.output)
    );
    Ok(())
}

fn maybe_publish_stats(
    args: &Args,
    stats: &diskgraph_ingest::IngestStats,
    source_str: &str,
    dest: Dest,
) -> Result<Option<PublishOutcome>> {
    let dataset = args
        .stats_dataset
        .clone()
        .or_else(|| std::env::var("DISKGRAPH_STATS_DATASET").ok())
        .unwrap_or_else(|| dataset_from_source(&args.source, source_str));
    let run_id = args.stats_run_id.clone()
        .or_else(|| std::env::var("DISKGRAPH_STATS_RUN_ID").ok())
        .unwrap_or_else(|| default_run_id(&run_id_user(), &run_id_git_prefix()));
    let stats_dest = args.stats_dest.clone();
    let config = PublishConfig {
        source: source_str.to_string(),
        output: args.output.clone(),
        dataset,
        run_id,
        stats_dest,
        stats_required: args.stats_required,
        profile_dir: args.profile.clone().or_else(|| {
            if args.mem_profile {
                Some(PathBuf::from("."))
            } else {
                None
            }
        }),
        dest,
    };
    let outcome = publish_stats(stats, config)?;
    Ok(outcome)
}

fn run_id_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn run_id_git_prefix() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();
    let sha = match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        _ => "nogit".to_string(),
    };
    sha.chars().take(8).collect()
}

fn format_delta(current_ms: u64, baseline_ms: Option<u64>) -> String {
    let Some(baseline_ms) = baseline_ms else {
        return String::new();
    };
    if baseline_ms == 0 {
        return String::new();
    }
    let delta_ms = current_ms.saturating_sub(baseline_ms);
    let pct = (delta_ms as f64 / baseline_ms as f64) * 100.0;
    let (color, reset) = if pct <= 5.0 {
        ("\x1b[32m", "\x1b[0m")
    } else if pct <= 15.0 {
        ("\x1b[33m", "\x1b[0m")
    } else {
        ("\x1b[31m", "\x1b[0m")
    };
    format!(
        "  {color}+{} (+{pct:.1}%){reset}",
        format_duration(delta_ms),
        reset = reset
    )
}

fn dataset_from_source(source: &PathBuf, source_str: &str) -> String {
    if is_cloud_url(source_str) {
        let trimmed = source_str.trim_end_matches('/');
        return trimmed
            .rsplit('/')
            .next()
            .unwrap_or(trimmed)
            .to_string();
    }
    if source.is_dir() {
        return source.file_name().and_then(|n| n.to_str()).unwrap_or("dataset").to_string();
    }
    if source.ends_with("diskgraph_config.json") {
        if let Some(parent) = source.parent() {
            if let Some(name) = parent.file_name().and_then(|n| n.to_str()) {
                return name.to_string();
            }
        }
    }
    source
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("dataset")
        .to_string()
}

#[cfg(feature = "mem-profile")]
mod mem_watchdog {
    use std::ffi::CString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    #[cfg(target_os = "linux")]
    use std::os::unix::ffi::OsStrExt;

    const MEM_WATCHDOG_INTERVAL: Duration = Duration::from_millis(500);
    const MEM_WATCHDOG_THRESHOLD: f64 = 0.80;

    pub struct MemWatchdog {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Drop for MemWatchdog {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    pub fn start(dump_dir: PathBuf) -> MemWatchdog {
        log_jemalloc_conf();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let handle = thread::spawn(move || mem_watchdog_loop(dump_dir, stop_thread));
        MemWatchdog {
            stop,
            handle: Some(handle),
        }
    }

    fn mem_watchdog_loop(dump_dir: PathBuf, stop: Arc<AtomicBool>) {
        if !cfg!(target_os = "linux") {
            tracing::warn!("memory watchdog only supported on linux");
            return;
        }
        let total_bytes = match total_memory_bytes() {
            Some(total) => total,
            None => {
                tracing::warn!("memory watchdog disabled: unable to read total memory");
                return;
            }
        };
        let threshold_bytes = (total_bytes as f64 * MEM_WATCHDOG_THRESHOLD) as u64;
        let dumped = AtomicBool::new(false);
        while !stop.load(Ordering::Relaxed) {
            if let Some(rss_anon) = rss_anon_bytes() {
                if rss_anon >= threshold_bytes
                    && dumped
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                {
                    if let Err(err) = dump_heap_profile(&dump_dir) {
                        tracing::warn!("heap profile dump failed: {err}");
                    }
                }
            }
            thread::sleep(MEM_WATCHDOG_INTERVAL);
        }
    }

    fn dump_heap_profile(dump_dir: &PathBuf) -> Result<()> {
        if !jemalloc_prof_supported() {
            tracing::warn!("jemalloc profiling is not compiled in");
            return Ok(());
        }
        if !jemalloc_prof_enabled() {
            tracing::warn!("jemalloc opt.prof=false; set _RJEM_MALLOC_CONF=prof:true");
            return Ok(());
        }
        if let Err(err) = unsafe { jemalloc_ctl::raw::write(b"prof.active\0", true) } {
            tracing::warn!("jemalloc prof.active enable failed: {err}");
        }
        fs::create_dir_all(dump_dir)?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let dump_path = dump_dir.join(format!("heap-{}.heap", ts));
        let c_path = CString::new(dump_path.as_os_str().as_bytes())?;
        unsafe {
            jemalloc_ctl::raw::write(b"prof.dump\0", c_path.as_ptr())
                .map_err(|e| anyhow::anyhow!("prof.dump failed: {e}"))?;
        }
        tracing::info!("heap profile written to {}", dump_path.display());
        Ok(())
    }

    fn jemalloc_prof_supported() -> bool {
        unsafe { jemalloc_ctl::raw::read::<bool>(b"config.prof\0") }.unwrap_or(false)
    }

    fn jemalloc_prof_enabled() -> bool {
        unsafe { jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") }.unwrap_or(false)
    }

    fn log_jemalloc_conf() {
        let prof = unsafe { jemalloc_ctl::raw::read::<bool>(b"config.prof\0") }.unwrap_or(false);
        let opt_prof = unsafe { jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") }.unwrap_or(false);
        tracing::warn!("jemalloc config.prof={}, opt.prof={}", prof, opt_prof);
        let conf = unsafe { jemalloc_ctl::raw::read::<*const std::os::raw::c_char>(b"config.malloc_conf\0") };
        if let Ok(ptr) = conf {
            if ptr.is_null() {
                tracing::warn!("jemalloc config.malloc_conf is null");
                return;
            }
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(ptr);
                tracing::warn!("jemalloc config.malloc_conf: {}", cstr.to_string_lossy());
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn rss_anon_bytes() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("RssAnon:") {
                return parse_kb_line(value);
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn rss_anon_bytes() -> Option<u64> {
        None
    }

    #[cfg(target_os = "linux")]
    fn total_memory_bytes() -> Option<u64> {
        let mem_total = read_meminfo_total().unwrap_or(0);
        let cgroup_limit = read_cgroup_limit().unwrap_or(u64::MAX);
        let total = mem_total.min(cgroup_limit);
        if total == 0 || total == u64::MAX {
            None
        } else {
            Some(total)
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn total_memory_bytes() -> Option<u64> {
        None
    }

    #[cfg(target_os = "linux")]
    fn read_meminfo_total() -> Option<u64> {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(value) = line.strip_prefix("MemTotal:") {
                return parse_kb_line(value);
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn read_cgroup_limit() -> Option<u64> {
        if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory.max") {
            let trimmed = limit.trim();
            if trimmed != "max" {
                if let Ok(bytes) = trimmed.parse::<u64>() {
                    return Some(bytes);
                }
            }
        }
        if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
            let trimmed = limit.trim();
            if let Ok(bytes) = trimmed.parse::<u64>() {
                return Some(bytes);
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn parse_kb_line(value: &str) -> Option<u64> {
        let mut iter = value.split_whitespace();
        let num = iter.next()?.parse::<u64>().ok()?;
        Some(num.saturating_mul(1024))
    }
}

#[cfg(not(feature = "mem-profile"))]
mod mem_watchdog {
    pub struct MemWatchdog;
}

fn explore_command_hint(dest: &DestArg, output: &str) -> String {
    let _ = dest;
    format!("./diskgraph-index/scripts/explore_local.sh {}", output)
}

fn is_cloud_url(path: &str) -> bool {
    path.starts_with("s3://") || path.starts_with("gs://") || path.starts_with("az://")
}

fn ensure_config_url(path: &str) -> String {
    if path.ends_with("diskgraph_config.json") {
        path.to_string()
    } else {
        format!("{}/diskgraph_config.json", path.trim_end_matches('/'))
    }
}

fn download_cloud_config_bytes(url: &str) -> Result<Vec<u8>> {
    let transfer = build_transfer_for_cloud_path(url)?;
    let object_path = object_path_from_cloud_url(url)?;
    let pooled = transfer.submit_download_bytes(object_path)?.wait()?;
    Ok(pooled.bytes().to_vec())
}
