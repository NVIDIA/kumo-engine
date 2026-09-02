use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::config::{PeakMemStats, PEAK_MEM_STAGE_COUNT};
use diskgraph_ingest::diagnostics::{
    render_diagnostic, render_process_fallback, restrict, write_diagnostic_atomically,
    DiagnosticCategory, DiagnosticEnvelope, RestrictedDiagnostic, DIAGNOSTIC_FILENAME,
    WORKER_DIAGNOSTIC_FILENAME,
};
use diskgraph_ingest::progress::{build_progress, format_bytes, format_duration};
use diskgraph_ingest::stats_publish::{
    default_run_id, fetch_perf_baseline, publish_stats, PublishConfig, PublishOutcome,
};
use diskgraph_ingest::{
    build_transfer_for_cloud_path, ingest_with_stats_from_source,
    object_path_from_cloud_url, Dest, GraphConfig, IngestOptions,
};
use std::fs;

#[cfg(all(feature = "mem-profile", target_os = "linux"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Ingest raw CSV/Parquet into diskgraph index format."
)]
struct Args {
    /// Internal worker mode used by the phase-aware OOM supervisor.
    #[arg(long, hide = true)]
    worker: bool,
    /// Maximum worker restarts after a retryable exit.
    #[arg(long, default_value_t = 8)]
    oom_retries: usize,
    /// Print the resolved default worker-thread count (cgroup-aware) and exit.
    /// The Kumo runner queries this so its thread default matches the binary's.
    /// Exclusive: pass on its own, no other args.
    #[arg(long, exclusive = true)]
    print_resolved_threads: bool,
    /// Path or URL to diskgraph_config.json, or a directory containing it.
    #[arg(long, required_unless_present = "print_resolved_threads")]
    source: Option<PathBuf>,
    /// Output destination root (path or URL).
    #[arg(long, required_unless_present = "print_resolved_threads")]
    output: Option<String>,
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
    /// Optional mode-0600 Databricks unified-authentication config file.
    #[arg(long, value_name = "FILE")]
    databricks_config_file: Option<PathBuf>,
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
    /// Collect per-column statistics during encode.
    #[arg(long, default_value_t = false)]
    col_stats: bool,
    /// Category count strategy: none, hybrid, graduation.
    #[arg(long, value_enum, default_value = "none")]
    cat_count_strategy: CatCountStrategyArg,
    /// Enable per-row 30% subsampling for category counts (scale ÷0.3).
    #[arg(long, default_value_t = false)]
    row_subsample: bool,
    /// Resume from this phase. Predecessor phases must already have completion
    /// markers in `<output>/_tmp/_phase_complete_<name>`. Default: unset
    /// (fresh ingest, run all phases from scratch).
    #[arg(long, value_enum, value_name = "PHASE")]
    resume_from_phase: Option<PhaseArg>,
    /// Override concurrent source loaders/downloads during Encode.
    /// By default Databricks uses four downloads per cgroup-aware worker,
    /// capped at 512.
    #[arg(long)]
    context_threads: Option<usize>,
    /// Override Encode worker concurrency
    /// (default: ~1.05 × available_parallelism).
    #[arg(long)]
    encode_threads: Option<usize>,
    /// Override worker pool size for the pk_map phase
    /// (default: ~1.05 × available_parallelism).
    #[arg(long)]
    pk_map_threads: Option<usize>,
    /// Override worker pool size for the edges_map phase
    /// (default: ~1.05 × available_parallelism).
    #[arg(long)]
    edges_map_threads: Option<usize>,
    /// Override worker pool size for the index phase
    /// (default: ~1.05 × available_parallelism).
    #[arg(long)]
    index_threads: Option<usize>,
    /// Optional dataplane mirror destination. When set, every output
    /// file ingest writes locally is also streamed to this URL (s3://,
    /// gs://, az://, or local path). End-of-ingest finalize writes
    /// `_MANIFEST.json` and `_SUCCESS` last on both backends, in that
    /// order. Mirrors the pyo3 `ingest()` kwarg of the same name; left
    /// unset means local-only ingest.
    #[arg(long, value_name = "URL")]
    output_dataplane_dir: Option<String>,
    /// Required when `output_dataplane_dir` is set. The byte payload
    /// written to the local + dataplane `_SUCCESS` markers as the
    /// atomic last-write of the snapshot. Typically a JSON blob the
    /// caller uses to identify the snapshot. Left unset means no
    /// success marker is written; callers using the dataplane mirror
    /// MUST supply this (the library aborts the ingest with a clear
    /// error otherwise — markerless mirror state is unverifiable).
    #[arg(long, value_name = "STRING")]
    success_marker_content: Option<String>,
    /// Path to a parent DG index. Required when the source declares
    /// resolution columns that resolve into the parent's primary keys
    /// (e.g. seed-table extensions whose ENTITY1 resolves to parent.entity
    /// node-ids). Threads through as `IngestOptions.parent_index_path`.
    #[arg(long, value_name = "DIR")]
    parent_index_path: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CatCountStrategyArg {
    None,
    Hybrid,
    Graduation,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum PhaseArg {
    Encode,
    PkMap,
    EdgesMap,
    Index,
}

impl From<PhaseArg> for diskgraph_ingest::IngestPhase {
    fn from(p: PhaseArg) -> Self {
        match p {
            PhaseArg::Encode => Self::Encode,
            PhaseArg::PkMap => Self::PkMap,
            PhaseArg::EdgesMap => Self::EdgesMap,
            PhaseArg::Index => Self::Index,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DestArg {
    Local,
    S3,
    Gcs,
    Adls,
}

/// Increase the soft open-file limit to the hard limit. The encode phase holds
/// one temp file open per (table, shard) — ~800+ on a 75-shard graph — which
/// exceeds common container defaults (Docker: 1024 soft / 524288 hard).
#[cfg(unix)]
fn increase_nofile_limit() {
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) != 0 || lim.rlim_cur >= lim.rlim_max {
            return;
        }
        let old = lim.rlim_cur;
        lim.rlim_cur = lim.rlim_max;
        if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) == 0 {
            tracing::info!(old, new = lim.rlim_max, "raised soft RLIMIT_NOFILE");
        } else {
            tracing::warn!(soft = old, "could not raise soft RLIMIT_NOFILE");
        }
    }
}

#[cfg(not(unix))]
fn increase_nofile_limit() {}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,pprof=warn")
        .init();
    increase_nofile_limit();
    let args = Args::parse();
    // Answer the runner's thread-count query before opening any source.
    if args.print_resolved_threads {
        println!("{}", diskgraph_ingest::default_worker_threads());
        return Ok(());
    }
    if !args.worker {
        let diagnostic_paths = local_diagnostic_paths(&args);
        if let Some((final_path, worker_path)) = &diagnostic_paths {
            remove_diagnostic(final_path);
            remove_diagnostic(worker_path);
        }
        let result = supervise_ingest(&args);
        if let Some((final_path, worker_path)) = &diagnostic_paths {
            remove_diagnostic(worker_path);
            if result.is_ok() {
                remove_diagnostic(final_path);
            } else if !final_path.is_file() {
                let diagnostic = render_process_fallback(
                    0,
                    "supervisor",
                    None,
                    None,
                    false,
                    false,
                );
                persist_diagnostic(final_path, &diagnostic);
            }
        }
        return result;
    }
    let worker_diagnostic_path = if matches!(args.dest, DestArg::Local) {
        args.output
            .as_deref()
            .map(PathBuf::from)
            .map(|output| output.join(WORKER_DIAGNOSTIC_FILENAME))
    } else {
        None
    };
    if let Some(path) = &worker_diagnostic_path {
        remove_diagnostic(path);
    }
    let worker_output = args.output.as_deref().map(PathBuf::from);
    match run_worker(args) {
        Ok(()) => {
            if let Some(path) = &worker_diagnostic_path {
                remove_diagnostic(path);
            }
            Ok(())
        }
        Err(error) => {
            if let Some(path) = &worker_diagnostic_path {
                let attempt = std::env::var("DISKGRAPH_SUPERVISOR_ATTEMPT")
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .unwrap_or(1);
                let phase = worker_output
                    .as_deref()
                    .and_then(|output| diskgraph_ingest::first_incomplete_phase(output).ok())
                    .map_or("worker", |phase| phase.name());
                let diagnostic = render_diagnostic(&error, attempt, phase);
                persist_diagnostic(path, &diagnostic);
            }
            if diskgraph_ingest::is_non_retryable_ingest_error(&error) {
                eprintln!("Error: {error:?}");
                std::process::exit(NON_RETRYABLE_WORKER_EXIT_CODE);
            }
            Err(error)
        }
    }
}

fn run_worker(args: Args) -> Result<()> {
    // `source`/`output` are required unless --print-resolved-threads (handled above).
    let source = args.source.clone().expect("source is required");
    let output = args.output.clone().expect("output is required");
    let source_str = source.to_string_lossy();
    let source_path = if source.is_dir() {
        source.join("diskgraph_config.json")
    } else {
        source.clone()
    };
    if !is_cloud_url(&source_str) && !source_path.is_file() {
        return Err(anyhow::Error::new(RestrictedDiagnostic::new(
            "config.not_found",
            "DiskGraph ingest configuration was not found",
            "read_ingest_config",
            DiagnosticCategory::Io,
            false,
            anyhow::anyhow!("diskgraph config not found at {}", source_path.display()),
        )));
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
            let dir = args.profile.clone().unwrap_or_else(|| PathBuf::from("."));
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
    let peak_tracker = PeakMemTracker::start();
    let options = IngestOptions {
        profile_dir: args.profile.clone(),
        databricks_config_file: args.databricks_config_file.clone(),
        progress: build_progress(),
        parallel: args.parallel,
        index_mem_budget_bytes: args.index_mem_budget_bytes,
        peak_mem_stats: peak_tracker.as_ref().map(|t| Arc::clone(&t.stats)),
        col_stats: args.col_stats,
        cat_count_strategy: match args.cat_count_strategy {
            CatCountStrategyArg::None => diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            CatCountStrategyArg::Hybrid => {
                diskgraph_ingest::col_stats::CategoryCountStrategy::Hybrid
            }
            CatCountStrategyArg::Graduation => {
                diskgraph_ingest::col_stats::CategoryCountStrategy::Graduation
            }
        },
        row_subsample: args.row_subsample,
        resume_from_phase: supervisor_phase_override()
            .or_else(|| args.resume_from_phase.map(Into::into)),
        source_loader_threads: supervisor_thread_override("SOURCE_LOADER")
            .or(args.context_threads),
        source_encoder_threads: supervisor_thread_override("SOURCE_ENCODER")
            .or(args.encode_threads),
        pk_map_threads: supervisor_thread_override("PK_MAP").or(args.pk_map_threads),
        edges_map_threads: supervisor_thread_override("EDGES_MAP").or(args.edges_map_threads),
        index_threads: supervisor_thread_override("INDEX").or(args.index_threads),
        parent_index_path: args.parent_index_path.clone(),
        resolve_columns: Vec::new(),
    };
    let dest = match args.dest {
        DestArg::Local => Dest::Local,
        DestArg::S3 => Dest::S3,
        DestArg::Gcs => Dest::Gcs,
        DestArg::Adls => Dest::Adls,
    };
    // Both branches funnel through `ingest_with_stats_from_source` so the
    // optional dataplane / success-marker args are plumbed uniformly.
    // The local-source branch inlines what the public `ingest_with_stats`
    // wrapper used to do (read the metadata file + derive base_dir);
    // we don't call it directly because its signature doesn't take the
    // dataplane args and changing it would touch every test that uses it.
    let dataplane = args.output_dataplane_dir.as_deref();
    let marker = args.success_marker_content.as_deref();
    let (_metadata, stats) = if is_cloud_url(&source_str) {
        let config_url = ensure_config_url(&source_str);
        let metadata_bytes = restrict(
            download_cloud_config_bytes(&config_url),
            "object_store.config_read_failed",
            "DiskGraph could not read its ingest configuration from object storage",
            "read_ingest_config",
            DiagnosticCategory::Provider,
            true,
        )?;
        let source: GraphConfig = serde_json::from_slice(&metadata_bytes)?;
        ingest_with_stats_from_source(
            source, None, &output, dest, options, dataplane, marker,
        )?
    } else {
        let metadata_bytes = restrict(
            fs::read(&source_path)
                .map_err(|error| anyhow::anyhow!("read {}: {error}", source_path.display())),
            "filesystem.config_read_failed",
            "DiskGraph could not read its ingest configuration",
            "read_ingest_config",
            DiagnosticCategory::Io,
            false,
        )?;
        let source: GraphConfig = serde_json::from_slice(&metadata_bytes)?;
        let base_dir = source_path.parent();
        ingest_with_stats_from_source(
            source, base_dir, &output, dest, options, dataplane, marker,
        )?
    };
    drop(peak_tracker);
    let _ = mem_watchdog;
    let publish_outcome = match maybe_publish_stats(&args, &stats, &source_str, dest) {
        Ok(outcome) => outcome,
        Err(err) => {
            if args.stats_required {
                return Err(anyhow::Error::new(RestrictedDiagnostic::new(
                    "stats.publish_failed",
                    "DiskGraph could not publish required ingest statistics",
                    "publish_ingest_stats",
                    DiagnosticCategory::Provider,
                    true,
                    err,
                )));
            }
            tracing::warn!("stats publish failed: {err}");
            None
        }
    };
    println!("{} {}", color("Output:", Color::CyanBold), stats.output);
    println!(
        "{} {} ({} nodes)",
        color("Node tables:", Color::CyanBold),
        stats.node_tables,
        stats.total_nodes
    );
    println!(
        "{} {} ({} edges)",
        color("Edge relations:", Color::CyanBold),
        stats.edge_relations,
        stats.total_edges
    );
    if !stats.node_counts.is_empty() {
        println!("{}", color("Nodes per table:", Color::CyanBold));
        let mut entries: Vec<_> = stats.node_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", color(name, Color::Blue), count);
        }
    }
    if !stats.edge_counts.is_empty() {
        println!("{}", color("Edges per relation:", Color::CyanBold));
        let mut entries: Vec<_> = stats.edge_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", color(name, Color::Blue), count);
        }
    }
    if stats.pk_duplicate_total > 0 {
        println!("{}", color("Duplicate primary keys:", Color::YellowBold));
        println!(
            "  {} {}",
            color("Total skipped:", Color::YellowBold),
            stats.pk_duplicate_total
        );
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
            println!("  {}: {}", color(name, Color::YellowBold), count);
            if let Some(samples) = samples_by_table.get(name) {
                if !samples.is_empty() {
                    println!(
                        "    {} {}",
                        color("Samples:", Color::YellowBold),
                        samples.join(", ")
                    );
                }
            }
        }
    }
    if !stats.dropped_rows_counts.is_empty() {
        println!("{}", color("Dropped rows:", Color::YellowBold));
        let mut entries: Vec<_> = stats.dropped_rows_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", color(name, Color::YellowBold), count);
        }
    }
    if stats.key_parse_failures > 0 || stats.key_lookup_failures > 0 {
        println!("{}", color("Key matching failures:", Color::YellowBold));
        println!(
            "  {} {}",
            color("Parse failures:", Color::YellowBold),
            stats.key_parse_failures
        );
        println!(
            "  {} {}",
            color("Lookup misses:", Color::YellowBold),
            stats.key_lookup_failures
        );
        if !stats.key_failures_by_context.is_empty() {
            println!("  {}", color("By context:", Color::YellowBold));
            let mut entries: Vec<_> = stats.key_failures_by_context.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (name, counts) in entries {
                if counts.parse_failures == 0 && counts.lookup_failures == 0 {
                    continue;
                }
                println!(
                    "    {}: parse {}, lookup {}",
                    color(name, Color::YellowBold),
                    counts.parse_failures,
                    counts.lookup_failures
                );
            }
        }
        if !stats.key_failure_samples.is_empty() {
            println!("  {}", color("Examples:", Color::YellowBold));
            for sample in &stats.key_failure_samples {
                let (label_part, value_part) = split_example(sample);
                print!("    {}", color(label_part, Color::YellowBold));
                println!("{}", value_part);
            }
        }
    }
    if let Some(trimmed) = stats.trimmed_strings {
        println!(
            "{} {}",
            color("Trimmed strings (>32KB):", Color::YellowBold),
            trimmed
        );
    }
    let total_bytes = stats
        .perf
        .bytes_read
        .saturating_add(stats.perf.bytes_written);
    let wall_secs = (stats.perf.wall_time_ms as f64) / 1000.0;
    let bytes_per_sec = if wall_secs > 0.0 {
        (total_bytes as f64) / wall_secs
    } else {
        0.0
    };
    let baseline = publish_outcome.as_ref().and_then(|outcome| {
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
    println!("{}", color("Performance:", Color::GreenBold));
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
            "  {} min of last 30 runs for {} on {} (git {})",
            color("Baseline:", Color::GreenBold),
            outcome.summary.dataset,
            instance,
            git
        );
    }
    println!(
        "  {} {}{}",
        color("Wall time:", Color::GreenBold),
        format_duration(stats.perf.wall_time_ms),
        format_delta(
            stats.perf.wall_time_ms,
            baseline.as_ref().map(|b| b.wall_time_ms)
        )
    );
    println!(
        "  {} {}",
        color("Bytes read:", Color::GreenBold),
        format_bytes(stats.perf.bytes_read)
    );
    println!(
        "  {} {}",
        color("Bytes written:", Color::GreenBold),
        format_bytes(stats.perf.bytes_written)
    );
    println!(
        "  {} {}",
        color("Total bytes:", Color::GreenBold),
        format_bytes(total_bytes)
    );
    println!(
        "  {} {}/s",
        color("Throughput:", Color::GreenBold),
        format_bytes(bytes_per_sec as u64)
    );
    if stats.perf.temp_bytes_total > 0 {
        println!(
            "  {} {}",
            color("Temp bytes (edges):", Color::GreenBold),
            format_bytes(stats.perf.temp_bytes_total)
        );
        if stats.perf.temp_fkeys_bytes > 0 {
            println!(
                "    {} {}",
                color("fkeys:", Color::GreenBold),
                format_bytes(stats.perf.temp_fkeys_bytes)
            );
        }
        if stats.perf.temp_edges_src_bytes > 0 {
            println!(
                "    {} {}",
                color("edges_src:", Color::GreenBold),
                format_bytes(stats.perf.temp_edges_src_bytes)
            );
        }
        if stats.perf.temp_edges_map_bytes > 0 {
            println!(
                "    {} {}",
                color("edges_map:", Color::GreenBold),
                format_bytes(stats.perf.temp_edges_map_bytes)
            );
        }
    }
    if stats.perf.temp_node_data_bytes > 0 {
        println!(
            "  {} {}",
            color("Temp bytes (nodes):", Color::GreenBold),
            format_bytes(stats.perf.temp_node_data_bytes)
        );
    }
    println!("  {}", color("Stage timings (wall):", Color::GreenBold));
    println!(
        "    {} {}{}",
        color("context:", Color::GreenBold),
        format_duration(stats.perf.context_wall_ms),
        format_delta(
            stats.perf.context_wall_ms,
            baseline.as_ref().map(|b| b.context_wall_ms)
        )
    );
    println!(
        "    {} {}{}",
        color("encode:", Color::GreenBold),
        format_duration(stats.perf.encode_wall_ms),
        format_delta(
            stats.perf.encode_wall_ms,
            baseline.as_ref().map(|b| b.encode_wall_ms)
        )
    );
    println!(
        "    {} {}{}",
        color("pk_map:", Color::GreenBold),
        format_duration(stats.perf.pk_map_wall_ms),
        format_delta(
            stats.perf.pk_map_wall_ms,
            baseline.as_ref().map(|b| b.pk_map_wall_ms)
        )
    );
    println!(
        "    {} {}{}",
        color("edges_map:", Color::GreenBold),
        format_duration(stats.perf.edges_map_wall_ms),
        format_delta(
            stats.perf.edges_map_wall_ms,
            baseline.as_ref().map(|b| b.edges_map_wall_ms)
        )
    );
    println!(
        "    {} {}{}",
        color("index:", Color::GreenBold),
        format_duration(stats.perf.index_wall_ms),
        format_delta(
            stats.perf.index_wall_ms,
            baseline.as_ref().map(|b| b.index_wall_ms)
        )
    );
    println!(
        "    {} {}{}",
        color("finalize:", Color::GreenBold),
        format_duration(stats.perf.finalize_wall_ms),
        format_delta(
            stats.perf.finalize_wall_ms,
            baseline.as_ref().map(|b| b.finalize_wall_ms)
        )
    );
    println!(
        "{} {}",
        color("Next:", Color::CyanBold),
        explore_command_hint(&args.dest, &stats.output)
    );
    Ok(())
}

const SUPERVISOR_RESUME_ENV: &str = "DISKGRAPH_SUPERVISOR_RESUME_PHASE";
// EX_CONFIG: the worker validated its input and a retry cannot change it.
const NON_RETRYABLE_WORKER_EXIT_CODE: i32 = 78;

fn local_diagnostic_paths(args: &Args) -> Option<(PathBuf, PathBuf)> {
    if !matches!(args.dest, DestArg::Local) {
        return None;
    }
    let output = PathBuf::from(args.output.as_deref()?);
    Some((
        output.join(DIAGNOSTIC_FILENAME),
        output.join(WORKER_DIAGNOSTIC_FILENAME),
    ))
}

fn remove_diagnostic(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => tracing::warn!("failed to remove a stale DiskGraph diagnostic"),
    }
}

fn persist_diagnostic(path: &Path, diagnostic: &DiagnosticEnvelope) {
    if write_diagnostic_atomically(path, diagnostic).is_err() {
        tracing::warn!("failed to persist a DiskGraph diagnostic");
    }
}

fn exit_status_parts(status: ExitStatus) -> (Option<i32>, Option<i32>) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        (status.code(), status.signal())
    }
    #[cfg(not(unix))]
    {
        (status.code(), None)
    }
}

fn publish_terminal_diagnostic(
    final_path: &Path,
    worker_path: &Path,
    attempt: u32,
    phase: diskgraph_ingest::IngestPhase,
    status: ExitStatus,
    oom: bool,
) {
    remove_diagnostic(final_path);
    if fs::rename(worker_path, final_path).is_ok() {
        return;
    }
    let (exit_code, signal) = exit_status_parts(status);
    let fallback = render_process_fallback(
        attempt,
        phase.name(),
        exit_code,
        signal,
        oom,
        false,
    );
    persist_diagnostic(final_path, &fallback);
}

fn supervisor_thread_override(phase: &str) -> Option<usize> {
    std::env::var(format!("DISKGRAPH_SUPERVISOR_{phase}_THREADS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn supervisor_phase_override() -> Option<diskgraph_ingest::IngestPhase> {
    std::env::var(SUPERVISOR_RESUME_ENV)
        .ok()
        .and_then(|value| diskgraph_ingest::IngestPhase::from_name(&value))
}

fn halve_threads(value: usize) -> usize {
    value.div_ceil(2).max(1)
}

fn backoff_source_loader(configured: Option<usize>, automatic: usize) -> usize {
    halve_threads(configured.unwrap_or(automatic))
}

fn is_oom_exit(status: ExitStatus) -> bool {
    if status.code() == Some(137) {
        return true;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.signal() == Some(libc::SIGKILL)
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn is_non_retryable_exit(status: ExitStatus) -> bool {
    status.code() == Some(NON_RETRYABLE_WORKER_EXIT_CODE)
}

fn retry_phase_for_exit(
    oom: bool,
    non_retryable: bool,
    phase: diskgraph_ingest::IngestPhase,
) -> Option<diskgraph_ingest::IngestPhase> {
    (!non_retryable && (oom || phase == diskgraph_ingest::IngestPhase::Encode)).then_some(phase)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PublicationMarkerIdentity {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

fn publication_marker_identity(root: &std::path::Path) -> Option<PublicationMarkerIdentity> {
    let metadata =
        fs::metadata(root.join(diskgraph_ingest::success_marker::SUCCESS_MARKER_FILENAME)).ok()?;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    Some(PublicationMarkerIdentity {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
    })
}

fn worker_committed_publication(
    root: &std::path::Path,
    before: Option<&PublicationMarkerIdentity>,
) -> bool {
    publication_marker_identity(root)
        .as_ref()
        .is_some_and(|after| Some(after) != before)
}

fn supervise_ingest(args: &Args) -> Result<()> {
    let executable = std::env::current_exe()?;
    let raw_args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let default_threads = diskgraph_ingest::default_worker_threads().max(1);
    let default_source_loader = diskgraph_ingest::databricks::default_download_parallelism();
    let mut source_loader = args.context_threads.map(|threads| threads.max(1));
    let mut source_encoder = args.encode_threads.unwrap_or(default_threads).max(1);
    let mut pk_map = args.pk_map_threads.unwrap_or(default_threads).max(1);
    let mut edges_map = args.edges_map_threads.unwrap_or(default_threads).max(1);
    let mut index = args.index_threads.unwrap_or(default_threads).max(1);
    let output = PathBuf::from(args.output.as_deref().expect("output is required"));
    let diagnostic_path = output.join(DIAGNOSTIC_FILENAME);
    let worker_diagnostic_path = output.join(WORKER_DIAGNOSTIC_FILENAME);
    let mut resume_override = args.resume_from_phase.map(Into::into);
    if resume_override.is_none() && matches!(args.dest, DestArg::Local) {
        resume_override = diskgraph_ingest::initial_resume_phase(&output)?;
    }

    for attempt in 0..=args.oom_retries {
        let mut command = Command::new(&executable);
        command
            .args(&raw_args)
            .arg("--worker")
            .env(
                "DISKGRAPH_SUPERVISOR_SOURCE_ENCODER_THREADS",
                source_encoder.to_string(),
            )
            .env("DISKGRAPH_SUPERVISOR_PK_MAP_THREADS", pk_map.to_string())
            .env(
                "DISKGRAPH_SUPERVISOR_EDGES_MAP_THREADS",
                edges_map.to_string(),
            )
            .env("DISKGRAPH_SUPERVISOR_INDEX_THREADS", index.to_string())
            .env("DISKGRAPH_SUPERVISOR_ATTEMPT", (attempt + 1).to_string())
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(source_loader) = source_loader {
            command.env(
                "DISKGRAPH_SUPERVISOR_SOURCE_LOADER_THREADS",
                source_loader.to_string(),
            );
        } else {
            command.env_remove("DISKGRAPH_SUPERVISOR_SOURCE_LOADER_THREADS");
        }
        if let Some(phase) = resume_override {
            command.env(SUPERVISOR_RESUME_ENV, phase.name());
        } else {
            command.env_remove(SUPERVISOR_RESUME_ENV);
        }

        #[cfg(unix)]
        unsafe {
            use std::os::unix::process::CommandExt;
            command.pre_exec(|| {
                #[cfg(target_os = "linux")]
                {
                    let supervisor_pid = libc::getppid();
                    if supervisor_pid == 1 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "ingest supervisor exited before worker startup",
                        ));
                    }
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != supervisor_pid {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "ingest supervisor exited during worker startup",
                        ));
                    }
                }
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        // A marker left by an earlier invocation is not evidence that this
        // worker published its requested source/configuration. Capture its
        // identity immediately before spawning; publication atomically
        // replaces the marker, which gives the current attempt a new identity.
        let publication_before = publication_marker_identity(&output);
        let mut child = command.spawn()?;
        let child_pid = child.id();
        let status = child.wait()?;
        if status.success() {
            remove_diagnostic(&diagnostic_path);
            remove_diagnostic(&worker_diagnostic_path);
            return Ok(());
        }

        if worker_committed_publication(&output, publication_before.as_ref()) {
            // Publication is the terminal commit. A worker killed between
            // that commit and process exit must never reopen Encode or mutate
            // a graph already advertised as complete.
            remove_diagnostic(&diagnostic_path);
            remove_diagnostic(&worker_diagnostic_path);
            return Ok(());
        }

        #[cfg(unix)]
        unsafe {
            // The worker is a session/process-group leader; terminate any
            // helper that survived the worker's OOM or abrupt exit.
            libc::kill(-(child_pid as i32), libc::SIGKILL);
        }

        let oom = is_oom_exit(status);
        let incomplete_phase = diskgraph_ingest::first_incomplete_phase(&output)?;
        let Some(phase) =
            retry_phase_for_exit(oom, is_non_retryable_exit(status), incomplete_phase)
        else {
            publish_terminal_diagnostic(
                &diagnostic_path,
                &worker_diagnostic_path,
                (attempt + 1) as u32,
                incomplete_phase,
                status,
                oom,
            );
            if is_non_retryable_exit(status) {
                std::process::exit(NON_RETRYABLE_WORKER_EXIT_CODE);
            }
            anyhow::bail!("DiskGraph ingest worker exited with {status}");
        };
        if attempt == args.oom_retries {
            if phase == diskgraph_ingest::IngestPhase::Encode {
                diskgraph_ingest::discard_provisional_encode_outputs(&output)?;
            }
            publish_terminal_diagnostic(
                &diagnostic_path,
                &worker_diagnostic_path,
                (attempt + 1) as u32,
                phase,
                status,
                oom,
            );
            anyhow::bail!(
                "DiskGraph ingest exhausted {} worker retries",
                args.oom_retries
            );
        }

        remove_diagnostic(&worker_diagnostic_path);

        match phase {
            diskgraph_ingest::IngestPhase::Encode => {
                source_loader = Some(backoff_source_loader(source_loader, default_source_loader));
                source_encoder = halve_threads(source_encoder);
            }
            diskgraph_ingest::IngestPhase::PkMap => pk_map = halve_threads(pk_map),
            diskgraph_ingest::IngestPhase::EdgesMap => edges_map = halve_threads(edges_map),
            diskgraph_ingest::IngestPhase::Index => index = halve_threads(index),
        }
        resume_override = Some(phase);
        tracing::warn!(
            attempt = attempt + 1,
            phase = phase.name(),
            source_loader = source_loader.unwrap_or(default_source_loader),
            source_encoder,
            pk_map,
            edges_map,
            index,
            oom,
            "ingest worker failed; restarting from first incomplete phase"
        );
    }
    unreachable!("supervisor loop always returns")
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
        .unwrap_or_else(|| dataset_from_source(std::path::Path::new(source_str), source_str));
    let run_id = args
        .stats_run_id
        .clone()
        .or_else(|| std::env::var("DISKGRAPH_STATS_RUN_ID").ok())
        .unwrap_or_else(|| default_run_id(&run_id_user(), &run_id_git_prefix()));
    let stats_dest = args.stats_dest.clone();
    let config = PublishConfig {
        source: source_str.to_string(),
        output: args.output.clone().expect("output is required"),
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
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "nogit".to_string(),
    };
    sha.chars().take(8).collect()
}

struct PeakMemTracker {
    stop: Arc<AtomicBool>,
    stats: Arc<PeakMemStats>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PeakMemTracker {
    fn start() -> Option<Self> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PeakMemStats::new());
        let thread_stop = Arc::clone(&stop);
        let thread_stats = Arc::clone(&stats);
        let handle = thread::spawn(move || peak_mem_loop(thread_stop, thread_stats));
        Some(Self {
            stop,
            stats,
            handle: Some(handle),
        })
    }
}

impl Drop for PeakMemTracker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn peak_mem_loop(stop: Arc<AtomicBool>, stats: Arc<PeakMemStats>) {
    let interval = Duration::from_millis(500);
    while !stop.load(Ordering::Relaxed) {
        if let Some(bytes) = rss_anon_bytes() {
            stats.update_overall(bytes);
            let stage = stats.stage_index();
            if stage < PEAK_MEM_STAGE_COUNT {
                stats.update_stage(stage, bytes);
            }
        }
        thread::sleep(interval);
    }
}

#[cfg(target_os = "linux")]
fn rss_anon_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
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
fn parse_kb_line(value: &str) -> Option<u64> {
    let mut iter = value.split_whitespace();
    let num = iter.next()?.parse::<u64>().ok()?;
    Some(num.saturating_mul(1024))
}

fn format_delta(current_ms: u64, baseline_ms: Option<u64>) -> String {
    let Some(baseline_ms) = baseline_ms else {
        return String::new();
    };
    if baseline_ms == 0 {
        return String::new();
    }
    let delta_ms = current_ms as i64 - baseline_ms as i64;
    let pct = (delta_ms as f64 / baseline_ms as f64) * 100.0;
    let (color, reset) = if pct <= 5.0 {
        ("\x1b[32m", "\x1b[0m")
    } else if pct <= 15.0 {
        ("\x1b[33m", "\x1b[0m")
    } else {
        ("\x1b[31m", "\x1b[0m")
    };
    let (sign, abs_ms, pct_label) = if delta_ms < 0 {
        ("-", delta_ms.unsigned_abs(), format!("{pct:.1}%"))
    } else {
        ("+", delta_ms as u64, format!("+{pct:.1}%"))
    };
    format!(
        "  {color}{sign}{} ({pct_label}){reset}",
        format_duration(abs_ms),
        reset = reset
    )
}

enum Color {
    CyanBold,
    GreenBold,
    YellowBold,
    Blue,
}

fn color(text: &str, color: Color) -> String {
    let (code, reset) = match color {
        Color::CyanBold => ("\x1b[36;1m", "\x1b[0m"),
        Color::GreenBold => ("\x1b[32;1m", "\x1b[0m"),
        Color::YellowBold => ("\x1b[33;1m", "\x1b[0m"),
        Color::Blue => ("\x1b[34m", "\x1b[0m"),
    };
    format!("{code}{text}{reset}")
}

fn split_example(sample: &str) -> (&str, &str) {
    if let Some((_label, value)) = sample.rsplit_once(": ") {
        let label_end = sample.len().saturating_sub(value.len());
        let (label_part, value_part) = sample.split_at(label_end);
        (label_part, value_part)
    } else {
        (sample, "")
    }
}

fn dataset_from_source(source: &std::path::Path, source_str: &str) -> String {
    if is_cloud_url(source_str) {
        let trimmed = source_str.trim_end_matches('/');
        return trimmed.rsplit('/').next().unwrap_or(trimmed).to_string();
    }
    if source.is_dir() {
        return source
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset")
            .to_string();
    }
    if let Some(name) = source.file_name().and_then(|n| n.to_str()) {
        if name != "diskgraph_config.json" {
            if let Some(stripped) = name.strip_suffix(".json") {
                if !stripped.is_empty() {
                    return stripped.to_string();
                }
            }
            return name.to_string();
        }
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
    // Absolute threshold (bytes) when set via env var DISKGRAPH_MEM_KILL_BYTES;
    // otherwise fall back to fraction of total RAM. Default fraction lowered
    // from 0.80 → 0.70 so the watchdog fires before OOM on a no-swap host.
    const MEM_WATCHDOG_THRESHOLD: f64 = 0.70;
    const REPEAT_DUMP_INTERVAL: Duration = Duration::from_secs(10);

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
        // Absolute kill threshold can be set via DISKGRAPH_MEM_KILL_BYTES (e.g.
        // 350GiB → 375809638400). Falls back to MEM_WATCHDOG_THRESHOLD * total.
        let kill_bytes: u64 = std::env::var("DISKGRAPH_MEM_KILL_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| (total_bytes as f64 * MEM_WATCHDOG_THRESHOLD) as u64);
        // Optional dump-only threshold (lower than kill). Defaults to 0.85 *
        // kill_bytes so we get an early heap snapshot before exiting.
        let dump_bytes: u64 = std::env::var("DISKGRAPH_MEM_DUMP_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or((kill_bytes as f64 * 0.85) as u64);
        tracing::info!(
            kill_bytes,
            dump_bytes,
            total_bytes,
            "mem_watchdog: armed (DISKGRAPH_MEM_KILL_BYTES / DISKGRAPH_MEM_DUMP_BYTES override env)"
        );
        let mut last_dump = std::time::Instant::now()
            .checked_sub(REPEAT_DUMP_INTERVAL)
            .unwrap_or_else(std::time::Instant::now);
        while !stop.load(Ordering::Relaxed) {
            if let Some(rss_anon) = rss_anon_bytes() {
                if rss_anon >= kill_bytes {
                    tracing::error!(
                        rss_anon,
                        kill_bytes,
                        "mem_watchdog: RSS-anon exceeded kill threshold; dumping heap and exiting"
                    );
                    if let Err(err) = dump_heap_profile(&dump_dir) {
                        tracing::warn!("final heap profile dump failed: {err}");
                    }
                    // Flush tracing buffers before exiting.
                    std::thread::sleep(Duration::from_millis(100));
                    std::process::exit(137);
                }
                if rss_anon >= dump_bytes
                    && last_dump.elapsed() >= REPEAT_DUMP_INTERVAL
                {
                    tracing::warn!(
                        rss_anon,
                        dump_bytes,
                        "mem_watchdog: above dump threshold; capturing heap snapshot"
                    );
                    if let Err(err) = dump_heap_profile(&dump_dir) {
                        tracing::warn!("heap profile dump failed: {err}");
                    }
                    last_dump = std::time::Instant::now();
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
        let conf = unsafe {
            jemalloc_ctl::raw::read::<*const std::os::raw::c_char>(b"config.malloc_conf\0")
        };
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

#[cfg(test)]
mod supervisor_tests {
    use super::*;

    #[test]
    fn thread_backoff_is_monotonic_and_never_zero() {
        assert_eq!(halve_threads(64), 32);
        assert_eq!(halve_threads(3), 2);
        assert_eq!(halve_threads(2), 1);
        assert_eq!(halve_threads(1), 1);
        assert_eq!(backoff_source_loader(None, 44), 22);
        assert_eq!(backoff_source_loader(Some(7), 44), 4);
    }

    #[test]
    fn restart_phase_is_the_first_missing_durable_marker() {
        let directory = tempfile::tempdir().unwrap();
        let temp = directory.path().join("_tmp");
        fs::create_dir_all(&temp).unwrap();
        assert_eq!(
            diskgraph_ingest::first_incomplete_phase(directory.path()).unwrap(),
            diskgraph_ingest::IngestPhase::Encode
        );

        for expected in [
            diskgraph_ingest::IngestPhase::PkMap,
            diskgraph_ingest::IngestPhase::EdgesMap,
            diskgraph_ingest::IngestPhase::Index,
        ] {
            let predecessor = expected.predecessors().last().unwrap();
            fs::write(
                temp.join(format!("_phase_complete_{}", predecessor.name())),
                b"complete",
            )
            .unwrap();
            assert_eq!(
                diskgraph_ingest::first_incomplete_phase(directory.path()).unwrap(),
                expected
            );
        }

        fs::write(temp.join("_phase_complete_index"), b"complete").unwrap();
        assert_eq!(
            diskgraph_ingest::first_incomplete_phase(directory.path()).unwrap(),
            diskgraph_ingest::IngestPhase::Index,
            "publication failure reruns Index from its durable predecessor"
        );
    }

    #[test]
    fn initial_supervisor_resume_requires_a_valid_incomplete_encode_chain() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let temp = root.join("_tmp");
        fs::create_dir_all(&temp).unwrap();

        // Provisional Encode output is not resumable and will be wiped by the
        // fresh worker path.
        assert_eq!(diskgraph_ingest::initial_resume_phase(root).unwrap(), None);

        fs::write(temp.join("_phase_complete_encode"), b"identity").unwrap();
        assert_eq!(
            diskgraph_ingest::initial_resume_phase(root).unwrap(),
            Some(diskgraph_ingest::IngestPhase::PkMap)
        );

        fs::write(temp.join("_phase_complete_pk_map"), b"").unwrap();
        assert_eq!(
            diskgraph_ingest::initial_resume_phase(root).unwrap(),
            Some(diskgraph_ingest::IngestPhase::EdgesMap)
        );

        fs::write(temp.join("_phase_complete_index"), b"").unwrap();
        let error = diskgraph_ingest::initial_resume_phase(root).unwrap_err();
        assert!(error.to_string().contains("after missing 'edges_map'"));

        fs::write(temp.join("_phase_complete_edges_map"), b"").unwrap();
        assert_eq!(
            diskgraph_ingest::initial_resume_phase(root).unwrap(),
            Some(diskgraph_ingest::IngestPhase::Index),
            "a crash after Index but before publication reruns finalization",
        );
    }

    #[test]
    fn initial_supervisor_does_not_resume_a_published_snapshot() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let temp = root.join("_tmp");
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("_phase_complete_encode"), b"identity").unwrap();
        fs::write(
            root.join(diskgraph_ingest::success_marker::SUCCESS_MARKER_FILENAME),
            b"published",
        )
        .unwrap();

        assert_eq!(diskgraph_ingest::initial_resume_phase(root).unwrap(), None);
    }

    #[test]
    fn only_retryable_encode_failures_and_ooms_restart() {
        use diskgraph_ingest::IngestPhase::{EdgesMap, Encode, Index, PkMap};

        assert_eq!(retry_phase_for_exit(false, false, Encode), Some(Encode));
        assert_eq!(retry_phase_for_exit(false, true, Encode), None);
        assert_eq!(retry_phase_for_exit(false, false, PkMap), None);
        assert_eq!(retry_phase_for_exit(false, false, EdgesMap), None);
        assert_eq!(retry_phase_for_exit(false, false, Index), None);
        assert_eq!(retry_phase_for_exit(true, false, PkMap), Some(PkMap));
    }

    #[test]
    fn only_a_marker_replaced_by_the_current_worker_counts_as_publication() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory
            .path()
            .join(diskgraph_ingest::success_marker::SUCCESS_MARKER_FILENAME);
        fs::write(&marker, b"stale").unwrap();
        let before = publication_marker_identity(directory.path()).unwrap();

        assert!(!worker_committed_publication(
            directory.path(),
            Some(&before)
        ));

        let replacement = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        fs::write(replacement.path(), b"fresh").unwrap();
        replacement.persist(&marker).unwrap();
        assert!(worker_committed_publication(
            directory.path(),
            Some(&before)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn oom_exit_classification_accepts_sigkill_and_137_only() {
        use std::os::unix::process::ExitStatusExt;

        assert!(is_oom_exit(ExitStatus::from_raw(libc::SIGKILL)));
        assert!(is_oom_exit(ExitStatus::from_raw(137 << 8)));
        assert!(!is_oom_exit(ExitStatus::from_raw(1 << 8)));
        assert!(is_non_retryable_exit(ExitStatus::from_raw(
            NON_RETRYABLE_WORKER_EXIT_CODE << 8
        )));
        assert!(!is_non_retryable_exit(ExitStatus::from_raw(1 << 8)));
    }

    #[cfg(unix)]
    #[test]
    fn terminal_publication_promotes_worker_detail_or_synthesizes_process_fallback() {
        use std::os::unix::process::ExitStatusExt;

        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join(DIAGNOSTIC_FILENAME);
        let worker_path = directory.path().join(WORKER_DIAGNOSTIC_FILENAME);
        let error = anyhow::anyhow!("actionable internal invariant");
        let diagnostic = render_diagnostic(&error, 1, "encode");
        write_diagnostic_atomically(&worker_path, &diagnostic).unwrap();

        publish_terminal_diagnostic(
            &final_path,
            &worker_path,
            1,
            diskgraph_ingest::IngestPhase::Encode,
            ExitStatus::from_raw(1 << 8),
            false,
        );

        let promoted: serde_json::Value =
            serde_json::from_slice(&fs::read(&final_path).unwrap()).unwrap();
        assert_eq!(promoted["visibility"], "ordinary");
        assert_eq!(promoted["summary"], "actionable internal invariant");
        assert!(!worker_path.exists());

        remove_diagnostic(&final_path);
        publish_terminal_diagnostic(
            &final_path,
            &worker_path,
            2,
            diskgraph_ingest::IngestPhase::PkMap,
            ExitStatus::from_raw(libc::SIGKILL),
            true,
        );
        let fallback: serde_json::Value =
            serde_json::from_slice(&fs::read(&final_path).unwrap()).unwrap();
        assert_eq!(fallback["visibility"], "restricted");
        assert_eq!(fallback["code"], "ingest.worker_memory_pressure");
        assert_eq!(fallback["fields"]["signal"], libc::SIGKILL);
    }
}
