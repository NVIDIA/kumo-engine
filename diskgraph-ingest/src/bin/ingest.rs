use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{
    ingest_with_stats, Dest, IngestOptions, ProgressCallback, ProgressEvent, ProgressKind,
    ProgressPhase,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ahash::AHashMap as HashMap;
use std::fs;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(author, version, about = "Ingest raw CSV/Parquet into diskgraph index format.")]
struct Args {
    /// Path to source_metadata.json
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
    let source_path = if args.source.is_dir() {
        args.source.join("source_metadata.json")
    } else {
        args.source.clone()
    };
    if !source_path.is_file() {
        anyhow::bail!(
            "source metadata not found at {}",
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
    let options = IngestOptions {
        profile_dir: args.profile.clone(),
        progress: build_progress(),
    };
    let dest = match args.dest {
        DestArg::Local => Dest::Local,
        DestArg::S3 => Dest::S3,
        DestArg::Gcs => Dest::Gcs,
        DestArg::Adls => Dest::Adls,
    };
    let (_metadata, stats) = ingest_with_stats(&source_path, &args.output, dest, options)?;
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
        let mut entries: Vec<_> = stats.pk_duplicate_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
            if let Some(samples) = stats.pk_duplicate_samples.get(name) {
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
    println!("Performance:");
    println!("  Wall time: {}", format_duration(stats.perf.wall_time_ms));
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
    println!("  Stage timings (wall):");
    println!(
        "    context: {}",
        format_duration(stats.perf.context_wall_ms)
    );
    println!(
        "    shuffle: {}",
        format_duration(stats.perf.shuffle_wall_ms)
    );
    println!(
        "    pk_map: {}",
        format_duration(stats.perf.pk_map_wall_ms)
    );
    println!(
        "    edges: {}",
        format_duration(stats.perf.build_wall_ms)
    );
    println!(
        "    index: {}",
        format_duration(stats.perf.index_wall_ms)
    );
    println!(
        "    finalize: {}",
        format_duration(stats.perf.finalize_wall_ms)
    );
    println!("  Stage timings (summed across threads):");
    println!(
        "    context: {} ({} tasks)",
        format_duration(stats.perf.context_task_ms),
        stats.perf.context_tasks
    );
    println!(
        "    shuffle: {} ({} tasks)",
        format_duration(stats.perf.shuffle_task_ms),
        stats.perf.shuffle_tasks
    );
    println!(
        "    pk_map: {} ({} tasks)",
        format_duration(stats.perf.pk_map_task_ms),
        stats.perf.pk_map_tasks
    );
    println!(
        "    edges: {} ({} tasks)",
        format_duration(stats.perf.build_task_ms),
        stats.perf.build_tasks
    );
    println!(
        "    index: {} ({} tasks)",
        format_duration(stats.perf.index_task_ms),
        stats.perf.index_tasks
    );
    if let Some(profile_dir) = args.profile.as_ref() {
        let mut profiles = Vec::new();
        for name in ["context", "shuffle", "pk_map", "edges", "index"] {
            let svg = profile_dir.join(format!("{name}-profile.svg"));
            let pb = profile_dir.join(format!("{name}-profile.pb"));
            if svg.exists() {
                profiles.push(svg);
            }
            if pb.exists() {
                profiles.push(pb);
            }
        }
        if !profiles.is_empty() {
            println!("Profiles:");
            for path in &profiles {
                println!("  {}", path.display());
            }
            let mut pb_paths = Vec::new();
            for name in ["context", "shuffle", "pk_map", "edges", "index"] {
                let pb = profile_dir.join(format!("{name}-profile.pb"));
                if pb.exists() {
                    pb_paths.push(pb);
                }
            }
            if !pb_paths.is_empty() {
                println!("Go pprof:");
                for path in pb_paths {
                    println!("  go tool pprof {}", path.display());
                }
            }
        }
    }
    println!(
        "Next: {}",
        explore_command_hint(&args.dest, &stats.output)
    );
    Ok(())
}

fn explore_command_hint(dest: &DestArg, output: &str) -> String {
    let _ = dest;
    format!("./diskgraph-index/scripts/explore_local.sh {}", output)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgressKey {
    phase: ProgressPhase,
    table: String,
    shard: Option<usize>,
}

struct ProgressState {
    mp: MultiProgress,
    bars: HashMap<ProgressKey, ProgressBar>,
}

fn build_progress() -> Option<ProgressCallback> {
    if !std::io::stderr().is_terminal() {
        return None;
    }
    let state = Arc::new(Mutex::new(ProgressState {
        mp: MultiProgress::new(),
        bars: HashMap::new(),
    }));
    let cb = Arc::new(move |event: ProgressEvent| {
        let mut guard = state.lock().unwrap();
        let label = progress_label(&event);
        let key = ProgressKey {
            phase: event.phase,
            table: event.table.clone(),
            shard: event.shard,
        };
        match event.kind {
            ProgressKind::Start => {
                if guard.bars.contains_key(&key) {
                    return;
                }
                let bar = match event.total {
                    Some(total) => {
                        let bar = guard.mp.add(ProgressBar::new(total));
                        bar.set_style(bar_style());
                        bar
                    }
                    None => {
                        let bar = guard.mp.add(ProgressBar::new_spinner());
                        bar.set_style(spinner_style());
                        bar
                    }
                };
                bar.set_message(label);
                if let Some(total) = event.total {
                    bar.set_prefix(format!("({}/{})", format_count(0), format_count(total)));
                }
                guard.bars.insert(key, bar);
            }
            ProgressKind::Advance => {
                if let Some(bar) = guard.bars.get(&key) {
                    bar.inc(event.delta);
                    if let Some(total) = bar.length() {
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    } else {
                        let count = format_count(bar.position());
                        bar.set_message(format!("{} {}", label, count));
                    }
                }
            }
            ProgressKind::Finish => {
                if let Some(bar) = guard.bars.remove(&key) {
                    if let Some(total) = event.total.or_else(|| bar.length()) {
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    }
                    bar.finish_with_message(label);
                }
            }
        }
    });
    Some(cb)
}

fn progress_label(event: &ProgressEvent) -> String {
    let phase = match event.phase {
        ProgressPhase::Context => "context",
        ProgressPhase::Shuffle => "shuffle",
        ProgressPhase::PkMap => "pk_map",
        ProgressPhase::Edges => "edges",
        ProgressPhase::Index => "index",
    };
    match event.shard {
        Some(shard) => format!("{} {} shard-{:05}", phase, event.table, shard),
        None => format!("{} {}", phase, event.table),
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut value = bytes as f64;
    let mut unit = units[0];
    for next in &units[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{} {}", bytes, unit)
    } else {
        format!("{:.1} {}", value, unit)
    }
}

fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        return format!("{}ms", ms);
    }
    let secs = ms / 1000;
    let rem_ms = ms % 1000;
    if secs < 60 {
        return format!("{}.{}s", secs, rem_ms / 100);
    }
    let mins = secs / 60;
    let rem_secs = secs % 60;
    format!("{}m {}s", mins, rem_secs)
}

fn format_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}
