use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{
    ingest_with_stats, Dest, IngestOptions, MappingKind, ProgressCallback, ProgressEvent,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::fs;
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
    /// Primary key mapping strategy.
    #[arg(long, value_enum, default_value = "hash")]
    mapping: MappingArg,
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MappingArg {
    Hash,
    Sorted,
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
    let total_tasks = count_expected_tasks(&source_path).unwrap_or(0);
    let progress = if total_tasks > 0 {
        progress_ui(total_tasks)
    } else {
        None
    };
    let options = IngestOptions {
        mapping_kind: match args.mapping {
            MappingArg::Hash => MappingKind::Hash,
            MappingArg::Sorted => MappingKind::Sorted,
        },
        progress,
        profile_dir: args.profile.clone(),
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
        "    pk_map: {}",
        format_duration(stats.perf.pk_map_wall_ms)
    );
    println!(
        "    build: {}",
        format_duration(stats.perf.build_wall_ms)
    );
    println!(
        "    finalize: {}",
        format_duration(stats.perf.finalize_wall_ms)
    );
    println!("  Stage timings (summed across tasks):");
    println!(
        "    pk_map: {} ({} tasks)",
        format_duration(stats.perf.pk_map_task_ms),
        stats.perf.pk_map_tasks
    );
    println!(
        "    build: {} ({} tasks)",
        format_duration(stats.perf.build_task_ms),
        stats.perf.build_tasks
    );
    println!(
        "      nodes: {} ({} tasks)",
        format_duration(stats.perf.node_task_ms),
        stats.perf.node_tasks
    );
    println!(
        "      edges: {} ({} tasks)",
        format_duration(stats.perf.edge_task_ms),
        stats.perf.edge_tasks
    );
    if let Some(profile_dir) = args.profile.as_ref() {
        let mut profiles = Vec::new();
        for name in ["pk_map", "nodes", "edges"] {
            let svg = profile_dir.join(format!("{name}.svg"));
            let pb = profile_dir.join(format!("{name}.pb"));
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
            for name in ["pk_map", "nodes", "edges"] {
                let pb = profile_dir.join(format!("{name}.pb"));
                if pb.exists() {
                    pb_paths.push(pb);
                }
            }
            if !pb_paths.is_empty() {
                println!("Go pprof:");
                for path in pb_paths {
                    println!("  go tool pprof -http=:0 {}", path.display());
                }
            }
        }
    }
    Ok(())
}

fn progress_ui(total_tasks: u64) -> Option<ProgressCallback> {
    if !std::io::stdout().is_terminal() {
        return None;
    }
    struct TaskState {
        pb: ProgressBar,
        total: Option<u64>,
        count: u64,
    }

    let multi = Arc::new(MultiProgress::new());
    let global = multi.add(ProgressBar::new(total_tasks));
    global.set_style(bar_style());
    global.set_message("Overall");
    global.set_prefix(format!("0 / {}", format_count(total_tasks)));
    let tasks: Arc<Mutex<HashMap<String, TaskState>>> = Arc::new(Mutex::new(HashMap::new()));

    let progress: ProgressCallback = Arc::new({
        let multi = Arc::clone(&multi);
        let tasks = Arc::clone(&tasks);
        let global = global.clone();
        move |event| match event {
            ProgressEvent::Start { key, total } => {
                let mut map = tasks.lock().unwrap();
                if map.contains_key(&key) {
                    return;
                }
                if let Some(total) = total {
                    let pb = multi.insert_before(&global, ProgressBar::new(total));
                    pb.set_style(bar_style());
                    pb.set_message(key.clone());
                    pb.set_prefix(format!("0 / {}", format_count(total)));
                    map.insert(
                        key,
                        TaskState {
                            pb,
                            total: Some(total),
                            count: 0,
                        },
                    );
                } else {
                    let pb = multi.insert_before(&global, ProgressBar::new_spinner());
                    pb.set_style(spinner_style());
                    pb.set_message(key.clone());
                    pb.enable_steady_tick(std::time::Duration::from_millis(120));
                    map.insert(
                        key,
                        TaskState {
                            pb,
                            total: None,
                            count: 0,
                        },
                    );
                }
            }
            ProgressEvent::Advance { key, delta } => {
                let mut map = tasks.lock().unwrap();
                if let Some(task) = map.get_mut(&key) {
                    task.count = task.count.saturating_add(delta);
                    if let Some(total) = task.total {
                        task.pb.set_position(task.count.min(total));
                        task.pb.set_prefix(format!(
                            "{} / {}",
                            format_count(task.count),
                            format_count(total),
                        ));
                    }
                }
            }
            ProgressEvent::Finish { key, total } => {
                let mut map = tasks.lock().unwrap();
                if let Some(task) = map.remove(&key) {
                    if let Some(total) = total.or(task.total) {
                        task.pb.set_position(total);
                        task.pb.set_prefix(format!(
                            "{} / {}",
                            format_count(total),
                            format_count(total),
                        ));
                    }
                    task.pb.finish_with_message(key.clone());
                    global.inc(1);
                    let done = global.position();
                    global.set_prefix(format!(
                        "{} / {}",
                        format_count(done),
                        format_count(total_tasks),
                    ));
                }
            }
        }
    });
    Some(progress)
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix} ({eta})")
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

fn count_expected_tasks(path: &std::path::Path) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    let meta: diskgraph_ingest::SourceMetadata = serde_json::from_slice(&bytes).ok()?;
    let mut total = 0u64;
    for table in &meta.tables {
        if table.edge_table.is_none() {
            total += 1; // node data
            if table.primary_key.is_some() {
                total += 1; // pk map
            }
        } else {
            total += 1; // edge table scan
            total += 2; // edge table forward/reverse
        }
    }
    total += meta.relationships.len() as u64 * 2;
    Some(total.max(1))
}
