use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{
    ingest_with_stats, Dest, IngestOptions, MappingKind, ProgressCallback, ProgressEvent,
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
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
    tracing_subscriber::fmt().with_env_filter("info").init();
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
    let progress = progress_ui();
    let options = IngestOptions {
        mapping_kind: match args.mapping {
            MappingArg::Hash => MappingKind::Hash,
            MappingArg::Sorted => MappingKind::Sorted,
        },
        progress,
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
    Ok(())
}

fn progress_ui() -> Option<ProgressCallback> {
    let total = Arc::new(Mutex::new(0u64));
    let pb = ProgressBar::new(0);
    pb.set_message("Ingest progress");
    pb.set_style(
        ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {pos}/{len} ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_draw_target(ProgressDrawTarget::stdout_with_hz(5));
    let progress: ProgressCallback = Arc::new({
        let pb = pb.clone();
        let total = Arc::clone(&total);
        move |event| match event {
            ProgressEvent::Start {
                key,
                total: event_total,
            } => {
                if let Some(add_total) = event_total {
                    let mut guard = total.lock().unwrap();
                    *guard = guard.saturating_add(add_total);
                    pb.set_length(*guard);
                } else {
                    let _ = key;
                }
            }
            ProgressEvent::Advance { key, delta } => {
                let _ = key;
                pb.inc(delta);
            }
            ProgressEvent::Finish { key, total } => {
                let _ = key;
                let _ = total;
            }
        }
    });
    Some(progress)
}
