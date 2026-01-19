mod config;
mod generator;
mod output;
mod progress;
mod uploader;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use crate::config::DatasetConfig;
use crate::output::OutputTarget;

#[derive(Parser, Debug)]
#[command(author, version, about = "Synthetic parquet generator from JSON config.")]
struct Cli {
    /// JSON config file.
    #[arg(short, long, value_name = "PATH")]
    config: PathBuf,
    /// Output directory where table folders are written.
    #[arg(short, long, value_name = "PATH")]
    output: String,
    /// Number of worker threads (defaults to available cores).
    #[arg(long)]
    workers: Option<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    let workers = cli
        .workers
        .unwrap_or_else(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1));
    let output = OutputTarget::parse(&cli.output, workers)?;
    generator::run(config, output, workers)
}

fn load_config(path: &PathBuf) -> Result<DatasetConfig> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read config {}", path.display()))?;
    let config: DatasetConfig = serde_json::from_str(&raw)
        .with_context(|| format!("parse config {}", path.display()))?;
    Ok(config)
}
