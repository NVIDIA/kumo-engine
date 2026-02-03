//! CLI tool to convert Kumo GraphConfig to DiskGraph GraphConfig.
//!
//! Usage:
//!   convert-kumo-config --customer allstate --job trainingjob-abc123 --region ue1
//!   convert-kumo-config --customer allstate --job trainingjob-abc123 --region uw2 --output config.json

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};

use diskgraph_ingest::config_converter::{convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Region {
    /// US East 1 (kumo-prod-ue1-dataplane-bucket)
    Ue1,
    /// US West 2 (kumo-prod-dataplane-bucket)
    Uw2,
}

impl Region {
    fn bucket(&self) -> &'static str {
        match self {
            Region::Ue1 => "kumo-prod-ue1-dataplane-bucket",
            Region::Uw2 => "kumo-prod-dataplane-bucket",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "convert-kumo-config")]
#[command(about = "Convert Kumo GraphConfig to DiskGraph GraphConfig")]
struct Args {
    /// Customer name (e.g., allstate, blend360)
    #[arg(short, long)]
    customer: String,

    /// Job ID (e.g., trainingjob-abc123)
    #[arg(short, long)]
    job: String,

    /// AWS region / S3 bucket
    #[arg(short, long, value_enum)]
    region: Region,

    /// Output DiskGraph config JSON file (writes to stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// AWS profile to use for S3 access
    #[arg(long, default_value = "normal-profile")]
    profile: String,
}

/// Download a file from S3 using AWS CLI
fn s3_download(bucket: &str, key: &str, profile: &str) -> Result<String> {
    let s3_path = format!("s3://{}/{}", bucket, key);
    let output = Command::new("aws")
        .args(["s3", "cp", &s3_path, "-", "--profile", profile])
        .output()
        .with_context(|| format!("Failed to run aws s3 cp for {}", s3_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to download {}: {}", s3_path, stderr);
    }

    String::from_utf8(output.stdout)
        .with_context(|| format!("Invalid UTF-8 in downloaded file: {}", s3_path))
}

/// List S3 directory and return first entry
fn s3_list_first(bucket: &str, prefix: &str, profile: &str) -> Result<String> {
    let s3_path = format!("s3://{}/{}", bucket, prefix);
    let output = Command::new("aws")
        .args(["s3", "ls", &s3_path, "--profile", profile])
        .output()
        .with_context(|| format!("Failed to run aws s3 ls for {}", s3_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to list {}: {}", s3_path, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse "PRE dirname/" format
    for line in stdout.lines() {
        if let Some(dir) = line.strip_prefix("                           PRE ") {
            return Ok(dir.trim_end_matches('/').to_string());
        }
    }

    bail!("No directories found in {}", s3_path)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let bucket = args.region.bucket();
    let customer = &args.customer;
    let job_id = &args.job;

    eprintln!("Looking up job {} for customer {} in {}...", job_id, customer, bucket);

    // Find the graph directory (e.g., NA-abc123)
    let graphs_prefix = format!("{}/prod/jobs/{}/graphs/", customer, job_id);
    let graph_dir = s3_list_first(bucket, &graphs_prefix, &args.profile)
        .with_context(|| format!("Failed to find graph directory in {}", graphs_prefix))?;

    eprintln!("Found graph: {}", graph_dir);

    // Download the config.json
    let config_key = format!("{}{}/config.json", graphs_prefix, graph_dir);
    eprintln!("Downloading config from {}...", config_key);
    let config_json = s3_download(bucket, &config_key, &args.profile)?;

    // Parse Kumo config
    let kumo_config: KumoGraphConfig =
        serde_json::from_str(&config_json).context("Failed to parse Kumo GraphConfig JSON")?;

    // Build data-persist-dir path
    let data_persist_dir = format!("s3://{}/{}/prod/data-persist-dir", bucket, customer);

    // Use job_id as dataset name
    let dataset_name = job_id.clone();

    eprintln!("Converting with data-persist-dir: {}", data_persist_dir);

    // Convert to DiskGraph config
    let options = ConvertOptions {
        data_persist_dir,
        dataset_name,
    };

    let diskgraph_config =
        convert_kumo_to_diskgraph(&kumo_config, &options).context("Failed to convert config")?;

    // Serialize output
    let output_json =
        serde_json::to_string_pretty(&diskgraph_config).context("Failed to serialize output")?;

    // Write output
    match &args.output {
        Some(path) => {
            fs::write(path, &output_json)
                .with_context(|| format!("Failed to write output file: {}", path.display()))?;
            eprintln!("Wrote DiskGraph config to {}", path.display());
        }
        None => {
            io::stdout()
                .write_all(output_json.as_bytes())
                .context("Failed to write to stdout")?;
            println!();
        }
    }

    Ok(())
}
