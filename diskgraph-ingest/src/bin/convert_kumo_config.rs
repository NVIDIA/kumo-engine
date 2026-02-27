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

use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, estimate_num_shards, ConvertOptions, KumoGraphConfig,
};
use diskgraph_ingest::progress::format_bytes;

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
    #[arg(long)]
    profile: Option<String>,

    /// Number of shards for the index (default: not set, ingest binary picks)
    #[arg(long)]
    num_shards: Option<usize>,
}

/// Download a file from S3 using AWS CLI
fn s3_download(bucket: &str, key: &str, profile: Option<&str>) -> Result<String> {
    let s3_path = format!("s3://{}/{}", bucket, key);
    let mut args = vec!["s3", "cp", &s3_path, "-"];
    if let Some(p) = profile {
        args.extend_from_slice(&["--profile", p]);
    }
    let output = Command::new("aws")
        .args(&args)
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
fn s3_list_first(bucket: &str, prefix: &str, profile: Option<&str>) -> Result<String> {
    let s3_path = format!("s3://{}/{}", bucket, prefix);
    let mut args = vec!["s3", "ls", &s3_path];
    if let Some(p) = profile {
        args.extend_from_slice(&["--profile", p]);
    }
    let output = Command::new("aws")
        .args(&args)
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

/// List parquet files under an S3 prefix, returning (full S3 URL, size_bytes).
fn s3_list_parquet(s3_url: &str, profile: Option<&str>) -> Result<Vec<(String, u64)>> {
    let url = s3_url.trim_end_matches('/');
    let url_slash = format!("{}/", url);
    let mut args = vec!["s3", "ls", url_slash.as_str()];
    if let Some(p) = profile {
        args.extend_from_slice(&["--profile", p]);
    }
    let output = Command::new("aws")
        .args(&args)
        .output()
        .with_context(|| format!("Failed to run aws s3 ls for {}", url))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to list {}: {}", url, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut files: Vec<(String, u64)> = stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return None;
            }
            let size = parts[2].parse::<u64>().ok()?;
            let filename = parts[3];
            if filename.ends_with(".parquet") {
                Some((format!("{}/{}", url, filename), size))
            } else {
                None
            }
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let bucket = args.region.bucket();
    let customer = &args.customer;
    let job_id = &args.job;

    eprintln!(
        "Looking up job {} for customer {} in {}...",
        job_id, customer, bucket
    );

    // Find the graph directory (e.g., NA-abc123)
    let graphs_prefix = format!("{}/prod/jobs/{}/graphs/", customer, job_id);
    let graph_dir = s3_list_first(bucket, &graphs_prefix, args.profile.as_deref())
        .with_context(|| format!("Failed to find graph directory in {}", graphs_prefix))?;

    eprintln!("Found graph: {}", graph_dir);

    // Download the config.json
    let config_key = format!("{}{}/config.json", graphs_prefix, graph_dir);
    eprintln!("Downloading config from {}...", config_key);
    let config_json = s3_download(bucket, &config_key, args.profile.as_deref())?;

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
        table_data_versions: None,
    };

    let mut diskgraph_config =
        convert_kumo_to_diskgraph(&kumo_config, &options).context("Failed to convert config")?;

    // Expand S3 directory prefixes to individual parquet file paths.
    // The converter outputs fpv3 directory paths but the ingest binary
    // expects individual file paths (it does HEAD per path, not prefix listing).
    let mut total_bytes = 0u64;
    for table in &mut diskgraph_config.tables {
        let dir_paths: Vec<String> = table.get_paths().to_vec();
        let mut expanded: Vec<String> = Vec::new();
        for dir_path in &dir_paths {
            if dir_path.starts_with("s3://") {
                let parquet_files = s3_list_parquet(dir_path, args.profile.as_deref())
                    .with_context(|| format!("Failed to list parquet files in {}", dir_path))?;
                if parquet_files.is_empty() {
                    bail!(
                        "No parquet files found for table {} in {}",
                        table.name,
                        dir_path
                    );
                }
                eprintln!(
                    "  {}: {} parquet files",
                    table.name,
                    parquet_files.len()
                );
                for (path, size) in parquet_files {
                    expanded.push(path);
                    total_bytes = total_bytes.saturating_add(size);
                }
            } else {
                expanded.push(dir_path.clone());
            }
        }
        *table.get_paths_mut() = expanded;
    }

    // Set num_shards if provided, otherwise infer from total size
    if let Some(shards) = args.num_shards {
        diskgraph_config.num_shards = Some(shards);
        eprintln!("Set num_shards to {} (manual override)", shards);
    } else {
        let shards = estimate_num_shards(total_bytes);
        diskgraph_config.num_shards = Some(shards);
        eprintln!(
            "Inferred num_shards={} for total data size {}",
            shards,
            format_bytes(total_bytes)
        );
    }

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
