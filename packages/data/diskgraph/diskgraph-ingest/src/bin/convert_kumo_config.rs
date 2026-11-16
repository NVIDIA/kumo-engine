// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI tool to convert Kumo GraphConfig to DiskGraph GraphConfig.
//!
//! Usage:
//!   convert-kumo-config --customer example --job trainingjob-example --bucket example-bucket

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::Parser;

use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
};

#[derive(Parser, Debug)]
#[command(name = "convert-kumo-config")]
#[command(about = "Convert Kumo GraphConfig to DiskGraph GraphConfig")]
struct Args {
    /// Customer directory within the bucket.
    #[arg(short, long)]
    customer: String,

    /// Job ID (e.g., trainingjob-abc123)
    #[arg(short, long)]
    job: String,

    /// S3 bucket containing the job and its source data.
    #[arg(short, long)]
    bucket: String,

    /// Output DiskGraph config JSON file (writes to stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// AWS profile to use for S3 access
    #[arg(long)]
    profile: Option<String>,
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

fn main() -> Result<()> {
    let args = Args::parse();

    let bucket = &args.bucket;
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
        source_paths_base: None,
        source_paths: None,
        source_specs: None,
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

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn storage_is_chosen_explicitly() {
        let job = [
            "convert-kumo-config",
            "--customer",
            "example",
            "--job",
            "trainingjob-example",
        ];
        assert!(Args::try_parse_from(job).is_err());
        assert!(
            Args::try_parse_from(job.into_iter().chain(["--bucket", "example-bucket"])).is_ok()
        );
    }
}
