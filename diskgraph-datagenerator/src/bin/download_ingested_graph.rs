use anyhow::{anyhow, Result};
use clap::Parser;
use diskgraph_datagenerator::downloader::{download_job, DownloadOptions};
use std::path::PathBuf;

const DEFAULT_OUT_ROOT: &str = "/mnt/localssd/datasets";

#[derive(Parser, Debug)]
#[command(about = "Download ingested graph parquet files for a training or bp job.")]
struct Args {
    /// Inputs: <customer_name> <job_id> or <job_url>.
    #[arg(value_name = "INPUT")]
    inputs: Vec<String>,
    /// Output root directory (defaults to /mnt/localssd/datasets or DISKGRAPH_DATA_ROOT/datasets).
    #[arg(long, default_value = DEFAULT_OUT_ROOT)]
    out_root: PathBuf,
    /// Skip downloading parquet files.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    
    // Original logic: handle either one URL or (customer, job_id)
    let job_spec = match args.inputs.as_slice() {
        [url] if url.starts_with("http") => url.clone(),
        [customer, job] => format!("{} {}", customer, job),
        _ => {
            return Err(anyhow!(
                "usage: download-ingested-graph <customer_name> <job_id> | <job_url>"
            ));
        }
    };

    download_job(&job_spec, args.options).await?;
    
    Ok(())
}

fn parse_args() -> Result<ResolvedArgs> {
    let raw = Args::parse();
    if raw.inputs.is_empty() {
        return Err(anyhow!(
            "usage: download-ingested-graph <customer_name> <job_id> | <job_url>"
        ));
    }

    Ok(ResolvedArgs {
        inputs: raw.inputs,
        options: DownloadOptions {
            out_root: resolve_out_root(raw.out_root),
            dry_run: raw.dry_run,
            clean: true,
            bucket: None,
        },
    })
}

fn resolve_out_root(out_root: PathBuf) -> PathBuf {
    if out_root == PathBuf::from(DEFAULT_OUT_ROOT) {
        if let Ok(data_root) = std::env::var("DISKGRAPH_DATA_ROOT") {
            let data_root = data_root.trim();
            if !data_root.is_empty() {
                return PathBuf::from(data_root).join("datasets");
            }
        }
    }
    out_root
}

#[derive(Debug)]
struct ResolvedArgs {
    inputs: Vec<String>,
    options: DownloadOptions,
}
