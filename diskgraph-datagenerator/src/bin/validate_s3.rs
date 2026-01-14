use anyhow::{Context, Result};
use clap::Parser;
use diskgraph_datagenerator::downloader::{download_job, DownloadOptions};
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Validate ingestion from Kumo customer/job specs.")]
struct Args {
    /// Job specs (e.g., "customer job_id" or "s3://..." if supported by downloader).
    #[arg(required = true)]
    jobs: Vec<String>,

    /// Output root directory for indices.
    #[arg(long)]
    output_root: Option<PathBuf>,

    /// Temporary directory for downloads.
    #[arg(long)]
    download_root: Option<PathBuf>,

    /// Keep downloaded data and indices after validation.
    #[arg(long, default_value_t = false)]
    keep: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let output_root = args.output_root.unwrap_or_else(|| {
        std::env::var("DISKGRAPH_DATA_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/diskgraph-validate/index"))
    });
    let download_root = args.download_root.unwrap_or_else(|| {
        std::env::var("DISKGRAPH_DATA_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/diskgraph-validate/raw"))
    });

    if !output_root.exists() {
        fs::create_dir_all(&output_root).context("create output root")?;
    }
    if !download_root.exists() {
        fs::create_dir_all(&download_root).context("create download root")?;
    }

    let mut results = Vec::new();

    for job in &args.jobs {
        println!("\n>>> Processing Job: {}", job);
        let (customer, job_id) = extract_job_info(job);

        loop {
            let start = Instant::now();
            match validate_job(job, &download_root, &output_root, args.keep).await {
                Ok(report) => {
                    let duration = start.elapsed();
                    println!(">>> Finished Job: {} in {:?}", job, duration);
                    results.push((job.clone(), Ok(report)));
                    break;
                }
                Err(e) => {
                    if is_network_error(&e) {
                        eprintln!(
                            ">>> Network Issue for Customer: {}, Job: {}. Error: {:?}. Retrying in 10s...",
                            customer, job_id, e
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                        continue;
                    } else {
                        eprintln!(
                            ">>> Internal Error for Customer: {}, Job: {}. Error: {:?}",
                            customer, job_id, e
                        );
                        results.push((job.clone(), Err(e)));
                        break;
                    }
                }
            }
        }
    }

    print_report(&results);

    Ok(())
}

fn extract_job_info(job_spec: &str) -> (String, String) {
    if job_spec.starts_with("http") {
        if let Ok(url) = url::Url::parse(job_spec) {
            let host = url.host_str().unwrap_or("unknown");
            let customer = host.split('.').next().unwrap_or("unknown").to_string();
            // simple heuristic to find job id in path
            let path = url.path();
            let job_id = if let Some(idx) = path.find("trainingjob-") {
                path[idx..].split('/').next().unwrap_or("unknown").to_string()
            } else if let Some(idx) = path.find("bp-job-") {
                path[idx..].split('/').next().unwrap_or("unknown").to_string()
            } else {
                "unknown".to_string()
            };
            return (customer, job_id);
        }
    } else {
        let parts: Vec<&str> = job_spec.split_whitespace().collect();
        if parts.len() >= 2 {
            return (parts[0].to_string(), parts[1].to_string());
        }
    }
    ("unknown".to_string(), "unknown".to_string())
}

fn is_network_error(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("timeout")
        || msg.contains("connection")
        || msg.contains("dns")
        || msg.contains("reset")
        || msg.contains("broken pipe")
        || msg.contains("hyper")
        || msg.contains("reqwest")
        || msg.contains("aws_sdk_s3")
        || msg.contains("service error")
        || msg.contains("500 internal server error")
        || msg.contains("503 service unavailable")
        || msg.contains("502 bad gateway")
        || msg.contains("504 gateway timeout")
}

struct ValidationReport {
    total_nodes: u64,
    total_edges: u64,
    ingest_time_ms: u64,
    index_size_bytes: u64,
}

async fn validate_job(
    job_spec: &str,
    download_root_base: &Path,
    output_root_base: &Path,
    keep: bool,
) -> Result<ValidationReport> {
    // 1. Download
    println!("Downloading to {}...", download_root_base.display());
    
    let options = DownloadOptions {
        out_root: download_root_base.to_path_buf(),
        ..Default::default()
    };

    let data_dir = download_job(job_spec, options).await?;
    println!("Found data at: {}", data_dir.display());

    // 2. Infer
    println!("Inferring metadata...");
    let source_root = parse_root_source(data_dir.to_str().unwrap())?;
    let infer_res = infer_source_metadata(&source_root, InferOptions::default())?;
    
    let metadata_path = data_dir.join("source_metadata.json");
    fs::write(&metadata_path, serde_json::to_vec_pretty(&infer_res.metadata)?)?;

    // 3. Ingest
    let index_name = format!("{}_index", infer_res.metadata.dataset_name);
    let output_path = output_root_base.join(&index_name);
    if output_path.exists() {
        println!("Index already exists at {}, clearing...", output_path.display());
        fs::remove_dir_all(&output_path)?;
    }

    println!("Ingesting to {}...", output_path.display());
    let options = IngestOptions {
        parallel: true,
        ..Default::default()
    };
    let (_, stats) = ingest_with_stats(
        &metadata_path, 
        output_path.to_str().unwrap(), 
        Dest::Local, 
        options
    )?;

    // 4. Measure Size
    let index_size = get_dir_size(&output_path)?;

    // Cleanup
    if !keep {
        println!("Cleaning up...");
        let _ = fs::remove_dir_all(&data_dir);
        let _ = fs::remove_dir_all(&output_path);
    } else {
        println!("Keeping artifacts.");
    }

    Ok(ValidationReport {
        total_nodes: stats.total_nodes,
        total_edges: stats.total_edges,
        ingest_time_ms: stats.perf.wall_time_ms,
        index_size_bytes: index_size,
    })
}

fn get_dir_size(path: &Path) -> Result<u64> {
    let mut total = 0;
    if !path.exists() {
        return Ok(0);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += get_dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

fn print_report(results: &[(String, Result<ValidationReport>)]) {
    println!("\nIngestion Validation Summary:");
    let line = "-".repeat(120);
    println!("{}", line);
    println!(
        "{:<40} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Job Spec", "Nodes", "Edges", "Time (ms)", "Size (MB)"
    );
    println!("{}", line);
    for (url, res) in results {
        match res {
            Ok(report) => {
                println!(
                    "{:<40} | {:>12} | {:>12} | {:>12} | {:>12.2}",
                    truncate_str(url, 40),
                    format_count(report.total_nodes),
                    format_count(report.total_edges),
                    report.ingest_time_ms,
                    report.index_size_bytes as f64 / 1024.0 / 1024.0
                );
            }
            Err(e) => {
                println!("{:<40} | ERROR: {}", truncate_str(url, 40), truncate_error(e));
            }
        }
    }
    println!("{}", line);
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max + 3..])
    }
}

fn truncate_error(e: &anyhow::Error) -> String {
    let s = format!("{:?}", e);
    if s.len() > 60 {
        format!("{}...", &s[..57])
    } else {
        s
    }
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
