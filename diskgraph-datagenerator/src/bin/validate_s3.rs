use anyhow::{Context, Result};
use clap::Parser;
use diskgraph_datagenerator::downloader::{download_job, parse_job_spec, DownloadOptions};
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions};
use diskgraph_ingest::progress::{build_progress, format_count};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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

    /// Output path for the incremental JSON validation report.
    #[arg(long, default_value = "validation_report.json")]
    report_file: PathBuf,
}

#[derive(Serialize, Deserialize, Clone)]
struct JobResult {
    job_spec: String,
    customer: String,
    job_id: String,
    status: String,
    error: Option<String>,
    #[serde(flatten)]
    report: Option<ValidationReport>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let output_root = args.output_root.unwrap_or_else(|| {
        std::env::var("DISKGRAPH_DATA_ROOT")
            .map(|v| PathBuf::from(v).join("index"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/diskgraph-validate/index"))
    });
    let download_root = args.download_root.unwrap_or_else(|| {
        std::env::var("DISKGRAPH_DATA_ROOT")
            .map(|v| PathBuf::from(v).join("raw"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/diskgraph-validate/raw"))
    });

    if !output_root.exists() {
        fs::create_dir_all(&output_root).context("create output root")?;
    }
    if !download_root.exists() {
        fs::create_dir_all(&download_root).context("create download root")?;
    }

    let mut results: Vec<JobResult> = load_existing_report(&args.report_file)?;
    let mut completed_jobs: HashSet<(String, String)> = results
        .iter()
        .filter(|r| r.status == "Success")
        .map(|r| (r.customer.clone(), r.job_id.clone()))
        .collect();

    for job in &args.jobs {
        let (customer, job_id) = parse_job_spec(job).unwrap_or(("unknown".to_string(), "unknown".to_string()));
        if completed_jobs.contains(&(customer.clone(), job_id.clone())) {
            println!("\n>>> Skipping already validated job: {} ({}/{})", job, customer, job_id);
            continue;
        }

        println!("\n>>> Processing Job: {}", job);

        // Remove any prior entry for this job (e.g. if it failed previously)
        results.retain(|r| r.customer != customer || r.job_id != job_id);

        loop {
            let start = Instant::now();
            match validate_job(job, &download_root, &output_root, args.keep).await {
                Ok(report) => {
                    let duration = start.elapsed();
                    println!(">>> Finished Job: {} in {:?}", job, duration);
                    let result = JobResult {
                        job_spec: job.clone(),
                        customer: customer.clone(),
                        job_id: job_id.clone(),
                        status: "Success".to_string(),
                        error: None,
                        report: Some(report),
                    };
                    results.push(result);
                    completed_jobs.insert((customer.clone(), job_id.clone()));
                    save_incremental_report(&results, &args.report_file)?;
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
                        let result = JobResult {
                            job_spec: job.clone(),
                            customer: customer.clone(),
                            job_id: job_id.clone(),
                            status: "Error".to_string(),
                            error: Some(format!("{:?}", e)),
                            report: None,
                        };
                        results.push(result);
                        save_incremental_report(&results, &args.report_file)?;
                        break;
                    }
                }
            }
        }
    }

    print_report(&results);

    Ok(())
}

fn load_existing_report(path: &Path) -> Result<Vec<JobResult>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path)?;
    // If file is empty or invalid JSON, return empty vec but warn?
    // For now, if it fails to parse, we assume it's corrupt or not compatible and fail hard to avoid overwrite.
    // Or we could return empty to start fresh. Let's fail hard for safety.
    let results: Vec<JobResult> = serde_json::from_reader(file).context("parse existing report")?;
    Ok(results)
}

fn save_incremental_report(results: &[JobResult], path: &Path) -> Result<()> {
    let file = fs::File::create(path).context("create report file")?;
    serde_json::to_writer_pretty(file, results).context("write report file")?;
    Ok(())
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

#[derive(Serialize, Deserialize, Clone)]
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

    // Check for graph config
    let graph_config_path = data_dir.join("config.json");
    let graph_config = if graph_config_path.exists() {
        println!("Found graph config at {}", graph_config_path.display());
        let bytes = fs::read(&graph_config_path)?;
        Some(serde_json::from_slice(&bytes)?)
    } else {
        None
    };

    let source_root = parse_root_source(data_dir.to_str().unwrap())?;
    let mut infer_opts = InferOptions::default();
    infer_opts.graph_config = graph_config;

    let infer_res = infer_source_metadata(&source_root, infer_opts)?;

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
        progress: build_progress(),
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

fn print_report(results: &[JobResult]) {
    println!("\nIngestion Validation Summary:");
    let line = "-".repeat(120);
    println!("{}", line);
    println!(
        "{:<40} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Job Spec", "Nodes", "Edges", "Time (ms)", "Size (MB)"
    );
    println!("{}", line);
    for res in results {
        match &res.report {
            Some(report) => {
                println!(
                    "{:<40} | {:>12} | {:>12} | {:>12} | {:>12.2}",
                    truncate_str(&res.job_spec, 40),
                    format_count(report.total_nodes),
                    format_count(report.total_edges),
                    report.ingest_time_ms,
                    report.index_size_bytes as f64 / 1024.0 / 1024.0
                );
            }
            None => {
                println!(
                    "{:<40} | ERROR: {}",
                    truncate_str(&res.job_spec, 40),
                    res.error.as_deref().unwrap_or("Unknown error")
                );
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
