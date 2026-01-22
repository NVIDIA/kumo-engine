use anyhow::{Context, Result};
use clap::Parser;
use diskgraph_datagenerator::downloader::{download_job, parse_job_spec, DownloadOptions};
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions, IngestStats};
use diskgraph_ingest::progress::{build_progress, format_count};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::process::Stdio;
use tokio::process::Command;

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

    /// If set, does not clean the download directory before downloading (resumes/verifies existing files).
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Output path for the incremental JSON validation report.
    #[arg(long, default_value = "validation_report.json")]
    report_file: PathBuf,

    /// Enable parallel ingestion.
    #[arg(long, default_value_t = true)]
    parallel: bool,

    /// Memory budget for index build edge sort (bytes).
    #[arg(long)]
    index_mem_budget_bytes: Option<u64>,

    /// Number of shards for the dataset.
    #[arg(long)]
    num_shards: Option<usize>,
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
            match validate_job(
                job,
                &download_root,
                &output_root,
                args.keep,
                args.resume,
                args.parallel,
                args.index_mem_budget_bytes,
                args.num_shards,
            )
            .await
            {
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
        || msg.contains("object_store")
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
    s3_stats_verification: Option<Vec<String>>,
}

async fn validate_job(
    job_spec: &str,
    download_root_base: &Path,
    output_root_base: &Path,
    keep: bool,
    resume: bool,
    parallel: bool,
    index_mem_budget_bytes: Option<u64>,
    num_shards: Option<usize>,
) -> Result<ValidationReport> {
    // 1. Download
    println!("Downloading to {}...", download_root_base.display());

    let options = DownloadOptions {
        out_root: download_root_base.to_path_buf(),
        clean: !resume,
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

    let mut infer_res = infer_source_metadata(&source_root, infer_opts)?;

    if let Some(shards) = num_shards {
        println!("Overriding num_shards to {}", shards);
        infer_res.metadata.num_shards = Some(shards);
    }

    let metadata_path = data_dir.join("diskgraph_config.json");
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
        parallel,
        index_mem_budget_bytes,
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

    // 5. Verify S3 Stats
    let (customer, job_id) = parse_job_spec(job_spec)?;
    println!("Verifying S3 Stats...");
    let s3_verification = match verify_s3_stats(&data_dir, &customer, &job_id, &stats).await {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("Warning: Failed to verify S3 stats: {:?}", e);
            Some(vec![format!("Error: {:?}", e)])
        }
    };

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
        s3_stats_verification: s3_verification,
    })
}

async fn verify_s3_stats(data_dir: &Path, customer: &str, job_id: &str, stats: &IngestStats) -> Result<Vec<String>> {
    let mut messages = Vec::new();
    let bucket = if customer.eq_ignore_ascii_case("doordash") {
        "kumo-doordash-cache"
    } else {
        "kumo-prod-dataplane-bucket"
    };
    let customer_path = if customer.eq_ignore_ascii_case("doordash") {
        "kumo"
    } else {
        customer
    };

    // 1. Verify Node Counts (Table Stats)
    let mut dir_reader = fs::read_dir(data_dir)?;
    while let Some(entry) = dir_reader.next() {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let attrs_path = path.join("attributes.json");
        if !attrs_path.exists() {
            continue;
        }

        let attrs_bytes = fs::read(&attrs_path)?;
        let attrs: serde_json::Value = serde_json::from_slice(&attrs_bytes)?;
        let meta = attrs.get("table_metadata_config")
            .ok_or_else(|| anyhow::anyhow!("missing table_metadata_config in {:?}", attrs_path))?;

        let source_name = meta.get("source_name").and_then(|v| v.as_str()).unwrap_or_default();
        let data_source = meta.get("data_source").and_then(|v| v.as_str()).unwrap_or_default();
        let data_version = meta.get("data_version").and_then(|v| v.as_str()).unwrap_or_default();
        let table_name = meta.get("table_name").and_then(|v| v.as_str()).unwrap_or_default();

        let s3_key_base = format!(
            "{}/prod/data-persist-dir/data_backend_cache/tables/{}/{}/{}/table_stats",
            customer_path, data_source, source_name, data_version
        );
        let s3_path = format!("s3://{}/{}", bucket, s3_key_base);

        let output = Command::new("/home/ubuntu/.local/bin/aws")
            .arg("s3")
            .arg("cp")
            .arg(&s3_path)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await?;

        if !output.status.success() {
             messages.push(format!("SKIP: {} (S3 stats not found at {})", table_name, s3_path));
             continue;
        }

        let s3_stats: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let s3_rows = s3_stats.get("NUM_ROWS__False")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_u64());

        if let Some(s3_rows) = s3_rows {
            if let Some(&ingest_rows) = stats.node_counts.get(table_name) {
                 if diff_pct(s3_rows, ingest_rows) < 0.01 {
                     messages.push(format!("PASS: {} (S3: {}, Ingest: {})", table_name, s3_rows, ingest_rows));
                 } else {
                     messages.push(format!("FAIL: {} (S3: {}, Ingest: {}, Diff: {:.2}%)", table_name, s3_rows, ingest_rows, diff_pct(s3_rows, ingest_rows) * 100.0));
                 }
            } else {
                 messages.push(format!("INFO: {} (S3 rows: {}, not in node_counts - likely edge table)", table_name, s3_rows));
            }
        }
    }

    // 2. Verify Edge Counts (Graph Stats)
    // Fetch prediction_job_input.json (or training_job_input.json for training jobs) to get graph_id and version
    let input_filename = if job_id.starts_with("trainingjob") {
        "training_job_input.json"
    } else {
        "prediction_job_input.json"
    };
    let input_key = format!("{}/prod/jobs/{}/inputs/{}", customer_path, job_id, input_filename);
    let input_s3_path = format!("s3://{}/{}", bucket, input_key);

    let output = Command::new("/home/ubuntu/.local/bin/aws")
            .arg("s3")
            .arg("cp")
            .arg(&input_s3_path)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await?;

    if !output.status.success() {
        messages.push(format!("SKIP: Edge stats ({} not found at {})", input_filename, input_s3_path));
        return Ok(messages);
    }

    let input_json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    // Prioritize predictive_query.graph.name for graph_id, fallback to config/graph_id or spec/graph_id
    let graph_id = input_json.pointer("/spec/predictive_query/graph/name").and_then(|v| v.as_str())
        .or_else(|| input_json.pointer("/config/graph_id").and_then(|v| v.as_str()))
        .or_else(|| input_json.pointer("/spec/graph_id").and_then(|v| v.as_str()));

    // Try multiple locations for graph_data_version
    let graph_version = input_json.pointer("/graph_version_info/graph_data_version").and_then(|v| v.as_str())
        .or_else(|| input_json.pointer("/config/graph_data_version").and_then(|v| v.as_str()))
        .or_else(|| input_json.pointer("/config/graph_version_info/graph_data_version").and_then(|v| v.as_str()))
        .or_else(|| input_json.pointer("/spec/graph_version_info/graph_data_version").and_then(|v| v.as_str()));

    if let (Some(gid), Some(gver)) = (graph_id, graph_version) {
        let edge_stats_key = format!(
            "{}/prod/data-persist-dir/data_backend_cache/graphs/{}/{}/edge_stats",
            customer_path, gid, gver
        );
        let edge_stats_s3 = format!("s3://{}/{}", bucket, edge_stats_key);

        let output = Command::new("/home/ubuntu/.local/bin/aws")
            .arg("s3")
            .arg("cp")
            .arg(&edge_stats_s3)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await?;

        if output.status.success() {
            let edge_stats: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            if let Some(obj) = edge_stats.as_object() {
                let mut ingest_edges_matched = HashSet::new();

                for (s3_key, val_obj) in obj {
                    if !s3_key.ends_with("EDGE_COUNT") { continue; }
                    let s3_count = val_obj.get("value").and_then(|v| v.as_u64()).unwrap_or(0);

                    let mut found_any_name_match = false;
                    for (relation_name, &ingest_count) in &stats.edge_counts {
                        // Strip _reverse suffix for name matching
                        let base_relation = relation_name.trim_end_matches("_reverse");
                        let parts: Vec<&str> = base_relation.split("_to_").collect();
                        if parts.len() == 2 {
                            let src_table = parts[0].split('_').next().unwrap_or("");
                            let dst_table = parts[1].split('_').next().unwrap_or("");

                            if s3_key.contains(src_table) && s3_key.contains(dst_table) {
                                found_any_name_match = true;
                                ingest_edges_matched.insert(relation_name.clone());

                                if s3_count == ingest_count {
                                    messages.push(format!("PASS: Edge {} (S3: {}, Ingest: {})", relation_name, s3_count, ingest_count));
                                } else {
                                    messages.push(format!("FAIL: Edge {} count mismatch (S3: {}, Ingest: {})", relation_name, s3_count, ingest_count));
                                }
                            }
                        }
                    }

                    if !found_any_name_match {
                        messages.push(format!("INFO: S3 Edge {} (S3 rows: {}, not in ingest edge_counts)", s3_key, s3_count));
                    }
                }

                for (relation_name, &ingest_count) in &stats.edge_counts {
                    if !ingest_edges_matched.contains(relation_name) {
                        messages.push(format!("FAIL: Edge {} (Ingest: {}, No matching S3 relation found)", relation_name, ingest_count));
                    }
                }
            }
        } else {
             messages.push(format!("SKIP: Edge stats (edge_stats not found at {})", edge_stats_s3));
        }
    } else {
        messages.push("SKIP: Edge stats (Could not determine graph_id/version)".to_string());
    }

    Ok(messages)
}

fn diff_pct(a: u64, b: u64) -> f64 {
    let diff = (a as i64 - b as i64).abs() as f64;
    let max = a.max(b) as f64;
    if max == 0.0 { 0.0 } else { diff / max }
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
        "{:<40} | {:>12} | {:>12} | {:>12} | {:>12} | S3 Verify",
        "Job Spec", "Nodes", "Edges", "Time (ms)", "Size (MB)"
    );
    println!("{}", line);
    for res in results {
        match &res.report {
            Some(report) => {
                    let verify_status = if let Some(msgs) = &report.s3_stats_verification {
                    let fails = msgs.iter().filter(|m| m.starts_with("FAIL")).count();
                    let skips = msgs.iter().filter(|m| m.starts_with("SKIP")).count();
                    if fails > 0 { format!("{} FAIL", fails) } 
                    else if skips > 0 && msgs.iter().all(|m| m.starts_with("SKIP") || m.starts_with("INFO")) { "SKIP".to_string() }
                    else { "PASS".to_string() }
                } else {
                    "N/A".to_string()
                };

                println!(
                    "{:<40} | {:>12} | {:>12} | {:>12} | {:>12.2} | {}",
                    truncate_str(&res.job_spec, 40),
                    format_count(report.total_nodes),
                    format_count(report.total_edges),
                    report.ingest_time_ms,
                    report.index_size_bytes as f64 / 1024.0 / 1024.0,
                    verify_status
                );

                if let Some(msgs) = &report.s3_stats_verification {
                     if verify_status.contains("FAIL") || std::env::var("VERBOSE").is_ok() {
                         for msg in msgs {
                             if msg.starts_with("FAIL") {
                                 println!("    - {}", msg);
                             }
                         }
                     }
                }
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
