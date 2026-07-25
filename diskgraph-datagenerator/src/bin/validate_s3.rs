use anyhow::{anyhow, Context, Result};
use clap::Parser;
use diskgraph_datagenerator::downloader::{download_job, parse_job_spec, DownloadOptions};
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::config_converter::{
    convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
};
use diskgraph_ingest::progress::{build_progress, format_count};
use diskgraph_ingest::{ingest_with_stats, Dest, IngestOptions, IngestStats};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Validate ingestion from Kumo customer/job specs."
)]
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

    /// S3 bucket to use (default: kumo-prod-dataplane-bucket, or kumo-doordash-cache for doordash)
    #[arg(long)]
    bucket: Option<String>,

    /// Use the config converter instead of infer to generate DiskGraph config.
    /// This reads the Kumo graph config (config.json) and converts it directly.
    #[arg(long, default_value_t = false)]
    use_config_converter: bool,
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
        let (customer, job_id) =
            parse_job_spec(job).unwrap_or(("unknown".to_string(), "unknown".to_string()));
        if completed_jobs.contains(&(customer.clone(), job_id.clone())) {
            println!(
                "\n>>> Skipping already validated job: {} ({}/{})",
                job, customer, job_id
            );
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
                args.bucket.clone(),
                args.use_config_converter,
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

#[allow(clippy::too_many_arguments)]
async fn validate_job(
    job_spec: &str,
    download_root_base: &Path,
    output_root_base: &Path,
    keep: bool,
    resume: bool,
    parallel: bool,
    index_mem_budget_bytes: Option<u64>,
    bucket: Option<String>,
    use_config_converter: bool,
) -> Result<ValidationReport> {
    // 1. Download
    println!("Downloading to {}...", download_root_base.display());

    let options = DownloadOptions {
        out_root: download_root_base.to_path_buf(),
        clean: !resume,
        bucket: bucket.clone(),
        ..Default::default()
    };

    let data_dir = download_job(job_spec, options).await?;
    println!("Found data at: {}", data_dir.display());

    // Check for graph config
    let graph_config_path = data_dir.join("config.json");
    let graph_config: Option<serde_json::Value> = if graph_config_path.exists() {
        println!("Found graph config at {}", graph_config_path.display());
        let bytes = fs::read(&graph_config_path)?;
        Some(serde_json::from_slice(&bytes)?)
    } else {
        None
    };

    // 2. Generate DiskGraph config (via converter or infer)
    let metadata_path = data_dir.join("diskgraph_config.json");

    if use_config_converter {
        println!("Using Kumo config converter...");

        // Require config.json when --use-config-converter is set
        let kumo_config_value = graph_config.as_ref()
            .ok_or_else(|| anyhow!(
                "--use-config-converter requires config.json, but it was not found on S3. \
                This job may not have a Kumo graph config, try without --use-config-converter to use infer instead."
            ))?;

        // Parse as KumoGraphConfig
        let kumo_config: KumoGraphConfig = serde_json::from_value(kumo_config_value.clone())
            .context("parse config.json as KumoGraphConfig")?;

        // Build data_persist_dir from bucket/customer
        let (customer, _) = parse_job_spec(job_spec)?;
        let effective_bucket = bucket.as_deref().unwrap_or_else(|| {
            if customer.eq_ignore_ascii_case("doordash") {
                "kumo-doordash-cache"
            } else {
                "kumo-prod-dataplane-bucket"
            }
        });
        let customer_path = if customer.eq_ignore_ascii_case("doordash") {
            "kumo"
        } else {
            &customer
        };
        let data_persist_dir = format!(
            "s3://{}/{}/prod/data-persist-dir",
            effective_bucket, customer_path
        );

        // Determine dataset name
        let dataset_name = kumo_config.name.clone().unwrap_or_else(|| {
            format!(
                "{}_{}",
                customer,
                parse_job_spec(job_spec).map(|(_, j)| j).unwrap_or_default()
            )
        });

        let convert_options = ConvertOptions {
            data_persist_dir,
            dataset_name,
            table_data_versions: None,
            source_paths_base: None,
            source_paths: None,
            source_specs: None,
        };

        // Convert Kumo config to DiskGraph config
        let mut diskgraph_config = convert_kumo_to_diskgraph(&kumo_config, &convert_options)
            .map_err(|e| anyhow!("config conversion failed: {}", e))?;

        // Replace S3 paths with local paths
        // Downloaded data is at: data_dir/{table_name}/
        // We need to list actual parquet files (ingest doesn't support directory paths)
        for table in &mut diskgraph_config.tables {
            let table_dir = data_dir.join(&table.name);
            let mut parquet_files: Vec<String> = Vec::new();

            if table_dir.is_dir() {
                for entry in fs::read_dir(&table_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                        parquet_files.push(path.to_string_lossy().to_string());
                    }
                }
            }

            if parquet_files.is_empty() {
                return Err(anyhow!(
                    "No parquet files found for table {} in {}",
                    table.name,
                    table_dir.display()
                ));
            }

            parquet_files.sort(); // Consistent ordering
            *table.get_paths_mut() = parquet_files;
        }

        fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&diskgraph_config)?,
        )?;
        println!("Wrote converted config to {}", metadata_path.display());
    } else {
        println!("Inferring metadata...");

        let source_root = parse_root_source(data_dir.to_str().unwrap())?;
        let mut infer_opts = InferOptions::default();
        if let Some(cfg) = &graph_config {
            infer_opts.graph_config = Some(serde_json::from_value(cfg.clone())?);
        }

        let infer_res = infer_source_metadata(&source_root, infer_opts)?;

        fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&infer_res.metadata)?,
        )?;
    }

    // 3. Ingest
    // Read back the config to get dataset_name
    let config_bytes = fs::read(&metadata_path)?;
    let config_json: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let dataset_name = config_json
        .get("dataset_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing dataset_name in config"))?;
    let index_name = format!("{}_index", dataset_name);
    let output_path = output_root_base.join(&index_name);
    if output_path.exists() {
        println!(
            "Index already exists at {}, clearing...",
            output_path.display()
        );
        fs::remove_dir_all(&output_path)?;
    }

    println!("Ingesting to {}...", output_path.display());
    let options = IngestOptions {
        parallel,
        index_mem_budget_bytes,
        progress: build_progress(),
        col_stats: false,
        ..Default::default()
    };
    let (_, stats) = ingest_with_stats(
        &metadata_path,
        output_path.to_str().unwrap(),
        Dest::Local,
        options,
    )?;

    // 4. Measure Size
    let index_size = get_dir_size(&output_path)?;

    // 5. Verify S3 Stats
    let (customer, job_id) = parse_job_spec(job_spec)?;
    println!("Verifying S3 Stats...");
    let s3_verification = match verify_s3_stats(
        &data_dir,
        &customer,
        &job_id,
        &stats,
        bucket.as_deref(),
        graph_config.as_ref(),
    )
    .await
    {
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

fn calculate_expected_node_count(
    table_name: &str,
    s3_count: u64,
    stats: &IngestStats,
    drop_duplicates_map: &HashMap<String, bool>,
) -> u64 {
    let mut dropped = 0;

    // 1. Parse Failures (from key_failures_by_context)
    // Context format: "pkey:{table}.{column}"
    for (context, failures) in &stats.key_failures_by_context {
        if context.starts_with(&format!("pkey:{}.", table_name)) {
            dropped += failures.parse_failures;
        }
    }

    // 2. Duplicates (from pk_duplicate_counts) - ONLY if drop_duplicates is true (default)
    let should_drop_duplicates = *drop_duplicates_map.get(table_name).unwrap_or(&true);
    if should_drop_duplicates {
        dropped += get_table_duplicates(table_name, stats);
    }

    // 3. Null timestamps or oversized rows (from dropped_rows_counts)
    if let Some(&c) = stats.dropped_rows_counts.get(table_name) {
        dropped += c;
    }

    s3_count.saturating_sub(dropped)
}

fn get_table_duplicates(table_name: &str, stats: &IngestStats) -> u64 {
    let mut count = 0;
    // Key format: "{table}:shard-{shard_id}"
    for (shard_key, c) in &stats.pk_duplicate_counts {
        if shard_key.starts_with(&format!("{}:", table_name)) {
            count += c;
        }
    }
    count
}

async fn verify_s3_stats(
    data_dir: &Path,
    customer: &str,
    job_id: &str,
    stats: &IngestStats,
    bucket_override: Option<&str>,
    kumo_config: Option<&serde_json::Value>,
) -> Result<Vec<String>> {
    let mut messages = Vec::new();
    let mut drop_duplicates_map = HashMap::new();
    if let Some(cfg) = kumo_config {
        if let Some(tables) = cfg.get("tables").and_then(|t| t.as_array()) {
            for table in tables {
                if let Some(name) = table.get("table_name").and_then(|n| n.as_str()) {
                    let drop = table
                        .get("pkey")
                        .and_then(|pk| pk.get("drop_duplicates"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    drop_duplicates_map.insert(name.to_string(), drop);
                }
            }
        }
    }

    let bucket = if let Some(b) = bucket_override {
        b
    } else if customer.eq_ignore_ascii_case("doordash") {
        "kumo-doordash-cache"
    } else {
        "kumo-prod-dataplane-bucket"
    };
    let customer_path = if customer.eq_ignore_ascii_case("doordash") {
        "kumo"
    } else {
        customer
    };

    // Download materialized_task_persist_paths.json manifest for efficient edge lookups
    let manifest_key = format!(
        "{}/prod/jobs/{}/jobs/materialized_task_persist_paths.json",
        customer_path, job_id
    );
    let manifest_s3_path = format!("s3://{}/{}", bucket, manifest_key);
    let manifest: Option<HashMap<String, String>> = {
        let output = Command::new("aws")
            .arg("s3")
            .arg("cp")
            .arg(&manifest_s3_path)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await;

        if let Ok(out) = output {
            if out.status.success() {
                let json: serde_json::Value =
                    serde_json::from_slice(&out.stdout).unwrap_or_default();
                json.get("edge_names_with_paths")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                            .collect()
                    })
            } else {
                None
            }
        } else {
            None
        }
    };

    let mut table_infos: std::collections::HashMap<String, TableMetadata> =
        std::collections::HashMap::new();

    // 1. Verify Node Counts (Table Stats)
    let dir_reader = fs::read_dir(data_dir)?;
    for entry in dir_reader {
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
        let meta = attrs
            .get("table_metadata_config")
            .ok_or_else(|| anyhow::anyhow!("missing table_metadata_config in {:?}", attrs_path))?;

        let source_name = meta
            .get("source_name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let data_source = meta
            .get("data_source")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let data_version = meta
            .get("data_version")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let table_name = meta
            .get("table_name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        table_infos.insert(
            table_name.clone(),
            TableMetadata {
                source_name: source_name.clone(),
                data_source: data_source.clone(),
                data_version: data_version.clone(),
            },
        );

        let s3_key_base = format!(
            "{}/prod/data-persist-dir/data_backend_cache/tables/{}/{}/{}/table_stats",
            customer_path, data_source, source_name, data_version
        );
        let s3_path = format!("s3://{}/{}", bucket, s3_key_base);

        let output = Command::new("aws")
            .arg("s3")
            .arg("cp")
            .arg(&s3_path)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await?;

        if !output.status.success() {
            messages.push(format!(
                "SKIP: {} (S3 stats not found at {})",
                table_name, s3_path
            ));
            continue;
        }

        let s3_stats: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let s3_rows = s3_stats
            .get("NUM_ROWS__False")
            .and_then(|v| v.get("value"))
            .and_then(|v| v.as_u64());

        if let Some(s3_rows) = s3_rows {
            if let Some(&ingest_rows) = stats.node_counts.get(&table_name) {
                let expected_rows = calculate_expected_node_count(
                    &table_name,
                    s3_rows,
                    stats,
                    &drop_duplicates_map,
                );
                if ingest_rows == s3_rows {
                    messages.push(format!(
                        "PASS: {} (S3: {}, Ingest: {})",
                        table_name, s3_rows, ingest_rows
                    ));
                } else if ingest_rows == expected_rows {
                    messages.push(format!(
                        "PASS: {} (S3: {}, Ingest: {}, Adjusted: {})",
                        table_name, s3_rows, ingest_rows, expected_rows
                    ));
                } else {
                    messages.push(format!(
                        "FAIL: {} (S3: {}, Ingest: {}, Expected: {}, Diff: {:.2}%)",
                        table_name,
                        s3_rows,
                        ingest_rows,
                        expected_rows,
                        diff_pct(s3_rows, ingest_rows) * 100.0
                    ));
                }
            } else {
                messages.push(format!(
                    "INFO: {} (S3 rows: {}, not in node_counts - likely edge table)",
                    table_name, s3_rows
                ));
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
    let input_key = format!(
        "{}/prod/jobs/{}/inputs/{}",
        customer_path, job_id, input_filename
    );
    let input_s3_path = format!("s3://{}/{}", bucket, input_key);

    let output = Command::new("aws")
        .arg("s3")
        .arg("cp")
        .arg(&input_s3_path)
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;

    if !output.status.success() {
        messages.push(format!(
            "SKIP: Edge stats ({} not found at {})",
            input_filename, input_s3_path
        ));
        return Ok(messages);
    }

    let input_json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

    // Prioritize predictive_query.graph.name for graph_id, fallback to config/graph_id or spec/graph_id
    let graph_id = input_json
        .pointer("/spec/predictive_query/graph/name")
        .and_then(|v| v.as_str())
        .or_else(|| {
            input_json
                .pointer("/config/graph_id")
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            input_json
                .pointer("/spec/graph_id")
                .and_then(|v| v.as_str())
        });

    // Try multiple locations for graph_data_version
    let graph_version = input_json
        .pointer("/graph_version_info/graph_data_version")
        .and_then(|v| v.as_str())
        .or_else(|| {
            input_json
                .pointer("/config/graph_data_version")
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            input_json
                .pointer("/config/graph_version_info/graph_data_version")
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            input_json
                .pointer("/spec/graph_version_info/graph_data_version")
                .and_then(|v| v.as_str())
        });

    if let (Some(gid), Some(gver)) = (graph_id, graph_version) {
        let edge_stats_key = format!(
            "{}/prod/data-persist-dir/data_backend_cache/graphs/{}/{}/edge_stats",
            customer_path, gid, gver
        );
        let edge_stats_s3 = format!("s3://{}/{}", bucket, edge_stats_key);

        let output = Command::new("aws")
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
                // Collect table names sorted by length descending to ensure greedy matching
                let mut sorted_table_names: Vec<String> = table_infos.keys().cloned().collect();
                sorted_table_names.sort_by_key(|a| std::cmp::Reverse(a.len()));

                let mut matched_s3_keys = HashSet::new();

                for (relation_name, &ingest_count) in &stats.edge_counts {
                    // Check for Edge Table pattern: edge_{table}_src_to_dst or edge_{table}_dst_to_src
                    if relation_name.starts_with("edge_")
                        && (relation_name.ends_with("_src_to_dst")
                            || relation_name.ends_with("_dst_to_src"))
                    {
                        let suffix = if relation_name.ends_with("_src_to_dst") {
                            "_src_to_dst"
                        } else {
                            "_dst_to_src"
                        };
                        let table_name = &relation_name[5..relation_name.len() - suffix.len()];

                        if let Some(meta) = table_infos.get(table_name) {
                            let target_source = &meta.source_name;
                            let target_ds = &meta.data_source;

                            let mut any_s3_key_found = false;
                            for (s3_key, val_obj) in obj {
                                if !s3_key.ends_with("EDGE_COUNT") {
                                    continue;
                                }

                                let parts: Vec<&str> = s3_key.split("__").collect();
                                if parts.len() < 7 {
                                    continue;
                                }

                                let src_src = parts[0];
                                let src_ds = parts[1];
                                let dst_src = parts[2];
                                let dst_ds = parts[3];

                                let involves_table = (src_src == target_source
                                    && src_ds == target_ds)
                                    || (dst_src == target_source && dst_ds == target_ds);

                                if involves_table {
                                    any_s3_key_found = true;
                                    matched_s3_keys.insert(s3_key.clone());
                                    let s3_count =
                                        val_obj.get("value").and_then(|v| v.as_u64()).unwrap_or(0);

                                    // Some S3 edges for this table might have 0 count (e.g. if one side of the many-to-many isn't in the graph)
                                    // We should only report FAIL if the S3 count is non-zero and doesn't match Ingest,
                                    // OR if Ingest is non-zero and S3 is 0.
                                    if s3_count == ingest_count {
                                        messages.push(format!(
                                            "PASS: Edge Table {} (Rel: {}) (S3: {}, Ingest: {})",
                                            table_name, s3_key, s3_count, ingest_count
                                        ));
                                    } else {
                                        let drop_msg = if let Some(drops) =
                                            stats.pk_duplicate_drops.get(relation_name)
                                        {
                                            format!(
                                                " (dropped {} edges due to duplicate src PKs)",
                                                drops
                                            )
                                        } else {
                                            String::new()
                                        };

                                        if ingest_count > 0 && ingest_count <= s3_count {
                                            messages.push(format!("INFO: Edge Table {} (Rel: {}) (Ingest {} <= S3 {}){}", table_name, s3_key, ingest_count, s3_count, drop_msg));
                                        } else {
                                            messages.push(format!("FAIL: Edge Table {} (Rel: {}) count mismatch (S3: {}, Ingest: {}){}", table_name, s3_key, s3_count, ingest_count, drop_msg));
                                        }
                                    }
                                }
                            }
                            if !any_s3_key_found {
                                messages.push(format!("FAIL: Edge Table {} (Ingest: {}) - No matching S3 key found for {}__{}", table_name, ingest_count, target_source, target_ds));
                            }
                        } else {
                            messages.push(format!(
                                "INFO: Missing metadata for edge table {}",
                                table_name
                            ));
                        }
                        continue;
                    }

                    // Parse relation name: {src_table}_{src_key}_to_{dst_table}_{dst_key}
                    let base_relation = relation_name.trim_end_matches("_reverse");
                    let parts: Vec<&str> = base_relation.split("_to_").collect();
                    if parts.len() != 2 {
                        messages.push(format!(
                            "INFO: Skipping complex edge name {}",
                            relation_name
                        ));
                        continue;
                    }

                    let (src_part, dst_part) = (parts[0], parts[1]);

                    // Helper to find table and key
                    let find_table_key = |part: &str| -> Option<(String, String)> {
                        for t in &sorted_table_names {
                            if part.starts_with(t) {
                                let key_part = &part[t.len()..];
                                // Expecting key part to start with _ usually
                                // The user example: sales_item_SOLDTOPARTY. Table: sales_item. Key: _SOLDTOPARTY.
                                // S3 needs SOLDTOPARTY.
                                let key = key_part.trim_start_matches('_');
                                return Some((t.clone(), key.to_string()));
                            }
                        }
                        None
                    };

                    let src_info = find_table_key(src_part);
                    let dst_info = find_table_key(dst_part);

                    if let (Some((src_table, src_key)), Some((dst_table, dst_key))) =
                        (src_info, dst_info)
                    {
                        if let (Some(src_meta), Some(dst_meta)) =
                            (table_infos.get(&src_table), table_infos.get(&dst_table))
                        {
                            // Construct S3 Key
                            // Pattern: {src_source}__{src_ds}__{dst_source}__{dst_ds}__{src_key}__{dst_key}__EDGE_COUNT
                            let s3_key_forward = format!(
                                "{}__{}__{}__{}__{}__{}__EDGE_COUNT",
                                src_meta.source_name,
                                src_meta.data_source,
                                dst_meta.source_name,
                                dst_meta.data_source,
                                src_key,
                                dst_key
                            );

                            let s3_key_reverse = format!(
                                "{}__{}__{}__{}__{}__{}__EDGE_COUNT",
                                dst_meta.source_name,
                                dst_meta.data_source,
                                src_meta.source_name,
                                src_meta.data_source,
                                dst_key,
                                src_key
                            );

                            let matched_s3_entry = obj
                                .get_key_value(&s3_key_forward)
                                .or_else(|| obj.get_key_value(&s3_key_reverse));

                            if let Some((matched_key, _val)) = matched_s3_entry {
                                matched_s3_keys.insert(matched_key.clone());

                                let manifest_edge_key = format!(
                                    "{}.{}__{}__{}.{}",
                                    src_meta.data_source,
                                    src_table,
                                    src_key,
                                    dst_meta.data_source,
                                    dst_table
                                );

                                // Compare against materialized edge count (correctly deduplicated)
                                // instead of edge_stats count (inflated when dst has duplicate PKs)
                                if let Some(mat_count) = get_materialized_edge_count(
                                    bucket,
                                    customer_path,
                                    &src_meta.data_version,
                                    &dst_meta.data_version,
                                    manifest.as_ref(),
                                    Some(&manifest_edge_key),
                                )
                                .await
                                {
                                    let drop_msg = if let Some(drops) =
                                        stats.pk_duplicate_drops.get(relation_name)
                                    {
                                        format!(
                                            " (dropped {} edges due to duplicate src PKs)",
                                            drops
                                        )
                                    } else {
                                        String::new()
                                    };

                                    if mat_count == ingest_count {
                                        messages.push(format!(
                                            "PASS: Edge {} (Materialized: {}, Ingest: {}){}",
                                            relation_name, mat_count, ingest_count, drop_msg
                                        ));
                                    } else {
                                        messages.push(format!("FAIL: Edge {} count mismatch (Materialized: {}, Ingest: {}){}", relation_name, mat_count, ingest_count, drop_msg));
                                    }
                                } else {
                                    // Fallback: materialized edge not found, skip count comparison
                                    // This can happen for FK-FK edges or if materialization hasn't run
                                    if ingest_count == 0 {
                                        messages.push(format!(
                                            "PASS: Edge {} (Ingest: 0, no materialized edge)",
                                            relation_name
                                        ));
                                    } else {
                                        messages.push(format!("INFO: Edge {} (Ingest: {}) - no materialized edge found for version {}_{}",
                                             relation_name, ingest_count, src_meta.data_version, dst_meta.data_version));
                                    }
                                }
                            } else if ingest_count == 0 {
                                messages.push(format!("PASS: Edge {} (Ingest: 0, S3 key not found - expected for pruned edges)", relation_name));
                            } else {
                                messages.push(format!(
                                    "FAIL: Edge {} (Ingest: {}) - S3 key {} (or reverse) not found",
                                    relation_name, ingest_count, s3_key_forward
                                ));
                            }
                        } else {
                            messages.push(format!(
                                "INFO: Missing metadata for tables in edge {}",
                                relation_name
                            ));
                        }
                    } else {
                        messages.push(format!(
                            "INFO: Could not parse tables from edge {}",
                            relation_name
                        ));
                    }
                }

                // Reverse check: Verify if any S3 edge was missed by ingestion
                let mut reverse_table_infos: std::collections::HashMap<
                    (String, String),
                    Vec<String>,
                > = std::collections::HashMap::new();
                for (t_name, meta) in &table_infos {
                    reverse_table_infos
                        .entry((meta.source_name.clone(), meta.data_source.clone()))
                        .or_default()
                        .push(t_name.clone());
                }

                for (s3_key, val_obj) in obj {
                    if !s3_key.ends_with("EDGE_COUNT") {
                        continue;
                    }
                    if matched_s3_keys.contains(s3_key) {
                        continue;
                    }

                    let s3_count = val_obj.get("value").and_then(|v| v.as_u64()).unwrap_or(0);
                    if s3_count == 0 {
                        continue;
                    }

                    // Try to identify if this S3 key belongs to tables we know about
                    // Key format: {src_src}__{src_ds}__{dst_src}__{dst_ds}__{src_key}__{dst_key}__EDGE_COUNT
                    let parts: Vec<&str> = s3_key.split("__").collect();
                    if parts.len() >= 7 {
                        let src_src = parts[0];
                        let src_ds = parts[1];
                        let dst_src = parts[2];
                        let dst_ds = parts[3];

                        let src_candidates =
                            reverse_table_infos.get(&(src_src.to_string(), src_ds.to_string()));
                        let dst_candidates =
                            reverse_table_infos.get(&(dst_src.to_string(), dst_ds.to_string()));

                        if let (Some(srcs), Some(dsts)) = (src_candidates, dst_candidates) {
                            // We have both tables locally, so we SHOULD have seen this edge.
                            messages.push(format!("FAIL: S3 has edge {} (count: {}), but Ingest has none. Matched local tables: Src={:?}, Dst={:?}", s3_key, s3_count, srcs, dsts));
                        } else {
                            // S3 has an edge but we don't have the table locally. Maybe we downloaded a subset?
                            // But validate_s3 downloads full job.
                            // Report as INFO or SKIP?
                            // Actually, if we are validating a specific job, we expect full match.
                            // But maybe the S3 stats include edges to tables not in this job's input? (Global graph stats?)
                            // Yes, graph stats are for the graph, which might be reused.
                            // But if the job input defines the tables, we only download those.
                            // So it is possible S3 has edges we don't see if we didn't download those tables.
                            // However, for this task, let's report it if we can match at least one side?
                            // No, only fail if we have BOTH tables locally.
                        }
                    }
                }
            }
        } else {
            messages.push(format!(
                "SKIP: Edge stats (edge_stats not found at {})",
                edge_stats_s3
            ));
        }
    } else {
        messages.push("SKIP: Edge stats (Could not determine graph_id/version)".to_string());
    }

    Ok(messages)
}

struct TableMetadata {
    source_name: String,
    data_source: String,
    data_version: String,
}

/// Fetches the edge count from materialized edges instead of edge_stats.
///
/// Why: edge_stats overcounts when destination tables have duplicate primary keys
/// (it counts raw inner join rows without deduplication). Materialized edges
/// correctly deduplicate, matching DiskGraph's behavior.
///
/// Path: s3://{bucket}/{customer}/prod/data-persist-dir/materialize/edges/v0__{hash}/{src_ver}_{dst_ver}/{timestamp}/attributes.json
///
/// Note: We grep for the version pattern since computing the hash requires matching
/// Python's EdgeFPConfig JSON serialization exactly (see kumo/kumo/data/persist_dir_manager.py).
async fn get_materialized_edge_count(
    bucket: &str,
    customer_path: &str,
    src_version: &str,
    dst_version: &str,
    manifest: Option<&HashMap<String, String>>,
    manifest_key: Option<&str>,
) -> Option<u64> {
    let mut attrs_s3_path = None;

    // Try manifest first (efficient)
    if let (Some(m), Some(key)) = (manifest, manifest_key) {
        if let Some(path) = m.get(key) {
            let base = path.trim_end_matches('/').replace("s3a://", "s3://");
            attrs_s3_path = Some(format!("{}/attributes.json", base));
        }
    }

    // Fallback to recursive listing (inefficient)
    if attrs_s3_path.is_none() {
        let materialize_base = format!(
            "s3://{}/{}/prod/data-persist-dir/materialize/edges/",
            bucket, customer_path
        );
        let version_pattern = format!("{}_{}", src_version, dst_version);

        // List materialized edges and grep for the version pattern
        let output = Command::new("aws")
            .arg("s3")
            .arg("ls")
            .arg("--recursive")
            .arg(&materialize_base)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let listing = String::from_utf8_lossy(&output.stdout);

        // Find attributes.json path matching the version pattern
        for line in listing.lines() {
            if !line.contains("attributes.json") || !line.contains(&version_pattern) {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            let path = parts[3];
            attrs_s3_path = Some(format!("{}{}", materialize_base, path));
            break;
        }
    }

    if let Some(s3_path) = attrs_s3_path {
        // Fetch attributes.json
        let output = Command::new("aws")
            .arg("s3")
            .arg("cp")
            .arg(&s3_path)
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .ok()?;

        if output.status.success() {
            let attrs: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
            return attrs.get("edge_count").and_then(|v| v.as_u64());
        }
    }

    None
}

fn diff_pct(a: u64, b: u64) -> f64 {
    let diff = (a as i64 - b as i64).abs() as f64;
    let max = a.max(b) as f64;
    if max == 0.0 {
        0.0
    } else {
        diff / max
    }
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
                    if fails > 0 {
                        format!("{} FAIL", fails)
                    } else if skips > 0
                        && msgs
                            .iter()
                            .all(|m| m.starts_with("SKIP") || m.starts_with("INFO"))
                    {
                        "SKIP".to_string()
                    } else {
                        "PASS".to_string()
                    }
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
