//! CLI tool to convert Kumo training job configs to DiskGraph source metadata.
//!
//! Usage:
//!     kumo-to-diskgraph --job-id trainingjob-xxx --env prod --tenant coinbase --output config.json
//!
//! With ingestion:
//!     kumo-to-diskgraph --job-id trainingjob-xxx --env prod --tenant coinbase --run-ingest --aws-profile normal-profile

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use diskgraph_ingest::progress::build_progress;
use diskgraph_ingest::source_metadata::{
    ColumnSpec, IngestedType, RelationshipSpec, SourceMetadata, TableFormat, TableSpec,
    TimestampUnit,
};
use diskgraph_ingest::{ingest_with_stats_from_source, Dest, IngestOptions};
use serde::Deserialize;

/// Environment configurations for Kumo S3 buckets
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Environment {
    Prod,
    Staging,
    Dev,
}

impl Environment {
    fn bucket(&self) -> &'static str {
        match self {
            Environment::Prod => "kumo-prod-dataplane-bucket",
            Environment::Staging => "kumo-staging-dataplane-bucket",
            Environment::Dev => "kumo-dev-dataplane-bucket",
        }
    }

    fn tenant_path(&self, tenant: &str) -> String {
        match self {
            Environment::Prod => format!("{}/prod", tenant),
            Environment::Staging => "uat".to_string(),
            Environment::Dev => format!("test/{}", tenant),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert Kumo training job to DiskGraph config")]
struct Args {
    /// Kumo training job ID (e.g., trainingjob-xxxxx)
    #[arg(long)]
    job_id: String,

    /// Kumo environment
    #[arg(long, value_enum)]
    env: Environment,

    /// Kumo tenant name (e.g., coinbase, demo)
    #[arg(long)]
    tenant: String,

    /// Output path for DiskGraph config
    #[arg(long, default_value = "diskgraph_config.json")]
    output: PathBuf,

    /// Dataset name (default: derived from job ID)
    #[arg(long)]
    dataset_name: Option<String>,

    /// Run ingestion after generating config
    #[arg(long)]
    run_ingest: bool,

    /// Directory for downloaded parquet files (default: ./temp_data)
    #[arg(long, default_value = "temp_data")]
    data_dir: PathBuf,

    /// Output directory for DiskGraph index (default: ./diskgraph_output)
    #[arg(long, default_value = "diskgraph_output")]
    diskgraph_output: PathBuf,

    /// AWS CLI profile to use for S3 operations
    #[arg(long)]
    aws_profile: Option<String>,
}

// ============================================================================
// Kumo JSON Types (from spec.predictive_query.graph)
// ============================================================================

#[derive(Debug, Deserialize)]
struct TrainingJobInput {
    spec: Spec,
}

#[derive(Debug, Deserialize)]
struct Spec {
    predictive_query: PredictiveQuery,
}

#[derive(Debug, Deserialize)]
struct PredictiveQuery {
    graph: Graph,
}

#[derive(Debug, Deserialize)]
struct Graph {
    #[allow(dead_code)]
    data_source_dict: HashMap<String, DataSource>,
    tables: Vec<KumoTable>,
}

#[derive(Debug, Deserialize)]
struct DataSource {
    #[allow(dead_code)]
    root_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KumoTable {
    table_name: String,
    data_source: String,
    source_name: Option<String>,
    data_version: Option<String>,
    pkey: Option<PrimaryKey>,
    fkeys: Vec<ForeignKey>,
    cols: Vec<KumoColumn>,
    time_col: Option<TimeCol>,
}

#[derive(Debug, Deserialize)]
struct PrimaryKey {
    name: String,
    dtype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ForeignKey {
    name: String,
    dtype: Option<String>,
    dst_table: String,
    dst_key: String,
}

#[derive(Debug, Deserialize)]
struct KumoColumn {
    name: String,
    dtype: Option<String>,
    timestamp_unit: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TimeCol {
    name: String,
    #[allow(dead_code)]
    dtype: Option<String>,
}

// ============================================================================
// Conversion Logic
// ============================================================================

fn map_kumo_dtype_to_diskgraph(dtype: &str) -> IngestedType {
    match dtype.to_lowercase().as_str() {
        "bool" => IngestedType::Bool,
        "int" | "int64" => IngestedType::Int64,
        "int32" => IngestedType::Int32,
        "int16" => IngestedType::Int16,
        "int8" => IngestedType::Int8,
        "uint64" => IngestedType::UInt64,
        "uint32" => IngestedType::UInt32,
        "uint16" => IngestedType::UInt16,
        "uint8" => IngestedType::UInt8,
        "float" | "float64" => IngestedType::Float64,
        "float32" => IngestedType::Float32,
        "string" => IngestedType::String,
        "date" => IngestedType::Date,
        "datetime" | "timestamp" | "timestamp_ns" => IngestedType::Timestamp,
        "binary" => IngestedType::Bytes,
        "uuid" => IngestedType::Uuid,
        _ => {
            tracing::warn!("Unknown dtype '{}', defaulting to string", dtype);
            IngestedType::String
        }
    }
}

fn map_kumo_timestamp_unit(unit: &str) -> Option<TimestampUnit> {
    match unit.to_lowercase().as_str() {
        "s" | "seconds" => Some(TimestampUnit::Seconds),
        "ms" | "millis" | "milliseconds" => Some(TimestampUnit::Millis),
        "us" | "micros" | "microseconds" => Some(TimestampUnit::Micros),
        "ns" | "nanos" | "nanoseconds" => Some(TimestampUnit::Nanos),
        _ => None,
    }
}

/// Parse dst_table like "hnm_simple.customers_1" -> "customers_1"
fn parse_dst_table(dst_table: &str) -> &str {
    dst_table.rsplit('.').next().unwrap_or(dst_table)
}

/// Build the persist-dir ingest path for a table (fpv3 format)
/// Pattern: s3://{bucket}/{tenant_path}/data-persist-dir/ingest/fpv3.{data_source}.{source_name}/{data_version}/
fn build_persist_dir_path_fpv3(
    bucket: &str,
    tenant_path: &str,
    data_source: &str,
    source_name: &str,
    data_version: &str,
) -> String {
    format!(
        "s3://{}/{}/data-persist-dir/ingest/fpv3.{}.{}/{}/",
        bucket, tenant_path, data_source, source_name, data_version
    )
}

/// Find the persist-dir path, trying multiple patterns
fn find_persist_dir_path(
    bucket: &str,
    tenant_path: &str,
    data_source: &str,
    source_name: &str,
    data_version: Option<&str>,
    table_name: &str,
    profile: Option<&str>,
) -> Result<String> {
    let old_ingest_base = format!(
        "s3://{}/{}/data-persist-dir-old/data-persist-dir/ingest/",
        bucket, tenant_path
    );

    // If we have data_version, try version-based paths
    if let Some(version) = data_version {
        // Try fpv3 path first (new format)
        let fpv3_path = build_persist_dir_path_fpv3(bucket, tenant_path, data_source, source_name, version);
        if s3_path_exists(&fpv3_path, profile) {
            tracing::info!("  Found at fpv3 path: {}", fpv3_path);
            return Ok(fpv3_path);
        }

        // Try fpv1 path by version
        tracing::info!("  Searching for fpv1 path with version {} ...", version);
        if let Some(fpv1_path) = find_fpv1_path_by_version(&old_ingest_base, version, profile) {
            tracing::info!("  Found at fpv1 path: {}", fpv1_path);
            return Ok(fpv1_path);
        }
    }

    // Try to find by table name (for kumo_views_connector tables without data_version)
    // Pattern: kumo_views_connector.{table_name}/
    let view_path = format!("{}{}.{}/", old_ingest_base, data_source, table_name);
    tracing::info!("  Trying view path: {}", view_path);
    if s3_path_exists(&view_path, profile) {
        tracing::info!("  Found at view path: {}", view_path);
        return Ok(view_path);
    }

    // Also try with source_name if different from table_name
    if source_name != table_name {
        let view_path2 = format!("{}{}.{}/", old_ingest_base, data_source, source_name);
        tracing::info!("  Trying view path (source_name): {}", view_path2);
        if s3_path_exists(&view_path2, profile) {
            tracing::info!("  Found at view path: {}", view_path2);
            return Ok(view_path2);
        }
    }

    Err(anyhow!(
        "Could not find data for table {} (data_source={}, data_version={:?}) in any known path",
        table_name, data_source, data_version
    ))
}

/// Check if an S3 path exists (has files)
fn s3_path_exists(s3_path: &str, profile: Option<&str>) -> bool {
    let mut cmd = Command::new("aws");
    cmd.args(["s3", "ls", s3_path, "--recursive", "--page-size", "1"]);
    if let Some(p) = profile {
        cmd.args(["--profile", p]);
    }

    match cmd.output() {
        Ok(output) => output.status.success() && !output.stdout.is_empty(),
        Err(_) => false,
    }
}

/// Find fpv1 path that contains the given data_version
fn find_fpv1_path_by_version(base_path: &str, data_version: &str, profile: Option<&str>) -> Option<String> {
    // List all fpv1 folders
    let mut cmd = Command::new("aws");
    cmd.args(["s3", "ls", base_path]);
    if let Some(p) = profile {
        cmd.args(["--profile", p]);
    }

    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Look for fpv1 folders and check if they contain the data_version
    for line in stdout.lines() {
        if let Some(folder) = line.split_whitespace().last() {
            if folder.starts_with("fpv1.") {
                let folder_path = format!("{}{}{}/", base_path, folder, data_version);
                if s3_path_exists(&folder_path, profile) {
                    return Some(folder_path);
                }
            }
        }
    }

    None
}

// ============================================================================
// S3 Operations (using AWS CLI)
// ============================================================================

fn s3_download(s3_url: &str, profile: Option<&str>) -> Result<Vec<u8>> {
    tracing::info!("Downloading: {}", s3_url);

    let mut cmd = Command::new("aws");
    if let Some(p) = profile {
        cmd.args(["--profile", p]);
    }
    cmd.args(["s3", "cp", s3_url, "-"]);

    let output = cmd.output().context("Failed to run aws s3 cp")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("aws s3 cp failed: {}", stderr));
    }

    Ok(output.stdout)
}

/// List parquet files in an S3 directory
fn s3_list_parquet_files(s3_url: &str, profile: Option<&str>) -> Result<Vec<String>> {
    tracing::info!("Listing parquet files in: {}", s3_url);

    let mut cmd = Command::new("aws");
    if let Some(p) = profile {
        cmd.args(["--profile", p]);
    }
    cmd.args(["s3", "ls", s3_url]);

    let output = cmd.output().context("Failed to run aws s3 ls")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("aws s3 ls failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut files = Vec::new();
    for line in stdout.lines() {
        // Lines look like: "2026-01-24 15:04:06     804800 part-00000-xxx.snappy.parquet"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let filename = parts[3];
            if filename.ends_with(".parquet") {
                files.push(format!("{}{}", s3_url.trim_end_matches('/').to_string() + "/", filename));
            }
        }
    }

    Ok(files)
}

/// Download a directory from S3 using aws s3 sync
fn s3_sync(s3_url: &str, local_dir: &PathBuf, profile: Option<&str>) -> Result<()> {
    tracing::info!("Syncing {} -> {}", s3_url, local_dir.display());

    std::fs::create_dir_all(local_dir)?;

    let mut cmd = Command::new("aws");
    if let Some(p) = profile {
        cmd.args(["--profile", p]);
    }
    cmd.args(["s3", "sync", s3_url, &local_dir.to_string_lossy()]);

    let output = cmd.output().context("Failed to run aws s3 sync")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("aws s3 sync failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        tracing::info!("{}", stdout.trim());
    }

    Ok(())
}

// ============================================================================
// Main Conversion
// ============================================================================

fn convert_to_diskgraph(
    job_input: &TrainingJobInput,
    dataset_name: &str,
    bucket: &str,
    tenant_path: &str,
    profile: Option<&str>,
) -> Result<SourceMetadata> {
    let graph = &job_input.spec.predictive_query.graph;

    let mut diskgraph_tables = Vec::new();
    let mut relationships = Vec::new();

    for table in &graph.tables {
        tracing::info!("\nProcessing table: {}", table.table_name);

        let data_source = &table.data_source;
        let source_name = table.source_name.as_deref().unwrap_or(&table.table_name);
        let data_version = table.data_version.as_deref();

        tracing::info!("  Data source: {}", data_source);
        tracing::info!("  Source name: {}", source_name);
        tracing::info!("  Data version: {:?}", data_version);

        // Find persist-dir path (tries multiple patterns)
        let s3_path = find_persist_dir_path(bucket, tenant_path, data_source, source_name, data_version, &table.table_name, profile)?;

        // Collect all columns: pkey + fkeys + cols
        let mut all_columns: Vec<ColumnSpec> = Vec::new();

        // Add primary key column
        if let Some(pkey) = &table.pkey {
            let dtype = pkey.dtype.as_deref().unwrap_or("string");
            all_columns.push(ColumnSpec {
                name: pkey.name.clone(),
                source_dtype: None,
                ingested_dtype: map_kumo_dtype_to_diskgraph(dtype),
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            });
        }

        // Add foreign key columns and extract relationships
        for fkey in &table.fkeys {
            let dtype = fkey.dtype.as_deref().unwrap_or("string");
            all_columns.push(ColumnSpec {
                name: fkey.name.clone(),
                source_dtype: None,
                ingested_dtype: map_kumo_dtype_to_diskgraph(dtype),
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            });

            // Create relationship: this table's fkey -> dst_table's dst_key
            let dst_table_name = parse_dst_table(&fkey.dst_table);
            tracing::info!(
                "  FK: {}.{} -> {}.{}",
                table.table_name,
                fkey.name,
                dst_table_name,
                fkey.dst_key
            );

            relationships.push(RelationshipSpec {
                name: None,
                src_table: table.table_name.clone(),
                src_column: fkey.name.clone(),
                dst_table: dst_table_name.to_string(),
                dst_column: fkey.dst_key.clone(),
            });
        }

        // Add regular columns
        for col in &table.cols {
            let dtype = col.dtype.as_deref().unwrap_or("string");
            let timestamp_unit = col
                .timestamp_unit
                .as_deref()
                .and_then(map_kumo_timestamp_unit);

            all_columns.push(ColumnSpec {
                name: col.name.clone(),
                source_dtype: None,
                ingested_dtype: map_kumo_dtype_to_diskgraph(dtype),
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit,
                precision: None,
                scale: None,
            });
        }

        // Get primary key name
        let primary_key = table.pkey.as_ref().map(|pk| pk.name.clone());

        // Get time column name
        let create_time = table.time_col.as_ref().map(|tc| tc.name.clone());

        diskgraph_tables.push(TableSpec {
            name: table.table_name.clone(),
            format: TableFormat::Parquet,
            paths: vec![s3_path],
            columns: all_columns,
            primary_key,
            create_time,
            weight: None,
            edge_table: None,
            rowpack: None,
        });
    }

    Ok(SourceMetadata {
        dataset_name: dataset_name.to_string(),
        tables: diskgraph_tables,
        relationships,
        edge_sort_keys: None,
        rowpack: None,
        num_shards: None,
        edge_format: None,
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let args = Args::parse();

    let bucket = args.env.bucket();
    let tenant_path = args.env.tenant_path(&args.tenant);
    let profile = args.aws_profile.as_deref();

    tracing::info!("S3 bucket: {}", bucket);
    tracing::info!("Tenant path: {}", tenant_path);
    if let Some(p) = profile {
        tracing::info!("AWS profile: {}", p);
    }

    // Download training job input
    let job_input_path = format!(
        "s3://{}/{}/jobs/{}/inputs/training_job_input.json",
        bucket, tenant_path, args.job_id
    );
    tracing::info!("\nLoading training job: {}", args.job_id);

    let bytes = s3_download(&job_input_path, profile)?;
    let job_input: TrainingJobInput =
        serde_json::from_slice(&bytes).context("Failed to parse training_job_input.json")?;

    // Determine dataset name
    let dataset_name = args
        .dataset_name
        .unwrap_or_else(|| args.job_id.replace("trainingjob-", ""));

    // Convert to DiskGraph format (using persist-dir paths)
    tracing::info!("\nConverting to DiskGraph format...");
    let mut source_metadata = convert_to_diskgraph(
        &job_input,
        &dataset_name,
        bucket,
        &tenant_path,
        profile,
    )?;

    // If run_ingest, download parquets and update paths
    if args.run_ingest {
        tracing::info!(
            "\nDownloading parquet files to {}...",
            args.data_dir.display()
        );
        std::fs::create_dir_all(&args.data_dir)?;

        for table in &mut source_metadata.tables {
            let mut new_paths = Vec::new();
            for path in &table.paths {
                if path.starts_with("s3://") {
                    // Download to local directory
                    let local_dir = args.data_dir.join(&table.name);
                    s3_sync(path, &local_dir, profile)?;

                    // Glob for parquet files in the directory
                    let local_dir = local_dir.canonicalize()?;
                    for entry in std::fs::read_dir(&local_dir)? {
                        let entry = entry?;
                        let file_path = entry.path();
                        if file_path.extension().map(|e| e == "parquet").unwrap_or(false) {
                            new_paths.push(file_path.to_string_lossy().to_string());
                        }
                    }
                } else {
                    new_paths.push(path.clone());
                }
            }
            if new_paths.is_empty() {
                return Err(anyhow!("No parquet files found for table {}", table.name));
            }
            tracing::info!("  {}: {} parquet files", table.name, new_paths.len());
            table.paths = new_paths;
        }
    }

    // Write output config
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let json = serde_json::to_string_pretty(&source_metadata)?;
    std::fs::write(&args.output, &json)?;

    tracing::info!(
        "\n\nDiskGraph config written to: {}",
        args.output.display()
    );
    tracing::info!("  Tables: {}", source_metadata.tables.len());
    tracing::info!("  Relationships: {}", source_metadata.relationships.len());

    // Run ingestion if requested
    if args.run_ingest {
        tracing::info!("\nRunning ingestion...");

        let options = IngestOptions {
            profile_dir: None,
            progress: build_progress(),
            parallel: true,
            index_mem_budget_bytes: None,
        };

        let output_str = args.diskgraph_output.to_string_lossy().to_string();
        let (_metadata, stats) = ingest_with_stats_from_source(
            source_metadata,
            Some(args.output.parent().unwrap_or(&PathBuf::from("."))),
            &output_str,
            Dest::Local,
            options,
        )?;

        tracing::info!("\n\nIngestion complete!");
        tracing::info!("  Output: {}", stats.output);
        tracing::info!(
            "  Node tables: {} ({} nodes)",
            stats.node_tables,
            stats.total_nodes
        );
        tracing::info!(
            "  Edge relations: {} ({} edges)",
            stats.edge_relations,
            stats.total_edges
        );
    }

    Ok(())
}
