use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use chrono::{NaiveDate, NaiveDateTime};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, CloudTransfer,
    CloudTransferConfig, GcsConfig, S3Config,
};
use diskgraph_ingest::graph_config::{
    ColumnSpec, DataSource, EdgeTableSpec, FKeySpec, ForeignKeySpec, IngestedType, GraphConfig,
    TableFormat, TableSpec, TimestampUnit,
};
use diskgraph_ingest::config_converter::map_kumo_dtype;
use diskgraph_ingest::utils::is_edge_table_compatible;
use arrow_array::{
    Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, RecordBatch,
    StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{ArrowError, DataType, Schema};
use futures::StreamExt;
use object_store::{path::Path as ObjectPath, ObjectStore};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::runtime::Runtime;
use url::Url;
use serde::Deserialize;

#[cfg(test)]
mod tests_infer;

const DEFAULT_SAMPLE_ROWS: usize = 2_000;
const DEFAULT_CSV_SAMPLE_BYTES: u64 = 256 * 1024;
const DEFAULT_CSV_SAMPLE_ROWS: usize = 2_000;

// Internal type for FK inference - converted to FKeySpec at the end.
// dst_column is always the destination table's primary key, so we look it up when needed.
#[derive(Clone, Debug)]
struct InferredFKey {
    src_table: String,
    src_column: String,
    dst_table: String,
}

/// Convert inferred FKeys to FKeySpec embedded in tables
fn convert_inferred_fkeys_to_fkeyspecs(
    table_specs: &mut [TableSpec],
    inferred_fkeys: &[InferredFKey],
) {
    // Group by src_table
    let mut by_table: HashMap<&str, Vec<&InferredFKey>> = HashMap::new();
    for fk in inferred_fkeys {
        by_table
            .entry(fk.src_table.as_str())
            .or_default()
            .push(fk);
    }

    // Add fkeys to each table
    for table in table_specs {
        if let Some(fks) = by_table.get(table.name.as_str()) {
            for fk in fks {
                // Check if fkey already exists
                let exists = table.fkeys.iter().any(|existing| {
                    existing.src_column == fk.src_column && existing.dst_table == fk.dst_table
                });
                if !exists {
                    table.fkeys.push(FKeySpec {
                        src_column: fk.src_column.clone(),
                        dst_table: fk.dst_table.clone(),
                        forward: None,
                        reverse: None,
                    });
                }
            }
        }
    }
}

/// Input graph configuration provided by the caller to guide inference.
/// This is different from the output GraphConfig (from diskgraph-ingest).
#[derive(Clone, Debug, Deserialize)]
pub struct InputGraphConfig {
    pub tables: Vec<InputGraphTable>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputGraphTable {
    pub table_name: String,
    pub data_source: String,
    pub cols: Vec<InputGraphCol>,
    pub fkeys: Vec<InputGraphFKey>,
    pub pkey: Option<InputGraphPKey>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputGraphPKey {
    pub name: String,
    pub dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputGraphCol {
    pub name: String,
    pub dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputGraphFKey {
    pub name: String,
    pub dst_table: String,
    pub dst_key: String,
    pub dtype: String,
}

#[derive(Clone, Debug)]
pub struct InferOptions {
    pub sample_rows: usize,
    pub csv_sample_bytes: u64,
    pub csv_sample_rows: usize,
    pub validate_dir_schemas: bool,
    pub graph_config: Option<InputGraphConfig>,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self {
            sample_rows: DEFAULT_SAMPLE_ROWS,
            csv_sample_bytes: DEFAULT_CSV_SAMPLE_BYTES,
            csv_sample_rows: DEFAULT_CSV_SAMPLE_ROWS,
            validate_dir_schemas: false,
            graph_config: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferWarnings {
    pub ignored_files: Vec<String>,
    pub schema_mismatches: Vec<SchemaMismatch>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub metadata: GraphConfig,
    pub warnings: InferWarnings,
}

#[derive(Clone, Debug)]
pub struct SchemaMismatch {
    pub path: String,
    pub diffs: Vec<SchemaDiff>,
}

#[derive(Clone, Debug)]
pub struct SchemaDiff {
    pub index: usize,
    pub left: Option<ColumnSpec>,
    pub right: Option<ColumnSpec>,
}

#[derive(Clone, Debug)]
struct TableInput {
    name: String,
    format: TableFormat,
    paths: Vec<String>,
    columns: Vec<ColumnSpec>,
    total_bytes: u64,
}

#[derive(Clone, Debug)]
pub enum RootSource {
    Local(PathBuf),
    S3 { bucket: String, prefix: String },
    Gcs { bucket: String, prefix: String },
    Adls { account: String, container: String, prefix: String },
}

pub fn parse_root_source(input: &str) -> Result<RootSource> {
    if let Ok(url) = Url::parse(input) {
        match url.scheme() {
            "s3" => Ok(RootSource::S3 {
                bucket: url.host_str().unwrap_or_default().to_string(),
                prefix: url.path().trim_start_matches('/').to_string(),
            }),
            "gs" | "gcs" => Ok(RootSource::Gcs {
                bucket: url.host_str().unwrap_or_default().to_string(),
                prefix: url.path().trim_start_matches('/').to_string(),
            }),
            "az" | "adls" => {
                let account = url.host_str().unwrap_or_default().to_string();
                let mut parts = url.path().trim_start_matches('/').splitn(2, '/');
                let container = parts.next().unwrap_or_default().to_string();
                let prefix = parts.next().unwrap_or("").to_string();
                Ok(RootSource::Adls {
                    account,
                    container,
                    prefix,
                })
            }
            "file" => Ok(RootSource::Local(PathBuf::from(
                url.to_file_path()
                    .map_err(|_| anyhow!("invalid file url {}", input))?,
            ))),
            _ => Err(anyhow!("unsupported source scheme {}", url.scheme())),
        }
    } else {
        Ok(RootSource::Local(PathBuf::from(input)))
    }
}

pub fn infer_source_metadata(root: &RootSource, options: InferOptions) -> Result<InferenceResult> {
    let mut warnings = InferWarnings {
        ignored_files: Vec::new(),
        schema_mismatches: Vec::new(),
        notes: Vec::new(),
    };
    let mut tables = match root {
        RootSource::Local(path) => infer_local_tables(path, &options, &mut warnings)?,
        RootSource::S3 { bucket, prefix } => infer_remote_tables(
            root,
            bucket,
            prefix,
            &options,
            &mut warnings,
        )?,
        RootSource::Gcs { bucket, prefix } => infer_remote_tables(
            root,
            bucket,
            prefix,
            &options,
            &mut warnings,
        )?,
        RootSource::Adls {
            account,
            container,
            prefix,
        } => infer_remote_tables(
            root,
            &format!("{}/{}", account, container),
            prefix,
            &options,
            &mut warnings,
        )?,
    };

    if let Some(config) = &options.graph_config {
        let config_tables: HashMap<String, &InputGraphTable> = config
            .tables
            .iter()
            .map(|t| (t.table_name.clone(), t))
            .collect();

        tables.retain(|t| config_tables.contains_key(&t.name));

        for table in &mut tables {
            if let Some(cfg) = config_tables.get(&table.name) {
                let mut allowed_cols: HashSet<String> = cfg
                    .cols
                    .iter()
                    .map(|c| c.name.clone())
                    .collect();

                if let Some(pk) = &cfg.pkey {
                    allowed_cols.insert(pk.name.clone());
                }
                for fk in &cfg.fkeys {
                    allowed_cols.insert(fk.name.clone());
                }

                table.columns.retain(|c| allowed_cols.contains(&c.name));

                let mut col_dtypes: HashMap<String, String> = cfg
                    .cols
                    .iter()
                    .map(|c| (c.name.clone(), c.dtype.clone()))
                    .collect();

                if let Some(pk) = &cfg.pkey {
                    col_dtypes.insert(pk.name.clone(), pk.dtype.clone());
                }
                for fk in &cfg.fkeys {
                    col_dtypes.insert(fk.name.clone(), fk.dtype.clone());
                }

                for col in &mut table.columns {
                    if let Some(dtype_str) = col_dtypes.get(&col.name) {
                        if let Some(ingested_type) = map_kumo_dtype(dtype_str) {
                            col.ingested_dtype = ingested_type;
                            if !matches!(col.ingested_dtype, IngestedType::String | IngestedType::Bytes) {
                                col.fixed_len = None;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut table_specs: Vec<TableSpec> = tables
        .iter()
        .map(|table| TableSpec {
            name: table.name.clone(),
            columns: table.columns.clone(),
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: table.paths.clone() },
            fkeys: Vec::new(),
        })
        .collect();

    if let Some(config) = &options.graph_config {
        let config_tables: HashMap<String, &InputGraphTable> = config
            .tables
            .iter()
            .map(|t| (t.table_name.clone(), t))
            .collect();
        for table in &mut table_specs {
            if let Some(cfg) = config_tables.get(&table.name) {
                if let Some(pk) = &cfg.pkey {
                    if table.columns.iter().any(|c| c.name == pk.name) {
                        table.primary_key = Some(pk.name.clone());
                    }
                }
            }
        }
    }

    let mut inferred_fkeys = Vec::new();
    // Pass config to infer_primary_keys to respect explicit nulls
    infer_primary_keys(&mut table_specs, options.graph_config.as_ref());
    infer_create_time(&mut table_specs);

    if let Some(config) = &options.graph_config {
        let data_sources: HashSet<String> = config
            .tables
            .iter()
            .map(|t| t.data_source.clone())
            .collect();
        infer_foreign_keys_with_config(&mut table_specs, &mut inferred_fkeys, &mut warnings.notes, config, &data_sources);
    } else {
        infer_foreign_keys(&table_specs, &mut inferred_fkeys, &mut warnings.notes, None);
    }
    unify_fkey_column_types(&mut table_specs, &inferred_fkeys, &mut warnings.notes);
    drop_join_table_primary_keys(&mut table_specs, &inferred_fkeys);
    prune_fkeys_without_primary_key(&table_specs, &mut inferred_fkeys);
    if options.graph_config.is_none() {
        infer_primary_key_relationships(&table_specs, &mut inferred_fkeys);
    }
    infer_edge_tables(&mut table_specs, &inferred_fkeys);
    prune_fkeys_for_edge_tables(&table_specs, &mut inferred_fkeys);

    // Convert inferred FKeys to FKeySpec embedded in tables
    convert_inferred_fkeys_to_fkeyspecs(&mut table_specs, &inferred_fkeys);

    let total_bytes: u64 = tables.iter().map(|table| table.total_bytes).sum();
    let num_shards = estimate_num_shards(total_bytes);
    let metadata = GraphConfig {
        dataset_name: root_dataset_name(root),
        tables: table_specs,
        rowpack: None,
        num_shards: Some(num_shards),
        edge_format: None,
    };
    metadata.validate()?;

    Ok(InferenceResult { metadata, warnings })
}

fn root_dataset_name(root: &RootSource) -> String {
    match root {
        RootSource::Local(path) => path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_string(),
        RootSource::S3 { bucket, prefix }
        | RootSource::Gcs { bucket, prefix } => {
            if prefix.is_empty() {
                bucket.clone()
            } else {
                prefix
                    .trim_end_matches('/')
                    .split('/')
                    .last()
                    .unwrap_or(bucket)
                    .to_string()
            }
        }
        RootSource::Adls { prefix, .. } => {
            if prefix.is_empty() {
                "dataset".to_string()
            } else {
                prefix
                    .trim_end_matches('/')
                    .split('/')
                    .last()
                    .unwrap_or("dataset")
                    .to_string()
            }
        }
    }
}

fn estimate_num_shards(total_bytes: u64) -> usize {
    if total_bytes == 0 {
        return 1;
    }
    let shard_bytes = shard_target_bytes(total_bytes);
    ((total_bytes + shard_bytes - 1) / shard_bytes) as usize
}

fn shard_target_bytes(total_bytes: u64) -> u64 {
    const FALLBACK_SHARD_BYTES: u64 = 1 * 1024 * 1024 * 1024;
    let mem_total = read_system_memory_bytes();
    let cpu_count = std::thread::available_parallelism().ok().map(|n| n.get() as u64);
    match (mem_total, cpu_count) {
        (Some(mem_total), Some(cpu_count)) if cpu_count > 0 => {
            let per_cpu = mem_total / cpu_count;
            let target = smooth_shard_target(per_cpu, total_bytes);
            if target == 0 {
                FALLBACK_SHARD_BYTES
            } else {
                target
            }
        }
        _ => FALLBACK_SHARD_BYTES,
    }
}

fn smooth_shard_target(per_cpu_bytes: u64, total_bytes: u64) -> u64 {
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let divisor = if total_gb <= 40.0 {
        let t = ((total_gb - 10.0) / 30.0).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        16.0 - 12.0 * smooth
    } else if total_gb <= 1024.0 {
        let t = ((total_gb - 40.0) / (1024.0 - 40.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        4.0 - 3.0 * smooth
    } else if total_gb <= 10_240.0 {
        let t = ((total_gb - 1024.0) / (10_240.0 - 1024.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        1.0 - 0.5 * smooth
    } else {
        0.75
    };
    (per_cpu_bytes as f64 / divisor).round() as u64
}

fn read_system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb = rest.trim().split_whitespace().next()?.parse::<u64>().ok()?;
                return kb.checked_mul(1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn build_table_config_map(config: &Option<InputGraphConfig>) -> HashMap<String, HashSet<String>> {
    let mut map = HashMap::new();
    if let Some(cfg) = config {
        for table in &cfg.tables {
             let mut cols = HashSet::new();
             for c in &table.cols { cols.insert(c.name.clone()); }
             if let Some(pk) = &table.pkey { cols.insert(pk.name.clone()); }
             for fk in &table.fkeys { cols.insert(fk.name.clone()); }
             map.insert(table.table_name.clone(), cols);
        }
    }
    map
}

fn infer_local_tables(
    root: &Path,
    options: &InferOptions,
    warnings: &mut InferWarnings,
) -> Result<Vec<TableInput>> {
    let config_map = build_table_config_map(&options.graph_config);
    let mut tables = Vec::new();
    let entries = std::fs::read_dir(root).with_context(|| format!("read {}", root.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            let allowed_cols = config_map.get(&name);
            let table = infer_dir_table(&name, &path, root, options, warnings, allowed_cols)?;
            if let Some(table) = table {
                tables.push(table);
            }
        } else {
            let allowed_cols = split_table_file(&name).and_then(|(t, _)| config_map.get(t));
            if let Some(table) = infer_file_table(&name, &path, root, options, allowed_cols)? {
                tables.push(table);
            }
        }
    }
    if tables.is_empty() {
        return Err(anyhow!("no tables found under {}", root.display()));
    }
    Ok(tables)
}

fn infer_file_table(
    name: &str,
    path: &Path,
    root: &Path,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Option<TableInput>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "csv" => {
            eprintln!("infer_config: reading {}", path.display());
            let size = std::fs::metadata(path)?.len();
            let columns = infer_csv_schema(path, options, allowed_cols)
                .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?;
            let rel_path = path.strip_prefix(root).unwrap_or(path);
            Ok(Some(TableInput {
                name: name.trim_end_matches(".csv").to_string(),
                format: TableFormat::Csv,
                paths: vec![rel_path.to_string_lossy().to_string()],
                columns,
                total_bytes: size,
            }))
        }
        "parquet" => {
            eprintln!("infer_config: reading {}", path.display());
            let size = std::fs::metadata(path)?.len();
            let columns = infer_parquet_schema_local(path, options, allowed_cols)
                .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?;
            let rel_path = path.strip_prefix(root).unwrap_or(path);
            Ok(Some(TableInput {
                name: name.trim_end_matches(".parquet").to_string(),
                format: TableFormat::Parquet,
                paths: vec![rel_path.to_string_lossy().to_string()],
                columns,
                total_bytes: size,
            }))
        }
        _ => Ok(None),
    }
}

fn infer_dir_table(
    name: &str,
    dir: &Path,
    root: &Path,
    options: &InferOptions,
    warnings: &mut InferWarnings,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Option<TableInput>> {
    let mut table: Option<TableInput> = None;
    let entries = std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let format = match ext.as_str() {
            "csv" => TableFormat::Csv,
            "parquet" => TableFormat::Parquet,
            _ => continue,
        };
        let size = path.metadata()?.len();
        match &mut table {
            Some(existing) => {
                if !options.validate_dir_schemas {
                    let rel_path = path.strip_prefix(root).unwrap_or(&path);
                    existing.paths.push(rel_path.to_string_lossy().to_string());
                    existing.total_bytes = existing.total_bytes.saturating_add(size);
                    continue;
                }
                eprintln!("infer_config: reading {}", path.display());
                let columns = match format {
                    TableFormat::Csv => infer_csv_schema(&path, options, allowed_cols)
                        .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?,
                    TableFormat::Parquet => infer_parquet_schema_local(&path, options, allowed_cols)
                        .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?,
                };
                if !same_table_format(&existing.format, &format)
                    || !columns_match(&existing.columns, &columns)
                {
                    if options.validate_dir_schemas {
                        warnings.schema_mismatches.push(SchemaMismatch {
                            path: path.to_string_lossy().to_string(),
                            diffs: schema_diffs(&existing.columns, &columns),
                        });
                        warnings
                            .ignored_files
                            .push(path.to_string_lossy().to_string());
                        continue;
                    }
                }
                let rel_path = path.strip_prefix(root).unwrap_or(&path);
                existing.paths.push(rel_path.to_string_lossy().to_string());
                existing.total_bytes = existing.total_bytes.saturating_add(size);
            }
            None => {
                eprintln!("infer_config: reading {}", path.display());
                let columns = match format {
                    TableFormat::Csv => infer_csv_schema(&path, options, allowed_cols)
                        .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?,
                    TableFormat::Parquet => infer_parquet_schema_local(&path, options, allowed_cols)
                        .with_context(|| format!("infer schema for table {} at {}", name, path.display()))?,
                };
                let rel_path = path.strip_prefix(root).unwrap_or(&path);
                table = Some(TableInput {
                    name: name.to_string(),
                    format,
                    paths: vec![rel_path.to_string_lossy().to_string()],
                    columns,
                    total_bytes: size,
                });
            }
        }
    }
    Ok(table)
}

fn infer_remote_tables(
    root: &RootSource,
    bucket: &str,
    prefix: &str,
    options: &InferOptions,
    warnings: &mut InferWarnings,
) -> Result<Vec<TableInput>> {
    let config_map = build_table_config_map(&options.graph_config);
    let (rt, store) = build_store(root)?;
    let transfer = CloudTransfer::new(Arc::clone(&store), CloudTransferConfig::from_env());
    let mut tables: HashMap<String, TableInput> = HashMap::new();
    let mut prefixes = HashSet::new();
    let normalized = prefix.trim_matches('/');
    let list_prefix = if normalized.is_empty() {
        None
    } else {
        Some(ObjectPath::from(normalized))
    };
    let mut stream = store.list(list_prefix.as_ref());
    while let Some(obj) = rt.block_on(stream.next()) {
        let obj = obj?;
        let path_str = obj.location.to_string();
        let rel_path = if normalized.is_empty() {
            path_str.clone()
        } else {
            path_str
                .strip_prefix(&format!("{}/", normalized))
                .unwrap_or(&path_str)
                .to_string()
        };
        let parts: Vec<&str> = rel_path.split('/').collect();
        if parts.is_empty() {
            continue;
        }
        if parts.len() == 1 {
            if let Some((table_name, format)) = split_table_file(parts[0]) {
                let allowed_cols = config_map.get(table_name);
                let format_clone = format.clone();
                if let Some(entry) = tables.get_mut(table_name) {
                    if !options.validate_dir_schemas {
                        entry.paths.push(format!(
                            "{}://{}/{}",
                            root_scheme(root),
                            bucket,
                            path_str
                        ));
                        entry.total_bytes = entry.total_bytes.saturating_add(obj.size);
                        continue;
                    }
                }
                eprintln!("infer_config: reading {}", path_str);
                let columns = match format {
                    TableFormat::Parquet => infer_parquet_schema_remote(
                        &transfer,
                        &obj.location,
                        obj.size,
                        options,
                        allowed_cols,
                    )
                    .with_context(|| format!("infer schema for table {} at {}", table_name, path_str))?,
                    TableFormat::Csv => infer_csv_schema_remote(
                        &transfer,
                        &obj.location,
                        obj.size,
                        options,
                        allowed_cols,
                    )
                    .with_context(|| format!("infer schema for table {} at {}", table_name, path_str))?,
                };
                let entry = tables.entry(table_name.to_string()).or_insert_with(|| TableInput {
                    name: table_name.to_string(),
                    format: format_clone,
                    paths: Vec::new(),
                    columns: columns.clone(),
                    total_bytes: 0,
                });
                if !same_table_format(&entry.format, &format)
                    || !columns_match(&entry.columns, &columns)
                {
                    warnings.schema_mismatches.push(SchemaMismatch {
                        path: path_str.clone(),
                        diffs: schema_diffs(&entry.columns, &columns),
                    });
                    warnings.ignored_files.push(path_str);
                } else {
                    entry.paths.push(format!(
                        "{}://{}/{}",
                        root_scheme(root),
                        bucket,
                        path_str
                    ));
                    entry.total_bytes = entry.total_bytes.saturating_add(obj.size);
                }
            }
        } else {
            prefixes.insert(parts[0].to_string());
            let file = parts.last().unwrap_or(&"");
            if let Some((_, format)) = split_table_file(file) {
                let table_name = parts[0];
                let allowed_cols = config_map.get(table_name);
                let format_clone = format.clone();
                if let Some(entry) = tables.get_mut(table_name) {
                    if !options.validate_dir_schemas {
                        entry.paths.push(format!(
                            "{}://{}/{}",
                            root_scheme(root),
                            bucket,
                            path_str
                        ));
                        entry.total_bytes = entry.total_bytes.saturating_add(obj.size);
                        continue;
                    }
                }
                eprintln!("infer_config: reading {}", path_str);
                let columns = match format {
                    TableFormat::Parquet => infer_parquet_schema_remote(
                        &transfer,
                        &obj.location,
                        obj.size,
                        options,
                        allowed_cols,
                    )
                    .with_context(|| format!("infer schema for table {} at {}", table_name, path_str))?,
                    TableFormat::Csv => infer_csv_schema_remote(
                        &transfer,
                        &obj.location,
                        obj.size,
                        options,
                        allowed_cols,
                    )
                    .with_context(|| format!("infer schema for table {} at {}", table_name, path_str))?,
                };
                let entry = tables.entry(table_name.to_string()).or_insert_with(|| TableInput {
                    name: table_name.to_string(),
                    format: format_clone,
                    paths: Vec::new(),
                    columns: columns.clone(),
                    total_bytes: 0,
                });
                if !same_table_format(&entry.format, &format)
                    || !columns_match(&entry.columns, &columns)
                {
                    warnings.schema_mismatches.push(SchemaMismatch {
                        path: path_str.clone(),
                        diffs: schema_diffs(&entry.columns, &columns),
                    });
                    warnings.ignored_files.push(path_str);
                } else {
                    entry.paths.push(format!(
                        "{}://{}/{}",
                        root_scheme(root),
                        bucket,
                        path_str
                    ));
                    entry.total_bytes = entry.total_bytes.saturating_add(obj.size);
                }
            }
        }
    }
    if tables.is_empty() {
        return Err(anyhow!("no tables found under {}", prefix));
    }
    Ok(tables.into_values().collect())
}

fn root_scheme(root: &RootSource) -> &'static str {
    match root {
        RootSource::Local(_) => "file",
        RootSource::S3 { .. } => "s3",
        RootSource::Gcs { .. } => "gs",
        RootSource::Adls { .. } => "az",
    }
}

fn split_table_file(file: &str) -> Option<(&str, TableFormat)> {
    let lower = file.to_lowercase();
    if lower.ends_with(".parquet") {
        Some((file.trim_end_matches(".parquet"), TableFormat::Parquet))
    } else if lower.ends_with(".csv") {
        Some((file.trim_end_matches(".csv"), TableFormat::Csv))
    } else {
        None
    }
}

fn columns_match(a: &[ColumnSpec], b: &[ColumnSpec]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (lhs, rhs) in a.iter().zip(b.iter()) {
        if !columns_equivalent(lhs, rhs) {
            return false;
        }
    }
    true
}

fn columns_equivalent(lhs: &ColumnSpec, rhs: &ColumnSpec) -> bool {
    lhs.name == rhs.name
        && column_type_eq(&lhs.ingested_dtype, &rhs.ingested_dtype)
        && lhs.fixed_len == rhs.fixed_len
        && lhs.precision == rhs.precision
        && lhs.scale == rhs.scale
}

fn schema_diffs(a: &[ColumnSpec], b: &[ColumnSpec]) -> Vec<SchemaDiff> {
    let mut diffs = Vec::new();
    let min_len = a.len().min(b.len());
    for idx in 0..min_len {
        if !columns_equivalent(&a[idx], &b[idx]) {
            diffs.push(SchemaDiff {
                index: idx,
                left: Some(a[idx].clone()),
                right: Some(b[idx].clone()),
            });
        }
    }
    if a.len() > min_len {
        for idx in min_len..a.len() {
            diffs.push(SchemaDiff {
                index: idx,
                left: Some(a[idx].clone()),
                right: None,
            });
        }
    } else if b.len() > min_len {
        for idx in min_len..b.len() {
            diffs.push(SchemaDiff {
                index: idx,
                left: None,
                right: Some(b[idx].clone()),
            });
        }
    }
    diffs
}

fn same_table_format(a: &TableFormat, b: &TableFormat) -> bool {
    matches!(
        (a, b),
        (TableFormat::Csv, TableFormat::Csv) | (TableFormat::Parquet, TableFormat::Parquet)
    )
}

fn column_type_eq(a: &IngestedType, b: &IngestedType) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn build_store(root: &RootSource) -> Result<(Runtime, Arc<dyn ObjectStore>)> {
    let rt = Runtime::new()?;
    let store: Arc<dyn ObjectStore> = match root {
        RootSource::S3 { bucket, .. } => {
            let client = build_s3_client(S3Config {
                bucket: bucket.clone(),
                region: None,
                endpoint: None,
                allow_http: None,
                virtual_hosted: None,
                profile: None,
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
            })?;
            client.store
        }
        RootSource::Gcs { bucket, .. } => {
            let client = build_gcs_client(GcsConfig {
                bucket: bucket.clone(),
            })?;
            client.store
        }
        RootSource::Adls { account, container, .. } => {
            let client = build_adls_client(AdlsConfig {
                account: account.clone(),
                container: container.clone(),
                endpoint: None,
                allow_http: None,
                access_key: None,
                sas_key: None,
                bearer_token: None,
                client_id: None,
                client_secret: None,
                tenant_id: None,
            })?;
            client.store
        }
        RootSource::Local(_) => return Err(anyhow!("local root does not need object store")),
    };
    Ok((rt, store))
}

fn infer_parquet_schema_local(
    path: &Path,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let mut columns = infer_parquet_schema_from_arrow(schema.as_ref(), allowed_cols)?;
    if options.sample_rows > 0 {
    let mut reader = builder.with_batch_size(1024).build()?;
    apply_parquet_string_samples(&mut columns, &mut reader, options.sample_rows)?;
    }
    Ok(columns)
}

fn infer_parquet_schema_remote(
    transfer: &CloudTransfer,
    path: &ObjectPath,
    _size: u64,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let pooled = transfer.submit_download_bytes(path.clone())?.wait()?;
    let bytes = pooled.bytes();
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let schema = builder.schema();
    let mut columns = infer_parquet_schema_from_arrow(schema.as_ref(), allowed_cols)?;
    if columns.is_empty() {
        return Ok(columns);
    }
    if options.sample_rows > 0 {
        let mut reader = builder.with_batch_size(1024).build()?;
        apply_parquet_string_samples(&mut columns, &mut reader, options.sample_rows)?;
    }
    drop(pooled);
    Ok(columns)
}

pub(crate) fn infer_parquet_schema_from_arrow(
    schema: &Schema,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let mut columns = Vec::new();
    for field in schema.fields() {
        if let Some(allowed) = allowed_cols {
            if !allowed.contains(field.name()) {
                continue;
            }
        }
        let (fixed_len, precision, scale) = arrow_type_metadata(field.data_type())?;
        let mut ingested_dtype = match field.data_type() {
            DataType::Float64 => IngestedType::Float32,
            DataType::List(field) | DataType::LargeList(field) => match field.data_type() {
                DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                    IngestedType::Float16Array
                }
                _ => return Err(anyhow!("unsupported list type {}", field.data_type())),
            },
            DataType::Int8 => IngestedType::Int8,
            DataType::Int16 => IngestedType::Int16,
            DataType::Int32 => IngestedType::Int32,
            DataType::Int64 => IngestedType::Int64,
            DataType::UInt8 => IngestedType::UInt8,
            DataType::UInt16 => IngestedType::UInt16,
            DataType::UInt32 => IngestedType::UInt32,
            DataType::UInt64 => IngestedType::UInt64,
            DataType::Float32 => IngestedType::Float32,
            DataType::Boolean => IngestedType::Bool,
            DataType::Utf8 | DataType::LargeUtf8 => IngestedType::String,
            DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_) => IngestedType::Bytes,
            DataType::Date32 => IngestedType::Date,
            DataType::Timestamp(_, _) => IngestedType::Timestamp,
            DataType::Decimal128(_, _) => IngestedType::Decimal128,
            DataType::Decimal256(_, _) => IngestedType::Decimal256,
            _ => {
                return Err(anyhow!(
                    "unsupported parquet type {} for column {}",
                    field.data_type(),
                    field.name()
                ))
            }
        };
        if matches!(
            field.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_)
        ) {
            if let Some(len) = fixed_len {
                let name_lower = field.name().to_lowercase();
                if len == 16 && name_lower.contains("ip") {
                    ingested_dtype = IngestedType::Ip;
                } else if len == 16 {
                    ingested_dtype = IngestedType::Uuid;
                } else if len == 32 {
                    ingested_dtype = IngestedType::Uuid256;
                } else if len == 4 && name_lower.contains("ip") {
                    ingested_dtype = IngestedType::Ip;
                }
            }
        }
        columns.push(ColumnSpec {
            name: field.name().to_string(),
            ingested_dtype,
            source_dtype: Some(field.data_type().to_string()),
            fixed_len,
            timestamp_format: None,
            timestamp_unit: None,
            precision,
            scale,
        });
    }
    Ok(columns)
}

fn apply_parquet_string_samples<I>(
    columns: &mut [ColumnSpec],
    batches: &mut I,
    sample_rows: usize,
) -> Result<()>
where
    I: Iterator<Item = std::result::Result<RecordBatch, ArrowError>>,
{
    let mut stats: Vec<Option<ColumnStats>> =
        (0..columns.len()).map(|_| Some(ColumnStats::default())).collect();
    let mut seen = 0usize;
    while let Some(batch) = batches.next() {
        let batch = batch.map_err(|err| anyhow!(err))?;
        if seen >= sample_rows {
            break;
        }
        let rows = batch.num_rows().min(sample_rows - seen);
        for idx in 0..columns.len() {
            if let Some(st) = stats[idx].as_mut() {
                let array = batch.column(idx);
                match array.data_type() {
                    DataType::Utf8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| anyhow!("string cast"))?;
                        sample_string_array(arr, rows, st);
                    }
                    DataType::LargeUtf8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .ok_or_else(|| anyhow!("large string cast"))?;
                        sample_large_string_array(arr, rows, st);
                    }
                    DataType::Int64 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| anyhow!("int64 cast"))?;
                        sample_int64_array(arr, rows, st);
                    }
                    DataType::Int32 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .ok_or_else(|| anyhow!("int32 cast"))?;
                        sample_int32_array(arr, rows, st);
                    }
                    DataType::Int16 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<Int16Array>()
                            .ok_or_else(|| anyhow!("int16 cast"))?;
                        sample_int16_array(arr, rows, st);
                    }
                    DataType::Int8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<Int8Array>()
                            .ok_or_else(|| anyhow!("int8 cast"))?;
                        sample_int8_array(arr, rows, st);
                    }
                    DataType::UInt64 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .ok_or_else(|| anyhow!("uint64 cast"))?;
                        sample_uint64_array(arr, rows, st);
                    }
                    DataType::UInt32 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<UInt32Array>()
                            .ok_or_else(|| anyhow!("uint32 cast"))?;
                        sample_uint32_array(arr, rows, st);
                    }
                    DataType::UInt16 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<UInt16Array>()
                            .ok_or_else(|| anyhow!("uint16 cast"))?;
                        sample_uint16_array(arr, rows, st);
                    }
                    DataType::UInt8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<UInt8Array>()
                            .ok_or_else(|| anyhow!("uint8 cast"))?;
                        sample_uint8_array(arr, rows, st);
                    }
                    _ => continue,
                }
            }
        }
        seen += rows;
    }
    apply_string_stats_overrides(columns, &stats);
    Ok(())
}


fn sample_string_array(array: &StringArray, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe(array.value(i));
    }
}

fn sample_large_string_array(array: &LargeStringArray, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe(array.value(i));
    }
}

fn sample_int64_array(array: &Int64Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i));
    }
}

fn sample_int32_array(array: &Int32Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn sample_int16_array(array: &Int16Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn sample_int8_array(array: &Int8Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn sample_uint64_array(array: &UInt64Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        let value = array.value(i);
        if value <= i64::MAX as u64 {
            stats.observe_int_value(value as i64);
        }
    }
}

fn sample_uint32_array(array: &UInt32Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn sample_uint16_array(array: &UInt16Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn sample_uint8_array(array: &UInt8Array, rows: usize, stats: &mut ColumnStats) {
    for i in 0..rows {
        if array.is_null(i) {
            continue;
        }
        stats.observe_int_value(array.value(i) as i64);
    }
}

fn apply_string_stats_overrides(columns: &mut [ColumnSpec], stats: &[Option<ColumnStats>]) {
    for (idx, col) in columns.iter_mut().enumerate() {
        let Some(st) = stats[idx].as_ref() else {
            continue;
        };
        if st.seen == 0 {
            continue;
        }
        let inferred = st.to_column_spec(&col.name);
        let inferred_dtype = inferred.ingested_dtype.clone();
        if matches!(inferred_dtype, IngestedType::Timestamp | IngestedType::Date) {
            col.ingested_dtype = inferred_dtype.clone();
            col.timestamp_format = inferred.timestamp_format;
            col.timestamp_unit = inferred.timestamp_unit;
        }
        if matches!(col.ingested_dtype, IngestedType::String | IngestedType::Bytes) {
            if matches!(
                inferred_dtype,
                IngestedType::Uuid | IngestedType::Uuid256 | IngestedType::Ip
            ) {
                col.ingested_dtype = inferred_dtype;
                col.fixed_len = None;
            } else if inferred.fixed_len.is_some() {
                col.fixed_len = inferred.fixed_len;
            }
        }
    }
}

fn arrow_type_metadata(
    source_dtype: &DataType,
) -> Result<(Option<usize>, Option<u32>, Option<i32>)> {
    let mut precision = None;
    let mut scale = None;
    let mut fixed_len = None;
    match source_dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Boolean
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::Date32
        | DataType::Timestamp(_, _) => {}
        DataType::FixedSizeBinary(len) => {
            fixed_len = Some(*len as usize);
        }
        DataType::Decimal128(p, s) => {
            precision = Some(*p as u32);
            scale = Some(*s as i32);
        }
        DataType::Decimal256(p, s) => {
            precision = Some(*p as u32);
            scale = Some(*s as i32);
        }
        DataType::List(field) | DataType::LargeList(field) => match field.data_type() {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {}
            _ => {
                return Err(anyhow!(
                    "unsupported list element type {} for field {}",
                    field.data_type(),
                    field.name()
                ))
            }
        },
        _ => return Err(anyhow!("unsupported arrow type {}", source_dtype)),
    };
    Ok((fixed_len, precision, scale))
}

fn infer_csv_schema(
    path: &Path,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;
    let headers = reader
        .headers()
        .context("csv headers")?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    let mut keep_indices = Vec::new();
    if let Some(allowed) = allowed_cols {
        for (i, h) in headers.iter().enumerate() {
            if allowed.contains(h) {
                keep_indices.push(i);
            }
        }
    } else {
        keep_indices.extend(0..headers.len());
    }

    let mut stats: Vec<ColumnStats> = keep_indices.iter().map(|_| ColumnStats::default()).collect();
    for result in reader.records().take(options.csv_sample_rows) {
        let record = result?;
        for (stat_idx, &col_idx) in keep_indices.iter().enumerate() {
            if let Some(value) = record.get(col_idx) {
                stats[stat_idx].observe(value);
            }
        }
    }
    Ok(keep_indices
        .iter()
        .zip(stats.iter())
        .map(|(&i, st)| st.to_column_spec(&headers[i]))
        .collect())
}

fn infer_csv_schema_remote(
    transfer: &CloudTransfer,
    path: &ObjectPath,
    size: u64,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let max = size.min(options.csv_sample_bytes);
    let buffer = transfer.submit_download(path.clone())?.wait()?;
    let bytes = Bytes::from(buffer.into_inner());
    let limited = bytes.slice(0..max as usize);
    infer_csv_schema_from_bytes(&limited, options, allowed_cols)
}

fn infer_csv_schema_from_bytes(
    bytes: &Bytes,
    options: &InferOptions,
    allowed_cols: Option<&HashSet<String>>,
) -> Result<Vec<ColumnSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(bytes.as_ref());
    let headers = reader
        .headers()
        .context("csv headers")?
        .iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>();

    let mut keep_indices = Vec::new();
    if let Some(allowed) = allowed_cols {
        for (i, h) in headers.iter().enumerate() {
            if allowed.contains(h) {
                keep_indices.push(i);
            }
        }
    } else {
        keep_indices.extend(0..headers.len());
    }

    let mut stats: Vec<ColumnStats> = keep_indices.iter().map(|_| ColumnStats::default()).collect();
    for result in reader.records().take(options.csv_sample_rows) {
        let record = result?;
        for (stat_idx, &col_idx) in keep_indices.iter().enumerate() {
            if let Some(value) = record.get(col_idx) {
                stats[stat_idx].observe(value);
            }
        }
    }
    Ok(keep_indices
        .iter()
        .zip(stats.iter())
        .map(|(&i, st)| st.to_column_spec(&headers[i]))
        .collect())
}

#[derive(Clone, Default)]
pub(crate) struct ColumnStats {
    seen: usize,
    int_ok: bool,
    float_ok: bool,
    date_ok: bool,
    ts_ok: bool,
    bool_ok: bool,
    int_min: Option<i64>,
    int_max: Option<i64>,
    ts_format: Option<String>,
    date_format: Option<String>,
    str_min_len: Option<usize>,
    str_max_len: Option<usize>,
    uuid_ok: bool,
    uuid_seen: bool,
    uuid256_ok: bool,
    uuid256_seen: bool,
    ip_ok: bool,
    ip_seen: bool,
}

impl ColumnStats {
    fn observe_int_value(&mut self, value: i64) {
        self.seen += 1;
        if self.seen == 1 {
            self.int_ok = true;
            self.float_ok = true;
            self.date_ok = false;
            self.ts_ok = false;
            self.bool_ok = false;
        }
        if self.int_ok {
            self.int_min = Some(self.int_min.map(|m| m.min(value)).unwrap_or(value));
            self.int_max = Some(self.int_max.map(|m| m.max(value)).unwrap_or(value));
        }
    }

    pub(crate) fn observe(&mut self, value: &str) {
        if value.is_empty() {
            return;
        }
        self.seen += 1;
        if self.seen == 1 {
            self.int_ok = true;
            self.float_ok = true;
            self.date_ok = true;
            self.ts_ok = true;
            self.bool_ok = true;
            self.uuid_ok = true;
            self.uuid_seen = false;
            self.uuid256_ok = true;
            self.uuid256_seen = false;
            self.ip_ok = true;
            self.ip_seen = false;
        }
        let len = value.len();
        self.str_min_len = Some(self.str_min_len.map(|m| m.min(len)).unwrap_or(len));
        self.str_max_len = Some(self.str_max_len.map(|m| m.max(len)).unwrap_or(len));
        self.uuid_seen = true;
        if self.uuid_ok && !looks_like_uuid(value) {
            self.uuid_ok = false;
        }
        self.uuid256_seen = true;
        if self.uuid256_ok && !looks_like_uuid256(value) {
            self.uuid256_ok = false;
        }
        self.ip_seen = true;
        if self.ip_ok && !looks_like_ip(value) {
            self.ip_ok = false;
        }
        if self.int_ok {
            if let Some(v) = parse_int(value) {
                self.int_min = Some(self.int_min.map(|m| m.min(v)).unwrap_or(v));
                self.int_max = Some(self.int_max.map(|m| m.max(v)).unwrap_or(v));
            } else {
                self.int_ok = false;
            }
        }
        if self.float_ok && parse_float(value).is_none() {
            self.float_ok = false;
        }
        if self.date_ok && parse_date(value).is_none() {
            self.date_ok = false;
            self.date_format = None;
        } else if self.date_ok {
            match &self.date_format {
                None => self.date_format = Some("%Y-%m-%d".to_string()),
                Some(existing) if existing == "%Y-%m-%d" => {}
                Some(_) => self.date_format = None,
            }
        }
        if self.ts_ok && parse_timestamp(value).is_none() {
            self.ts_ok = false;
            self.ts_format = None;
        } else if self.ts_ok {
            if let Some(fmt) = parse_timestamp_format(value) {
                match &self.ts_format {
                    None => self.ts_format = Some(fmt.to_string()),
                    Some(existing) if existing == fmt => {}
                    Some(_) => self.ts_format = None,
                }
            } else {
                self.ts_ok = false;
                self.ts_format = None;
            }
        }
        if self.bool_ok && parse_bool(value).is_none() {
            self.bool_ok = false;
        }
    }

    pub(crate) fn to_column_spec(&self, name: &str) -> ColumnSpec {
        let name_lower = name.to_lowercase();
        let mut ingested_dtype = IngestedType::String;
        let mut timestamp_format = None;
        let mut timestamp_unit = None;
        let mut fixed_len = None;
        if self.bool_ok {
            ingested_dtype = IngestedType::Bool;
        } else if self.ts_ok {
            if self.date_ok
                && matches!(self.ts_format.as_deref(), Some("%Y-%m-%d"))
            {
                ingested_dtype = IngestedType::Date;
                timestamp_format = self.date_format.clone();
            } else {
                ingested_dtype = IngestedType::Timestamp;
                // Preserve the parse format for string-backed timestamps.
                timestamp_format = self.ts_format.clone();
            }
        } else if time_name_score(&name_lower) > 0
            && self.int_ok
            && int_looks_like_timestamp(self.int_min, self.int_max).is_some()
        {
            ingested_dtype = IngestedType::Timestamp;
            // Record the epoch unit for integer timestamps.
            timestamp_unit = int_looks_like_timestamp(self.int_min, self.int_max);
        } else if self.date_ok {
            ingested_dtype = IngestedType::Date;
            // Preserve the parse format for string-backed dates.
            timestamp_format = self.date_format.clone();
        } else if self.int_ok {
            ingested_dtype = IngestedType::Int64;
        } else if self.float_ok {
            ingested_dtype = IngestedType::Float32;
        }
        if matches!(ingested_dtype, IngestedType::String) {
            if self.uuid_seen && self.uuid_ok {
                ingested_dtype = IngestedType::Uuid;
            } else if self.uuid256_seen && self.uuid256_ok {
                ingested_dtype = IngestedType::Uuid256;
            } else if self.ip_seen && self.ip_ok {
                ingested_dtype = IngestedType::Ip;
            } else if let (Some(min_len), Some(max_len)) = (self.str_min_len, self.str_max_len) {
                if min_len == max_len {
                    fixed_len = Some(min_len);
                }
            }
        }
        ColumnSpec {
            name: name.to_string(),
            ingested_dtype,
            source_dtype: Some("string".to_string()),
            fixed_len,
            timestamp_format,
            timestamp_unit,
            precision: None,
            scale: None,
        }
    }
}

fn looks_like_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    match bytes.len() {
        32 => bytes.iter().all(|b| is_hex(*b)),
        36 => bytes
            .iter()
            .enumerate()
            .all(|(idx, b)| match idx {
                8 | 13 | 18 | 23 => *b == b'-',
                _ => is_hex(*b),
            }),
        _ => false,
    }
}

fn looks_like_uuid256(value: &str) -> bool {
    let s = value.trim();
    if s.len() != 64 {
        return false;
    }
    s.as_bytes().iter().all(|b| is_hex(*b))
}

fn looks_like_ip(value: &str) -> bool {
    value.trim().parse::<std::net::IpAddr>().is_ok()
}

fn is_hex(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

fn parse_int(value: &str) -> Option<i64> {
    value.parse::<i64>().ok()
}

fn parse_float(value: &str) -> Option<f64> {
    value.parse::<f64>().ok()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "false" | "1" | "0" | "yes" | "no" => Some(true),
        _ => None,
    }
}

fn parse_date(value: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").ok()
}

fn parse_timestamp(value: &str) -> Option<NaiveDateTime> {
    let bytes = value.as_bytes();
    parse_ymd(bytes)
        .or_else(|| parse_ymd_hm(bytes))
        .or_else(|| parse_ymd_hms(bytes, false))
        .or_else(|| parse_ymd_hms(bytes, true))
        .or_else(|| parse_ymd_hms_t(bytes, false))
        .or_else(|| parse_ymd_hms_t(bytes, true))
        .or_else(|| parse_ymd_hms_offset(bytes, false, false))
        .or_else(|| parse_ymd_hms_offset(bytes, true, false))
        .or_else(|| parse_ymd_hms_offset(bytes, false, true))
        .or_else(|| parse_ymd_hms_offset(bytes, true, true))
        .or_else(|| parse_ymd_hms_t_offset(bytes, false))
        .or_else(|| parse_ymd_hms_t_offset(bytes, true))
}

fn parse_timestamp_format(value: &str) -> Option<&'static str> {
    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S%:z",
        "%Y-%m-%d %H:%M:%S %:z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%:z",
    ] {
        if parse_timestamp_with_format(value, fmt).is_some() {
            return Some(fmt);
        }
    }
    None
}

fn parse_timestamp_with_format(value: &str, fmt: &str) -> Option<NaiveDateTime> {
    let bytes = value.as_bytes();
    match fmt {
        "%Y-%m-%d" => parse_ymd(bytes),
        "%Y-%m-%d %H:%M" => parse_ymd_hm(bytes),
        "%Y-%m-%d %H:%M:%S" => parse_ymd_hms(bytes, false),
        "%Y-%m-%d %H:%M:%S%.f" => parse_ymd_hms(bytes, true),
        "%Y-%m-%dT%H:%M:%S" => parse_ymd_hms_t(bytes, false),
        "%Y-%m-%dT%H:%M:%S%.f" => parse_ymd_hms_t(bytes, true),
        "%Y-%m-%d %H:%M:%S%z" => parse_ymd_hms_offset(bytes, false, false),
        "%Y-%m-%d %H:%M:%S %z" => parse_ymd_hms_offset(bytes, false, true),
        "%Y-%m-%d %H:%M:%S%:z" => parse_ymd_hms_offset(bytes, true, false),
        "%Y-%m-%d %H:%M:%S %:z" => parse_ymd_hms_offset(bytes, true, true),
        "%Y-%m-%dT%H:%M:%S%z" => parse_ymd_hms_t_offset(bytes, false),
        "%Y-%m-%dT%H:%M:%S%:z" => parse_ymd_hms_t_offset(bytes, true),
        _ => NaiveDateTime::parse_from_str(value, fmt).ok(),
    }
}

fn parse_ymd(bytes: &[u8]) -> Option<NaiveDateTime> {
    if bytes.len() != 10 {
        return None;
    }
    let year = parse_fixed_u32(bytes, 0, 4)? as i32;
    let month = parse_fixed_u32(bytes, 5, 2)?;
    let day = parse_fixed_u32(bytes, 8, 2)?;
    timestamp_from_parts(year, month, day, 0, 0, 0, 0)
}

fn parse_ymd_hm(bytes: &[u8]) -> Option<NaiveDateTime> {
    if bytes.len() != 16 {
        return None;
    }
    let year = parse_fixed_u32(bytes, 0, 4)? as i32;
    let month = parse_fixed_u32(bytes, 5, 2)?;
    let day = parse_fixed_u32(bytes, 8, 2)?;
    let hour = parse_fixed_u32(bytes, 11, 2)?;
    let minute = parse_fixed_u32(bytes, 14, 2)?;
    timestamp_from_parts(year, month, day, hour, minute, 0, 0)
}

fn parse_ymd_hms(bytes: &[u8], allow_fraction: bool) -> Option<NaiveDateTime> {
    if bytes.len() < 19 {
        return None;
    }
    let year = parse_fixed_u32(bytes, 0, 4)? as i32;
    let month = parse_fixed_u32(bytes, 5, 2)?;
    let day = parse_fixed_u32(bytes, 8, 2)?;
    let hour = parse_fixed_u32(bytes, 11, 2)?;
    let minute = parse_fixed_u32(bytes, 14, 2)?;
    let second = parse_fixed_u32(bytes, 17, 2)?;
    let mut nanos = 0u32;
    if allow_fraction {
        if bytes.len() < 21 {
            return None;
        }
        let (fraction, digits, next) = parse_fraction(bytes, 20)?;
        nanos = fraction_to_nanos(fraction, digits);
        if next != bytes.len() {
            return None;
        }
    } else if bytes.len() != 19 {
        return None;
    }
    timestamp_from_parts(year, month, day, hour, minute, second, nanos)
}

fn parse_ymd_hms_t(bytes: &[u8], allow_fraction: bool) -> Option<NaiveDateTime> {
    parse_ymd_hms(bytes, allow_fraction)
}

fn parse_ymd_hms_offset(bytes: &[u8], colon: bool, spaced: bool) -> Option<NaiveDateTime> {
    let base_len = 19 + if spaced { 1 } else { 0 };
    let tz_len = if colon { 6 } else { 5 };
    let total_len = base_len + tz_len;
    if bytes.len() != total_len {
        return None;
    }
    let year = parse_fixed_u32(bytes, 0, 4)? as i32;
    let month = parse_fixed_u32(bytes, 5, 2)?;
    let day = parse_fixed_u32(bytes, 8, 2)?;
    let hour = parse_fixed_u32(bytes, 11, 2)?;
    let minute = parse_fixed_u32(bytes, 14, 2)?;
    let second = parse_fixed_u32(bytes, 17, 2)?;
    let mut pos = 19 + if spaced { 1 } else { 0 };
    let sign = match *bytes.get(pos)? {
        b'+' => 1i64,
        b'-' => -1i64,
        _ => return None,
    };
    pos += 1;
    let off_hour = parse_fixed_u32(bytes, pos, 2)? as i64;
    pos += 2;
    if colon {
        pos += 1;
    }
    let off_min = parse_fixed_u32(bytes, pos, 2)? as i64;
    let (secs, nanos) = timestamp_from_parts_secs(year, month, day, hour, minute, second, 0)?;
    let offset_secs = sign * (off_hour * 3_600 + off_min * 60);
    chrono::DateTime::from_timestamp(secs.saturating_sub(offset_secs), nanos).map(|dt| dt.naive_utc())
}

fn parse_ymd_hms_t_offset(bytes: &[u8], colon: bool) -> Option<NaiveDateTime> {
    parse_ymd_hms_offset(bytes, colon, false)
}

fn timestamp_from_parts(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    nanos: u32,
) -> Option<NaiveDateTime> {
    let (secs, nanos) = timestamp_from_parts_secs(year, month, day, hour, minute, second, nanos)?;
    chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.naive_utc())
}

fn timestamp_from_parts_secs(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    nanos: u32,
) -> Option<(i64, u32)> {
    if hour >= 24 || minute >= 60 || second >= 60 || nanos >= 1_000_000_000 {
        return None;
    }
    let days = days_from_civil(year, month, day)?;
    let secs = days
        .saturating_mul(86_400)
        .saturating_add(hour as i64 * 3_600)
        .saturating_add(minute as i64 * 60)
        .saturating_add(second as i64);
    Some((secs, nanos))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    let dim = days_in_month(year, month)?;
    if day == 0 || day > dim {
        return None;
    }
    let mut y = year as i64;
    let m = month as i64;
    let d = day as i64;
    y -= if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe - 719468)
}

fn days_in_month(year: i32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn parse_fixed_u32(bytes: &[u8], pos: usize, len: usize) -> Option<u32> {
    if pos + len > bytes.len() {
        return None;
    }
    let mut val: u32 = 0;
    for idx in pos..pos + len {
        let b = bytes[idx];
        if !b.is_ascii_digit() {
            return None;
        }
        val = val.saturating_mul(10).saturating_add((b - b'0') as u32);
    }
    Some(val)
}

fn parse_fraction(bytes: &[u8], pos: usize) -> Option<(u32, usize, usize)> {
    let mut idx = pos;
    let mut val: u32 = 0;
    let mut digits = 0usize;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        if digits < 9 {
            val = val.saturating_mul(10).saturating_add((bytes[idx] - b'0') as u32);
        }
        digits += 1;
        idx += 1;
    }
    if digits == 0 {
        return None;
    }
    Some((val, digits.min(9), idx))
}

fn fraction_to_nanos(value: u32, digits: usize) -> u32 {
    const POW10: [u32; 10] = [
        1_000_000_000,
        100_000_000,
        10_000_000,
        1_000_000,
        100_000,
        10_000,
        1_000,
        100,
        10,
        1,
    ];
    if digits == 0 {
        0
    } else if digits >= 9 {
        value
    } else {
        value.saturating_mul(POW10[digits])
    }
}

pub(crate) fn infer_primary_keys(tables: &mut [TableSpec], config: Option<&InputGraphConfig>) {
    let fallback_freq = tables.len().max(1) as i32;
    let mut freq = HashMap::new();

    // Build map of tables where PK is explicitly disabled (null in config)
    let disabled_pks: HashSet<String> = if let Some(cfg) = config {
        cfg.tables.iter()
            .filter(|t| t.pkey.is_none())
            .map(|t| t.table_name.clone())
            .collect()
    } else {
        HashSet::new()
    };
    for table in tables.iter() {
        for col in &table.columns {
            if !is_pk_dtype(&col.ingested_dtype) {
                continue;
            }
            let lower = col.name.to_lowercase();
            if is_key_like_name(&lower) {
                *freq.entry(lower).or_insert(0usize) += 1;
            }
        }
    }

    for table in tables.iter_mut() {
        if table.primary_key.is_some() {
            continue;
        }
        if disabled_pks.contains(&table.name) {
            continue;
        }
        if should_prefer_generic_id(table) {
            if let Some(id_col) = generic_id_column(table) {
                table.primary_key = Some(id_col);
                continue;
            }
        }
        let mut best: Option<(String, i32)> = None;
        let mut tie = false;
        let table_tokens = normalized_tokens(&table.name);
        let key_like_cols: Vec<String> = table
            .columns
            .iter()
            .filter(|col| is_pk_dtype(&col.ingested_dtype))
            .map(|col| col.name.to_lowercase())
            .filter(|name| is_key_like_name(name))
            .collect();
        let has_many_key_like = key_like_cols.len() > 2;
        let require_strong_for_small = key_like_cols.len() >= 2 && table.columns.len() <= 3;
        for col in &table.columns {
            if !is_pk_dtype(&col.ingested_dtype) {
                continue;
            }
            let lower = col.name.to_lowercase();
            let matches_table = column_matches_table_name(&table.name, &lower);
            let name_match = matches_table_name_field(&table.name, &lower);
            if !is_key_like_name(&lower) && !matches_table && !name_match {
                continue;
            }
            if has_composite_code_version(&lower, &table.columns) {
                continue;
            }
            let count = if matches_table || name_match {
                1
            } else {
                freq.get(&lower)
                    .map(|v| (*v).min(i32::MAX as usize) as i32)
                    .unwrap_or(fallback_freq)
            };
            if count <= 0 {
                continue;
            }
            let mut score = 1000 / count;
            let strong_match = is_strong_pk_match(&table.name, &lower);
            if matches_table {
                score += 900;
            }
            if strong_match {
                score += 12;
            } else if name_match {
                score += 4;
            } else if has_many_key_like {
                continue;
            } else if require_strong_for_small {
                continue;
            }
            if strong_match {
                score += 1000;
            } else if name_match {
                score += 120;
            }
            if column_matches_tokens(&lower, &table_tokens) {
                score += 3;
            }
            match &best {
                Some((_, best_score)) if score == *best_score => {
                    tie = true;
                }
                Some((_, best_score)) if score < *best_score => {}
                _ => {
                    best = Some((col.name.clone(), score));
                    tie = false;
                }
            }
        }
        if let Some((pk, _)) = best {
            if !tie {
                table.primary_key = Some(pk);
            }
        }
        if table.primary_key.is_none() {
            if let Some(pk) = table_name_column(table) {
                table.primary_key = Some(pk);
            }
        }
    }
}

fn infer_foreign_keys_with_config(
    tables: &[TableSpec],
    inferred_fkeys: &mut Vec<InferredFKey>,
    notes: &mut Vec<String>,
    config: &InputGraphConfig,
    data_sources: &HashSet<String>,
) {
    let config_tables: HashMap<String, &InputGraphTable> = config
        .tables
        .iter()
        .map(|t| (t.table_name.clone(), t))
        .collect();

    let discovered_tables: HashSet<String> = tables.iter().map(|t| t.name.clone()).collect();

    // Build a map of table name -> primary key for validation
    let pk_map: HashMap<&str, &str> = tables
        .iter()
        .filter_map(|t| t.primary_key.as_ref().map(|pk| (t.name.as_str(), pk.as_str())))
        .collect();

    for table in tables {
        if let Some(cfg) = config_tables.get(&table.name) {
            if cfg.fkeys.is_empty() {
                eprintln!("INFO: No fkeys defined for table {}", table.name);
            }
            for fk in &cfg.fkeys {
                if !table.columns.iter().any(|c| c.name == fk.name) {
                    let msg = format!(
                        "Config fkey {} in table {} refers to missing column",
                        fk.name, table.name
                    );
                    eprintln!("WARN: {}", msg);
                    notes.push(msg);
                    continue;
                }

                let mut dst_table = fk.dst_table.clone();
                if !discovered_tables.contains(&dst_table) {
                    if let Some(dot_idx) = dst_table.find('.') {
                        let prefix = &dst_table[..dot_idx];
                        if data_sources.contains(prefix) {
                            let stripped = &dst_table[dot_idx + 1..];
                            if discovered_tables.contains(stripped) {
                                eprintln!(
                                    "INFO: Mapping prefixed dst_table {} to discovered table {}",
                                    dst_table, stripped
                                );
                                dst_table = stripped.to_string();
                            }
                        }
                    }
                }

                // Verify dst_key matches the destination table's primary key
                if let Some(&dst_pk) = pk_map.get(dst_table.as_str()) {
                    if dst_pk != fk.dst_key {
                        let msg = format!(
                            "Config fkey {}.{} -> {}.{} skipped: dst_key {} doesn't match primary key {}",
                            table.name, fk.name, dst_table, fk.dst_key, fk.dst_key, dst_pk
                        );
                        eprintln!("WARN: {}", msg);
                        notes.push(msg);
                        continue;
                    }
                }

                eprintln!(
                    "INFO: Adding config fkey {} from {} to {}",
                    fk.name, table.name, dst_table
                );

                inferred_fkeys.push(InferredFKey {
                    src_table: table.name.clone(),
                    src_column: fk.name.clone(),
                    dst_table,
                });
            }
        }
    }
}

pub(crate) fn infer_foreign_keys(
    tables: &[TableSpec],
    inferred_fkeys: &mut Vec<InferredFKey>,
    notes: &mut Vec<String>,
    _config: Option<&GraphConfig>,
) {
    let mut table_map = HashMap::new();
    let mut pk_name_map: HashMap<String, Vec<&TableSpec>> = HashMap::new();
    let mut name_alias_map: HashMap<String, Vec<&TableSpec>> = HashMap::new();
    for table in tables {
        table_map.insert(table.name.to_lowercase(), table);
        if let Some(pk) = &table.primary_key {
            pk_name_map
                .entry(pk.to_lowercase())
                .or_default()
                .push(table);
        }
        let lower = table.name.to_lowercase();
        name_alias_map.entry(lower.clone()).or_default().push(table);
        let singular = singularize(&lower);
        name_alias_map.entry(singular.clone()).or_default().push(table);
        let name_field = format!("{}_name", lower);
        name_alias_map.entry(name_field).or_default().push(table);
        let singular_name_field = format!("{}_name", singular);
        name_alias_map
            .entry(singular_name_field)
            .or_default()
            .push(table);
    }
    for table in tables {
        let table_pk = table.primary_key.as_deref();
        let table_tokens = normalized_tokens(&table.name);
        let fk_candidate_count = table
            .columns
            .iter()
            .filter(|col| {
                let lower = col.name.to_lowercase();
                is_key_like_name(&lower) || name_alias_map.contains_key(&lower)
            })
            .count();
        for col in &table.columns {
            let col_name = col.name.to_lowercase();
            let is_key_like = is_key_like_name(&col_name);
            if !is_key_like {
                let matches_alias = name_alias_map.contains_key(&col_name);
                if !matches_alias {
                    continue;
                }
            } else if matches!(col_name.as_str(), "id" | "uuid" | "pk")
                && fk_candidate_count >= 2
            {
                continue;
            }
            if table_pk == Some(col.name.as_str()) && fk_candidate_count < 2 {
                continue;
            }
            if let Some(dest_tables) = pk_name_map.get(&col_name) {
                let candidates: Vec<&TableSpec> = if table_pk == Some(col.name.as_str()) {
                    dest_tables
                        .iter()
                        .copied()
                        .filter(|t| !t.name.eq_ignore_ascii_case(&table.name))
                        .collect()
                } else {
                    dest_tables.clone()
                };
                if let Some((dst_table, _dst_pk)) =
                    choose_best_fk_dest(table, col, &candidates, &table_tokens, notes)
                {
                    if !fkey_already_declared(
                        inferred_fkeys,
                        &table.name,
                        &col.name,
                        &dst_table,
                    ) {
                        inferred_fkeys.push(InferredFKey {
                            src_table: table.name.clone(),
                            src_column: col.name.clone(),
                            dst_table,
                        });
                    }
                    continue;
                }
            }
            if !is_key_like {
                if let Some(dest_tables) = name_alias_map.get(&col_name) {
                    if let Some((dst_table, _dst_pk)) =
                        choose_best_fk_dest(table, col, dest_tables, &table_tokens, notes)
                    {
                        if !fkey_already_declared(
                            inferred_fkeys,
                            &table.name,
                            &col.name,
                            &dst_table,
                        ) {
                            inferred_fkeys.push(InferredFKey {
                                src_table: table.name.clone(),
                                src_column: col.name.clone(),
                                dst_table,
                            });
                        }
                    }
                    continue;
                }
            }
            let mut best: Option<(String, String, i32)> = None;
            let mut tie = false;
            for (dest_name, dest_table) in &table_map {
                let pk = match &dest_table.primary_key {
                    Some(pk) => pk,
                    None => continue,
                };
                if dest_table.name.eq_ignore_ascii_case(&table.name)
                    && table_pk == Some(col.name.as_str())
                {
                    continue;
                }
                if dest_table.name.eq_ignore_ascii_case(&table.name) {
                    let table_lower = table.name.to_lowercase();
                    let singular = singularize(&table_lower);
                    if !col_name.contains(&table_lower) && !col_name.contains(&singular) {
                        continue;
                    }
                }
                let pk_lower = pk.to_lowercase();
                let mut matches = col_name == pk_lower;
                if !matches {
                    if is_generic_key_name(&pk_lower) {
                        matches = generic_pk_column_match(&col.name, dest_name);
                    } else {
                        matches = col_name.ends_with(&format!("_{}", pk_lower));
                    }
                }
                    if !matches {
                        let col_base = key_name_base(&col_name).unwrap_or("");
                        if !col_base.is_empty() {
                            let dest_tokens = normalized_tokens(dest_name);
                            let base_matches_dest =
                                dest_tokens.iter().any(|t| overlap_ratio(col_base, t) >= 0.6);
                            if base_matches_dest {
                                if is_generic_key_name(&pk_lower) {
                                    if !column_has_type_like(&col.name) {
                                        matches = true;
                                    }
                                } else if let Some(pk_base) = key_name_base(&pk_lower) {
                                    if col_base == pk_base {
                                        matches = true;
                                    }
                                }
                            }
                        }
                    }
                if !matches {
                    continue;
                }
                if let Some(pk_col) = dest_table.columns.iter().find(|c| c.name == *pk) {
                    if !compatible_key_columns(col, pk_col) {
                        notes.push(format!(
                            "skipped foreign key {}.{} -> {} (incompatible key types)",
                            table.name, col.name, dest_table.name
                        ));
                        continue;
                    }
                    let dest_tokens = normalized_tokens(dest_name);
                    let col_tokens = normalized_column_tokens(&col_name);
                    let score =
                        token_overlap_score(&table_tokens, &col_tokens, &dest_tokens, &table.name);
                    match &best {
                        Some((_, _, best_score)) if score == *best_score => {
                            tie = true;
                        }
                        Some((_, _, best_score)) if score < *best_score => {}
                        _ => {
                            best = Some((
                                dest_table.name.clone(),
                                pk.clone(),
                                score,
                            ));
                            tie = false;
                        }
                    }
                } else {
                    notes.push(format!(
                        "skipped foreign key {}.{} -> {} (pk column not found)",
                        table.name, col.name, dest_table.name
                    ));
                }
            }
            if let Some((dst_table, _dst_pk, _)) = best {
                if !tie {
                    if !fkey_already_declared(
                        inferred_fkeys,
                        &table.name,
                        &col.name,
                        &dst_table,
                    ) {
                        inferred_fkeys.push(InferredFKey {
                            src_table: table.name.clone(),
                            src_column: col.name.clone(),
                            dst_table,
                        });
                    }
                }
            }
        }
    }
}

pub(crate) fn infer_primary_key_relationships(tables: &[TableSpec], inferred_fkeys: &mut Vec<InferredFKey>) {
    let mut pk_map: HashMap<String, Vec<&TableSpec>> = HashMap::new();
    for table in tables {
        if let Some(pk) = &table.primary_key {
            pk_map
                .entry(pk.to_lowercase())
                .or_default()
                .push(table);
        }
    }
    for (pk_name, tables_with_pk) in pk_map {
        if tables_with_pk.len() < 2 {
            continue;
        }
        if is_generic_key_name(&pk_name) {
            continue;
        }
        let base = key_name_base(&pk_name).unwrap_or("");
        for i in 0..tables_with_pk.len() {
            for j in (i + 1)..tables_with_pk.len() {
                let a = tables_with_pk[i];
                let b = tables_with_pk[j];
                // Check if this pk-pk relationship already exists
                if pk_relationship_exists(inferred_fkeys, &a.name, &b.name) {
                    continue;
                }
                let a_pk = a.primary_key.as_ref().unwrap();
                let b_pk = b.primary_key.as_ref().unwrap();
                let a_col = a.columns.iter().find(|c| c.name == *a_pk);
                let b_col = b.columns.iter().find(|c| c.name == *b_pk);
                let (Some(a_col), Some(b_col)) = (a_col, b_col) else {
                    continue;
                };
                if !compatible_key_columns(a_col, b_col) {
                    continue;
                }
                let a_name = a.name.to_lowercase();
                let b_name = b.name.to_lowercase();
                let a_tokens = normalized_tokens(&a.name);
                let b_tokens = normalized_tokens(&b.name);
                let table_overlap = overlap_ratio(&a_name, &b_name) >= 0.6;
                let base_overlap = !base.is_empty()
                    && a_tokens.iter().any(|t| overlap_ratio(base, t) >= 0.6)
                    && b_tokens.iter().any(|t| overlap_ratio(base, t) >= 0.6);
                if !(table_overlap || base_overlap) {
                    continue;
                }
                // For pk-pk relationships, src_column is the source table's PK
                let (src, dst) = if a.name <= b.name { (a, b) } else { (b, a) };
                inferred_fkeys.push(InferredFKey {
                    src_table: src.name.clone(),
                    src_column: src.primary_key.clone().unwrap(),
                    dst_table: dst.name.clone(),
                });
            }
        }
    }
}

pub(crate) fn drop_join_table_primary_keys(tables: &mut [TableSpec], inferred_fkeys: &[InferredFKey]) {
    let mut fk_counts: HashMap<&str, usize> = HashMap::new();
    let mut fk_cols: HashSet<(&str, &str)> = HashSet::new();
    for fk in inferred_fkeys {
        *fk_counts.entry(fk.src_table.as_str()).or_default() += 1;
        fk_cols.insert((fk.src_table.as_str(), fk.src_column.as_str()));
    }
    for table in tables {
        let Some(pk) = table.primary_key.as_deref() else {
            continue;
        };
        if !is_join_table_name(&table.name) {
            continue;
        }
        let fk_count = fk_counts.get(table.name.as_str()).copied().unwrap_or(0);
        if fk_count >= 2 && fk_cols.contains(&(table.name.as_str(), pk)) {
            table.primary_key = None;
        }
    }
}

/// Prune FKeys where neither end points to a primary key.
/// For standard FKs: src_column is not a PK, but dst_table must have a PK.
/// For PK-PK relationships: src_column is src_table's PK.
pub(crate) fn prune_fkeys_without_primary_key(tables: &[TableSpec], inferred_fkeys: &mut Vec<InferredFKey>) {
    let pk_map: HashMap<&str, &str> = tables
        .iter()
        .filter_map(|t| t.primary_key.as_ref().map(|pk| (t.name.as_str(), pk.as_str())))
        .collect();

    inferred_fkeys.retain(|fk| {
        // Keep if src_column is src_table's PK (pk-pk relationship)
        let src_is_pk = pk_map.get(fk.src_table.as_str())
            .map_or(false, |pk| *pk == fk.src_column);
        // Keep if dst_table has a PK (standard FK relationship)
        let dst_has_pk = pk_map.contains_key(fk.dst_table.as_str());
        src_is_pk || dst_has_pk
    });
}

pub(crate) fn infer_edge_tables(tables: &mut [TableSpec], inferred_fkeys: &[InferredFKey]) {
    let mut fkeys_by_table: HashMap<&str, Vec<&InferredFKey>> = HashMap::new();
    for fk in inferred_fkeys {
        fkeys_by_table
            .entry(fk.src_table.as_str())
            .or_default()
            .push(fk);
    }
    for table in tables {
        if table.edge_table.is_some() {
            continue;
        }
        let Some(table_fks) = fkeys_by_table.get(table.name.as_str()) else {
            continue;
        };
        let mut fks_by_col: HashMap<&str, &InferredFKey> = HashMap::new();
        for fk in table_fks {
            fks_by_col.entry(fk.src_column.as_str()).or_insert(*fk);
        }
        if fks_by_col.len() != 2 {
            continue;
        }

        // Use shared edge table compatibility check
        let fkey_col_names: HashSet<&str> = fks_by_col.keys().copied().collect();
        if !is_edge_table_compatible(&table.columns, &fkey_col_names, table.create_time.as_deref()) {
            continue;
        }

        let mut ordered: Vec<&InferredFKey> = Vec::new();
        for col in &table.columns {
            if let Some(fk) = fks_by_col.get(col.name.as_str()) {
                ordered.push(*fk);
            }
        }
        if ordered.len() != 2 {
            ordered = fks_by_col.values().copied().collect();
        }
        let src_fk = ordered[0];
        let dst_fk = ordered[1];
        table.edge_table = Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: src_fk.src_column.clone(),
                ref_table: src_fk.dst_table.clone(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: dst_fk.src_column.clone(),
                ref_table: dst_fk.dst_table.clone(),
            },
            forward: None,
            reverse: None,
        });
    }
}

pub(crate) fn prune_fkeys_for_edge_tables(tables: &[TableSpec], inferred_fkeys: &mut Vec<InferredFKey>) {
    let edge_tables: HashSet<&str> = tables
        .iter()
        .filter(|t| t.edge_table.is_some())
        .map(|t| t.name.as_str())
        .collect();

    if edge_tables.is_empty() {
        return;
    }
    inferred_fkeys.retain(|fk| {
        !edge_tables.contains(fk.src_table.as_str())
            && !edge_tables.contains(fk.dst_table.as_str())
    });
}

/// Check if a pk-pk relationship already exists between two tables (in either direction).
fn pk_relationship_exists(
    inferred_fkeys: &[InferredFKey],
    table_a: &str,
    table_b: &str,
) -> bool {
    inferred_fkeys.iter().any(|fk| {
        (fk.src_table == table_a && fk.dst_table == table_b)
            || (fk.src_table == table_b && fk.dst_table == table_a)
    })
}

/// Check if a foreign key is already declared (same src_table, src_column, dst_table).
fn fkey_already_declared(
    inferred_fkeys: &[InferredFKey],
    src_table: &str,
    src_col: &str,
    dst_table: &str,
) -> bool {
    inferred_fkeys.iter().any(|fk| {
        fk.src_table == src_table && fk.src_column == src_col && fk.dst_table == dst_table
    })
}

fn choose_best_fk_dest(
    src_table: &TableSpec,
    src_col: &ColumnSpec,
    dest_tables: &[&TableSpec],
    src_tokens: &[String],
    notes: &mut Vec<String>,
) -> Option<(String, String)> {
    let mut best: Option<(String, String, i32, usize)> = None;
    let mut tie = false;
    for dest_table in dest_tables {
        let pk = match &dest_table.primary_key {
            Some(pk) => pk,
            None => continue,
        };
        if let Some(pk_col) = dest_table.columns.iter().find(|c| c.name == *pk) {
            if !compatible_key_columns(src_col, pk_col) {
                notes.push(format!(
                    "skipped foreign key {}.{} -> {} (incompatible key types)",
                    src_table.name, src_col.name, dest_table.name
                ));
                continue;
            }
            let dest_tokens = normalized_tokens(&dest_table.name);
            let col_tokens = normalized_column_tokens(&src_col.name.to_lowercase());
            let score = token_overlap_score(src_tokens, &col_tokens, &dest_tokens, &src_table.name);
            let name_len = dest_table.name.len();
            match &best {
                Some((_, _, best_score, best_len))
                    if score == *best_score && name_len == *best_len =>
                {
                    tie = true;
                }
                Some((_, _, best_score, best_len))
                    if score < *best_score
                        || (score == *best_score && name_len > *best_len) => {}
                _ => {
                    best = Some((
                        dest_table.name.clone(),
                        pk.clone(),
                        score,
                        name_len,
                    ));
                    tie = false;
                }
            }
        }
    }
    best.and_then(|(name, pk, _score, _len)| if tie { None } else { Some((name, pk)) })
}

fn is_key_like_name(name: &str) -> bool {
    name == "id"
        || name == "uuid"
        || name == "pk"
        || name == "code"
        || name == "key"
        || name.ends_with("_id")
        || name.ends_with("id")
        || name.ends_with("_cd")
        || name.ends_with("_code")
        || name.ends_with("_key")
}

fn is_generic_key_name(name: &str) -> bool {
    matches!(name, "id" | "code" | "cd" | "key")
}

fn normalized_tokens(name: &str) -> Vec<String> {
    let mut tokens = split_tokens(name);
    if tokens.len() > 1 && tokens[0].len() <= 2 {
        tokens.remove(0);
    }
    let mut out = Vec::new();
    for token in tokens {
        let lower = token.to_lowercase();
        if lower == "id" || lower == "code" || lower == "key" {
            continue;
        }
        out.push(lower.clone());
        out.push(singularize(&lower));
    }
    out.retain(|t| !t.is_empty());
    out.sort();
    out.dedup();
    out
}

fn normalized_column_tokens(name: &str) -> Vec<String> {
    let mut tokens = split_tokens(name);
    let mut out = Vec::new();
    for token in tokens.drain(..) {
        let mut lower = token.to_lowercase();
        if lower == "id" || lower == "code" || lower == "key" {
            continue;
        }
        if let Some(stripped) = lower.strip_suffix("id") {
            if stripped.len() >= 2 {
                lower = stripped.to_string();
            }
        } else if let Some(stripped) = lower.strip_suffix("code") {
            if stripped.len() >= 2 {
                lower = stripped.to_string();
            }
        } else if let Some(stripped) = lower.strip_suffix("cd") {
            if stripped.len() >= 2 {
                lower = stripped.to_string();
            }
        } else if let Some(stripped) = lower.strip_suffix("key") {
            if stripped.len() >= 2 {
                lower = stripped.to_string();
            }
        }
        out.push(lower.clone());
        out.push(singularize(&lower));
    }
    out.retain(|t| !t.is_empty());
    out.sort();
    out.dedup();
    out
}

fn token_overlap_score(
    source: &[String],
    column: &[String],
    dest: &[String],
    source_name: &str,
) -> i32 {
    let source_lower = source_name.to_lowercase();
    let mut score = 0;
    for token in dest {
        if source.binary_search(token).is_ok() {
            score += 3;
        }
        if source_lower.starts_with(token) {
            score += 4;
        } else if source_lower.contains(token) {
            score += 2;
        }
        if column.binary_search(token).is_ok() {
            score += 2;
        }
    }
    score
}

fn matches_table_key(table_name: &str, col_lower: &str) -> bool {
    let name = table_name.to_lowercase();
    let singular = singularize(&name);
    col_lower == format!("{}_id", name)
        || col_lower == format!("{}_id", singular)
        || col_lower == format!("{}id", singular)
}

fn is_strong_pk_match(table_name: &str, col_lower: &str) -> bool {
    if col_lower == "id" || col_lower == "uuid" || col_lower == "pk" {
        return true;
    }
    if matches_table_key(table_name, col_lower) {
        return true;
    }
    let name = table_name.to_lowercase().replace('_', "");
    let singular = singularize(&name);
    let col_compact = col_lower.replace('_', "");
    if col_compact == name || col_compact == singular {
        return true;
    }
    let tokens = normalized_tokens(table_name);
    if tokens
        .iter()
        .filter(|t| t.len() >= 3)
        .any(|t| {
            col_lower == format!("{}_id", t)
                || col_lower == format!("{}_code", t)
                || col_lower == format!("{}_cd", t)
        })
    {
        return true;
    }
    if let Some(base) = key_name_base(col_lower) {
        if tokens
            .iter()
            .any(|t| overlap_ratio(base, t) >= 0.6)
        {
            return true;
        }
    }
    false
}

fn key_like_base(name: &str) -> Option<&str> {
    if let Some(base) = name.strip_suffix("_id") {
        return Some(base);
    }
    if let Some(base) = name.strip_suffix("_code") {
        return Some(base);
    }
    if let Some(base) = name.strip_suffix("_cd") {
        return Some(base);
    }
    if let Some(base) = name.strip_suffix("_key") {
        return Some(base);
    }
    None
}

fn split_tokens(name: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut prev_is_lower = false;
    let mut prev_is_digit = false;
    for ch in name.chars() {
        if ch == '_' || ch == '-' || ch == ' ' {
            if !current.is_empty() {
                tokens.push(current);
                current = String::new();
            }
            prev_is_lower = false;
            prev_is_digit = false;
            continue;
        }
        let is_upper = ch.is_ascii_uppercase();
        let is_lower = ch.is_ascii_lowercase();
        let is_digit = ch.is_ascii_digit();
        if !current.is_empty() {
            if (is_upper && prev_is_lower) || (is_digit && !prev_is_digit) || (!is_digit && prev_is_digit)
            {
                tokens.push(current);
                current = String::new();
            }
        }
        current.push(ch);
        prev_is_lower = is_lower;
        prev_is_digit = is_digit;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn key_name_base(name: &str) -> Option<&str> {
    key_like_base(name).or_else(|| {
        if name.ends_with("id") {
            return name.strip_suffix("id");
        }
        if name.ends_with("code") {
            return name.strip_suffix("code");
        }
        if name.ends_with("cd") {
            return name.strip_suffix("cd");
        }
        if name.ends_with("key") {
            return name.strip_suffix("key");
        }
        None
    })
}

fn overlap_ratio(a: &str, b: &str) -> f32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.is_empty() || b_chars.is_empty() {
        return 0.0;
    }
    let mut best = 0usize;
    for i in 0..a_chars.len() {
        for j in 0..b_chars.len() {
            let mut k = 0usize;
            while i + k < a_chars.len()
                && j + k < b_chars.len()
                && a_chars[i + k] == b_chars[j + k]
            {
                k += 1;
            }
            if k > best {
                best = k;
            }
        }
    }
    let denom = a_chars.len().max(b_chars.len());
    best as f32 / denom as f32
}

fn column_matches_tokens(col_lower: &str, tokens: &[String]) -> bool {
    for token in tokens {
        if col_lower.contains(token) {
            return true;
        }
    }
    false
}

fn table_name_column(table: &TableSpec) -> Option<String> {
    let table_lower = table.name.to_lowercase();
    let singular = singularize(&table_lower);
    for col in &table.columns {
        let lower = col.name.to_lowercase();
        if lower == table_lower || lower == singular {
            if is_pk_dtype(&col.ingested_dtype) {
                return Some(col.name.clone());
            }
        }
    }
    None
}

fn column_matches_table_name(table_name: &str, col_lower: &str) -> bool {
    let table_lower = table_name.to_lowercase();
    let singular = singularize(&table_lower);
    col_lower == table_lower || col_lower == singular
}

fn matches_table_name_field(table_name: &str, col_lower: &str) -> bool {
    let table_lower = table_name.to_lowercase();
    let singular = singularize(&table_lower);
    col_lower == format!("{}_name", table_lower) || col_lower == format!("{}_name", singular)
}

fn is_join_table_name(name: &str) -> bool {
    name.contains('_') || name.contains('-') || name.contains(' ')
}

fn table_has_specific_key(table: &TableSpec) -> bool {
    table.columns.iter().any(|col| {
        let lower = col.name.to_lowercase();
        matches_table_key(&table.name, &lower) || column_matches_table_name(&table.name, &lower)
    })
}

fn generic_id_column(table: &TableSpec) -> Option<String> {
    for col in &table.columns {
        let lower = col.name.to_lowercase();
        if matches!(lower.as_str(), "id" | "uuid" | "pk") && is_pk_dtype(&col.ingested_dtype) {
            return Some(col.name.clone());
        }
    }
    None
}

fn should_prefer_generic_id(table: &TableSpec) -> bool {
    if table_has_specific_key(table) {
        return false;
    }
    if table.name.contains('_') || table.name.contains('-') || table.name.contains(' ') {
        return false;
    }
    generic_id_column(table).is_some()
}

fn generic_pk_column_match(col_name: &str, dest_table: &str) -> bool {
    let col_tokens = normalized_column_tokens(col_name);
    let dest_tokens = normalized_primary_tokens(dest_table);
    if col_tokens.is_empty() || dest_tokens.is_empty() {
        return false;
    }
    if has_type_like_token(&col_tokens)
        && !dest_tokens.iter().any(|t| is_type_like_token(t))
    {
        return false;
    }
    dest_tokens
        .iter()
        .all(|t| col_tokens.binary_search(t).is_ok())
}

fn has_type_like_token(tokens: &[String]) -> bool {
    tokens.iter().any(|t| is_type_like_token(t))
}

fn is_type_like_token(token: &str) -> bool {
    matches!(
        token,
        "type" | "kind" | "class" | "category" | "status" | "level" | "role"
    )
}

fn column_has_type_like(name: &str) -> bool {
    let tokens = normalized_column_tokens(name);
    has_type_like_token(&tokens)
}

fn normalized_primary_tokens(name: &str) -> Vec<String> {
    let mut tokens = split_tokens(name);
    let mut out = Vec::new();
    for token in tokens.drain(..) {
        let lower = token.to_lowercase();
        let singular = singularize(&lower);
        if !singular.is_empty() {
            out.push(singular);
        }
    }
    out.sort();
    out.dedup();
    out
}

fn has_composite_code_version(candidate: &str, columns: &[ColumnSpec]) -> bool {
    let Some(prefix) = candidate.strip_suffix("_code") else {
        return false;
    };
    columns.iter().any(|col| col.name.to_lowercase() == format!("{}_version", prefix))
}

pub(crate) fn compatible_key_columns(a: &ColumnSpec, b: &ColumnSpec) -> bool {
    use IngestedType::*;
    match (&a.ingested_dtype, &b.ingested_dtype) {
        (String, String) | (Bytes, Bytes) | (String, Bytes) | (Bytes, String) => true,
        (Uuid, Uuid) | (Uuid256, Uuid256) | (Ip, Ip) => true,
        (a_type, b_type) if is_numeric(a_type) && is_numeric(b_type) => true,
        (a_type, b_type) if is_numeric(a_type) && matches!(b_type, String | Bytes) => true,
        (a_type, b_type) if is_numeric(b_type) && matches!(a_type, String | Bytes) => true,
        _ => false,
    }
}

fn is_pk_dtype(t: &IngestedType) -> bool {
    matches!(
        t,
        IngestedType::String
            | IngestedType::Bytes
            | IngestedType::Uuid
            | IngestedType::Uuid256
            | IngestedType::Ip
    ) || is_numeric(t)
}

fn is_numeric(t: &IngestedType) -> bool {
    matches!(
        t,
        IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64
            | IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64
            | IngestedType::Float32
            | IngestedType::Float64
            | IngestedType::Decimal128
            | IngestedType::Decimal256
    )
}

pub(crate) fn infer_create_time(tables: &mut [TableSpec]) {
    for table in tables.iter_mut() {
        if table.create_time.is_some() {
            continue;
        }
        let time_cols: Vec<&ColumnSpec> = table
            .columns
            .iter()
            .filter(|col| matches!(col.ingested_dtype, IngestedType::Timestamp | IngestedType::Date))
            .collect();
        if time_cols.len() == 1 {
            table.create_time = Some(time_cols[0].name.clone());
            continue;
        }
        let mut best: Option<(String, i32)> = None;
        let mut tie = false;
        for col in time_cols {
            let name = col.name.to_lowercase();
            let score = time_name_score(&name);
            if score <= 0 {
                continue;
            }
            match &best {
                Some((_, best_score)) if score == *best_score => {
                    tie = true;
                }
                Some((_, best_score)) if score < *best_score => {}
                _ => {
                    best = Some((col.name.clone(), score));
                    tie = false;
                }
            }
        }
        if let Some((name, _)) = best {
            if !tie {
                table.create_time = Some(name);
            }
        }
    }
}

fn time_name_score(name: &str) -> i32 {
    let mut score = 0;
    let positives = [
        "create",
        "created",
        "creation",
        "signup",
        "onboard",
        "onboarded",
        "register",
        "registered",
        "acquire",
        "acquired",
        "activation",
        "activated",
        "install",
        "installed",
        "click",
        "clicked",
        "impression",
        "view",
        "viewed",
        "session_start",
        "lead",
        "lead_created",
        "conversion",
        "converted",
        "purchase",
        "purchased",
        "checkout",
        "checkin",
        "ship",
        "shipped",
        "fulfill",
        "fulfilled",
        "invoice",
        "invoiced",
        "bill",
        "billed",
        "pay",
        "paid",
        "capture",
        "captured",
        "funded",
        "settle",
        "settled",
        "auth",
        "authorize",
        "authorized",
        "credit",
        "credited",
        "debit",
        "debited",
        "risk",
        "risked",
        "approve",
        "approved",
        "accept",
        "accepted",
        "signed",
        "signature",
        "engage",
        "engaged",
        "assign",
        "assigned",
        "start",
        "begin",
        "open",
        "opened",
        "ingest",
        "load",
        "event",
        "occur",
        "request",
        "submit",
        "receive",
        "received",
        "posted",
        "settled",
        "cleared",
        "authorized",
        "authorize",
        "entry",
        "enter",
        "insert",
        "inserted",
        "issue",
        "effective",
        "update",
        "updated",
        "modify",
        "modified",
        "record",
        "recorded",
        "first_seen",
        "first",
        "start_date",
        "start_time",
        "created_at",
        "created_on",
        "created_date",
        "created_time",
        "opened_at",
        "opened_on",
        "timestamp",
        "time",
        "date",
    ];
    let negatives = [
        "end",
        "stop",
        "close",
        "closed",
        "out",
        "exit",
        "expire",
        "expired",
        "verify",
        "verified",
        "store",
        "stored",
        "finish",
        "finished",
        "complete",
        "completed",
        "cancel",
        "canceled",
        "abandon",
        "abandoned",
        "delete",
        "deleted",
        "void",
        "voided",
        "chargeback",
        "refunded",
        "refund",
        "reversed",
        "reverse",
        "deny",
        "denied",
        "reject",
        "rejected",
        "fail",
        "failed",
        "bounce",
        "bounced",
        "unsub",
        "unsubscribe",
        "unsubscribed",
        "churn",
        "churned",
        "return",
        "returned",
        "cancelled",
        "last_seen",
        "last",
        "updated_at",
        "updated_on",
        "updated_date",
        "closed_at",
        "closed_on",
        "closed_date",
        "end_date",
        "end_time",
        "stop_date",
        "stop_time",
        "expire_date",
        "expire_time",
        "death",
        "dod",
    ];
    for token in positives {
        if name.contains(token) {
            score += 2;
        }
    }
    for token in negatives {
        if name.contains(token) {
            score -= 3;
        }
    }
    if name.ends_with("_at") || name.ends_with("_time") || name.ends_with("_date") {
        score += 1;
    }
    if name == "timestamp" || name == "time" || name == "date" {
        score += 2;
    }
    score
}

fn int_looks_like_timestamp(min: Option<i64>, max: Option<i64>) -> Option<TimestampUnit> {
    let (Some(min), Some(max)) = (min, max) else {
        return None;
    };
    let ranges = [
        (
            946_684_800i64,
            4_102_444_800i64,
            TimestampUnit::Seconds,
        ),
        (
            946_684_800_000i64,
            4_102_444_800_000i64,
            TimestampUnit::Millis,
        ),
        (
            946_684_800_000_000i64,
            4_102_444_800_000_000i64,
            TimestampUnit::Micros,
        ),
        (
            946_684_800_000_000_000i64,
            4_102_444_800_000_000_000i64,
            TimestampUnit::Nanos,
        ),
    ];
    for (lo, hi, unit) in ranges {
        if min >= lo && max <= hi {
            return Some(unit);
        }
    }
    None
}

fn singularize(name: &str) -> String {
    if name.ends_with("ies") {
        return format!("{}y", &name[..name.len() - 3]);
    }
    if name.ends_with('s') && name.len() > 1 {
        return name[..name.len() - 1].to_string();
    }
    name.to_string()
}

/// Unify column types across FKs so that src_column and dst primary key have compatible types.
fn unify_fkey_column_types(
    tables: &mut [TableSpec],
    inferred_fkeys: &[InferredFKey],
    notes: &mut Vec<String>,
) {
    let table_map: HashMap<String, usize> = tables
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    let mut updates: Vec<(usize, String, IngestedType)> = Vec::new();

    for fk in inferred_fkeys {
        let src_idx = match table_map.get(&fk.src_table) {
            Some(&i) => i,
            None => continue,
        };
        let dst_idx = match table_map.get(&fk.dst_table) {
            Some(&i) => i,
            None => continue,
        };

        // Get destination table's primary key (dst_column is always the PK)
        let dst_pk = match &tables[dst_idx].primary_key {
            Some(pk) => pk.clone(),
            None => continue,
        };

        let src_dtype = tables[src_idx]
            .columns
            .iter()
            .find(|c| c.name == fk.src_column)
            .map(|c| c.ingested_dtype.clone());
        let dst_dtype = tables[dst_idx]
            .columns
            .iter()
            .find(|c| c.name == dst_pk)
            .map(|c| c.ingested_dtype.clone());

        if let (Some(src), Some(dst)) = (src_dtype, dst_dtype) {
            if src != dst {
                if let Some(unified) = unify_types(&src, &dst) {
                    if unified != src {
                        notes.push(format!(
                            "Downgrading column {}.{} from {:?} to {:?} to match {}.{}",
                            fk.src_table, fk.src_column, src, unified, fk.dst_table, dst_pk
                        ));
                        updates.push((src_idx, fk.src_column.clone(), unified.clone()));
                    }
                    if unified != dst {
                        notes.push(format!(
                            "Downgrading column {}.{} from {:?} to {:?} to match {}.{}",
                            fk.dst_table, dst_pk, dst, unified, fk.src_table, fk.src_column
                        ));
                        updates.push((dst_idx, dst_pk.clone(), unified.clone()));
                    }
                }
            }
        }
    }

    for (table_idx, col_name, new_type) in updates {
        if let Some(col) = tables[table_idx]
            .columns
            .iter_mut()
            .find(|c| c.name == col_name)
        {
            col.ingested_dtype = new_type.clone();
            if matches!(new_type, IngestedType::String | IngestedType::Bytes) {
                col.fixed_len = None;
            }
        }
    }
}

fn unify_types(a: &IngestedType, b: &IngestedType) -> Option<IngestedType> {
    use IngestedType::*;
    if a == b {
        return Some(a.clone());
    }
    match (a, b) {
        (String, Uuid) | (Uuid, String) => Some(String),
        (String, Uuid256) | (Uuid256, String) => Some(String),
        (String, Ip) | (Ip, String) => Some(String),
        (Bytes, Uuid) | (Uuid, Bytes) => Some(Bytes),
        (Bytes, Uuid256) | (Uuid256, Bytes) => Some(Bytes),
        (Bytes, Ip) | (Ip, Bytes) => Some(Bytes),
        (Int64, Int32) | (Int32, Int64) => Some(Int64),
        _ => None,
    }
}

// map_kumo_dtype is now imported from diskgraph_ingest::config_converter
