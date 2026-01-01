use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use chrono::{NaiveDate, NaiveDateTime};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use diskgraph_ingest::source_metadata::{
    ColumnSpec, ColumnType, KeyMode, RelationshipSpec, SourceMetadata, TableFormat, TableSpec,
    TimestampUnit,
};
use arrow_array::{
    Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, RecordBatch,
    StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{ArrowError, DataType, Schema};
use futures::StreamExt;
use object_store::{path::Path as ObjectPath, ObjectStore};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::errors::ParquetError;
use tokio::runtime::Runtime;
use url::Url;

const DEFAULT_SAMPLE_ROWS: usize = 2_000;
const DEFAULT_CSV_SAMPLE_BYTES: u64 = 256 * 1024;
const DEFAULT_CSV_SAMPLE_ROWS: usize = 2_000;

#[derive(Clone, Debug)]
pub struct InferOptions {
    pub sample_rows: usize,
    pub csv_sample_bytes: u64,
    pub csv_sample_rows: usize,
    pub validate_dir_schemas: bool,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self {
            sample_rows: DEFAULT_SAMPLE_ROWS,
            csv_sample_bytes: DEFAULT_CSV_SAMPLE_BYTES,
            csv_sample_rows: DEFAULT_CSV_SAMPLE_ROWS,
            validate_dir_schemas: true,
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
    pub metadata: SourceMetadata,
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
    let tables = match root {
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

    let mut table_specs: Vec<TableSpec> = tables
        .iter()
        .map(|table| TableSpec {
            name: table.name.clone(),
            format: table.format.clone(),
            paths: table.paths.clone(),
            columns: table.columns.clone(),
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        })
        .collect();

    let mut relationships = Vec::new();
    infer_primary_keys(&mut table_specs);
    infer_create_time(&mut table_specs);
    infer_foreign_keys(&table_specs, &mut relationships, &mut warnings.notes);
    drop_join_table_primary_keys(&mut table_specs, &relationships);
    infer_primary_key_relationships(&table_specs, &mut relationships);

    let metadata = SourceMetadata {
        dataset_name: root_dataset_name(root),
        tables: table_specs,
        relationships,
        rowpack: None,
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

fn infer_local_tables(
    root: &Path,
    options: &InferOptions,
    warnings: &mut InferWarnings,
) -> Result<Vec<TableInput>> {
    let mut tables = Vec::new();
    let entries = std::fs::read_dir(root).with_context(|| format!("read {}", root.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            let table = infer_dir_table(&name, &path, options, warnings)?;
            if let Some(table) = table {
                tables.push(table);
            }
        } else if let Some(table) = infer_file_table(&name, &path, options)? {
            tables.push(table);
        }
    }
    if tables.is_empty() {
        return Err(anyhow!("no tables found under {}", root.display()));
    }
    Ok(tables)
}

fn infer_file_table(name: &str, path: &Path, options: &InferOptions) -> Result<Option<TableInput>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "csv" => {
            let columns = infer_csv_schema(path, options)?;
            Ok(Some(TableInput {
                name: name.trim_end_matches(".csv").to_string(),
                format: TableFormat::Csv,
                paths: vec![path.to_string_lossy().to_string()],
                columns,
            }))
        }
        "parquet" => {
            let columns = infer_parquet_schema_local(path, options)?;
            Ok(Some(TableInput {
                name: name.trim_end_matches(".parquet").to_string(),
                format: TableFormat::Parquet,
                paths: vec![path.to_string_lossy().to_string()],
                columns,
            }))
        }
        _ => Ok(None),
    }
}

fn infer_dir_table(
    name: &str,
    dir: &Path,
    options: &InferOptions,
    warnings: &mut InferWarnings,
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
        let columns = match format {
            TableFormat::Csv => infer_csv_schema(&path, options)?,
            TableFormat::Parquet => infer_parquet_schema_local(&path, options)?,
        };
        match &mut table {
            Some(existing) => {
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
                existing.paths.push(path.to_string_lossy().to_string());
            }
            None => {
                table = Some(TableInput {
                    name: name.to_string(),
                    format,
                    paths: vec![path.to_string_lossy().to_string()],
                    columns,
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
    let (rt, store) = build_store(root)?;
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
                let format_clone = format.clone();
                let columns = match format {
                    TableFormat::Parquet => infer_parquet_schema_remote(
                        &rt,
                        Arc::clone(&store),
                        &obj.location,
                        obj.size,
                        options,
                    )?,
                    TableFormat::Csv => infer_csv_schema_remote(
                        &rt,
                        store.as_ref(),
                        &obj.location,
                        obj.size,
                        options,
                    )?,
                };
                let entry = tables.entry(table_name.to_string()).or_insert_with(|| TableInput {
                    name: table_name.to_string(),
                    format: format_clone,
                    paths: Vec::new(),
                    columns: columns.clone(),
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
                        rel_path
                    ));
                }
            }
        } else {
            prefixes.insert(parts[0].to_string());
            let file = parts.last().unwrap_or(&"");
            if let Some((_, format)) = split_table_file(file) {
                let table_name = parts[0];
                let format_clone = format.clone();
                let columns = match format {
                    TableFormat::Parquet => infer_parquet_schema_remote(
                        &rt,
                        Arc::clone(&store),
                        &obj.location,
                        obj.size,
                        options,
                    )?,
                    TableFormat::Csv => infer_csv_schema_remote(
                        &rt,
                        store.as_ref(),
                        &obj.location,
                        obj.size,
                        options,
                    )?,
                };
                let entry = tables.entry(table_name.to_string()).or_insert_with(|| TableInput {
                    name: table_name.to_string(),
                    format: format_clone,
                    paths: Vec::new(),
                    columns: columns.clone(),
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
                        rel_path
                    ));
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
        && column_type_eq(&lhs.source_dtype, &rhs.source_dtype)
        && lhs.fixed_len == rhs.fixed_len
        && lhs.key_mode == rhs.key_mode
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

fn column_type_eq(a: &ColumnType, b: &ColumnType) -> bool {
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

fn infer_parquet_schema_local(path: &Path, options: &InferOptions) -> Result<Vec<ColumnSpec>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let mut columns = infer_parquet_schema_from_arrow(schema.as_ref())?;
    if options.sample_rows > 0 {
    let mut reader = builder.with_batch_size(1024).build()?;
    apply_parquet_string_samples(&mut columns, &mut reader, options.sample_rows)?;
    }
    Ok(columns)
}

fn infer_parquet_schema_remote(
    rt: &Runtime,
    store: Arc<dyn ObjectStore>,
    path: &ObjectPath,
    size: u64,
    options: &InferOptions,
) -> Result<Vec<ColumnSpec>> {
    let reader = ParquetObjectReader::new(store, path.clone())
        .with_file_size(size)
        .with_runtime(rt.handle().clone());
    let builder = rt.block_on(ParquetRecordBatchStreamBuilder::new(reader))?;
    let schema = builder.schema();
    let mut columns = infer_parquet_schema_from_arrow(schema.as_ref())?;
    if columns.is_empty() {
        return Ok(columns);
    }
    let sample_rows = options.sample_rows;
    if sample_rows > 0 {
        let stream = builder.with_batch_size(1024).build()?;
        rt.block_on(apply_parquet_string_samples_stream(
            &mut columns,
            stream,
            sample_rows,
        ))?;
    }
    Ok(columns)
}

fn infer_parquet_schema_from_arrow(schema: &Schema) -> Result<Vec<ColumnSpec>> {
    let mut columns = Vec::new();
    for field in schema.fields() {
        let (dtype, fixed_len, precision, scale) = arrow_type_to_column_type(field.data_type())?;
        columns.push(ColumnSpec {
            name: field.name().to_string(),
            source_dtype: dtype.clone(),
            ingested_dtype: dtype,
            fixed_len,
            timestamp_format: None,
            timestamp_unit: None,
            key_mode: None,
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
    let mut stats: Vec<Option<ColumnStats>> = columns
        .iter()
        .map(|col| {
            if matches!(
                col.source_dtype,
                ColumnType::String
                    | ColumnType::Int8
                    | ColumnType::Int16
                    | ColumnType::Int32
                    | ColumnType::Int64
                    | ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64
            ) {
                Some(ColumnStats::default())
            } else {
                None
            }
        })
        .collect();
    let mut seen = 0usize;
    while let Some(batch) = batches.next() {
        let batch = batch.map_err(|err| anyhow!(err))?;
        if seen >= sample_rows {
            break;
        }
        let rows = batch.num_rows().min(sample_rows - seen);
        for (idx, col) in columns.iter().enumerate() {
            if !matches!(
                col.source_dtype,
                ColumnType::String
                    | ColumnType::Int8
                    | ColumnType::Int16
                    | ColumnType::Int32
                    | ColumnType::Int64
                    | ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64
            ) {
                continue;
            }
            if let Some(st) = stats[idx].as_mut() {
                let array = batch.column(idx);
                if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                    sample_string_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
                    sample_large_string_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                    sample_int64_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                    sample_int32_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
                    sample_int16_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
                    sample_int8_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
                    sample_uint64_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
                    sample_uint32_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
                    sample_uint16_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
                    sample_uint8_array(arr, rows, st);
                }
            }
        }
        seen += rows;
    }
    apply_string_stats_overrides(columns, &stats);
    Ok(())
}

async fn apply_parquet_string_samples_stream<S>(
    columns: &mut [ColumnSpec],
    mut stream: S,
    sample_rows: usize,
) -> Result<()>
where
    S: futures::Stream<Item = std::result::Result<RecordBatch, ParquetError>> + Unpin,
{
    let mut stats: Vec<Option<ColumnStats>> = columns
        .iter()
        .map(|col| {
            if matches!(
                col.source_dtype,
                ColumnType::String
                    | ColumnType::Int8
                    | ColumnType::Int16
                    | ColumnType::Int32
                    | ColumnType::Int64
                    | ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64
            ) {
                Some(ColumnStats::default())
            } else {
                None
            }
        })
        .collect();
    let mut seen = 0usize;
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|err| anyhow!(err))?;
        if seen >= sample_rows {
            break;
        }
        let rows = batch.num_rows().min(sample_rows - seen);
        for (idx, col) in columns.iter().enumerate() {
            if !matches!(
                col.source_dtype,
                ColumnType::String
                    | ColumnType::Int8
                    | ColumnType::Int16
                    | ColumnType::Int32
                    | ColumnType::Int64
                    | ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64
            ) {
                continue;
            }
            if let Some(st) = stats[idx].as_mut() {
                let array = batch.column(idx);
                if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                    sample_string_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
                    sample_large_string_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                    sample_int64_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                    sample_int32_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
                    sample_int16_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
                    sample_int8_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
                    sample_uint64_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
                    sample_uint32_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
                    sample_uint16_array(arr, rows, st);
                } else if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
                    sample_uint8_array(arr, rows, st);
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
        if matches!(inferred.ingested_dtype, ColumnType::Timestamp | ColumnType::Date) {
            col.ingested_dtype = inferred.ingested_dtype;
            col.timestamp_format = inferred.timestamp_format;
            col.timestamp_unit = inferred.timestamp_unit;
        }
    }
}

fn arrow_type_to_column_type(
    source_dtype: &DataType,
) -> Result<(ColumnType, Option<usize>, Option<u32>, Option<i32>)> {
    let mut precision = None;
    let mut scale = None;
    let mut fixed_len = None;
    let out = match source_dtype {
        DataType::Int8 => ColumnType::Int8,
        DataType::Int16 => ColumnType::Int16,
        DataType::Int32 => ColumnType::Int32,
        DataType::Int64 => ColumnType::Int64,
        DataType::UInt8 => ColumnType::UInt8,
        DataType::UInt16 => ColumnType::UInt16,
        DataType::UInt32 => ColumnType::UInt32,
        DataType::UInt64 => ColumnType::UInt64,
        DataType::Float32 => ColumnType::Float32,
        DataType::Float64 => ColumnType::Float64,
        DataType::Boolean => ColumnType::Bool,
        DataType::Utf8 | DataType::LargeUtf8 => ColumnType::String,
        DataType::Binary | DataType::LargeBinary => ColumnType::Bytes,
        DataType::FixedSizeBinary(len) => {
            fixed_len = Some(*len as usize);
            ColumnType::Bytes
        }
        DataType::Date32 => ColumnType::Date,
        DataType::Timestamp(_, _) => ColumnType::Timestamp,
        DataType::Decimal128(p, s) => {
            precision = Some(*p as u32);
            scale = Some(*s as i32);
            ColumnType::Decimal128
        }
        DataType::Decimal256(p, s) => {
            precision = Some(*p as u32);
            scale = Some(*s as i32);
            ColumnType::Decimal256
        }
        DataType::List(field) | DataType::LargeList(field) => match field.data_type() {
            DataType::Float32 => ColumnType::Float32Array,
            DataType::Float64 => ColumnType::Float64Array,
            _ => return Err(anyhow!("unsupported list type {}", field.data_type())),
        },
        _ => return Err(anyhow!("unsupported parquet type {}", source_dtype)),
    };
    Ok((out, fixed_len, precision, scale))
}

fn infer_csv_schema(path: &Path, options: &InferOptions) -> Result<Vec<ColumnSpec>> {
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
    let mut stats: Vec<ColumnStats> = headers.iter().map(|_| ColumnStats::default()).collect();
    for result in reader.records().take(options.csv_sample_rows) {
        let record = result?;
        for (idx, value) in record.iter().enumerate() {
            if idx >= stats.len() {
                continue;
            }
            stats[idx].observe(value);
        }
    }
    Ok(headers
        .iter()
        .zip(stats.iter())
        .map(|(name, st)| {
            let mut col = st.to_column_spec(name);
            col.source_dtype = ColumnType::String;
            col
        })
        .collect())
}

fn infer_csv_schema_remote(
    rt: &Runtime,
    store: &dyn ObjectStore,
    path: &ObjectPath,
    size: u64,
    options: &InferOptions,
) -> Result<Vec<ColumnSpec>> {
    let max = size.min(options.csv_sample_bytes);
    let range = 0..max;
    let bytes = rt.block_on(store.get_range(path, range))?;
    infer_csv_schema_from_bytes(&bytes, options)
}

fn infer_csv_schema_from_bytes(bytes: &Bytes, options: &InferOptions) -> Result<Vec<ColumnSpec>> {
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
    let mut stats: Vec<ColumnStats> = headers.iter().map(|_| ColumnStats::default()).collect();
    for result in reader.records().take(options.csv_sample_rows) {
        let record = result?;
        for (idx, value) in record.iter().enumerate() {
            if idx >= stats.len() {
                continue;
            }
            stats[idx].observe(value);
        }
    }
    Ok(headers
        .iter()
        .zip(stats.iter())
        .map(|(name, st)| st.to_column_spec(name))
        .collect())
}

#[derive(Clone, Default)]
struct ColumnStats {
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

    fn observe(&mut self, value: &str) {
        if value.trim().is_empty() {
            return;
        }
        self.seen += 1;
        if self.seen == 1 {
            self.int_ok = true;
            self.float_ok = true;
            self.date_ok = true;
            self.ts_ok = true;
            self.bool_ok = true;
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

    fn to_column_spec(&self, name: &str) -> ColumnSpec {
        let name_lower = name.to_lowercase();
        let mut source_dtype = ColumnType::String;
        let mut ingested_dtype = ColumnType::String;
        let mut timestamp_format = None;
        let mut timestamp_unit = None;
        if self.bool_ok {
            source_dtype = ColumnType::Bool;
            ingested_dtype = ColumnType::Bool;
        } else if self.ts_ok {
            source_dtype = ColumnType::String;
            ingested_dtype = ColumnType::Timestamp;
            // Preserve the parse format for string-backed timestamps.
            timestamp_format = self.ts_format.clone();
        } else if time_name_score(&name_lower) > 0
            && self.int_ok
            && int_looks_like_timestamp(self.int_min, self.int_max).is_some()
        {
            source_dtype = ColumnType::Int64;
            ingested_dtype = ColumnType::Timestamp;
            // Record the epoch unit for integer timestamps.
            timestamp_unit = int_looks_like_timestamp(self.int_min, self.int_max);
        } else if self.date_ok {
            source_dtype = ColumnType::String;
            ingested_dtype = ColumnType::Date;
            // Preserve the parse format for string-backed dates.
            timestamp_format = self.date_format.clone();
        } else if self.int_ok {
            source_dtype = ColumnType::Int64;
            ingested_dtype = ColumnType::Int64;
        } else if self.float_ok {
            source_dtype = ColumnType::Float64;
            ingested_dtype = ColumnType::Float64;
        }
        ColumnSpec {
            name: name.to_string(),
            source_dtype,
            ingested_dtype,
            fixed_len: None,
            timestamp_format,
            timestamp_unit,
            key_mode: None,
            precision: None,
            scale: None,
        }
    }
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
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(value) {
        return Some(ts.naive_utc());
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Some(ts);
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(ts);
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M") {
        return Some(ts);
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Some(ts);
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(ts);
    }
    None
}

fn parse_timestamp_format(value: &str) -> Option<&'static str> {
    if chrono::DateTime::parse_from_rfc3339(value).is_ok() {
        return Some("rfc3339");
    }
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
    ] {
        if NaiveDateTime::parse_from_str(value, fmt).is_ok() {
            return Some(fmt);
        }
    }
    None
}

fn infer_primary_keys(tables: &mut [TableSpec]) {
    let mut freq = HashMap::new();
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
            if !is_key_like_name(&lower) && !matches_table {
                continue;
            }
            if has_composite_code_version(&lower, &table.columns) {
                continue;
            }
            let count = if matches_table {
                1
            } else {
                *freq.get(&lower).unwrap_or(&usize::MAX) as i32
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
            } else if has_many_key_like {
                continue;
            } else if require_strong_for_small {
                continue;
            }
            if strong_match {
                score += 1000;
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

fn infer_foreign_keys(
    tables: &[TableSpec],
    rels: &mut Vec<RelationshipSpec>,
    notes: &mut Vec<String>,
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
        name_alias_map.entry(singular).or_default().push(table);
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
                if let Some(best) =
                    choose_best_fk_dest(table, col, &candidates, &table_tokens, notes)
                {
                    rels.push(RelationshipSpec {
                        name: None,
                        src_table: table.name.clone(),
                        src_column: col.name.clone(),
                        dst_table: best.0,
                        dst_column: best.1,
                    });
                    continue;
                }
            }
            if !is_key_like {
                if let Some(dest_tables) = name_alias_map.get(&col_name) {
                    if let Some(best) =
                        choose_best_fk_dest(table, col, dest_tables, &table_tokens, notes)
                    {
                        rels.push(RelationshipSpec {
                            name: None,
                            src_table: table.name.clone(),
                            src_column: col.name.clone(),
                            dst_table: best.0,
                            dst_column: best.1,
                        });
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
            if let Some((dest_table, dest_pk, _)) = best {
                if !tie {
                    rels.push(RelationshipSpec {
                        name: None,
                        src_table: table.name.clone(),
                        src_column: col.name.clone(),
                        dst_table: dest_table,
                        dst_column: dest_pk,
                    });
                }
            }
        }
    }
}

fn infer_primary_key_relationships(tables: &[TableSpec], rels: &mut Vec<RelationshipSpec>) {
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
                if relationship_exists(rels, &a.name, &a.primary_key, &b.name, &b.primary_key) {
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
                let (src, dst) = if a.name <= b.name { (a, b) } else { (b, a) };
                rels.push(RelationshipSpec {
                    name: None,
                    src_table: src.name.clone(),
                    src_column: src.primary_key.clone().unwrap(),
                    dst_table: dst.name.clone(),
                    dst_column: dst.primary_key.clone().unwrap(),
                });
            }
        }
    }
}

fn drop_join_table_primary_keys(tables: &mut [TableSpec], rels: &[RelationshipSpec]) {
    let mut fk_counts: HashMap<&str, usize> = HashMap::new();
    let mut fk_cols: HashSet<(&str, &str)> = HashSet::new();
    for rel in rels {
        *fk_counts.entry(rel.src_table.as_str()).or_default() += 1;
        fk_cols.insert((rel.src_table.as_str(), rel.src_column.as_str()));
    }
    for table in tables {
        let Some(pk) = table.primary_key.as_deref() else {
            continue;
        };
        let fk_count = fk_counts.get(table.name.as_str()).copied().unwrap_or(0);
        if fk_count >= 2 && fk_cols.contains(&(table.name.as_str(), pk)) {
            table.primary_key = None;
        }
    }
}

fn relationship_exists(
    rels: &[RelationshipSpec],
    src_table: &str,
    src_col: &Option<String>,
    dst_table: &str,
    dst_col: &Option<String>,
) -> bool {
    let (Some(src_col), Some(dst_col)) = (src_col, dst_col) else {
        return false;
    };
    rels.iter().any(|rel| {
        (rel.src_table == src_table && rel.src_column == *src_col && rel.dst_table == dst_table
            && rel.dst_column == *dst_col)
            || (rel.src_table == dst_table
                && rel.src_column == *dst_col
                && rel.dst_table == src_table
                && rel.dst_column == *src_col)
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

fn compatible_key_columns(a: &ColumnSpec, b: &ColumnSpec) -> bool {
    use ColumnType::*;
    match (&a.ingested_dtype, &b.ingested_dtype) {
        (String, String) | (Bytes, Bytes) | (String, Bytes) | (Bytes, String) => {
            let a_mode = string_key_mode(a);
            let b_mode = string_key_mode(b);
            match (a_mode, b_mode) {
                (KeyStringMode::Uuid, KeyStringMode::Uuid) => true,
                (KeyStringMode::Fixed(a_len), KeyStringMode::Fixed(b_len)) => a_len == b_len,
                (KeyStringMode::Var, KeyStringMode::Var) => true,
                _ => false,
            }
        }
        (a_type, b_type) if is_numeric(a_type) && is_numeric(b_type) => true,
        (a_type, b_type) if is_numeric(a_type) && matches!(b_type, String | Bytes) => {
            string_key_mode(b).is_var()
        }
        (a_type, b_type) if is_numeric(b_type) && matches!(a_type, String | Bytes) => {
            string_key_mode(a).is_var()
        }
        _ => false,
    }
}

fn is_pk_dtype(t: &ColumnType) -> bool {
    matches!(t, ColumnType::String | ColumnType::Bytes) || is_numeric(t)
}

#[derive(Clone, Copy)]
enum KeyStringMode {
    Var,
    Fixed(usize),
    Uuid,
}

impl KeyStringMode {
    fn is_var(self) -> bool {
        matches!(self, KeyStringMode::Var)
    }
}

fn string_key_mode(col: &ColumnSpec) -> KeyStringMode {
    if col.key_mode == Some(KeyMode::Uuid) {
        return KeyStringMode::Uuid;
    }
    let len = col.fixed_len;
    if col.key_mode == Some(KeyMode::Fixed) || len.is_some() {
        return KeyStringMode::Fixed(len.unwrap_or(0));
    }
    KeyStringMode::Var
}

fn is_numeric(t: &ColumnType) -> bool {
    matches!(
        t,
        ColumnType::Int8
            | ColumnType::Int16
            | ColumnType::Int32
            | ColumnType::Int64
            | ColumnType::UInt8
            | ColumnType::UInt16
            | ColumnType::UInt32
            | ColumnType::UInt64
            | ColumnType::Float32
            | ColumnType::Float64
            | ColumnType::Decimal128
            | ColumnType::Decimal256
    )
}

fn infer_create_time(tables: &mut [TableSpec]) {
    for table in tables.iter_mut() {
        if table.create_time.is_some() {
            continue;
        }
        let time_cols: Vec<&ColumnSpec> = table
            .columns
            .iter()
            .filter(|col| matches!(col.ingested_dtype, ColumnType::Timestamp | ColumnType::Date))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn table(name: &str, cols: &[(&str, ColumnType)]) -> TableSpec {
        TableSpec {
            name: name.to_string(),
            format: TableFormat::Csv,
            paths: vec![],
            columns: cols
                .iter()
                .map(|(n, t)| ColumnSpec {
                    name: n.to_string(),
                    source_dtype: t.clone(),
                    ingested_dtype: t.clone(),
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                })
                .collect(),
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        }
    }

    fn rel_exists(
        rels: &[RelationshipSpec],
        src_table: &str,
        src_col: &str,
        dst_table: &str,
        dst_col: &str,
    ) -> bool {
        rels.iter().any(|rel| {
            rel.src_table == src_table
                && rel.src_column == src_col
                && rel.dst_table == dst_table
                && rel.dst_column == dst_col
        })
    }

    #[test]
    fn infer_pk_id() {
        let mut t = table("users", &[("id", ColumnType::Int64)]);
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("id"));
    }

    #[test]
    fn infer_pk_table_id() {
        let mut t = table("orders", &[("order_id", ColumnType::Int64)]);
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("order_id"));
    }

    #[test]
    fn infer_pk_last_token_only_when_single_id() {
        let mut t = table("kaggle_hm_articles", &[("article_id", ColumnType::Int64)]);
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("article_id"));
    }

    #[test]
    fn infer_pk_uuid() {
        let mut t = table("sessions", &[("uuid", ColumnType::String)]);
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("uuid"));
    }

    #[test]
    fn infer_pk_prefers_customer_id_over_postal_code() {
        let mut t = TableSpec {
            name: "kaggle_hm_customers".to_string(),
            format: TableFormat::Parquet,
            paths: vec!["/tmp/customers.parquet".to_string()],
            columns: vec![
                ColumnSpec {
                    name: "customer_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                },
                ColumnSpec {
                    name: "postal_code".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                },
            ],
            primary_key: None,
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        };
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("customer_id"));
    }

    #[test]
    fn infer_pk_ignores_user_id_in_join_table() {
        let mut t = table(
            "user_interest",
            &[("user_id", ColumnType::Int64), ("interest_id", ColumnType::Int64)],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key, None);
    }

    #[test]
    fn rel_avito_infers_ads_and_users_primary_keys() {
        let mut tables = vec![
            table(
                "AdsInfo",
                &[
                    ("AdID", ColumnType::Int64),
                    ("LocationID", ColumnType::Int64),
                    ("CategoryID", ColumnType::Int64),
                ],
            ),
            table(
                "UserInfo",
                &[
                    ("UserID", ColumnType::Int64),
                    ("UserAgentID", ColumnType::Int64),
                    ("UserDeviceID", ColumnType::Int64),
                ],
            ),
        ];
        infer_primary_keys(&mut tables);
        let ads = tables.iter().find(|t| t.name == "AdsInfo").unwrap();
        let users = tables.iter().find(|t| t.name == "UserInfo").unwrap();
        assert_eq!(ads.primary_key.as_deref(), Some("AdID"));
        assert_eq!(users.primary_key.as_deref(), Some("UserID"));
    }

    #[test]
    fn rel_event_infers_join_keys_and_foreign_keys() {
        let mut tables = vec![
            table(
                "events",
                &[
                    ("event_id", ColumnType::Int64),
                    ("user_id", ColumnType::Int64),
                    ("start_time", ColumnType::Timestamp),
                ],
            ),
            table(
                "users",
                &[
                    ("user_id", ColumnType::Int64),
                    ("joinedAt", ColumnType::Timestamp),
                ],
            ),
            table(
                "event_attendees",
                &[
                    ("event", ColumnType::Int64),
                    ("user_id", ColumnType::Int64),
                    ("start_time", ColumnType::Timestamp),
                ],
            ),
            table(
                "event_interest",
                &[
                    ("user", ColumnType::Int64),
                    ("event", ColumnType::Int64),
                    ("timestamp", ColumnType::Timestamp),
                ],
            ),
        ];
        infer_primary_keys(&mut tables);
        infer_create_time(&mut tables);
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        drop_join_table_primary_keys(&mut tables, &rels);
        infer_primary_key_relationships(&tables, &mut rels);

        let attendees = tables
            .iter()
            .find(|t| t.name == "event_attendees")
            .unwrap();
        assert_eq!(attendees.primary_key, None);
        assert!(rel_exists(
            &rels,
            "event_attendees",
            "event",
            "events",
            "event_id"
        ));
        assert!(rel_exists(
            &rels,
            "event_attendees",
            "user_id",
            "users",
            "user_id"
        ));
        assert!(rel_exists(
            &rels,
            "event_interest",
            "event",
            "events",
            "event_id"
        ));
        assert!(rel_exists(
            &rels,
            "event_interest",
            "user",
            "users",
            "user_id"
        ));
    }

    #[test]
    fn rel_salt_prefers_customer_id_and_address_fk() {
        let mut tables = vec![
            table("address", &[("ADDRESSID", ColumnType::Int64)]),
            table(
                "customer",
                &[
                    ("CUSTOMER", ColumnType::Int64),
                    ("ADDRESSID", ColumnType::Int64),
                ],
            ),
        ];
        infer_primary_keys(&mut tables);
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);

        let customer = tables.iter().find(|t| t.name == "customer").unwrap();
        assert_eq!(customer.primary_key.as_deref(), Some("CUSTOMER"));
        assert!(rel_exists(
            &rels,
            "customer",
            "ADDRESSID",
            "address",
            "ADDRESSID"
        ));
    }

    #[test]
    fn rel_stack_adds_related_post_fk_and_skips_history_type_fk() {
        let mut tables = vec![
            table("posts", &[("Id", ColumnType::Int64)]),
            table("users", &[("Id", ColumnType::Int64)]),
            table(
                "postLinks",
                &[
                    ("Id", ColumnType::Int64),
                    ("PostId", ColumnType::Int64),
                    ("RelatedPostId", ColumnType::Int64),
                    ("CreationDate", ColumnType::Timestamp),
                ],
            ),
            table(
                "postHistory",
                &[
                    ("Id", ColumnType::Int64),
                    ("PostId", ColumnType::Int64),
                    ("UserId", ColumnType::Int64),
                    ("PostHistoryTypeId", ColumnType::Int64),
                    ("CreationDate", ColumnType::Timestamp),
                ],
            ),
        ];
        infer_primary_keys(&mut tables);
        infer_create_time(&mut tables);
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        drop_join_table_primary_keys(&mut tables, &rels);
        infer_primary_key_relationships(&tables, &mut rels);

        assert!(rel_exists(
            &rels,
            "postLinks",
            "RelatedPostId",
            "posts",
            "Id"
        ));
        assert!(!rel_exists(
            &rels,
            "postHistory",
            "PostHistoryTypeId",
            "postHistory",
            "Id"
        ));
    }

    #[test]
    fn rel_trial_handles_join_tables_and_fk_direction() {
        let mut tables = vec![
            table("conditions", &[("condition_id", ColumnType::Int64)]),
            table("studies", &[("nct_id", ColumnType::String)]),
            table(
                "designs",
                &[("id", ColumnType::Int64), ("nct_id", ColumnType::String)],
            ),
            table(
                "conditions_studies",
                &[
                    ("id", ColumnType::Int64),
                    ("nct_id", ColumnType::String),
                    ("condition_id", ColumnType::Int64),
                    ("date", ColumnType::Date),
                ],
            ),
        ];
        infer_primary_keys(&mut tables);
        infer_create_time(&mut tables);
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        drop_join_table_primary_keys(&mut tables, &rels);
        infer_primary_key_relationships(&tables, &mut rels);

        let join = tables
            .iter()
            .find(|t| t.name == "conditions_studies")
            .unwrap();
        assert_eq!(join.primary_key, None);
        assert!(rel_exists(
            &rels,
            "conditions_studies",
            "condition_id",
            "conditions",
            "condition_id"
        ));
        assert!(rel_exists(
            &rels,
            "conditions_studies",
            "nct_id",
            "studies",
            "nct_id"
        ));
        assert!(!rel_exists(
            &rels,
            "conditions_studies",
            "id",
            "designs",
            "id"
        ));
    }

    #[test]
    fn infer_fk_relationship() {
        let tables = vec![
            TableSpec {
                name: "users".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "user_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("user_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "orders".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "user_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].src_table, "orders");
        assert_eq!(rels[0].dst_table, "users");
    }

    #[test]
    fn infer_fk_skips_incompatible() {
        let tables = vec![
            TableSpec {
                name: "devices".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "device_id".to_string(),
                    source_dtype: ColumnType::Bytes,
                    ingested_dtype: ColumnType::Bytes,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: Some(16),
                    key_mode: Some(KeyMode::Fixed),
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("device_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "sessions".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "device_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert!(rels.is_empty());
        assert!(!notes.is_empty());
    }

    #[test]
    fn infer_pk_prefers_rare_key() {
        let admits = table(
            "admits",
            &[
                ("subject_id", ColumnType::Int64),
                ("hadm_id", ColumnType::Int64),
                ("admittime", ColumnType::Timestamp),
                ("dischtime", ColumnType::Timestamp),
            ],
        );
        let pats = table("pats", &[("subject_id", ColumnType::Int64)]);
        let encounters = table("encounters", &[("subject_id", ColumnType::Int64)]);
        let mut tables = vec![admits, pats, encounters];
        infer_primary_keys(&mut tables);
        assert_eq!(tables[0].primary_key.as_deref(), Some("hadm_id"));
    }

    #[test]
    fn infer_fk_prefers_best_token_match() {
        let tables = vec![
            TableSpec {
                name: "d_items".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("itemid".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "d_lab".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("itemid".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "labevents".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].src_table, "labevents");
        assert_eq!(rels[0].dst_table, "d_lab");
    }

    #[test]
    fn infer_fk_self_reference_suffix() {
        let tables = vec![TableSpec {
            name: "poe".to_string(),
            format: TableFormat::Csv,
            paths: vec![],
            columns: vec![
                ColumnSpec {
                    name: "poe_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                },
                ColumnSpec {
                    name: "discontinue_of_poe_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                },
            ],
            primary_key: Some("poe_id".to_string()),
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        }];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].src_table, "poe");
        assert_eq!(rels[0].dst_table, "poe");
        assert_eq!(rels[0].src_column, "discontinue_of_poe_id");
    }

    #[test]
    fn infer_pk_skips_code_with_version() {
        let mut t = table(
            "d_icd_diag",
            &[
                ("icd_code", ColumnType::String),
                ("icd_version", ColumnType::Int16),
            ],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key, None);
    }

    #[test]
    fn infer_pk_token_id_match() {
        let mut t = table(
            "icu_stays",
            &[
                ("subject_id", ColumnType::Int64),
                ("hadm_id", ColumnType::Int64),
                ("stay_id", ColumnType::Int64),
            ],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("stay_id"));
    }

    #[test]
    fn infer_pk_token_cd_match() {
        let mut t = table(
            "hcpcs",
            &[
                ("subject_id", ColumnType::Int64),
                ("hadm_id", ColumnType::Int64),
                ("hcpcs_cd", ColumnType::String),
            ],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("hcpcs_cd"));
    }

    #[test]
    fn infer_pk_small_two_ids_requires_strong_match() {
        let mut t = table(
            "encounters",
            &[
                ("subject_id", ColumnType::Int64),
                ("hadm_id", ColumnType::Int64),
                ("date", ColumnType::Date),
            ],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key, None);
    }

    #[test]
    fn infer_fk_base_match_allows_code() {
        let tables = vec![
            TableSpec {
                name: "d_hcpcs".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "code".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("code".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "hcpcs".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "hcpcs_cd".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].src_table, "hcpcs");
        assert_eq!(rels[0].dst_table, "d_hcpcs");
    }

    #[test]
    fn infer_fk_avoids_generic_code_match() {
        let tables = vec![
            TableSpec {
                name: "d_hcpcs".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "code".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("code".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "d_icd_diag".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "icd_code".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert!(rels.is_empty());
    }

    #[test]
    fn infer_pk_overlap_ratio_match() {
        let mut t = table(
            "admits",
            &[
                ("subject_id", ColumnType::Int64),
                ("hadm_id", ColumnType::Int64),
                ("admittime", ColumnType::Timestamp),
                ("dischtime", ColumnType::Timestamp),
            ],
        );
        infer_primary_keys(std::slice::from_mut(&mut t));
        assert_eq!(t.primary_key.as_deref(), Some("hadm_id"));
    }

    #[test]
    fn infer_fk_prefers_shorter_table_on_tie() {
        let tables = vec![
            TableSpec {
                name: "poe".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "poe_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("poe_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "poe_detail".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "poe_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("poe_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "emar".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "poe_id".to_string(),
                    source_dtype: ColumnType::String,
                    ingested_dtype: ColumnType::String,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].dst_table, "poe");
    }

    #[test]
    fn infer_fk_exact_pk_name_match() {
        let tables = vec![
            TableSpec {
                name: "admits".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "hadm_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("hadm_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "services".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "hadm_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].dst_table, "admits");
    }

    #[test]
    fn infer_fk_itemid_prefers_lab_table() {
        let tables = vec![
            TableSpec {
                name: "d_items".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("itemid".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "d_lab".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("itemid".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "labevents".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "itemid".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    timestamp_format: None,
                    timestamp_unit: None,
                    fixed_len: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: None,
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].dst_table, "d_lab");
    }

    #[test]
    fn infer_pk_pk_relationship_when_table_names_overlap() {
        let tables = vec![
            TableSpec {
                name: "user".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "user_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("user_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "user_profile".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "user_id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("user_id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        infer_primary_key_relationships(&tables, &mut rels);
        assert_eq!(rels.len(), 1);
        assert!(rels[0].src_table == "user" || rels[0].src_table == "user_profile");
    }

    #[test]
    fn infer_pk_pk_skips_generic_id() {
        let tables = vec![
            TableSpec {
                name: "alpha".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
            TableSpec {
                name: "beta".to_string(),
                format: TableFormat::Csv,
                paths: vec![],
                columns: vec![ColumnSpec {
                    name: "id".to_string(),
                    source_dtype: ColumnType::Int64,
                    ingested_dtype: ColumnType::Int64,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    key_mode: None,
                    precision: None,
                    scale: None,
                }],
                primary_key: Some("id".to_string()),
                create_time: None,
                weight: None,
                edge_table: None,
                rowpack: None,
            },
        ];
        let mut rels = Vec::new();
        let mut notes = Vec::new();
        infer_foreign_keys(&tables, &mut rels, &mut notes);
        infer_primary_key_relationships(&tables, &mut rels);
        assert!(rels.is_empty());
    }

    #[test]
    fn infer_create_time_from_timestamp_col() {
        let mut t = table(
            "events",
            &[("created_at", ColumnType::Timestamp), ("id", ColumnType::Int64)],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("created_at"));
    }

    #[test]
    fn csv_infers_timestamp_from_name_and_ints() {
        let mut stats = ColumnStats::default();
        for v in ["1693526400", "1693612800", "1693699200"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("created_at");
        assert!(matches!(col.ingested_dtype, ColumnType::Timestamp));
    }

    #[test]
    fn csv_rejects_timestamp_out_of_range() {
        let mut stats = ColumnStats::default();
        for v in ["12", "13", "14"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("created_at");
        assert!(matches!(col.ingested_dtype, ColumnType::Int64));
    }

    #[test]
    fn csv_infers_date_from_iso() {
        let mut stats = ColumnStats::default();
        for v in ["2024-01-01", "2024-01-02"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("event_date");
        assert!(matches!(col.ingested_dtype, ColumnType::Date));
    }

    #[test]
    fn csv_infers_bool() {
        let mut stats = ColumnStats::default();
        for v in ["true", "false", "true"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("flag");
        assert!(matches!(col.ingested_dtype, ColumnType::Bool));
    }

    #[test]
    fn csv_infers_timestamp_from_common_format() {
        let mut stats = ColumnStats::default();
        for v in ["2024-01-01 12:34:56", "2024-01-01 12:35:56"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("charttime");
        assert!(matches!(col.ingested_dtype, ColumnType::Timestamp));
    }

    #[test]
    fn csv_infers_timestamp_from_fractional_seconds() {
        let mut stats = ColumnStats::default();
        for v in ["2024-01-01 12:34:56.123", "2024-01-01 12:35:56.987"] {
            stats.observe(v);
        }
        let col = stats.to_column_spec("storetime");
        assert!(matches!(col.ingested_dtype, ColumnType::Timestamp));
    }

    #[test]
    fn infer_create_time_prefers_start_over_end() {
        let mut t = table(
            "events",
            &[
                ("start_time", ColumnType::Timestamp),
                ("end_time", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("start_time"));
    }

    #[test]
    fn infer_create_time_skips_negative_only() {
        let mut t = table(
            "events",
            &[
                ("end_time", ColumnType::Timestamp),
                ("stop_time", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time, None);
    }

    #[test]
    fn infer_create_time_prefers_created_over_updated() {
        let mut t = table(
            "accounts",
            &[
                ("created_at", ColumnType::Timestamp),
                ("updated_at", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("created_at"));
    }

    #[test]
    fn infer_create_time_ignores_negative_token() {
        let mut t = table(
            "orders",
            &[
                ("authorized_time", ColumnType::Timestamp),
                ("voided_time", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("authorized_time"));
    }

    #[test]
    fn infer_create_time_prefers_positive_when_present() {
        let mut t = table(
            "shipping",
            &[
                ("shipped_at", ColumnType::Timestamp),
                ("returned_at", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("shipped_at"));
    }

    #[test]
    fn infer_create_time_prefers_chart_over_store() {
        let mut t = table(
            "chart",
            &[
                ("charttime", ColumnType::Timestamp),
                ("storetime", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("charttime"));
    }

    #[test]
    fn infer_create_time_prefers_start_over_store() {
        let mut t = table(
            "inputevents",
            &[
                ("starttime", ColumnType::Timestamp),
                ("storetime", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("starttime"));
    }

    #[test]
    fn infer_create_time_prefers_in_over_out() {
        let mut t = table(
            "transfers",
            &[
                ("intime", ColumnType::Timestamp),
                ("outtime", ColumnType::Timestamp),
            ],
        );
        infer_create_time(std::slice::from_mut(&mut t));
        assert_eq!(t.create_time.as_deref(), Some("intime"));
    }

    #[test]
    fn compatible_numeric_to_var_string() {
        let a = ColumnSpec {
            name: "user_id".to_string(),
            source_dtype: ColumnType::Int64,
            ingested_dtype: ColumnType::Int64,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            key_mode: None,
            precision: None,
            scale: None,
        };
        let b = ColumnSpec {
            name: "id".to_string(),
            source_dtype: ColumnType::String,
            ingested_dtype: ColumnType::String,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            key_mode: None,
            precision: None,
            scale: None,
        };
        assert!(compatible_key_columns(&a, &b));
    }

    #[test]
    fn incompatible_fixed_vs_var() {
        let a = ColumnSpec {
            name: "device_id".to_string(),
            source_dtype: ColumnType::Bytes,
            ingested_dtype: ColumnType::Bytes,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: Some(16),
            key_mode: Some(KeyMode::Fixed),
            precision: None,
            scale: None,
        };
        let b = ColumnSpec {
            name: "id".to_string(),
            source_dtype: ColumnType::Bytes,
            ingested_dtype: ColumnType::Bytes,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            key_mode: None,
            precision: None,
            scale: None,
        };
        assert!(!compatible_key_columns(&a, &b));
    }

    #[test]
    fn compatible_uuid() {
        let a = ColumnSpec {
            name: "id".to_string(),
            source_dtype: ColumnType::String,
            ingested_dtype: ColumnType::String,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: Some(16),
            key_mode: Some(KeyMode::Uuid),
            precision: None,
            scale: None,
        };
        let b = ColumnSpec {
            name: "id".to_string(),
            source_dtype: ColumnType::Bytes,
            ingested_dtype: ColumnType::Bytes,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: Some(16),
            key_mode: Some(KeyMode::Uuid),
            precision: None,
            scale: None,
        };
        assert!(compatible_key_columns(&a, &b));
    }
}
