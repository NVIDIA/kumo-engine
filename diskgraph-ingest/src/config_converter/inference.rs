use anyhow::Result;
use arrow_array::RecordBatchReader;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::file::reader::ChunkReader;
use std::collections::{HashMap, HashSet};

use crate::graph_config::{GraphConfig, IngestedType, TableFormat, TimestampUnit};
use crate::io::{build_input_transfer_for_table, download_cloud_bytes_condition, is_cloud_path};
use diskgraph_fs::{CloudObjectCondition, CloudTransfer};

/// Detection result for a single timestamp column.
enum TimestampDetection {
    /// String format detected (e.g., "%Y-%m-%d %H:%M:%S").
    Format(String),
    /// Epoch unit detected from integer magnitude.
    Unit(TimestampUnit),
}

fn build_parquet_sample_reader<T, E>(
    source: T,
    path: &str,
    map_build_error: impl FnOnce(parquet::errors::ParquetError) -> E,
) -> std::result::Result<Option<ParquetRecordBatchReader>, E>
where
    T: ChunkReader + 'static,
{
    let builder = match ParquetRecordBatchReaderBuilder::try_new(source) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!("Skipping {}: failed to read parquet: {error}", path);
            return Ok(None);
        }
    };
    let reader = builder
        .with_batch_size(1024)
        .build()
        .map_err(map_build_error)?;
    Ok(Some(reader))
}

fn open_parquet_sample_reader<E>(
    transfer: Option<&CloudTransfer>,
    path: &str,
    condition: Option<CloudObjectCondition>,
    map_build_error: impl FnOnce(parquet::errors::ParquetError) -> E,
) -> std::result::Result<Option<ParquetRecordBatchReader>, E> {
    if let Some(transfer) = transfer.filter(|_| is_cloud_path(path)) {
        let pooled = match download_cloud_bytes_condition(transfer, path, condition) {
            Ok(pooled) => pooled,
            Err(error) => {
                tracing::warn!("Skipping {}: download failed: {error}", path);
                return Ok(None);
            }
        };
        return build_parquet_sample_reader(pooled.bytes(), path, map_build_error);
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!("Skipping {}: failed to open: {error}", path);
            return Ok(None);
        }
    };
    build_parquet_sample_reader(file, path, map_build_error)
}

/// Detect timestamp format strings and epoch units for timestamp/date columns.
///
/// For each table in the config, finds columns with `ingested_dtype ∈ {Timestamp, Date}`
/// that are missing both `timestamp_format` and `timestamp_unit`. Iterates through the
/// table's parquet files, reading one batch (1024 rows) from each, until all candidate
/// columns have been resolved or files are exhausted. For each candidate column:
///
/// 1. **String columns**: tries `detect_timestamp_format` on the first non-null value.
///    If no format matches but the string parses as an integer, falls through to (3).
/// 2. **Integer columns**: falls through to (3).
/// 3. **Epoch unit heuristic**: interprets the integer value's magnitude to determine
///    whether it represents seconds, milliseconds, microseconds, or nanoseconds.
///
/// Files that are empty (0 rows) or where a candidate column is entirely null are
/// automatically skipped, and detection continues with the next file.
///
/// Modifies `config` in place, setting `col.timestamp_format` or `col.timestamp_unit`.
pub fn detect_timestamp_formats(
    config: &mut GraphConfig,
    input_transfer: &Option<CloudTransfer>,
) -> Result<()> {
    for table in &mut config.tables {
        // Find candidate columns: Timestamp or Date with no format and no unit set yet.
        let candidates: Vec<(usize, String)> = table
            .columns
            .iter()
            .enumerate()
            .filter(|(_, col)| {
                matches!(
                    col.ingested_dtype,
                    IngestedType::Timestamp | IngestedType::Date
                ) && col.timestamp_format.is_none()
                    && col.timestamp_unit.is_none()
            })
            .map(|(i, col)| (i, col.name.clone()))
            .collect();

        if candidates.is_empty() {
            continue;
        }

        if table.format() != TableFormat::Parquet {
            continue;
        }

        let owned_transfer = if input_transfer.is_none() {
            build_input_transfer_for_table(table)?
        } else {
            None
        };
        let table_transfer = input_transfer.as_ref().or(owned_transfer.as_ref());
        let parquet_files = table.source_objects();
        let parquet_file_count = parquet_files.len().min(MAX_FILES);
        if parquet_file_count == 0 {
            continue;
        }

        // Accumulated state across files.
        let mut votes: HashMap<usize, Vec<String>> = HashMap::new();
        let mut sample_counts: HashMap<usize, usize> = HashMap::new();
        // Columns whose Arrow type is a native temporal type (Date32, Date64,
        // Timestamp, etc.) — rowpack handles these without format/unit hints.
        let mut native_temporal: HashSet<usize> = HashSet::new();
        // Info about non-null values we couldn't classify (for warnings).
        let mut unrecognized_info: HashMap<usize, String> = HashMap::new();

        /// Maximum number of non-null values to sample per column for majority vote.
        const MAX_SAMPLES: usize = 10;
        /// Maximum number of files to download when searching for non-null values.
        const MAX_FILES: usize = 15;

        macro_rules! sample_from_reader {
            ($reader:expr, $schema:expr) => {{
                // Resolve column indices and their arrow data types.
                let col_info: Vec<Option<(usize, arrow_schema::DataType)>> = candidates
                    .iter()
                    .map(|(_, name)| {
                        $schema
                            .index_of(name)
                            .ok()
                            .map(|idx| (idx, $schema.field(idx).data_type().clone()))
                    })
                    .collect();

                for batch in $reader {
                    let batch = batch?;
                    for (cand_idx, info) in col_info.iter().enumerate() {
                        let (arr_idx, dt) = match info {
                            Some((i, dt)) => (*i, dt),
                            None => continue,
                        };
                        let col_key = candidates[cand_idx].0;
                        if native_temporal.contains(&col_key) {
                            continue;
                        }
                        if *sample_counts.get(&col_key).unwrap_or(&0) >= MAX_SAMPLES {
                            continue;
                        }
                        let array = batch.column(arr_idx);
                        for row in 0..batch.num_rows() {
                            if *sample_counts.get(&col_key).unwrap_or(&0) >= MAX_SAMPLES {
                                break;
                            }
                            if array.is_null(row) {
                                continue;
                            }
                            *sample_counts.entry(col_key).or_insert(0) += 1;
                            match dt {
                                // Native Arrow temporal types — rowpack handles
                                // these directly from the Arrow type, no hint needed.
                                arrow_schema::DataType::Date32
                                | arrow_schema::DataType::Date64
                                | arrow_schema::DataType::Timestamp(_, _)
                                | arrow_schema::DataType::Time32(_)
                                | arrow_schema::DataType::Time64(_) => {
                                    native_temporal.insert(col_key);
                                    break; // done with this column
                                }
                                arrow_schema::DataType::Utf8
                                | arrow_schema::DataType::LargeUtf8
                                | arrow_schema::DataType::Binary => {
                                    let text: Option<&str> = if let Some(s) =
                                        array.as_any().downcast_ref::<arrow_array::StringArray>()
                                    {
                                        Some(s.value(row))
                                    } else if let Some(s) = array
                                        .as_any()
                                        .downcast_ref::<arrow_array::LargeStringArray>(
                                    ) {
                                        Some(s.value(row))
                                    } else if let Some(s) =
                                        array.as_any().downcast_ref::<arrow_array::BinaryArray>()
                                    {
                                        std::str::from_utf8(s.value(row)).ok()
                                    } else {
                                        None
                                    };
                                    if let Some(text) = text {
                                        if let Some(fmt) =
                                            diskgraph_rowpack::detect_timestamp_format(text)
                                        {
                                            votes
                                                .entry(col_key)
                                                .or_default()
                                                .push(format!("fmt:{}", fmt));
                                        } else if let Ok(v) = text.trim().parse::<i64>() {
                                            if let Some(det) = detect_int_timestamp(v) {
                                                votes
                                                    .entry(col_key)
                                                    .or_default()
                                                    .push(detection_to_vote_key(&det));
                                            }
                                        } else {
                                            // Non-null string, not a recognized timestamp
                                            // format, not parseable as integer.
                                            let sample =
                                                if text.len() > 50 { &text[..50] } else { text };
                                            unrecognized_info.entry(col_key).or_insert_with(|| {
                                                format!(
                                                    "unrecognized string format, sample: \"{}\"",
                                                    sample
                                                )
                                            });
                                        }
                                    }
                                }
                                arrow_schema::DataType::Int8
                                | arrow_schema::DataType::Int16
                                | arrow_schema::DataType::Int32
                                | arrow_schema::DataType::Int64
                                | arrow_schema::DataType::UInt8
                                | arrow_schema::DataType::UInt16
                                | arrow_schema::DataType::UInt32
                                | arrow_schema::DataType::UInt64 => {
                                    if let Some(v) = read_int_value(array, row) {
                                        if let Some(det) = detect_int_timestamp(v) {
                                            votes
                                                .entry(col_key)
                                                .or_default()
                                                .push(detection_to_vote_key(&det));
                                        }
                                    }
                                }
                                _ => {
                                    unrecognized_info.entry(col_key).or_insert_with(|| {
                                        format!("unsupported arrow type: {:?}", dt)
                                    });
                                }
                            }
                        }
                    }
                    // One batch (1024 rows) is enough per file.
                    break;
                }
            }};
        }

        // Iterate through files until all candidates are resolved.
        // A candidate is resolved when it has: a vote, a native temporal type,
        // or a non-null value we couldn't classify (more files won't help).
        // Cap at MAX_FILES to avoid downloading hundreds of files for all-null columns.
        for object in parquet_files.take(MAX_FILES) {
            let path = object.path;
            let Some(reader) = open_parquet_sample_reader(
                table_transfer,
                path,
                object.condition(),
                anyhow::Error::from,
            )?
            else {
                continue;
            };
            let schema = reader.schema();
            sample_from_reader!(reader, schema);

            let all_resolved = candidates.iter().all(|(col_idx, _)| {
                votes.contains_key(col_idx)
                    || native_temporal.contains(col_idx)
                    || sample_counts.get(col_idx).copied().unwrap_or(0) > 0
            });
            if all_resolved {
                break;
            }
        }

        // Emit warnings (or errors for time columns) for columns we couldn't classify.
        for (col_idx, col_name) in &candidates {
            if votes.contains_key(col_idx) || native_temporal.contains(col_idx) {
                continue;
            }
            let is_create_time_col = table.create_time.as_ref() == Some(col_name);
            let is_end_time_col = table.end_time.as_ref() == Some(col_name);
            let is_time_col = is_create_time_col || is_end_time_col;

            if let Some(info) = unrecognized_info.get(col_idx) {
                if is_time_col {
                    let role = if table.create_time.as_ref() == Some(col_name) {
                        "create_time"
                    } else {
                        "end_time"
                    };
                    tracing::warn!(
                        table = %table.name, column = %col_name, role = role,
                        "could not detect timestamp format: {}. Rows with unparseable \
                         time values will be treated as invalid during ingest.",
                        info,
                    );
                    continue;
                }
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    "could not detect timestamp format: {}", info,
                );
            } else if sample_counts.get(col_idx).copied().unwrap_or(0) == 0 {
                if is_create_time_col {
                    return Err(anyhow::anyhow!(
                        "table '{}' column '{}' (create_time): column is entirely null across {} file(s). \
                         A time column must contain parseable timestamp values.",
                        table.name,
                        col_name,
                        parquet_file_count,
                    ));
                }
                if is_end_time_col {
                    tracing::warn!(
                        table = %table.name, column = %col_name, role = "end_time",
                        "column is entirely null across {} file(s); treating null end times as open-ended intervals",
                        parquet_file_count,
                    );
                    continue;
                }
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    "column is entirely null across {} file(s)",
                    parquet_file_count,
                );
            }
        }

        // Majority vote across accumulated samples, then apply to config.
        let mut detected: HashMap<usize, TimestampDetection> = HashMap::new();
        for (col_key, col_votes) in &votes {
            if let Some(winner) = majority_vote(col_votes) {
                detected.insert(*col_key, winner);
            }
        }
        for (col_idx, detection) in detected {
            match detection {
                TimestampDetection::Format(fmt) => {
                    table.columns[col_idx].timestamp_format = Some(fmt);
                }
                TimestampDetection::Unit(unit) => {
                    table.columns[col_idx].timestamp_unit = Some(unit);
                }
            }
        }
    }
    Ok(())
}

/// Detect timestamp representation from an integer value.
///
/// First checks for encoded date patterns (YYYYMMDD, YYYYMMDDHHMMSS, etc.)
/// via `detect_encoded_int_format`. Only falls back to epoch-magnitude
/// heuristic if no encoded date pattern matches.
///
/// Epoch heuristic ranges (covers ~1970 to ~2286):
/// - `|v| < 1e10`  → Seconds
/// - `|v| < 1e13`  → Milliseconds
/// - `|v| < 1e16`  → Microseconds
/// - `|v| >= 1e16` → Nanoseconds
fn detect_int_timestamp(value: i64) -> Option<TimestampDetection> {
    // Check encoded date formats first (e.g. 20230408 → "%Y%m%d")
    if let Some(fmt) = diskgraph_rowpack::detect_encoded_int_format(value) {
        return Some(TimestampDetection::Format(fmt.to_string()));
    }
    // Fall back to epoch unit detection
    detect_epoch_unit(value).map(TimestampDetection::Unit)
}

/// Detect epoch unit from integer magnitude. Delegates to the shared
/// `TimestampUnit::from_epoch_magnitude`.
fn detect_epoch_unit(value: i64) -> Option<TimestampUnit> {
    TimestampUnit::from_epoch_magnitude(value)
}

/// Convert a `TimestampDetection` to a string key for majority voting.
fn detection_to_vote_key(det: &TimestampDetection) -> String {
    match det {
        TimestampDetection::Format(f) => format!("fmt:{}", f),
        TimestampDetection::Unit(u) => format!("unit:{:?}", u),
    }
}

/// Parse a vote key back into a `TimestampDetection`.
fn vote_key_to_detection(key: &str) -> Option<TimestampDetection> {
    if let Some(fmt) = key.strip_prefix("fmt:") {
        Some(TimestampDetection::Format(fmt.to_string()))
    } else if let Some(unit_str) = key.strip_prefix("unit:") {
        let unit = match unit_str {
            "Seconds" => TimestampUnit::Seconds,
            "Millis" => TimestampUnit::Millis,
            "Micros" => TimestampUnit::Micros,
            "Nanos" => TimestampUnit::Nanos,
            _ => return None,
        };
        Some(TimestampDetection::Unit(unit))
    } else {
        None
    }
}

/// Pick the most common vote. On ties, prefer Format over Unit.
fn majority_vote(votes: &[String]) -> Option<TimestampDetection> {
    if votes.is_empty() {
        return None;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in votes {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let max_count = *counts.values().max().unwrap();
    // Collect all keys with the max count
    let mut best: Vec<&str> = counts
        .iter()
        .filter(|(_, &c)| c == max_count)
        .map(|(&k, _)| k)
        .collect();
    // On ties, prefer Format over Unit
    best.sort_by(|a, b| {
        let a_is_fmt = a.starts_with("fmt:");
        let b_is_fmt = b.starts_with("fmt:");
        b_is_fmt.cmp(&a_is_fmt) // true (fmt) sorts before false (unit)
    });
    vote_key_to_detection(best[0])
}

/// Read an integer value from an arrow array at a given row.
fn read_int_value(array: &dyn arrow_array::Array, row: usize) -> Option<i64> {
    use arrow_array::*;
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        Some(a.value(row))
    } else if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        Some(a.value(row) as i64)
    } else if let Some(a) = array.as_any().downcast_ref::<Int16Array>() {
        Some(a.value(row) as i64)
    } else if let Some(a) = array.as_any().downcast_ref::<Int8Array>() {
        Some(a.value(row) as i64)
    } else if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        Some(a.value(row) as i64)
    } else if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        Some(a.value(row) as i64)
    } else if let Some(a) = array.as_any().downcast_ref::<UInt16Array>() {
        Some(a.value(row) as i64)
    } else {
        array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| a.value(row) as i64)
    }
}

/// Compute total bytes across all table paths in a GraphConfig.
///
/// For local paths, walks directories and sums file sizes.
/// For cloud paths, lists objects under the prefix and sums sizes.
pub fn compute_total_bytes(
    config: &GraphConfig,
    input_transfer: &Option<CloudTransfer>,
) -> Result<u64> {
    let mut total: u64 = 0;
    for table in &config.tables {
        for path in table.source_paths() {
            if is_cloud_path(path) {
                if let Some(transfer) = input_transfer {
                    total += compute_cloud_dir_bytes(transfer, path)?;
                }
            } else {
                total += compute_local_dir_bytes(path);
            }
        }
    }
    Ok(total)
}

/// Sum file sizes under a local directory (recursing into subdirectories) or a single file.
///
/// Partitioned datasets nest their parquet files under subdirectories (e.g.
/// `dt=.../part-*.parquet`), so a non-recursive sum would measure near-0 and under-shard.
pub(crate) fn compute_local_dir_bytes(dir_path: &str) -> u64 {
    let dir_path = dir_path.trim_start_matches("file://");
    let dir_path = dir_path.trim_end_matches('/');
    match std::fs::metadata(dir_path) {
        Ok(meta) if meta.is_file() => meta.len(),
        Ok(_) => sum_dir_bytes_recursive(std::path::Path::new(dir_path)),
        Err(_) => 0,
    }
}

/// Recursively sum the sizes of all files under `path`.
fn sum_dir_bytes_recursive(path: &std::path::Path) -> u64 {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_dir() {
                total = total.saturating_add(sum_dir_bytes_recursive(&entry.path()));
            } else {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

/// Sum object sizes under a cloud prefix.
fn compute_cloud_dir_bytes(transfer: &CloudTransfer, path: &str) -> Result<u64> {
    let prefix = crate::io::object_path_from_cloud_url(path)?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let objects = rt.block_on(transfer.list_objects(&prefix))?;
    Ok(objects.iter().map(|o| o.size).sum())
}

/// Estimate the optimal number of shards based on total dataset size.
///
/// This function attempts to balance two competing goals:
/// 1. **Parallelism**: More shards allow for higher concurrency during ingestion and training.
/// 2. **Overhead**: Too many shards increase metadata overhead and file system fragmentation.
///
/// The estimation logic adapts to the *system's* capabilities (available RAM and CPU cores)
/// as well as the *dataset's* size. It uses a "smooth" scaling function to determine
/// the target shard size.
pub fn estimate_num_shards(total_bytes: u64) -> usize {
    if total_bytes == 0 {
        return 1;
    }
    let shard_bytes = shard_target_bytes(total_bytes);
    total_bytes.div_ceil(shard_bytes) as usize
}

/// Calculate the target size (in bytes) for a single shard.
///
/// If system memory or CPU count cannot be determined, it falls back to a safe default (1 GB).
///
/// Memory is cgroup-aware. Shard sizing deliberately uses CPU *capacity*
/// (the affinity/cpuset mask) rather than the CFS ``cpu.max`` quota. A lower
/// CPU bandwidth quota must reduce worker pools, but it must not create
/// fewer/larger shards and weaken the memory-safety floor. When there is no
/// cgroup limit these reduce to the host values.
fn shard_target_bytes(total_bytes: u64) -> u64 {
    const FALLBACK_SHARD_BYTES: u64 = 1024 * 1024 * 1024; // 1 GB
                                                          // total_memory_bytes() = min(host MemTotal, memory.max on this process's
                                                          // own cgroup path). cpu_capacity_parallelism() honors CPU affinity/cpuset
                                                          // while intentionally ignoring cpu.max bandwidth quota for shard sizing.
    let mem_total = crate::total_memory_bytes().or_else(read_system_memory_bytes);
    let capacity_parallelism = crate::cpu_capacity_parallelism() as u64;
    match mem_total {
        Some(mem_total) => {
            shard_target_bytes_for_capacity(mem_total, capacity_parallelism, total_bytes)
        }
        None => FALLBACK_SHARD_BYTES,
    }
}

fn shard_target_bytes_for_capacity(
    mem_total: u64,
    capacity_parallelism: u64,
    total_bytes: u64,
) -> u64 {
    const FALLBACK_SHARD_BYTES: u64 = 1024 * 1024 * 1024;
    if capacity_parallelism == 0 {
        return FALLBACK_SHARD_BYTES;
    }
    let target = smooth_shard_target(mem_total / capacity_parallelism, total_bytes);
    if target == 0 {
        FALLBACK_SHARD_BYTES
    } else {
        target
    }
}

/// Calculate the target shard size using a smooth interpolation function.
///
/// The goal is to determine what fraction (or multiple) of per-CPU RAM should be used
/// for a single shard.
///
/// Logic:
/// - **Small Datasets (< 40 GB)**: We want *small* shards to ensure we utilize all cores.
///   Divisor ranges from 16.0 down to 4.0.
///   Target Shard Size: `(RAM / 16)` to `(RAM / 4)`.
///
/// - **Medium Datasets (40 GB - 1 TB)**: We transition to "standard" sizing.
///   Divisor ranges from 4.0 down to 1.0.
///   Target Shard Size: `(RAM / 4)` to `RAM`.
///
/// - **Large Datasets (1 TB - 10 TB)**: We increase shard size to reduce metadata overhead.
///   Divisor ranges from 1.0 down to 0.5.
///   Target Shard Size: `RAM` to `2 * RAM`.
///
/// - **Huge Datasets (> 10 TB)**: Constant factor.
///   Divisor is 0.75.
///   Target Shard Size: `1.33 * RAM`.
fn smooth_shard_target(per_cpu_bytes: u64, total_bytes: u64) -> u64 {
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let divisor = if total_gb <= 40.0 {
        // Range: 10GB - 40GB
        // Smooth interpolation t: 0.0 -> 1.0
        let t = ((total_gb - 10.0) / 30.0).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t); // Smoothstep function
                                              // Divisor: 16.0 -> 4.0
        16.0 - 12.0 * smooth
    } else if total_gb <= 1024.0 {
        // Range: 40GB - 1TB
        let t = ((total_gb - 40.0) / (1024.0 - 40.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        // Divisor: 4.0 -> 1.0
        4.0 - 3.0 * smooth
    } else if total_gb <= 10_240.0 {
        // Range: 1TB - 10TB
        let t = ((total_gb - 1024.0) / (10_240.0 - 1024.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        // Divisor: 1.0 -> 0.5
        1.0 - 0.5 * smooth
    } else {
        // > 10TB
        0.75
    };
    (per_cpu_bytes as f64 / divisor).round() as u64
}

/// Read total system memory from `/proc/meminfo` (Linux only).
fn read_system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(value) = line.strip_prefix("MemTotal:") {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] == "kB" {
                    return parts[0].parse::<u64>().ok().map(|kb| kb * 1024);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Multicategorical delimiter detection
// ---------------------------------------------------------------------------

use crate::errors::ConvertError;
use crate::graph_config::SemanticType;
use arrow_array::Array;

/// Candidate delimiters tried for `String + Multicategorical` columns.
/// Order matters for deterministic reporting; the detector does not prefer
/// earlier candidates, only uses the list for enumeration.
const MULTICAT_DELIMITER_CANDIDATES: &[&str] = &[",", "|", ";", "\t"];

/// Minimum sample support for auto-detecting a delimiter. Below this floor,
/// punctuation is treated as noise and the column stays single-token-per-row.
const MULTICAT_DETECT_MIN_FRACTION: f64 = 0.01;
/// Suspicion threshold for explicitly-declared delimiters. Explicit configs
/// are accepted if the delimiter appears at least once, but hit rates below
/// this threshold get a warning.
const MULTICAT_EXPLICIT_SUSPICIOUS_FRACTION: f64 = 0.20;
/// Max number of non-null sample rows to pull per column.
const MULTICAT_MAX_SAMPLES: usize = 1_000;
/// Max number of files to inspect while looking for multicat delimiter samples.
const MULTICAT_MAX_FILES: usize = 15;

fn multicat_string_value(array: &dyn Array, row: usize) -> Option<&str> {
    if let Some(str_arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
        Some(str_arr.value(row))
    } else if let Some(str_arr) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        Some(str_arr.value(row))
    } else {
        None
    }
}

struct ExplicitMulticatCandidate {
    name: String,
    declared: String,
    total: usize,
    hits: usize,
    miss_sample: Option<String>,
}

/// Bounded, transport-independent validation folded into the normal Encode
/// pass. Once each candidate has 1,000 non-empty values, observation is a
/// no-op, so this never grows with result size.
pub(crate) struct ExplicitMulticatDelimiterValidator {
    table: String,
    candidates: Vec<ExplicitMulticatCandidate>,
    capture_samples: bool,
}

impl ExplicitMulticatDelimiterValidator {
    pub(crate) fn new(table: &crate::graph_config::TableSpec) -> Self {
        Self::with_sample_capture(table, false)
    }

    fn with_sample_capture(table: &crate::graph_config::TableSpec, capture_samples: bool) -> Self {
        let candidates = table
            .columns
            .iter()
            .filter_map(|column| {
                let scalar_string_source = matches!(
                    column.ingested_dtype,
                    IngestedType::String | IngestedType::StringArray | IngestedType::Int64Array,
                );
                (scalar_string_source
                    && column.semantic_type == Some(SemanticType::Multicategorical))
                .then(|| {
                    column
                        .multicategorical_delimiter
                        .as_deref()
                        .map(|declared| ExplicitMulticatCandidate {
                            name: column.name.clone(),
                            declared: declared.to_string(),
                            total: 0,
                            hits: 0,
                            miss_sample: None,
                        })
                })
                .flatten()
            })
            .collect();
        Self {
            table: table.name.clone(),
            candidates,
            capture_samples,
        }
    }

    fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    fn is_complete(&self) -> bool {
        self.candidates
            .iter()
            .all(|candidate| candidate.total >= MULTICAT_MAX_SAMPLES)
    }

    pub(crate) fn observe(&mut self, batch: &arrow_array::RecordBatch) {
        for candidate in &mut self.candidates {
            if candidate.total >= MULTICAT_MAX_SAMPLES {
                continue;
            }
            let Ok(array_index) = batch.schema().index_of(&candidate.name) else {
                continue;
            };
            let array = batch.column(array_index);
            for row in 0..batch.num_rows() {
                if candidate.total >= MULTICAT_MAX_SAMPLES {
                    break;
                }
                if array.is_null(row) {
                    continue;
                }
                let Some(text) = multicat_string_value(array.as_ref(), row) else {
                    continue;
                };
                if text.is_empty() {
                    continue;
                }
                candidate.total += 1;
                if text.contains(&candidate.declared) {
                    candidate.hits += 1;
                } else if self.capture_samples && candidate.miss_sample.is_none() {
                    candidate.miss_sample = Some(text.to_string());
                }
            }
        }
    }

    pub(crate) fn finish(self) -> Result<(), ConvertError> {
        for candidate in self.candidates {
            if candidate.total == 0 {
                tracing::warn!(
                    table = %self.table,
                    column = %candidate.name,
                    "explicit multicategorical_delimiter could not be validated: \
                     no non-empty sample rows in the bounded validation pass",
                );
                continue;
            }
            if candidate.hits == 0 {
                return Err(ConvertError::MulticategoricalDelimiterMismatch {
                    table: self.table.clone(),
                    column: candidate.name,
                    declared: candidate.declared,
                    hits: 0,
                    total: candidate.total,
                    required: 1,
                    sample: candidate
                        .miss_sample
                        .unwrap_or_else(|| "<omitted>".to_string()),
                });
            }
            let suspicious_floor =
                (MULTICAT_EXPLICIT_SUSPICIOUS_FRACTION * candidate.total as f64).ceil() as usize;
            if candidate.hits < suspicious_floor {
                tracing::warn!(
                    table = %self.table,
                    column = %candidate.name,
                    declared = %candidate.declared,
                    hits = candidate.hits,
                    total = candidate.total,
                    "explicit multicategorical_delimiter appears in fewer rows than expected",
                );
            }
        }
        Ok(())
    }
}

/// Detect the per-column `multicategorical_delimiter` for every
/// `String + Multicategorical` column that didn't come with an explicit
/// value on the upstream Kumo config.
///
/// Decision rule:
/// - Pick the delimiter candidate that appears in the most sampled rows, as
///   long as it appears in at least ceil(1% of sampled non-null rows).
/// - No candidate reaches the floor → `delimiter = None` (legitimate
///   single-token-per-row multicat).
///
/// Ties are resolved by `MULTICAT_DELIMITER_CANDIDATES` order.
///
/// All-null columns fall through with a warning (nothing to detect against);
/// ingest will treat the column as single-token-per-row.
pub fn detect_multicategorical_delimiters(
    config: &mut GraphConfig,
    input_transfer: &Option<CloudTransfer>,
) -> Result<(), ConvertError> {
    for table in &mut config.tables {
        // Candidate columns: String + Multicategorical, no explicit delimiter.
        let candidates: Vec<(usize, String)> = table
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.ingested_dtype == IngestedType::String
                    && c.semantic_type == Some(SemanticType::Multicategorical)
                    && c.multicategorical_delimiter.is_none()
            })
            .map(|(i, c)| (i, c.name.clone()))
            .collect();

        if candidates.is_empty() {
            continue;
        }
        if table.format() != TableFormat::Parquet {
            continue;
        }
        let owned_transfer = if input_transfer.is_none() {
            build_input_transfer_for_table(table)
                .map_err(|error| ConvertError::InferenceError(format!("{error:#}")))?
        } else {
            None
        };
        let table_transfer = input_transfer.as_ref().or(owned_transfer.as_ref());
        let parquet_files = table.source_objects();
        if parquet_files.len() == 0 {
            continue;
        }

        // Per-column per-candidate occurrence counts, and total non-null
        // sample rows for that column.
        let mut per_col_totals: HashMap<usize, usize> = HashMap::new();
        let mut per_col_hits: HashMap<usize, HashMap<&'static str, usize>> = HashMap::new();

        // Sample multiple parquet files. Early files can be empty, all-null,
        // or non-representative partitions, so continue until each candidate
        // has enough non-null samples or the bounded file budget is exhausted.
        for object in parquet_files.take(MULTICAT_MAX_FILES) {
            let path = object.path;

            macro_rules! scan_reader {
                ($reader:expr, $schema:expr) => {{
                    let col_info: Vec<Option<usize>> = candidates
                        .iter()
                        .map(|(_, name)| $schema.index_of(name).ok())
                        .collect();
                    for batch in $reader {
                        let batch =
                            batch.map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                        for (cand_idx, info) in col_info.iter().enumerate() {
                            let Some(arr_idx) = info else { continue };
                            let col_key = candidates[cand_idx].0;
                            if *per_col_totals.get(&col_key).unwrap_or(&0) >= MULTICAT_MAX_SAMPLES {
                                continue;
                            }
                            let array = batch.column(*arr_idx);
                            for row in 0..batch.num_rows() {
                                if *per_col_totals.get(&col_key).unwrap_or(&0)
                                    >= MULTICAT_MAX_SAMPLES
                                {
                                    break;
                                }
                                if array.is_null(row) {
                                    continue;
                                }
                                let Some(text) = multicat_string_value(array.as_ref(), row) else {
                                    continue;
                                };
                                *per_col_totals.entry(col_key).or_insert(0) += 1;
                                let hits = per_col_hits.entry(col_key).or_default();
                                for cand in MULTICAT_DELIMITER_CANDIDATES {
                                    if text.contains(cand) {
                                        *hits.entry(*cand).or_insert(0) += 1;
                                    }
                                }
                            }
                        }
                    }
                }};
            }

            let Some(reader) =
                open_parquet_sample_reader(table_transfer, path, object.condition(), |error| {
                    ConvertError::InferenceError(format!("{error:#}"))
                })?
            else {
                continue;
            };
            let schema = reader.schema();
            scan_reader!(reader, schema);

            // Stop as soon as we have enough samples for every candidate.
            let all_done = candidates.iter().all(|(col_idx, _)| {
                *per_col_totals.get(col_idx).unwrap_or(&0) >= MULTICAT_MAX_SAMPLES
            });
            if all_done {
                break;
            }
        }

        // Decision per candidate column.
        for (col_idx, col_name) in &candidates {
            let total = *per_col_totals.get(col_idx).unwrap_or(&0);
            let hits = per_col_hits.remove(col_idx).unwrap_or_default();
            let decision = decide_multicat_delimiter(total, &hits);
            match decision {
                MulticatDelimiterDecision::NoDelimiter => {
                    // Leave `multicategorical_delimiter = None`.
                    if total == 0 {
                        tracing::warn!(
                            table = %table.name, column = %col_name,
                            "multicategorical column was entirely null in sampled file — \
                             treating as single-token-per-row",
                        );
                    } else {
                        tracing::info!(
                            table = %table.name, column = %col_name,
                            "no delimiter candidate reached the threshold; \
                             treating as single-token-per-row",
                        );
                    }
                }
                MulticatDelimiterDecision::Winner(sep) => {
                    table.columns[*col_idx].multicategorical_delimiter = Some(sep.to_string());
                }
            }
        }
    }
    Ok(())
}

/// Validate that an explicitly-declared `multicategorical_delimiter` actually
/// matches the source data. Today an unmatched delimiter silently produces
/// one-token-per-row tokenization at every read site; post-#28478 the same
/// mistake bakes into the rowpack on disk because ingest pre-splits, so a
/// wrong choice is irreversible without re-ingest. Surface it loudly at
/// convert time with column + sample-row context instead.
///
/// Sampling and threshold mirror `detect_multicategorical_delimiters` so
/// auto-detect and explicit-declare paths agree on what "delimiter present
/// in the data" means.
pub fn validate_explicit_multicat_delimiters(
    config: &GraphConfig,
    input_transfer: &Option<CloudTransfer>,
) -> Result<(), ConvertError> {
    for table in &config.tables {
        let mut validator = ExplicitMulticatDelimiterValidator::with_sample_capture(table, true);
        if validator.is_empty() {
            continue;
        }
        if table.format() != TableFormat::Parquet {
            continue;
        }
        let owned_transfer = if input_transfer.is_none() {
            build_input_transfer_for_table(table)
                .map_err(|error| ConvertError::InferenceError(format!("{error:#}")))?
        } else {
            None
        };
        let table_transfer = input_transfer.as_ref().or(owned_transfer.as_ref());
        let parquet_files = table.source_objects();
        if parquet_files.len() == 0 {
            continue;
        }

        for object in parquet_files.take(MULTICAT_MAX_FILES) {
            let path = object.path;
            let Some(reader) =
                open_parquet_sample_reader(table_transfer, path, object.condition(), |error| {
                    ConvertError::InferenceError(format!("{error:#}"))
                })?
            else {
                continue;
            };
            for batch in reader {
                validator.observe(
                    &batch.map_err(|error| ConvertError::InferenceError(format!("{error:#}")))?,
                );
            }

            if validator.is_complete() {
                break;
            }
        }
        validator.finish()?;
    }
    Ok(())
}

#[derive(Debug)]
enum MulticatDelimiterDecision {
    NoDelimiter,
    Winner(&'static str),
}

/// Pure decision function — exposed for unit testing so we can cover the
/// threshold logic without parquet fixtures.
fn decide_multicat_delimiter(
    total: usize,
    hits: &HashMap<&'static str, usize>,
) -> MulticatDelimiterDecision {
    if total == 0 {
        return MulticatDelimiterDecision::NoDelimiter;
    }
    let min_count = (MULTICAT_DETECT_MIN_FRACTION * total as f64).ceil() as usize;

    let mut best: Option<(&'static str, usize)> = None;
    for cand in MULTICAT_DELIMITER_CANDIDATES {
        let count = hits.get(cand).copied().unwrap_or(0);
        if count == 0 {
            continue;
        }
        match best {
            Some((_, best_count)) if best_count >= count => {}
            _ => best = Some((cand, count)),
        }
    }

    match best {
        Some((winner, count)) if count >= min_count => MulticatDelimiterDecision::Winner(winner),
        None => MulticatDelimiterDecision::NoDelimiter,
        _ => MulticatDelimiterDecision::NoDelimiter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn test_estimate_num_shards_zero() {
        assert_eq!(estimate_num_shards(0), 1);
    }

    #[test]
    fn test_effective_memory_never_exceeds_host() {
        // total_memory_bytes() = min(host, cgroup) and must be <= host MemTotal. This
        // guards the primary cgroup fix that feeds shard sizing.
        if let (Some(eff), Some(host)) = (crate::total_memory_bytes(), read_system_memory_bytes()) {
            assert!(
                eff <= host,
                "effective mem {eff} must not exceed host mem {host}"
            );
        }
    }

    #[test]
    fn test_effective_parallelism_never_exceeds_capacity() {
        let eff = crate::effective_parallelism();
        let capacity = crate::cpu_capacity_parallelism();
        assert!(
            eff <= capacity,
            "effective cpus {eff} must not exceed affinity capacity {capacity}"
        );
        assert!(eff >= 1);
        #[cfg(target_os = "linux")]
        if let Some(quota) = crate::read_cgroup_cpu_quota() {
            if quota < capacity {
                assert_eq!(eff, quota.max(1));
            }
        }
    }

    #[test]
    fn test_smaller_cgroup_mem_yields_more_shards() {
        // Sizing shards for a smaller memory slice must produce >= as many shards as a
        // larger one (smaller shards). Uses smooth_shard_target directly to simulate the
        // per-cpu memory that shrinks with a memory cgroup.
        let total = 182 * GB; // ~DoorDash source size
        let cpus = 64u64;
        let host_per_cpu = 497 * GB / cpus;
        let cgroup_per_cpu = 128 * GB / cpus;
        let host_shard = smooth_shard_target(host_per_cpu, total);
        let cgroup_shard = smooth_shard_target(cgroup_per_cpu, total);
        assert!(
            cgroup_shard < host_shard,
            "cgroup shard {cgroup_shard} should be smaller than host shard {host_shard}"
        );
        let host_shards = total.div_ceil(host_shard);
        let cgroup_shards = total.div_ceil(cgroup_shard);
        assert!(
            cgroup_shards > host_shards,
            "cgroup should plan more shards: {cgroup_shards} vs {host_shards}"
        );
    }

    #[test]
    fn test_cpu_quota_does_not_weaken_memory_shard_floor() {
        let total = 182 * GB;
        let memory = 128 * GB;
        let capacity_parallelism = 64;
        let target = shard_target_bytes_for_capacity(memory, capacity_parallelism, total);
        let expected_shards = total.div_ceil(target);

        // Worker quotas affect effective_parallelism(), not the memory-driven
        // shard target. Keep the shard floor fixed throughout a CPU sweep.
        for worker_quota in [64u64, 32, 16, 8, 4] {
            assert!(worker_quota <= capacity_parallelism);
            let actual = total.div_ceil(shard_target_bytes_for_capacity(
                memory,
                capacity_parallelism,
                total,
            ));
            assert_eq!(actual, expected_shards);
        }

        // Guard the exact old failure mode: substituting an 8-core cpu.max
        // quota for capacity would plan fewer, larger shards.
        let quota_sized = total.div_ceil(shard_target_bytes_for_capacity(memory, 8, total));
        assert!(quota_sized < expected_shards);
    }

    #[test]
    fn test_smooth_shard_target_small() {
        let per_cpu = 2 * GB; // 2GB per core
                              // Case: 10GB total (Start of small range)
                              // t=0, smooth=0, divisor=16. Target = 2GB/16 = 128MB.
        let target_10gb = smooth_shard_target(per_cpu, 10 * GB);
        assert_eq!(target_10gb, 128 * 1024 * 1024);

        // Case: 40GB total (End of small range)
        // t=1, smooth=1, divisor=4. Target = 2GB/4 = 512MB.
        let target_40gb = smooth_shard_target(per_cpu, 40 * GB);
        assert_eq!(target_40gb, 512 * 1024 * 1024);
    }

    #[test]
    fn test_smooth_shard_target_medium() {
        let per_cpu = 4 * GB; // 4GB per core
                              // Case: 40GB total (Start of medium range)
                              // Divisor starts at 4.0. Target = 4GB/4 = 1GB.
        let target_40gb = smooth_shard_target(per_cpu, 40 * GB);
        assert_eq!(target_40gb, GB);

        // Case: 1TB total (End of medium range)
        // Divisor ends at 1.0. Target = 4GB/1 = 4GB.
        let target_1tb = smooth_shard_target(per_cpu, 1024 * GB);
        assert_eq!(target_1tb, 4 * GB);
    }

    #[test]
    fn test_smooth_shard_target_large() {
        let per_cpu = 8 * GB;
        // Case: 1TB total (Start of large range)
        // Divisor starts at 1.0. Target = 8GB.
        let target_1tb = smooth_shard_target(per_cpu, 1024 * GB);
        assert_eq!(target_1tb, 8 * GB);

        // Case: 10TB total (End of large range)
        // Divisor ends at 0.5. Target = 8GB/0.5 = 16GB.
        let target_10tb = smooth_shard_target(per_cpu, 10_240 * GB);
        assert_eq!(target_10tb, 16 * GB);
    }

    #[test]
    fn test_smooth_shard_target_huge() {
        let per_cpu = 8 * GB;
        // Case: 20TB (> 10TB)
        // Divisor fixed at 0.75. Target = 8GB/0.75 = 10.66GB.
        let target_huge = smooth_shard_target(per_cpu, 20_000 * GB);
        // 8 / 0.75 = 32/3 = 10.666...
        let expected = (8.0 * 1024.0 * 1024.0 * 1024.0 / 0.75) as u64;
        // Allow off-by-one due to float rounding differences in implementation vs test
        assert!(
            target_huge >= expected - 1 && target_huge <= expected + 1,
            "Target {} not within 1 byte of expected {}",
            target_huge,
            expected
        );
    }

    #[test]
    fn test_monotonicity_within_ranges() {
        // Ensure shard size generally increases or stays same as dataset grows within defined ranges.
        // Note: There is a known discontinuity at 10TB where shard size drops (divisor 0.5 -> 0.75),
        // effectively increasing shard count for very large datasets.
        let per_cpu = 2 * GB;

        // Range 1: Small to Medium (< 1TB)
        let mut prev_target = 0;
        let mut bytes = GB;
        while bytes < 1_000 * GB {
            let target = smooth_shard_target(per_cpu, bytes);
            assert!(
                target >= prev_target,
                "Target shard size decreased at {} GB: {} < {}",
                bytes / GB,
                target,
                prev_target
            );
            prev_target = target;
            bytes += 10 * GB;
        }

        // Range 2: Large (1TB - 10TB)
        prev_target = 0; // Reset
        bytes = 1_024 * GB;
        while bytes <= 10_240 * GB {
            let target = smooth_shard_target(per_cpu, bytes);
            assert!(
                target >= prev_target,
                "Target shard size decreased at {} GB: {} < {}",
                bytes / GB,
                target,
                prev_target
            );
            prev_target = target;
            bytes += 100 * GB;
        }
    }

    #[test]
    fn test_detect_epoch_unit_zero() {
        assert_eq!(detect_epoch_unit(0), None);
    }

    #[test]
    fn test_detect_epoch_unit_seconds() {
        // Unix epoch 2024-03-01 ~= 1709251200
        assert_eq!(detect_epoch_unit(1709251200), Some(TimestampUnit::Seconds));
        // Small value (year 2001)
        assert_eq!(
            detect_epoch_unit(1_000_000_000),
            Some(TimestampUnit::Seconds)
        );
        // Negative (before epoch)
        assert_eq!(detect_epoch_unit(-86400), Some(TimestampUnit::Seconds));
    }

    #[test]
    fn test_detect_epoch_unit_millis() {
        // 1709251200 * 1000
        assert_eq!(
            detect_epoch_unit(1_709_251_200_000),
            Some(TimestampUnit::Millis)
        );
    }

    #[test]
    fn test_detect_epoch_unit_micros() {
        // 1709251200 * 1_000_000
        assert_eq!(
            detect_epoch_unit(1_709_251_200_000_000),
            Some(TimestampUnit::Micros)
        );
    }

    #[test]
    fn test_detect_epoch_unit_nanos() {
        // 1709251200 * 1_000_000_000
        assert_eq!(
            detect_epoch_unit(1_709_251_200_000_000_000),
            Some(TimestampUnit::Nanos)
        );
    }

    #[test]
    fn test_detect_epoch_unit_boundary() {
        // Just below 1e10 → seconds
        assert_eq!(
            detect_epoch_unit(9_999_999_999),
            Some(TimestampUnit::Seconds)
        );
        // At 1e10 → millis
        assert_eq!(
            detect_epoch_unit(10_000_000_000),
            Some(TimestampUnit::Millis)
        );
        // Just below 1e13 → millis
        assert_eq!(
            detect_epoch_unit(9_999_999_999_999),
            Some(TimestampUnit::Millis)
        );
        // At 1e13 → micros
        assert_eq!(
            detect_epoch_unit(10_000_000_000_000),
            Some(TimestampUnit::Micros)
        );
    }

    #[test]
    fn test_detect_int_timestamp_yyyymmdd() {
        let result = detect_int_timestamp(20230408);
        match result {
            Some(TimestampDetection::Format(fmt)) => assert_eq!(fmt, "%Y%m%d"),
            other => panic!(
                "Expected Format(\"%Y%m%d\"), got {:?}",
                other.map(|d| match d {
                    TimestampDetection::Format(f) => format!("Format({})", f),
                    TimestampDetection::Unit(u) => format!("Unit({:?})", u),
                })
            ),
        }
    }

    #[test]
    fn test_detect_int_timestamp_yyyymmddhhmmss() {
        let result = detect_int_timestamp(20230408120530);
        match result {
            Some(TimestampDetection::Format(fmt)) => assert_eq!(fmt, "%Y%m%d%H%M%S"),
            other => panic!(
                "Expected Format(\"%Y%m%d%H%M%S\"), got {:?}",
                other.map(|d| match d {
                    TimestampDetection::Format(f) => format!("Format({})", f),
                    TimestampDetection::Unit(u) => format!("Unit({:?})", u),
                })
            ),
        }
    }

    #[test]
    fn test_detect_int_timestamp_epoch_seconds() {
        // 1709251200 is 10 digits — not a valid date pattern, should be epoch seconds
        let result = detect_int_timestamp(1709251200);
        match result {
            Some(TimestampDetection::Unit(TimestampUnit::Seconds)) => {}
            other => panic!(
                "Expected Unit(Seconds), got {:?}",
                other.map(|d| match d {
                    TimestampDetection::Format(f) => format!("Format({})", f),
                    TimestampDetection::Unit(u) => format!("Unit({:?})", u),
                })
            ),
        }
    }

    // =====================================================================
    // Multicategorical delimiter decision-function tests
    //
    // These cover the threshold logic in `decide_multicat_delimiter` without
    // requiring parquet fixtures. End-to-end detection against real parquet
    // is exercised by the integration tests in `diskgraph-ingest/tests/`.
    // =====================================================================

    fn hits(pairs: &[(&'static str, usize)]) -> HashMap<&'static str, usize> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn multicat_decide_zero_samples_is_no_delimiter() {
        let h = hits(&[]);
        assert!(matches!(
            decide_multicat_delimiter(0, &h),
            MulticatDelimiterDecision::NoDelimiter
        ));
    }

    #[test]
    fn multicat_decide_clear_majority_comma_wins() {
        // 80 / 100 rows have comma, none have others.
        let h = hits(&[(",", 80)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_pipe_wins_against_noise() {
        // 60% pipe, 0.9% comma: the most common observed delimiter wins.
        let h = hits(&[("|", 600), (",", 9)]);
        match decide_multicat_delimiter(1000, &h) {
            MulticatDelimiterDecision::Winner("|") => {}
            other => panic!("expected Winner(\"|\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_picks_most_common_candidate() {
        // Comma in 50% of rows and pipe in 30%: comma wins because it is
        // most common.
        let h = hits(&[(",", 50), ("|", 30)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_single_sparse_candidate_below_50pct_wins() {
        // Sparse multicategorical columns are valid: if exactly one delimiter
        // is observed, it should win even if most rows are single-token.
        let h = hits(&[(",", 30)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_single_observed_candidate_wins() {
        // A delimiter at the 1% floor wins if it is the most common candidate.
        let h = hits(&[(",", 10)]);
        match decide_multicat_delimiter(1000, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_observed_candidate_below_one_percent_is_no_delimiter() {
        let h = hits(&[(",", 9)]);
        assert!(matches!(
            decide_multicat_delimiter(1000, &h),
            MulticatDelimiterDecision::NoDelimiter
        ));
    }

    #[test]
    fn multicat_decide_competing_candidates_pick_highest_count() {
        let h = hits(&[(",", 10), ("|", 11)]);
        match decide_multicat_delimiter(1000, &h) {
            MulticatDelimiterDecision::Winner("|") => {}
            other => panic!("expected Winner(\"|\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_tie_uses_candidate_order() {
        let h = hits(&[(",", 10), ("|", 10)]);
        match decide_multicat_delimiter(1000, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_no_observed_candidate_is_no_delimiter() {
        let h = hits(&[]);
        assert!(matches!(
            decide_multicat_delimiter(1000, &h),
            MulticatDelimiterDecision::NoDelimiter
        ));
    }

    #[test]
    fn multicat_decide_thresholds_scale_with_total() {
        // At very small sample counts, the ceil() in the 1% floor matters.
        // 3 rows, 2 with comma, 0 others → winner.
        let h = hits(&[(",", 2)]);
        match decide_multicat_delimiter(3, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    fn write_file(path: &std::path::Path, size: usize) {
        std::fs::write(path, vec![0u8; size]).unwrap();
    }

    #[test]
    fn test_local_dir_bytes_flat_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_file(&dir.path().join("a.parquet"), 100);
        write_file(&dir.path().join("b.parquet"), 250);
        assert_eq!(compute_local_dir_bytes(dir.path().to_str().unwrap()), 350);
    }

    #[test]
    fn test_local_dir_bytes_nested_partitions() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("dt=2020-01-01");
        let p2 = dir.path().join("dt=2020-01-02");
        std::fs::create_dir_all(&p1).unwrap();
        std::fs::create_dir_all(&p2).unwrap();
        write_file(&dir.path().join("top.parquet"), 10);
        write_file(&p1.join("part-0.parquet"), 100);
        write_file(&p1.join("part-1.parquet"), 200);
        write_file(&p2.join("part-0.parquet"), 300);
        // Sum of all nested files, not just the top-level file.
        assert_eq!(compute_local_dir_bytes(dir.path().to_str().unwrap()), 610);
    }

    #[test]
    fn test_local_dir_bytes_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("data.parquet");
        write_file(&file, 512);
        assert_eq!(compute_local_dir_bytes(file.to_str().unwrap()), 512);
    }

    #[test]
    fn test_local_dir_bytes_file_scheme_prefix() {
        let dir = tempfile::tempdir().unwrap();
        write_file(&dir.path().join("a.parquet"), 128);
        let bare = dir.path().to_str().unwrap();
        let with_scheme = format!("file://{bare}");
        assert_eq!(
            compute_local_dir_bytes(&with_scheme),
            compute_local_dir_bytes(bare)
        );
    }
}
