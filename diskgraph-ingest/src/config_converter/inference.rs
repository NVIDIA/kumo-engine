use anyhow::Result;
use std::collections::{HashMap, HashSet};

use crate::graph_config::{GraphConfig, IngestedType, TableFormat, TimestampUnit};
use crate::io::{download_cloud_bytes, is_cloud_path};
use diskgraph_fs::CloudTransfer;

/// Detection result for a single timestamp column.
enum TimestampDetection {
    /// String format detected (e.g., "%Y-%m-%d %H:%M:%S").
    Format(String),
    /// Epoch unit detected from integer magnitude.
    Unit(TimestampUnit),
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

        let all_paths = table.get_paths();
        if all_paths.is_empty() {
            continue;
        }

        let parquet_files = all_paths;
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
        for path in parquet_files.iter().take(MAX_FILES) {
            let pooled = if let Some(transfer) = input_transfer {
                if is_cloud_path(path) {
                    match download_cloud_bytes(transfer, path) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            tracing::warn!("Skipping {}: download failed: {e}", path);
                            continue;
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(pooled) = pooled {
                let bytes = pooled.bytes();
                let builder =
                    match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        bytes,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                            continue;
                        }
                    };
                let schema = builder.schema().clone();
                let reader = builder.with_batch_size(1024).build()?;
                sample_from_reader!(reader, schema);
            } else {
                let file = match std::fs::File::open(path) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!("Skipping {}: failed to open: {e}", path);
                        continue;
                    }
                };
                let builder =
                    match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        file,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                            continue;
                        }
                    };
                let schema = builder.schema().clone();
                let reader = builder.with_batch_size(1024).build()?;
                sample_from_reader!(reader, schema);
            }

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
            let is_time_col = table.create_time.as_ref() == Some(col_name)
                || table.end_time.as_ref() == Some(col_name);

            if let Some(info) = unrecognized_info.get(col_idx) {
                if is_time_col {
                    let role = if table.create_time.as_ref() == Some(col_name) {
                        "create_time"
                    } else {
                        "end_time"
                    };
                    return Err(anyhow::anyhow!(
                        "table '{}' column '{}' ({}): could not detect timestamp format. \
                         {}. Set timestamp_format explicitly in the config.",
                        table.name,
                        col_name,
                        role,
                        info,
                    ));
                }
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    "could not detect timestamp format: {}", info,
                );
            } else if sample_counts.get(col_idx).copied().unwrap_or(0) == 0 {
                if is_time_col {
                    let role = if table.create_time.as_ref() == Some(col_name) {
                        "create_time"
                    } else {
                        "end_time"
                    };
                    return Err(anyhow::anyhow!(
                        "table '{}' column '{}' ({}): column is entirely null across {} file(s). \
                         A time column must contain parseable timestamp values.",
                        table.name,
                        col_name,
                        role,
                        parquet_files.len().min(MAX_FILES),
                    ));
                }
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    "column is entirely null across {} file(s)",
                    parquet_files.len().min(MAX_FILES),
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

/// Detect epoch unit from an integer timestamp value based on magnitude.
fn detect_epoch_unit(value: i64) -> Option<TimestampUnit> {
    let abs = value.unsigned_abs();
    if abs == 0 {
        return None;
    }
    Some(if abs < 10_000_000_000 {
        // < 1e10 → seconds (max ~2286)
        TimestampUnit::Seconds
    } else if abs < 10_000_000_000_000 {
        // < 1e13 → milliseconds
        TimestampUnit::Millis
    } else if abs < 10_000_000_000_000_000 {
        // < 1e16 → microseconds
        TimestampUnit::Micros
    } else {
        // >= 1e16 → nanoseconds
        TimestampUnit::Nanos
    })
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
        for path in table.get_paths() {
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


/// Sum file sizes in a local directory or single file.
fn compute_local_dir_bytes(dir_path: &str) -> u64 {
    let dir_path = dir_path.trim_end_matches('/');
    let entries = match std::fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(_) => {
            // Maybe it's a single file, not a directory
            return std::fs::metadata(dir_path).map(|m| m.len()).unwrap_or(0);
        }
    };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                total += meta.len();
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
fn shard_target_bytes(total_bytes: u64) -> u64 {
    const FALLBACK_SHARD_BYTES: u64 = 1024 * 1024 * 1024; // 1 GB
    let mem_total = read_system_memory_bytes();
    let cpu_count = std::thread::available_parallelism()
        .ok()
        .map(|n| n.get() as u64);
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

/// Majority threshold for a candidate to win outright (fraction of non-null
/// sample rows containing the delimiter).
const MULTICAT_WIN_FRACTION: f64 = 0.50;
/// Ambiguity threshold — a candidate at or above this fraction that is not
/// the winner triggers `MulticategoricalDelimiterAmbiguous`.
const MULTICAT_AMBIGUOUS_FRACTION: f64 = 0.20;
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

/// Detect the per-column `multicategorical_delimiter` for every
/// `String + Multicategorical` column that didn't come with an explicit
/// value on the upstream Kumo config.
///
/// Decision rule (§4.2 Step 0 of the multicat design doc):
/// - A candidate `wins` if it appears in ≥50% of sampled rows AND no other
///   candidate appears in ≥20% of sampled rows → bake that delimiter in.
/// - No candidate reaches 20% → `delimiter = None` (legitimate
///   single-token-per-row multicat).
/// - 2+ candidates ≥20% OR a single candidate in `[20%, 50%)` with no
///   other candidates → `MulticategoricalDelimiterAmbiguous` so users see
///   the problem before ingest starts.
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
        let all_paths = table.get_paths();
        if all_paths.is_empty() {
            continue;
        }
        let parquet_files = all_paths;

        // Per-column per-candidate occurrence counts, and total non-null
        // sample rows for that column.
        let mut per_col_totals: HashMap<usize, usize> = HashMap::new();
        let mut per_col_hits: HashMap<usize, HashMap<&'static str, usize>> = HashMap::new();

        // Sample multiple parquet files. Early files can be empty, all-null,
        // or non-representative partitions, so continue until each candidate
        // has enough non-null samples or the bounded file budget is exhausted.
        for path in parquet_files.iter().take(MULTICAT_MAX_FILES) {
            let pooled = if let Some(transfer) = input_transfer {
                if is_cloud_path(path) {
                    match download_cloud_bytes(transfer, path) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            tracing::warn!("Skipping {}: download failed: {e}", path);
                            continue;
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

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

            if let Some(pooled) = pooled {
                let bytes = pooled.bytes();
                let builder =
                    match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        bytes,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                            continue;
                        }
                    };
                let schema = builder.schema().clone();
                let reader = builder
                    .with_batch_size(1024)
                    .build()
                    .map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                scan_reader!(reader, schema);
            } else {
                let file = match std::fs::File::open(path) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!("Skipping {}: failed to open: {e}", path);
                        continue;
                    }
                };
                let builder =
                    match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                        file,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                            continue;
                        }
                    };
                let schema = builder.schema().clone();
                let reader = builder
                    .with_batch_size(1024)
                    .build()
                    .map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                scan_reader!(reader, schema);
            }

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
                MulticatDelimiterDecision::Ambiguous(candidates_with_counts) => {
                    return Err(ConvertError::MulticategoricalDelimiterAmbiguous {
                        table: table.name.clone(),
                        column: col_name.clone(),
                        candidates: candidates_with_counts,
                    });
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
        // Validate any multicat column whose source comes off parquet as a
        // scalar string with a declared delimiter — whether the rowpack
        // logical type is `String` (delimited string of string tokens),
        // `StringArray` (delim-string post-normalize), or `Int64Array`
        // (delim-string of int tokens). Native `List<...>` source columns
        // never carry a delimiter and don't need this check; the access
        // layer reads their elements directly without splitting.
        let candidates: Vec<(usize, String, String)> = table
            .columns
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let scalar_string_source = matches!(
                    c.ingested_dtype,
                    IngestedType::String | IngestedType::StringArray | IngestedType::Int64Array,
                );
                if scalar_string_source
                    && c.semantic_type == Some(SemanticType::Multicategorical)
                {
                    c.multicategorical_delimiter
                        .as_deref()
                        .map(|d| (i, c.name.clone(), d.to_string()))
                } else {
                    None
                }
            })
            .collect();
        if candidates.is_empty() {
            continue;
        }
        if table.format() != TableFormat::Parquet {
            continue;
        }
        let all_paths = table.get_paths();
        if all_paths.is_empty() {
            continue;
        }
        let parquet_files = all_paths;

        // Per-column occurrence counts + the first sample row that didn't
        // contain the declared delimiter. The first miss is plenty for the
        // error message — diagnostic, not exhaustive.
        let mut totals: HashMap<usize, usize> = HashMap::new();
        let mut hits: HashMap<usize, usize> = HashMap::new();
        let mut miss_sample: HashMap<usize, String> = HashMap::new();

        for path in parquet_files.iter().take(MULTICAT_MAX_FILES) {
            let pooled = if let Some(transfer) = input_transfer {
                if is_cloud_path(path) {
                    match download_cloud_bytes(transfer, path) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            tracing::warn!("Skipping {}: download failed: {e}", path);
                            continue;
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            macro_rules! scan_reader {
                ($reader:expr, $schema:expr) => {{
                    let col_info: Vec<Option<usize>> = candidates
                        .iter()
                        .map(|(_, name, _)| $schema.index_of(name).ok())
                        .collect();
                    for batch in $reader {
                        let batch = batch
                            .map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                        for (cand_idx, info) in col_info.iter().enumerate() {
                            let Some(arr_idx) = info else { continue };
                            let (col_key, _, declared) = &candidates[cand_idx];
                            if *totals.get(col_key).unwrap_or(&0) >= MULTICAT_MAX_SAMPLES {
                                continue;
                            }
                            let array = batch.column(*arr_idx);
                            for row in 0..batch.num_rows() {
                                if *totals.get(col_key).unwrap_or(&0) >= MULTICAT_MAX_SAMPLES {
                                    break;
                                }
                                if array.is_null(row) {
                                    continue;
                                }
                                let Some(text) = multicat_string_value(array.as_ref(), row)
                                else {
                                    continue;
                                };
                                if text.is_empty() {
                                    continue;
                                }
                                *totals.entry(*col_key).or_insert(0) += 1;
                                if text.contains(declared.as_str()) {
                                    *hits.entry(*col_key).or_insert(0) += 1;
                                } else {
                                    miss_sample
                                        .entry(*col_key)
                                        .or_insert_with(|| text.to_string());
                                }
                            }
                        }
                    }
                }};
            }

            if let Some(pooled) = pooled {
                let bytes = pooled.bytes();
                let builder = match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                        continue;
                    }
                };
                let schema = builder.schema().clone();
                let reader = builder
                    .with_batch_size(1024)
                    .build()
                    .map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                scan_reader!(reader, schema);
            } else {
                let file = match std::fs::File::open(path) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!("Skipping {}: failed to open: {e}", path);
                        continue;
                    }
                };
                let builder = match parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("Skipping {}: failed to read parquet: {e}", path);
                        continue;
                    }
                };
                let schema = builder.schema().clone();
                let reader = builder
                    .with_batch_size(1024)
                    .build()
                    .map_err(|e| ConvertError::InferenceError(format!("{e:#}")))?;
                scan_reader!(reader, schema);
            }

            let all_done = candidates.iter().all(|(col_idx, _, _)| {
                *totals.get(col_idx).unwrap_or(&0) >= MULTICAT_MAX_SAMPLES
            });
            if all_done {
                break;
            }
        }

        for (col_idx, col_name, declared) in &candidates {
            let total = *totals.get(col_idx).unwrap_or(&0);
            let hit = *hits.get(col_idx).unwrap_or(&0);
            // Empty column / all-null sample → can't validate, skip silently.
            // The auto-detect path also tolerates this with a tracing warn.
            if total == 0 {
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    "explicit multicategorical_delimiter could not be validated: \
                     no non-empty sample rows in the bounded scan",
                );
                continue;
            }
            // The hit threshold for explicit-delim validation is intentionally
            // looser than the auto-detect winner threshold
            // (`MULTICAT_WIN_FRACTION = 50%`). Auto-detect picks among
            // candidates and needs a clear majority; here the user has already
            // committed to a specific delimiter, so the only failure mode we
            // need to catch is total mismatch (typo, wrong column). A column
            // where most rows have a single tag and only a few rows are
            // multi-token is valid: each single-token row tokenizes correctly
            // as a one-element list under GE-parity drop-empty. Reject only
            // when `hits == 0` — i.e. the declared delimiter NEVER appears in
            // any sampled non-empty row. Anything else gets a tracing warn so
            // a sparse-but-real configuration is preserved while still
            // surfacing suspiciously low rates.
            if hit == 0 {
                let sample = miss_sample
                    .remove(col_idx)
                    .unwrap_or_else(|| "<no sample>".to_string());
                return Err(ConvertError::MulticategoricalDelimiterMismatch {
                    table: table.name.clone(),
                    column: col_name.clone(),
                    declared: declared.clone(),
                    hits: hit,
                    total,
                    required: 1,
                    sample,
                });
            }
            let suspicious_floor =
                (MULTICAT_AMBIGUOUS_FRACTION * total as f64).ceil() as usize;
            if hit < suspicious_floor {
                tracing::warn!(
                    table = %table.name, column = %col_name,
                    declared = %declared,
                    hits = hit,
                    total = total,
                    "explicit multicategorical_delimiter appears in fewer rows than \
                     expected; data may be mostly single-token, but verify the \
                     declared delimiter is correct",
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
enum MulticatDelimiterDecision {
    NoDelimiter,
    Winner(&'static str),
    Ambiguous(Vec<(String, usize)>),
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
    let win_cutoff = (MULTICAT_WIN_FRACTION * total as f64).ceil() as usize;
    let ambig_cutoff = (MULTICAT_AMBIGUOUS_FRACTION * total as f64).ceil() as usize;

    let mut winner: Option<&'static str> = None;
    let mut above_ambig: Vec<(&'static str, usize)> = Vec::new();
    for cand in MULTICAT_DELIMITER_CANDIDATES {
        let count = hits.get(cand).copied().unwrap_or(0);
        if count >= ambig_cutoff {
            above_ambig.push((cand, count));
        }
        if count >= win_cutoff {
            // Pick the first qualifier; ties are resolved by the
            // "no other candidate ≥ ambig_cutoff" invariant below.
            if winner.is_none() {
                winner = Some(cand);
            }
        }
    }
    match winner {
        Some(w) => {
            let others_above_ambig = above_ambig.iter().any(|(c, _)| *c != w);
            if others_above_ambig {
                let counts: Vec<(String, usize)> = above_ambig
                    .into_iter()
                    .map(|(c, n)| (c.to_string(), n))
                    .collect();
                MulticatDelimiterDecision::Ambiguous(counts)
            } else {
                MulticatDelimiterDecision::Winner(w)
            }
        }
        None => {
            if above_ambig.is_empty() {
                // No candidate reached even the ambiguity floor — treat as
                // single-token-per-row.
                MulticatDelimiterDecision::NoDelimiter
            } else {
                // At least one candidate between 20% and 50% — unstable.
                let counts: Vec<(String, usize)> = above_ambig
                    .into_iter()
                    .map(|(c, n)| (c.to_string(), n))
                    .collect();
                MulticatDelimiterDecision::Ambiguous(counts)
            }
        }
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
        // 60% pipe, 10% comma (noise below 20% floor).
        let h = hits(&[("|", 60), (",", 10)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Winner("|") => {}
            other => panic!("expected Winner(\"|\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_ambiguous_two_candidates_above_20pct() {
        // Comma in 50% of rows AND pipe in 30% — winner candidate exists
        // but is ambiguous because pipe is above the 20% floor.
        let h = hits(&[(",", 50), ("|", 30)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Ambiguous(counts) => {
                // Both candidates surface in the error payload.
                let has_comma = counts.iter().any(|(c, _)| c == ",");
                let has_pipe = counts.iter().any(|(c, _)| c == "|");
                assert!(has_comma && has_pipe);
            }
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_ambiguous_single_candidate_below_50pct() {
        // Only one candidate, but it's in the [20%, 50%) band — not a stable
        // enough signal to commit to a delimiter; surface as ambiguous so
        // the user adjudicates.
        let h = hits(&[(",", 30)]);
        match decide_multicat_delimiter(100, &h) {
            MulticatDelimiterDecision::Ambiguous(_) => {}
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_no_candidate_above_20pct_is_no_delimiter() {
        // Every candidate below floor → treat as single-token-per-row.
        let h = hits(&[(",", 5), ("|", 10), (";", 2)]);
        assert!(matches!(
            decide_multicat_delimiter(100, &h),
            MulticatDelimiterDecision::NoDelimiter
        ));
    }

    #[test]
    fn multicat_decide_thresholds_scale_with_total() {
        // At very small sample counts, the ceil() in cutoffs matters. 3 rows,
        // 2 with comma (66%, above 50%), 0 others → winner.
        let h = hits(&[(",", 2)]);
        match decide_multicat_delimiter(3, &h) {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }
}
