use anyhow::Result;
use std::collections::HashMap;

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

        // Resolve directory prefixes to individual parquet file paths.
        let parquet_files = resolve_parquet_files(&all_paths[0], input_transfer)?;
        if parquet_files.is_empty() {
            continue;
        }

        // Accumulated state across files.
        let mut votes: HashMap<usize, Vec<String>> = HashMap::new();
        let mut sample_counts: HashMap<usize, usize> = HashMap::new();
        /// Maximum number of non-null values to sample per column for majority vote.
        const MAX_SAMPLES: usize = 10;

        macro_rules! sample_from_reader {
            ($reader:expr, $schema:expr) => {{
                // Resolve column indices and their arrow data types.
                let col_info: Vec<Option<(usize, arrow_schema::DataType)>> = candidates
                    .iter()
                    .map(|(_, name)| {
                        $schema.index_of(name).ok().map(|idx| {
                            (idx, $schema.field(idx).data_type().clone())
                        })
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
                                arrow_schema::DataType::Utf8
                                | arrow_schema::DataType::LargeUtf8
                                | arrow_schema::DataType::Binary => {
                                    let text: Option<&str> = if let Some(s) =
                                        array.as_any().downcast_ref::<arrow_array::StringArray>()
                                    {
                                        Some(s.value(row))
                                    } else if let Some(s) = array
                                        .as_any()
                                        .downcast_ref::<arrow_array::LargeStringArray>()
                                    {
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
                                _ => {}
                            }
                        }
                    }
                    // One batch (1024 rows) is enough per file.
                    break;
                }
            }};
        }

        // Iterate through files until all candidates have at least one vote.
        for path in &parquet_files {
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
                let builder = match
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
                {
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
                let builder = match
                    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                {
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

            // Check if all candidates have at least one vote — no need to read more files.
            let all_resolved = candidates
                .iter()
                .all(|(col_idx, _)| votes.contains_key(col_idx));
            if all_resolved {
                break;
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
    } else if let Some(a) = array.as_any().downcast_ref::<UInt8Array>() {
        Some(a.value(row) as i64)
    } else {
        None
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

/// Resolve a path (which may be a directory prefix) to all `.parquet` files within it.
///
/// - Cloud directory prefixes (ending with `/`): lists objects and collects all `.parquet` files.
/// - Cloud file paths: returned as a single-element vec if they end with `.parquet`.
/// - Local directories: walks entries and collects all `.parquet` files, sorted by name.
/// - Local file paths: returned as a single-element vec if they end with `.parquet`.
fn resolve_parquet_files(
    path: &str,
    transfer: &Option<CloudTransfer>,
) -> Result<Vec<String>> {
    if is_cloud_path(path) {
        if path.ends_with('/') || !path.ends_with(".parquet") {
            // Directory prefix — list and collect all parquet files.
            if let Some(transfer) = transfer {
                let prefix = crate::io::object_path_from_cloud_url(path)?;
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                let objects = rt.block_on(transfer.list_objects(&prefix))?;
                let base_url = path.trim_end_matches('/');
                let bucket_root = base_url
                    .find("://")
                    .and_then(|i| base_url[i + 3..].find('/').map(|j| &base_url[..i + 3 + j]))
                    .unwrap_or(base_url);
                let mut files: Vec<String> = objects
                    .iter()
                    .filter_map(|obj| {
                        let loc = obj.location.as_ref().to_string();
                        if loc.ends_with(".parquet") {
                            Some(format!("{}/{}", bucket_root, loc))
                        } else {
                            None
                        }
                    })
                    .collect();
                files.sort();
                return Ok(files);
            }
            Ok(Vec::new())
        } else {
            Ok(vec![path.to_string()])
        }
    } else {
        // Local path
        let p = std::path::Path::new(path.trim_end_matches('/'));
        if p.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(p)?
                .flatten()
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "parquet")
                        .unwrap_or(false)
                })
                .collect();
            entries.sort_by_key(|e| e.file_name());
            Ok(entries
                .iter()
                .map(|e| e.path().to_string_lossy().to_string())
                .collect())
        } else if path.ends_with(".parquet") {
            Ok(vec![path.to_string()])
        } else {
            Ok(Vec::new())
        }
    }
}

/// Sum file sizes in a local directory or single file.
fn compute_local_dir_bytes(dir_path: &str) -> u64 {
    let dir_path = dir_path.trim_end_matches('/');
    let entries = match std::fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(_) => {
            // Maybe it's a single file, not a directory
            return std::fs::metadata(dir_path)
                .map(|m| m.len())
                .unwrap_or(0);
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
    Ok(objects.iter().map(|o| o.size as u64).sum())
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
        assert_eq!(target_40gb, 1 * GB);

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
        let mut bytes = 1 * GB;
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
        assert_eq!(detect_epoch_unit(1_000_000_000), Some(TimestampUnit::Seconds));
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
            other => panic!("Expected Format(\"%Y%m%d\"), got {:?}", other.map(|d| match d {
                TimestampDetection::Format(f) => format!("Format({})", f),
                TimestampDetection::Unit(u) => format!("Unit({:?})", u),
            })),
        }
    }

    #[test]
    fn test_detect_int_timestamp_yyyymmddhhmmss() {
        let result = detect_int_timestamp(20230408120530);
        match result {
            Some(TimestampDetection::Format(fmt)) => assert_eq!(fmt, "%Y%m%d%H%M%S"),
            other => panic!("Expected Format(\"%Y%m%d%H%M%S\"), got {:?}", other.map(|d| match d {
                TimestampDetection::Format(f) => format!("Format({})", f),
                TimestampDetection::Unit(u) => format!("Unit({:?})", u),
            })),
        }
    }

    #[test]
    fn test_detect_int_timestamp_epoch_seconds() {
        // 1709251200 is 10 digits — not a valid date pattern, should be epoch seconds
        let result = detect_int_timestamp(1709251200);
        match result {
            Some(TimestampDetection::Unit(TimestampUnit::Seconds)) => {}
            other => panic!("Expected Unit(Seconds), got {:?}", other.map(|d| match d {
                TimestampDetection::Format(f) => format!("Format({})", f),
                TimestampDetection::Unit(u) => format!("Unit({:?})", u),
            })),
        }
    }
}
