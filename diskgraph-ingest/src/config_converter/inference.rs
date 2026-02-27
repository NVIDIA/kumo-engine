use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::graph_config::{GraphConfig, IngestedType, TableFormat};
use crate::io::{download_cloud_bytes, is_cloud_path};
use diskgraph_fs::CloudTransfer;

/// Run a future on the current tokio runtime if one exists, otherwise create a
/// temporary single-threaded runtime.  This avoids the "cannot start a runtime
/// from within a runtime" panic when inference functions are called from an
/// async context (e.g. validate-s3-ingest).
fn block_on_async<F: std::future::Future>(f: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(f))
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(f)
    }
}

/// Detect timestamp format strings for string-backed timestamp/date columns.
///
/// For each table in the config, finds columns with `ingested_dtype ∈ {Timestamp, Date}`
/// whose `timestamp_format` is not yet set. Opens the first parquet file, reads
/// one batch, and uses `diskgraph_rowpack::detect_timestamp_format` on the first
/// non-null value of each candidate column.
///
/// Modifies `config` in place, setting `col.timestamp_format` for detected columns.
pub fn detect_timestamp_formats(
    config: &mut GraphConfig,
    input_transfer: &Option<CloudTransfer>,
) -> Result<()> {
    for table in &mut config.tables {
        // Find candidate columns: Timestamp or Date with no format set yet.
        let candidates: Vec<(usize, String)> = table
            .columns
            .iter()
            .enumerate()
            .filter(|(_, col)| {
                matches!(
                    col.ingested_dtype,
                    IngestedType::Timestamp | IngestedType::Date
                ) && col.timestamp_format.is_none()
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

        // Resolve the first path to an actual parquet file.
        // Paths from the converter are directory prefixes (e.g., "s3://.../version/")
        // so we need to list and pick the first .parquet file.
        let first_parquet = resolve_first_parquet_file(&all_paths[0], input_transfer)?;
        let path = match first_parquet {
            Some(p) => p,
            None => continue, // No parquet files found, skip this table
        };

        // Open the first parquet file (cloud or local).
        let pooled = if let Some(transfer) = input_transfer {
            if is_cloud_path(&path) {
                Some(download_cloud_bytes(transfer, &path)?)
            } else {
                None
            }
        } else {
            None
        };

        // Detected formats: col_index -> format string
        let mut detected: HashMap<usize, String> = HashMap::new();

        macro_rules! detect_from_reader {
            ($reader:expr, $schema:expr) => {{
                // Resolve column indices; skip candidates whose parquet type is not string-like.
                let col_indices: Vec<Option<usize>> = candidates
                    .iter()
                    .map(|(_, name)| {
                        if let Ok(idx) = $schema.index_of(name) {
                            match $schema.field(idx).data_type() {
                                arrow_schema::DataType::Utf8
                                | arrow_schema::DataType::LargeUtf8
                                | arrow_schema::DataType::Binary => Some(idx),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                // Read one batch and detect from first non-null value per column.
                for batch in $reader {
                    let batch = batch?;
                    for (cand_idx, arr_idx) in col_indices.iter().enumerate() {
                        let arr_idx = match arr_idx {
                            Some(i) => *i,
                            None => continue,
                        };
                        if detected.contains_key(&candidates[cand_idx].0) {
                            continue;
                        }
                        let array = batch.column(arr_idx);
                        for row in 0..batch.num_rows() {
                            if array.is_null(row) {
                                continue;
                            }
                            let value: Option<&str> = if let Some(s) =
                                array.as_any().downcast_ref::<arrow_array::StringArray>()
                            {
                                Some(s.value(row))
                            } else if let Some(s) =
                                array.as_any().downcast_ref::<arrow_array::LargeStringArray>()
                            {
                                Some(s.value(row))
                            } else if let Some(s) =
                                array.as_any().downcast_ref::<arrow_array::BinaryArray>()
                            {
                                std::str::from_utf8(s.value(row)).ok()
                            } else {
                                None
                            };
                            if let Some(value) = value {
                                if let Some(fmt) =
                                    diskgraph_rowpack::detect_timestamp_format(value)
                                {
                                    detected
                                        .insert(candidates[cand_idx].0, fmt.to_string());
                                }
                            }
                            break; // Only need first non-null value
                        }
                    }
                    // One batch is enough.
                    break;
                }
            }};
        }

        if let Some(pooled) = pooled {
            let bytes = pooled.bytes();
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)?;
            let schema = builder.schema().clone();
            let reader = builder.with_batch_size(1024).build()?;
            detect_from_reader!(reader, schema);
        } else {
            let file = std::fs::File::open(&path)
                .with_context(|| format!("open {} for timestamp detection", path))?;
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
            let schema = builder.schema().clone();
            let reader = builder.with_batch_size(1024).build()?;
            detect_from_reader!(reader, schema);
        }

        // Apply detected formats to the config.
        for (col_idx, fmt) in detected {
            table.columns[col_idx].timestamp_format = Some(fmt);
        }
    }
    Ok(())
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

/// Resolve a path (which may be a directory prefix) to the first .parquet file.
///
/// - Cloud directory prefixes (ending with `/`): lists objects and picks the first `.parquet`.
/// - Cloud file paths: returned as-is if they end with `.parquet`.
/// - Local directories: walks entries and picks the first `.parquet`.
/// - Local file paths: returned as-is if they end with `.parquet`.
fn resolve_first_parquet_file(
    path: &str,
    transfer: &Option<CloudTransfer>,
) -> Result<Option<String>> {
    if is_cloud_path(path) {
        if path.ends_with('/') || !path.ends_with(".parquet") {
            // Directory prefix — list and find first parquet file.
            if let Some(transfer) = transfer {
                let prefix = crate::io::object_path_from_cloud_url(path)?;
                let objects = block_on_async(transfer.list_objects(&prefix))?;
                // Find first .parquet object and reconstruct full cloud URL.
                let base_url = path.trim_end_matches('/');
                // Strip the prefix portion from the base URL to get the bucket root.
                let bucket_root = base_url
                    .find("://")
                    .and_then(|i| base_url[i + 3..].find('/').map(|j| &base_url[..i + 3 + j]))
                    .unwrap_or(base_url);
                for obj in &objects {
                    let loc = obj.location.as_ref().to_string();
                    if loc.ends_with(".parquet") {
                        return Ok(Some(format!("{}/{}", bucket_root, loc)));
                    }
                }
                return Ok(None);
            }
            Ok(None)
        } else {
            Ok(Some(path.to_string()))
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
                .first()
                .map(|e| e.path().to_string_lossy().to_string()))
        } else if path.ends_with(".parquet") {
            Ok(Some(path.to_string()))
        } else {
            Ok(None)
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
    let objects = block_on_async(transfer.list_objects(&prefix))?;
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
}
