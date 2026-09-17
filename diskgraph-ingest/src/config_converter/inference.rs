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

/// How one sampled value votes.
enum TemporalVote {
    /// Arrow-native temporal type: the value carries its own unit, no hint needed.
    Native,
    Detected(TimestampDetection),
    /// Non-null but unclassifiable; carries a sample for the warning.
    Unrecognized(String),
    Abstain,
}

/// Budget is counted in votes, not rows: a run of sentinels must not exhaust it
/// and hide the timestamps behind it. Sized to kumo's inference sample
/// (`NUM_ROWS_FOR_APPROX_STATS`), so the format is decided by the whole sample
/// the caller supplies rather than by its first handful of classifiable rows —
/// the detected format is persisted and never re-inferred, so a decision made
/// on too little evidence is permanent.
const TEMPORAL_MAX_VOTES: usize = 10_000;

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

        /// Maximum number of files to download when searching for non-null values.
        const MAX_FILES: usize = 15;

        macro_rules! sample_from_reader {
            ($reader:expr, $schema:expr) => {{
                let col_info: Vec<Option<usize>> = candidates
                    .iter()
                    .map(|(_, name)| $schema.index_of(name).ok())
                    .collect();

                for batch in $reader {
                    let batch = batch?;
                    for (cand_idx, info) in col_info.iter().enumerate() {
                        let Some(arr_idx) = *info else {
                            continue;
                        };
                        let col_key = candidates[cand_idx].0;
                        if native_temporal.contains(&col_key) {
                            continue;
                        }
                        if votes.get(&col_key).map_or(0, Vec::len) >= TEMPORAL_MAX_VOTES {
                            continue;
                        }
                        let array = batch.column(arr_idx);
                        for row in 0..batch.num_rows() {
                            if votes.get(&col_key).map_or(0, Vec::len) >= TEMPORAL_MAX_VOTES {
                                break;
                            }
                            if array.is_null(row) {
                                continue;
                            }
                            *sample_counts.entry(col_key).or_insert(0) += 1;
                            match classify_temporal_value(array.as_ref(), row) {
                                TemporalVote::Native => {
                                    native_temporal.insert(col_key);
                                    break; // done with this column
                                }
                                TemporalVote::Detected(det) => {
                                    votes
                                        .entry(col_key)
                                        .or_default()
                                        .push(detection_to_vote_key(&det));
                                }
                                TemporalVote::Unrecognized(info) => {
                                    unrecognized_info.entry(col_key).or_insert(info);
                                }
                                TemporalVote::Abstain => {}
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

/// Coarser units rank higher: read as seconds, a millisecond column overflows
/// to null and is reported; read as millis, a second column silently becomes a
/// plausible 1970 timestamp. Ties go to the failure you can see.
fn unit_vote_rank(key: &str) -> u8 {
    match key {
        "unit:Seconds" => 4,
        "unit:Millis" => 3,
        "unit:Micros" => 2,
        "unit:Nanos" => 1,
        _ => 0,
    }
}

/// Pick the most common vote. Ties prefer Format over Unit, then the coarser
/// unit, then the first vote cast. The winner is persisted and never
/// re-inferred, so the tie-break has to be total and deterministic.
fn majority_vote(votes: &[String]) -> Option<TimestampDetection> {
    if votes.is_empty() {
        return None;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in votes {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let max_count = *counts.values().max().unwrap();
    // Seed from `votes`, not `counts`: first-seen order, not hash order.
    let mut best: Vec<&str> = Vec::new();
    for v in votes {
        let key = v.as_str();
        if counts[key] == max_count && !best.contains(&key) {
            best.push(key);
        }
    }
    // Stable, so equal-rank keys keep first-seen order.
    best.sort_by(|a, b| {
        let a_is_fmt = a.starts_with("fmt:");
        let b_is_fmt = b.starts_with("fmt:");
        b_is_fmt
            .cmp(&a_is_fmt) // true (fmt) sorts before false (unit)
            .then_with(|| unit_vote_rank(b).cmp(&unit_vote_rank(a)))
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

/// Rewrite pandas' epoch encodings — float64 once a null appears, `i64::MIN`
/// for NaT — as a null-carrying `Int64Array`, so a sentinel or an out-of-range
/// float becomes null rather than a wrong instant. `None` when already clean.
fn normalize_epoch_column(array: &dyn arrow_array::Array) -> Option<arrow_array::ArrayRef> {
    use arrow_array::{Float32Array, Float64Array, Int64Array};
    let truncate =
        |v: f64| (v.is_finite() && v > i64::MIN as f64 && v < i64::MAX as f64).then_some(v as i64);
    let values: Vec<Option<i64>> = if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        a.iter().map(|v| v.and_then(truncate)).collect()
    } else if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        a.iter()
            .map(|v| v.and_then(|v| truncate(f64::from(v))))
            .collect()
    } else {
        let a = array.as_any().downcast_ref::<Int64Array>()?;
        if !a.iter().any(|v| v == Some(i64::MIN)) {
            return None;
        }
        a.iter().map(|v| v.filter(|&v| v != i64::MIN)).collect()
    };
    Some(std::sync::Arc::new(Int64Array::from(values)))
}

fn string_value(array: &dyn arrow_array::Array, row: usize) -> Option<&str> {
    use arrow_array::*;
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        Some(a.value(row))
    } else if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
        Some(a.value(row))
    } else {
        array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .and_then(|a| std::str::from_utf8(a.value(row)).ok())
    }
}

/// The single per-value temporal decision; every caller funnels through here.
fn classify_temporal_value(array: &dyn arrow_array::Array, row: usize) -> TemporalVote {
    use arrow_schema::DataType::*;
    match array.data_type() {
        Date32 | Date64 | Timestamp(_, _) | Time32(_) | Time64(_) => TemporalVote::Native,
        Utf8 | LargeUtf8 | Binary => {
            let Some(text) = string_value(array, row) else {
                return TemporalVote::Abstain;
            };
            if let Some(fmt) = diskgraph_rowpack::detect_timestamp_format(text) {
                TemporalVote::Detected(TimestampDetection::Format(fmt.to_string()))
            } else if let Ok(value) = text.trim().parse::<i64>() {
                match detect_int_timestamp(value) {
                    Some(det) => TemporalVote::Detected(det),
                    None => TemporalVote::Abstain,
                }
            } else {
                let sample: String = text.chars().take(50).collect();
                TemporalVote::Unrecognized(format!(
                    "unrecognized string format, sample: \"{}\"",
                    sample
                ))
            }
        }
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => {
            match read_int_value(array, row).and_then(detect_int_timestamp) {
                Some(det) => TemporalVote::Detected(det),
                None => TemporalVote::Abstain,
            }
        }
        dt => TemporalVote::Unrecognized(format!("unsupported arrow type: {:?}", dt)),
    }
}

/// Majority-vote a column's temporal hint from its first non-null values.
/// `None` for a native-temporal, empty or unclassifiable column.
fn detect_temporal_column(array: &dyn arrow_array::Array) -> Option<TimestampDetection> {
    let mut votes: Vec<String> = Vec::new();
    for row in 0..array.len() {
        if votes.len() >= TEMPORAL_MAX_VOTES {
            break;
        }
        if array.is_null(row) {
            continue;
        }
        match classify_temporal_value(array, row) {
            TemporalVote::Native => return None,
            TemporalVote::Detected(det) => votes.push(detection_to_vote_key(&det)),
            _ => {}
        }
    }
    majority_vote(&votes)
}

/// The hint Rust chose and the epoch values it produced under it. Detection
/// sets at most one of `format`/`unit`; a caller that declares both gets both.
pub struct ResolvedTemporal {
    pub epochs: Vec<Option<i64>>,
    pub format: Option<String>,
    pub unit: Option<TimestampUnit>,
}

/// Detect (unless declared) and apply the temporal hint for one column,
/// yielding epoch seconds, or epoch days when `as_days`. Best-effort: a null or
/// unparseable value yields `None`, never an epoch.
pub fn resolve_temporal(
    array: &arrow_array::ArrayRef,
    declared_format: Option<&str>,
    declared_unit: Option<TimestampUnit>,
    as_days: bool,
) -> ResolvedTemporal {
    let normalized = normalize_epoch_column(array.as_ref());
    let array = normalized.as_ref().unwrap_or(array);
    let (format, unit) = match (declared_format, declared_unit) {
        (None, None) => match detect_temporal_column(array.as_ref()) {
            Some(TimestampDetection::Format(fmt)) => (Some(fmt), None),
            Some(TimestampDetection::Unit(unit)) => (None, Some(unit)),
            None => (None, None),
        },
        (fmt, unit) => (fmt.map(str::to_string), unit),
    };
    let hints = diskgraph_rowpack::RowAccessHints {
        timestamp_format: format.clone(),
        timestamp_unit: unit.map(TimestampUnit::to_arrow),
        ..Default::default()
    };
    let mut access = diskgraph_rowpack::RowAccess::new(
        vec![hints],
        diskgraph_rowpack::RowAccessStats::default(),
    );
    access.set_cols(std::slice::from_ref(array));
    let epochs = (0..array.len())
        .map(|row| {
            access.set_row(row);
            if as_days {
                access.get_date(0).ok().flatten().map(i64::from)
            } else {
                access.get_timestamp(0).ok().flatten()
            }
        })
        .collect();
    ResolvedTemporal {
        epochs,
        format,
        unit,
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
/// Order is the tie-break: on equal hit counts the earlier candidate wins.
pub const MULTICAT_DELIMITER_CANDIDATES: &[&str] = &[",", "|", ";", "\t"];

/// Minimum sample support for auto-detecting a delimiter. Below this floor,
/// punctuation is treated as noise and the column stays single-token-per-row.
const MULTICAT_DETECT_MIN_FRACTION: f64 = 0.01;
/// Suspicion threshold for explicitly-declared delimiters. Explicit configs
/// are accepted if the delimiter appears at least once, but hit rates below
/// this threshold get a warning.
const MULTICAT_EXPLICIT_SUSPICIOUS_FRACTION: f64 = 0.20;
/// Max number of non-null sample rows used for automatic detection.
const MULTICAT_MAX_SAMPLES: usize = 1_000;
/// Explicit declarations need a larger bounded sample so sparse valid values
/// are not rejected by the stricter zero-hit mismatch check.
const MULTICAT_EXPLICIT_MAX_SAMPLES: usize = 10_000;
/// Max number of files to inspect while looking for multicat delimiter samples.
const MULTICAT_MAX_FILES: usize = 15;

struct ExplicitMulticatCandidate {
    name: String,
    declared: String,
    total: usize,
    hits: usize,
    miss_sample: Option<String>,
}

/// Bounded, transport-independent validation folded into the normal Encode
/// pass. Once each candidate has 10,000 non-empty values, observation is a
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
            .all(|candidate| candidate.total >= MULTICAT_EXPLICIT_MAX_SAMPLES)
    }

    pub(crate) fn observe(&mut self, batch: &arrow_array::RecordBatch) {
        for candidate in &mut self.candidates {
            if candidate.total >= MULTICAT_EXPLICIT_MAX_SAMPLES {
                continue;
            }
            let Ok(array_index) = batch.schema().index_of(&candidate.name) else {
                continue;
            };
            let array = batch.column(array_index);
            for row in 0..batch.num_rows() {
                if candidate.total >= MULTICAT_EXPLICIT_MAX_SAMPLES {
                    break;
                }
                if array.is_null(row) {
                    continue;
                }
                let Some(text) = string_value(array.as_ref(), row) else {
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
/// - Every non-empty sampled row is `[...]` → `delimiter = None`; bracketed
///   JSON-like lists require a native list dtype, not a scalar delimiter.
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

        let mut per_col: HashMap<usize, MulticatSamples> = HashMap::new();

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
                            let samples = per_col.entry(col_key).or_default();
                            if samples.is_full() {
                                continue;
                            }
                            let array = batch.column(*arr_idx);
                            for row in 0..batch.num_rows() {
                                if samples.is_full() {
                                    break;
                                }
                                if array.is_null(row) {
                                    continue;
                                }
                                let Some(text) = string_value(array.as_ref(), row) else {
                                    continue;
                                };
                                samples.observe(text);
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
            let all_done = candidates
                .iter()
                .all(|(col_idx, _)| per_col.get(col_idx).is_some_and(MulticatSamples::is_full));
            if all_done {
                break;
            }
        }

        // Decision per candidate column.
        for (col_idx, col_name) in &candidates {
            let samples = per_col.remove(col_idx).unwrap_or_default();
            match samples.decide() {
                MulticatDelimiterDecision::NoDelimiter => {
                    // Leave `multicategorical_delimiter = None`.
                    if samples.total == 0 {
                        tracing::warn!(
                            table = %table.name, column = %col_name,
                            "multicategorical column was entirely null in sampled file — \
                             treating as single-token-per-row",
                        );
                    } else if samples.is_bracketed_encoding() {
                        tracing::warn!(
                            table = %table.name, column = %col_name,
                            "multicategorical column is a bracketed JSON-like list, which is \
                             not a supported scalar encoding — treating as single-token-per-row",
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

/// The delimiter for one column of sampled rows, or `None`.
pub fn detect_multicat_delimiter<'a>(
    rows: impl IntoIterator<Item = &'a str>,
) -> Option<&'static str> {
    let mut samples = MulticatSamples::default();
    for row in rows {
        samples.observe(row);
    }
    match samples.decide() {
        MulticatDelimiterDecision::Winner(delimiter) => Some(delimiter),
        MulticatDelimiterDecision::NoDelimiter => None,
    }
}

/// Whether an already-chosen delimiter clears the same support floor. Several
/// delimiters can clear it at once, so this is not "is it the winner".
pub fn multicat_delimiter_is_proven<'a>(
    rows: impl IntoIterator<Item = &'a str>,
    delimiter: &str,
) -> bool {
    let mut samples = MulticatSamples::default();
    for row in rows {
        samples.observe(row);
    }
    samples.proves(delimiter)
}

#[derive(Debug)]
enum MulticatDelimiterDecision {
    NoDelimiter,
    Winner(&'static str),
}

#[derive(Default)]
struct MulticatSamples {
    total: usize,
    non_empty: usize,
    bracketed: usize,
    hits: HashMap<&'static str, usize>,
}

impl MulticatSamples {
    fn observe(&mut self, text: &str) {
        self.total += 1;
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            self.non_empty += 1;
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                self.bracketed += 1;
            }
        }
        for cand in MULTICAT_DELIMITER_CANDIDATES {
            if text.contains(cand) {
                *self.hits.entry(*cand).or_insert(0) += 1;
            }
        }
    }

    fn is_full(&self) -> bool {
        self.total >= MULTICAT_MAX_SAMPLES
    }

    /// Splitting a JSON-like list would leave brackets and quotes in the tokens.
    fn is_bracketed_encoding(&self) -> bool {
        self.non_empty > 0 && self.bracketed == self.non_empty
    }

    /// Minimum hit count a delimiter needs, or `None` when nothing can be
    /// proven against these rows.
    fn min_count(&self) -> Option<usize> {
        (self.total > 0 && !self.is_bracketed_encoding())
            .then(|| (MULTICAT_DETECT_MIN_FRACTION * self.total as f64).ceil() as usize)
    }

    fn proves(&self, delimiter: &str) -> bool {
        self.min_count()
            .is_some_and(|min| self.hits.get(delimiter).copied().unwrap_or(0) >= min)
    }

    fn decide(&self) -> MulticatDelimiterDecision {
        let Some(min_count) = self.min_count() else {
            return MulticatDelimiterDecision::NoDelimiter;
        };

        let mut best: Option<(&'static str, usize)> = None;
        for cand in MULTICAT_DELIMITER_CANDIDATES {
            let count = self.hits.get(cand).copied().unwrap_or(0);
            if count == 0 {
                continue;
            }
            match best {
                Some((_, best_count)) if best_count >= count => {}
                _ => best = Some((cand, count)),
            }
        }

        match best {
            Some((winner, count)) if count >= min_count => {
                MulticatDelimiterDecision::Winner(winner)
            }
            _ => MulticatDelimiterDecision::NoDelimiter,
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
    // These cover the threshold and encoding logic in `MulticatSamples`
    // without parquet fixtures. End-to-end detection against real parquet
    // is exercised by the integration tests in `diskgraph-ingest/tests/`.
    // =====================================================================

    fn counted(total: usize, pairs: &[(&'static str, usize)]) -> MulticatSamples {
        MulticatSamples {
            total,
            non_empty: total,
            hits: pairs.iter().copied().collect(),
            ..Default::default()
        }
    }

    fn observed(rows: &[&str]) -> MulticatSamples {
        let mut samples = MulticatSamples::default();
        for row in rows {
            samples.observe(row);
        }
        samples
    }

    #[test]
    fn multicat_decide_zero_samples_is_no_delimiter() {
        let s = counted(0, &[]);
        assert!(matches!(s.decide(), MulticatDelimiterDecision::NoDelimiter));
    }

    #[test]
    fn multicat_decide_clear_majority_comma_wins() {
        // 80 / 100 rows have comma, none have others.
        let s = counted(100, &[(",", 80)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_pipe_wins_against_noise() {
        // 60% pipe, 0.9% comma: the most common observed delimiter wins.
        let s = counted(1000, &[("|", 600), (",", 9)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner("|") => {}
            other => panic!("expected Winner(\"|\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_picks_most_common_candidate() {
        // Comma in 50% of rows and pipe in 30%: comma wins because it is
        // most common.
        let s = counted(100, &[(",", 50), ("|", 30)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_single_sparse_candidate_below_50pct_wins() {
        // Sparse multicategorical columns are valid: if exactly one delimiter
        // is observed, it should win even if most rows are single-token.
        let s = counted(100, &[(",", 30)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_single_observed_candidate_wins() {
        // A delimiter at the 1% floor wins if it is the most common candidate.
        let s = counted(1000, &[(",", 10)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_observed_candidate_below_one_percent_is_no_delimiter() {
        let s = counted(1000, &[(",", 9)]);
        assert!(matches!(s.decide(), MulticatDelimiterDecision::NoDelimiter));
    }

    #[test]
    fn multicat_decide_competing_candidates_pick_highest_count() {
        let s = counted(1000, &[(",", 10), ("|", 11)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner("|") => {}
            other => panic!("expected Winner(\"|\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_tie_uses_candidate_order() {
        let s = counted(1000, &[(",", 10), ("|", 10)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_decide_no_observed_candidate_is_no_delimiter() {
        let s = counted(1000, &[]);
        assert!(matches!(s.decide(), MulticatDelimiterDecision::NoDelimiter));
    }

    #[test]
    fn multicat_decide_thresholds_scale_with_total() {
        // At very small sample counts, the ceil() in the 1% floor matters.
        // 3 rows, 2 with comma, 0 others → winner.
        let s = counted(3, &[(",", 2)]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_bracketed_json_rows_are_no_delimiter() {
        let s = observed(&[r#"["a","b"]"#, r#"["c"]"#, r#"["d","e","f"]"#]);
        assert!(matches!(s.decide(), MulticatDelimiterDecision::NoDelimiter));
    }

    #[test]
    fn multicat_bracketed_rows_ignore_empty_and_whitespace() {
        let s = observed(&[r#"  ["a","b"]  "#, "", "   ", r#"["c","d"]"#]);
        assert!(matches!(s.decide(), MulticatDelimiterDecision::NoDelimiter));
    }

    #[test]
    fn multicat_one_unbracketed_row_keeps_detection() {
        let s = observed(&[r#"["a","b"]"#, "c,d"]);
        match s.decide() {
            MulticatDelimiterDecision::Winner(",") => {}
            other => panic!("expected Winner(\",\"), got {:?}", other),
        }
    }

    #[test]
    fn multicat_all_empty_rows_are_not_bracketed() {
        let s = observed(&["", "  "]);
        assert!(!s.is_bracketed_encoding());
    }

    #[test]
    fn multicat_plain_rows_still_detect_comma() {
        let s = observed(&["a,b", "c,d", "e"]);
        match s.decide() {
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

#[cfg(test)]
mod tests_resolve {
    use super::*;
    use arrow_array::ArrayRef;
    use std::sync::Arc;

    const JUL_4_2019: i64 = 1_562_198_400;

    fn strings(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(arrow_array::StringArray::from(values.to_vec()))
    }

    fn ints(values: &[Option<i64>]) -> ArrayRef {
        Arc::new(arrow_array::Int64Array::from(values.to_vec()))
    }

    #[test]
    fn a_tie_resolves_the_same_way_whatever_order_the_votes_arrive_in() {
        let s = "unit:Seconds".to_string();
        let ms = "unit:Millis".to_string();
        for votes in [
            vec![s.clone(), s.clone(), ms.clone(), ms.clone()],
            vec![ms.clone(), ms.clone(), s.clone(), s.clone()],
        ] {
            assert!(matches!(
                majority_vote(&votes),
                Some(TimestampDetection::Unit(TimestampUnit::Seconds))
            ));
        }
    }

    #[test]
    fn a_tie_still_prefers_a_format_over_a_unit() {
        let votes = vec!["unit:Seconds".to_string(), "fmt:%Y-%m-%d".to_string()];
        assert!(matches!(
            majority_vote(&votes),
            Some(TimestampDetection::Format(_))
        ));
    }

    #[test]
    fn an_even_split_of_second_and_millisecond_epochs_reads_seconds() {
        // Resolving to millis would silently read the 10-digit values as 1970.
        let array = ints(&[
            Some(1_659_304_801),
            Some(1_659_304_901),
            Some(1_659_304_801_388),
            Some(1_659_304_901_388),
        ]);
        let r = resolve(&array, None, None);
        assert_eq!(r.unit, Some(TimestampUnit::Seconds));
    }

    fn resolve(
        array: &ArrayRef,
        fmt: Option<&str>,
        unit: Option<TimestampUnit>,
    ) -> ResolvedTemporal {
        resolve_temporal(array, fmt, unit, false)
    }

    #[test]
    fn resolve_detects_string_format_and_parses() {
        let array = strings(&[Some("2019-07-04 00:00:00"), None]);
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None]);
        assert_eq!(r.format.as_deref(), Some("%Y-%m-%d %H:%M:%S"));
        assert_eq!(r.unit, None);
    }

    /// kumo#30447: an offset format must not null a `Z`-suffixed value.
    #[test]
    fn resolve_zulu_value_under_offset_format() {
        let array = strings(&[Some("2019-07-04T00:00:00Z")]);
        let r = resolve(&array, Some("%Y-%m-%dT%H:%M:%S%z"), None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019)]);
        assert_eq!(r.format.as_deref(), Some("%Y-%m-%dT%H:%M:%S%z"));
    }

    #[test]
    fn resolve_honors_declared_format_over_detection() {
        let array = strings(&[Some("07/04/2019")]);
        let r = resolve(&array, Some("%m/%d/%Y"), None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019)]);
        assert_eq!(r.format.as_deref(), Some("%m/%d/%Y"));
    }

    #[test]
    fn resolve_detects_epoch_unit_from_magnitude() {
        let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![
            Some(JUL_4_2019 * 1_000),
            None,
        ]));
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None]);
        assert_eq!(r.unit, Some(TimestampUnit::Millis));
        assert_eq!(r.format, None);
    }

    #[test]
    fn resolve_honors_declared_unit_over_magnitude() {
        let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![JUL_4_2019 * 1_000]));
        let r = resolve(&array, None, Some(TimestampUnit::Micros));
        assert_eq!(r.epochs, vec![Some(1_562_199)]);
        assert_eq!(r.unit, Some(TimestampUnit::Micros));
    }

    #[test]
    fn resolve_detects_encoded_int_date() {
        let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![20190704]));
        let r = resolve(&array, None, None);
        assert_eq!(r.format.as_deref(), Some("%Y%m%d"));
        assert_eq!(r.unit, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019)]);
    }

    #[test]
    fn resolve_native_column_carries_no_hint() {
        let array: ArrayRef = Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
            Some(JUL_4_2019 * 1_000_000),
            None,
        ]));
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None]);
        assert_eq!(r.format, None);
        assert_eq!(r.unit, None);
    }

    #[test]
    fn resolve_all_null_column_yields_no_hint_and_no_epochs() {
        let array = strings(&[None, None]);
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![None, None]);
        assert_eq!(r.format, None);
        assert_eq!(r.unit, None);
    }

    #[test]
    fn resolve_unparseable_value_is_none_not_zero() {
        let array = strings(&[Some("2019-07-04"), Some("not a date")]);
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None]);
    }

    /// A run of sentinels at the head of a column must not exhaust the sample
    /// budget and hide the timestamps behind it.
    #[test]
    fn resolve_looks_past_leading_sentinels() {
        let mut values = vec![Some("0"); TEMPORAL_MAX_VOTES * 3];
        values.push(Some("2019-07-04"));
        let r = resolve(&strings(&values), None, None);
        assert_eq!(r.format.as_deref(), Some("%Y-%m-%d"));
    }

    #[test]
    fn resolve_gives_up_on_a_column_of_pure_sentinels() {
        let values = vec![Some("0"); TEMPORAL_MAX_VOTES * 3];
        let r = resolve(&strings(&values), None, None);
        assert_eq!(r.format, None);
        assert_eq!(r.unit, None);
    }

    /// A colonless offset must not be swallowed as a sub-second fraction.
    /// (The resolved token keeps the space separator: the fast paths accept
    /// either separator, so detection reports the first equivalent format.)
    #[test]
    fn resolve_colonless_offset_keeps_the_offset() {
        let array = strings(&[Some("2019-07-04T00:00:00-0530")]);
        let r = resolve(&array, None, None);
        assert!(r.format.as_deref().is_some_and(|f| f.ends_with("%z")));
        assert_eq!(r.epochs, vec![Some(JUL_4_2019 + 5 * 3600 + 1800)]);
    }

    #[test]
    fn resolve_float_epoch_column() {
        let array: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![
            Some(JUL_4_2019 as f64 * 1_000.0),
            Some(f64::NAN),
            None,
        ]));
        let r = resolve(&array, None, None);
        assert_eq!(r.unit, Some(TimestampUnit::Millis));
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None, None]);
    }

    /// pandas encodes NaT as `i64::MIN`; it must not decode to year 1677.
    #[test]
    fn resolve_nat_sentinel_is_null_not_an_epoch() {
        let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![
            Some(JUL_4_2019),
            Some(i64::MIN),
        ]));
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None]);
        assert_eq!(r.unit, Some(TimestampUnit::Seconds));
    }

    /// A float too large for `i64` must be null, not a saturated cast.
    #[test]
    fn resolve_out_of_range_float_is_null() {
        let array: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![
            Some(JUL_4_2019 as f64),
            Some(1e300),
            Some(f64::INFINITY),
        ]));
        let r = resolve(&array, None, None);
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), None, None]);
    }

    /// A declared unit must reach the apply path even when a format is also
    /// declared, exactly as `build_row_access_hints` passes both to ingest.
    #[test]
    fn resolve_applies_declared_unit_alongside_declared_format() {
        let array: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![JUL_4_2019 * 1_000]));
        let r = resolve(
            &array,
            Some("%Y-%m-%d %H:%M:%S"),
            Some(TimestampUnit::Millis),
        );
        assert_eq!(r.epochs, vec![Some(JUL_4_2019)]);
        assert_eq!(r.unit, Some(TimestampUnit::Millis));
    }

    /// A 10-digit epoch-seconds string is an epoch, not a `%Y-%m-%d` date.
    #[test]
    fn resolve_string_epoch_column_is_a_unit_not_a_date_format() {
        let array = strings(&[Some("1562198400"), Some("1562284800")]);
        let r = resolve_temporal(&array, None, None, false);
        assert_eq!(r.format, None);
        assert_eq!(r.unit, Some(TimestampUnit::Seconds));
        assert_eq!(r.epochs, vec![Some(JUL_4_2019), Some(JUL_4_2019 + 86_400)]);
    }

    /// `as_days` routes to the date path; a partial day still rounds up.
    #[test]
    fn resolve_days_returns_epoch_days() {
        let array = strings(&[
            Some("2019-07-04 00:00:00"),
            Some("2019-07-04 12:00:00"),
            None,
        ]);
        let r = resolve_temporal(&array, None, None, true);
        let day = JUL_4_2019 / 86_400;
        assert_eq!(r.epochs, vec![Some(day), Some(day + 1), None]);
        assert_eq!(r.format.as_deref(), Some("%Y-%m-%d %H:%M:%S"));
    }

    #[test]
    fn detect_multicat_delimiter_picks_the_supported_candidate() {
        assert_eq!(detect_multicat_delimiter(["a,b", "c,d"]), Some(","));
        assert_eq!(detect_multicat_delimiter(["a|b", "c|d"]), Some("|"));
        assert_eq!(detect_multicat_delimiter(["a", "b"]), None);
        assert_eq!(detect_multicat_delimiter(["[\"a\", \"b\"]"]), None);
    }

    /// A loser can still be proven: proof is the support floor, not the vote.
    #[test]
    fn multicat_delimiter_is_proven_accepts_a_non_winning_candidate() {
        let rows = ["a,b|c", "d,e"];
        assert_eq!(detect_multicat_delimiter(rows), Some(","));
        assert!(multicat_delimiter_is_proven(rows, "|"));
        assert!(!multicat_delimiter_is_proven(rows, ";"));
        assert!(!multicat_delimiter_is_proven(rows, ":"));
        assert!(!multicat_delimiter_is_proven(["[\"a\", \"b\"]"], ","));
        assert!(!multicat_delimiter_is_proven([], ","));
    }
}
