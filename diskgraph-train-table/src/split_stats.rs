//! Per-stage column-stats accumulator: observes TIMESTAMP/TARGET1 into
//! Welford / TDigest / HLL accumulators keyed by stage, alongside the fused
//! pass-B routing.
//!
//! Concurrency: each rayon worker owns a `PerStageStatsAccumulator` (heap state
//! = four `PerStageStats` slots); after the parallel scope the orchestrator
//! folds them via `merge`. No sync primitives during workers.
//!
//! Buckets: `train / val / test / unused` = `0 / 1 / 2 / 3` (rows routed
//! nowhere count as `unused`). [`build_split_stats_payload`] serializes to the
//! `tt_split_stats_v1` JSON sidecar (currently unused by TT-NG gen, retained
//! for the DG-main path).

use std::path::{Path, PathBuf};

use arrow_array::{
    Array, Float64Array, Int64Array, ListArray, RecordBatch, StringArray, TimestampNanosecondArray,
};
use diskgraph_core::accumulators::{HyperLogLog, TDigestEstimator, WelfordAccumulator};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use serde_json::{json, Value as JsonValue};

use crate::errors::{TTGenError, TTGenResult};
use crate::manifest::sha256_file;
use crate::pass_b::{StagePartsOut, TIMESTAMP_COL};
use crate::ttgen_resolve::FUTURE_LABELS_OUTPUT_ALIAS;

const NUM_HISTOGRAM_BINS: usize = 50;
const ENTITY_COL: &str = "ENTITY1";

struct KumoStageStats {
    row_count: u64,
    entity_hll: HyperLogLog,
    target_hll: HyperLogLog,
    target_welford: WelfordAccumulator,
    target_tdigest: TDigestEstimator,
    timestamp_min: Option<i64>,
    timestamp_max: Option<i64>,
}

impl Default for KumoStageStats {
    fn default() -> Self {
        Self {
            row_count: 0,
            entity_hll: HyperLogLog::new(),
            target_hll: HyperLogLog::new(),
            target_welford: WelfordAccumulator::new(),
            target_tdigest: TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA),
            timestamp_min: None,
            timestamp_max: None,
        }
    }
}

impl KumoStageStats {
    fn observe_target(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.target_welford.observe(value);
        self.target_tdigest.observe(value);
        self.target_hll.observe_f64(value);
    }

    fn observe_timestamp(&mut self, value: i64) {
        self.timestamp_min = Some(self.timestamp_min.map_or(value, |v| v.min(value)));
        self.timestamp_max = Some(self.timestamp_max.map_or(value, |v| v.max(value)));
    }
}

pub fn write_forecasting_split_stats_artifact(
    output_dir: &Path,
    stage_parts: &[StagePartsOut; 3],
) -> TTGenResult<Option<(PathBuf, String)>> {
    let mut stages = [
        KumoStageStats::default(),
        KumoStageStats::default(),
        KumoStageStats::default(),
    ];
    for (stage, parts) in stage_parts.iter().enumerate() {
        observe_stage(output_dir, parts, &mut stages[stage])?;
    }
    if stages.iter().all(|s| s.row_count == 0) {
        return Ok(None);
    }

    let histograms = [
        build_stage_histogram(output_dir, &stage_parts[0], &stages[0])?,
        build_stage_histogram(output_dir, &stage_parts[1], &stages[1])?,
        build_stage_histogram(output_dir, &stage_parts[2], &stages[2])?,
    ];
    let payload = json!({
        "version": 1,
        "kind": "tt_split_stats",
        "train_table_stats": stage_to_kumo_json(&stages[0], &histograms[0], true),
        "val_table_stats": stage_to_kumo_json(&stages[1], &histograms[1], false),
        "test_table_stats": stage_to_kumo_json(&stages[2], &histograms[2], false),
    });
    let path = output_dir.join("split_stats.json");
    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| TTGenError::SystemError(format!("split-stats serialize: {e}")))?;
    std::fs::write(&path, json)?;
    let sha = sha256_file(&path)?;
    Ok(Some((path, sha)))
}

fn observe_stage(
    output_dir: &Path,
    parts: &StagePartsOut,
    stats: &mut KumoStageStats,
) -> TTGenResult<()> {
    if parts.row_count == 0 {
        return Ok(());
    }
    for part in &parts.parts {
        let path = output_dir.join(part);
        let reader = projected_reader(
            &path,
            &[ENTITY_COL, TIMESTAMP_COL, FUTURE_LABELS_OUTPUT_ALIAS],
        )?;
        for batch in reader {
            observe_batch(
                &batch.map_err(|e| TTGenError::Parquet(e.to_string()))?,
                stats,
            )?;
        }
    }
    Ok(())
}

fn observe_batch(batch: &RecordBatch, stats: &mut KumoStageStats) -> TTGenResult<()> {
    stats.row_count += batch.num_rows() as u64;
    observe_entities(batch, stats)?;
    observe_timestamps(batch, stats)?;
    let label_idx = batch
        .schema()
        .index_of(FUTURE_LABELS_OUTPUT_ALIAS)
        .map_err(|_| {
            TTGenError::SchemaError(format!(
                "forecasting split stats require {FUTURE_LABELS_OUTPUT_ALIAS}"
            ))
        })?;
    for_each_label(batch.column(label_idx).as_ref(), |value| {
        stats.observe_target(value)
    })
}

fn observe_entities(batch: &RecordBatch, stats: &mut KumoStageStats) -> TTGenResult<()> {
    let idx = batch
        .schema()
        .index_of(ENTITY_COL)
        .map_err(|_| TTGenError::SchemaError(format!("split stats require {ENTITY_COL}")))?;
    let arr = batch.column(idx);
    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        for i in 0..a.len() {
            if !a.is_null(i) {
                stats.entity_hll.observe_i64(a.value(i));
            }
        }
        return Ok(());
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        for i in 0..a.len() {
            if !a.is_null(i) {
                stats.entity_hll.observe_bytes(a.value(i).as_bytes());
            }
        }
        return Ok(());
    }
    Err(TTGenError::SchemaError(format!(
        "unsupported {ENTITY_COL} dtype for split stats: {:?}",
        arr.data_type()
    )))
}

fn observe_timestamps(batch: &RecordBatch, stats: &mut KumoStageStats) -> TTGenResult<()> {
    let idx = batch
        .schema()
        .index_of(TIMESTAMP_COL)
        .map_err(|_| TTGenError::SchemaError(format!("split stats require {TIMESTAMP_COL}")))?;
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| {
            TTGenError::SchemaError(format!(
                "{TIMESTAMP_COL} must be timestamp[ns] for split stats"
            ))
        })?;
    for i in 0..arr.len() {
        if !arr.is_null(i) {
            stats.observe_timestamp(arr.value(i));
        }
    }
    Ok(())
}

fn for_each_label<F>(arr: &dyn Array, mut f: F) -> TTGenResult<()>
where
    F: FnMut(f64),
{
    let list = arr.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
        TTGenError::SchemaError(format!(
            "{FUTURE_LABELS_OUTPUT_ALIAS} must be List for split stats"
        ))
    })?;
    for row in 0..list.len() {
        if list.is_null(row) {
            continue;
        }
        let values = list.value(row);
        if let Some(a) = values.as_any().downcast_ref::<Float64Array>() {
            for i in 0..a.len() {
                if !a.is_null(i) {
                    f(a.value(i));
                }
            }
        } else if let Some(a) = values.as_any().downcast_ref::<Int64Array>() {
            for i in 0..a.len() {
                if !a.is_null(i) {
                    f(a.value(i) as f64);
                }
            }
        } else {
            return Err(TTGenError::SchemaError(format!(
                "unsupported {FUTURE_LABELS_OUTPUT_ALIAS} value dtype: {:?}",
                values.data_type()
            )));
        }
    }
    Ok(())
}

fn build_stage_histogram(
    output_dir: &Path,
    parts: &StagePartsOut,
    stats: &KumoStageStats,
) -> TTGenResult<Vec<u64>> {
    let count = stats.target_welford.count as usize;
    let mut counts = vec![0u64; NUM_HISTOGRAM_BINS];
    if count == 0 || parts.row_count == 0 {
        return Ok(counts);
    }
    let min = stats.target_welford.min;
    let width = histogram_bin_width(stats);
    for part in &parts.parts {
        let reader = projected_reader(
            &output_dir.join(part),
            &[FUTURE_LABELS_OUTPUT_ALIAS],
        )?;
        for batch in reader {
            let batch = batch.map_err(|e| TTGenError::Parquet(e.to_string()))?;
            let idx = batch
                .schema()
                .index_of(FUTURE_LABELS_OUTPUT_ALIAS)
                .map_err(|_| {
                    TTGenError::SchemaError(format!(
                        "forecasting split stats require {FUTURE_LABELS_OUTPUT_ALIAS}"
                    ))
                })?;
            for_each_label(batch.column(idx).as_ref(), |value| {
                if value.is_finite() {
                    let raw = ((value - min) / width).floor() as isize;
                    let idx = raw.clamp(0, NUM_HISTOGRAM_BINS as isize - 1) as usize;
                    counts[idx] += 1;
                }
            })?;
        }
    }
    Ok(counts)
}

fn projected_reader(
    path: &Path,
    columns: &[&str],
) -> TTGenResult<ParquetRecordBatchReader> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut indices = Vec::with_capacity(columns.len());
    for col in columns {
        let idx = builder.schema().index_of(col).map_err(|_| {
            TTGenError::SchemaError(format!(
                "split stats require {col} in {}",
                path.display()
            ))
        })?;
        indices.push(idx);
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    Ok(builder.with_projection(mask).build()?)
}

fn stage_to_kumo_json(
    stats: &KumoStageStats,
    histogram_counts: &[u64],
    required: bool,
) -> JsonValue {
    if stats.row_count == 0 && !required {
        return JsonValue::Null;
    }
    let target_count = stats.target_welford.count;
    let mut td = stats.target_tdigest.clone();
    let p25 = if target_count > 0 {
        td.quantile_at(0.25)
    } else {
        0.0
    };
    let p50 = if target_count > 0 {
        td.quantile_at(0.50)
    } else {
        0.0
    };
    let p75 = if target_count > 0 {
        td.quantile_at(0.75)
    } else {
        0.0
    };
    let min = if target_count > 0 {
        stats.target_welford.min
    } else {
        0.0
    };
    let max = if target_count > 0 {
        stats.target_welford.max
    } else {
        0.0
    };
    let width = histogram_bin_width(stats);
    let bins: Vec<f64> = (0..=NUM_HISTOGRAM_BINS)
        .map(|i| min + width * i as f64)
        .collect();
    let (mode_bin, mode_count) = histogram_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(idx, count)| (idx, *count))
        .unwrap_or((0, 0));
    let mode = min + width * (mode_bin as f64 + 0.5);
    json!({
        "num_train_rows": stats.row_count,
        "num_entities": stats.entity_hll.estimate() as u64,
        "num_targets": stats.target_hll.estimate() as u64,
        "min_date": stats.timestamp_min.map(ns_to_iso),
        "max_date": stats.timestamp_max.map(ns_to_iso),
        "anchor_date": stats.timestamp_min.map(ns_to_iso),
        "target_distribution": {
            "target_mean": if target_count > 0 { stats.target_welford.mean() } else { 0.0 },
            "target_min": min,
            "target_max": max,
            "target_percentile_25": p25,
            "target_percentile_75": p75,
            "target_median": p50,
            "target_mode": mode,
            "target_mode_percentage": if target_count > 0 {
                mode_count as f64 / target_count as f64
            } else {
                0.0
            },
            "target_mode_count": mode_count,
            "histogram": {
                "bins": bins,
                "counts": histogram_counts,
                "bin_width": width,
                "num_bins": NUM_HISTOGRAM_BINS,
            },
        },
    })
}

fn histogram_bin_width(stats: &KumoStageStats) -> f64 {
    if stats.target_welford.count == 0 {
        return 1.0 / NUM_HISTOGRAM_BINS as f64;
    }
    let range = stats.target_welford.max - stats.target_welford.min;
    if range.is_finite() && range > 0.0 {
        range / NUM_HISTOGRAM_BINS as f64
    } else {
        1.0 / NUM_HISTOGRAM_BINS as f64
    }
}

fn ns_to_iso(ns: i64) -> String {
    let secs = ns.div_euclid(1_000_000_000);
    let nanos = ns.rem_euclid(1_000_000_000) as u32;
    chrono::DateTime::from_timestamp(secs, nanos)
        .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
        .unwrap_or_else(|| ns.to_string())
}

/// Streaming accumulators for a single column of a single stage.
pub struct ColumnAccumulator {
    pub welford: WelfordAccumulator,
    pub tdigest: TDigestEstimator,
    pub hll: HyperLogLog,
    pub null_count: u64,
}

impl Default for ColumnAccumulator {
    fn default() -> Self {
        Self {
            welford: WelfordAccumulator::new(),
            tdigest: TDigestEstimator::new(TDigestEstimator::DEFAULT_DELTA),
            hll: HyperLogLog::new(),
            null_count: 0,
        }
    }
}

impl ColumnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a finite f64. Non-finite values (NaN/±Inf) are dropped by
    /// Welford and TDigest (their `observe` already filters); HLL hashes
    /// the bit pattern unconditionally so the unique-count includes them.
    /// We mirror the Welford/TDigest behaviour and skip non-finite for
    /// HLL too — keeps the unique-count tied to what made it into the
    /// distribution stats.
    #[inline]
    pub fn observe_f64(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.welford.observe(value);
        self.tdigest.observe(value);
        self.hll.observe_f64(value);
    }

    /// Observe a single null. Increments only `null_count`.
    #[inline]
    pub fn observe_null(&mut self) {
        self.null_count += 1;
    }

    /// Merge another accumulator into this one (sequential fold after the
    /// rayon scope; no sync primitives).
    pub fn merge(&mut self, other: ColumnAccumulator) {
        let ColumnAccumulator {
            welford,
            mut tdigest,
            hll,
            null_count,
        } = other;
        self.welford.merge(&welford);
        self.tdigest.merge(&mut tdigest);
        self.hll.merge(&hll);
        self.null_count += null_count;
    }
}

/// Per-stage stats: row count + accumulators for the columns of interest.
///
/// Starts with TARGET (TARGET1, when present in tt.parquet) and
/// TIMESTAMP. Other columns can be added later; the schema below carries
/// only those two top-level keys.
pub struct PerStageStats {
    pub row_count: u64,
    pub target: ColumnAccumulator,
    pub target_present: bool,
    pub timestamp: ColumnAccumulator,
}

impl Default for PerStageStats {
    fn default() -> Self {
        Self {
            row_count: 0,
            target: ColumnAccumulator::new(),
            target_present: false,
            timestamp: ColumnAccumulator::new(),
        }
    }
}

impl PerStageStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&mut self, other: PerStageStats) {
        let PerStageStats {
            row_count,
            target,
            target_present,
            timestamp,
        } = other;
        self.row_count += row_count;
        self.target.merge(target);
        self.target_present = self.target_present || target_present;
        self.timestamp.merge(timestamp);
    }
}

/// Per-worker accumulator: 4 stage buckets keyed by stage byte
/// (0=train, 1=val, 2=test, 3=unused — anything outside the alphabet).
///
/// Workers fold their per-batch values into the appropriate bucket;
/// after the rayon scope the orchestrator merges per-worker
/// accumulators into a single global one.
pub struct PerStageStatsAccumulator {
    pub per_stage: [PerStageStats; 4],
}

impl Default for PerStageStatsAccumulator {
    fn default() -> Self {
        Self {
            per_stage: [
                PerStageStats::new(),
                PerStageStats::new(),
                PerStageStats::new(),
                PerStageStats::new(),
            ],
        }
    }
}

impl PerStageStatsAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bucket index for a stage byte. Unknown bytes bucket to 3 (unused).
    #[inline]
    pub fn bucket(stage: u8) -> usize {
        if stage <= 2 {
            stage as usize
        } else {
            3
        }
    }

    /// Merge another accumulator into this one, slot-by-slot.
    pub fn merge(&mut self, other: PerStageStatsAccumulator) {
        let PerStageStatsAccumulator { per_stage } = other;
        let [a, b, c, d] = per_stage;
        let merged = [a, b, c, d];
        for (lhs, rhs) in self.per_stage.iter_mut().zip(merged) {
            lhs.merge(rhs);
        }
    }
}

impl PerStageStatsAccumulator {
    /// Extract per-stage seed-time `(min, max)` of the TIMESTAMP column,
    /// keyed by stage name. Only `train`/`val`/`test` (indices 0/1/2) are
    /// considered; `unused` (index 3) is skipped. A stage is emitted only
    /// when its timestamp accumulator observed at least one non-null
    /// finite value (`welford.count > 0`). The f64 Welford min/max are
    /// rounded to nearest int64 (TIMESTAMP is canonically int64 ns; the
    /// f64 accumulator carries a tiny precision caveat at ns scale).
    ///
    /// Returns `None` when no stage carried a timestamp, so the manifest
    /// field elides cleanly.
    pub fn stage_seed_times(&self) -> Option<std::collections::HashMap<String, [i64; 2]>> {
        let mut map = std::collections::HashMap::new();
        for (idx, name) in [(0usize, "train"), (1, "val"), (2, "test")] {
            let col = &self.per_stage[idx].timestamp;
            if col.welford.count > 0 {
                let min = col.welford.min.round() as i64;
                let max = col.welford.max.round() as i64;
                map.insert(name.to_string(), [min, max]);
            }
        }
        if map.is_empty() {
            None
        } else {
            Some(map)
        }
    }
}

// ─── On-disk JSON schema (`tt_split_stats_v1`) ─────────────────────────────

/// Build the JSON payload for `split_stats.json`.
///
/// Schema (kind=`tt_split_stats`, format=`tt_split_stats_v1`):
/// ```text
/// {
///   "kind": "tt_split_stats",
///   "format": "tt_split_stats_v1",
///   "stages": {
///     "train":  { row_count, target?, timestamp },
///     "val":    { ... },
///     "test":   { ... },
///     "unused": { ... }
///   }
/// }
/// ```
///
/// Each per-column block carries: `mean`, `stddev`, `min`, `max`,
/// `quantiles` (q25/q50/q75 — `None` when accumulator was empty),
/// `num_unique` (HLL estimate, integer), and `null_count`. The `target`
/// block is elided when the column was not present in tt.parquet (e.g.
/// `OutputMode::PassThrough` plans).
pub fn build_split_stats_payload(accum: &PerStageStatsAccumulator) -> JsonValue {
    json!({
        "kind": "tt_split_stats",
        "format": "tt_split_stats_v1",
        "stages": {
            "train":  per_stage_to_json(&accum.per_stage[0]),
            "val":    per_stage_to_json(&accum.per_stage[1]),
            "test":   per_stage_to_json(&accum.per_stage[2]),
            "unused": per_stage_to_json(&accum.per_stage[3]),
        }
    })
}

fn per_stage_to_json(stats: &PerStageStats) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert("row_count".into(), json!(stats.row_count));
    if stats.target_present {
        obj.insert("target".into(), column_to_json(&stats.target));
    }
    obj.insert("timestamp".into(), column_to_json(&stats.timestamp));
    JsonValue::Object(obj)
}

fn column_to_json(col: &ColumnAccumulator) -> JsonValue {
    // Quantiles: TDigest needs `&mut` to flush its buffer. Clone first so
    // we never mutate the caller-visible accumulator during serialization.
    let mut td = col.tdigest.clone();
    let q = td.quantiles();
    let (mean, stddev, min, max) = if col.welford.count > 0 {
        (
            Some(col.welford.mean()),
            Some(col.welford.std_dev()),
            Some(col.welford.min),
            Some(col.welford.max),
        )
    } else {
        (None, None, None, None)
    };
    let num_unique = col.hll.estimate() as u64;
    json!({
        "count": col.welford.count,
        "null_count": col.null_count,
        "mean": mean,
        "stddev": stddev,
        "min": min,
        "max": max,
        "quantiles": q.map(|(p25, p50, p75)| json!({"q25": p25, "q50": p50, "q75": p75})),
        "num_unique": num_unique,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_accumulator_observe_basic() {
        let mut c = ColumnAccumulator::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            c.observe_f64(v);
        }
        assert_eq!(c.welford.count, 5);
        assert!((c.welford.mean() - 3.0).abs() < 1e-10);
        assert_eq!(c.null_count, 0);
    }

    #[test]
    fn test_column_accumulator_skip_non_finite() {
        let mut c = ColumnAccumulator::new();
        c.observe_f64(1.0);
        c.observe_f64(f64::NAN);
        c.observe_f64(f64::INFINITY);
        c.observe_f64(2.0);
        assert_eq!(c.welford.count, 2);
    }

    #[test]
    fn test_column_accumulator_observe_null() {
        let mut c = ColumnAccumulator::new();
        c.observe_f64(1.0);
        c.observe_null();
        c.observe_null();
        assert_eq!(c.welford.count, 1);
        assert_eq!(c.null_count, 2);
    }

    #[test]
    fn test_column_accumulator_merge() {
        let mut a = ColumnAccumulator::new();
        let mut b = ColumnAccumulator::new();
        for v in [1.0, 2.0, 3.0] {
            a.observe_f64(v);
        }
        for v in [4.0, 5.0] {
            b.observe_f64(v);
        }
        a.observe_null();
        b.observe_null();
        b.observe_null();
        a.merge(b);
        assert_eq!(a.welford.count, 5);
        assert_eq!(a.null_count, 3);
        assert!((a.welford.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_index() {
        assert_eq!(PerStageStatsAccumulator::bucket(0), 0);
        assert_eq!(PerStageStatsAccumulator::bucket(1), 1);
        assert_eq!(PerStageStatsAccumulator::bucket(2), 2);
        assert_eq!(PerStageStatsAccumulator::bucket(3), 3);
        assert_eq!(PerStageStatsAccumulator::bucket(255), 3);
        assert_eq!(PerStageStatsAccumulator::bucket(42), 3);
    }

    #[test]
    fn test_per_stage_stats_merge() {
        let mut a = PerStageStatsAccumulator::new();
        let mut b = PerStageStatsAccumulator::new();
        // a: 3 rows train, 1 row val
        a.per_stage[0].row_count = 3;
        a.per_stage[0].target.observe_f64(10.0);
        a.per_stage[0].target.observe_f64(20.0);
        a.per_stage[0].target_present = true;
        a.per_stage[0].timestamp.observe_f64(100.0);
        a.per_stage[0].timestamp.observe_f64(200.0);
        a.per_stage[0].timestamp.observe_f64(300.0);
        a.per_stage[1].row_count = 1;
        a.per_stage[1].timestamp.observe_f64(400.0);

        // b: 2 rows train (different values), 4 rows test
        b.per_stage[0].row_count = 2;
        b.per_stage[0].target.observe_f64(30.0);
        b.per_stage[0].target.observe_f64(40.0);
        b.per_stage[0].target_present = true;
        b.per_stage[0].timestamp.observe_f64(150.0);
        b.per_stage[0].timestamp.observe_f64(250.0);
        b.per_stage[2].row_count = 4;
        b.per_stage[2].timestamp.observe_f64(500.0);
        b.per_stage[2].timestamp.observe_f64(600.0);
        b.per_stage[2].timestamp.observe_f64(700.0);
        b.per_stage[2].timestamp.observe_f64(800.0);

        a.merge(b);
        // Train bucket: 5 rows, target = mean(10,20,30,40)=25
        assert_eq!(a.per_stage[0].row_count, 5);
        assert_eq!(a.per_stage[0].target.welford.count, 4);
        assert!((a.per_stage[0].target.welford.mean() - 25.0).abs() < 1e-10);
        assert_eq!(a.per_stage[0].timestamp.welford.count, 5);
        // mean(100,200,300,150,250) = 200
        assert!((a.per_stage[0].timestamp.welford.mean() - 200.0).abs() < 1e-10);
        // Val bucket: 1 row, timestamp = 400.0
        assert_eq!(a.per_stage[1].row_count, 1);
        assert_eq!(a.per_stage[1].timestamp.welford.count, 1);
        // Test bucket: 4 rows
        assert_eq!(a.per_stage[2].row_count, 4);
        assert_eq!(a.per_stage[2].timestamp.welford.count, 4);
        // Unused untouched
        assert_eq!(a.per_stage[3].row_count, 0);
    }

    #[test]
    fn test_build_split_stats_payload_basic() {
        let mut a = PerStageStatsAccumulator::new();
        a.per_stage[0].row_count = 3;
        a.per_stage[0].target_present = true;
        a.per_stage[0].target.observe_f64(10.0);
        a.per_stage[0].target.observe_f64(20.0);
        a.per_stage[0].target.observe_f64(30.0);
        a.per_stage[0].timestamp.observe_f64(100.0);
        a.per_stage[0].timestamp.observe_f64(200.0);
        a.per_stage[0].timestamp.observe_f64(300.0);
        a.per_stage[1].row_count = 0;
        a.per_stage[2].row_count = 0;
        a.per_stage[3].row_count = 0;

        let v = build_split_stats_payload(&a);
        assert_eq!(v["kind"], "tt_split_stats");
        assert_eq!(v["format"], "tt_split_stats_v1");
        let stages = v["stages"].as_object().unwrap();
        for k in ["train", "val", "test", "unused"] {
            assert!(stages.contains_key(k), "missing stage key {k}");
        }
        // Train bucket has the data
        let train = &stages["train"];
        assert_eq!(train["row_count"].as_u64(), Some(3));
        assert!((train["target"]["mean"].as_f64().unwrap() - 20.0).abs() < 1e-10);
        assert_eq!(train["target"]["count"].as_u64(), Some(3));
        assert!(train["target"]["quantiles"].is_object());
        assert_eq!(train["target"]["null_count"].as_u64(), Some(0));
        assert!((train["timestamp"]["mean"].as_f64().unwrap() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_split_stats_payload_target_elided_when_absent() {
        // No target_present flag set anywhere → the `target` key must be
        // elided from each stage block.
        let a = PerStageStatsAccumulator::new();
        let v = build_split_stats_payload(&a);
        for stage in ["train", "val", "test", "unused"] {
            let s = v["stages"][stage].as_object().unwrap();
            assert!(
                !s.contains_key("target"),
                "stage {stage} should elide target",
            );
            assert!(s.contains_key("timestamp"));
            assert_eq!(s["row_count"].as_u64(), Some(0));
        }
    }

    #[test]
    fn test_stage_seed_times_extraction_and_omission() {
        let mut a = PerStageStatsAccumulator::new();
        // train: timestamps {100, 250, 175} → min 100, max 250
        a.per_stage[0].timestamp.observe_f64(100.0);
        a.per_stage[0].timestamp.observe_f64(250.0);
        a.per_stage[0].timestamp.observe_f64(175.0);
        // val: single timestamp 400 → min == max == 400
        a.per_stage[1].timestamp.observe_f64(400.0);
        // test: {500, 800} → min 500, max 800
        a.per_stage[2].timestamp.observe_f64(500.0);
        a.per_stage[2].timestamp.observe_f64(800.0);
        // unused bucket has a timestamp but must never be emitted.
        a.per_stage[3].timestamp.observe_f64(999.0);

        let map = a.stage_seed_times().expect("non-empty");
        assert_eq!(map.get("train"), Some(&[100i64, 250]));
        assert_eq!(map.get("val"), Some(&[400i64, 400]));
        assert_eq!(map.get("test"), Some(&[500i64, 800]));
        assert!(!map.contains_key("unused"));
        assert_eq!(map.len(), 3);

        // JSON shape: top-level object of stage -> [min, max] int64.
        let v = serde_json::to_value(&map).unwrap();
        assert_eq!(v["train"], serde_json::json!([100, 250]));
        assert_eq!(v["val"], serde_json::json!([400, 400]));
        assert_eq!(v["test"], serde_json::json!([500, 800]));
        assert!(v.get("unused").is_none());
    }

    #[test]
    fn test_stage_seed_times_omits_empty_stages() {
        let mut a = PerStageStatsAccumulator::new();
        // Only test has timestamps; train/val have none (count==0) → omitted.
        a.per_stage[2].timestamp.observe_f64(42.0);
        let map = a.stage_seed_times().expect("non-empty");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("test"), Some(&[42i64, 42]));
        assert!(!map.contains_key("train"));
        assert!(!map.contains_key("val"));
    }

    #[test]
    fn test_stage_seed_times_none_when_no_timestamps() {
        // No stage observed a timestamp → None (field elides in manifest).
        let a = PerStageStatsAccumulator::new();
        assert!(a.stage_seed_times().is_none());
    }

    #[test]
    fn test_stage_seed_times_rounds_to_nearest_i64() {
        let mut a = PerStageStatsAccumulator::new();
        // f64 min/max with fractional ns → round to nearest.
        a.per_stage[0].timestamp.observe_f64(100.4);
        a.per_stage[0].timestamp.observe_f64(250.6);
        let map = a.stage_seed_times().unwrap();
        assert_eq!(map.get("train"), Some(&[100i64, 251]));
    }

    #[test]
    fn test_build_split_stats_payload_empty_quantiles_null() {
        // An empty timestamp accumulator → quantiles must serialize as null.
        let a = PerStageStatsAccumulator::new();
        let v = build_split_stats_payload(&a);
        for stage in ["train", "val", "test", "unused"] {
            assert!(v["stages"][stage]["timestamp"]["quantiles"].is_null());
            assert!(v["stages"][stage]["timestamp"]["mean"].is_null());
        }
    }
}
