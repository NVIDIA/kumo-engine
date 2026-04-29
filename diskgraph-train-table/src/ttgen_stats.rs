use std::collections::{HashMap, HashSet};

const SPLIT_RATIO_EPSILON: f64 = 1e-6;

use diskgraph_core::accumulators::{
    build_multicat_emit_stats, kumo_entry, kumo_key, kumo_safe_div, HyperLogLog,
    TDigestEstimator, WelfordAccumulator,
};
use diskgraph_core::time::{Abs, Dur, UnitSystem};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct IntColumnStats {
    pub count: u64,
    pub null_count: u64,
    pub num_unique: Option<u64>,
    pub min: Option<i64>,
    pub max: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct BoolColumnStats {
    pub count: u64,
    pub null_count: u64,
    pub true_count: u64,
    pub false_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableExactStatsPayload {
    pub entity: IntColumnStats,
    pub timestamp: IntColumnStats,
    pub target: IntColumnStats,
    pub frequency: Option<IntColumnStats>,
    pub entity_include_bit: Option<BoolColumnStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableSplitMaskCounts {
    pub train: u64,
    pub val: u64,
    pub test: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[serde(bound = "")]
pub struct TimestampSplitAssignmentV3<U: UnitSystem> {
    pub timestamp: Abs<U>,
    pub stage: String,
}

/// Raw i64 serialization form of a timestamp-stage assignment, used in the
/// output payload consumed by Python.  The generic internal representation
/// is [`TimestampSplitAssignmentV3<U>`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TimestampSplitAssignmentV3Raw {
    pub timestamp: i64,
    pub stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableSplitPayloadArtifactsV3 {
    pub strategy: String,
    pub split_kind: Option<String>,
    pub ratio: Option<Vec<String>>,
    pub offsets: Option<Vec<i64>>,
    pub ranges: Option<Vec<(i64, i64)>>,
    pub backtesting_timestamp: Option<i64>,
    pub split_column: Option<String>,
    pub manual_train_values: Option<Vec<String>>,
    pub manual_val_values: Option<Vec<String>>,
    pub manual_test_values: Option<Vec<String>>,
    pub authoritative_supported_family: bool,
    pub timestamp_stage_assignments: Option<Vec<TimestampSplitAssignmentV3Raw>>,
    pub index_stage_sizes: Option<TrainTableSplitMaskCounts>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableSplitPayload {
    pub version: u32,
    pub status: String,
    pub supports_native_execution: bool,
    pub split_kind: Option<String>,
    pub derivation_method: Option<String>,
    pub num_total_rows: u64,
    pub mask_counts: Option<TrainTableSplitMaskCounts>,
    pub artifacts_v3: Option<TrainTableSplitPayloadArtifactsV3>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableStatsPayload {
    pub version: u32,
    pub num_total_rows: u64,
    pub num_train_rows: u64,
    pub num_entities: u64,
    pub num_targets: u64,
    pub num_links: Option<u64>,
    pub exact_stats: TrainTableExactStatsPayload,
    pub split_payload: Option<TrainTableSplitPayload>,
    /// Per-timestamp row counts: {timestamp_seconds: row_count}.
    /// Used by the pybind layer to build accurate stats_per_timeframe.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub timestamp_counts: HashMap<i64, u64>,
    /// Kumo-format column stats for encoder inference (TARGET, FREQUENCY, ENTITY_INCLUDE_BIT).
    /// Format matches ingest's kumo_col_stats_v1 for compatibility with
    /// seed_train_table_col_stats_from_manifest().
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsed_train_table_col_stats: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TTExecutionSummary {
    pub rows_written: u64,
    pub stats: Option<TrainTableStatsPayload>,
}

pub(crate) struct ChunkStats<U: UnitSystem> {
    pub(crate) num_total_rows: u64,
    pub(crate) num_train_rows: u64,
    pub(crate) num_links: u64,
    pub(crate) entities: HashSet<u64>,
    pub(crate) targets: HashSet<u64>,
    pub(crate) timestamps: HashSet<Abs<U>>,
    pub(crate) timestamp_counts: HashMap<Abs<U>, u64>,
    pub(crate) timestamp_min: Option<Abs<U>>,
    pub(crate) timestamp_max: Option<Abs<U>>,
    pub(crate) entity_min: Option<i64>,
    pub(crate) entity_max: Option<i64>,
    pub(crate) target_min: Option<i64>,
    pub(crate) target_max: Option<i64>,
    pub(crate) frequency_min: Option<i64>,
    pub(crate) frequency_max: Option<i64>,
    pub(crate) include_true_count: u64,
    pub(crate) include_false_count: u64,
    // Distribution stats for TARGET column (numerical targets)
    pub(crate) target_welford: Option<WelfordAccumulator>,
    pub(crate) target_tdigest: Option<TDigestEstimator>,
    pub(crate) target_hll: Option<HyperLogLog>,
    pub(crate) target_invalid_count: u64,
    // Distribution stats for FREQUENCY column (link prediction)
    pub(crate) frequency_welford: Option<WelfordAccumulator>,
    pub(crate) frequency_tdigest: Option<TDigestEstimator>,
    pub(crate) frequency_hll: Option<HyperLogLog>,
    pub(crate) frequency_invalid_count: u64,
    // Distribution stats for ENTITY column (categorical/ID + optional numerical)
    pub(crate) entity_hll: Option<HyperLogLog>,
    pub(crate) entity_welford: Option<WelfordAccumulator>,
    pub(crate) entity_tdigest: Option<TDigestEstimator>,
    // Distribution stats for TIMESTAMP column (numerical, epoch seconds)
    pub(crate) timestamp_welford: Option<WelfordAccumulator>,
    pub(crate) timestamp_tdigest: Option<TDigestEstimator>,
    pub(crate) timestamp_hll: Option<HyperLogLog>,
}

impl<U: UnitSystem> Default for ChunkStats<U> {
    fn default() -> Self {
        Self {
            num_total_rows: 0,
            num_train_rows: 0,
            num_links: 0,
            entities: HashSet::new(),
            targets: HashSet::new(),
            timestamps: HashSet::new(),
            timestamp_counts: HashMap::new(),
            timestamp_min: None,
            timestamp_max: None,
            entity_min: None,
            entity_max: None,
            target_min: None,
            target_max: None,
            frequency_min: None,
            frequency_max: None,
            include_true_count: 0,
            include_false_count: 0,
            target_welford: None,
            target_tdigest: None,
            target_hll: None,
            target_invalid_count: 0,
            frequency_welford: None,
            frequency_tdigest: None,
            frequency_hll: None,
            frequency_invalid_count: 0,
            entity_hll: None,
            entity_welford: None,
            entity_tdigest: None,
            timestamp_welford: None,
            timestamp_tdigest: None,
            timestamp_hll: None,
        }
    }
}

pub(crate) struct StatsMerger<U: UnitSystem> {
    num_train_rows: u64,
    num_links: u64,
    entities: HashSet<u64>,
    targets: HashSet<u64>,
    timestamps: HashSet<Abs<U>>,
    timestamp_counts: HashMap<Abs<U>, u64>,
    timestamp_min: Option<Abs<U>>,
    timestamp_max: Option<Abs<U>>,
    entity_min: Option<i64>,
    entity_max: Option<i64>,
    target_min: Option<i64>,
    target_max: Option<i64>,
    frequency_min: Option<i64>,
    frequency_max: Option<i64>,
    include_true_count: u64,
    include_false_count: u64,
    // Distribution stats for TARGET column (numerical targets)
    target_welford: Option<WelfordAccumulator>,
    target_tdigest: Option<TDigestEstimator>,
    target_hll: Option<HyperLogLog>,
    target_invalid_count: u64,
    // Distribution stats for FREQUENCY column (link prediction)
    frequency_welford: Option<WelfordAccumulator>,
    frequency_tdigest: Option<TDigestEstimator>,
    frequency_hll: Option<HyperLogLog>,
    frequency_invalid_count: u64,
    // Distribution stats for ENTITY column (categorical/ID + optional numerical)
    entity_hll: Option<HyperLogLog>,
    entity_welford: Option<WelfordAccumulator>,
    entity_tdigest: Option<TDigestEstimator>,
    // Distribution stats for TIMESTAMP column (numerical, epoch seconds)
    timestamp_welford: Option<WelfordAccumulator>,
    timestamp_tdigest: Option<TDigestEstimator>,
    timestamp_hll: Option<HyperLogLog>,
}

impl<U: UnitSystem> Default for StatsMerger<U> {
    fn default() -> Self {
        Self {
            num_train_rows: 0,
            num_links: 0,
            entities: HashSet::new(),
            targets: HashSet::new(),
            timestamps: HashSet::new(),
            timestamp_counts: HashMap::new(),
            timestamp_min: None,
            timestamp_max: None,
            entity_min: None,
            entity_max: None,
            target_min: None,
            target_max: None,
            frequency_min: None,
            frequency_max: None,
            include_true_count: 0,
            include_false_count: 0,
            target_welford: None,
            target_tdigest: None,
            target_hll: None,
            target_invalid_count: 0,
            frequency_welford: None,
            frequency_tdigest: None,
            frequency_hll: None,
            frequency_invalid_count: 0,
            entity_hll: None,
            entity_welford: None,
            entity_tdigest: None,
            timestamp_welford: None,
            timestamp_tdigest: None,
            timestamp_hll: None,
        }
    }
}

impl<U: UnitSystem> StatsMerger<U> {
    pub(crate) fn merge(&mut self, chunk: ChunkStats<U>) {
        self.num_train_rows += chunk.num_train_rows;
        self.num_links += chunk.num_links;
        self.entities.extend(chunk.entities);
        self.targets.extend(chunk.targets);
        self.timestamps.extend(chunk.timestamps);
        for (ts, count) in chunk.timestamp_counts {
            *self.timestamp_counts.entry(ts).or_insert(0) += count;
        }

        self.timestamp_min = min_opt_abs(self.timestamp_min, chunk.timestamp_min);
        self.timestamp_max = max_opt_abs(self.timestamp_max, chunk.timestamp_max);
        self.entity_min = min_opt(self.entity_min, chunk.entity_min);
        self.entity_max = max_opt(self.entity_max, chunk.entity_max);
        self.target_min = min_opt(self.target_min, chunk.target_min);
        self.target_max = max_opt(self.target_max, chunk.target_max);
        self.frequency_min = min_opt(self.frequency_min, chunk.frequency_min);
        self.frequency_max = max_opt(self.frequency_max, chunk.frequency_max);
        self.include_true_count += chunk.include_true_count;
        self.include_false_count += chunk.include_false_count;

        // Merge target distribution accumulators
        if let Some(chunk_w) = chunk.target_welford {
            self.target_welford.get_or_insert_with(WelfordAccumulator::new).merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.target_tdigest {
            self.target_tdigest.get_or_insert_with(|| TDigestEstimator::new(200)).merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.target_hll {
            self.target_hll.get_or_insert_with(HyperLogLog::new).merge(&chunk_h);
        }
        self.target_invalid_count += chunk.target_invalid_count;
        // Merge frequency distribution accumulators
        if let Some(chunk_w) = chunk.frequency_welford {
            self.frequency_welford.get_or_insert_with(WelfordAccumulator::new).merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.frequency_tdigest {
            self.frequency_tdigest.get_or_insert_with(|| TDigestEstimator::new(200)).merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.frequency_hll {
            self.frequency_hll.get_or_insert_with(HyperLogLog::new).merge(&chunk_h);
        }
        self.frequency_invalid_count += chunk.frequency_invalid_count;
        // Merge entity distribution accumulators
        if let Some(chunk_h) = chunk.entity_hll {
            self.entity_hll.get_or_insert_with(HyperLogLog::new).merge(&chunk_h);
        }
        if let Some(chunk_w) = chunk.entity_welford {
            self.entity_welford.get_or_insert_with(WelfordAccumulator::new).merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.entity_tdigest {
            self.entity_tdigest.get_or_insert_with(|| TDigestEstimator::new(200)).merge(&mut chunk_t);
        }
        // Merge timestamp distribution accumulators
        if let Some(chunk_w) = chunk.timestamp_welford {
            self.timestamp_welford.get_or_insert_with(WelfordAccumulator::new).merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.timestamp_tdigest {
            self.timestamp_tdigest.get_or_insert_with(|| TDigestEstimator::new(200)).merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.timestamp_hll {
            self.timestamp_hll.get_or_insert_with(HyperLogLog::new).merge(&chunk_h);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize(
        mut self,
        num_total_rows: u64,
        is_link_prediction: bool,
        has_frequency_col: bool,
        has_include_bit_col: bool,
        split_ratio: Option<&[f64]>,
        split_kind: Option<&str>,
        split_is_temporal: bool,
        split_offsets: Option<&[Dur<U>]>,
        split_ranges: Option<&[(Abs<U>, Abs<U>)]>,
        split_backtesting_timestamp: Option<Abs<U>>,
        split_column: Option<&str>,
        manual_train_values: Option<&[String]>,
        manual_val_values: Option<&[String]>,
        manual_test_values: Option<&[String]>,
        target_category_counts: Option<HashMap<String, u64>>,
        target_stype: &str,
        entity_category_counts: Option<HashMap<String, u64>>,
    ) -> TrainTableStatsPayload {
        let num_entities = self.entities.len() as u64;
        let num_targets = self.targets.len() as u64;

        let entity_stats = IntColumnStats {
            count: self.num_train_rows,
            null_count: 0,
            num_unique: Some(num_entities),
            min: self.entity_min,
            max: self.entity_max,
        };

        let timestamp_stats = IntColumnStats {
            count: self.num_train_rows,
            null_count: 0,
            num_unique: Some(self.timestamps.len() as u64),
            min: self.timestamp_min.map(|t| t.raw()),
            max: self.timestamp_max.map(|t| t.raw()),
        };

        let target_stats = IntColumnStats {
            count: self.num_train_rows,
            null_count: 0,
            num_unique: Some(num_targets),
            min: self.target_min,
            max: self.target_max,
        };

        let frequency_stats = if has_frequency_col {
            Some(IntColumnStats {
                count: self.num_train_rows,
                null_count: 0,
                num_unique: None,
                min: self.frequency_min,
                max: self.frequency_max,
            })
        } else {
            None
        };

        let include_bit_stats = if has_include_bit_col {
            Some(BoolColumnStats {
                count: self.include_true_count + self.include_false_count,
                null_count: 0,
                true_count: self.include_true_count,
                false_count: self.include_false_count,
            })
        } else {
            None
        };

        let split_derivation = derive_split_mask_counts(
            num_total_rows,
            split_ratio,
            split_kind,
            split_is_temporal,
            split_offsets,
            split_ranges,
            split_backtesting_timestamp,
            split_column,
            manual_train_values,
            manual_val_values,
            manual_test_values,
            &self.timestamp_counts,
        );

        // Convert typed timestamp_counts to raw i64 only at the final output boundary.
        let raw_timestamp_counts: HashMap<i64, u64> = self
            .timestamp_counts
            .iter()
            .map(|(k, v)| (k.raw(), *v))
            .collect();

        let parsed_train_table_col_stats = build_inline_col_stats(
            &self.target_welford,
            &mut self.target_tdigest,
            &self.target_hll,
            self.target_invalid_count,
            &self.frequency_welford,
            &mut self.frequency_tdigest,
            &self.frequency_hll,
            self.frequency_invalid_count,
            self.include_true_count,
            self.include_false_count,
            num_total_rows,
            "v1",
            target_category_counts,
            target_stype,
            &self.entity_hll,
            &self.entity_welford,
            &mut self.entity_tdigest,
            entity_category_counts,
            &self.timestamp_welford,
            &mut self.timestamp_tdigest,
            &self.timestamp_hll,
        );

        TrainTableStatsPayload {
            version: 2,
            num_total_rows,
            num_train_rows: self.num_train_rows,
            num_entities,
            num_targets,
            num_links: if is_link_prediction {
                Some(self.num_links)
            } else {
                None
            },
            timestamp_counts: raw_timestamp_counts,
            parsed_train_table_col_stats,
            exact_stats: TrainTableExactStatsPayload {
                entity: entity_stats,
                timestamp: timestamp_stats,
                target: target_stats,
                frequency: frequency_stats,
                entity_include_bit: include_bit_stats,
            },
            split_payload: Some(TrainTableSplitPayload {
                version: 3,
                status: if split_derivation.supports_native_execution {
                    "ready_row_artifacts_v3".to_string()
                } else {
                    "counts_only_no_masks".to_string()
                },
                supports_native_execution: split_derivation.supports_native_execution,
                split_kind: split_kind.map(|s| s.to_string()),
                derivation_method: split_derivation.method.clone(),
                num_total_rows,
                mask_counts: split_derivation.mask_counts.clone(),
                artifacts_v3: Some(build_split_artifacts_v3(
                    split_kind,
                    split_ratio,
                    split_offsets.map(|xs| xs.iter().map(|d| d.raw()).collect()),
                    split_ranges.map(|xs| xs.iter().map(|(lo, hi)| (lo.raw(), hi.raw())).collect()),
                    split_backtesting_timestamp.map(|ts| ts.raw()),
                    split_column,
                    manual_train_values,
                    manual_val_values,
                    manual_test_values,
                    &split_derivation,
                )),
            }),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_inline_col_stats(
    target_welford: &Option<WelfordAccumulator>,
    target_tdigest: &mut Option<TDigestEstimator>,
    target_hll: &Option<HyperLogLog>,
    target_invalid_count: u64,
    frequency_welford: &Option<WelfordAccumulator>,
    frequency_tdigest: &mut Option<TDigestEstimator>,
    frequency_hll: &Option<HyperLogLog>,
    frequency_invalid_count: u64,
    include_true_count: u64,
    include_false_count: u64,
    num_total_rows: u64,
    data_version: &str,
    target_category_counts: Option<HashMap<String, u64>>,
    target_stype: &str,
    entity_hll: &Option<HyperLogLog>,
    entity_welford: &Option<WelfordAccumulator>,
    entity_tdigest: &mut Option<TDigestEstimator>,
    entity_category_counts: Option<HashMap<String, u64>>,
    timestamp_welford: &Option<WelfordAccumulator>,
    timestamp_tdigest: &mut Option<TDigestEstimator>,
    timestamp_hll: &Option<HyperLogLog>,
) -> Option<JsonValue> {
    let mut stats = serde_json::Map::new();
    let dv = data_version;

    let is_categorical_target = matches!(
        target_stype,
        "categorical" | "boolean" | "ID"
    );
    let is_multicat_target = target_stype == "multicategorical";

    // TARGET categorical stats
    if is_categorical_target {
        if let Some(ref cat_counts) = target_category_counts {
            let count: u64 = cat_counts.values().sum();
            let stype = "kumo.categorical";
            let name = "TARGET";

            stats.insert(
                kumo_key(name, stype, "COUNT", false),
                kumo_entry(json!(count), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_NA", false),
                kumo_entry(json!(0u64), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NA_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "INVALID_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );

            // CATEGORY_COUNTS: [[key, count], ...]
            let mut pairs: Vec<(&String, &u64)> = cat_counts.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1)); // descending by count
            let category_array: Vec<JsonValue> = pairs
                .iter()
                .map(|(k, c)| json!([k, c]))
                .collect();
            stats.insert(
                kumo_key(name, stype, "CATEGORY_COUNTS", false),
                kumo_entry(json!(category_array), false, dv, count),
            );

            let num_unique = cat_counts.len() as u64;
            if let Some(h) = target_hll {
                let hll_estimate = (h.estimate() as u64).min(count);
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(hll_estimate.max(num_unique)), false, dv, count),
                );
            } else {
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(num_unique), false, dv, count),
                );
            }
        }
    }

    // TARGET multicategorical stats
    if is_multicat_target {
        if let Some(ref cat_counts) = target_category_counts {
            let count: u64 = cat_counts.values().sum();
            let stype = "kumo.multicategorical";
            let name = "TARGET";

            // Sort category counts count-desc, token-asc
            let mut sorted_cats: Vec<(String, u64)> = cat_counts
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            sorted_cats.sort_by(|a, b| {
                b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });

            let fallback_hll = HyperLogLog::new();
            let hll = target_hll.as_ref().unwrap_or(&fallback_hll);

            let pairs = build_multicat_emit_stats(
                name,
                stype,
                dv,
                count,
                0, // null_count — ttgen targets are never null
                hll,
                Some("|"), // standard LIST_DISTINCT delimiter
                &sorted_cats,
            );
            for (key, value) in pairs {
                stats.insert(key, value);
            }
        }
    }

    // TARGET numerical stats
    if !is_categorical_target && !is_multicat_target {
        if let Some(w) = target_welford {
            let count = w.count + target_invalid_count;
            let stype = "kumo.numerical";
            let name = "TARGET";
            stats.insert(
                kumo_key(name, stype, "COUNT", false),
                kumo_entry(json!(count), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_NA", false),
                kumo_entry(json!(0u64), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NA_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "INVALID_FRACTION", false),
                kumo_entry(json!(kumo_safe_div(target_invalid_count, count)), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MEAN", false),
                kumo_entry(json!(w.mean()), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "STD", false),
                kumo_entry(json!(w.std_dev()), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MIN", false),
                kumo_entry(json!(w.min), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MAX", false),
                kumo_entry(json!(w.max), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "KURTOSIS", false),
                kumo_entry(json!(w.kurtosis()), false, dv, count),
            );

            if let Some(ref mut td) = *target_tdigest {
                let (q25, q50, q75) = td.quantiles().unwrap_or((0.0, 0.0, 0.0));
                stats.insert(
                    kumo_key(name, stype, "QUANTILES", false),
                    kumo_entry(json!([q25, q50, q75]), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "QUANTILE25", false),
                    kumo_entry(json!(q25), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "MEDIAN", false),
                    kumo_entry(json!(q50), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "QUANTILE75", false),
                    kumo_entry(json!(q75), false, dv, count),
                );
            }

            if let Some(h) = target_hll {
                let num_unique = (h.estimate() as u64).min(count);
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(num_unique), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "UNIQUE_FRACTION", false),
                    kumo_entry(json!(kumo_safe_div(num_unique, count)), false, dv, count),
                );
            }
        }
    }

    // FREQUENCY numerical stats (same pattern)
    if let Some(w) = frequency_welford {
        let count = w.count + frequency_invalid_count;
        let stype = "kumo.numerical";
        let name = "__kumo_internal_frequency_column__";
        stats.insert(
            kumo_key(name, stype, "COUNT", false),
            kumo_entry(json!(count), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "NUM_NA", false),
            kumo_entry(json!(0u64), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "NA_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "INVALID_FRACTION", false),
            kumo_entry(json!(kumo_safe_div(frequency_invalid_count, count)), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MEAN", false),
            kumo_entry(json!(w.mean()), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "STD", false),
            kumo_entry(json!(w.std_dev()), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MIN", false),
            kumo_entry(json!(w.min), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MAX", false),
            kumo_entry(json!(w.max), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "KURTOSIS", false),
            kumo_entry(json!(w.kurtosis()), false, dv, count),
        );

        if let Some(ref mut td) = *frequency_tdigest {
            let (q25, q50, q75) = td.quantiles().unwrap_or((0.0, 0.0, 0.0));
            stats.insert(
                kumo_key(name, stype, "QUANTILES", false),
                kumo_entry(json!([q25, q50, q75]), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "QUANTILE25", false),
                kumo_entry(json!(q25), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MEDIAN", false),
                kumo_entry(json!(q50), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "QUANTILE75", false),
                kumo_entry(json!(q75), false, dv, count),
            );
        }

        if let Some(h) = frequency_hll {
            let num_unique = (h.estimate() as u64).min(count);
            stats.insert(
                kumo_key(name, stype, "NUM_UNIQUE", false),
                kumo_entry(json!(num_unique), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "UNIQUE_FRACTION", false),
                kumo_entry(json!(kumo_safe_div(num_unique, count)), false, dv, count),
            );
        }
    }

    // ENTITY_INCLUDE_BIT boolean stats (as categorical CATEGORY_COUNTS)
    let total_include = include_true_count + include_false_count;
    if total_include > 0 {
        let name = "__kumo_internal_entity_inclusive_bit_column__";
        let stype = "kumo.categorical";
        stats.insert(
            kumo_key(name, stype, "COUNT", false),
            kumo_entry(json!(total_include), false, dv, total_include),
        );
        stats.insert(
            kumo_key(name, stype, "NUM_NA", false),
            kumo_entry(json!(0u64), false, dv, total_include),
        );
        stats.insert(
            kumo_key(name, stype, "NA_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, total_include),
        );
        stats.insert(
            kumo_key(name, stype, "INVALID_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, total_include),
        );
        stats.insert(
            kumo_key(name, stype, "CATEGORY_COUNTS", false),
            kumo_entry(
                json!([[true, include_true_count], [false, include_false_count]]),
                false,
                dv,
                total_include,
            ),
        );
        stats.insert(
            kumo_key(name, stype, "NUM_UNIQUE", false),
            kumo_entry(json!(2u64), false, dv, total_include),
        );
    }

    // ENTITY categorical stats (ID column — with CATEGORY_COUNTS)
    {
        let count = num_total_rows;
        let stype = "kumo.categorical";
        let name = "ENTITY";

        if let Some(ref cat_counts) = entity_category_counts {
            let cat_total: u64 = cat_counts.values().sum();
            let effective_count = if cat_total > 0 { cat_total } else { count };
            stats.insert(
                kumo_key(name, stype, "COUNT", false),
                kumo_entry(json!(effective_count), false, dv, effective_count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_NA", false),
                kumo_entry(json!(0u64), false, dv, effective_count),
            );
            stats.insert(
                kumo_key(name, stype, "NA_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, effective_count),
            );
            stats.insert(
                kumo_key(name, stype, "INVALID_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, effective_count),
            );

            // CATEGORY_COUNTS: [[key, count], ...]
            let mut pairs: Vec<(&String, &u64)> = cat_counts.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1)); // descending by count
            let category_array: Vec<JsonValue> = pairs
                .iter()
                .map(|(k, c)| json!([k, c]))
                .collect();
            stats.insert(
                kumo_key(name, stype, "CATEGORY_COUNTS", false),
                kumo_entry(json!(category_array), false, dv, effective_count),
            );

            let num_unique_exact = cat_counts.len() as u64;
            if let Some(h) = entity_hll {
                let hll_estimate = (h.estimate() as u64).min(effective_count);
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(hll_estimate.max(num_unique_exact)), false, dv, effective_count),
                );
            } else {
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(num_unique_exact), false, dv, effective_count),
                );
            }
        } else if let Some(h) = entity_hll {
            // Fallback: no category counts available, emit HLL-only stats
            let num_unique = (h.estimate() as u64).min(count);
            stats.insert(
                kumo_key(name, stype, "COUNT", false),
                kumo_entry(json!(count), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_NA", false),
                kumo_entry(json!(0u64), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NA_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "INVALID_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_UNIQUE", false),
                kumo_entry(json!(num_unique), false, dv, count),
            );
        }
    }

    // ENTITY numerical stats (when entity is Int64 — welford will have observations)
    if let Some(w) = entity_welford {
        if w.count > 0 {
            let count = w.count;
            let stype = "kumo.numerical";
            let name = "ENTITY";
            stats.insert(
                kumo_key(name, stype, "COUNT", false),
                kumo_entry(json!(count), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NUM_NA", false),
                kumo_entry(json!(0u64), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "NA_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "INVALID_FRACTION", false),
                kumo_entry(json!(0.0), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MEAN", false),
                kumo_entry(json!(w.mean()), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "STD", false),
                kumo_entry(json!(w.std_dev()), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MIN", false),
                kumo_entry(json!(w.min), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MAX", false),
                kumo_entry(json!(w.max), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "KURTOSIS", false),
                kumo_entry(json!(w.kurtosis()), false, dv, count),
            );

            if let Some(ref mut td) = *entity_tdigest {
                let (q25, q50, q75) = td.quantiles().unwrap_or((0.0, 0.0, 0.0));
                stats.insert(
                    kumo_key(name, stype, "QUANTILES", false),
                    kumo_entry(json!([q25, q50, q75]), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "QUANTILE25", false),
                    kumo_entry(json!(q25), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "MEDIAN", false),
                    kumo_entry(json!(q50), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "QUANTILE75", false),
                    kumo_entry(json!(q75), false, dv, count),
                );
            }

            if let Some(h) = entity_hll {
                let num_unique = (h.estimate() as u64).min(count);
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(json!(num_unique), false, dv, count),
                );
                stats.insert(
                    kumo_key(name, stype, "UNIQUE_FRACTION", false),
                    kumo_entry(json!(kumo_safe_div(num_unique, count)), false, dv, count),
                );
            }
        }
    }

    // TIMESTAMP numerical stats (epoch seconds — full stats including quantiles)
    if let Some(w) = timestamp_welford {
        let count = w.count;
        let stype = "kumo.numerical";
        let name = "TIMESTAMP";
        stats.insert(
            kumo_key(name, stype, "COUNT", false),
            kumo_entry(json!(count), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "NUM_NA", false),
            kumo_entry(json!(0u64), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "NA_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "INVALID_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MEAN", false),
            kumo_entry(json!(w.mean()), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "STD", false),
            kumo_entry(json!(w.std_dev()), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MIN", false),
            kumo_entry(json!(w.min), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "MAX", false),
            kumo_entry(json!(w.max), false, dv, count),
        );
        stats.insert(
            kumo_key(name, stype, "KURTOSIS", false),
            kumo_entry(json!(w.kurtosis()), false, dv, count),
        );

        if let Some(ref mut td) = *timestamp_tdigest {
            let (q25, q50, q75) = td.quantiles().unwrap_or((0.0, 0.0, 0.0));
            stats.insert(
                kumo_key(name, stype, "QUANTILES", false),
                kumo_entry(json!([q25, q50, q75]), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "QUANTILE25", false),
                kumo_entry(json!(q25), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "MEDIAN", false),
                kumo_entry(json!(q50), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "QUANTILE75", false),
                kumo_entry(json!(q75), false, dv, count),
            );
        }

        if let Some(h) = timestamp_hll {
            let num_unique = (h.estimate() as u64).min(count);
            stats.insert(
                kumo_key(name, stype, "NUM_UNIQUE", false),
                kumo_entry(json!(num_unique), false, dv, count),
            );
            stats.insert(
                kumo_key(name, stype, "UNIQUE_FRACTION", false),
                kumo_entry(json!(kumo_safe_div(num_unique, count)), false, dv, count),
            );
        }
    }

    if stats.is_empty() {
        None
    } else {
        Some(json!({
            "version": 1,
            "kind": "tt_train_table_col_stats",
            "stats_format": "kumo_col_stats_v1",
            "row_count": num_total_rows,
            "stats": stats,
            "provided_fields": ["stats"]
        }))
    }
}

struct SplitMaskDerivation<U: UnitSystem> {
    method: Option<String>,
    mask_counts: Option<TrainTableSplitMaskCounts>,
    timestamp_stage_assignments: Option<Vec<TimestampSplitAssignmentV3<U>>>,
    index_stage_sizes: Option<TrainTableSplitMaskCounts>,
    supports_native_execution: bool,
}

#[allow(clippy::too_many_arguments)]
fn build_split_artifacts_v3<U: UnitSystem>(
    split_kind: Option<&str>,
    split_ratio: Option<&[f64]>,
    split_offsets: Option<Vec<i64>>,
    split_ranges: Option<Vec<(i64, i64)>>,
    split_backtesting_timestamp: Option<i64>,
    split_column: Option<&str>,
    manual_train_values: Option<&[String]>,
    manual_val_values: Option<&[String]>,
    manual_test_values: Option<&[String]>,
    split_derivation: &SplitMaskDerivation<U>,
) -> TrainTableSplitPayloadArtifactsV3 {
    TrainTableSplitPayloadArtifactsV3 {
        strategy: "rule_based_v3".to_string(),
        split_kind: split_kind.map(str::to_string),
        ratio: split_ratio.map(|xs| xs.iter().map(|x| x.to_string()).collect()),
        offsets: split_offsets,
        ranges: split_ranges,
        backtesting_timestamp: split_backtesting_timestamp,
        split_column: split_column.map(str::to_string),
        manual_train_values: manual_train_values.map(|xs| xs.to_vec()),
        manual_val_values: manual_val_values.map(|xs| xs.to_vec()),
        manual_test_values: manual_test_values.map(|xs| xs.to_vec()),
        authoritative_supported_family: split_derivation.supports_native_execution,
        timestamp_stage_assignments: split_derivation
            .timestamp_stage_assignments
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| TimestampSplitAssignmentV3Raw {
                        timestamp: a.timestamp.raw(),
                        stage: a.stage.clone(),
                    })
                    .collect()
            }),
        index_stage_sizes: split_derivation.index_stage_sizes.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_split_mask_counts<U: UnitSystem>(
    num_total_rows: u64,
    split_ratio: Option<&[f64]>,
    split_kind: Option<&str>,
    split_is_temporal: bool,
    split_offsets: Option<&[Dur<U>]>,
    split_ranges: Option<&[(Abs<U>, Abs<U>)]>,
    split_backtesting_timestamp: Option<Abs<U>>,
    split_column: Option<&str>,
    manual_train_values: Option<&[String]>,
    manual_val_values: Option<&[String]>,
    manual_test_values: Option<&[String]>,
    typed_timestamp_counts: &HashMap<Abs<U>, u64>,
) -> SplitMaskDerivation<U> {
    match (split_kind, split_is_temporal) {
        (Some("FixedSplit"), false) => {
            let counts = validate_split_ratio(split_ratio)
                .and_then(|ratio| derive_ratio_mask_counts(num_total_rows, ratio));
            SplitMaskDerivation {
                method: Some("fixed_index_ranges".to_string()),
                mask_counts: counts.clone(),
                timestamp_stage_assignments: None,
                index_stage_sizes: counts,
                supports_native_execution: true,
            }
        }
        (Some("ApproxDateOffsetSplit"), true) => {
            let ratio = validate_split_ratio(split_ratio);
            let stage_map = ratio.and_then(|r| {
                derive_approx_date_offset_timestamp_stage_assignments(r, typed_timestamp_counts)
            });
            let counts = stage_map
                .as_ref()
                .map(|assignments| mask_counts_from_timestamp_assignments(assignments, typed_timestamp_counts));
            SplitMaskDerivation {
                method: Some("approx_date_offset_timeframes".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: stage_map,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("DateOffsetSplit"), true) => {
            let stage_map =
                derive_date_offset_timestamp_stage_assignments(split_offsets, typed_timestamp_counts);
            let counts = stage_map
                .as_ref()
                .map(|assignments| mask_counts_from_timestamp_assignments(assignments, typed_timestamp_counts));
            SplitMaskDerivation {
                method: Some("date_offset_boundaries".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: stage_map,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("TimeRangeSplit"), true) => {
            let stage_map =
                derive_time_range_timestamp_stage_assignments(split_ranges, typed_timestamp_counts);
            let counts = stage_map
                .as_ref()
                .map(|assignments| mask_counts_from_timestamp_assignments(assignments, typed_timestamp_counts));
            SplitMaskDerivation {
                method: Some("time_range_boundaries".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: stage_map,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("BacktestingSplit"), true) => {
            let stage_map = derive_backtesting_timestamp_stage_assignments(
                split_backtesting_timestamp,
                typed_timestamp_counts,
            );
            let counts = stage_map
                .as_ref()
                .map(|assignments| mask_counts_from_timestamp_assignments(assignments, typed_timestamp_counts));
            SplitMaskDerivation {
                method: Some("backtesting_timestamp".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: stage_map,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("TemporalSplit"), true) => {
            let counts = validate_split_ratio(split_ratio)
                .and_then(|ratio| derive_ratio_mask_counts(num_total_rows, ratio));
            SplitMaskDerivation {
                method: Some("temporal_ratio_sort_reconstruct".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: None,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("ColumnSplit"), false) => {
            let counts = validate_split_ratio(split_ratio)
                .and_then(|ratio| derive_ratio_mask_counts(num_total_rows, ratio));
            SplitMaskDerivation {
                method: Some("column_ratio_sort_reconstruct".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: None,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("ManualSplit"), false) => {
            let counts = derive_manual_split_mask_counts(
                split_column,
                manual_train_values,
                manual_val_values,
                manual_test_values,
                typed_timestamp_counts,
            );
            SplitMaskDerivation {
                method: Some("manual_value_membership_reconstruct".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: None,
                index_stage_sizes: None,
                supports_native_execution: true,
            }
        }
        (Some("RandomSplit"), false) | (Some("RandomLinkSplit"), false) => {
            let counts = validate_split_ratio(split_ratio)
                .and_then(|ratio| derive_ratio_mask_counts(num_total_rows, ratio));
            SplitMaskDerivation {
                method: Some("ratio_rounding".to_string()),
                mask_counts: counts,
                timestamp_stage_assignments: None,
                index_stage_sizes: None,
                supports_native_execution: false,
            }
        }
        _ => SplitMaskDerivation {
            method: None,
            mask_counts: None,
            timestamp_stage_assignments: None,
            index_stage_sizes: None,
            supports_native_execution: false,
        },
    }
}

fn validate_split_ratio(split_ratio: Option<&[f64]>) -> Option<&[f64]> {
    let ratio = split_ratio?;
    if ratio.len() < 2 || ratio.len() > 3 {
        return None;
    }
    if ratio.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return None;
    }
    let sum: f64 = ratio.iter().sum();
    if sum > 1.0 + SPLIT_RATIO_EPSILON {
        return None;
    }
    Some(ratio)
}

fn derive_ratio_mask_counts(
    num_total_rows: u64,
    ratio: &[f64],
) -> Option<TrainTableSplitMaskCounts> {
    let mut ratio_vals: Vec<f64> = ratio.iter().take(3).map(|x| x.max(0.0)).collect();
    if ratio_vals.is_empty() {
        return Some(TrainTableSplitMaskCounts {
            train: 0,
            val: 0,
            test: 0,
        });
    }

    let total: f64 = ratio_vals.iter().sum();
    if total <= 0.0 {
        return Some(TrainTableSplitMaskCounts {
            train: 0,
            val: 0,
            test: 0,
        });
    }
    for x in &mut ratio_vals {
        *x /= total;
    }

    let row_count = num_total_rows as usize;
    let ideal: Vec<f64> = ratio_vals.iter().map(|x| x * row_count as f64).collect();
    let mut sizes: Vec<usize> = ideal.iter().map(|v| v.floor() as usize).collect();

    let assigned_floor: usize = sizes.iter().sum();
    let mut remaining = row_count.saturating_sub(assigned_floor);

    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = ideal[a] - sizes[a] as f64;
        let rb = ideal[b] - sizes[b] as f64;
        rb.partial_cmp(&ra)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut k = 0usize;
    while remaining > 0 && !order.is_empty() {
        sizes[order[k]] += 1;
        remaining -= 1;
        k += 1;
        if k == order.len() {
            k = 0;
        }
    }

    let train = sizes.first().copied().unwrap_or(0) as u64;
    let val = sizes.get(1).copied().unwrap_or(0) as u64;
    let test = sizes.get(2).copied().unwrap_or(0) as u64;
    Some(TrainTableSplitMaskCounts { train, val, test })
}

fn derive_manual_split_mask_counts<U: UnitSystem>(
    split_column: Option<&str>,
    manual_train_values: Option<&[String]>,
    manual_val_values: Option<&[String]>,
    manual_test_values: Option<&[String]>,
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> Option<TrainTableSplitMaskCounts> {
    // Current TT summary only carries timestamp-level value distributions.
    // Therefore ManualSplit parity is supported only when splitting on
    // TIMESTAMP/PSEUDO_TIMESTAMP and values are epoch-second integers.
    let col = split_column?;
    if col != "TIMESTAMP" && col != "PSEUDO_TIMESTAMP" {
        return None;
    }

    let parse = |vals: Option<&[String]>| -> Option<Vec<Abs<U>>> {
        vals.map(|xs| {
            xs.iter()
                .filter_map(|x| x.parse::<i64>().ok().map(Abs::from_raw))
                .collect::<Vec<_>>()
        })
    };

    let train_vals = parse(manual_train_values).unwrap_or_default();
    let val_vals = parse(manual_val_values).unwrap_or_default();
    let test_vals = parse(manual_test_values).unwrap_or_default();

    let sum_for = |vals: &[Abs<U>]| -> u64 {
        vals.iter()
            .map(|v| *timestamp_counts.get(v).unwrap_or(&0))
            .sum()
    };

    Some(TrainTableSplitMaskCounts {
        train: sum_for(&train_vals),
        val: sum_for(&val_vals),
        test: sum_for(&test_vals),
    })
}

fn stage_name(idx: usize) -> String {
    match idx {
        0 => "train".to_string(),
        1 => "val".to_string(),
        _ => "test".to_string(),
    }
}

fn mask_counts_from_timestamp_assignments<U: UnitSystem>(
    assignments: &[TimestampSplitAssignmentV3<U>],
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> TrainTableSplitMaskCounts {
    let mut counts = TrainTableSplitMaskCounts {
        train: 0,
        val: 0,
        test: 0,
    };
    for a in assignments {
        let c = *timestamp_counts.get(&a.timestamp).unwrap_or(&0);
        match a.stage.as_str() {
            "train" => counts.train += c,
            "val" => counts.val += c,
            "test" => counts.test += c,
            _ => {}
        }
    }
    counts
}

fn derive_approx_date_offset_timestamp_stage_assignments<U: UnitSystem>(
    ratio: &[f64],
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> Option<Vec<TimestampSplitAssignmentV3<U>>> {
    if timestamp_counts.is_empty() {
        return None;
    }
    let mut latest_to_earliest: Vec<(Abs<U>, u64)> = timestamp_counts
        .iter()
        .map(|(ts, c)| (*ts, *c))
        .collect();
    latest_to_earliest.sort_by_key(|entry| std::cmp::Reverse(entry.0));

    let earliest_to_latest: Vec<(Abs<U>, u64)> = latest_to_earliest.into_iter().rev().collect();
    let counts_only: Vec<u64> = earliest_to_latest.iter().map(|(_, c)| *c).collect();
    let num_timeframes = counts_only.len();
    let num_split = ratio.len();
    if num_timeframes < num_split {
        return None;
    }

    let total: f64 = counts_only.iter().map(|c| *c as f64).sum();
    if total <= 0.0 {
        return None;
    }

    let mut cumulative = Vec::with_capacity(num_timeframes);
    let mut acc = 0.0f64;
    for count in &counts_only {
        acc += (*count as f64) / total;
        cumulative.push(acc);
    }

    let mut train_val_index = search_sorted_left(&cumulative, ratio[0]);
    train_val_index = train_val_index.min(num_timeframes - num_split);

    let mut val_test_index = None;
    if num_split == 3 {
        let mut idx = search_sorted_left(&cumulative, ratio[0] + ratio[1]);
        idx = idx.min(num_timeframes - 2);
        idx = idx.max(train_val_index + 1);
        val_test_index = Some(idx);
    }

    let mut assignments = Vec::with_capacity(num_timeframes);
    for (i, (ts, _)) in earliest_to_latest.into_iter().enumerate() {
        let stage = if i <= train_val_index {
            stage_name(0)
        } else if num_split == 2 || i <= val_test_index.unwrap_or(i) {
            stage_name(1)
        } else {
            stage_name(2)
        };
        assignments.push(TimestampSplitAssignmentV3 {
            timestamp: ts,
            stage,
        });
    }
    Some(assignments)
}

fn derive_date_offset_timestamp_stage_assignments<U: UnitSystem>(
    split_offsets: Option<&[Dur<U>]>,
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> Option<Vec<TimestampSplitAssignmentV3<U>>> {
    let offsets = split_offsets?;
    if timestamp_counts.is_empty() || offsets.is_empty() || offsets.len() > 2 {
        return None;
    }

    let mut ts_values: Vec<Abs<U>> = timestamp_counts.keys().copied().collect();
    ts_values.sort_unstable();
    let min_ts = *ts_values.first()?;
    let max_ts = *ts_values.last()?;

    let mut boundaries = vec![min_ts];
    for off in offsets {
        // Abs<U> + Dur<U> = Abs<U> — no manual scaling needed.
        boundaries.push(max_ts + *off);
    }
    boundaries.push(max_ts);

    if boundaries.len() < 3 {
        return None;
    }

    let mut assignments = Vec::with_capacity(ts_values.len());
    for ts in ts_values {
        for i in 0..(boundaries.len() - 1) {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            let in_bucket = if i == 0 {
                ts > start && ts <= end
            } else {
                ts >= start && ts <= end
            };
            if in_bucket {
                assignments.push(TimestampSplitAssignmentV3 {
                    timestamp: ts,
                    stage: stage_name(i),
                });
                break;
            }
        }
    }
    Some(assignments)
}

fn derive_time_range_timestamp_stage_assignments<U: UnitSystem>(
    split_ranges: Option<&[(Abs<U>, Abs<U>)]>,
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> Option<Vec<TimestampSplitAssignmentV3<U>>> {
    let ranges = split_ranges?;
    if ranges.len() < 2 || ranges.len() > 3 {
        return None;
    }
    let mut ts_values: Vec<Abs<U>> = timestamp_counts.keys().copied().collect();
    ts_values.sort_unstable();

    let mut assignments = Vec::with_capacity(ts_values.len());
    for ts in ts_values {
        for (idx, (start, end)) in ranges.iter().enumerate() {
            if ts >= *start && ts <= *end {
                assignments.push(TimestampSplitAssignmentV3 {
                    timestamp: ts,
                    stage: stage_name(idx),
                });
                break;
            }
        }
    }
    Some(assignments)
}

fn derive_backtesting_timestamp_stage_assignments<U: UnitSystem>(
    split_backtesting_timestamp: Option<Abs<U>>,
    timestamp_counts: &HashMap<Abs<U>, u64>,
) -> Option<Vec<TimestampSplitAssignmentV3<U>>> {
    let test_ts = split_backtesting_timestamp?;
    if !timestamp_counts.contains_key(&test_ts) {
        return Some(Vec::new());
    }

    Some(vec![TimestampSplitAssignmentV3 {
        timestamp: test_ts,
        stage: stage_name(2),
    }])
}

fn search_sorted_left(cumulative: &[f64], value: f64) -> usize {
    for (idx, v) in cumulative.iter().enumerate() {
        if *v >= value {
            return idx;
        }
    }
    cumulative.len().saturating_sub(1)
}

fn min_opt(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn max_opt(a: Option<i64>, b: Option<i64>) -> Option<i64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn min_opt_abs<U: UnitSystem>(a: Option<Abs<U>>, b: Option<Abs<U>>) -> Option<Abs<U>> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn max_opt_abs<U: UnitSystem>(a: Option<Abs<U>>, b: Option<Abs<U>>) -> Option<Abs<U>> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskgraph_core::time::Seconds;

    #[test]
    fn test_finalize_contains_exact_stats_payload() {
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 3,
            num_links: 2,
            ..Default::default()
        };
        chunk.entities.insert(10);
        chunk.entities.insert(20);
        chunk.targets.insert(100);
        chunk.timestamps.insert(Abs::from_raw(1000));
        chunk.timestamps.insert(Abs::from_raw(2000));
        chunk.timestamp_min = Some(Abs::from_raw(1000));
        chunk.timestamp_max = Some(Abs::from_raw(2000));
        chunk.entity_min = Some(10);
        chunk.entity_max = Some(20);
        chunk.target_min = Some(100);
        chunk.target_max = Some(100);
        chunk.frequency_min = Some(1);
        chunk.frequency_max = Some(2);
        chunk.include_true_count = 3;
        chunk.include_false_count = 1;

        merger.merge(chunk);
        let out = merger.finalize(
            4,
            true,
            true,
            true,
            Some(&[0.5, 0.25, 0.25]),
            Some("RandomSplit"),
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "numerical",
            None,
        );

        assert_eq!(out.version, 2);
        assert_eq!(out.num_total_rows, 4);
        assert_eq!(out.num_train_rows, 3);
        assert_eq!(out.num_entities, 2);
        assert_eq!(out.num_targets, 1);
        assert_eq!(out.num_links, Some(2));
        assert_eq!(out.exact_stats.timestamp.min, Some(1000));
        assert_eq!(out.exact_stats.timestamp.max, Some(2000));
        assert_eq!(out.exact_stats.entity_include_bit.unwrap().false_count, 1);
        assert_eq!(out.split_payload.as_ref().map(|p| p.version), Some(3));
        assert_eq!(
            out.split_payload.as_ref().map(|p| p.status.as_str()),
            Some("counts_only_no_masks")
        );
        assert_eq!(
            out.split_payload
                .as_ref()
                .map(|p| p.supports_native_execution),
            Some(false)
        );
        assert_eq!(
            out.split_payload.as_ref().map(|p| p.num_total_rows),
            Some(4)
        );
        assert!(!out
            .split_payload
            .as_ref()
            .and_then(|p| p.artifacts_v3.as_ref())
            .map(|a| a.authoritative_supported_family)
            .unwrap_or(false));
        let counts = out
            .split_payload
            .as_ref()
            .and_then(|p| p.mask_counts.as_ref())
            .expect("mask counts should be present");
        assert_eq!(counts.train, 2);
        assert_eq!(counts.val, 1);
        assert_eq!(counts.test, 1);
    }

    #[test]
    fn test_finalize_approx_date_offset_split_counts() {
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        let mut chunk: ChunkStats<Seconds> = ChunkStats::default();

        // Latest -> earliest frame counts: [4500, 4500, 4500, 4500]
        // ApproxDateOffsetSplit(0.8, 0.1, 0.1) should allocate full
        // timeframes and result in 9000/4500/4500 under Kumo logic.
        for ts in [40_i64, 30, 20, 10] {
            chunk.timestamps.insert(Abs::from_raw(ts));
            chunk.timestamp_counts.insert(Abs::from_raw(ts), 4_500);
        }
        chunk.num_train_rows = 18_000;

        merger.merge(chunk);
        let out = merger.finalize(
            18_000,
            true,
            false,
            false,
            Some(&[0.8, 0.1, 0.1]),
            Some("ApproxDateOffsetSplit"),
            true,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "numerical",
            None,
        );

        let counts = out
            .split_payload
            .as_ref()
            .and_then(|p| p.mask_counts.as_ref())
            .expect("mask counts should be present for ApproxDateOffsetSplit");
        assert_eq!(counts.train, 9_000);
        assert_eq!(counts.val, 4_500);
        assert_eq!(counts.test, 4_500);
    }

    #[test]
    fn test_finalize_backtesting_split_counts() {
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        let mut chunk: ChunkStats<Seconds> = ChunkStats::default();
        chunk.timestamps.insert(Abs::from_raw(10));
        chunk.timestamps.insert(Abs::from_raw(20));
        chunk.timestamp_counts.insert(Abs::from_raw(10), 7);
        chunk.timestamp_counts.insert(Abs::from_raw(20), 3);
        chunk.num_train_rows = 10;

        merger.merge(chunk);
        let out = merger.finalize(
            10,
            true,
            false,
            false,
            Some(&[1.0, 0.0]),
            Some("BacktestingSplit"),
            true,
            None,
            None,
            Some(Abs::from_raw(20)),
            None,
            None,
            None,
            None,
            None,
            "numerical",
            None,
        );
        let counts = out
            .split_payload
            .as_ref()
            .and_then(|p| p.mask_counts.as_ref())
            .expect("mask counts should be present for BacktestingSplit");
        assert_eq!(counts.train, 0);
        assert_eq!(counts.val, 0);
        assert_eq!(counts.test, 3);
    }

    #[test]
    fn test_finalize_fixed_split_counts_match_largest_remainder() {
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        let chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 10,
            ..Default::default()
        };
        merger.merge(chunk);

        let out = merger.finalize(
            10,
            true,
            false,
            false,
            Some(&[0.34, 0.33, 0.33]),
            Some("FixedSplit"),
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "numerical",
            None,
        );

        let counts = out
            .split_payload
            .as_ref()
            .and_then(|p| p.mask_counts.as_ref())
            .expect("mask counts should be present for FixedSplit");
        assert_eq!(counts.train, 4);
        assert_eq!(counts.val, 3);
        assert_eq!(counts.test, 3);
    }
}
