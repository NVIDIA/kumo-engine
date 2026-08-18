use std::collections::HashMap;

use arrow_array::{Array, ListArray};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SequenceFeatureStats {
    pub count: u64,
    pub null_count: u64,
    pub min_len: Option<i64>,
    pub max_len: Option<i64>,
}

impl SequenceFeatureStats {
    pub fn observe(&mut self, array: &dyn Array, row: usize) {
        if array.is_null(row) {
            self.null_count += 1;
            return;
        }
        if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
            let len = list.value_length(row) as i64;
            // Treat a zero-length list as a null embedding.
            if len == 0 {
                self.null_count += 1;
                return;
            }
            self.count += 1;
            self.min_len = Some(self.min_len.map_or(len, |current| current.min(len)));
            self.max_len = Some(self.max_len.map_or(len, |current| current.max(len)));
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.null_count += other.null_count;
        if let Some(other_min) = other.min_len {
            self.min_len = Some(self.min_len.map_or(other_min, |m| m.min(other_min)));
        }
        if let Some(other_max) = other.max_len {
            self.max_len = Some(self.max_len.map_or(other_max, |m| m.max(other_max)));
        }
    }
}

use diskgraph_core::accumulators::{
    build_multicat_emit_stats, kumo_entry, kumo_key, kumo_safe_div, HyperLogLog, LocalColumnCounts,
    TDigestEstimator, WelfordAccumulator,
};
use diskgraph_core::time::{Abs, UnitSystem};
use diskgraph_core::tt_builder::Value;

use crate::split_decider::SplitDecider;

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

/// Bounded Kumo-contract train-table stats fields. Shape derived from
/// the legacy `build_tt_stats_artifact_value` formatter on origin/main and
/// `kumo.stats.train_table_stats.{TemporalNodeTrainTableStats,
/// TemporalLinkTrainTableStats}`. Kumo's
/// `seed_train_stats_cache_from_native_artifact` reads these eight keys
/// directly from the top-level stats.json dict.
///
/// `target_distribution` is stored as `JsonValue` because its concrete shape
/// is task-type-discriminated and contains f64s that block `Eq`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct KumoContractStats {
    pub num_target_rows: u64,
    pub num_timeframes: u64,
    pub num_timeframes_post_filter: u64,
    pub avg_entities_per_timeframe: u64,
    /// Top-level minimum anchor date (RFC3339 string, derived from
    /// `timestamp_min`). Required by kumo's `TemporalTrainTableStats`.
    /// `null` when no rows / no temporal data.
    pub min_date: JsonValue,
    /// Top-level maximum anchor date (RFC3339 string, derived from
    /// `timestamp_max`). Required by kumo's `TemporalTrainTableStats`.
    pub max_date: JsonValue,
    /// Top-level anchor date — same as `max_date` per kumo's convention.
    pub anchor_date: JsonValue,
    /// Always `null` today — the diskgraph path doesn't compute degree
    /// stats. Display-only in kumo (per UI grep).
    pub node_degree_stats: JsonValue,
    /// Always `null` today — display-only in kumo.
    pub scoring_table_column_stats: JsonValue,
    /// Categorical: `{count: {key: int}, percentage: {key: float}}`.
    /// Numerical: `{target_mean, target_min, target_max, target_percentile_25,
    /// target_percentile_75, target_median, target_mode, target_mode_percentage,
    /// target_mode_count, histogram: {bins, counts, bin_width, num_bins}}`.
    pub target_distribution: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableStatsPayload {
    pub version: u32,
    pub num_total_rows: u64,
    pub num_train_rows: u64,
    pub num_entities: u64,
    pub num_targets: u64,
    pub num_links: Option<u64>,
    pub exact_stats: TrainTableExactStatsPayload,
    /// Kumo-format column stats for node-task encoder inference. Link
    /// prediction has no train-table target encoder and omits this sidecar.
    /// Format matches ingest's kumo_col_stats_v1 for compatibility with
    /// seed_train_table_col_stats_from_manifest().
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsed_train_table_col_stats: Option<JsonValue>,
    /// Bounded Kumo cache fields, flattened in executor summaries as well as
    /// the on-disk artifact so no nested contract copy is serialized.
    #[serde(flatten)]
    pub kumo_contract: KumoContractStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TTExecutionSummary {
    pub rows_written: u64,
    pub stats: Option<TrainTableStatsPayload>,
}

// Cardinality tracking is HLL-only. Removed prior `HashSet<u64>` fields
// (entities / targets / timestamps) that grew unboundedly with the chunk's
// distinct cardinality (OOM risk at 50B-row scale: ~5 GB per worker for
// entities alone, ~80 GB across 16 workers, then ~800 GB peak in the
// merger). HLL gives ~0.5% error within bounded ~12 KB per accumulator
// regardless of input cardinality.
pub(crate) struct ChunkStats<U: UnitSystem> {
    pub(crate) num_total_rows: u64,
    pub(crate) num_train_rows: u64,
    pub(crate) num_links: u64,
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
    /// Mirrors the StatsMerger field. Set once per chunk from
    /// ``plan.primary_target_dtype() == DataType::Boolean`` and OR'd into the
    /// merger so CATEGORY_COUNTS emits JSON booleans (not lowercase
    /// strings) for bool targets.
    pub(crate) target_is_bool_dtype: bool,
    pub(crate) sequence_feature_stats: HashMap<String, SequenceFeatureStats>,
}

impl<U: UnitSystem> Default for ChunkStats<U> {
    fn default() -> Self {
        Self {
            num_total_rows: 0,
            num_train_rows: 0,
            num_links: 0,
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
            target_is_bool_dtype: false,
            sequence_feature_stats: HashMap::new(),
        }
    }
}

pub(crate) struct StatsMerger<U: UnitSystem> {
    num_train_rows: u64,
    num_links: u64,
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
    /// True when the TARGET Arrow column was BooleanArray. Required so
    /// CATEGORY_COUNTS gets emitted as JSON booleans (matching GE/Spark).
    pub(crate) target_is_bool_dtype: bool,
    pub(crate) sequence_feature_stats: HashMap<String, SequenceFeatureStats>,
}

impl<U: UnitSystem> Default for StatsMerger<U> {
    fn default() -> Self {
        Self {
            num_train_rows: 0,
            num_links: 0,
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
            target_is_bool_dtype: false,
            sequence_feature_stats: HashMap::new(),
        }
    }
}

impl<U: UnitSystem> StatsMerger<U> {
    pub(crate) fn merge(&mut self, chunk: ChunkStats<U>) {
        self.num_train_rows += chunk.num_train_rows;
        self.num_links += chunk.num_links;
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
            self.target_welford
                .get_or_insert_with(WelfordAccumulator::new)
                .merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.target_tdigest {
            self.target_tdigest
                .get_or_insert_with(|| TDigestEstimator::new(200))
                .merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.target_hll {
            self.target_hll
                .get_or_insert_with(HyperLogLog::new)
                .merge(&chunk_h);
        }
        self.target_invalid_count += chunk.target_invalid_count;
        // Merge frequency distribution accumulators
        if let Some(chunk_w) = chunk.frequency_welford {
            self.frequency_welford
                .get_or_insert_with(WelfordAccumulator::new)
                .merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.frequency_tdigest {
            self.frequency_tdigest
                .get_or_insert_with(|| TDigestEstimator::new(200))
                .merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.frequency_hll {
            self.frequency_hll
                .get_or_insert_with(HyperLogLog::new)
                .merge(&chunk_h);
        }
        self.frequency_invalid_count += chunk.frequency_invalid_count;
        // Merge entity distribution accumulators
        if let Some(chunk_h) = chunk.entity_hll {
            self.entity_hll
                .get_or_insert_with(HyperLogLog::new)
                .merge(&chunk_h);
        }
        if let Some(chunk_w) = chunk.entity_welford {
            self.entity_welford
                .get_or_insert_with(WelfordAccumulator::new)
                .merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.entity_tdigest {
            self.entity_tdigest
                .get_or_insert_with(|| TDigestEstimator::new(200))
                .merge(&mut chunk_t);
        }
        // Merge timestamp distribution accumulators
        if let Some(chunk_w) = chunk.timestamp_welford {
            self.timestamp_welford
                .get_or_insert_with(WelfordAccumulator::new)
                .merge(&chunk_w);
        }
        if let Some(mut chunk_t) = chunk.timestamp_tdigest {
            self.timestamp_tdigest
                .get_or_insert_with(|| TDigestEstimator::new(200))
                .merge(&mut chunk_t);
        }
        if let Some(chunk_h) = chunk.timestamp_hll {
            self.timestamp_hll
                .get_or_insert_with(HyperLogLog::new)
                .merge(&chunk_h);
        }
        // Sticky: any chunk that observed a bool target column flips the
        // merger flag for the rest of the run.
        self.target_is_bool_dtype |= chunk.target_is_bool_dtype;

        for (name, stats) in chunk.sequence_feature_stats {
            self.sequence_feature_stats
                .entry(name)
                .or_default()
                .merge(&stats);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize(
        mut self,
        num_total_rows: u64,
        is_link_prediction: bool,
        has_frequency_col: bool,
        has_include_bit_col: bool,
        target_category_counts: Option<HashMap<String, u64>>,
        target_stype: &str,
        entity_category_counts: Option<HashMap<String, u64>>,
    ) -> TrainTableStatsPayload {
        // Cardinality is HLL-only. For PassThrough or any path
        // that left the HLL as None, fall back to 0 — same defensive value
        // the prior empty-HashSet branch returned.
        let num_entities = self
            .entity_hll
            .as_ref()
            .map(|h| h.estimate() as u64)
            .unwrap_or(0);
        let num_targets = self
            .target_hll
            .as_ref()
            .map(|h| h.estimate() as u64)
            .unwrap_or(0);
        let estimated_num_unique_timestamps = self
            .timestamp_hll
            .as_ref()
            .map(|h| h.estimate() as u64)
            .unwrap_or(0);

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
            num_unique: Some(estimated_num_unique_timestamps),
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

        // LP targets are graph node IDs. Their exact category distribution is
        // neither an encoder input nor a registered LP statistic, so keep the
        // categorical contract empty while retaining scalar/HLL counts below.
        let is_categorical_target_for_dist = matches!(
            target_stype,
            "categorical" | "boolean" | "ID" | "multicategorical"
        );
        let target_distribution_value = if is_link_prediction && is_categorical_target_for_dist {
            json!({})
        } else if is_categorical_target_for_dist {
            build_target_distribution_categorical(target_category_counts.as_ref())
        } else {
            build_target_distribution_numerical(&self.target_welford, &mut self.target_tdigest)
        };

        // Exact timestamp counts are split-decider-local when needed. Global
        // diagnostics use the bounded HLL estimate instead.
        let num_timeframes_u64 = estimated_num_unique_timestamps;
        let avg_entities_per_timeframe = if num_timeframes_u64 > 0 {
            (self.num_train_rows as f64 / num_timeframes_u64 as f64).round() as u64
        } else {
            0
        };

        // Format epoch-second bounds as RFC3339 strings for Kumo's cache.
        use chrono::DateTime;
        let fmt_ts = |t: Abs<U>| -> JsonValue {
            let secs = t.to_secs();
            DateTime::from_timestamp(secs, 0)
                .map(|dt| json!(dt.to_rfc3339()))
                .unwrap_or_else(|| json!(secs))
        };
        let min_date_value = self.timestamp_min.map(fmt_ts).unwrap_or(JsonValue::Null);
        let max_date_value = self.timestamp_max.map(fmt_ts).unwrap_or(JsonValue::Null);
        // anchor_date == max_date by kumo convention.
        let anchor_date_value = max_date_value.clone();

        let kumo_contract = KumoContractStats {
            // num_target_rows: kumo's TemporalNodeTrainTableStats / NUM_TARGET_ROWS
            // is the count of target-bearing rows. For node-pred this is
            // num_train_rows (each emitted train row carries a target). For
            // link-pred kumo also accepts num_links semantically; the legacy
            // formatter wired num_train_rows in both cases, so we match.
            num_target_rows: if is_link_prediction {
                self.num_links
            } else {
                self.num_train_rows
            },
            num_timeframes: num_timeframes_u64,
            // No post-filter pass today; equal to num_timeframes.
            num_timeframes_post_filter: num_timeframes_u64,
            avg_entities_per_timeframe,
            min_date: min_date_value,
            max_date: max_date_value,
            anchor_date: anchor_date_value,
            node_degree_stats: JsonValue::Null,
            scoring_table_column_stats: JsonValue::Null,
            target_distribution: target_distribution_value,
        };

        let parsed_train_table_col_stats = if is_link_prediction {
            None
        } else {
            build_inline_col_stats(
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
                self.target_is_bool_dtype,
                &self.entity_hll,
                &self.entity_welford,
                &mut self.entity_tdigest,
                entity_category_counts,
                &self.timestamp_welford,
                &mut self.timestamp_tdigest,
                &self.timestamp_hll,
                self.sequence_feature_stats.clone(),
            )
        };

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
            parsed_train_table_col_stats,
            kumo_contract,
            exact_stats: TrainTableExactStatsPayload {
                entity: entity_stats,
                timestamp: timestamp_stats,
                target: target_stats,
                frequency: frequency_stats,
                entity_include_bit: include_bit_stats,
            },
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
    target_is_bool_dtype: bool,
    entity_hll: &Option<HyperLogLog>,
    entity_welford: &Option<WelfordAccumulator>,
    entity_tdigest: &mut Option<TDigestEstimator>,
    entity_category_counts: Option<HashMap<String, u64>>,
    timestamp_welford: &Option<WelfordAccumulator>,
    timestamp_tdigest: &mut Option<TDigestEstimator>,
    timestamp_hll: &Option<HyperLogLog>,
    sequence_feature_stats: HashMap<String, SequenceFeatureStats>,
) -> Option<JsonValue> {
    let mut stats = serde_json::Map::new();
    let dv = data_version;

    for (name, s) in sequence_feature_stats {
        let total = s.count + s.null_count;
        let stype = "kumo.sequence";
        stats.insert(
            kumo_key(&name, stype, "COUNT", false),
            kumo_entry(json!(total), false, dv, total),
        );
        stats.insert(
            kumo_key(&name, stype, "NUM_NA", false),
            kumo_entry(json!(s.null_count), false, dv, total),
        );
        stats.insert(
            kumo_key(&name, stype, "NA_FRACTION", false),
            kumo_entry(json!(kumo_safe_div(s.null_count, total)), false, dv, total),
        );
        stats.insert(
            kumo_key(&name, stype, "INVALID_FRACTION", false),
            kumo_entry(json!(0.0), false, dv, total),
        );
        stats.insert(
            kumo_key(&name, stype, "SEQUENCE_MIN_LENGTH", false),
            kumo_entry(json!(s.min_len.unwrap_or(0)), false, dv, total),
        );
        stats.insert(
            kumo_key(&name, stype, "SEQUENCE_MAX_LENGTH", false),
            kumo_entry(json!(s.max_len.unwrap_or(0)), false, dv, total),
        );
    }

    let is_categorical_target = matches!(target_stype, "categorical" | "boolean" | "ID");
    let is_multicat_target = target_stype == "multicategorical";

    // TARGET categorical stats
    if is_categorical_target {
        if let Some(ref cat_counts) = target_category_counts {
            let count: u64 = cat_counts.values().sum();
            let stype = "kumo.categorical";
            let name = "TARGET1";

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
            //
            // For boolean targets we emit JSON booleans (true/false), not
            // strings, so the kumo encoder builds its mapping with bool keys
            // — matching pandas/GE behaviour. Otherwise pd.Series.map() over
            // a bool column with string keys yields all-NaN -> all -1, and
            // the trainer fails the val-imbalance check on the first batch.
            let mut pairs: Vec<(&String, &u64)> = cat_counts.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1)); // descending by count
                                                // target_stype "boolean" OR Arrow BooleanArray rehydration both
                                                // signal that the kumo encoder will see a bool-dtype column.
            let target_is_boolean = target_stype == "boolean" || target_is_bool_dtype;
            let category_array: Vec<JsonValue> = pairs
                .iter()
                .map(|(k, c)| {
                    let key_value = if target_is_boolean {
                        match k.as_str() {
                            "true" => json!(true),
                            "false" => json!(false),
                            _ => json!(k),
                        }
                    } else {
                        json!(k)
                    };
                    json!([key_value, c])
                })
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
            let name = "TARGET1";

            // Sort category counts count-desc, token-asc
            let mut sorted_cats: Vec<(String, u64)> =
                cat_counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
            sorted_cats.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

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
            let name = "TARGET1";
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
                kumo_entry(
                    json!(kumo_safe_div(target_invalid_count, count)),
                    false,
                    dv,
                    count,
                ),
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
            kumo_entry(
                json!(kumo_safe_div(frequency_invalid_count, count)),
                false,
                dv,
                count,
            ),
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
        let name = "ENTITY1";

        // Empty means not collected. Emit no CATEGORY_COUNTS key at all, so a
        // consumer fails on the missing key instead of reading an empty
        // vocabulary as "no categories".
        if let Some(ref cat_counts) = entity_category_counts.filter(|c| !c.is_empty()) {
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
            let category_array: Vec<JsonValue> = pairs.iter().map(|(k, c)| json!([k, c])).collect();
            stats.insert(
                kumo_key(name, stype, "CATEGORY_COUNTS", false),
                kumo_entry(json!(category_array), false, dv, effective_count),
            );

            let num_unique_exact = cat_counts.len() as u64;
            if let Some(h) = entity_hll {
                let hll_estimate = (h.estimate() as u64).min(effective_count);
                stats.insert(
                    kumo_key(name, stype, "NUM_UNIQUE", false),
                    kumo_entry(
                        json!(hll_estimate.max(num_unique_exact)),
                        false,
                        dv,
                        effective_count,
                    ),
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
            let name = "ENTITY1";
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

// Kumo-contract target distribution.

// Kumo-contract target distribution.
/// `{count: {key: int}, percentage: {key: float}}`. Mirrors kumo's
/// `CategoricalDictTargetDistribution`. Returns `{}` when no category
/// counts are available (defensive — kumo treats empty dict as missing
/// distribution, which is acceptable for display).
fn build_target_distribution_categorical(counts: Option<&HashMap<String, u64>>) -> JsonValue {
    let counts = match counts {
        Some(c) if !c.is_empty() => c,
        _ => return json!({}),
    };
    let total: u64 = counts.values().sum();
    let mut count_map = serde_json::Map::new();
    let mut pct_map = serde_json::Map::new();
    // Stable ordering by key for deterministic JSON output.
    let mut entries: Vec<(&String, &u64)> = counts.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (k, v) in entries {
        count_map.insert(k.clone(), json!(v));
        let pct = if total > 0 {
            *v as f64 / total as f64
        } else {
            0.0
        };
        pct_map.insert(k.clone(), json!(pct));
    }
    json!({
        "count": JsonValue::Object(count_map),
        "percentage": JsonValue::Object(pct_map),
    })
}

/// Build the numerical-shape `target_distribution` value. Kumo's
/// `NumericalDictTargetDistribution` shape is:
/// `{target_mean, target_min, target_max, target_percentile_25,
/// target_percentile_75, target_median, target_mode, target_mode_percentage,
/// target_mode_count, histogram: {bins, counts, bin_width, num_bins}}`.
///
/// Streaming accumulators give us mean/min/max (Welford) and the three
/// percentiles (t-digest). The mode-related fields are stubbed
/// (display-only in kumo per consumer grep). The histogram is stubbed
/// to an empty bins/counts pair: t-digest can sample edges but does not
/// give exact per-bin counts without re-streaming, and kumo's consumers
/// don't read it for training/split logic — only for UI display.
fn build_target_distribution_numerical(
    welford: &Option<WelfordAccumulator>,
    tdigest: &mut Option<TDigestEstimator>,
) -> JsonValue {
    let (mean, min, max) = match welford.as_ref() {
        Some(w) if w.count > 0 => (w.mean(), w.min, w.max),
        _ => (0.0_f64, 0.0_f64, 0.0_f64),
    };
    let (p25, p50, p75) = match tdigest.as_mut() {
        Some(t) => (
            t.quantile_at(0.25),
            t.quantile_at(0.50),
            t.quantile_at(0.75),
        ),
        None => (0.0_f64, 0.0_f64, 0.0_f64),
    };
    json!({
        "target_mean": mean,
        "target_min": min,
        "target_max": max,
        "target_percentile_25": p25,
        "target_percentile_75": p75,
        "target_median": p50,
        // mode is not streamed for floats; stubbed (display-only in kumo).
        "target_mode": min,
        "target_mode_percentage": 0.0,
        "target_mode_count": 0,
        // Histogram: t-digest has centroids but not exact bin counts.
        // Kumo's consumers grep showed display-only — stub empty.
        "histogram": {
            "bins": JsonValue::Array(Vec::new()),
            "counts": JsonValue::Array(Vec::new()),
            "bin_width": 0.0,
            "num_bins": 0,
        },
    })
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

// ─── TT-NG RowObserver / RowProcessor abstraction ────────────────────────────
//
// This block introduces the trait shape described in design doc §5.2:
//
//   - `TTRowRef<'a, U>` is a zero-copy bundle of refs to the per-row data
//     that observers consume. It mirrors today's `update_exact_stats`
//     argument list, modulo per-worker scratch state (see note below).
//
//   - `RowObserver<U>` is the trait both `ChunkStats<U>` and real
//     `SplitDecider` impls satisfy. Used purely for shape uniformity
//     at the type level. NOT used via dyn dispatch.
//
//   - `RowProcessor<U, D>` is the concrete tuple wrapping `ChunkStats<U>`
//     and a `D: SplitDecider`. Monomorphized per (U, D) pair. No vtable.
//
// ## Local counts: NOT in `TTRowRef`
//
// `local_target_counts` and `local_entity_counts` are PER-WORKER SCRATCH
// state, not row data. They are passed by `&mut` through `RowProcessor::observe`
// because today's `update_exact_stats` writes into them. We deliberately
// keep them OUT of `TTRowRef` (they are not row content) and OUT of
// `RowProcessor` fields to minimize blast radius.
//
// ## Performance contract
//
// `RowProcessor::observe` is `#[inline]` so cross-crate inlining can elide
// the wrapper. Inside, both `self.stats.observe(...)` and
// `self.decider.accumulate(...)` are statically dispatched via the
// concrete `D: SplitDecider` parameter — no `Box<dyn>`, no vtable.

/// Zero-copy view of one TT row used by `RowObserver` impls.
///
/// Field set matches what today's `update_exact_stats` consumes. Lifetime
/// `'a` ties this view to the worker-local scratch buffers it borrows
/// from (`entity_pk`, `target`).
///
/// Optional `split_column_value` lets `ColumnDecider` impls observe
/// per-row split-column values during pass A. `None` for every
/// non-ColumnSplit family (zero-cost — same field layout, just an extra
/// pointer that's unused).
#[allow(dead_code)]
pub(crate) struct TTRowRef<'a, U: UnitSystem> {
    pub(crate) entity_pk: &'a Value,
    pub(crate) ts: Abs<U>,
    pub(crate) target: &'a Value,
    pub(crate) frequency: Option<i64>,
    pub(crate) target_is_numerical: bool,
    pub(crate) target_is_categorical: bool,
    pub(crate) target_stype: &'a str,
    /// Per-row split-column value, populated by callers when the plan's
    /// split family is a `ColumnSplit` (kind `"ColumnSplit"`). `None` for
    /// every other family. Read by `ColumnDecider` impls in `accumulate`
    /// and (via `decide`) when they don't have a `BatchView` cache.
    pub(crate) split_column_value: Option<&'a Value>,
}

/// Per-row observer. Implemented by `ChunkStats<U>` (in this file) and
/// by call-site wrappers; used only for type-level shape uniformity.
///
/// `Send` so observers can cross rayon worker boundaries.
#[allow(dead_code)]
pub(crate) trait RowObserver<U: UnitSystem>: Send {
    /// Observe one row. `local_target_counts` / `local_entity_counts` are
    /// per-worker scratch buffers (NOT row data) that today's stats path
    /// writes category-count info into.
    fn observe(
        &mut self,
        row: &TTRowRef<'_, U>,
        local_target_counts: &mut Option<LocalColumnCounts>,
        local_entity_counts: &mut Option<LocalColumnCounts>,
    );

    /// Fold another observer's state into self.
    fn merge(&mut self, other: Self)
    where
        Self: Sized;
}

impl<U: UnitSystem> RowObserver<U> for ChunkStats<U> {
    #[inline]
    fn observe(
        &mut self,
        row: &TTRowRef<'_, U>,
        local_target_counts: &mut Option<LocalColumnCounts>,
        local_entity_counts: &mut Option<LocalColumnCounts>,
    ) {
        // Delegate to the existing per-row stats logic. Trait-shape only;
        // no behavior change.
        crate::ttgen_executor::update_exact_stats(
            self,
            row.entity_pk,
            row.ts,
            row.target,
            row.frequency,
            row.target_is_numerical,
            row.target_is_categorical,
            row.target_stype,
            local_target_counts,
            local_entity_counts,
        );
    }

    fn merge(&mut self, other: ChunkStats<U>) {
        merge_chunk_stats_into(self, other);
    }
}

/// Concrete composition of `ChunkStats<U>` and a `D: SplitDecider`.
///
/// Monomorphized per (U, D). Both inner observers are statically dispatched
/// from `observe`. No `dyn`, no `Box`, no vtable.
#[allow(dead_code)]
pub(crate) struct RowProcessor<U: UnitSystem, D: SplitDecider<U>> {
    pub(crate) stats: ChunkStats<U>,
    pub(crate) decider: D,
}

#[allow(dead_code)]
impl<U: UnitSystem, D: SplitDecider<U>> RowProcessor<U, D> {
    pub(crate) fn new(stats: ChunkStats<U>, decider: D) -> Self {
        Self { stats, decider }
    }

    /// Observe one row. Drives both the stats accumulator and the
    /// split decider's per-row accumulate. Statically dispatched.
    #[inline]
    pub(crate) fn observe(
        &mut self,
        row: &TTRowRef<'_, U>,
        local_target_counts: &mut Option<LocalColumnCounts>,
        local_entity_counts: &mut Option<LocalColumnCounts>,
    ) {
        // Both calls are monomorphized — the compiler resolves D's accumulate
        // at compile time and elides NoSplitDecider's no-op entirely.
        <ChunkStats<U> as RowObserver<U>>::observe(
            &mut self.stats,
            row,
            local_target_counts,
            local_entity_counts,
        );
        self.decider.accumulate(row);
    }

    /// Fold another `RowProcessor` into self. Combines stats + decider.
    pub(crate) fn merge(&mut self, other: RowProcessor<U, D>) {
        let RowProcessor { stats, decider } = other;
        <ChunkStats<U> as RowObserver<U>>::merge(&mut self.stats, stats);
        self.decider.merge(&decider);
    }
}

/// Shared helper: fold `other` into `self` for two `ChunkStats` instances.
/// Mirrors the field-by-field logic in `StatsMerger::merge` but on a
/// `ChunkStats` accumulator (used by `<ChunkStats as RowObserver>::merge`).
#[allow(dead_code)] // Unused outside the `RowObserver` impl + tests.
fn merge_chunk_stats_into<U: UnitSystem>(dst: &mut ChunkStats<U>, src: ChunkStats<U>) {
    dst.num_total_rows += src.num_total_rows;
    dst.num_train_rows += src.num_train_rows;
    dst.num_links += src.num_links;
    dst.timestamp_min = min_opt_abs(dst.timestamp_min, src.timestamp_min);
    dst.timestamp_max = max_opt_abs(dst.timestamp_max, src.timestamp_max);
    dst.entity_min = min_opt(dst.entity_min, src.entity_min);
    dst.entity_max = max_opt(dst.entity_max, src.entity_max);
    dst.target_min = min_opt(dst.target_min, src.target_min);
    dst.target_max = max_opt(dst.target_max, src.target_max);
    dst.frequency_min = min_opt(dst.frequency_min, src.frequency_min);
    dst.frequency_max = max_opt(dst.frequency_max, src.frequency_max);
    dst.include_true_count += src.include_true_count;
    dst.include_false_count += src.include_false_count;

    if let Some(s) = src.target_welford {
        dst.target_welford
            .get_or_insert_with(WelfordAccumulator::new)
            .merge(&s);
    }
    if let Some(mut s) = src.target_tdigest {
        dst.target_tdigest
            .get_or_insert_with(|| TDigestEstimator::new(200))
            .merge(&mut s);
    }
    if let Some(s) = src.target_hll {
        dst.target_hll
            .get_or_insert_with(HyperLogLog::new)
            .merge(&s);
    }
    dst.target_invalid_count += src.target_invalid_count;

    if let Some(s) = src.frequency_welford {
        dst.frequency_welford
            .get_or_insert_with(WelfordAccumulator::new)
            .merge(&s);
    }
    if let Some(mut s) = src.frequency_tdigest {
        dst.frequency_tdigest
            .get_or_insert_with(|| TDigestEstimator::new(200))
            .merge(&mut s);
    }
    if let Some(s) = src.frequency_hll {
        dst.frequency_hll
            .get_or_insert_with(HyperLogLog::new)
            .merge(&s);
    }
    dst.frequency_invalid_count += src.frequency_invalid_count;

    if let Some(s) = src.entity_hll {
        dst.entity_hll
            .get_or_insert_with(HyperLogLog::new)
            .merge(&s);
    }
    if let Some(s) = src.entity_welford {
        dst.entity_welford
            .get_or_insert_with(WelfordAccumulator::new)
            .merge(&s);
    }
    if let Some(mut s) = src.entity_tdigest {
        dst.entity_tdigest
            .get_or_insert_with(|| TDigestEstimator::new(200))
            .merge(&mut s);
    }

    if let Some(s) = src.timestamp_welford {
        dst.timestamp_welford
            .get_or_insert_with(WelfordAccumulator::new)
            .merge(&s);
    }
    if let Some(mut s) = src.timestamp_tdigest {
        dst.timestamp_tdigest
            .get_or_insert_with(|| TDigestEstimator::new(200))
            .merge(&mut s);
    }
    if let Some(s) = src.timestamp_hll {
        dst.timestamp_hll
            .get_or_insert_with(HyperLogLog::new)
            .merge(&s);
    }

    dst.target_is_bool_dtype |= src.target_is_bool_dtype;

    for (name, stats) in src.sequence_feature_stats {
        dst.sequence_feature_stats
            .entry(name)
            .or_default()
            .merge(&stats);
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
        // Cardinality is HLL-only. Seed the per-chunk HLLs the way
        // `execute_chunk` would, then observe the same handful of
        // (entity, target, timestamp) values. With M = 16384 registers
        // the estimate is exact for these small fixtures.
        chunk.entity_hll = Some(HyperLogLog::new());
        chunk.target_hll = Some(HyperLogLog::new());
        chunk.timestamp_hll = Some(HyperLogLog::new());
        chunk.entity_hll.as_mut().unwrap().observe_i64(10);
        chunk.entity_hll.as_mut().unwrap().observe_i64(20);
        chunk.target_hll.as_mut().unwrap().observe_i64(100);
        chunk.timestamp_hll.as_mut().unwrap().observe_i64(1000);
        chunk.timestamp_hll.as_mut().unwrap().observe_i64(2000);
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
        let out = merger.finalize(4, true, true, true, None, "numerical", None);

        assert_eq!(out.version, 2);
        assert_eq!(out.num_total_rows, 4);
        assert_eq!(out.num_train_rows, 3);
        assert_eq!(out.num_entities, 2);
        assert_eq!(out.num_targets, 1);
        assert_eq!(out.num_links, Some(2));
        assert_eq!(out.exact_stats.timestamp.min, Some(1000));
        assert_eq!(out.exact_stats.timestamp.max, Some(2000));
        assert_eq!(out.exact_stats.entity_include_bit.unwrap().false_count, 1);
    }
    #[test]
    fn test_merge_propagates_target_is_bool_dtype_when_chunk_sets_it() {
        // A single chunk that observed a bool target dtype must flip the
        // merger flag. This is the regression guard for the ttgen path.
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        assert!(!merger.target_is_bool_dtype, "merger starts false");

        let chunk: ChunkStats<Seconds> = ChunkStats {
            target_is_bool_dtype: true,
            ..Default::default()
        };
        merger.merge(chunk);
        assert!(
            merger.target_is_bool_dtype,
            "merge must OR target_is_bool_dtype from chunk into merger"
        );
    }

    #[test]
    fn test_merge_target_is_bool_dtype_is_sticky_across_chunks() {
        // Once any chunk flips the flag, subsequent chunks (even with
        // false) must not unset it. Defensive in case some chunks see
        // an empty target column and default to false.
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();

        merger.merge(ChunkStats::<Seconds> {
            target_is_bool_dtype: true,
            ..Default::default()
        });
        merger.merge(ChunkStats::<Seconds> {
            target_is_bool_dtype: false,
            ..Default::default()
        });
        assert!(
            merger.target_is_bool_dtype,
            "false-valued chunk must not clear the flag"
        );
    }

    #[test]
    fn test_finalize_emits_json_booleans_for_bool_target() {
        // End-to-end: a categorical target with target_is_bool_dtype = true
        // and string-keyed category counts ("true"/"false", as ttgen records
        // them via Display for bool) must surface as JSON booleans in the
        // emitted CATEGORY_COUNTS array.
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        merger.merge(ChunkStats::<Seconds> {
            num_train_rows: 10,
            target_is_bool_dtype: true,
            ..Default::default()
        });

        let mut counts = HashMap::new();
        counts.insert("false".to_string(), 7);
        counts.insert("true".to_string(), 3);

        let payload = merger.finalize(10, false, false, false, Some(counts), "categorical", None);

        let outer = payload
            .parsed_train_table_col_stats
            .as_ref()
            .expect("parsed_train_table_col_stats must be present");
        let stats = outer.get("stats").expect("outer must have 'stats' wrapper");
        let key = stats
            .as_object()
            .unwrap()
            .keys()
            .find(|k| k.starts_with("TARGET1__") && k.contains("CATEGORY_COUNTS"))
            .expect("TARGET1 CATEGORY_COUNTS key must exist");
        let entry = stats.get(key).unwrap();
        let pairs = entry.get("value").and_then(|v| v.as_array()).unwrap();
        assert!(!pairs.is_empty());
        for pair in pairs {
            let arr = pair.as_array().unwrap();
            let key = &arr[0];
            assert!(
                key.is_boolean(),
                "expected JSON boolean key, got {key:?} (type {:?})",
                key
            );
        }
    }

    #[test]
    fn test_finalize_emits_string_keys_when_target_is_bool_dtype_is_false() {
        // Mirror of the previous test: when the flag is NOT set and stype
        // is "categorical" (kumo's convention for binary classification),
        // emit must NOT promote string keys to JSON booleans. Asserts the
        // fix is gated correctly and doesn't over-fire on plain string
        // categoricals that happen to use "true"/"false" as labels.
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        merger.merge(ChunkStats::<Seconds> {
            num_train_rows: 10,
            target_is_bool_dtype: false,
            ..Default::default()
        });

        let mut counts = HashMap::new();
        counts.insert("false".to_string(), 7);
        counts.insert("true".to_string(), 3);

        let payload = merger.finalize(10, false, false, false, Some(counts), "categorical", None);
        let outer = payload.parsed_train_table_col_stats.unwrap();
        let stats = outer.get("stats").expect("outer must have 'stats' wrapper");
        let key = stats
            .as_object()
            .unwrap()
            .keys()
            .find(|k| k.starts_with("TARGET1__") && k.contains("CATEGORY_COUNTS"))
            .expect("TARGET1 CATEGORY_COUNTS key must exist");
        let pairs = stats
            .get(key)
            .unwrap()
            .get("value")
            .unwrap()
            .as_array()
            .unwrap();
        for pair in pairs {
            let key = &pair.as_array().unwrap()[0];
            assert!(
                key.is_string(),
                "expected JSON string key when flag is false, got {key:?}"
            );
        }
    }

    // ─── RowObserver / RowProcessor / TTRowRef tests ────────────────────────
    //
    // These tests exercise the TT-NG observer/processor abstraction.
    // They reference types defined in the production section of this file
    // (`TTRowRef`, `RowObserver`, `RowProcessor`) and `NoSplitDecider` from
    // the sibling `split_decider` module.

    use crate::split_decider::{NoSplitDecider, SplitDecider, UNUSED_STAGE};
    use diskgraph_core::accumulators::LocalColumnCounts;
    use diskgraph_core::tt_builder::Value;

    fn make_row<'a>(
        entity_pk: &'a Value,
        ts: Abs<Seconds>,
        target: &'a Value,
    ) -> TTRowRef<'a, Seconds> {
        TTRowRef {
            entity_pk,
            ts,
            target,
            frequency: None,
            target_is_numerical: true,
            target_is_categorical: false,
            target_stype: "numerical",
            split_column_value: None,
        }
    }

    #[test]
    fn test_row_processor_constructs() {
        let processor: RowProcessor<Seconds, NoSplitDecider> =
            RowProcessor::new(ChunkStats::default(), NoSplitDecider);
        assert_eq!(processor.stats.num_total_rows, 0);
        assert_eq!(processor.stats.num_train_rows, 0);
        // Decider is a ZST — verify type stays no-state.
        assert_eq!(std::mem::size_of_val(&processor.decider), 0);
    }

    /// Seed all per-chunk HLL accumulators the way `execute_chunk` would.
    /// Tests need this because cardinality is HLL-only and the observe
    /// paths short-circuit when an HLL is `None`.
    fn seed_hlls(stats: &mut ChunkStats<Seconds>) {
        stats.entity_hll = Some(HyperLogLog::new());
        stats.target_hll = Some(HyperLogLog::new());
        stats.timestamp_hll = Some(HyperLogLog::new());
    }

    #[test]
    fn test_row_processor_observe_delegates_to_chunk_stats() {
        // Build a RowProcessor and a fresh ChunkStats; feed an identical
        // row via processor.observe(...) and via the existing
        // update_exact_stats path. Resulting cardinality estimates and
        // min/max state must match.
        let entity = Value::Int(42);
        let target = Value::Int(7);
        let ts = Abs::from_raw(1_000);
        let row = make_row(&entity, ts, &target);

        let mut p_stats = ChunkStats::default();
        seed_hlls(&mut p_stats);
        let mut processor: RowProcessor<Seconds, NoSplitDecider> =
            RowProcessor::new(p_stats, NoSplitDecider);
        let mut local_target_counts: Option<LocalColumnCounts> = None;
        let mut local_entity_counts: Option<LocalColumnCounts> = None;
        processor.observe(&row, &mut local_target_counts, &mut local_entity_counts);

        // Compare against an independent ChunkStats path that uses the
        // same observation logic via the trait impl.
        let mut reference_stats: ChunkStats<Seconds> = ChunkStats::default();
        seed_hlls(&mut reference_stats);
        let mut local_target_counts2: Option<LocalColumnCounts> = None;
        let mut local_entity_counts2: Option<LocalColumnCounts> = None;
        <ChunkStats<Seconds> as RowObserver<Seconds>>::observe(
            &mut reference_stats,
            &row,
            &mut local_target_counts2,
            &mut local_entity_counts2,
        );

        // HLL state must match bit-for-bit between the two paths since
        // both observed the same row through the same observe() chain.
        assert_eq!(
            processor
                .stats
                .entity_hll
                .as_ref()
                .map(|h| h.estimate() as u64),
            reference_stats
                .entity_hll
                .as_ref()
                .map(|h| h.estimate() as u64),
        );
        assert_eq!(
            processor
                .stats
                .timestamp_hll
                .as_ref()
                .map(|h| h.estimate() as u64),
            reference_stats
                .timestamp_hll
                .as_ref()
                .map(|h| h.estimate() as u64),
        );
        assert_eq!(processor.stats.timestamp_min, reference_stats.timestamp_min);
        assert_eq!(processor.stats.timestamp_max, reference_stats.timestamp_max);
        assert_eq!(processor.stats.entity_min, reference_stats.entity_min);
        assert_eq!(processor.stats.entity_max, reference_stats.entity_max);
        assert_eq!(processor.stats.target_min, reference_stats.target_min);
        assert_eq!(processor.stats.target_max, reference_stats.target_max);

        // Sanity: HLL recorded one distinct entity / timestamp.
        // For default-precision HLL (M = 16384 registers), estimates are
        // exact for cardinalities this small.
        assert_eq!(
            processor.stats.entity_hll.as_ref().unwrap().estimate() as u64,
            1
        );
        assert_eq!(
            processor.stats.timestamp_hll.as_ref().unwrap().estimate() as u64,
            1
        );
    }

    #[test]
    fn test_row_processor_merge_combines_stats_and_decider() {
        // Two RowProcessors observe disjoint rows; merging should combine
        // their HLL state. NoSplitDecider's merge is no-op.
        let entity_a = Value::Int(1);
        let entity_b = Value::Int(2);
        let target_a = Value::Int(10);
        let target_b = Value::Int(20);

        let mut s_a = ChunkStats::default();
        seed_hlls(&mut s_a);
        let mut s_b = ChunkStats::default();
        seed_hlls(&mut s_b);
        let mut p_a: RowProcessor<Seconds, NoSplitDecider> = RowProcessor::new(s_a, NoSplitDecider);
        let mut p_b: RowProcessor<Seconds, NoSplitDecider> = RowProcessor::new(s_b, NoSplitDecider);

        let mut ltc_a: Option<LocalColumnCounts> = None;
        let mut lec_a: Option<LocalColumnCounts> = None;
        let mut ltc_b: Option<LocalColumnCounts> = None;
        let mut lec_b: Option<LocalColumnCounts> = None;

        p_a.observe(
            &make_row(&entity_a, Abs::from_raw(100), &target_a),
            &mut ltc_a,
            &mut lec_a,
        );
        p_b.observe(
            &make_row(&entity_b, Abs::from_raw(200), &target_b),
            &mut ltc_b,
            &mut lec_b,
        );

        // Cardinality is HLL-only. For tiny fixtures
        // (cardinality << 16384 registers) HLL is exact, so we can
        // assert the post-merge estimate equals the union cardinality.
        let n_entities_before_a = p_a.stats.entity_hll.as_ref().unwrap().estimate() as u64;
        let n_targets_before_a = p_a.stats.target_hll.as_ref().unwrap().estimate() as u64;
        p_a.merge(p_b);

        assert_eq!(
            p_a.stats.entity_hll.as_ref().unwrap().estimate() as u64,
            n_entities_before_a + 1
        );
        assert_eq!(
            p_a.stats.target_hll.as_ref().unwrap().estimate() as u64,
            n_targets_before_a + 1
        );
        assert_eq!(p_a.stats.timestamp_min, Some(Abs::from_raw(100)));
        assert_eq!(p_a.stats.timestamp_max, Some(Abs::from_raw(200)));

        // Decider check: still UNUSED.
        let row = make_row(&entity_a, Abs::from_raw(0), &target_a);
        assert_eq!(p_a.decider.decide(&row), UNUSED_STAGE);
    }

    #[test]
    fn test_hll_cardinality_estimate_on_small_fixture() {
        // Regression guard for a representative low-cardinality fixture.
        // HyperLogLog is probabilistic even below its 16384-register size;
        // these deterministic inputs happen to estimate to their exact counts.
        let mut chunk: ChunkStats<Seconds> = ChunkStats::default();
        seed_hlls(&mut chunk);

        // 5 distinct entities (with one duplicate), 3 distinct targets,
        // 4 distinct timestamps (with one duplicate).
        let entities = [1_i64, 2, 3, 4, 5, 1];
        let targets = [10_i64, 20, 30, 10, 20, 30];
        let tses = [100_i64, 200, 300, 400, 100, 200];
        for ((&e, &t), &ts) in entities.iter().zip(targets.iter()).zip(tses.iter()) {
            chunk.entity_hll.as_mut().unwrap().observe_i64(e);
            chunk.target_hll.as_mut().unwrap().observe_i64(t);
            chunk.timestamp_hll.as_mut().unwrap().observe_i64(ts);
        }

        assert_eq!(
            chunk.entity_hll.as_ref().unwrap().estimate() as u64,
            5,
            "unexpected entity estimate for fixture"
        );
        assert_eq!(
            chunk.target_hll.as_ref().unwrap().estimate() as u64,
            3,
            "unexpected target estimate for fixture"
        );
        assert_eq!(
            chunk.timestamp_hll.as_ref().unwrap().estimate() as u64,
            4,
            "unexpected timestamp estimate for fixture"
        );

        // Now drive the merger->finalize path and ensure num_entities /
        // num_targets / exact_stats.timestamp.num_unique reflect the
        // HLL-derived counts.
        chunk.num_train_rows = 6;
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        merger.merge(chunk);
        let out = merger.finalize(6, false, false, false, None, "numerical", None);
        assert_eq!(out.num_entities, 5);
        assert_eq!(out.num_targets, 3);
        assert_eq!(out.exact_stats.timestamp.num_unique, Some(4));
    }

    #[test]
    fn test_chunk_stats_impls_row_observer() {
        // Compile-time check: ChunkStats<U> implements RowObserver<U>.
        fn _assert_impls<U: UnitSystem, T: RowObserver<U>>() {}
        _assert_impls::<Seconds, ChunkStats<Seconds>>();
    }

    #[test]
    fn test_send_bounds() {
        // Compile-time check: required Send bounds for cross-thread use.
        fn _assert_send<T: Send>() {}
        _assert_send::<ChunkStats<Seconds>>();
        _assert_send::<NoSplitDecider>();
        _assert_send::<RowProcessor<Seconds, NoSplitDecider>>();
    }

    // ─── kumo-contract stats payload tests ──────────────────────────────────

    /// Helper: drive merger->finalize with the minimum knobs the kumo
    /// contract needs. All other args take their no-op defaults.
    #[allow(clippy::too_many_arguments)]
    fn finalize_minimal(
        chunk: ChunkStats<Seconds>,
        num_total_rows: u64,
        is_link_prediction: bool,
        target_category_counts: Option<HashMap<String, u64>>,
        target_stype: &str,
    ) -> TrainTableStatsPayload {
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        let train_rows = chunk.num_train_rows;
        merger.merge(chunk);
        merger.finalize(
            num_total_rows.max(train_rows),
            is_link_prediction,
            false,
            false,
            target_category_counts,
            target_stype,
            None,
        )
    }

    /// The entity column is a foreign key resolved to graph node ids; it is
    /// never encoded, so nothing reads its CATEGORY_COUNTS. Collecting them
    /// shared the target's string budget, and a high-cardinality entity could
    /// exhaust it and drop the target's counts, which ARE required.
    ///
    /// Drained-but-empty is what production produces once collection stops, so
    /// that is what this feeds in. The key must be absent rather than empty: a
    /// consumer added later should fail on the missing key, not read an empty
    /// vocabulary as "no categories".
    #[test]
    fn test_finalize_omits_entity_category_counts() {
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 5,
            ..Default::default()
        };
        chunk.timestamp_min = Some(Abs::from_raw(100));
        chunk.timestamp_max = Some(Abs::from_raw(100));

        let mut target_counts = HashMap::new();
        target_counts.insert("a".to_string(), 3u64);
        target_counts.insert("b".to_string(), 2u64);

        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        merger.merge(chunk);
        let payload = merger.finalize(
            5,
            false,
            false,
            false,
            Some(target_counts),
            "categorical",
            // What the executor hands over once entity counting is disabled.
            Some(HashMap::new()),
        );

        let stats = payload
            .parsed_train_table_col_stats
            .as_ref()
            .and_then(|v| v.get("stats"))
            .and_then(|v| v.as_object())
            .cloned()
            .expect("train-table col stats must be emitted");

        for key in stats.keys() {
            assert!(
                !(key.starts_with("ENTITY1") && key.contains("CATEGORY_COUNTS")),
                "entity CATEGORY_COUNTS must not be emitted, found {key:?}"
            );
        }
        // The target's counts must survive: they size the classification head.
        assert!(
            stats
                .keys()
                .any(|k| k.starts_with("TARGET1") && k.contains("CATEGORY_COUNTS")),
            "target CATEGORY_COUNTS must still be emitted"
        );
    }

    #[test]
    fn test_finalize_emits_kumo_contract_keys() {
        // The eleven keys kumo's seed_train_stats_cache_from_native_artifact
        // looks up directly.
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 5,
            timestamp_hll: Some(HyperLogLog::new()),
            ..Default::default()
        };
        chunk.timestamp_hll.as_mut().unwrap().observe_i64(100);
        chunk.timestamp_hll.as_mut().unwrap().observe_i64(200);
        chunk.timestamp_min = Some(Abs::from_raw(100));
        chunk.timestamp_max = Some(Abs::from_raw(200));

        let payload = finalize_minimal(chunk, 5, false, None, "numerical");
        let kc = &payload.kumo_contract;

        // Spot-check each of the required keys is populated and
        // serializes to the expected top-level field name. We check the
        // serialized JSON of StatsArtifact to ensure they end up at top.
        let artifact = crate::manifest::StatsArtifact::from_payload(&payload);
        let json = serde_json::to_value(&artifact).expect("StatsArtifact serializes");
        let obj = json.as_object().expect("StatsArtifact is a JSON object");
        for key in [
            "num_target_rows",
            "num_timeframes",
            "num_timeframes_post_filter",
            "avg_entities_per_timeframe",
            "min_date",
            "max_date",
            "anchor_date",
            "node_degree_stats",
            "scoring_table_column_stats",
            "target_distribution",
        ] {
            assert!(
                obj.contains_key(key),
                "kumo-contract key {key:?} must be present at top level of stats.json"
            );
        }

        // Sanity-check derived values
        assert_eq!(kc.num_target_rows, 5);
        assert_eq!(kc.num_timeframes, 2);
        assert_eq!(kc.num_timeframes_post_filter, 2);
        // 5 / 2 = 2.5 → rounds to 3
        assert_eq!(kc.avg_entities_per_timeframe, 3);
        assert!(kc.node_degree_stats.is_null());
        assert!(kc.scoring_table_column_stats.is_null());
        // Date fields populated from timestamp_min / timestamp_max.
        assert!(kc.min_date.is_string(), "min_date must be a string");
        assert!(kc.max_date.is_string(), "max_date must be a string");
        // anchor_date == max_date by kumo convention.
        assert_eq!(kc.anchor_date, kc.max_date);
        for removed in [
            "payload",
            "timestamp_counts",
            "stats_per_timeframe",
            "timestamp_stage_assignments",
            "kumo_contract",
        ] {
            assert!(
                !obj.contains_key(removed),
                "expanded field {removed:?} must not be in stats.json"
            );
        }
    }

    #[test]
    fn test_stats_artifact_size_is_bounded_by_timestamp_cardinality() {
        fn artifact_bytes(cardinality: i64) -> Vec<u8> {
            let mut timestamp_hll = HyperLogLog::new();
            for ts in 0..cardinality {
                timestamp_hll.observe_i64(ts);
            }
            let chunk: ChunkStats<Seconds> = ChunkStats {
                num_train_rows: cardinality as u64,
                timestamp_hll: Some(timestamp_hll),
                timestamp_min: Some(Abs::from_raw(0)),
                timestamp_max: Some(Abs::from_raw(cardinality.saturating_sub(1))),
                ..Default::default()
            };
            let payload = finalize_minimal(chunk, cardinality as u64, false, None, "numerical");
            serde_json::to_vec(&crate::manifest::StatsArtifact::from_payload(&payload)).unwrap()
        }

        let ten = artifact_bytes(10);
        let hundred_thousand = artifact_bytes(100_000);
        assert!(hundred_thousand.len() < 2_048);
        assert!(
            hundred_thousand.len() <= ten.len() + 32,
            "artifact grew with timestamp cardinality: {} -> {} bytes",
            ten.len(),
            hundred_thousand.len()
        );
    }

    #[test]
    fn test_finalize_target_distribution_categorical() {
        // Categorical target: target_distribution should be
        // {count: {key: int}, percentage: {key: float}} matching kumo's
        // CategoricalDictTargetDistribution.
        let chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 10,
            ..Default::default()
        };
        let mut counts = HashMap::new();
        counts.insert("A".to_string(), 6);
        counts.insert("B".to_string(), 3);
        counts.insert("C".to_string(), 1);

        let payload = finalize_minimal(chunk, 10, false, Some(counts), "categorical");
        let dist = &payload.kumo_contract.target_distribution;
        let count = dist.get("count").expect("count key").as_object().unwrap();
        let pct = dist
            .get("percentage")
            .expect("percentage key")
            .as_object()
            .unwrap();

        assert_eq!(count.get("A").unwrap().as_u64(), Some(6));
        assert_eq!(count.get("B").unwrap().as_u64(), Some(3));
        assert_eq!(count.get("C").unwrap().as_u64(), Some(1));

        let pa = pct.get("A").unwrap().as_f64().unwrap();
        let pb = pct.get("B").unwrap().as_f64().unwrap();
        let pc = pct.get("C").unwrap().as_f64().unwrap();
        assert!((pa - 0.6).abs() < 1e-9);
        assert!((pb - 0.3).abs() < 1e-9);
        assert!((pc - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_finalize_link_prediction_omits_exact_target_metadata() {
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 10,
            num_links: 12,
            ..Default::default()
        };
        let mut target_hll = HyperLogLog::new();
        target_hll.observe_i64(10);
        target_hll.observe_i64(20);
        chunk.target_hll = Some(target_hll);

        // Even if an older caller supplies exact counts, the LP artifact must
        // not distribute or duplicate graph node IDs.
        let mut counts = HashMap::new();
        counts.insert("10".to_string(), 6);
        counts.insert("20".to_string(), 4);

        let payload = finalize_minimal(chunk, 10, true, Some(counts), "ID");

        assert_eq!(payload.num_train_rows, 10);
        assert_eq!(payload.num_links, Some(12));
        assert_eq!(payload.num_targets, 2);
        assert_eq!(payload.kumo_contract.target_distribution, json!({}));
        assert!(payload.parsed_train_table_col_stats.is_none());
        assert_eq!(payload.exact_stats.target.num_unique, Some(2));
    }

    #[test]
    fn test_finalize_target_distribution_numerical() {
        // Numerical target: target_distribution should be
        // {target_mean, target_min, target_max, target_percentile_25,
        // target_median, target_percentile_75, target_mode,
        // target_mode_percentage, target_mode_count, histogram: {...}}
        // Drive Welford + t-digest with a known dataset: 1..=10.
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 10,
            ..Default::default()
        };
        chunk.target_welford = Some(WelfordAccumulator::new());
        chunk.target_tdigest = Some(TDigestEstimator::new(200));
        for v in 1..=10 {
            chunk.target_welford.as_mut().unwrap().observe(v as f64);
            chunk.target_tdigest.as_mut().unwrap().observe(v as f64);
        }

        let payload = finalize_minimal(chunk, 10, false, None, "numerical");
        let dist = &payload.kumo_contract.target_distribution;

        let mean = dist.get("target_mean").unwrap().as_f64().unwrap();
        let dmin = dist.get("target_min").unwrap().as_f64().unwrap();
        let dmax = dist.get("target_max").unwrap().as_f64().unwrap();
        // mean of 1..=10 is 5.5
        assert!((mean - 5.5).abs() < 1e-9);
        assert!((dmin - 1.0).abs() < 1e-9);
        assert!((dmax - 10.0).abs() < 1e-9);

        // p50 should be near 5.5; bound loosely (t-digest is approximate).
        let p50 = dist.get("target_median").unwrap().as_f64().unwrap();
        assert!((p50 - 5.5).abs() < 1.5, "p50 should be near 5.5, got {p50}");
        // Histogram is stubbed: empty bins/counts.
        let hist = dist.get("histogram").unwrap();
        assert_eq!(hist.get("num_bins").unwrap().as_i64(), Some(0));
        assert_eq!(hist.get("bin_width").unwrap().as_f64(), Some(0.0));
    }
    #[test]
    fn test_finalize_emits_top_level_dates() {
        let mut chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 2,
            ..Default::default()
        };
        chunk.timestamp_min = Some(Abs::from_raw(1_700_000_000));
        chunk.timestamp_max = Some(Abs::from_raw(1_800_000_000));

        let payload = finalize_minimal(chunk, 2, false, None, "numerical");
        let kc = &payload.kumo_contract;
        let mn = kc.min_date.as_str().expect("min_date must be a string");
        let mx = kc.max_date.as_str().expect("max_date must be a string");
        let an = kc
            .anchor_date
            .as_str()
            .expect("anchor_date must be a string");

        // RFC3339 strings; lexicographic order matches chronological.
        assert!(mn < mx, "min_date {mn:?} should be < max_date {mx:?}");
        assert_eq!(an, mx, "anchor_date must equal max_date");
        // Sanity: the strings parse as RFC3339 (start with year digits).
        assert!(
            mn.starts_with("20"),
            "expected RFC3339-shaped min_date, got {mn:?}"
        );
        assert!(
            mx.starts_with("20"),
            "expected RFC3339-shaped max_date, got {mx:?}"
        );
    }

    #[test]
    fn test_finalize_top_level_dates_null_when_no_data() {
        // No timestamps observed → date fields are null.
        let chunk: ChunkStats<Seconds> = ChunkStats {
            num_train_rows: 0,
            ..Default::default()
        };
        let payload = finalize_minimal(chunk, 0, false, None, "numerical");
        let kc = &payload.kumo_contract;
        assert!(kc.min_date.is_null());
        assert!(kc.max_date.is_null());
        assert!(kc.anchor_date.is_null());
    }
}

#[test]
fn test_sequence_feature_stats_observation_and_merge() {
    use arrow_array::builder::{Float64Builder, ListBuilder};
    use std::sync::Arc;

    let mut stats1 = SequenceFeatureStats::default();
    let mut stats2 = SequenceFeatureStats::default();

    // Build first array: one list of length 2, one null, one list of length 4, and one empty list.
    let array1 = {
        let mut b = ListBuilder::new(Float64Builder::new());
        b.values().append_value(1.0);
        b.values().append_value(2.0);
        b.append(true);
        b.append(false); // null
        b.values().append_value(3.0);
        b.values().append_value(4.0);
        b.values().append_value(5.0);
        b.values().append_value(6.0);
        b.append(true);
        b.append(true); // empty list
        Arc::new(b.finish()) as Arc<dyn Array>
    };

    stats1.observe(array1.as_ref(), 0);
    stats1.observe(array1.as_ref(), 1);
    stats1.observe(array1.as_ref(), 3);
    stats1.observe(array1.as_ref(), 2);

    assert_eq!(stats1.count, 2);
    assert_eq!(stats1.null_count, 2);
    assert_eq!(stats1.min_len, Some(2));
    assert_eq!(stats1.max_len, Some(4));

    // Build second array: one list of length 1, one list of length 5
    let array2 = {
        let mut b = ListBuilder::new(Float64Builder::new());
        b.values().append_value(7.0);
        b.append(true);
        b.values().append_value(8.0);
        b.values().append_value(9.0);
        b.values().append_value(10.0);
        b.values().append_value(11.0);
        b.values().append_value(12.0);
        b.append(true);
        Arc::new(b.finish()) as Arc<dyn Array>
    };

    stats2.observe(array2.as_ref(), 0);
    stats2.observe(array2.as_ref(), 1);

    assert_eq!(stats2.count, 2);
    assert_eq!(stats2.null_count, 0);
    assert_eq!(stats2.min_len, Some(1));
    assert_eq!(stats2.max_len, Some(5));

    // Merge stats2 into stats1
    stats1.merge(&stats2);

    assert_eq!(stats1.count, 4);
    assert_eq!(stats1.null_count, 2);
    assert_eq!(stats1.min_len, Some(1));
    assert_eq!(stats1.max_len, Some(5));
}

#[cfg(test)]
mod sequence_feature_emission_tests {
    use super::*;
    use arrow_array::builder::{Float64Builder, ListBuilder};
    use diskgraph_core::time::Seconds;
    use std::sync::Arc;

    fn observe(rows: &[Option<&[f64]>]) -> SequenceFeatureStats {
        let mut builder = ListBuilder::new(Float64Builder::new());
        for row in rows {
            match row {
                Some(values) => {
                    for value in *values {
                        builder.values().append_value(*value);
                    }
                    builder.append(true);
                }
                None => builder.append(false),
            }
        }
        let array = Arc::new(builder.finish()) as Arc<dyn Array>;
        let mut stats = SequenceFeatureStats::default();
        for row in 0..array.len() {
            stats.observe(array.as_ref(), row);
        }
        stats
    }

    fn emit(chunks: Vec<SequenceFeatureStats>) -> JsonValue {
        let num_rows = chunks
            .iter()
            .map(|stats| stats.count + stats.null_count)
            .sum();
        let mut merger: StatsMerger<Seconds> = StatsMerger::default();
        for stats in chunks {
            merger.merge(ChunkStats {
                sequence_feature_stats: HashMap::from([("FEATURE".to_string(), stats)]),
                ..Default::default()
            });
        }
        merger
            .finalize(num_rows, false, false, false, None, "numerical", None)
            .parsed_train_table_col_stats
            .expect("sequence statistics must emit the Kumo envelope")
    }

    fn value<'a>(emitted: &'a JsonValue, stat: &str) -> &'a JsonValue {
        let key = kumo_key("FEATURE", "kumo.sequence", stat, false);
        emitted
            .get("stats")
            .and_then(|stats| stats.get(&key))
            .and_then(|entry| entry.get("value"))
            .unwrap_or_else(|| panic!("missing emitted sequence statistic {key}"))
    }

    #[test]
    fn empty_and_null_rows_emit_as_missing_without_changing_fixed_width() {
        let stats = observe(&[
            Some(&[]),
            Some(&[1.0, 2.0, 3.0]),
            None,
            Some(&[4.0, 5.0, 6.0]),
        ]);
        let emitted = emit(vec![stats]);

        assert_eq!(value(&emitted, "COUNT").as_u64(), Some(4));
        assert_eq!(value(&emitted, "NUM_NA").as_u64(), Some(2));
        assert_eq!(value(&emitted, "NA_FRACTION").as_f64(), Some(0.5));
        assert_eq!(value(&emitted, "SEQUENCE_MIN_LENGTH").as_i64(), Some(3));
        assert_eq!(value(&emitted, "SEQUENCE_MAX_LENGTH").as_i64(), Some(3));
    }

    #[test]
    fn all_missing_rows_emit_zero_width_bounds() {
        let emitted = emit(vec![observe(&[Some(&[]), None, Some(&[])])]);

        assert_eq!(value(&emitted, "COUNT").as_u64(), Some(3));
        assert_eq!(value(&emitted, "NUM_NA").as_u64(), Some(3));
        assert_eq!(value(&emitted, "NA_FRACTION").as_f64(), Some(1.0));
        assert_eq!(value(&emitted, "SEQUENCE_MIN_LENGTH").as_i64(), Some(0));
        assert_eq!(value(&emitted, "SEQUENCE_MAX_LENGTH").as_i64(), Some(0));
    }

    #[test]
    fn mixed_nonempty_widths_emit_exact_bounds_and_missing_fraction() {
        let emitted = emit(vec![observe(&[
            Some(&[1.0, 2.0]),
            Some(&[]),
            Some(&[3.0, 4.0, 5.0, 6.0]),
        ])]);

        assert_eq!(value(&emitted, "COUNT").as_u64(), Some(3));
        assert_eq!(value(&emitted, "NUM_NA").as_u64(), Some(1));
        assert_eq!(value(&emitted, "NA_FRACTION").as_f64(), Some(1.0 / 3.0));
        assert_eq!(value(&emitted, "SEQUENCE_MIN_LENGTH").as_i64(), Some(2));
        assert_eq!(value(&emitted, "SEQUENCE_MAX_LENGTH").as_i64(), Some(4));
    }

    #[test]
    fn emitted_contract_is_invariant_to_partitioning_and_merge_order() {
        let first = observe(&[Some(&[]), Some(&[1.0, 2.0, 3.0])]);
        let second = observe(&[None, Some(&[4.0, 5.0, 6.0])]);
        let unpartitioned = observe(&[
            Some(&[]),
            Some(&[1.0, 2.0, 3.0]),
            None,
            Some(&[4.0, 5.0, 6.0]),
        ]);

        let expected = emit(vec![unpartitioned]);
        assert_eq!(emit(vec![first.clone(), second.clone()]), expected);
        assert_eq!(emit(vec![second, first]), expected);
    }
}
