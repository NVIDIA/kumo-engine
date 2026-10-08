// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Row-level validation policy for the custom-TT source reader.
//!
//! A customer-supplied training table is validated in the same single pass
//! that ingests it. Every condition below is one the reader would otherwise
//! handle silently (skip the row, substitute a label, route it to no split)
//! and then report success. [`IngestOnError`] decides what happens instead:
//!
//! * [`IngestOnError::Abort`] -- fail the run on the first
//!   offending row, naming the condition, column and source location.
//! * [`IngestOnError::SkipRowOnError`] (default) -- drop the row and count it in
//!   [`RowFailures`], reported as `row_failures` in the gen result and the
//!   bundle manifest.
//!
//! Each skipped row is counted once; every condition it fails is reported.
//! [`RowFailureKind::EmptyTargetList`] is the one condition that skips (and
//! warns) in both modes: an empty list is a legitimate value that trains on
//! nothing, so it is surfaced rather than refused unless it is every row.
//! A run that keeps no rows fails in both modes.

use std::collections::BTreeMap;

use diskgraph_core::time::TimeUnit;
use serde::{Deserialize, Serialize};

use crate::errors::{TTGenError, TTGenResult};

/// What to do with a row that fails validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IngestOnError {
    /// Fail the run on the first offending row.
    Abort,
    /// Drop the offending row and count it.
    #[default]
    SkipRowOnError,
}

impl IngestOnError {
    pub const ABORT: &'static str = "abort";
    pub const SKIP_ROW_ON_ERROR: &'static str = "skip_row_on_error";

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            Self::ABORT => Some(Self::Abort),
            Self::SKIP_ROW_ON_ERROR => Some(Self::SkipRowOnError),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Abort => Self::ABORT,
            Self::SkipRowOnError => Self::SKIP_ROW_ON_ERROR,
        }
    }
}

/// A row-level condition the custom-TT reader checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowFailureKind {
    NullEntity,
    NullTarget,
    NullTargetItem,
    EmptyTargetList,
    NullTimestamp,
    NullWeight,
    NonFiniteWeight,
    NegativeWeight,
    WeightLengthMismatch,
    NullSplit,
    InvalidSplit,
    UnknownEntity,
    NonCanonicalEntity,
    UnavailableTarget,
}

impl RowFailureKind {
    pub fn describe(self) -> &'static str {
        match self {
            Self::NullEntity => "entity key is null",
            Self::NullTarget => "target is null",
            Self::NullTargetItem => "target list contains a null element",
            Self::EmptyTargetList => "target list is empty",
            Self::NullTimestamp => "anchor timestamp is null",
            Self::NullWeight => "weight is null",
            Self::NonFiniteWeight => "weight is not finite",
            Self::NegativeWeight => "weight is negative",
            Self::WeightLengthMismatch => "weight list length differs from the target list",
            Self::NullSplit => "split value is null",
            Self::InvalidSplit => "split value is not one of the configured split values",
            Self::UnknownEntity => "entity key is absent from the graph",
            Self::NonCanonicalEntity => "entity node is not canonical",
            Self::UnavailableTarget => "target is absent or temporally invalid in the graph",
        }
    }

    /// Counted as a skip in both modes rather than aborting the run.
    pub fn always_skips(self) -> bool {
        matches!(
            self,
            Self::EmptyTargetList
                | Self::UnknownEntity
                | Self::NonCanonicalEntity
                | Self::UnavailableTarget
        )
    }
}

/// Where a row sits in the source: the parquet part, its row group, and the
/// row's index within that row group.
#[derive(Debug, Clone, Copy)]
pub struct RowLocation<'a> {
    pub part: &'a str,
    pub row_group: usize,
    pub row: u64,
}

impl std::fmt::Display for RowLocation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} row group {} row {}",
            self.part, self.row_group, self.row
        )
    }
}

/// Samples kept per `(condition, column)`.
const SAMPLES_PER_ENTRY: usize = 5;

#[derive(Debug, Clone, Default, PartialEq)]
struct Entry {
    count: u64,
    samples: Vec<String>,
}

/// Per-chunk accumulator, merged across row-group workers.
#[derive(Debug, Clone, Default)]
pub struct RowFailures {
    rows_read: u64,
    skipped_rows: u64,
    entries: BTreeMap<(RowFailureKind, String), Entry>,
}

impl RowFailures {
    pub fn add_rows_read(&mut self, rows: u64) {
        self.rows_read += rows;
    }

    /// Record all conditions before applying the policy once for this row.
    /// `Ok(())` means skip the row; `Err` aborts the run.
    pub fn reject_row(
        &mut self,
        on_error: IngestOnError,
        conditions: &[(RowFailureKind, String)],
        at: RowLocation<'_>,
    ) -> TTGenResult<()> {
        debug_assert!(!conditions.is_empty());
        let unique: std::collections::BTreeSet<_> = conditions.iter().collect();
        for (kind, column) in &unique {
            let entry = self.entries.entry((*kind, column.clone())).or_default();
            entry.count += 1;
            if entry.samples.len() < SAMPLES_PER_ENTRY {
                entry.samples.push(at.to_string());
            }
        }
        if on_error == IngestOnError::Abort
            && conditions.iter().any(|(kind, _)| !kind.always_skips())
        {
            let details = unique
                .iter()
                .map(|(kind, column)| format!("{} in column {column:?}", kind.describe()))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(TTGenError::SchemaError(format!(
                "custom-TT source: {details} at {at}. Fix the source, or set \
                 ingest_on_error=skip_row_on_error to drop such rows."
            )));
        }
        self.skipped_rows += 1;
        Ok(())
    }

    #[cfg(test)]
    fn reject(
        &mut self,
        on_error: IngestOnError,
        kind: RowFailureKind,
        column: &str,
        at: RowLocation<'_>,
    ) -> TTGenResult<()> {
        self.reject_row(on_error, &[(kind, column.to_string())], at)
    }

    pub fn merge(&mut self, other: RowFailures) {
        self.rows_read += other.rows_read;
        self.skipped_rows += other.skipped_rows;
        for (key, theirs) in other.entries {
            let entry = self.entries.entry(key).or_default();
            entry.count += theirs.count;
            for sample in theirs.samples {
                if entry.samples.len() < SAMPLES_PER_ENTRY {
                    entry.samples.push(sample);
                }
            }
        }
    }

    pub fn rows_read(&self) -> u64 {
        self.rows_read
    }

    pub fn skipped_rows(&self) -> u64 {
        self.skipped_rows
    }

    pub fn report(&self, on_error: IngestOnError) -> RowFailuresReport {
        RowFailuresReport {
            on_error: on_error.name().to_string(),
            rows_read: self.rows_read,
            skipped_rows: self.skipped_rows(),
            by_condition: self
                .entries
                .iter()
                .map(|((kind, column), entry)| ConditionCount {
                    condition: *kind,
                    column: column.clone(),
                    count: entry.count,
                    samples: entry.samples.clone(),
                })
                .collect(),
        }
    }
}

/// What the custom-TT reader tallies per chunk beyond the row stats: the
/// failures, and the weights that survived (so that a table with no positive
/// weight at all can be refused once every chunk has been merged).
#[derive(Debug, Clone)]
pub struct SourceRowStats {
    pub failures: RowFailures,
    pub weight_values: u64,
    pub weight_max: f64,
    pub integer_timestamp_max: Option<i64>,
}

impl Default for SourceRowStats {
    fn default() -> Self {
        Self {
            failures: RowFailures::default(),
            weight_values: 0,
            weight_max: f64::NEG_INFINITY,
            integer_timestamp_max: None,
        }
    }
}

impl SourceRowStats {
    pub fn observe_integer_timestamp(&mut self, value: i64) {
        self.integer_timestamp_max = Some(
            self.integer_timestamp_max
                .map_or(value, |old| old.max(value)),
        );
    }

    pub fn observe_weight(&mut self, weight: f64) {
        self.weight_values += 1;
        self.weight_max = self.weight_max.max(weight);
    }

    pub fn merge(&mut self, other: SourceRowStats) {
        self.failures.merge(other.failures);
        self.weight_values += other.weight_values;
        self.weight_max = self.weight_max.max(other.weight_max);
        if let Some(max) = other.integer_timestamp_max {
            self.observe_integer_timestamp(max);
        }
    }
}

/// Trusted generated-table footer metadata. Only equal-sized sources are compared.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimestampReference {
    pub rows: u64,
    pub max_ns: i64,
}

impl TimestampReference {
    /// A whole-column unit mismatch cannot be repaired by dropping one row.
    /// Refuse it in both ingestion modes, without guessing from epoch magnitude.
    pub fn validate(&self, source: &SourceRowStats, declared: TimeUnit) -> TTGenResult<()> {
        let Some(max) = source.integer_timestamp_max else {
            return Ok(());
        };
        if self.rows != source.failures.rows_read() || max == 0 || self.max_ns == 0 {
            return Ok(());
        }
        let ratio = self.max_ns as f64 / max as f64;
        let declared_name = match declared {
            TimeUnit::Seconds => "seconds",
            TimeUnit::Milliseconds => "milliseconds",
            TimeUnit::Microseconds => "microseconds",
            TimeUnit::Nanoseconds => "nanoseconds",
            _ => return Ok(()),
        };
        for (unit, factor) in [
            ("seconds", 1_000_000_000.0),
            ("milliseconds", 1_000_000.0),
            ("microseconds", 1_000.0),
            ("nanoseconds", 1.0),
        ] {
            if unit != declared_name && (ratio - factor).abs() <= factor * 0.01 {
                return Err(TTGenError::SchemaError(format!(
                    "custom-TT source: TIMESTAMP looks like {}, not {}: maximum integer \
                     timestamp {max} against generated-table maximum {} nanoseconds. \
                     Fix the source timestamp unit or input_timestamp_unit; rows cannot \
                     be skipped to repair a whole-column unit mismatch.",
                    unit, declared_name, self.max_ns
                )));
            }
        }
        Ok(())
    }
}

/// One `(condition, column)` bucket of skipped rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionCount {
    pub condition: RowFailureKind,
    pub column: String,
    pub count: u64,
    /// Up to five source locations, as `"<part> row group <n> row <m>"`.
    pub samples: Vec<String>,
}

/// The `row_failures` payload of the gen result and the bundle manifest.
///
/// `skipped_rows` counts distinct rows; condition counts can overlap. Valid
/// rows removed by graph filters are also included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowFailuresReport {
    pub on_error: String,
    pub rows_read: u64,
    pub skipped_rows: u64,
    pub by_condition: Vec<ConditionCount>,
}

impl RowFailuresReport {
    fn warning_message(&self, kept_rows: u64) -> Option<String> {
        if self.skipped_rows == 0 {
            return None;
        }
        let breakdown = self
            .by_condition
            .iter()
            .map(|c| format!("{}[{}]={}", c.condition.describe(), c.column, c.count))
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "custom-TT source: {} of {} rows were skipped; training on {} rows. Reasons: {}",
            self.skipped_rows, self.rows_read, kept_rows, breakdown
        ))
    }

    pub fn warn_if_any(&self, kept_rows: u64) {
        if let Some(message) = self.warning_message(kept_rows) {
            tracing::warn!(on_error = %self.on_error, "{message}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(row: u64) -> RowLocation<'static> {
        RowLocation {
            part: "part-0.parquet",
            row_group: 1,
            row,
        }
    }

    #[test]
    fn names_round_trip_and_unknown_is_rejected() {
        for mode in [IngestOnError::Abort, IngestOnError::SkipRowOnError] {
            assert_eq!(IngestOnError::from_name(mode.name()), Some(mode));
        }
        assert_eq!(IngestOnError::from_name("skip"), None);
        assert_eq!(IngestOnError::default(), IngestOnError::SkipRowOnError);
    }

    #[test]
    fn abort_names_condition_column_and_location() {
        let mut failures = RowFailures::default();
        let err = failures
            .reject(
                IngestOnError::Abort,
                RowFailureKind::NullEntity,
                "ENTITY1",
                at(7),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("entity key is null"), "{err}");
        assert!(err.contains("ENTITY1"), "{err}");
        assert!(err.contains("part-0.parquet row group 1 row 7"), "{err}");
        assert_eq!(failures.skipped_rows(), 0);
    }

    #[test]
    fn skip_counts_and_caps_samples() {
        let mut failures = RowFailures::default();
        for row in 0..9 {
            failures
                .reject(
                    IngestOnError::SkipRowOnError,
                    RowFailureKind::NullTarget,
                    "TARGET1",
                    at(row),
                )
                .unwrap();
        }
        let report = failures.report(IngestOnError::SkipRowOnError);
        assert_eq!(report.skipped_rows, 9);
        assert_eq!(report.by_condition.len(), 1);
        assert_eq!(report.by_condition[0].samples.len(), SAMPLES_PER_ENTRY);
    }

    #[test]
    fn empty_target_list_skips_even_when_aborting() {
        let mut failures = RowFailures::default();
        failures
            .reject(
                IngestOnError::Abort,
                RowFailureKind::EmptyTargetList,
                "TARGET1",
                at(0),
            )
            .unwrap();
        assert_eq!(failures.skipped_rows(), 1);
    }

    #[test]
    fn warning_counts_rows_once_and_lists_all_reasons() {
        let mut failures = RowFailures::default();
        failures.add_rows_read(10);
        failures
            .reject_row(
                IngestOnError::default(),
                &[
                    (RowFailureKind::NullTarget, "TARGET1".to_string()),
                    (RowFailureKind::NullWeight, "WEIGHT".to_string()),
                ],
                at(0),
            )
            .unwrap();
        let message = failures
            .report(IngestOnError::default())
            .warning_message(9)
            .unwrap();
        assert!(
            message.contains("1 of 10 rows were skipped; training on 9 rows"),
            "{message}"
        );
        assert!(message.contains("target is null[TARGET1]=1"), "{message}");
        assert!(message.contains("weight is null[WEIGHT]=1"), "{message}");
        assert!(RowFailures::default()
            .report(IngestOnError::default())
            .warning_message(10)
            .is_none());
    }

    #[test]
    fn merge_adds_counts_per_bucket() {
        let mut a = RowFailures::default();
        let mut b = RowFailures::default();
        a.add_rows_read(10);
        b.add_rows_read(5);
        for failures in [&mut a, &mut b] {
            failures
                .reject(
                    IngestOnError::SkipRowOnError,
                    RowFailureKind::NullTimestamp,
                    "TIMESTAMP",
                    at(1),
                )
                .unwrap();
        }
        a.merge(b);
        let report = a.report(IngestOnError::SkipRowOnError);
        assert_eq!((report.rows_read, report.skipped_rows), (15, 2));
    }
}
