// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::graph_config::GraphConfig;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const INGEST_REPORT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestReport {
    #[serde(default = "legacy_ingest_report_version")]
    pub version: u32,
    pub stats: IngestStats,
    /// Reference to the separately published resolved graph config. Present
    /// for version 2 reports and bounded independently of source file count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_config: Option<IngestSourceConfig>,
    /// Version 1 reports embedded the complete resolved config here. New
    /// reports omit it, but readers retain support for historical snapshots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<GraphConfig>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestSourceConfig {
    pub path: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representative_data_path: Option<String>,
}

const fn legacy_ingest_report_version() -> u32 {
    1
}

impl IngestReport {
    pub fn compact(stats: IngestStats, source_config: IngestSourceConfig) -> Self {
        Self {
            version: INGEST_REPORT_VERSION,
            stats,
            source_config: Some(source_config),
            source: None,
        }
    }

    /// Validate the versioned report envelope. Config content validation is a
    /// reader responsibility because only a reader has access to the index
    /// root containing the referenced file.
    pub fn validate_envelope(&self) -> Result<()> {
        match self.version {
            1 => {
                ensure!(
                    self.source.is_some(),
                    "version 1 ingest report is missing embedded source config"
                );
                ensure!(
                    self.source_config.is_none(),
                    "version 1 ingest report must not contain a source config reference"
                );
            }
            INGEST_REPORT_VERSION => {
                let source = self.source_config.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("version 2 ingest report is missing source config reference")
                })?;
                ensure!(
                    source.path == crate::graph_config::DISKGRAPH_CONFIG_FILENAME,
                    "ingest report source config path must be {}",
                    crate::graph_config::DISKGRAPH_CONFIG_FILENAME
                );
                ensure!(
                    is_sha256_hex(&source.sha256),
                    "ingest report source config checksum is not lowercase SHA-256"
                );
                ensure!(
                    self.source.is_none(),
                    "version 2 ingest report must not embed source config"
                );
            }
            version => anyhow::bail!("unsupported ingest report version {version}"),
        }
        Ok(())
    }

    pub fn representative_data_path(&self) -> Option<String> {
        if self.validate_envelope().is_err() {
            return None;
        }
        if let Some(source) = &self.source_config {
            return source.representative_data_path.clone();
        }
        self.source.as_ref().and_then(representative_data_path)
    }
}

pub fn representative_data_path(source: &GraphConfig) -> Option<String> {
    let mut first_local = None;
    for table in &source.tables {
        for path in table.source_paths() {
            if is_cloud_path(path) {
                return Some(path.to_string());
            }
            if first_local.is_none() {
                first_local = Some(path.to_string());
            }
        }
    }
    first_local
}

fn is_cloud_path(path: &str) -> bool {
    path.starts_with("s3://") || path.starts_with("gs://") || path.starts_with("az://")
}

pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(Default))]
pub struct IngestStats {
    pub output: String,
    pub node_tables: usize,
    pub edge_relations: usize,
    pub total_nodes: u64,
    pub total_edges: u64,
    pub node_counts: HashMap<String, u64>,
    pub edge_counts: HashMap<String, u64>,
    pub key_parse_failures: u64,
    pub key_lookup_failures: u64,
    /// FK reference values that DID resolve through the minimal-perfect-hash
    /// pkmap but whose stored parent PK does not match the reference value —
    /// i.e. MPH false positives (orphan keys colliding onto an in-range
    /// parent slot). Only counted where membership verification runs
    /// (external-parent FK edges and resolution-mode columns).
    #[serde(default)]
    pub key_membership_failures: u64,
    pub key_failure_samples: Vec<String>,
    pub key_failures_by_context: HashMap<String, KeyFailureCounts>,
    pub pk_duplicate_counts: HashMap<String, u64>,
    pub pk_duplicate_drops: HashMap<String, u64>,
    pub pk_null_timestamp_drops: HashMap<String, u64>,
    pub edge_null_timestamp_drops: HashMap<String, u64>,
    pub pk_duplicate_total: u64,
    pub pk_duplicate_samples: HashMap<String, Vec<String>>,
    pub dropped_rows_counts: HashMap<String, u64>,
    /// Logical Arrow rows consumed from each input table.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub source_rows_by_table: HashMap<String, u64>,
    /// Logical Arrow bytes consumed from each input table.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub source_bytes_by_table: HashMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trimmed_strings: Option<u64>,
    pub perf: PerfStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_shards: Option<usize>,
    /// TT-NG temporal-LP timestamp grouping: per-stage-table `bucket_index.bin`
    /// plus a bucket list, present only for tables with `group_by_timestamp =
    /// true`. Empty (and omitted from JSON) otherwise.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub ts_groupings: HashMap<String, crate::ts_grouping::StageGrouping>,
    /// TT-NG binary-class seed sidecars, present only for requested tables whose
    /// label column contains valid binary values.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub binary_label_indexes: HashMap<String, crate::binary_label::BinaryLabelIndex>,
    /// TT-NG row-weight seed sidecars, present only for requested tables whose
    /// weight column contains valid non-negative numeric values.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sample_weight_indexes: HashMap<String, crate::sample_weight::SampleWeightIndex>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KeyFailureCounts {
    pub parse_failures: u64,
    pub lookup_failures: u64,
    /// MPH false positives dropped by membership verification (see
    /// [`IngestStats::key_membership_failures`]).
    #[serde(default)]
    pub membership_failures: u64,
    /// Non-null FK values that were not valid keys (wrong type or too long).
    #[serde(default)]
    pub non_null_parse_failures: u64,
}

impl KeyFailureCounts {
    pub fn merge(&mut self, other: &KeyFailureCounts) {
        self.parse_failures = self.parse_failures.saturating_add(other.parse_failures);
        self.lookup_failures = self.lookup_failures.saturating_add(other.lookup_failures);
        self.membership_failures = self
            .membership_failures
            .saturating_add(other.membership_failures);
        self.non_null_parse_failures = self
            .non_null_parse_failures
            .saturating_add(other.non_null_parse_failures);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KeyFailureState {
    pub parse_failures: u64,
    pub lookup_failures: u64,
    #[serde(default)]
    pub membership_failures: u64,
    pub samples: Vec<String>,
    pub by_context: HashMap<String, KeyFailureCounts>,
}

impl KeyFailureState {
    pub fn merge(&mut self, other: KeyFailureState) {
        self.parse_failures = self.parse_failures.saturating_add(other.parse_failures);
        self.lookup_failures = self.lookup_failures.saturating_add(other.lookup_failures);
        self.membership_failures = self
            .membership_failures
            .saturating_add(other.membership_failures);
        for (context, counts) in other.by_context {
            self.by_context.entry(context).or_default().merge(&counts);
        }
        for sample in other.samples {
            if self.samples.len() >= 10 {
                break;
            }
            self.samples.push(sample);
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PerfStats {
    pub wall_time_ms: u64,
    #[serde(default)]
    pub encode_wall_nanos: u64,
    /// Time the Encode consumer spent waiting for its source transport to
    /// produce the next Arrow batch. Subtracting this from Encode wall time is
    /// a bounded encode-ceiling measurement; it does not require retaining or
    /// rewriting the input dataset.
    #[serde(default)]
    pub source_input_wait_nanos: u64,
    pub context_wall_ms: u64,
    pub encode_wall_ms: u64,
    pub pk_map_wall_ms: u64,
    pub edges_map_wall_ms: u64,
    pub index_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_total_bytes: Option<u64>,
    #[serde(default)]
    pub encode_memory_high_water_bytes: u64,
    #[serde(default)]
    pub source_rows: u64,
    pub temp_edges_src_bytes: u64,
    pub temp_fkeys_bytes: u64,
    pub temp_edges_map_bytes: u64,
    pub temp_node_data_bytes: u64,
    pub temp_bytes_total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_context_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_encode_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_pk_map_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_edges_map_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_index_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_finalize_rss_anon_bytes: Option<u64>,
}

#[cfg(test)]
mod ingest_report_tests {
    use super::*;
    use serde_json::json;

    fn graph_config(paths: Vec<String>) -> GraphConfig {
        serde_json::from_value(json!({
            "dataset_name": "bounded-report-test",
            "tables": [{
                "name": "events",
                "columns": [],
                "source": {"type": "parquet", "paths": paths}
            }]
        }))
        .unwrap()
    }

    #[test]
    fn historical_unversioned_report_reads_embedded_source() {
        let report: IngestReport = serde_json::from_value(json!({
            "stats": IngestStats::default(),
            "source": graph_config(vec![
                "/local/first.parquet".to_string(),
                "s3://bucket/cloud.parquet".to_string(),
            ])
        }))
        .unwrap();

        assert_eq!(report.version, 1);
        report.validate_envelope().unwrap();
        assert_eq!(
            report.representative_data_path().as_deref(),
            Some("s3://bucket/cloud.parquet")
        );
    }

    #[test]
    fn compact_report_is_bounded_independently_of_source_path_count() {
        let paths: Vec<_> = (0..10_000)
            .map(|index| format!("s3://bucket/path/{index:05}.parquet"))
            .collect();
        let source = graph_config(paths);
        let historical = serde_json::to_vec(&json!({
            "stats": IngestStats::default(),
            "source": source,
        }))
        .unwrap();
        let report = IngestReport::compact(
            IngestStats::default(),
            IngestSourceConfig {
                path: crate::graph_config::DISKGRAPH_CONFIG_FILENAME.to_string(),
                sha256: "a".repeat(64),
                representative_data_path: representative_data_path(&source),
            },
        );
        report.validate_envelope().unwrap();
        let compact = serde_json::to_vec(&report).unwrap();

        assert!(historical.len() > 300_000);
        assert!(
            compact.len() < 2_000,
            "compact report was {} bytes",
            compact.len()
        );
        let value: serde_json::Value = serde_json::from_slice(&compact).unwrap();
        assert!(value.get("source").is_none());
        assert_eq!(value["version"], INGEST_REPORT_VERSION);
    }

    #[test]
    fn compact_report_rejects_malformed_or_ambiguous_source_reference() {
        for checksum in ["a".repeat(63), "A".repeat(64), "g".repeat(64)] {
            let report = IngestReport::compact(
                IngestStats::default(),
                IngestSourceConfig {
                    path: crate::graph_config::DISKGRAPH_CONFIG_FILENAME.to_string(),
                    sha256: checksum,
                    representative_data_path: None,
                },
            );
            assert!(report.validate_envelope().is_err());
        }

        let mut report = IngestReport::compact(
            IngestStats::default(),
            IngestSourceConfig {
                path: "../diskgraph_config.json".to_string(),
                sha256: "a".repeat(64),
                representative_data_path: None,
            },
        );
        assert!(report.validate_envelope().is_err());
        report.source = Some(graph_config(Vec::new()));
        assert!(report.validate_envelope().is_err());
        report.version = 1;
        assert!(report.validate_envelope().is_err());
    }

}
