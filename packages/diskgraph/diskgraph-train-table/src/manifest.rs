// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TT-NG manifest (`manifest.json`) — schema, serialization, and writer.
//!
//! Written once at the very end of
//! `execute_tt_pass_with_source`, after pass A, the optional regroup,
//! pass B, and the bridge pass have all completed.
//!
//! ## Schema
//!
//! Reshaped to match kumo's reader (`kumo/engine/diskgraph_native_tt_artifacts.py`).
//! The on-disk JSON uses nested objects with `path` + `format` strings:
//!
//! ```json
//! {
//!   "version": 4,
//!   "row_count": 1234,
//!   "source_mode": "pquery",
//!   "split_mode": "compute",
//!   "train_table":           {"path": "tt.parquet", "format": "tt_parquet_dataset_v1", "row_count": 1234, "size_bytes": 4567, "parts": [...]},
//!   "split":                 {"path": "...", "format": "split_assignment_row_stage_bin"},
//!   "stats":                 {"path": "...", "format": "tt_stats_json"},
//!   "split_stats":           {"path": "...", "format": "tt_split_stats_json"} | null,
//!   "train_table_col_stats": {"path": "...", "format": "tt_train_table_col_stats_json"} | null,
//!   "train_table_edges":     {...edge bundle JSON...} | null,
//!   "arange_sidecar":        {"path": "..."} | null,
//!   "fact_pseudo_time":      {"path": "...", "format": "tt_fact_pseudo_time_v1", "node_type": "..."} | null,
//!   "checksums": {"stats_sha256": "...", ...}
//! }
//! ```
//!
//! Optional sections (`split`, `split_stats`, `train_table_col_stats`,
//! `train_table_edges`, `arange_sidecar`, `fact_pseudo_time`) elide via
//! `skip_serializing_if = "Option::is_none"`.
//!
//! ## Concurrency
//!
//! The manifest writer is single-threaded; nothing else is touching the
//! output directory at the call site. No sync primitives.
//!
//! ## Schema stability
//!
//! Version 3 has no fact visibility requirement; version 4 requires it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use crate::errors::{TTGenError, TTGenResult};
use crate::ttgen_stats::TrainTableStatsPayload;

pub const TT_MANIFEST_VERSION: u32 = 3;
pub const FACT_PSEUDO_TIME_MANIFEST_VERSION: u32 = 4;

// ─── Format identifiers (kumo's reader checks these strings exactly) ────────

/// `train_table.format` value: produced parquet file.
pub const TT_PARQUET_FORMAT: &str = "tt_parquet_dataset_v1";
/// `split.format` value: row-stage assignment binary blob (DGSPLT).
pub const SPLIT_FORMAT: &str = "split_assignment_row_stage_bin";
/// `stats.format` value: top-level training-table stats JSON.
pub const STATS_FORMAT: &str = "tt_stats_json";
/// `split_stats.format` value (optional in manifest).
pub const SPLIT_STATS_FORMAT: &str = "tt_split_stats_json";
/// `train_table_col_stats.format` value (optional in manifest).
pub const TRAIN_TABLE_COL_STATS_FORMAT: &str = "tt_train_table_col_stats_json";
/// `weight_sidecar.format` value (optional in manifest, custom-TT only).
/// File on disk has DGWGHT magic + version + dtype + row_count + payload.
pub const WEIGHT_SIDECAR_FORMAT: &str = "tt_weight_sidecar_v1";
/// `fact_pseudo_time.format` value (static LP only). DGPTIME header + u8/row.
pub const FACT_PSEUDO_TIME_FORMAT: &str = "tt_fact_pseudo_time_v1";

/// Reference to a single artifact: `path` + `format`. The format string
/// is checked verbatim by kumo's reader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactRef {
    pub path: String,
    pub format: String,
}

/// Reference to the large train-table parquet. Unlike bounded sidecars, this
/// artifact is validated from file metadata instead of by re-reading all
/// bytes after the writer has closed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainTableArtifactRef {
    pub path: String,
    pub format: String,
    pub row_count: u64,
    pub size_bytes: u64,
    pub parts: Vec<TrainTablePartRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainTablePartRef {
    pub path: String,
    pub row_count: u64,
    pub size_bytes: u64,
}

impl TrainTableArtifactRef {
    pub fn new(
        path: impl Into<String>,
        format: impl Into<String>,
        row_count: u64,
        size_bytes: u64,
        parts: Vec<TrainTablePartRef>,
    ) -> Self {
        Self {
            path: path.into(),
            format: format.into(),
            row_count,
            size_bytes,
            parts,
        }
    }
}

impl ArtifactRef {
    pub fn new(path: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            format: format.into(),
        }
    }
}

/// Per-row pseudo times of the static LP fact `node_type`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FactPseudoTimeRef {
    pub path: String,
    pub format: String,
    pub node_type: String,
}

/// Sidecar reference (no format check on the kumo side; just a path).
/// Used for the transitional `arange.bin` artifact (no longer written on
/// the TT-NG path; the field stays for the kumo JSON contract).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SidecarRef {
    pub path: String,
}

/// Checksums for bounded sidecars. The potentially very large train-table
/// parquet is intentionally absent: manifest v3 validates closed-part sizes
/// and parquet metadata without a second full-content read.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ManifestChecksums {
    pub split_sha256: String,
    pub stats_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub split_stats_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub train_table_col_stats_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub train_table_edges_sha256: Option<String>,
    /// SHA-256 of `weight.bin` (custom-TT path only). `None` when no
    /// weight sidecar was emitted (PQ path, or custom-TT plan with
    /// `weight_col == None`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub weight_sidecar_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub fact_pseudo_time_sha256: Option<String>,
}

/// Manifest written to `output_dir/manifest.json` at the end of every
/// TT-NG run. Schema matches kumo's reader expectations (nested
/// `path` + `format` objects).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TTManifest {
    /// Version 4 requires fact visibility; ordinary artifacts remain version 3.
    pub version: u32,
    /// Total rows in the `tt.parquet` dataset (after regroup, if applicable).
    pub row_count: u64,
    /// `"pquery"` for the standard PQ entity-table walk; `"custom_tt"`
    /// when the executor was driven by a customer-supplied parquet.
    pub source_mode: String,
    /// `"compute"` when the fused pass-B ran, `"use_bin"` when the
    /// customer supplied a split assignment, `"none"` on the no-split path.
    pub split_mode: String,
    /// `"TimeRangeSplit"`, `"BacktestingSplit"`, etc. — the value of
    /// `plan.split.kind`. `None` when the plan has no `split`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub split_kind: Option<String>,
    /// Per-stage row counts from the fused pass-B output.
    /// `None` on the no-split (prediction) path.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stage_counts: Option<HashMap<String, u64>>,

    /// Reference to the `tt.parquet` directory dataset (always present).
    pub train_table: TrainTableArtifactRef,
    /// Reference to `split.bin` (header-prefixed). `None` when no split
    /// was emitted (e.g. `SplitMode::None`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub split: Option<ArtifactRef>,
    /// Reference to `stats.json` (top-level TT stats, kind=`tt_stats`).
    pub stats: ArtifactRef,
    /// Per-stage stats (optional; currently emits `None`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub split_stats: Option<ArtifactRef>,
    /// Per-stage seed-time `(min, max)` of the train-table TIMESTAMP
    /// column, keyed by stage name (`"train"`/`"val"`/`"test"`). Values
    /// are `[min, max]` int64 nanoseconds. A stage appears only when it
    /// has rows with a non-null finite timestamp; `None` when no split
    /// was computed (or no stage carried a timestamp).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stage_seed_times: Option<HashMap<String, [i64; 2]>>,
    /// Encoder-inference column stats. Optional; emitted when
    /// `parsed_train_table_col_stats` is present in the stats payload.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub train_table_col_stats: Option<ArtifactRef>,
    /// Edge artifact bundle. Always emitted by a separate
    /// `export_tt_artifacts` call, NOT by `execute_tt_pass`. `None`
    /// here; consumers fall back to a default disabled bundle when
    /// absent.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub train_table_edges: Option<JsonValue>,
    /// Transitional sidecar for the legacy ML consumer
    /// (`arange.bin`). No longer written on the TT-NG path (set to
    /// `None`); the field stays for the kumo JSON contract.
    /// **Will be removed when ML migrates.**
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub arange_sidecar: Option<SidecarRef>,
    /// Custom-TT weight sidecar (`weight.bin`). `None` when the plan
    /// did not declare `weight_col`. When present, format is
    /// `tt_weight_sidecar_v1` (DGWGHT magic + version + dtype +
    /// row_count + payload). Kumo's bridge reads this and appends a
    /// `weight` column to `tt.parquet` post-bridge.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub weight_sidecar: Option<ArtifactRef>,
    /// Static LP fact-row pseudo times (`fact_pseudo_time.bin`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub fact_pseudo_time: Option<FactPseudoTimeRef>,
    /// Checksums for every referenced artifact.
    pub checksums: ManifestChecksums,
}

fn validate_manifest_version(manifest: &TTManifest) -> TTGenResult<()> {
    let expected = if manifest.fact_pseudo_time.is_some() {
        FACT_PSEUDO_TIME_MANIFEST_VERSION
    } else {
        TT_MANIFEST_VERSION
    };
    if manifest.version != expected {
        return Err(TTGenError::SystemError(format!(
            "manifest version mismatch: expected={expected} got={}; version 4 requires fact visibility support",
            manifest.version
        )));
    }
    Ok(())
}

/// Write `manifest` to `output_dir/manifest.json` (pretty-printed).
/// Returns the absolute path on success. Single-threaded; no sync
/// primitives.
pub fn write_manifest(output_dir: &Path, manifest: &TTManifest) -> TTGenResult<PathBuf> {
    validate_manifest_version(manifest)?;
    let path = output_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| TTGenError::SystemError(format!("manifest serialize: {e}")))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Read and deserialize a manifest from `path`. Used by tests and
/// downstream consumers.
pub fn read_manifest(path: &Path) -> TTGenResult<TTManifest> {
    let s = std::fs::read_to_string(path)?;
    let manifest = serde_json::from_str::<TTManifest>(&s)
        .map_err(|e| TTGenError::SystemError(format!("manifest deserialize: {e}")))?;
    validate_manifest_version(&manifest)?;
    Ok(manifest)
}

// ─── Stats artifact (`stats.json`) ──────────────────────────────────────────

/// Bounded on-disk shape for `stats.json`.
///
/// Split assignment is owned by pass B and its exact bounded results are
/// persisted in `split_info.json` and `split_stats.json`. This diagnostic
/// deliberately contains no per-timestamp data and remains constant-size as
/// timestamp cardinality grows.
#[derive(Debug, Serialize)]
pub struct StatsArtifact<'a> {
    pub version: u32,
    pub kind: String,
    pub num_total_rows: u64,
    pub num_train_rows: u64,
    pub num_entities: u64,
    pub num_targets: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub num_links: Option<u64>,
    #[serde(flatten)]
    pub kumo_contract: &'a crate::ttgen_stats::KumoContractStats,
}

impl<'a> StatsArtifact<'a> {
    pub fn from_payload(payload: &'a TrainTableStatsPayload) -> Self {
        Self {
            version: 2,
            kind: "tt_stats".to_string(),
            num_total_rows: payload.num_total_rows,
            num_train_rows: payload.num_train_rows,
            num_entities: payload.num_entities,
            num_targets: payload.num_targets,
            num_links: payload.num_links,
            kumo_contract: &payload.kumo_contract,
        }
    }
}

/// Write `stats.json` to `output_dir`, returning its path.
pub fn write_stats_artifact(
    output_dir: &Path,
    payload: &TrainTableStatsPayload,
) -> TTGenResult<PathBuf> {
    let path = output_dir.join("stats.json");
    let artifact = StatsArtifact::from_payload(payload);
    let file = std::fs::File::create(&path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, &artifact)
        .map_err(|e| TTGenError::SystemError(format!("stats artifact serialize: {e}")))?;
    use std::io::Write;
    writer.flush()?;
    Ok(path)
}

// ─── Train-table col stats artifact (optional sidecar) ─────────────────────

/// Write `train_table_col_stats.json` to `output_dir` when the stats
/// payload carries `parsed_train_table_col_stats`. Returns
/// `Ok(Some(path))` if a file was written, `Ok(None)` if the payload had
/// nothing to write.
pub fn write_train_table_col_stats_artifact(
    output_dir: &Path,
    payload: &TrainTableStatsPayload,
) -> TTGenResult<Option<PathBuf>> {
    let stats = match &payload.parsed_train_table_col_stats {
        Some(s) => s.clone(),
        None => return Ok(None),
    };
    let artifact = serde_json::json!({
        "version": 1,
        "kind": "tt_train_table_col_stats",
        "stats_format": "kumo_col_stats_v1",
        "stats": stats,
    });
    let path = output_dir.join("train_table_col_stats.json");
    let json = serde_json::to_string_pretty(&artifact)
        .map_err(|e| TTGenError::SystemError(format!("col-stats serialize: {e}")))?;
    std::fs::write(&path, json)?;
    Ok(Some(path))
}

// ─── Checksum helpers ──────────────────────────────────────────────────────

/// SHA-256 of a byte slice, hex-encoded.
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

/// SHA-256 of a single file, hex-encoded. Kumo's reader compares
/// hex-strings exactly. Streams the file in 1 MiB chunks to avoid
/// allocating the entire payload for large parquet files.
pub fn sha256_file(path: &Path) -> TTGenResult<String> {
    use std::io::Read;
    let mut hasher = Sha256::new();
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{:02x}", b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttgen_stats::{IntColumnStats, KumoContractStats, TrainTableExactStatsPayload};

    fn empty_int_col() -> IntColumnStats {
        IntColumnStats {
            count: 0,
            null_count: 0,
            num_unique: None,
            min: None,
            max: None,
        }
    }

    fn empty_kumo_contract() -> KumoContractStats {
        KumoContractStats {
            num_target_rows: 0,
            num_timeframes: 0,
            num_timeframes_post_filter: 0,
            avg_entities_per_timeframe: 0,
            min_date: JsonValue::Null,
            max_date: JsonValue::Null,
            anchor_date: JsonValue::Null,
            node_degree_stats: JsonValue::Null,
            scoring_table_column_stats: JsonValue::Null,
            target_distribution: JsonValue::Object(serde_json::Map::new()),
        }
    }

    fn dummy_stats() -> TrainTableStatsPayload {
        TrainTableStatsPayload {
            version: 2,
            num_total_rows: 10,
            num_train_rows: 7,
            num_entities: 3,
            num_targets: 4,
            num_links: None,
            exact_stats: TrainTableExactStatsPayload {
                entity: empty_int_col(),
                timestamp: empty_int_col(),
                target: empty_int_col(),
                frequency: None,
                entity_include_bit: None,
            },
            parsed_train_table_col_stats: None,
            kumo_contract: empty_kumo_contract(),
        }
    }

    fn full_manifest_with_split() -> TTManifest {
        let stage_counts = {
            let mut m = HashMap::new();
            m.insert("train".to_string(), 7u64);
            m.insert("val".to_string(), 2u64);
            m.insert("test".to_string(), 1u64);
            m.insert("unused".to_string(), 0u64);
            m
        };
        TTManifest {
            version: TT_MANIFEST_VERSION,
            row_count: 10,
            source_mode: "pquery".into(),
            split_mode: "compute".into(),
            split_kind: Some("TimeRangeSplit".into()),
            stage_counts: Some(stage_counts),
            train_table: TrainTableArtifactRef::new(
                "tt.parquet",
                TT_PARQUET_FORMAT,
                10,
                1234,
                vec![TrainTablePartRef {
                    path: "tt.parquet/part-00000.parquet".into(),
                    row_count: 10,
                    size_bytes: 1234,
                }],
            ),
            split: Some(ArtifactRef::new("split.bin", SPLIT_FORMAT)),
            stats: ArtifactRef::new("stats.json", STATS_FORMAT),
            split_stats: None,
            stage_seed_times: None,
            train_table_col_stats: None,
            train_table_edges: None,
            arange_sidecar: Some(SidecarRef {
                path: "arange.bin".into(),
            }),
            weight_sidecar: None,
            fact_pseudo_time: None,
            checksums: ManifestChecksums {
                split_sha256: "b".into(),
                stats_sha256: "c".into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_manifest_roundtrip_full() {
        let dir = tempfile::tempdir().unwrap();
        let m = full_manifest_with_split();
        let path = write_manifest(dir.path(), &m).unwrap();
        let m2 = read_manifest(&path).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn fact_visibility_requires_manifest_v4() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = full_manifest_with_split();
        manifest.fact_pseudo_time = Some(FactPseudoTimeRef {
            path: "fact_pseudo_time.bin".into(),
            format: FACT_PSEUDO_TIME_FORMAT.into(),
            node_type: "orders".into(),
        });
        assert!(write_manifest(dir.path(), &manifest).is_err());
        manifest.version = FACT_PSEUDO_TIME_MANIFEST_VERSION;
        let path = write_manifest(dir.path(), &manifest).unwrap();
        assert_eq!(read_manifest(&path).unwrap(), manifest);
        manifest.fact_pseudo_time = None;
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(read_manifest(&path).is_err());
        assert!(write_manifest(dir.path(), &manifest).is_err());
    }

    #[test]
    fn test_manifest_v1_is_rejected_without_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = full_manifest_with_split();
        m.version = 1;
        let error = write_manifest(dir.path(), &m).unwrap_err();
        assert!(error.to_string().contains("expected=3 got=1"));

        let path = dir.path().join("manifest.json");
        let raw = serde_json::to_string(&m).unwrap();
        std::fs::write(&path, raw).unwrap();
        let error = read_manifest(&path).unwrap_err();
        assert!(error.to_string().contains("expected=3 got=1"));
    }

    #[test]
    fn test_manifest_optional_fields_elided() {
        let dir = tempfile::tempdir().unwrap();
        let m = TTManifest {
            version: TT_MANIFEST_VERSION,
            row_count: 0,
            source_mode: "custom_tt".into(),
            split_mode: "none".into(),
            split_kind: None,
            stage_counts: None,
            train_table: TrainTableArtifactRef::new(
                "tt.parquet",
                TT_PARQUET_FORMAT,
                0,
                4567,
                Vec::new(),
            ),
            split: None,
            stats: ArtifactRef::new("stats.json", STATS_FORMAT),
            split_stats: None,
            stage_seed_times: None,
            train_table_col_stats: None,
            train_table_edges: None,
            arange_sidecar: None,
            weight_sidecar: None,
            fact_pseudo_time: None,
            checksums: ManifestChecksums {
                split_sha256: "".into(),
                stats_sha256: "y".into(),
                ..Default::default()
            },
        };
        let path = write_manifest(dir.path(), &m).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        // Absent optionals shouldn't appear in JSON.
        assert!(!raw.contains("\"split\""), "split section should be elided");
        assert!(!raw.contains("split_stats"));
        assert!(!raw.contains("train_table_col_stats"));
        assert!(!raw.contains("train_table_edges"));
        assert!(!raw.contains("arange_sidecar"));
        assert!(!raw.contains("split_kind"));
        assert!(!raw.contains("stage_counts"));
        let m2 = read_manifest(&path).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_manifest_schema_matches_kumo_expectations() {
        // Asserts the on-disk JSON has the top-level keys kumo's
        // `validate_artifact_bundle` reaches for. Mirrors the keys
        // checked in `kumo/engine/diskgraph_native_tt_artifacts.py`.
        let dir = tempfile::tempdir().unwrap();
        let m = full_manifest_with_split();
        let path = write_manifest(dir.path(), &m).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();
        let obj = v.as_object().expect("manifest is JSON object");
        // Required keys.
        for key in [
            "version",
            "row_count",
            "train_table",
            "split",
            "stats",
            "checksums",
        ] {
            assert!(
                obj.contains_key(key),
                "manifest missing required key {key}, got keys: {:?}",
                obj.keys().collect::<Vec<_>>()
            );
        }
        // version must be integer 3 (kumo rejects any other shape).
        assert_eq!(obj["version"].as_u64(), Some(3));
        // train_table carries aggregate metadata plus the exact part inventory.
        let tt = obj["train_table"].as_object().unwrap();
        assert!(tt.contains_key("path"));
        assert_eq!(tt["format"].as_str(), Some(TT_PARQUET_FORMAT));
        assert_eq!(tt["row_count"].as_u64(), Some(10));
        assert_eq!(tt["parts"].as_array().unwrap().len(), 1);
        assert_eq!(tt["size_bytes"].as_u64(), Some(1234));
        // split must have path + format=split_assignment_row_stage_bin.
        let sp = obj["split"].as_object().unwrap();
        assert!(sp.contains_key("path"));
        assert_eq!(sp["format"].as_str(), Some(SPLIT_FORMAT));
        // stats must have path + format=tt_stats_json.
        let st = obj["stats"].as_object().unwrap();
        assert!(st.contains_key("path"));
        assert_eq!(st["format"].as_str(), Some(STATS_FORMAT));
        // Checksums remain required only for bounded sidecars.
        let cs = obj["checksums"].as_object().unwrap();
        assert!(!cs.contains_key("train_table_sha256"));
        for key in ["split_sha256", "stats_sha256"] {
            assert!(cs.contains_key(key), "checksums missing {key}");
        }
    }

    #[test]
    fn test_stats_artifact_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dummy_stats();
        let path = write_stats_artifact(dir.path(), &payload).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["version"].as_u64(), Some(2));
        assert_eq!(v["kind"].as_str(), Some("tt_stats"));
        assert_eq!(v["num_total_rows"].as_u64(), Some(10));
        assert_eq!(v["num_train_rows"].as_u64(), Some(7));
        assert_eq!(v["num_entities"].as_u64(), Some(3));
        assert_eq!(v["num_targets"].as_u64(), Some(4));
        for removed in [
            "provided_fields",
            "payload",
            "timestamp_counts",
            "stats_per_timeframe",
            "timestamp_stage_assignments",
            "kumo_contract",
        ] {
            assert!(v.get(removed).is_none(), "unexpected field {removed}");
        }
    }

    #[test]
    fn test_sha256_file_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, b"abc").unwrap();
        // sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let got = sha256_file(&path).unwrap();
        assert_eq!(
            got,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
