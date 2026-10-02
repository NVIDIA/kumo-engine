//! KUMO-CONTRACT BRIDGE — DELETE WHEN KUMO MIGRATES OFF SIDECAR JSON.
//!
//! Transitional `train_table_col_stats.json` sidecar; end-state is kumo reading
//! col stats from the permanent `summary['stats']` payload (KumoContractStats).
//! Current readers: kumo/engine/diskgraph_native_tt_artifacts.py
//! (parse_train_table_col_stats_artifact) and kumo/task/task.py
//! (seed_train_table_col_stats_from_manifest). When they migrate: rm this file,
//! drop the `write_all` call in ttgen_executor.rs, and remove the
//! `train_table_col_stats` field from ArtifactRefs in manifest.rs.
//!
//! Emits kumo `tt_train_table_col_stats` v1 (`stats_format:
//! "kumo_col_stats_v1"`), forwarded as-is from the payload's
//! `parsed_train_table_col_stats` map (already in the
//! `{col__stype__stat__approx: {value, approx, data_version, num_entries}}`
//! shape; built upstream by `ttgen_stats::build_inline_col_stats`).

use std::path::Path;

use serde_json::json;

use crate::errors::TTGenResult;
use crate::manifest::{
    sha256_file, ArtifactRef, ManifestChecksums, TTManifest, TRAIN_TABLE_COL_STATS_FORMAT,
};
use crate::ttgen_stats::TrainTableStatsPayload;

/// Emit the `train_table_col_stats.json` kumo-contract sidecar to
/// `output_dir`, populate the corresponding `ArtifactRef` and
/// `train_table_col_stats_sha256` fields on `manifest`, and return.
///
/// The sidecar is written only when the payload carries a non-empty
/// `parsed_train_table_col_stats`. Otherwise the manifest field stays
/// `None`.
pub fn write_all(
    payload: &TrainTableStatsPayload,
    output_dir: &Path,
    manifest: &mut TTManifest,
) -> TTGenResult<()> {
    if let Some((path, sha)) = write_train_table_col_stats_artifact(payload, output_dir)? {
        manifest.train_table_col_stats = Some(ArtifactRef::new(
            path.file_name()
                .expect("train-table col-stats path has a filename")
                .to_string_lossy()
                .into_owned(),
            TRAIN_TABLE_COL_STATS_FORMAT,
        ));
        manifest.checksums.train_table_col_stats_sha256 = Some(sha);
    }

    Ok(())
}

/// Convenience: build a fresh `ManifestChecksums` with only the three
/// required fields set; used by `write_all` callers that need to defer
/// the optional fields. Public so tests can construct manifests cleanly.
#[allow(dead_code)]
pub fn empty_checksums(
    split_sha256: String,
    stats_sha256: String,
) -> ManifestChecksums {
    ManifestChecksums {
        split_sha256,
        stats_sha256,
        split_stats_sha256: None,
        train_table_col_stats_sha256: None,
        train_table_edges_sha256: None,
        weight_sidecar_sha256: None,
        fact_pseudo_time_sha256: None,
    }
}

// ─── train_table_col_stats.json ────────────────────────────────────────────

/// Emit `train_table_col_stats.json` (when payload has col-stats data)
/// and return `Some((path, sha256_hex))`. Returns `None` when the
/// payload's `parsed_train_table_col_stats` is empty.
///
/// Schema (kumo `parse_train_table_col_stats_artifact`):
/// ```json
/// {
///   "version": 1,
///   "kind": "tt_train_table_col_stats",
///   "stats_format": "kumo_col_stats_v1",
///   "stats": {
///     "<col>__<stype>__<stat>__<approx>": {
///       "value": ...,
///       "approx": bool,
///       "data_version": str,
///       "num_entries": int
///     }, ...
///   }
/// }
/// ```
///
/// The inner stat objects are produced upstream by
/// `ttgen_stats::build_inline_col_stats` via
/// `diskgraph_core::accumulators::{kumo_key, kumo_entry,
/// build_multicat_emit_stats}` and forwarded here verbatim. Numerical
/// columns emit COUNT / NUM_NA / NA_FRACTION / MEAN / STD / MIN / MAX /
/// QUANTILES / NUM_UNIQUE; categorical / boolean / ID emit COUNT /
/// CATEGORY_COUNTS / NUM_UNIQUE; multicategorical emits the MULTI_*
/// family.
fn write_train_table_col_stats_artifact(
    payload: &TrainTableStatsPayload,
    output_dir: &Path,
) -> TTGenResult<Option<(std::path::PathBuf, String)>> {
    // `parsed_train_table_col_stats` is already wrapped in the
    // kumo-format envelope by `build_inline_col_stats` (it returns
    // `Some({version, kind, stats_format, row_count, stats,
    // provided_fields})`). Extract the inner `stats` map and wrap it in
    // the artifact envelope kumo's reader expects on disk.
    let envelope = match &payload.parsed_train_table_col_stats {
        Some(s) => s,
        None => return Ok(None),
    };
    let stats = match envelope.get("stats") {
        Some(s) if s.is_object() => s,
        _ => return Ok(None),
    };
    // Kumo's parse_train_table_col_stats_artifact requires `stats` to be
    // a non-empty object (each entry has `value`). If we somehow got an
    // empty map, skip the artifact: emitting an empty file would parse
    // but seed zero col stats in kumo, masking a bug elsewhere.
    if stats.as_object().map(|m| m.is_empty()).unwrap_or(true) {
        return Ok(None);
    }

    let json_value = json!({
        "version": 1,
        "kind": "tt_train_table_col_stats",
        "stats_format": "kumo_col_stats_v1",
        "stats": stats,
    });

    let path = output_dir.join("train_table_col_stats.json");
    let json = serde_json::to_string_pretty(&json_value).map_err(|e| {
        crate::errors::TTGenError::SystemError(format!("col-stats serialize: {e}"))
    })?;
    std::fs::write(&path, json)?;
    let sha = sha256_file(&path)?;
    Ok(Some((path, sha)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        write_manifest, ArtifactRef, ManifestChecksums, SidecarRef, TTManifest,
        SPLIT_FORMAT, STATS_FORMAT, TT_PARQUET_FORMAT,
    };
    use crate::ttgen_stats::{
        IntColumnStats, KumoContractStats, TrainTableExactStatsPayload,
        TrainTableStatsPayload,
    };
    use serde_json::Value as JsonValue;

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

    /// `col_stats_map` is the inner `{key: kumo_entry}` map (the same
    /// shape `build_inline_col_stats` puts under `.stats`); we wrap it
    /// in the legacy envelope `{version, kind, stats_format, stats,
    /// ...}` here to match what `build_inline_col_stats` actually
    /// produces in production. The bridge re-extracts `.stats` from
    /// this envelope and re-wraps for on-disk emission.
    fn dummy_payload(
        num_train_rows: u64,
        num_entities: u64,
        col_stats_map: Option<JsonValue>,
    ) -> TrainTableStatsPayload {
        let parsed_train_table_col_stats = col_stats_map.map(|stats| {
            json!({
                "version": 1,
                "kind": "tt_train_table_col_stats",
                "stats_format": "kumo_col_stats_v1",
                "row_count": num_train_rows,
                "stats": stats,
                "provided_fields": ["stats"]
            })
        });
        TrainTableStatsPayload {
            version: 2,
            num_total_rows: num_train_rows,
            num_train_rows,
            num_entities,
            num_targets: num_train_rows,
            num_links: None,
            exact_stats: TrainTableExactStatsPayload {
                entity: empty_int_col(),
                timestamp: empty_int_col(),
                target: empty_int_col(),
                frequency: None,
                entity_include_bit: None,
            },
            parsed_train_table_col_stats,
            kumo_contract: empty_kumo_contract(),
        }
    }

    fn baseline_manifest() -> TTManifest {
        TTManifest {
            version: crate::manifest::TT_MANIFEST_VERSION,
            row_count: 1623,
            source_mode: "pquery".into(),
            split_mode: "compute".into(),
            split_kind: Some("TimeRangeSplit".into()),
            stage_counts: None,
            train_table: crate::manifest::TrainTableArtifactRef::new(
                "tt.parquet",
                TT_PARQUET_FORMAT,
                1623,
                1234,
                vec![crate::manifest::TrainTablePartRef {
                    path: "tt.parquet/part-00000.parquet".into(),
                    row_count: 1623,
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

    /// Round-trip the train_table_col_stats.json envelope. This test
    /// uses a synthetic numerical-style stats payload to assert the
    /// 4-tuple key format and the `{value: ..}` inner object that kumo's
    /// reader requires.
    #[test]
    fn test_col_stats_schema_numerical() {
        let dir = tempfile::tempdir().unwrap();
        // Mirror the kumo_key / kumo_entry shape produced by
        // build_inline_col_stats for a numerical TARGET1 column.
        let stats_map = json!({
            "TARGET1__kumo.numerical__COUNT__False": {
                "value": 1322,
                "approx": false,
                "data_version": "v1",
                "num_entries": 1322
            },
            "TARGET1__kumo.numerical__MEAN__False": {
                "value": 12.5,
                "approx": false,
                "data_version": "v1",
                "num_entries": 1322
            },
            "TARGET1__kumo.numerical__STD__False": {
                "value": 3.2,
                "approx": false,
                "data_version": "v1",
                "num_entries": 1322
            },
            "TARGET1__kumo.numerical__QUANTILES__False": {
                "value": [5.0, 12.0, 20.0],
                "approx": false,
                "data_version": "v1",
                "num_entries": 1322
            },
        });

        let payload = dummy_payload(1322, 1234, Some(stats_map.clone()));
        let result = write_train_table_col_stats_artifact(&payload, dir.path()).unwrap();
        let (path, _sha) = result.expect("col-stats artifact must be written");
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();

        assert_eq!(v["version"].as_u64(), Some(1));
        assert_eq!(v["kind"].as_str(), Some("tt_train_table_col_stats"));
        assert_eq!(v["stats_format"].as_str(), Some("kumo_col_stats_v1"));

        let stats = v["stats"].as_object().unwrap();
        assert!(stats.contains_key("TARGET1__kumo.numerical__COUNT__False"));
        assert!(stats.contains_key("TARGET1__kumo.numerical__MEAN__False"));
        assert!(stats.contains_key("TARGET1__kumo.numerical__STD__False"));
        assert!(stats.contains_key("TARGET1__kumo.numerical__QUANTILES__False"));

        // Each entry must be a `{value: ...}` object — kumo's reader rule.
        for (key, entry) in stats {
            let obj = entry.as_object().unwrap_or_else(|| {
                panic!("entry {key:?} must be an object")
            });
            assert!(obj.contains_key("value"), "entry {key:?} must contain `value`");
        }

        // Spot-check actual values.
        assert_eq!(
            stats["TARGET1__kumo.numerical__COUNT__False"]["value"].as_u64(),
            Some(1322)
        );
        assert!(
            (stats["TARGET1__kumo.numerical__MEAN__False"]["value"].as_f64().unwrap() - 12.5).abs()
                < 1e-9
        );
    }

    /// Categorical TARGET → CATEGORY_COUNTS + NUM_UNIQUE keys present,
    /// inner `value` is a list of [token, count] pairs.
    #[test]
    fn test_col_stats_schema_categorical() {
        let dir = tempfile::tempdir().unwrap();
        let stats_map = json!({
            "TARGET1__kumo.categorical__COUNT__False": {
                "value": 100,
                "approx": false,
                "data_version": "v1",
                "num_entries": 100
            },
            "TARGET1__kumo.categorical__NUM_UNIQUE__False": {
                "value": 3,
                "approx": false,
                "data_version": "v1",
                "num_entries": 100
            },
            "TARGET1__kumo.categorical__CATEGORY_COUNTS__False": {
                "value": [["a", 60], ["b", 30], ["c", 10]],
                "approx": false,
                "data_version": "v1",
                "num_entries": 100
            },
        });

        let payload = dummy_payload(100, 80, Some(stats_map));
        let (path, _sha) = write_train_table_col_stats_artifact(&payload, dir.path())
            .unwrap()
            .expect("col-stats written");
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();
        let stats = v["stats"].as_object().unwrap();
        assert!(stats.contains_key("TARGET1__kumo.categorical__NUM_UNIQUE__False"));
        let cat = &stats["TARGET1__kumo.categorical__CATEGORY_COUNTS__False"];
        assert!(cat["value"].is_array());
        assert_eq!(cat["value"].as_array().unwrap().len(), 3);
        // First pair: [token, count]
        let first = &cat["value"][0];
        assert!(first.is_array());
        assert_eq!(first[0].as_str(), Some("a"));
        assert_eq!(first[1].as_u64(), Some(60));
    }

    /// Multicategorical TARGET → MULTI_CATEGORY_COUNTS +
    /// MULTI_CATEGORIES_SEPARATOR + NUM_UNIQUE_MULTI keys present.
    /// Mirrors the `build_multicat_emit_stats` output shape.
    #[test]
    fn test_col_stats_schema_multicategorical() {
        let dir = tempfile::tempdir().unwrap();
        let stats_map = json!({
            "TARGET1__kumo.multicategorical__COUNT__False": {
                "value": 50,
                "approx": false,
                "data_version": "v1",
                "num_entries": 50
            },
            "TARGET1__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False": {
                "value": "|",
                "approx": false,
                "data_version": "v1",
                "num_entries": 50
            },
            "TARGET1__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False": {
                "value": [["x", 30], ["y", 20]],
                "approx": false,
                "data_version": "v1",
                "num_entries": 50
            },
            "TARGET1__kumo.multicategorical__NUM_UNIQUE_MULTI__True": {
                "value": 2,
                "approx": true,
                "data_version": "v1",
                "num_entries": 50
            },
        });

        let payload = dummy_payload(50, 40, Some(stats_map));
        let (path, _sha) = write_train_table_col_stats_artifact(&payload, dir.path())
            .unwrap()
            .expect("col-stats written");
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();
        let stats = v["stats"].as_object().unwrap();
        assert!(stats.contains_key(
            "TARGET1__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False"
        ));
        assert!(stats.contains_key(
            "TARGET1__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"
        ));
        // NUM_UNIQUE_MULTI is approx=true.
        let unique = &stats["TARGET1__kumo.multicategorical__NUM_UNIQUE_MULTI__True"];
        assert_eq!(unique["approx"], JsonValue::Bool(true));
        // Separator value is the literal pipe string.
        let sep = &stats["TARGET1__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False"];
        assert_eq!(sep["value"].as_str(), Some("|"));
    }

    /// `write_all` end-to-end: drives col-stats sidecar + manifest mutation.
    /// Verifies manifest references are populated and SHA-256 field is a
    /// non-empty hex string of length 64.
    #[test]
    fn test_col_stats_only() {
        let dir = tempfile::tempdir().unwrap();
        let stats_map = json!({
            "TARGET1__kumo.numerical__COUNT__False": {
                "value": 100, "approx": false, "data_version": "v1", "num_entries": 100
            }
        });
        let payload = dummy_payload(130, 90, Some(stats_map));
        let mut manifest = baseline_manifest();

        write_all(&payload, dir.path(), &mut manifest).unwrap();

        // split_stats stays None (removed from write_all).
        assert!(manifest.split_stats.is_none());
        assert!(manifest.checksums.split_stats_sha256.is_none());

        // train_table_col_stats populated.
        let cs = manifest
            .train_table_col_stats
            .as_ref()
            .expect("train_table_col_stats set");
        assert_eq!(cs.format, TRAIN_TABLE_COL_STATS_FORMAT);
        assert!(cs.path.ends_with("train_table_col_stats.json"));
        let cs_sha = manifest
            .checksums
            .train_table_col_stats_sha256
            .as_ref()
            .expect("col-stats sha set");
        assert_eq!(cs_sha.len(), 64);
        assert!(cs_sha.chars().all(|c| c.is_ascii_hexdigit()));

        // SHA recorded in manifest equals SHA of the actual file.
        let actual_cs_sha =
            sha256_file(&dir.path().join("train_table_col_stats.json")).unwrap();
        assert_eq!(&actual_cs_sha, cs_sha);

        // Manifest round-trips through write_manifest / read_manifest.
        let mpath = write_manifest(dir.path(), &manifest).unwrap();
        let m2 = crate::manifest::read_manifest(&mpath).unwrap();
        assert_eq!(m2.train_table_col_stats, manifest.train_table_col_stats);
    }

    /// `write_all` emits col-stats artifact when payload has it.
    #[test]
    fn test_write_all_col_stats_emitted() {
        let dir = tempfile::tempdir().unwrap();
        let stats_map = json!({
            "TARGET1__kumo.numerical__COUNT__False": {
                "value": 5, "approx": false, "data_version": "v1", "num_entries": 5
            }
        });
        let payload = dummy_payload(5, 5, Some(stats_map));
        let mut manifest = baseline_manifest();
        manifest.split = None;

        write_all(&payload, dir.path(), &mut manifest).unwrap();
        assert!(manifest.split_stats.is_none());
        assert!(manifest.checksums.split_stats_sha256.is_none());
        assert!(manifest.train_table_col_stats.is_some());
    }

    /// `write_all` with no col-stats payload → no artifact emitted.
    #[test]
    fn test_write_all_no_col_stats() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dummy_payload(1, 1, None);
        let mut manifest = baseline_manifest();

        write_all(&payload, dir.path(), &mut manifest).unwrap();
        assert!(manifest.split_stats.is_none());
        assert!(manifest.train_table_col_stats.is_none());
        assert!(manifest.checksums.train_table_col_stats_sha256.is_none());
    }

    /// Empty `parsed_train_table_col_stats` map (defensive guard) →
    /// artifact is NOT emitted; an empty stats object would parse but
    /// seed zero col stats in kumo, masking a bug elsewhere.
    #[test]
    fn test_write_all_empty_col_stats_map_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let payload = dummy_payload(1, 1, Some(json!({})));
        let mut manifest = baseline_manifest();

        write_all(&payload, dir.path(), &mut manifest).unwrap();
        assert!(manifest.train_table_col_stats.is_none());
    }

    /// Mirror kumo's `parse_train_table_col_stats_artifact` validation
    /// rules from the Rust side: re-read and assert the same gates.
    #[test]
    fn test_col_stats_kumo_validator_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let stats_map = json!({
            "TARGET1__kumo.numerical__COUNT__False": {
                "value": 1, "approx": false, "data_version": "v1", "num_entries": 1
            },
            "TARGET1__kumo.numerical__MEAN__False": {
                "value": 0.5, "approx": false, "data_version": "v1", "num_entries": 1
            }
        });
        let payload = dummy_payload(1, 1, Some(stats_map));
        let (path, _) =
            write_train_table_col_stats_artifact(&payload, dir.path())
                .unwrap()
                .expect("artifact written");
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: JsonValue = serde_json::from_str(&raw).unwrap();

        // kumo: (version, kind) == (1, 'tt_train_table_col_stats') and
        // stats_format == 'kumo_col_stats_v1'.
        assert_eq!(v["version"].as_u64(), Some(1));
        assert_eq!(v["kind"].as_str(), Some("tt_train_table_col_stats"));
        assert_eq!(v["stats_format"].as_str(), Some("kumo_col_stats_v1"));

        let stats = v["stats"].as_object().expect("stats must be a dict");
        for (key, value) in stats {
            // kumo: keys are non-empty strings.
            assert!(!key.is_empty());
            // kumo: each stat payload is an object with `value`.
            let obj = value
                .as_object()
                .unwrap_or_else(|| panic!("stat {key:?} payload must be object"));
            assert!(
                obj.contains_key("value"),
                "stat {key:?} missing `value` key"
            );
            // 4-tuple key format used by kumo's
            // seed_train_table_col_stats_from_manifest at line 594:
            // `parts = stat_key.rsplit('__', 3)`.
            let parts: Vec<&str> = key.rsplitn(4, "__").collect();
            assert_eq!(
                parts.len(),
                4,
                "key {key:?} must split into 4 __-separated parts"
            );
        }
    }
}
