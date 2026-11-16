// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Root directory under an ingest output where kumo-format col stats live.
pub const COL_STATS_DIR: &str = "kumo_col_stats";
/// Subdirectory holding the full (encoder-ready) per-table stats.
pub const COL_STATS_FULL: &str = "col_stats_full";
/// Subdirectory holding the minimal (encoder-free) per-table stats.
pub const COL_STATS_MIN: &str = "col_stats_min";
/// Filename of the graph-wide edge-stats JSON, in the ingest output root.
pub const EDGE_STATS_FILENAME: &str = "edge_stats.json";

#[derive(Debug, Error)]
pub enum ColStatsReadError {
    #[error("invalid col stats variant: {0:?} (expected \"full\" or \"min\")")]
    InvalidVariant(String),

    #[error("IO error reading {0}: {1}")]
    Io(String, std::io::Error),

    #[error("invalid JSON in {0}: {1}")]
    InvalidJson(String, serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColStatsVariant {
    Full,
    Min,
}

impl ColStatsVariant {
    pub fn from_str_input(s: &str) -> Result<Self, ColStatsReadError> {
        match s {
            "full" => Ok(Self::Full),
            "min" => Ok(Self::Min),
            other => Err(ColStatsReadError::InvalidVariant(other.to_string())),
        }
    }

    pub fn subdir(self) -> &'static str {
        match self {
            Self::Full => COL_STATS_FULL,
            Self::Min => COL_STATS_MIN,
        }
    }
}

/// Return the contents of `{index_path}/{COL_STATS_DIR}/{variant}/{table_name}.json`
/// as a JSON string, or `None` if the file doesn't exist.
///
/// Returns the raw bytes verbatim — no re-serialisation — so callers see the
/// exact JSON the ingestor wrote.
pub fn read_col_stats(
    index_path: &Path,
    table_name: &str,
    variant: ColStatsVariant,
) -> Result<Option<String>, ColStatsReadError> {
    let path = index_path
        .join(COL_STATS_DIR)
        .join(variant.subdir())
        .join(format!("{table_name}.json"));
    read_json_file(&path)
}

/// Return the contents of `{index_path}/{EDGE_STATS_FILENAME}` as a JSON
/// string, or `None` if the file doesn't exist.
pub fn read_edge_stats(index_path: &Path) -> Result<Option<String>, ColStatsReadError> {
    let path = index_path.join(EDGE_STATS_FILENAME);
    read_json_file(&path)
}

fn read_json_file(path: &Path) -> Result<Option<String>, ColStatsReadError> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(ColStatsReadError::Io(path.display().to_string(), err)),
    };
    // Defensive parse: the ingestor should always write valid JSON. If we
    // ever read something malformed, surface it as a data error rather than
    // hand garbage back to Python.
    if let Err(err) = serde_json::from_str::<Value>(&raw) {
        return Err(ColStatsReadError::InvalidJson(path.display().to_string(), err));
    }
    Ok(Some(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_col_stats(dir: &Path, variant: ColStatsVariant, table: &str, body: &str) {
        let subdir = dir.join(COL_STATS_DIR).join(variant.subdir());
        fs::create_dir_all(&subdir).unwrap();
        fs::write(subdir.join(format!("{table}.json")), body).unwrap();
    }

    #[test]
    fn roundtrip_full() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"col":"stats"}"#;
        write_col_stats(dir.path(), ColStatsVariant::Full, "USERS", body);
        let got = read_col_stats(dir.path(), "USERS", ColStatsVariant::Full).unwrap();
        assert_eq!(got.as_deref(), Some(body));
    }

    #[test]
    fn roundtrip_min() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"min":true}"#;
        write_col_stats(dir.path(), ColStatsVariant::Min, "LOAN", body);
        let got = read_col_stats(dir.path(), "LOAN", ColStatsVariant::Min).unwrap();
        assert_eq!(got.as_deref(), Some(body));
    }

    #[test]
    fn missing_table_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let got = read_col_stats(dir.path(), "ABSENT", ColStatsVariant::Full).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn missing_variant_dir_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        // Only write col_stats_full; ask for col_stats_min.
        write_col_stats(dir.path(), ColStatsVariant::Full, "USERS", "{}");
        let got = read_col_stats(dir.path(), "USERS", ColStatsVariant::Min).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn missing_index_path_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let ghost = dir.path().join("does-not-exist");
        let got = read_col_stats(&ghost, "USERS", ColStatsVariant::Full).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn invalid_variant_string() {
        match ColStatsVariant::from_str_input("approx") {
            Err(ColStatsReadError::InvalidVariant(got)) => assert_eq!(got, "approx"),
            other => panic!("expected InvalidVariant, got {:?}", other),
        }
    }

    #[test]
    fn corrupt_json_raises_data_error() {
        let dir = tempfile::tempdir().unwrap();
        write_col_stats(dir.path(), ColStatsVariant::Full, "BAD", "not valid json {{{");
        match read_col_stats(dir.path(), "BAD", ColStatsVariant::Full) {
            Err(ColStatsReadError::InvalidJson(path, _)) => {
                assert!(path.contains("BAD.json"), "path was: {path}");
            }
            other => panic!("expected InvalidJson, got {:?}", other),
        }
    }

    #[test]
    fn edge_stats_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"edges":[]}"#;
        fs::write(dir.path().join(EDGE_STATS_FILENAME), body).unwrap();
        let got = read_edge_stats(dir.path()).unwrap();
        assert_eq!(got.as_deref(), Some(body));
    }

    #[test]
    fn edge_stats_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_edge_stats(dir.path()).unwrap().is_none());
    }

    #[test]
    fn edge_stats_corrupt_raises_data_error() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(EDGE_STATS_FILENAME), "not valid").unwrap();
        match read_edge_stats(dir.path()) {
            Err(ColStatsReadError::InvalidJson(_, _)) => {}
            other => panic!("expected InvalidJson, got {:?}", other),
        }
    }
}
