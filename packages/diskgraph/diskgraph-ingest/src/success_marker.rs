// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use filetime::FileTime;
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;
use tempfile::NamedTempFile;
use thiserror::Error;

pub const SUCCESS_MARKER_FILENAME: &str = "_SUCCESS";

#[derive(Debug, Error)]
pub enum SuccessMarkerError {
    #[error("invalid success marker JSON: {0}")]
    InvalidPayload(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Deserialize)]
struct SuccessMarker {
    index_path: String,
    col_stats_path: String,
    ingest_stats_path: String,
    kumo_stats_dir: String,
}

pub fn read_success_marker(output_dir: &Path) -> Result<Option<String>, SuccessMarkerError> {
    let success_path = output_dir.join(SUCCESS_MARKER_FILENAME);
    let raw = match fs::read_to_string(&success_path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    let marker: SuccessMarker = match serde_json::from_str(&raw) {
        Ok(marker) => marker,
        Err(err) => {
            tracing::warn!(
                "Failed to parse _SUCCESS marker at {}: {}",
                success_path.display(),
                err
            );
            return Ok(None);
        }
    };

    for (path, name, expect_dir) in [
        (marker.col_stats_path.as_str(), "col_stats_path", false),
        (
            marker.ingest_stats_path.as_str(),
            "ingest_stats_path",
            false,
        ),
        (marker.index_path.as_str(), "index_path", true),
        (marker.kumo_stats_dir.as_str(), "kumo_stats_dir", true),
    ] {
        let exists = if expect_dir {
            Path::new(path).is_dir()
        } else {
            Path::new(path).is_file()
        };
        if !exists {
            tracing::warn!("Cached diskgraph output missing {} at {}", name, path);
            return Ok(None);
        }
    }

    Ok(Some(raw))
}

pub fn write_success_marker(
    output_dir: &Path,
    payload_json: &str,
) -> Result<(), SuccessMarkerError> {
    let _ = serde_json::from_str::<serde_json::Value>(payload_json)?;

    fs::create_dir_all(output_dir)?;
    let success_path = output_dir.join(SUCCESS_MARKER_FILENAME);
    let mut tmp_file = NamedTempFile::new_in(output_dir)?;
    tmp_file.write_all(payload_json.as_bytes())?;
    tmp_file.as_file().sync_all()?;
    tmp_file
        .persist(&success_path)
        .map_err(|err| SuccessMarkerError::Io(err.error))?;
    tracing::info!("Wrote _SUCCESS marker to {}", success_path.display());
    Ok(())
}

pub fn touch_success_marker(output_dir: &Path) -> Result<bool, SuccessMarkerError> {
    let success_path = output_dir.join(SUCCESS_MARKER_FILENAME);
    if !success_path.is_file() {
        return Ok(false);
    }

    let mtime = FileTime::from_system_time(SystemTime::now());
    filetime::set_file_mtime(&success_path, mtime)?;
    Ok(true)
}
#[cfg(test)]
mod tests {
    use super::*;

    fn valid_marker(dir: &Path) -> String {
        let col_stats = dir.join("col_stats.json");
        let ingest_stats = dir.join("ingest_stats.json");
        let kumo_stats = dir.join("kumo_col_stats");
        fs::write(&col_stats, "{}").unwrap();
        fs::write(&ingest_stats, "{}").unwrap();
        fs::create_dir_all(&kumo_stats).unwrap();

        serde_json::json!({
            "index_path": dir,
            "col_stats_path": col_stats,
            "ingest_stats_path": ingest_stats,
            "kumo_stats_dir": kumo_stats,
        })
        .to_string()
    }

    #[test]
    fn test_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let payload = valid_marker(dir.path());

        write_success_marker(dir.path(), &payload).unwrap();
        let read_back = read_success_marker(dir.path()).unwrap();

        assert_eq!(read_back.as_deref(), Some(payload.as_str()));
    }

    #[test]
    fn test_missing_marker_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_success_marker(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_corrupt_json_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join(SUCCESS_MARKER_FILENAME),
            "not valid json {{{",
        )
        .unwrap();

        assert!(read_success_marker(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_missing_referent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let payload = valid_marker(dir.path());
        let col_stats = dir.path().join("col_stats.json");
        write_success_marker(dir.path(), &payload).unwrap();
        fs::remove_file(col_stats).unwrap();

        assert!(read_success_marker(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_touch_missing_marker_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!touch_success_marker(dir.path()).unwrap());
    }
}
