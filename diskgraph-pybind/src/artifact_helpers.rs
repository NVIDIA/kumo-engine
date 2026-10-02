// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for artifact-bundle writers.
//!
//! This module collects utilities that used to live privately in
//! `tt_executor.rs` and are shared between the TT artifact bundle, the
//! prediction artifact bundle, and the task-dataset bundle.

use anyhow::Result;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// --- checksums ---------------------------------------------------------------

pub fn sha256_hex_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn sha256_hex_file(path: &str) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut f = File::open(path)?;
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

fn collect_files_recursive(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Compute SHA256 for a file or a deterministic directory digest.
///
/// Directory digest format (Linux path semantics):
/// `sha256(rel_path + '\0' + file_sha256_hex + '\0' for each file sorted lexicographically)`.
pub fn sha256_hex_path(path: &Path) -> Result<String> {
    if path.is_file() {
        return sha256_hex_file(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf8 path not supported"))?,
        );
    }

    let mut hasher = Sha256::new();
    let files = collect_files_recursive(path)?;
    for file in files {
        let rel = file
            .strip_prefix(path)
            .map_err(|_| anyhow::anyhow!("failed to strip prefix while hashing path"))?;
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0u8]);
        let file_hash = sha256_hex_file(
            file.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf8 path not supported"))?,
        )?;
        hasher.update(file_hash.as_bytes());
        hasher.update([0u8]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{:02x}", b)).collect())
}

/// Merge a nested `checksums` JSON object from `source` into `target`, prefixing
/// each key with `"{prefix}__"`.
///
/// This is the pattern used when attaching side-artifact bundles that carry
/// their own checksums map (for example `train_table_edges` or
/// `prediction_table_col_stats`).  If `source` has no `checksums` key or the
/// value is not an object, this is a no-op.
pub fn merge_prefixed_checksums(target: &mut Map<String, Value>, source: &Value, prefix: &str) {
    if let Some(source_checksums) = source.get("checksums").and_then(|v| v.as_object()) {
        for (k, v) in source_checksums {
            target.insert(format!("{prefix}__{k}"), v.clone());
        }
    }
}

// --- I/O ---------------------------------------------------------------------

pub fn ensure_artifacts_dir(artifacts_dir: &str) -> Result<()> {
    std::fs::create_dir_all(artifacts_dir)?;
    Ok(())
}

pub fn write_artifact_file(path: &str, data: &[u8]) -> Result<()> {
    let mut f = File::create(path)?;
    f.write_all(data)?;
    Ok(())
}

pub fn write_manifest_json(path: &str, manifest: &Value) -> Result<()> {
    let mut f = File::create(path)?;
    f.write_all(serde_json::to_string(manifest)?.as_bytes())?;
    Ok(())
}

// --- manifest ref helpers ----------------------------------------------------

/// Build a `{"path": ..., "format": ...}` artifact reference object.
pub fn artifact_ref(path: &str, format: &str) -> Value {
    json!({"path": path, "format": format})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn sha256_hex_file_matches_bytes() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("abc.bin");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_hex_file(p.to_str().unwrap()).unwrap(),
            sha256_hex_bytes(b"abc")
        );
    }

    #[test]
    fn sha256_hex_path_on_file_matches_bytes() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("abc.bin");
        fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_hex_path(&p).unwrap(), sha256_hex_bytes(b"abc"));
    }

    #[test]
    fn sha256_hex_path_on_directory_is_deterministic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), b"aaa").unwrap();
        fs::write(dir.path().join("b.txt"), b"bbb").unwrap();
        let h1 = sha256_hex_path(dir.path()).unwrap();
        let h2 = sha256_hex_path(dir.path()).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn sha256_hex_path_directory_hash_depends_on_content() {
        let dir_a = TempDir::new().unwrap();
        fs::write(dir_a.path().join("a.txt"), b"aaa").unwrap();
        let dir_b = TempDir::new().unwrap();
        fs::write(dir_b.path().join("a.txt"), b"bbb").unwrap();
        assert_ne!(
            sha256_hex_path(dir_a.path()).unwrap(),
            sha256_hex_path(dir_b.path()).unwrap()
        );
    }

    #[test]
    fn merge_prefixed_checksums_inserts_with_prefix() {
        let source = json!({"checksums": {"inner": "abc123", "other": "def456"}});
        let mut target = Map::new();
        merge_prefixed_checksums(&mut target, &source, "train_table_edges");
        assert_eq!(
            target.get("train_table_edges__inner"),
            Some(&Value::String("abc123".into()))
        );
        assert_eq!(
            target.get("train_table_edges__other"),
            Some(&Value::String("def456".into()))
        );
    }

    #[test]
    fn merge_prefixed_checksums_noop_when_missing() {
        let mut target = Map::new();
        merge_prefixed_checksums(&mut target, &json!({}), "whatever");
        assert!(target.is_empty());
    }

    #[test]
    fn merge_prefixed_checksums_noop_when_not_object() {
        let mut target = Map::new();
        merge_prefixed_checksums(&mut target, &json!({"checksums": "not an object"}), "p");
        assert!(target.is_empty());
    }

    #[test]
    fn artifact_ref_shape() {
        let r = artifact_ref("/some/path", "my_format");
        assert_eq!(r.get("path").unwrap().as_str(), Some("/some/path"));
        assert_eq!(r.get("format").unwrap().as_str(), Some("my_format"));
    }

    #[test]
    fn write_and_read_artifact_file_roundtrips() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("artifact.bin");
        write_artifact_file(p.to_str().unwrap(), b"hello").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"hello");
    }

    #[test]
    fn write_manifest_json_writes_serialized_form() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("manifest.json");
        let m = json!({"a": 1, "b": "two"});
        write_manifest_json(p.to_str().unwrap(), &m).unwrap();
        let on_disk: Value = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(on_disk, m);
    }

    #[test]
    fn ensure_artifacts_dir_creates_nested_paths() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a/b/c");
        ensure_artifacts_dir(nested.to_str().unwrap()).unwrap();
        assert!(nested.is_dir());
    }
}
