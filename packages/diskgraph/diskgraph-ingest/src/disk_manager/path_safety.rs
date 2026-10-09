// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// Validate that a path component (graph key or version) is safe for use in
/// filesystem paths. Rejects traversal attacks, null bytes, and empty strings.
pub fn validate_path_component(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("path component must not be empty");
    }
    if name.contains('\0') {
        bail!("path component must not contain null bytes: {:?}", name);
    }
    if name.contains('/') || name.contains('\\') {
        bail!(
            "path component must not contain path separators: {:?}",
            name
        );
    }
    if name == "." || name == ".." {
        bail!("path component must not be '.' or '..': {:?}", name);
    }
    if name.starts_with('.') {
        bail!("path component must not start with '.': {:?}", name);
    }
    Ok(())
}

/// Canonicalize `child` and verify it is strictly under `parent`.
/// Returns the canonical child path. Rejects symlink escapes and traversal.
pub fn canonicalize_under(parent: &Path, child: &Path) -> Result<PathBuf> {
    let canon_parent = parent
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("cannot canonicalize parent {:?}: {}", parent, e))?;
    let canon_child = child
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("cannot canonicalize child {:?}: {}", child, e))?;

    if !canon_child.starts_with(&canon_parent) {
        bail!(
            "path {:?} is not under {:?} (canonicalized: {:?} vs {:?})",
            child,
            parent,
            canon_child,
            canon_parent
        );
    }

    // Must be strictly under, not equal to parent
    if canon_child == canon_parent {
        bail!("path {:?} equals parent {:?}, expected a child", child, parent);
    }

    Ok(canon_child)
}

/// Build a safe path under `data_root` from graph_key and graph_version,
/// validating both components.
pub fn safe_index_path(data_root: &Path, graph_key: &str, graph_version: &str) -> Result<PathBuf> {
    validate_path_component(graph_key)?;
    validate_path_component(graph_version)?;
    Ok(data_root.join(graph_key).join(graph_version))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_components() {
        assert!(validate_path_component("my-graph").is_ok());
        assert!(validate_path_component("graph_v2").is_ok());
        assert!(validate_path_component("2025-03-27T12:00:00Z").is_ok());
    }

    #[test]
    fn test_rejects_traversal() {
        assert!(validate_path_component("..").is_err());
        assert!(validate_path_component(".").is_err());
        assert!(validate_path_component("foo/bar").is_err());
        assert!(validate_path_component("foo\\bar").is_err());
    }

    #[test]
    fn test_rejects_hidden() {
        assert!(validate_path_component(".hidden").is_err());
        assert!(validate_path_component(".lock").is_err());
    }

    #[test]
    fn test_rejects_empty_and_null() {
        assert!(validate_path_component("").is_err());
        assert!(validate_path_component("foo\0bar").is_err());
    }

    #[test]
    fn test_safe_index_path() {
        let root = Path::new("/data");
        let path = safe_index_path(root, "my-graph", "v1").unwrap();
        assert_eq!(path, PathBuf::from("/data/my-graph/v1"));
    }

    #[test]
    fn test_safe_index_path_rejects_bad_key() {
        let root = Path::new("/data");
        assert!(safe_index_path(root, "../etc", "v1").is_err());
        assert!(safe_index_path(root, "ok", "..").is_err());
    }

    #[test]
    fn test_canonicalize_under_valid() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path();
        let child_dir = parent.join("subdir");
        std::fs::create_dir_all(&child_dir).unwrap();

        let result = canonicalize_under(parent, &child_dir).unwrap();
        assert_eq!(result, child_dir.canonicalize().unwrap());
    }

    #[test]
    fn test_canonicalize_under_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path();
        let link_path = parent.join("escape");
        std::os::unix::fs::symlink("/tmp", &link_path).unwrap();

        let result = canonicalize_under(parent, &link_path);
        assert!(result.is_err(), "symlink escaping parent should be rejected");
    }

    #[test]
    fn test_canonicalize_under_nonexistent_child() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path();
        let nonexistent = parent.join("does-not-exist");

        let result = canonicalize_under(parent, &nonexistent);
        assert!(result.is_err(), "non-existent child should fail canonicalization");
    }

    #[test]
    fn test_canonicalize_under_child_equals_parent() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path();

        let result = canonicalize_under(parent, parent);
        assert!(result.is_err(), "child equal to parent should be rejected");
    }
}
