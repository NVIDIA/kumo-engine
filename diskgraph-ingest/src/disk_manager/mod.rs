//! Diskgraph disk space manager.
//!
//! Manages local SSD space for diskgraph indexes. Before each ingest, scans
//! existing indexes and evicts unused ones (LRU) to make room for the new
//! ingest. Designed for "1 job at a time" concurrency model.

pub mod eviction;
pub mod path_safety;
pub mod scan;
pub mod types;

use self::eviction::{execute_eviction, plan_eviction};
use self::path_safety::validate_path_component;
use self::scan::scan_indexes;
use self::types::{DiskUsage, EnsureSpaceRequest, EnsureSpaceResult, IndexInfo};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Typed errors for disk management operations.
#[derive(Debug, Error)]
pub enum DiskManagerError {
    /// Invalid path component (traversal, null bytes, etc.)
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Not enough disk space even after evicting everything evictable.
    #[error("Insufficient disk space: need {needed} bytes, only {available} available after eviction")]
    InsufficientSpace { needed: u64, available: u64 },

    /// IO error during scan/eviction.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type DiskManagerResult<T> = std::result::Result<T, DiskManagerError>;

/// Manages disk space for diskgraph indexes on local storage.
pub struct DiskManager {
    data_root: PathBuf,
}

impl DiskManager {
    pub fn new(data_root: &Path) -> Self {
        // Canonicalize to resolve symlinks and .. segments.
        // Fall back to the original path if canonicalization fails
        // (e.g., path doesn't exist yet).
        let resolved = data_root.canonicalize().unwrap_or_else(|_| data_root.to_path_buf());
        Self {
            data_root: resolved,
        }
    }

    /// Ensure enough disk space is available for a new ingest.
    ///
    /// If space is insufficient, evicts unused indexes in LRU order
    /// (orphans first, then least-recently-used valid indexes).
    /// Never evicts the current ingest target.
    pub fn ensure_space(&self, req: &EnsureSpaceRequest) -> DiskManagerResult<EnsureSpaceResult> {
        validate_path_component(&req.current_graph_key)
            .map_err(|e| DiskManagerError::InvalidPath(e.to_string()))?;
        validate_path_component(&req.current_graph_version)
            .map_err(|e| DiskManagerError::InvalidPath(e.to_string()))?;

        let available = available_space(&self.data_root)?;

        tracing::info!(
            needed_bytes = req.needed_bytes,
            available_bytes = available,
            current_target = %format!("{}/{}", req.current_graph_key, req.current_graph_version),
            "disk space check"
        );

        if available >= req.needed_bytes {
            return Ok(EnsureSpaceResult {
                evicted: Vec::new(),
                available_bytes: available,
                needed_bytes: req.needed_bytes,
                freed_bytes: 0,
            });
        }

        let deficit = req.needed_bytes - available;
        tracing::info!(deficit, "insufficient space, scanning for eviction");

        let indexes = scan_indexes(&self.data_root)?;
        tracing::info!(
            index_count = indexes.len(),
            orphan_count = indexes.iter().filter(|i| i.is_orphan()).count(),
            "scan complete"
        );

        let plan = plan_eviction(
            &indexes,
            req.needed_bytes,
            available,
            &req.current_graph_key,
            &req.current_graph_version,
        );

        if plan.is_empty() {
            return Err(DiskManagerError::InsufficientSpace {
                needed: req.needed_bytes,
                available,
            });
        }

        tracing::info!(
            eviction_count = plan.len(),
            eviction_bytes = plan.iter().map(|e| e.size_bytes).sum::<u64>(),
            "executing eviction plan"
        );

        execute_eviction(&self.data_root, &plan, req.needed_bytes)
    }

    /// Scan all index directories under data_root.
    pub fn scan_indexes(&self) -> DiskManagerResult<Vec<IndexInfo>> {
        Ok(scan_indexes(&self.data_root)?)
    }

    /// Get disk usage statistics.
    pub fn disk_usage(&self) -> DiskManagerResult<DiskUsage> {
        let total = total_space(&self.data_root)?;
        let available = available_space(&self.data_root)?;
        Ok(DiskUsage {
            total_bytes: total,
            available_bytes: available,
            used_bytes: total.saturating_sub(available),
        })
    }
}

/// Available bytes on the filesystem containing `path` (via statvfs).
pub fn available_space(path: &Path) -> std::io::Result<u64> {
    let stat = statvfs(path)?;
    // statvfs field widths are platform-dependent (u32 on aarch64-apple-darwin,
    // u64 on linux); widen via u64 multiply so the function's return type is
    // honored uniformly across targets.
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// Total bytes on the filesystem containing `path` (via statvfs).
pub fn total_space(path: &Path) -> std::io::Result<u64> {
    let stat = statvfs(path)?;
    Ok((stat.f_blocks as u64).saturating_mul(stat.f_frsize as u64))
}

fn statvfs(path: &Path) -> std::io::Result<libc::statvfs> {
    let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(stat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;

    fn create_index(root: &Path, key: &str, ver: &str, data_size: usize, mtime_secs: i64) {
        let dir = root.join(key).join(ver);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("data.bin"), vec![0u8; data_size]).unwrap();
        let success = dir.join("_SUCCESS");
        fs::write(&success, b"{}").unwrap();
        let ft = filetime::FileTime::from_unix_time(mtime_secs, 0);
        filetime::set_file_mtime(&success, ft).unwrap();
    }

    fn create_orphan(root: &Path, key: &str, ver: &str, data_size: usize) {
        let dir = root.join(key).join(ver);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("partial.bin"), vec![0u8; data_size]).unwrap();
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_scan_and_ensure_space_enough() {
        let dir = tempfile::tempdir().unwrap();
        create_index(dir.path(), "graph-A", "v1", 100, 1000);

        let dm = DiskManager::new(dir.path());
        let result = dm.ensure_space(&EnsureSpaceRequest {
            needed_bytes: 100,
            current_graph_key: "graph-B".into(),
            current_graph_version: "v1".into(),
        }).unwrap();

        assert!(result.evicted.is_empty());
        assert_eq!(result.freed_bytes, 0);
        assert_eq!(result.needed_bytes, 100);
        assert!(result.available_bytes > 0);
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_scan_finds_valid_and_orphan() {
        let dir = tempfile::tempdir().unwrap();
        create_index(dir.path(), "graph-A", "v1", 100, 1000);
        create_orphan(dir.path(), "graph-B", "v1", 50);

        let dm = DiskManager::new(dir.path());
        let indexes = dm.scan_indexes().unwrap();
        assert_eq!(indexes.len(), 2);

        assert!(!indexes.iter().find(|i| i.graph_key == "graph-A").unwrap().is_orphan());
        assert!(indexes.iter().find(|i| i.graph_key == "graph-B").unwrap().is_orphan());
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_eviction_removes_dirs_and_cleans_parent() {
        let dir = tempfile::tempdir().unwrap();
        create_index(dir.path(), "graph-A", "v1", 100, 1000);

        let plan = vec![types::EvictedIndex {
            path: dir.path().join("graph-A").join("v1"),
            graph_key: "graph-A".into(),
            graph_version: "v1".into(),
            size_bytes: 100,
            was_orphan: false,
        }];
        let result = eviction::execute_eviction(dir.path(), &plan, 100).unwrap();
        assert_eq!(result.evicted.len(), 1);
        assert!(!dir.path().join("graph-A").exists());
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_eviction_lru_order_real_fs() {
        let dir = tempfile::tempdir().unwrap();
        create_index(dir.path(), "graph-A", "v1", 100, 1000);
        create_index(dir.path(), "graph-B", "v1", 100, 3000);
        create_index(dir.path(), "graph-C", "v1", 100, 2000);

        let dm = DiskManager::new(dir.path());
        let indexes = dm.scan_indexes().unwrap();

        let plan = eviction::plan_eviction(&indexes, u64::MAX, 0, "graph-D", "v1");
        assert_eq!(plan[0].graph_key, "graph-A");
        assert_eq!(plan[1].graph_key, "graph-C");
        assert_eq!(plan[2].graph_key, "graph-B");
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_ensure_space_rejects_bad_graph_key() {
        let dir = tempfile::tempdir().unwrap();
        let dm = DiskManager::new(dir.path());
        let err = dm.ensure_space(&EnsureSpaceRequest {
            needed_bytes: 100,
            current_graph_key: "../etc".into(),
            current_graph_version: "v1".into(),
        }).unwrap_err();
        assert!(matches!(err, DiskManagerError::InvalidPath(_)));
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_disk_usage() {
        let dir = tempfile::tempdir().unwrap();
        let dm = DiskManager::new(dir.path());
        let usage = dm.disk_usage().unwrap();
        assert!(usage.total_bytes > 0);
        assert!(usage.available_bytes > 0);
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_eviction_under_real_disk_pressure() {
        let dir = tempfile::tempdir().unwrap();

        // Each index is several MB so that freeing even one index reclaims
        // enough real filesystem blocks to close the deficit below with margin.
        // CI runners can consume a few KiB between the pre-eviction statvfs
        // call and the post-eviction statvfs call.
        let index_size = 4_000_000;
        create_index(dir.path(), "graph-old", "v1", index_size, 1000);
        create_index(dir.path(), "graph-mid", "v1", index_size, 2000);
        create_index(dir.path(), "graph-new", "v1", index_size, 3000);

        // Record the paths before eviction so we can verify deletion.
        let old_path = dir.path().join("graph-old").join("v1");
        let mid_path = dir.path().join("graph-mid").join("v1");
        assert!(old_path.exists());
        assert!(mid_path.exists());

        let dm = DiskManager::new(dir.path());

        // Query real filesystem available space, then request slightly more
        // than what's available so that ensure_space must evict indexes to
        // close the deficit. Keep the deficit well below one evictable index
        // so the test is not sensitive to filesystem accounting jitter.
        let available = available_space(dir.path()).unwrap();
        let deficit = 1_000_000u64;
        let needed = available + deficit;

        let result = dm
            .ensure_space(&EnsureSpaceRequest {
                needed_bytes: needed,
                current_graph_key: "graph-current".into(),
                current_graph_version: "v1".into(),
            })
            .unwrap();

        // At least some indexes should have been evicted.
        assert!(
            !result.evicted.is_empty(),
            "expected eviction but none occurred"
        );
        assert!(
            result.freed_bytes > 0,
            "expected freed_bytes > 0, got {}",
            result.freed_bytes
        );

        // Eviction should respect LRU order: oldest first.
        assert_eq!(
            result.evicted[0].graph_key, "graph-old",
            "oldest index should be evicted first"
        );

        // Verify evicted directories are actually deleted from disk.
        for evicted in &result.evicted {
            assert!(
                !evicted.path.exists(),
                "evicted directory should be deleted: {:?}",
                evicted.path
            );
        }

        // The oldest index's parent dir should also be cleaned up.
        assert!(
            !dir.path().join("graph-old").exists(),
            "empty parent dir should be removed after eviction"
        );
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_ensure_space_insufficient_only_current_target() {
        let dir = tempfile::tempdir().unwrap();
        // Create one index that matches the current target — it must not be evicted.
        create_index(dir.path(), "graph-A", "v1", 1_000, 1000);

        let dm = DiskManager::new(dir.path());
        let available = available_space(dir.path()).unwrap();
        // Request more than available so eviction is attempted, but the only
        // index matches current_graph_key/version and is therefore protected.
        let needed = available + 10_000_000;

        let err = dm
            .ensure_space(&EnsureSpaceRequest {
                needed_bytes: needed,
                current_graph_key: "graph-A".into(),
                current_graph_version: "v1".into(),
            })
            .unwrap_err();

        assert!(
            matches!(err, DiskManagerError::InsufficientSpace { .. }),
            "expected InsufficientSpace, got {:?}",
            err
        );
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_ensure_space_nonexistent_root() {
        let fake_root = PathBuf::from(format!(
            "/tmp/nonexistent-dm-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!fake_root.exists());

        let dm = DiskManager::new(&fake_root);
        let err = dm
            .ensure_space(&EnsureSpaceRequest {
                needed_bytes: 100,
                current_graph_key: "graph-A".into(),
                current_graph_version: "v1".into(),
            })
            .unwrap_err();

        assert!(
            matches!(err, DiskManagerError::Io(_)),
            "expected Io error, got {:?}",
            err
        );
    }

    #[serial(disk_manager_real_fs)]
    #[test]
    fn test_eviction_keeps_parent_with_siblings() {
        // Tests that evicting one version does NOT remove the parent
        // graph-key directory when a sibling version still exists.
        // Uses execute_eviction directly (not ensure_space) to avoid
        // depending on real disk pressure, which is flaky on CI.
        let dir = tempfile::tempdir().unwrap();
        create_index(dir.path(), "graph-A", "v1", 100, 1000);
        create_index(dir.path(), "graph-A", "v2", 100, 3000);

        // Plan to evict only v1.
        let plan = vec![types::EvictedIndex {
            path: dir.path().join("graph-A").join("v1"),
            graph_key: "graph-A".into(),
            graph_version: "v1".into(),
            size_bytes: 100,
            was_orphan: false,
        }];

        let result = eviction::execute_eviction(dir.path(), &plan, 100).unwrap();
        assert_eq!(result.evicted.len(), 1);
        assert_eq!(result.evicted[0].graph_version, "v1");

        // v1 directory should be gone.
        assert!(
            !dir.path().join("graph-A").join("v1").exists(),
            "v1 should have been deleted"
        );
        // The graph-A parent dir must still exist because v2 remains.
        assert!(
            dir.path().join("graph-A").exists(),
            "parent dir should still exist when sibling v2 remains"
        );
        // v2 itself is still on disk.
        assert!(
            dir.path().join("graph-A").join("v2").exists(),
            "v2 should not have been evicted"
        );
    }
}
