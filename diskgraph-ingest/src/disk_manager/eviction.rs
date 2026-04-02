use super::path_safety::canonicalize_under;
use super::types::{EnsureSpaceResult, EvictedIndex, IndexInfo};
use super::DiskManagerError;
use std::path::Path;

/// Plan which indexes to evict to free `needed_bytes`.
///
/// Eviction order:
/// 1. Orphans first (no `_SUCCESS` marker — crashed/partial ingests)
/// 2. LRU: least recently used first (oldest `_SUCCESS` mtime)
///
/// Stops as soon as enough space would be freed. Skips the current ingest target.
pub fn plan_eviction(
    indexes: &[IndexInfo],
    needed_bytes: u64,
    available_bytes: u64,
    current_graph_key: &str,
    current_graph_version: &str,
) -> Vec<EvictedIndex> {
    if available_bytes >= needed_bytes {
        return Vec::new();
    }

    let deficit = needed_bytes - available_bytes;

    let mut orphans: Vec<&IndexInfo> = Vec::new();
    let mut valid: Vec<&IndexInfo> = Vec::new();

    for idx in indexes {
        if idx.graph_key == current_graph_key && idx.graph_version == current_graph_version {
            continue;
        }
        if idx.is_orphan() {
            orphans.push(idx);
        } else {
            valid.push(idx);
        }
    }

    valid.sort_by_key(|idx| idx.success_mtime_secs.unwrap_or(0));

    let mut to_evict = Vec::new();
    let mut freed = 0u64;

    for idx in orphans.iter().chain(valid.iter()) {
        if freed >= deficit {
            break;
        }
        freed += idx.size_bytes;
        to_evict.push(EvictedIndex {
            path: idx.path.clone(),
            graph_key: idx.graph_key.clone(),
            graph_version: idx.graph_version.clone(),
            size_bytes: idx.size_bytes,
            was_orphan: idx.is_orphan(),
        });
    }

    to_evict
}

/// Execute an eviction plan: delete the directories and return the result.
pub fn execute_eviction(
    data_root: &Path,
    plan: &[EvictedIndex],
    needed_bytes: u64,
) -> Result<EnsureSpaceResult, DiskManagerError> {
    let mut freed_bytes = 0u64;
    let mut evicted = Vec::new();

    for entry in plan {
        if let Err(e) = canonicalize_under(data_root, &entry.path) {
            tracing::warn!("skipping eviction of {:?}: {}", entry.path, e);
            continue;
        }

        match std::fs::remove_dir_all(&entry.path) {
            Ok(()) => {
                tracing::info!(
                    graph_key = %entry.graph_key,
                    graph_version = %entry.graph_version,
                    size_bytes = entry.size_bytes,
                    was_orphan = entry.was_orphan,
                    "evicted index"
                );
                freed_bytes += entry.size_bytes;
                evicted.push(entry.clone());
            }
            Err(e) => {
                tracing::warn!("failed to evict {:?}: {}", entry.path, e);
            }
        }

        // Clean up empty graph_key parent directory
        if let Some(parent) = entry.path.parent() {
            if parent != data_root {
                if let Ok(mut rd) = std::fs::read_dir(parent) {
                    if rd.next().is_none() {
                        let _ = std::fs::remove_dir(parent);
                    }
                }
            }
        }
    }

    let available_bytes = super::available_space(data_root).unwrap_or(0);

    if available_bytes < needed_bytes {
        return Err(DiskManagerError::InsufficientSpace {
            needed: needed_bytes,
            available: available_bytes,
        });
    }

    Ok(EnsureSpaceResult {
        evicted,
        available_bytes,
        needed_bytes,
        freed_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_index(key: &str, ver: &str, size: u64, mtime: Option<u64>) -> IndexInfo {
        IndexInfo {
            path: PathBuf::from(format!("/data/{}/{}", key, ver)),
            graph_key: key.to_string(),
            graph_version: ver.to_string(),
            size_bytes: size,
            has_success_marker: mtime.is_some(),
            success_mtime_secs: mtime,
        }
    }

    #[test]
    fn test_no_eviction_when_enough_space() {
        let plan = plan_eviction(
            &[make_index("A", "v1", 1000, Some(100))],
            500, 1000, "A", "v2",
        );
        assert!(plan.is_empty());
    }

    #[test]
    fn test_orphans_evicted_first() {
        let indexes = vec![
            make_index("A", "v1", 500, Some(100)),
            make_index("A", "v2", 300, None),
            make_index("B", "v1", 200, Some(200)),
        ];
        let plan = plan_eviction(&indexes, 900, 500, "C", "v1");
        assert_eq!(plan.len(), 2);
        assert!(plan[0].was_orphan);
        assert!(!plan[1].was_orphan);
        assert_eq!(plan[1].graph_key, "A");
    }

    #[test]
    fn test_skips_current_target() {
        let indexes = vec![
            make_index("A", "v1", 500, Some(100)),
            make_index("A", "v2", 500, Some(200)),
        ];
        let plan = plan_eviction(&indexes, 800, 400, "A", "v2");
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].graph_version, "v1");
    }

    #[test]
    fn test_lru_order() {
        let indexes = vec![
            make_index("A", "v1", 100, Some(300)),
            make_index("A", "v2", 100, Some(100)),
            make_index("A", "v3", 100, Some(200)),
        ];
        let plan = plan_eviction(&indexes, 500, 200, "B", "v1");
        assert_eq!(plan[0].graph_version, "v2");
        assert_eq!(plan[1].graph_version, "v3");
        assert_eq!(plan[2].graph_version, "v1");
    }

    #[test]
    fn test_stops_when_enough_freed() {
        let indexes = vec![
            make_index("A", "v1", 500, Some(100)),
            make_index("A", "v2", 500, Some(200)),
        ];
        let plan = plan_eviction(&indexes, 900, 500, "B", "v1");
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn test_evicts_across_graph_keys() {
        let indexes = vec![
            make_index("A", "v1", 300, Some(100)),
            make_index("B", "v1", 300, Some(200)),
        ];
        let plan = plan_eviction(&indexes, 800, 200, "D", "v1");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].graph_key, "A");
        assert_eq!(plan[1].graph_key, "B");
    }

    #[test]
    fn test_execute_eviction_removes_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let idx = root.join("graph-A").join("v1");
        std::fs::create_dir_all(&idx).unwrap();
        std::fs::write(idx.join("data.bin"), vec![0u8; 100]).unwrap();

        let plan = vec![EvictedIndex {
            path: idx.clone(),
            graph_key: "graph-A".into(),
            graph_version: "v1".into(),
            size_bytes: 100,
            was_orphan: false,
        }];

        let result = execute_eviction(root, &plan, 100).unwrap();
        assert_eq!(result.evicted.len(), 1);
        assert!(!idx.exists());
    }
}
