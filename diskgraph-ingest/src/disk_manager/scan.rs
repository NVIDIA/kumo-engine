use super::path_safety::validate_path_component;
use super::types::IndexInfo;
use std::fs;
use std::io;
use std::path::Path;
use std::time::UNIX_EPOCH;

const SUCCESS_MARKER: &str = "_SUCCESS";

/// Scan all index directories under `data_root`.
///
/// Expected layout: `{data_root}/{graph_key}/{graph_version}/`
/// Each version directory is an index. Valid indexes have a `_SUCCESS` marker.
/// Directories without `_SUCCESS` are orphans (crashed/partial ingests).
pub fn scan_indexes(data_root: &Path) -> io::Result<Vec<IndexInfo>> {
    let mut indexes = Vec::new();

    let key_entries = match fs::read_dir(data_root) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!("cannot read data_root {:?}: {}", data_root, e);
            return Ok(indexes);
        }
    };

    for key_entry in key_entries {
        let key_entry = match key_entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !key_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        // Reject symlinks explicitly (file_type may follow symlinks on some filesystems)
        if key_entry.path().symlink_metadata()
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            continue;
        }
        let key_name = key_entry.file_name().to_string_lossy().to_string();
        if validate_path_component(&key_name).is_err() {
            continue;
        }

        let key_path = key_entry.path();
        let ver_entries = match fs::read_dir(&key_path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for ver_entry in ver_entries {
            let ver_entry = match ver_entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !ver_entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            // Reject symlinks explicitly (file_type may follow symlinks on some filesystems)
            if ver_entry.path().symlink_metadata()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                continue;
            }
            let ver_name = ver_entry.file_name().to_string_lossy().to_string();
            if validate_path_component(&ver_name).is_err() {
                continue;
            }

            let ver_path = ver_entry.path();
            let success_path = ver_path.join(SUCCESS_MARKER);
            let has_success = success_path.is_file();

            let success_mtime_secs = if has_success {
                fs::metadata(&success_path)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
            } else {
                None
            };

            let size_bytes = dir_size(&ver_path).unwrap_or(0);

            indexes.push(IndexInfo {
                path: ver_path,
                graph_key: key_name.clone(),
                graph_version: ver_name,
                size_bytes,
                has_success_marker: has_success,
                success_mtime_secs,
            });
        }
    }

    Ok(indexes)
}

/// Recursively compute the size of a directory.
pub fn dir_size(path: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_file() {
            total += entry.metadata()?.len();
        } else if ft.is_dir() {
            total += dir_size(&entry.path())?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_finds_valid_and_orphan_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // graph-A/v1: valid (has _SUCCESS)
        let v1 = root.join("graph-A").join("v1");
        fs::create_dir_all(&v1).unwrap();
        fs::write(v1.join("data.bin"), vec![0u8; 100]).unwrap();
        fs::write(v1.join("_SUCCESS"), b"{}").unwrap();

        // graph-A/v2: orphan (no _SUCCESS)
        let v2 = root.join("graph-A").join("v2");
        fs::create_dir_all(&v2).unwrap();
        fs::write(v2.join("data.bin"), vec![0u8; 50]).unwrap();

        let indexes = scan_indexes(root).unwrap();
        assert_eq!(indexes.len(), 2);

        let valid = indexes.iter().find(|i| i.graph_version == "v1").unwrap();
        assert!(valid.has_success_marker);
        assert!(!valid.is_orphan());

        let orphan = indexes.iter().find(|i| i.graph_version == "v2").unwrap();
        assert!(!orphan.has_success_marker);
        assert!(orphan.is_orphan());
    }

    #[test]
    fn test_scan_skips_hidden_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        fs::create_dir_all(root.join(".hidden").join("v1")).unwrap();
        fs::create_dir_all(root.join("graph-A")).unwrap();

        let indexes = scan_indexes(root).unwrap();
        assert!(indexes.is_empty());
    }

    #[test]
    fn test_scan_empty_root() {
        let dir = tempfile::tempdir().unwrap();
        let indexes = scan_indexes(dir.path()).unwrap();
        assert!(indexes.is_empty());
    }
}
