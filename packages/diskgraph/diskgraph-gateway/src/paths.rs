// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validate worker-local artifact and sidecar paths.
use std::{
    fs,
    path::{Component, Path, PathBuf},
};
use tonic::Status;

pub(crate) fn io_error(error: std::io::Error) -> Status {
    Status::internal(error.to_string())
}

pub(crate) fn relative_path(name: &str) -> Result<PathBuf, Status> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        || path
            .iter()
            .map(|s| s.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
            != name
    {
        return Err(Status::invalid_argument("Invalid task file path"));
    }
    Ok(path.to_path_buf())
}

pub(crate) fn task_path(root: &Path, name: &str) -> Result<PathBuf, Status> {
    let path = relative_path(name)?;
    let parts: Vec<_> = path.iter().collect();
    if parts.len() != 2
        || (parts[0] != "train_tables" && parts[0] != "pred_tables")
        || parts[1].to_string_lossy().starts_with('.')
    {
        return Err(Status::invalid_argument(
            "Expected train_tables/<id> or pred_tables/<id>",
        ));
    }
    let namespace = root.join(parts[0]);
    for candidate in [&namespace, &root.join(&path)] {
        match fs::symlink_metadata(candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(Status::invalid_argument(
                    "Task paths must not contain symlinks",
                ))
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error(error)),
        }
    }
    let extension = root.join(path);
    if !extension.join("index_metadata.json").is_file() {
        return Err(Status::failed_precondition(format!(
            "Training-table artifact `{name}` is not ready; materialize it on the worker before sampling"
        )));
    }
    Ok(extension)
}

pub(crate) fn resolve_beneath(root: &Path, name: &str) -> Result<PathBuf, Status> {
    let path = root
        .join(relative_path(name)?)
        .canonicalize()
        .map_err(io_error)?;
    if !path.starts_with(root) {
        return Err(Status::invalid_argument("Task path escapes its root"));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_and_noncanonical_paths() {
        for path in [
            "",
            "../escape",
            "/absolute",
            "a/../b",
            "a//b",
            "a/./b",
            "a/",
        ] {
            assert!(relative_path(path).is_err(), "{path}");
        }
    }

    #[test]
    fn task_requires_prepared_worker_local_metadata() {
        let root = tempfile::tempdir().unwrap();
        let extension = root.path().join("train_tables/task");
        let error = task_path(root.path(), "train_tables/task").unwrap_err();
        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        fs::create_dir_all(&extension).unwrap();
        assert!(task_path(root.path(), "train_tables/task").is_err());
        fs::write(extension.join("index_metadata.json"), b"{}").unwrap();
        assert_eq!(
            task_path(root.path(), "train_tables/task").unwrap(),
            extension
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_linked_extensions_and_sidecar_escapes() {
        let root = tempfile::tempdir().unwrap();
        let extension = root.path().join("train_tables/task");
        fs::create_dir_all(&extension).unwrap();
        fs::write(extension.join("index_metadata.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(&extension, root.path().join("train_tables/linked")).unwrap();
        assert!(task_path(root.path(), "train_tables/linked").is_err());
        fs::write(root.path().join("outside"), b"data").unwrap();
        std::os::unix::fs::symlink(root.path().join("outside"), extension.join("sidecar")).unwrap();
        assert!(resolve_beneath(&extension, "sidecar").is_err());
    }
    #[test]
    fn paths_cannot_escape_the_served_graph() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("task")).unwrap();
        assert!(resolve_beneath(root.path(), "task").is_ok());
        assert!(resolve_beneath(root.path(), "../task").is_err());
        assert!(resolve_beneath(root.path(), "/task").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/", root.path().join("outside")).unwrap();
            assert!(resolve_beneath(root.path(), "outside").is_err());
        }
    }
}
