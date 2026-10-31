// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable task registration and path validation.
use crate::protocol::CHUNK_BYTES;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
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
    Ok(root.join(path))
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

fn files(
    root: &Path,
    directory: &Path,
    output: &mut Vec<(String, PathBuf, u64)>,
) -> Result<(), Status> {
    for entry in fs::read_dir(directory).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let metadata = entry.path().symlink_metadata().map_err(io_error)?;
        if metadata.file_type().is_symlink() {
            return Err(Status::invalid_argument(
                "Task extensions must not contain symlinks",
            ));
        }
        if metadata.is_dir() {
            files(root, &entry.path(), output)?;
        } else if metadata.is_file() {
            let path = entry.path();
            let name = path
                .strip_prefix(root)
                .unwrap()
                .to_str()
                .ok_or_else(|| Status::invalid_argument("Non UTF-8 task path"))?
                .replace(std::path::MAIN_SEPARATOR, "/");
            output.push((name, path, metadata.len()));
        }
    }
    Ok(())
}

pub(crate) fn hash_task(root: &Path) -> Result<String, Status> {
    let mut inventory = Vec::new();
    files(root, root, &mut inventory)?;
    if inventory.is_empty() {
        return Err(Status::invalid_argument("Task extension contains no files"));
    }
    inventory.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; CHUNK_BYTES];
    for (name, path, size) in inventory {
        hash.update(serde_json::to_vec(&(name, size)).unwrap());
        let mut file = fs::File::open(path).map_err(io_error)?;
        loop {
            let count = file.read(&mut buffer).map_err(io_error)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) struct Upload {
    pub destination: PathBuf,
    pub staging: Option<PathBuf>,
    hash: Sha256,
    previous: String,
    remaining: u64,
    file: Option<fs::File>,
}
impl Upload {
    pub fn new(destination: PathBuf, expected: &str) -> Result<Self, Status> {
        let staging = if destination.exists() {
            if hash_task(&destination)? != expected {
                return Err(Status::invalid_argument("Task extension identity mismatch"));
            }
            None
        } else {
            let parent = destination.parent().unwrap();
            fs::create_dir_all(parent).map_err(io_error)?;
            let path = parent.join(format!(".upload-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).map_err(io_error)?;
            Some(path)
        };
        Ok(Self {
            destination,
            staging,
            hash: Sha256::new(),
            previous: String::new(),
            remaining: 0,
            file: None,
        })
    }
    pub fn header(&mut self, name: String, size: u64) -> Result<(), Status> {
        let relative = relative_path(&name)?;
        if self.remaining != 0 || name <= self.previous {
            return Err(Status::invalid_argument(
                "Invalid task file order or length",
            ));
        }
        self.file = None;
        if let Some(root) = &self.staging {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).map_err(io_error)?;
            self.file = Some(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .map_err(io_error)?,
            );
        }
        self.hash
            .update(serde_json::to_vec(&(&name, size)).unwrap());
        self.previous = name;
        self.remaining = size;
        Ok(())
    }
    pub fn data(&mut self, bytes: &[u8]) -> Result<(), Status> {
        if bytes.is_empty() || bytes.len() > CHUNK_BYTES || bytes.len() as u64 > self.remaining {
            return Err(Status::invalid_argument("Invalid task file chunk"));
        }
        if let Some(file) = &mut self.file {
            file.write_all(bytes).map_err(io_error)?;
        }
        self.hash.update(bytes);
        self.remaining -= bytes.len() as u64;
        Ok(())
    }
    pub fn finish(mut self, expected: &str) -> Result<(), Status> {
        if self.previous.is_empty() || self.remaining != 0 {
            return Err(Status::invalid_argument("Incomplete task transfer"));
        }
        if format!("{:x}", self.hash.finalize()) != expected {
            return Err(Status::invalid_argument("Task transfer checksum mismatch"));
        }
        self.file.take();
        if let Some(path) = self.staging {
            fs::rename(path, self.destination).map_err(io_error)?;
        }
        Ok(())
    }
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
    fn upload_publishes_only_a_complete_matching_task() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("data"), b"abc").unwrap();
        let hash = hash_task(&source).unwrap();
        let destination = temporary.path().join("train_tables/task");
        let mut upload = Upload::new(destination.clone(), &hash).unwrap();
        upload.header("data".into(), 3).unwrap();
        upload.data(b"abc").unwrap();
        assert!(!destination.exists());
        upload.finish(&hash).unwrap();
        assert_eq!(hash_task(&destination).unwrap(), hash);
        let mut invalid =
            Upload::new(temporary.path().join("train_tables/invalid"), &hash).unwrap();
        invalid.header("data".into(), 3).unwrap();
        invalid.data(b"ab").unwrap();
        assert!(invalid.finish(&hash).is_err());
        assert!(!temporary.path().join("train_tables/invalid").exists());
    }
}
