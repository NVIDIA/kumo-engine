//! Streaming dataplane mirror for ingest output, plus the manifest and
//! end-of-ingest ordering primitives that callers (and `verify_snapshot_integrity`)
//! rely on.
//!
//! The mirror is the optional secondary write channel attached to
//! [`OutputTarget::Local`]. When set, every output file written during
//! ingest is also `submit_upload`'d to the dataplane via [`CloudTransfer`].
//! Uploads run in parallel with subsequent ingest writes (per-file
//! streaming, not post-ingest batch). At the end of ingest the controller
//! drains all uploads, then writes `_MANIFEST.json` and `_SUCCESS` in that
//! order, both locally and on the dataplane (if mirroring), so consumers
//! can rely on a strict invariant: **`_SUCCESS` is the last byte written;
//! if it exists, every file in `_MANIFEST.json` is fully uploaded.**
//!
//! `_MANIFEST.json` is a separate integrity contract. Its presence and
//! correctness is checked by [`verify_snapshot_integrity`], which the
//! consumer (Python `_ensure_snapshot`) calls after a Python-driven
//! dataplane→local copy. The check catches:
//!   - lifecycle-mid-deletion races on the dataplane (manifest entry
//!     missing or wrong size after Python's parallel copy completes);
//!   - partial copy failures the Python downloader missed;
//!   - operator-induced deletion races between an in-flight ingest and
//!     a stale-snapshot cleanup sweep.
//!
//! Per-file content hashes are deliberately not part of the manifest:
//! the threats above produce missing or truncated objects, not
//! silently-corrupted ones; size-only verification is sufficient and
//! avoids the per-byte CPU cost of hashing GB-scale binary index files
//! during ingest. Add hashes if a corruption class emerges in practice.

use anyhow::{Context, Result};
use object_store::path::Path as ObjectPath;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use diskgraph_fs::{CloudTransfer, CloudTransferTicket};

use crate::io::{build_object_store_for_path, object_path_for_path};
use crate::success_marker::{write_success_marker, SUCCESS_MARKER_FILENAME};

/// Filename for the per-snapshot manifest written by ingest and consumed
/// by `verify_snapshot_integrity`. Lives at the snapshot root, alongside
/// `_SUCCESS` (`SUCCESS_MARKER_FILENAME`, see [`crate::success_marker`]).
pub const MANIFEST_FILENAME: &str = "_MANIFEST.json";

/// Manifest schema version embedded in every `_MANIFEST.json`. Bumped on
/// any breaking change to the manifest shape so `verify_snapshot_integrity`
/// can fail loud-and-fast against unsupported snapshots rather than
/// silently mis-parsing.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// One entry in `_MANIFEST.json`. The list of entries is the closure of
/// every output file ingest writes EXCEPT `_MANIFEST.json` itself and
/// `_SUCCESS`, which are the ordering markers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Path relative to the snapshot root, forward-slash separated. Same
    /// shape used as the local filesystem path (joined onto the local
    /// root) and the upload path (composed onto the dataplane dest).
    pub rel_path: String,
    /// On-disk size of the file in bytes. Compared against
    /// `fs::metadata(local).len()` during integrity verification.
    pub size_bytes: u64,
}

/// `_MANIFEST.json` payload — schema-versioned, sorted-deterministic
/// across runs of the same input so the file diffs cleanly.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub files: Vec<ManifestEntry>,
}

impl Manifest {
    pub fn new(mut files: Vec<ManifestEntry>) -> Self {
        files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            files,
        }
    }
}

/// Optional secondary write channel attached to [`OutputTarget::Local`]. All
/// methods are `&self`; concurrent uploads coordinate via [`CloudTransfer`]'s
/// internal queue. A short-lived [`Mutex`] guards only the ticket vector.
pub struct SnapshotMirror {
    transfer: CloudTransfer,
    /// Path prefix to compose with per-file rel_paths at submit time.
    /// Empty for local backends, `bucket/path/snapshot` for cloud.
    dest_dir: ObjectPath,
    /// Tickets for every non-marker upload submitted during ingest.
    /// Drained by [`Self::wait_all_pending`].
    tickets: Mutex<Vec<CloudTransferTicket>>,
}

impl SnapshotMirror {
    /// Build a mirror rooted at `dataplane_dir`, which may be a cloud URL
    /// (`s3://`, `gs://`, `az://`) or a local-filesystem path. Creates
    /// the local prefix dir if missing (cloud buckets are pre-existing,
    /// so the cloud branch is a no-op for that step).
    pub fn new(dataplane_dir: &str) -> Result<Self> {
        let store = build_object_store_for_path(dataplane_dir)
            .with_context(|| format!("build object store for dataplane dir {dataplane_dir}"))?;
        let dest_dir = object_path_for_path(dataplane_dir)
            .with_context(|| format!("derive dest ObjectPath for {dataplane_dir}"))?;
        let transfer = CloudTransfer::new(store, diskgraph_fs::CloudTransferConfig::from_env());
        Ok(Self {
            transfer,
            dest_dir,
            tickets: Mutex::new(Vec::new()),
        })
    }

    /// Compose `<dest_dir>/<rel_path>` for upload submission. Encapsulated
    /// here so callers don't reach into `dest_dir` directly and can stay
    /// agnostic to whether the dest_dir is empty (local) or non-empty
    /// (cloud).
    fn dest_path(&self, rel_path: &str) -> ObjectPath {
        // ObjectPath::from accepts a slash-separated path; constructing
        // from a fully composed string avoids the join-vs-child API
        // deprecation churn and supports multi-segment rel_paths
        // (e.g., ``"kumo_col_stats/col_stats_full/customers.json"``).
        if self.dest_dir.as_ref().is_empty() {
            ObjectPath::from(rel_path)
        } else {
            ObjectPath::from(format!(
                "{}/{}",
                self.dest_dir.as_ref(),
                rel_path
            ))
        }
    }

    /// Submit an upload of `bytes` to `<dest_dir>/<rel_path>`. The upload
    /// runs in parallel with subsequent ingest work; the returned ticket
    /// is captured for the end-of-ingest drain. Errors at submit time
    /// (queue closed) propagate immediately.
    pub fn submit(&self, rel_path: &str, bytes: Vec<u8>) -> Result<()> {
        let ticket = self.transfer.submit_upload(self.dest_path(rel_path), bytes)?;
        self.tickets.lock().push(ticket);
        Ok(())
    }

    /// Like [`Self::submit`], but sources the upload bytes from a
    /// local file instead of an in-memory `Vec`. Used by tee'd sinks
    /// after their local file has been flushed, and by the
    /// post-`build_index` reconciliation sweep — both already have
    /// the file durable on disk and would otherwise have to read it
    /// into memory just to hand it off. Routing through
    /// `submit_upload_from_path` bounds peak RSS to
    /// `part_size × inflight_parts` per CloudTransfer.
    pub fn submit_from_path(&self, rel_path: &str, src_path: &std::path::Path) -> Result<()> {
        let ticket = self
            .transfer
            .submit_upload_from_path(self.dest_path(rel_path), src_path.to_path_buf())?;
        self.tickets.lock().push(ticket);
        Ok(())
    }

    /// Wait for every pending non-marker upload to complete. Returns the
    /// first failure if any; remaining tickets are still waited on so a
    /// transient error doesn't leak in-flight bytes.
    pub fn wait_all_pending(&self) -> Result<()> {
        let tickets = std::mem::take(&mut *self.tickets.lock());
        let mut first_err: Option<anyhow::Error> = None;
        for ticket in tickets {
            if let Err(e) = ticket.wait() {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(err) = first_err {
            return Err(err.context("wait_all_pending: at least one upload failed"));
        }
        Ok(())
    }

    /// Synchronous upload + wait. Used for the manifest and `_SUCCESS`
    /// markers, which the controller writes strictly after every other
    /// upload has drained — so the natural API is "submit and wait now".
    pub fn upload_blocking(&self, rel_path: &str, bytes: Vec<u8>) -> Result<()> {
        self.transfer
            .submit_upload(self.dest_path(rel_path), bytes)?
            .wait()
            .with_context(|| format!("upload_blocking: failed uploading {rel_path}"))
    }
}

impl std::fmt::Debug for SnapshotMirror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotMirror")
            .field("dest_dir", &self.dest_dir)
            .field("pending_tickets", &self.tickets.lock().len())
            .finish()
    }
}

/// Manifest accumulator threaded through every output write site.
///
/// Pushes are O(1) under a short-lived lock; the mutex contention is
/// negligible because writes happen on a single ingest thread and the
/// hot path is well-amortized by per-file IO. Read-out at end of ingest
/// is a single `take` followed by [`Manifest::new`] which sorts.
#[derive(Default)]
pub struct ManifestAccumulator {
    entries: Mutex<Vec<ManifestEntry>>,
}

impl ManifestAccumulator {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    pub fn record(&self, rel_path: &str, size_bytes: u64) {
        self.entries.lock().push(ManifestEntry {
            rel_path: rel_path.to_string(),
            size_bytes,
        });
    }

    pub fn take(&self) -> Vec<ManifestEntry> {
        std::mem::take(&mut *self.entries.lock())
    }

    /// Snapshot of currently-recorded rel_paths. Used by the post-build
    /// reconciliation sweep to skip files that already went through
    /// [`OutputTarget::write_bytes`] / [`OutputTarget::sink`].
    pub fn known_paths(&self) -> std::collections::HashSet<String> {
        self.entries
            .lock()
            .iter()
            .map(|e| e.rel_path.clone())
            .collect()
    }
}

impl std::fmt::Debug for ManifestAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifestAccumulator")
            .field("entries", &self.entries.lock().len())
            .finish()
    }
}

/// Result of [`finalize_snapshot`]: the manifest written, plus the local
/// paths of the two markers (handy for callers that want to assert
/// post-conditions in tests).
#[derive(Debug)]
pub struct FinalizeResult {
    pub manifest: Manifest,
    pub manifest_local_path: std::path::PathBuf,
    pub success_local_path: std::path::PathBuf,
}

/// Drain pending uploads, then write `_MANIFEST.json` and `_SUCCESS` in
/// strict order — locally always, on the dataplane mirror if present.
///
/// The strict ordering — wait-all-pending → manifest → success — is the
/// invariant `verify_snapshot_integrity` and the consumer-side restore
/// path rely on. If `_SUCCESS` exists at a snapshot root, every file in
/// `_MANIFEST.json` was already durably written on the same backend.
///
/// **Storage invariant: L1 ⇒ L2.** When a mirror is configured, ingest
/// success means the snapshot is durable on BOTH backends. A drain or
/// upload failure here propagates as `Err`, the caller's
/// `cleanup_guard` rmtrees L1, and ingest fails — by design. The
/// alternative (preserve L1, leave L2 partial) would force every
/// downstream consumer (eviction policy, restore path, pool-state
/// model) to track per-snapshot "L2 status" as a separate axis,
/// fragmenting the storage layer. CloudTransfer's per-file retry
/// (`DISKGRAPH_CLOUD_FILE_RETRY_MAX`, default 2) absorbs transient
/// blips; a sustained S3 outage failing the whole ingest is the
/// correct outcome, not a regression — the pool can't operate without
/// L2 regardless.
///
/// `success_marker_content` is the JSON payload Python builds (the
/// `DiskgraphIngestResult` shape from `kumo.engine.diskgraph.success_marker`).
/// Local write goes through [`write_success_marker`] which uses tmp+rename
/// for atomicity. The dataplane upload uses the same bytes.
pub fn finalize_snapshot(
    local_root: &std::path::Path,
    mirror: Option<&SnapshotMirror>,
    accumulator: &ManifestAccumulator,
    success_marker_content: &str,
) -> Result<FinalizeResult> {
    if let Some(m) = mirror {
        m.wait_all_pending()
            .context("finalize_snapshot: drain pending uploads")?;
    }

    let manifest = Manifest::new(accumulator.take());
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("serialize _MANIFEST.json")?;
    let manifest_local = local_root.join(MANIFEST_FILENAME);
    std::fs::write(&manifest_local, &manifest_bytes)
        .with_context(|| format!("write {}", manifest_local.display()))?;
    if let Some(m) = mirror {
        m.upload_blocking(MANIFEST_FILENAME, manifest_bytes)
            .context("finalize_snapshot: upload _MANIFEST.json")?;
    }

    write_success_marker(local_root, success_marker_content)
        .with_context(|| format!("write _SUCCESS at {}", local_root.display()))?;
    let success_local = local_root.join(SUCCESS_MARKER_FILENAME);
    if let Some(m) = mirror {
        m.upload_blocking(SUCCESS_MARKER_FILENAME, success_marker_content.as_bytes().to_vec())
            .context("finalize_snapshot: upload _SUCCESS")?;
    }

    Ok(FinalizeResult {
        manifest,
        manifest_local_path: manifest_local,
        success_local_path: success_local,
    })
}

/// Reasons a snapshot fails the post-restore integrity check.
///
/// Distinct from `DiskgraphDataError` (user data problem) and
/// `DiskgraphSystemError` (infra fault) because the caller's response
/// differs: corruption is unexpected and warrants alerting + falling
/// through to a fresh ingest, while data errors propagate to the user.
#[derive(Debug, thiserror::Error)]
pub enum CorruptionError {
    #[error("missing {SUCCESS_MARKER_FILENAME} at {0}")]
    MissingSuccess(std::path::PathBuf),

    #[error("missing {MANIFEST_FILENAME} at {0}")]
    MissingManifest(std::path::PathBuf),

    #[error("invalid {MANIFEST_FILENAME} at {path}: {detail}")]
    InvalidManifest {
        path: std::path::PathBuf,
        detail: String,
    },

    #[error(
        "unsupported manifest schema version at {path}: file={file_version} controller={controller_version}"
    )]
    UnsupportedManifestVersion {
        path: std::path::PathBuf,
        file_version: u32,
        controller_version: u32,
    },

    #[error(
        "manifest mismatch at {root}: {missing_count} missing file(s), {size_mismatch_count} size mismatch(es)"
    )]
    ManifestMismatch {
        root: std::path::PathBuf,
        missing: Vec<String>,
        size_mismatches: Vec<(String, u64, u64)>, // (rel_path, expected, actual)
        missing_count: usize,
        size_mismatch_count: usize,
    },

    #[error("io error verifying {path}: {detail}")]
    Io {
        path: std::path::PathBuf,
        detail: String,
    },

    #[error(
        "manifest at {manifest_path} contains rejected rel_path {rel_path:?}: {reason}"
    )]
    UnsafeManifestEntry {
        manifest_path: std::path::PathBuf,
        rel_path: String,
        reason: String,
    },
}

/// Validate a `_MANIFEST.json` rel_path before joining it under the
/// snapshot root. The manifest is read from dataplane content
/// (potentially tampered, partially corrupted, or written by a future
/// schema version), so we treat rel_paths as untrusted input. Rejecting
/// shapes that escape or alias the snapshot root protects
/// `verify_snapshot_integrity` from being tricked into stat'ing files
/// outside `output_dir` and matches the boundary
/// `OutputTarget::write_bytes` / `OutputTarget::sink` enforce on the
/// producer side (where rel_paths are always normalized via
/// `path.strip_prefix(root)`).
///
/// Rules: rel_path must be non-empty, relative (no leading `/`, no
/// drive letter), use forward slashes, and contain only "normal"
/// components (no `..`, no `.`, no empty segments from `a//b`). Symlink
/// safety is out of scope — `verify_snapshot_integrity` only stats
/// metadata; symlink-following is a downstream consumer's
/// responsibility if any.
pub(crate) fn validate_manifest_rel_path(rel: &str) -> Result<(), &'static str> {
    if rel.is_empty() {
        return Err("empty");
    }
    if rel.starts_with('/') {
        return Err("absolute (leading `/`)");
    }
    // Reject drive-letter Windows roots (`C:\…`) defensively even though
    // we don't run on Windows — keeps the boundary clean if that ever
    // changes, and a Windows-style path leaking into a Unix manifest is
    // suspicious data on its own.
    if rel.contains(':') {
        return Err("contains `:` (windows drive letter or stream)");
    }
    if rel.contains('\\') {
        return Err("contains `\\` (windows separator)");
    }
    for segment in rel.split('/') {
        if segment.is_empty() {
            return Err("empty path segment (consecutive `/`)");
        }
        if segment == ".." {
            return Err("contains `..` segment");
        }
        if segment == "." {
            return Err("contains `.` segment");
        }
    }
    Ok(())
}

/// Verify the snapshot rooted at `output_dir` against its `_MANIFEST.json`.
/// Used by the consumer side after a Python-driven dataplane→local copy
/// completes, to catch lifecycle-mid-delete races and partial copies
/// before the snapshot is handed to a workload.
///
/// Pass on success; [`CorruptionError`] on any deviation from the
/// manifest contract. Errors are precise about what failed so the
/// caller's alerting carries actionable detail (which file is missing,
/// expected vs actual size, etc.).
pub fn verify_snapshot_integrity(output_dir: &std::path::Path) -> std::result::Result<(), CorruptionError> {
    let success_path = output_dir.join(SUCCESS_MARKER_FILENAME);
    if !success_path.exists() {
        return Err(CorruptionError::MissingSuccess(success_path));
    }

    let manifest_path = output_dir.join(MANIFEST_FILENAME);
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(CorruptionError::MissingManifest(manifest_path));
        }
        Err(e) => {
            return Err(CorruptionError::Io {
                path: manifest_path,
                detail: e.to_string(),
            });
        }
    };
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| {
        CorruptionError::InvalidManifest {
            path: manifest_path.clone(),
            detail: e.to_string(),
        }
    })?;
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(CorruptionError::UnsupportedManifestVersion {
            path: manifest_path,
            file_version: manifest.schema_version,
            controller_version: MANIFEST_SCHEMA_VERSION,
        });
    }

    // Validate every rel_path against the snapshot-root boundary
    // BEFORE the file-existence loop. A malformed entry (absolute
    // path, `..` traversal, etc.) would otherwise let
    // `output_dir.join(rel)` return a path outside `output_dir`, and
    // `fs::metadata` would happily stat it — passing integrity for a
    // file the snapshot doesn't actually own. Reject the manifest
    // outright on first offender; a single bad entry means the
    // manifest can't be trusted and the caller should fall through to
    // a fresh ingest.
    for entry in &manifest.files {
        if let Err(reason) = validate_manifest_rel_path(&entry.rel_path) {
            return Err(CorruptionError::UnsafeManifestEntry {
                manifest_path: manifest_path.clone(),
                rel_path: entry.rel_path.clone(),
                reason: reason.to_string(),
            });
        }
    }

    let mut missing: Vec<String> = Vec::new();
    let mut size_mismatches: Vec<(String, u64, u64)> = Vec::new();
    for entry in &manifest.files {
        let path = output_dir.join(&entry.rel_path);
        match std::fs::metadata(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                missing.push(entry.rel_path.clone());
            }
            Err(e) => {
                return Err(CorruptionError::Io {
                    path,
                    detail: e.to_string(),
                });
            }
            Ok(meta) => {
                let actual = meta.len();
                if actual != entry.size_bytes {
                    size_mismatches.push((entry.rel_path.clone(), entry.size_bytes, actual));
                }
            }
        }
    }
    if !missing.is_empty() || !size_mismatches.is_empty() {
        return Err(CorruptionError::ManifestMismatch {
            root: output_dir.to_path_buf(),
            missing_count: missing.len(),
            size_mismatch_count: size_mismatches.len(),
            missing,
            size_mismatches,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn manifest_round_trip() {
        let m = Manifest::new(vec![
            ManifestEntry {
                rel_path: "b.json".into(),
                size_bytes: 2,
            },
            ManifestEntry {
                rel_path: "a.bin".into(),
                size_bytes: 1,
            },
        ]);
        // Sorted on construction for stable output.
        assert_eq!(m.files[0].rel_path, "a.bin");
        assert_eq!(m.files[1].rel_path, "b.json");
        let bytes = serde_json::to_vec(&m).unwrap();
        let parsed: Manifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    fn write_marker_files(root: &std::path::Path, files: &[(&str, &[u8])]) {
        for (rel, bytes) in files {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(bytes).unwrap();
        }
    }

    fn write_manifest(root: &std::path::Path, entries: Vec<ManifestEntry>) {
        let manifest = Manifest::new(entries);
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        std::fs::write(root.join(MANIFEST_FILENAME), bytes).unwrap();
    }

    #[test]
    fn verify_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(
            tmp.path(),
            &[
                (SUCCESS_MARKER_FILENAME, b"{}"),
                ("a.json", b"hello"),
                ("nested/b.bin", b"world!"),
            ],
        );
        write_manifest(
            tmp.path(),
            vec![
                ManifestEntry {
                    rel_path: "a.json".into(),
                    size_bytes: 5,
                },
                ManifestEntry {
                    rel_path: "nested/b.bin".into(),
                    size_bytes: 6,
                },
            ],
        );
        verify_snapshot_integrity(tmp.path()).expect("happy-path verify must pass");
    }

    #[test]
    fn verify_missing_success() {
        let tmp = tempfile::tempdir().unwrap();
        write_manifest(tmp.path(), vec![]);
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        assert!(matches!(err, CorruptionError::MissingSuccess(_)));
    }

    #[test]
    fn verify_missing_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        assert!(matches!(err, CorruptionError::MissingManifest(_)));
    }

    #[test]
    fn verify_invalid_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        std::fs::write(tmp.path().join(MANIFEST_FILENAME), b"{ not json").unwrap();
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        assert!(matches!(err, CorruptionError::InvalidManifest { .. }));
    }

    #[test]
    fn verify_unsupported_schema_version() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        let bad = serde_json::json!({
            "schema_version": 999,
            "files": [],
        });
        std::fs::write(
            tmp.path().join(MANIFEST_FILENAME),
            serde_json::to_vec_pretty(&bad).unwrap(),
        )
        .unwrap();
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            CorruptionError::UnsupportedManifestVersion { .. }
        ));
    }

    #[test]
    fn verify_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        write_manifest(
            tmp.path(),
            vec![ManifestEntry {
                rel_path: "missing.json".into(),
                size_bytes: 5,
            }],
        );
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        match err {
            CorruptionError::ManifestMismatch {
                missing,
                size_mismatch_count,
                ..
            } => {
                assert_eq!(missing, vec!["missing.json".to_string()]);
                assert_eq!(size_mismatch_count, 0);
            }
            other => panic!("expected ManifestMismatch, got {other:?}"),
        }
    }

    /// Path validator unit tests — covers every shape we reject so a
    /// future "be more permissive" change has to update the rejection
    /// surface deliberately.
    #[test]
    fn validate_manifest_rel_path_accepts_normal() {
        for ok in [
            "a.json",
            "nested/b.bin",
            "deeply/nested/path/with-many-parts/f.dat",
            "nodes/users/shard-00000/offsets.bin",
            "_MANIFEST.json", // (markers themselves wouldn't be in the
                              // file list, but the validator doesn't
                              // discriminate by name — only by shape)
        ] {
            assert!(
                validate_manifest_rel_path(ok).is_ok(),
                "expected {ok:?} to be accepted"
            );
        }
    }

    #[test]
    fn validate_manifest_rel_path_rejects_empty() {
        assert!(validate_manifest_rel_path("").is_err());
    }

    #[test]
    fn validate_manifest_rel_path_rejects_absolute() {
        assert!(validate_manifest_rel_path("/etc/passwd").is_err());
        assert!(validate_manifest_rel_path("/").is_err());
    }

    #[test]
    fn validate_manifest_rel_path_rejects_parent_dir() {
        assert!(validate_manifest_rel_path("../escape").is_err());
        assert!(validate_manifest_rel_path("nested/../escape").is_err());
        assert!(validate_manifest_rel_path("a/b/..").is_err());
        // Prefix-only check would let `..foo` through; segment-based
        // check is what we actually want.
        assert!(
            validate_manifest_rel_path("..foo").is_ok(),
            "literal `..foo` filename is fine — only the `..` segment alone is rejected"
        );
    }

    #[test]
    fn validate_manifest_rel_path_rejects_current_dir() {
        assert!(validate_manifest_rel_path("./relative").is_err());
        assert!(validate_manifest_rel_path("a/./b").is_err());
    }

    #[test]
    fn validate_manifest_rel_path_rejects_empty_segments() {
        assert!(validate_manifest_rel_path("a//b").is_err());
        assert!(validate_manifest_rel_path("a/").is_err());
    }

    #[test]
    fn validate_manifest_rel_path_rejects_windows_shapes() {
        assert!(validate_manifest_rel_path("C:\\foo").is_err());
        assert!(validate_manifest_rel_path("C:/foo").is_err());
        assert!(validate_manifest_rel_path("a\\b").is_err());
    }

    /// Verify the full integration: a tampered manifest with an
    /// `..`-bearing rel_path is rejected by `verify_snapshot_integrity`
    /// BEFORE the file-existence check fires, so an attacker can't
    /// trick verify into stat'ing files outside the snapshot root.
    #[test]
    fn verify_rejects_unsafe_manifest_entry() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        // Hand-roll a manifest that the OutputTarget producer-side
        // path would never emit — strip_prefix(root) normalizes
        // everything to plain rel paths. This simulates a tampered
        // dataplane mirror.
        write_manifest(
            tmp.path(),
            vec![ManifestEntry {
                rel_path: "../escaped".into(),
                size_bytes: 0,
            }],
        );
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        match err {
            CorruptionError::UnsafeManifestEntry {
                rel_path, reason, ..
            } => {
                assert_eq!(rel_path, "../escaped");
                assert!(
                    reason.contains(".."),
                    "reason should name the offending segment shape; got: {reason}"
                );
            }
            other => panic!("expected UnsafeManifestEntry, got {other:?}"),
        }
    }

    #[test]
    fn verify_rejects_absolute_manifest_entry() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(tmp.path(), &[(SUCCESS_MARKER_FILENAME, b"{}")]);
        write_manifest(
            tmp.path(),
            vec![ManifestEntry {
                rel_path: "/etc/passwd".into(),
                size_bytes: 0,
            }],
        );
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        assert!(matches!(err, CorruptionError::UnsafeManifestEntry { .. }));
    }

    #[test]
    fn verify_size_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker_files(
            tmp.path(),
            &[(SUCCESS_MARKER_FILENAME, b"{}"), ("data.bin", b"actual")],
        );
        write_manifest(
            tmp.path(),
            vec![ManifestEntry {
                rel_path: "data.bin".into(),
                size_bytes: 999, // claim != reality
            }],
        );
        let err = verify_snapshot_integrity(tmp.path()).unwrap_err();
        match err {
            CorruptionError::ManifestMismatch {
                size_mismatches, ..
            } => {
                assert_eq!(size_mismatches.len(), 1);
                let (rel, expected, actual) = &size_mismatches[0];
                assert_eq!(rel, "data.bin");
                assert_eq!(*expected, 999);
                assert_eq!(*actual, 6);
            }
            other => panic!("expected ManifestMismatch, got {other:?}"),
        }
    }

    /// Mirror end-to-end against a local-FS dataplane: submit upload,
    /// drain, then upload+wait the markers — all roundtrip back.
    #[test]
    fn mirror_finalize_round_trip() -> anyhow::Result<()> {
        let local_root = tempfile::tempdir()?;
        let dataplane_root = tempfile::tempdir()?;
        let dataplane_dir = dataplane_root.path().to_str().unwrap();
        let mirror = SnapshotMirror::new(dataplane_dir)?;
        let accumulator = ManifestAccumulator::new();

        // Simulate two output writes that ingest would mirror.
        std::fs::write(local_root.path().join("a.json"), b"hello")?;
        accumulator.record("a.json", 5);
        mirror.submit("a.json", b"hello".to_vec())?;
        std::fs::create_dir_all(local_root.path().join("dir"))?;
        std::fs::write(local_root.path().join("dir/b.bin"), b"world!")?;
        accumulator.record("dir/b.bin", 6);
        mirror.submit("dir/b.bin", b"world!".to_vec())?;

        let result = finalize_snapshot(
            local_root.path(),
            Some(&mirror),
            &accumulator,
            "{\"local\":\"ok\"}",
        )?;
        assert_eq!(result.manifest.files.len(), 2);
        assert!(result.manifest_local_path.exists());
        assert!(result.success_local_path.exists());

        // Dataplane received every file + both markers, and the local
        // snapshot passes integrity verification.
        for rel in ["a.json", "dir/b.bin", MANIFEST_FILENAME, SUCCESS_MARKER_FILENAME] {
            let dp = dataplane_root.path().join(rel);
            assert!(dp.exists(), "{} missing on dataplane", rel);
        }
        verify_snapshot_integrity(local_root.path())?;
        Ok(())
    }

    /// finalize_snapshot still runs locally with no mirror — the
    /// invariant is that `_MANIFEST.json` + `_SUCCESS` always exist
    /// locally on a successful ingest, mirror or not.
    #[test]
    fn finalize_without_mirror_writes_local_markers() -> anyhow::Result<()> {
        let local_root = tempfile::tempdir()?;
        let accumulator = ManifestAccumulator::new();
        std::fs::write(local_root.path().join("a.json"), b"hello")?;
        accumulator.record("a.json", 5);
        finalize_snapshot(local_root.path(), None, &accumulator, "{}")?;
        assert!(local_root.path().join(MANIFEST_FILENAME).exists());
        assert!(local_root.path().join(SUCCESS_MARKER_FILENAME).exists());
        verify_snapshot_integrity(local_root.path())?;
        Ok(())
    }
}
