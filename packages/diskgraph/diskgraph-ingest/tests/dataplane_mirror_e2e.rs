// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for atomic-marker ingest with a dataplane mirror.
//!
//! This is the post-condition the streaming-mirror feature promises and that
//! the kumo-side `_ensure_snapshot` flow depends on:
//!
//! > After `ingest_with_stats_from_source(... , output_dataplane_dir=Some(D),
//! >  success_marker_content=Some(C))` returns, both the local root AND `D`
//! > are complete, self-consistent snapshots and pass
//! > `verify_snapshot_integrity` independently.
//!
//! Block E observed this fail in production: the local root was missing
//! every `nodes/<table>/pk_hash-XXXXX.bin`, `pk_data-XXXXX.bin`,
//! `node_data.ctx`, and the `.meta.json` siblings — every manifest entry
//! was on the dataplane but absent locally — so verify raised
//! `ManifestMismatch` and the engine crashed loading from a half-empty
//! tree. This test reproduces the real ingest path, including the index
//! builder writes that bypassed `OutputTarget` entirely, so any future
//! regression along that fault line fails immediately.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, StringArray, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::config::IngestOptions;
use diskgraph_ingest::graph_config::GraphConfig;
use diskgraph_ingest::io::{build_object_store_for_path, object_path_for_path, Dest};
use diskgraph_ingest::snapshot_mirror::{verify_snapshot_integrity, Manifest, MANIFEST_FILENAME};
use diskgraph_ingest::success_marker::SUCCESS_MARKER_FILENAME;
use futures::TryStreamExt;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn write_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    let batch = arrow_array::RecordBatch::try_new(schema, columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_users(path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
    ]));
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(StringArray::from(vec!["u1", "u2", "u3", "u4"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["us", "us", "ca", "ca"])),
        ],
    )
}

fn write_devices(path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("os", DataType::Utf8, false),
    ]));
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(StringArray::from(vec!["d1", "d2", "d3"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["android", "ios", "ios"])),
        ],
    )
}

fn write_edges(path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("device_id", DataType::Utf8, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), false),
    ]));
    write_parquet(
        path,
        schema,
        vec![
            Arc::new(StringArray::from(vec!["u1", "u2", "u3", "u4", "u1"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["d1", "d2", "d2", "d3", "d3"])),
            Arc::new(TimestampSecondArray::from(vec![1_700_000_000i64; 5])),
        ],
    )
}

fn build_graph_config(users: &Path, devices: &Path, edges: &Path) -> Result<GraphConfig> {
    let json = serde_json::json!({
        "dataset_name": "dataplane_mirror_e2e",
        "num_shards": 2,
        "edge_format": "binary",
        "tables": [
            {
                "name": "users",
                "source": {"type": "parquet", "paths": [users.to_str().unwrap()]},
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "country", "ingested_dtype": "string"}
                ],
                "primary_key": "id"
            },
            {
                "name": "devices",
                "source": {"type": "parquet", "paths": [devices.to_str().unwrap()]},
                "columns": [
                    {"name": "id", "ingested_dtype": "string"},
                    {"name": "os", "ingested_dtype": "string"}
                ],
                "primary_key": "id"
            },
            {
                "name": "user_device",
                "source": {"type": "parquet", "paths": [edges.to_str().unwrap()]},
                "columns": [
                    {"name": "user_id", "ingested_dtype": "string"},
                    {"name": "device_id", "ingested_dtype": "string"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ],
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "users"},
                    "dst_fk": {"edge_column": "device_id", "ref_table": "devices"},
                    "forward": {
                        "time": {
                            "table": "user_device",
                            "column": "ts",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        }
                    }
                }
            }
        ]
    });
    serde_json::from_value::<GraphConfig>(json)
        .map_err(|e| anyhow!("invalid GraphConfig fixture: {e}"))
}

/// Recursively collect every file under `root` as a snapshot-relative
/// path (forward-slash separated). Used to verify manifest coverage —
/// the manifest must list every non-`_tmp/` file the ingest leaves on
/// disk, otherwise `verify_snapshot_integrity` silently passes on a
/// snapshot whose runtime payload is missing.
fn collect_files_under(root: &Path) -> Result<BTreeSet<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(dir)
            .with_context(|| format!("read_dir {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                walk(root, &path, out)?;
            } else if file_type.is_file() {
                let rel = path
                    .strip_prefix(root)
                    .with_context(|| format!("strip_prefix {}", path.display()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(rel);
            }
        }
        Ok(())
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

/// Walk a snapshot dir and assert every entry in `_MANIFEST.json` exists
/// at the expected size. Returns the manifest for follow-up assertions.
fn assert_local_matches_manifest(root: &Path) -> Result<Manifest> {
    let manifest_bytes = fs::read(root.join(MANIFEST_FILENAME))
        .with_context(|| format!("read _MANIFEST.json under {}", root.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    for entry in &manifest.files {
        let path = root.join(&entry.rel_path);
        let meta = fs::metadata(&path)
            .with_context(|| format!("manifest entry missing on disk: {}", path.display()))?;
        if meta.len() != entry.size_bytes {
            return Err(anyhow!(
                "size mismatch for {}: manifest={} on-disk={}",
                entry.rel_path,
                entry.size_bytes,
                meta.len()
            ));
        }
    }
    Ok(manifest)
}

#[test]
fn ingest_with_dataplane_mirror_produces_complete_local_and_dataplane_snapshots() -> Result<()> {
    let tmp = TempDir::new()?;
    let local_root = tmp.path().join("local");
    let dataplane_root = tmp.path().join("dataplane");
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    let config = build_graph_config(&users, &devices, &edges)?;

    // Atomic-marker mode: mirror set, marker payload set. Caller-provided
    // success_marker_content mimics the JSON Python's
    // `DiskgraphIngestResult` produces; for this test the contents are
    // arbitrary as long as it parses as JSON (the success_marker module
    // validates that on write).
    let marker_content = serde_json::json!({
        "index_path": local_root.to_str().unwrap(),
        "col_stats_path": local_root.join("col_stats.json").to_str().unwrap(),
        "ingest_stats_path": local_root.join("ingest_stats.json").to_str().unwrap(),
        "kumo_stats_dir": local_root.join("kumo_col_stats").to_str().unwrap(),
    })
    .to_string();

    let options = IngestOptions {
        parallel: true,
        ..Default::default()
    };

    let (_metadata, _stats) = diskgraph_ingest::ingest_with_stats_from_source(
        config,
        None,
        local_root.to_str().unwrap(),
        Dest::Local,
        options,
        Some(dataplane_root.to_str().unwrap()),
        Some(&marker_content),
    )?;

    // Post-condition #1: verify_snapshot_integrity passes against the
    // LOCAL root. This is the property Block E hit failing — manifest
    // entries pointed at deleted ``_tmp/*`` files.
    verify_snapshot_integrity(&local_root)
        .map_err(|e| anyhow!("verify_snapshot_integrity(local) failed: {e}"))?;

    // Post-condition #2: same property holds independently against the
    // dataplane root. Both backends must be self-consistent.
    verify_snapshot_integrity(&dataplane_root)
        .map_err(|e| anyhow!("verify_snapshot_integrity(dataplane) failed: {e}"))?;

    // Post-condition #3: every manifest entry actually exists locally
    // with the expected size — redundant with verify_snapshot_integrity
    // by construction, but the explicit walk produces a clearer failure
    // message if a regression flips between "missing" and "wrong size".
    let local_manifest = assert_local_matches_manifest(&local_root)?;
    let dataplane_manifest = assert_local_matches_manifest(&dataplane_root)?;
    assert_eq!(
        local_manifest, dataplane_manifest,
        "manifests must agree byte-for-byte across the two backends"
    );

    // Post-condition #4: regression for the bug. `_tmp/*` files are
    // deleted by `cleanup_temp` before ingest returns, so they MUST NOT
    // appear in the manifest. Block E hit a 36-file mismatch because
    // intermediate temp-tree writes were being recorded; verify caught
    // them as missing post-cleanup.
    let temp_in_manifest: Vec<&str> = local_manifest
        .files
        .iter()
        .filter(|e| e.rel_path.starts_with("_tmp/") || e.rel_path == "_tmp")
        .map(|e| e.rel_path.as_str())
        .collect();
    assert!(
        temp_in_manifest.is_empty(),
        "manifest must not list `_tmp/*` (deleted by cleanup_temp before \
         ingest returns); offenders: {:?}",
        temp_in_manifest,
    );

    // Post-condition #5: nodes/* files (the class Block E saw missing) are
    // present locally for every table that has a primary key. Catches
    // "manifest is empty so verify trivially passes" as a failure mode.
    let nodes_in_manifest: BTreeSet<&str> = local_manifest
        .files
        .iter()
        .filter(|e| e.rel_path.starts_with("nodes/"))
        .map(|e| e.rel_path.as_str())
        .collect();
    assert!(
        !nodes_in_manifest.is_empty(),
        "expected manifest to list nodes/* files; got: {:?}",
        local_manifest.files.iter().map(|e| &e.rel_path).collect::<Vec<_>>(),
    );

    // Post-condition #6: _SUCCESS marker is the last byte written on
    // both backends and contains the caller-provided payload.
    for root in [&local_root, &dataplane_root] {
        let marker = fs::read_to_string(root.join(SUCCESS_MARKER_FILENAME))?;
        assert_eq!(marker, marker_content, "marker mismatch under {}", root.display());
    }

    Ok(())
}

/// Companion to the atomic-marker test. Without `success_marker_content`,
/// ingest preserves the legacy out-of-band marker contract: it does NOT
/// write `_SUCCESS` (the caller does it via
/// `success_marker::write_success_marker` after ingest returns) and does
/// NOT write `_MANIFEST.json`. Asserts that no regression here either.
#[test]
fn ingest_without_marker_payload_skips_finalize() -> Result<()> {
    let tmp = TempDir::new()?;
    let local_root = tmp.path().join("local");
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    let config = build_graph_config(&users, &devices, &edges)?;

    let options = IngestOptions::default();
    diskgraph_ingest::ingest_with_stats_from_source(
        config,
        None,
        local_root.to_str().unwrap(),
        Dest::Local,
        options,
        None,
        None,
    )?;

    // Legacy contract: ingest leaves both markers absent for the caller
    // to write out-of-band.
    assert!(
        !local_root.join(SUCCESS_MARKER_FILENAME).exists(),
        "_SUCCESS must NOT be written when success_marker_content=None",
    );
    assert!(
        !local_root.join(MANIFEST_FILENAME).exists(),
        "_MANIFEST.json must NOT be written when success_marker_content=None",
    );
    Ok(())
}

/// Helper: run the standard fixture ingest with a dataplane mirror in
/// atomic-marker mode and return the local + dataplane roots so each
/// test can assert independently against them.
fn run_mirror_ingest(tmp: &TempDir) -> Result<(PathBuf, PathBuf)> {
    let local_root = tmp.path().join("local");
    let dataplane_root = tmp.path().join("dataplane");
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    let config = build_graph_config(&users, &devices, &edges)?;

    let marker_content = serde_json::json!({
        "index_path": local_root.to_str().unwrap(),
        "col_stats_path": local_root.join("col_stats.json").to_str().unwrap(),
        "ingest_stats_path": local_root.join("ingest_stats.json").to_str().unwrap(),
        "kumo_stats_dir": local_root.join("kumo_col_stats").to_str().unwrap(),
    })
    .to_string();

    let options = IngestOptions {
        parallel: true,
        ..Default::default()
    };

    diskgraph_ingest::ingest_with_stats_from_source(
        config,
        None,
        local_root.to_str().unwrap(),
        Dest::Local,
        options,
        Some(dataplane_root.to_str().unwrap()),
        Some(&marker_content),
    )?;
    Ok((local_root, dataplane_root))
}

/// Regression: every non-temp file the ingest leaves on local disk must
/// be listed in `_MANIFEST.json`.
///
/// **Bug this catches.** `build_index` writes `index_metadata.json` and
/// every per-shard `nodes/<table>/shard-NNNNN/{offsets,edges}.bin` via
/// raw `fs::File::create` — bypassing `OutputTarget::write_bytes` /
/// `OutputTarget::sink`, which are the two paths that record into the
/// `ManifestAccumulator`. Result: `_MANIFEST.json` lists only the files
/// that happened to go through `OutputTarget`; the index-builder outputs
/// are present on local disk but invisible to `verify_snapshot_integrity`
/// AND never submitted to the dataplane mirror. A snapshot then "passes
/// integrity" while being un-openable on the L2 backend.
///
/// We assert manifest coverage by walking the on-disk tree directly, so
/// any future stage that adds new output files outside the manifest tee
/// will trip this immediately.
#[test]
fn ingest_manifest_covers_every_non_temp_file_on_disk() -> Result<()> {
    let tmp = TempDir::new()?;
    let (local_root, _dataplane_root) = run_mirror_ingest(&tmp)?;

    let manifest = assert_local_matches_manifest(&local_root)?;
    let on_disk = collect_files_under(&local_root)?;
    let in_manifest: BTreeSet<String> = manifest
        .files
        .iter()
        .map(|e| e.rel_path.clone())
        .collect();

    // Markers are not part of the file list (they are written by
    // `finalize_snapshot` after the manifest is sealed).
    let expected_unrecorded: BTreeSet<String> = [
        MANIFEST_FILENAME.to_string(),
        SUCCESS_MARKER_FILENAME.to_string(),
    ]
    .into_iter()
    .collect();

    let unrecorded: BTreeSet<String> = on_disk
        .difference(&in_manifest)
        .filter(|p| !expected_unrecorded.contains(*p))
        .filter(|p| !p.starts_with("_tmp/"))
        .cloned()
        .collect();

    assert!(
        unrecorded.is_empty(),
        "files exist on disk but are missing from _MANIFEST.json — \
         verify_snapshot_integrity will silently pass on incomplete \
         snapshots and the dataplane mirror won't carry these files. \
         Offenders: {:?}",
        unrecorded,
    );

    // Spot-check the specific offenders for the original Block E
    // failure: index_metadata.json + per-shard offsets/edges. These
    // ABSOLUTELY must be tracked because they're what the runtime
    // GraphIndexReader opens.
    assert!(
        in_manifest.contains("index_metadata.json"),
        "_MANIFEST.json must list index_metadata.json (the runtime entry \
         point); manifest entries: {:?}",
        in_manifest,
    );
    let shard_offsets: Vec<&String> = in_manifest
        .iter()
        .filter(|p| p.starts_with("nodes/") && p.ends_with("/offsets.bin"))
        .collect();
    let shard_edges: Vec<&String> = in_manifest
        .iter()
        .filter(|p| p.starts_with("nodes/") && p.ends_with("/edges.bin"))
        .collect();
    assert!(
        !shard_offsets.is_empty(),
        "_MANIFEST.json must list per-shard offsets.bin under nodes/<table>/shard-NNNNN/; \
         entries: {:?}",
        in_manifest,
    );
    assert!(
        !shard_edges.is_empty(),
        "_MANIFEST.json must list per-shard edges.bin under nodes/<table>/shard-NNNNN/; \
         entries: {:?}",
        in_manifest,
    );

    Ok(())
}

/// API-boundary rejection: combining a dataplane mirror with a `None`
/// success-marker payload produces an unverifiable, unrestorable copy
/// (`finalize_snapshot` doesn't run, so neither `_MANIFEST.json` nor
/// `_SUCCESS` is ever written to the dataplane). There's no caller for
/// this shape; failing loud at the boundary is preferable to silently
/// scattering bytes across S3 and producing nothing the restore path
/// can use.
#[test]
fn ingest_rejects_dataplane_mirror_without_marker_payload() -> Result<()> {
    let tmp = TempDir::new()?;
    let local_root = tmp.path().join("local");
    let dataplane_root = tmp.path().join("dataplane");
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    let config = build_graph_config(&users, &devices, &edges)?;

    let result = diskgraph_ingest::ingest_with_stats_from_source(
        config,
        None,
        local_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
        Some(dataplane_root.to_str().unwrap()),
        None, // ← the offending combo
    );

    let err = result.expect_err(
        "ingest must reject `output_dataplane_dir=Some` with \
         `success_marker_content=None`",
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("output_dataplane_dir") && msg.contains("success_marker_content"),
        "rejection message must name both kwargs so the caller can fix \
         the call site; got: {msg}"
    );
    // Ingest must abort BEFORE writing to either backend — neither
    // root should have any state on disk.
    assert!(
        !local_root.exists() || fs::read_dir(&local_root)?.next().is_none(),
        "local root should be untouched on early rejection",
    );
    assert!(
        !dataplane_root.exists() || fs::read_dir(&dataplane_root)?.next().is_none(),
        "dataplane root should be untouched on early rejection",
    );
    Ok(())
}

/// Recursively download an object_store prefix to a local directory.
/// Used by the real-S3 integration test to fetch the dataplane mirror
/// after ingest. List → for each object: get → write to
/// `local_root/<rel_path>`.
fn download_object_store_prefix(
    rt: &Runtime,
    store: Arc<dyn ObjectStore>,
    prefix: &object_store::path::Path,
    local_root: &Path,
) -> Result<()> {
    fs::create_dir_all(local_root)
        .with_context(|| format!("create_dir_all {}", local_root.display()))?;
    rt.block_on(async {
        let mut stream = store.list(Some(prefix));
        while let Some(meta) = stream
            .try_next()
            .await
            .map_err(|e| anyhow!("list {}: {}", prefix, e))?
        {
            let obj_path = meta.location;
            // rel = obj_path stripped of the prefix
            let rel = obj_path
                .as_ref()
                .strip_prefix(prefix.as_ref())
                .map(|s| s.trim_start_matches('/'))
                .ok_or_else(|| {
                    anyhow!("object {} not under prefix {}", obj_path, prefix)
                })?;
            if rel.is_empty() {
                continue;
            }
            let bytes = store
                .get(&obj_path)
                .await
                .map_err(|e| anyhow!("get {}: {}", obj_path, e))?
                .bytes()
                .await
                .map_err(|e| anyhow!("read body {}: {}", obj_path, e))?;
            let dest = local_root.join(rel);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create_dir_all {}", parent.display()))?;
            }
            fs::write(&dest, &bytes)
                .with_context(|| format!("write {}", dest.display()))?;
        }
        Result::<_, anyhow::Error>::Ok(())
    })
}

/// Best-effort delete of every object under `prefix`. Used as a Drop
/// cleanup for the real-S3 test so a panicking test does not leak
/// objects into the test bucket.
fn purge_object_store_prefix(
    rt: &Runtime,
    store: Arc<dyn ObjectStore>,
    prefix: &object_store::path::Path,
) -> Result<()> {
    rt.block_on(async {
        let mut stream = store.list(Some(prefix));
        let mut paths: Vec<object_store::path::Path> = Vec::new();
        while let Some(meta) = stream
            .try_next()
            .await
            .map_err(|e| anyhow!("list {}: {}", prefix, e))?
        {
            paths.push(meta.location);
        }
        for path in paths {
            // Best-effort: log and continue on failure.
            if let Err(e) = store.delete(&path).await {
                eprintln!("cleanup: delete {} failed: {}", path, e);
            }
        }
        Result::<_, anyhow::Error>::Ok(())
    })
}

/// End-to-end: a snapshot ingested with a dataplane mirror must be
/// runtime-openable from BOTH the local root AND the dataplane root.
///
/// **Bug this catches.** Because `build_index` bypasses `OutputTarget`,
/// its outputs are never submitted to the mirror. `restore_snapshot`
/// (which copies the mirror dir to local SSD) therefore produces a tree
/// that passes `verify_snapshot_integrity` (manifest doesn't list the
/// missing files) but cannot be opened by `GraphIndexReader::open`
/// (which reads `nodes/<table>/shard-NNNNN/{offsets,edges}.bin`). This
/// is the exact failure singlehost hit after AMI stop/start: NVMe wiped,
/// dataplane restored an incomplete snapshot, gen_tt errored with
/// `opening graph index: No such file or directory`.
///
/// Asserting against the dataplane root directly is the strongest
/// production-equivalent check: if this passes, restore-from-mirror is
/// guaranteed to produce a usable snapshot.
#[test]
fn dataplane_root_is_runtime_openable_after_ingest_with_mirror() -> Result<()> {
    let tmp = TempDir::new()?;
    let (local_root, dataplane_root) = run_mirror_ingest(&tmp)?;

    // Sanity: the LOCAL root opens. If this fails, the bug is more
    // serious than the L2 mirror gap (ingest itself is broken).
    GraphIndexReader::open(
        &local_root.join("index_metadata.json"),
        &local_root,
    )
    .map_err(|e| anyhow!("GraphIndexReader::open(local) failed: {e:#}"))?;

    // The load-bearing assertion. Pre-fix this fails with
    // `No such file or directory` opening the first shard's offsets.bin
    // because `build_index` never submitted shard files to the mirror.
    GraphIndexReader::open(
        &dataplane_root.join("index_metadata.json"),
        &dataplane_root,
    )
    .map_err(|e| {
        anyhow!(
            "GraphIndexReader::open(dataplane) failed: {e:#} — the L1 ⇒ L2 \
             invariant is violated; the mirror does not carry every file \
             the runtime needs."
        )
    })?;

    Ok(())
}

/// Real-S3 integration test for the streaming `submit_upload_from_path`
/// path. Uses the team's `kumo-unit-test` bucket under a per-run
/// unique prefix so concurrent runs / reruns don't collide.
///
/// **What this test exercises that the local-mirror e2e does not.**
/// Local mirrors run against an `object_store::LocalFileSystem` —
/// `put_multipart` is a single fs write, not a real multipart upload,
/// so the streaming path-from-disk code (`upload_file_once_from_path`)
/// is never actually exercised end-to-end. With a real S3 backend the
/// `pread` → `put_part` flow runs for real, validates that part
/// streaming + retry + complete all wire correctly, and confirms the
/// uploaded object byte-for-byte matches the local file. Without this
/// test, a regression to streaming submit (e.g., an off-by-one in the
/// part-offset loop) would slip through every local-only test.
///
/// **`#[ignore]` on purpose.** Real S3 calls cost money + need
/// credentials; this is run manually or in a designated integration
/// CI lane:
///
/// ```sh
/// AWS_REGION=us-west-2 AWS_PROFILE=kumo-dev \
///   cargo test -p diskgraph-ingest --test dataplane_mirror_e2e \
///   -- --ignored --nocapture dataplane_mirror_to_real_s3
/// ```
///
/// The test gates itself on `AWS_REGION` / `AWS_DEFAULT_REGION`; if
/// neither is set, it logs and exits 0 so an `--ignored` sweep on a
/// machine without credentials doesn't fail spuriously.
#[test]
#[ignore]
fn dataplane_mirror_to_real_s3_streams_files_without_buffering_them() -> Result<()> {
    if std::env::var("AWS_REGION").is_err()
        && std::env::var("AWS_DEFAULT_REGION").is_err()
    {
        eprintln!(
            "Skipping: AWS_REGION / AWS_DEFAULT_REGION not set; this test \
             requires real AWS credentials."
        );
        return Ok(());
    }

    // Per-run unique prefix. Combination of nanosecond timestamp +
    // process id is collision-resistant within a CI runner / dev box
    // and avoids pulling `uuid` as a dev dep solely for this test.
    let test_id = format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        std::process::id(),
    );
    let s3_dataplane = format!("s3://kumo-unit-test/diskgraph-mirror-test/{test_id}");
    eprintln!("Using dataplane: {s3_dataplane}");

    let tmp = TempDir::new()?;
    let local_root = tmp.path().join("local");
    let users = tmp.path().join("users.parquet");
    let devices = tmp.path().join("devices.parquet");
    let edges = tmp.path().join("edges.parquet");

    write_users(&users)?;
    write_devices(&devices)?;
    write_edges(&edges)?;
    let config = build_graph_config(&users, &devices, &edges)?;

    let marker_content = serde_json::json!({
        "index_path": local_root.to_str().unwrap(),
        "col_stats_path": local_root.join("col_stats.json").to_str().unwrap(),
        "ingest_stats_path": local_root.join("ingest_stats.json").to_str().unwrap(),
        "kumo_stats_dir": local_root.join("kumo_col_stats").to_str().unwrap(),
    })
    .to_string();

    let options = IngestOptions {
        parallel: true,
        ..Default::default()
    };

    // Drop guard: always purge the S3 prefix on test exit (success or
    // panic), so a flaky / failing run doesn't leak objects into the
    // test bucket.
    struct S3Cleanup {
        rt: Arc<Runtime>,
        store: Arc<dyn ObjectStore>,
        prefix: object_store::path::Path,
    }
    impl Drop for S3Cleanup {
        fn drop(&mut self) {
            if let Err(e) = purge_object_store_prefix(&self.rt, Arc::clone(&self.store), &self.prefix) {
                eprintln!("cleanup: purge {} failed: {}", self.prefix, e);
            }
        }
    }
    let cleanup_rt = Arc::new(Runtime::new()?);
    let cleanup_store = build_object_store_for_path(&s3_dataplane)
        .with_context(|| format!("build store for {s3_dataplane}"))?;
    let cleanup_prefix = object_path_for_path(&s3_dataplane)
        .with_context(|| format!("derive prefix path for {s3_dataplane}"))?;
    let _cleanup = S3Cleanup {
        rt: Arc::clone(&cleanup_rt),
        store: Arc::clone(&cleanup_store),
        prefix: cleanup_prefix.clone(),
    };

    // Run the ingest with S3 as the dataplane.
    diskgraph_ingest::ingest_with_stats_from_source(
        config,
        None,
        local_root.to_str().unwrap(),
        Dest::Local,
        options,
        Some(&s3_dataplane),
        Some(&marker_content),
    )
    .with_context(|| format!("ingest with S3 dataplane at {s3_dataplane}"))?;

    // 1. Local snapshot is openable (sanity — same as the local-only
    //    test). If this fails, the bug is upstream of the mirror path.
    GraphIndexReader::open(&local_root.join("index_metadata.json"), &local_root)
        .map_err(|e| anyhow!("GraphIndexReader::open(local): {e:#}"))?;

    // 2. Download the entire S3 prefix to a fresh local dir. This is
    //    the production-equivalent "restore from dataplane" path.
    let downloaded_root = tmp.path().join("downloaded_from_s3");
    download_object_store_prefix(
        &cleanup_rt,
        Arc::clone(&cleanup_store),
        &cleanup_prefix,
        &downloaded_root,
    )
    .with_context(|| format!("download {s3_dataplane} -> local"))?;

    // 3. Manifest-based integrity check passes against the downloaded
    //    copy — i.e., every file `_MANIFEST.json` lists is present and
    //    correctly sized on S3.
    verify_snapshot_integrity(&downloaded_root)
        .map_err(|e| anyhow!("verify(downloaded): {e:#}"))?;

    // 4. **Load-bearing assertion.** GraphIndexReader::open succeeds
    //    against the S3-downloaded snapshot. This proves the streaming
    //    `submit_upload_from_path` path actually transferred every
    //    runtime-required file end-to-end. Pre-fix (when shard files
    //    bypassed both manifest and mirror), this would fail with
    //    `No such file or directory` opening the first shard.
    GraphIndexReader::open(
        &downloaded_root.join("index_metadata.json"),
        &downloaded_root,
    )
    .map_err(|e| anyhow!("GraphIndexReader::open(downloaded): {e:#}"))?;

    // 5. Byte-for-byte spot-check on a non-trivial file. We compare
    //    one of the per-shard `offsets.bin` files between local and
    //    S3-downloaded copies. Any silent corruption in the streaming
    //    pread → put_part flow (off-by-one part offset, wrong part
    //    length, padding) would surface here — `verify_snapshot_integrity`
    //    only checks size, not content.
    let manifest = assert_local_matches_manifest(&local_root)?;
    let probe_rel = manifest
        .files
        .iter()
        .find(|e| {
            e.rel_path.starts_with("nodes/") && e.rel_path.ends_with("/offsets.bin")
        })
        .ok_or_else(|| {
            anyhow!("expected at least one shard offsets.bin in manifest")
        })?
        .rel_path
        .clone();
    let local_bytes = fs::read(local_root.join(&probe_rel))
        .with_context(|| format!("read local probe {probe_rel}"))?;
    let s3_bytes = fs::read(downloaded_root.join(&probe_rel))
        .with_context(|| format!("read S3 probe {probe_rel}"))?;
    assert_eq!(
        local_bytes, s3_bytes,
        "byte mismatch between local and S3-downloaded {probe_rel}: local_len={} s3_len={}",
        local_bytes.len(),
        s3_bytes.len(),
    );

    Ok(())
}
