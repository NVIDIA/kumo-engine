//! Integration tests proving that resuming from any phase produces a
//! byte-identical final index compared to running the same ingest end-to-end
//! without resume. This is the core correctness invariant the resume code
//! must satisfy.
//!
//! Strategy: for each phase boundary P, run two ingests against the same
//! synthetic dataset:
//!
//! 1. Baseline: a fresh end-to-end ingest with `resume_from_phase = None`.
//! 2. Split: a partial ingest stopped right after phase P completes (the
//!    `_tmp/_phase_complete_<P>` marker has been written) followed by an
//!    in-process resume from `P+1`.
//!
//! The "stop after P" leg is performed by spawning the `ingest` binary as a
//! subprocess and SIGKILLing it the moment the marker file appears. This
//! preserves on-disk state (the binary's `CleanupGuard` only runs on graceful
//! Drop, and SIGKILL skips destructors). The in-process resume then runs
//! `ingest_with_stats_from_source` with the appropriate `resume_from_phase`.
//!
//! Final outputs are compared byte-for-byte for files that should be
//! deterministic (`index_metadata.json`, `nodes/<t>/shard-N/{offsets.bin,
//! edges.bin,build_stats.json}`).

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::{ingest_with_stats, IngestOptions, IngestPhase};
use parquet::arrow::arrow_writer::ArrowWriter;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Synthetic dataset
// ---------------------------------------------------------------------------

/// Build a tiny deterministic graph as parquet files in `input_root` and
/// return the path to a `diskgraph_config.json` that points at them.
///
/// Layout:
/// - `users` table: 10 rows, PK = `user_id`, has `create_time`.
/// - `items` table: 5 rows, PK = `item_id`.
/// - `orders` fact table: 20 rows, FKs `user_id -> users` and
///   `item_id -> items`, plus a `ts` timestamp.
fn write_synthetic_graph(input_root: &Path, dataset_name: &str) -> Result<PathBuf> {
    fs::create_dir_all(input_root)?;

    // Users
    let mut user_ids: Vec<i64> = (0..10).collect();
    let mut user_names: Vec<String> = (0..10).map(|i| format!("user-{i:02}")).collect();
    let mut user_ts: Vec<i64> = (0..10)
        .map(|i| 1_700_000_000_i64 + (i as i64) * 60)
        .collect();
    // Exercise phase-owned resume counters as well as index bytes: PkMap
    // records this duplicate, while the first row remains the canonical PK.
    user_ids.push(3);
    user_names.push("user-duplicate".to_string());
    user_ts.push(1_700_001_000);
    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "create_time",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            ),
        ]),
        vec![
            Arc::new(Int64Array::from(user_ids)) as ArrayRef,
            Arc::new(StringArray::from(user_names)) as ArrayRef,
            Arc::new(TimestampSecondArray::from(user_ts)) as ArrayRef,
        ],
    )?;

    // Items
    let item_ids: Vec<i64> = (0..5).collect();
    let item_names: Vec<String> = (0..5).map(|i| format!("item-{i:02}")).collect();
    write_parquet(
        &input_root.join("items.parquet"),
        Schema::new(vec![
            Field::new("item_id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(Int64Array::from(item_ids)) as ArrayRef,
            Arc::new(StringArray::from(item_names)) as ArrayRef,
        ],
    )?;

    // Orders: FK fact table (no PK on this table — match dd_graph_run2 pattern).
    let mut order_user_ids: Vec<i64> = Vec::with_capacity(20);
    let mut order_item_ids: Vec<i64> = Vec::with_capacity(20);
    let mut order_ts: Vec<i64> = Vec::with_capacity(20);
    for i in 0..20i64 {
        // One orphan FK exercises EdgesMap key-failure state restoration.
        order_user_ids.push(if i == 19 { 999 } else { i % 10 });
        order_item_ids.push(i % 5);
        order_ts.push(1_700_000_000_i64 + i * 30);
    }
    write_parquet(
        &input_root.join("orders.parquet"),
        Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("item_id", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            ),
        ]),
        vec![
            Arc::new(Int64Array::from(order_user_ids)) as ArrayRef,
            Arc::new(Int64Array::from(order_item_ids)) as ArrayRef,
            Arc::new(TimestampSecondArray::from(order_ts)) as ArrayRef,
        ],
    )?;

    let users_path = input_root.join("users.parquet").to_string_lossy().to_string();
    let items_path = input_root.join("items.parquet").to_string_lossy().to_string();
    let orders_path = input_root.join("orders.parquet").to_string_lossy().to_string();

    let config = serde_json::json!({
        "dataset_name": dataset_name,
        "num_shards": 2,
        "tables": [
            {
                "name": "Users",
                "source": {"type": "parquet", "paths": [users_path]},
                "primary_key": "user_id",
                "create_time": "create_time",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"},
                    {"name": "create_time", "ingested_dtype": "timestamp"}
                ]
            },
            {
                "name": "Items",
                "source": {"type": "parquet", "paths": [items_path]},
                "primary_key": "item_id",
                "columns": [
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "name", "ingested_dtype": "string"}
                ]
            },
            {
                "name": "Orders",
                "source": {"type": "parquet", "paths": [orders_path]},
                "create_time": "ts",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "ts", "ingested_dtype": "timestamp"}
                ],
                "fkeys": [
                    {"src_column": "user_id", "dst_table": "Users"},
                    {"src_column": "item_id", "dst_table": "Items"}
                ]
            }
        ]
    });
    let config_path = input_root.join("diskgraph_config.json");
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;
    Ok(config_path)
}

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase markers
// ---------------------------------------------------------------------------

fn marker_path(root: &Path, phase: IngestPhase) -> PathBuf {
    root.join("_tmp")
        .join(format!("_phase_complete_{}", phase.name()))
}

// ---------------------------------------------------------------------------
// Subprocess runner: stops the binary right after phase `stop_after`
// completes (its marker appears).
// ---------------------------------------------------------------------------

fn ingest_binary() -> &'static str {
    env!("CARGO_BIN_EXE_ingest")
}

/// Run the ingest binary against `config_path`, watching for the
/// `_phase_complete_<stop_after>` marker. The instant it appears, SIGKILL
/// the child so on-disk state freezes there: `_tmp/` is preserved (no
/// `cleanup_temp`), the predecessor + stop_after markers are present, and
/// no successor phase has begun. Returns once the child is reaped.
fn run_partial(
    config_path: &Path,
    output_root: &Path,
    stop_after: IngestPhase,
) -> Result<()> {
    fs::create_dir_all(output_root)?;
    let marker = marker_path(output_root, stop_after);

    let mut child = Command::new(ingest_binary())
        // The test must observe and kill the phase worker itself. Killing the
        // outer OOM supervisor would intentionally leave its isolated worker
        // session running long enough to race the in-process resume below.
        .arg("--worker")
        .arg("--source")
        .arg(config_path)
        .arg("--output")
        .arg(output_root)
        .arg("--dest")
        .arg("local")
        // Single-threaded keeps wall-time predictable on the small fixture
        // and avoids races on marker observation.
        .arg("--context-threads")
        .arg("1")
        .arg("--encode-threads")
        .arg("1")
        .arg("--pk-map-threads")
        .arg("1")
        .arg("--edges-map-threads")
        .arg("1")
        .arg("--index-threads")
        .arg("1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {}", ingest_binary()))?;

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        // Has phase P's marker landed yet?
        if marker.exists() {
            kill_and_wait(&mut child)?;
            // Predecessor markers must also be present at this point.
            for pred in stop_after.predecessors_inclusive() {
                let p = marker_path(output_root, pred);
                if !p.exists() {
                    bail!(
                        "after kill: expected marker {} to exist but it did not",
                        p.display()
                    );
                }
            }
            return Ok(());
        }
        // Did the child exit on its own (success or error)? If it ran past
        // our stop point we still take its on-disk state — the `_tmp/`
        // cleanup is the last thing it does, but markers are gone after a
        // *successful* run. Detect that and fail loudly so the test author
        // can pick a smaller fixture or earlier phase.
        if let Some(status) = child
            .try_wait()
            .with_context(|| "try_wait on partial ingest")?
        {
            // Child exited before we observed the marker. Either the run
            // raced past us (cleanup_temp swept the marker) or it errored.
            // In both cases, we cannot reliably build the desired partial
            // state from this attempt.
            bail!(
                "partial ingest exited (status={status:?}) before marker {} was observed; \
                 try a slower fixture or use a finer poll interval",
                marker.display()
            );
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "partial ingest did not produce marker {} within 60s",
                marker.display()
            );
        }
        // Tight poll: phases on a 30-row fixture finish in well under a
        // millisecond once started, so any sleep here risks racing past
        // the marker we want.
        std::thread::yield_now();
    }
}

trait PhaseExt {
    fn predecessors_inclusive(self) -> Vec<IngestPhase>;
    fn successor(self) -> Option<IngestPhase>;
}

impl PhaseExt for IngestPhase {
    fn predecessors_inclusive(self) -> Vec<IngestPhase> {
        // ALL_PHASES is in execution order; take while p <= self.
        diskgraph_ingest::ALL_PHASES
            .iter()
            .copied()
            .filter(|p| *p <= self)
            .collect()
    }
    fn successor(self) -> Option<IngestPhase> {
        let phases = diskgraph_ingest::ALL_PHASES;
        let idx = phases.iter().position(|p| *p == self)?;
        phases.get(idx + 1).copied()
    }
}

fn kill_and_wait(child: &mut Child) -> Result<()> {
    let _ = child.kill();
    let _ = child.wait();
    Ok(())
}

// ---------------------------------------------------------------------------
// Byte-identical comparison
// ---------------------------------------------------------------------------

/// Files we expect to be byte-identical between baseline and split runs.
const COMPARE_BASENAMES: &[&str] = &["offsets.bin", "edges.bin", "build_stats.json"];

fn assert_indexes_byte_identical(baseline: &Path, split: &Path) -> Result<()> {
    // index_metadata.json — the user-visible final manifest.
    let a = baseline.join("index_metadata.json");
    let b = split.join("index_metadata.json");
    compare_files_byte_for_byte(&a, &b)?;

    // For every node table directory present under nodes/ in baseline,
    // walk shards and compare offsets/edges/build_stats. Edge-table-only
    // names (no shards/) are skipped.
    let nodes_a = baseline.join("nodes");
    if !nodes_a.is_dir() {
        bail!("baseline missing nodes/: {}", nodes_a.display());
    }
    for entry in fs::read_dir(&nodes_a)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let table_name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 dir name"))?
            .to_owned();
        let table_a = entry.path();
        let table_b = split.join("nodes").join(&table_name);
        if !table_b.is_dir() {
            bail!(
                "split missing nodes/{table_name}/: {}",
                table_b.display()
            );
        }
        // Walk shard-NNNNN/ directories under each table.
        for shard_entry in fs::read_dir(&table_a)? {
            let shard_entry = shard_entry?;
            if !shard_entry.file_type()?.is_dir() {
                continue;
            }
            let shard_name = shard_entry
                .file_name()
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 shard name"))?
                .to_owned();
            if !shard_name.starts_with("shard-") {
                continue;
            }
            let shard_a = shard_entry.path();
            let shard_b = table_b.join(&shard_name);
            if !shard_b.is_dir() {
                bail!(
                    "split missing nodes/{table_name}/{shard_name}/: {}",
                    shard_b.display()
                );
            }
            for base in COMPARE_BASENAMES {
                let fa = shard_a.join(base);
                let fb = shard_b.join(base);
                if !fa.exists() && !fb.exists() {
                    continue;
                }
                compare_files_byte_for_byte(&fa, &fb)?;
            }
        }
    }
    Ok(())
}

fn assert_ingest_stats_equivalent(baseline: &Path, split: &Path) -> Result<()> {
    fn comparable_report(root: &Path) -> Result<serde_json::Value> {
        let path = root.join("ingest_stats.json");
        let mut report: serde_json::Value = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )?;
        let stats = report
            .get_mut("stats")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| anyhow!("{} has no stats object", path.display()))?;
        // These identify the destination and measure the individual process;
        // every data-quality and graph-result field must remain identical.
        stats.remove("output");
        stats.remove("perf");
        Ok(report)
    }

    let baseline_report = comparable_report(baseline)?;
    let split_report = comparable_report(split)?;
    if baseline_report != split_report {
        bail!("ingest_stats.json differs after excluding output and perf fields");
    }
    Ok(())
}

fn compare_files_byte_for_byte(a: &Path, b: &Path) -> Result<()> {
    let ba = fs::read(a).with_context(|| format!("read baseline {}", a.display()))?;
    let bb = fs::read(b).with_context(|| format!("read split {}", b.display()))?;
    if ba.len() != bb.len() {
        bail!(
            "size mismatch: {} ({} bytes) vs {} ({} bytes)",
            a.display(),
            ba.len(),
            b.display(),
            bb.len()
        );
    }
    if let Some((i, (xa, xb))) = ba
        .iter()
        .zip(bb.iter())
        .enumerate()
        .find(|(_, (xa, xb))| xa != xb)
    {
        bail!(
            "byte mismatch at offset {i} between {} and {}: 0x{xa:02x} vs 0x{xb:02x}",
            a.display(),
            b.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

/// Generic helper: build the synthetic graph, do a fresh end-to-end ingest
/// (baseline), then a partial run that stops right after `stop_after` and
/// finishes via in-process resume from `stop_after.successor()`. Compare.
fn run_phase_boundary_test(stop_after: IngestPhase) -> Result<()> {
    let resume_from = stop_after
        .successor()
        .ok_or_else(|| anyhow!("no successor for {stop_after:?}; nothing to resume"))?;

    let tmp = tempdir()?;
    let input_root = tmp.path().join("input");
    let baseline_root = tmp.path().join("baseline");
    let split_root = tmp.path().join("split");

    let config_path = write_synthetic_graph(
        &input_root,
        &format!("resume_eq_{}", stop_after.name()),
    )?;

    // 1. Baseline: full fresh run.
    ingest_with_stats(
        &config_path,
        baseline_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;

    // 2. Split arm 1: partial run stopped after `stop_after`.
    run_partial(&config_path, &split_root, stop_after)?;

    // Sanity: the marker for stop_after.successor() must NOT exist (we
    // killed before it could complete). The successor's marker is the
    // last thing the successor phase writes, so its presence is the
    // authoritative signal that the partial run reached completion past
    // our intended stop point.
    let succ_marker = marker_path(&split_root, resume_from);
    if succ_marker.exists() {
        let leaked = tmp.keep();
        bail!(
            "successor marker {} exists after partial run — partial did not stop in time \n  split: {}",
            succ_marker.display(),
            leaked.join("split").display(),
        );
    }
    // Note: we deliberately do NOT bail when index_metadata.json exists
    // here. That file is Index's *first* output (`build_index`'s
    // `fs::File::create` runs immediately after the phase begins), not
    // its last, so its presence proves only "Index started," not
    // "pipeline completed." The race window between
    // `_phase_complete_edges_map` appearing and SIGKILL landing is
    // sub-millisecond and on busy CI runners SIGKILL routinely loses
    // it — the kill arrives after Index has touched
    // `index_metadata.json` and any number of partial per-shard files.
    // That state is recoverable: `wipe_phase_outputs("index")` removes
    // exactly those artifacts at resume entry, and Index re-runs
    // cleanly. The successor-marker check above already detects the
    // unrecoverable case (the binary ran past Index and through
    // `cleanup_temp` would have removed all `_tmp/` markers — which
    // `run_partial`'s predecessor check catches on its own).

    // 3. Split arm 2: in-process resume from `resume_from`.
    ingest_with_stats(
        &config_path,
        split_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            resume_from_phase: Some(resume_from),
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;

    // 4. Both must produce a final index_metadata.json + per-shard files.
    if !baseline_root.join("index_metadata.json").exists() {
        bail!("baseline missing index_metadata.json");
    }
    if !split_root.join("index_metadata.json").exists() {
        bail!("split missing index_metadata.json");
    }
    if let Err(e) = assert_indexes_byte_identical(&baseline_root, &split_root) {
        // Leak the tempdir so the user can inspect it.
        let leaked = tmp.keep();
        return Err(anyhow!(
            "{e}\n  baseline: {}\n  split:    {}",
            leaked.join("baseline").display(),
            leaked.join("split").display(),
        ));
    }
    assert_ingest_stats_equivalent(&baseline_root, &split_root)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Sanity check: two fresh runs against the same fixture must produce
/// byte-identical output, otherwise the equivalence assertion below cannot
/// distinguish "resume bug" from "intrinsic non-determinism".
#[test]
fn fresh_vs_fresh_is_deterministic() -> Result<()> {
    let tmp = tempdir()?;
    let input_root = tmp.path().join("input");
    let a_root = tmp.path().join("a");
    let b_root = tmp.path().join("b");
    let config_path = write_synthetic_graph(&input_root, "resume_eq_determinism")?;

    for out in [&a_root, &b_root] {
        ingest_with_stats(
            &config_path,
            out.to_str().unwrap(),
            Dest::Local,
            IngestOptions {
                source_loader_threads: Some(1),
                source_encoder_threads: Some(1),
                pk_map_threads: Some(1),
                edges_map_threads: Some(1),
                index_threads: Some(1),
                parent_index_path: None,
                ..Default::default()
            },
        )?;
    }
    if let Err(e) = assert_indexes_byte_identical(&a_root, &b_root) {
        let leaked = tmp.keep();
        return Err(anyhow!(
            "fresh ingest is not deterministic: {e}\n  a: {}\n  b: {}",
            leaked.join("a").display(),
            leaked.join("b").display(),
        ));
    }
    Ok(())
}

/// Resume from `Encode` is a no-op: `Encode` is the very first phase, so
/// asking the pipeline to "resume from Encode" is functionally equivalent
/// to a fresh run except that the top-level output_dir wipe is skipped and
/// CleanupGuard is disabled. There is no "fresh up to before Encode" state
/// to construct, so this case is not testable with our partial-then-resume
/// pattern. Instead, we assert the documented equivalence directly: a
/// fresh run and a `resume_from_phase=Some(Encode)` run on a clean output
/// dir must produce a byte-identical final index.
#[test]
fn resume_from_encode_matches_fresh() -> Result<()> {
    let tmp = tempdir()?;
    let input_root = tmp.path().join("input");
    let baseline_root = tmp.path().join("baseline");
    let split_root = tmp.path().join("split");
    let config_path = write_synthetic_graph(&input_root, "resume_eq_context")?;

    ingest_with_stats(
        &config_path,
        baseline_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;
    fs::create_dir_all(&split_root)?;
    ingest_with_stats(
        &config_path,
        split_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            resume_from_phase: Some(IngestPhase::Encode),
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;
    assert_indexes_byte_identical(&baseline_root, &split_root)?;
    Ok(())
}

#[test]
fn resume_from_pk_map_matches_fresh() -> Result<()> {
    run_phase_boundary_test(IngestPhase::Encode)
}

#[test]
fn resume_from_edges_map_matches_fresh() -> Result<()> {
    run_phase_boundary_test(IngestPhase::PkMap)
}

#[test]
fn resume_from_index_matches_fresh() -> Result<()> {
    run_phase_boundary_test(IngestPhase::EdgesMap)
}

/// Verify the wipe-on-entry semantics for the resume-target phase: when
/// resuming from `pk_map`, any pre-existing pk_map output (e.g. from a
/// prior crashed run) must be overwritten, not retained. Write a canary
/// byte to a stale `pk_hash-N.bin`, resume, and check the canary is gone
/// AND the resulting file matches a fresh run byte-for-byte.
#[test]
fn resume_wipes_stale_pk_map_outputs() -> Result<()> {
    let tmp = tempdir()?;
    let input_root = tmp.path().join("input");
    let baseline_root = tmp.path().join("baseline");
    let split_root = tmp.path().join("split");
    let config_path = write_synthetic_graph(&input_root, "resume_eq_wipe")?;

    // Baseline.
    ingest_with_stats(
        &config_path,
        baseline_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;

    // Partial through Encode.
    run_partial(&config_path, &split_root, IngestPhase::Encode)?;

    // Plant a stale pk_hash file with canary content. This is what a
    // half-completed pk_map from a prior crash would have left behind.
    let stale = split_root.join("nodes/users/pk_hash-00000.bin");
    fs::create_dir_all(stale.parent().unwrap())?;
    let canary = b"CANARY-NOT-A-VALID-PK-HASH-FILE";
    fs::write(&stale, canary)?;
    let stale_size = fs::metadata(&stale)?.len();
    assert_eq!(stale_size, canary.len() as u64);

    // Resume — the pk_map wipe-on-entry must remove the canary, then
    // re-run pk_map and write the real file.
    ingest_with_stats(
        &config_path,
        split_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            resume_from_phase: Some(IngestPhase::PkMap),
            source_loader_threads: Some(1),
            source_encoder_threads: Some(1),
            pk_map_threads: Some(1),
            edges_map_threads: Some(1),
            index_threads: Some(1),
            parent_index_path: None,
            ..Default::default()
        },
    )?;

    // Canary must be gone.
    let post = fs::read(&stale)?;
    if post == canary {
        bail!(
            "wipe failed: pk_hash-00000.bin still contains canary after resume from pk_map"
        );
    }
    // And final output must match baseline byte-for-byte.
    assert_indexes_byte_identical(&baseline_root, &split_root)?;
    Ok(())
}
