// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling-throughput bench on the **production engine** path
//! (`DiskGraphEngine::sample`) — the one TT-NG actually runs.
//!
//! The measured ~1.31M-IOPS mmap ceiling
//! (`docs/explorations/pinterest_omnisage/mmap_iops_ceiling_and_io_uring_proposal.md`)
//! came from `dg-sample-bench`, the prototype worker's in-process sampler. Both
//! read through the same `ShardReader` mmap, so the ceiling should transfer —
//! but "should" isn't measured, so this harness reproduces the sweep on the
//! engine and, with `--prefetch`, measures how much of the gap batched
//! readahead recovers.
//!
//! Reports achieved read IOPS and average queue depth from `/proc/diskstats`
//! deltas over the measure window only, so the number is comparable to the fio
//! device ceiling. Queue depth is the diagnostic that matters: the mmap fault
//! path pins it near 0 (each thread blocks on its one read), which is *why* the
//! devices deliver a third of their capacity.
//!
//! Usage:
//!   dg-engine-bench --index <dir> --seed-type <name> [--layers 3] [--fanout 10]
//!                   [--num-seeds N] [--batch-size B] [--threads T]
//!                   [--prefetch off|on|both] [--verify] [--devices nvme1n1,...]
//!                   [--warmup-seeds W] [--seed S] [--temporal]
//!                   [--cpu-profile /tmp/measure.pb] [--profile-frequency 500]

use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, BinaryArray, FixedSizeBinaryArray, Int64Array, LargeBinaryArray, LargeStringArray,
    StringArray, UInt64Array,
};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_core::sampling::{
    AbsentKeyPolicy, BatchSamplingRequest, FanoutSpec, LayerSpec, SamplingFlags, SamplingMode,
    SamplingPlan,
};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::InMemoryOutput;
use diskgraph_engine::state::StatePool;
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_index::pkmap::{self, Key, KeyType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use pprof::protos::Message;
use pprof::ProfilerGuard;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;

// Optional whole-process allocation accounting for the measured sampling
// window. It is disabled during normal benchmarks because the counter RMWs
// intentionally perturb allocation-heavy paths; use --count-allocations as a
// diagnostic arm, not for the headline throughput number.
struct CountingAllocator;

#[cfg(feature = "jemalloc")]
static BASE_ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(not(feature = "jemalloc"))]
static BASE_ALLOCATOR: std::alloc::System = std::alloc::System;

#[cfg(feature = "jemalloc")]
const ALLOCATOR_NAME: &str = "jemalloc";
#[cfg(not(feature = "jemalloc"))]
const ALLOCATOR_NAME: &str = "system";

static ALLOC_TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = BASE_ALLOCATOR.alloc(layout);
        if ALLOC_TRACKING.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = BASE_ALLOCATOR.alloc_zeroed(layout);
        if ALLOC_TRACKING.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ALLOC_TRACKING.load(Ordering::Relaxed) {
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        BASE_ALLOCATOR.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = BASE_ALLOCATOR.realloc(ptr, layout, new_size);
        if ALLOC_TRACKING.load(Ordering::Relaxed) {
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        out
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Default)]
struct AllocStats {
    alloc_calls: u64,
    alloc_bytes: u64,
    dealloc_calls: u64,
    dealloc_bytes: u64,
}

fn alloc_stats() -> AllocStats {
    AllocStats {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

impl std::ops::Sub for AllocStats {
    type Output = AllocStats;

    fn sub(self, rhs: AllocStats) -> AllocStats {
        AllocStats {
            alloc_calls: self.alloc_calls.saturating_sub(rhs.alloc_calls),
            alloc_bytes: self.alloc_bytes.saturating_sub(rhs.alloc_bytes),
            dealloc_calls: self.dealloc_calls.saturating_sub(rhs.dealloc_calls),
            dealloc_bytes: self.dealloc_bytes.saturating_sub(rhs.dealloc_bytes),
        }
    }
}

// ---------------------------------------------------------------- CLI

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Result<Self> {
        const KNOWN: &[&str] = &[
            "index",
            "seed-type",
            "layers",
            "fanout",
            "num-seeds",
            "warmup-seeds",
            "batch-size",
            "threads",
            "engine-threads",
            "seed",
            "prefetch",
            "verify",
            "devices",
            "temporal",
            "plan",
            "fused",
            "backend",
            "io-qd",
            "io-rings",
            "io-batch-us",
            "io-batch-mult",
            "io-coalesce-pages",
            "io-eager-node-pages",
            "direct-cache-gb",
            "fetch-features",
            "prefault-offsets",
            "cpu-profile",
            "profile-frequency",
            "count-allocations",
            "prewarm-measurement",
            "pkey-e2e",
            "seed-source-config",
            "max-neighbors-per-seed",
            "seed-rank-coverage",
            "seed-rank-permutation-seed",
            "degree-stats-samples",
            "degree-stats-relations",
        ];
        let mut map = HashMap::new();
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let raw = &argv[i];
            let key = raw
                .strip_prefix("--")
                .ok_or_else(|| anyhow!("unexpected positional argument `{raw}`"))?
                .to_string();
            if !KNOWN.contains(&key.as_str()) {
                return Err(anyhow!("unknown option --{key}"));
            }
            // Bare `--flag` (next token is another flag, or we're at the end).
            let is_flag = i + 1 >= argv.len() || argv[i + 1].starts_with("--");
            let value = if is_flag {
                "true".to_string()
            } else {
                argv[i + 1].clone()
            };
            i += if is_flag { 1 } else { 2 };
            if map.insert(key.clone(), value).is_some() {
                return Err(anyhow!("duplicate option --{key}"));
            }
        }
        Ok(Self(map))
    }

    fn req(&self, k: &str) -> Result<&str> {
        self.0
            .get(k)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("missing --{k}"))
    }

    fn opt<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T> {
        match self.0.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| anyhow!("bad value for --{k}: {v}")),
        }
    }

    fn flag(&self, k: &str) -> bool {
        self.0.get(k).is_some_and(|v| v == "true" || v == "1")
    }
}

// --------------------------------------------------------- primary-key seeds

#[derive(Deserialize)]
struct SourceGraphConfig {
    tables: Vec<SourceTable>,
}

#[derive(Deserialize)]
struct SourceTable {
    name: String,
    primary_key: Option<String>,
    source: SourceLocation,
}

#[derive(Deserialize)]
struct SourceLocation {
    #[serde(default)]
    paths: Vec<String>,
}

#[derive(Clone, Copy)]
enum SeedBatch<'a> {
    NodeIds(&'a [u64]),
    PrimaryKeys(&'a [Key]),
}

enum SeedStorage {
    NodeIds(Vec<u64>),
    PrimaryKeys(Vec<Key>),
}

#[derive(Clone)]
struct SeedRankCoverage {
    /// Cumulative `(key_fraction, request_fraction)` points.
    points: Vec<(f64, f64)>,
    permutation_seed: u64,
}

impl SeedRankCoverage {
    fn parse(raw: &str, permutation_seed: u64) -> Result<Self> {
        let mut points = Vec::new();
        let mut previous_key = 0.0;
        let mut previous_request = 0.0;
        for entry in raw.split(',').filter(|entry| !entry.is_empty()) {
            let (key, request) = entry.split_once(':').ok_or_else(|| {
                anyhow!("seed rank entry `{entry}` must be KEY_FRACTION:REQUEST_FRACTION")
            })?;
            let key: f64 = key
                .parse()
                .with_context(|| format!("invalid key fraction in `{entry}`"))?;
            let request: f64 = request
                .parse()
                .with_context(|| format!("invalid request fraction in `{entry}`"))?;
            if !key.is_finite()
                || !request.is_finite()
                || key <= previous_key
                || request <= previous_request
                || key > 1.0
                || request > 1.0
            {
                return Err(anyhow!(
                    "seed rank coverage must increase strictly within (0, 1]"
                ));
            }
            points.push((key, request));
            previous_key = key;
            previous_request = request;
        }
        if points.last().copied() != Some((1.0, 1.0)) {
            return Err(anyhow!("seed rank coverage must end with `1:1`"));
        }
        Ok(Self {
            points,
            permutation_seed,
        })
    }

    fn source_row(&self, quantile: f64, total_rows: u64, permutation: (u64, u64)) -> u64 {
        let mut previous_key = 0.0;
        let mut previous_request = 0.0;
        let mut rank_fraction = 1.0;
        for &(key, request) in &self.points {
            if quantile < request {
                let position = (quantile - previous_request) / (request - previous_request);
                rank_fraction = previous_key + position * (key - previous_key);
                break;
            }
            previous_key = key;
            previous_request = request;
        }
        let rank = ((rank_fraction * total_rows as f64) as u64).min(total_rows - 1);
        seed_rank_affine_permute(rank, total_rows, permutation.0, permutation.1)
    }
}

fn seed_rank_mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d049bb133111eb);
    mixed ^ (mixed >> 31)
}

fn seed_rank_gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn seed_rank_affine_parameters(modulus: u64, seed: u64) -> (u64, u64) {
    if modulus <= 1 {
        return (0, 0);
    }
    let mut multiplier = (seed_rank_mix64(seed) % modulus) | 1;
    while seed_rank_gcd(multiplier, modulus) != 1 {
        multiplier = (multiplier + 2) % modulus;
        if multiplier == 0 {
            multiplier = 1;
        }
    }
    (
        multiplier,
        seed_rank_mix64(seed ^ 0xbb67ae8584caa73b) % modulus,
    )
}

fn seed_rank_affine_permute(value: u64, modulus: u64, multiplier: u64, offset: u64) -> u64 {
    if modulus <= 1 {
        0
    } else {
        ((value as u128 * multiplier as u128 + offset as u128) % modulus as u128) as u64
    }
}

impl SeedStorage {
    fn batch(&self) -> SeedBatch<'_> {
        match self {
            Self::NodeIds(values) => SeedBatch::NodeIds(values),
            Self::PrimaryKeys(values) => SeedBatch::PrimaryKeys(values),
        }
    }
}

impl SeedBatch<'_> {
    fn len(self) -> usize {
        match self {
            Self::NodeIds(values) => values.len(),
            Self::PrimaryKeys(values) => values.len(),
        }
    }

    fn slice(self, start: usize, end: usize) -> Self {
        match self {
            Self::NodeIds(values) => Self::NodeIds(&values[start..end]),
            Self::PrimaryKeys(values) => Self::PrimaryKeys(&values[start..end]),
        }
    }
}

fn local_source_path(config_dir: &Path, raw: &str) -> Result<PathBuf> {
    if raw.contains("://") && !raw.starts_with("file://") {
        return Err(anyhow!(
            "primary-key seed source must be local Parquet, got `{raw}`"
        ));
    }
    let path = PathBuf::from(raw.strip_prefix("file://").unwrap_or(raw));
    Ok(if path.is_absolute() {
        path
    } else {
        config_dir.join(path)
    })
}

fn node_id_from_known_source_pk(table: &TableReader, key: &Key) -> Result<Option<u64>> {
    let Some(pk_map) = table.pk_map() else {
        return Ok(None);
    };
    if key.key_type() != pk_map.key_type() {
        return Ok(None);
    }
    let shard_idx = pkmap::shard_for_key(key, table.shard_count());
    let Some(local) = pk_map
        .shard(shard_idx)
        .ok_or_else(|| anyhow!("missing PK shard {shard_idx}"))?
        .lookup(key)
    else {
        return Ok(None);
    };
    let min_id = table
        .shard_id_range(shard_idx)
        .map(|(min_id, _)| min_id)
        .unwrap_or(0);
    Ok(Some(min_id.saturating_add(local as u64)))
}

fn key_from_array(
    array: &dyn Array,
    row: usize,
    key_type: KeyType,
    table: &TableReader,
) -> Result<Key> {
    if array.is_null(row) {
        return Err(anyhow!("sampled NULL primary key at source row {row}"));
    }
    match key_type {
        KeyType::VarBytes => {
            if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
                return Ok(Key::Bytes(values.value(row).as_bytes().to_vec()));
            }
            if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
                return Ok(Key::Bytes(values.value(row).as_bytes().to_vec()));
            }
            if let Some(values) = array.as_any().downcast_ref::<BinaryArray>() {
                return Ok(Key::Bytes(values.value(row).to_vec()));
            }
            if let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() {
                return Ok(Key::Bytes(values.value(row).to_vec()));
            }
        }
        KeyType::Uuid => {
            if let Some(values) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                let raw: [u8; 16] = values
                    .value(row)
                    .try_into()
                    .map_err(|_| anyhow!("UUID primary key is not 16 bytes at source row {row}"))?;
                return Ok(Key::Uuid(raw));
            }
        }
        KeyType::Uuid256 => {
            if let Some(values) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                let raw: [u8; 32] = values.value(row).try_into().map_err(|_| {
                    anyhow!("UUID256 primary key is not 32 bytes at source row {row}")
                })?;
                return Ok(Key::Uuid256(raw));
            }
        }
        KeyType::Ip => {
            if let Some(values) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                let raw: [u8; 16] = values
                    .value(row)
                    .try_into()
                    .map_err(|_| anyhow!("IP primary key is not 16 bytes at source row {row}"))?;
                return Ok(Key::Ip(raw));
            }
        }
        KeyType::Int64 => {
            if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
                let value = u64::try_from(values.value(row))
                    .with_context(|| format!("negative integer primary key at source row {row}"))?;
                return Ok(Key::Int(value));
            }
            if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
                return Ok(Key::Int(values.value(row)));
            }
        }
    }

    // The source configs used by these benchmarks encode UUID/IP values as
    // strings and integer keys as Arrow numerics. Arrow's display conversion
    // keeps this setup-only path generic without affecting the timed window.
    let text = arrow::util::display::array_value_to_string(array, row)
        .with_context(|| format!("format primary key at source row {row}"))?;
    table
        .parse_pk_key(&text)
        .with_context(|| format!("parse primary key `{text}` at source row {row}"))
}

/// Draw source-table row positions with replacement, then read only the
/// primary-key column. The default is uniform over the full table. An optional
/// cumulative rank CDF reproduces a measured request skew and uses the same
/// rank-to-key permutation as the organic generator. Source reading is setup
/// work and finishes before benchmark telemetry.
fn load_primary_keys(
    config_path: &Path,
    seed_type: &str,
    count: usize,
    rng_seed: u64,
    table: &TableReader,
    rank_coverage: Option<&SeedRankCoverage>,
) -> Result<Vec<Key>> {
    let config: SourceGraphConfig = serde_json::from_slice(
        &std::fs::read(config_path)
            .with_context(|| format!("read source config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse source config {}", config_path.display()))?;
    let source = config
        .tables
        .into_iter()
        .find(|candidate| candidate.name == seed_type)
        .ok_or_else(|| anyhow!("seed table `{seed_type}` not found in source config"))?;
    let pk_column = source
        .primary_key
        .ok_or_else(|| anyhow!("seed table `{seed_type}` has no primary key"))?;
    if source.source.paths.is_empty() {
        return Err(anyhow!("seed table `{seed_type}` has no Parquet paths"));
    }
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let mut paths: Vec<PathBuf> = source
        .source
        .paths
        .iter()
        .map(|raw| local_source_path(config_dir, raw))
        .collect::<Result<_>>()?;
    // Generator shards are contiguous primary-key ranges. Preserve that
    // global row order even if a hand-written source config lists files in a
    // different order; uniform selection is unaffected by the sort.
    paths.sort();

    let mut file_rows = Vec::with_capacity(paths.len());
    let mut total_rows = 0u64;
    for path in &paths {
        let builder = ParquetRecordBatchReaderBuilder::try_new(
            File::open(path).with_context(|| format!("open source Parquet {}", path.display()))?,
        )
        .with_context(|| format!("read Parquet footer {}", path.display()))?;
        let rows = u64::try_from(builder.metadata().file_metadata().num_rows())
            .context("negative Parquet row count")?;
        total_rows = total_rows
            .checked_add(rows)
            .ok_or_else(|| anyhow!("source table row count overflow"))?;
        file_rows.push(rows);
    }
    if total_rows == 0 {
        return Err(anyhow!("seed table `{seed_type}` is empty"));
    }

    let mut rng = StdRng::seed_from_u64(rng_seed);
    let rank_permutation = rank_coverage
        .map(|coverage| seed_rank_affine_parameters(total_rows, coverage.permutation_seed));
    let mut selected: Vec<(u64, usize)> = (0..count)
        .map(|output| {
            let row = match rank_coverage {
                Some(coverage) => coverage.source_row(
                    rng.gen::<f64>(),
                    total_rows,
                    rank_permutation.expect("rank coverage prepared its permutation"),
                ),
                None => rng.gen_range(0..total_rows),
            };
            (row, output)
        })
        .collect();
    selected.sort_unstable_by_key(|&(row, _)| row);
    let selected_min = selected.first().map(|entry| entry.0).unwrap_or(0);
    let selected_max = selected.last().map(|entry| entry.0).unwrap_or(0);
    let key_type = table
        .pk_map()
        .ok_or_else(|| anyhow!("seed table `{seed_type}` has no PK hash index"))?
        .key_type();
    let mut output: Vec<Option<Key>> = (0..count).map(|_| None).collect();
    let mut selected_cursor = 0usize;
    let mut file_start = 0u64;
    let started = Instant::now();

    struct SourceFileWork<'a> {
        path: &'a Path,
        file_start: u64,
        selection_start: usize,
        selection_end: usize,
    }
    let mut work = Vec::with_capacity(paths.len());
    for (path, rows) in paths.iter().zip(&file_rows) {
        let file_end = file_start + rows;
        let selection_start = selected_cursor;
        while selected_cursor < selected.len() && selected[selected_cursor].0 < file_end {
            selected_cursor += 1;
        }
        if selection_start != selected_cursor {
            work.push(SourceFileWork {
                path,
                file_start,
                selection_start,
                selection_end: selected_cursor,
            });
        }
        file_start = file_end;
    }
    if selected_cursor != selected.len() {
        return Err(anyhow!(
            "not all sampled source rows were assigned to files"
        ));
    }

    // This is untimed benchmark setup, but a multi-billion-row source table
    // should not be decoded through one core for every backend process. Each
    // generated shard owns a disjoint row range, so files can be scanned in
    // parallel and their selected keys merged by the original output index.
    let setup_threads = work.len().clamp(1, 32);
    let setup_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(setup_threads)
        .thread_name(|index| format!("dg-pk-load-{index}"))
        .build()
        .context("build source primary-key loader pool")?;
    let results: Vec<Vec<(usize, Key)>> = setup_pool.install(|| {
        work.par_iter()
            .map(|item| -> Result<Vec<(usize, Key)>> {
                let file = File::open(item.path).with_context(|| {
                    format!("open source primary-key file {}", item.path.display())
                })?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                    .with_context(|| format!("open Parquet reader {}", item.path.display()))?;
                let column_idx = builder.schema().index_of(&pk_column).with_context(|| {
                    format!(
                        "primary-key column `{pk_column}` in {}",
                        item.path.display()
                    )
                })?;
                let projection = ProjectionMask::roots(builder.parquet_schema(), [column_idx]);
                let reader = builder
                    .with_projection(projection)
                    .with_batch_size(1 << 20)
                    .build()
                    .with_context(|| format!("build primary-key reader {}", item.path.display()))?;

                let mut found = Vec::with_capacity(item.selection_end - item.selection_start);
                let mut local_row_start = 0u64;
                let mut cursor = item.selection_start;
                for batch in reader {
                    let batch = batch
                        .with_context(|| format!("read primary keys {}", item.path.display()))?;
                    let local_row_end = local_row_start + batch.num_rows() as u64;
                    let values = batch.column(0);
                    while cursor < item.selection_end {
                        let local = selected[cursor].0 - item.file_start;
                        if local >= local_row_end {
                            break;
                        }
                        let row = usize::try_from(local - local_row_start)
                            .context("primary-key row offset exceeds usize")?;
                        let key = key_from_array(values.as_ref(), row, key_type, table)?;
                        found.push((selected[cursor].1, key));
                        cursor += 1;
                    }
                    local_row_start = local_row_end;
                    if cursor == item.selection_end {
                        break;
                    }
                }
                if cursor != item.selection_end {
                    return Err(anyhow!(
                        "source Parquet {} ended before selected row {}",
                        item.path.display(),
                        selected[cursor].0 - item.file_start
                    ));
                }
                Ok(found)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    for file_results in results {
        for (index, key) in file_results {
            output[index] = Some(key);
        }
    }
    let output: Vec<Key> = output
        .into_iter()
        .enumerate()
        .map(|(idx, value)| value.ok_or_else(|| anyhow!("missing sampled primary key {idx}")))
        .collect::<Result<_>>()?;
    eprintln!(
        "[bench] loaded {} real primary keys {} from {} source rows in {:.3}s using {} setup threads; source-row span {}..={} ({:.6}%..{:.6}%), touched {}/{} Parquet files",
        output.len(),
        if rank_coverage.is_some() {
            "with configured rank skew"
        } else {
            "uniformly"
        },
        total_rows,
        started.elapsed().as_secs_f64(),
        setup_threads,
        selected_min,
        selected_max,
        100.0 * selected_min as f64 / total_rows as f64,
        100.0 * selected_max as f64 / total_rows as f64,
        work.len(),
        paths.len(),
    );
    Ok(output)
}

fn report_degree_stats(
    engine: &DiskGraphEngine,
    seed_type: &str,
    samples: usize,
    relation_names: &[String],
    rng_seed: u64,
) -> Result<()> {
    if samples == 0 {
        return Err(anyhow!("--degree-stats-samples must be at least 1"));
    }
    let type_id = *engine
        .node_type_to_id
        .get(seed_type)
        .ok_or_else(|| anyhow!("unknown degree-stat node type `{seed_type}`"))?;
    let shards = engine
        .readers
        .get(type_id)
        .ok_or_else(|| anyhow!("node type `{seed_type}` has no readers"))?;
    let max_id = shards
        .last()
        .map(|shard| shard.max_id)
        .ok_or_else(|| anyhow!("node type `{seed_type}` has no non-empty shards"))?;
    let selected_relations: std::collections::HashSet<usize> = if relation_names.is_empty() {
        engine
            .relation_src_type
            .iter()
            .enumerate()
            .filter_map(|(relation, source)| (*source == type_id).then_some(relation))
            .collect()
    } else {
        relation_names
            .iter()
            .map(|name| {
                engine
                    .rel_name_to_id
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow!("unknown degree-stat relation `{name}`"))
            })
            .collect::<Result<_>>()?
    };
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut degrees = Vec::with_capacity(samples);
    for _ in 0..samples {
        let global_id = rng.gen_range(0..=max_id);
        let shard_idx = engine.find_shard(shards, global_id)?;
        let shard = &shards[shard_idx];
        let view = shard
            .reader
            .edges_for_node(global_id - shard.min_id)
            .with_context(|| format!("read degree for {seed_type} node {global_id}"))?;
        let mut degree = 0u64;
        for segment in view.relation_segments()? {
            let local_relation = segment.layout.relation_id as usize;
            let merged_relation = engine
                .local_to_merged_rel
                .get(type_id)
                .and_then(|relations| relations.get(local_relation))
                .copied()
                .unwrap_or(u16::MAX) as usize;
            if selected_relations.contains(&merged_relation) {
                degree = degree.saturating_add(segment.count as u64);
            }
        }
        degrees.push(degree);
    }
    degrees.sort_unstable();
    let quantile = |per_mille: usize| -> u64 {
        let index = (degrees.len() - 1) * per_mille / 1000;
        degrees[index]
    };
    let sum: u128 = degrees.iter().map(|degree| *degree as u128).sum();
    let mean = sum as f64 / degrees.len() as f64;
    let relation_list = if relation_names.is_empty() {
        "all outgoing relations".to_string()
    } else {
        relation_names.join(",")
    };
    println!(
        "[degree-stats] type={seed_type} samples={} relations={} p50={} p90={} p99={} p99.9={} max_sample={} mean={mean:.4}",
        degrees.len(),
        relation_list,
        quantile(500),
        quantile(900),
        quantile(990),
        quantile(999),
        degrees.last().copied().unwrap_or(0),
    );
    Ok(())
}

// ------------------------------------------------------- /proc/diskstats

/// The counters we difference across the measure window. `time_in_queue` is
/// the weighted busy time whose derivative is the average queue depth — the
/// statistic that separates "the device is saturated" from "the device is
/// idle waiting for someone to ask it for something".
#[derive(Default, Clone, Copy)]
struct DiskStats {
    reads: u64,
    sectors: u64,
    io_ticks_ms: u64,
    time_in_queue_ms: u64,
}

#[derive(Default, Clone, Copy)]
struct FaultStats {
    minor: u64,
    major: u64,
}

#[derive(Default, Clone, Copy)]
struct CpuStats {
    user_s: f64,
    system_s: f64,
}

impl std::ops::Sub for CpuStats {
    type Output = CpuStats;

    fn sub(self, rhs: CpuStats) -> CpuStats {
        CpuStats {
            user_s: (self.user_s - rhs.user_s).max(0.0),
            system_s: (self.system_s - rhs.system_s).max(0.0),
        }
    }
}

fn read_cpu_stats() -> CpuStats {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return CpuStats::default();
    };
    let Some((_, tail)) = stat.rsplit_once(") ") else {
        return CpuStats::default();
    };
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    if ticks <= 0.0 {
        return CpuStats::default();
    }
    CpuStats {
        // tail begins at field 3 (`state`); utime/stime are fields 14/15.
        user_s: fields
            .get(11)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0) as f64
            / ticks,
        system_s: fields
            .get(12)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0) as f64
            / ticks,
    }
}

impl std::ops::Sub for FaultStats {
    type Output = FaultStats;
    fn sub(self, rhs: FaultStats) -> FaultStats {
        FaultStats {
            minor: self.minor.saturating_sub(rhs.minor),
            major: self.major.saturating_sub(rhs.major),
        }
    }
}

fn read_faults() -> FaultStats {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return FaultStats::default();
    };
    let Some((_, tail)) = stat.rsplit_once(") ") else {
        return FaultStats::default();
    };
    let fields: Vec<&str> = tail.split_whitespace().collect();
    FaultStats {
        minor: fields.get(7).and_then(|v| v.parse().ok()).unwrap_or(0),
        major: fields.get(9).and_then(|v| v.parse().ok()).unwrap_or(0),
    }
}

impl std::ops::Sub for DiskStats {
    type Output = DiskStats;
    fn sub(self, rhs: DiskStats) -> DiskStats {
        DiskStats {
            reads: self.reads.saturating_sub(rhs.reads),
            sectors: self.sectors.saturating_sub(rhs.sectors),
            io_ticks_ms: self.io_ticks_ms.saturating_sub(rhs.io_ticks_ms),
            time_in_queue_ms: self.time_in_queue_ms.saturating_sub(rhs.time_in_queue_ms),
        }
    }
}

/// Sum `/proc/diskstats` over `devices`. Empty `devices` means every whole
/// NVMe device (`nvmeXnY`, not its partitions) — the usual instance-store
/// layout. Returns zeros off Linux, where the throughput numbers still stand
/// but the I/O attribution does not.
fn read_diskstats(devices: &[String]) -> DiskStats {
    let Ok(content) = std::fs::read_to_string("/proc/diskstats") else {
        return DiskStats::default();
    };
    let mut total = DiskStats::default();
    for line in content.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 14 {
            continue;
        }
        let name = f[2];
        let wanted = if devices.is_empty() {
            name.starts_with("nvme") && !name.contains('p')
        } else {
            devices.iter().any(|d| d == name)
        };
        if !wanted {
            continue;
        }
        let get = |i: usize| f[i].parse::<u64>().unwrap_or(0);
        total.reads += get(3);
        total.sectors += get(5);
        total.io_ticks_ms += get(12);
        total.time_in_queue_ms += get(13);
    }
    total
}

/// Page cache actually available to this run, in GiB.
///
/// Prefers **this cgroup's** `memory.stat file`, because a memory-capped run is
/// how the disk-heavy regime gets created and the global `Cached` figure says
/// nothing about what the cap left us. Falls back to `/proc/meminfo Cached`
/// when uncapped. The regime label has to come from measured cache, never from
/// the cap itself — a capped run whose working set still fits is not disk-heavy.
fn page_cache_gb() -> f64 {
    if let Some(v) = cgroup_file_cache_gb() {
        return v;
    }
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return f64::NAN;
    };
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Cached:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
            {
                return kb / 1024.0 / 1024.0;
            }
        }
    }
    f64::NAN
}

fn cgroup_file_cache_gb() -> Option<f64> {
    // cgroup v2: "0::/user.slice/...scope"
    let self_cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = self_cgroup
        .lines()
        .find_map(|l| l.strip_prefix("0::"))?
        .trim();
    if rel == "/" {
        return None; // root cgroup — no cap, the global figure is the honest one
    }
    let stat = std::fs::read_to_string(format!("/sys/fs/cgroup{rel}/memory.stat")).ok()?;
    let bytes: f64 = stat
        .lines()
        .find_map(|l| l.strip_prefix("file "))?
        .trim()
        .parse()
        .ok()?;
    Some(bytes / (1 << 30) as f64)
}

// ---------------------------------------------------------------- bench

struct Phase {
    label: String,
    wall: f64,
    nodes: u64,
    edges: u64,
    seeds: u64,
    input_pk_lookups: u64,
    returned_pkeys: u64,
    returned_pk_bytes: u64,
    pk_lookup_cpu_ns: u64,
    pk_output_cpu_ns: u64,
    disk: DiskStats,
    /// Page cache at the START and END of the measure window. The pair is the
    /// steady-state check: if the cache is still filling, the run is measuring
    /// the cold ramp, and a ramp reports inflated throughput and a badly
    /// understated `reads_per_node` (every page is a first-touch miss, so the
    /// working set has not spread yet).
    cache_gb_start: f64,
    cache_gb: f64,
    direct_cache_gb_start: f64,
    gate: (u64, u32),
    digest: u64,
    page_io: diskgraph_index::page_io::PageIoStats,
    faults: FaultStats,
    allocations: AllocStats,
    cpu: CpuStats,
}

/// How much cache growth during the window still counts as steady, and the
/// shortest window worth believing.
///
/// The i8g reference run excludes a **~30 s** shared-structure ramp, so a
/// window shorter than that cannot have cleared it.
const STEADY_CACHE_GROWTH: f64 = 0.10;
const STEADY_MIN_SECONDS: f64 = 30.0;

impl Phase {
    /// `None` when the run looks like steady state; otherwise why it doesn't.
    fn ramp_warning(&self) -> Option<String> {
        let direct_cache_gb = self.page_io.cache_entries as f64 * 4096.0 / (1u64 << 30) as f64;
        let (growth_start, growth_end, cache_name) = if self.page_io.cache_capacity_pages > 0 {
            (
                self.direct_cache_gb_start,
                direct_cache_gb,
                "userspace direct cache",
            )
        } else {
            (self.cache_gb_start, self.cache_gb, "page cache")
        };
        let growth = if growth_start > 0.0 {
            (growth_end - growth_start) / growth_start
        } else if growth_end > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if self.wall < STEADY_MIN_SECONDS {
            return Some(format!(
                "measure window {:.1}s < {STEADY_MIN_SECONDS:.0}s — shorter than the known \
                 shared-structure ramp; throughput is inflated and reads/node understated",
                self.wall
            ));
        }
        if growth > STEADY_CACHE_GROWTH {
            return Some(format!(
                "{cache_name} grew {:.1}% during the window ({:.1} → {:.1} GiB) — still filling, \
                 so this is the cold ramp, not steady state",
                100.0 * growth,
                growth_start,
                growth_end
            ));
        }
        None
    }
}

impl Phase {
    fn report(&self) {
        let per_s = |v: u64| v as f64 / self.wall.max(1e-9);
        let iops = self.disk.reads as f64 / self.wall.max(1e-9);
        let cache_lookups = self
            .page_io
            .cache_hits
            .saturating_add(self.page_io.cache_misses);
        let cache_hit_rate = if cache_lookups == 0 {
            0.0
        } else {
            100.0 * self.page_io.cache_hits as f64 / cache_lookups as f64
        };
        let direct_cache_gb = self.page_io.cache_entries as f64 * 4096.0 / (1u64 << 30) as f64;
        let direct_cache_capacity_gb =
            self.page_io.cache_capacity_pages as f64 * 4096.0 / (1u64 << 30) as f64;
        println!(
            "\n=== {} ===\n\
             wall               : {:.3}s   ({} seeds)\n\
             throughput         : {:.0} nodes/s  |  {:.0} edges/s  |  {:.0} examples/s\n\
             real primary keys : {:.0} input lookups/s | {:.0} returned/s ({:.2} GiB/s), {:.2} lookup + {:.2} output CPU-s\n\
             process CPU        : {:.1} cores ({:.1} user + {:.1} system CPU-s; {:.0}% of {} logical CPUs)\n\
             subgraph           : {:.1} nodes/seed  |  {:.1} edges/seed\n\
             device reads       : {} ({:.0} IOPS, {:.3} reads/node, {:.2} GiB read)\n\
             queue depth (avg)  : {:.1}   |  device busy {:.0}%\n\
             page cache         : {:.1} → {:.1} GiB (start → end of window)\n\
             userspace cache    : {:.1} → {:.1} / {:.1} GiB (start → end / capacity)\n\
             prefetch gate      : {} passes run, smoothed hit rate {}%\n\
             cache page demand  : {} lookups ({:.0}/s, {:.3}/node), {} hits + {} misses ({:.3} misses/node), {:.2}% hit\n\
             explicit storage IO: {} miss pages / {} SQEs ({} multi-page SQEs covering {} pages), {} shard jobs / {} envelopes / {} owner batches (max {} pages), {} bytes, {} loading bypasses, {} evictions\n\
             explicit frames    : {} allocated, {} reused, {} recycled, {} dropped\n\
             heap activity      : {} allocs ({:.4}/node), {:.1} MiB allocated ({:.1} B/node), {} frees\n\
             process faults     : {} minor, {} major\n\
             output digest      : {:016x}",
            self.label,
            self.wall,
            self.seeds,
            per_s(self.nodes),
            per_s(self.edges),
            per_s(self.seeds),
            per_s(self.input_pk_lookups),
            per_s(self.returned_pkeys),
            self.returned_pk_bytes as f64 / (1u64 << 30) as f64 / self.wall.max(1e-9),
            self.pk_lookup_cpu_ns as f64 / 1e9,
            self.pk_output_cpu_ns as f64 / 1e9,
            (self.cpu.user_s + self.cpu.system_s) / self.wall.max(1e-9),
            self.cpu.user_s,
            self.cpu.system_s,
            100.0 * (self.cpu.user_s + self.cpu.system_s)
                / self.wall.max(1e-9)
                / num_cpus() as f64,
            num_cpus(),
            self.nodes as f64 / self.seeds.max(1) as f64,
            self.edges as f64 / self.seeds.max(1) as f64,
            self.disk.reads,
            iops,
            self.disk.reads as f64 / self.nodes.max(1) as f64,
            self.disk.sectors as f64 * 512.0 / (1 << 30) as f64,
            self.disk.time_in_queue_ms as f64 / 1000.0 / self.wall.max(1e-9),
            100.0 * self.disk.io_ticks_ms as f64 / 1000.0 / self.wall.max(1e-9),
            self.cache_gb_start,
            self.cache_gb,
            self.direct_cache_gb_start,
            direct_cache_gb,
            direct_cache_capacity_gb,
            self.gate.0,
            self.gate.1,
            cache_lookups,
            per_s(cache_lookups),
            cache_lookups as f64 / self.nodes.max(1) as f64,
            self.page_io.cache_hits,
            self.page_io.cache_misses,
            self.page_io.cache_misses as f64 / self.nodes.max(1) as f64,
            cache_hit_rate,
            self.page_io.submitted,
            self.page_io.io_ops,
            self.page_io.multi_page_ops,
            self.page_io.multi_page_pages,
            self.page_io.runtime_jobs,
            self.page_io.runtime_envelopes,
            self.page_io.runtime_batches,
            self.page_io.max_batch_pages,
            self.page_io.bytes,
            self.page_io.coalesced,
            self.page_io.evictions,
            self.page_io.frames_allocated,
            self.page_io.frames_reused,
            self.page_io.frames_recycled,
            self.page_io.frames_dropped,
            self.allocations.alloc_calls,
            self.allocations.alloc_calls as f64 / self.nodes.max(1) as f64,
            self.allocations.alloc_bytes as f64 / (1 << 20) as f64,
            self.allocations.alloc_bytes as f64 / self.nodes.max(1) as f64,
            self.allocations.dealloc_calls,
            self.faults.minor,
            self.faults.major,
            self.digest,
        );
        if let Some(why) = self.ramp_warning() {
            println!("*** NOT STEADY STATE: {why}");
            println!(
                "*** Raise --warmup-seeds until the cache stops growing, then --num-seeds \
                      until the window is comfortably past {STEADY_MIN_SECONDS:.0}s."
            );
        }
    }
}

fn validate_page_io_accounting(stats: &diskgraph_index::page_io::PageIoStats) -> Result<()> {
    if stats.io_ops > stats.submitted {
        return Err(anyhow!(
            "page-I/O accounting mismatch: {} SQEs exceed {} submitted pages",
            stats.io_ops,
            stats.submitted
        ));
    }
    if stats.multi_page_ops > stats.io_ops || stats.multi_page_pages > stats.submitted {
        return Err(anyhow!(
            "page-I/O coalescing counters are inconsistent: {} multi-page SQEs / {} pages, {} total SQEs / {} submitted pages",
            stats.multi_page_ops,
            stats.multi_page_pages,
            stats.io_ops,
            stats.submitted
        ));
    }
    if stats.multi_page_ops > 0 && stats.multi_page_pages < stats.multi_page_ops * 2 {
        return Err(anyhow!(
            "page-I/O accounting mismatch: {} multi-page SQEs cover only {} pages",
            stats.multi_page_ops,
            stats.multi_page_pages
        ));
    }
    if stats.bytes > stats.submitted.saturating_mul(4096) {
        return Err(anyhow!(
            "page-I/O accounting mismatch: {} bytes exceed {} submitted 4 KiB pages",
            stats.bytes,
            stats.submitted
        ));
    }
    // With the fixed userspace cache enabled, every cache miss produces one
    // submitted page. io_uring may combine adjacent pages into fewer SQEs,
    // but that must not alter the page-level hit/miss denominator.
    if stats.cache_capacity_pages > 0 && stats.cache_misses != stats.submitted {
        return Err(anyhow!(
            "direct-cache accounting mismatch: {} misses but {} submitted pages",
            stats.cache_misses,
            stats.submitted
        ));
    }
    Ok(())
}

fn hash_array_data(data: &arrow::array::ArrayData, hasher: &mut impl Hasher) {
    data.len().hash(hasher);
    data.offset().hash(hasher);
    for buffer in data.buffers() {
        buffer.as_slice().hash(hasher);
    }
    for child in data.child_data() {
        hash_array_data(child, hasher);
    }
}

fn hash_primary_key_batches(batches: &[arrow_array::RecordBatch]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for batch in batches {
        batch.num_rows().hash(&mut hasher);
        for column in batch.columns() {
            hash_array_data(&column.to_data(), &mut hasher);
        }
    }
    hasher.finish()
}

#[allow(clippy::too_many_arguments)]
fn run_phase(
    label: &str,
    engine: &DiskGraphEngine,
    request_threads: usize,
    state_pool: &StatePool,
    seed_type: &str,
    seeds: SeedBatch<'_>,
    seed_table: Option<&TableReader>,
    batch_size: usize,
    plan: &SamplingPlan,
    flags: &SamplingFlags,
    fetch_features: bool,
    return_pkeys: bool,
    primary_key_columns: &[Option<String>],
    devices: &[String],
    digest_output: bool,
) -> Result<Phase> {
    let nodes = AtomicU64::new(0);
    let edges = AtomicU64::new(0);
    let digest = AtomicU64::new(0);
    let input_pk_lookups = AtomicU64::new(0);
    let returned_pkeys = AtomicU64::new(0);
    let returned_pk_bytes = AtomicU64::new(0);
    let pk_lookup_cpu_ns = AtomicU64::new(0);
    let pk_output_cpu_ns = AtomicU64::new(0);
    let cache_gb_start = page_cache_gb();
    let d0 = read_diskstats(devices);
    let p0 = diskgraph_index::page_io::stats();
    let f0 = read_faults();
    let a0 = alloc_stats();
    let c0 = read_cpu_stats();
    let t0 = Instant::now();
    let next_batch = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let first_error = Mutex::new(None);
    let batch_count = seeds.len().div_ceil(batch_size);
    std::thread::scope(|scope| {
        for worker in 0..request_threads {
            std::thread::Builder::new()
                .name(format!("dg-bench-request-{worker}"))
                .stack_size(4 << 20)
                .spawn_scoped(scope, || {
                    let mut request = BatchSamplingRequest {
                        dataset_id: diskgraph_core::dataset::DatasetId::nil(),
                        seed_nodes: [(seed_type.to_string(), Vec::with_capacity(batch_size))]
                            .into(),
                        seed_timestamps: None,
                        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
                        plan: plan.clone(),
                        flags: flags.clone(),
                    };
                    let mut output = InMemoryOutput::new();
                    // Keep feature-id staging capacity for the lifetime of the
                    // request worker. Rebuilding a Vec<Vec<u64>> for every
                    // sample made the benchmark itself allocate in the hot
                    // path even after SamplingState had become reusable.
                    let mut feature_ids: Vec<Vec<u64>> = (0..engine.node_decoders.len())
                        .map(|_| Vec::new())
                        .collect();
                    while !cancelled.load(Ordering::Relaxed) {
                        let batch_idx = next_batch.fetch_add(1, Ordering::Relaxed);
                        if batch_idx >= batch_count {
                            break;
                        }
                        let start = batch_idx * batch_size;
                        let end = (start + batch_size).min(seeds.len());
                        let chunk = seeds.slice(start, end);
                        let result = (|| -> Result<()> {
                            let batch_seeds = request
                                .seed_nodes
                                .get_mut(seed_type)
                                .expect("benchmark seed type initialized once per worker");
                            batch_seeds.clear();
                            match chunk {
                                SeedBatch::NodeIds(ids) => batch_seeds.extend_from_slice(ids),
                                SeedBatch::PrimaryKeys(keys) => {
                                    let table = seed_table.ok_or_else(|| {
                                        anyhow!("primary-key seeds require a seed table reader")
                                    })?;
                                    let lookup_started = Instant::now();
                                    for key in keys {
                                        let node_id = node_id_from_known_source_pk(table, key)
                                            .map_err(|error| {
                                                anyhow!("primary-key lookup failed: {error}")
                                            })?
                                            .ok_or_else(|| {
                                                anyhow!(
                                                    "known source primary key was absent from the index"
                                                )
                                            })?;
                                        batch_seeds.push(node_id);
                                    }
                                    input_pk_lookups
                                        .fetch_add(keys.len() as u64, Ordering::Relaxed);
                                    pk_lookup_cpu_ns.fetch_add(
                                        lookup_started.elapsed().as_nanos().min(u64::MAX as u128)
                                            as u64,
                                        Ordering::Relaxed,
                                    );
                                }
                            }
                            output.clear();
                            let mut state = state_pool.acquire();
                            engine
                                .sample(&request, &mut state, &mut output)
                                .map_err(|e| anyhow!("sampling batch failed: {e}"))?;
                            if fetch_features {
                                for ids in &mut feature_ids {
                                    ids.clear();
                                }
                                for (type_id, type_state) in &state.type_states {
                                    if type_state.nodes.is_empty()
                                        || engine
                                            .node_decoders
                                            .get(*type_id)
                                            .and_then(|decoder| decoder.as_ref())
                                            .is_none()
                                    {
                                        continue;
                                    }
                                    let ids = &mut feature_ids[*type_id];
                                    ids.reserve(type_state.nodes.len().saturating_sub(ids.len()));
                                    ids.extend(type_state.nodes.iter().map(|node| node.global_id));
                                }
                                for (type_id, node_ids) in feature_ids.iter().enumerate() {
                                    if node_ids.is_empty() {
                                        continue;
                                    }
                                    engine
                                        .fetch_features_for_type(
                                            type_id,
                                            node_ids,
                                            &[],
                                            &mut output,
                                        )
                                        .map_err(|error| {
                                            anyhow!("feature fetch failed: {error}")
                                        })?;
                                }
                            }
                            if return_pkeys {
                                for ids in &mut feature_ids {
                                    ids.clear();
                                }
                                for (type_id, type_state) in &state.type_states {
                                    if type_state.nodes.is_empty() {
                                        continue;
                                    }
                                    let ids = &mut feature_ids[*type_id];
                                    ids.reserve(type_state.nodes.len().saturating_sub(ids.len()));
                                    ids.extend(type_state.nodes.iter().map(|node| node.global_id));
                                }
                                let output_started = Instant::now();
                                for (type_id, node_ids) in feature_ids.iter().enumerate() {
                                    if node_ids.is_empty() {
                                        continue;
                                    }
                                    let pk_column = primary_key_columns
                                        .get(type_id)
                                        .and_then(Option::as_ref)
                                        .ok_or_else(|| {
                                            anyhow!(
                                                "sampled node type {} has no primary-key column",
                                                engine
                                                    .id_to_node_type
                                                    .get(type_id)
                                                    .map(String::as_str)
                                                    .unwrap_or("<unknown>")
                                            )
                                        })?;
                                    let first_batch = output.feature_batches.len();
                                    engine
                                        .fetch_features_for_type(
                                            type_id,
                                            node_ids,
                                            std::slice::from_ref(pk_column),
                                            &mut output,
                                        )
                                        .map_err(|error| {
                                            anyhow!("primary-key output fetch failed: {error}")
                                        })?;
                                    let new_batches = &output.feature_batches[first_batch..];
                                    let rows: u64 = new_batches
                                        .iter()
                                        .map(|batch| batch.num_rows() as u64)
                                        .sum();
                                    let bytes: u64 = new_batches
                                        .iter()
                                        .map(|batch| batch.get_array_memory_size() as u64)
                                        .sum();
                                    returned_pkeys.fetch_add(rows, Ordering::Relaxed);
                                    returned_pk_bytes.fetch_add(bytes, Ordering::Relaxed);
                                    if digest_output {
                                        digest.fetch_add(
                                            hash_primary_key_batches(new_batches),
                                            Ordering::Relaxed,
                                        );
                                    }
                                }
                                pk_output_cpu_ns.fetch_add(
                                    output_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                                    Ordering::Relaxed,
                                );
                            }
                            nodes.fetch_add(state.stats.total_nodes, Ordering::Relaxed);
                            edges.fetch_add(state.stats.total_edges, Ordering::Relaxed);
                            if digest_output {
                                // Hash WHICH nodes were drawn for WHICH seed — the thing the
                                // determinism gate exists to protect — as a multiset sum, so
                                // it is invariant to emission order.
                                //
                                // Deliberately NOT `output.edges`: those carry *local* ids,
                                // which are assigned by dedup insertion order rather than by
                                // sampling identity, so they differ run-to-run even when the
                                // identical neighbours were drawn (two prefetch-OFF runs
                                // disagree on them, single-threaded and seeded). Global ids
                                // paired with the seed index are the invariant.
                                let mut sum = 0u64;
                                for (type_id, ts) in state.type_states.iter() {
                                    for n in ts.nodes.iter() {
                                        let mut h =
                                            std::collections::hash_map::DefaultHasher::new();
                                        (*type_id, n.seed_index, n.global_id).hash(&mut h);
                                        sum = sum.wrapping_add(h.finish());
                                    }
                                }
                                digest.fetch_add(sum, Ordering::Relaxed);
                            }
                            Ok(())
                        })();
                        if let Err(error) = result {
                            cancelled.store(true, Ordering::Relaxed);
                            let mut first = first_error.lock().unwrap();
                            if first.is_none() {
                                *first = Some(error);
                            }
                            break;
                        }
                    }
                })
                .expect("failed to spawn fixed benchmark request worker");
        }
    });
    let allocations = alloc_stats() - a0;
    if let Some(error) = first_error.into_inner().unwrap() {
        return Err(error);
    }
    let wall = t0.elapsed().as_secs_f64();
    let disk = read_diskstats(devices) - d0;
    let p1 = diskgraph_index::page_io::stats();
    let f1 = read_faults();
    let c1 = read_cpu_stats();

    let phase = Phase {
        label: label.to_string(),
        wall,
        nodes: nodes.load(Ordering::Relaxed),
        edges: edges.load(Ordering::Relaxed),
        seeds: seeds.len() as u64,
        input_pk_lookups: input_pk_lookups.load(Ordering::Relaxed),
        returned_pkeys: returned_pkeys.load(Ordering::Relaxed),
        returned_pk_bytes: returned_pk_bytes.load(Ordering::Relaxed),
        pk_lookup_cpu_ns: pk_lookup_cpu_ns.load(Ordering::Relaxed),
        pk_output_cpu_ns: pk_output_cpu_ns.load(Ordering::Relaxed),
        disk,
        cache_gb_start,
        cache_gb: page_cache_gb(),
        direct_cache_gb_start: p0.cache_entries as f64 * 4096.0 / (1u64 << 30) as f64,
        gate: diskgraph_index::prefetch::stats(),
        digest: digest.load(Ordering::Relaxed),
        page_io: diskgraph_index::page_io::PageIoStats {
            submitted: p1.submitted.saturating_sub(p0.submitted),
            io_ops: p1.io_ops.saturating_sub(p0.io_ops),
            multi_page_ops: p1.multi_page_ops.saturating_sub(p0.multi_page_ops),
            multi_page_pages: p1.multi_page_pages.saturating_sub(p0.multi_page_pages),
            runtime_jobs: p1.runtime_jobs.saturating_sub(p0.runtime_jobs),
            runtime_envelopes: p1.runtime_envelopes.saturating_sub(p0.runtime_envelopes),
            runtime_batches: p1.runtime_batches.saturating_sub(p0.runtime_batches),
            max_batch_pages: p1.max_batch_pages,
            cache_hits: p1.cache_hits.saturating_sub(p0.cache_hits),
            cache_misses: p1.cache_misses.saturating_sub(p0.cache_misses),
            coalesced: p1.coalesced.saturating_sub(p0.coalesced),
            evictions: p1.evictions.saturating_sub(p0.evictions),
            bytes: p1.bytes.saturating_sub(p0.bytes),
            frames_allocated: p1.frames_allocated.saturating_sub(p0.frames_allocated),
            frames_reused: p1.frames_reused.saturating_sub(p0.frames_reused),
            frames_recycled: p1.frames_recycled.saturating_sub(p0.frames_recycled),
            frames_dropped: p1.frames_dropped.saturating_sub(p0.frames_dropped),
            cache_capacity_pages: p1.cache_capacity_pages,
            cache_entries: p1.cache_entries,
        },
        faults: f1 - f0,
        allocations,
        cpu: c1 - c0,
    };
    if return_pkeys && phase.returned_pkeys != phase.nodes {
        return Err(anyhow!(
            "primary-key output cardinality mismatch: sampled {} nodes but returned {} keys",
            phase.nodes,
            phase.returned_pkeys
        ));
    }
    validate_page_io_accounting(&phase.page_io)?;
    Ok(phase)
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let index: PathBuf = PathBuf::from(args.req("index")?);
    let seed_type = args.req("seed-type")?.to_string();
    let layers: usize = args.opt("layers", 3usize)?;
    let fanout: u32 = args.opt("fanout", 10u32)?;
    let num_seeds: usize = args.opt("num-seeds", 200_000usize)?;
    let warmup_seeds: usize = args.opt("warmup-seeds", 20_000usize)?;
    // Three all-relation hops produce about 1K output nodes/root on the alt
    // graph. Keep enough roots together to amortize shard planning without
    // allowing each concurrent request to retain millions of output nodes.
    let batch_size: usize = args.opt("batch-size", 256usize)?;
    let threads: usize = args.opt("threads", num_cpus())?;
    let rng_seed: u64 = args.opt("seed", 42u64)?;
    let prefetch = args.opt("prefetch", "off".to_string())?;
    let backend = args.opt("backend", "mmap".to_string())?;
    let io_qd: usize = args.opt("io-qd", 256usize)?;
    let io_rings: usize = args.opt("io-rings", 8usize)?;
    let io_batch_us: u64 = args.opt("io-batch-us", 0u64)?;
    let io_batch_mult: usize = args.opt("io-batch-mult", 1usize)?;
    let io_coalesce_pages: usize = args.opt("io-coalesce-pages", 16usize)?;
    let io_eager_node_pages: usize = args.opt("io-eager-node-pages", 0usize)?;
    let direct_cache_gb: usize = args.opt("direct-cache-gb", 0usize)?;
    let fetch_features = args.flag("fetch-features");
    let pkey_e2e = args.flag("pkey-e2e");
    let seed_source_config = args.opt("seed-source-config", String::new())?;
    let seed_rank_coverage_raw = args.opt("seed-rank-coverage", String::new())?;
    let seed_rank_permutation_seed = args.opt("seed-rank-permutation-seed", 0u64)?;
    let degree_stats_samples = args.opt("degree-stats-samples", 0usize)?;
    let degree_stats_relations: Vec<String> = args
        .opt("degree-stats-relations", String::new())?
        .split(',')
        .filter(|relation| !relation.is_empty())
        .map(str::to_string)
        .collect();
    let seed_rank_coverage = if seed_rank_coverage_raw.is_empty() {
        None
    } else {
        Some(SeedRankCoverage::parse(
            &seed_rank_coverage_raw,
            seed_rank_permutation_seed,
        )?)
    };
    if pkey_e2e && fetch_features {
        return Err(anyhow!(
            "--pkey-e2e fetches only returned primary keys and cannot be combined with --fetch-features"
        ));
    }
    if pkey_e2e && seed_source_config.is_empty() {
        return Err(anyhow!("--pkey-e2e requires --seed-source-config"));
    }
    if seed_rank_coverage.is_some() && !pkey_e2e {
        return Err(anyhow!("--seed-rank-coverage requires --pkey-e2e"));
    }
    let cpu_profile = args.opt("cpu-profile", String::new())?;
    let profile_frequency: i32 = args.opt("profile-frequency", 500i32)?;
    let count_allocations = args.flag("count-allocations");
    let prewarm_measurement = args.flag("prewarm-measurement");
    if profile_frequency <= 0 {
        return Err(anyhow!("--profile-frequency must be positive"));
    }
    if !matches!(backend.as_str(), "mmap" | "buffered" | "direct") {
        return Err(anyhow!("--backend must be mmap|buffered|direct"));
    }
    std::env::set_var("DG_EDGE_BACKEND", &backend);
    std::env::set_var("DG_IO_QD", io_qd.to_string());
    std::env::set_var("DG_IO_RINGS", io_rings.to_string());
    std::env::set_var("DG_IO_BATCH_US", io_batch_us.to_string());
    std::env::set_var("DG_IO_BATCH_MULT", io_batch_mult.to_string());
    std::env::set_var("DG_IO_COALESCE_PAGES", io_coalesce_pages.to_string());
    std::env::set_var("DG_IO_EAGER_NODE_PAGES", io_eager_node_pages.to_string());
    std::env::set_var("DG_DIRECT_CACHE_GB", direct_cache_gb.to_string());
    let verify = args.flag("verify");
    let devices: Vec<String> = args
        .opt("devices", String::new())?
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();

    let metadata: DatasetMetadata =
        serde_json::from_slice(&std::fs::read(index.join("index_metadata.json"))?)
            .context("parse index_metadata.json")?;
    let max_id = metadata
        .node_types
        .iter()
        .find(|n| n.name == seed_type)
        .ok_or_else(|| {
            anyhow!(
                "seed type `{seed_type}` not in index; have: {:?}",
                metadata
                    .node_types
                    .iter()
                    .map(|n| &n.name)
                    .collect::<Vec<_>>()
            )
        })?
        .max_id;

    // The engine's pool must be sized too, not left at 1: `process_layer` runs
    // its per-shard map phase via `engine_pool.install(...)`, so a 1-thread
    // engine pool funnels EVERY concurrent batch through a single thread while
    // the outer workers block — the sampler then looks CPU-bound at ~4% of the
    // device ceiling, which is a harness artifact, not a read-path property.
    let engine_threads: usize = args.opt("engine-threads", threads)?;
    // Request concurrency uses fixed ordinary threads below. Do not put it in
    // a second Rayon pool: a Rayon worker blocked in the engine pool's
    // `install()` steals and recursively launches more requests, making state,
    // page-lease, stack, and allocator memory effectively unbounded.
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        std::env::set_var("RUST_MIN_STACK", (8 << 20).to_string());
    }
    let seed_table = if pkey_e2e {
        let only_tables = [seed_type.clone()].into_iter().collect();
        let reader = GraphIndexReader::open_subset(
            &index.join("index_metadata.json"),
            &index,
            &only_tables,
        )?;
        Some(
            reader
                .table(&seed_type)
                .ok_or_else(|| anyhow!("seed table `{seed_type}` was not opened from the index"))?,
        )
    } else {
        None
    };
    let mut engine = DiskGraphEngine::load(&index, Some(rng_seed), Some(engine_threads))?;
    let max_neighbors_per_seed: Option<u32> = args
        .0
        .get("max-neighbors-per-seed")
        .map(|value| value.parse())
        .transpose()
        .context("invalid --max-neighbors-per-seed")?;
    if max_neighbors_per_seed == Some(0) {
        return Err(anyhow!("--max-neighbors-per-seed must be at least 1"));
    }
    engine.set_max_neighbors_per_seed(max_neighbors_per_seed);
    if degree_stats_samples > 0 {
        return report_degree_stats(
            &engine,
            &seed_type,
            degree_stats_samples,
            &degree_stats_relations,
            rng_seed,
        );
    }
    let primary_key_columns: Vec<Option<String>> = engine
        .id_to_node_type
        .iter()
        .map(|name| {
            metadata
                .node_types
                .iter()
                .find(|node_type| &node_type.name == name)
                .and_then(|node_type| node_type.primary_key_map.as_ref())
                .map(|pk| pk.column.clone())
        })
        .collect();
    if args.flag("prefault-offsets") {
        let started = Instant::now();
        let checksum = engine.prefault_offsets();
        println!(
            "[bench] prefaulted offsets in {:.3}s checksum={checksum:016x}",
            started.elapsed().as_secs_f64()
        );
    }
    // No named relations: `default_fanout` traverses every relation the node
    // type has, which is what a disk-heavy sweep wants.
    let mode = if args.flag("temporal") {
        SamplingMode::TemporalMostRecent
    } else {
        SamplingMode::Uniform
    };
    // `--plan "rel:fanout,rel:fanout;rel:fanout,..."` — `;` separates layers.
    //
    // PER-LAYER relations are not a nicety, they decide the regime. The i8g
    // reference's `pin-uniform` fans out from `pin` at hop 0 and then uses the
    // REVERSE relations at hop 1, so the boards/queries/items it reached fan
    // back out into pins. Reusing hop-0's relations at hop 1 instead leaves
    // those node types as dead ends: a far narrower and much more
    // cache-friendly traversal, which measured `reads_per_node` 0.121 against
    // the reference's 0.60 — i.e. not the disk-bound workload at all.
    //
    // With no `--plan`, `default_fanout` traverses every relation at every
    // layer, which is a different (also cache-friendly) workload again.
    let plan_arg = args.opt("plan", String::new())?;
    let spec_of = |f: u32| FanoutSpec {
        fanout: f,
        mode: mode.clone(),
        absent_key_policy: AbsentKeyPolicy::UniformFallback,
    };
    let plan = if plan_arg.is_empty() {
        SamplingPlan {
            layers: (0..layers)
                .map(|_| LayerSpec {
                    fanouts: HashMap::new(),
                    default_fanout: Some(spec_of(fanout)),
                })
                .collect(),
        }
    } else {
        let mut layers_out = Vec::new();
        for layer in plan_arg.split(';').filter(|l| !l.is_empty()) {
            let mut fanouts = HashMap::new();
            for entry in layer.split(',').filter(|e| !e.is_empty()) {
                let (rel, f) = entry
                    .rsplit_once(':')
                    .ok_or_else(|| anyhow!("--plan entry `{entry}` must be `relation:fanout`"))?;
                let f: u32 = f.parse().map_err(|_| anyhow!("bad fanout in `{entry}`"))?;
                fanouts.insert(rel.to_string(), spec_of(f));
            }
            layers_out.push(LayerSpec {
                fanouts,
                default_fanout: None,
            });
        }
        SamplingPlan { layers: layers_out }
    };
    let canonical_plan = if plan_arg.is_empty() {
        format!("default:{layers}x{fanout}:{mode:?}")
    } else {
        plan_arg.clone()
    };
    let mut plan_hasher = std::collections::hash_map::DefaultHasher::new();
    canonical_plan.hash(&mut plan_hasher);
    let plan_digest = plan_hasher.finish();
    let flags = SamplingFlags {
        replace: false,
        directed: true,
        disjoint: true,
        return_edge_id: false,
        adaptive: false,
        fused_feature_fetching: args.flag("fused"),
        swap_edge_direction: false,
    };

    // Every phase gets its OWN fresh seeds. Reusing seeds would let a later
    // phase read a page cache the earlier one warmed, which is exactly the
    // contamination that made the first round of these numbers wrong.
    let measured_streams = if prefetch == "both" { 2 } else { 1 };
    let total_seeds = warmup_seeds + num_seeds * measured_streams;
    let all = if pkey_e2e {
        SeedStorage::PrimaryKeys(load_primary_keys(
            Path::new(&seed_source_config),
            &seed_type,
            total_seeds,
            rng_seed,
            seed_table
                .as_deref()
                .expect("pkey mode opened the seed table"),
            seed_rank_coverage.as_ref(),
        )?)
    } else {
        let mut rng = StdRng::seed_from_u64(rng_seed);
        SeedStorage::NodeIds(
            (0..total_seeds)
                .map(|_| rng.gen_range(0..=max_id))
                .collect(),
        )
    };
    let all = all.batch();
    let warmup = all.slice(0, warmup_seeds);
    let seeds_a = all.slice(warmup_seeds, warmup_seeds + num_seeds);
    let seeds_b = if measured_streams == 2 {
        all.slice(warmup_seeds + num_seeds, total_seeds)
    } else {
        seeds_a
    };

    let effective_layers = plan.layers.len();
    let seed_rank_skew = seed_rank_coverage.is_some();
    println!(
        "[bench] index={} seed_type={seed_type} max_id={max_id} layers={effective_layers} fanout={fanout} \
         mode={mode:?} threads={threads} engine_threads={engine_threads} batch={batch_size} \
         seeds={num_seeds} prefetch={prefetch} backend={backend} fetch_features={fetch_features} io_rings={io_rings} io_qd={io_qd} io_batch_mult={io_batch_mult} io_batch_us={io_batch_us} io_coalesce_pages={io_coalesce_pages} io_eager_node_pages={io_eager_node_pages} \
         direct_cache_gb={direct_cache_gb} pkey_e2e={pkey_e2e} max_neighbors_per_seed={max_neighbors_per_seed:?} seed_rank_skew={seed_rank_skew} allocator={ALLOCATOR_NAME}",
        index.display()
    );
    println!(
        "[bench] canonical_plan={} plan_digest={:016x}",
        canonical_plan, plan_digest
    );
    println!(
        "[bench] page cache before warmup: {:.1} GiB",
        page_cache_gb()
    );

    let phases: Vec<(&str, bool, SeedBatch<'_>)> = match prefetch.as_str() {
        "off" => vec![("prefetch=off", false, seeds_a)],
        "on" => vec![("prefetch=on", true, seeds_a)],
        // A/B in one process on disjoint fresh seeds. Same cache state going
        // in, so the delta is the prefetch effect and not a cold-start
        // difference — but the shared-structure pages the first phase warms do
        // help the second, so treat `both` as directional and confirm the
        // headline number with two separate `off`/`on` processes.
        "both" => vec![
            ("prefetch=off", false, seeds_a),
            ("prefetch=on", true, seeds_b),
        ],
        other => return Err(anyhow!("--prefetch must be off|on|both, got `{other}`")),
    };

    // Production keeps and reuses sampling states. Reuse the same pool across
    // warmup and measurement so the discarded phase actually pays capacity
    // growth and allocator costs instead of throwing that work away.
    let state_pool = StatePool::new();
    diskgraph_index::prefetch::set_enabled(false);
    let w = run_phase(
        "warmup (discarded)",
        &engine,
        threads,
        &state_pool,
        &seed_type,
        warmup,
        seed_table.as_deref(),
        batch_size,
        &plan,
        &flags,
        fetch_features,
        pkey_e2e,
        &primary_key_columns,
        &devices,
        false,
    )?;
    // The warmup's own cache trajectory is what tells you whether it was long
    // enough: if the cache is still climbing when the warmup ends, the measure
    // window inherits the ramp.
    eprintln!(
        "[bench] warmup done in {:.1}s ({} seeds), page cache {:.1} → {:.1} GiB (+{:.1}%)",
        w.wall,
        w.seeds,
        w.cache_gb_start,
        w.cache_gb,
        100.0 * (w.cache_gb - w.cache_gb_start) / w.cache_gb_start.max(1e-9),
    );

    let mut results = Vec::new();
    if prewarm_measurement {
        eprintln!(
            "[bench] prewarming the exact measurement seeds; final rates are hot-path diagnostics, not independent-random steady-state rates"
        );
        for (label, on, seeds) in &phases {
            diskgraph_index::prefetch::set_enabled(*on);
            let prewarm = run_phase(
                &format!("{label} measurement prewarm (discarded)"),
                &engine,
                threads,
                &state_pool,
                &seed_type,
                *seeds,
                seed_table.as_deref(),
                batch_size,
                &plan,
                &flags,
                fetch_features,
                pkey_e2e,
                &primary_key_columns,
                &devices,
                false,
            )?;
            prewarm.report();
        }
    }
    // Deliberately start profiling after warmup: initialization, prefaulting,
    // and cache ramp are different workloads from the steady sampling path.
    let profile_guard = if cpu_profile.is_empty() {
        None
    } else {
        eprintln!("[bench] profiling measured phase at {profile_frequency} Hz -> {cpu_profile}");
        Some(ProfilerGuard::new(profile_frequency).context("start measured-phase profiler")?)
    };
    ALLOC_TRACKING.store(count_allocations, Ordering::Relaxed);
    for (label, on, seeds) in &phases {
        diskgraph_index::prefetch::set_enabled(*on);
        let phase = run_phase(
            label,
            &engine,
            threads,
            &state_pool,
            &seed_type,
            *seeds,
            seed_table.as_deref(),
            batch_size,
            &plan,
            &flags,
            fetch_features,
            pkey_e2e,
            &primary_key_columns,
            &devices,
            verify,
        )?;
        phase.report();
        results.push(phase);
    }
    ALLOC_TRACKING.store(false, Ordering::Relaxed);

    if let Some(guard) = profile_guard {
        let report = guard
            .report()
            .build()
            .context("build measured-phase profile")?;
        let profile = report.pprof().context("encode measured-phase profile")?;
        std::fs::write(&cpu_profile, profile.encode_to_vec())
            .with_context(|| format!("write measured-phase profile to {cpu_profile}"))?;
        eprintln!("[bench] wrote measured-phase profile to {cpu_profile}");
    }

    if results.len() == 2 {
        let (a, b) = (&results[0], &results[1]);
        let rate = |p: &Phase| p.nodes as f64 / p.wall.max(1e-9);
        let iops = |p: &Phase| p.disk.reads as f64 / p.wall.max(1e-9);
        println!(
            "\n=== A/B ===\nnodes/s {:.0} -> {:.0}  ({:+.1}%)\nIOPS    {:.0} -> {:.0}  ({:+.1}%)",
            rate(a),
            rate(b),
            100.0 * (rate(b) / rate(a) - 1.0),
            iops(a),
            iops(b),
            100.0 * (iops(b) / iops(a) - 1.0),
        );
    }

    // Determinism gate at index scale: same seeds, both modes, digests must
    // match. A throughput win that changed which neighbours were drawn would
    // be worthless, so this runs before any number is believed.
    if verify {
        println!("\n=== determinism verify (same seeds) ===");
        let vseeds = seeds_a.slice(0, num_seeds.min(20_000));
        let once = |label: &str, on: bool| -> Result<Phase> {
            diskgraph_index::prefetch::set_enabled(on);
            run_phase(
                label,
                &engine,
                threads,
                &state_pool,
                &seed_type,
                vseeds,
                seed_table.as_deref(),
                batch_size,
                &plan,
                &flags,
                fetch_features,
                pkey_e2e,
                &primary_key_columns,
                &devices,
                true,
            )
        };
        // Two OFF runs first. Without that control an unstable digest looks
        // like a prefetch bug when it is really the engine's own run-to-run
        // ordering, and the whole gate is uninterpretable.
        let off_a = once("verify off (control A)", false)?;
        let off_b = once("verify off (control B)", false)?;
        let on = once("verify on", true)?;

        let control_stable = off_a.digest == off_b.digest;
        println!(
            "digest  offA={:016x}  offB={:016x}  on={:016x}",
            off_a.digest, off_b.digest, on.digest
        );
        println!(
            "counts  nodes off={}/{} on={}   edges off={}/{} on={}",
            off_a.nodes, off_b.nodes, on.nodes, off_a.edges, off_b.edges, on.edges
        );

        if let Some(cap) = max_neighbors_per_seed {
            let max_nodes = (u64::from(cap) + 1).saturating_mul(vseeds.len() as u64);
            let node_counts = [off_a.nodes, off_b.nodes, on.nodes];
            if node_counts.iter().any(|count| *count > max_nodes) {
                return Err(anyhow!(
                    "CAPPED BFS FAILURE: a verification run returned more than {cap} distinct \
                     neighbors per seed"
                ));
            }
            let node_min = node_counts.iter().copied().min().unwrap_or(0);
            let node_max = node_counts.iter().copied().max().unwrap_or(0);
            let node_drift = node_max.saturating_sub(node_min) as f64 / node_max.max(1) as f64;
            if node_drift > 0.005 {
                return Err(anyhow!(
                    "CAPPED BFS FAILURE: node cardinality drifted {:.3}% across identical-root \
                     verification runs (allowed 0.5%)",
                    node_drift * 100.0
                ));
            }
            let edge_counts = [off_a.edges, off_b.edges, on.edges];
            let edge_min = edge_counts.iter().copied().min().unwrap_or(0);
            let edge_max = edge_counts.iter().copied().max().unwrap_or(0);
            let edge_drift = edge_max.saturating_sub(edge_min) as f64 / edge_max.max(1) as f64;
            println!(
                "VERIFY OK — capped approximate BFS respected the {cap}-neighbor bound; node \
                 count drift {:.3}% and edge count drift {:.3}%. Frontier completion order may \
                 change which equal-hop edges are retained, so byte digests are informational.",
                node_drift * 100.0,
                edge_drift * 100.0,
            );
            return Ok(());
        }

        if off_a.nodes != on.nodes || off_a.edges != on.edges {
            return Err(anyhow!(
                "DETERMINISM FAILURE: prefetch changed the sampled node/edge COUNTS — \
                 every throughput number from this build is void"
            ));
        }
        match (control_stable, off_a.digest == on.digest) {
            (true, true) => println!(
                "VERIFY OK — prefetch is output-identical on {} seeds",
                vseeds.len()
            ),
            (true, false) => {
                return Err(anyhow!(
                    "DETERMINISM FAILURE: off is self-consistent but prefetch changed the output \
                     ordering — every throughput number from this build is void"
                ))
            }
            (false, _) => println!(
                "INCONCLUSIVE — the engine's own output ordering is not stable run-to-run \
                 (offA != offB), so the digest cannot attribute anything to prefetch. \
                 Node and edge counts DO match ({} nodes, {} edges), which is the part this \
                 harness can still check.",
                on.nodes, on.edges
            ),
        }
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_rank_coverage_matches_pinterest_breakpoints() {
        let coverage = SeedRankCoverage::parse("0.0106:0.60,0.0389:0.80,1:1", 1102).unwrap();
        let rows = 1_000_000u64;
        let (a, b) = seed_rank_affine_parameters(rows, 1102);

        let at_sixty = coverage.source_row(0.60, rows, (a, b));
        let rank_at_sixty = (0.0106 * rows as f64) as u64;
        assert_eq!(
            at_sixty,
            seed_rank_affine_permute(rank_at_sixty, rows, a, b)
        );

        let at_seventy = coverage.source_row(0.70, rows, (a, b));
        let position = (0.70 - 0.60) / (0.80 - 0.60);
        let rank_at_seventy = ((0.0106 + position * (0.0389 - 0.0106)) * rows as f64) as u64;
        assert_eq!(
            at_seventy,
            seed_rank_affine_permute(rank_at_seventy, rows, a, b)
        );

        let almost_all = coverage.source_row(1.0 - f64::EPSILON, rows, (a, b));
        assert_eq!(almost_all, seed_rank_affine_permute(rows - 1, rows, a, b));
    }

    #[test]
    fn seed_rank_coverage_requires_a_complete_cdf() {
        assert!(SeedRankCoverage::parse("0.01:0.6", 1).is_err());
        assert!(SeedRankCoverage::parse("0.02:0.6,0.01:0.8,1:1", 1).is_err());
    }

    #[test]
    fn direct_page_accounting_distinguishes_lookups_pages_and_sqes() {
        let valid = diskgraph_index::page_io::PageIoStats {
            submitted: 20,
            io_ops: 10,
            multi_page_ops: 5,
            multi_page_pages: 15,
            cache_hits: 80,
            cache_misses: 20,
            bytes: 20 * 4096,
            cache_capacity_pages: 1,
            ..Default::default()
        };
        validate_page_io_accounting(&valid).unwrap();

        let mut invalid = valid;
        invalid.cache_misses = 21;
        assert!(validate_page_io_accounting(&invalid)
            .unwrap_err()
            .to_string()
            .contains("21 misses but 20 submitted pages"));
    }
}
