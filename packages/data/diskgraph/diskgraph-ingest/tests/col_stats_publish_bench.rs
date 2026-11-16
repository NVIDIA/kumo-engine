// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peak-RSS and wall-time bench for publishing column statistics.
//!
//! cargo test --release -p diskgraph-ingest --test col_stats_publish_bench -- --ignored --nocapture
//! Sizes: DG_COLSTATS_BENCH_COLS (80), DG_COLSTATS_BENCH_DISTINCT (1000000),
//! DG_COLSTATS_BENCH_ROWS (2 x distinct). Output: DG_COLSTATS_BENCH_OUT (tempdir).

use diskgraph_ingest::col_stats::{
    write_col_stats, CategoryCountStrategy, ColStatsTable, FileColStats, TableCategoryCounts,
};
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType, SemanticType};
use diskgraph_ingest::io::{Dest, OutputTarget};
use diskgraph_rowpack::ColumnValueObserver;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn spec(name: String, dtype: IngestedType, stype: SemanticType, delim: Option<&str>) -> ColumnSpec {
    ColumnSpec {
        name,
        source_dtype: None,
        ingested_dtype: dtype,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: Some(stype),
        multicategorical_delimiter: delim.map(str::to_string),
    }
}

/// Every 8th column is an int64 Id, every 8th+1 a comma-delimited
/// multicategorical, the rest string categoricals; plus one numerical.
fn table_columns(first: usize, count: usize) -> Vec<ColumnSpec> {
    let mut cols: Vec<ColumnSpec> = (first..first + count)
        .map(|c| match c % 8 {
            0 => spec(
                format!("id_{c:02}"),
                IngestedType::Int64,
                SemanticType::Id,
                None,
            ),
            1 => spec(
                format!("tags_{c:02}"),
                IngestedType::String,
                SemanticType::Multicategorical,
                Some(","),
            ),
            _ => spec(
                format!("cat_{c:02}"),
                IngestedType::String,
                SemanticType::Categorical,
                None,
            ),
        })
        .collect();
    cols.push(spec(
        format!("num_{first:02}"),
        IngestedType::Float64,
        SemanticType::Numerical,
        None,
    ));
    cols
}

fn populate(
    cols: &[ColumnSpec],
    distinct: u64,
    rows: u64,
) -> (FileColStats, Arc<TableCategoryCounts>) {
    let stypes: Vec<_> = cols.iter().map(|c| c.semantic_type).collect();
    let table_cat = Arc::new(TableCategoryCounts::new(
        CategoryCountStrategy::Hybrid,
        &stypes,
        false,
    ));
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = 100_000u64;
    let next = AtomicU64::new(0);
    let mut parts: Vec<(u64, FileColStats)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    let mut buf = String::new();
                    loop {
                        let start = next.fetch_add(chunk, Ordering::Relaxed);
                        if start >= rows {
                            break;
                        }
                        let mut file =
                            FileColStats::new_with_categories(cols, Arc::clone(&table_cat));
                        for r in start..(start + chunk).min(rows) {
                            for (c, col) in cols.iter().enumerate() {
                                let salt = (c as u64) << 48;
                                let k = if r < distinct {
                                    r
                                } else {
                                    splitmix(r ^ salt) % distinct
                                };
                                match col.semantic_type {
                                    Some(SemanticType::Id) => {
                                        file.observe_i64(c, (k as i64) * 7919 - 1_000_000)
                                    }
                                    Some(SemanticType::Multicategorical) => {
                                        use std::fmt::Write;
                                        buf.clear();
                                        let k2 = splitmix(k ^ salt) % distinct;
                                        write!(buf, "t{c:02}-{k:014x},t{c:02}-{k2:014x}").unwrap();
                                        file.observe_string(c, &buf);
                                    }
                                    Some(SemanticType::Numerical) => {
                                        file.observe_f64(c, (k % 1000) as f64)
                                    }
                                    _ => {
                                        use std::fmt::Write;
                                        buf.clear();
                                        write!(buf, "v{c:02}-{k:016x}").unwrap();
                                        file.observe_string(c, &buf);
                                    }
                                }
                            }
                        }
                        file.flush_categories();
                        done.push((start, file));
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    // Fixed merge order keeps float stats reproducible.
    parts.sort_by_key(|(start, _)| *start);
    let mut merged = FileColStats::new_with_categories(cols, Arc::clone(&table_cat));
    for (_, mut part) in parts {
        merged.merge(&mut part);
    }
    (merged, table_cat)
}

#[test]
#[ignore]
fn bench_col_stats_publish() {
    let ncols = env_usize("DG_COLSTATS_BENCH_COLS", 80);
    let distinct = env_usize("DG_COLSTATS_BENCH_DISTINCT", 1_000_000) as u64;
    let rows = env_usize("DG_COLSTATS_BENCH_ROWS", 2 * distinct as usize) as u64;
    let tmp = tempfile::tempdir().unwrap();
    let out = std::env::var("DG_COLSTATS_BENCH_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| tmp.path().to_path_buf());
    std::fs::create_dir_all(&out).unwrap();

    let split = ncols * 3 / 4;
    let t0 = Instant::now();
    let tables: Vec<_> = [("facts", 0, split), ("dims", split, ncols - split)]
        .into_iter()
        .map(|(name, first, count)| {
            let cols = table_columns(first, count);
            let (stats, tc) = populate(&cols, distinct, rows);
            (name.to_string(), cols, stats, tc)
        })
        .collect();
    eprintln!(
        "populated {ncols} cols x {distinct} distinct ({rows} rows) in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    let target = OutputTarget::parse(out.to_str().unwrap(), Dest::Local).unwrap();
    let before_kb = rss_kb();
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(before_kb));
    let sampler = {
        let (stop, peak) = (Arc::clone(&stop), Arc::clone(&peak));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(rss_kb(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    let start = Instant::now();
    publish(&target, tables);
    let wall = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    let peak_kb = peak.load(Ordering::Relaxed);
    eprintln!(
        "publish: wall {:.2}s, rss before {:.2} GiB, peak {:.2} GiB, peak delta {:.2} GiB",
        wall.as_secs_f64(),
        before_kb as f64 / (1 << 20) as f64,
        peak_kb as f64 / (1 << 20) as f64,
        (peak_kb - before_kb) as f64 / (1 << 20) as f64,
    );
}

fn publish(
    target: &OutputTarget,
    tables: Vec<(
        String,
        Vec<ColumnSpec>,
        FileColStats,
        Arc<TableCategoryCounts>,
    )>,
) {
    let outputs = tables
        .into_iter()
        .map(|(name, cols, stats, tc)| ColStatsTable {
            name,
            stats,
            column_names: cols.iter().map(|c| c.name.clone()).collect(),
            ingested_types: cols.iter().map(|c| c.ingested_dtype.clone()).collect(),
            table_cat: Some(tc),
            text_sample_stats: None,
            data_version: Some("v1".to_string()),
        })
        .collect();
    write_col_stats(target, outputs).unwrap();
}
