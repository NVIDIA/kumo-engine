// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-level 4 KiB page benchmark for mmap, buffered io_uring, and
//! O_DIRECT plus the experimental userspace cache.

use anyhow::{anyhow, Result};
use diskgraph_index::page_io::{read_prepared_pages, reset_stats, stats, EdgeIoBackend, PAGE_SIZE};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn parse_args() -> Result<HashMap<String, String>> {
    const KNOWN: &[&str] = &[
        "file",
        "directory",
        "file-count",
        "backend",
        "pages",
        "iterations",
        "qd",
        "rings",
        "batch-us",
        "batch-mult",
        "threads",
        "cache-gb",
        "seed",
    ];
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut out = HashMap::new();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i]
            .strip_prefix("--")
            .ok_or_else(|| anyhow!("unexpected positional argument `{}`", argv[i]))?;
        if !KNOWN.contains(&key) {
            return Err(anyhow!("unknown option --{key}"));
        }
        let value = argv
            .get(i + 1)
            .ok_or_else(|| anyhow!("missing value for --{key}"))?;
        if out.insert(key.to_string(), value.clone()).is_some() {
            return Err(anyhow!("duplicate option --{key}"));
        }
        i += 2;
    }
    Ok(out)
}

fn collect_edge_files(directory: &Path, out: &mut Vec<(u64, PathBuf)>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_edge_files(&entry.path(), out)?;
        } else if file_type.is_file() && entry.file_name() == "edges.bin" {
            let len = entry.metadata()?.len();
            if len != 0 {
                out.push((len, entry.path()));
            }
        }
    }
    Ok(())
}

fn opt<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> Result<T> {
    match args.get(key) {
        Some(value) => value
            .parse()
            .map_err(|_| anyhow!("bad value for --{key}: {value}")),
        None => Ok(default),
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let file_count: usize = opt(&args, "file-count", 128usize)?;
    let paths = match (args.get("file"), args.get("directory")) {
        (Some(file), None) => vec![PathBuf::from(file)],
        (None, Some(directory)) => {
            let mut files = Vec::new();
            collect_edge_files(Path::new(directory), &mut files)?;
            files.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
            files.truncate(file_count);
            if files.is_empty() {
                return Err(anyhow!("no edges.bin files under --directory {directory}"));
            }
            files.into_iter().map(|(_, path)| path).collect()
        }
        (Some(_), Some(_)) => return Err(anyhow!("use exactly one of --file or --directory")),
        (None, None) => return Err(anyhow!("missing --file or --directory")),
    };
    let backend_name = args.get("backend").map(String::as_str).unwrap_or("mmap");
    let backend = match backend_name {
        "mmap" => EdgeIoBackend::Mmap,
        "buffered" => EdgeIoBackend::Buffered,
        "direct" => EdgeIoBackend::Direct,
        other => return Err(anyhow!("bad --backend `{other}`")),
    };
    let pages_per_iteration: usize = opt(&args, "pages", 256usize)?;
    let iterations: usize = opt(&args, "iterations", 16_000usize)?;
    let qd: usize = opt(&args, "qd", 32usize)?;
    let rings: usize = opt(&args, "rings", 32usize)?;
    let batch_us: u64 = opt(&args, "batch-us", 0u64)?;
    let batch_mult: usize = opt(&args, "batch-mult", 8usize)?;
    let threads: usize = opt(&args, "threads", 32usize)?;
    let cache_gb: usize = opt(&args, "cache-gb", 0usize)?;
    let seed: u64 = opt(&args, "seed", 42u64)?;
    std::env::set_var("DG_IO_QD", qd.to_string());
    std::env::set_var("DG_IO_RINGS", rings.to_string());
    std::env::set_var("DG_IO_BATCH_US", batch_us.to_string());
    std::env::set_var("DG_IO_BATCH_MULT", batch_mult.to_string());
    std::env::set_var("DG_DIRECT_CACHE_GB", cache_gb.to_string());

    let mut buffered = Vec::with_capacity(paths.len());
    let mut direct = Vec::with_capacity(paths.len());
    let mut mmaps = Vec::with_capacity(paths.len());
    let mut file_lens = Vec::with_capacity(paths.len());
    let mut file_pages = Vec::with_capacity(paths.len());
    for path in &paths {
        let file = Arc::new(File::open(path)?);
        let len = file.metadata()?.len();
        let pages = len / PAGE_SIZE as u64 + u64::from(len % PAGE_SIZE as u64 != 0);
        if pages == 0 {
            return Err(anyhow!("input file is empty: {}", path.display()));
        }
        direct.push(if backend == EdgeIoBackend::Direct {
            Some(Arc::new(
                OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
                    .open(path)?,
            ))
        } else {
            None
        });
        mmaps.push(if backend == EdgeIoBackend::Mmap {
            Some(unsafe { Mmap::map(file.as_ref())? })
        } else {
            None
        });
        buffered.push(file);
        file_lens.push(len);
        file_pages.push(pages);
    }

    let total = pages_per_iteration.saturating_mul(iterations);
    let mut page_numbers = Vec::with_capacity(total);
    let mut batch_files = Vec::with_capacity(iterations);
    let mut state = seed;
    for _ in 0..iterations {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let file_idx = state as usize % paths.len();
        batch_files.push(file_idx);
        for _ in 0..pages_per_iteration {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            page_numbers.push(state % file_pages[file_idx]);
        }
    }

    reset_stats();
    let started = Instant::now();
    let checksum = std::thread::scope(|scope| -> Result<u64> {
        let mut handles = Vec::with_capacity(threads);
        for worker in 0..threads {
            let buffered = &buffered;
            let direct = &direct;
            let mmaps = &mmaps;
            let file_lens = &file_lens;
            let batch_files = &batch_files;
            let page_numbers = &page_numbers;
            handles.push(scope.spawn(move || -> Result<u64> {
                let mut checksum = 0u64;
                for batch_idx in (worker..iterations).step_by(threads) {
                    let file_idx = batch_files[batch_idx];
                    let start = batch_idx * pages_per_iteration;
                    let batch = &page_numbers[start..start + pages_per_iteration];
                    if let Some(map) = mmaps[file_idx].as_ref() {
                        for &page in batch {
                            let offset = page as usize * PAGE_SIZE;
                            checksum = checksum.wrapping_add(map[offset] as u64);
                        }
                    } else {
                        let prepared = read_prepared_pages(
                            &buffered[file_idx],
                            direct[file_idx].as_ref(),
                            file_idx as u64 + 1,
                            file_lens[file_idx],
                            backend,
                            batch,
                        )?;
                        checksum = checksum.wrapping_add(prepared.page_count() as u64);
                    }
                }
                Ok(checksum)
            }));
        }
        let mut checksum = 0u64;
        for handle in handles {
            checksum = checksum.wrapping_add(
                handle
                    .join()
                    .map_err(|_| anyhow!("page benchmark worker panicked"))??,
            );
        }
        Ok(checksum)
    })?;
    let wall = started.elapsed().as_secs_f64();
    let io = stats();
    println!(
        "backend={} target={} files={} threads={} rings={} qd_per_ring={} batch_mult={} batch_us={} logical_pages={} wall_s={:.6} logical_pages_s={:.0} submitted={} submitted_iops={:.0} jobs={} envelopes={} owner_batches={} max_batch_pages={} cache_hits={} cache_misses={} loading_bypasses={} bytes={} frames_allocated={} frames_reused={} frames_recycled={} frames_dropped={} checksum={}",
        backend.name(),
        args.get("directory").unwrap_or_else(|| args.get("file").unwrap()),
        paths.len(),
        threads,
        rings,
        qd,
        batch_mult,
        batch_us,
        total,
        wall,
        total as f64 / wall,
        io.submitted,
        io.submitted as f64 / wall,
        io.runtime_jobs,
        io.runtime_envelopes,
        io.runtime_batches,
        io.max_batch_pages,
        io.cache_hits,
        io.cache_misses,
        io.coalesced,
        io.bytes,
        io.frames_allocated,
        io.frames_reused,
        io.frames_recycled,
        io.frames_dropped,
        checksum,
    );
    Ok(())
}
