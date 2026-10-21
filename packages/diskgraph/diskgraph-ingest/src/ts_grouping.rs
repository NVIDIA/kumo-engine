// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TT-NG temporal-LP timestamp grouping (ingest side).
//!
//! When enabled for a stage table, ingest produces an on-disk artifact that
//! groups the stage's seed node-ids by their `TIMESTAMP` value:
//!
//! * `<table>/bucket_index.bin` — a flat little-endian `u64` array of FINAL
//!   global node-ids, laid out so all node-ids sharing a timestamp are
//!   contiguous (buckets back-to-back, in ascending-timestamp order).
//! * a per-stage list of [`TsBucket`] `{ ts, offset, count }` describing each
//!   bucket's slice of `bucket_index.bin`.
//!
//! ## Where the timestamp is captured
//! The node's final local position within a shard is the order the node-writer
//! worker writes its batch — NOT the per-file encode `row_id` (encode runs
//! files in parallel and resets `row_id` per file). So the timestamp travels
//! WITH the node: the encode pass attaches a per-row `TIMESTAMP` value to the
//! node batch, and the writer appends each timestamp (LE `i64`) to a per-shard
//! spill file `_tmp/ts_group/<table>/shard-<s>.ts` IN WRITE ORDER. The k-th
//! record in a shard's spill is the timestamp of the node whose final id is
//! `shard.min_id + k`.
//!
//! ## Merge
//! After the index is finalized (shard offsets / min-ids known), [`merge_stage`]
//! performs two bounded parallel passes over fixed-size chunks of the spills:
//!   1. count timestamps up to a fixed cardinality cap; prefix-sum (ascending
//!      ts) → per-bucket `offset[ts]`.
//!   2. preallocate `bucket_index.bin` = `total * 8` bytes; `cursor = offset`.
//!   3. per chunk, in node order, claim `count[ts]` slots at `cursor[ts]`;
//!      counting-sort the chunk's node ids in memory and write one run per ts.
//!   4. emit the [`TsBucket`] list.
//!
//! Encode retains no per-timestamp maps. Merge RAM is one count map capped by
//! [`MAX_TIMESTAMP_BUCKETS`] plus one chunk per worker; write waves shrink as
//! timestamp cardinality grows.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;

/// Relative temp path of a stage's per-shard timestamp spill file.
pub(crate) fn spill_rel_path(table: &str, shard: usize) -> String {
    format!(
        "{}/ts_group/{}/shard-{:05}.ts",
        crate::consts::TEMP_DIR_NAME,
        table.to_lowercase(),
        shard
    )
}

/// On-disk name of the per-stage bucket-index file, relative to the stage's
/// extension table dir is `<table>/bucket_index.bin`; this is just the leaf.
pub const BUCKET_INDEX_FILENAME: &str = "bucket_index.bin";

/// One timestamp bucket: all node-ids with `ts` occupy `bucket_index.bin`
/// slice `[offset, offset + count)` (units: node-ids, so byte range is `*8`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TsBucket {
    pub ts: i64,
    pub offset: u64,
    pub count: u64,
}

/// Per-stage grouping result: the bucket list plus the relative path of the
/// emitted `bucket_index.bin` (relative to the extension root).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageGrouping {
    pub bucket_index_rel_path: String,
    pub buckets: Vec<TsBucket>,
}

/// Per-shard inputs to the merge: the shard's `min_id` (final global node-id of
/// its first node) and the path to its timestamp spill file (records are LE
/// `i64`, one per node, in node order).
pub(crate) struct ShardSpill {
    pub min_id: u64,
    pub spill_path: PathBuf,
}

/// Bounds both grouping metadata and the returned bucket vector. A timestamp
/// per row is not a useful sampling grouping and must fail before publication
/// rather than turn input cardinality into unbounded memory.
const MAX_TIMESTAMP_BUCKETS: usize = 1_000_000;
const SPILL_BUFFER_BYTES: usize = 64 * 1024;
const CHUNK_RECORDS: u64 = 1 << 19;
const WORKER_COUNT_ENTRIES: usize = 1 << 16;

struct Chunk {
    file: usize,
    start: u64,
    len: u64,
    first_node_id: u64,
}

struct LoadedChunk {
    slot_of: Vec<u32>,
    counts: Vec<(i64, u64)>,
}

/// Build a stage's `bucket_index.bin` + bucket list from its per-shard spills.
///
/// `bucket_index_path` is the absolute output path; `bucket_index_rel_path` is
/// what the manifest records. RAM is O(D + CHUNK_RECORDS * threads), where D
/// is the number of distinct timestamps.
pub(crate) fn merge_stage(
    shards: &[ShardSpill],
    bucket_index_path: &Path,
    bucket_index_rel_path: String,
    threads: usize,
) -> Result<StageGrouping> {
    merge_stage_with_limit(
        shards,
        bucket_index_path,
        bucket_index_rel_path,
        threads,
        CHUNK_RECORDS,
        MAX_TIMESTAMP_BUCKETS,
    )
}

fn merge_stage_with_limit(
    shards: &[ShardSpill],
    bucket_index_path: &Path,
    bucket_index_rel_path: String,
    threads: usize,
    chunk_records: u64,
    maximum_buckets: usize,
) -> Result<StageGrouping> {
    let threads = threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("build ts-group merge pool")?;
    let mut chunks = Vec::new();
    for (file, shard) in shards.iter().enumerate() {
        let length = std::fs::metadata(&shard.spill_path)
            .with_context(|| format!("stat ts-group spill {}", shard.spill_path.display()))?
            .len();
        if length % 8 != 0 {
            return Err(anyhow!(
                "truncated ts-group spill {}",
                shard.spill_path.display()
            ));
        }
        let records = length / 8;
        let mut start = 0;
        while start < records {
            let len = chunk_records.min(records - start);
            chunks.push(Chunk {
                file,
                start,
                len,
                first_node_id: shard.min_id.saturating_add(start),
            });
            start += len;
        }
    }
    let read = |chunk: &Chunk| -> Result<Vec<u8>> {
        let path = &shards[chunk.file].spill_path;
        let mut bytes = vec![0u8; chunk.len as usize * 8];
        std::fs::File::open(path)
            .and_then(|spill| read_exact_at(&spill, &mut bytes, chunk.start * 8))
            .with_context(|| format!("read ts-group spill {}", path.display()))?;
        Ok(bytes)
    };

    // 1. Count directly from the disk spills. Encode intentionally retains no
    // cardinality-sized count map, so its byte budget remains authoritative.
    let shared_counts = std::sync::Mutex::new(HashMap::new());
    let flush = |counts: &mut HashMap<i64, u64>| -> Result<()> {
        let mut shared = shared_counts.lock().expect("count map lock");
        for (ts, count) in counts.drain() {
            add_count(&mut shared, ts, count, maximum_buckets)?;
        }
        Ok(())
    };
    let workers = threads.min(chunks.len()).max(1);
    let flush_entries = WORKER_COUNT_ENTRIES.min(chunk_records as usize);
    let count = |worker: usize| -> Result<()> {
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for chunk in chunks.iter().skip(worker).step_by(workers) {
            for ts in records(&read(chunk)?) {
                *counts.entry(ts).or_insert(0) += 1;
                if counts.len() >= flush_entries {
                    flush(&mut counts)?;
                }
            }
        }
        flush(&mut counts)
    };
    pool.install(|| (0..workers).into_par_iter().try_for_each(count))?;
    let mut global_counts = shared_counts.into_inner().expect("count map lock");
    // Deterministic bucket order: ascending timestamp.
    let mut tss: Vec<i64> = global_counts.keys().copied().collect();
    tss.sort_unstable();

    let mut buckets = Vec::with_capacity(tss.len());
    let mut acc: u64 = 0;
    for ts in &tss {
        let count = global_counts[ts];
        buckets.push(TsBucket {
            ts: *ts,
            offset: acc,
            count,
        });
        // Reuse the count map allocation as the node-id cursor map so merge
        // does not retain counts, offsets, and cursors simultaneously.
        global_counts.insert(*ts, acc);
        acc = acc
            .checked_add(count)
            .ok_or_else(|| anyhow!("timestamp grouping row count overflowed"))?;
    }
    drop(tss);
    let total = acc;

    // 2. Preallocate bucket_index.bin = total * 8 bytes.
    if let Some(parent) = bucket_index_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir for {}", bucket_index_path.display()))?;
    }
    let out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(bucket_index_path)
        .with_context(|| format!("create {}", bucket_index_path.display()))?;
    out.set_len(total.saturating_mul(8))
        .with_context(|| format!("preallocate {}", bucket_index_path.display()))?;

    // 3. Each chunk is counting-sorted in memory and written as one contiguous
    // run per timestamp. Runs are claimed in chunk order, which is node order.
    // A chunk holds one count slot per distinct timestamp, so high-cardinality
    // stages get fewer concurrent chunks.
    let wave_threads = threads.min((chunk_records as usize * 4 / buckets.len().max(1)).max(1));
    let mut cursor = global_counts;
    let load = |chunk: &Chunk| -> Result<LoadedChunk> {
        let mut slots = HashMap::new();
        let mut counts = Vec::new();
        let slot_of = records(&read(chunk)?)
            .map(|ts| {
                let slot = *slots.entry(ts).or_insert_with(|| {
                    counts.push((ts, 0u64));
                    counts.len() as u32 - 1
                });
                counts[slot as usize].1 += 1;
                slot
            })
            .collect();
        Ok(LoadedChunk { slot_of, counts })
    };
    let write = |(chunk, loaded, starts): (&Chunk, LoadedChunk, Vec<u64>)| -> Result<()> {
        let mut ends = Vec::with_capacity(loaded.counts.len());
        let mut acc = 0usize;
        for (_, count) in &loaded.counts {
            ends.push(acc);
            acc += *count as usize;
        }
        let mut bytes = vec![0u8; loaded.slot_of.len() * 8];
        for (local, slot) in loaded.slot_of.iter().enumerate() {
            let end = &mut ends[*slot as usize];
            let node_id = chunk.first_node_id.saturating_add(local as u64);
            bytes[*end * 8..*end * 8 + 8].copy_from_slice(&node_id.to_le_bytes());
            *end += 1;
        }
        for ((_, count), (end, start)) in loaded.counts.iter().zip(ends.iter().zip(starts)) {
            let begin = end - *count as usize;
            write_all_at(&out, &bytes[begin * 8..end * 8], start * 8)
                .with_context(|| format!("write {}", bucket_index_path.display()))?;
        }
        Ok(())
    };
    for wave in chunks.chunks(wave_threads) {
        let loaded: Vec<LoadedChunk> =
            pool.install(|| wave.par_iter().map(load).collect::<Result<_>>())?;
        let mut work = Vec::with_capacity(wave.len());
        for (chunk, loaded) in wave.iter().zip(loaded) {
            let mut starts = Vec::with_capacity(loaded.counts.len());
            for (ts, count) in &loaded.counts {
                let next = cursor.get_mut(ts).ok_or_else(|| {
                    anyhow!("ts {ts} present in spill but absent from cursor map")
                })?;
                starts.push(*next);
                *next += count;
            }
            work.push((chunk, loaded, starts));
        }
        pool.install(|| work.into_par_iter().try_for_each(write))?;
    }

    // 4. Buckets already built (step 1). Delete spills.
    for shard in shards {
        let _ = std::fs::remove_file(&shard.spill_path);
    }

    Ok(StageGrouping {
        bucket_index_rel_path,
        buckets,
    })
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(bytes, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        match file.seek_read(bytes, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                offset += read as u64;
                bytes = &mut bytes[read..];
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &File, bytes: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(bytes, offset)
}

#[cfg(windows)]
fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        match file.seek_write(bytes, offset) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(written) => {
                offset += written as u64;
                bytes = &bytes[written..];
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn records(bytes: &[u8]) -> impl Iterator<Item = i64> + '_ {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|record| i64::from_le_bytes(*record))
}

fn add_count(
    counts: &mut HashMap<i64, u64>,
    ts: i64,
    count: u64,
    maximum_buckets: usize,
) -> Result<()> {
    if !counts.contains_key(&ts) && counts.len() >= maximum_buckets {
        return Err(anyhow!(
            "timestamp grouping exceeds the supported limit of {maximum_buckets} distinct timestamps"
        ));
    }
    let entry = counts.entry(ts).or_insert(0);
    *entry = entry
        .checked_add(count)
        .ok_or_else(|| anyhow!("timestamp bucket row count overflowed"))?;
    Ok(())
}

/// Append a batch of timestamps (LE `i64`, in write order) to a shard's spill
/// file, creating it on first use. Used by the node-writer worker.
pub(crate) fn append_spill(file: &mut std::fs::File, timestamps: &[i64]) -> Result<()> {
    // Pack in a fixed-size window: one large Encode batch must not allocate a
    // second cardinality-sized timestamp buffer outside the shared budget.
    let mut buf = Vec::with_capacity(SPILL_BUFFER_BYTES);
    for chunk in timestamps.chunks(SPILL_BUFFER_BYTES / 8) {
        buf.clear();
        for ts in chunk {
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        file.write_all(&buf).context("append ts-group spill")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positioned_io_preserves_other_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("positioned.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        write_all_at(&file, b"ABCD", 3).unwrap();
        write_all_at(&file, b"Z", 0).unwrap();
        write_all_at(&file, b"", 100).unwrap();
        let mut bytes = [0; 4];
        read_exact_at(&file, &mut bytes, 3).unwrap();
        assert_eq!(&bytes, b"ABCD");
        assert_eq!(
            read_exact_at(&file, &mut bytes, 8).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        read_exact_at(&file, &mut [], 100).unwrap();
        drop(file);
        assert_eq!(std::fs::read(path).unwrap(), b"Z12ABCD789");
    }

    /// Helper: write a shard spill from a list of timestamps (node order).
    fn write_spill(dir: &Path, shard: usize, tss: &[i64]) -> PathBuf {
        let path = dir.join(format!("shard-{shard}.ts"));
        let mut f = std::fs::File::create(&path).unwrap();
        for ts in tss {
            f.write_all(&ts.to_le_bytes()).unwrap();
        }
        path
    }

    fn read_bin(path: &Path) -> Vec<u64> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect()
    }

    /// Assert every node-id with timestamp `t` is contiguous in `bin` within
    /// `[offset_t, offset_t + count_t)`, offsets partition `[0,total)` with no
    /// gaps/overlaps, and `expected[node_id] == t`.
    fn assert_grouped(bin: &[u64], buckets: &[TsBucket], expected_ts: &HashMap<u64, i64>) {
        let total: u64 = buckets.iter().map(|b| b.count).sum();
        assert_eq!(bin.len() as u64, total, "bin length == total bucket count");
        // Offsets partition [0,total): sort by offset, check contiguity.
        let mut sorted = buckets.to_vec();
        sorted.sort_by_key(|b| b.offset);
        let mut cursor = 0u64;
        for b in &sorted {
            assert_eq!(b.offset, cursor, "no gap/overlap before ts={}", b.ts);
            for i in b.offset..b.offset + b.count {
                let node_id = bin[i as usize];
                assert_eq!(
                    expected_ts.get(&node_id).copied(),
                    Some(b.ts),
                    "node {node_id} in bucket ts={} must actually have ts={}",
                    b.ts,
                    b.ts
                );
            }
            cursor += b.count;
        }
        assert_eq!(cursor, total, "buckets cover exactly [0,total)");
    }

    // Test 1 — group-by correctness (single shard, mixed timestamps).
    #[test]
    fn merge_groups_by_timestamp_single_shard() {
        let tmp = tempfile::tempdir().unwrap();
        // node-order timestamps; min_id = 0 so node_id == local position.
        let tss = vec![30i64, 10, 20, 10, 30, 10];
        let path = write_spill(tmp.path(), 0, &tss);
        let expected: HashMap<u64, i64> = tss
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, *t))
            .collect();

        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[ShardSpill {
                min_id: 0,
                spill_path: path,
            }],
            &out,
            "stage/bucket_index.bin".to_string(),
            4,
        )
        .unwrap();

        let bin = read_bin(&out);
        assert_grouped(&bin, &g.buckets, &expected);
        // Buckets are ascending ts with exact counts.
        assert_eq!(
            g.buckets,
            vec![
                TsBucket { ts: 10, offset: 0, count: 3 },
                TsBucket { ts: 20, offset: 3, count: 1 },
                TsBucket { ts: 30, offset: 4, count: 2 },
            ]
        );
    }

    // Test 2 — +min_id correctness across two shards with distinct min_ids.
    #[test]
    fn merge_applies_shard_min_id() {
        let tmp = tempfile::tempdir().unwrap();
        // shard 0: min_id 0, 2 nodes; shard 1: min_id 100, 3 nodes.
        let p0 = write_spill(tmp.path(), 0, &[5i64, 7]);
        let p1 = write_spill(tmp.path(), 1, &[7i64, 5, 7]);
        // Expected final node-ids: shard0 -> {0:5, 1:7}, shard1 -> {100:7,101:5,102:7}
        let mut expected: HashMap<u64, i64> = HashMap::new();
        expected.insert(0, 5);
        expected.insert(1, 7);
        expected.insert(100, 7);
        expected.insert(101, 5);
        expected.insert(102, 7);

        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[
                ShardSpill {
                    min_id: 0,
                    spill_path: p0,
                },
                ShardSpill {
                    min_id: 100,
                    spill_path: p1,
                },
            ],
            &out,
            "stage/bucket_index.bin".to_string(),
            4,
        )
        .unwrap();

        let bin = read_bin(&out);
        assert_grouped(&bin, &g.buckets, &expected);
        // ts=5 has nodes {0,101}; ts=7 has {1,100,102}.
        let bucket5 = g.buckets.iter().find(|b| b.ts == 5).unwrap();
        let got5: std::collections::HashSet<u64> = bin
            [bucket5.offset as usize..(bucket5.offset + bucket5.count) as usize]
            .iter()
            .copied()
            .collect();
        assert_eq!(got5, [0u64, 101].into_iter().collect());
    }

    // Test 3 — completeness / no-dup: every node-id appears exactly once.
    #[test]
    fn merge_is_complete_and_dedup() {
        let tmp = tempfile::tempdir().unwrap();
        let p0 = write_spill(tmp.path(), 0, &[1i64, 2, 1, 3]);
        let p1 = write_spill(tmp.path(), 1, &[2i64, 2]);
        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[
                ShardSpill {
                    min_id: 0,
                    spill_path: p0,
                },
                ShardSpill {
                    min_id: 50,
                    spill_path: p1,
                },
            ],
            &out,
            "s/bucket_index.bin".to_string(),
            4,
        )
        .unwrap();
        let bin = read_bin(&out);
        let total: u64 = g.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 6, "4 + 2 retained rows");
        assert_eq!(bin.len(), 6);
        let uniq: std::collections::HashSet<u64> = bin.iter().copied().collect();
        assert_eq!(uniq.len(), 6, "each node-id appears exactly once");
        // Exactly the expected node-id set: shard0 [0..4), shard1 [50..52).
        let expected: std::collections::HashSet<u64> =
            [0u64, 1, 2, 3, 50, 51].into_iter().collect();
        assert_eq!(uniq, expected);
    }

    // Test 5 — a timestamp present in only some shards still merges (union).
    #[test]
    fn merge_handles_ts_missing_from_some_shards() {
        let tmp = tempfile::tempdir().unwrap();
        // ts=99 only in shard 1; ts=1 only in shard 0; ts=2 in both.
        let p0 = write_spill(tmp.path(), 0, &[1i64, 2]);
        let p1 = write_spill(tmp.path(), 1, &[2i64, 99]);
        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[
                ShardSpill {
                    min_id: 0,
                    spill_path: p0,
                },
                ShardSpill {
                    min_id: 10,
                    spill_path: p1,
                },
            ],
            &out,
            "s/bucket_index.bin".to_string(),
            4,
        )
        .unwrap();
        let tss: Vec<i64> = g.buckets.iter().map(|b| b.ts).collect();
        assert_eq!(tss, vec![1, 2, 99], "union of all shard timestamps");
        let b99 = g.buckets.iter().find(|b| b.ts == 99).unwrap();
        assert_eq!(b99.count, 1);
        let bin = read_bin(&out);
        assert_eq!(bin[b99.offset as usize], 11, "node 11 (10+1) has ts=99");
    }

    #[test]
    fn parallel_chunked_merge_matches_sequential_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut shards = Vec::new();
        let mut expected: Vec<(i64, u64)> = Vec::new();
        for (shard, (min_id, rows)) in [(1_000u64, 2_000usize), (0, 1_501), (5_000, 0), (40_000, 7)]
            .into_iter()
            .enumerate()
        {
            let tss: Vec<i64> = (0..rows)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) % 101) as i64 * 86_400
                })
                .collect();
            expected.extend(
                tss.iter()
                    .enumerate()
                    .map(|(k, ts)| (*ts, min_id + k as u64)),
            );
            shards.push((min_id, tss, shard));
        }
        expected.sort_by_key(|(ts, _)| *ts);
        let mut expected_buckets: Vec<TsBucket> = Vec::new();
        for (index, (ts, _)) in expected.iter().enumerate() {
            match expected_buckets.last_mut() {
                Some(bucket) if bucket.ts == *ts => bucket.count += 1,
                _ => expected_buckets.push(TsBucket {
                    ts: *ts,
                    offset: index as u64,
                    count: 1,
                }),
            }
        }
        assert_eq!(expected_buckets.len(), 101);
        let expected_bytes: Vec<u8> = expected
            .iter()
            .flat_map(|(_, node_id)| node_id.to_le_bytes())
            .collect();
        for (threads, chunk_records) in [(1, CHUNK_RECORDS), (4, 3), (3, 64), (8, 500)] {
            let spills = shards
                .iter()
                .map(|(min_id, tss, shard)| ShardSpill {
                    min_id: *min_id,
                    spill_path: write_spill(tmp.path(), *shard, tss),
                })
                .collect::<Vec<_>>();
            let out = tmp
                .path()
                .join(format!("bucket_index-{threads}-{chunk_records}.bin"));
            let grouping = merge_stage_with_limit(
                &spills,
                &out,
                "s/bucket_index.bin".to_string(),
                threads,
                chunk_records,
                MAX_TIMESTAMP_BUCKETS,
            )
            .unwrap();
            assert_eq!(std::fs::read(&out).unwrap(), expected_bytes);
            assert_eq!(grouping.buckets, expected_buckets);
        }
    }

    #[test]
    fn merge_rejects_timestamp_cardinality_above_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_spill(tmp.path(), 0, &[1, 2, 3]);
        let out = tmp.path().join("bucket_index.bin");
        for (threads, chunk_records) in [(4, CHUNK_RECORDS), (3, 1)] {
            let error = merge_stage_with_limit(
                &[ShardSpill {
                    min_id: 0,
                    spill_path: path.clone(),
                }],
                &out,
                "s/bucket_index.bin".to_string(),
                threads,
                chunk_records,
                2,
            )
            .unwrap_err();
            assert!(error.to_string().contains("2 distinct timestamps"));
            assert!(!out.exists());
        }
        let grouping = merge_stage_with_limit(
            &[ShardSpill {
                min_id: 0,
                spill_path: path,
            }],
            &out,
            "s/bucket_index.bin".to_string(),
            3,
            1,
            3,
        )
        .unwrap();
        assert_eq!(grouping.buckets.len(), 3);
    }

    #[test]
    fn merge_open_files_do_not_scale_with_shards() {
        let Some(dir) = std::env::var_os("TS_GROUPING_NOFILE_CHILD") else {
            let tmp = tempfile::tempdir().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "ts_grouping::tests::merge_open_files_do_not_scale_with_shards",
                ])
                .env("TS_GROUPING_NOFILE_CHILD", tmp.path())
                .status()
                .unwrap();
            assert!(status.success());
            return;
        };
        let dir = PathBuf::from(dir);
        let limit = libc::rlimit {
            rlim_cur: 64,
            rlim_max: 64,
        };
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
        let shards: Vec<ShardSpill> = (0..100)
            .map(|shard| ShardSpill {
                min_id: shard as u64,
                spill_path: write_spill(&dir, shard, &[shard as i64 % 3]),
            })
            .collect();
        let grouping =
            merge_stage(&shards, &dir.join("bucket_index.bin"), String::new(), 4).unwrap();
        assert_eq!(grouping.buckets.len(), 3);
    }
}
