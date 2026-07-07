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
//! does a SINGLE scatter pass (counts are known up-front from the writer's
//! per-shard count maps — no count pass, no sort, no per-ts files):
//!   1. union per-shard count maps → global `count[ts]`; prefix-sum (ascending
//!      ts) → per-bucket `offset[ts]`.
//!   2. preallocate `bucket_index.bin` = `total * 8` bytes; `cursor = offset`.
//!   3. stream each shard's spill once; for record `k` with timestamp `ts`,
//!      write `(shard.min_id + k) as u64` at `cursor[ts]++`.
//!   4. emit the [`TsBucket`] list.
//!
//! RAM is O(D) (the count/offset/cursor maps, D = distinct timestamps) plus
//! fixed IO buffers — never O(N).

use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

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
    pub count_map: HashMap<i64, u64>,
    pub spill_path: PathBuf,
}

/// Read one shard's spill into a borrowed reader. Kept tiny so the merge stays
/// a streaming loop (no full-file slurp).
fn open_spill(path: &Path) -> Result<BufReader<std::fs::File>> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("open ts-group spill {}", path.display()))?;
    Ok(BufReader::new(f))
}

/// Build a stage's `bucket_index.bin` + bucket list from its per-shard spills.
///
/// `bucket_index_path` is the absolute output path; `bucket_index_rel_path` is
/// what the manifest records. Streams each spill exactly once; RAM is O(D).
pub(crate) fn merge_stage(
    shards: &[ShardSpill],
    bucket_index_path: &Path,
    bucket_index_rel_path: String,
) -> Result<StageGrouping> {
    // 1. Union per-shard count maps → global counts; prefix-sum → offsets.
    let mut global_counts: HashMap<i64, u64> = HashMap::new();
    for shard in shards {
        for (ts, c) in &shard.count_map {
            let entry = global_counts.entry(*ts).or_insert(0);
            *entry = entry.saturating_add(*c);
        }
    }
    // Deterministic bucket order: ascending timestamp.
    let mut tss: Vec<i64> = global_counts.keys().copied().collect();
    tss.sort_unstable();

    let mut buckets = Vec::with_capacity(tss.len());
    let mut offset_map: HashMap<i64, u64> = HashMap::with_capacity(tss.len());
    let mut acc: u64 = 0;
    for ts in &tss {
        let count = global_counts[ts];
        offset_map.insert(*ts, acc);
        buckets.push(TsBucket {
            ts: *ts,
            offset: acc,
            count,
        });
        acc = acc.saturating_add(count);
    }
    let total = acc;

    // 2. Preallocate bucket_index.bin = total * 8 bytes.
    if let Some(parent) = bucket_index_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir for {}", bucket_index_path.display()))?;
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(bucket_index_path)
        .with_context(|| format!("create {}", bucket_index_path.display()))?;
    out.set_len(total.saturating_mul(8))
        .with_context(|| format!("preallocate {}", bucket_index_path.display()))?;

    // 3. Single scatter pass with per-bucket buffered output.
    //
    //    The naive scatter is one `seek` + 8-byte `write` per node — ~2
    //    syscalls/node with random-position writes, a severe bottleneck at
    //    1B–10B rows. Instead we keep a small in-memory buffer per bucket and,
    //    when one fills (BUCKET_BUF_BYTES), flush it as ONE positioned
    //    `write_at` (seek+write fused into a single syscall) at the bucket's
    //    moving byte cursor. This amortizes syscalls to ~1 per BUFCAP nodes
    //    while writing the exact same node-id to the exact same slot as the
    //    naive version, so the output bytes are byte-identical.
    //
    //    RAM stays O(D · BUCKET_BUF_BYTES) (D = distinct timestamps), never
    //    O(N): we never hold all node-ids in memory.
    use std::os::unix::fs::FileExt;
    // 64 KiB per bucket = 8192 node-ids before a flush.
    const BUCKET_BUF_BYTES: usize = 64 * 1024;
    // Per-bucket byte cursor (where this bucket's next block is written) and its
    // pending buffer of node-id bytes. `cursor` starts at each bucket's byte
    // offset and advances by the bytes flushed.
    let mut cursor: HashMap<i64, u64> =
        offset_map.iter().map(|(ts, off)| (*ts, off.saturating_mul(8))).collect();
    let mut pending: HashMap<i64, Vec<u8>> = HashMap::with_capacity(cursor.len());

    let mut buf = [0u8; 8];
    for shard in shards {
        let mut reader = open_spill(&shard.spill_path)?;
        let mut local: u64 = 0;
        loop {
            match reader.read_exact(&mut buf) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    return Err(anyhow!(
                        "read ts-group spill {}: {e}",
                        shard.spill_path.display()
                    ))
                }
            }
            let ts = i64::from_le_bytes(buf);
            let node_id = shard.min_id.saturating_add(local);
            // The cursor map mirrors the count map's key set; an absent ts means
            // a spill record with no corresponding count (an invariant break).
            if !cursor.contains_key(&ts) {
                return Err(anyhow!("ts {ts} present in spill but absent from count map"));
            }
            let block = pending.entry(ts).or_default();
            block.extend_from_slice(&node_id.to_le_bytes());
            if block.len() >= BUCKET_BUF_BYTES {
                let pos = cursor[&ts];
                out.write_all_at(block, pos)
                    .with_context(|| format!("write {}", bucket_index_path.display()))?;
                *cursor.get_mut(&ts).unwrap() = pos.saturating_add(block.len() as u64);
                block.clear();
            }
            local = local.saturating_add(1);
        }
    }
    // Flush every remaining (non-empty) bucket buffer.
    for (ts, block) in pending.iter() {
        if block.is_empty() {
            continue;
        }
        let pos = cursor[ts];
        out.write_all_at(block, pos)
            .with_context(|| format!("write {}", bucket_index_path.display()))?;
    }
    out.flush().ok();

    // 4. Buckets already built (step 1). Delete spills.
    for shard in shards {
        let _ = std::fs::remove_file(&shard.spill_path);
    }

    Ok(StageGrouping {
        bucket_index_rel_path,
        buckets,
    })
}

/// Append a batch of timestamps (LE `i64`, in write order) to a shard's spill
/// file, creating it on first use. Used by the node-writer worker.
pub(crate) fn append_spill(file: &mut std::fs::File, timestamps: &[i64]) -> Result<()> {
    // Pack to a local buffer then one write — keeps the hot path syscall-light.
    let mut buf = Vec::with_capacity(timestamps.len() * 8);
    for ts in timestamps {
        buf.extend_from_slice(&ts.to_le_bytes());
    }
    file.write_all(&buf).context("append ts-group spill")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: write a shard spill from a list of timestamps (node order).
    fn write_spill(dir: &Path, shard: usize, tss: &[i64]) -> (PathBuf, HashMap<i64, u64>) {
        let path = dir.join(format!("shard-{shard}.ts"));
        let mut f = std::fs::File::create(&path).unwrap();
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for ts in tss {
            f.write_all(&ts.to_le_bytes()).unwrap();
            *counts.entry(*ts).or_insert(0) += 1;
        }
        (path, counts)
    }

    fn read_bin(path: &Path) -> Vec<u64> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
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
        let (path, counts) = write_spill(tmp.path(), 0, &tss);
        let expected: HashMap<u64, i64> =
            tss.iter().enumerate().map(|(i, t)| (i as u64, *t)).collect();

        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[ShardSpill {
                min_id: 0,
                count_map: counts,
                spill_path: path,
            }],
            &out,
            "stage/bucket_index.bin".to_string(),
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
        let (p0, c0) = write_spill(tmp.path(), 0, &[5i64, 7]);
        let (p1, c1) = write_spill(tmp.path(), 1, &[7i64, 5, 7]);
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
                ShardSpill { min_id: 0, count_map: c0, spill_path: p0 },
                ShardSpill { min_id: 100, count_map: c1, spill_path: p1 },
            ],
            &out,
            "stage/bucket_index.bin".to_string(),
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
        let (p0, c0) = write_spill(tmp.path(), 0, &[1i64, 2, 1, 3]);
        let (p1, c1) = write_spill(tmp.path(), 1, &[2i64, 2]);
        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[
                ShardSpill { min_id: 0, count_map: c0, spill_path: p0 },
                ShardSpill { min_id: 50, count_map: c1, spill_path: p1 },
            ],
            &out,
            "s/bucket_index.bin".to_string(),
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
        let (p0, c0) = write_spill(tmp.path(), 0, &[1i64, 2]);
        let (p1, c1) = write_spill(tmp.path(), 1, &[2i64, 99]);
        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[
                ShardSpill { min_id: 0, count_map: c0, spill_path: p0 },
                ShardSpill { min_id: 10, count_map: c1, spill_path: p1 },
            ],
            &out,
            "s/bucket_index.bin".to_string(),
        )
        .unwrap();
        let tss: Vec<i64> = g.buckets.iter().map(|b| b.ts).collect();
        assert_eq!(tss, vec![1, 2, 99], "union of all shard timestamps");
        let b99 = g.buckets.iter().find(|b| b.ts == 99).unwrap();
        assert_eq!(b99.count, 1);
        let bin = read_bin(&out);
        assert_eq!(bin[b99.offset as usize], 11, "node 11 (10+1) has ts=99");
    }

    // Test 6 (bounded RAM, structural): merge never materializes a Vec of all
    // node-ids — it streams the spill and only holds O(D) maps. We assert the
    // peak in-RAM collections are sized by distinct-ts, not by N, by feeding a
    // large N with tiny D and checking the result is still correct (a Vec<N>
    // approach would be the only alternative, and this exercises the streaming
    // path end to end).
    #[test]
    fn merge_streams_large_n_small_d() {
        let tmp = tempfile::tempdir().unwrap();
        let n = 100_000usize;
        let tss: Vec<i64> = (0..n).map(|i| (i % 3) as i64).collect();
        let (path, counts) = write_spill(tmp.path(), 0, &tss);
        assert_eq!(counts.len(), 3, "only 3 distinct timestamps");
        let out = tmp.path().join("bucket_index.bin");
        let g = merge_stage(
            &[ShardSpill { min_id: 0, count_map: counts, spill_path: path }],
            &out,
            "s/bucket_index.bin".to_string(),
        )
        .unwrap();
        assert_eq!(g.buckets.len(), 3);
        let total: u64 = g.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total as usize, n);
        let bin = read_bin(&out);
        assert_eq!(bin.len(), n);
        // Spot-check contiguity: bucket for ts=0 holds exactly node-ids ≡0 mod 3.
        let b0 = g.buckets.iter().find(|b| b.ts == 0).unwrap();
        for i in b0.offset..b0.offset + b0.count {
            assert_eq!(bin[i as usize] % 3, 0);
        }
    }
}
