# Local-NVMe feature-heavy ecommerce: mmap versus cached Direct I/O (2026-08-19)

## Result

The 10% feature-heavy ecommerce index is 1,233.58 GiB (5.03x the host's
approximately 245 GiB available RAM), so this is a storage-bound local-NVMe
comparison. A 240-GiB-requested / 224-GiB-usable Direct cache is **1.754x**
faster end-to-end than mmap while using 52.3% fewer total CPU-seconds.

| scale | source rows | index GiB | mmap nodes/s | Direct nodes/s | Direct / mmap | mmap IOPS | Direct IOPS | Direct hit | mmap cores | Direct cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 2.8B | 1,233.58 | 313,642 | **550,116** | **1.754x** | 149,544 | **250,641** | 74.72% | 9.0 | 7.6 |

## Workload and validity

- One 1.8-TB local NVMe (`nvme1n1`) mounted at `/mnt/localssd`; the VM has
  249 GiB physical RAM. The mmap and initial 200-GiB Direct baseline used
  `MemoryMax=240G`; the 220/240-GiB Direct sweep used a safe `MemoryMax=245G`,
  with no swap.
- Four-hop all-relation user sampling, fanout 10, batch 32, 64 request and
  engine threads, with feature materialization enabled. Canonical plan:
  `default:4x10:Uniform` (`5d2e1672de275d8d`).
- Each backend was a fresh process after file-scoped `POSIX_FADV_DONTNEED` for
  all 4,495 index files (1,233.58 GiB). Both used an independent 1,000,000-root
  warmup and a 300,000-root measured window.
- mmap: 405.684s, 419,715 edges/s, 100% device busy, kernel cache
  230.9->231.2 GiB during measurement, and main digest `463575f63aa66cec`.
- Direct: 231.295s, 736,166 edges/s, 100% device busy, and a requested 240-GiB
  cache with 224.0 GiB usable geometry. Its cache was 223.5->223.7/224.0 GiB
  during measurement (0.05% growth), so it was both sufficiently sized and
  steady. It issued 57,972,059 physical reads (221.37 GiB) and served
  171,485,601 page-cache hits.
- Both runs produced the same main digest and passed the 20,000-root
  output-identical verification digest `eb86343e75025e8b`.

The Direct cache is deliberately userspace-only: the process's kernel page
cache stayed 5.5 GiB throughout Direct, while mmap used about 231 GiB. Direct
nevertheless delivered 1.676x mmap's physical IOPS and used 1,748.1 CPU-s
versus mmap's 3,666.0 CPU-s.

## Direct-cache geometry sweep

The fixed cache uses a power-of-two set count. Consequently, request values are
not continuous: 200 and 220 GiB both select 8,388,608 sets x 6 ways (192 GiB),
while 240 GiB selects the next useful geometry, 8,388,608 x 7 (224 GiB).

| requested GiB | usable GiB | nodes/s | IOPS | hit rate | measured cache |
|---:|---:|---:|---:|---:|---|
| 200 | 192 | 478,854 | 250,634 | 70.94% | 191.8->191.9 / 192.0 GiB |
| 220 | 192 | 478,866 | 250,643 | 70.94% | 191.8->191.9 / 192.0 GiB |
| 240 | 224 | **550,116** | 250,641 | **74.72%** | 223.5->223.7 / 224.0 GiB |

The 220-GiB result validates that the identical 192-GiB geometry is stable and
repeatable. The extra 32 GiB usable at the 240-GiB request raises natural cache
hits by 3.78 percentage points and end-to-end throughput by 14.9%. The 220 and
240 points used `MemoryMax=245G`; the original 200 point used 240G and is
otherwise the same fresh-process workload.

## Resumable-index scratch reclamation

The initial 10% Index attempt with the pre-#443 code was stopped at 97% of the
1.8-TB volume because it retained approximately 1.02 TB of `_tmp/nodes` plus
mapped-edge scratch while final output accumulated. PR #443 was backported as
`a3190f2f` before resuming Index.

The resumed run reused completed Encode, PK-map, and EdgesMap phases, rebuilt
the old markerless provisional Index output safely, then published durable
per-shard markers before deleting each shard's consumed input. It completed
Index in 23m42s, removed `_tmp` on finalization, and left 500 GiB free (72%
volume utilization). The final index has 2.8B nodes and 13.104B directed edges.
Representative user shards reclaimed about 2.81 GB each immediately after
their output commit.

Focused validation of the backport passed:

- `cargo test -p diskgraph-index --test graph_index` (2 tests);
- `cargo test -p diskgraph-ingest --test resume_equivalence` (6 tests).

Reproduction:

```bash
DG_BENCH_PLAN=default DG_BENCH_LAYERS=4 DG_BENCH_FETCH_FEATURES=1 \
DG_BENCH_BATCH_SIZE=32 DG_BENCH_THREADS=64 DG_BENCH_BACKENDS='mmap direct' \
DG_BENCH_DIRECT_CACHE_GB=240 DG_BENCH_NUM_SEEDS=300000 \
DG_BENCH_WARMUP_SEEDS=1000000 DG_BENCH_MEMORY_MAX=245G \
DG_BENCH_DEVICES=nvme1n1 DG_BENCH_DIRECT_IO_QD=32 \
DG_BENCH_DIRECT_IO_RINGS=16 DG_BENCH_EVICT_CACHE=1 \
scripts/bench_sampling_iops.sh /mnt/localssd/diskgraph-index/ecommerce_features_10pct-pr424-20260819
```
