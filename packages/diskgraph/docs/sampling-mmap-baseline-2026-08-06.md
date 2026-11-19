# DiskGraph mmap sampling baseline — 2026-08-06

> Superseded benchmark warning: a later CLI audit found that this run's
> `--relations` option was silently ignored by `dg-engine-bench`. The 11.456M
> result is useful historical context but is not the canonical two-relation
> baseline. See `docs/reports/diskgraph-sampler-io-benchmark-2026-08-06.md` for
> the repaired strict-plan comparison.

## Outcome

On the local 76.48B-directed-edge index, the best retained production-engine
mmap point was **11.456M sampled nodes/s at 1.021M physical read IOPS**. The
average aggregate device queue depth was **59.5** (about 14.9 per RAID member).

That is:

- 15.6% of the independently validated 6.56M-IOPS raw md RAID0 ceiling
  (5.54M IOPS unused; 6.42x ceiling/current), and
- 12.8% of the four drives' approximately 8M-IOPS summed industry rating
  (6.98M IOPS unused; 7.83x ceiling/current).

The live queue is not literally zero, but it is only 5.8% of the aggregate
QD1024 used to reach the raw-array ceiling. mmap major faults still couple one
outstanding miss to a blocked sampling thread, so increasing sampling workers
past the CPU/IO knee adds scheduling overhead without enough useful queue
depth.

## Dataset inventory

`/mnt/localssd` contained three source datasets and one standalone completed
DiskGraph index:

| path | allocated size | role |
|---|---:|---|
| `datasets/alt_detection_25pct` | 3.898 TB | source parquet |
| `datasets/mimic` | 7.30 GB | source data |
| `datasets/mimic_safe` | 12.1 MB | source data |
| `diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor` | 1.058 TB | completed index |

The selected index is the only graph in the requested approximately-100B-edge
class. `ingest_stats.json` records 19,645,490,124 nodes and 76,475,264,044
stored directed edges. The four source edge tables contain 38,237,632,022 rows;
DiskGraph stores both directions, accounting for the 76.48B total.

Index payload accounting found 7,506 `edges.bin` files totalling 866.874 GiB
and 7,506 `offsets.bin` files totalling 38.880 GiB. There are 1,251 shards for
each of six node types.

## Host and method

- CPU: AMD Ryzen Threadripper PRO 9995WX, 96 physical cores / 192 threads.
- Memory: 502 GiB.
- Storage: XFS on md127 RAID0, 512 KiB chunk, four Samsung 9100 PRO 8TB NVMe.
- Harness: `dg-engine-bench` from commit `94200867`, calling the production
  `DiskGraphEngine::sample` path.
- Cache control: each retained point used file-specific
  `POSIX_FADV_DONTNEED`, followed by a temporary `MemoryMax=96G` user scope.
- Measurement: 2M discarded warmup seeds then 10M fresh measured seeds.
- Plan: `users`, uniform sampling, two hops, fanout 10, relations
  `edge_user_ip_edges_src_to_dst` and
  `edge_user_ip_edges_dst_to_src`, batch size 1,000.
- Resulting workload: 84.4 nodes/seed and 85.0 edges/seed.
- Device attribution: `/proc/diskstats` deltas for exactly nvme0n1 through
  nvme3n1; OS NVMes were excluded.

The eight Kumo devservers were stopped before retained measurements. All
dataset/index access was read-only.

## Retained sampler sweep

| outer + engine workers | nodes/s | examples/s | read IOPS | reads/node | avg aggregate QD | aggregate member busy |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 6,921,468 | 81,985 | 631,768 | 0.091 | 35.2 | 371% |
| **96** | **11,455,955** | **135,696** | **1,021,247** | **0.089** | **59.5** | **344%** |
| 128 | 11,067,185 | 131,091 | 999,927 | 0.090 | 63.5 | 313% |
| 192 | 9,520,272 | 112,768 | 861,703 | 0.091 | 65.5 | 269% |

The maximum retained rate is therefore 11.456M nodes/s at 96 workers. The
decline at 128 and 192 while queue depth changes only modestly establishes a
real oversubscription knee rather than a too-short run artifact.

### Phase-0 prefetch check

A separate cache-reset run at the winning 96-worker point enabled the branch's
batched `MADV_WILLNEED` prefetch gate:

| mode | nodes/s | read IOPS | reads/node | avg aggregate QD | gate passes |
|---|---:|---:|---:|---:|---:|
| prefetch off | **11,455,955** | 1,021,247 | 0.089 | 59.5 | 0 |
| prefetch on | 11,243,575 | 952,262 | 0.085 | 63.5 | 19,370,630 |

Prefetch reduced physical reads by 6.8% and modestly deepened the queue, but
nodes/s fell 1.85%. The gate was active, so this is not a disabled-feature
artifact: grouping/advice overhead costs more than the avoided reads on this
plan. Phase 0 does not close the mmap gap here.

## fio ceiling checks

A read-only fio run through one 441MB XFS `edges.bin` file used 32 workers x
QD32 (aggregate QD1024), io_uring, O_DIRECT, 4KiB random reads, a five-second
ramp, and a 30-second measurement:

- 2.544M IOPS / 9,936 MiB/s
- mean completion latency 390 us; mean total latency 402 us
- 99.7% md utilization
- 95.2% system CPU

This is a useful single-inode/XFS ceiling, not the hardware ceiling. The
corrected raw-md result supplied for this host is 6.56M IOPS at 32 workers and
aggregate QD1024; QD2048 reaches 6.62M but roughly doubles latency. The latter
is the correct redesign target.

Raw md and member-device reruns were attempted but the devices are `root:disk`
mode 0660 and the benchmark user is not in group `disk`; sudo is unavailable
in this execution environment. The commands are encoded in
`scripts/bench_fio_iops.sh` and remain strictly read-only.

## Reproduction

Build the harness from the async-I/O benchmark branch:

```bash
cargo build --release -p diskgraph-engine --bin dg-engine-bench
```

Run the retained sweep:

```bash
DG_BENCH_PLAN='edge_user_ip_edges_src_to_dst:10;edge_user_ip_edges_dst_to_src:10' \
  scripts/bench_sampling_iops.sh \
  /mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor
```

Monitor independently in another terminal:

```bash
scripts/diskstats.sh -i 2 -d '^(nvme[0-3]n1|md127)$'
```

Run ceiling checks from a shell with raw-device read permission:

```bash
FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/md127
FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/nvme0n1
FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/nvme1n1
FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/nvme2n1
FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/nvme3n1
```

## Redesign baseline and next gates

The benchmark stack now covers deterministic production sampling, cache
control, nodes/s, reads/node, physical IOPS, bandwidth, utilization, queue
depth, a concurrency sweep, and read-only fio saturation.

For buffered async I/O or direct I/O, hold the sampling plan, seed, cache cap,
thread points, and determinism digest constant. Add explicit counters for
submitted reads, completions, in-flight high-water mark, coalesced/duplicate
reads, cache hits, and completion latency. Compare at equal aggregate QD points
(64, 128, 256, 512, 1024), not merely equal thread counts.

Do not project the 6.42x IOPS gap directly into a 6.42x nodes/s claim. At the
measured 0.089 reads/node, storage alone permits a much higher rate, so CPU,
decode, dedup, memory bandwidth, or output assembly will become the next
bottleneck as async I/O raises IOPS. Buffered io_uring should be the first
redesign arm because it preserves useful page-cache hits; O_DIRECT needs a
userspace cache or an explicit decision to sacrifice the cache-resident regime.
