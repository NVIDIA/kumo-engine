# DiskGraph sampler I/O redesign baseline — 2026-08-06

## Outcome

Keep mmap as the production backend for now. Explicit io_uring successfully
raises outstanding I/O, but neither explicit backend beats mmap end to end on
the canonical 25% graph workload.

| backend | retained nodes/s | vs mmap | physical IOPS | reads/node | aggregate QD | cache result |
|---|---:|---:|---:|---:|---:|---|
| exact legacy mmap | **9,244,459** | 100% | 730,611 | 0.079 | 43.2 | about 26.6% kernel hits |
| buffered io_uring | 7,932,731 | 85.8% | 647,235 | 0.082 | 291.6 | about 24.2% kernel hits |
| O_DIRECT io_uring, cacheless | **8,640,104** | **93.5%** | **929,078** | 0.108 | 122.7 | none |
| O_DIRECT io_uring, 16 GiB cache | 7,215,159 | 78.0% | 640,601 | 0.089 | 87.8 | 17.6% userspace hits |

The maximum retained steady sample rate is therefore **9.244M nodes/s** on
mmap; the maximum retained explicit-I/O rate is **8.640M nodes/s** on cacheless
Direct I/O. A different valid long Direct tuning run reached 9.000M nodes/s,
but the table retains the run using the same canonical 400k-warmup seed stream
as mmap and buffered. A 10.04M buffered short-window point is excluded because
its page cache was still growing and its window was only 8.4 seconds.

The best explicit arm is cacheless Direct I/O at 8.64M nodes/s, within 6.5% of
mmap. It raises physical IOPS by 27% and removes edge-data major faults, but its
929k IOPS are only 14.1% of the measured 6.573M multi-file XFS ceiling. The
bottleneck has moved above the storage device, into page planning, request
grouping, handoff, page-object management, and result consumption.

The simple userspace cache validates the original concern: Linux's cache is
materially more efficient here. Saving 17.6% of Direct-I/O misses costs 16.5%
of cacheless throughput, while mmap combines a larger hit fraction with much less
per-hit work. A more sophisticated userspace cache is not the next justified
step without a graph-layout change or trace evidence showing much stronger
reuse.

## Dataset and host

The requested approximately-100B-edge index is:

`/mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor`

- 19,645,490,124 nodes;
- 76,475,264,044 stored directed edges (38,237,632,022 source rows stored in
  both directions);
- 1.058 TB total allocated index size;
- 7,506 `edges.bin` files totaling 866.874 GiB;
- 7,506 `offsets.bin` files totaling 38.880 GiB;
- six node types with 1,251 shards each;
- every node type in this index declares node rowpack data;
- all offsets use two-byte deltas.

The host has a 96-core/192-thread AMD Threadripper PRO 9995WX, 502 GiB RAM, and
XFS on md127 RAID0 over four Samsung 9100 PRO 8 TB NVMe drives. md uses a
512 KiB chunk (2 MiB full stripe). The benchmark process retained enough file
descriptors for all shard files; `RLIMIT_NOFILE` is 524,288.

The devservers were stopped before the measurements. Every index and block-I/O
operation in this work was read-only.

## Current index layout

Each shard has a dense local-node-ID namespace. `offsets.bin` maps local node
`i` to the byte interval `[offset[i], offset[i+1])` in `edges.bin`.

Offsets are stored in canonical groups of 64:

```text
[u16 count][u64 absolute base][count - 1 fixed-width deltas]
```

At the two-byte width used here, a full chunk is 136 bytes, or 2.125 bytes per
offset. That compactness is why the node-to-record directory for 19.6B nodes is
38.88 GiB rather than roughly 146 GiB for plain `u64` offsets. The reader now
computes canonical chunk positions arithmetically instead of retaining a
24-byte heap descriptor per chunk; this removes about 6.86 GiB of duplicate
RAM. Offset pages remain mmap-backed and are prefaulted before measurement.

An `edges.bin` node record is:

```text
[1-4 byte count for each relation]
[2/3/4 byte rowpack length, when node data exists]
[fixed-width packed relation 0 entries]
...
[fixed-width packed relation N entries]
[rowpack PK and feature bytes]
```

Records and their prefixes are not page aligned. A two-byte prefix or an edge
entry can cross a 4 KiB boundary. Zero-edge/no-rowpack nodes may have repeated
offsets and consume no edge-blob bytes. A table with only PK/FKs is still not
necessarily topology-only: FKs become graph relations, while a stored PK may
remain in the rowpack tail. Feature-heavy tables have a much larger tail in the
same record.

The explicit planner uses the resident offsets twice:

1. read every page intersecting the bounded relation-count/rowpack-length
   prefix, including page-straddling prefixes;
2. derive relation byte spans, reproduce the production RNG's selected entry
   indices, and read only pages containing those selected entries.

It does not read whole hub adjacency lists for fanout 10, and it does not fetch
the feature tail for this topology-only benchmark. This is essential: reading
the full topology span would amplify I/O relative to mmap, whose sampler jumps
directly to selected fixed-width entries.

## Benchmark contract

The repaired canonical plan is:

```text
edge_user_ip_edges_src_to_dst:10;
edge_user_ip_edges_dst_to_src:10
```

Its printed digest is `e11d45bd83962f04`. The earlier 11.456M mmap result is
not the new baseline: its script passed `--relations`, but the old benchmark did
not parse that flag and silently sampled its default relation set. The harness
now rejects unknown/duplicate arguments, accepts only `--plan`, prints the
canonical plan and digest, and fails the phase on the first sampling error.

Headline workload and memory settings were identical across backends; thread,
batch, and ring geometry were tuned per backend to report its maximum retained
rate:

- 4,000,000 measured seeds after 400,000 discarded warmup seeds; the separate
  16 GiB cache diagnostic used 2,000,000 warmup seeds so the cache was full
  before measurement;
- mmap: 144 outer/engine threads, batch 100;
- buffered: 144 outer/engine threads, batch 50, 8 rings x QD128;
- Direct: 160 outer/engine threads, batch 50, 16 rings x QD48;
- seed 42, uniform sampling, same seed stream;
- `MemoryMax=96G`, `MemorySwapMax=0`;
- offset prefault before warmup;
- file-specific `POSIX_FADV_DONTNEED` before every independent process;
- `/proc/diskstats` deltas for nvme0n1 through nvme3n1 over only the measured
  window.

The harness now also reuses one `StatePool` across discarded warmup and measured
phases. An earlier version created a new pool for measurement, so its warmup did
not pay graph-state capacity growth and its memory trajectory was misleading.

Each point produced about 84.4 output nodes and 85.0 edges per seed. Small
production checks over identical seeds produced 84,592 nodes, 85,162 edges,
and digest `267756e5024b65ae` for mmap, buffered, and Direct I/O. The long
performance runs intentionally disabled digest construction after this
correctness gate.

These are one retained long run per headline arm, not publication-quality
confidence intervals. Every measured window exceeds 34 seconds, but three
randomized repetitions remain a release gate.

The retained points can be reproduced with the checked-in driver (the canonical
plan is its default):

```bash
index=/mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor
DG_BENCH_BACKENDS=mmap DG_BENCH_THREADS=144 DG_BENCH_BATCH_SIZE=100 \
  DG_BENCH_NUM_SEEDS=4000000 DG_BENCH_WARMUP_SEEDS=400000 \
  scripts/bench_sampling_iops.sh "$index"
DG_BENCH_BACKENDS=buffered DG_BENCH_THREADS=144 DG_BENCH_BATCH_SIZE=50 \
  DG_BENCH_NUM_SEEDS=4000000 DG_BENCH_WARMUP_SEEDS=400000 \
  scripts/bench_sampling_iops.sh "$index"
DG_BENCH_BACKENDS=direct DG_BENCH_THREADS=160 DG_BENCH_BATCH_SIZE=50 \
  DG_BENCH_DIRECT_CACHE_GB=0 DG_BENCH_NUM_SEEDS=4000000 \
  DG_BENCH_WARMUP_SEEDS=400000 scripts/bench_sampling_iops.sh "$index"
```

Set `DG_BENCH_DIRECT_CACHE_GB=16` and
`DG_BENCH_WARMUP_SEEDS=2000000` for the cache diagnostic. The harness records
member IOPS/QD itself; `scripts/diskstats.sh -i 2 -d
'^(nvme[0-3]n1|md127)$'` provides an independent live view.

## Detailed retained results

### mmap

- 337.7M output nodes in 36.533 s: 9.244M nodes/s;
- 26,691,327 physical reads: 730,611 IOPS;
- aggregate QD43.2 and 101.82 GiB read;
- 26,691,328 major and 9,247,277 minor process faults;
- cgroup page-cache charge 51.8 to 54.8 GiB during measurement.

Using the explicit planner's identical 36,369,231 required page references as
the logical denominator gives an estimated 26.6% kernel-cache hit fraction.
The current workload therefore does not have queue depth zero: it has about
10.6 requests outstanding per drive, still far below fio's QD256 per drive.

### Buffered io_uring

- 337.7M output nodes in 42.574 s: 7.933M nodes/s;
- 36,369,231 logical page submissions but 27,555,270 physical reads;
- 647,235 physical IOPS, aggregate QD291.6, 105.25 GiB read;
- estimated 24.2% kernel-cache hit fraction;
- no major faults in the explicit edge path.

Cross-shard wave batching reduced 31.91M logical shard jobs to 1.94M producer
envelopes and 221k owner read batches. During the measured phase 2,742 aligned
frames were allocated, 36.37M were reused/recycled, and none were dropped for a
full recycle reserve. The file-cache charge was stable at 51.8 to 51.5 GiB.

Phase 1 no longer triggers a blocking mmap fault for each requested page.
Owners asynchronously submit a header wave and later a selected-entry wave;
the calling sample task waits at each wave boundary while other sample batches
remain runnable. This is real I/O overlap, but not a fully continuation-driven
sampler.

io_uring does not make a page-cache hit slow at the storage layer. It does make
that hit much more expensive than a resident mmap load in this implementation:
the request crosses the producer queue, SQ/CQ, kernel buffered-read machinery,
an explicit 4 KiB destination, an Arc-backed prepared page, and lookup/copy
logic. The measured end-to-end result is 14.2% below mmap despite similar cache
effectiveness.

### Cacheless Direct I/O

- 337.7M output nodes in 39.088 s: 8.640M nodes/s;
- 36,369,231 submitted pages and 36,316,058 device reads;
- 929,078 IOPS, aggregate QD122.7, 138.74 GiB read;
- zero major faults in the explicit edge path.

Tuning found a clear application-specific knee at 16 rings x QD48, owner batch
target eight ring depths, batch 50, and 160 sampler threads. Increasing queue
capacity beyond the knee raised physical QD but lowered nodes/s, matching fio's
finding that aggregate QD2048 adds negligible IOPS while increasing latency.
Cross-shard batching raised the cacheless long rate from 6.72M to 8.64M; a
separate valid long tuning point reached 9.00M with a different warmup stream.

### Direct I/O with 16 GiB userspace cache

- 337.7M output nodes in 46.800 s: 7.215M nodes/s;
- 6,392,058 hits, 29,964,483 misses, 6,418 coalesced waiters;
- 17.6% hit fraction, 640,601 physical IOPS, QD87.8;
- 29,964,483 FIFO evictions after the cache-filled warmup;
- 3,609 major faults and a cgroup file-cache charge drop from 38.9 to 19.6 GiB.

The cache has 256 mutex-sharded maps, per-key LOADING miss coalescing, FIFO
replacement, and one Arc lease per prepared unique page. There is no global LRU
lock. A 200-thread unit test confirms one physical owner for a same-page miss.
Nevertheless, hashing, partition locking, Arc traffic, replacement, and the
footprint/fragmentation of millions of separately aligned frames outweigh the
saved I/O. The offset refaults also show why equal nominal payload bytes are not
equal resident memory.

## Storage ceilings and IOPS gap

The corrected read-only fio control uses 128 large real `edges.bin` files,
32 processes, io_uring, O_DIRECT, 4 KiB random reads, QD32 per worker
(aggregate QD1024), a 5-second ramp, and 30-second measurement:

- **6.573M IOPS / 25.1 GiB/s**;
- 155.6 us mean latency, p50 125 us, p95 343 us, p99 594 us;
- md127 utilization 99.7%; all four members balanced and essentially 100% busy.

This exact fio point was repeated after the long sampler sweep and still
reached 6.573M IOPS. NVMe composite/sensor temperatures were 38-54 C across the
four members, ruling out thermal throttling as the reason sampler IOPS were
lower.

QD2048 reaches about 6.62M IOPS but roughly doubles latency, so QD1024 is the
correct knee. The 6.573M XFS result matches the supplied 6.56M raw-md result:
the earlier 2.7M ceiling was worker/inode concurrency, not RAID0 or XFS. It is
74.7% of the drives' 8.8M advertised aggregate. Against the more conservative
industry rule of thumb in the task—about 2M IOPS per drive, or 8.0M total—the
measured filesystem knee is 82.2%.

| path | physical IOPS | of 6.573M XFS | of advertised 8.8M | unused vs XFS |
|---|---:|---:|---:|---:|
| mmap | 730,611 | 11.1% | 8.3% | 5.84M |
| buffered io_uring | 647,235 | 9.8% | 7.4% | 5.93M |
| Direct I/O, cacheless | **929,078** | **14.1%** | **10.6%** | 5.64M |
| Direct I/O, 16 GiB cache | 640,601 | 9.7% | 7.3% | 5.93M |

The explicit engine cannot currently generate near-ceiling IOPS even with QD
available. Little's-law latency is roughly 132 us for the cacheless Direct arm
(QD123 / 929k IOPS), versus about 156 us at the much deeper fio knee. More queue
depth alone is not the fix: the application submits only about 930k planned
pages/s. Buffered depth sweeps raised physical QD from about 300 to 400/510 at
aggregate SQ capacities 1536/2048 but reduced nodes/s because latency and owner
work increased.

The provider microbenchmark is healthier. On one 441 MB shard, 32 clients and
32 rings x QD32 sustain **2.123M submitted pages/s** with 256-page batches and
zero recycle drops, or 83.5% of the 2.544M single-file fio ceiling. The provider
therefore has measurable overhead but is more than twice the end-to-end
cacheless Direct rate of 929k pages/s. The remaining gap comes primarily from
the engine's many small two-stage shard groups and page-consumption path. A
multi-file captured-trace provider replay is the next isolating benchmark.

Raw per-member fio remains blocked because `/dev/nvme*` is `root:disk` mode
0660 and this account lacks raw-device permission. The ready script uses 32
workers and aggregate QD1024 and is strictly read-only:

```bash
FIO_ALLOW_RAW_READ=1 FIO_WORKERS=32 FIO_IODEPTH=32 \
  scripts/bench_fio_raw_matrix.sh
```

## Cache unit: 4 KiB page or node ID?

Use 4 KiB pages for any next cache experiment on the current format.

- Physical I/O is 4 KiB, so a node cache still needs a bounce buffer and must
  decide which subset of a variable record to retain.
- The dense node ID already maps through resident compact offsets. A separate
  cache pointer/tag per 19.6B nodes would duplicate the most valuable index
  structure and consume prohibitive memory.
- Page keys need only `(file_id, page_number)` and naturally share bytes among
  nearby small/key-only node records and repeated sampled entries.
- Hubs should cache selected pages, not whole adjacency records.
- The current 17.6% hit result is not high enough to justify finer-grained
  lookup, allocation, contention, and eviction state.

This decision changes if the physical layout changes. For feature-heavy tables,
interleaving rowpack tails with topology lowers useful topology density and can
make a cached page carry irrelevant features. The stronger redesign is:

```text
resident compact node offsets/counts
             |
             +--> topology.bin: counts + packed adjacency
             |
             +--> rowpack.bin: PK/features, fetched only on demand
```

With topology separated, a page cache remains simple and dense. Only then use
captured traces to compare an exact-node L1, page L2, and hybrid at equal total
bytes. The existing offset map should remain the node-to-record directory for
all variants; do not add one pointer per node to the cache.

For truly key-only schemas, the split makes topology pages especially dense.
For feature-heavy schemas, it prevents feature bytes from evicting adjacency.
That format leverage is more promising than recreating Linux page cache around
the existing interleaved record.

## Recommendation

1. Keep the exact mmap path as default for now. It is currently 7.0% faster
   than the retained cacheless Direct arm and shares cache across processes.
   Cacheless Direct is close enough to justify the next completion-driven
   scheduling prototype; buffered is not the preferred production candidate.
2. Do not combine io_uring with mmap expecting asynchronous page faults.
   io_uring reads complete into explicit buffers; mmap faults themselves remain
   synchronous. Buffered io_uring has now tested that alternative and lost.
3. Preserve the resident 38.88 GiB compact offset map and the removal of the
   6.86 GiB duplicate chunk directory.
4. Build a captured multi-file page-trace replay before changing engine
   scheduling. It must approach at least 70% of 6.573M IOPS before storage can
   be the claimed end-to-end limiter.
5. Profile the explicit engine path for cycles in header planning, the remaining
   3.75M producer envelopes, queues, Arc/page lookup, and consumption. Replace
   the two global header/data wave waits with completion-driven microbatches;
   cacheless Direct is now close enough to mmap for this work to be justified.
6. If Direct I/O is revisited, use slab-allocated frames and a fixed 4-/8-way
   set cache with compact tags, batch pinning, and uncached miss bypass. Do not
   use one aligned allocation, HashMap node, mutex acquisition, and Arc RMW per
   hit. Validate it first against a holdout trace.
7. Prefer a topology/rowpack format split over a node-level cache. Rebenchmark
   key-only and feature-heavy projections separately.
8. Run the raw member-device matrix when permissions are available, then run
   three randomized repetitions of every retained sampler point before a
   production decision.

## Implementation and validation inventory

The branch contains:

- `diskgraph-index/src/page_io.rs`: persistent io_uring owners, aligned frame
  recycling, EOF-short handling, prepared pages, O_DIRECT, and the sharded
  cache;
- `diskgraph-index/src/reader.rs`: arithmetic offset chunks, boundary-safe
  headers, resident-offset prefaulting, and async page preparation;
- `diskgraph-engine/src/engine.rs`: sampling-aware non-fused topology adapter
  that preserves the exact mmap control;
- `diskgraph-engine/src/bin/dg-engine-bench.rs`: strict/error-propagating
  three-backend production harness and I/O/fault counters;
- `diskgraph-index/src/bin/dg-page-io-bench.rs`: provider microbenchmark;
- `scripts/bench_sampling_iops.sh`, `scripts/bench_fio_iops.sh`,
  `scripts/bench_fio_raw_matrix.sh`, and `scripts/evict_file_cache.py`.

Validation completed:

- 23 `diskgraph-index` library tests;
- 29 index end-to-end tests;
- 47 `diskgraph-engine` library tests;
- production mmap/buffered/direct count and digest equality;
- actual buffered and O_DIRECT 1,373-byte EOF-tail reads;
- boundary/dedup/valid-length tests;
- 200-thread direct-cache miss coalescing;
- multi-file batch response mapping and failure-wakeup coverage;
- strict CLI and shell syntax checks.

Six independent design reviews—three per proposal, covering kernel/I/O
semantics, concurrency/cache design, and benchmark simplicity/validity—caused
the following material changes before implementation: exact sampled-entry page
planning, page-straddling prefix support, valid EOF-short reads, a separate
topology parser instead of truncated `EdgeListView`, retained owned FDs,
unchanged legacy mmap control, honest blocking-wave semantics, strict benchmark
errors/arguments, fixed total memory, enough fio workers/QD, batch-only page
leases, miss I/O independent of cache admission, and deferral of epochs/node-L1
complexity.

The explicit implementation is still experimental. It supports canonical
non-fused topology sampling; fused topology/feature fetch is rejected rather
than silently falling back. Production hardening still needs runtime shutdown,
registered-resource experiments, direct-alignment discovery, continuation
scheduling if justified, feature-heavy tests, and repeated statistical runs.
