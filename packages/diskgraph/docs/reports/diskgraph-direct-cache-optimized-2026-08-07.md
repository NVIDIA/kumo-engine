# DiskGraph cached O_DIRECT optimization baseline — 2026-08-07

## Outcome

Cached O_DIRECT now beats the retained production mmap baseline on the exact
same two-layer plan. It has separately validated storage-pressure,
maximum-throughput, and lower-CPU operating points:

| arm | nodes/s | logical pages/s | cache hit | physical IOPS | avg QD | drive busy | process CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| retained mmap | 9.244M | 0.996M | ~26.6% kernel-estimated | 0.731M | 43 | — | not recorded |
| Direct, 16 GiB, I/O-saturated | 45.684M | 4.919M | 15.77% | 4.137M | 586 | 96.8% | 53.5 cores (28%) |
| Direct, 320 GiB, high-hit | 160.591M | 17.291M | 95.61% | 0.758M | 97 | 71.8% | 103.6 cores (54%) |

The 16 GiB point is the storage-pressure proof: real sampler misses, not
synthetic reads, keep the four drives 96.8% busy while the process uses only
53.5 of 192 logical CPUs. It is 4.94x mmap by output nodes/s.

The 320 GiB point is the mmap-replacement proof: the cache holds 267.8 GiB of
edge pages after warmup, the offset/page-cache charge remains 44.9 GiB, and
95.61% of explicit page references hit userspace. It is 17.37x mmap by output
nodes/s and 17.36x by logical page lookup rate.

### Strict hot-cache CPU gate

The 320 GiB setting is the userspace cache capacity, not a separate 320 GiB
index. `/mnt/localssd` contains one completed standalone DiskGraph index, the
1.058 TiB index listed below. Before any further full-scale work, mmap and
Direct were given the same 150M-seed warmup and 60M-seed measured workload.
Both used the same canonical plan, jemalloc build, deterministic seed stream,
and 4,000-seed frontier batches.

| arm | coord / engine threads | wall | nodes/s | avg process cores | total CPU-s | physical IOPS | resident cache at end |
|---|---:|---:|---:|---:|---:|---:|---:|
| fully hot mmap control | 32 / 32 | 50.802s | 99.704M | 40.6 | 2,063.9 | 2 | 372.1 GiB file cache |
| Direct, wider CPU margin | 28 / 28 | 49.246s | 102.855M | 36.1 | 1,779.9 | 0.485M | 267.8 / 320 GiB edge cache |
| Direct, retained balanced point | 30 / 30 | 46.429s | 109.096M | 39.3 | 1,825.1 | 0.514M | 267.8 / 320 GiB edge cache |
| Direct, throughput-biased control | 32 / 32 | 43.991s | 115.141M | 42.6 | 1,872.3 | 0.543M | 267.8 / 320 GiB edge cache |

The retained 30-thread point is 9.42% faster than fully hot mmap while using
3.2% fewer simultaneous process CPU cores and 11.6% fewer total CPU-seconds
for the identical output workload. The 28-thread point independently gives an
11.1% average-core reduction and 13.8% fewer CPU-seconds while remaining 3.16%
faster. The 32-thread Direct row is included to show scaling, but it is not the
lower-instantaneous-CPU acceptance point.

Mmap ended with a 372.1 GiB useful file cache. Direct uses O_DIRECT for edge
pages, so file-cache pages left by the preceding mmap run do not serve its edge
lookups; the Direct working set reported by its own cache counters was 267.8
GiB. The binpacked offsets remain mmapped and prefaulted in both arms.

This lower-CPU gate passed. No subsequent full-scale benchmark was started as
part of this gate.

## Workload, data, and correctness gate

Index:

`/mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor`

- 19.645B nodes;
- 76.475B directed edges (about 38B undirected source edges);
- 1.058 TiB total index;
- 866.874 GiB `edges.bin` payload;
- 38.880 GiB binpacked offset indexes.

Canonical sampling plan:

```text
edge_user_ip_edges_src_to_dst:10;edge_user_ip_edges_dst_to_src:10
plan_digest=e11d45bd83962f04
```

Mmap and Direct were checked with the same 20,000 seeds after the sampler's
allocation-free Floyd uniform-without-replacement implementation was added.
Both produced:

```text
nodes  = 1,690,963
edges  = 1,702,426
digest = c2df5ab9ad1ce7c3
```

Two Direct control runs and the prefetch-on control also produced the same
digest. The full engine library test suite passes (48/48), as do the eight
focused page-I/O/cache tests.

## Hardware ceiling and remaining IOPS gap

The four-drive md RAID0 array reaches 6.573M 4 KiB random-read IOPS with 32 fio
workers and aggregate QD1024. QD2048 reaches only 6.62M while approximately
doubling latency. The advertised aggregate is about 8.8M IOPS, so fio itself
reaches roughly 75% of the advertised value.

The end-to-end 16 GiB sampler point reaches 4.137M IOPS, 62.9% of the observed
fio ceiling and 47.0% of the advertised ceiling. Its organic QD is 586 versus
fio's QD1024. Implied service time is about 142 us (`586 / 4.137M`), comparable
to fio at QD1024 (about 156 us). The primary residual gap is therefore fewer
outstanding misses, not slow individual O_DIRECT requests. The array is already
96.8% busy during the sampling window.

The high-hit arm should not saturate storage: only 4.39% of 17.291M logical
page references/s miss, so 0.758M physical IOPS is the expected result. Its
logical rate is 2.63x the physical fio ceiling because RAM serves most pages.

Raw per-drive fio is still needed to separate md/XFS/file-system overhead from
the device firmware limit.

## What was actually bottlenecking the cache path

The fixed-set lookup was not the dominant CPU cost. A measured all-hit pprof
profile attributed under 1% of CPU to `DirectCache::scan`. Before the fixes,
the major costs were:

1. glibc allocator/free-list locks and coalescing;
2. nested cross-pool Rayon work stealing and over-recursive request execution;
3. sparse 50-seed frontiers allocating page plans and output/scratch vectors
   once per nonempty shard;
4. one binary search per selected edge in `PreparedPages`;
5. unconditional cache-fill mutex/`notify_all`, despite almost no waiters;
6. per-page atomic statistics and owner-selection RMWs.

The allocation counter measured 3.067 allocations and 740.5 allocated bytes
per output node on the original 50-seed path. Frontier batching and ownership
cleanup reduce this to 0.663 allocations and 465.6 bytes per node. The counter
itself intentionally adds an atomic RMW per allocation and is not used for
headline throughput.

An allocator-only A/B was decisive: installed tcmalloc raised the hot path from
12.14M to 25.25M nodes/s. The checked-in Rust `jemalloc` feature reached
30.94M on the same short replay and is used for the retained optimized runs.
This is not a claim that residual allocations are desirable; it removes their
global serialization while the remaining plan arenas are made reusable.

## Implemented hot-path changes

- Replaced the benchmark's outer Rayon pool with a fixed number of ordinary
  coordinator threads. This removes cross-pool `install()` recursion and bounds
  pooled sampling states.
- Increased the real frontier batch from 50 to 4,000 seeds. This amortizes
  sparse-shard plans while preserving independent sampler requests.
- Flattened per-node relation plans into two contiguous arenas and replaced a
  per-shard header `HashMap` with a cursor over sorted node/header arrays.
- Replaced heap-based uniform index sampling with exact Floyd sampling using
  inline storage for the usual fanout 10.
- Moved already-sorted page vectors through planning/request boundaries instead
  of repeatedly cloning, sorting, and deduplicating them.
- Stored page numbers separately from prepared-page handles, reducing lookup
  metadata traffic, and added a monotone page cursor so selected edges pay one
  lower-bound per relation rather than one binary search per edge.
- Added a true all-hit ready path that bypasses the io_uring runtime entirely.
- Removed unconditional fill-side mutex/condition-variable notification.
  A duplicate lookup of a still-LOADING page performs a transient read; this
  occurred only 7,145 times against 521.4M hits in the final run.
- Batched cache-hit statistics per plan and owner selection per batch; reference
  metadata is written only on `0 -> 1`.
- Replaced the tag/pin retirement spin protocol with one `AtomicU16` pin word
  whose high bit is exclusive eviction ownership. Hits pin then revalidate;
  eviction acquires `0 -> EXCLUSIVE`; publication installs pin count 1 before
  release-publishing the tag.
- Added `Drop` cleanup for unfinished Direct read plans so cancellation cannot
  leak permanent LOADING slots.
- Padded and expanded miss-lock shards. No lock is taken on a hit.
- Accepted valid short final-page O_DIRECT reads and retained prefix-straddle
  handling.

## Current cache design and why it remains simple

The 320 GiB geometry is 16,777,216 sets x 5 ways (83.9M 4 KiB slots). Each
entry has a full 64-bit file/page tag, a durable pin/exclusive word, valid
length, approximate referenced byte, and data frame. Lookup is bounded:

```text
hash file/page key -> scan at most five tags -> pin -> revalidate -> bytes
```

There is no lock, heap allocation, runtime queue, or replacement work on the
cache-hit lookup. Miss serialization and replacement remain off that path.
The pprof result says a denser graph-page directory may be a useful later
simplification (about 0.85 GiB for one 32-bit entry per corpus page), but it is
not justified as the next performance project while lookup itself is under 1%
of CPU.

Caching 4 KiB pages is currently the right baseline. It matches O_DIRECT
alignment, keeps lifetime/reclamation bounded, and reuses the resident
binpacked offset index to locate exact selected records. Node-level caching
would add a finer-grained concurrent index and variable-sized allocator while
the measured five-way page lookup is already cheap. A node-level representation
should be reconsidered only if trace measurements show material 4 KiB read
amplification for tiny adjacency records.

## Index-layout implications

Each shard's mmapped `offsets.bin` maps local node id to an `edges.bin` record.
The record begins with encoded relation counts, followed by fixed-layout edge
entries for each relation, followed by the optional node-data tail. Explicit
sampling is two-wave:

1. read the small prefix pages needed to parse relation byte ranges;
2. read only pages containing selected topology entries.

For key-only tables (primary/foreign keys and no feature tail), records end at
topology and this layout maximizes node density. For feature-heavy tables, the
node-data tail may dominate record bytes, but topology selection deliberately
does not read it. The 38.88 GiB offset mmap remains resident/file-backed and is
prefaulted independently; it is not duplicated in the edge cache.

Explicit O_DIRECT fused feature fetching is still rejected rather than silently
misreading topology-only prepared bytes. A future Direct feature path needs a
separate cache/admission policy because feature payload access and graph
topology reuse have different traces.

## Reproduction

Build the benchmark with the scalable allocator explicitly:

```bash
cargo build --release -p diskgraph-engine --bin dg-engine-bench --features jemalloc
```

I/O-saturated arm:

```bash
target/release/dg-engine-bench \
  --index /mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor \
  --seed-type users \
  --plan 'edge_user_ip_edges_src_to_dst:10;edge_user_ip_edges_dst_to_src:10' \
  --warmup-seeds 1000000 --num-seeds 5000000 --batch-size 1000 \
  --threads 64 --engine-threads 64 --backend direct \
  --io-rings 16 --io-qd 64 --io-batch-mult 8 \
  --direct-cache-gb 16 \
  --devices nvme0n1,nvme1n1,nvme2n1,nvme3n1
```

High-hit arm:

```bash
target/release/dg-engine-bench \
  --index /mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor \
  --seed-type users \
  --plan 'edge_user_ip_edges_src_to_dst:10;edge_user_ip_edges_dst_to_src:10' \
  --warmup-seeds 150000000 --num-seeds 60000000 --batch-size 1000 \
  --threads 64 --engine-threads 64 --backend direct \
  --io-rings 16 --io-qd 64 --io-batch-mult 8 \
  --direct-cache-gb 320 --prefault-offsets \
  --devices nvme0n1,nvme1n1,nvme2n1,nvme3n1
```

Strict lower-CPU gate:

```bash
target/release/dg-engine-bench \
  --index /mnt/localssd/diskgraph-index/alt_detection_25pct-fixed-7265af98-full-monitor \
  --seed-type users \
  --plan 'edge_user_ip_edges_src_to_dst:10;edge_user_ip_edges_dst_to_src:10' \
  --warmup-seeds 150000000 --num-seeds 60000000 --batch-size 4000 \
  --threads 30 --engine-threads 30 --backend direct \
  --io-rings 16 --io-qd 64 --io-batch-mult 8 \
  --direct-cache-gb 320 --prefault-offsets \
  --devices nvme0n1,nvme1n1,nvme2n1,nvme3n1
```

## Remaining engineering work

- Move the remaining per-shard plan vectors into capacity-reused arenas. The
  end-to-end path is fast and no longer allocator-serialized with jemalloc, but
  the measured 0.663 allocations/node is not the desired zero-allocation
  steady state.
- Batch owner completion counters and transient-frame accounting further; they
  are miss-path work and no longer limit the high-hit arm.
- Add production-binary allocator wiring. A Rust library cannot select the
  process-global allocator for its embedding Python/server executable.
- Run raw fio against each drive and retain three randomized repetitions for
  publication-quality confidence intervals.
- Implement and benchmark explicit feature reads separately before enabling
  fused feature fetching for Direct I/O.
