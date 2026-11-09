# 100% identity graph: mmap versus cached O_DIRECT — 2026-08-07

## Result

Cached O_DIRECT is 4.616x faster than equivalently configured mmap on the
full-scale, three-hop all-edge workload. It consumes 67.4% less total CPU for
the fixed output, although it uses more cores concurrently to overlap I/O.

Both retained arms use 160 request threads, 128 engine threads, batch 512,
3M discarded warmup roots, and 3M independently measured roots. Each arm
starts in a fresh no-swap cgroup after per-index cache eviction, and both
output-verification gates pass.

| metric | mmap | cached O_DIRECT | Direct / mmap |
|---|---:|---:|---:|
| measured wall time | 200.818s | **43.505s** | **0.217x** |
| sampled nodes/s | 22.611M | **104.369M** | **4.616x** |
| sampled edges/s | 23.975M | **110.669M** | **4.616x** |
| roots/s | 14,939 | **68,957** | **4.616x** |
| process CPU, average cores | 44.5 | 67.0 | 1.506x |
| total process CPU | 8,943.4 CPU-s | **2,914.6 CPU-s** | **0.326x** |
| CPU-s per million nodes | 1.969 | **0.642** | **0.326x** |
| physical 4K read IOPS | 1.306M | **6.119M** | **4.687x** |
| aggregate queue depth | 78.6 | **807.5** | **10.274x** |
| average busy per drive | 89.0% | **99.3%** | — |
| physical reads/node | **0.058** | 0.059 | 1.016x |
| bytes read in window | **1,000.12 GiB** | 1,016.77 GiB | 1.017x |
| edge-cache hit rate | kernel not instrumented | 17.08% | — |
| cache at window end | 429.0 GiB file cache | 256.0 GiB Direct + 155.5 GiB offsets | — |
| minor faults | **34,586,202** | 104,462,698 | 3.020x |
| major faults | 262,174,694 | **0** | — |
| output digest | `afb40d6dce1cac0c` | `afb40d6dce1cac0c` | identical |

The fixed output is about 1,513.5 sampled nodes and 1,604.9 sampled edges per
root. Direct finishes 78.3% sooner and uses 67.4% fewer CPU-seconds. Its
concurrent CPU demand is higher, but 67.0 cores is only 35% of the 192 logical
CPUs while all four drives average 99.3% busy. The intended operating condition—
storage saturated, CPU not saturated—is satisfied.

Direct performs 1.6% more physical reads than mmap. Explicit asynchronous I/O
still wins because it maintains 10.2x the queue depth and overlaps misses rather
than blocking request threads in page faults.

## Dataset generation and ingest

The deterministic generator config is
`diskgraph-datagenerator/configs/alt_detection_full_scale.json`. It now writes
Zstd-compressed Parquet explicitly rather than relying on the uncompressed
Arrow default. A representative probe was 208,826,368 bytes compressed versus
420,118,528 bytes uncompressed, a 50.3% reduction; the random edge table alone
was 160,439,873 versus 321,450,964 bytes.

| artifact | value |
|---|---:|
| generator rows | 231,532,488,578 |
| Parquet shards | 18,496 |
| compressed source directory | 7,811,354,595,328 bytes |
| indexed nodes | 78,581,960,494 |
| directed indexed edges | 305,901,056,168 |
| retained index size | 4,384,905,142,272 bytes |
| generator command wall time | 28m40s |
| successful resumed worker wall time | 73m47s |
| end-to-end ingest wall time, including retry | 2h31m03s |
| ingest key parse/lookup/membership failures | 0 / 0 / 0 |
| dropped PK/timestamp rows | 0 |

Paths:

```text
/mnt/localssd/datasets/alt_detection_full_scale
/mnt/localssd/diskgraph-index/alt_detection_full_scale-zstd-1c1f7a9f-full-monitor
```

The first 202-worker `pk_map` attempt reached 493,559,736 KiB (about 470.7 GiB)
anonymous RSS and was killed by the global OOM killer. The phase-aware
supervisor preserved the completed 63m18s encode output and retried `pk_map`
at 101 workers. The
successful retry peaked at 295.6 GB in `pk_map`; later phases completed at the
normal worker count. Successful phase times were 25m49s `pk_map`, 21m49s
`edges_map`, 25m41s `index`, and 103ms finalize. Ingest produced 16.278 TB of
temporary data; the observed destination peak was about 16.56 TB before
cleanup. Compressed source generation kept the run comfortably within the
32.0 TB filesystem.

There are 49,967 duplicate-primary-key observations in the auxiliary
`alt_labels` table, whose generated label key is deliberately low-cardinality.
No rows were dropped, all 50,000 label rows are present, and this disconnected
table is not part of the user-start sampling plan.

## Workload and isolation

The seed type is `users`; fanout is 10 on every applicable relation:

```text
users -> {ip_addresses, mac_addresses, emails, btids}
      -> users
      -> {ip_addresses, mac_addresses, emails, btids}
```

The canonical plan digest is `43d5badeb6e6f065`. Each retained arm:

- starts after `POSIX_FADV_DONTNEED` eviction of all 62,217 index files;
- prefaults all 166,987,088,020 bytes (155.5 GiB) of binpacked offsets;
- runs in a fresh cgroup with `MemoryMax=500G` and `MemorySwapMax=0`;
- uses seed 42 and verifies output determinism on 20,000 roots;
- runs without competing devservers or storage workloads.

Direct uses a 256 GiB fixed cache, 32 rings, QD32/ring (aggregate configured
capacity 1024), I/O batch multiplier 8, and no coalescing delay. The 50% graph
used a 320 GiB cache, but that is not safe here: 320 GiB plus 155.5 GiB of
resident offsets leaves no room on a 502 GiB host for runtime metadata and
frontiers. The 256 GiB cache plus offsets occupies about 411.5 GiB before
runtime overhead and remains resident without swap or offset refault pressure.

## Ring, batching, and concurrency selection

The original 16-ring x QD64 geometry had the right aggregate configured depth
but was not the best way to expose it. A cacheless provider benchmark over the
same 128 largest edge files used by fio measured:

| provider arm | submitted IOPS | fraction of fio |
|---|---:|---:|
| fio, 32 workers x QD32 | **6.577M** | 100.0% |
| page runtime, 16 rings x QD64 | 5.373M | 81.7% |
| page runtime, 32 rings x QD32 | **6.004M** | **91.3%** |
| page runtime, 8 rings x QD128 | 3.533M | 53.7% |

The 32 x QD32 runtime also sustains 6.007M submitted IOPS when opened across
all 8,673 nonempty edge files, so shard-file fan-out and XFS are not the
end-to-end limiter. The production workload then selected frontier batching
and coordinator concurrency:

| threads | roots/batch | Direct nodes/s | IOPS | QD | average cores |
|---:|---:|---:|---:|---:|---:|
| 128 | 256 | 85.753M | 5.035M | 594.9 | 123.1 |
| 128 | **512** | **87.337M** | **5.139M** | **616.0** | **111.3** |
| 128 | 1024 | 76.711M | 4.535M | 566.6 | 102.9 |
| 160 | 512 | 88.381M | 5.200M | 611.1 | 126.2 |

These are short selection points from before the runtime hot-path cleanup.
Batch 512 improves both throughput and CPU over 256, while batch 1024 crosses
the memory/frame-lifetime knee and regresses. At that stage, moving from 128 to
160 request threads also bought only 1.2% throughput because every request
still entered a second work-stealing engine pool. After removing that nested
scheduler, request threads became inexpensive blocked I/O contexts: the
retained 160/512 run uses 67.0 cores, reaches 104.369M nodes/s, and supplies
enough independent header/data waves to keep the existing rings full.

## Runtime hot-path closure

The former complete sampler reached 5.054M IOPS against the same runtime's
6.007M multi-file ceiling. Profiling and source audit found that the cache and
ring owner were doing diagnostic synchronization and shared-ownership work per
physical 4K read, even though neither is part of the cache algorithm:

- global cache-miss, eviction, byte, frame-reuse, and frame-recycle atomic
  updates;
- an `Arc<File>` clone/drop for every page-sized submission;
- an `Arc` clone/drop of the same owner recycler for every returned frame;
- allocation of a temporary CQE vector on every completion pass.

The retained implementation batches cache events once per shard plan, shards
those event counters by sampling thread, batches owner byte/frame counters,
uses the `ReadJob`'s owned file to keep a raw descriptor live for page specs,
borrows the process-lifetime owner recycler, and consumes CQEs directly. It
does not add a scheduler, cache tier, continuation, or synthetic read.

| full-scale Direct arm | nodes/s | device IOPS | QD | average cores | CPU-s/M nodes |
|---|---:|---:|---:|---:|---:|
| before hot-path cleanup | 86.253M | 5.054M | 608.8 | 109.8 | 1.273 |
| per-page runtime cleanup | 98.058M | 5.739M | 734.1 | 108.9 | 1.111 |
| scheduler/data cleanup | 104.228M | 6.111M | 805.6 | 67.0 | 0.643 |
| **retained allocation cleanup** | **104.369M** | **6.119M** | **807.5** | **67.0** | **0.642** |

The final profile found that 24% of sampled CPU was spent stealing work and
advancing epochs in nested Rayon pools. The benchmark already has a fixed set
of request threads, while the ring-owner threads supply I/O concurrency. The
explicit path now consumes its ready shard list directly instead of scheduling
thousands of tiny jobs into a second pool. Request and output vectors retain
their capacity between calls, map insertion uses one entry probe, and stable
frontier buffers no longer shrink and regrow around a fixed threshold.

The page owner also moves uniquely-owned completed-page descriptors to the
cache publisher instead of allocating an `Arc<PageData>` for every physical
read. Only rare transient bypasses need a boxed descriptor. An `EINTR` retry
around `io_uring_enter` fixes the profiler-signal case without changing normal
submission. These are ownership and allocation removals, not a new cache or
scheduler.

A tempting attempt to record edges directly while decoding each prepared shard
was rejected: it extended cache-page pin lifetimes and reduced the adjacent
short result from 102.24M to 95.93M nodes/s. The retained path deliberately
releases prepared pages before its existing ordered state/output commit. The
IOPS increase in the retained code is demand-driven: nodes/s rises in the same
proportion while CPU falls sharply. Increasing the owner batch target from 8x
to 16x also regressed nodes/s, so the lower-latency 8x target remains.

## Allocation stop line

The steady sampler's remaining small-allocation churn was profiled with a
diagnostic-only jemalloc profiling feature and measured separately with global
allocation counters. The retained cleanup:

- appends every ready shard into one request-owned raw-edge vector and reuses
  one pair of decode buffers instead of allocating three vectors per shard;
- plans headers directly from already-sorted shard nodes and stores local ID
  plus record range in one exactly sized vector;
- exactly sizes header pages, parsed headers, range pages, cache miss plans,
  and batch-part vectors; and
- returns one flat completed-page vector per owner envelope with compact
  `(request index, page count)` spans instead of one page vector per shard.

| allocation diagnostic | earlier retained | final | change |
|---|---:|---:|---:|
| allocations / sampled node | 0.1930 | **0.0904** | **-53.2%** |
| allocated bytes / sampled node | 289.2 B | **258.0 B** | **-10.8%** |
| allocations over ~303M nodes | 58.49M | **27.38M** | **-31.11M** |

These counters perturb the allocator and scheduler, so their throughput is not
a retained performance point. The normal-binary steady run remains at the
storage ceiling and improves marginally from 104.228M to 104.369M nodes/s.

The final call-site profile shows that nearly all remaining calls are one
ownership vector per nonempty shard for header records/pages, parsed headers,
range pages, relation plans, and cache lookup results. Those vectors coexist
across two asynchronous I/O waves. Removing them requires a cross-shard flat
arena with spans or a shared object pool. The former is a substantial lifetime
rewrite; the latter adds contention and can retain hundreds of GiB across 160
concurrent requests. Neither is justified while Direct uses only 35% of CPUs
and already reaches 93% of fio. Literal zero allocation remains possible, but
is not a useful optimization target under the measured bottleneck.

## Cache locality

The 17.08% Direct hit rate is not caused by restricting roots to a small key
subset. Roots are uniformly distributed over all 8.707B users, and each hop
uses all four edge families. The 256 GiB cache is only about 6% of the
multi-terabyte edge corpus. The retained 3M-root window records 54,911,691
hits, 266,540,496 misses, and 265,592,248 evictions: it is full and in steady
replacement, not still filling. The hit rate falls from 40.47% on the fresh
50% graph to 17.08% here because the graph doubles while the safe cache budget
decreases from 320 to 256 GiB. The
remaining locality comes from repeated frontier nodes/hubs and page sharing
within batched three-hop requests.

## Storage ceiling

The retained fio ceiling is 6.577M 4K random-read IOPS at aggregate QD1024;
the advertised four-drive aggregate is about 8.8M IOPS.

- mmap reaches 19.8% of fio and 14.8% of advertised IOPS;
- Direct reaches 93.0% of fio and 69.5% of advertised IOPS;
- Direct's remaining gap to fio is 0.458M IOPS (7.0%);
- Direct averages 1.530M IOPS/drive versus 1.644M/drive under fio and about
  2.2M/drive advertised.

The page runtime reaches 6.007M IOPS across every edge shard, and the complete
sampler now reaches 6.119M with QD808: **101.9% of the isolated runtime while
also performing cache lookup, three-hop planning, decoding, sampling, and
output materialization**. The small excess is normal run/file-mix variation;
the sampler/runtime gap is closed. This came from removing redundant scheduling
and allocations and then supplying the existing runtime with 160 inexpensive
request contexts. No synthetic reads, deeper rings, or continuation component
was added.

## Scale comparison

| metric | 50% graph | 100% graph | 100% / 50% |
|---|---:|---:|---:|
| indexed nodes | 39.291B | 78.582B | 2.000x |
| directed edges | 152.951B | 305.901B | 2.000x |
| mmap nodes/s | 40.083M | 22.611M | 0.564x |
| Direct nodes/s | 140.853M | 104.369M | 0.741x |
| Direct hit rate | 40.47% | 17.08% | 0.422x |
| Direct physical IOPS | 5.929M | 6.119M | 1.032x |

The full graph lowers hit rate and therefore reaches the storage ceiling with
fewer nodes/s while driving 3.2% more physical IOPS. The explicit backend's
advantage grows from 3.514x at 50% scale to 4.616x here. The complete fresh
matrix is recorded in
[`diskgraph-all-edge-3hop-scale-matrix-2026-08-08.md`](diskgraph-all-edge-3hop-scale-matrix-2026-08-08.md).
