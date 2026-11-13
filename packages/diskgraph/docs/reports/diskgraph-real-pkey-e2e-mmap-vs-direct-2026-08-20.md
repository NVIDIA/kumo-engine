# Real-primary-key sampling: mmap versus cached O_DIRECT

This report measures the complete CPU-side lookup contract: choose source rows
uniformly from the original Parquet node table, resolve each real primary key
through the DiskGraph primary-key map, sample a three-hop graph neighborhood,
and return the real primary keys of every sampled node. It complements the
integer-ID sampler results elsewhere in this directory.

The identity results cover 1%, 5%, and 25% scales on the same host. A separate
[organic Pinterest report](diskgraph-pinterest-organic-capped-bfs-2026-08-21.md)
uses the same real-primary-key contract with an 80-neighbor capped traversal.

## Test protocol

- Host: AMD Ryzen Threadripper PRO 9995WX, 96 physical cores / 192 logical
  CPUs, 502.973 GiB RAM.
- Storage: four-NVMe md/XFS RAID0 at `/mnt/localssd`; the retained fio reference
  is 6.577M 4 KiB random-read IOPS at aggregate QD1024.
- Backends: the process selects either mmap or cached O_DIRECT at startup.
  Direct uses 32 io_uring owners at QD32 per owner, 16-page adjacent-read
  coalescing, and a cache geometry selected by end-to-end nodes/s under the
  same 500 GiB process limit.
- Memory: `MemoryMax=500G`, `MemorySwapMax=0`; offsets are prefaulted before
  warmup. Each point runs in a fresh process after file-scoped
  `POSIX_FADV_DONTNEED` eviction of the index.
- Seeds: source row positions are drawn uniformly with replacement across all
  Parquet files for the source node table. The benchmark reads only the primary
  key column, then performs the timed primary-key lookup during sampling.
- Output: only each sampled node table's primary-key column is materialized.
  The reported returned-PK rate is therefore also the sampled-node rate.
- Identity plan: three hops, fanout 10 on all four identity relations in the
  valid direction for each hop. Plan digest: `43d5badeb6e6f065`.
- Retention gate: at least 30 seconds of measured sampling, flat cache occupancy,
  no swap, and one returned primary key for every sampled node. mmap and Direct
  use independently selected thread counts but the same graph, source-key
  stream, batch size, fanout, RNG seed, and memory limit.

The engine does not currently emit a stable order for a multithreaded sample:
two mmap control repetitions of the same keys produce different raw digests.
Consequently, backend correctness is checked with identical node/edge counts
and PK cardinality, not a misleading byte-order digest. This benchmark fails
on a missing input key or missing/null returned primary key.

## Results

| graph | index GiB | index / RAM | backend | threads | nodes/s | source keys/s | returned PK GiB/s | physical IOPS | avg QD | cache hit | avg cores | CPU-s / M nodes | speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity 1% | 38.003 | 0.076x | mmap | 192 | 30.038M | 19.842k | 1.12 | 0.000027M | 0.1 | kernel | 170.3 | 5.670 | 1.000x |
| identity 1% | 38.003 | 0.076x | cached O_DIRECT | 160 | **81.047M** | **53.537k** | **3.02** | 0.019895M | 0.9 | 99.98% | 154.6 | **1.908** | **2.698x** |
| identity 5% | 193.547 | 0.385x | mmap | 192 | 21.523M | 14.218k | 0.80 | 0.000038M | 0.1 | kernel | 171.6 | 7.973 | 1.000x |
| identity 5% | 193.547 | 0.385x | cached O_DIRECT | 160 | **64.087M** | **42.337k** | **2.39** | 0.103305M | 6.1 | 99.85% | 147.2 | **2.297** | **2.978x** |
| identity 25% | 985.290 | 1.959x | mmap | 192 | 4.314M | 2.849k | 0.16 | 1.486M | 64.5 | kernel | 101.9 | 23.622 | 1.000x |
| identity 25% | 985.290 | 1.959x | cached O_DIRECT | 128 | **13.939M** | **9.207k** | **0.52** | **6.026M** | **559.7** | 60.75% | **59.9** | **4.297** | **3.231x** |

### Identity 1%

The source contains 87,070,770 user keys. The complete index contains
785,819,605 nodes and 3,059,010,562 directed edges and occupies 38.003 GiB,
7.6% of host RAM. Both retained points use 300,000 warmup source keys followed
by the same 2,000,000 measured source keys in batches of 512. Each key produces
1,513.8 sampled nodes and 1,605.2 sampled edges on average.

mmap completes in 100.795 seconds at 30.038M nodes/s. The kernel page cache is
flat at 35.2 GiB during measurement, disk traffic falls to 27 IOPS, and the
process averages 170.3 cores. Cached O_DIRECT completes in 37.357 seconds at
81.047M nodes/s. Its resident cache is flat at 33.2 GiB, it averages 154.6
cores, and it performs 19,895 physical IOPS. The Direct path is 2.698x faster
and reduces CPU time per million returned nodes from 5.670 to 1.908 seconds
(66.4%).

This graph is entirely resident after warmup; the SSD is not the limiter. The
result isolates the CPU cost of the two lookup paths. The fixed userspace cache
avoids the kernel virtual-memory/page-cache bookkeeping on each random file-page
lookup while retaining the same 4 KiB payload granularity. The measured Direct
path also reports zero tracked heap allocations per sampled node.

### Identity 5%

The source contains 435,353,851 user keys. The index contains 3,929,098,024
nodes and 15,295,052,808 directed edges and occupies 193.547 GiB, 38.5% of
host RAM. Both retained points use 2,000,000 warmup source keys followed by the
same 1,500,000 measured source keys. Each key produces 1,513.7 sampled nodes
and 1,605.1 sampled edges on average.

mmap completes in 105.498 seconds at 21.523M nodes/s. Its page cache is flat
at 179.4 GiB during measurement and the process averages 171.6 cores. Cached
O_DIRECT completes in 35.430 seconds at 64.087M nodes/s. Its cache is flat at
169.3 GiB, it averages 147.2 cores, and its hit rate is 99.85%. Direct is
2.978x faster and reduces CPU time per million returned nodes from 7.973 to
2.297 seconds (71.2%). Both paths are CPU-bound after warmup.

Returning PKs adds roughly one cached node-body page lookup per sampled node.
That exposed set conflicts that the topology-only workload barely exercised.
The cache keeps 16,777,216 sets and increases associativity in 64 GiB steps:

| configured cache | geometry | resident payload | nodes/s | physical IOPS | hit rate |
|---:|---:|---:|---:|---:|---:|
| 256 GiB | 4 ways | 156.3 GiB | 45.892M | 2.187M | 95.67% |
| 320 GiB | 5 ways | 164.6 GiB | 54.213M | 0.935M | 98.43% |
| 384 GiB | 6 ways | 168.0 GiB | 60.492M | 0.338M | 99.49% |
| 448 GiB | 7 ways | 169.3 GiB | **63.725M** | **0.105M** | **99.85%** |
| 512 GiB | 33,554,432 sets x 4 ways | 167.9 GiB | 59.236M | 0.374M | 99.42% |

These are 500,000-key diagnostics after identical 2,000,000-key warmups; the
448 GiB row is the retained geometry. The configured value is an addressable
slot cap, not committed payload memory. RSS was about 223 GiB during tuning,
and the retained payload occupied 169.3 GiB under `MemoryMax=500G` with no
swap. At 512 GiB the implementation doubles the set count and returns to four
ways, so conflict misses and end-to-end time regress. This explains the
otherwise surprising non-monotonic result; it is not a telemetry error.

### Identity 25%

The source contains 2,176,769,256 user keys. The index contains
19,645,490,124 nodes and 76,475,264,044 directed edges and occupies 985.290
GiB, 1.959 times host RAM. Both retained points use 1,000,000 warmup source
keys followed by the same 350,000 measured keys. Each key produces 1,514.0
sampled nodes and 1,605.4 sampled edges on average.

mmap completes in 122.843 seconds at 4.314M nodes/s. It averages 1.486M
physical reads/s at queue depth 64.5 and 101.9 cores. Its page cache is flat
at about 432 GiB during measurement. Cached O_DIRECT completes in 38.016
seconds at 13.939M nodes/s. It averages 6.026M physical reads/s at queue depth
559.7 and 59.9 cores. Its 320 GiB cache is flat at 319.5--319.6 GiB and reports
a 60.75% hit rate.

Direct is 3.231x faster and raises storage utilization from 22.6% to 91.6% of
the retained 6.577M fio reference. CPU time falls from 23.622 to 4.297 seconds
per million returned nodes, an 81.8% reduction. The Direct path performs 0.432
physical reads per returned node versus 0.344 for mmap; its advantage comes
from completing those useful misses at four times the rate, not from issuing
fewer reads. Both paths report zero tracked heap allocations in the measured
sampling window.

The 1,000,000-key warmup was selected from cache behavior, not by copying the
larger-scale topology protocol. It performs about 1.5 billion returned-node
lookups and leaves the explicit cache effectively full before measurement. A
10,000,000-key diagnostic was abandoned because it expanded to roughly 15
billion nodes and added tens of minutes without changing the capacity state.

## Reproduction

The benchmark entry point is `scripts/bench_sampling_iops.sh`. Real-primary-key
mode is enabled with:

```bash
DG_BENCH_PKEY_E2E=1 \
DG_BENCH_SEED_SOURCE_CONFIG=/mnt/localssd/datasets/GRAPH/diskgraph_config.json \
scripts/bench_sampling_iops.sh /mnt/localssd/diskgraph-index/GRAPH
```

The source scan and Parquet decoding finish before telemetry begins. The timed
input lookup, graph sample, and returned-primary-key materialization are all
inside the measurement window.
