# Hotspot Pinterest graph: real-primary-key mmap versus cached O_DIRECT

This benchmark reproduces the original hotspot-based pruned Pinterest graph,
starts from real `pin_id` values read uniformly from the original Parquet
table, and returns the real primary key of every sampled node. The completed
index is 1,177.447 GiB, or 2.341 times the host's 502.973 GiB RAM.

Cached O_DIRECT reaches 9.962M nodes/s, versus 2.462M for mmap. Direct is
4.046 times faster, raises storage command throughput from 1.076M to 5.826M
IOPS, and reduces CPU time per million returned nodes from 33.953 to 6.294
seconds.

## Dataset generation and ingest

The canonical generator configuration is
`pinterest_omnisage_pruned.json` (SHA-256
`7e38e0552d6912257045cf30b3c18a0869c8c87f2c34d51a10b68cfe31408e02`).
The only generation change was enabling Zstd Parquet compression. Generation
completed in 3m29s and produced 11,200 Parquet files occupying 1.633 TiB.

The source contains 15.21B entity rows and 102B edge rows:

| table | rows |
|---|---:|
| pin | 4.36B |
| board | 4.70B |
| item | 2.82B |
| query | 3.33B |
| board-pin | 68.10B |
| pin-pin | 12.80B |
| query-pin | 8.40B |
| item-item | 0.20B |
| pin-item | 0.80B |
| board-item | 11.70B |

The six relationship tables are explicit edge tables. Ingest emits both
directions for each relationship, so the final index contains 15.21B nodes
and 204B directed adjacency entries across 12 relation directions. It reports
zero failed keys, duplicate primary keys, dropped duplicates, and dropped
timestamps.

Parallel ingest used 638 shards and 202 workers. It completed in 41m49s,
peaked at 303.9 GiB process RSS, and produced a byte-exact 1,264,273,813,732-B
index (1,177.447 GiB). Stage times were 7m09s encode, 10m11s primary-key map,
8m21s edge map, and 15m59s index construction. Every node table has all 638
`edges.bin`, `offsets.bin`, primary-key data, and primary-key hash shards; no
partial or temporary publication remains.

## Benchmark contract

- Host: AMD Ryzen Threadripper PRO 9995WX, 96 physical / 192 logical CPUs,
  502.973 GiB RAM.
- Storage: four-NVMe md/XFS RAID0. The retained filesystem reference is
  6.577M 4 KiB random-read IOPS at aggregate QD1024.
- Isolation: each backend runs in a fresh `MemoryMax=500G`,
  `MemorySwapMax=0` user scope. Clean file pages belonging to the index are
  evicted with `POSIX_FADV_DONTNEED` before each process. Offset files are
  prefaulted before warmup.
- Input: 30M real `pin_id` values are selected uniformly from all 4.36B source
  rows. The loader touches all 512 pin Parquet files and spans source rows
  65 through 4,359,999,983 (0.000001% through 100.000000%).
- Output: primary-key lookup, sampling, and returned-primary-key
  materialization are timed. The harness requires one non-null real primary
  key for every returned node.
- Traversal: two breadth-first hops. Pins first traverse fanout 10 over
  board-pin, pin-pin, query-pin, and pin-item; the second hop follows the
  reverse directions. Batch size is 512. Plan digest:
  `19ba1ebdfd392b6d`.
- Window: 20M warmup roots followed by the same 10M measured roots for each
  backend. mmap and Direct return the same verification node and edge counts.
  Raw byte-order digests are not used as a correctness claim because two mmap
  controls demonstrate nondeterministic output ordering.
- mmap uses 256 sampling and engine contexts. Direct uses 384 contexts,
  32 fixed io_uring owners at QD32 each, adjacent-read coalescing up to 16
  pages, and a fixed 320 GiB cache.

## Results

| backend | contexts | nodes/s | edges/s | roots/s | physical IOPS | fio utilization | avg QD | reads/node | cache hit | avg cores | CPU-s/M nodes | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmap | 256 | 2.462M | 2.614M | 29.690k | 1.076M | 16.4% | 69.0 | 0.437 | kernel | 83.6 | 33.953 | 1.000x |
| cached O_DIRECT | 384 | **9.962M** | **10.576M** | **120.124k** | **5.826M** | **88.6%** | **684.5** | 0.585 | 55.16% | **62.7** | **6.294** | **4.046x** |

Direct returns 4.046 times as many nodes per second while using 25% fewer
cores. Normalized CPU time falls by 81.5%. Its four SSDs remain fully busy in
the measured window, but CPU utilization is only 33% of the 192 logical CPUs.
The result is storage-bound rather than CPU-bound.

mmap keeps 453.0 GiB in the kernel page cache and therefore reads fewer pages
per returned node. Direct ends at 319.9 GiB in its data cache and about 53 GiB
in the kernel page cache. Despite the smaller data cache, explicit batched
submission raises device IOPS by 5.41 times and sustains nearly ten million
returned-node primary-key lookups per second.

Both backends report zero tracked heap allocations in the measured sampling
window. Direct cache occupancy moves only from 319.7 to 319.9 GiB; mmap's page
cache remains at 453.0 GiB. Neither retained result depends on a growing cache.

## Page and I/O accounting

The Direct run records:

- 1,083,826,614 logical cache-page lookups, or 1.307 per returned node;
- 597,796,587 hits and 486,030,027 misses, for a 55.16% hit rate and 0.586
  miss pages per returned node;
- exactly 486,030,027 pages submitted to the storage path;
- 484,947,809 SQEs after adjacent pages are coalesced; and
- 484,993,349 physical device reads, including non-edge index traffic.

The relationship between nodes/s and IOPS is therefore expected:

```text
1.307 lookups/node × (1 - 0.5516 hit rate) = 0.586 misses/node
9.962M nodes/s × 0.585 physical reads/node = 5.826M physical IOPS
```

One returned node does not imply one storage operation. Multiple node records
share a 4 KiB page, cached pages require no SSD read, and some returned nodes
do not require another adjacency expansion. The independent device counters,
cache counters, and submitted-page counters agree; the high nodes/s result is
not caused by misclassified cache hits or omitted I/O.

## Concurrency selection

Five-million-root diagnostics following identical two-million-root warmups
were used only to select Direct concurrency:

| Direct contexts | nodes/s | physical IOPS | avg QD | avg cores |
|---:|---:|---:|---:|---:|
| 96 | 8.272M | 4.888M | 575 | 47.0 |
| 192 | 9.237M | 5.464M | 642 | 55.0 |
| 256 | 9.525M | 5.638M | 662 | 58.6 |
| 320 | 9.724M | 5.761M | 677 | 61.0 |
| 384 | **9.857M** | **5.845M** | **692** | **62.5** |
| 512 | 9.817M | 5.831M | 688 | 65.3 |
| 640 | 9.737M | 5.793M | 682 | 68.3 |

Reads per node and cache hit rate remain nearly constant through 384, so the
gain is proportional to useful storage throughput. At 512 and 640, nodes/s
regresses while CPU time, loading bypasses, and transient frame pressure rise.
The retained Direct setting is therefore 384.

The mmap diagnostic increased from 2.236M nodes/s at 128 contexts to 2.643M
at 256. A 320-context diagnostic reached 2.752M but raised average cores from
80.4 to 103.4 and CPU time per node by 23%. The retained mmap control uses 256
contexts as the requested sufficient baseline rather than extending
oversubscription further.

## Reproduction

```bash
# mmap control
DG_BENCH_BACKENDS=mmap DG_BENCH_THREADS=256 \
DG_BENCH_WARMUP_SEEDS=20000000 DG_BENCH_NUM_SEEDS=10000000 \
scripts/bench_pinterest_pruned_pkey.sh INDEX SOURCE/diskgraph_config.json

# cached O_DIRECT
DG_BENCH_BACKENDS=direct DG_BENCH_THREADS=384 \
DG_BENCH_WARMUP_SEEDS=20000000 DG_BENCH_NUM_SEEDS=10000000 \
scripts/bench_pinterest_pruned_pkey.sh INDEX SOURCE/diskgraph_config.json
```
