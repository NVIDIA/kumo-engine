# Local NVMe mmap versus cached O_DIRECT scale check — 2026-08-18

## Result

This additive canonical three-hop identity-sampler rerun uses the existing
alt_detection Parquet data at s3://kumo-unit-test/diskgraph-datagenerator/.
The 1% index fits in memory; 10% is the largest practical local-scratch point:
394.12 GiB, 1.61x the 245 GiB RAM available to this host.

| scale | index GiB | index / available RAM | mmap nodes/s | Direct nodes/s | Direct / mmap | mmap IOPS | Direct IOPS | Direct hit | mmap cores | Direct cores | Direct total CPU / mmap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 38.00 | 0.155x | 98.958M | **110.967M** | **1.121x** | 0.002M | 0.076M | 99.02% | 56.3 | 61.4 | 0.967x |
| 10% | 394.12 | 1.609x | 20.516M | **22.309M** | **1.087x** | 0.191M | **0.251M** | 84.09% | 18.4 | 10.1 | **0.506x** |

Cached Direct is 12.1% faster at 1% and 8.7% faster at 10%. At 10%, the larger
cache drives the single NVMe at 100% busy while using about half mmap's total
CPU time.

## Host and isolation

- /dev/nvme1n1: one 1.8-TB local NVMe, XFS at /mnt/localssd with
  noatime,nodiratime;
- 64 logical AMD EPYC 7R32 CPUs; 249 GiB physical / about 245 GiB available;
- fresh no-swap MemoryMax=220G scope per 1% process and 240G for 10% Direct;
- release dg-engine-bench with jemalloc from PR head 7580acc6;
- file-scoped POSIX_FADV_DONTNEED before each backend; telemetry limited to
  nvme1n1.

The read-only scripts/bench_fio_iops.sh control was attempted but fio is not
installed. No fio ceiling is claimed; device busy is sampler telemetry.

## Workload, correctness, and steady state

Both backends use batch 512, seed 42, prefaulted offsets, prefetch off, and:

    users -> {ip_addresses, mac_addresses, emails, btids}
          -> users
          -> {ip_addresses, mac_addresses, emails, btids}

The plan digest is 43d5badeb6e6f065. Every retained pair has matching main
output digests and passed 20,000-root output-identical verification. Direct
uses 16 rings at QD32/ring, batch multiplier 8, no coalescing delay, and eager
reads off; mmap is the legacy control.

| scale | mmap threads | Direct threads | Direct cache | warmup / measured roots | main digest | verify digest | measurement cache state |
|---:|---:|---:|---:|---|---|---|---|
| 1% | 48 | 64 | 64 GiB | 2M / 10M | 1cbe8c0707b558f2 | 16c857aaf74d3315 | mmap 33.8->34.8 GiB; Direct 31.0->31.9/64 GiB |
| 10% | 64 | 64 | 200 GiB requested / 192 GiB geometry | 10M / 5M | f214de90b8699bc1 | 0cf57f9dbf303516 | mmap 212.7->212.5 GiB; Direct 187.0->188.8/192 GiB |

All retained windows exceed 30 seconds and change applicable cache by less than
5%. The 10% Direct and mmap warmups took 1,073.6s and 822.1s, avoiding a
cache-fill comparison.

## Retained detail

| scale | backend | wall | nodes/s | edges/s | IOPS | QD | busy | cores | page cache | Direct cache |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1% | mmap | 153.707s | 98.958M | 104.402M | 1,752 | 0.2 | 6% | 56.3 | 33.8->34.8 GiB | — |
| 1% | Direct | 136.380s | **110.967M** | **117.667M** | 76,214 | 28.9 | 44% | 61.4 | 1.6->1.6 GiB | 31.0->31.9/64 GiB |
| 10% | mmap | 368.906s | **20.516M** | **21.755M** | 191,431 | 42.1 | 89% | 18.4 | 212.7->212.5 GiB | — |
| 10% | Direct | 339.263s | **22.309M** | **23.655M** | **250,630** | **444.9** | **100%** | 10.1 | 15.6->15.6 GiB | 187.0->188.8/192 GiB |

The retained 10% Direct run requested 200 GiB; cache-set geometry provides
192 GiB of payload. It read 324.57 GiB versus mmap's 269.39 GiB, with
449,936,476 hits and 85,084,912 misses (84.09% hit rate). The 1.8-GiB growth
is below 1%, confirming the larger cache is steady and safely sized.

## Intermediate 5% diagnostic (not retained)

A 193.47-GiB 5% index was ingested but its 5M-root diagnostic was still filling
cache: mmap 77.913M nodes/s with page cache 154.7->172.5 GiB (+11.5%), and
Direct 62.237M with cache 125.7->133.9/160 GiB (+6.5%). Both had output digest
f63960e0d345d582 and passed verification, but neither is a headline result.

## Dataset availability and ingest

The default generator prefix already contained 1% (155,897,628,427 bytes), 5%
(779,500,379,962), 10% (1,559,005,042,144), 15%, 50%, and 100%; 2%, 3%, and
25% were absent. The 15% source was validated as available and its ingest was
started, but it cannot complete on this single 1.8-TB volume with the current
resumable ingest pipeline. Before its final edges-map phase, it had already
consumed 1.6 TB: _tmp/nodes 410 GiB, _tmp/edges_src 544 GiB, and
_tmp/edge_keys 589 GiB. The implementation deliberately retains edges_src,
fkeys, fkeys_dst, and edge_keys until after edges_map succeeds, then cleans
them. This is peak staging space, not S3 download duplication or a missing
dataset; the stopped state remains resumable.

The final 10% ingest used:

    target/release/ingest \
      --source s3://kumo-unit-test/diskgraph-datagenerator/alt_detection_10pct/diskgraph_config.json \
      --output /mnt/localssd/diskgraph-index/alt_detection_10pct-pr424-20260818 \
      --dest local --parallel

It contains 7,858,196,049 nodes and 30,590,105,618 directed edges, with zero
parse, lookup, or membership failures. The reproducible 1% and 5% indexes were
removed before the 10% ingest to stay within scratch capacity.

## Reproduction

    DG_BENCH_THREADS=64 DG_BENCH_BACKENDS='mmap direct' \
    DG_BENCH_DIRECT_CACHE_GB=200 DG_BENCH_NUM_SEEDS=5000000 \
    DG_BENCH_WARMUP_SEEDS=10000000 DG_BENCH_MEMORY_MAX=240G \
    DG_BENCH_DEVICES=nvme1n1 DG_BENCH_DIRECT_IO_QD=32 \
    DG_BENCH_DIRECT_IO_RINGS=16 DG_BENCH_EVICT_CACHE=1 \
    scripts/bench_sampling_iops.sh \
      /mnt/localssd/diskgraph-index/alt_detection_10pct-pr424-20260818
