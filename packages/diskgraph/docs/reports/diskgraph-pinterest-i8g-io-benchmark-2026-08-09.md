# Pinterest OmniSage on i8g: mmap versus cached O_DIRECT — 2026-08-09

## Result

The current production-engine sampler was measured on the requested pruned and
unpruned Pinterest OmniSage graphs on one `i8g.24xlarge`. Cached O_DIRECT is
3.023x faster than mmap on the pruned graph and 2.435x faster on the unpruned
graph. Both Direct points reach about 92.5% of this host's measured filesystem
fio ceiling while using fewer average cores and much less total CPU per sampled
node.

| graph | index GiB | index / 742.162 GiB RAM | backend | nodes/s | physical IOPS | QD | cache | average cores | total CPU / mmap |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| pruned | 1177.44 | 1.587x | mmap | 19.329M | 0.913M | 91.3 | 690.9→691.2 GiB kernel | 69.7 | 1.000x |
| pruned | 1177.44 | 1.587x | cached O_DIRECT | **58.426M** | **3.119M** | **883.5** | 600.8→614.5/640 GiB; 60.30% hit | **37.0** | **0.176x** |
| unpruned | 3200.39 | 4.312x | mmap | 11.799M | 1.290M | 169.3 | 680.6→680.5 GiB kernel | 63.9 | 1.000x |
| unpruned | 3200.39 | 4.312x | cached O_DIRECT | **28.731M** | **3.115M** | **917.0** | 625.6→637.2/640 GiB; 22.65% hit | **23.2** | **0.149x** |

The retained mmap and Direct arms for each graph use identical warmup roots,
measurement roots, plan, batch, RNG seed, and output digest. Every measured
window is longer than 30 seconds and every cache grows by less than 5% during
that window. No retained point is a cold-ramp or short-window result.

## Host and storage ceiling

- AWS `i8g.24xlarge`: 96 physical Graviton4/Neoverse-V2 cores and 742.162 GiB
  RAM, with no swap;
- six 3.75-TB third-generation Nitro SSDs in RAID0 as `/dev/md0`;
- XFS mounted at `/mnt/localssd` with `noatime,nodiratime`;
- benchmark binary built from commit `73c16045` with `--release` and the
  `jemalloc` feature;
- clean index pages evicted with file-scoped `POSIX_FADV_DONTNEED` before each
  independent process; and
- sampler disk telemetry covers `nvme1n1` through `nvme6n1`; the EBS root disk
  `nvme0n1` is excluded.

The checked-in read-only fio harness selected the 128/120 largest real
`edges.bin` files and measured 4 KiB random O_DIRECT reads through the same
md/XFS filesystem:

| fio geometry | IOPS | bandwidth | average member utilization | interpretation |
|---|---:|---:|---:|---|
| 32 workers × QD32 = QD1024 | 3.091M | 11.8 GiB/s | 97.0% | lower-depth control |
| 12 workers × QD128 = QD1536 | **3.369M** | **12.8 GiB/s** | **99.69%** | retained filesystem knee |

The Direct sampler reaches 3.119M IOPS on pruned and 3.115M on unpruned, or
92.6% and 92.5% of the 3.369M fio knee. All six drives report effectively 100%
busy during both windows. The mmap controls reach only 27.1% and 38.3% of fio,
despite using more CPU.

## Dataset and workload

The deterministic generator configs are stored in
`s3://kumo-omnisage-synth-926922431314/configs/`:

- `pinterest_omnisage_pruned.json`, seed 20260804; and
- `pinterest_omnisage_unpruned.json`, seed 20260804.

The pre-generated timestamp-sorted index snapshots were used because they were
available and compatible with the current reader:

```text
s3://kumo-omnisage-synth-926922431314/snapshots/idx_pruned_ts/
s3://kumo-omnisage-synth-926922431314/snapshots/idx_unpruned_ts/
```

After each CRT S3 sync, a dry-run sync reported no missing/different objects.
The pruned snapshot contained 8,556 objects and exactly 1,264,271,546,274
bytes. The unpruned snapshot contained 14,604 objects and exactly
3,436,395,588,419 bytes. Both loaded without conversion and passed mmap/Direct
output verification. The pruned index was deleted before downloading the
unpruned index.

Sampling starts from uniformly distributed `pin` roots and follows the same
two-hop, four-family direction reversal as the historical `pin-uniform` task:

```text
hop 0: pin -> {board, pin, query, item}
hop 1: {board, pin, query, item} -> pin
```

The exact current-engine plan is:

```text
edge_board_pin_dst_to_src:10,
edge_pin_pin_src_to_dst:10,
edge_query_pin_dst_to_src:10,
edge_pin_item_src_to_dst:10;
edge_board_pin_src_to_dst:10,
edge_pin_pin_dst_to_src:10,
edge_query_pin_src_to_dst:10,
edge_pin_item_dst_to_src:10
```

Its canonical digest is `19ba1ebdfd392b6d`. The plan yields 82.9 output
nodes/root on pruned and 170.5 nodes/root on unpruned. It samples topology only;
no node features or marshalling are included.

Every retained process uses batch 512, RNG seed 42, prefaulted offsets,
`--prefetch off`, a no-swap `MemoryMax=700G` user scope, and the exact legacy
mmap implementation as the control. Direct uses 32 rings × QD32, owner batch
multiplier eight, no coalescing delay, at most 16 adjacent pages per SQE, eager
record reads disabled, and a 640-GiB five-way fixed cache. That payload is
86.2% of physical RAM and leaves about 102 GiB for the roughly 30-GiB resident
offset set, cache metadata, frontiers, allocator state, and the OS.

## Retained pruned pair

Both backends discard 50M warmup roots and measure the same subsequent 25M
roots. mmap peaks at 256 request/engine contexts; Direct uses 96. The paired
main digest is `861c8ea81fa42b11`, and the adjacent 20k-root verification digest
is `bc5aee0413cfe1fd` for mmap and Direct.

| metric | mmap | cached O_DIRECT | Direct / mmap |
|---|---:|---:|---:|
| measured wall | 107.258s | **35.484s** | **0.331x** |
| nodes/s | 19.329M | **58.426M** | **3.023x** |
| edges/s | 20.522M | **62.031M** | **3.023x** |
| roots/s | 233,083 | **704,534** | **3.023x** |
| average cores | 69.7 | **37.0** | 0.531x |
| CPU-s / million nodes | 3.606 | **0.633** | **0.176x** |
| physical IOPS | 0.913M | **3.119M** | **3.416x** |
| aggregate QD | 91.3 | **883.5** | 9.677x |
| physical reads/node | **0.047** | 0.053 | 1.128x |
| cache during window | 690.9→691.2 GiB | 600.8→614.5/640 GiB | stable |
| Direct hit rate | n/a | 60.30% | — |
| major faults | 97,931,502 | **0** | — |

The Direct window uses 82.4% less total CPU per node while saturating storage.
Its userspace cache grows only 2.3% during measurement; mmap's kernel cache is
flat. Direct performs 12.8% more physical reads/node but finishes 67% sooner by
maintaining almost ten times the queue depth.

## Retained unpruned pair

Both backends discard 15M warmup roots and measure the same subsequent 6M
roots. mmap peaks at 384 request/engine contexts; Direct uses 96. The paired
main digest is `9ef2c4d262fc794f`, and the adjacent 20k-root verification digest
is `f030647e077e4232` for both backends.

| metric | mmap | cached O_DIRECT | Direct / mmap |
|---|---:|---:|---:|
| measured wall | 86.715s | **35.612s** | **0.411x** |
| nodes/s | 11.799M | **28.731M** | **2.435x** |
| edges/s | 12.128M | **29.532M** | **2.435x** |
| roots/s | 69,192 | **168,484** | **2.435x** |
| average cores | 63.9 | **23.2** | 0.363x |
| CPU-s / million nodes | 5.416 | **0.807** | **0.149x** |
| physical IOPS | 1.290M | **3.115M** | **2.415x** |
| aggregate QD | 169.3 | **917.0** | 5.416x |
| physical reads/node | 0.109 | **0.108** | 0.991x |
| cache during window | 680.6→680.5 GiB | 625.6→637.2/640 GiB | stable |
| Direct hit rate | n/a | 22.65% | — |
| major faults | 111,835,076 | **0** | — |

The backends read essentially the same number of physical pages per output
node, making this the cleanest async-I/O comparison. Direct is 2.435x faster,
drives 2.415x the physical command rate, and uses 85.1% less total CPU per
node. Its cache grows only 1.9% during the window and mmap's cache is flat.

## Concurrency selection

Short cold/mixed diagnostics were used only to select concurrency. They are not
headline results because their caches were still growing and their windows
were under 30 seconds.

| graph/backend | contexts tested | selected knee | evidence |
|---|---|---:|---|
| pruned mmap | 96, 128, 192, 256, 384, 512 | 256 | 11.655M at 256; 11.569M at 384; 10.933M at 512 while CPU rose |
| pruned Direct | 48, 64, 96, 128, 160, 192 | 96 | all storage-bound near 25.8–25.9M and 3.1M IOPS; 96 marginally highest |
| unpruned mmap | 192, 256, 384, 512, 640 | 384 | 12.376M at 384; 11.757M at 512; 11.111M at 640 while CPU rose |
| unpruned Direct | 48, 96, 128 | 96 | 23.996M, 24.130M, and 24.065M; all near the storage ceiling |

The mmap curves reproduce the expected blocking-fault oversubscription knee:
adding contexts eventually raises CPU and queue depth while reducing useful
throughput. Direct saturates the six drives with 48–96 inexpensive contexts
and remains far below CPU saturation.

## Reproduction

Download one snapshot at a time, benchmark it, and delete it before the next:

```bash
aws s3 sync \
  s3://kumo-omnisage-synth-926922431314/snapshots/idx_pruned_ts/ \
  /mnt/localssd/idx_pruned_ts/ --no-progress --only-show-errors
```

The checked-in `scripts/bench_sampling_iops.sh` is the benchmark entry point.
Set `DG_BENCH_SEED_TYPE=pin`, `DG_BENCH_MEMORY_MAX=700G`,
`DG_BENCH_DEVICES=nvme1n1,nvme2n1,nvme3n1,nvme4n1,nvme5n1,nvme6n1`, and
`DG_BENCH_PLAN` to the exact two-layer string above. Retained settings are:

| graph | backend | threads | cache | warmup roots | measured roots |
|---|---|---:|---:|---:|---:|
| pruned | mmap | 256 | kernel | 50M | 25M |
| pruned | Direct | 96 | 640 GiB | 50M | 25M |
| unpruned | mmap | 384 | kernel | 15M | 6M |
| unpruned | Direct | 96 | 640 GiB | 15M | 6M |

The fio controls are reproduced with:

```bash
FIO_WORKERS=32 FIO_IODEPTH=32 FIO_RUNTIME=30 FIO_RAMP_TIME=5 \
  scripts/bench_fio_iops.sh INDEX
FIO_WORKERS=12 FIO_IODEPTH=128 FIO_FILE_COUNT=120 \
  FIO_RUNTIME=30 FIO_RAMP_TIME=5 scripts/bench_fio_iops.sh INDEX
```
