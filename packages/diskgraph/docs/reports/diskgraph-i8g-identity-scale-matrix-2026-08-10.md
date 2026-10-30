# Identity scale matrix on i8g: mmap versus cached O_DIRECT — 2026-08-10

## Result

Fresh 1%, 5%, 25%, and 50% identity graphs were generated, inferred, ingested,
and benchmarked on one `i8g.24xlarge`. These results are additive to, and do
not replace, the preceding Threadripper matrix. Cached O_DIRECT is faster at
every current-i8g scale. It reaches 404.096M nodes/s on the resident 1% graph,
290.582M at 25%, and 105.324M while saturating the six-SSD array at 50%.

| scale | index GiB | index / 742.162 GiB RAM | mmap nodes/s | Direct nodes/s | speedup | mmap IOPS | Direct IOPS | Direct hit | mmap cores | Direct cores | Direct total CPU / mmap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 37.987 | 0.051x | 180.155M | **404.096M** | **2.243x** | ~0 | ~0 | 99.99997% | 86.0 | 95.6 | **0.495x** |
| 5% | 193.470 | 0.261x | 168.907M | **385.669M** | **2.283x** | 0.001M | 0.014M | 99.95% | 76.7 | 94.4 | **0.538x** |
| 25% | 985.224 | 1.328x | 110.809M | **290.582M** | **2.622x** | 0.287M | 2.058M | 89.98% | 79.3 | 85.3 | **0.410x** |
| 50% | 2041.671 | 2.751x | 56.032M | **105.324M** | **1.880x** | 1.338M | **3.110M** | 58.24% | 59.1 | **38.4** | **0.346x** |

The 1% and 5% working sets become fully resident, so their retained windows
correctly show almost no physical I/O and are CPU-bound. At 25%, Direct is a
mixed CPU/storage point: each drive averages 94.5% busy. At 50%, Direct reaches
92.3% of the 3.369M-IOPS md/XFS fio knee and each drive averages 99.7% busy.
The current host therefore meets the expected CPU and disk behavior at every
scale rather than reporting synthetic IOPS or a hardware-limited anomaly.

## Hardware and isolation

- AWS `i8g.24xlarge`: 96 physical Graviton4/Neoverse-V2 cores and 742.162 GiB
  RAM, with no swap;
- six 3.75-TB third-generation Nitro SSDs in RAID0 as `/dev/md0`;
- XFS at `/mnt/localssd` with `noatime,nodiratime`;
- release sampler binary with `jemalloc` from commit `73c16045`;
- fresh no-swap `MemoryMax=700G` scope for every sampler process;
- clean index pages evicted with file-scoped `POSIX_FADV_DONTNEED` before each
  independent process; and
- disk telemetry restricted to `nvme1n1` through `nvme6n1`, excluding the EBS
  root disk.

The read-only fio control on this exact md/XFS stack reaches 3.091M 4 KiB
random-read IOPS at 32xQD32 and 3.369M at the retained 12xQD128 filesystem
knee. The latter holds every member at 99.69% average utilization and is the
software denominator used here.

## Workload and correctness

Every row uses the same canonical three-hop plan, starting from uniformly
distributed `users` roots with fanout 10 on every applicable edge family:

```text
users -> {ip_addresses, mac_addresses, emails, btids}
      -> users
      -> {ip_addresses, mac_addresses, emails, btids}
```

The canonical plan digest is `43d5badeb6e6f065`. Every retained pair uses
batch 512, RNG seed 42, `--prefetch off`, prefaulted offsets, 32 Direct rings
at QD32/ring, batch multiplier 8, no coalescing delay, eager-node reads off,
and the exact legacy mmap implementation as the control. Warmup and measured
root streams are identical between backends at a scale; measurement roots are
disjoint from warmup roots.

| scale | mmap threads | Direct threads | Direct cache | warmup roots | measured roots | main digest | verify digest |
|---:|---:|---:|---:|---:|---:|---|---|
| 1% | 96 | 96 | 640 GiB cap | 10M | 10M | `d50e1f33ab782ca3` | `482c0f4878cb9f1b` |
| 5% | 64 | 96 | 640 GiB cap | 20M | 10M | `b8d560293de88b72` | `639567ef057338dd` |
| 25% | 160 | 128 | 640 GiB cap | 30M | 7M | `0dda331d00af8079` | `c04259dec2c9f2d0` |
| 50% | 256 | 160 | 512 GiB cap | 30M | 5M | `55f6efd6e26a7625` | `cbb511a4ba0dd030` |

All eight retained measurement windows exceed 30 seconds. Kernel or userspace
cache growth is below 5% in every window, all output and verification digests
match within each pair, and every process reports zero swap.

## Retained measurement detail

| scale | backend | wall | nodes/s | edges/s | roots/s | IOPS | QD | aggregate busy | average cores | cache state |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1% | mmap | 84.004s | 180.155M | 191.032M | 119,042 | ~0 | 0.0 | 0% | 86.0 | file 34.8→34.8 GiB |
| 1% | Direct | 37.451s | **404.096M** | **428.494M** | **267,017** | ~0 | 0.0 | 0% | 95.6 | 33.3→33.3/640 GiB |
| 5% | mmap | 89.610s | 168.907M | 179.104M | 111,594 | 0.001M | 0.1 | 8% | 76.7 | file 177.3→177.6 GiB |
| 5% | Direct | 39.246s | **385.669M** | **408.951M** | **254,805** | 0.014M | 1.0 | 73% | 94.4 | 169.1→169.4/640 GiB |
| 25% | mmap | 95.619s | 110.809M | 117.498M | 73,207 | 0.287M | 23.6 | 231% | 79.3 | file 682.2→681.2 GiB |
| 25% | Direct | 36.463s | **290.582M** | **308.124M** | **191,976** | **2.058M** | **372.6** | **567%** | 85.3 | 577.7→585.1/640 GiB |
| 50% | mmap | 135.064s | 56.032M | 59.415M | 37,019 | 1.338M | 138.3 | 474% | 59.1 | file 670.7→666.7 GiB |
| 50% | Direct | 71.854s | **105.324M** | **111.682M** | **69,585** | **3.110M** | **940.8** | **598%** | **38.4** | 510.9→511.2/512 GiB |

Aggregate busy is the sum across six drives; divide by six for average member
utilization. Total CPU seconds for Direct are 49.5%, 53.8%, 41.0%, and 34.6%
of mmap at 1%, 5%, 25%, and 50%, respectively. Direct can use more cores while
finishing much sooner at resident scales, yet still consume less total CPU.

## Cache sizing and steady state

The Direct CLI cap is intentionally large on this 742-GiB host. At 1%, 5%, and
25%, `--direct-cache-gb 640` configures five ways over 33,554,432 sets: exactly
640 GiB of payload capacity. The actual stable occupancies are 33.3, 169.4,
and 585.1 GiB because working-set reachability and set conflicts govern used
slots. These runs use the available RAM without manufacturing cache traffic.

The 50% index has 77.8 GiB of prefaulted offsets. A 640-GiB payload plus those
offsets is already 717.8 GiB before cache metadata, pooled state, and runtime
memory, which cannot fit safely inside `MemoryMax=700G` or 742 GiB physical
RAM. Two 30M-root 640-GiB attempts—including a dedicated one-hour attempt—did
not finish warmup after entering high-occupancy memory pressure and emitted no
measurement. They are rejected, not headline results.

The largest viable cache geometry is four ways, or 512 GiB. With 77.8 GiB of
offsets it devotes at least 589.8 GiB to index-resident state while leaving
real runtime/system headroom. A separate 15M-root diagnostic reached
506.9→509.2 GiB over its measured window; the retained root-matched 30M warmup
starts measurement at 510.9 GiB and grows only 0.3 GiB (0.06%). Thus 512 GiB
both makes substantial use of RAM and reaches a measurable steady state.

## Concurrency selection

Short cache-ramp diagnostics selected the knees; none are used as headline
rates:

- 1%: both backends peak at 96 threads; Direct and mmap regress at 128.
- 5%: mmap peaks at 64 and regresses at 96/128; Direct peaks at 96 and
  regresses at 128.
- 25%: mmap improves through 160 and regresses 5.8% at 192. Direct peaks at
  128 and regresses 1.7% at 160.
- 50%: mmap improves through 256 and regresses 1.7% at 320. Direct reaches the
  storage ceiling at 160; 192 is flat/slightly lower while adding no useful
  drive headroom.

The tuning curves and retained rates diverge at large scale because a short
fill has a changing hit rate. Only long-run, stable-cache values are retained.

## Comparison with the earlier Threadripper matrix

The preceding report used 96 Threadripper cores with SMT (192 logical CPUs),
502.97 GiB RAM, four different SSDs, a 6.577M-IOPS fio knee, and 256/320-GiB
Direct caches. The current report uses 96 physical Graviton4 cores, 742.162
GiB RAM, six Nitro SSDs, a 3.369M-IOPS fio knee, and 640/512-GiB caches. The
rows are therefore hardware comparisons, not replacements or controlled ISA
microbenchmarks.

| scale | Threadripper mmap | i8g mmap | i8g delta | Threadripper Direct | i8g Direct | i8g delta | Threadripper speedup | i8g speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 243.117M | 180.155M | -25.9% | 345.843M | **404.096M** | **+16.8%** | 1.423x | **2.243x** |
| 5% | 210.022M | 168.907M | -19.6% | 319.607M | **385.669M** | **+20.7%** | 1.522x | **2.283x** |
| 25% | 80.222M | **110.809M** | **+38.1%** | 209.260M | **290.582M** | **+38.9%** | 2.609x | **2.622x** |
| 50% | 40.083M | **56.032M** | **+39.8%** | **140.853M** | 105.324M | -25.2% | **3.514x** | 1.880x |

The resident Direct path favors i8g despite slower current-host mmap. At 25%,
the extra RAM/cache and host balance improve both backends. At 50%, i8g mmap
is faster, but Direct is capped by a filesystem knee roughly half the earlier
host's; it still reaches 92.3% of its own fio ceiling with ample CPU idle.

## Fresh generation and ingest

Each scale was generated, inferred, ingested, benchmarked, and permanently
deleted before the next scale. All configs use deterministic Zstd Parquet.
Only the final 50% artifacts remain until the identity report is committed,
after which they are wiped before e-commerce generation.

| scale | generator rows | Parquet files | source bytes | indexed nodes | directed edges | index bytes | generation | ingest wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 2,315,324,886 | 185 | 77,831,570,163 | 785,819,605 | 3,059,010,562 | 40,788,491,304 | 13.56s | 1m12.52s |
| 5% | 11,576,624,428 | 925 | 389,159,716,943 | 3,929,098,024 | 15,295,052,808 | 207,736,650,878 | 55.59s | 5m00.30s |
| 25% | 57,883,122,146 | 4,624 | 1,951,409,728,146 | 19,645,490,124 | 76,475,264,044 | 1,057,876,114,385 | 4m45.28s | 25m02.58s |
| 50% | 115,766,244,291 | 9,248 | 3,904,713,328,594 | 39,290,980,248 | 152,950,528,086 | 2,192,227,146,201 | 9m45.02s | 54m55.55s |

Ingests report zero primary-key duplicates on sampled node tables and zero null
timestamp drops. The only duplicate observations are in deliberately
low-cardinality, disconnected `alt_labels`; they do not enter the `users`
sampling plan. Every ingest reports zero swap.

## Reproduction

After generate and infer, the ingest command is:

```bash
target/release/ingest \
  --source /mnt/localssd/datasets/alt_detection_SCALE/diskgraph_config.json \
  --output /mnt/localssd/diskgraph-index/alt_detection_SCALE-zstd-i8g-20260809 \
  --dest local --parallel
```

Run `scripts/bench_sampling_iops.sh` on that index with the retained thread,
cache, warmup, and measured-root settings above. Set
`DG_BENCH_MEMORY_MAX=700G`, `DG_BENCH_BATCH_SIZE=512`, and
`DG_BENCH_DEVICES=nvme1n1,nvme2n1,nvme3n1,nvme4n1,nvme5n1,nvme6n1`.
The wrapper enforces the canonical plan, fresh-process eviction, no swap,
prefaulted offsets, disk telemetry, and deterministic verification.
