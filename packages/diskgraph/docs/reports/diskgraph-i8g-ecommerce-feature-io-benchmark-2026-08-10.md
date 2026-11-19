# Feature-heavy e-commerce scale matrix on i8g — 2026-08-10

## Result

Fresh 1%, 5%, 25%, and 50% feature-heavy e-commerce graphs were generated,
inferred, ingested, and benchmarked on one `i8g.24xlarge`. These results are
additive to, and do not replace, the earlier
[Threadripper feature-heavy matrix](diskgraph-ecommerce-feature-io-benchmark-2026-08-08.md).

The resident 1% and 5% rows are effectively tied: mmap leads Direct by 4.1%
at 1%, and by 0.2% at 5%. Once the graph is larger than RAM, cached O_DIRECT
is 2.390x faster at 25% and 2.267x faster at 50%, while consuming 65.5% and
76.6% less total sampler CPU, respectively.

| scale | index GiB | index / 742.2 GiB RAM | mmap nodes/s | Direct nodes/s | Direct / mmap | mmap IOPS | Direct IOPS | Direct hit | mmap cores | Direct cores | Direct total CPU / mmap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 122.84 | 0.166x | **7.500M** | 7.196M | 0.959x | 0.017M | 0.016M | 99.87% | 65.6 | 53.0 | **0.842x** |
| 5% | 616.19 | 0.830x | **6.936M** | 6.922M | 0.998x | 0.063M | 0.180M | 98.56% | 69.9 | 53.9 | **0.773x** |
| 25% | 3107.34 | 4.187x | 2.717M | **6.495M** | **2.390x** | 1.119M | **2.791M** | 76.70% | 92.1 | **75.9** | **0.345x** |
| 50% | 6220.20 | 8.380x | 1.926M | **4.366M** | **2.267x** | 1.419M | **3.106M** | 61.34% | 90.7 | **48.2** | **0.234x** |

The 25% and 50% Direct rows average 99.7% and 99.8% member-device busy,
respectively. The 50% row reaches 87.0% of the 3.569M-IOPS fio ceiling while
retaining CPU headroom. The limiting resource is therefore the six-SSD array,
not an undersized CPU pool.

## Hardware and storage controls

- AWS `i8g.24xlarge`: 96 physical Graviton4/Neoverse-V2 cores, 742.2 GiB RAM,
  and no swap;
- six 3.4-TiB EC2 NVMe instance-store devices in RAID0 as `/dev/md0`, with a
  1 MiB chunk;
- XFS at `/mnt/localssd`, with a 1 MiB data stripe unit, 6 MiB stripe width,
  and `noatime,nodiratime,allocsize=1m,logbufs=8,logbsize=256k`;
- fresh no-swap `MemoryMax=700G` scope for every sampler process; and
- telemetry restricted to `nvme0n1,nvme2n1,nvme3n1,nvme4n1,nvme5n1,nvme6n1`.
  The EBS root device, `nvme1n1`, is excluded.

Read-only fio was rerun after the array was recreated. The filesystem reached
3.409M 4 KiB random-read IOPS (13.0 GiB/s) at 32 jobs x QD32 and 3.569M IOPS
(13.6 GiB/s) at 12 jobs x QD128. At the latter point every member averaged
99.7–100.0% utilization and throughput was balanced across all six devices.
These controls verify the expected disk bandwidth and queueing behavior on the
exact storage stack used for the sampler rows.

## Workload and correctness

The workload matches the preceding feature-heavy report: four node tables,
1–2 KiB user/item feature records, hotspot-distributed order relations, and a
four-hop fanout-10 Uniform sample starting at `users`. `--fetch-features`
materializes features for every sampled node type. The canonical plan is
`default:4x10:Uniform` with digest `5d2e1672de275d8d`.

Every retained pair uses batch 32, RNG seed 42, `--prefetch off`, I/O coalesce
16, batch multiplier 8, zero coalescing delay, eager-node reads off, 32 Direct
rings at QD32/ring, and a 640-GiB Direct cache. Warmup and measured root streams
are identical between backends at a scale, and measurement roots are disjoint
from warmup roots.

| scale | mmap threads | Direct threads | warmup roots | measured roots | output digest | verify digest |
|---:|---:|---:|---:|---:|---|---|
| 1% | 80 | 48 | 3.0M | 700k | `22dbe312d2543e16` | `c849d900488c2eec` |
| 5% | 96 | 48 | 6.0M | 600k | `86a7cea9efcd252e` | `430d8bd45780aa77` |
| 25% | 640 | 160 | 2.0M | 500k | `79d9003d90db2a74` | `1174fad9b8853ec5` |
| 50% | 640 | 128 | 2.0M | 500k | `fbdb1448bb9ebbfc` | `908326cfd454a9da` |

All output and verification digests match within each mmap/Direct pair. All
eight retained windows exceed 30 seconds, and every process reports zero swap.

## Retained measurement detail

| scale | backend | warmup wall | measured wall | nodes/s | edges/s | IOPS | QD | aggregate busy | average cores | cache state |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1% | mmap | 189.4s | 39.558s | **7.500M** | **10.037M** | 0.017M | 1.2 | 101% | 65.6 | file 96.0→98.5 GiB |
| 1% | Direct | 178.7s | 41.232s | 7.196M | 9.629M | 0.016M | 1.1 | 73% | 53.0 | 103.5→106.0/640 GiB |
| 5% | mmap | 459.4s | 36.656s | **6.936M** | **9.282M** | 0.063M | 4.3 | 290% | 69.9 | file 449.0→457.9 GiB |
| 5% | Direct | 379.7s | 36.729s | 6.922M | 9.263M | 0.180M | 13.0 | 297% | 53.9 | 414.9→421.3/640 GiB |
| 25% | mmap | 299.8s | 78.053s | 2.717M | 3.636M | 1.119M | 99.8 | 588% | 92.1 | file 679.3→679.0 GiB |
| 25% | Direct | 149.4s | 32.655s | **6.495M** | **8.691M** | **2.791M** | 652.1 | 598% | **75.9** | 624.9→630.1/640 GiB |
| 50% | mmap | 391.3s | 110.003s | 1.926M | 2.578M | 1.419M | 151.0 | 594% | 90.7 | file 665.9→666.2 GiB |
| 50% | Direct | 209.3s | 48.525s | **4.366M** | **5.843M** | **3.106M** | 889.6 | 599% | **48.2** | 639.2→639.7/640 GiB |

Aggregate busy is summed across the six devices; divide by six for mean member
utilization. Physical bytes read in the measured windows were 2.52/2.56 GiB at
1%, 8.86/25.21 GiB at 5%, 333.26/348.13 GiB at 25%, and 595.51/577.21 GiB at
50% for mmap/Direct.

## Cache sizing and steady state

The 640-GiB Direct cache is intentional: it commits 86% of physical RAM to
payload capacity and can exploit substantially more memory than the earlier
256/320-GiB Threadripper runs. It is also viable under `MemoryMax=700G`. At
50%, the retained Direct process was observed near 674 GiB RSS with about
26 GiB of cgroup headroom while cache occupancy stabilized at 639.7 GiB.

The long warmups fill the naturally reachable working sets before measurement.
Retained-window cache growth is 2.4%, 1.5%, 0.83%, and 0.08% for Direct at
1%, 5%, 25%, and 50%; mmap page-cache growth is 2.6%, 2.0%, -0.04%, and 0.05%.
The two large-scale rows are therefore measured with effectively flat memory,
and even the resident rows remain below the 5% steady-state gate.

At 1% the reachable working set occupies only about 106 GiB despite the larger
cap. At 5% it occupies 421 GiB and reaches 98.56% hits. The 25% and 50% caches
fill to 630 and 640 GiB, giving 76.70% and 61.34% hits while leaving sufficient
runtime and system memory. A smaller artificial cap would discard available
RAM and misrepresent this host.

## Concurrency selection

Only stable, longer-than-30-second windows are retained. Short cache-ramp
diagnostics selected the concurrency knees:

- at 25%, Direct improves from 6.052M nodes/s at 96 threads to 6.293M at 128
  and 6.324M at 160; mmap rises through 640 threads (2.685M in tuning) and
  regresses at 768;
- at 50%, Direct is flat from 128 through 192 threads, with 128 the best point;
  mmap peaks at 640 and regresses at 768; and
- 1% and 5% use the lower-concurrency resident-path knees shown above.

A first 5% Direct orchestration attempt was terminated before it entered the
timed window. It is excluded; the reported result is a clean fresh-process
rerun with the matching digest. Diagnostics that the benchmark wrapper flagged
for sub-30-second windows are likewise excluded from headline results.

## Comparison with the earlier Threadripper matrix

The preceding host had 96 Threadripper cores with SMT, 502.97 GiB RAM, four
different SSDs, a 6.577M-IOPS fio knee, and 256/320-GiB Direct caches. This
host has 96 physical Graviton4 cores, 742.2 GiB RAM, a six-Nitro-SSD 3.569M-
IOPS knee, and a 640-GiB cache. These are hardware comparisons, not replacements
or controlled ISA microbenchmarks.

| scale | Threadripper mmap | i8g mmap | i8g delta | Threadripper Direct | i8g Direct | i8g delta | Threadripper speedup | i8g speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | **8.082M** | 7.500M | -7.2% | **9.297M** | 7.196M | -22.6% | **1.150x** | 0.959x |
| 5% | 5.877M | **6.936M** | **+18.0%** | **8.188M** | 6.922M | -15.5% | **1.393x** | 0.998x |
| 25% | 1.651M | **2.717M** | **+64.6%** | 5.976M | **6.495M** | **+8.7%** | **3.620x** | 2.390x |
| 50% | 1.302M | **1.926M** | **+47.9%** | **5.554M** | 4.366M | -21.4% | **4.265x** | 2.267x |

i8g mmap is substantially faster at 5–50%, and i8g Direct is fastest at 25%.
At 50%, Direct is storage-bound at 3.106M IOPS on a fio ceiling roughly half
the Threadripper host's; CPU remains available. The lower Direct/mmap speedups
on i8g thus combine a stronger mmap path with a lower raw storage ceiling, not
a failure to use cache, CPU, or disk.

## Fresh generation and ingest

Each scale was generated with deterministic seed `20260808`, inferred,
ingested, and benchmarked before advancing to the next. Temporary 1%, 5%, and
25% data were removed after their retained results. Generation and ingest used
no swap.

| scale | source rows | Parquet files | source bytes | index bytes | generation wall | generation peak RSS | ingest wall | ingest peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 280M | 86 | 20,196,064,705 | 131,894,861,302 | 2m04.88s | 27.6 GiB | 2m01.70s | 41.7 GiB |
| 5% | 1.4B | 435 | 101,773,874,622 | 661,630,397,753 | 8m32.65s | 64.5 GiB | 9m46.68s | 139.5 GiB |
| 25% | 7.0B | 2,176 | 511,143,312,816 | 3,336,476,063,287 | 41m16.04s | 64.5 GiB | 48m32.30s | 238.1 GiB |
| 50% | 14.0B | 4,352 | 1,023,482,563,561 | 6,678,887,763,257 | 1h21m45s | 63.4 GiB | 1h37m31s | 481.3 GiB |

The 50% dataset directory occupies 1,023,593,245,551 bytes including its
configuration metadata; the table reports the 1,023,482,563,561 bytes of
Parquet input accounted by infer/ingest. The final index contains 14B nodes and
65.52B directed relation entries. The ingest read/wrote 17.2 TB at an average
3.0 GB/s and reported 79m52s encode, 4m00s primary-key map, 3m06s edge map,
and 10m21s index assembly.

## Reproduction

The scale configs are
`diskgraph-datagenerator/configs/ecommerce_features_{1,5,25,50}pct.json`.
After generation and infer, ingest each config into its own local index, then
run `scripts/bench_sampling_iops.sh` with the retained settings above and:

```bash
DG_BENCH_PLAN=default \
DG_BENCH_MEMORY_MAX=700G \
DG_BENCH_BATCH_SIZE=32 \
DG_BENCH_DEVICES=nvme0n1,nvme2n1,nvme3n1,nvme4n1,nvme5n1,nvme6n1 \
./scripts/bench_sampling_iops.sh /mnt/localssd/INDEX
```

The full logs and copied metadata are retained on the EBS root volume under
`/home/zdrach/benchmark-logs/kumo-i8g-ecommerce-20260810-retry`.
