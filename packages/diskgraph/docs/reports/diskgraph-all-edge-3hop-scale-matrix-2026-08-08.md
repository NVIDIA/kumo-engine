# Three-hop all-edge scale matrix: mmap versus cached O_DIRECT — 2026-08-08

## Result

Fresh compressed datasets were generated, ingested, and benchmarked at 1%,
5%, 25%, and 50% identity-graph scale. The 100% row uses the best latest
equivalent run from the preceding full-scale investigation. Cached O_DIRECT is
faster at every scale; its advantage grows from 1.423x when the complete edge
corpus is resident to 4.619x on the full graph.

| scale | index GiB | index / 502.97 GiB RAM | mmap nodes/s | Direct nodes/s | speedup | mmap IOPS | Direct IOPS | Direct hit | mmap cores | Direct cores | Direct total CPU / mmap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 38.003 | 0.076x | 243.117M | **345.843M** | **1.423x** | 0.006M | 0.011M | 99.96% | 154.4 | 157.2 | **0.716x** |
| 5% | 193.547 | 0.385x | 210.022M | **319.607M** | **1.522x** | 0.023M | 0.481M | 97.87% | 149.5 | 156.6 | **0.688x** |
| 25% | 985.289 | 1.959x | 80.222M | **209.260M** | **2.609x** | 1.105M | 4.608M | 68.84% | 83.9 | 116.6 | **0.533x** |
| 50% | 2041.801 | 4.059x | 40.083M | **140.853M** | **3.514x** | 1.528M | 5.929M | 40.47% | 111.3 | 87.4 | **0.223x** |
| 100% | 4083.761 | 8.119x | 22.611M | **104.437M** | **4.619x** | 1.306M | 6.123M | 17.09% | 44.5 | 66.5 | **0.324x** |

Index size relative to RAM is the governing context for the transition. The
1% and 5% indexes fit within RAM and are compute-bound: raising concurrency beyond the retained
knee increased CPU and reduced nodes/s. At 25%, Direct averaged 96.0% busy per
drive. At 50% and 100%, Direct averaged 99.3% busy per drive while consuming
only 46% and 35% of the host's 192 logical CPUs. The intended large-scale
operating condition—storage saturated while CPU remains available—is met.

The retained 50% Direct run reaches 90.2% of the 6.577M-IOPS md/XFS fio
reference. The best 100% run reaches 93.1%. Neither run adds synthetic reads:
IOPS and nodes/s rise together.

## Workload and isolation

Every row uses the same canonical three-hop plan, starting from uniformly
distributed `users` roots with fanout 10 on every applicable edge family:

```text
users -> {ip_addresses, mac_addresses, emails, btids}
      -> users
      -> {ip_addresses, mac_addresses, emails, btids}
```

The canonical plan digest is `43d5badeb6e6f065`. Every retained pair:

- uses batch 512, RNG seed 42, `--prefetch off`, and prefaulted offsets;
- starts each backend in a fresh process and no-swap `MemoryMax=500G` scope;
- evicts all clean pages belonging to that index before each process;
- uses the exact legacy mmap implementation as the control;
- uses 32 Direct rings at QD32/ring, batch multiplier 8, and no coalescing
  delay; and
- passes the deterministic 20,000-root verification gate.

Concurrency and warmup were tuned independently per scale. Backend thread
counts may differ because the retained point is each backend's own nodes/s
maximum; the plan, roots, batch, and output remain identical.

| scale | mmap threads | Direct threads | Direct cache | warmup roots | measured roots | output digest |
|---:|---:|---:|---:|---:|---:|---|
| 1% | 128 | 160 | 256 GiB | 2M | 7M | `9d390ea773f0bbf5` |
| 5% | 128 | 160 | 256 GiB | 10M | 7M | `15171695b52ab327` |
| 25% | 160 | 128 | 320 GiB | 10M | 5M | `d710bdb093464712` |
| 50% | 256 | 192 | 320 GiB | 3M | 5M | `1dd02a060126e5fe` |
| 100% | 160 | 160 | 256 GiB | 3M | 3M | `afb40d6dce1cac0c` |

## Retained measurement detail

| scale | backend | wall | nodes/s | edges/s | roots/s | IOPS | QD | avg drive busy | avg cores | cache state |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1% | mmap | 43.577s | 243.117M | 257.795M | 160,635 | 0.006M | 0.5 | 7.5% | 154.4 | file 33.8→34.8 GiB |
| 1% | Direct | 30.633s | **345.843M** | **366.724M** | **228,509** | 0.011M | 1.0 | 12.5% | 157.2 | 32.2→33.2/256 GiB |
| 5% | mmap | 50.437s | 210.022M | 222.701M | 138,786 | 0.023M | 1.5 | 18.8% | 149.5 | file 172.5→176.9 GiB |
| 5% | Direct | 33.144s | **319.607M** | **338.901M** | **211,202** | 0.481M | 65.1 | 59.3% | 156.6 | 152.6→155.9/256 GiB |
| 25% | mmap | 94.329s | 80.222M | 85.064M | 53,006 | 1.105M | 67.7 | 82.8% | 83.9 | file 455.0→454.8 GiB |
| 25% | Direct | 36.162s | **209.260M** | **221.892M** | **138,267** | 4.608M | 589.6 | 96.0% | 116.6 | 315.0→317.7/320 GiB |
| 50% | mmap | 188.805s | 40.083M | 42.502M | 26,482 | 1.528M | 103.8 | 84.3% | 111.3 | file 448.2→448.0 GiB |
| 50% | Direct | 53.728s | **140.853M** | **149.356M** | **93,061** | **5.929M** | **756.2** | **99.3%** | 87.4 | 316.6→319.8/320 GiB |
| 100% | mmap | 200.818s | 22.611M | 23.975M | 14,939 | 1.306M | 78.6 | 89.0% | 44.5 | file 429.0 GiB end |
| 100% | Direct | 43.477s | **104.437M** | **110.741M** | **69,002** | **6.123M** | **808.8** | **99.3%** | 66.5 | 256 GiB cap |

The 100% row is the best latest repeat. The adjacent fully detailed full-scale
report retains a 104.369M-nodes/s, 6.119M-IOPS run from the same geometry; the
0.06% difference is ordinary run variation.

## Saturation selection

The retained points are not arbitrary fixed-thread comparisons:

- 1%: mmap peaked at 128 threads; Direct peaked at 160 and regressed at 192.
- 5%: mmap again peaked at 128; Direct peaked at 160 and regressed at 192.
- 25%: mmap peaked at 160. Direct peaked at 128 with a 320 GiB cache; 160
  threads regressed. The drives averaged 96% busy.
- 50%: Direct improved through 192 threads, then flattened at 224 and regressed
  at 256. mmap needed 256 contexts to maximize the blocking-fault path; 320 and
  384 raised CPU while slightly lowering nodes/s.
- 100%: Direct reaches the storage ceiling with 160 inexpensive request
  contexts; the detailed full-scale report documents that selection.

At 25%, batch 256 improved Direct but regressed mmap and changed the
batch-scoped RNG digest. Batch 512 is retained for both backends so every pair
is output-identical. The report selects nodes/s, not IOPS in isolation.

## Storage ceiling and remaining gap

The filesystem reference is 6.577M 4 KiB random-read IOPS at aggregate QD1024
using 32 fio workers. QD2048 reaches about 6.62M but roughly doubles latency.
The four-drive advertised aggregate is approximately 8.8M IOPS.

| scale | Direct IOPS | fraction of fio | gap to fio | fraction of advertised | interpretation |
|---:|---:|---:|---:|---:|---|
| 1% | 0.011M | 0.2% | 6.566M | 0.1% | corpus resident; CPU-bound |
| 5% | 0.481M | 7.3% | 6.096M | 5.5% | 97.87% hit; CPU-bound |
| 25% | 4.608M | 70.1% | 1.969M | 52.4% | mixed, drives 96% busy |
| 50% | 5.929M | 90.2% | 0.648M | 67.4% | storage-saturated |
| 100% | 6.123M | 93.1% | 0.454M | 69.6% | storage-saturated |

The industry number is not the right software denominator: fio through the
same md/XFS stack reaches 74.7% of advertised. At 100%, the complete sampler
is within 6.9% of fio while also performing cache lookup, dependent three-hop
planning, decoding, sampling, and output materialization.

## Why nodes/s greatly exceeds IOPS

Nodes/s counts every sampled output node; IOPS counts only physical 4 KiB
reads. A page can serve several selected entries and one root emits about 1,514
nodes. Cache hits need no physical read. The measured Direct physical reads per
sampled node rise with scale:

| scale | Direct hit rate | physical reads/node |
|---:|---:|---:|
| 1% | 99.96% | 0.00003 |
| 5% | 97.87% | 0.00150 |
| 25% | 68.84% | 0.02202 |
| 50% | 40.47% | 0.04209 |
| 100% | 17.09% | 0.0586 |

Thus 140.9M nodes/s at 50% and 5.93M IOPS are consistent: there is one physical
read per roughly 23.8 output nodes, plus 40.5% logical page hits.

## Dataset generation and ingest

All four regenerated configs explicitly use Zstd Parquet compression. Each
scale was generated, inferred, ingested, benchmarked, and then deleted before
the next scale; the final 50% artifacts remain on `/mnt/localssd`. The 100%
artifacts were deleted after retaining their best result to make room for this
matrix.

| scale | generator rows | Parquet files | source bytes | indexed nodes | directed edges | index bytes | generation | ingest wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 2,315,324,886 | 185 | 77,831,570,164 | 785,819,605 | 3,059,010,562 | 40,805,148,418 | 10.33s | 50.06s |
| 5% | 11,576,624,428 | 925 | 389,159,716,943 | 3,929,098,024 | 15,295,052,808 | 207,819,577,389 | 50.84s | 5m08s |
| 25% | 57,883,122,146 | 4,624 | 1,951,409,728,146 | 19,645,490,124 | 76,475,264,044 | 1,057,946,436,268 | 5m51s | 34m23s |
| 50% | 115,766,244,291 | 9,248 | 3,904,713,328,595 | 39,290,980,248 | 152,950,528,086 | 2,192,367,242,411 | 23m45s | 1h33m05s |
| 100% | 231,532,488,578 | 18,496 | 7,811,354,595,328 | 78,581,960,494 | 305,901,056,168 | 4,384,905,142,272 | 28m40s | 2h31m03s including retry |

All ingests report zero key parse, lookup, and membership failures. Duplicate
observations occur only in the deliberately low-cardinality, disconnected
`alt_labels` table and do not affect the user-start sampling plan.

## Reproduction

After generate and infer, the ingest command is:

```bash
target/release/ingest \
  --source /mnt/localssd/datasets/alt_detection_SCALE/diskgraph_config.json \
  --output /mnt/localssd/diskgraph-index/alt_detection_SCALE-zstd-f7d99f8d-matrix \
  --dest local --parallel
```

The benchmark wrapper is `scripts/bench_sampling_iops.sh`. Set the scale's
thread count, backend, cache size, warmup, and measured roots from the retained
configuration table above. The wrapper prints and enforces the canonical plan,
evicts only index-owned file pages, starts a fresh no-swap scope, records disk
telemetry, and fails on sampling or determinism errors.
