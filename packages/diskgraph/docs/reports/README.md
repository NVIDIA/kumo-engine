# Direct I/O sampler benchmark reports

For current Engine integration findings and release gates, see the
[PR #424 production-readiness review](pr-424-production-readiness-2026-10-30.md).

These reports trace the sampler from the repaired mmap/io_uring baseline,
through userspace-cache CPU optimization, to the retained three-hop all-edge
results at 1%, 5%, 25%, 50%, and 100% identity-graph scale.

The stage table below records the earlier Threadripper host. The newer i8g
Pinterest, identity, and feature-heavy reports are additive hardware reruns
and carry their own fio ceilings, cache sizing, steady-state gates, and
cross-host comparisons.

Workloads differ between the first two stages and the all-edge scaling runs.
Use the plan column and the linked report before making cross-row comparisons.

## Stage summary

| stage | workload / graph | index GiB | index / 502.97 GiB RAM | backend | nodes/s | physical IOPS | QD | cache hit | average cores | total CPU vs mmap |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| repaired baseline | two-hop IP, 25% | 985.289 | 1.959x | mmap | **9.244M** | 0.731M | 43.2 | ~26.6% estimated | — | 1.000x |
| repaired baseline | two-hop IP, 25% | 985.289 | 1.959x | buffered io_uring | 7.933M | 0.647M | 291.6 | ~24.2% estimated | — | — |
| repaired baseline | two-hop IP, 25% | 985.289 | 1.959x | cacheless O_DIRECT | 8.640M | 0.929M | 122.7 | — | — | — |
| repaired baseline | two-hop IP, 25% | 985.289 | 1.959x | O_DIRECT, 16 GiB cache | 7.215M | 0.641M | 87.8 | 17.6% | — | — |
| optimized miss path | two-hop IP, 25% | 985.289 | 1.959x | O_DIRECT, 16 GiB cache | **45.684M** | **4.137M** | 586 | 15.77% | 53.5 | — |
| optimized high-hit | two-hop IP, 25% | 985.289 | 1.959x | O_DIRECT, 320 GiB cache | **160.591M** | 0.758M | 97 | 95.61% | 103.6 | — |
| retained all-edge | three-hop all-edge, 1% | 38.003 | 0.076x | mmap | 243.117M | 0.006M | 0.5 | kernel | 154.4 | 1.000x |
| retained all-edge | three-hop all-edge, 1% | 38.003 | 0.076x | cached O_DIRECT | **345.843M** | 0.011M | 1.0 | 99.96% | 157.2 | **0.716x** |
| retained all-edge | three-hop all-edge, 5% | 193.547 | 0.385x | mmap | 210.022M | 0.023M | 1.5 | kernel | 149.5 | 1.000x |
| retained all-edge | three-hop all-edge, 5% | 193.547 | 0.385x | cached O_DIRECT | **319.607M** | 0.481M | 65.1 | 97.87% | 156.6 | **0.688x** |
| retained all-edge | three-hop all-edge, 25% | 985.289 | 1.959x | mmap | 80.222M | 1.105M | 67.7 | kernel | 83.9 | 1.000x |
| retained all-edge | three-hop all-edge, 25% | 985.289 | 1.959x | cached O_DIRECT | **209.260M** | **4.608M** | 589.6 | 68.84% | 116.6 | **0.533x** |
| retained all-edge | three-hop all-edge, 50% | 2041.801 | 4.059x | mmap | 40.083M | 1.528M | 103.8 | kernel | 111.3 | 1.000x |
| retained all-edge | three-hop all-edge, 50% | 2041.801 | 4.059x | cached O_DIRECT | **140.853M** | **5.929M** | 756.2 | 40.47% | 87.4 | **0.223x** |
| retained all-edge | three-hop all-edge, 100% | 4083.761 | 8.119x | mmap | 22.611M | 1.306M | 78.6 | kernel | 44.5 | 1.000x |
| retained all-edge | three-hop all-edge, 100% | 4083.761 | 8.119x | cached O_DIRECT | **104.437M** | **6.123M** | 808.8 | 17.09% | 66.5 | **0.324x** |

The storage reference is 6.577M 4 KiB random-read IOPS at aggregate QD1024
through md/XFS, with 6.62M at QD2048 but approximately twice the latency. The
four-drive advertised aggregate is about 8.8M IOPS.

## Full reports

- [Hotspot Pinterest real-primary-key mmap versus cached O_DIRECT](diskgraph-pinterest-hotspot-real-pkey-2026-08-21.md)
  records the original pruned generator, corrected explicit edge tables,
  204B-entry ingest, full-source uniform real keys, returned primary-key
  materialization, independent concurrency selection, and a 4.046x Direct
  speedup on the 1,177-GiB index.
- [Organic Pinterest capped-neighborhood mmap versus cached O_DIRECT](diskgraph-pinterest-organic-capped-bfs-2026-08-21.md)
  records the stateless broken-power-law generator, measured inbound/outbound
  degree validation, real-primary-key capped BFS, the initial 515-GiB result,
  and a size-controlled 1,217-GiB rerun. At the larger scale Direct is
  5.340x faster for uniform roots and 4.994x faster for Pinterest-style skewed
  roots; the request skew reduces reads per node but does not increase the 4
  KiB page-hit rate.
- [Real-primary-key mmap versus cached O_DIRECT](diskgraph-real-pkey-e2e-mmap-vs-direct-2026-08-20.md)
  measures uniform primary keys read from the original Parquet node table,
  timed primary-key-map lookup, three-hop sampling, and returned primary-key
  materialization. It records independently tuned mmap/Direct results for the
  1%, 5%, and 25% identity graphs.
- [Local-NVMe feature-heavy ecommerce mmap versus cached O_DIRECT](diskgraph-localssd-ecommerce-feature-mmap-vs-direct-2026-08-19.md)
  records the 1.23-TiB, 10%-scale ecommerce feature workload on a single local
  NVMe and a 240-GiB memory cap. It also validates the #443 per-shard scratch
  reclamation backport: a stopped pre-#443 Index resumes and completes with
  `_tmp` removed, then cached Direct is 1.526x faster than mmap.
- [Local-NVMe mmap versus cached O_DIRECT scale check](diskgraph-localssd-mmap-vs-direct-2026-08-18.md)
  records the existing S3 dataset inventory, RAM-fitting 1% and largest-safe
  10% local-scratch reruns on one 1.8-TB NVMe, stable-cache gates, identical
  output verification, and the 5% cache-ramp diagnostic that is intentionally
  not retained. Direct wins by 12.1% at 1% but loses to mmap at the 1.61x-RAM
  10% point while saturating the single device.
- [Pinterest OmniSage pruned/unpruned on i8g.24xlarge](diskgraph-pinterest-i8g-io-benchmark-2026-08-09.md)
  is the current two-hop Pinterest comparison. It records the strict direction-
  reversing plan, 1.18/3.20-TiB index snapshots, 640-GiB Direct cache, current
  six-Nitro-SSD fio ceiling, steady-cache gates, and paired mmap/Direct results
  on 742 GiB RAM.
- [Identity 1%/5%/25%/50% scale matrix on i8g.24xlarge](diskgraph-i8g-identity-scale-matrix-2026-08-10.md)
  records fresh generation and ingest, independently tuned mmap/Direct
  concurrency, 640/512-GiB Direct caches, long steady-state warmups, identical
  output gates, current-host CPU/disk validation, and the additive comparison
  with the earlier Threadripper matrix.
- [Feature-heavy e-commerce 1%/5%/25%/50% scale matrix on i8g.24xlarge](diskgraph-i8g-ecommerce-feature-io-benchmark-2026-08-10.md)
  records fresh generation and ingest, a viable 640-GiB Direct cache,
  flat-memory warmup gates, independently tuned concurrency, current-host fio
  validation, identical output gates, and the additive Threadripper comparison.
- [Feature-heavy e-commerce 1%/5%/25%/50%/100% scale matrix](diskgraph-ecommerce-feature-io-benchmark-2026-08-08.md)
  is the earlier Threadripper 1–2 KiB feature-I/O comparison. It records four-hop all-edge
  sampling, feature materialization, offset-guided multi-page eager reads, and
  the index/502.973-GiB-RAM ratio that governs the resident-to-storage-bound
  transition. The completed 100% point covers a 12,449.492-GiB index
  (24.752x RAM) and records the transition from IOPS to read-bandwidth
  saturation.
- [Fresh 1%/5%/25%/50%/100% three-hop all-edge scale matrix](diskgraph-all-edge-3hop-scale-matrix-2026-08-08.md)
  is the current comparison. It records compressed generation, independent
  concurrency selection, CPU and disk saturation, identical-output gates, and
  the remaining fio gap at every scale.
- [Detailed 100% three-hop all-edge mmap versus cached O_DIRECT](diskgraph-all-edge-3hop-100pct-mmap-vs-direct-2026-08-07.md)
  retains the adjacent full-scale profiler, cache, allocation, and runtime-gap
  investigation.
- [Cached O_DIRECT optimization baseline](diskgraph-direct-cache-optimized-2026-08-07.md)
  contains the allocator, batching, cache, hot-hit CPU, and storage-pressure
  investigations that made Direct faster than mmap.
- [Sampler I/O redesign baseline](diskgraph-sampler-io-benchmark-2026-08-06.md)
  documents the repaired strict-plan baseline, index layout, initial buffered
  and Direct implementations, and why the first userspace cache regressed.
- [Superseded mmap sampling baseline](../sampling-mmap-baseline-2026-08-06.md)
  is retained as historical context and is clearly marked for the old silently
  ignored `--relations` argument.

## Reproduction entry points

- `scripts/bench_sampling_iops.sh`: production sampler mmap/buffered/Direct
  comparison with per-index eviction and no-swap cgroups.
- `scripts/bench_pinterest_organic_pkey.sh`: real-primary-key three-hop
  all-relation Pinterest preset with an 80-neighbor cap.
- `scripts/bench_pinterest_pruned_pkey.sh`: real-primary-key version of the
  original hotspot-based pruned Pinterest two-hop traversal.
- `scripts/bench_fio_iops.sh`: filesystem or explicitly acknowledged raw-device
  read-only fio ceiling.
- `scripts/bench_fio_raw_matrix.sh`: multi-QD raw-device matrix.
- `dg-page-io-bench --directory INDEX --file-count N`: cacheless multi-shard
  runtime isolation without graph planning.
- `scripts/evict_file_cache.py`: file-scoped `POSIX_FADV_DONTNEED` eviction.
