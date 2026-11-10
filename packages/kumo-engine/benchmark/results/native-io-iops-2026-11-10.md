# DiskGraph: observed IOPS during instrumented training and prediction

**90 instrumented workflows completed and passed portable replay and independent raw-counter audit.** These are instrumented workload observations, not SSD capacity ratings. The [original untraced latency report](https://github.com/NVIDIA/kumo-engine/blob/3e65cd8cb667faba35d40685c367de7511438120/packages/kumo-engine/benchmark/results/native-io-2026-11-09.md) remains a separate, unchanged experiment.

The benchmark executes real public fit → first prediction → repeat prediction → reload/prediction workflows on three generated relational datasets. It compares mmap, buffered io_uring and Direct I/O under resident and memory-constrained conditions. Higher IOPS is not intrinsically faster: a cache can improve latency while avoiding storage requests entirely.

## Results

Every entry below is the median of five per-trial rates, rather than a rate calculated from median counts or pooled time. Physical read/write/total columns count final completed device requests. Native requests count completed Read/Readv SQEs after page coalescing; they are a different unit. Native mmap requests are unavailable, not zero. Measured physical mmap zeros are valid warm-cache observations.

### Training

| Task | Regime | Backend | Physical read IOPS | Physical write IOPS | Physical total IOPS | Native read requests/s |
|---|---|---|---:|---:|---:|---:|
| Binary classification | Resident | mmap | 0.0 | 16.1 | 16.1 | N/A |
| Binary classification | Resident | buffered | 0.0 | 12.3 | 12.3 | 231,420 |
| Binary classification | Resident | direct | 24,085.6 | 10.0 | 24,095.6 | 24,086 |
| Binary classification | Constrained | mmap | 7,304.5 | 1.8 | 7,306.3 | N/A |
| Binary classification | Constrained | buffered | 32,107.2 | 11.6 | 32,118.7 | 216,085 |
| Binary classification | Constrained | direct | 38,934.0 | 9.9 | 38,943.9 | 38,931 |
| Regression | Resident | mmap | 0.0 | 1.6 | 1.6 | N/A |
| Regression | Resident | buffered | 0.0 | 1.2 | 1.2 | 224,057 |
| Regression | Resident | direct | 25,459.5 | 1.1 | 25,460.5 | 25,460 |
| Regression | Constrained | mmap | 7,477.2 | 0.3 | 7,477.5 | N/A |
| Regression | Constrained | buffered | 32,988.8 | 1.1 | 32,989.9 | 206,178 |
| Regression | Constrained | direct | 37,788.1 | 1.0 | 37,789.2 | 37,785 |
| Temporal link prediction | Resident | mmap | 0.0 | 24.3 | 24.3 | N/A |
| Temporal link prediction | Resident | buffered | 0.0 | 24.1 | 24.1 | 28,858 |
| Temporal link prediction | Resident | direct | 7,777.7 | 22.2 | 7,799.9 | 7,778 |
| Temporal link prediction | Constrained | mmap | 4,630.3 | 11.7 | 4,642.4 | N/A |
| Temporal link prediction | Constrained | buffered | 6,358.5 | 23.1 | 6,381.6 | 27,670 |
| Temporal link prediction | Constrained | direct | 10,635.6 | 22.3 | 10,657.9 | 10,633 |

### Fit through first prediction (E2E)

| Task | Regime | Backend | Physical read IOPS | Physical write IOPS | Physical total IOPS | Native read requests/s |
|---|---|---|---:|---:|---:|---:|
| Binary classification | Resident | mmap | 0.0 | 34.9 | 34.9 | N/A |
| Binary classification | Resident | buffered | 0.0 | 27.7 | 27.7 | 204,057 |
| Binary classification | Resident | direct | 23,021.5 | 23.1 | 23,044.6 | 23,022 |
| Binary classification | Constrained | mmap | 7,120.2 | 5.4 | 7,125.6 | N/A |
| Binary classification | Constrained | buffered | 26,051.2 | 23.7 | 26,080.4 | 170,922 |
| Binary classification | Constrained | direct | 32,924.1 | 24.4 | 32,949.6 | 32,363 |
| Regression | Resident | mmap | 0.0 | 7.2 | 7.2 | N/A |
| Regression | Resident | buffered | 0.0 | 5.7 | 5.7 | 207,154 |
| Regression | Resident | direct | 24,425.3 | 4.9 | 24,430.2 | 24,426 |
| Regression | Constrained | mmap | 7,238.5 | 0.9 | 7,239.4 | N/A |
| Regression | Constrained | buffered | 27,178.5 | 4.6 | 27,183.1 | 165,988 |
| Regression | Constrained | direct | 31,652.1 | 4.1 | 31,656.2 | 31,073 |
| Temporal link prediction | Resident | mmap | 0.0 | 36.6 | 36.6 | N/A |
| Temporal link prediction | Resident | buffered | 0.0 | 36.3 | 36.3 | 26,045 |
| Temporal link prediction | Resident | direct | 7,697.3 | 33.6 | 7,730.8 | 7,771 |
| Temporal link prediction | Constrained | mmap | 4,295.5 | 18.8 | 4,313.4 | N/A |
| Temporal link prediction | Constrained | buffered | 5,118.0 | 29.9 | 5,147.9 | 18,257 |
| Temporal link prediction | Constrained | direct | 8,968.9 | 31.4 | 9,000.3 | 7,994 |

Total IOPS is computed as `(completed reads + completed writes) / phase seconds` for each trial before taking its median; its displayed median need not equal the sum of the separate read/write medians. The artifact retains unrounded floats; displayed physical rates use one decimal and native rates use explicit rounding.

The artifact also retains actual completed read/write payload bytes and mean completed-read payload KiB/request for every phase/trial. Mean read payload is `completed_read_bytes / completed_reads / 1024`; it is N/A when no reads complete. Group payload medians use defined per-trial values and retain both `payload_defined_trials` and `payload_total_trials`. Coverage accompanies each payload median below. A zero median read IOPS does not establish that every trial had zero reads. The following are medians of per-trial actual completed payload volumes and defined per-trial mean payloads. Read/write volumes use MiB (2²⁰ bytes); unrounded bytes and counts remain in the artifact. Binary and link-prediction resident buffered E2E each have reads in only one of five trials.

| Task | Regime | Backend | Training read/write MiB | Training mean read KiB (defined/5) | E2E read/write MiB | E2E mean read KiB (defined/5) |
|---|---|---|---:|---:|---:|---:|
| Binary classification | Resident | mmap | 0.0 / 70.0 | N/A (0/5) | 0.0 / 181.4 | N/A (0/5) |
| Binary classification | Resident | buffered | 0.0 / 69.9 | N/A (0/5) | 0.0 / 181.4 | 4.000 (1/5) |
| Binary classification | Resident | direct | 7,109.1 / 69.9 | 5.206 (5/5) | 7,600.1 / 181.4 | 5.161 (5/5) |
| Binary classification | Constrained | mmap | 9,511.8 / 70.1 | 4.007 (5/5) | 9,884.7 / 229.2 | 4.028 (5/5) |
| Binary classification | Constrained | buffered | 8,658.2 / 70.0 | 5.429 (5/5) | 8,988.8 / 181.4 | 5.420 (5/5) |
| Binary classification | Constrained | direct | 11,591.4 / 69.9 | 5.170 (5/5) | 12,280.8 / 222.2 | 5.139 (5/5) |
| Regression | Resident | mmap | 0.0 / 6.2 | N/A (0/5) | 0.0 / 30.4 | N/A (0/5) |
| Regression | Resident | buffered | 0.0 / 6.2 | N/A (0/5) | 0.0 / 30.4 | N/A (0/5) |
| Regression | Resident | direct | 7,172.3 / 6.2 | 5.244 (5/5) | 7,471.5 / 30.4 | 5.232 (5/5) |
| Regression | Constrained | mmap | 9,476.6 / 6.3 | 4.007 (5/5) | 9,792.4 / 30.5 | 4.024 (5/5) |
| Regression | Constrained | buffered | 8,891.4 / 6.2 | 5.453 (5/5) | 9,196.7 / 30.4 | 5.441 (5/5) |
| Regression | Constrained | direct | 11,202.9 / 6.2 | 5.290 (5/5) | 11,690.8 / 30.4 | 5.269 (5/5) |
| Temporal link prediction | Resident | mmap | 0.0 / 192.3 | N/A (0/5) | 0.0 / 338.9 | N/A (0/5) |
| Temporal link prediction | Resident | buffered | 0.0 / 192.3 | N/A (0/5) | 0.0 / 338.9 | 4.000 (1/5) |
| Temporal link prediction | Resident | direct | 2,722.1 / 192.3 | 5.111 (5/5) | 3,955.3 / 338.9 | 6.324 (5/5) |
| Temporal link prediction | Constrained | mmap | 2,352.9 / 192.3 | 4.021 (5/5) | 3,148.8 / 402.7 | 4.269 (5/5) |
| Temporal link prediction | Constrained | buffered | 2,214.7 / 192.3 | 5.290 (5/5) | 2,951.4 / 398.8 | 5.379 (5/5) |
| Temporal link prediction | Constrained | direct | 3,595.7 / 192.3 | 4.970 (5/5) | 5,422.9 / 402.7 | 5.879 (5/5) |

## What is measured

The collector records BIO-origin cgroup identity when a request is issued after merging, then counts its final `block_io_done` callback on physical NVMe device `/dev/nvme0n1` (`259:0`). Only requests attributed to the exact trial cgroup qualify. Requests from other cgroups and SSD-internal operations are outside this definition. Requeues retain request identity; merged-away requests reconcile separately. The collector uses bounded request maps and per-CPU counters rather than streaming one event to userspace per completion.

For each phase/trial, physical read, write and total rates divide the corresponding final-completion counts by that phase's **BPF marker-window seconds**. Native read requests/s divides the native completed coalesced Read/Readv SQE delta by the **workload's monotonic interval seconds**. Native submitted pages are not completed SQEs. No operation count is inferred from bytes/4,096. Training and E2E are reported separately; explicit E2E markers span fit through the first prediction, including the intervening gap.

Client calls and kernel marker transitions have recorded monotonic brackets; raw clocks and interval-alignment offsets are retained. A request issued before a phase but completed inside it counts in that phase. Carry-in/out reads and pending reads/writes at boundaries remain visible. Completed payload bytes include partial callbacks and are assigned to the request's final-callback phase; they are not bytes transferred exclusively during that window. Read/write/total IOPS cover completions inside the marked workload window, including target-cgroup background writeback that finishes then. They exclude future writeback after the window ends and do not measure foreground write submissions or all eventual writes caused by the workflow. Write completions can occur after the API returns, so these write IOPS do not establish durability. The workload adds no fsync to change this behavior.

Raw per-device cgroup `io.stat` BIO-start/byte counters and process storage-byte counters provide separate consistency evidence. BIO starts, final request completions, native SQEs and logical pages are not interchangeable. We do not have a measured denominator for useful logical byte demand; these counters alone support no read-amplification ratio or causal claim about readahead/wasted reads.

## Controls and source scope

There are 3 tasks × 2 regimes × 3 backends × 5 repetitions = **90 fresh processes**. Binary/regression run 1,280 training batches; temporal link prediction runs 256 batches, with 512 actual optimizer updates. Workload plans, logical seed streams, predictions, task shard layouts, backend configuration, native artifact, original prepared-data oracles and resource eligibility must pass the unchanged canonical validators.

The ARM CPU host uses model cores `5-9,15-19`, Torch threads 1, Rayon/native threads 4, workers 0, standalone loaders, 8 I/O rings, queue depth 256 and batch multiplier 1. Ingestion controls pin context/encode threads and temporal-link shard layout for comparable logical work. Observer userspace runs on efficiency cores `0-4,10-14`; kernel probe cost still executes in kernel event context. These settings do not represent every production configuration or GPU workload.

Resident trials use 64 GiB cgroups and native cache capacity 16 GiB. Constrained trials use 4 GiB, native cache 2 GiB for binary/regression and 1 GiB for link prediction. Swap is disabled. Each trial receives a fresh private-inode copy verified against the original pristine oracle; only owned files are evicted before cgroup creation. Resident warming happens inside the cgroup: its OS page cache begins warm, while the new process's Direct cache begins cold. All 45 constrained trials passed actual training pressure checks; all 90 training intervals exceeded 30 seconds (minimum 33.871 seconds), with no OOM events.

The instrumented runs use transient SYSTEM services with cgroup I/O accounting; the original untraced runs used USER scopes. Effective hierarchy/affinity evidence is preserved; the latency experiments remain separate.

| Scope | Exact revision |
|---|---|
| Measured instrumentation/runtime source | `ac7778cd134902178077d1ff179129e3ea40d91d` |
| Optimized native build source | `75ca0e2fd6797827feda61d637b59b75e2d6f678` |
| Loaded native SHA-256 | `a5949f96723df2684c6cdbd5884bcc25c72129c8a629bfffa49bc36f9552140f` |
| Core port draft [PR #169](https://github.com/NVIDIA/kumo-engine/pull/169) | `c1a04a305f569cf4fc6a1284e054b9c4277f3625` |

The measured native candidate includes the core port plus read-only diagnostics and opt-in ingestion controls used by the benchmark. It is not a claim that the benchmark's full native tree is byte-identical to the core PR. Original prepared-data, wheel/build, measurement-source and four canonical-validator hashes are pinned in the artifact. Instrumentation adds optional phase markers/metadata; the native build remains unchanged from `75ca`. A later report commit may be a descendant of measured source; offline replay requires the pinned source **bytes**, rather than requiring that later HEAD equal `ac7778`.

## Observer-cost qualification

The synthetic 20-trial qualification remains **HOLD**. Five fixed-count traced/untraced pairs per backend retained all CRC-verified owned-file trials. Direct showed +17.505% median elapsed inflation at a median 428,214 final requests/s, with 0.682% untraced relative MAD. Cold mmap showed +3.995% inflation with 10.829% untraced relative MAD, so its precision gate failed; differential inflation was 13.510 percentage points. Some Direct windows were shorter than the planned five seconds (minimum 4.099 seconds). No trial was discarded or threshold widened.

Eight separately authorized public-workflow diagnostics passed their **scoped E2E** gates: two balanced pairs/backend, binary 256 batches, resident 64 GiB/cache 16 GiB, markers enabled in both traced and untraced variants. Median paired E2E inflation was +0.349% for mmap and +2.526% for Direct; differential 2.176 percentage points; untraced relative MAD 0.257%/0.241%. Training inflation was +0.602%/+3.149%. Maximum recorded model phase-average physical rate was 39,883 requests/s; these averages do not bound instantaneous bursts.

Two pairs do not prove low overhead across every task, regime or GPU. Do not linearly extrapolate the high-rate stress cost, apply an overhead correction factor, or describe tracing as zero overhead. The 90 observations are explicitly **instrumented IOPS**; retain their times and counter evidence while using the separate original untraced experiment for latency conclusions. Across the 90 runs, the maximum recorded phase-average read rate was 39,340.854 requests/s; none exceeded the short public diagnostic maximum of 39,882.531. This comparison does not bound bursts or establish unbiased rates in every regime.

## Portable evidence and replay

The artifact packages all 90 exact original row texts/SHA through unchanged `native_io_results`, plus each exact raw trace and exit-zero STOP text/SHA bound to the original row SHA and trace/result paths. Separate archives preserve all 20 stress and eight public diagnostic trials, source/command/proof records, build provenance and independent audits. Graphs, Parquet, model weights and tensor traces are excluded.

The thin IOPS reader checks canonical native/work/prediction/model/control/resource eligibility, all 90 unique task/regime/backend/repetition coordinates, unique measured process/cgroup/service identities, measurement/native/C/BTF/device pins, exactly 12 phase markers, clocks, loss/error diagnostics, request reconciliation, pending/drain state and positive constrained attribution. It recomputes raw read/write/total rates, native SQE deltas, actual payload bytes and five-trial medians. Mean payload is descriptive, with zero-read trials undefined.

Full replay passed both in place and after relocation to a different scratch directory. The relocated replay ran as the unprivileged service user with historical paths, DiskGraph imports and subprocess/git access blocked. It required no prepared graph, native SO, privileged collector or live kernel state. Original absolute paths remain identity bindings only. All 15 actual-archive corruption cases were rejected, including fully rehashed STOP failure, pending/loss, marker/clock, request reconciliation, collector/BTF/cgroup identity, zero constrained attribution and negative native SQE mutations; restoring the archive restored positive replay. Focused helper tests passed 40/40; the complete benchmark suite passed 156/156. Changed Python flake8 and isort checks passed.

### Replay and artifact contents

From the repository root, with repo-local Python dependencies installed:

```bash
python packages/kumo-engine/benchmark/native_io_iops_results.py \
  --repo-root . \
  --replay packages/kumo-engine/benchmark/results/native-io-iops-2026-11-10
```

Expected: `IOPS_REPLAY_PASS`, 90 records, six canonical variants and 18 backend groups. Replay reads archived rows and sidecars plus exact current source bytes; it launches no model or collector.

- [Artifact manifest](native-io-iops-2026-11-10/iops-manifest.json): source snapshots, helper SHA and archive hashes.
- [Canonical manifest](native-io-iops-2026-11-10/rows/manifest.json), [controls](native-io-iops-2026-11-10/rows/controls.json), and [original rows](native-io-iops-2026-11-10/rows/raw-trials.jsonl.gz).
- [Raw trace/STOP archive](native-io-iops-2026-11-10/raw-traces.jsonl.gz) and [derived observations](native-io-iops-2026-11-10/observations.json), including all per-trial clocks, counts, bytes and six phase rates.
- Separate [stress20](native-io-iops-2026-11-10/evidence-stress20.jsonl.gz) and [public8](native-io-iops-2026-11-10/evidence-public8.jsonl.gz) observer-cost evidence. Their different qualification outcomes remain explicit.
- [Relocated replay proof](native-io-iops-2026-11-10/evidence-relocated-replay-proof.jsonl.gz), [actual raw-tamper proof](native-io-iops-2026-11-10/evidence-real-tamper-proof.jsonl.gz), and [independent full audit](native-io-iops-2026-11-10/evidence-independent-final.jsonl.gz).
- [Thin replay reader](../native_io_iops_results.py) and [artifact guide](native-io-iops-2026-11-10/README.md). The manifest pins the reader bytes; the signed delivery commit pins reader, report and all evidence.

All evidence archives contain exact original text/SHA records, not only summarized conclusions. The artifact guide lists the full evidence inventory. The original untraced latency report is preserved, with a brief link to this separate measurement.
