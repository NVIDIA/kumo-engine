# PyTorch suite roadmap

Development is consolidated in [PR #353](https://github.com/NVIDIA/kumo-engine/pull/353).
This roadmap records scope and remaining acceptance work. It does not certify a
passing package revision. Commands and report locations live in the [run guide](README.md);
requirement owners live in the [coverage map](docs/coverage-audit.md).
Historical progress and superseded execution results remain in Git history.

## Agreed scope

- Keep models, fixtures, dataset inventories and runners outside the installed package.
  Use Engine's existing build/setup and package tests. Arena is a provenance/data
  source, with no runtime checkout or import dependency.
- Test DiskGraph against independently calculated answers. InMemory is prohibited
  as a test backend or oracle; its existing internal `ServingAssembler` dependency
  is permitted. Do not modify package behavior to make this testing PR pass.
- Own populations, shuffling, planning and iteration through native `AnchorTable`.
  Production loading never calls `task.batches(requests)`.
- Keep every required assertion even when behavior is unsupported. Only the
  [approved explicit refusals](docs/required-coverage.md#capability-refusals) pass;
  crashes, wrong results, late failures, silent fallback, skips and xfails fail.
- Equal-time events may permute, including eligible alternatives at clipping
  boundaries. Still check chronology, cutoff, membership, multiplicity and values.
  Graph comparisons match identities/connectivity, not tensor order.
- PR, scheduled and manual full-suite runs use the same plan. Weekly coverage is
  outside scope. T01 real-data campaigns are explicitly on demand; its fixtures
  remain required in the default suite.
- Target roughly 20 minutes through parallel independent campaigns, without
  truncating required populations, repetitions or checks. This is not a timeout.
- Benchmark only DiskGraph. Keep CUDA model training and GPU collectors; the
  copied Polars/GPU reference engines have been removed. HI Small retains
  fitting-memory checks but no model/index/loader timing campaigns.

## Binding coverage

| Dataset | Active bindings | Campaigns |
| --- | --- | --- |
| AML HI Small | E01 GraphSAGE, E02 Combined, E03 PRAGMA, E04/E05 nuFormer | Correctness, two 100-update CPU training repetitions, lifecycle, full population; 25k/400k fitting memory |
| AML LI Medium | L01 Combined | Correctness/features, histories, training, lifecycle, full population; training/index/loader/length-planning benchmarks |
| Synthetic Medium | S01 PRAGMA, S02 nuFormer, S04 Combined | Correctness/features, training, lifecycle, full population |
| MBD | M03 nuFormer, M04 GraphSAGE | Correctness/features, training, lifecycle, full population |
| Streaming 25B | T01 GraphSAGE | Same four binding campaigns, on demand; full sources with a fixed 10,000-viewer cohort |

The [catalog](workloads/bindings.json) is the execution authority. L02/L03/L04
real-data campaigns remain disabled; their fixture assertions remain. Add one
binding at a time through the shared dataset, anchor, model and runner interfaces.
See [feature requirements and assertion owners](docs/binding-coverage.md).

## Coverage ownership

| Boundary | Owner |
| --- | --- |
| Generic API, encoding, sampling, clocks, fitting, persistence | Existing package tests and DiskGraph conformance |
| Source/anchor mapping, task targets, model consumption and feature checks | `model_checks/`, binding campaigns and independent source oracles |
| Difficult feature interactions and documentation ambiguities | `difficult_behavior/` with authored evidence and expected outcomes |
| Fixed-fixture learning vs real-data 100-update acceptance | `model_checks/test_learning.py` vs shared `training_acceptance.py` |
| Data acquisition, orchestration, reporting and refusal guards | `checks/`; CPU/difficult inventories reject missing or skipped cases |
| Stage metrics, cache/state/index comparisons and collector validity | `benchmarking/`, native benchmark fixtures, Linux/GPU validation jobs |

Preserve the [selective Arena golden-case mapping](docs/arena-golden-audit.md).
Do not delete an assertion merely because a different test uses a similar API.

## Remaining acceptance work

1. Validate the final integrated revision with the full suite. Review package,
   conformance, difficult-behavior, real-data, training, lifecycle and profiler
   results collectively, with child logs retained for failures.
2. Keep package defects visible. The last locally reproduced L01 length-planning
   check failed because the native iterator omitted the required merged-history
   plan; it therefore did not establish second-epoch length-cache reuse.
3. Obtain passing full-population and lifecycle evidence for the selected real
   bindings. A declaration, tiny fixture, bounded run or accepted refusal does
   not prove full-data execution. T01 also needs sufficient runner storage;
   administrator runner provisioning remains outside this change.
4. Confirm all applicable benchmark metrics and experimental conditions below.
   A missing collector or failed reading remains incomplete, never zero.
5. Resolve direct PR admission to the RTX runner separately. Keep the PR trigger
   and required GPU jobs. Required-check activation and post-merge schedule
   verification remain deferred administrative/acceptance steps; see
   [automation](docs/required-coverage.md#github-merge-enforcement).

The testing additions from #361/#362/#363 were consolidated into #353. Fitting
optimizations #351/#352 remain separate; no inference about their current PR
status is made here. The suite's merge gate supersedes the outdated #284 design.

## Complete metric checklist

The benchmark guide defines [collection and reporting](docs/benchmarking.md).
The following preserves the supplied nightly measurement requirements, excluding
reference-engine comparison removed by the user's later scope decision.

**Stage measurements.** For every workload, record enclosing elapsed time and
peak host memory separately for ingestion, materialization and batch production.
Also instrument the named operations below explicitly. Preserve parent/child
intervals and operation identity instead of summing nested time. The design
will retain stage-memory attribution for these operations where separable;
fused work must identify its shared scope rather than invent exclusive costs.

| Stage | Every named operation requiring an explicit elapsed-time measurement |
| --- | --- |
| Ingestion | Source reading; source conversion; new DiskGraph index construction; index reopening separately from construction |
| Materialization | Anchor preparation; input fitting; target fitting; vocabulary fitting; numerical-statistics fitting; BPE fitting; task setup; saved-state loading/reuse separately from fresh fitting |
| Native batch production | Enclosing batch production; DiskGraph/Rust sampling; retrieval; aggregate computation |
| Python batch production | Planning; post-processing, including merge, encoding, tokenization, label production, mask production and tensor assembly |

**Counts, sizes, rates, memory and latency.** Counts underlying rates are retained
alongside durations, never inferred from configured batch limits.

| Metric | Required definition |
| --- | --- |
| Source rows | Actual source/input and converted-output row counts, with table identity |
| Source bytes | Source/input and converted-output logical bytes, with scope; not mislabelled as physical device I/O |
| Index size | Measured index bytes after construction and on reuse |
| Rows examined | Actual rows considered by the observed operation, including filtered-out work; count at the relevant source/native boundary |
| Rows decoded | Actual decoded rows at the relevant boundary; distinguish repeated decoding from unique source rows |
| Real example count and samples/s | Emitted example occurrences excluding distributed request padding; primary rate for every workload |
| Input token count and tokens/s | Non-padding input sequence positions: encoded values, BPE pieces and markers, each counted once; exclude targets and parallel field-ID arrays; numerical columns are not tokens |
| Emitted node count and nodes/s | Actual graph node occurrences, preserving output multiplicity |
| Emitted edge count and edges/s | Actual graph edge occurrences, preserving output multiplicity |
| Actual event counts | Emitted event occurrences with representation-specific provenance; no estimate from token count or history cap |
| Actual output counts | Produced batches and output elements by representation; retain partial tails and separate terminal requests |
| Batch sizes | Actual examples and applicable tokens/nodes/edges in each emitted batch, including padding information |
| Output sizes | Actual emitted tensor shapes, element counts and bytes, with a stated convention for shared storage |
| Peak host memory | Separate ingestion/materialization/batch-production peaks including native allocations and workload children; memory scope and peak-measurement method retained |
| Peak allocated GPU memory | Framework allocated-memory high-water counter for each measured window |
| Peak reserved GPU memory | Framework reserved-memory high-water counter for each measured window, separate from allocated memory |
| First-batch latency | Startup-through-first-produced-batch elapsed time, plus separate iterator-construction and first-request timings |
| p50 batch latency | Median of individual produced-request elapsed times, with population/sample count stated |
| p95 batch latency | 95th percentile of produced-request elapsed times, with percentile method and sample count stated |
| Training steps/s | Completed training iterations divided by completed execution-window time; retain step count and optimizer-update count separately |
| Batch-wait time | Explicit host time spent awaiting loader results; terminal requests identified separately |
| Transfer time | Explicit host-to-device transfer host spans and device-event elapsed time with clock/stream scope |
| Sampled GPU utilization | Device-identified utilization observations and summary, with interval, timestamps, gaps and measured steady-state coverage |

**Required experimental separations and records.** These are not additional
throughput counters; they are mandatory conditions for interpreting the metrics.

- First-batch versus measured steady-state execution, with a defined warmup and
  observation window; do not automatically call all later batches steady state.
- First epoch versus later epochs, with separate time/count/memory records.
- Controlled cold caches versus warm caches, declaring OS page-cache, index,
  application and device cache scope and verifying the performed procedure.
- Fresh input/target fitting versus loaded/reused fitted state.
- Repeated independent runs; retain every repetition and observed variation.
- Code revision/source identity and exact dependency versions/pins.
- Hardware identity/configuration, including CPU, memory, storage and GPU.
- Actual batch/output sizes, memory scope and rows examined/decoded as above.
- Probe settings: enabled instrumentation, sampling cadence, clocks, warmup,
  repetition/epoch settings and measured instrumentation overhead.
- Dataset/anchor identities, selected population, splits, model, precision,
  threads, seed and fitted-state identity so each result can be reproduced.

## Measurement acceptance

Each metric needs a source, unit, scope and validation. Nested host/native/device
spans may overlap and must not be summed into a fabricated total. Unexplained
time is not Python time. Preserve raw observations, code/dependency identities,
configuration, counts and failure evidence in artifacts.

Host peaks include native allocations and workload children. Label sampled
peaks explicitly; the user approved sampled maxima on the existing runner.
Keep anonymous memory, total host memory and allocated/reserved device peaks
separate. Source/index bytes are not physical SSD I/O throughput.

Require measured first/later epochs, warmup/steady-state windows, independent
repetitions, cold/warm cache evidence, fresh/reused fitting and one index build
followed by two reopens. Cache or fitted-state reuse never follows merely from
starting another process. Calibrate performance thresholds only after collecting
variation on matching workloads and hardware. Fitting-memory growth is reported
without a pass/fail ratio limit.
