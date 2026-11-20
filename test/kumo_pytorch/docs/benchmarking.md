# General benchmark interface

The runner under [benchmarking/](../benchmarking/) belongs to the testing suite,
not the installed package. It has no Arena dependency and makes no decisions
based on dataset, model-family or backend names. A workload adapter supplies the
actual operations and semantic counts. New datasets/models do not require edits
to the runner, metric catalog or report generator.

## Execution interface

A trusted Python factory `factory(config, workdir) -> Workload` receives a JSON
configuration and a new scratch directory per repetition. It supplies:

| Member | Responsibility |
| --- | --- |
| `identity` | Dataset/workload identity and relevant data checksums, population, backend, model and settings |
| `prepare(recorder)` | Prepare sources, anchors, indexes, fitted state and consumer; instrument actual operations |
| `batches(epoch)` | Return the complete declared epoch; preserve workload semantics and partial tails |
| `count(batch)` | Read-only `BatchCounts` from actual emitted metadata, validity and output shapes |
| `step(batch, recorder)` | Optional model/objective/backward/update; return whether an optimizer update occurred |
| `device`, `synchronize()` | Declare execution device and finish its work at timing-window boundaries; non-CPU execution requires an explicit callback |
| `close()` | Release resources even after partially failed preparation |
| `before_epoch(epoch, recorder)` | Optional explicit experiment setup outside the measured execution window |
| `inapplicable` | Per-metric reasons for genuinely absent operations; never a way to excuse missing instrumentation |

`BatchCounts` requires sample and request-padding counts. Optional token, graph,
event and output-size fields use `None` for missing measurements; zero means
measured zero work. Missing counts in any batch make that epoch's corresponding
total/rate unavailable. Count token positions once, excluding tensor padding,
targets and parallel field-ID arrays. Count graph/event output occurrences, not
unique dataset entities. Define output-size handling of views/shared storage in
the adapter identity. The runner never traverses unknown payloads to guess.

Adapters use `with recorder.span("source.read"):` for observed operations and
`recorder.measure("source.rows", actual_rows)` for counters. The
[catalog](../benchmarking/metrics.py) names all accepted required fields and units.
Spans retain parent IDs and start/stop timestamps; nested durations are not added.
Unexpected names, negative/nonfinite measurements and invalid counts fail.
Semantic correctness checks belong outside production timing loops; assertions
inside integration-test callbacks validate the interface, not throughput.

## Kumo adapter

[kumo_workload](../benchmarking/kumo.py) implements the lifecycle using public
Kumo APIs. An `inputs(recorder)` callback returns `KumoInputs` containing the
contract, dataset, backend, a Parquet `AnchorTable`, materialization options and
batching options. Select `fit_split` for fitting and `split` for iteration.
The runner owns the epoch argument. It calls
`task.batches(split=..., epoch=..., output_format="model")` without requests;
the task owns its population, shuffling, planning and iteration.

DiskGraph benchmarks fail unless they exhaust the native model iterator.
Reports retain native iteration and merged-history evidence for each epoch.
AML sequence-only contracts enable length bucketing: the first epoch must
compute every anchor length; subsequent epochs must reuse the same native
cache and report hits for every anchor, with no recomputation. Graph/combined
contracts retain fixed-size native batching; they do not claim length-cache
coverage. Exclusive native length-planning time is unavailable in the current
API. The labelled host iterator boundary includes planning and retrieval and
must not be added to nested native timings.

An optional `consumer_factory(task, recorder)` creates any model/optimizer and
returns its step callback. It is not limited to the copied AML consumers.
Use the `inputs` callback to perform and measure source/anchor work rather than
loading data in the outer factory. The adapter closes task and plan contexts on
success or failure. Prepare and materialize timers remain enclosing calls;
internal/native phases have separate scoped instrumentation described below.

The native [integration cases](../model_checks/test_benchmark_kumo.py) demonstrate
a non-AML product dataset and linear model on DiskGraph, with
two epochs and a partial last batch. They are tiny interface tests, not data-scale
or model-performance evidence.

`load_fitted_state` loads a checked Kumo artifact inside the materialization
window, with a separate `state.load` timer. Complete reused state must retain its
fingerprint. `save_fitted_state` writes a new artifact outside that window;
existing destinations are rejected. Records retain the state, contract and
fitting-anchor fingerprints. Anchor comparisons hash the table binding and
Parquet bytes independently of each run's temporary path; the package's task
anchor fingerprint is retained separately. Partial input-only reuse is labelled separately.

## Catalog bindings and process comparisons

[benchmarking/kumo.py](../benchmarking/kumo.py) resolves dataset, anchor, input
and model declarations from the shared binding catalog. The enabled real-data
benchmark campaigns currently cover LI Medium / Combined (L01). HI Small's
timing/profiling campaigns are disabled; fitting-memory checks remain enabled.
The adapter verifies
the selected dataset inventory, selects physical anchor rows per split, preserves full selected-root
histories and executes the complete selected train split. This is a bounded
population, not a full-AML benchmark. Graph workloads retain v2 anchors and the
original prediction time; sequence workloads use v1. The adapter downloads no data.

Source rows/bytes include the selected source tables. Byte counts are encoded
Parquet file lengths, not decoded Arrow array sizes or physical SSD I/O.
Converted byte counts use the same file-length scope. Observations retain
source, prepared-table and anchor identities. Read timing covers checksum
verification; conversion includes its own source reads and output validation.
These enclosing measurements are not exclusive costs. Events count emitted
history occurrences, or occurrences of declared event nodes for graph-only inputs.
Tokens count valid input value/marker positions, excluding targets and parallel
field IDs. Output size counts logical tensor occurrences in inputs and targets,
including masks/layout arrays, excluding metadata and unique-storage accounting.
Shared task losses and model consumers are reused. CPU input validation remains
measured model-adaptation overhead before device transfer.

The [example configuration](../workloads/aml-benchmark.json) selects runner-visible
CUDA. Set its data directory to the runner's verified download directory and
select its registered `binding` (for example `E02`). If present, `workload` and
`family` must agree with that binding:

```sh
python -m test.kumo_pytorch.benchmark \
  --adapter test.kumo_pytorch.benchmarking.kumo:factory \
  --config test/kumo_pytorch/workloads/aml-benchmark.json \
  --output /path/to/new-run --repetitions 1 --epochs 2
python -m test.kumo_pytorch.benchmarking.campaign \
  --adapter test.kumo_pytorch.benchmarking.kumo:factory \
  --config test/kumo_pytorch/workloads/aml-benchmark.json \
  --output /path/to/new-campaign --repetitions 3 --epochs 2 --compare-fitting
```

The [campaign runner](../benchmarking/campaign.py) launches each run in a fresh
interpreter, saves its PID/log/report and stops timed-out process groups.
Optional fitting comparison gives an adapter `save_fitted_state` for each fresh
run, then `load_fitted_state` for its paired reuse run. The Kumo adapter supports
these paths; other adapters must explicitly implement and record reuse. The
campaign checks state/contract/anchor fingerprints and delivered sample counts,
then reports materialization and state-load durations separately. State loading
is nested inside materialization; those durations must not be added. OS caches
are shared across processes unless an explicit cache experiment is selected.
Missing evidence or worker failures cannot become a successful comparison.

Campaign `report.md` links every worker report; `results.json` records comparison
evidence. Incomplete benchmark coverage still exits 2; failed execution exits 1.
The nightly configures L01's CUDA profiling, index reuse, DiskGraph loader timing
and native length-planning campaigns in `benchmark-aml`, after the Linux and GPU
collector checks. It downloads checksum-pinned sources and anchors using the
existing Engine OIDC role. Real-data acceptance requires a run of the candidate
workflow; fixture validation does not establish full-population acceptance.

## Running and viewing results

From the repository root, with your adapter module importable:

```sh
python -m test.kumo_pytorch.benchmark \
  --adapter your_module:make_workload --config configuration.json \
  --output test/kumo_pytorch/.runs/my-benchmark \
  --repetitions 3 --epochs 2
```

The output directory must be new. The CLI prints per-epoch rates/latency and a
report path. Exit codes: 1 for execution failure, 2 for incomplete metric or
experiment coverage; 0 is reserved for complete coverage. Current experiments
remain incomplete even when all measured operations succeed. It is not a
performance-threshold pass. Settings are JSON; the adapter is trusted executable
Python, not a restricted configuration language.

| Artifact | Contents |
| --- | --- |
| `report.md` | Per-epoch summary, every required metric, scope/coverage, experiment gaps and failures |
| `results.json` | Structured results and observations with separate execution/coverage status |
| `observations.jsonl` | Raw stage/request timings, per-batch counts and failed-operation evidence |
| `config.json` | Requested configuration, runtime/dependency versions, Git revision/dirty state and adapter/benchmark source hashes |
| `coverage.json` | Every metric checked for every required repetition/epoch, with applicability reasons and missing evidence |

`work/` is scratch, not a report artifact: it can contain prepared datasets and
indexes. Never upload the whole run directory as a report. Batch payloads and
model tensors are not retained by the recorder. Identity/configuration records
must not include credentials. Reports do not yet establish full hardware or
transitive adapter-source provenance.

## Timing and current coverage

Each request timer stops before counting and record serialization. Nested
observer records are deferred until the request/iterator timer stops, with a
10,000-record per-request limit that fails explicitly on overflow. Those costs
remain in enclosing execution time. Terminal exhaustion contributes to window
and wait time, not produced-batch counts or latency percentiles. Rates divide
actual counts by the completed execution window. Failed windows retain partial
counts without successful rates. p50 is the median; p95 uses nearest rank.
First request, startup-through-first-batch and later-request distributions are
distinct. Later requests do not establish steady state by themselves.

For loader-only execution, batch-production time is the whole loader window.
For training, it is the disjoint iterator/request intervals; the full execution
window additionally includes model work. Model host durations do not claim
completed device time. Device work completes before starting and before stopping
the execution window, without an extra fence at every batch.

Direct runner repetitions create fresh mutable state within one OS process;
the campaign command supplies separate processes. Separate epoch records
establish first/later-epoch evidence. Fitting comparisons and explicit file-cache
conditions have their own evidence. Explicit post-warmup windows retain counts,
rates, p50/p95 latency and latency coefficient of variation. The defaults exclude
two batches and require at least four subsequent batches; CUDA runs also require
two successful utilization samples inside that window. This characterizes the
selected window without assuming statistical equilibrium. One completion fence
at the warmup boundary separates device work from the measured window.
The raw recorder streams records;
exact percentiles retain scalar request durations, and offline reporting loads
observations into memory. Neither observer overhead nor report memory is yet
calibrated for large runs.

Every selected AML operation now has a collector or an explicit contract-based
applicability declaration. Missing collectors or failed readings still fail
coverage. Benchmarking uses DiskGraph only. The Polars and GPU reference
backends and their comparison metrics were removed at the user's request.
See the [complete checklist](../PLAN.md#complete-metric-checklist).

## Internal observations

Native graph profiling is always collected, including NodeRange batches.
The runtime batch's `native_profile` dictionary exposes `sampling_ns`,
`retrieval_ns`, `rows_examined`, `rows_decoded` and `fused`. Sampling/fetch
iterators expose their operation's subset through the `native_profile` property
after exhaustion. Collection adds two timer start/stop pairs per native graph call
and row-counter updates during existing decoding; it performs no extra source
scan. Fused retrieval shares the sampling interval and must not be added to it.
Decode counts precede duplicate-output expansion. This cost has not been
isolated in a production-scale overhead measurement.

The [scoped observers](../benchmarking/kumo_observers.py) install only during a
Kumo workload and restore module/class/instance functions when it closes, including
on failure. Only the creating Python thread is observed; concurrent observer
sessions in one process are rejected. Native worker threads remain part of the
underlying call. No engine source files are edited by these wrappers.

Collected boundaries include input/target fitting, vocabulary/statistics fitting,
the generic BPE fit path, Python row merge, scalar encoding, token construction,
masking, labels, column assembly, batch planning and enclosing post-processing.
These are timings of the named functions, not exclusive CPU costs: column assembly
includes nested encoding; post-processing may include graph calls. Each record
retains its function and nesting identity. Unsupported execution paths remain gaps.

DiskGraph index build/reopen measurements surround actual ensure/validation
calls, not historical manifest times. Logical index bytes count unique regular-file
inodes and exclude symlinks. Fresh source profiles preserve all backend fields.
Temporal-query profiles supply native traversal, decode/select and aggregation
elapsed times, decoded rows and candidate frame rows before filtering/selection.
Graph APIs expose completed native sampling/retrieval durations and row counts,
including asynchronous and fused-runtime paths. Pending/failed calls do not supply
completed measurements. Fused sampling/retrieval share one explicitly labelled
interval and must not be added; native elapsed can include output backpressure.
Graph examined rows count adjacency source-row visits plus decoded feature-row
occurrences. Index-pruned edges are excluded. These are declared operation
counters, not physical storage reads or unique records. Full native snapshots and
measurement ownership IDs are retained. Older native APIs lacking these fields
produce unavailable measurements, never inferred Python or Rust durations.

## Explicit file-page-cache experiments

[FilePageCache](../benchmarking/file_cache.py) is a `before_epoch` callback for
an explicit list of owned regular files. AML's `cache_mode: "cold-warm"` selects
its index files before the first epoch and alternates conditions on that fixed
selection before complete epochs. New files created by iteration, including
native length caches, appear as `unconditioned_new_files` in later snapshots;
they are outside this file-page comparison. Length planning and second-epoch
reuse still run and are checked by the native profiler. The callback never
clears machine-wide caches. At the idle epoch boundary, cold preparation syncs
the selected files, releases this worker's read-only file mappings with
`MADV_DONTNEED`, and requests advisory eviction. Writable mappings are rejected;
anonymous mappings and unselected files are untouched. Warm preparation reads
the selected files. The entire selection is then
checked with `mincore`. An unmet condition fails and retains its observations;
advisory eviction alone is not proof of coldness. See the Linux contracts for
[advice](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html) and
[residency snapshots](https://man7.org/linux/man-pages/man2/mincore.2.html).

Reports record file identities, total/resident pages, setup boundaries and
completed epoch scopes. A cold/warm pair requires unchanged file identities,
sizes and modification times. Missing or changed selected files still fail
conditioning or leave the comparison incomplete.
This describes selected Linux file pages at the measurement boundary, not
application buffers, metadata, device caches or an enduring residency guarantee.
Epoch/model state can also change, so the pair does not isolate a causal cache
speedup. Source ingestion is outside this index-cache experiment.

## Linux stage peak memory

`--memory-cgroup /path/to/workload-cgroup` enables the
[peak collector](../benchmarking/memory.py) for an existing, dedicated Linux
cgroup v2 containing the benchmark and its children. It checks membership and
rejects unrelated live processes. It does not create groups, move processes,
change limits or clear caches. Provision the dedicated group outside the runner.

The default `--memory-method kernel` resets and reads `memory.peak` through the same open descriptor,
following the [kernel contract](https://docs.kernel.org/admin-guide/cgroup-v2.html).
This measures memory-controller charges for the workload and descendant groups,
including file cache and kernel charges; it is not RSS. Live allocations at reset
and observation overhead remain included. Records preserve reset/read boundaries
and scope. Ingestion and materialization have separate intervals; batch production
retains iterator/request interval peaks, whose maximum is reported separately
from training. It never sums independent process peaks.

Failed/reset-denied/unsupported kernel collection produces explicit unavailable/error
measurements, leaving required coverage incomplete. It never silently falls back
to sampling. Tests cover descriptor/control-flow behavior and
the real Linux validator checks transient parent allocations, independent stage
resets and allocations from children that exit before the stage ends:

```sh
python -m test.kumo_pytorch.benchmarking.validate_memory \
  --cgroup /path/to/dedicated-cgroup --output memory-validation.json
```

This passed in an isolated Linux 7.0.14 container. The full suite's `host-metrics`
job provisions a dedicated private container cgroup and validates stage peaks
and warm/cold/warm file residency. Only that temporary container has mount
capability; it has no network or Docker socket. Unsupported kernel peak reset
fails explicitly. Both validators passed locally in that Linux environment;
the dedicated Linux CI validation also passed. `--isolate-memory` runs every
campaign worker
through the [cgroup supervisor](../benchmarking/cgroup_run.py), which provisions
a dedicated group, moves the child before exec and drops to the runner UID/GID.
The supervisor requires passwordless sudo and a writable cgroup v2 parent with
its memory controller already enabled. It changes no parent controller or host
limits. Timeout cleanup kills only that worker’s descendants and removes its
group. The [isolation validator](../benchmarking/validate_isolation.py) passed
on Linux with an unprivileged worker, complete benchmark stages, two epochs,
transient children and timeout cleanup. The nightly repeats this validation.
Container validation is collector evidence,
not a workload benchmark or a requirement to execute on a developer's machine.

The initial RTX campaign host reported Linux `6.8.0-147-generic`. Its
[`memory.peak` interface is read-only](https://docs.kernel.org/6.8/admin-guide/cgroup-v2.html),
so all three host-stage peak measurements were unavailable despite successful
worker cgroup creation. The collector requires the
[per-descriptor reset interface documented in Linux 6.12](https://docs.kernel.org/6.12/admin-guide/cgroup-v2.html)
or a compatible backport, plus delegated write access. A newer container image
does not change the host kernel. This requirement applies to exact kernel
peaks; the nightly now explicitly selects sampled peaks as described below.


### Explicit sampled host peaks

The nightly uses `--memory-method sampled --memory-interval 0.01`, following the
user-approved sampled-peak approach. This reads `memory.current` for the isolated
worker cgroup, including descendants, native allocations, file-cache and kernel
charges. Its scope differs from Arena's summed process RSS; it does not double
count shared pages by summing process RSS. No kernel reset is needed.

Each ingestion, materialization and iterator/request interval gets independent
boundary and periodic reads. Reports label the maximum **sampled**, not an exact
kernel high-water mark, and retain the interval, sample count, largest gap, stage
boundaries and raw readings. Allocations between reads can be missed; short
stages may have boundary samples only. Boundary reads include observation
overhead. The method is saved in every worker/campaign configuration; changing
it changes the measurement semantics and should not be treated as a performance
change in the package. Read errors, sample-capacity overflow and failed stages
remain unavailable/errors and fail coverage.

The GPU campaign job also runs the memory validator in a separate worker cgroup
on that same host. Sampled validation holds allocations for 0.1 seconds to check
that the sampler captures parent and child allocations and starts a new peak
for each stage. It does not claim to validate between-sample spikes. The artifact
includes `sampled-memory-validation.json`. The separate exact-peak Linux tests
remain enabled on their compatible runner.

[CI run 43864367740](https://github.com/NVIDIA/kumo-engine/actions/runs/43864367740)
validated sampled parent/child allocations on Linux 6.8 and all three sequence
campaigns. Every sequence worker covered 73/73 metric obligations. The target
interval was 10 ms; actual gaps reached 320.18 ms and are retained in reports.
This is measured sampled coverage, not an exact high-water guarantee. See the
[measurement acceptance rules](../PLAN.md#measurement-acceptance).

## GPU collection and nightly runner

CUDA workloads enable [GPU collectors](../benchmarking/gpu.py) automatically.
The adapter selects a runner-visible device and supplies completion callbacks.
Collectors reset allocated/reserved peaks before each execution window and read
them after completion. Native PyTorch allocator statistics include live tensors
at reset. Other allocator semantics remain explicitly unsupported.

Host-to-device copies enclosed by `transfer.host` receive CUDA event pairs on
the current stream. Events resolve after window completion, with no per-batch
synchronization. Stream changes, unresolved events and capacity limits fail
measurement rather than inventing a value. NVML samples device-wide utilization
using the selected CUDA device's UUID, never an assumed matching ordinal.
Reports retain every sample, failures, interval and coverage gaps. These samples
are not a utilization time integral. `--gpu-interval` defaults to one second.
The NVML Python binding is a test-runner dependency, not a package dependency.

The full suite's `gpu-checks` job uses the documented engine
[RTX PRO 6000 group](../../../docs/development/engine-pr-ci.md) and locked
`cu130` environment. PR, scheduled and manual plans now require the same GPU
coverage. Direct PR admission remains an open infrastructure prerequisite:
the documented NVIDIA runner rejects `pull_request` jobs. The user chose to
resolve access separately; these jobs are never skipped as a passing fallback.
No local device, hostname or UUID is configured as a prerequisite. Runtime hardware
records describe the machine that actually ran the benchmark.

The job runs `python -m test.kumo_pytorch.benchmarking.validate_gpu --output DIR`
on real CUDA training, requires transfer/utilization/peak evidence in both
epochs, and checks that allocated peaks reset. Missing CUDA/NVML fails; there is
no CPU fallback or skip. The job summary and `kumo-pytorch-benchmark-gpu` artifact
retain the report and observations. CI run 43857600971 passed this collector
validation and all three existing CUDA package tests on the RTX runner.
Collector acceptance is separate from full PDF coverage;
the fixture's overall benchmark report correctly remains incomplete.

## Validation

Native Kumo adapter checks run in the existing model-check job. The PR trigger is unchanged.

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_benchmark_kumo.py -q
```

The command requires compatible Kumo/DiskGraph and PyTorch dependencies.
No datasets or remote services are required by these fixtures. CPU validation
does not establish GPU measurement correctness.

## Suite execution and acceptance

[The suite command](../benchmarking/suite.py) accepts a JSON list of
`{name, adapter, config}` cases; it has no dataset/model dispatch. Each case runs
fresh/reused fitted-state pairs in independent interpreters. The AML nightly
constructs this list from the existing workload catalog, uses 512 anchors per
split with full histories, and performs three pairs of two epochs per case.
At batch size four this provides 128 batches per epoch and 126 after warmup.
The previous 64-anchor selection produced later nuFormer windows of about
0.1 seconds, too short for two in-window GPU samples at the 0.1-second polling
interval. The larger selection extends actual workload execution; the sampling
interval and required-evidence checks are unchanged.
This is a bounded population experiment, not full-AML population acceptance.

The suite and campaign return 0 only when all required metric coverage and
experiment evidence are complete, 2 for missing evidence and 1 for execution
failure. Individual worker reports can remain incomplete because the enclosing
campaign owns independent repetitions and paired fitting comparisons.

The Actions summary links to the artifact workflow: download
`kumo-pytorch-aml-benchmarks`, open the root `report.md`, then follow case
and worker reports. Artifacts contain JSON/JSONL, Markdown and logs, excluding
source/index payloads and persisted fitting artifacts. No result is uploaded to
S3. PR and nightly select the same CUDA jobs; direct PR runner admission is still unresolved. A candidate workflow must be published to an internal branch
before GitHub can execute it. The [roadmap](../PLAN.md#remaining-acceptance-work) tracks remaining acceptance.
Historical decimal failures do not describe the current integrated package.

## DiskGraph loader timing

Select `backend: "diskgraph"`, `training: false`, `device: "cpu"` and
`cache_mode: "uncontrolled"` for loader-only timing. L01 enables this campaign;
HI Small's timing campaigns remain disabled. Polars and GPU reference backends,
their adapters and their dedicated metric requirements have been removed.
CUDA model training and GPU memory/utilization collection remain enabled.

`--measurement-profile timing` selects loader timings and sample counts/rates
from the metric catalog. It rejects model training and disables memory/GPU
utilization collectors. The default `full` profile retains all DiskGraph and
training measurements. Both profiles require independent repetitions, complete
first/later epochs, explicit post-warmup windows and fresh/reused fitted-state
evidence. Timing-only runs do not claim controlled cold/warm caches.

```sh
python -m test.kumo_pytorch.benchmarking.suite \
  --config /path/to/loader-cases.json --output /path/to/new-timings \
  --measurement-profile timing --repetitions 3 --epochs 2
```

The root `report.md` presents loader campaigns side by side. `timings.json`
retains individual values, medians and ranges, separated by fresh/reused fitting
and epoch. Only workers that completed execution contribute to summary values;
failed runs retain errors and partial observations in linked reports and still
fail the suite. Source/index payloads are excluded from artifacts.

## One index build and two reopens

Run the generic suite with `--index-lifecycle --repetitions 3` to execute three
separate interpreters per case: build, reopen, reopen. Each measures ingestion,
materialization and complete selected batch epochs. Fitting is fresh in all
three runs; the existing fitted-state reuse comparison remains a separate campaign.
Adapters opting into this mode accept `index_dir`, `index_action` (`build` or
`reopen`) and `prepared_dir`, and emit exactly one matching `index.build` or
`index.reopen` timing. Missing timings or unexpected rebuilds fail the campaign.
The AML adapter refuses an existing build destination or a missing reopen
index and opens reused indexes with ingestion forbidden. Shared prepared sources
keep the dataset identity fixed. This protocol is independent of dataset/model.

Index reuse is distinct from OS page-cache warmth. The existing per-epoch cold/
warm file-page experiments and their recorded residency checks remain enabled;
no claim that every reopened page is resident is made. The nightly adds this
experiment for every declared AML combination. Results appear in the job summary
and the `kumo-pytorch-aml-benchmarks` artifact under `index/<case>`, including
per-worker stage metrics and build/reopen evidence. Index and source payloads
are excluded from uploads.

## Anonymous-memory peaks

Full-profile reports require `ingestion.peak_anonymous_bytes`,
`materialization.peak_anonymous_bytes` and `batch_production.peak_anonymous_bytes`
alongside the existing total host-memory peaks. These sample the cgroup v2
`memory.stat` `anon` counter, including descendants but excluding file and shmem
charges. Boundary and periodic readings, intervals and gaps are retained.
Anonymous peaks are explicitly sampled even when total peaks use `memory.peak`;
counter refresh can lag allocations and samples can miss short-lived peaks.
Total and anonymous maxima are independent, not a simultaneous decomposition.
A missing/malformed counter remains unavailable and fails measurement completeness.
The Linux collector validation exercises held parent/child anonymous allocations
and independent stage peaks. Local mocked collectors are not live Linux evidence.
The fitting-memory experiment also retains anonymous peaks and size medians,
without introducing a growth threshold.

## Parallel nightly execution

The full suite derives the same PR/nightly selection from
`workloads/bindings.json`; see the [organization](../README.md#organization-and-adding-bindings):

| Experiment | Independent jobs | Work retained within each job |
| --- | ---: | --- |
| Full-profile CUDA training | 1 | L01: three fresh/reused fitting pairs, two epochs per worker, cold/warm caches |
| Index lifecycle | 1 | L01: one build followed by two reopens of the same index, two epochs per worker |
| Loader timing | 1 | L01 / DiskGraph: three fresh/reused pairs, two epochs |
| Native length planning/cache | 1 | L01: native planning and second-epoch cache reuse |
| Fitting memory | 15 | HI Small's five bindings times three repetitions; both 25k/400k sizes use the same index within each repetition |
| Population acceptance | 15 | HI Small: five smoke/full pairs; LI Medium: four full runs and L01 smoke |
| CPU training acceptance | 9 | Both datasets use the same 100-update runner and report schema |
| LI lifecycle and histories | 2 | L01: fresh-process lifecycle and independent source-history checks |

HI Small no longer runs model profiling, index-reuse timing or loader timings
on any backend. Its other checks, including fitting memory, are unchanged.

Measurements run sequentially within their experiment on a dedicated runner;
parallel timing jobs do not share a GPU. All timing backends retain the same
runner group, limits and thread settings. CPU population and real-AML lifecycle
checks run independently of profiling. Existing package tests, conformance,
retail lifecycle, GPU collectors and five-model CPU training acceptance remain.
Matrix jobs use `fail-fast: false`, preserving other evidence after failures.

Each shard uploads a uniquely named `kumo-pytorch-aml-parallel-*` artifact. The
combined-report job downloads them, rejects missing or mismatched cases and
repetitions, and retains one `kumo-pytorch-campaign-results` artifact. Its root
`report.md` links every experiment, and `timings.json`/the timing tables combine
the enabled loader timing runs. Payloads, indexes and fitted-state files are excluded.
Training/population shards use `kumo-pytorch-dataset-*` artifacts and join the
same summary. Existing lifecycle/history scenarios retain separate artifacts.

Roughly 20 minutes end to end is a target, not a timeout or permission to reduce
coverage. This layout removes accumulated campaign/repetition time, but runner
queueing and the slowest individual workload still determine completion. The
previous combined 400k fit took about 21 minutes by itself; each repetition now
has its own runner. Cold native builds and full-population execution can also
exceed the target. A new CI run is required to establish the resulting duration.
