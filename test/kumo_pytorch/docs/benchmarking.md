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
contract, dataset, backend, fitting anchors, reusable epoch anchors, materialize
options and batching options. Select `fit_split` explicitly as required by Kumo.
The runner owns the epoch argument. Anchors may be tables or reusable request
collections; do not supply an exhaustible generator as the repeated population.

An optional `consumer_factory(task, recorder)` creates any model/optimizer and
returns its step callback. It is not limited to the copied AML consumers.
Use the `inputs` callback to perform and measure source/anchor work rather than
loading data in the outer factory. The adapter closes task and plan contexts on
success or failure. Prepare and materialize timers remain enclosing calls;
internal/native phases have separate scoped instrumentation described below.

The native [integration cases](../model_checks/test_benchmark_kumo.py) demonstrate
a non-AML product dataset and linear model on both InMemory and DiskGraph, with
two epochs and a partial last batch. They are tiny interface tests, not data-scale
or model-performance evidence.

`load_fitted_state` loads a checked Kumo artifact inside the materialization
window, with a separate `state.load` timer. Complete reused state must retain its
fingerprint. `save_fitted_state` writes a new artifact outside that window;
existing destinations are rejected. Records retain the state, contract and
fitting-anchor fingerprints. Partial input-only reuse is labelled separately.

## AML binding and process comparisons

[aml_benchmark.py](../aml_benchmark.py) binds all five registered AML
workload/model combinations to this interface. It verifies downloaded HI-small
inputs, selects physical anchor rows per split, preserves full selected-root
histories and executes the complete selected train split. This is a bounded
population, not a full-AML benchmark. Graph workloads retain v2 anchors and the
original prediction time; sequence workloads use v1. The adapter downloads no data.

Source rows/bytes include transactions and account bindings. Observations retain
source, prepared-table and anchor identities. Read timing covers checksum
verification; conversion includes its own source reads and output validation.
These enclosing measurements are not exclusive costs. Events count emitted
history occurrences, or transaction-node occurrences for graph-only inputs.
Tokens count valid input value/marker positions, excluding targets and parallel
field IDs. Output size counts logical tensor occurrences in inputs and targets,
including masks/layout arrays, excluding metadata and unique-storage accounting.
Existing AML losses and model consumers are reused. CPU input validation remains
measured model-adaptation overhead before device transfer.

The [example configuration](../workloads/aml-benchmark.json) selects runner-visible
CUDA. Set its data directory to the runner's verified download directory and
select another registered `workload`/`family` as needed:

```sh
python -m test.kumo_pytorch.benchmark \
  --adapter test.kumo_pytorch.aml_benchmark:factory \
  --config test/kumo_pytorch/workloads/aml-benchmark.json \
  --output /path/to/new-run --repetitions 1 --epochs 2
python -m test.kumo_pytorch.benchmarking.campaign \
  --adapter test.kumo_pytorch.aml_benchmark:factory \
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
The real-data adapter and campaigns have not run on full AML or the GPU runner.
The nightly now configures all five AML CUDA campaigns in `benchmark-aml`,
after the Linux and GPU collector checks. It downloads checksum-pinned sources
and anchors using the existing Engine OIDC role. The CI role’s access and actual
GPU campaign execution still require a run of this candidate workflow.

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
coverage. Equivalent reference-engine performance comparison remains deferred,
as requested; a DiskGraph report does not claim reference-engine acceptance.
See the [complete checklist](../PLAN.md#complete-metric-checklist).

## Internal observations

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
its index files and alternates conditions before complete epochs. It never
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
completed epoch scopes. A cold/warm pair requires the same file identities.
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

This passed in an isolated Linux 7.0.14 container. The nightly's `benchmark-memory`
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
[results](../PLAN.md#gpu-and-sampled-memory-validation).

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

The nightly workflow's `benchmark-gpu` job uses the documented engine
[RTX PRO 6000 group](../../../docs/development/engine-pr-ci.md) and locked
`cu130` environment. It runs on schedules and manually selected internal branches; NVIDIA runners
do not admit `pull_request` jobs. The existing CPU PR trigger remains. No local
device, hostname or UUID is configured as a prerequisite. Runtime hardware
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

Dependency-free runner/report checks join the existing nightly tooling job;
Kumo adapter checks join its existing model-check job. The PR trigger is unchanged.

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/checks/test_benchmark.py -q
python -m pytest -c test/kumo_pytorch/pytest.ini --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_benchmark_kumo.py -q
```

The second command requires compatible Kumo/DiskGraph and PyTorch dependencies.
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
`kumo-pytorch-aml-benchmarks`, open `aml-benchmarks/report.md`, then follow case
and worker reports. Artifacts contain JSON/JSONL, Markdown and logs, excluding
source/index payloads and persisted fitting artifacts. No result is uploaded to
S3. The CPU PR trigger remains active; real-data CUDA jobs require a schedule
or manual dispatch. A candidate workflow must be published to an internal branch
before GitHub can execute it. The [collective CI review](../PLAN.md#collective-ci-review)
records actual execution, expected decimal failures and remaining measurements.

## Polars loader timing comparison

The test-only [Polars engine](../reference_engine/README.md) uses the same Kumo
contracts, prepared source tables and selected anchors as DiskGraph. The nightly
adds CPU loader-only campaigns for each declared AML input family on both
backends, independently of the existing CUDA model/training campaigns. Graph and
combined keep v2 anchors. Batch size, thread count and source/anchor selection are
shared; sampled graphs and individual batches need not be identical. No numerical
parity or model-quality assertion is made across engines.

Set `backend` to `polars` or `diskgraph`, `training` to `false`, `device` to `cpu`
and `cache_mode` to `uncontrolled` in the AML adapter config. Polars does not
accept a DiskGraph `index_dir`. Start workers with `POLARS_MAX_THREADS` matching
`threads`; the nightly installs `polars==2.1.0` as a test-only dependency.

The generic `--measurement-profile timing` option selects loader timings and
sample counts/rates from the metric catalog. It rejects model training and GPU
execution and disables memory collectors. The default `full` profile and its
73-metric obligations remain unchanged. Each timing campaign still requires
independent repetitions, complete first/later epochs, explicit post-warmup
windows and fresh/reused fitted-state evidence. Cache-control experiments are
outside this comparison: neither a new interpreter nor resident Polars tables
establish a cold-cache claim.

`reference.load` times the actual engine `open` call. `reference.fit` encloses
materialization, `reference.plan` creates the batch iterator and
`reference.batch` encloses its requests. These overlap their enclosing ingestion,
materialization and batch-production timers and must not be added. DiskGraph
index construction/reopen and Polars resident-table loading are reported in
separate columns, with unsupported breakdowns visible in worker reports.

```sh
POLARS_MAX_THREADS=4 python -m test.kumo_pytorch.benchmarking.suite \
  --config /path/to/paired-loader-cases.json --output /path/to/new-timings \
  --measurement-profile timing --repetitions 3 --epochs 2
```

Cases retain the standard `name`, `adapter`, `config` schema. Each backend gets
its own case and fresh process/state artifacts. Both use the same runner host;
results include runtime/dependency pins, source identities and actual emitted
counts. Source read/conversion and preparation are timed anew in each worker.

The Actions job summary and `kumo-pytorch-aml-benchmarks` artifact include
`aml-timings/report.md`, which puts loader campaigns side by side. `timings.json`
retains individual values, medians and ranges, separated by fresh/reused fitting
and epoch. Only workers that completed execution contribute to summary values;
failed runs keep their errors and partial observations in linked reports and
still fail the suite. The existing DiskGraph decimal failures remain failures.
