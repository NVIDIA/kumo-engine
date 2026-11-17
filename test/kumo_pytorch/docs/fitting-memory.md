# Fitting-memory scalability

Criterion 3 compares fresh materialization at **25,000 and 400,000 training
anchors**. The nightly/manual `aml-fitting-memory` job runs all five AML input
contracts on one Linux CPU runner. Tiny harness tests also run through the
existing PR jobs. No model training or batch iteration is involved.

## Experiment

Sources, contract, history capacity, encoder cutoff, native threads and index
are fixed within each contract's comparison. The smaller training population
is a verified prefix of the larger one. Source identities and anchor-file hashes
are retained; insufficient anchors fail instead of duplicating rows. Graph
contracts retain the v2 anchor cutoff. Preparation keeps original targets,
including null targets, and excludes validation/test anchors from fitting.

One worker builds the index. Each measured worker then reopens it with
`ingest="never"` and calls `Plan.materialize` without fitted-state reuse.
The measured boundary includes native anchor binding, input/target fitting and
task setup; imports, source preparation, anchor projection and index build/reopen
are outside it. All five contracts use the full prepared AML sources.

Each size runs three times, in fresh processes and separate Linux cgroup v2
groups. Order alternates between repetitions. Cache state is **uncontrolled**;
the index and any column-statistics caches persist. This is not evidence for
the separate cold/warm-cache requirement.

The default measurement is explicitly labelled **sampled cgroup memory.current**,
with 50 ms sampling and boundary samples. It includes the worker's native
allocations, descendants, Python baseline, measurement overhead and file-cache/
kernel charges attributed to that cgroup. Shared cached pages charged elsewhere
are not included. Samples can miss short peaks. Reports retain sample counts,
gaps, stage-start bytes and raw observations; they do not claim an exact kernel
high-water mark. Separate labelled samples of `memory.stat` `anon` report
anonymous-memory peaks and per-size medians, excluding file/shmem charges;
counter refresh can lag allocations. Explicit `--memory-method kernel`
requires resettable kernel peaks and fails if unavailable; there is no fallback.

Reports retain the median absolute stage peak at each size and their growth
ratio, with no pass/fail growth limit. A passed status means measurements are
complete and comparable, not that memory growth is bounded. Stage-start memory
is diagnostic, not subtracted. Every repetition is required;
worker errors, unavailable measurements, changed specifications, reused cgroups
or mismatched measurement methods fail the experiment. Unsupported package
behavior remains an ordinary failure.

## Execution and results

From the repository root on Linux with the Engine test environment, a delegated
memory controller at `/sys/fs/cgroup`, and passwordless sudo for the existing
worker supervisor:

```sh
python -m test.kumo_pytorch.aml_fitting_memory \
  --data-dir /path/to/pinned-aml --output /path/to/new-fitting-run \
  --repetitions 3 --memory-method sampled
```

Results appear in the Actions job summary and the
`kumo-pytorch-aml-fitting-memory` artifact: `report.md`, structured aggregate and
worker reports, per-repetition observations, logs and input specifications.
Payloads and indexes are excluded. Reports retain partial progress on failures.

The dataset-independent runner is
`python -m test.kumo_pytorch.benchmarking.fitting_memory --spec SPEC --output OUT`.
The specification supplies a serialized Kumo contract and fingerprint, Parquet
source paths, a new index path, two anchor-file paths/checksums keyed by count,
`AnchorTable` column bindings, thread count, fitting split and common encoder
cutoff. AML's `make_spec` is one adapter; the runner has no dataset/model dispatch.

Local tiny-fixture checks validate preparation, native fitting and report gates
with simulated collectors. They are not live Linux or 25k/400k acceptance.
Real-data CI measurements remain outstanding until the new job is executed
on this revision.
