# PR #424 production readiness — October 30 investigation, November 3 qualification

## November 4 integration and evidence scope

The dependency audit found that the imported Unix benchmark profiler used
`pprof 0.13.0`, affected by [RUSTSEC-2024-0408](https://rustsec.org/advisories/RUSTSEC-2024-0408.html).
The optional `--cpu-profile` path calls the affected report builder. Aligning
Engine's profiler with ingest's existing `pprof 0.14.1` removes that affected
version while preserving Unix gating and protobuf output. This correction
requires fresh candidate qualification; earlier runs do not qualify it.

The candidate integrates Engine `main` at
`502adef65ab0011545eeeb242610412c6ed8e7e3`. This adds Arrow/Parquet 59.3.2,
PyO3 0.29.3, the saved-index-compatible epserde security backport and changes
to training, prediction and model persistence. The November 3 results below
qualify their recorded `88fd6042` or `d4088a12` revisions; they do not qualify
this combined revision. Qualification of the integration requires fresh
native and installed-wheel checks, all task/backend CPU and GPU runs, and
platform build/runtime checks.

The refresh exposed an inherited Windows build failure: snapshot checksums
imported Unix `FileExt` unconditionally. [PR #198](https://github.com/NVIDIA/kumo-engine/pull/198)
uses independent per-range seek/read handles while preserving checksum bytes,
ordering and errors. [PR #197](https://github.com/NVIDIA/kumo-engine/pull/197)
classifies 42 copied epserde files by exact path, retains upstream attribution,
and completes the header of one NVIDIA-authored regression test; new owned
files remain checked. The integrated prerequisite order is #197, #198, #196,
then #169. These corrections require qualification of the combined revision;
the earlier results below retain their recorded revision scope.

The maintained [PR #169 change map](https://github.com/NVIDIA/kumo-engine/pull/169#issuecomment-6611981159)
and PR description record the current revision and its completed CI evidence.
[PR #196](https://github.com/NVIDIA/kumo-engine/pull/196) is a separate baseline
prerequisite correcting BigQuery test eligibility: explicit
`KUMO_RUN_GCP_TESTS=0` must prevent credential setup when the workflow disables
GCP authentication. This preserves enabled-mode credential failures and does
not provision cloud access.

## November 3 qualified candidate

The reproduced correctness and portability defects are fixed. At candidate
`88fd60424ea8f3617327f635531ef82823a537c3`, native Linux, installed macOS,
all 24 real CPU task/backend fits and all 24 GPU task/backend fits pass.
No unresolved production-code defect is established by the completed checks.
The candidate is ready for controlled Linux opt-in under the documented
resource/capability requirements. Keep the PR draft until general CI clears;
this report does not claim the entire CI matrix is green. The deployment
default remains `DG_EDGE_BACKEND=mmap`.

## November 3 current status

The candidate incorporates Engine `main` through `f8ba7e17`, preserving the
extracted native runtime, timestamp clocks, ingestion portability and
experimental Windows/WSL qualification. The merged native baseline passes
930 tests and related Core checks pass 158 tests. The results below test the
final source head `88fd6042`, including the Windows portability adaptation.

| Qualification | Observed result |
| --- | --- |
| [Final-head Linux native and Python CI](https://github.com/NVIDIA/kumo-engine/actions/runs/41700694125) | Ordinary Rust, all-target Clippy with warnings denied, candidate-wheel build/import and installed Python checks pass. The strict native gate executes all ten I/O/cache/error/fork/provider cases without skips; actual submissions and cache hits are asserted. |
| [Real CPU task/backend matrix](https://github.com/NVIDIA/kumo-engine/actions/runs/41700694125/job/126147535511) | All 24 fits pass: eight task types across mmap, buffered and cached direct I/O (eight pytest cases, 325.78 seconds). Non-link fits execute two optimizer steps; link fits execute four. |
| [Installed macOS ARM64 wheel](https://github.com/NVIDIA/kumo-engine/actions/runs/41700694125/job/126143730551) | Actual `cp311-abi3-macosx_14_0_arm64` wheel builds and installs; all 11 default/mmap ingestion, sampling, feature and configuration checks pass, with no skips. |
| [Finished-wheel direct-I/O GPU tasks](https://github.com/NVIDIA/kumo-engine/actions/runs/41700700752) | All eight task types pass fit, repeat prediction and save/reload prediction: 1,600 optimizer steps and 1,280 observed CUDA training batches. Maximum reload absolute difference is 0.002417, below the declared 0.005 tolerance. The job requests `direct`, enables a 1 GiB per-process cache and logs 696 real direct-cache initializations. |
| [Finished-wheel buffered GPU tasks](https://github.com/NVIDIA/kumo-engine/actions/runs/41701343545) | All eight tasks pass fit, repeat and save/reload prediction: 1,660 optimizer steps and 1,360 observed CUDA training batches. Worst reload absolute difference is 0.0001518726, below 0.005. Both explicit GPU runs cover 8,930 entities with unchanged prediction decisions. |
| [Finished-wheel default-mmap GPU control](https://github.com/NVIDIA/kumo-engine/actions/runs/41700708272/job/126160089452) | All eight tasks pass: 1,640 optimizer steps, 1,320 observed CUDA training batches and 8,930 entities with unchanged decisions. Worst reload absolute difference is 0.000175655, below 0.005. |
| [Full platform qualification](https://github.com/NVIDIA/kumo-engine/actions/runs/41700708272) | All four wheel producers pass. Windows MSVC passes wheel/payload verification, ten ingestion path-safety tests and one unsupported-profiling rejection test. All twelve installed CPU qualifications pass: native Windows, WSL and Linux ARM64 on Python 3.11–3.14. Each completes four optimizer steps and 100 finite predictions with save/reload parity. Four hosted platform requirements jobs remain queued; their exact cross-resolver commands pass locally against the same verified wheel artifacts (144 combinations). |

The CPU matrix covers binary, multiclass and multilabel classification,
multilabel ranking, regression, forecasting, static links and temporal links.
It validates complete finite predictions for eight canonical entities, all
classes/RHS items and all 12 forecast horizons, plus own-model repeated and
reloaded predictions. Backend parity uses identical saved model weights,
snapshot and entity/time inputs at absolute tolerance `1e-6`, relative
tolerance zero. Each backend/task runs in a fresh process with asserted
candidate native and Core/SDK import paths. Training is real and uses the
normal Core automatic mode selection; no forced `fused=False` workaround is
applied by the harness.

The model matrix does not expose per-fit native I/O counters. Those counters
are Rust-only and process-local; the separate mandatory native checks establish
explicit submissions, cache hits and byte parity. Source dispatch propagates
explicit I/O errors without retrying mmap. Projection/parent/train-table parity
is covered by the native Python companion checks rather than inferred from
model prediction parity. The GPU reload tolerance is intentionally distinct
from the tighter CPU shared-model backend tolerance. Across all three backends
the GPU checks execute 24 real fits, 4,900
optimizer steps and 3,960 observed CUDA training batches, with unchanged
prediction decisions. The runs execute different step counts; their durations
do not establish a backend performance comparison.

## Exact wheel and resolver provenance

The four completed platform producers in run `41700708272` are tied to
`88fd60424ea8f3617327f635531ef82823a537c3` by the GitHub artifact API. Downloaded
archive SHA-256 values were checked against that API before extracting the
single `1.0.0.dev0+88fd604` wheel. Extracted wheel SHA-256 values are:

| Target | Artifact ID | Wheel SHA-256 |
| --- | --- | --- |
| Linux x86-64 | 13264346687 | `f3de80405b5fa894c33c2c3374858787c64e46515d34c4a3b654105dbae02318` |
| Linux ARM64 | 13264286253 | `e164a3f5e8c146b1e0f5203becd58c20c74da4afc4dfcfd191bfc00a2eb40c19` |
| macOS ARM64 | 13264431683 | `4cb4b199d79056436c3b199c55ab0b9f5846fd5f4dcc5b78de50f31ed5d742ae` |
| Windows x86-64 | 13265018533 | `c43efe7f1ef9f47ccbe6278d90fdc4443c1d1254206fb8002c4462d9da430ead` |

The unmodified `packages/kumo-engine/scripts/check_pypi_requirements.py` passes
locally for each actual wheel and its exact CI target: Linux
`x86_64-manylinux_2_34` and `aarch64-manylinux_2_34`, macOS
`aarch64-apple-darwin`, and Windows `x86_64-pc-windows-msvc`. It resolves the
base wheel and each of eight extras for Python 3.11, 3.12, 3.13 and 3.14:
**144 successful resolutions**, using uv 0.12.23 and PyPI. It installs no
packages and executes no wheel runtime. This reproduces the four queued CI
resolver commands; native operating-system proof comes from the separate
hosted qualifications above.

The local evidence bundle is `/tmp/dg424-requirements-88fd/verification.md`,
with `artifact-provenance.json`, per-platform logs and downloaded wheels in the
same directory. To reproduce after downloading a producer artifact and
verifying its wheel hash, run from the repository root:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 python3 \
  packages/kumo-engine/scripts/check_pypi_requirements.py \
  /path/to/kumo_engine-1.0.0.dev0+88fd604-cp311-abi3-PLATFORM.whl \
  --platform TARGET
```

## Implemented corrections

- Failed plans relinquish cache reservations exactly once; error, cancellation
  and replacement interleavings have executable regressions.
- Runtime/cache state is process-owned. Fork during initialization, locked
  cache state, populated cache and inherited in-flight work are checked. Native
  RuntimeIndex/iterator access rejects inherited owners before touching their
  thread state; Core reopens a cached index when a worker PID changes. The
  actual warmed-parent dataset/DataLoader/runtime-adapter checks pass.
- Native automatic fusion checks backend capability, including zero-hop
  requests. Core automatic training selects standalone mode for explicit I/O,
  preserving mmap's existing worker/thread policy. Explicit unsupported fused
  requests and invalid backend text/UTF-8 return checked errors; 118 Core
  automatic-configuration cases pass.
- Linux dependencies/syscalls are target-gated. Supported macOS and newly
  merged experimental Windows preserve default mmap; explicit non-Linux I/O
  returns unsupported-operation errors. Unix page advice and profiling remain
  Unix-only. The existing Windows/WSL release boundary is unchanged.
- Provider benchmark backends checksum the same literal page bytes, including
  duplicate requests. The mandatory native parity regression passes.
- Unused profiler dependencies that failed existing notice policy were
  removed; generated artifact/license inventories were refreshed without
  changing licensing policy.

Engine's existing sorted `rand::seq::index::sample` behavior remains the seeded
selection contract. The source proposal's Floyd replacement is omitted because
it changes sampled neighborhoods for the same seed. Imported source timings
therefore do not establish performance for this candidate.

## Current action items

1. Keep the PR draft until outstanding general CI jobs and the four hosted
   requirements jobs complete. Broader Ubuntu 24 jobs are queued/unallocated;
   the scheduler cause is unknown. There is no observed failure in the port's
   completed qualification gates. A separate base-main `f8ba7e17` connector
   control passes 1,911 tests and fails two tests requiring restricted GCP
   BigQuery identities; those failures are outside this change's runtime path.
   Do not label queued jobs as passes or infer that all CI is green.
2. Before enabling an explicit backend in a deployment, verify that its Linux
   host permits `io_uring` and its storage supports the selected direct mode.
   Configure the environment before process/worker startup and budget the
   direct cache across all active reading processes. The [package README](../../README.md#sampling-io-configuration) records
   the actual defaults: mmap, cache disabled, eight rings and queue depth 256.
3. Establish a workload-specific performance case before changing deployment
   defaults. Compare identical training/prediction inputs and output digests,
   realistic workers and aggregate memory, resident and storage-bound cases,
   disk utilization and swap. The correctness and GPU checks prove execution
   and output behavior; they establish no production throughput improvement.

No additional production-code blocker has been demonstrated by the completed
qualification. Historical multi-terabyte reports remain imported provenance,
not independently reproduced speedup evidence.

Run the CPU matrix from the repository root with this checkout's source
packages, CPU dependencies and candidate native wheel installed:

```bash
DG_TEST_NATIVE_TASKS=1 python -m pytest -q -s -c /dev/null \
  --confcutdir=packages/kumo-engine/test \
  packages/kumo-engine/test/test_native_io_tasks.py
```

Run the strict native gate from `packages/diskgraph` on a Linux host permitting
`io_uring` and direct I/O:

```bash
bash scripts/check_native_io.sh
```

## October 30 historical investigation

The remaining sections preserve the original reproductions and original
action plan. Their failure results and pending-assurance statements describe
the pre-fix candidate, not the November 3 source. The corrections and current
gates above supersede those historical release blockers.

## Provenance and scope

- Engine base: `dbe45326705c706d1f87beb2b1dd320218edbc70`, freshly fetched
  `upstream/main`.
- Proposal: [kumo-diskgraph PR #424](https://github.com/kumo-ai/kumo-diskgraph/pull/424),
  source head `f84d30cf3b9a31884c46b3b71c19d233c80fbcc3`.
- Source merge base: `cc6279e841c15b74204f5cf637ff4e8fdf5dcfc2`.
  The PR's current base tip `62a07645` is not the patch baseline.
- The source PR is an unmerged draft with conflicts, 51 commits and 60 files.
  Its head reports four successful CodeQL/security checks; it reports no Rust
  test, Python integration or wheel-build check at that revision.
- The candidate carries the backend, sampler optimizations, benchmark tools,
  generators and original reports. Already-landed ingest memory budgeting and
  resumable scratch cleanup remain Engine's implementations.

The integration retains Arrow/Parquet 58.4, feature projection pushdown,
per-seed edge recording and parallel deduplication, resolved timestamp anchors,
computed offset chunk positions and atomic bitmap backup publication. New
explicit sampling feeds the existing relation remapping and commit paths.
At the user's request, seeded selection retains Engine's existing sorted
`rand::seq::index::sample` results across all backends. The proposal's Floyd
replacement changes high-degree sampled neighborhoods for the same seed and
is omitted; source benchmark timings therefore do not establish candidate
performance.
The port retains Engine's newer resident-Parquet memory-budget default, adds
the proposed explicit budget override and buffered compressed scratch output,
and retains the newer nested-list prefix sizing implementation and tests.

## October 30 verified findings

| Finding | Evidence and consequence | Required action |
| --- | --- | --- |
| Failed cache plans cancel another owner's reservation | `failed_plan_preserves_replacement` injects failure, reacquires the same cache key, then drops the old plan. The old destructor clears the replacement's LOADING reservation. | Consume/revoke reservation ownership exactly once; test error, cancellation and replacement interleavings. |
| Fork inherits unusable runtime/cache state | `fork_before_first_read` passes; `fork_after_first_read` times out at three seconds. `fork_with_populated_cache` terminates the child with SIGSEGV: cache metadata survives but its MADV_DONTFORK arena does not. | Reset both runtime and cache after fork, or reject inherited state with a clear checked error; test both cases. |
| Automatic fused dispatch selects an unsupported backend combination | Explicit backends reject fused feature fetching; the Python automatic heuristic can still select it. Source engine tests have four fused failures per explicit backend. | Make automatic dispatch consider backend capability; either implement explicit fused reads or reject only explicitly requested unsupported mode. |
| macOS ARM64 compilation breaks | A dependency-only `io-uring=0.7.13` cross-check for `aarch64-apple-darwin` fails with 12 errors, including missing `MAP_POPULATE`, `MADV_DONTFORK` and io_uring syscall constants. Engine declares macOS ARM64 as a supported wheel target. | Gate Linux dependencies and implementation by target; preserve macOS mmap and return a clear unsupported error for explicit I/O. |

These are observed failures, not conclusions drawn from benchmark throughput.
The macOS compilation failure blocks release even with the default mmap backend.
The [training loader](../../../kumo-core/src/kumo_core/diskgraph/lightning_data.py)
explicitly uses `multiprocessing_context="fork"` when worker count is positive.
The reproduced lifecycle failure applies when native I/O initializes in the
parent before those workers fork; it does not establish that every training
invocation initializes I/O in that order.
The ignored Rust diagnostics and strict Python expected failures intentionally
make the remaining blockers executable without hanging the ordinary suite.

### Resolved integration gap: profiler dependencies

The source's unused index `pprof` dependency enabled `flamegraph`, adding
`inferno 0.11.21` to both shipped artifact graphs. Notice generation rejected
its `CDDL-1.0` license under the existing policy. Removing that unused
declaration preserves the benchmark code and eliminates the rejected graph;
no license policy was changed.

Regenerated artifact and notice inventories add `io-uring 0.7.13` to the Linux
extension and ingest graphs, and benchmark `pprof 0.13.0` to the extension graph.
Notice validation passes for 389 components. Original upstream license text is
retained, including its indentation.

## October 30 reproduction and validation

Run Rust commands from `packages/diskgraph` with the package toolchain and
dependencies installed. Native diagnostics require Linux permitting io_uring
and a filesystem supporting O_DIRECT; denied native I/O is a failure rather
than a successful skip.

```bash
cargo test -p diskgraph-index -p diskgraph-engine -p diskgraph-ingest
cargo test -p diskgraph-index -p diskgraph-engine -p diskgraph-ingest -p diskgraph-datagenerator
cargo test -p diskgraph-index --test page_io_readiness -- --ignored --exact native_roundtrip --test-threads=1
DG_DIRECT_CACHE_GB=1 cargo test -p diskgraph-index --test page_io_readiness -- --ignored --exact native_roundtrip --test-threads=1
cargo test -p diskgraph-index --lib page_io::readiness_tests::failed_plan_preserves_replacement -- --ignored --exact
cargo test -p diskgraph-index --test page_io_readiness -- --ignored --exact fork_before_first_read --test-threads=1
cargo test -p diskgraph-index --test page_io_readiness -- --ignored --exact fork_after_first_read --test-threads=1
DG_DIRECT_CACHE_GB=1 cargo test -p diskgraph-index --test page_io_readiness -- --ignored --exact fork_with_populated_cache --test-threads=1
cargo check -p diskgraph-index --lib --target aarch64-apple-darwin
```

Install the candidate Python wheel, then run from the repository root:

```bash
DG_TEST_NATIVE_IO=1 pytest -q packages/diskgraph/diskgraph-pybind/python/tests/test_native_io_readiness.py --basetemp=/tmp/dg424-python-readiness
python3 tools/migrations/diskgraph/check_native_runtime.py --backend buffered --manifest /tmp/dg424-python-readiness/seed_e2e0/manifest.json
python3 tools/migrations/diskgraph/check_native_runtime.py --backend buffered --manifest /tmp/dg424-python-readiness/seed_e2e0/manifest.json --preinitialize
python3 packages/diskgraph/scripts/generate_artifact_manifest.py --repo-root packages/diskgraph --check packages/diskgraph/third_party/artifact-manifest.json
python3 scripts/gen_third_party.py --check THIRD-PARTY.txt
python3 -m unittest test.scripts.test_gen_third_party test.scripts.test_refresh_diskgraph_notice_inventory
```

The Python checks ingest real parent and train-table fixtures, compare exact
topology/Arrow outputs across backends, reuse runtime state, and verify feature
projection. Baseline covers three crates; candidate additionally covers the
data generator, so their totals are not a count of added tests.
The core diagnostic also needs the checkout's `kumo_core` and dependencies;
the pytest command creates its parent/train-table manifest. Repeat the core
command with `--backend mmap` and `--backend direct` for the other controls.

| Check | Result |
| --- | --- |
| Latest-main native baseline | 871 passed, 8 ignored |
| Candidate four-crate native suites, including final index rerun | 901 passed, 14 ignored, zero failures; final index alone has 72 passed and 6 ignored |
| Candidate ingest rerun after buffered scratch-output restoration | 757 passed, 7 ignored, zero failures |
| Candidate affected-package `cargo check --all-targets` | Passed |
| Candidate four-crate Clippy, all targets, warnings denied | Passed on Rust 1.96.1; pinned CI Rust 1.98.1 unavailable locally |
| Candidate native lifecycle diagnostics | Roundtrip, concurrent cancellation and fork-before-read pass; fork-after-read times out; populated-cache fork gets SIGSEGV; failed-plan replacement clears the new reservation |
| Candidate Python native suite, final built wheel | 3 passed, 2 strict expected failures; includes parent/train-table and projected parity plus upstream seeded golden replay |
| Actual Kumo iterable → fork DataLoader worker → runtime adapter | Fresh mmap/buffered/direct each return 2 batches and 2 HeteroData conversions; buffered/direct after parent reads time out at 3 seconds |
| Source engine with mmap | 49 passed, including ingest/export integration |
| Source engine with buffered or direct | 43 passed, 5 failed per backend: four fused failures and one missing-node error classification |
| Notice/inventory unit tests | 14 passed |
| Linux artifact manifest and generated notice | Current; notice has 389 components |
| License headers, tracked and new files | 5,916 files checked, 59 exclusions, zero errors |

Two inherited native unit tests return success when io_uring setup returns
`EPERM`; this sandbox denied setup while the host permitted it. Ordinary green
tests therefore do not establish explicit-backend execution. The new opt-in
diagnostics fail denied I/O and assert submission/cache-hit counters, but are
not wired into a native-capable Linux CI job yet.

The page-provider benchmark also uses different checksums: mmap sums the first
byte of pages, while explicit paths add the page count. A 64 MiB zero-filled
fixture produces mmap checksum zero and explicit checksum 128,000 for the same
128,000 pages. Use the strict byte comparisons for correctness; unify this
benchmark's digest before using it as a parity gate.

## October 30 proposed fixes and release gate (completed)

1. **Cache error ownership:** fix failed-plan cleanup; the replacement-reservation
   diagnostic must pass, followed by concurrent cancellation/read verification.
2. **Worker lifecycle:** reset/check runtime and cache state after fork. The
   bounded timeout and populated-cache SIGSEGV diagnostics must pass, plus
   the actual Python DataLoader/adapter check after a parent read.
3. **Backend dispatch:** make automatic fused selection capability-aware;
   parent/train-table and projected Python parity must pass without expected
   failures. Preserve explicit error classification tests.
4. **macOS packaging:** isolate Linux-specific I/O; cross-check the index and
   build/import the supported macOS ARM64 wheel while exercising default mmap.
5. **Native assurance:** run opt-in roundtrip/cache/error/worker checks in a
   native-capable Linux job, require nonzero I/O and cache-hit counters, and
   compare consistent byte digests across provider benchmark backends.

Only after correctness gates pass, measure real training/prediction workloads
with realistic worker counts and total process memory limits. Cache allocation
is per process, so worker count must be included in memory sizing. Re-run
steady-state resident and storage-bound comparisons with identical workloads,
output digests, zero swap and recorded disk utilization.

The historical multi-terabyte performance reports are imported source evidence.
This investigation does not independently reproduce or endorse their speedup
claims, and small local correctness fixtures establish no production speedup.
