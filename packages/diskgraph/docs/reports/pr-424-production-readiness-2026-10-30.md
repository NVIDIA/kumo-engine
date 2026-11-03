# PR #424 production readiness — October 30, 2026

Production enablement remains gated on validation of the latest merged
candidate. The October 30 investigation reproduced cache recovery, process
lifecycle, automatic dispatch and macOS build failures. The November 3 follow-up
records their fixes and completed checks. The default remains
`DG_EDGE_BACKEND=mmap`.

## November 3 follow-up

The candidate now incorporates Engine `main` through `f8ba7e17`, including the
extracted native runtime, timestamp clock and platform-ingestion changes.
The latest main merge passes 930 native tests and 158 related Core tests.
The October 30 findings below remain the historical reproduction record;
this follow-up tracks fixes.

At correctness head `d6057381`, the [hosted Linux CI run](https://github.com/NVIDIA/kumo-engine/actions/runs/41696146582)
passes ordinary Rust, strict native I/O (ten checks without skips), Clippy,
candidate-wheel build and DiskGraph Python checks. The same head's
[macOS job](https://github.com/NVIDIA/kumo-engine/actions/runs/41696146582/job/126127793077)
builds and installs the actual `cp311-abi3-macosx_14_0_arm64` wheel on macOS
15.8.2 ARM64, Rust 1.98.1 and Python 3.12.10: all 11 checks pass, including mmap
ingestion/fused feature sampling and explicit-backend rejection. No wheel hash
was printed. These results precede the latest main merge and Windows changes;
their final-head platform, explicit GPU and CPU task gates remain pending.

- Cache cleanup now relinquishes reservation ownership once. Native runtime and
  cache state are process-owned, including fork during initialization, locked
  cache state, populated cache and inherited in-flight work. The host index and
  engine suites pass 138 tests with 11 opt-in tests ignored; all selected native
  error/lifecycle diagnostics pass when run explicitly. All-target Clippy passes.
- Linux-only dependencies and I/O primitives are gated, and mmap remains the
  non-Linux implementation. Latest main adds experimental native Windows builds.
  The proposal's page-I/O/prefetch code and benchmark profiler now preserve that
  build boundary: Unix profiling stays Unix-only and unsupported explicit I/O
  returns checked errors. Windows GNU compile checks cover the actual source
  and unit-test code; a full local cross-build lacks a C cross-compiler, and the actual MSVC wheel
  build/import remains a CI gate. The WSL-only Windows release boundary stays
  unchanged.
- Native automatic fused dispatch is capability-aware. Actual CPU training also
  exposed Core's automatic caller forcing `fused=True`. Backend-aware Core
  configuration fixes that gap while preserving mmap's worker/thread policy;
  118 configuration tests pass. The training test exercises the actual default.
- The provider benchmark now checks the same page bytes for all three backends;
  its bounded native run passes. This establishes digest parity, not a speedup.
- All eight task types complete real CPU mmap fits, validate eight prediction
  entities and match repeated/reloaded predictions. Export row order can vary
  across independently generated fixtures, so selecting their first rows does
  not establish equivalent prediction inputs. The full 24-fit matrix selects
  canonical entity/time keys, retains each backend's training/reload checks and
  compares predictions using identical saved model, snapshot and input artifacts
  across backends at absolute tolerance 1e-6. All 24 CPU fits pass (eight pytest
  cases, 199.61 seconds): binary, multiclass, multilabel, multilabel ranking,
  regression, forecasting, static links and temporal links. Each fit completes
  at least two optimizer steps; predictions cover eight entities, all classes
  or RHS items, and all 12 forecast horizons. Own-model repeat/reload parity and
  shared-model backend parity pass. The matrix does not measure per-fit native
  I/O counters, projection payload size, GPU behavior or large-graph throughput.
- The updated native wrapper and Core fork lifecycle checks pass 22 tests with
  two platform-specific skips, including a warmed dataset reused after fork.
  Existing runtime end-to-end tests pass 44 tests, and related Core tests pass
  155 tests.

The 24-fit wheel SHA-256 is
`8e6c715ae213c3ecf2ab7356cdb494173e2478472e16215cbe5319dac7f7bf0a`.
The final wheel adds strict rejection of invalid UTF-8 backend configuration;
normal backend strings are unchanged. Its SHA-256 is
`36632960b79540e46e3a0602c0c8fa7ec74d9b5b5e57c4ccba7874c5a9293670`,
and its focused native/Core suite passes 22 tests with two non-Linux skips.
These artifact results also precede the latest main merge and Windows changes.
Fresh candidate-wheel qualification and GPU task checks remain pending.

Run the CPU matrix from the repository root, with this checkout's source
packages and CPU dependencies installed plus the candidate native wheel:

```bash
DG_TEST_NATIVE_TASKS=1 python -m pytest -q -s -c /dev/null \
  --confcutdir=packages/kumo-engine/test \
  packages/kumo-engine/test/test_native_io_tasks.py
```

The earlier fork PR's 41 failed checks all stopped at AWS credential setup,
before tests. The development branch and replacement draft now live in
`NVIDIA/kumo-engine`; their current CI results must be assessed independently.

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

## Verified findings

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

## Reproduction and validation

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

## Focused follow-up PRs and release gate

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
