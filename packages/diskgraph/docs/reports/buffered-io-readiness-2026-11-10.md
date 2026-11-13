# Buffered I/O readiness review

This change adds an opt-in Linux `buffered` sampling backend and preserves
`mmap` as the default. Direct I/O, its userspace page cache, and its benchmark
scenarios are outside this branch's scope.

## Implementation boundary

Buffered reads use `io_uring` and the Linux page cache. Dedicated owner threads
submit page reads, retain buffers until completion, and return prepared bytes
to the sampler. Planning batches header and selected-entry pages instead of
reading every adjacency in full. Offset lookup and metadata retain their
existing mapped representation.

Sampling keeps the existing seeded neighbor-selection algorithm. Automatic
sampling and Core configuration select standalone feature fetching with
Buffered; explicitly requesting fused native sampling raises a configuration
error. Backend errors propagate instead of silently switching to mmap.

Each process owns its runtime and buffer pool. The fork and cancellation
boundaries require native tests in addition to ordinary Rust unit tests; a
successful build alone does not exercise `io_uring` permissions or inherited
runtime state. See [configuration](../../README.md#sampling-io-configuration)
and the executable gate at `packages/diskgraph/scripts/check_native_io.sh`
(run it from the repository root).

## Candidate source qualification

The candidate source is signed commit
`8878f21d2ffc198f6ef26b00def4884988dfd819`, based on main
`58890141ae0b7d77524ea5562039ef48c7979249`. Source qualification passed:

| Check | Result |
| --- | --- |
| Index and sampler engine unit tests | 41 and 60 passed |
| Developer benchmark CLI tests | 4 passed |
| Real Linux Buffered lifecycle/provider gate | 9 executed, zero skips |
| Affected native all-target build and strict Clippy | Passed |
| Core backend configuration tests | 91 passed |
| GPU workflow shell/configuration regressions | 26 passed |
| Core Lightning data-module tests | 161 passed |
| Changed Python flake8 and isort | Passed |
| Documentation tests, references, and catalog | 36 tests passed; 317 Markdown files checked; catalog matches |
| Source headers and Python license metadata | 6,372 candidate files and 14 manifests passed; 21 checker tests passed |

Native checks used Linux aarch64, Rust stable 1.96.1, and
`RUSTFLAGS="-C target-feature=+aes,+neon"`. The repository's pinned Rust
1.99.1 toolchain was unavailable locally. macOS and GPU behavior require
their CI jobs; they were not executed locally.

The benchmark's `--verify` mode now rejects inconclusive comparisons instead
of treating missing comparable output as a successful check.

### Fatal I/O ownership and recovery

Review found that a failed submission/wait could release buffers while the
kernel still held submitted requests. The candidate now retires that owner
on an undrainable submission/wait failure or panic and retains the ring,
file handles, buffers, and vectored-read descriptors until process exit.
Future work sent to the affected owner fails. Restart the affected process
after this terminal owner error. Other owners remain independent.

Retention is at most one failed batch per owner, not a fixed byte limit; a
batch can exceed the ring queue depth. Ordinary read-completion errors are
drained normally and remain recoverable. Four fault-injection scenarios cover
both error and panic after real submission through sparse reads and coalesced
vectored reads. They verify retained resources, no premature buffer recycling,
rejection of later work, and no repeated retention. The
[liburing cancellation contract](https://kernel.googlesource.com/pub/scm/linux/kernel/git/axboe/liburing/+/refs/heads/master/man/io_uring_cancelation.7)
requires waiting for completion before freeing request resources; closing a
ring alone does not establish that the kernel has stopped using them.

## Built-artifact qualification

An optimized release wheel was built from tested source
`8878f21d2ffc198f6ef26b00def4884988dfd819` and installed in a fresh Python
3.14.7 environment. The extension's build-commit identifier matched that
source, and the bundled ingest binary matched the rebuilt release executable.
The tests used the installed native package with this checkout's Engine and
Core Python sources. Documentation changes after that source commit do not
change the tested native inputs.

| Artifact | SHA-256 |
| --- | --- |
| `diskgraph-0.1.0-cp311-abi3-linux_aarch64.whl` | `b08c2230ba6ae5bc24669e9da993a7ca16cddebac6d2df7e22ad55c25129bd63` |
| `diskgraph/_c.abi3.so` | `c4a437bbdbc1f82c206ebc525084755c235efefcd3ea6724c84ead5b0b19732f` |
| `diskgraph/bin/ingest` | `b758122463394dd8af3d1a2da66fe7238ea99ed6bdb49696009139896a5e0626` |
| Native Git inputs, 397 records | `ffeb69f33d0772116782f6c98c6ed3439530d54d3b7252a592a6f0d9f2fa948f` |

The installed-wheel native Python suite passed all 10 checks with zero skips,
including mmap/Buffered ownership, fork recovery, feature projection, and
seeded parity. Portability checks passed 10 cases; the non-Linux-only
unsupported-platform test was expectedly skipped on Linux. That skip is
separate from the critical native suite, which requires zero skips.

Reproduce the native-input digest from the published PR head using the
NUL-separated records from `git ls-tree -rz HEAD -- packages/diskgraph`,
excluding paths containing `/docs/` or ending in `.md` or `.html`. Sort the
complete remaining records lexicographically as bytes, join with NUL, append
a final NUL, and compute SHA-256. Each record retains its Git mode, type,
blob ID, and path. The published head has the same 397 input records and
digest as the initial tested build; report-only edits do not change them.
The initial build's commit identifier above is retained as build provenance,
not a revision reviewers need to fetch.

With the candidate wheel and CPU test dependencies installed, run from the
repository root using that environment's Python. Native tests require a
Linux host permitting `io_uring`; denied operations fail:

```sh
export PYTHONPATH="$PWD:$PWD/packages/kumo-engine:$PWD/packages/kumo-core/src:$PWD/packages/kumo-api:$PWD/packages/kumo-pql:$PWD/packages/kumo-engine/test"
DG_TEST_NATIVE_IO=1 DG_TEST_CORE_IO=1 KUMO_RUN_GCP_TESTS=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -c /dev/null \
  --confcutdir=packages/diskgraph/diskgraph-pybind/python/tests \
  --junitxml=/tmp/diskgraph-buffered-native.xml \
  packages/diskgraph/diskgraph-pybind/python/tests/test_native_io_readiness.py \
  packages/diskgraph/diskgraph-pybind/python/tests/test_backend_portability.py
```

All eight public Engine task cases passed across mmap and Buffered: 16 fresh
process/backend cells, zero failures, errors, or skips, in 128.33 seconds.
They cover binary, multiclass, multilabel, multilabel ranking, regression,
forecasting, static link prediction, and temporal link prediction. Each cell
trained and produced predictions for eight entities; the six non-link tasks
completed two optimizer steps and the link tasks completed four.

Repeat and save/load predictions matched within an absolute tolerance of
`1e-6`, with zero relative tolerance and decision checks. Cross-backend
prediction parity uses the same retained mmap-fitted model and inputs, while
each backend also independently completes its own training and reload checks.
This tests reading/model execution equivalence without claiming that separate
training runs have identical weights.

Run the public matrix from the same environment and source paths:

```sh
DG_TEST_NATIVE_TASKS=1 KUMO_RUN_GCP_TESTS=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -s -c /dev/null \
  --confcutdir=packages/kumo-engine/test \
  packages/kumo-engine/test/test_native_io_tasks.py
```

These small correctness checks provide no new performance measurement for
this binary. The final PR's documentation and build-commit metadata can
change after the tested source commit; the 397 functional input records
above must remain identical.

## Evidence from the earlier candidate

The [full proposal investigation](https://github.com/NVIDIA/kumo-engine/blob/c1a04a305f569cf4fc6a1284e054b9c4277f3625/packages/diskgraph/docs/reports/pr-424-production-readiness-2026-10-30.md)
records the original port and readiness fixes. The
[repeated end-to-end benchmark report](https://github.com/NVIDIA/kumo-engine/blob/3e65cd8cb667faba35d40685c367de7511438120/packages/kumo-engine/benchmark/results/native-io-2026-11-09.md)
retains raw evidence, controls, and reproduction commands for that candidate.
Its mmap and Buffered rows produced these median fit plus first-prediction
times, in seconds, over five repetitions per configuration:

| Task | Memory regime | mmap | Buffered |
| --- | --- | ---: | ---: |
| Classification | Resident | 44.14 | 54.85 |
| Classification | Constrained | 358.38 | 64.99 |
| Regression | Resident | 40.44 | 52.00 |
| Regression | Constrained | 341.25 | 63.35 |
| Link prediction | Resident | 75.28 | 77.55 |
| Link prediction | Constrained | 174.34 | 104.59 |

These are measurements of the earlier pinned candidate, not new measurements
of this branch. The controlled workloads used ARM CPU training, local NVMe,
standalone fetching, zero DataLoader workers, one Torch thread, fixed work,
private immutable inputs, and matching predictions. Resident means a fresh
process with files in the kernel page cache; constrained runs used a 4 GiB
memory limit and demonstrated physical reads and reclaim. The full report
records omitted production configurations and instrumentation limitations.

## Release decisions

Keep mmap as the default: the earlier Buffered path improved the constrained
cases but regressed resident classification and regression. Repeat performance
qualification with the intended hardware, concurrency, storage, and memory
budget before enabling Buffered in a deployment.

Automatic worker sizing does not yet account for cgroup limits. The earlier
controlled benchmark avoided that policy by selecting zero DataLoader workers;
it did not qualify automatic worker throughput or OOM behavior. Worker sizing
should be addressed separately before enabling automatic Buffered training in
tightly constrained deployments.

This branch intentionally carries no Direct cache tuning, cache-lifetime
contract, Direct benchmark scripts, or historic Direct performance claims.
