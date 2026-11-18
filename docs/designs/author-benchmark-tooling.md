# Author validation and paired benchmark tooling

Status: proposed implementation. Tracking: [#384](https://github.com/NVIDIA/kumo-engine/issues/384).

Authors and coding agents can already run public tests, generated-data scale
benchmarks, and specialized A/B experiments. Entry points and result contracts
are spread across component guides; some old entry points provision/format
storage or require private CI fixtures. This change adds discovery, bounded
execution, and reviewable receipts while reusing the existing harnesses.
It does not change contribution policy, accepted-issue rules, PR templates,
branch protection, or the release/test matrix.

## Reuse existing harnesses

| Existing capability | Reuse |
| --- | --- |
| `scripts/test_public.py` | Credential-free Docker reader groups and their reports |
| `packages/diskgraph/scripts/benchmarks/shared_temporal_ranges.py` | Paired oracle-checked temporal-query operation benchmark |
| `benchmark/commerce/generator/` | Public synthetic smoke/scale data and manifest verification |
| `benchmark/commerce/run_e2e.py` | Snapshot/training/prediction stages and realized-work information |
| `packages/diskgraph/scripts/benchmarks/commerce/run_matrix.py` | Bounded sampling matrices with explicit processes/threads/cache modes |
| Small output/workflow checks in `test/scripts/` | Local checks also used by OSS Readiness |

`scripts/author_checks.py` discovers JSON profiles, probes prerequisites, prints
resolved argv, and executes only an explicit selection. It records commands,
Git state, Python diagnostics, host CPU/memory information, raw logs and outcome.
Smoke data/snapshot profiles require an explicit disk-space budget: a preflight
floor, not a quota. Selected existing harnesses may build/download public
dependencies; profile limits state that effect. No cloud or destructive storage
profile is included. The tool does not install prerequisites or environments.

`scripts/benchmark_pair.py` times complete commands in alternating paired order.
A plan identifies workload, seed/workers/cache/device context, inputs, per-side
interpreter/env, optional native/build artifacts, expected checker receipt,
timeout, and pair/warmup counts. The author supplies the experiment's checker.
A fresh JSON receipt must match explicit, type-sensitive expected content;
exit zero alone is insufficient. Checkers run outside the timed trials;
timed-output correctness remains the harness's responsibility.

The [usage guide](../development/author-validation.md) starts with reusing
existing data and its manifest/evaluation inputs, including exact S3 locations
for the Engine CI archive and the seven generated scale graphs. It provides
download-once, checksum/validation and reuse commands; generation is an optional
fallback. It provides exact smoke and opt-in scale commands, a runnable self-comparison control, and the distinction
between complete-command timing and specialized operation/GPU measurements.

## Failure and evidence semantics

Each attempt gets a new private-mode report directory. Benchmark results live
outside both measured Git roots. Reports start as `running`; only all-successful
comparisons publish timing summaries. Failed commands, wrong/missing checker
receipts, timeouts, interrupts, and changed inputs/artifacts fail the attempt.
Partial logs remain. Child processes have separate process groups. Normal
worker/resource-tracker shutdown gets a bounded drain period before lingering
groups are terminated. Windows authors use WSL.

Reports retain individual samples, per-pair ratios and their median/range; no
noisy cross-machine performance gate is introduced. Fewer than three pairs and
subsecond commands receive warnings. Callers synchronize device work. An `ok`
receipt implies no memory, model-quality, GPU/kernel speedup, statistical
significance, or production-scale claim.

Only explicitly listed files are hashed. A verified manifest can identify a
large fixture, but its contents are not recursively checked. Git cannot identify
untracked/ignored contents; list important native libraries and wheels as
artifacts. Full dependency versions, inherited environment, thermal state,
cache residency, and competing load remain external evidence. Separate venv
interpreters retain their executable paths; symlinks are not resolved away.
No environment dump or credential collection runs.

## Review lessons and validation

Small changes can affect allocations, I/O stage boundaries, output ordering,
tail latency, and realistic memory use. Compare the correct baseline, including
stacked changes; preserve controls and report tiny/noisy improvements honestly.
An operation-only gain does not establish an end-to-end training gain. Reduced
work or changed outputs can masquerade as speedups. The temporal-range harness
already demonstrates source-derived oracles, controls, native identities, and
AB/BA measurement; these tools retain rather than replace that specialization.

Focused tests cover discovery, local path parameters, prerequisite diagnostics,
no-mutation dry runs, status propagation, stale-output rejection, disk preflight,
paired order, checker content/types, malformed plans, venv identity, input changes,
and descendant cleanup. OSS Readiness runs them without cloud/model training.
A generated-commerce smoke and verifier check a real public harness; the
self-comparison control checks report generation. Neither establishes scale
performance. Existing commands and CI remain available; use is optional.
