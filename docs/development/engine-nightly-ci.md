# Engine nightly check

`Kumo Engine Nightly` builds a selected product revision and runs sixteen
primary fits on two registered RTX hosts. Each of eight tasks runs once with
normal answers and once with scrambled answers. Every fit uses seed zero and
the existing full `100M` profile. Quality scores report only.

The schedule is fixed at 10:00 UTC, which is 03:00 PDT and 02:00 PST. A manual
run accepts `product_ref`, with `main` as the default. GitHub requires the
[manual workflow](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)
to exist on the default branch. Trusted copied PR branches also run when the
nightly workflow, guide, runtime, comparator, shared GPU helper, or their
contract tests change.

## Source and wheel identity

The workflow checks out its harness at the workflow revision. A separate
checkout resolves `product_ref` to the product commit. The existing builder
builds that product tree. The existing PR helpers record its commit, tree,
version, and wheel SHA-256, then reject source changes during the build.
Each shard also records its checked harness commit, run ID, and run attempt
in `setup/harness.json`. The harness commit must equal the workflow commit.

Each RTX job installs that exact archive in a fresh Python 3.12 environment.
It installs the frozen dependency profile first through the public package
index. It then installs the wheel without changing those dependencies. The
runtime checks the archive, every installed runtime file, the installed
version, all dependency versions, and the actual RTX GPU.

## Trusted host configuration

Set these repository variables before a run. Paths must be absolute and must
refer to the same retained resources on both hosts.

| Repository variable | Required value |
| --- | --- |
| `KUMO_ENGINE_NIGHTLY_RTX_RUNNER` | One registered pool label that selects the two admitted RTX hosts |
| `KUMO_ENGINE_NIGHTLY_ROOT` | Writable study root, including the shared `gpu-study.lock` |
| `KUMO_ENGINE_NIGHTLY_DATA` | Existing full profile data root, with each graph's `100M` directory |
| `KUMO_ENGINE_NIGHTLY_SCALE_SOURCE` | Existing pinned PR 172 `scale-bench` source directory |
| `KUMO_ENGINE_NIGHTLY_FIXTURES` | Frozen registry root with `{task}/{normal,seed-0}/descriptor.json` |
| `KUMO_ENGINE_NIGHTLY_DEPENDENCY_PROFILE` | Frozen JSON map of exactly 98 non-Engine runtime package versions |
| `KUMO_ENGINE_NIGHTLY_REFERENCE_CACHE` | Approved version 1.0 cache root with all sixteen cases and `manifest.json` |
| `KUMO_ENGINE_NIGHTLY_REFERENCE_SHA256` | SHA-256 of that approved cache manifest |

The configuration job fails before it selects a GPU runner if a variable is
missing. There is no T4 fallback. The workflow provisions no runners, creates
no training data, and does not register or change hosts. The current access
audit could not list NVIDIA runners because the API returned 403. It also
found no runner agent on the two retained RTX hosts. A usable registration
and pool label remain prerequisites. An SSH connection does not prove that
a host can receive GitHub jobs.

## Required work and retained evidence

The node shard runs binary, multiclass, multilabel, and regression tasks. The
other shard runs forecasting, multilabel ranking, static links, and temporal
links. Each shard performs eight primary fits. The host lock prevents a
second benchmark from using the same host at the same time. SSH benchmark
launches must honor the same lock.

One shard also runs the existing installation, documented example, and
behavior phases. Their fit ledgers count extra fits separately. The workflow
does not call the PR worker that performs another eight primary fits.
Pytest and its test tools use a separate target directory. Each correctness
phase appends that directory after the locked runtime packages. Its install
receipt is retained separately. The measured runtime stays unchanged.
Current behavior coverage is incomplete for B7 model reuse: weights, new
entities, backfill, and compatible graphs. The final required status reports
that gap and fails. This draft does not claim complete nightly qualification.

The comparison job runs no fits. It uses a fresh environment with the same
wheel and dependency profile. Its environment setup, CUDA checks, and report
comparison use the same host lock. It requires all sixteen candidate cases and
the pinned approved reference cache. Score differences remain diagnostic.
Missing cases, changed identities, invalid predictions, failed model reloads,
and failed execution remain errors. No fixed runtime gate is asserted.

GitHub artifacts retain the wheel, build log, source identity, installation
receipts, dependency profile, fit ledgers, task plans, raw prediction data,
published models, comparison report, and failure logs. The pack command
excludes full input data, prepared native tables, and feature caches. Those
large files remain on the host. There is no reference calibration or cache
publication step in this workflow.
