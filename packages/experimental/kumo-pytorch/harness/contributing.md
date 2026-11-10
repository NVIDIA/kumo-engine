# Contributing to kumo-pytorch

kumo-pytorch is developed by its maintainers and isn't accepting outside
contributions yet. Bug reports and suggestions are welcome as GitHub issues, as
for the rest of kumo-engine ([CONTRIBUTING.md](../../../../CONTRIBUTING.md)).
This guide is how the maintainers, and the agents working with them, change the
package.

Contributors agree on the intended answer before implementing new behavior.
They preserve a tiny example of that answer, then test it through the same API
a user calls. Every PR is approved by a package owner before it merges, and
changes to agreed behavior or the checking system are approved by a harness
owner who isn't the author.

The process checks a small, named set of examples. A green check does not
establish that all kumo-pytorch inputs work.

A **guarantee** is the English rule. A **golden example** is tiny input data and
an independently agreed answer. An **engine** is a way of running the request.
DiskGraph is the shipped engine, and the spec is checked on it; the in-memory
engine is a test tool that the harness compares with DiskGraph
([rules.md](rules.md#differential-testing)). A pull request (PR) proposes
repository changes for review; CI is the set of automatic checks GitHub runs on
that proposal.

Paths in this guide are relative to the package directory,
`packages/experimental/kumo-pytorch/`, unless they start with `/`, which means
the root of the kumo-engine repository.

## The harness today

What exists and how each piece is enforced. Anything not listed here isn't
enforced; where another document describes more, this guide governs.

| Part | What exists | Where | Enforced by |
|---|---|---|---|
| **Conformance suite** | Guarantees (English rules), hand-calculated golden cases, and each engine's expected answer or refusal per case; seeded generated cases from a dimension registry, checked by properties; a regression corpus of repro files; run on DiskGraph and on the in-memory engine for the differential comparison | `conformance/`, `tests/conformance/` | `kumo-pytorch-gate` |
| **Content policy** | Engine expectations only move forward unless a guarantee changes too; every golden case is linked from its guarantee | `/.github/scripts/kumo-pytorch-policy.py`, tested by `tests/conformance/test_content_policy.py` | `kumo-pytorch-gate` |
| **Rules** | Decision rules (S, V, D, C, X) and escalation triggers (E1–E6) | [rules.md](rules.md) | Each rule names its enforcement: CI, the independent reviewer, or the harness owners' approval |
| **Instructions** | `AGENTS.md` as the index; this guide; the self-review loop and its notes; the independent reviewer's brief | `AGENTS.md`, `harness/` | Human review of each PR; the reviewer agent runs inside the self-review loop |
| **Ownership** | Package owners for the whole package; harness owners for the guarantees, goldens, conformance machinery, these instructions and the kumo-pytorch CI | [/CODEOWNERS](../../../../CODEOWNERS) | kumo-engine's branch rules: code-owner approval on every PR |
| **Checks after merge** | The kumo-pytorch workflows on `main` and nightly; a failure opens or updates an issue labeled `kumo-pytorch-ci` | `/.github/workflows/kumo-pytorch-*.yml` | The author of the failing commit fixes or reverts it ([How a PR merges](#how-a-pr-merges)) |

## Start here

kumo-pytorch is a package in kumo-engine's `uv` workspace. At the root of a
checkout, install the CPU build of PyTorch and the workspace, then kumo-pytorch
with its `dev` extra, then run the conformance suite from the package
directory:

```bash
uv sync --locked --extra cpu
uv sync --locked --inexact --no-install-package torch --package kumo-pytorch --extra dev
source .venv/bin/activate
cd packages/experimental/kumo-pytorch
python -m conformance
```

The second sync builds DiskGraph from `/packages/diskgraph`, so it needs the
Rust toolchain ([installation](../docs/installation.md#run-the-tests-and-notebooks-from-a-checkout)).
Add `--no-install-package diskgraph` to skip it; DiskGraph tests then skip.

`python -m conformance` checks the English-document links, JSON examples, the
runner's failure checks, and the examples on the in-memory engine. It
explicitly reports that DiskGraph has not been executed. To run both engines:

```bash
python -m conformance --engine all
```

A missing native package fails an explicitly requested DiskGraph run; it never
turns into a skipped pass.

The [file guide](../conformance/README.md) defines every field of the guarantee
and golden files.

## One rule, nearby examples: a worked example

An English guarantee and its golden JSON examples share one directory:

```text
conformance/spec/guarantees/G-ROWS-001/
  guarantee.md
  golden/C-ROWS-001.json
  golden/C-ROWS-002.json
```

The directory supplies ownership. The heading names the directory's ID, and the
document links every example. Each JSON supplies rows, an actual saved request,
anchors, expected raw values, and a hand-worked reason. No separate case-to-rule
record, support field, manifest, or bug-answer copy is needed.

The answer is agreed independently of whether an engine can produce it yet.
Independent assertions in `conformance/expectations.json` state each engine's
current obligation; every case has one for both engines. The DiskGraph
assertion is the check of the spec; the in-memory assertion is the differential
cross-check, and a failure there alone is triaged by D2. CI checks the set of
cases against those assertions, so an unregistered JSON can't be mistaken for
tested coverage.

[G-ROWS-001](../conformance/spec/guarantees/G-ROWS-001/guarantee.md) states that
a one-relation count includes matching rows at or before each request's
cutoff. User 1 has events at 2, 9, 10, 12; user 2 has an event at 11.

| Request | Eligible event times | Expected count |
|---|---|---|
| user 1 at 9 | 2, 9 | 2 |
| user 1 at 10 | 2, 9, 10 | 3 |
| user 1 at 12 | 2, 9, 10, 12 | 4 |
| user 2 at 10 | none | 0 |

Both JSON files contain those rows, anchors, and answers. C-ROWS-001 also checks
bounded counts: `start < time <= stop`, including an empty recent window when
the only nearby event lies exactly at its start. In C-ROWS-001, the
relation matches `users.user_id` to `events.user_id`. In C-ROWS-002, it matches
`users.group_id` to `events.group_id`, neither of which is its table's primary
key. Both are valid requests with the same intended counts.

| Baseline case | In-memory | DiskGraph |
|---|---|---|
| C-ROWS-001 | accepted; exact golden answer | accepted; exact golden answer |
| C-ROWS-002 | accepted; exact golden answer | existing refusal: relation needs exactly one primary-key side |

The second case is a valid request that one engine can't execute. These are a
worked baseline for existing behavior, not unfinished features; keep them as
real regression checks. `conformance/expectations.json` is the live support
assertion; a later promotion doesn't change the count rule.

### What `check()`, `prepare()` and `check_contract()` mean

Constructing `DataLoaderSpec` describes the request. It doesn't select an engine
or read a dataset. Structural validation checks that the description is well
formed.

Public `check(request, dataset, engine)` inspects the combination and returns
`ValidationIssue` objects. It combines the contract's own validation,
source-binding checks, and the engine's optional `check_sources()` and
`check_contract()` checks. DiskGraph raises its refusals from its declared list,
`DiskGraphEngine.unsupported` (`src/kumo/pytorch/engines/unsupported.py`), so
each carries the entry's code and next step.

Public `prepare()` always calls `check()` first. If problems exist, it raises
`ValidationError` before calling `engine.open()`. Otherwise it opens the data
and returns a plan; `materialize()` binds sample requests and fitting;
`preview()` exposes actual raw samples and a batch.

```text
Describe a valid request
        |
        v
check(request, dataset, engine)
        |
        +-- problems --> prepare() raises ValidationError; no engine opening
        |
        +-- accepted --> prepare() opens data
                         --> materialize(requests)
                         --> preview() --> compare approved raw values
```

For C-ROWS-002, the real `DiskGraphEngine.check_contract()` reports the relation
shape at `schema.relations[0]`. The test doesn't recreate that rule or mock it.
It traps `engine.open()`, binds nonexistent Parquet files, and requires both
`check()` and `prepare()` to report the specified limitation with a suggestion.
A late failure or unrelated error can't pass.

## Principles of this flow

1. **Agree on new meaning before implementing it.** A human reviews the English
   rule, boundaries, compatibility, and independently calculated answers.
2. **Keep meaning separate from engine support.** The correct answer stays fixed
   when an engine moves from refusing a request to producing its answer.
3. **Use the public API.** Every golden contract first passes through the real
   `to_json()` → `from_json()` methods, preserving values, explicit sequences,
   and both fingerprints. Required examples also rearrange dictionary keys and
   use the loaded contracts for `check()`, then either verify an early refusal
   from `prepare()` or use `prepare()` → `materialize()` → `preview()` to
   compare actual raw values.
4. **Execute every required case.** A complete run rejects missing case/engine
   checks, skips, expected failures, unexpected passes, and deselections.
5. **Get the right approval.** Changing approved answers, inputs, rules,
   refusals, or checking code needs a harness owner's approval. Without a
   guarantee change, engine expectations can only move from a refusal to
   `"answer"`.

Agents also follow the decision rules in [rules.md](rules.md).

## First PR for new behavior: settle the answer and prove current support

Start the discussion in an issue when useful, then open a PR containing the
following files. The reviewed files are the durable decision that the
implementation follows.

```text
conformance/spec/guarantees/G-<topic>-<number>/
  guarantee.md
  golden/
    C-<topic>-<number>.json
conformance/expectations.json
src/kumo/pytorch/...
```

Use an existing guarantee when it already determines the answer. Create a new
directory only for a distinct rule. Refining an existing rule requires a
compatibility explanation: which existing requests change meaning, and why?
New configurations don't automatically need new guarantees.

The document has four sections: **Rule**, **Boundaries**, **Scope**, and
**Golden examples**. State results, including equality, empty outcomes, or other
relevant boundaries. Name open questions and keep them outside the approved
scope. Link every JSON example in the directory from the Golden examples list.

Each golden file supplies actual toy rows, a saved `DataLoaderSpec`, sample
requests, the correct result, and the hand-worked calculation. Calculate the
answer from the rule and rows, without running an engine to obtain it.

Add each case to `conformance/expectations.json`. For each engine, require
`"answer"` or an object with `path` and `message_contains` for a specific
refusal. An unimplemented engine must refuse; an already-supporting engine must
return the golden answer. These are independent test assertions, not another
runtime support policy.

If the existing API can't describe the request, add the smallest request
objects/options and serialization needed. A valid description should be
constructible and saveable even before an engine can execute it. Add the
runtime support check so unsupported engines refuse it. Don't implement the
feature's calculation in this PR.

When DiskGraph can't run a new operation yet, add an entry to its unsupported
list with a code, a plain description and a next step, refuse it from
`check_contract()` with that entry, add its minimal case to
`tests/test_diskgraph_unsupported.py`, and regenerate `docs/limitations.md`
(`python -m tests.limitations_reference --write`). Don't add a CI-only checker.
The in-memory engine declares nothing; it refuses only what it can't read.

When the new API produces a kind of output the checker can't yet compare, the
same PR first extends the shared comparison/validation helpers, with tests of
right and wrong values. Otherwise the case is rejected; an unasserted output is
never counted as coverage.

**What CI does on this PR:**
1. Discover all guarantee directories; check IDs, document sections, example
   links, uniqueness, and that every engine has an assertion.
2. Load every JSON; save and reload its contract with the public `to_json()` and
   `from_json()`. Require unchanged values, explicit sequences and both
   fingerprints; check complete tables, anchors, answers, and supported output
   fields. Engine assertions use the loaded contract.
3. For engines asserted as refusing, verify the specific public refusal before
   any data is read or opened. For engines asserted as supporting, require the
   actual golden answer. Repeat with reversed and deterministically shuffled
   dictionary keys (arrays keep their order). An unexpected outcome fails.
4. Test the machinery itself: wrong answers, accepted refused requests,
   wrong/additional refusal reasons, row reads, malformed records and missing
   native dependencies must fail.
5. Run the content policy: the expectation changes come with the guarantee
   change that explains them, and the new golden files are linked from it.
6. Run the existing suite (including the doc-code and link tests), style checks
   and the package build on CPU; run the whole suite again with DiskGraph.

CI doesn't calculate an unsupported feature's intended answer and can't judge
whether the English is right or the API addition minimal. A harness owner
approves the answer, checks compatibility and scope, and rejects feature
calculation code in this first PR.

## Following PR: implement the unchanged answer

Link the merged first PR, guarantee, case, and engine. Implement the behavior
and change that engine's expectation from its refusal object to `"answer"`.
Remove or narrow its runtime guard. Keep the approved rule, golden inputs, and
answer unchanged. If these are the only expectation edits and no harness-owned
files change, a package owner's approval is enough. CI must actually execute
the promoted engine and produce the approved answer, and all other listed
examples must keep passing.

If implementing reveals ambiguity or a wrong answer key, return to the first
PR's kind of review. Don't adjust the answer until the engine passes.

The checker compares raw entity values and explicitly selected typed batch
buffers and fitted-state records. An additional output shape needs reviewed
comparison support and tests showing that wrong or missing values fail; a
selected projection does not certify unasserted outputs.

## Bugs, refactors, and changes to this process

A bug whose answer is unambiguous under an existing rule can use one PR. Add a
small reproducing case and a neighboring boundary case, link the issue and
guarantee, and show the wrong result on current `main` and the right result on
the PR. CI executes the proposed version only; the reviewer checks the base/head
proof. Adding a reproducing golden case needs a harness owner's approval. If
meaning is unsettled, settle it first.

An explicitly agreed behavior change may update the rule, examples, and code in
one PR approved by a harness owner; explain compatibility. A temporary loss of
support uses the same route: link a tracking issue and explain the affected
requests and recovery plan in the guarantee and PR. The engine must refuse
safely, not silently return a different answer. Normal new-feature refusals
aren't regressions.

A refactor preserves the answers and support expectations. Changing this guide,
the loader, comparisons, or CI needs a harness owner's approval and tests that
demonstrate an allowed result and a rejected mistake.

## How a PR merges

Two checks are required on every PR into `main`:

- **`kumo-pytorch-gate`** runs the kumo-pytorch checks when the PR can affect
  the package (see [What CI runs](#what-the-commands-and-ci-jobs-execute)) and
  passes at once otherwise.
- **`kumo-core-gpu-ci`** is kumo-engine's GPU check. It starts by itself for
  signed commits; for an unsigned commit a maintainer has to comment
  `/ok to test <full SHA>` after each push, so sign your commits.

Approvals come from [/CODEOWNERS](../../../../CODEOWNERS):

- Every PR needs one approval from a package owner.
- A PR that touches the guarantees, golden cases, conformance machinery,
  `tests/conftest.py`, `AGENTS.md`, `CLAUDE.md`, `harness/` or the kumo-pytorch
  CI also needs approval from a harness owner. GitHub never counts the author's
  own approval, so a harness owner's harness change needs the other harness
  owner.
- A push after an approval dismisses it, rebases included, and every review
  thread must be resolved. Ask for review when the PR is done.

PRs merge by squash. A stack of PRs is rebased onto `main` after the PR below
it merges, and each one is approved again.

**Who merges.** The human author decides when a PR merges. An agent merges only
when its author explicitly tells it to, usually with
`gh pr merge --auto --squash` so the PR merges once the approval and checks
arrive, and never approves on a human's behalf.

**After merge.** Each push to `main` that touches kumo-pytorch, DiskGraph or the
workspace's `pyproject.toml` and `uv.lock` runs the kumo-pytorch tests,
DiskGraph and end-to-end workflows again, and the nightly workflow runs every
day. These catch two PRs that pass alone but fail together, a PR whose checks
ran against an older `main`, and changes elsewhere that break kumo-pytorch. A
failed run opens an issue labeled `kumo-pytorch-ci` for that workflow, or
comments on the open one. The author of the first failing commit fixes it; a
commit still failing after one working day is reverted. Nightly failures are
triaged by the maintainer on point that week.

## PR evidence and checks

Follow kumo-engine's [PR template](../../../../.github/PULL_REQUEST_TEMPLATE.md)
and keep its change-map comment current
([review requirements](../../../../docs/development/review.md)). Link the rule
and examples, explain which engines must answer or refuse, show the relevant
commands and results, and list limitations. Agents also add a Self-review
section to the description ([self-review.md](self-review.md)).

Before pushing, run from the package directory:

```bash
python -m conformance
python -m pytest -q
uvx ruff@0.17.3 check .
uvx ruff@0.17.3 format --check .
uv build --package kumo-pytorch
python ../../../.github/scripts/kumo-pytorch-policy.py --base origin/main
```

The last command runs the content policy against your local `origin/main`
(fetch it first). kumo-engine's `Linting` workflow checks that every tracked
`.py`, `.yml`, `.toml` and other file type listed in
`/scripts/license_header_policy.json` starts with this header, after a shebang
or encoding line if any (`python scripts/check_license_headers.py` from the
repository root):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Run the DiskGraph command from [Start here](#start-here) for changes that affect
it. The self-review loop runs an independent reviewer agent in a fresh context
([independent-review.md](independent-review.md)). Record unresolved findings;
an agent's approval is never a human approval.

### What the commands and CI jobs execute

`python -m conformance` runs the validation and machinery tests and the
in-memory obligations: golden cases, generated properties and the corpus.
`--engine diskgraph` requests native obligations; `--engine all` requests both.
Native imports are required even for a native refusal case, so missing packages
never substitute for a tested native setup.

The command supplies `--conformance-complete`. Before collection, its pytest
hook derives the required test IDs from the independently loaded cases,
expectations and property obligations, not from the selected tests, and rejects
missing obligations at the end. Filtering out behavior tests can't pass because
the machinery tests still pass. Plain targeted pytest remains available without
this requirement. Tests outside `tests/conformance/` may skip on CPU when an
optional dependency (DiskGraph or PyG) or a GPU is missing. The DiskGraph jobs
run with `--require-imports`, so a skip for a missing dependency fails there;
only GPU skips remain.

**`kumo-pytorch-gate`** (`/.github/workflows/kumo-pytorch-gate.yml`) runs on
every PR. Its first job decides whether the PR can affect kumo-pytorch: a
change to this package, `/packages/diskgraph`, `/scripts/diskgraph`, the
kumo-pytorch workflows, actions and policy script, the DiskGraph wheel
workflow, or the workspace's `pyproject.toml` or `uv.lock`. If so, it runs:

- **`CPU / kumo-pytorch-tests`** (`kumo-pytorch-tests.yml`): conformance on the
  in-memory engine, the test suite, Ruff lint and format checks, and a wheel
  build.
- **`DiskGraph / …`** (`kumo-pytorch-diskgraph.yml`): one DiskGraph wheel built
  from the commit's `/packages/diskgraph` (cached by its source tree, so most
  PRs reuse it), then `kumo-pytorch-diskgraph` runs conformance on both engines
  and the whole suite with `--require-imports`; `kumo-pytorch-notebooks`
  executes the `examples/` notebooks and requires their outputs to match the
  committed files; `kumo-pytorch-install` builds the sdist and the wheel from
  it, installs it with DiskGraph's wheel into a fresh environment, and runs the
  README quick start and the notebooks from that install.
- **`kumo-pytorch-policy`**: the content policy, on the PR's changes since its
  merge base with `main`.

The final **`kumo-pytorch-gate`** job fails unless all of them succeeded, or
none had to run.

The other workflows aren't required:

- **`kumo-pytorch end-to-end`** (`kumo-pytorch-t2a.yml`; after merges to `main`
  that can affect kumo-pytorch, on manual dispatch, and on PRs touching `e2e/`,
  `tests/e2e/` or its workflow) runs the H&M flows in `e2e/flows/` on DiskGraph
  at S: train, persist, then batch inference and serving in fresh processes,
  for a GNN and for PRAGMA. Lifecycle checks are exact; model checks use the
  bands stated in `tests/e2e/tiers.py`. Each flow and seed trains in its own job
  (`python -m tests.e2e.runs`, the matrix from `tests/e2e/tiers.py`); one job
  then runs the checks over all of them. Locally, `pytest --t2a tests/e2e` does
  both in one process. A failure uploads the predictions and metrics.
- **`kumo-pytorch nightly end-to-end`** (`kumo-pytorch-t2b.yml`; nightly at
  09:00 UTC, on manual dispatch, and on PRs touching that workflow,
  `tests/e2e/tiers.py` or `tests/e2e/runs.py`) runs the same flows and checks
  at M, training on a fixed sample of the anchors and scoring every test anchor,
  with the M bands in `tests/e2e/tiers.py`; and, in parallel,
  `python -m conformance --engine all --seeds 12 --seed-start K`: every
  property on 12 new seed batches a day on both engines (K from the date,
  logged in the step summary), plus the pinned batches. Locally,
  `pytest --t2b tests/e2e`. It uploads the predictions and metrics.

## Current limits

- Cases use at most eight integer-second anchors. The checker asserts raw
  entity values and explicitly selected encoded buffers and fitted-state
  records, including exact array dtypes, shapes, values and boundaries. Native
  fixtures use declared Arrow schemas, including empty and all-NULL tables.
  Unasserted outputs are not established coverage; additional formats require
  reviewed loader/comparison support and tests.
- Numeric comparison is exact and treats 2 and 2.0 as the same raw number;
  booleans remain distinct.
- This doesn't certify tensor layouts, model quality, scale, or every existing
  API feature. Only the named engine/case obligations are checked.
