# Agent instructions

kumo-pytorch (`kumo.pytorch`) turns a declarative data-loader contract over
relational tables into point-in-time-correct training batches. It ships one
engine, DiskGraph, and the spec is checked on it. The in-memory engine is a
test tool that the harness compares with DiskGraph; it is never the reference
for what is correct, and discrepancies are triaged by
[harness/rules.md](harness/rules.md) D1–D4.

kumo-pytorch is developed by its maintainers and isn't accepting outside
contributions yet; bug reports and suggestions are welcome as GitHub issues.

Paths below are relative to this package (`packages/experimental/kumo-pytorch/`)
unless they start with a `/`, which means the repository root.

## Read first

| Doc | What it's for |
|---|---|
| [harness/contributing.md](harness/contributing.md) | What the harness enforces today; the procedure: changing behavior, implementing it, bug fixes, how a PR merges, what CI runs, commands |
| [harness/rules.md](harness/rules.md) | The decision rules (S, V, D, C, X) and when to stop and escalate (E1–E6) |
| [harness/self-review.md](harness/self-review.md) | The loop to run before every PR and the decision on every push; the self-review notes and their `Reviewed at:` line |
| [harness/independent-review.md](harness/independent-review.md) | The fresh-context reviewer the loop runs, and its checklist |
| [conformance/README.md](conformance/README.md) | Guarantee and golden-case file formats; engine expectations |
| [/docs/development/review.md](../../../docs/development/review.md) | kumo-engine's PR template and the change-map comment every PR keeps |

## The harness in five lines

1. Guarantees (English rules) and golden cases (tiny, hand-calculated answers)
   are the spec. New meaning is approved before it's implemented.
2. Every PR needs one approval from a package owner. Changes to guarantees,
   goldens, the conformance machinery, the harness docs and the kumo-pytorch CI
   also need approval from a harness owner who isn't the author (see
   [/CODEOWNERS](../../../CODEOWNERS)). The `kumo-pytorch-gate` check must pass.
3. Never weaken the oracle to make code pass; escalate instead.
4. Run the self-review loop before every PR, and on every push decide whether
   to run it again; keep the PR description's Self-review section current.
5. Merge only when the PR's author explicitly says to. Never approve on a
   human's behalf.

kumo-engine's PR template has no Self-review section; add one to the PR
description yourself ([harness/self-review.md](harness/self-review.md)).

Preserve edits that aren't yours. If something here conflicts with a direct
instruction from the PR's author, ask.
