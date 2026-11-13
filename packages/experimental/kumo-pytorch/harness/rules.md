# Rules

The conformance suite checks *outcomes*. These rules govern *decisions*: the
choices an agent makes the same way every time, including when to stop and ask
instead of choosing.

- Each rule has one right answer for kumo-pytorch and names its enforcement:
  **CI** (a check fails), **Review** (the independent reviewer's checklist,
  [independent-review.md](independent-review.md)) or **Owners** (approval
  by a harness owner through CODEOWNERS, and the merge flow in
  [contributing.md](contributing.md#how-a-pr-merges)).
- A rule that can become a test becomes a test. A rule with no enforcement is
  deleted.
- IDs are stable. Cite them in self-review notes, review findings and escape
  records.

## Semantics and the contract

| ID | Rule | Enforced by |
|---|---|---|
| **S1** | **Spec before code.** No behavior is merged before its guarantee text and golden cases are approved. The first PR may add only the API surface and refusal needed to express the request. | Owners |
| **S2** | **Refuse rather than guess.** When an engine can't do exactly what a contract asks, refuse at `check()` with the path and a suggestion. Never silently approximate, drop, coerce or truncate. | CI: refusal assertions; Review |
| **S3** | **Point-in-time at every hop.** Historical cutoffs include equality (`<= cutoff`); bounded windows use `start < time <= stop`. Cutoffs apply at every timed table on a path, not only the destination. A change that touches time adds future-row cases at every hop. | Review |
| **S4** | **No engine-dependent output.** Units, dtypes, value ranges and ordering guarantees are part of the contract, and the contract binds the engine. DiskGraph is the shipped engine: the goldens and guarantees are checked on it, and its output may vary only where a guarantee leaves room. The in-memory engine is a test tool, not a peer or a reference; its output is compared with DiskGraph's under D1–D4 and never decides what is correct. | CI: goldens on DiskGraph; Review |
| **S5** | **One source of truth for support.** Refusals come from the engine's own checks (`check_sources()`, `check_contract()`); DiskGraph's come from its declared unsupported list; no CI-only guards, no ad-hoc refusals elsewhere. | Review |
| **S6** | **A behavior change is a spec change.** Changing a documented guarantee, a golden case or a default that affects output needs a harness owner's approval. | Owners; Review |

## Tests and the oracle

| ID | Rule | Enforced by |
|---|---|---|
| **V1** | **Repro first.** A bug fix starts with a golden case that fails on the merge base; the PR shows the wrong base result and the right head result. | Review |
| **V2** | **Generalize.** Add a neighboring boundary case next to the reproducing one, so the fix covers the bug's neighbors, not only the point. | Review |
| **V3** | **Never weaken the oracle.** Don't edit goldens, guarantees, expectations or comparison helpers to make code pass. If the oracle looks wrong, escalate (E2). | CI: `kumo-pytorch-gate` content policy (expectations only move forward); Owners |
| **V4** | **No skips.** A required case that is skipped, deselected or expected to fail fails the run. Missing dependencies don't prove support. | CI |
| **V5** | **Deterministic or it's a bug.** Seed everything; a flaky case is a defect, never noise to retry. | Review |
| **V6** | **Tolerances come from the spec,** never from observed output. | Review; Owners |
| **V7** | **Answers are independent of engines.** A golden answer is calculated by hand from the rule and the rows, never copied from an engine run. | Review; Owners |

## Differential testing

The in-memory engine is a test tool, also handy for trying things out on tiny data. The
spec says nothing about it, and it is not the oracle: the guarantees and
goldens are. The harness runs the same golden cases and generated properties on
the in-memory engine and on DiskGraph and compares the two. Triage every
discrepancy between them, and every golden or property failure on only one of
them, with these rules, in order.

| ID | Rule | Enforced by |
|---|---|---|
| **D1** | **Check the rule first.** Compare each engine's output with the golden answer or the guarantee text. If DiskGraph disagrees with the rule, it is a DiskGraph bug: fix it (V1), make DiskGraph refuse the construct at `check()` and record the refusal in `expectations.json` (S2; a refusal that replaces an expected answer is a temporary loss of support, which needs a harness owner's approval, a tracking issue and a recovery plan: see [contributing.md](contributing.md#bugs-refactors-and-changes-to-this-process)), or record a known gap in `registry.KNOWN_GAPS` with a repro under `conformance/known_gaps/`. | CI: goldens and known-gap replays on DiskGraph; Review |
| **D2** | **DiskGraph matches the rule and the in-memory engine doesn't: an in-memory bug.** Fix it in the in-memory engine, or record it as a known in-memory divergence in `registry.IN_MEMORY_DIVERGENCES` with a repro under `conformance/in_memory_divergences/`. An in-memory bug never changes a guarantee, a golden answer or a DiskGraph expectation. A DiskGraph fix that exposes one records the divergence rather than waiting for the in-memory fix. `expectations.json` has no divergence entry for a golden case, so an in-memory bug on a golden case is fixed in the in-memory engine in the same PR. | CI: divergence replays; Review |
| **D3** | **The rule leaves the behavior open: an allowed discrepancy.** Where a guarantee leaves the behavior unspecified (which tied neighbor a fanout keeps, the rounding of approximate floating means and moments, an order the spec doesn't fix), the comparison tolerates exactly that, explicitly. The tolerance names the guarantee that leaves room and the reason, next to the code, and is listed in the allowed-discrepancy table in `conformance/README.md`. A tolerance never covers more than the guarantee leaves open (V6), and never stands in for a bug on either engine. | Review |
| **D4** | **No rule covers the behavior: a spec question.** Escalate (E1) with both outputs and a proposed rule. Don't resolve it by copying either engine's output into a golden (V7), by adding a tolerance, or by recording a gap or a divergence. | Review; Owners |

The three records stay separate. Known gaps (`GAP-*`) are DiskGraph's
shortfalls against the spec and nothing else. Known in-memory divergences
(`DIV-*`) are in-memory bugs; they replay like known gaps, so a fix shows up as
a failing test that lifts the entry, and they keep the generator out of their
region on the in-memory engine only. Allowed discrepancies are not defects:
they live in the comparison code as named tolerances and in the README table.

## Scope and code health

| ID | Rule | Enforced by |
|---|---|---|
| **C1** | **One coherent goal per PR, at whatever size it takes.** Include the refinements the goal needs; put unrelated goals in separate PRs. Size is not a criterion. | Review |
| **C2** | **Reuse before adding.** Search for an existing mechanism before adding one; never add a second mechanism for the same concept. | Review |
| **C3** | **Delete what a change replaces.** No dead code, no parallel old path, no unused parameters. | Review |
| **C4** | **Public surface is a contract.** Public API, config names and output schemas change only with an explicit compatibility note. | Review |
| **C5** | **Docs and examples move with behavior.** A change to user-visible behavior updates the affected guarantee text, docs and examples in the same PR. | Review |
| **C6** | **Errors name the stage, the resource or the path.** No bare exceptions from user-reachable code. | Review |

## Dependencies

| ID | Rule | Enforced by |
|---|---|---|
| **X1** | **DiskGraph changes run DiskGraph for real.** A change to `/packages/diskgraph` builds DiskGraph from that commit and `python -m conformance --engine all` passes on it: the goldens and properties on DiskGraph, and the differential comparison with the in-memory engine. | CI: `kumo-pytorch-gate` |
| **X2** | **No new third-party dependency without a stated need** and a check that it installs in the target environments (CPU, GPU, partner notebooks). | Review |

## Escalation: stop and ask

Escalate instead of deciding when:

| ID | Trigger |
|---|---|
| **E1** | **Spec ambiguity:** two reasonable readings of a guarantee give different outputs. |
| **E2** | **The oracle looks wrong:** a golden case contradicts the rule, or a fix needs a change to a guarantee, a golden case or the conformance machinery. |
| **E3** | **Public surface or user-visible behavior** would change (S6, C4). |
| **E4** | **A reported failure can't be reproduced** after a few distinct attempts. |
| **E5** | **The self-review loop doesn't converge** within its budget ([self-review.md](self-review.md)). |
| **E6** | **Two rules conflict,** or following a rule would clearly cause harm in this case. |

**How:** comment on the issue or PR with the question, the options, the
evidence for each and a recommendation. Continue work that doesn't depend on
the answer. Record the resolution where the next agent will find it: in the
guarantee if it settled meaning, in this file if it settled a recurring
decision.

## How rules change

- Rules come from escapes: when a defect got through because of a judgment gap
  (a duplicated mechanism, an unjustified tolerance), the fix adds or sharpens
  a rule.
- Rules that never fire get reviewed: delete them, or fix their enforcement.
- Rules that only compensate for a model weakness (step lists, budgets) are
  expected to expire as models improve; rules that encode what the project
  wants (S3, V3, human approval of spec changes) stay.
- Changing this file needs a harness owner's approval.
