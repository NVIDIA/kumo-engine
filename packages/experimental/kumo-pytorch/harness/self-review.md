# Self-review

The author agent runs this loop before opening a PR, and decides on every push
whether to run it again. It gives the human reviewer a change that has already
been checked against the rules, and notes that say how.

## When

- **Before opening a PR,** for every PR.
- **On every later push,** decide whether the change since the last
  self-reviewed commit needs the loop again (see [Pushing after the PR is
  open](#pushing-after-the-pr-is-open)).

## The loop

1. **Classify the change.** Which paths does it touch? Paths owned by the
   harness owners in [CODEOWNERS](../../../../CODEOWNERS) need their approval.
   Which kind is it: new meaning, an implementation of approved meaning, a bug
   fix, a refactor, a dependency change? The kind decides the evidence
   ([contributing.md](contributing.md)).
2. **Run the checks locally** until they pass: conformance, the test suite,
   lint and format, the doc-code and link tests (part of the suite), the
   package build. Run DiskGraph
   for changes that can affect it. The commands are in [contributing.md](contributing.md).
3. **Walk the rules** ([rules.md](rules.md)) against the diff and add what's
   missing. Does it touch time (S3)? Refusals (S2, S5)? Output or defaults
   (S6, C4)? Is it a fix (V1, V2)? Does it add a mechanism (C2) or leave dead
   code (C3)? Do docs and examples still match behavior (C5)?
4. **Run the independent reviewer** ([independent-review.md](independent-review.md))
   in a fresh context. Give it the diff, the relevant guarantees, the rules
   and the draft notes, not the author's reasoning.
5. **Handle each finding.**
   - Reproducible as a case? Add the case (a golden case needs a harness
     owner's approval), fix, and go back to step 2.
   - Otherwise fix it, or record in the notes why it doesn't apply.
6. **Exit** when all hold:
   - the checks pass locally;
   - the reviewer has no blocking findings;
   - the evidence for the change's kind is present;
   - the notes are complete.

**Budget:** three reviewer rounds. If it doesn't converge, stop and escalate
(E5) with the notes so far: the task, the spec or the harness is unclear, and a
human should see that.

## The notes

Add a **Self-review** section to the PR description, after the sections of
kumo-engine's [PR template](../../../../.github/PULL_REQUEST_TEMPLATE.md). The
template doesn't have one, so write it yourself:

```markdown
## Self-review

- Rounds: 2. Reviewer: fresh context, same diff, guarantees and rules.
- Fixed: added a future-row case at the second hop (S3); removed the old
  helper (C3).
- Dismissed: "unbounded loop in X": bounded by `max_samples`, covered by
  C-ROWS-002.
- Not re-reviewed: <for later pushes, what changed and why no new loop>.

Reviewed at: <full SHA of the head commit>
```

`Reviewed at:` names the commit the notes cover. The human reviewer checks
that it matches the PR's head commit, or that the notes say why the later
commits didn't need the loop.

## Pushing after the PR is open

Every push moves the head past `Reviewed at:`. Decide each time:

- **A tiny follow-up to a reviewed commit that doesn't change behavior** (a
  typo, a comment, formatting, a test name): no new loop. Update `Reviewed at:` to the new head and add a
  line saying what changed.
- **Anything that changes behavior, adds code, or touches a new area:** run the
  loop again, then update the notes and `Reviewed at:`.

Update `Reviewed at:` after pushing (the new head SHA is known then).

## Must not

- Don't name a commit in `Reviewed at:` that the notes don't cover.
- Don't weaken the oracle to exit the loop (V3); escalate (E2) instead.
- Don't treat the reviewer's approval as a human approval.
