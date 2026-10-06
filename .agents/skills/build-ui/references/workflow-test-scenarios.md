# Build UI Workflow Test Scenarios

Use these scenarios to validate the workflow itself after changes to the skill.

## Existing-Page Bug Fix

- existing page or component
- no new route
- existing tests and mocks

Expected result:

- adds regression coverage
- does not over-wire unrelated files

## Greenfield Page With Route Wiring

- new page with route, loader, and sidebar entry

Expected result:

- catches `navRoutes`, `Routes.tsx`, `Loaders.ts`, and `useSidebarData.ts`

## Missing Generated Hook

- endpoint not present in `gen/v1/`

Expected result:

- checks `gen/v0/`, then spec or codegen status, then stops or narrows scope explicitly

## Figma-Backed UI Change

- issue includes a Figma link

Expected result:

- extracts useful design guidance
- proceeds safely if Figma access fails

## No Backend Or No Browser Tooling

- backend or browser tooling unavailable

Expected result:

- code and test validation still works
- browser verification is clearly marked partial or skipped

## PR Creation Path

- implementation and validation succeeded
- push, rebase, or label choice still required

Expected result:

- pauses at Checkpoint 2
- does not create a PR without explicit approval

## Headless CI Mode

- workflow triggered from a GitHub comment or workflow dispatch
- no interactive checkpoint approval available
- browser testing uses Playwright screenshot fallback instead of interactive browser tooling

Expected result:

- applies `references/headless-mode.md`
- auto-approves only when no hard blocker exists
- uses headless reporting behavior instead of interactive checkpoint prompts

## PR Mode Continuation

- target is an existing PR
- linked issue may or may not exist
- branch already exists

Expected result:

- inspects PR claims vs verified evidence
- chooses `inspect-only`, `continue-existing-pr`, or `supersede-with-new-work` explicitly
- continues on `{headRefName}` when the user wants to continue existing work

## Bare Number Target

- target is a bare integer such as `27326`
- no explicit PR URL or PR-prefixed form is provided

Expected result:

- treats the target as issue mode
- never infers PR mode from a bare integer
- uses the existing issue-mode related PR check to discover overlapping open PRs
- requires an explicit PR URL or PR-prefixed form for direct PR inspection or continuation

## Shared Primitive Modification

- ticket changes a shared `@ui/` component
- the change adds new optional props and updates all callers
- many consumer files and tests need updating

Expected result:

- enumerates all callers before editing the shared component
- new props are optional with defaults
- all callers are updated consistently
- a grep for the old pattern confirms nothing was missed
- ripple test updates are planned and executed
- incidental bugs found in callers are logged as adjacent fixes

## Shared Component Blast Radius Review

- change touches `src/components/@ui/` or another widely reused module
- blast radius is not obvious from the issue itself

Expected result:

- callers are enumerated before implementation
- Checkpoint 1 records the blast radius explicitly
- workflow recommends a split when caller count or feature-area spread is too large

## Catalog Dependency

- design requires a catalog variant that does not exist yet
- implementation cannot proceed cleanly without it

Expected result:

- recon identifies the catalog gap
- Checkpoint 1 stops and presents the decision
- if proceeding, the PR includes the in-tree catalog change and affected UI callers
- no silent approximation with ad hoc inline styles

## Large PR Splitting

- ticket maps to many modified files across multiple feature areas

Expected result:

- Checkpoint 1 recommends splitting
- presents a split proposal
- user decides whether to split or proceed

## Multi-Caller Copy Or Style Update

- ticket updates dialog text or button variants across many locations
- some locations have incidental copy bugs

Expected result:

- all instances are updated consistently
- grep confirms no old patterns remain
- incidental bugs are recorded as adjacent fixes
- the PR description includes a summary table for the updated pattern
- mock dispatch entries are updated if test behavior changed

## Resume With Partial Work

- a previous run stopped after implementation or validation
- branch contains intentional partial progress
- user reruns with `--resume`

Expected result:

- reuses the existing branch
- compares requested resume stage against run-state artifact and branch state
- fixes forward instead of discarding partial changes

## Resume Mismatch

- user requests `--resume validation`
- branch state or run-state artifact shows planning or implementation is incomplete

Expected result:

- mismatch is reported explicitly
- workflow trusts actual branch state over requested stage
- safe next action is given instead of blindly skipping phases

## Pre-Existing Failure Claim

- a validation command fails
- the failure is claimed to be unrelated to the current diff

Expected result:

- workflow provides concrete evidence for `pre-existing`
- if evidence is missing, failure stays `introduced-by-change` or `flaky-or-unclear`

## Final Diff Sanity

- implementation and validation are complete
- PR creation is in scope

Expected result:

- generated files, lockfile deltas, screenshot staging, and debug artifacts are checked
- scope drift is caught before PR creation
