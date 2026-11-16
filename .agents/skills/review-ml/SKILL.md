---
name: review-ml
description: Review PR code changes against ML standards. Use when reviewing PRs that touch ML pipeline code (packages/runtime/kumo-core, apps/studio/kumo/pquery, apps/studio/kumo/train, apps/studio/kumo/task, apps/studio/kumo/automl, apps/studio/kumo/explain). Applies lessons learned from past ML reviews.
---

# ML Code Review

You are a terse ML code reviewer. Be brief. No filler.

## Step 1: Get the diff

Run `gh pr diff <number>` to get the code changes. Filter to only these paths:
- `apps/studio/kumo/pquery/`, `apps/studio/kumo/train/`, `apps/studio/kumo/task/`
- `apps/studio/kumo/automl/`, `apps/studio/kumo/explain/`
- `packages/runtime/kumo-core/src/kumo_core/`, `apps/studio/kumo/config/predictive_query.py`, `apps/studio/kumo/config/train.py`
- `apps/studio/kumo/graph/table_data.py`
- `apps/studio/bin/kumo-auto`, `apps/studio/bin/kumo-pred`

If NO files match those paths, reply with exactly:

> No ML pipeline files changed. `/review-ml` not applicable.

Nothing else. Stop.

## Step 2: Review

For each matching file, check the diff against the linked review conventions. Only comment on violations you actually see — do not repeat rules that aren't violated.

## Step 3: Post comments

Post only when the user explicitly requests a posted GitHub review.
Otherwise return findings in the conversation. For an authorized posted review,
submit a single GitHub PR review with inline comments on the specific diff lines:

```bash
gh api repos/{owner}/{repo}/pulls/{number}/reviews \
  --method POST -f event=COMMENT \
  -f body='ML Review' \
  -f 'comments[][path]=file.py' \
  -f 'comments[][line]=42' \
  -f 'comments[][side]=RIGHT' \
  -f 'comments[][body]=Comment'
```

If that fails, fall back to one short summary comment. Never post more than 15 lines total in fallback mode.

---

## Rules

Read [ML review conventions](../../../docs/development/ml.md).

## Reference

Pattern examples: `docs/development/ml-review-patterns.md`
