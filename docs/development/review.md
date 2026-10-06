# Review and pull requests

Read [CODEOWNERS](../../CODEOWNERS) for component ownership. Review a change
against actual consumers, preserved behavior, and appropriate validation.
Keep unrelated improvements separate.

## PR change map

Follow the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md). Keep one
conversation comment beginning with `<!-- kumo-pr-change-map -->`, account for
every changed file, and link to it from the description. Update that comment
after new commits; keep summaries in the description and rationale in the map.
The PR author or authorized bot must maintain it.

To validate locally:

```bash
gh api repos/kumo-ai/kumo/pulls/NUMBER > PR.json
gh api --paginate --slurp repos/kumo-ai/kumo/pulls/NUMBER/files | jq 'add' > FILES.json
gh api --paginate --slurp repos/kumo-ai/kumo/issues/NUMBER/comments | jq 'add' > COMMENTS.json
node .github/scripts/pr-change-table.cjs PR.json FILES.json COMMENTS.json
```

The check validates metadata and file coverage; reviewers assess the design
and explanation. New commits and comment changes rerun it using the checker
on the default branch.
