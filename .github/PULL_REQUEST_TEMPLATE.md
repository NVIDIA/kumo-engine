Fixes #\<issue_number>

Design discussion or issue: *public link if applicable*

## Summary
For all **moderate and large changes**, please provide a list of what specific changes that are included in this PR. For larger changes, include "how" as well, detailing design patterns you are using or any other information which helps reviewers get a high level view of the change before they look into the code.

## Change map
Create one PR conversation comment starting with `<!-- kumo-pr-change-map -->`
and containing the table below. Edit that same comment when the diff changes.
Link to the comment here for reviewers. The PR author or authorized
`github-actions[bot]` automation must author it; other marked comments are ignored.
Do not duplicate the table in the description or commit messages; concise
summaries and rationale can remain there.

In that comment, identify the observed failure or explicitly requested release
requirement, existing behavior, smallest sufficient change, focused verification,
and stopping condition. Keep optional refinements separate.

```markdown
<!-- kumo-pr-change-map -->
## Change map
| File | Function / scope | What changed | Why |
| --- | --- | --- | --- |
| `<path>` | <functions or scope> | <behavior change> | <reason> |
```

Use exact repository-relative paths in backticks, including tests, docs,
generated files and deletions; use the new path for renames. Explicitly named
related files may share a row. At <=1000 additions + deletions describe functions
or non-function scope; at >=1001 the Function / scope column may be omitted.
Remove this example from the description after creating the comment.

## Testing Done
If this ticket resolves a **P0** bug, please create a test plan to avoid further regression or explain why a test plan is not applicable.

For any changes, it's the code owner's responsibility and judgement to create new tests, unit or integration, and to conduct testing locally or in CI.

## Hotfix Context
For detailed instructions on release handeling see [README.md](https://github.com/kumo-ai/kumo/blob/master/scripts/release/README.md)

**If P0 Fix on master:**
- Related P0 bug / incident ticket(s): <!-- GitHub issue links -->
- Brief symptoms / customer impact: <!-- Plain language: what breaks for customers? -->
- Affected deployment types: <!-- Omit customer names and private incident details. -->
- Affected versions: <!-- Must check: 1) earliest version bug appeared, 2) ensuring forward-ported to all affected versions without gaps -->

**If Backport PR:**
- Related master PR: `<PR_URL>`

## Checklist
- [ ] I listed all the **breaking changes** introduced by the PR
- [ ] I made sure to **update examples and documentation** with my changes (if necessary)
- [ ] I have added and modified unit tests required to cover changes made by this PR.
