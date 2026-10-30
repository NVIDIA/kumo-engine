---
name: skill-zips
description: Check, or rebuild and attach, the kumo-data-science agent skill zips on a kumo-engine GitHub release.
---

# Skill zips

Every `kumo-engine-v<version>` release carries one agent skill zip per host, attached by the
`skill-zips` job in `.github/workflows/kumo-engine-wheel.yml`:
`kumo-data-science-coco.zip`, `kumo-data-science-generic.zip`, `kumo-data-science-genie.zip`.
The docs download button and install scripts link to them by tag. Use this skill to confirm a
release's zips are there and correct, or to attach them when the job did not.

Argument: the version, e.g. `1.1.0`. The tag is `kumo-engine-v<version>`.

## Steps

1. List the release's assets:
   `gh release view kumo-engine-v<version> -R NVIDIA/kumo-engine --json assets --jq '.assets[].name'`

2. Rebuild the zips from the tag, in a scratch worktree so no local edit leaks into the build:
   ```bash
   git fetch origin tag kumo-engine-v<version>
   git worktree add --detach <scratch>/tag kumo-engine-v<version>
   python3 <scratch>/tag/agent-skills/scripts/package_skill.py --zip <scratch>/built
   ```
   The tag must carry `package_skill.py --zip`. A release cut before it has no skill zips and
   cannot be given any this way; say so and stop.

3. If the release has all three zips, download them into a new empty directory
   (`gh release download kumo-engine-v<version> -R NVIDIA/kumo-engine -p 'kumo-data-science-*.zip' -D <scratch>/released`)
   and compare each with its rebuild (`cmp`). The build is reproducible, so any difference means
   the attached zip was not built from the tag. Report the result, with the `SKILL-BUILD` line from
   each zip's `kumo-data-science/SKILL.md`.

4. If any zip is missing, show the user the rebuilt zips (file counts, `SKILL-BUILD`) and **ask
   before uploading**: attaching to a public release publishes the files. On approval:
   `gh release upload kumo-engine-v<version> -R NVIDIA/kumo-engine <scratch>/built/*.zip`
   Never pass `--clobber` over a zip that step 3 found to differ without telling the user first.

5. Remove the scratch worktree: `git worktree remove <scratch>/tag`.

Never upload zips built from a branch, or from a checkout with local changes: the zip must be the
skill built from the release's own commit, as the wheel's copy is.
