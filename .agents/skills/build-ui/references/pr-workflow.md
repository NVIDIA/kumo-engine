# Build UI PR Workflow

Use this file during Phase 9 for PR body construction, screenshot handling, and breaking-change checks.

## Required PR Sections

Use the repo PR template at `.github/PULL_REQUEST_TEMPLATE.md`.

Required sections:

- `Fixes #`
- `Summary`
- `Testing Done`
- `Breaking Changes`
- `Documentation Update`
- `Unit Tests`

If the change is a hotfix, also include:

- symptom
- customer impact
- affected version

If the PR modifies shared primitives or has cross-repo dependencies, add a `Notes` section with dependencies and merge order.

## Breaking Change Detection

Before creating the PR, scan the diff for:

- removed or renamed exports from shared modules such as `@ui/`, `hooks/`, `stores/`, or `utils/`
- changed function signatures on exported hooks or components
- removed or renamed `navRoutes` entries
- changed `ReactQueryKeys` enum values
- modified `mockAxios.ts` dispatch patterns that external tests depend on

If none are found, write `None.` in the breaking changes section.

## PR Body Rules

- Keep the PR body aligned with the actual repo template.
- Add screenshots, Figma comparison, and a11y notes only when they add signal.
- Do not require screenshots in git to create the PR.
- Prefer the shared image-branch publisher over browser-uploaded attachments so local and headless screenshot handling stay consistent.
- Never commit screenshot files to the branch unless the user explicitly opts in.
- Do not treat screenshot publication as complete until the PR body or PR comment renders the image links.
- If the same dialog or UI pattern changed across multiple surfaces, include a summary table listing each instance and its key attributes.
- If incidental bugs were fixed as adjacent fixes, list them with a short justification.

## Shared Screenshot Publishing Flow

Once Phase 9 screenshot publication is authorized, and screenshots are captured by `ui/scripts/ci-screenshot.mjs` or otherwise recorded in a compatible `results.json` manifest:

1. Screenshots are captured by `ui/scripts/ci-screenshot.mjs` and saved to `/tmp/build-ui-screenshots/{issue}/`.
   The script writes `/tmp/build-ui-screenshots/{issue}/results.json` with per-route status, matched selectors, URLs, and artifact paths.
   If the capture fails, the script also saves failure artifacts under `/tmp/build-ui-screenshots/{issue}/failed/`.
2. Run `ui/scripts/build-ui-publish-screenshots.mjs` to commit primary PNGs to the `images` branch via the GitHub Contents API and obtain commit-pinned raw URLs.
3. The script writes `screenshot-section.md` with inline `<img>` tags referencing those URLs.
4. In headless CI, the workflow posts a PR comment (or issue comment if no PR exists) and uploads the screenshot directory as a workflow artifact for debugging and retention.
5. In local PR creation, create the PR first, then run the same publisher with `--pr-number <number> --patch-pr --comment-type pr --comment-target <number>` so the PR body and comment use the same screenshot section as CI.
6. Screenshots render directly on the PR page and inherit repository visibility. Images committed to a public repository are public; review them for credentials and private data before publishing.
7. No browser-based GitHub upload or attachment URL extraction is needed.

This flow is simpler and more reliable than the browser-based upload flow. Use it for both headless CI and local build-ui PR creation whenever screenshots are available and GitHub auth can write to the repo.

Example publish command from the repository root after PR creation; replace `OWNER/REPO` with the intended destination:

```bash
npm --prefix ui run screenshots:build-ui -- \
  --repo OWNER/REPO \
  --issue {issue} \
  --plan /tmp/build-ui-artifacts/{issue}/screenshot-plan.json \
  --pr-number {pr} \
  --publish
```

The helper runs `ui/scripts/ci-screenshot.mjs` first and then calls `ui/scripts/build-ui-publish-screenshots.mjs` with the standard manifest, resolution, override, and section paths. Use `--skip-capture --publish` only when `results.json` already exists.

## Failure Handling

- If screenshot image upload succeeds but body patching fails, leave the screenshot section in a PR comment and report the failure.
- If screenshot image upload succeeds but commenting fails, preserve the generated `screenshot-section.md` and classify the failure explicitly.
- If body patching succeeds but render verification fails, retry once and then report `screenshot-render-check: failed`.
