---
name: review-ui-catalog
description: Review PR code changes against UI catalog conventions. Use when reviewing PRs that touch components, stories, or tests in kumo-ui-catalog.
---

# UI Catalog Code Review

**IMPORTANT:** Always execute a fresh review every time this skill is invoked, even if you already reviewed this PR before. Never skip steps or silently do nothing — the user expects a full review on every invocation. Ignore any previous review output in the conversation.

You are a terse UI catalog code reviewer. Be brief. No filler.

## Severity Levels

- **[HIGH]** — Always flag. Bugs, broken patterns, or hard conventions with zero exceptions.
- **[MEDIUM]** — Flag with context. Strong conventions that may have rare justified exceptions.

Only comment on violations you actually see. Do not repeat rules that aren't violated.

## Invocation

```
/review-ui-catalog <pr_number>              # Posts review to GitHub
/review-ui-catalog <pr_number> --dry-run    # Prints findings to stdout only
/review-ui-catalog <pr_number> --force      # Posts even if a previous review exists
```

When `--dry-run` is passed, skip posting. Print findings to stdout:

```
[SEVERITY] **Rule Name** (file:line) explanation
```

## GitHub App Context

When triggered by `@claude` in a GitHub PR comment, your response is already posted as a PR comment automatically. In this context:

- Skip Step 0 (duplicate check) — always run the review fresh.
- Skip Step 4 (posting) — do NOT run `gh api` or `gh pr comment`. Just output the review as your response.
- All other steps (diff, context, pre-scan, rule review) run normally.

## Step 0: Setup

### Resolve repository

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

### Check for duplicate reviews

```bash
EXISTING=$(gh api "repos/${REPO}/pulls/<number>/reviews" \
  --jq '[.[] | select(.body | startswith("**Catalog Review**"))] | length')
```

If `$EXISTING > 0` and `--force` was NOT passed, reply:

> Catalog Review already posted on this PR. Run with `--force` to post again.

## Step 1: Get the diff

```bash
gh pr view <number> --json title,body,isDraft
gh pr diff <number>
```

PR diffs use monorepo-root paths. Filter to these paths:

- `packages/kumo-ui-catalog/src/components/ui/`, `packages/kumo-ui-catalog/src/components/blocks/`
- `packages/kumo-ui-catalog/src/stories/`, `packages/kumo-ui-catalog/src/icons/`, `packages/kumo-ui-catalog/src/animations/`
- `packages/kumo-ui-catalog/src/hooks/`, `packages/kumo-ui-catalog/src/lib/`, `packages/kumo-ui-catalog/src/utils/`, `packages/kumo-ui-catalog/src/types/`
- `packages/kumo-ui-catalog/src/index.ts`, `packages/kumo-ui-catalog/src/colors.css`

Exclude: `dist/`, `node_modules/`, `*.lock`

If NO files match, reply:

> No catalog source files changed. `/review-ui-catalog` not applicable.

**Comment budget:** 10 comments for normal PRs, 15 for >15 files.

## Step 2: Load context

Always load:

- Root `AGENTS.md` and `packages/kumo-ui-catalog/CLAUDE.md`
- `packages/kumo-ui-catalog/.agents/skills/review-ui-catalog/references/patterns.md` — correct/incorrect code patterns

## Step 2.5: Pre-scan (automated)

Run the helper script to surface mechanical violations:

```bash
bash packages/kumo-ui-catalog/.agents/skills/review-ui-catalog/scripts/diff-scan.sh <number>
```

Output: TSV to stdout (`rule_id\tfile_path\tline_number\tmatched_text`), summary to stderr.

Each line is a confirmed violation — include directly as a review comment. Pre-scan hits do NOT count against the comment budget.

The script catches: `inline-styles`, `hardcoded-colors`, `cn-usage`, `barrel-self-import`, `focus-states`, `any-type`, `strict-compliance`, `story-tags`.

If the script fails, report the failure and review the diff manually; do not treat
a failed scan as zero findings.

## Step 3: Review against rules (LLM)

In the rules below, `src/` means `packages/kumo-ui-catalog/src/`; use the
full repository-relative path when reading diffs or posting line comments.

Skip rules already covered by pre-scan hits — don't double-flag the same line.

### Component Structure

**[HIGH] CVA Pattern** (`cva-pattern`): UI primitives in `src/components/ui/` must use CVA for variants. Flag components that define variant styles without CVA, or use inline conditional classes instead of CVA variants.

**[HIGH] asChild Prop** (`as-child`): UI primitives must accept `asChild` prop and use Radix `Slot` for polymorphic rendering. Flag UI components missing `asChild` support.

**[HIGH] data-slot Attribute** (`data-slot`): All components must have `data-slot="kebab-case-name"` on their root element. Flag components missing this attribute.

**[MEDIUM] Variant Export** (`component-export`): UI primitives should export their `*Variants` object alongside the component when variants are useful to consumers (e.g., `buttonVariants` for styling links as buttons). Not all components need this — only flag when a consumer use case is obvious.

**[HIGH] Props Pattern** (`props-pattern`): UI primitives must use `React.ComponentProps<"element">` for HTML props, not custom interfaces extending HTML elements. Block components should use explicit props interfaces.

**[HIGH] cn() Usage** (`cn-usage`): All className merging must use `cn()` from `@/lib/utils`. Flag manual string concatenation or template literals for className.

### Barrel Export

**[HIGH] Missing Export** (`missing-export`): New components must be exported from `src/index.ts`. Flag new component files without a corresponding export line.

**[HIGH] Type Export** (`type-export`): New types/interfaces intended for consumers must have `export type` in `src/index.ts`. Flag public types only exported from their source file.

### Imports

**[HIGH] Barrel Self-Import** (`barrel-self-import`): Components must NOT import from `src/index.ts` (barrel). Use direct `@/components/ui/` or `@/components/blocks/` paths. Flag any `from "@kumo-ai/kumo-ui-catalog"` or `from "../index"` in source files.

**[MEDIUM] Import Order** (`import-order`): Imports should follow: React/externals → Radix/headless → Internal components → Utils/types. Flag obviously jumbled imports.

### Styling

**[HIGH] Inline Styles** (`inline-styles`): Flag `style={{}}` in component source. Use Tailwind utilities or CSS variables.

**[HIGH] Hardcoded Colors** (`hardcoded-colors`): Flag hex color values, `rgb()`, or `hsl()` in component code. Use Tailwind color classes or CSS variables from `colors.css`.

**[MEDIUM] Focus States** (`focus-states`): Flag `focus:` without `focus-visible:`. Interactive elements must use `focus-visible:` for keyboard-only focus styling.

**[MEDIUM] Disabled Pattern** (`disabled-pattern`): Interactive elements should include `disabled:opacity-50 disabled:pointer-events-none` or equivalent. Flag interactive components without disabled styling.

**[MEDIUM] Icon Sizing** (`icon-sizing`): Icons inside components should be sized via parent `[&_svg]:size-N` pattern, not directly on the icon element.

### Stories & Tests

**[HIGH] Missing Story** (`missing-story`): New components must have a Storybook story. Flag new component files without a corresponding `.stories.tsx`.

**[HIGH] Story Tags** (`story-tags`): Stories must include `tags: ["autodocs"]`. Flag stories missing this tag.

**[MEDIUM] Story Title Path** (`story-title`): Story title must follow `Components/UI/Name` or `Components/Blocks/Name` convention. Flag incorrect paths.

**[MEDIUM] Play Function** (`play-function`): Interactive component stories should have at least one story with a `play` function for interaction testing. Flag interactive components (buttons, inputs, toggles) without play functions.

**[MEDIUM] Visual Test** (`visual-test`): New components should have a visual regression test using `composeStories()` pattern. Flag new components without one.

### TypeScript

**[HIGH] Any Type** (`any-type`): Flag new `any` types without justification.

**[MEDIUM] Strict Compliance** (`strict-compliance`): Flag `@ts-ignore` or `@ts-expect-error` without explanatory comments.

### Accessibility

**[HIGH] ARIA Attributes** (`aria-attributes`): Interactive components must use proper ARIA attributes. Radix handles most cases — flag custom interactive elements missing `role`, `aria-label`, or `aria-expanded`.

**[MEDIUM] Validation States** (`validation-states`): Form components should support `aria-invalid` with visual styling (`aria-invalid:ring-destructive/20`).

### General

**[HIGH] Scope** (`scope`): Flag unrelated formatting or refactoring mixed in. Flag edits to `dist/`.

**[MEDIUM] Bundle Size** (`bundle-size`): Flag new heavy dependencies. The catalog must stay lean — prefer Radix primitives over full component libraries.

**[MEDIUM] Breaking Changes** (`breaking-changes`): Flag removed exports, renamed props, or changed variant values without a migration note in the PR description.

## Step 4: Post comments

If `--dry-run`, skip this step.

```bash
gh api "repos/${REPO}/pulls/<number>/reviews" \
  --method POST \
  --input - <<'EOF'
{
  "event": "COMMENT",
  "body": "**Catalog Review** — N [HIGH], M [MEDIUM] across F files",
  "comments": [
    {
      "path": "packages/kumo-ui-catalog/src/components/ui/example.tsx",
      "line": 42,
      "side": "RIGHT",
      "body": "[HIGH] **CVA Pattern**: Component defines variant styles without CVA. Use `cva()` with variants object."
    }
  ]
}
EOF
```

### Comment format

`[SEVERITY] **Rule Name**: explanation`

Keep each comment to 1-3 sentences max.

### Fallback

If the review API call fails, post a summary comment:

```bash
gh pr comment <number> --body "**Catalog Review**

<list findings, format: \`file:line\` — [SEVERITY] **Rule**: explanation>

---
*Automated review against UI catalog standards.*"
```

---

## Inline Suppression

Developers can suppress a rule:

```tsx
<div style={{ width: "100%" }}> {/* review-ui-catalog-ignore: inline-styles */}
```

---

## Rules Quick Reference

| Rule ID              | Severity | Summary                          |
| -------------------- | -------- | -------------------------------- |
| `cva-pattern`        | HIGH     | UI primitives must use CVA       |
| `as-child`           | HIGH     | Must support asChild + Slot      |
| `data-slot`          | HIGH     | Root element needs data-slot     |
| `component-export`   | MEDIUM   | Export variants when useful      |
| `props-pattern`      | HIGH     | Use React.ComponentProps         |
| `cn-usage`           | HIGH     | Use cn() for className           |
| `missing-export`     | HIGH     | Export from index.ts             |
| `type-export`        | HIGH     | Export public types              |
| `barrel-self-import` | HIGH     | No barrel imports in source      |
| `inline-styles`      | HIGH     | No style={{}}                    |
| `hardcoded-colors`   | HIGH     | Use Tailwind/CSS vars            |
| `missing-story`      | HIGH     | New components need stories      |
| `story-tags`         | HIGH     | Include autodocs tag             |
| `any-type`           | HIGH     | No untyped any                   |
| `aria-attributes`    | HIGH     | Proper ARIA on interactives      |
| `scope`              | HIGH     | No unrelated changes             |
| `import-order`       | MEDIUM   | Follow import grouping           |
| `focus-states`       | MEDIUM   | Use focus-visible:               |
| `disabled-pattern`   | MEDIUM   | Include disabled styles          |
| `icon-sizing`        | MEDIUM   | Parent [&_svg] pattern           |
| `story-title`        | MEDIUM   | Correct title path               |
| `play-function`      | MEDIUM   | Interactive stories need play    |
| `visual-test`        | MEDIUM   | New components need visual tests |
| `strict-compliance`  | MEDIUM   | No unexplained ts-ignore         |
| `validation-states`  | MEDIUM   | Support aria-invalid             |
| `bundle-size`        | MEDIUM   | No heavy new deps                |
| `breaking-changes`   | MEDIUM   | Flag removed/renamed exports     |
