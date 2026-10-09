# KUMO-UI-CATALOG

The catalog will serve as a single source of truth and a centralized library for UI design tokens and components, ensuring consistency, maintainability, and scalability across the app.

UI Catalog will use Storybook to provide an accessible and interactive catalog and act as a reference for designers and developers.

### In-tree development and builds

The catalog is maintained in this monorepo. `ui/package.json` consumes it through
`file:../packages/kumo-ui-catalog`; no SSH checkout, GitHub npm registry token,
release branch or standalone publishing step is required. The package is marked
`private: true`.

With Node 22, from `packages/kumo-ui-catalog`:

```bash
npm ci --legacy-peer-deps
npm run lint-check
npm run build
npm run build-storybook
npm run test:ci
npm run test:visual
```

Browser tests require the matching Playwright browser installation. The root
[UI catalog workflow](../../.github/workflows/ui-catalog.yml) uses the matching
Playwright container. Follow the [UI instructions](../../ui/README.md) to build
and run the consuming application. Make catalog and consumer changes in the
same Kumo PR; release/version policy is tracked in #30655/#31135.

### Storybook

To run storybook

`npm run storybook`

### Agent Skills

Coding agents (Claude Code, Codex) have specialized skills for working with this repo.

#### `/code-ui-catalog` — Scaffold & Implement Components

Use this skill when creating new primitives, block components, stories, or tests.

```bash
/code-ui-catalog add a new Tag component with dismissible variant
/code-ui-catalog add visual tests for Badge component
/code-ui-catalog add a Stepper block component
```

**What it does:**

- Loads conventions from `CLAUDE.md` automatically
- Checks for existing Radix primitives and color tokens before writing
- Generates components following the CVA + Radix + data-slot pattern
- Creates stories with autodocs + play functions
- Creates visual regression tests with composeStories pattern
- Wires up exports in `src/index.ts`
- Runs type-check, lint, and tests to validate

#### `/review-ui-catalog` — Review PRs Against Catalog Standards

Use this skill to review PRs that touch catalog source files.

```bash
/review-ui-catalog 42              # Posts review comments to the PR on GitHub
/review-ui-catalog 42 --dry-run    # Prints findings to stdout only
/review-ui-catalog 42 --force      # Posts even if a previous review exists
@claude /review-ui-catalog         # Via GitHub App in a PR comment
```

**What it does:**

- Runs automated pre-scan (`diff-scan.sh`) for mechanical violations (inline styles, hardcoded colors, barrel self-imports, missing autodocs, etc.)
- LLM reviews against 27 rules covering component structure, exports, styling, stories, TypeScript, and accessibility
- Posts inline review comments with explanations on GitHub
- Respects `// review-ui-catalog-ignore: <rule-id>` inline suppression

**Rule categories:** CVA pattern, asChild/data-slot, barrel exports, import paths, inline styles, hardcoded colors, focus states, missing stories, autodocs tags, play functions, visual tests, TypeScript strict mode, ARIA attributes, bundle size, breaking changes.

Full rule list: `.agents/skills/review-ui-catalog/SKILL.md`

#### Agent Documentation

| File                                                      | Purpose                                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------- |
| `CLAUDE.md`                                               | Auto-loaded conventions for agents (patterns, commands, styling) |
| `.agents/skills/code-ui-catalog/SKILL.md`                 | `/code-ui-catalog` skill definition with templates               |
| `.agents/skills/review-ui-catalog/SKILL.md`               | `/review-ui-catalog` skill definition (27 rules)                 |
| `.agents/skills/review-ui-catalog/references/patterns.md` | Correct vs incorrect code examples                               |
| `.agents/skills/review-ui-catalog/scripts/diff-scan.sh`   | Automated pre-scan for mechanical violations                     |

#### References

- [Tailwind CSS](https://tailwindcss.com/docs/installation/using-vite)
- [Shadcn](https://ui.shadcn.com/docs/components/accordion)
- [Storybook](https://storybook.js.org/docs)
