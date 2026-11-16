# Generated artifacts

Edit the authored input and regenerate the affected output. Review generated
diffs with the source change. Run commands from the repository root unless
noted otherwise; they require the corresponding prepared toolchain.

| Authored input | Generator | Output and verification |
| --- | --- | --- |
| REST API and `kumoapi` schemas; `apps/studio/ui/orval.config.ts` | `make generate-ui-client-code` | `apps/studio/ui/src/gen/`; the target downloads schemas, removes old v0/v1 clients, runs Orval, schema injection, and formatting. Inspect the full diff and run affected UI checks |
| `packages/contracts/kumo-pql/kumopql/grammar/PQLGrammar.g4` | From `packages/contracts/kumo-pql`: `python scripts/generate_grammar.py` | Python grammar artifacts; use the pinned ANTLR version and parser tests |
| The same PQL grammar | `npm --prefix apps/studio/ui run compile-grammar-to-ts` | `apps/studio/ui/src/utils/PQLParsing/grammar/`; review autocomplete consumers too |
| UI catalog source exports | `.agents/skills/code-ui/scripts/generate-catalog-docs.sh` | Catalog component index; generated summaries require review against the source |
| `packages/data/diskgraph/third_party/artifact-manifest.json` and the Cargo source cache | `python development/tooling/scripts/refresh_diskgraph_notice_inventory.py`, then `python development/tooling/scripts/gen_third_party.py --output THIRD-PARTY.txt` | Rust components of `development/third-party/notices/inventory.json` and `THIRD-PARTY.txt`; run both after DiskGraph dependency changes and review the inventory diff; see [third-party notices](../../development/third-party/README.md) |
| Dependency declarations and resolution configuration | Repository/package lockfile tooling | Lockfiles and attribution; use the owning package/release workflow rather than editing resolved records by hand |
| `.agents/mcp-servers.json` (optional Codex entry) | `python3 .agents/scripts/sync-mcp.py` (manual) | MCP permissions in `.claude/settings.json` and local `.codex/config.toml`; review generated changes before using them |

`apps/studio/ui/src/utils/PQLParsing/` also contains authored parsing and autocomplete
code. Its grammar files and synchronized `pqDocs.json` are generated; the
remaining parsing and autocomplete code is authored.

## Editor help

Engine owns the authored `apps/studio/ui/editor-hints/editor-hints.json` registry, the shared validation schema in `apps/studio/ui/src/components/CodeEditor/editorHintRegistry.ts`, and the generator in `apps/studio/ui/scripts/generate-editor-help.mjs`.
Edit the registry for hover text, exact editor keys, aliases, highlight types and explicit documentation links.
The schema reuses the editor's configured and fallback keys in `tooltipKeys.ts`; there is no second coverage list in the website repository.
The registry records deliberate missing hints.

After installing the public UI dependencies using the [setup guide](setup.md), run from this repository root:

```bash
npm --prefix apps/studio/ui run generate:editor-help
npm --prefix apps/studio/ui run check:editor-help
npm --prefix apps/studio/ui run test -- --run src/components/CodeEditor/__tests__/editorHelp.test.ts
```

Generation validates the same schema, coverage and safe content as the runtime adapter, sorts entries deterministically and applies the lockfile's Prettier formatter.
`check:editor-help` fails for invalid source or stale generated output without writing files.
Commit the source and regenerated `apps/studio/ui/src/utils/PQLParsing/pqDocs.json` together; never edit the generated file by hand.
With dependencies installed, these commands require no network access, private checkout, Fern or credentials.
Engine builds consume the committed output without running the generator.
The adapter also accepts legacy tuple data; token dictionaries retain syntax metadata without authored hover text.

### Website consumption

The website can retain its Fern tutorials, navigation and publishing in its own repository while consuming this public reference asset:

```text
https://raw.githubusercontent.com/NVIDIA/kumo-engine/<full-commit-sha>/ui/src/utils/PQLParsing/pqDocs.json
```

Pin a full reviewed Engine commit, vendor the asset with its SHA-256 and render the versioned `schemaVersion`, `hints` and `noHints` fields into the website's reference pages.
Keep that import in a reviewed website change rather than fetching a moving branch during publishing.
The asset contains explicit documentation URLs and both Markdown and literal hint content; the website renderer must respect each entry's `format` and preserve its own reference navigation.
Website publication and Engine releases remain independent.

The existing docs-to-Engine synchronization and the companion docs PR #366 must be changed to consume Engine before enabling that migration or deleting reference sources.
Do not run the old docs synchronization against this registry: it would overwrite Engine-authored changes.
Verify each final documentation destination, including redirects, before deleting its website source.

Root `CLAUDE.md` links to `AGENTS.md`; component compatibility adapters import
their adjacent instructions. Skills have one authored source. The
[documentation hook](../../.agents/hooks/README.md#documentation-validator) can
sync an existing root skills table; it does not generate the skill sources or
compatibility adapters.

## DiskGraph source

This repository is the sole source of truth for DiskGraph. Edit
`packages/data/diskgraph` directly and follow its package-local build and test
instructions. There is no upstream synchronization or mirror verification.
The [import records](../../apps/studio/tools/migrations/diskgraph/README.md) are historical
provenance, not a constraint on subsequent changes.
