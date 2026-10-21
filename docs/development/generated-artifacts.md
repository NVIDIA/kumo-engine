# Generated artifacts

Edit the authored input and regenerate the affected output. Review generated
diffs with the source change. Run commands from the repository root unless
noted otherwise; they require the corresponding prepared toolchain.

| Authored input | Generator | Output and verification |
| --- | --- | --- |
| REST API and `kumoapi` schemas; `ui/orval.config.ts` | `make generate-ui-client-code` | `ui/src/gen/`; the target downloads schemas, removes old v0/v1 clients, runs Orval, schema injection, and formatting. Inspect the full diff and run affected UI checks |
| `packages/kumo-pql/kumopql/grammar/PQLGrammar.g4` | From `packages/kumo-pql`: `python scripts/generate_grammar.py` | Python grammar artifacts; use the pinned ANTLR version and parser tests |
| The same PQL grammar | `npm --prefix ui run compile-grammar-to-ts` | `ui/src/utils/PQLParsing/grammar/`; review autocomplete consumers too |
| UI catalog source exports | `.agents/skills/code-ui/scripts/generate-catalog-docs.sh` | Catalog component index; generated summaries require review against the source |
| `packages/diskgraph/third_party/artifact-manifest.json` and the Cargo source cache | `python scripts/refresh_diskgraph_notice_inventory.py`, then `python scripts/gen_third_party.py --output THIRD-PARTY.txt` | Rust components of `3rdparty/notices/inventory.json` and `THIRD-PARTY.txt`; run both after DiskGraph dependency changes and review the inventory diff; see [third-party notices](../../3rdparty/README.md) |
| Dependency declarations and resolution configuration | Repository/package lockfile tooling | Lockfiles and attribution; use the owning package/release workflow rather than editing resolved records by hand |
| `.agents/mcp-servers.json` (optional Codex entry) | `python3 .agents/scripts/sync-mcp.py` (manual) | MCP permissions in `.claude/settings.json` and local `.codex/config.toml`; review generated changes before using them |

`ui/src/utils/PQLParsing/` also contains authored parsing and autocomplete
code. Its grammar files and synchronized `pqDocs.json` are generated; the
remaining parsing and autocomplete code is authored.

Editor help in `ui/src/utils/PQLParsing/pqDocs.json` is generated from
`kumo-ai/docs` `ui-hints/editor-hints.json` by `scripts/build_pq_docs.py`.
Edit that authored registry for hover text, exact editor keys, aliases, highlight types and explicit documentation links.
The registry records deliberate missing hints and retains legacy tuples for compatible generation.
Run the docs generator with `--output <engine-checkout>/ui/src/utils/PQLParsing/pqDocs.json`, then run the CodeEditor tests; do not edit the JSON by hand.
The docs synchronization workflow opens reviewed Draft PRs against `NVIDIA/kumo-engine` with source revision provenance, after checking that its default branch supports the registry format.
The compatible Engine consumer and bundled registry must merge before the docs synchronization format switch or hint-only source deletion.
The adapter also accepts legacy tuple data; token dictionaries retain syntax metadata without authored hover text.

The website is maintained and published from `kumo-ai/docs`. Kumo builds use
the committed JSON and do not require that checkout, Fern, or a docs token.
Website publication and Kumo releases remain independent.

Root `CLAUDE.md` links to `AGENTS.md`; component compatibility adapters import
their adjacent instructions. Skills have one authored source. The
[documentation hook](../../.agents/hooks/README.md#documentation-validator) can
sync an existing root skills table; it does not generate the skill sources or
compatibility adapters.

## DiskGraph source

This repository is the sole source of truth for DiskGraph. Edit
`packages/diskgraph` directly and follow its package-local build and test
instructions. There is no upstream synchronization or mirror verification.
The [import records](../../tools/migrations/diskgraph/README.md) are historical
provenance, not a constraint on subsequent changes.
