# PQL import

`packages/kumo-pql` was imported from `kumo-ai/kumo-pql` at the revision in
`source_revision.txt`. `commit-map` maps upstream commits to the filtered
history under this directory; zero destinations are upstream CI-only commits
excluded from the import. The mapped source tip is an ancestor of this branch.
A regular merge is required to retain that history for blame.

The filtered import tip preserves source, tests and documentation byte for
byte. Subsequent monorepo adaptations remove unsupported RFM functionality,
use the in-tree API, generate the Python and UI parsers from one grammar,
and apply repository formatting. Kumo owns future changes.

To reproduce the history, check out the exact `source_revision.txt` commit in
an upstream clone before running:

```bash
git filter-repo \
  --path .gitignore --path .pre-commit-config.yaml \
  --path AGENTS.md --path ARCHITECTURE.md --path TESTING.md --path README.md \
  --path docs/ --path kumopql/ --path pyproject.toml --path setup.py --path test/ \
  --to-subdirectory-filter packages/kumo-pql
```

Verify the filtered tip is an ancestor of the PR head and compare its tree
against the upstream revision, not the clone's working directory. The original
import accidentally used an older checked-out revision while recording the
newer remote revision; the completed history and commit map correct that gap.
