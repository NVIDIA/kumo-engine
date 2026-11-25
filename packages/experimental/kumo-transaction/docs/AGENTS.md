# docs/AGENTS.md

- Use the terminology in `docs/reference/glossary.md`.
- Update public contract and migration guides when contracts, config composition, or support tiers change.
- Do not claim quality, performance, fairness, explainability, or production
  readiness without a reproducible recipe and recorded evidence.
- Contract documentation and code must change together.
- The site (`mkdocs.yml`) serves every page at its repository path, so write
  relative links that work on GitHub. Add a new page to `nav:`. Run
  `make docs`: the build is strict, so a broken link fails it.
- `docs/_site/api/` holds the API reference pages, which render docstrings.
  A public function or class documents itself in its docstring, not in a page.

Do not copy status tables into `README.md` or recipe READMEs.

- Recipe inventory: `recipes/catalog.yaml` and `recipes/README.md`.
- Support status: `docs/reference/compatibility-matrix.md`.
- Reproduction milestone status: `docs/reference/roadmap.md`.
- How to run a recipe: that recipe's `README.md`.
- User-visible changes: `CHANGELOG.md`.
- Public contracts and compatibility: `docs/concepts/`, `docs/reference/`, and `docs/migration/`.
