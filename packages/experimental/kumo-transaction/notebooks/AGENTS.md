# notebooks/AGENTS.md

Notebooks are explanatory only. There is one series per dataset:

- `blueprint_v1` — TabFormer; recipe `blueprint_v1`, following v1's five notebooks
- `pragmatiq` — pragmatiq; recipes `pragma_like` (03a) and `blueprint_v2`
  (00, 03b, and the rest), organized by building-block slot

Numbered files inside a series are sequential; a letter suffix (`03a`, `03b`)
marks alternatives for the same step, and `00` is an overview that runs on
its own.

- Do not put reusable logic here; it belongs in `src/tfm/`.
- Use synthetic or approved fixtures only.
- Keep execution deterministic (set seeds).
- Every notebook must execute on CPU in CI: on generated data, or in the
  series' smoke mode.
- Clear outputs before committing.
- Recipes remain the canonical executable workflow.
