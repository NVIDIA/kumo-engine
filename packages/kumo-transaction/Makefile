.PHONY: lint typecheck test-unit test-contract test-smoke test-notebooks docs docs-serve check

lint:
	uv run ruff check src tests recipes benchmarks scripts
	uv run ruff format --check src tests recipes benchmarks scripts

typecheck:
	uv run pyright src/tfm

test-unit:
	uv run pytest tests/unit -q

test-contract:
	uv run pytest tests/contract -q

test-smoke:
	uv run pytest tests/recipe_smoke -q

# Executes every notebook on CPU; needs `uv sync --extra dev --extra notebooks
# --extra baselines --extra umap`. Local dataset roots are unset so notebooks run
# on their generated fixtures, and the v2 lab series runs in smoke mode.
test-notebooks:
	env -u TFM_PRAGMATIQ_ROOT -u TFM_TABFORMER_ROOT -u TFM_V2_CHECKPOINT \
		TFM_NOTEBOOK_SMOKE=1 uv run pytest --nbmake notebooks -q

# The documentation site; needs `uv sync --extra docs`. --strict fails on a
# broken link or a missing page, as CI does.
docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

check: lint typecheck test-unit test-contract test-smoke
