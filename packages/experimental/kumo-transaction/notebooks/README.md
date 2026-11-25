# Notebooks

Install Python 3.12, rustup, a C/C++ compiler and Linux build tools before
syncing the `native` extra. See the [installation guide](../examples/kumo-engine/README.md#canonical-recipes).

One explanatory series per dataset. The recipes under
[`recipes/`](../recipes/README.md) are the canonical workflows; notebooks call
the same `src/tfm` code and hold no logic of their own.

```bash
uv sync --extra native --extra dev --extra notebooks
uv run jupyter notebook notebooks/
```

| Series | Dataset | Recipes | Runs on |
|---|---|---|---|
| [`blueprint_v1`](blueprint_v1) 01–05 (**start here**) | TabFormer | [`blueprint_v1`](../recipes/blueprint_v1) | CPU, on a generated book or 24 TabFormer users |
| [`pragmatiq`](pragmatiq) 00–06 | pragmatiq | [`pragma_like`](../recipes/pragma_like), [`blueprint_v2`](../recipes/blueprint_v2) | GPU lab with a downloaded book and checkpoint; CPU in smoke mode |

New to transaction foundation models? Start with `blueprint_v1`. It follows
the five notebooks of the public
[NVIDIA TFM Blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)
on the IBM TabFormer dataset, and notes in each notebook what the kit changed
and why. The `pragmatiq` series then takes a multi-source book apart slot by
slot ([building blocks](../docs/concepts/building-blocks.md)), comparing the
PRAGMA-like encoder (03a) with the Blueprint v2 decoder (03b).

CI executes every notebook on CPU (`make test-notebooks`), with the pragmatiq
series in its smoke mode (`TFM_NOTEBOOK_SMOKE=1`). The full-size runs are
recipe commands, which each series lists with their hardware and times.

## Writing a notebook

Follow the [style guide](STYLE.md).
