# Notebook style guide

Notebooks are written for developers and data scientists outside NVIDIA, for
example in a GTC training session or as part of an NVIDIA Blueprint. Assume
the reader knows Python, PyTorch, and basic machine learning, but has not seen
a transaction foundation model or this code. Define a term the first time you
use it.

## Time and hardware

- Every notebook runs on a laptop CPU in about a minute. That is the path CI
  executes and the default a session attendee gets.
- Anything larger is a recipe command in a markdown cell, with the hardware
  and wall-clock time from its recorded run (for example "one DGX Spark,
  about 40 minutes") and a link to the record under `benchmarks/results/`.
- A series may read a public book when its environment variable is set
  (`TFM_TABFORMER_ROOT`, `TFM_PRAGMATIQ_ROOT`), at a size that stays within
  the one-minute budget. It must still run without the variable.
- **Exception: lab series.** A series written for a GPU training lab
  (`blueprint_v2`) may default to a downloaded book and pretrained
  checkpoint on a GPU, fetched by a tested helper (`tfm.runtime.lab`). Its
  README states the hardware, the downloads, and each notebook's wall-clock
  time. It must also run every cell on CPU under `TFM_NOTEBOOK_SMOKE=1`, on
  generated data with an untrained model. CI runs that mode, and each
  notebook's first cell prints which mode it is in.

## Structure

- One series per recipe, in `notebooks/<recipe_id>/`, with a `README.md`
  listing each notebook and what it teaches. Files are numbered `01_`…`05_`
  and read in order; each still runs on its own.
- The first cell is a title (`# 01 · Topic`) and two or three sentences:
  which series this is, which slot it covers, and what the reader will know
  at the end.
- One idea per section: a short markdown cell that says what and why, then the
  code that shows it. If a code cell needs a comment to explain what it does,
  the markdown above it is missing a sentence.
- End with where the code lives in `src/tfm` and the recipe, and what the
  fixture's numbers do not show.
- When a series retells a published notebook, such as v1's, add a
  "What changed from v1" quote block to each notebook. List each change with
  its reason and a link to the ADR or errata entry.

## Code

- The first code cell carries the SPDX header, the imports it needs, the
  config path from `recipe_dir(...)`, a run directory under
  `repo_root() / "outputs" / "notebooks"`, and `seed_everything(config.seed)`.
  In a lab series, `fetch_lab_assets(...)` supplies the config and the run
  directory.
- Plot what the notebook teaches, with inline matplotlib. Label axes and say
  in the title what is recorded and what is computed live.
- Call `src/tfm` (usually `tfm.runtime`); do not define reusable functions in
  a notebook. If a notebook needs a helper, it belongs in `src/tfm` with a test.
- Show refused combinations with `try` / `except IncompatibleCompositionError`
  and print the message; the message is the lesson.
- Print small, labelled outputs. Prefer `metrics_markdown` for metrics.
- Commit notebooks with outputs cleared.

## Data and claims

- Default to generated synthetic data, on CPU, in under a minute per notebook.
  A real book is never the default path, except in a lab series (see "Time
  and hardware").
- No real customer, account, card, or merchant data; no dataset URLs that are
  not public.
- Fixture metrics show that the wiring works. Say so, and link the recorded
  result under `benchmarks/results/` when there is one.

## Testing

`make test-notebooks` executes every notebook with
[nbmake](https://github.com/treebeardtech/nbmake), and CI runs it with
`TFM_NOTEBOOK_SMOKE=1`. A notebook that cannot run on CPU in CI does not
belong in a series.
