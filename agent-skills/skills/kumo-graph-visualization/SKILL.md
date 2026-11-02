---
name: kumo-graph-visualization
description: Inspect and iterate declared Kumo graph definitions through Markdown, add a graph view to a modeling notebook, or display and export table relationships on request. Covers schema diagrams, layouts, SVG/PDF/Markdown exports, and optional source previews; not model explanation subgraphs.
---

# Graph inspection and visualization

Reuse the user's `kumo.engine.Graph`, or the graph definition being built for the authorized task.
A standalone diagram or schema inspection does not require a prediction target, horizon, training run, snapshot, or fit.
If the graph is missing or ambiguous, resolve which definition to use; do not invent tables, keys, or relationships.
New or changed graph definitions still follow the source access, graph confirmation, column exclusions, and host constraints in `kumo-data-science/accept/preflight.md` and `kumoai-reference/references/graph_construction.md`.
For a training request, preserve the router's task gates before generating code.

Check that the active runtime has `Graph.visualize`; older bundles may lack it.
If the runtime is unavailable or the method is absent, report the limitation rather than inventing an API or executing unrelated cells to obtain a graph.
The contracts below come from `kumo.engine.graph.Graph.visualize` and `kumo.engine._visualization.SVG`; they do not imply that every deployed bundle has the feature.

## Inspect and iterate using Markdown

Use the Markdown export to reason about a declared graph, including when generating a notebook.
In an available, authorized runtime with the graph already defined, export to a temporary local file and read its text:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as inspection_dir:
    schema_path = Path(inspection_dir) / 'graph.md'
    _ = graph.visualize(schema_path)
    graph_markdown = schema_path.read_text(encoding='utf-8')
print(graph_markdown)
```

The assignment suppresses automatic notebook display of the returned SVG object; the export still computes SVG internally.
Inspect the text with Python or file-reading tools, not a browser, screenshot, or interactive viewer.
The Markdown contains a Mermaid relationship diagram plus every declared column, data type, PK/FK/TS/TE role, and outgoing connection, including columns hidden in the interactive diagram.
Compare these declarations to the confirmed graph proposal, including column exclusions and FK-to-PK direction.
After an authorized definition change, rebuild the immutable graph and repeat the export; report differences without silently changing the confirmed schema.
This is a schema inspection, not proof of observed key uniqueness, referential integrity, row coverage, or sample distributions.
Do not call `snapshot()`, train, or query source rows just to inspect the declaration.
If the graph is not available for execution, review its definition and state that export inspection could not run; generating a notebook does not authorize running its connector or modeling cells.
Markdown export has the same visualization limits as other formats; if it raises `ValueError`, report the error and inspect the declarations directly without trimming the graph.

## Generate or display a notebook view

In a generated modeling notebook, include this separate optional-preview cell immediately after graph construction and `graph.validate()` for the human reader:

```python
from IPython.display import display

try:
    display(graph.visualize())
except ValueError as error:
    print(f'Graph preview unavailable: {error}')
```

Visualization limits must not stop an otherwise valid modeling workflow: the cell reports the actual `ValueError` and continues without changing the graph.
Explicit `display` is needed inside the `try` block; a bare expression there does not display automatically.
Generating the cell does not authorize running the notebook; preserve the router's generate-only and explicit-run rules.
For standalone requests, show the interactive view only when the user asks to see it.
Do not open or analyze the rendered view as part of the agent's graph iteration.
Read the graph visualization section of `hosts/<host>/runtime.md` for display and file access on the active host.

```python
graph.visualize(path=None, show_cols=False, show_samples=False, layout='auto')
```

- With no path, the call returns SVG text with notebook display support and writes no export file.
- The default diagram shows PK, relationship FK, time (TS), and end-time (TE) columns. `show_cols=True` adds ordinary columns up to 10 rows per table, or 5 above 10 tables; key/time columns always remain visible even beyond the cap.
- `layout` accepts only `auto`, `standard`, or `compact`. Auto selects from graph structure; the notebook selector can compare arrangements without changing the graph.
- Pan, zoom, Fit, fullscreen when the host permits it, and hover/focus details are notebook controls. Each displayed view has its own camera; changing the view does not edit the graph.
- More than 30 tables, self-relations, missing definitions, invalid edge endpoints, and missing or duplicate declared columns raise `ValueError`. Report the actual error; do not silently drop tables or edges to obtain a picture.

## Save a requested export

Pass the user's local destination on the filesystem where Kumo Engine runs; an existing file is overwritten and the parent directory must exist.
Use a fresh path unless replacing that file is intended.
These examples are alternatives; emit only the requested format:

```python
_ = graph.visualize('relationships.svg', show_cols=True, layout='compact')
_ = graph.visualize('relationships.pdf', show_cols=True, layout='compact')
_ = graph.visualize('relationships.md')
```

`.pdf` produces a vector diagram plus a paginated complete schema/connection index; `.md` and `.markdown` produce Mermaid and complete schema tables; other suffixes receive UTF-8 SVG, not HTML or PNG.
SVG and PDF use the initial layout requested in Python, not subsequent notebook interactions, and use a light palette.
Markdown diagrams use the viewer's Mermaid layout; the schema tables remain readable without Mermaid support.
The notebook's Export commands panel shows Python save commands; opening it does not write a file or start a browser download.
Report the actual runtime-side path and how to retrieve it through the active host; do not claim a browser download occurred.

## Optional source samples

Keep `show_samples=False` for Markdown inspection and generated notebook defaults.
Use `show_samples=True` only for an explicit request to preview source values, explaining that it can run warehouse queries with costs not bounded by the returned row limit.
It requests up to 30 preview rows per table and shows up to six non-null examples per visible column, independently selected, in notebook tooltips.
Unavailable, empty, and all-null previews are labeled; they are not evidence about the full dataset.
Samples can persist in saved notebook HTML or widget state, but SVG, PDF, and Markdown exports exclude them.
Requesting a sample-free export does not justify enabling previews.

Source: shipped package (`kumo.engine.graph.Graph.visualize`, `kumo.engine._visualization.SVG`, `kumo.engine._visualization.render`)
