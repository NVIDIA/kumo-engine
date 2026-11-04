---
name: kumo-graph-visualization
description: Provide graph visuals in Kumo notebooks and SDK scripts, display or export relationships on request, and inspect Markdown schemas during model iteration.
---

# Graph visualization

Use `Graph.visualize()` to give users a view of the tables and relationships their model uses.
Reuse the graph for the current task; graph changes follow `kumoai-reference/references/graph_construction.md`.
The visual shows the graph exactly as built and never changes it: add no primary key, column, or edge for the viewer.
Tables without a primary key, such as event tables, display as they are.
A request to view or export an existing graph needs no model-training intake.

## Generated notebooks

After graph construction and `graph.validate()`, include a separate cell before snapshot creation or training:

```python
from IPython.display import display

if not hasattr(graph, 'visualize'):
    print('Graph preview needs a kumo-engine build including #134; upgrade when available. Continuing.')
else:
    try:
        display(graph.visualize())
    except Exception as error:
        print(f'Graph preview unavailable: {type(error).__name__}: {error}')
```

Introduce the cell as an interactive view where the user can explore tables, columns, and relationships.
It requires no snapshot and reads no source rows by default.
Continue the workflow normally; pause for visual review only if the user asks.
A notebook displays the graph only; it saves no SVG unless the user asks for a file.

## Generated SDK scripts

This section applies to scripts only, never to notebooks.
Save `graph.svg` alongside the run's outputs before training and report its path.
Use the run's output directory; this example uses the current directory:

```python
from pathlib import Path

graph_path = Path('graph.svg').resolve()
if not hasattr(graph, 'visualize'):
    print('Graph preview needs a kumo-engine build including #134; upgrade when available. Continuing.')
else:
    try:
        _ = graph.visualize(graph_path)
        print(f'Graph diagram: {graph_path}')
    except Exception as error:
        print(f'Graph preview unavailable: {type(error).__name__}: {error}')
```

A missing visualization API, unsupported graph, or export failure should report the error and let modeling continue with the original graph.
The viewer supports up to 30 tables and does not support self-relations.

## Requested visuals and exports

Display the graph in a notebook or save the format the user requests: `.svg` for a diagram, `.pdf` for a diagram and complete schema index, or `.md` for a Mermaid diagram and schema tables.
`graph.visualize('graph.pdf', show_cols=True)` also shows ordinary columns in the diagram.
Use `layout='standard'` or `layout='compact'` to select an arrangement; the default is `'auto'`.
Export paths belong to the Python runtime; use `hosts/<host>/runtime.md` for display and file retrieval.

Keep samples off unless the user requests source values.
`show_samples=True` queries source data and includes sample values in saved notebook output; file exports omit them.

## Model iteration with Markdown

When considering which tables or relationships to change for a use case, export the graph for text inspection:

```python
_ = graph.visualize('graph.md')
```

Read the file to inspect all declared columns, key and time roles, and relationships, including columns hidden in the diagram.
The assignment suppresses automatic notebook display.
Use this schema alongside model evaluation results to plan graph experiments; it does not measure predictive usefulness.
Markdown inspection is optional during iteration, not a prerequisite for generating or running a model.

Source: shipped package (`kumo.engine.graph.Graph.visualize`, `kumo.engine._visualization.SVG`)
