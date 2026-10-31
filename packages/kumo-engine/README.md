<!-- This README is also the PyPI project description: keep every link absolute. -->
<p align="center">
  <img height="180" src="https://raw.githubusercontent.com/NVIDIA/kumo-engine/main/ui/src/images/kumo_logo_green.svg" alt="Kumo" />
</p>

______________________________________________________________________

# Kumo Engine

Kumo Engine (`kumo.engine`) trains graph neural networks on relational data, in-process on your own machine. Define tables and the relationships between them, describe what to predict with a predictive query, then train a model and run batch prediction.

## Requirements

- Linux x86_64 or ARM64 with glibc 2.34 or newer, or macOS 14 or newer on Apple silicon
- On Windows, use WSL2 x86_64 with a Linux distribution meeting that glibc requirement
- CPython 3.11 or newer
- For production workloads, an NVIDIA GPU with a compatible driver; see [prerequisites](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/prerequisites) and [GPU sizing](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/gpu-sizing)

Kumo Engine uses a GPU when one is available: an NVIDIA GPU through CUDA, or an Apple silicon GPU through PyTorch's MPS backend. Without one, training and prediction run on CPU, which is enough for small jobs such as the quickstart below. To choose the device, pass `accelerator=` (`"auto"`, `"cpu"`, `"gpu"`, or `"mps"`) to `Trainer`.

The bundled native engine requires CPU AES instructions: AES and SSE2 on
x86-64, or AES and NEON on ARM64. Check that your host or virtual machine
exposes these features before installing a wheel.

Windows support for this release is limited to WSL2 using the Linux x86_64
wheel. Run installation and Python commands inside WSL. Native Windows is not
supported, and native Windows wheels are not published in this release.

## Installation

```bash
pip install kumo-engine
```

Linux x86_64 requires GPU Faiss, which also handles CPU retrieval. macOS keeps
CPU Faiss and its existing Metal retrieval support. Faiss is optional on ARM64
Linux: without it, affected link-prediction retrieval raises an installation
error while other tasks remain available. On ARM64 Linux,
use `pip install "kumo-engine[cpu]"` for CPU retrieval with a compatible backend
wheel for your platform and Python version. For CUDA retrieval, install a
compatible GPU Faiss build directly, building from source on ARM64 Linux if
needed. The CPU extra retains the required platform backend on Linux x86_64
and macOS; it does not configure PyTorch or select the execution device.
Choose one backend per environment.

The base install reads local CSV and Parquet files. For other data sources, install the matching extra in the same environment:

| Data source | Command |
| --- | --- |
| Amazon S3 | `pip install "kumo-engine[s3]"` |
| Google Cloud Storage | `pip install "kumo-engine[gcs]"` |
| Azure Data Lake Storage Gen2 | `pip install "kumo-engine[adls]"` |
| BigQuery | `pip install "kumo-engine[bigquery]"` |
| Snowflake | `pip install "kumo-engine[snowflake]"` |
| Databricks | `pip install "kumo-engine[databricks]"` |

Text columns are encoded with word vectors. The `text` extra installs the default model, GloVe 6B 50d:

```bash
pip install "kumo-engine[text]"
```

Extras combine, for example `pip install "kumo-engine[s3,text]"`.

The wheels on PyPI are for Linux x86_64 and for macOS on Apple silicon. To build and install on Linux ARM64 (DGX Spark), follow the [DGX Spark guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/dgx-spark.md).

### Optional: faster training on NVIDIA GPUs

On an NVIDIA GPU, PyTorch Geometric's optional `torch-scatter` extension accelerates the min and max aggregations used in training. Its wheels are built for a specific PyTorch and CUDA version, so install it from PyG's wheel index for your environment, for example for PyTorch 2.9 with CUDA 12.8:

```bash
pip install torch-scatter --only-binary torch-scatter -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

Check your versions with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`, and see [PyG's installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the available combinations.

## Quickstart

Predict which users will place an order in the next 30 days, using a sample dataset of users and their orders.

```python
from kumo.engine import (Edge, FileConnector, Graph, PredictiveQuery, Table,
                         Trainer, datasets)

files = FileConnector(root_dir=datasets.shop())
users = Table.from_source_table(files["users"], primary_key="user_id")
orders = Table.from_source_table(files["orders"], primary_key="order_id",
                                 time_column="order_time")
graph = Graph(
    tables={"users": users, "orders": orders},
    edges=[Edge(src_table="orders", fkey="user_id", dst_table="users")],
)
snapshot = graph.snapshot()

pquery = PredictiveQuery(
    graph, "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
trainer = Trainer(pquery.suggest_model_plan(snapshot))
trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)

result = trainer.predict(pquery.generate_prediction_table(snapshot),
                         snapshot=snapshot, binary_classification_threshold=0.5)
print(result.frame.head())
```

To use your own data, point `FileConnector` at a directory of CSV or Parquet files instead of `datasets.shop()`.

`result.frame` is a pandas DataFrame with one row per user: `ENTITY`, the predicted label `TARGET_PRED`, and the class probabilities `False_PROB` and `True_PROB`.

`generate_prediction_table(snapshot)` predicts as of the latest `order_time` in the snapshot, so the 30 days start there. To predict from another time, pass `prediction_time=` a timezone-naive `datetime`, read as UTC.

`graph.snapshot()` ingests the tables into an on-disk graph index under the scratch directory, which also holds training runs, model bundles, and predictions. It defaults to `~/.cache/kumo-engine`; to put it on a large local disk, call `kumo.engine.configure(scratch_dir=...)` first.

## Visualize table relationships

Display declared tables and relationships in a notebook, or export SVG, PDF, or Markdown:

```python
graph.visualize()
graph.visualize("relationships.svg", show_cols=True)
graph.visualize("relationships.pdf", show_cols=True, layout="compact")
graph.visualize("relationships.md")
graph.visualize(layout="compact")
```

Visualization reads no source data by default and uses Kumo Engine's existing dependencies.
The return value is SVG text with notebook display support.
The default view includes primary keys, relationship foreign keys, and declared time (`TS`) and end-time (`TE`) columns.
`show_cols=True` adds other columns up to 10 rows per table, or 5 above 10 tables; key and time columns always remain visible.
Graphs are limited to 30 tables and require valid declared schemas without self-relations.

`layout="auto"` chooses Standard or Compact from graph structure; either is also selectable in the notebook.
Drag to pan, use the controls or Ctrl/Cmd+wheel to zoom, and choose Fit to reset the view.
Hover or focus a header or column for details without moving the graph.
The default notebook view follows the browser’s light or dark appearance; SVG and PDF exports use the light palette.
Notebook display automatically uses the widget protocol in detected Snowflake Workspace notebooks; Databricks and other hosts use HTML.
Installing the Snowflake connector extra does not select the display protocol.
HTML interactivity requires trusted output; fullscreen also needs permission from the notebook host.
Saved HTML output includes the complete, sample-free SVG image in the requested initial layout and “Rerun this cell to activate animation.”
Rerun the visualization cell once after upgrading and save the notebook to store this fallback; VS Code Restricted Mode can refuse HTML/SVG output entirely.
The Snowflake Workspace view uses the environment's `anywidget>=0.9,<1` and `ipywidgets` packages, imported only when displaying the result.
Snowflake Workspace Runtime 2.6 and later supplies these packages, including the documented GPU Runtime 2.9 target; Runtime 2.5 and earlier requires installation according to your environment's package policy.
See [Snowflake's widget installation guidance](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-in-workspaces/notebooks-in-workspaces-interactive-widgets#installation).
Kumo Engine does not install these packages as base dependencies or connector extras.
If the widget packages are unavailable or incompatible, the Workspace view shows a static graph and a diagnostic; SVG, PDF, and Markdown exports remain available.
The optional `graph.visualize().as_widget()` helper explicitly selects the same widget view on compatible hosts.
The widget view uses the same controls, and widget restoration still depends on the notebook host.
It inherits a host color scheme declared when mounted, with browser appearance as the fallback.
Each displayed view has its own camera; redisplay after restarting the kernel.
SVG exports contain the initial arrangement and native tooltips.
The Export commands button reveals a selectable PDF save command followed by a Markdown save command.
Run the desired command with your graph variable to save the file in the notebook filesystem.
Opening the panel does not generate a PDF, start a download, or write a file.
The PDF includes a paginated index of all declared tables, columns, data types, key/time roles, and connections, including columns omitted from the diagram.
Passing a `.pdf` path to `visualize` saves the same document on the filesystem where Kumo Engine runs, using the requested initial layout; it overwrites an existing file.
PDF export uses Matplotlib and its embedded DejaVu fonts; unsupported characters are shown as explicit Unicode escapes.
For example, `_ = graph.visualize("graph.pdf", layout="standard")` saves a PDF in the notebook's working directory; use `layout="compact"` for Compact.
Use `_ = graph.visualize("graph.md")` to save Markdown.
The assignment suppresses automatic display of the returned SVG.
In Snowflake Workspaces, retrieve the resulting file through the workspace file browser; saving requires write access to the directory.
This writes a notebook-side file and does not start a browser download.

Passing a `.md` or `.markdown` path writes a Mermaid relationship diagram and Markdown tables containing every declared column, data type, key/time role, and outgoing connection.
The file is saved on the machine running Kumo Engine and overwrites an existing file.
Mermaid diagrams render on GitHub and in compatible Markdown viewers, using the viewer's layout rather than the notebook arrangement.
The schema tables are readable without Mermaid support.

`show_samples=True` shows up to six non-null examples per visible column in two columns and up to three rows, sampled independently from up to 30 preview rows per table.
The default `show_samples=False` hides samples and reads no source data.
Previews can run warehouse queries and incur costs; the row limit does not bound bytes scanned.
Unavailable or empty previews are labeled.
Sample values are saved in notebook HTML or widget state, but never included in SVG, PDF, or Markdown exports.

### Developing another notebook widget

Use the private `_widget` helpers to define one feature specification for both notebook transports.
The shared shell supplies theme tokens, controls, view identity, and disposal; keep feature behavior inside its initializer.
This chart uses the same specification with trusted HTML and with real widget MIME:

```python
from IPython.display import HTML, display
from kumo.engine._widget import (
    CSS_THEME, button, render_anywidget, render_widget,
)

chart = dict(
    class_name="kumo-contribution",
    body=(
        '<div class="kumo-widget-toolbar">'
        + button("chart-toggle", "Toggle contribution", text="↔")
        + '</div><svg width="320" height="48" role="img" '
        'aria-label="Contribution"><rect y="8" height="24"/></svg>'
    ),
    style=f'.kumo-contribution rect{{fill:{CSS_THEME["accent"]}}}',
    data={"fraction": 0.75},
    script="""
const svg=host.querySelector('svg'),bar=svg.querySelector('rect');
const toggle=host.querySelector('.chart-toggle');
let fraction=data.fraction;
const update=()=>{bar.setAttribute('width',300*fraction);
svg.setAttribute('aria-label','Contribution: '+Math.round(100*fraction)+'%')};
toggle.addEventListener('click',()=>{fraction=1-fraction;update()},{signal});
update();toggle.disabled=false;
""",
)
display(HTML(render_widget(**chart)))
display(render_anywidget(**chart))  # uses environment-provided widget packages
```

`render_anywidget` lazily imports the optional dependency and provides an AFM `render({model, el})` entry point.
It transports markup and structured `data` in model state; executable feature code stays in the module.
Each view initializes its own closure, so repeated displays of one model have independent controls and camera state.
The specification is a snapshot; updating Python dictionaries does not update existing views.

- Escape user text in markup and use `data` for values; never interpret model state as code or include executable scripts in `body`.
  Keep frontend code self-contained, without CDN imports, `eval`, or `new Function`.
- Initializers receive `host`, `widgetId`, `data`, `signal`, and `onDispose(callback)`.
  Scope queries to `host`, attach listeners with `signal`, and enable controls after initialization.
  Register timer/animation cancellation, observer disconnection, pointer release, and object-URL revocation with `onDispose`; the shell invokes cleanup on HTML removal or AFM disposal.
- Scope CSS under the feature's root class and use `CSS_THEME` for interactive colors.
  Both shells retain `nodarkmode`; the widget transport inherits a declared host color scheme available at mount and otherwise follows browser appearance.
  Redisplay to inherit a scheme that becomes available only after mounting.
- Use `widgetId` for IDs created by the initializer; for IDs and references already in SVG markup, pass their shared unique prefix as `namespace` so each view rewrites it independently.
- Keep static export serialization in the feature, using the resolved `THEME` palette for light exports.
  Notebook-only sample values may enter saved HTML or widget state, but must stay out of static exports.
- HTML needs a trusted execution surface; anywidget needs a compatible active notebook host.
  Request fullscreen directly from a user action and handle rejection; use `host.matches(":fullscreen")` to recognize the actual fullscreen element inside Shadow DOM.
  Fullscreen remains subject to host permissions, and widget support alone does not prove that capability.
  Show explicit notebook-file export commands with accurate destinations; keep file generation outside display initialization.

AFM initializes once attached; temporary host detachment preserves the view until the host calls its disposer.
Disposed views must be rendered again.

## Documentation

- [ADLS connector guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-adls/README.md): setup, authentication, supported reads and exports.
- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview): prerequisites, GPU sizing, installation, and end-to-end workflows. Its User Code Guide covers connecting data, tables and graphs, training, and prediction.
- [Predictive query examples](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/workflows-and-solutions/predictive-query-examples): common prediction tasks written as predictive queries.

## Support

Report bugs and request features in [GitHub Issues](https://github.com/NVIDIA/kumo-engine/issues). To report a security vulnerability, follow [SECURITY.md](https://github.com/NVIDIA/kumo-engine/blob/main/SECURITY.md) instead of opening an issue.

## License

Kumo Engine is licensed under the [Apache License 2.0](https://github.com/NVIDIA/kumo-engine/blob/main/LICENSE). The wheel also bundles third-party components under their own licenses, listed in [THIRD-PARTY.txt](https://github.com/NVIDIA/kumo-engine/blob/main/THIRD-PARTY.txt).
