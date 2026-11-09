# Notebook guidelines

Help a developer answer a concrete business question with Kumo, from setup to useful predictions.
Keep the notebook focused on the code and the decisions specific to its use case.

| Kind | Purpose | Where it lives |
| --- | --- | --- |
| End-to-end notebook | One use case, runnable from setup through predictions. | `<offering_path>/<vertical>/notebooks/` |
| Module notebook | One workflow step, small and runnable on its own. | `<offering_path>/<vertical>/notebooks/` |

Choose the offering path from the API and model used by the example:

* `kumo-engine/` contains workflows using `import kumo.engine` or `from kumo.engine ...`, together with their helpers and feature previews.
* `structured-data-models/relational/` contains direct and client KumoRelational examples, including those installing `structured-data-models-client[relational]`.
* `structured-data-models/tabular/` contains direct and client KumoTabular examples and TabICLv2 examples, including tabular inference through `structured-data-models-client`.
* `structured-data-models/time-series/` contains forecasting and anomaly-detection model examples and previews.

Structured Data Models examples use `sdm` or `sdm_client`; select their family by model, since direct relational and tabular models share the `structured-data-models` package.
Keep client workflows identified in their title and filename.
Choose the vertical from the example's business use case, such as `fsi/`, `retail/` or `healthcare/`.
Use `cross-industry/` for general workflows that are not specific to an industry.
Keep deployment in the filename, using the example's actual setup and runtime.
Use README categories for focused modules.
Keep standalone scripts under `<offering_path>/<vertical>/sdk-code-examples/`.

Name examples `<platform>_<offering_slug>_<descriptive_example>.ipynb` or `.py`, using lowercase snake_case.
For example, use `kumo-engine/fsi/notebooks/databricks_kumo_engine_money_muling_detection.ipynb`.
Use hyphenated offering directories and underscore offering slugs (`kumo_engine` or `structured_data_models`).
Keep the model family in the directory path; it does not change the offering slug.
Keep helpers and assets in clearly named example-specific companion directories under the same vertical and artifact directory, outside the runnable-example index.
Preserve conventional package markers and required license or attribution filenames.
Update imports and links whenever an example moves.

The writing, formatting and data rules below apply to both kinds.
Use the [separate module guide](MODULE_NOTEBOOK_GUIDELINES.md) for module structure.

Install the latest published package using an unpinned pip install command.
For Databricks Engine notebooks, use `%pip install "kumo-engine[databricks]"`, followed by a separate `dbutils.library.restartPython()` cell.
Do not put wheel-release or source-validation references in the notebook header.
Keep detailed revision and validation evidence outside the notebook.
Do not add a reproducibility checklist, provenance dump or generic limitations section.
Validate active imports, signatures and workflow contracts against the agreed published package baseline before publishing an example.
Keep exact validation versions and evidence in the review record, outside customer examples.

## Writing and cell formatting

* Give each code cell a short explanation of its action and purpose, usually one sentence.
  A short introduction can cover a pair of closely related cells.
* Explain choices specific to the use case or model, such as the prediction window, split strategy, selected metric or a tuning setting.
  Do not narrate every line or repeat general API explanations.
* Link to the documentation page or section for the action being demonstrated.
  Use descriptive link text and the docs for the actual offering and runtime, rather than a generic overview link.
* Use short, descriptive headings in sentence case.
  Keep Markdown cells brief; avoid multi-paragraph introductions to routine code cells.
* Keep each code cell focused on one workflow step, apart from the synthetic-data block described below.
  Use meaningful variable names and comments that explain a non-obvious choice rather than restating the code.
* Show the output that helps the reader understand the result.
  Add a brief interpretation when its meaning depends on the use case.
* Use lowercase file names with underscores, without versions or dates.
  Use the offering, vertical and artifact type layout above.

### Spacing between cells

Keep the same visual separation between sections throughout a notebook, including when the preceding cell displays a table, chart, training log or other output.
In Databricks, start a Markdown cell containing a section or subsection heading after a code cell with one `<br>` on its own line, followed by a blank line and the heading.
Apply this consistently to all such headings, including after code cells that currently produce no output.
Keep one blank line between the heading and its explanatory text.
Do not add empty code cells or repeated blank lines as spacers; blank Markdown lines alone do not create reliable rendered spacing.
Review section transitions in the target notebook UI when accessible.
When execution is authorized, also review with outputs visible.
Record source-only or unexecuted review limits outside customer examples.

## Synthetic data: a distinct setup block

Keep all synthetic-data generation, including its helper functions, in one or two code cells maximum.
Place these cells together under **Generate synthetic example data**, separate from the model workflow.
Introduce them with a short note such as:

> Run the next two cells to create synthetic account and transaction tables.
> You can skim the generation code; the Kumo workflow starts below.

Use “next cell” when there is only one.
Keep graph construction, query definition, model configuration and training out of these cells.
After the data cells, add a clear heading such as **Build the graph** or **Train the model**, matching the first actual workflow step.
Introduce that step with a short sentence explaining what the reader will do with the generated data.
Do not hide model setup inside data-generation helpers.

## End-to-end notebook structure

Use the sections that serve the offering and use case.
Do not add training, graphs, predictive queries or explanations to a workflow that does not use them.

| Section | What to include |
| --- | --- |
| Header | A descriptive title, one sentence on the business question, offering, runtime, and the matching documentation link. State compute requirements and expected run time when relevant and known. |
| Setup | Installation and configuration for the selected offering and runtime. Group values the reader needs to change in a clearly labeled parameters cell. |
| Data | Public or synthetic data and a brief description of what its tables represent. Follow the synthetic-data block convention above when generating data. |
| Model workflow | The offering-specific steps below, with brief explanations of the choices this use case needs. |
| Evaluate | The relevant metrics and their meaning for the business question. Label smoke-test settings and synthetic-data metrics where they appear. |
| Batch predictions | Required for every end-to-end model example: prepare the scoring population, predict in a dedicated code cell, inspect the outputs, and save the results. |
| Explanations | Supported explanations that help interpret this model's predictions, when relevant. |
| Retain and reuse | Retain and reload the model or context when applicable. Link to documentation for scheduling or broader deployment guidance. |
| Cleanup | Remove only scratch data or resources created by this notebook. Never delete the reader's existing data. |

### Batch predictions

Every end-to-end model notebook must include a section titled **Batch predictions**, with a dedicated code cell that scores multiple records using the trained model or configured pretrained model/context.
For graph workflows, score rows added to the original graph and show their addition in a separate cell. For tabular or client workflows without a graph, prepare a separate label-free scoring table or batch. Simulate scoring rows in the same fashion as the original data.
Identify the scoring population and prepare its inputs without supplying held-out target values to prediction.
Use the offering's supported prediction API; batch prediction does not require a hosted batch-job service.
Display a small result sample, verify its alignment with the input IDs or row indices, and explain the prediction columns and expected output rows per input.
Show how to save the predictions to a file or supported destination.
Metrics alone do not satisfy this requirement.
Apply the same workflow as a clearly labeled section in end-to-end standalone scripts, preserving their script format.

### Kumo Engine training

Show the graph's tables and relationships, the predictive query, the training table and splits, model configuration, and training.
Every end-to-end model training notebook that uses a graph must include a dedicated graph-visualization cell.
Place it immediately after graph construction and before training so the reader can inspect the tables and relationships used by the model.
Use the actual graph object from the notebook, not a manually drawn diagram or a saved screenshot.
Introduce the cell with a short purpose statement; link to the relevant graph visualization documentation for details.
If graph visualization is unavailable in the supported package API, keep the demonstration commented and label it as a preview.
Apply the same treatment to other new, unavailable features, and do not describe a commented demonstration as runnable.
Explain the static or temporal choice and split strategy in terms of the business question.
Validate the graph and query using the supported APIs for the version used.
Use the supported `FAST` training preset for tutorial defaults.
Keep severe epoch or batch limits only in an optional, clearly labeled smoke mode.
Change tuning settings only when useful to this example and briefly explain why.
Include incremental prediction or other specialized patterns only when the use case needs them.

### Client KumoRelational

Show the relational inputs, prediction task and client inference workflow supported by the example.
Use the client's actual setup and API; do not impose Engine training steps on a pretrained inference example.

### Kumo Tabular

Show the table, features, labels, context and prediction task using the direct or client API selected by the example.
Use task-appropriate headings rather than graph or predictive-query sections.
Include context reconstruction when it helps the reader reuse the example.

## Data rules

Use synthetic source data with reproducible predictive patterns and valid table relationships.
AdventureWorks is an approved dataset; its examples retain their existing source-data setup.
Keep shared standalone generators in `synthetic-data/`, targeting roughly 1 MB of local files per dataset.
Notebooks keep generation in the one or two setup cells described above.
Preserve connector APIs with clearly named blank account configuration, and explain how generated files map to warehouse tables.
Generators write local files only; the reader loads warehouse tables separately.

* Use public or synthetic data.
  Never include customer data, real account or card numbers, or real merchant names.
* For a download, name the source and its license and verify a checksum.
* Identify synthetic data in the header.
  Label metrics from synthetic data “demonstration only” beside the results.
* For Kumo Engine or KumoRelational, generate related tables suitable for a graph.
  For Kumo Tabular, generate a single table of aggregated features representing the prediction task.
