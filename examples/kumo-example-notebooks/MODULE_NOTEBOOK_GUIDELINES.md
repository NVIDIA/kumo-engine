# Module notebook guidelines

Teach one workflow step in a small notebook that runs on its own.
Use the shared [writing, cell formatting and data rules](NOTEBOOK_GUIDELINES.md), including the limit of one or two synthetic-data cells.

## Structure

| Section | What to include |
| --- | --- |
| Header | The step the reader will learn, offering, runtime and a link to the relevant documentation. |
| Setup and example data | Only the setup and data needed to demonstrate the step. Clearly separate synthetic-data generation from the workflow. |
| Demonstrate the step | The focused example, with short explanations of its purpose and any use-case-specific choices. |
| Inspect the result | A small output or check that shows what the step accomplished. Explain an unexpected result only when it is part of the lesson. |
| Next step | A link to the related workflow or end-to-end notebook when one exists. |
| Cleanup | Remove temporary resources created by the example, when needed. |

## Keep the module focused

* Do not require another notebook's variables, files or execution history.
* Include training only if the demonstrated step needs a trained model.
* Avoid turning a focused example into a complete training tutorial or API reference.
* Use descriptive deployment-prefixed filenames in the offering, model-family and vertical layout defined in the shared guidelines.
  Put general workflow modules under `cross-industry/`.
  Update notebook links and indexes when renaming a module.
* Keep unavailable new-feature demonstrations commented and label them as previews.
  Describe the executable preparation separately from the preview.
