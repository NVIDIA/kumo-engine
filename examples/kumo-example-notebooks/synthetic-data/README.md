# Synthetic example data

These generators create local files with seeded predictive patterns and related-table keys.
They do not connect to a warehouse or upload data.
Install `numpy`, `pandas` and `pyarrow`, then run the generator command shown at the top of the example from the repository root.
Use a new output directory to avoid overwriting existing data.

| Directory | Examples |
| --- | --- |
| `engine_retail/` | Retail graphs, including customer lifetime value and TPC-H |
| `engine_hospitality/` | Restaurant, booking and recipe graphs |
| `engine_other/` | Financial, media, forecasting and general Engine workflows |
| `sdm/` | Tabular, relational and text tasks for Structured Data Models |

AdventureWorks examples retain their approved source dataset and do not use these generators.

Related examples share generators so that their table names, keys and relationships agree.
Generators target roughly 1 MB per dataset; small focused teaching modules can use less.
Notebook generation is contained in one or two setup cells, separate from graph construction and the model workflow.
Synthetic evaluation metrics are demonstration only.

## Use your own warehouse

Generate the files locally, then load them into a database and schema you control using your warehouse's file-loading tools.
Follow the example's file-to-table mapping, preserving column names, key types and timestamps.
Fill in the blank connector configuration with your own account, database, schema and authentication settings.
Keep credentials outside version control.
The example retains its platform's connector API so that you can later point it at your own data.
