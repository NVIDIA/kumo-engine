# Synthetic SDM data

Run generators from the collection root (`examples/kumo-example-notebooks`) with NumPy, pandas and PyArrow installed.
All output stays on the local filesystem; the scripts never upload data or connect to a warehouse.
Seeds fix the samples and targets contain noisy predictive signal.
Synthetic metrics are demonstration only.

- `python synthetic-data/sdm/generate.py DATASET --output data/DATASET` creates training and label-free scoring files for the corresponding SDK example.
  Run `--help` for dataset names; the example's header supplies its exact command and `--data-dir` argument.
  Wide molecular, spectral and transaction feature sets use a reduced width to keep files small.
- `readability.py --output data/readability` generates synthetic passages and readability targets in `passages.parquet`.
- `relational.py --output data/sales` generates the related customers, addresses, sales and item tables used by the SALT-style field-completion and RelBench-style temporal entity examples.
  Foreign keys resolve to generated customer and sales IDs.
- `notebook_tables.py breast_mass --output data/breast_mass` and `notebook_tables.py vehicle --output data/vehicle` reproduce the notebook measurement tables.
- `account_closure.py --output data/account_closure` generates the shared account-closure cohorts, with prospective outcomes left null.
- `time_series.py --output synthetic_data/time_series` generates seasonal hourly, energy, retail, weather and anomaly data for the commented model previews.

SDK workflows that invoke a generator locate it relative to their source file, so keep the repository layout intact.
Notebooks embed their generation code in one setup block and can be used independently.
The Snowflake notebook explains how to upload its locally generated CSV to your own stage and then read it with the Snowpark connector.
Model-weight downloads remain part of model setup; no external source datasets are needed.
