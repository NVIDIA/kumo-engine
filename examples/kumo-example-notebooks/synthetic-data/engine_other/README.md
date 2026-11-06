# Engine example data

From the collection root (`examples/kumo-example-notebooks`), run `python synthetic-data/engine_other/generate.py --dataset movielens --output-dir ./example-data/movielens`.
Install `numpy pandas pyarrow` first.
The destination must be new, so an existing dataset is never overwritten.
Every generator writes local Parquet files and uses seed 42 by default.
`--seed` changes the simulation without changing its schema.

| Dataset | Examples | Signal | Default bytes |
| --- | --- | --- | ---: |
| `movielens` | Average and pair ratings | Movie quality and genre preference | 936,222 |
| `netflix` | Snowflake movie-rating notebook | Movie quality and genre preference | 924,218 |
| `transformer` | Transformer temperature | Daily load cycle predicts temperature | 942,974 |
| `sunspots` | Monthly forecasting | Noisy eleven-year cycle | 48,891 |
| `home_credit` | Credit default | Debt burden changes default probability | 807,208 |
| `financial` | Loan status | Account income changes repayment probability | 899,194 |
| `stack` | Comment recommendations | User topic preference selects posts | 890,110 |
| `mlops` | Fit/score and scheduler companions | Loyalty controls repeat-order frequency | 966,130 |
| `modules` | File, S3, Snowflake, graph, collection and prompt modules | Loyalty and periodic sensor signal | 831,256 |
| `fraud` | Databricks fraud notebook | Merchant risk changes report probability | 526,892 |

The monthly single-entity series remains small: its 200 years already provide many cycles without inventing thousands of years merely to reach a byte target.
Small notebook modules retain their explicit teaching populations and row IDs.

Both money-muling notebooks share the account-network simulation in `money_muling.py`.
Run `python synthetic-data/engine_other/money_muling.py --output-dir ./example-data/money_muling` to write its five tables locally (475,880 bytes).
The notebooks include that generation inline, so they can be used independently of a repository checkout.
The hidden account propensity controls transfer activity and future report probability; future reports remain outside historical model features.

Customer lifetime-value and fashion examples reuse `../engine_retail/generate.py` with `--dataset customerltv` and `--dataset fashion` respectively.

## Connect generated files

Local examples pass the generated directory to `--source-root` or `SOURCE_ROOT`, as documented in the script.
For Snowflake, load each Parquet file into your own database and schema under its filename stem, preserving column types and case.
Quote mixed-case identifiers for the `Movies`, `Users` and `Ratings` example.
Set your connection name and source table for the connector inspection script; the notebook uses blank warehouse, database and schema fields.
For Databricks fraud, load `customers`, `merchants`, `transactions`, `fraud_reports` and `prediction_population` into your own catalog and schema with matching column names and types.
Fill in your workspace host, service-principal credentials, SQL warehouse, catalog and schema in the connector cell.
For S3, upload the files yourself to your bucket and set `SOURCE_ROOT`, `SOURCE_TABLE` and `SOURCE_URI` to that location.
These generators do not connect, upload or provision resources.

Synthetic metrics are demonstration only.
