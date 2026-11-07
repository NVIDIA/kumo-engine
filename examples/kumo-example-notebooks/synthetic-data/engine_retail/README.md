# Synthetic retail data

Run from the collection root (`examples/kumo-example-notebooks`) with `numpy`, `pandas` and `pyarrow` installed:

```bash
python synthetic-data/engine_retail/generate.py --dataset fashion --output-dir ./synthetic-fashion
```

Each invocation creates a new directory containing roughly 1 MB across all tables.
An existing directory is refused.
The default seed is 42; `--seed` selects another reproducible sample.
The generator never downloads data or writes warehouse objects.

| Dataset | Examples | Files |
| --- | --- | --- |
| `fashion` | Customer churn, H&M-shaped recommendations, paired queries, weighting | Customers, articles, transactions and article pairs |
| `customerltv` | Customer lifetime value and reusable-model workflows | Customer, stock and transaction |
| `coupon` | Coupon recommendations | Users, coupons, purchases and visits |
| `olist` | Customer recommendations | Orders, customers, products, payments and reviews |
| `instacart` | Basket size | Users and orders with item counts |
| `m5` | Sales forecasting | Sales entities/events, prices and calendar |
| `matching` | Product entity matching | Two product CSV tables and labeled train/valid/test pairs |
| `stores` | Composite user/location recommendations and SQL preprocessing | Users, stores and orders |
| `tpch` | Warehouse task notebooks | Customer, orders, lineitem, part and dimension tables |

Use the output directory as the owning script's `--source-root` or `--input-dir`.
Reload and evaluation scripts consume artifacts from the matching training workflow, not a newly generated model or fabricated predictions.

For Snowflake, manually load the generated files into your own database/schema, preserving numeric keys and timestamp types.
The fashion example expects `CUSTOMERS`, `ARTICLES` and `TRANSACTIONS`, with uppercase columns; use `transactions.parquet` for `TRANSACTIONS`.
SQL preprocessing expects `USERS`, `STORES` and `ORDERS` with uppercase columns.
TPC-H expects uppercase names on Snowflake and lowercase names on Databricks.
Configure your own connector credentials and warehouse identifiers after loading.

Latent customer segments determine product preferences, spending and early purchasing cessation.
Generated dates span 2023 through 2025, giving temporal tasks historical context and future label windows.
Metrics are demonstration only and do not estimate real-world model quality.

AdventureWorks examples retain their approved original dataset and preparation instructions.
