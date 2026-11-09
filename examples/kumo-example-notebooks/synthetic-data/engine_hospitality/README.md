# Synthetic hospitality data

Run from the collection root (`examples/kumo-example-notebooks`) with `numpy`, `pandas` and `pyarrow` installed:

```bash
python synthetic-data/engine_hospitality/generate.py --dataset restaurant --output-dir ./synthetic-restaurant
```

Choose `restaurant`, `recipes` or `booking`.
Each invocation creates about 1 MB of related tables in a new directory and refuses an existing destination.
Set `--seed` to change the reproducible sample; the default is 42.
Nothing is downloaded or uploaded.

Restaurant produces CSV users, restaurants, interactions and social relationships matching the restaurant scripts.
Use the output as `--input-dir`; prediction and reload workflows retain their prepared-table and model requirements.
Recipes produces Parquet customers, menus, recipes and orders with dates inside recipe availability.
Booking produces Parquet users, sessions and demographic buckets; browsing preferences correlate with booking destinations.
Use recipes and booking output as `--source-root`.

Restaurant cuisine preferences correlate with visits, and user segments vary activity frequency.
The history spans 2023 through 2024 and includes the examples' January 2024 scoring boundaries.
Metrics and predictions are demonstration only.

Restaurant interactions have four classes: `visit`, `review`, `favorite` and `reservation`.
User preferences influence the class, with event variation allowing multiple interaction types in a daily label window.
