# diskgraph-train-table

Generate a train table parquet from an ingested Diskgraph index and a predictive query.

## CLI

```
./target/debug/generate --config /path/to/job.json
```

## Query TUI

```
./diskgraph-train-table/scripts/ttui.sh /path/to/index
```

Non-interactive mode:

```
./diskgraph-train-table/scripts/ttui.sh /path/to/index -- -C "PREDICT ..."
```

## E2E example (H&M)

```
./diskgraph-train-table/scripts/hnm_temporal_e2e.sh
```

This script downloads the H&M dataset, infers `diskgraph_config.json`, ingests a local index, and generates a temporal train table with a filter.

## Job config (JSON)

```json
{
  "index_path": "/path/to/ingested/index",
  "query": "PREDICT SUM(transactions.price, 0, 7, days) FOR EACH users.user_id",
  "output_parquet": "/tmp/train.parquet",
  "split": {
    "type": "random",
    "fractions": [0.8, 0.1, 0.1],
    "seed": 123
  },
  "options": {
    "forecast_length": 1,
    "lag_timesteps": null,
    "year_over_year": false,
    "list_distinct_target_is_array": true,
    "drop_earliest_timeframe": true,
    "allow_empty_target": false,
    "temporal_sampling": "random"
  },
  "validation_output": null
}
```

### options.temporal_sampling

- `random`: sample uniformly across each entity's temporal range. Sample count is proportional to the target window size.
- `timeframe`: sample per target aggregation window (aligned windows).

## Output schema

Parquet columns:
- `entity` (preserves key type)
- `target` (float, count, or list)
- `timestamp` (unix seconds)
- `weight` (optional)
- `split` (train/val/test label)
