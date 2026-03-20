# diskgraph-train-table

Generate a training-table parquet from an ingested DiskGraph index and a predictive query plan.

## CLI

```bash
./target/debug/generate --config /path/to/job.json
```

## Query TUI

```bash
./diskgraph-train-table/scripts/train_table.sh /path/to/index
```

Non-interactive mode:

```bash
./diskgraph-train-table/scripts/train_table.sh /path/to/index -- -C "PREDICT ..."
```

## E2E example (H&M)

```bash
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

## Current output contract (native artifact pipeline)

The TT executor produces parquet batches (entity/timestamp/target and optional helper columns).
Downstream (`kumo`) appends contiguous `arange` and writes manifest/artifacts.

Split is **not** a parquet column in the native path.
Instead, split assignments are represented as a separate dense row-stage artifact:

- format: `split_assignment_row_stage_v1_bin` (legacy format label)
- payload: `u8[N]` aligned to TT `arange`
- stage ids:
  - `0 = train`
  - `1 = val`
  - `2 = test`
  - `255 = unused`

## Split computation ownership

For the DG-native path, split semantics are producer-owned by the DiskGraph side.
Kumo passes split intent (`split_kind` + params) and consumes producer output.

Currently supported families for dense row-stage assignment:

- `FixedSplit`
- `TemporalSplit`
- `ApproxDateOffsetSplit`
- `DateOffsetSplit`
- `TimeRangeSplit`
- `BacktestingSplit`
- `ColumnSplit`
- `ManualSplit`
- `RandomSplit`

Non-authoritative for now:

- `RandomLinkSplit` (array-level semantics not yet implemented natively)
