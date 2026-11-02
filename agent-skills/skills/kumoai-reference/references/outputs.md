# Predictions, embeddings and destinations

`Trainer.predict` selects one artifact per call. Without `output_config`, it produces local
prediction Parquet files. `num_rows` is the selected output row count; `.frame` collects all
files into pandas and should be used only for bounded results. Local files require explicit
retention to survive runtime teardown.

Inspect schema without collecting all rows:

```python
import pyarrow.parquet as parquet

for filename in predictions.files:
    print(filename, parquet.ParquetFile(filename).schema_arrow)
```

Binary/regression results have scalar output columns. Multiclass/multilabel and rankings
use long class rows. Forecasts use expanded timestamp rows. Embeddings include vector-valued
EMBEDDING columns and entity metadata; link embeddings can include ID, TYPE and NAME for
self-links. Preserve the actual schema and task rather than assuming one row per entity.

## Write predictions

Inputs: `output_connector` scoped to the confirmed destination database/schema, exact bare
`OUTPUT_TABLE`, explicit `WRITE_MODE`, and task-specific `PREDICT_OPTIONS`. WriteMode has
APPEND and OVERWRITE. Select the mode deliberately: File export OVERWRITE replaces existing data while retaining the table and grants when its schema matches;
APPEND can duplicate outputs on retry. A column mismatch fails rather than silently changing the existing schema. Check destination
existence and access first.

```python
import kumo.engine as kumo
from kumoapi.jobs import PredictionArtifactType, WriteMode

output = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.PREDICTIONS,
    table_name=OUTPUT_TABLE, write_mode=WRITE_MODE,
)
written = trainer.predict(
    population, snapshot=snapshot, output_config=output, **PREDICT_OPTIONS,
)
print(written.num_rows, written.files)
```

The exact table name is used; no suffix is appended. Quote case-sensitive identifiers through
Snowpark's identifier handling. Read back, with the session taken in the first cell
(`kumoai-reference/references/kumo_finetuning_sdk.md` §3.3), using `session.table([OUTPUT_DATABASE, OUTPUT_SCHEMA,
OUTPUT_TABLE])`, inspect schema and bounded values. For a new/replaced table compare its count
with `written.num_rows`. For APPEND compare the count delta against a recorded pre-write count
only without concurrent writers, or use a run-specific destination for unambiguous verification.
Check entity/anchor/class coverage too. A count alone does not prove semantic correctness.

## Requested embeddings

Predictions and embeddings are separate calls and destinations. For a fitted or loaded trainer,
using the same compatible snapshot and population:

```python
embedding_output = kumo.PredictionOutputConfig(
    connector=output_connector,
    artifact_type=PredictionArtifactType.EMBEDDINGS,
    table_name=EMBEDDING_TABLE, write_mode=WRITE_MODE,
)
embeddings = trainer.predict(
    population, snapshot=snapshot, output_config=embedding_output, **PREDICT_OPTIONS,
)
print(embeddings.num_rows, embeddings.files)
```

Preserve task-specific arguments, including the explicit binary threshold where the API
requires it. Do not promise embeddings for every module/task: report an unsupported selection
error without silently switching models or refitting. Link models can expose entity and item
vectors; inspect TYPE and identifiers before calling them customer/product embeddings.

## Serialization boundaries

On Snowflake, the public prediction path exports selected Parquet through Snowflake's file sink,
with schema inference and COPY. This differs from the separate in-memory DataFrame writer, which rejects
list/dict columns. Do not apply that rejection indiscriminately to prediction export, and do
not send nested cohort JSON through the scalar DataFrame path unchanged.

For vectors or other nested fields, inspect the destination's actual column type and a
bounded read-back value; inference is not a guarantee of the consumer's expected representation.
Use [stage retention](../../kumo-model-persistence/references/retention.md) to preserve the original
Parquet or JSON when a destination cannot represent the required shape. Conversion/flattening
requires an explicit schema decision. Do not claim that all artifact families fit an ordinary
table unchanged or automatically rerun prediction after an export failure.
