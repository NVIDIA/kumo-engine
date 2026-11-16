# Supported features and limitations

kumo-pytorch is alpha software. This page says what works today and
what isn't supported yet. If your contract uses something that isn't supported,
you get an error that names the declaration involved. Most of these errors come
from `check` or `prepare`, before any data is read; the table below says when
each DiskGraph refusal happens. Errors that depend on the data, such as integer overflow, are raised
when the affected values are read. The [behavior guarantees](../conformance/README.md) state the exact rules
the DiskGraph engine is tested against, and this page says where it doesn't
meet one yet. The [API reference](api/README.md) covers each declaration.

## Status

| Area | Status |
| --- | --- |
| Declarative contract, validation, JSON, fingerprints | Implemented |
| Point-in-time selection and leakage boundaries | Implemented |
| Encodings, fitting, layouts, targets, previews | Implemented |
| Artifacts, inference mode, serving from caller-supplied rows | Implemented. Serving builds batches in Python, the same way the in-memory engine does |
| `DiskGraphEngine` over local Parquet | Graphs with node features, event sequences, aggregates, generated labels and columnar model batches; see [DiskGraph](#diskgraph) |
| `InMemoryEngine` over `InMemoryRows` | Implemented, for trying things out on tiny data |
| Streaming, prefetch, caching, remote storage | Not implemented |
| Artifact and schema migration | Not implemented: an artifact loads only in the version that wrote it |

kumo-pytorch doesn't provide the following:

- Array, list and struct columns, or parsing of multi-category values stored as
  strings.
- UUID and IP address types.
- Fitted calculations other than the encoders, and choosing or weighting which
  observations an encoder is fitted on, other than `fit_split` and `encoder_cutoff`.
- Updating a saved artifact, or reusing only part of one.
- Online serving infrastructure, such as deployment, routing and row lookup.
- Indexes spread over several machines. The rank and world-size arguments split
  anchors across processes; they don't split the index.
- Checking an index's shard count when it's reused. A DiskGraph index opened
  with a different `num_shards` from the one it was built with is reused as
  built, and graph samples with ties may then differ from those of an index
  built with the new shard count.

To reproduce an experiment, pin the package version and keep the artifact (the
contract and its fitted state) that you trained with.

## DiskGraph

DiskGraph supports direct entity fields, historical event fields, typed scalar
calculations, supported aggregates and aggregate comparisons, nested grouping,
generated scalar and row labels, and recent graph neighborhoods with node
fields. It can also build model-ready tensors directly
(`output_format="model"`).

The table below lists what it doesn't support yet. Each code starts with `DG-`
for DiskGraph. The table is generated from
`DiskGraphEngine.unsupported`, the engine's own list. Each refusal carries the
entry's code as `ValidationIssue.code`, names the declaration involved, and
suggests the next step shown here. Entries refused at `check()` are refused by
both `check` and `prepare` before any data is read; the others are refused while
`prepare` binds the data.

<!-- generated from DiskGraphEngine.unsupported by tests/limitations_reference.py -->
| Code | Refused | Refused at | Instead |
|---|---|---|---|
| `DG-SOURCE-KIND` | Tables and anchor tables must be bound to Parquet files; other sources, such as `InMemoryRows`, are refused. | `check()` | Write the rows to a local Parquet file and bind it with `Parquet(path)`. |
| `DG-REMOTE-PARQUET` | The DiskGraph engine doesn't support remote Parquet paths, such as `s3://`, yet, for tables or anchor tables. | `check()` | Copy the files to a local disk and bind the local path. |
| `DG-RELATION-KEYS` | A relation must have exactly one side that is its own table's primary key; joins of two foreign keys, and of two primary keys, are refused. | `check()` | For two foreign keys, relate each to a key table whose primary key is the shared key; for two primary keys, add a copy of the key as an ordinary column of one table and relate through that column. |
| `DG-TIMED-INTERMEDIATE` | A `Rows` path can't pass through an intermediate table that has a clock, because the cutoff can't be applied at an intermediate hop. | `check()` | Use a direct relation if it gives the same rows, or precompute the joined rows into one table before writing the Parquet files. |
| `DG-ROW-FILTER` | Row filters, in `Rows.filters` or `EntitySpec.filters`, are refused. | `check()` | Precompute the condition as a column, or write only the matching rows to their own table, before writing the Parquet files. |
| `DG-ACTIVE-INTERVAL` | Active intervals (`Rows.active`) are refused. | `check()` | Write one event row per change of state, with its time as the table clock. |
| `DG-ENTITY-VALUE-KIND` | The DiskGraph engine doesn't support entity values that look up a field through a relation path or by its own time column yet. | `check()` | Precompute the looked-up value as a column of the root table. |
| `DG-JOINED-FIELD` | Event and row-target fields must read the event row itself; a field with a relation path or its own time column is refused. | `check()` | Precompute the joined column onto the event table before writing the Parquet files. |
| `DG-DATED-ENTITY-VALUES` | The DiskGraph engine doesn't support `SampleSpec.dated_entity_values` yet. | `check()` | Declare an event input (`EventSpec`) over the same `Rows` instead. |
| `DG-CURRENT-EVENTS` | The DiskGraph engine doesn't support `SampleSpec.current_events` yet. | `check()` | Read the event through an ordinary historical event input. |
| `DG-CANDIDATE-SETS` | Candidate sets, in `SampleSpec.candidate_sets` or as candidate columns of an anchor table, are refused. | `check()` | Build the candidates' features with a second contract whose entity is the candidate table, anchored at the same times, and pair them in your own code. |
| `DG-UNIFORM-GRAPH` | Uniform graph sampling isn't supported yet: DiskGraph's native uniform mode doesn't yet keep each example to rows at or before its own prediction time. | `check()` | Use `strategy="recent"`. |
| `DG-GRAPH-CLOCK-OVERRIDE` | The DiskGraph engine doesn't support a graph traversal `time_column` other than the destination table's clock yet. | `check()` | Use the destination table's declared clock. |
| `DG-STATIC-TIMED-TRAVERSAL` | A graph traversal into a table with a clock can't be `temporal="static"`. | `check()` | Use `temporal="inherit"`. |
| `DG-REDUCTION-OPERAND` | Reductions over decimal or bytes columns are refused; direct decimal field reads are supported. | `check()` | Cast decimals to float64 or integer units, and bytes to a string or integer key, before writing the Parquet files. |
| `DG-COLUMN-RENAME` | `Parquet(columns=...)` renames in a table binding are refused; physical column names must match the declared ones. Anchor tables accept renames. | `prepare()` | Rename the columns in the Parquet files to the declared names. |
| `DG-INT64-CLOCK` | A column declared as a timestamp must be stored as a Parquet timestamp; an int64 column of epoch values is refused. | `prepare()` | Write the column as a Parquet timestamp in the declared unit. |
| `DG-NESTED-COLUMN` | Parquet columns of list, struct or map type are refused; every declared column must be stored as a scalar. | `prepare()` | Flatten the values into scalar columns, or into rows of their own table, before writing the Parquet files. |
| `DG-DUPLICATE-LAYER-RELATION` | A graph layer can't sample the same relation twice, including two relations that resolve to the same stored relation. | `batches()` | Sample the relation once in the layer, with the combined fanout. |
<!-- end of generated list -->

## Scope and source data

You supply the examples, split membership, prediction times, external labels
and candidate lists. kumo-pytorch selects data and builds batches; choosing
training populations, generating negatives, training models and implementing
attention and losses are up to you.

Every column has a type (`DType`). With Parquet you can leave types out and
they're read from the files; with `InMemoryRows` you must give them. Supported
scalar types are
booleans, signed/unsigned integers, float32/64, strings, bytes, date32,
whole-second UTC timestamps, and decimal128/256. Arrays and structs are not
supported scalar column declarations. Every table names its time column (its
clock) or declares `NON_TEMPORAL`.

Entity and candidate lookup require the chosen table's declared primary key.
Other tables may set `primary_key=None`; identical source rows remain separate
occurrences. Their internal IDs and equal-time order are local to the index,
so tied limits can select different rows after a rebuild or in a serving subset.
Relations join
one column on each side; composite keys and arbitrary join predicates are not
expressible. DiskGraph applies its native ingestion rules: rows it rejects,
such as rows with a NULL clock, are dropped, and building the index reports how
many were dropped. The in-memory test engine requires declared primary keys to
be unique and non-NULL.

A column's type doesn't choose its encoding; you do. Values keep their declared
type: kumo-pytorch won't turn numbers into strings, parse numeric strings, or
treat booleans as integers. Decimal precision
and scale and integer ranges are checked. Raw floating values can contain NaN
or infinity; encoders that require finite observations reject them.

## Time

- Stored timestamps round toward positive infinity to their declared tick.
  Anchor (`as_of_time`) times round downward to the graph clock, including
  negative epochs. This preserves the reference engine's existing causal
  boundary; the reference engine remains seconds-only.
  Naïve datetimes mean UTC. Integer anchors are epoch seconds; integer timestamp
  field values use their declared ticks. Use typed timestamps for subsecond anchors.
- Windows use `(start, stop]`, measured in seconds from each anchor: the start
  is excluded and the stop is included.
  Historical context stops at or before zero; generated targets start at or
  after zero.
- A current event (`SampleSpec.current_events`) is the event stored at the
  anchor's own second, so its window must be exactly `(-1, 0]` with `limit=1`.
- Equal-time merged events sort by component name and primary key, or by local
  internal identity for a keyless table.
- Calendar calculations use UTC. `EventTimeDelta` is anchor minus event time,
  in its declared unit; it is not relative to the latest event.
- Graph sampling needs a time column both to filter neighbors by time
  (`temporal="inherit"`) and to keep the most recent ones (`strategy="recent"`),
  even for a `temporal="static"` traversal.
- On the DiskGraph engine, a graph traversal can't reach a table without a
  clock (`NON_TEMPORAL`). Such a traversal must be `temporal="static"`, and the
  only strategy that doesn't need a clock, `strategy="uniform"`, is refused
  (`DG-UNIFORM-GRAPH`). A clockless table can still be a neighborhood's root
  entity. If you want to reach a dimension table such as products, give it a
  clock, for example the time each row was first listed.
- Which of several equally timed graph neighbors a fanout keeps is unspecified,
  but the behavior guarantees require the choice not to depend on row order,
  how the tables are split into files, or how the index is sharded. The
  DiskGraph engine doesn't meet that yet: its choice depends on how the index
  was built, so rows dated after an example's prediction time, the row order in
  the files and the number of index shards can all change which tied neighbor
  a sample keeps. The kept neighbors are always dated at or before the
  prediction time.
- On the in-memory engine, and so also in serving, the cutoff of a `Rows` path
  applies only to the table the path ends at. If the path passes through an
  intermediate table that has a clock, rows reached through an intermediate row
  dated after the prediction time are still selected. The DiskGraph engine
  refuses such paths (`DG-TIMED-INTERMEDIATE`), so it never reads them.

## Calculations

Calculations in `SampleSpec` infer result types from their operations and
resolved operands. An optional explicit type is checked against that rule.
Constants, MapValues results, and external labels require explicit types.

| Operation | Accepted input | Result |
| --- | --- | --- |
| `count`, `count_distinct` | Selected rows; distinct uses its selected column | `int64` |
| `sum` | Numerical | Exact `int64` for signed integers, exact `uint64` for unsigned integers, `float64` for floats; exact decimal for decimals, which DiskGraph refuses |
| `mean`, `std_population`, `std_sample`, `var_population`, `var_sample` | Numerical | `float64` |
| `min`, `max` | Numerical, string, date32, timestamp | Same dtype |
| `mode` | Scalar | Same dtype |
| `recency` | Date32 or timestamp in historical context | `float64` seconds |
| `add`, `subtract`, `multiply` | Numerical operands, including mixed integer/float types | Arrow numeric promotion; optional compatible widening |
| `divide` | Numerical operands, including different numerical dtypes | `float64` |
| `log1p` | Numerical | `float64` |
| `abs` | Numerical | Same family, with explicit compatible widening |
| `sign` | Numerical | `int64`, containing -1, 0, or 1 |
| `CalendarPart` | Timestamp; date32 for date components | `int64` |
| `MapValues` | Typed scalar members | Explicit result dtype |
| `EventTimeDelta` | Event timestamp and anchor | `float64` |

Numerical excludes boolean. Arithmetic follows Arrow's promotion of operand
types; for example, int64 plus float64 produces float64. An explicit compatible
widening can increase integer width, widen float32 to float64, or increase
decimal precision. Decimal multiplication uses the combined operand scales.
Integer and decimal arithmetic checks actual result bounds. Floating-point
calculations preserve their ordinary rounding and overflow results, including
infinity; encoders may separately require finite input.

Map members are normalized to the operand dtype before checking overlap; they
must be finite and non-NULL. NULL operands remain NULL rather than selecting a
map entry or fallback.

Named calculation dependencies may appear in any declaration order, but must
exist and be acyclic. Operands must be scalar; a repeated `Field(selection="all")`
is not a calculation operand. Operand tuple order is meaningful.

### Missing and empty results

NULL inputs are ignored by reductions. An empty row set produces zero for
count, count-distinct, and sum. A nonempty row set whose summed values are all
NULL produces a missing sum. Other reductions without observations are
missing.

Population variance and standard deviation divide by `n`; sample variants
divide by `n - 1`. One observation gives population spread zero and missing
sample spread. Missing arithmetic operands remain missing. Division by zero
and `log1p(x)` for `x <= -1` produce missing values.

Mode ignores NULL. Equal-frequency values use the stable ordering
`(type name, repr(value))`; this is deterministic, not numerical tie-breaking.
Raw floating `count_distinct` and mode identify values by their IEEE bit
patterns: positive and negative zero are distinct. A `Vocabulary()` encoder, on
the other hand, treats +0 and -0 as the same category.

### Exact and floating-point sums

Integer sums accumulate exactly, including values beyond float64's exact
integer range: 2^53 plus 1 produces 9007199254740993.
Results outside the declared int64 or uint64 range fail rather than wrap.
Floating sums use float64 operands and accumulation, so two values of 1e308
produce infinity. DiskGraph refuses reductions over bytes or decimals. Raw
reads and scalar encodings of those types are separate operations.

Integer min, max, mode, count-distinct, and sum preserve exact integer values.
Mean, variance, and standard deviation explicitly return float64.

## Encodings and arrangements

| Encoding | Compatible raw dtype |
| --- | --- |
| `Raw()` | Every supported scalar dtype |
| `Vocabulary()` | Every supported scalar dtype |
| `ZScore()` | Numerical |
| `Buckets()` | Numerical |
| `Text()` | String |

`Raw()` preserves dtype. Inspectable batches can hold strings, bytes, dates,
timestamps, and decimals; model-format raw outputs require a directly supported
tensor dtype and otherwise refuse before fetching a batch. Returning a raw
date or decimal does not imply an automatic numerical interpretation.

Each field in an input has exactly one encoding, which you choose.
`UseEncoding(name=...)` shares an encoder between fields of the same type, and
inline encoders are never shared. An `IdSpace` gives several discrete encoders
one combined numbering without merging what they learned, and field-name IDs
tell the model which field a value came from. Special IDs such as padding and
masking are reserved when you declare them. [Encodings and
IDs](03-encodings-and-ids.md) explains all of these.

`Columns()` keeps each field as its own typed array. `Tokens()` needs discrete
encodings, keeps field order, and can expand a text value into several pieces.
`Composite()` gives each combination of field values one ID, and a combination
it hasn't seen gets the composite's unknown ID. Text and repeated fields can't
be part of a composite.

A `Flatten()` history concatenates events within each sample. Grouped packed
histories can have the same value buffer, but retain the event boundaries.
Padding and offsets describe storage, not model attention behavior.

Candidate inputs use columns with a padded or packed candidate layout.
Neighborhood inputs use graph output, and graph node fields support scalar
column encodings but not field-name arrays. `EventsOf` reads extra values for
the events of a grouped event input, and `OccurrencesOf` returns raw
occurrence values for a grouped entity-token input. Both line up with the rows
the input they refer to already selected, rather than selecting rows of their
own.

## Targets and exposure

External labels require an `ExternalTarget(dtype=...)` declaration. Absent
external labels remain missing. Supplied labels cannot replace generated
labels. Generated targets have explicit future windows.

Aligned event labels use the input's retained events and layout. Their source
fields must be withheld from inputs. Masking changes only its named input;
another input exposing the same qualified field produces an
`InputExposureWarning`, not automatic masking or rejection. Mask targets use
reserved replacement IDs and deterministic sample/field coordinates.

Candidate labels follow the candidate order you provide. Pair-specific
attributes and candidate generation aren't implemented. DiskGraph doesn't
support current events, dated values or candidate sets yet.

## Fitting, saved state, and batching

Encoders over computed values learn only from the selected fit anchors. Shared
input encoders collect observations from each selected input use; selecting
overlapping data in several inputs can contribute it several times. Encoders
over stored columns instead learn from each table row created at or before the
encoder cutoff, once per column. External or generated
labels do not silently enter a shared input vocabulary.

An artifact doesn't include a copy of your data. `Artifact.load` reloads an artifact written by this version; one whose
manifest records no file digests is refused. Inference mode emits no targets
and reads no target rows; serving from supplied rows uses the in-memory engine
and inspectable output.

Batching supports sample limits, actual emitted-position token limits,
length bucketing, deterministic shuffling, and rank partitioning. Token-budget
batching with `world_size > 1` is refused because ranks can otherwise emit
different numbers of steps. Short-rank repetitions are marked in
`metadata["is_padding"]`; real anchors are assigned disjointly before padding.
kumo-pytorch splits anchors across ranks; moving batches to GPUs and running
distributed training is up to your training loop.

## Engine support

`DiskGraphEngine` reads an index built from local Parquet files and builds
model output column by column. It raises an error for the joins, filters, graph
strategies and sampling features it doesn't support; see
[DiskGraph](#diskgraph). Where both engines support a contract, DiskGraph's
batches can still differ from the in-memory engine's, and so from serving's, in
a few ways: the order of graph nodes within a layer, `std`, `var` and `mean`
values within floating-point rounding, and which tied neighbor a `recent`
fanout keeps, as described under [Time](#time).
`InMemoryEngine` runs the contract in Python over `InMemoryRows`, for trying
things out on tiny data.
