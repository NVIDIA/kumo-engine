# Approved examples beside their English rules

This directory holds the behavior guarantees (English rules), their golden
cases (small inputs with answers calculated by hand), and the test helpers that
check them. It is not included in the `kumo-pytorch` wheel and does not decide
runtime engine support. The guarantees live in `spec/guarantees/`.
G-ROWS-001 is a worked example of existing behavior and an existing refusal; its
golden cases are real required checks.

kumo-pytorch defines engines through its `Engine` interface and ships one, the
DiskGraph engine. The guarantees bind the engine, and the conformance tests check
them on the DiskGraph engine. Each golden answer is calculated by hand from its rule
and is the normative check; where the DiskGraph engine falls short of a rule,
`expectations.json` records its refusal or `registry.py` records a known gap.
The in-memory engine is a test tool, also handy for trying things out on tiny data, and
the guarantees say nothing about it. The conformance tests run the same golden cases and
generated properties on it and compares the two engines, as described under
[Differential testing](#differential-testing). A discrepancy is judged against
the rule, never against either engine's output.

```text
conformance/spec/guarantees/
  G-ROWS-001/
    guarantee.md
    golden/
      C-ROWS-001.json
      C-ROWS-002.json
tests/conformance/
  test_guarantees.py        # independently requires answer or refusal per engine
  test_case_validation.py   # malformed examples must fail
  test_assertions.py        # wrong answers/refusals must fail
conformance/expectations.json  # independent engine assertions, not support declarations
```

## English file

Each guarantee directory has exactly one `guarantee.md`, beginning with
`# G-<TOPIC>-<NUMBER>: title`. Use uppercase letters, digits, and hyphens in
IDs. Its directory owns its examples; no case-to-guarantee mapping is repeated
inside JSON. Case IDs are unique across all guarantees, even after moves.

Use these four nonempty sections:

```markdown
## Rule

Describe the requested inputs and precise result.

## Boundaries

Explain equality, zero/empty outcomes, and other relevant edges.

## Scope

State assumptions, open questions, and compatibility with existing behavior.

## Golden examples

- [C-TOPIC-001](golden/C-TOPIC-001.json): explain what this example establishes.
```

The document must link every golden JSON in its directory and no missing case.
An existing guarantee can gain more examples without being copied. A
guarantee's meaning is settled independently of engine support.

## JSON file

The filename is the stable case ID. The following six fields are required;
unknown fields fail validation:

| Field | Content |
|---|---|
| `id` | Exactly the filename stem, such as `C-ROWS-001`. |
| `contract` | The complete result of the real `DataLoaderSpec.to_dict()`. Structural validation and a real `to_json()` → `from_json()` round trip must preserve values, explicit sequences, and both fingerprints. Engine assertions use the loaded contract. |
| `tables` | Every declared table mapped to a list of tiny rows. Each row supplies exactly the declared columns. Primary keys are unique non-null integers or strings. |
| `anchors` | One to eight objects containing only `sample_id`, `entity_id`, and integer-second `as_of_time`. Sample IDs are unique and root entities exist. |
| `expected` | Every sample ID mapped to `{"entity_values": {"field": agreed_value}}`. Assert every declared raw entity value, including NULL for absent values. Event-only cases use an empty entity-values mapping and assert their selected batch buffers. |
| `reason` | A nonempty, independent calculation of the expected values from the listed rows and rule. |

Three optional fields extend the observation without changing the contract:

| Field | Content |
| --- | --- |
| `batch` | Selected named `inputs` and/or `targets`, with complete typed-array assertions for the requested buffers. |
| `state` | Selected fitted-state records; every requested leaf must exist and equal its independent answer. |
| `physical_types` | Table/column Arrow type-name overrides for deliberately different physical fixtures. Ordinary fixtures use the declared DType. |

The one-to-eight limit is explicit because public `preview()` returns at most
eight samples. Duplicate JSON keys, NaN/infinity, missing answers, unknown
projections, orphan JSON, duplicate IDs, and broken document links fail. The
loader cannot determine whether an English calculation is correct; review
supplies that judgment.

Raw entity numbers compare exactly, allowing integer `2` and float `2.0` for
the same raw number; booleans are not accepted as numbers. Batch arrays are
stricter: each requested leaf names `dtype`, `shape`, and all `values`. The
checker reads the actual Torch/NumPy/Arrow type and dimensions, then compares
every element in order. Integer IDs cannot be asserted as floating values.
A floating projection may specify a nonnegative absolute `atol` justified by
its guarantee; no tolerance is inferred from observed output. A nested mapping
selects named buffers, never an arbitrary expression or a prefix of an array.
Missing requested buffers, incorrect shapes, and unexpected assertion options
fail. These checks establish only explicitly asserted outputs, not all model
behavior or model quality.

For example, this asserts every position of one two-element ID tensor:

```json
{"dtype": "int64", "shape": [2], "values": [2, 3]}
```

Source fixture scalars that JSON cannot represent use explicit tags:
`{"$bytes":"AAE="}`, `{"$decimal":"1.10"}`, `{"$date":"1970-01-02"}`,
`{"$timestamp":"1970-01-01T00:00:00+00:00"}`, or
`{"$float":"nan"}` (also `inf` and `-inf`). Ordinary strings are never parsed
as dates/decimals/numbers. Unknown or malformed tags fail loading; a tag does
not override the declared source dtype. The same tags can express exact expected
typed-array values. Portable JSON must not contain literal NaN or infinity.

Every case exercises the public JSON writer and reader before engine checking,
including cases whose engine expectation is refusal. The shared loader compares
declarations using an independent sorted-key encoding that ignores dictionary
placement but preserves array order. It also checks full and input-only
fingerprints. Each required engine assertion runs the original, reversed-key,
and fixed-seed shuffled-key forms through JSON loading and saving, then compares
them against the same approved answer or refusal. Arrays are never shuffled.
[G-CONTRACT-001](spec/guarantees/G-CONTRACT-001/guarantee.md) states the ordering
and identity rules; focused tests cover token layouts and calculated dependencies
that the raw-value golden format cannot fully certify.

The in-memory engine uses the listed rows directly. DiskGraph gets temporary
Parquet and an isolated index. Every column is written using its declared Arrow dtype,
including empty and all-NULL tables. Clock fixtures use integer epoch seconds
stored in the declared second-resolution UTC timestamp columns. Explicit
physical-type overrides are available when a case tests binding mismatches.

Generate the request portion by constructing the real Python API object and
calling `to_dict()`. Copying an existing full case is also a reasonable start.
Never manufacture a second shorthand request language or use an engine run to
fill in the expected answer.

## Executable obligations

An obligation is one required check on one engine: a golden case, a generated
property on a seed batch, or a recorded gap. Every discovered case must occur in
[expectations.json](expectations.json), with one assertion
for each of `diskgraph`, the shipped engine, whose assertion checks the spec,
and `in_memory`, whose assertion is the differential cross-check:

```json
{
    "C-TOPIC-001": {
        "in_memory": "answer",
        "diskgraph": {
            "path": "sample.some_field",
            "message_contains": "specific unsupported condition"
        }
    }
}
```

`"answer"` calls real `check()`, requires acceptance, executes
`prepare()` → `materialize()` → `preview()`, and compares every named raw value
by sample ID plus all selected batch/state projections. Batch assertions execute
one batch containing the case's anchors in declared order. Missing/duplicate
samples or wrong named values fail.

A refusal object calls real `check()`, requires exactly the specified path/reason
and a useful suggestion, then requires real `prepare()` to raise the same
`ValidationError`. Engine opening is trapped. In-memory row access raises;
Parquet files deliberately do not exist, so attempts to read them cannot serve
as passing refusals. An unexpected additional error also fails.

These expectations are ordinary test assertions. They do not generate runtime
checks, claim a complete support matrix, or choose an answer from engine output.
The catalog check fails if a JSON file is added without an obligation or an engine
decision is missing. The normal local command executes in-memory obligations;
CI explicitly executes all obligations. The complete command independently
enumerates the required case/engine test IDs and fails if any did not pass,
even if other tests passed. Skips, expected failures, unexpected passes, and
reported deselections also fail. Ordinary targeted `pytest` runs remain useful
for debugging, but are not complete-conformance evidence.

Declaration-invalid constructors and parsers are focused public-API tests, not
golden requests with an invented successful answer. `assert_execution_error`
also verifies data-dependent failures at the specific `bind`, `fit`, or `build`
stage. An earlier error cannot satisfy a later-stage assertion. Public check
refusals continue to require unreadable source rows, nonexistent Parquet paths,
an engine-open trap, and the requested path/reason plus a suggestion.

From `packages/experimental/kumo-pytorch`, with the kumo-engine environment
described in [installation](../docs/installation.md#run-the-tests-and-notebooks-from-a-checkout)
active:

```bash
python -m conformance
python -m conformance --engine all
```

## Differential testing

The in-memory engine runs the same golden cases as the DiskGraph engine and,
where they apply, the same generated properties, and the differential
properties compare the two engines' outputs case by case. Neither engine is the
reference: a discrepancy is triaged against the rule. If the DiskGraph engine
disagrees with the golden answer or the guarantee, it is a DiskGraph bug, which
is fixed or recorded as a known gap. If the DiskGraph engine follows the rule
and the in-memory engine does not, it is an in-memory bug, which is fixed in the
in-memory engine or recorded as a known in-memory divergence; it never changes a
guarantee or a golden answer. If the rule leaves the behavior open, the
discrepancy is allowed, and the comparison tolerates exactly that, with the
reason written beside the tolerance. If no rule covers the behavior, it is a
question for the spec, settled by a reviewed rule and golden case rather than by
copying either engine's output.

Known gaps (`registry.KNOWN_GAPS`, repros in `known_gaps/`) list the DiskGraph
engine's shortfalls against the spec. Known in-memory divergences
(`registry.IN_MEMORY_DIVERGENCES`, repros in `in_memory_divergences/`) list the
in-memory engine's bugs; they are not spec gaps. The allowed discrepancies are
the tolerances in this table, each named in the comparison code:

| Tolerance | Where | Room left by |
|---|---|---|
| Mean, variance and standard deviation (`SUMMATION_DEPENDENT`) compare within `MOMENT_TOLERANCE`, relative and absolute 1e-12; the engines sum in different orders, and the largest measured difference is 4 ULP (relative 5e-16) | `properties.py` | G-CALC-001: floating means and moments are approximate |
| Fitted-state floats compare within the same `MOMENT_TOLERANCE` (`state_differences` in `fitted_state`) | `properties.py` | G-CALC-001: the fitted means and moments are approximate floating results |
| A sample with a tie at a fanout limit compares the clocks of the root's first-layer neighbors, not which tied neighbors are kept (`_same_up_to_ties`) | `graph_cases.py` | G-GRAPH-001 rule 2: which tied neighbor is kept is unspecified |

The in-memory engine builds no index, reads no Parquet binding and has no model
output format (`registry.INAPPLICABLE`), so the properties about those run on
the DiskGraph engine alone.

## Generated cases and properties

Golden cases fix answers by hand; generated cases check properties that need no
expected answer.

| File | Content |
|---|---|
| `registry.py` | Typed, versioned dimensions (source files, time, values, schema shape, contract, execution), what each contract feature or reduction is sensitive to, `KNOWN_GAPS`, `IN_MEMORY_DIVERGENCES` and `ENGINE_REFUSALS` |
| `tests/conformance/test_dtype_matrix.py` | All 17 scalar types × five encoders: real binding/execution for accepted pairs and pre-open refusals for incompatible declarations, on each selected engine. This fixed matrix is separate from the generated row/time dimensions. |
| `generate.py` | Seeded cases built with the real API: a pairwise covering array per seed batch over the sensitive dimensions; every case declares every reduction and window; XS rows in memory, S rows on DiskGraph; valid and deliberately invalid contract strategies |
| `properties.py` | Point-in-time invariance, refusal timing, differential (DiskGraph vs the in-memory engine), permutation, partition into Parquet files, time shift, irrelevance, delivery (sharding, resume, determinism, shuffling), epoch work trend, truncated Parquet |
| `repro.py` | Repro files and the corpus loader |
| `outputs.py` | Readers for targets and batches, and the variants the functional-area properties build: planted label-window rows, packing and fitting contracts, split anchors, `available_from` |
| `variants.py` | Refusal-timing strategies that also add a table, change the binding (Parquet types, renames, paths) or need batches to run |
| `transfer.py` | The fresh process that reuses a bound contract from JSON |
| `fake_ingest.py` | A recorder in place of `diskgraph.ingest`, and what this package's ingest config and relation mapping must say |

Each property runs through the real `check()` → `prepare()` → `materialize()`
path and compares samples as sets keyed by `sample_id`, with the same exact
comparison as golden cases, except for the allowed discrepancies listed under
[Differential testing](#differential-testing).

DiskGraph cases also vary `execution.num_shards` {1, 3} and
`execution.num_threads` {1, 4}. Generators assign the combinations by case
position, aggregate, graph and sequence cases alike, so every combination occurs
in each seed batch; a per-batch property checks the batch's first four cases,
one per combination. Every DiskGraph obligation therefore runs each of them, at
the case's own settings (`properties.obligation_cases`).

`tests/conformance/test_fake_ingest.py` checks this package's side of ingest on the CPU
job, with no DiskGraph wheel: the config compiled for generated and hand-written
schemas (declared columns in order, primary key, `create_time`, one fkey per FK
column with each direction timed by its destination's clock, each column's
type whatever unit Parquet reads timestamps back in (a type outside the table is
a violation), `num_shards`, `dataset_name`, `edge_format`, threads and the fixed
ingest options), the relation mapping in both orientations, and the index lifecycle
(reuse; refusal when the dataset's files or the schema changed, when Parquet
changed during ingest, when the index appeared during ingest, or on an
incompatible manifest; nothing published or staged after a failure; an
incompatible index never overwritten). With the wheel installed, a test also
compares the recorder's signature and output files with the installed
`diskgraph.ingest`. An index opened again with another `num_shards` is reused
silently today, though graph ties differ by shard count: the `index_layout`
property checks it, and GAP-INDEX-LAYOUT-DG records it.

Point-in-time invariance covers the destination table and a static
intermediate table. No passing obligation checks a cutoff at a timed
intermediate yet: DiskGraph refuses it at `check()`, which the refusal-timing
property keeps checking, and the in-memory engine leaks rows there, the known
in-memory divergence `DIV-TIMED-HOP-MEM`.

The in-memory engine's window boundaries are checked by golden cases and by the
DiskGraph differential, so the CPU-only job alone can't see a boundary
regression in the in-memory engine.

Functional areas widen the same cases:

- **Shapes:** an FK→PK hop into a static parent (`depth1_parent`), two hops
  through a link table that reaches some events twice, so `Rows.distinct`
  changes answers (`depth2_sibling`), and three hops out, back and out again
  (`depth3_static`). Every case also declares a lag window (`stop < 0`); a
  `contract.group_by` value adds aggregates over a two-key `GroupBy`.
- **Targets:** every case declares count and sum targets over `(0, W]` and
  `(a, b]` with `a > 0`, and the first and last future row, with ties. Answers
  carry them (a row target's time relative to the cutoff); point-in-time
  invariance compares entity values only. `label_window` plants rows on both
  sides of every target edge and checks each target moves exactly as the
  window rule says (start exclusive, stop inclusive); for rows tied for first
  or last, their time is judged, not which row is chosen.
- **Model path:** `execution.output_format=model` makes the differential and
  point-in-time checks read DiskGraph's `batches(output_format="model")`,
  decoded to the same answers, against the in-memory engine's inspectable batches.
  `model_delivery` and `model_epoch_trend` rerun delivery and the epoch work
  trend on model batches. `packing` checks token budgets, bucketing, Padded
  against Packed, model against inspectable, over histories that are empty,
  one or two rows long or at the cap, and `drop_last` with several
  `chunk_batches`; under a token budget, `drop_last` may drop only the final
  batch (`execution.drop_last`).
- **Anchors, roots, fitted state:** `anchor_splits` (split names, unions and
  the null split, in table order; DiskGraph's shuffle equals the in-memory engine's),
  `root_availability` (an anchor at `available_from` is refused, a second later
  accepted; a missing root key is a clean error), `fitted_state` (the same fit
  inputs and fitted structure as the in-memory engine: vocabulary entries, not their
  IDs, and types exactly) and `contract_transfer` (a bound contract gives
  identical batches in a fresh process). Both use fitted encoders over entity
  columns and field tokens (`Fields` with `Tokens` rows), which DiskGraph fits
  straight from typed columns (`build_model`) and the in-memory engine from records.
- **Refusal timing:** strategies DiskGraph refuses at `check()` by design, each
  refusal naming its construct at a path as specific as
  `properties.REFUSAL_CONSTRUCTS` records; the late refusals recorded as known
  gaps (accepted at `check()`, refused at `prepare()`); and `batches()`
  arguments `check()` never sees (`max_tokens` with `world_size > 1`, model
  output of a `Raw` string column), which `registry.INAPPLICABLE` declares refused
  at `batches()`, before any batch, with a clear message.

Compared outputs are answers, never encoded values.

Every (property × engine × seed batch), the corpus and each known gap and
in-memory divergence is a required obligation, like a golden case/engine pair: `--conformance-complete`
fails if one is missing, skipped, filtered or expected to fail. The property and
seed set is pinned in `tests/conformance/conftest.py`. The CPU job runs the
in-memory obligations; the DiskGraph job runs both engines. A conformance test
that needs one engine is marked `@pytest.mark.engine("diskgraph")`: it isn't
created unless that engine is selected, and then it calls `require_engine`, so a
missing native package fails it instead of skipping it.

Longer runs check the same properties on more seed batches:
`python -m conformance --engine all --seeds 12 --seed-start K` runs batches
K .. K+11 on each engine, on top of the pinned ones, which always run (415 checks
in about 12 minutes on an Apple M-series laptop). Every batch it runs is
required, so `--seeds` can only add. The nightly workflow picks a new window
each day (K = days since 1970 × 12) and logs K in its step summary, so a failure
reproduces with the same command. Rotating these seeds only
changes which generated cases are checked; the e2e flows' model-training seeds
stay fixed, since varying those would add noise rather than coverage.

A failing case writes `repro-<hash>.json` (rows, contract, anchors, seed,
property, engine, position, and the package, DiskGraph, registry and Python
versions) to `conformance-repros/`, or to `$CONFORMANCE_REPROS`. The
assertion message names each file. `properties.replay()` reruns one.

`corpus/` holds repro files that also carry an `"issue"` link. Every entry runs
on each selected engine its property applies to. It starts empty.

When a property exposes a DiskGraph defect, the gap is recorded in
`registry.KNOWN_GAPS` with its engine, property, the dimension values it covers,
a reason, a repro under `known_gaps/` and the failure it shows. An in-memory
defect is recorded the same way in `registry.IN_MEMORY_DIVERGENCES`, with its
repro under `in_memory_divergences/`. Generators stay
out of that region, and a required test replays each repro and expects that
failure, so a fix shows up as a failing test that lifts the gap. Never narrow a
generator silently. The same holds for in-memory divergences.

The assertions call the package's actual runtime checks. Expected answers and refusals
remain independent test assertions; they do not grant runtime support or
replace an engine's own checks.

## Graph neighborhoods and event sequences

`graph_cases.py` and `sequence_cases.py` generate raw-sampling cases with the
same `GeneratedCase`, repro and known-gap machinery, over their own dimensions
(`registry.SAMPLING_DIMENSIONS`). Their batches cover every value pair of
those dimensions.

| Property | Checks |
|---|---|
| `graph_differential` | DiskGraph's node sets and distinct edges equal the in-memory engine's; a sample with a tie at a fanout limit compares the clocks of the root's first-layer neighbors; both engines' graphs are layer-by-layer expansions (below) |
| `graph_inference_exact` | what batch inference needs (G-INFERENCE-001): rows dated after every cutoff change none of DiskGraph's graphs, tied neighbors included; where the index build isn't deterministic (several shards and several threads), samples with a tie at a fanout limit are left out, the known gap GAP-GRAPH-TIES-DG |
| `graph_point_in_time` | no neighbor dated after its cutoff, except events reached through a static traversal; future rows change nothing (tied samples as above), and the run with them is a layer-by-layer expansion over the changed tables |
| `graph_structure` | fanout bound, `recent` keeps the most recent eligible neighbors, unique nodes, edge orientation, layer-by-layer expansion; DiskGraph refuses static traversals at `check()` |
| `graph_edge_uniqueness` | no edge repeats within a sample |
| `graph_determinism` | sampling one prepared index twice more gives the same graphs in the same order; separate builds are rule 2, checked by `graph_tie_invariance` |
| `graph_tie_invariance` | row order, file splits and another `num_shards` don't change the graphs |
| `sequence_differential` | DiskGraph's sequences equal the in-memory engine's on both output formats |
| `sequence_point_in_time` | no event after its cutoff; each `EventTimeDelta` is the cutoff minus the event time, in its unit; future rows change nothing |
| `sequence_permutation` | a second run, and row and request order, give the same sequences |

[G-GRAPH-001](spec/guarantees/G-GRAPH-001/guarantee.md) states the graph rules.
A layer-by-layer expansion is rebuilt from the engine's own output, so it holds
whichever tied neighbor was kept: layer 0's frontier is the root, a frontier
node's kept neighbors are the traversal's edges at it, and the next frontier is
every kept neighbor, expanded again if reached before, as the in-memory engine does. At
each frontier node the kept neighbors must be candidates (keys join, dated at or
before the cutoff unless static) with the clocks of the `fanout` most recent candidates;
every node is the root or an edge's end, and every edge leaves the frontier. A
relation traversed more than once (`REVISITING` paths) can't say which layer made
an edge, so there the kept clocks need only include the most recent candidates',
and each edge must be a candidate of some traversal of it from the frontier.
The model output format is decoded from the same `Padded` layout as the
inspectable one, so both compare against the in-memory engine. Each property shares one
covering array; a known gap removes cases only from the property that found it,
which adds rows for any pairs those cases alone covered.
