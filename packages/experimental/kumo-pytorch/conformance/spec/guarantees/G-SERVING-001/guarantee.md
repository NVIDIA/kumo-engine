# G-SERVING-001: serving assembles caller-supplied rows into the inference batch

## Rule

1. Assembly. `ServingAssembler(artifact)`, built once per artifact, and its
   `assemble(rows, requests)` build the inference-mode batches the contract
   defines (G-INFERENCE-001) for `requests` over exactly `rows`, which the
   caller supplies. Nothing is fetched: the supplied rows are the whole dataset
   for that call, and they are read in process. Nothing is fitted: the fitted
   state is the artifact's.
2. Sufficiency. `rows_spec`, a `ServingRowsSpec` derived from the contract
   alone, lists the rows to fetch for each request. If the supplied rows include
   every row it lists for each request, and those rows are the same as in the
   offline dataset, the served batches equal the inference batches that the
   engine's batch inference over the full dataset produces for the same
   requests, batch size and seed. They may differ only where the rules leave
   room: which of several tied neighbors a fanout keeps (G-GRAPH-001, rule 2)
   and the rounding of floating means and moments, which are approximate
   (G-CALC-001). Rows `rows_spec` does not list may be left out.
   Whether the caller's lookup returns the same rows as the offline data (fresh,
   none missing) is the caller's responsibility: a row the samples read and the
   caller leaves out is absent, as it would be from a dataset, so the batch
   differs and nothing detects it.
3. Refusals. Refused, naming the table, row and column: a table the schema does
   not declare; a required table or column left out; a column the table does not
   declare; a value that is not of its column's declared type (checked as for
   `InMemoryRows`); a NULL or repeated declared primary key within a keyed
   table; a request whose entity, or a candidate, has no row; and an empty
   request list.

## Boundaries

A table or column that inference never reads may be left out; it reads as an
empty table or as `None`, and the batches are unchanged. Every other declared
table must be named (an empty list when a request has no rows in it), and every
supplied row must carry its table's other columns, declared primary key, join keys and
time columns included, with `None` for a missing value.
Keyless tables require no PK column and retain separate identical rows under
[G-KEYLESS-001](../G-KEYLESS-001/guarantee.md).

For keyless tables, serving preserves occurrence multiplicity and selection
rules, but exact offline/serving equality is not promised for results depending
on local identity or source order. This includes opaque graph IDs, equal-time
ordering and limits, uniform graph draws, and floating reduction rounding.
Compare graph features and connectivity under a renaming of opaque IDs.
A fetched subset or reordered rows form a different prepared dataset; declare
a PK when stable cross-preparation ordering and choices are required. The
exact-batch guarantee for existing keyed contracts is unchanged.

Types are never inferred from the supplied rows: the artifact's contract pins
every column type, so a column that is `None` throughout a request neither
changes the batch's types nor is refused. Rows may be row mappings or an Arrow
table; both give the same batches, and an Arrow column must have the declared
type (dictionary-encoded or not, or one the row path accepts for it, or no type
when it is all NULL).

Serving emits the inspectable output format. One call with every request gives
one batch unless a token budget in the contract splits it.

## Scope

The golden case below fixes the raw input values for requests whose rows are
supplied one user at a time, checked by the conformance harness over the whole
table on the DiskGraph engine. Serving needs encoded batches, so it is covered
by tests in
[test_inference_lifecycle.py](../../../../tests/test_inference_lifecycle.py)
and [test_serving.py](../../../../tests/test_serving.py), each comparing served
batches value for value with batch inference over the full dataset read in
process: the golden case served from each user's own rows; fetching exactly
what `rows_spec` lists, without the optional tables and columns, for the
example, a bounded-window, an event-limit, a columns/events/graph and generated
multi-hop graph contracts, and, in
[test_serving_rows_drift.py](../../../../tests/test_serving_rows_drift.py), for
every golden case and generated tabular, graph and sequence case, where dropping
a listed row changes some batch in every family; serving one request at a time,
from that user's rows only, for a columns/events/graph contract and a key/value
token contract; Arrow input; a column that is `None` throughout; the plan built
once per `ServingAssembler`; and each refusal above. For a model trained on the
DiskGraph engine, served batches have the same layout as its model-format
inference batches. No test compares served values directly with the DiskGraph
engine's batch inference; those values are covered only through the harness's
differential comparison of batch inference read in process with the DiskGraph
engine's.

Deployment, request routing and fetching rows online are outside this package.

## Golden examples

- [C-SERVING-001](golden/C-SERVING-001.json): user 1's rows are events at 1, 4
  and 9 and user 2's are events at 2 and 12, so the counts are 2 for user 1 at
  5, 3 for user 1 at 10 and 1 for user 2 at 10, whether each user's rows are
  supplied alone or the whole table is.
