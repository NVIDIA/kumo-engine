# G-SERVING-001: serving uses DiskGraph inference batches

## Rule

1. Bind the artifact and an open DiskGraph plan for its contract to a
   `ServingAssembler`. The caller maintains the indexed data and owns the plan.
2. `assemble` takes Parquet anchors and returns native model-format inference
   batches. With the same artifact, index, anchors, batch size and seed as
   offline DiskGraph inference, inputs, targets and metadata must agree.
   Serving uses the saved encoders and emits no training targets.
3. A row-only request or a plan that cannot deliver native batches is refused
   before the first batch. Serving never falls back to Python request batching.

## Boundaries

`rows_spec` describes the contract's table and column dependencies and can be
inspected from an artifact alone. Serving requires a prepared plan and anchor
table. Keyless tables use the identity and ordering of the prepared index
under [G-KEYLESS-001](../G-KEYLESS-001/guarantee.md).

## Scope

The [serving tests](../../../../tests/test_serving.py) compare serving with
offline inference for time windows and event limits and exercise refusal paths.
Deployment, request routing and model inference are outside this guarantee.

## Golden examples

- [C-SERVING-001](golden/C-SERVING-001.json) counts two events for user 1 at
  time 5, three at time 10, and one for user 2 at time 10.
