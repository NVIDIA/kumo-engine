# Difficult Behavior: documented failures

These are normal regression tests in [test_regressions.py](../difficult_behavior/test_regressions.py).
They assert the documented successful behavior and currently fail against the
package. They do not assert that a bug must keep happening, use xfail, or change
package code. Passing neighboring cases establish that the fixture is usable.

Evidence below was read from the package documentation at Engine commit
`46211d923555f50afc7f002de9bb3ac89b6f8f4b`. Quoted rules are distinguished from
our application of those rules to the fixture. No expected output comes from
another engine. All loading uses Parquet and native AnchorTables.

The local regression run has **11 failures and 13 passes**: ten failing variants
across **seven confirmed defect families**, plus **one explicitly ambiguous
case**. Several layouts/values exposing one cause do not count as separate bugs.
The existing aliased-relation ingestion defect (`GAP-PARALLEL-FK-DG`) was also
reproduced, but its existing conformance ownership is retained.

| ID | Defect | Documented expected behavior | Observed failure |
| --- | --- | --- | --- |
| DB01 | Artifact reload changes task identity | Identical anchors/fitted state preserve fingerprints and allow iterator continuation | `this state was saved for a different task` |
| DB02 | Next-token budgeting uses the pre-shift length | Five emitted input slots fit a budget of five | `one sample exceeds Batching.max_tokens` |
| DB03 | Numerical fitting loses valid finite statistics | Nonconstant tiny inputs normalize; a constant finite population uses scale one | Tiny inputs use scale one; large constants crash on `None ** 2` |
| DB04 | Masking casts unsigned entity identities to signed integers | Valid uint64 keys remain usable with discrete masking | `Overflow when unpacking long long` |
| DB05 | Long event ages fail with a nanosecond graph clock | A 105-day interval yields 9,072,000 seconds | Arrow rejects casting elapsed nanoseconds to float64 |
| DB06 | Later graph layers fail to expand revisited nodes | The returned root is expanded using the third layer's larger fanout | Two eligible event nodes and their edges are absent |
| DB07 | Native length planning cannot handle typed NULL text constants | `Constant(None, string)` with `missing="emit"` contributes a missing token | `post-fit text token lengths missing for events.text` |
| A01 | Ambiguous scope of duplicate sampled-value restriction | Proposed: distinct named fields may read the same physical column; alternative: reject the alias before execution | Late token-metadata error; failure explicitly says **CONTRACT AMBIGUITY A01** |

## DB01 — Artifact identity and iterator continuation

[G-ARTIFACT-001, Rule](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-ARTIFACT-001/guarantee.md#rule):

> The task's fingerprints equal the writing task's.

That paragraph explicitly covers rematerializing the same anchors with
`fitted=artifact.fitted_state`. [TaskBatches](../../../packages/experimental/kumo-pytorch/docs/api/lifecycle.md#taskbatches)
also says a fresh iterator with the same configuration continues after the saved
position. The fixture has four anchors, values `[0,1,2,3]`, and batches of two.
After saving the first batch, the remaining IDs/values must be `[2,3]`.

The same-task control passes. Artifact reload over the same plan/index and
original AnchorTable fails to restore. Scalar materialization puts a newly
created temporary anchor path into the fingerprint; no user data changed.
Test: `test_artifact_reload_preserves_iterator_identity`.

## DB02 — Next-token budget after shifting

[Batching.max_tokens](../../../packages/experimental/kumo-pytorch/docs/api/batches.md#batching):

> budget of materialized slots of `length_input`, padding included.

[NextTokenTarget](../../../packages/experimental/kumo-pytorch/docs/api/targets.md#nexttokentarget)
and [G-TARGET-001](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-TARGET-001/guarantee.md#rule)
define shifting within a sample. The controlled six-ID sequence is
`[4,13,16,13,2,5]`. Next-token input is its first five IDs, with the final five
as labels. Thus the single example has five materialized input slots in both
Packed and Padded layouts. A budget of five must accommodate it.

Both layouts fail at five; the budget-six controls emit exactly five slots.
The native planner counts the sequence before the next-token reduction.
Test: `test_next_token_budget_counts_emitted_input_slots`.

## DB03 — Population normalization on finite binary64 inputs

[G-ENCODING-001, Boundaries](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-ENCODING-001/guarantee.md#boundaries):

> ZScore uses population mean/scale, ignores NULL while fitting

The same paragraph specifies scale one for constant populations. The
[ZScore API](../../../packages/experimental/kumo-pytorch/docs/api/encodings.md#zscore)
says it rejects nonfinite observations and uses a binary64 working domain.
All inputs here are finite, exactly represented powers of two.

For `[0, 2**-599]`, the population mean and standard deviation are `2**-600`;
the expected normalized values are exactly `[-1,1]`. The implementation
underflows its intermediate variance and uses scale one, emitting
`[-2**-600, 2**-600]`. The neighboring `[0, 2**-499]` case passes. No tolerance
is fitted to the observed result.

For `[2**600, 2**600]`, the population is constant, so the documented scale is
one and both outputs are zero. Fitting instead crashes when it squares a
missing standard deviation. This constant-population crash independently
establishes the defect even if reviewers want a more explicit stability bound
for the documented approximate numerical fitting. Both variants are counted as
one numerical-statistics family.
Test: `test_numerical_fitting_preserves_representable_normalization`.

## DB04 — uint64 identities with masking

[DType](../../../packages/experimental/kumo-pytorch/docs/api/schema.md#dtype)
lists `uint64` as supported. [TableSpec](../../../packages/experimental/kumo-pytorch/docs/api/schema.md#tablespec)
requires primary keys to be non-null, unique and declared in the schema; it does
not narrow them to signed integers. [G-TARGET-001](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-TARGET-001/guarantee.md#rule)
specifies that masking records original final IDs and replaces the selected
input using the declared reserved role.

The fixture has one uint64 entity key and one supplied categorical token.
[G-IDSPACE-001](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-IDSPACE-001/guarantee.md#boundaries)
allocates pad/unknown/missing/mask as 0/1/2/3; the ordinary value is 4.
Probability-one masking must emit input `[[3]]`, label `[[4]]`, valid and
corruption masks `[[true]]`, regardless of whether the entity key's high bit
is set. The helper writes the anchor key explicitly as Arrow uint64.

`2**63-1` passes; `2**63` and `2**64-1` fail when masking creates an implicit
signed torch tensor from entity identities.
Test: `test_masking_supports_declared_uint64_entity_keys`.

## DB05 — Event ages at nanosecond resolution

[EventTimeDelta](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#eventtimedelta):

> The value is `(anchor time - event time)` in `unit`

[DType](../../../packages/experimental/kumo-pytorch/docs/api/schema.md#dtype)
explicitly supports nanosecond timestamps, with integer field values in declared
ticks and integer anchor values in epoch seconds. The fixture uses an anchor at
20,000,000 seconds and an event exactly 105 days earlier. With requested unit
`seconds`, the answer is `105 * 86400 = 9,072,000`, exactly representable in float64.

Microsecond controls and the 104-day nanosecond control pass. At 105 days,
casting the intermediate `9,072,000,000,000,000` nanoseconds exceeds Arrow's
safe integer-to-float range before conversion to seconds. This is an execution
error, not a declared preflight refusal.
Test: `test_event_age_survives_long_history_at_fine_clock_resolution`.

## DB06 — Expanding a revisited graph node

[G-GRAPH-001, Rule](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-GRAPH-001/guarantee.md#rule):

> Each layer samples at most `fanout` neighbors per frontier node for each `GraphSample`.

The same rule specifies that `recent` keeps the most recent candidates and that
output nodes/edges are unique per sample. [NeighborhoodSpec](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#neighborhoodspec)
says layers expand in order from the entity. These are output uniqueness rules,
not permission to skip a later layer's declared expansion.

One user owns events at times 10, 20 and 30; cutoff is 100. Layer one takes its
latest event (fanout 1); layer two follows the reverse edge back to the user;
layer three takes that user's latest three events. Required nodes are `{1,2,3}`
and all three user-to-event edges, each emitted once. There are no ties.

Only event 3 is returned. The native frontier adds newly discovered nodes but
omits a previously visited node needed by the next layer. A final fanout of one
passes, which explains why unchanged-fanout revisit cases missed this behavior.
Test: `test_later_graph_layer_reexpands_visited_nodes`.

## DB07 — Typed NULL text constants under length planning

[Constant](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#constant):

> `Constant(value=None, dtype=...)` is a typed NULL.

[EventSpec](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#eventspec)
allows constant fields; [Tokens.missing](../../../packages/experimental/kumo-pytorch/docs/api/batches.md#tokens)
supports emitting NULL values. [Text encoding](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-ENCODING-001/guarantee.md#boundaries),
[G-IDSPACE-001](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-IDSPACE-001/guarantee.md#boundaries)
and [G-TARGET-001](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-TARGET-001/guarantee.md#rule)
define the missing ID, reserved markers and masked answers.

Three events with this constant produce `[BOS,missing,missing,missing,EOS]`.
After masking, input is `[4,3,3,3,5]`, labels are `[-100,2,2,2,-100]`, and only
the three value positions are supervised. Five slots fit the budget of twelve.
The native planner instead seeks stored text-length metadata for the constant.
The same NULL case without budgeting and non-NULL `"red"` controls pass.
Test: `test_native_length_planning_handles_typed_null_text_constants`.

## Ambiguous A01

[G-FIELDS-001, Rule](../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-FIELDS-001/guarantee.md#rule):

> One input cannot encode the same resolved sampled value twice, including overlapping qualified/unqualified aliases.

But [EventSpec.fields](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#eventspec)
defines named values per event, and [Field](../../../packages/experimental/kumo-pytorch/docs/api/samples.md#field)
reads a source column. It is unclear whether two distinct sampled names, both
reading `Field("text")`, count as the same resolved value for this restriction.

- **Proposed interpretation in this test:** the names are distinct sampled
  fields. Each emits the source's BPE pieces, in declaration order. For
  `"red bike"`, shared encoding and the fixture's ID space give
  `[4,13,16,13,16,5]` including BOS/EOS.
- **Alternative:** physical source aliases are also forbidden. Then declaration
  or preflight should explicitly reject them; accepting the contract and later
  failing on token metadata is not an adequate validation rule.

The single-field control passes. Two fields pass preflight/materialization and
fail in native length planning. Per the user's instruction, this case remains
an ordinary required failure, with **CONTRACT AMBIGUITY A01** and both readings
in its error message and full artifact traceback. It is not counted among the
seven confirmed defect families. A reviewer must settle the alias scope before
changing the expectation.
Test: `test_ambiguous_two_named_fields_reading_one_text_column`.

## Reproduce and review

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/difficult_behavior/test_regressions.py \
  -q --tb=long --junitxml=/tmp/difficult-regressions.xml
```

Use an individual node ID to isolate a case. The required workflow runs these
alongside the original interactions and oracle mutation checks, preserving all
failures rather than stopping after the first. `case.json` artifacts retain
source rows, anchors and the complete contract. Local runs use the existing
compatible native extension; the new branch-built Linux workflow remains to be
run after these changes are committed and pushed.
