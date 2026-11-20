# Feature interactions

These 82 cases are ordinary tests in
`packages/experimental/kumo-pytorch/tests/conformance/`: 60 package-interaction
and regression cases and 22 oracle mutation checks. They run with the existing
package conformance suite, using native DiskGraph, independent expected answers
and the same refusal and coverage policies as the other cases. They need no
downloaded datasets or GPU.

`--conformance-complete` requires every pinned variant to pass or verify an
approved refusal. Missing cases, unreviewed additions in these modules, skips,
xfails and unexecuted requirements fail the job. A known package defect stays a
failing assertion. The existing package and nightly workflows execute these requirements.

The [documented bug report](documented-regressions.md) records seven confirmed
defect families and one explicit contract ambiguity. Each expected answer cites
the package documentation. These regressions remain ordinary required failures.

## New interaction cases

| Area | Difficult combination | Independent checks |
| --- | --- | --- |
| Temporal relations | Partitioned sources, a future intermediate or destination row, and two-hop count/sum | Eligibility at each timed hop; cutoff equality; no future poison; empty count/sum are zero |
| Calculations | Window boundaries, optional filters, grouped or direct sums, cancellation above `2**53`, reversed dependency declarations | Exact int64 sum and derived result; excluded lower boundary/future row; NULL handling |
| Frozen encoding | Constant and all-NULL training columns, sparse shared categorical IDs, wide integer IDs, empty/NULL text, distinct Unicode forms, future-only observations | Explicit IDs and masks; frozen numerical statistics; artifact reload and inference with fitting forbidden |
| Candidate supervision | Empty and ragged candidate lists, unknown/missing features and partial labels | Candidate identity/order, layout, feature values, label values and validity; padding is not negative supervision |
| Histories + graphs | Empty and high-degree entities, two tied event streams, identical values on distinct events, per-anchor cutoffs, clipped histories, recent graph fanout, withheld labels and shared age inputs | Eligible membership, chronology, multiplicity, values, masks, packed offsets, age/label alignment; graph identity, features and connectivity independent of tensor order |
| Native token delivery | Controlled BPE, empty/NULL/unknown text, oversized history clipping, BOS/EOS, padded/packed layouts, masked/next-token targets, length bucketing and an interrupted epoch | Hand-calculated IDs/labels; no cross-sample shift; padding-inclusive token budget; actual native iterator/length planning; zero recomputation and full cache hits in epoch two; exact remaining batches after iterator restore |
| Directed graphs | Forward, reverse and both traversals, one/three layers, both relation-key orientations, cycles, self-loops, dangling foreign keys and time boundaries | Exact source-derived node/edge sets; relation orientation; no fabricated reciprocal edges, duplicates or cross-sample edges |
| Oracle validation | Deliberately corrupt identity, time, label, age, each follower/identity mask or event count | Each corruption must be rejected, including when ties allow legitimate alternatives and masked payload storage varies |

Ties do not impose an arbitrary primary-key or component order. The history
oracle checks the required timestamp multiset and eligible identities at the
clipping boundary; the graph oracle checks eligible neighbors and semantic
connectivity. Duplicate-valued events with distinct identities remain distinct.
Only the token cases claim native length-cache instrumentation. Mixed graph and
history cases verify the public AnchorTable result and batching policy, without
claiming that every internal assembly step is native.

The graph API has no `undirected` flag. `GraphSample.direction` accepts
`forward` and `reverse`; selecting both expands incoming and outgoing neighbors.
Edges still point left-to-right according to `RelationSpec`. These tests do not
expect automatic reciprocal edges or conflate traversal with emitted orientation.

## Feature audit and ownership

Reviewed against the public schema, sample, batch, encoding, target and lifecycle
interfaces. Existing tests retain their assertions; these additions exercise feature interactions
within the same conformance suite. The table
lists coverage owners, **not a promise that every feature is supported or every
combination is tested**. Paths below `tests/` refer to
`packages/experimental/kumo-pytorch/tests/`.

| Public feature family | Existing coverage owner | Difficult Behavior scope |
| --- | --- | --- |
| Schema, dtypes, temporal/non-temporal tables, relation keys, Parquet | `tests/test_entity_contract.py`, `tests/test_diskgraph_decimals.py`, G-TYPE-001 | Typed partitioned sources; exact wide integers, NULLs and timed joins |
| Native AnchorTable, splits, order, shuffle, rank/padding/drop-last | `tests/test_anchor_tables.py`, `tests/test_native_anchor_handoff.py` | Native table ownership, train/evaluation split, partial batches; distributed and shuffled cases stay with existing owners |
| Entity filters, availability, Rows/TimeWindow, relation traversal | G-ROWS-001, G-ENTITY-001, `tests/test_diskgraph_unsupported.py` | Future rows at two-hop boundaries and optional row filters |
| Aggregates, GroupBy/Reduce, Derived, Compare, CalendarPart, MapValues | G-CALC-001, `tests/test_graph_time_and_reductions.py` | Grouped/direct cancellation and dependency order; other calculation variants remain with existing owners |
| Dated occurrences, current events, reference clocks, OccurrencesOf | `tests/test_occurrence_projection.py`, G-HISTORY-001 | No additional cases in this increment; existing coverage retained |
| EventSource, merged histories, EventsOf, EventTimeDelta | `tests/test_multisource_history.py` | Empty/high-degree/tied merged histories, retained ages and labels |
| Neighborhoods, graph directions/layers, self-relations, sampling | `tests/test_graph_sweeps.py`, `tests/test_graph_self_relation.py`, `tests/graph_semantics.py` | Recent fanout, ties, forward/reverse/both directions, cyclic revisits and exact edge sets; shared semantic graph oracle |
| Raw, Vocabulary, ZScore, Buckets, Text, UseEncoding | `tests/test_encoding_runtime.py`, `tests/test_feature_encoding_sweeps.py`, G-ENCODING-001 | Wide/sparse IDs, shared mappings, frozen degenerate statistics, controlled BPE; bucket sweeps stay with existing owners |
| Fields, Graph, Columns, Tokens, Composite, field IDs and ID spaces | `tests/test_field_outputs.py`, G-COMPOSITE-001, G-IDSPACE-001 | Columns/tokens/graph, explicit reserved roles and clipping; composite/field-name combinations stay with existing owners |
| Separate, Padded, Packed, Nested, Grouped, Flatten | `tests/test_field_outputs.py`, `tests/test_target_layouts.py` | Separate columns, padded/packed histories and flattened token supervision; nested layout coverage stays with existing owners |
| External/row/structured labels, thresholds, matching groups | `tests/test_target_layouts.py`, G-TARGET-001 | Withheld aligned labels and candidate labels; remaining target families stay with existing owners |
| MaskedValuesTarget, NextTokenTarget | `tests/test_target_layouts.py` | Clipped BPE, empty samples, markers, budgets, epoch reuse and interrupted iteration |
| Candidate sets/targets | `tests/test_candidates.py` | Empty/unknown/partially labeled candidates with native AnchorTables; currently a preflight refusal |
| Contract serialization and fitted/artifact persistence | `tests/test_contract_serialization.py`, `tests/test_inference_lifecycle.py` | Round-trip token contract, frozen numerical/category state and inference |
| Model training, optimizer/RNG checkpoints, finetuning, serving | `tests/test_inference_lifecycle.py` | Iterator resume only; existing model/serving acceptance remains separate |

## Run and inspect

From `packages/experimental/kumo-pytorch` in the existing PyTorch test environment:

```sh
python -m conformance --engine diskgraph
```

For a focused run, pass a module or exact node ID to pytest, for example:

```sh
python -m pytest tests/conformance/test_behavior_regressions.py -q --tb=long
```

The package CI artifact `kumo-pytorch-test-results` retains JUnit results and
generated `case.json` files containing declarations, source rows, anchors and
seeds. Test output distinguishes executed passes, accepted refusals and failures.
Reproduce a failure at the run's commit using its exact pytest node ID.

An accepted refusal must identify a declared unsupported capability before
execution. It proves rejection, not correctness of the unsupported operation.
The normal [refusal policy](README.md#executable-obligations) remains in
force. The fixture helper prevents a late refusal from masking an execution
failure. Tests contain the expected success assertions for future support.

## Regression expectations

The 24 regression variants include controls, seven documented defect families,
and one explicitly labeled contract ambiguity. Expected answers and source
citations are retained in [documented regressions](documented-regressions.md).
Known failures remain ordinary failed tests; reviewers must settle ambiguous
semantics before changing their expectations.
