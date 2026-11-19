# Difficult Behavior

A separate **Difficult Behavior** Actions workflow checks small, awkward feature
combinations against independent answers. It uses the same commit's DiskGraph
wheel, environment setup, refusal policy, required-case inventory and readable
reporting as the full suite. It is separate from the real-data binding matrix
and the `kumo-pytorch-gate`; it does not download datasets or require a GPU.

The workflow runs the same 82 cases on relevant PRs, manual dispatch and the
09:00 UTC schedule after merge. There are 60 package-interaction/regression
cases and 22 oracle mutation checks. All cases are required: missing cases, unexpected
additions, skips, xfails and unexecuted required bodies fail the job. A known
package defect stays a failing assertion, with its reproducer retained.

The [documented bug report](difficult-behavior-bugs.md) records seven confirmed
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
interfaces. Existing ordinary tests remain their owners; the new workflow adds
interaction coverage rather than running the package suite twice. The table
lists coverage owners, **not a promise that every feature is supported or every
combination is tested**. Paths below `tests/` refer to
`packages/experimental/kumo-pytorch/tests/`.

| Public feature family | Existing coverage owner | New workflow scope |
| --- | --- | --- |
| Schema, dtypes, temporal/non-temporal tables, relation keys, Parquet | `tests/test_entity_contract.py`, `tests/test_diskgraph_decimals.py`, G-TYPE-001 | Typed partitioned sources; exact wide integers, NULLs and timed joins |
| Native AnchorTable, splits, order, shuffle, rank/padding/drop-last | `tests/test_anchor_tables.py`, `tests/test_native_anchor_handoff.py` | Native table ownership, train/evaluation split, partial batches; distributed and shuffled cases stay with existing owners |
| Entity filters, availability, Rows/TimeWindow, relation traversal | G-ROWS-001, G-ENTITY-001, `tests/test_diskgraph_unsupported.py` | Future rows at two-hop boundaries and optional row filters |
| Aggregates, GroupBy/Reduce, Derived, Compare, CalendarPart, MapValues | G-CALC-001, `tests/test_graph_time_and_reductions.py` | Grouped/direct cancellation and dependency order; other calculation variants remain with existing owners |
| Dated occurrences, current events, reference clocks, OccurrencesOf | `tests/test_occurrence_projection.py`, G-HISTORY-001 | No additional cases in this increment; existing coverage retained |
| EventSource, merged histories, EventsOf, EventTimeDelta | `tests/test_multisource_history.py`, `model_checks/test_history_sweeps.py` | Empty/high-degree/tied merged histories, retained ages and labels |
| Neighborhoods, graph directions/layers, self-relations, sampling | `tests/test_graph_sweeps.py`, `tests/test_graph_self_relation.py`, `tests/graph_semantics.py` | Recent fanout, ties, forward/reverse/both directions, cyclic revisits and exact edge sets; shared semantic graph oracle |
| Raw, Vocabulary, ZScore, Buckets, Text, UseEncoding | `tests/test_encoding_runtime.py`, `tests/test_feature_encoding_sweeps.py`, G-ENCODING-001 | Wide/sparse IDs, shared mappings, frozen degenerate statistics, controlled BPE; bucket sweeps stay with existing owners |
| Fields, Graph, Columns, Tokens, Composite, field IDs and ID spaces | `tests/test_field_outputs.py`, `model_checks/test_category_forecast.py`, G-COMPOSITE-001, G-IDSPACE-001 | Columns/tokens/graph, explicit reserved roles and clipping; composite/field-name combinations stay with existing owners |
| Separate, Padded, Packed, Nested, Grouped, Flatten | `tests/test_field_outputs.py`, `tests/test_target_layouts.py` | Separate columns, padded/packed histories and flattened token supervision; nested layout coverage stays with existing owners |
| External/row/structured labels, thresholds, matching groups | `tests/test_target_layouts.py`, `model_checks/test_relational_history.py`, G-TARGET-001 | Withheld aligned labels and candidate labels; remaining target families stay with existing owners |
| MaskedValuesTarget, NextTokenTarget | `tests/test_target_layouts.py`, `model_checks/test_masked_history.py` | Clipped BPE, empty samples, markers, budgets, epoch reuse and interrupted iteration |
| Candidate sets/targets | `tests/test_candidates.py` | Empty/unknown/partially labeled candidates with native AnchorTables; currently a preflight refusal |
| Contract serialization and fitted/artifact persistence | `tests/test_contract_serialization.py`, `tests/test_inference_lifecycle.py` | Round-trip token contract, frozen numerical/category state and inference |
| Model training, optimizer/RNG checkpoints, finetuning, serving | `model_checks/test_checkpoint_resume.py`, `model_checks/test_finetuning.py`, `tests/test_inference_lifecycle.py` | Iterator resume only; existing model/serving acceptance remains separate |

## Run and inspect

From the repository root in the existing PyTorch test environment:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/difficult_behavior \
  --required-coverage=difficult -q --tb=long --junitxml=/tmp/difficult-behavior.xml
```

The Actions summary distinguishes executed passes, accepted refusals and
failures. `kumo-pytorch-diagnostics-difficult-behavior` holds full stdout/stderr,
tracebacks and JUnit evidence. `difficult-behavior-cases` holds the JUnit report
and generated `case.json` files with declarations, original source rows,
anchors and seeds. Reproduce a failure by checking out the run's commit and
running its exact pytest node ID. No downloaded dataset or Arena checkout is
needed. CPU setup/build time is additional to test time.

An accepted refusal must identify a declared unsupported capability before
execution. It proves rejection, not correctness of the unsupported operation.
The normal [refusal policy](required-coverage.md#capability-refusals) remains in
force. The fixture helper prevents a late refusal from masking an execution
failure. Tests contain the expected success assertions for future support.

## Local validation, 2026-11-18

The full 82-case run produced **11 failures and 71 passes** in 27.8 seconds.
Ten failing variants expose seven confirmed defect families; the remaining
failure is explicitly labeled a contract ambiguity. The passes comprise 65
executed cases and 6 accepted preflight refusals (candidate sets, timed
intermediate traversal and row filters). No skips or xfails. The original
58 cases remain green; the 24 regression/control cases add 11 failures and
13 passing controls. See the [documentation evidence](difficult-behavior-bugs.md).

This run used the existing compatible local DiskGraph extension. It is local
fixture evidence, not a branch-built Linux CI
result or a claim of complete public API conformance. The new workflow builds
DiskGraph from the branch before testing it.

Independent review completed in two rounds. It identified missing follower-mask
assertions and an unchecked epoch-two anchor population; both are now covered,
including isolated mask mutations and complete anchor multiplicity checks.
A separate fresh-context review reproduced all 11 regression failures, accepted
the seven documented defect families, and verified that A01 remains explicitly
ambiguous rather than counted as a confirmed bug.
The 397 harness checks, unchanged 671-case CPU inventory collection, scoped
Flake8/isort, YAML/actionlint, license headers and documentation checks pass.
The shared reporter's 23 checks cover readable expectations and observations,
unchanged outcomes, full evidence retention and cancellation. Independent review
confirmed the final summaries preserve dictionary differences and ambiguity.

The first bug hunt exercised matching groups with unknown categories,
two-stage history limits, follower targets, parallel graph foreign keys, and
future row targets with BPE text. A subsequent targeted hunt found the
[documented failures](difficult-behavior-bugs.md). The existing
`GAP-PARALLEL-FK-DG` was reproduced; its existing conformance
owner is retained rather than duplicated here.
