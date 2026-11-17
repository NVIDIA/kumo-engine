# Arena golden coverage audit

Retain Engine's tests and port only additional assertions. No Arena checkout,
runner, catalog or dataset is required at runtime. New checks are DiskGraph-only; pre-existing Engine tests are unchanged.
Generic behavior belongs in
the package's repository tests; AML integration stays in `test/kumo_pytorch`.

Audit sources: Engine `23e49c3c70a73ce1b9a95488a5cf3e7cc733107e` and
[Arena `220aa187`](https://github.com/kumo-ai/kumo-engine-arena/tree/220aa187921ddb1d846985bec8f8edff1d3581c9/tests/golden).
Arena's catalog contains 173 families and 1,863 variants per engine. These are
declarations, not passing results. Its 799-variant G-EXEC-014 expands other cases
through the model-output path; it is not 799 unrelated behaviors.

## Family ownership map

Every Arena family is assigned below to existing Engine tests to compare. This
is a family-level map, **not a claim that every Arena variant is already covered**.
The batching table below records the assertions actually compared in detail.
Other groups still need variant-level comparison before further ports.
All family ranges below are inclusive.

Package file links are under `packages/experimental/kumo-pytorch/tests`.

| Arena families | Existing Engine assertion owners | Comparison still needed |
| --- | --- | --- |
| G-SCHEMA-001–007 | [test_entity_identity.py](../../../packages/experimental/kumo-pytorch/tests/test_entity_identity.py), [test_type_resolution.py](../../../packages/experimental/kumo-pytorch/tests/test_type_resolution.py), [test_time_columns.py](../../../packages/experimental/kumo-pytorch/tests/test_time_columns.py) | Physical PK failures and availability equality versus constructor-only rejection. |
| G-ROWS-001–012 | [test_temporal_declarations.py](../../../packages/experimental/kumo-pytorch/tests/test_temporal_declarations.py), [test_time_boundaries.py](../../../packages/experimental/kumo-pytorch/tests/test_time_boundaries.py), [test_multisource_history.py](../../../packages/experimental/kumo-pytorch/tests/test_multisource_history.py) | Each filter/null operand, join multiplicity and active membership boundary. G-ROWS-012 has unresolved Arena policy. |
| G-VALUE-001–017 | [test_graph_time_and_reductions.py](../../../packages/experimental/kumo-pytorch/tests/test_graph_time_and_reductions.py), [test_core_api.py](../../../packages/experimental/kumo-pytorch/tests/test_core_api.py), [test_typed_sample_values.py](../../../packages/experimental/kumo-pytorch/tests/test_typed_sample_values.py) | Every reduction on empty/all-null/singleton populations, joined-field selection and arithmetic domains. |
| G-EVENT-001–009 | [test_event_output_order.py](../../../packages/experimental/kumo-pytorch/tests/test_event_output_order.py), [test_history_cap_pushdown.py](../../../packages/experimental/kumo-pytorch/tests/test_history_cap_pushdown.py), [test_time_delta.py](../../../packages/experimental/kumo-pytorch/tests/test_time_delta.py) | Per-component versus global limits, repeated dated values and first/last future targets. |
| G-CAND-001–004 | [test_candidates.py](../../../packages/experimental/kumo-pytorch/tests/test_candidates.py), [test_anchor_tables.py](../../../packages/experimental/kumo-pytorch/tests/test_anchor_tables.py) | Candidate-rooted values, empty/omitted groups and repeated IDs across groups. |
| G-INPUT-001–022 | [test_input_design.py](../../../packages/experimental/kumo-pytorch/tests/test_input_design.py), [test_field_outputs.py](../../../packages/experimental/kumo-pytorch/tests/test_field_outputs.py), [test_varlen_metadata.py](../../../packages/experimental/kumo-pytorch/tests/test_varlen_metadata.py) | Exact values/masks/offsets for each layout and empty-history form; 29/30/31 truncation grid. |
| G-GRAPH-001–014 | [test_graph_sweeps.py](../../../packages/experimental/kumo-pytorch/tests/test_graph_sweeps.py), [test_graph_edge_uniqueness.py](../../../packages/experimental/kumo-pytorch/tests/test_graph_edge_uniqueness.py), [test_time_boundaries.py](../../../packages/experimental/kumo-pytorch/tests/test_time_boundaries.py), [conformance properties](../../../packages/experimental/kumo-pytorch/conformance/properties.py) | Ownership under rebatching, fanout equality and graph clocks. G-GRAPH-013 has an unresolved Arena oracle. |
| G-ENC-001–010 | [test_field_outputs.py](../../../packages/experimental/kumo-pytorch/tests/test_field_outputs.py), [test_feature_encoding_sweeps.py](../../../packages/experimental/kumo-pytorch/tests/test_feature_encoding_sweeps.py), [test_bound_contract.py](../../../packages/experimental/kumo-pytorch/tests/test_bound_contract.py) | Reserved-ID gaps, quantile equality/duplicate cuts, pooled versus independent fitting. |
| G-TOKEN-001–022 | [test_token_field_order.py](../../../packages/experimental/kumo-pytorch/tests/test_token_field_order.py), [test_tokenization.py](../../../packages/experimental/kumo-pytorch/tests/test_tokenization.py), [test_special_token_ids.py](../../../packages/experimental/kumo-pytorch/tests/test_special_token_ids.py), [test_text_from_index.py](../../../packages/experimental/kumo-pytorch/tests/test_text_from_index.py) | Whole-event truncation, timezone independence, BPE tie/frequency boundaries and composite missing/literal collisions. |
| G-TARGET-001–017 | [test_target_layouts.py](../../../packages/experimental/kumo-pytorch/tests/test_target_layouts.py), [test_masking.py](../../../packages/experimental/kumo-pytorch/tests/test_masking.py), [test_batch_api.py](../../../packages/experimental/kumo-pytorch/tests/test_batch_api.py) | Probability endpoints/allowlists, matching-group identity under rebatching and fitted thresholds. |
| G-CONTRACT-001–008 | [test_contract_serialization.py](../../../packages/experimental/kumo-pytorch/tests/test_contract_serialization.py), [test_public_contract.py](../../../packages/experimental/kumo-pytorch/tests/test_public_contract.py), [test_build.py](../../../packages/experimental/kumo-pytorch/tests/test_build.py) | Current typed-API compatibility matrix versus Arena's older representation names; invalid-version diagnostics. |
| G-BATCH-001–015 | [test_core_api.py](../../../packages/experimental/kumo-pytorch/tests/test_core_api.py), [test_native_length_planning.py](../../../packages/experimental/kumo-pytorch/tests/test_native_length_planning.py), [new boundary tests](../../../packages/experimental/kumo-pytorch/tests/conformance/test_batch_boundaries.py) | Detailed comparison below. |
| G-EXEC-001–014 | [test_anchor_tables.py](../../../packages/experimental/kumo-pytorch/tests/test_anchor_tables.py), [test_fitted_state_persistence.py](../../../packages/experimental/kumo-pytorch/tests/test_fitted_state_persistence.py), [test_inference_lifecycle.py](../../../packages/experimental/kumo-pytorch/tests/test_inference_lifecycle.py), [test_native_anchor_handoff.py](../../../packages/experimental/kumo-pytorch/tests/test_native_anchor_handoff.py) | Resume before/within/after a pass, empty ranks and exact model-output variant coverage. |
| G-HARNESS-001–002 | [test_assertions.py](../../../packages/experimental/kumo-pytorch/tests/conformance/test_assertions.py), [test_required_execution.py](../../../packages/experimental/kumo-pytorch/tests/conformance/test_required_execution.py), [standalone required coverage](required-coverage.md) | Existing deliberate wrong-answer and accounting checks should be extended only for new missing failure modes. Do not copy Arena's harness. |

## Batching comparison and first ports

All entries below refer to assertions, not matching test names. “Overlap” means
the existing witness is retained; it does not establish every engine/layout
combination. The new module has 29 DiskGraph checks. `--engine in_memory` creates no
new boundary obligations; `--engine diskgraph` and `--engine all` require them.

| Arena family / variants | Existing assertion and decision |
| --- | --- |
| G-BATCH-001: 0/1/15/16/17/32/33 samples | Native handoff covers partial batches; the full count grid was absent. Added `test_sample_count_boundaries`, `keep` variants, including empty requests and exact identities/values/labels. |
| G-BATCH-002: same grid, `drop_last=True` | Core tests cover dropping a final partial batch, not the full grid. Added the `drop` variants. |
| G-BATCH-003: size/drop/keep/default overrides | Overlap: core `test_drop_last_without_a_token_budget_drops_the_final_short_batch` and `test_drop_last_under_a_token_budget_keeps_a_full_final_batch` assert overrides. Full engine/variant audit remains open. |
| G-BATCH-004: mixed and empty histories | Overlap: `test_model_column_hotpath.py::test_padding_preserves_presence_and_empty_histories` and AML empty-history cases. Physically empty source versus nonempty ineligible source still needs a separate comparison. |
| G-BATCH-005/006: padded/packed budget equality | Overlap: core `test_length_bucketing_and_token_budget_control_batch_composition` and native length-planning checks. The exact pair of layout witnesses needs comparison; no duplicate added here. |
| G-BATCH-007: sample cap × token cap | Overlap: core and native length-planning tests plus AML batching sweeps. Keep these; compare individual budget limits before adding more. |
| G-BATCH-008: zero-length samples consume sample capacity | Empty-history coverage exists. The explicit zero-cost token-budget grid remains to be compared. |
| G-BATCH-009: 7; 1,7; 7,1; 1,2,7; exact 6, padded and packed | Core tests reject one oversized example; native planning tests reject an unseen long history. Added all ten position/layout witnesses, verifying no oversized example is emitted and the exact-budget control succeeds. |
| G-BATCH-010: four nested layouts | Overlap: `test_field_outputs.py::test_grouped_token_budget_counts_materialized_token_slots`, `test_single_grouped_history_exceeding_padding_budget_is_refused` and native nested-composition tests. Compare the complete four-layout matrix before porting. |
| G-BATCH-011: terminal/nonterminal budget-full batch | Existing core tests `test_drop_last_under_a_token_budget_drops_only_the_final_short_batch` and `test_drop_last_judges_the_final_batch_by_sample_count_not_tokens` already assert both answers. Retained unchanged. |
| G-BATCH-012: unbucketed/bucketed/minimum witness | Existing tests assert grouping, not this independently calculated padding cost. Added `[1,30,2,29]`: 56 versus 2 padding positions. The three-partition minimum assertion is folded into the two execution checks. |
| G-BATCH-013: truncate before length sorting | Existing tests cover retrieval caps and native lengths separately. Added `[40,3,31]` capped at 30: exact grouping plus retained history values. |
| G-BATCH-014: one/two-batch chunks | Added exact chunk-local sorting from the unbucketed stored-anchor order. No assumption that different engines have identical stored anchor order. |
| G-BATCH-015: candidate/label alignment under bucketing | Candidate alignment and bucketing are tested separately; their combined padded/packed witness remains to be compared. |

The six ported families are G-BATCH-001/002/009/012/013/014. Their 30 Arena
variants map to 29 DiskGraph execution checks because the partition-minimum
calculation is checked alongside execution. The fixture uses declared types,
tiny authored tables and explicit requests; it does not claim full-data,
GPU, model-output or native Parquet-anchor campaign acceptance.

## Required execution and policy

The existing conformance gate pins each new node ID independently of its test
parametrization. Missing, skipped, deselected and expected-failure cases cannot
produce a successful complete run. Fault-injection tests exercise those paths.
The package CPU and DiskGraph PR workflows already invoke `python -m conformance`;
the nightly invokes it on both engines with extra property seeds. No new workflow
or runtime framework is introduced. These test files remain outside `src/` and
outside the installed wheel.

Do not port catalog statuses as acceptance requirements. Rejection is a passing
answer only for invalid inputs that are required to be rejected. If required
functionality is unsupported, its answer test must fail normally, not skip or
xfail. Arena's two unresolved policies are recorded above, not silently adopted.

Arena's executable typed-API notes also differ from older prose: event windows
use `(start, stop]`, strict histories need explicit bounds, and `drop_last`
judges the final batch by sample count. Follow the current Engine contract and
the user's workload requirements; do not change expectations merely to match
observed output. Generic golden answers are independent per engine, not a demand
for identical Polars/DiskGraph batch grouping.

## Validation and remaining work

DiskGraph-only validation: all 105 focused package checks passed, zero skips:
29 new boundary checks, 26 required-execution harness checks, 45 coverage-sweep
cases, three fitting-boundary cases and two explicit-ID lifecycle cases.
Collection also matched all 29 independently pinned boundary IDs exactly.
CI still needs to rebuild and validate the branch's native revision. Local runs
use the checked-out Python sources and existing compatible DiskGraph extension;
they do not establish cloud or real-dataset acceptance.

The family ownership map is complete; the 1,863-variant comparison is not.
Continue with the open batching witnesses above, then schema/row/reduction
boundaries. Only DiskGraph assertions count for new coverage. Keep exact variant and
execution-path gaps explicit. A family with
an existing owner is not automatically complete, and this audit is not a reason
to replace the existing conformance, AML, lifecycle or training suites.
