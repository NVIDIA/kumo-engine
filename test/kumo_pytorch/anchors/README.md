# AML anchors

The [input inventory](../datasets/inputs.json) pins both account-daily anchor
versions for every AML variant. Each selected variant downloads its actual
manifests and tables under `.data/anchors/aml-data/v8/<variant>/account-daily/`.
The downloaded manifests contain the schemas, counts and provenance; no second
copy of that metadata is maintained here.

| Workload/model | Anchor version |
| --- | --- |
| Account involvement: GraphSAGE or combined | v2 |
| Account involvement: nuFormer | v1 |
| Masked-value pretraining: PRAGMA | v1 |
| Next-token pretraining: nuFormer | v1 |

V2 adds `graph_as_of_time` while preserving every original column and row.
For prediction time T, its inclusive graph execution clock is
`C_seconds = (T_epoch_ns - 1) // 1_000_000_000`. For AML daily cutoffs this is
T minus one second. Preserve T for label windows and temporal ages. Combined
history/aggregate windows also need translation to preserve their original
boundaries. The execution clock is metadata, not a feature.
