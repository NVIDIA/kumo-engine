# Recorded results

Every claim this kit makes about a model rests on a record here: a summary
(`.md`) and its numbers (`.json`) from a committed config. Run directories and
checkpoints stay outside git. What a record must contain is in the
[evaluation protocol](../../docs/concepts/evaluation-protocol.md#recording-a-result);
milestone status is in the [roadmap](../../docs/reference/roadmap.md).

None of these records is a quality, fraud, or production claim. Each one
measures one configuration on one dataset, and results from different
datasets are never combined
([datasets](../../docs/reference/datasets.md)).

## Current

| Record | Dataset | What it shows |
|---|---|---|
| [`blueprint_v1_seeds`](blueprint_v1_seeds.md) | TabFormer | Blueprint v1 at three seeds on every test row: the single-transaction embedding lifts raw-feature XGBoost by 145–192% PR-AUC; the history embedding's lift (−1% to +42%) does not hold |
| [`pragma_like_tabformer_seed42`](pragma_like_tabformer_seed42.md) | TabFormer | PRAGMA-like's encoder against v1's decoder on the same 250,000 test rows, seed 42: the encoder's history embedding adds nothing to raw features (−5%); v1's adds +37% |
| [`flat_encoder_tabformer_seed42`](flat_encoder_tabformer_seed42.md) | TabFormer | ADR 0019's flat BERT-style encoder on v1's tokens and the same 250,000 rows, seed 42: its history embedding lifts raw-feature XGBoost PR-AUC by 443% (v1 +37%, PRAGMA-like −5%); one seed, 277 frauds |
| [`blueprint_v2_seeds`](blueprint_v2_seeds.md) | pragmatiq | Blueprint v2 and PRAGMA-like at 11.5M parameters, three seeds: v2's embedding lifts raw-feature XGBoost on four of five tasks; neither reaches the hand-built aggregates |
| [`blueprint_v2_final_sweep`](blueprint_v2_final_sweep.md) | pragmatiq | Pretraining steps (1,500 or 3,000) by LoRA rank (8 or 16), three seeds, chosen by a rule fixed in advance: 3,000 steps with rank 16 wins; its surprise arm lifts raw-feature XGBoost on all four headline tasks at every seed, and its LoRA passes the hand-built reference on churn |
| [`blueprint_v2_hybrid_lab_kvt_seeds`](blueprint_v2_hybrid_lab_kvt_seeds.md) | pragmatiq | Blueprint v2 at 8.4M parameters, three seeds, with tuned tabular baselines: no lift that holds across seeds |
| [`blueprint_v2_hybrid_gb10_tuning`](blueprint_v2_hybrid_gb10_tuning.md) | pragmatiq | The search behind `blueprint_v2/config.yaml`: one pretraining seed, scored on a proxy |
| [`blueprint_v2_hybrid_lab_v1_flat_splice_seeds`](blueprint_v2_hybrid_lab_v1_flat_splice_seeds.md) | pragmatiq | v1's flat decoder on v2's rows, against the hybrid, on spliced identity theft |
| [`pragmatiq_fraud_separability`](pragmatiq_fraud_separability.md) | pragmatiq | Why fraud is spliced: one transaction identifies the book's own fraud labels (ADR 0015) |

## Systems checks

| Record | Dataset | What it shows |
|---|---|---|
| [`graph_backends_l4_cost`](graph_backends_l4_cost.md) | Seeded generated relational fixture | Short six-hop L4 cost comparison: identical-state Engine shared/disjoint, and the historical custom GNN on the same disjoint inputs; no full-pipeline or quality claim |

## Archive

Superseded records, kept because later records and ADRs cite them. Each says
what superseded it.

| Record | Superseded by | Why |
|---|---|---|
| [`tabformer_hf_decoder_gb10_seed42`](archive/tabformer_hf_decoder_gb10_seed42.md) | `blueprint_v1_seeds` | One seed on a 100,000-row sample (106 frauds), with a search per arm |
| [`blueprint_v2_hybrid_lab_seeds`](archive/blueprint_v2_hybrid_lab_seeds.md) | `blueprint_v2_hybrid_lab_kvt_seeds` | Scored on the book's own fraud labels, which a single event gives away |
| [`blueprint_v2_hybrid_lab_splice_seeds`](archive/blueprint_v2_hybrid_lab_splice_seeds.md) | `blueprint_v2_hybrid_lab_kvt_seeds` | Before the tokenizer kept `amount` and `mcc` on every event, and before tuned tabular baselines |
| [`blueprint_v2_hybrid_lab_v1_flat_seeds`](archive/blueprint_v2_hybrid_lab_v1_flat_seeds.md) | `blueprint_v2_hybrid_lab_v1_flat_splice_seeds` | Scored on the book's own fraud labels |

## Names in older records

Records keep the recipe and config names they were run under. The current
names are:

| In a record | Now |
|---|---|
| `tabformer_hf_decoder/config.gb10.yaml` | `blueprint_v1/config.yaml` |
| `tabformer_hf_decoder/config.yaml` | `blueprint_v1/config.small.yaml` |
| `blueprint_v2_hybrid/config.lab_tuned.yaml` | `blueprint_v2/config.yaml` |
| `blueprint_v2_hybrid/config.pragmatiq.yaml` | `blueprint_v2/config.scale.yaml` |
| `blueprint_v2_hybrid/config.lab.yaml` | `benchmarks/ablations/blueprint_v2_lab.yaml` |
| `blueprint_v2_hybrid/config.lab_v1_flat.yaml` | `benchmarks/ablations/blueprint_v2_lab_v1_flat.yaml` |
| `blueprint_v2_hybrid/pragmatiq_world.yaml` | `configs/data/pragmatiq/world.yaml` |
| `profile_history_kvt` | `pragma_like` |
| `history_window_probe`, `history_window_max_rows` | `evaluation.history_window.enabled`, `.max_rows` |
| `pretrain_window_events` | `data.pretrain_windows.events` |
| `single_event_raw` arm | dropped: it repeated the history row's raw arm |

The full list, including config keys, is in the
[migration guide](../../docs/migration/2.0.0a1-to-2.0.0a2.md).


## Graph benchmark raw artifacts

Graph execution/construction JSON files are imported compact manifests with
summaries, conditions, hashes, and byte counts. The upstream raw runs and
embedding coordinates are held outside this public repository and have not
been mirrored here. Private artifact URLs are omitted during import; the
recorded hashes, measurements, and other provenance remain unchanged.
The unverified resident-baseline comparison remains excluded from the reports.

If you already have the original payloads, verify their hashes before using
`scripts/compact_graph_benchmarks.py` to regenerate the manifests. This import
does not provide public raw downloads or independently reproduce these results.
See [resource measurements](graph_resource_supplement_l4.md) for the distinction
between historical latency and later memory measurements.
