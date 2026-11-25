# PRAGMA-like against Blueprint v1 on TabFormer (seed 42)

Does an encoder-only model give TabFormer a history embedding that adds to
raw features, where v1's causal decoder's does not reliably
([`blueprint_v1_seeds`](blueprint_v1_seeds.md))? `pragma_like`'s encoder, on
v1's users, split, probe rows, and XGBoost setting, beside v1's decoder
re-scored on the same rows. One seed, an exploration: not a quality, fraud, or
production claim.

| field | value |
|---|---|
| configs | [`benchmarks/ablations/pragma_like_tabformer.yaml`](../ablations/pragma_like_tabformer.yaml) and `recipes/blueprint_v1/config.yaml`, seed 42, both with `evaluation.history_window.max_rows: 250000` |
| code | commit `124ab93` plus the uncommitted `features.exclude_keys` option |
| data | IBM TabFormer through `tfm prepare-tabformer`, v1's precomputed date split, all 2,000 users, 128-event probe histories |
| rows | train side: every training-window fraud with 5 negatives each; evaluation side: a seeded uniform sample of 250,000 of v1's ~700,000 test rows, **277 frauds**. The raw and hand-built arms score identically in both runs, so the rows are the same |
| `pragma_like` | profile / event / history encoders, depths 2 / 4 / 4, d=256, 8 heads, TimeRoPE; token / event / key MLM; `[USR]` readout; 11.4M parameters. Key-value-time tokens: `mcc` and `merchant_state` categorical, merchant name, city, and zip hashed (4,096 buckets), `user` excluded; no profiles. 1,500 steps of 128 windows of 128 events, Muon and AdamW, bf16; loss 35.1 to 2.38 |
| v1 | `blueprint_v1/config.yaml` unchanged: Llama decoder, 25.9M parameters, next-event on 341-event windows, 3,000 steps; last 100 steps' mean loss 0.574 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock | `pragma_like`: prepare 300 s, train about 25 min, evaluate 1,594 s. v1: prepare 298 s, train 4,854 s, evaluate 3,152 s |
| raw output | [`pragma_like_tabformer_seed42.json`](pragma_like_tabformer_seed42.json) |

The full test set (every row, as in the v1 record) ran `pragma_like` out of
the GB10's 121 GB of host memory: the probe holds every row's key-value-time
batch at once. Hence the 250,000-row sample for both models.

## Results

`history_window` PR-AUC on the same 250,000 rows; lift over `xgboost_raw`.

| arm | `pragma_like` (encoder) | v1 (decoder) |
|---|---|---|
| `xgboost_raw` | 0.0414 | 0.0414 |
| raw + history embedding | 0.0395 (−5%) | **0.0566 (+37%)** |
| raw + transaction-alone embedding | 0.0422 (+2%) | 0.0336 (−19%) |
| `tabular_xgboost` (hand-built) | 0.0650 (+57%) | 0.0650 (+57%) |
| hand-built + history embedding | 0.0704 (+70%) | 0.0691 (+67%) |

ROC-AUC: raw 0.916; `pragma_like` raw + history 0.917; v1 raw + history
0.965; hand-built 0.975; hand-built + embedding 0.980 (`pragma_like`) and
0.981 (v1).

## Reading

- **The encoder adds nothing to raw features here.** Its history embedding
  moves PR-AUC by −5% and ROC-AUC by +0.001; v1's decoder, on the same rows,
  lifts PR-AUC by 37% and ROC-AUC by 0.05.
- **Both embeddings add a little to hand-built features** (+8% and +6% over
  `tabular_xgboost`), as v1's did on every row at three seeds.
- **277 frauds make single arms noisy.** v1's transaction-alone arm is −19%
  here and +146% on every test row of the original seed-42 checkpoint
  (`blueprint_v1_seeds`); its history arm is +37% here and −1% there. Compare
  arms within this table only.
- An encoder-only stack, as built in this kit for pragmatiq, does not rescue
  TabFormer's history embedding. A flat bidirectional encoder on v1's own
  tokens ([flat-encoder guide](../../recipes/blueprint_v1/README.md),
  proposed) would isolate the mask from the tokens and hierarchy, which this
  comparison changes all at once.

## Limitations

One seed. 277 evaluation frauds. The v1 checkpoint here is a retrain of the
seed-42 config, not the checkpoint behind `blueprint_v1_seeds`: re-scoring it
retrained the model, and the new weights replaced the original file (shared
by hard link with `outputs/evidence/blueprint_v1/{seeds,len_16,len_32}`,
`blueprint_v1_rawfix`, and `tabformer_hf_decoder/gb10_seeds`, none of which
is committed). `pragma_like` was sized for pragmatiq and not tuned for
TabFormer; LoRA was off for both.
