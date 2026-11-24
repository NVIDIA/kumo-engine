# Blueprint v2 and PRAGMA-like against raw-feature XGBoost, pragmatiq (one GPU, three seeds)

Milestone 2's headline comparison
([ADR 0018](../../docs/architecture/0018-one-headline-comparison.md)), scored by
the recipe's own evaluation rather than benchmark scripts: does XGBoost on raw
features plus the frozen embedding (`xgboost_raw_embedding`) beat XGBoost on
raw features alone (`xgboost_raw`)? Hand-built history aggregates
(`tabular_xgboost`) are reported beside them as the reference. Two
architectures of matched size are run on the same probe rows:

- `blueprint_v2`, the hybrid: PRAGMA's event and profile encoders under a
  causal history decoder.
- `pragma_like`, the encoder-only PRAGMA clone, sharing the same event and
  profile encoder modules.

This is not a quality, fraud, or production claim.

| field | value |
|---|---|
| configs | `recipes/blueprint_v2/config.yaml` and `recipes/pragma_like/config.yaml`, no overrides; seeds 42, 43, 44 |
| code | a snapshot of the working tree on top of commit `e871249`, taken before the config keys moved into sub-sections; the snapshot's configs differ from the committed ones only in those key names (`pretrain_window_events`, `muon_learning_rate`, `history_window_probe`, `history_window_max_rows`). The snapshot also still wrote the logistic-regression arms (`raw`, `embedding`, `combined`, `tabular_logistic`), which the current evaluation drops when XGBoost arms are on; they are in the JSON but not used here |
| data | `synth_100k_gen2_m31_seed0`, 10,000 users, most recent 128 events each, through the token cache; train cutoff 2024-06-01, validation cutoff 2024-09-01, test window to 2025-07-31 |
| fraud task | spliced identity theft, 20% victims, matched donors ([ADR 0015](../../docs/architecture/0015-dataset-strategy-and-spliced-fraud.md)) |
| tokens | key-value-time, 40 tokens per event, `categorical_keys: [mcc, app_version]`, three hashed keys at 4,096 buckets |
| `blueprint_v2` | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA, varlen; next-event fields plus token/event/key MLM; last-token readout; 11,513,856 parameters |
| `pragma_like` | profile / event / history depths 2 / 4 / 4, same widths; token/event/key MLM only; `USR`-token readout; 11,447,808 parameters |
| training | both: eager, bf16, batch 128, 1,500 steps, Muon (LR 0.01) on the hidden matrices and AdamW (LR 1e-3) on the rest, cosine schedule, 75 warmup; no non-finite steps |
| probes | every XGBoost arm uses one fixed setting (`tabular_baselines.xgboost_params`), so arms differ only in their features; LoRA rank 8 on qkv, out, and the feed-forward layers, early-stopped on a validation split |
| rows | identical for both models at each seed (same users, splice, and labels): fraud 375–386 positives of about 2,890 evaluation users; churn 295–299 of about 2,380; default 60–64 of about 2,810; LTV 1,679–1,733 of about 2,380; history window 618–640 of 2,885–3,067 rows |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | `blueprint_v2`: prepare 289–298 s, train 1,293–1,316 s, evaluate 1,150–1,370 s. `pragma_like`: prepare 294–298 s, train 1,422–1,424 s, evaluate 1,262–1,468 s |
| memory | peak GPU 21.5–21.6 GiB (`blueprint_v2`), 31.9 GiB (`pragma_like`); host memory not measured |
| raw output | [`blueprint_v2_seeds.json`](blueprint_v2_seeds.json), [`pragma_like_seeds.json`](pragma_like_seeds.json) |

Reproduce:

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv sync --extra baselines
for recipe in blueprint_v2 pragma_like; do
  uv run python benchmarks/seed_range.py \
    --config recipes/$recipe/config.yaml --seeds 42 43 44 \
    --run-root outputs/$recipe/seeds \
    --summary benchmarks/results/${recipe}_seeds.json --device cuda
  uv run python benchmarks/lift_table.py benchmarks/results/${recipe}_seeds.json \
    --reference xgboost_raw
done
```

## Results

PR-AUC lift over `xgboost_raw`, median [min, max] over the three seeds. Every
arm is scored on the same rows as its `xgboost_raw`, and `xgboost_raw` and
`tabular_xgboost` are the same for both models.

| Task | `blueprint_v2` raw + embedding | `pragma_like` raw + embedding | `blueprint_v2` raw + embedding + surprise | `blueprint_v2` LoRA | `pragma_like` LoRA | `tabular_xgboost` (reference) |
|---|---|---|---|---|---|---|
| fraud (spliced) | +9% [+7%, +16%] | +3% [-13%, +7%] | +111% [+97%, +120%] | +144% [+138%, +147%] | +7% [+2%, +8%] | +178% [+154%, +196%] |
| churn, 6 months | +18% [+17%, +25%] | +20% [+20%, +24%] | +16% [+13%, +19%] | +33% [+27%, +44%] | +38% [+23%, +41%] | +38% [+29%, +43%] |
| default, 12 months | +7% [-15%, +8%] | +7% [-1%, +9%] | +12% [-11%, +22%] | -1% [-5%, +9%] | -2% [-9%, +11%] | +41% [+21%, +42%] |
| LTV positive | +2% [+1%, +2%] | +1% [+1%, +2%] | +2% [+2%, +3%] | +2% [+1%, +2%] | +1% [+1%, +3%] | +3% [+3%, +4%] |
| fraud, per transaction (history window) | +26% [+4%, +32%] | +3% [-1%, +18%] | +73% [+46%, +83%] | +34% [+11%, +43%] | +34% [+21%, +36%] | +113% [+95%, +125%] |

`xgboost_raw` PR-AUC, the base of every lift: fraud 0.184–0.201, churn
0.255–0.284, default 0.228–0.270, LTV 0.908–0.914, history window 0.258–0.277.

Share of `tabular_xgboost`'s PR-AUC gain over `xgboost_raw` that
`xgboost_raw_embedding` recovers, median [min, max]:

| Task | `blueprint_v2` | `pragma_like` |
|---|---|---|
| fraud | 6% [4%, 9%] | 2% [-8%, 4%] |
| churn | 60% [43%, 67%] | 52% [45%, 82%] |
| default | 16% [-37%, 38%] | 16% [-3%, 44%] |
| LTV | 56% [53%, 64%] | 44% [28%, 52%] |
| history window | 23% [4%, 26%] | 2% [-1%, 16%] |

What these runs support, at three seeds on one synthetic book:

- **`blueprint_v2`'s embedding beats raw-feature XGBoost on four of the five
  tasks at every seed.** Those tasks are fraud, churn, LTV, and the
  per-transaction fraud probe. On default (about 60 positives) the lift
  changes sign across seeds, so it is not a result.
- **The frozen embedding does not reach the hand-built aggregates on any
  task.** It recovers about half of their gain on churn and LTV, and little
  on fraud. Appending the decoder's next-event surprise to raw features and
  the embedding closes most of the fraud gap (+111% against +178%).
  Fine-tuning with LoRA also closes most of it (+144%).
- **The hybrid's advantage over PRAGMA-like is on the fraud tasks, not on
  churn, default, or LTV.** On those three the two encoders are within each
  other's seed range. On spliced fraud, PRAGMA-like's frozen embedding moves
  raw features by -13% to +7% across seeds. Its LoRA fine-tune adds 7%
  against v2's 144%, on the same rows. The per-transaction probe shows the
  same pattern for the frozen embedding.
- **The single-event control does not lift.** `xgboost_raw_embedding:single_event`
  (the scored transaction embedded alone) is -6% [-11%, +3%] for
  `blueprint_v2` and -10% [-11%, -5%] for `pragma_like`. Where the embedding
  helps on this book, it is the history that helps.

AML has one or two evaluation positives per seed, below the five the report
needs to trust a ranking metric. Every AML XGBoost arm predicts a constant
(ROC-AUC exactly 0.5), so AML is a diagnostic, not a task, at 10,000 users.

## Not measured

- A larger book or model: `blueprint_v2/config.scale.yaml` was not run.
- Host memory, and the re-scoring of these checkpoints with the final
  working tree. The XGBoost and LoRA arms used here go through the same
  fitter there, but this record does not show they reproduce exactly.
- Whether the surprise readout's fraud gain is specific to spliced identity
  theft, which by construction changes what comes next.
