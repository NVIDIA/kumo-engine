# Blueprint v2 hybrid at lab size, with tuned tabular baselines and lift (one GPU, three seeds)

The config of
[`blueprint_v2_hybrid_lab_splice_seeds.md`](archive/blueprint_v2_hybrid_lab_splice_seeds.md),
rerun after the tokenizer fix (every event keeps `amount` and `mcc`, and
`mcc` is categorical), with the tuned tabular baselines as the reference for
lift ([evaluation protocol](../../docs/concepts/evaluation-protocol.md)).
It asks whether this 8.4M-parameter encoder adds anything to a tuned
gradient-boosted model on hand-built features, as PRAGMA (arXiv 2604.08649)
reports for its own models on Revolut data. At this size, on this book, it
does not. This is not a quality claim either way.

| field | value |
|---|---|
| config | `recipes/blueprint_v2_hybrid/config.lab.yaml`, no overrides; seeds 42, 43, 44 |
| code | trained and first evaluated at commit `bb64ba2`; re-evaluated from the same checkpoints (`--reuse-checkpoint`) at the commit that adds `tabular_xgboost_embedding`. Every earlier arm reproduced exactly (largest difference 0.0) |
| data | as the splice record: `synth_100k_gen2_m31_seed0`, 10,000 users sampled uniformly, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01 |
| fraud task | spliced identity theft, matched donors, as the splice record |
| tokens | key-value-time, 40 tokens per event, `categorical_keys: [mcc]` |
| model | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA; 8,356,096 parameters |
| objective | next-event fields plus token/event/key MLM |
| training | eager, bf16, batch 32, 3,000 steps, cosine schedule with 150 warmup, AdamW, peak LR 5e-4 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare 330–355 s, train 1,162–1,188 s, evaluate 1,240–1,276 s before the embedding arm; the re-evaluation with it took 34.7–35.1 min including data preparation |
| memory | 7.16–7.33 GiB peak GPU; 49.5–50.0 GB peak host in the re-evaluation |
| loss | 54.0–59.0 at step 0, 4.10–4.18 at step 2,999 |
| probes | logistic regression on the frozen embedding; LoRA rank 8 on qkv and out, 300 steps; tabular arms as in the evaluation protocol, 30 Optuna trials each |
| raw output | [`blueprint_v2_hybrid_lab_kvt_seeds.json`](blueprint_v2_hybrid_lab_kvt_seeds.json) |

Reproduce (at the current commit one sweep writes every arm):

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv sync --extra baselines
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v2_hybrid/config.lab.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v2_hybrid/lab_kvt_seeds \
  --summary benchmarks/results/blueprint_v2_hybrid_lab_kvt_seeds.json --device cuda
uv run python benchmarks/lift_table.py benchmarks/results/blueprint_v2_hybrid_lab_kvt_seeds.json
```

## Results

ROC-AUC and PR-AUC, median [min, max] over three seeds. `tabular_*` arms
read no encoder, except `tabular_xgboost_embedding`, which appends the
frozen embedding to the tabular features.

| arm | ROC-AUC | PR-AUC | eval rows | eval positives |
|---|---|---|---|---|
| `raw` (fraud, one row per entity) | 0.568 [0.542, 0.571] | 0.146 [0.135, 0.154] | 2891–2895 | 375–386 |
| `embedding` | 0.699 [0.698, 0.709] | 0.242 [0.220, 0.252] | 2891–2895 | 375–386 |
| `lora_fraud` | 0.707 [0.666, 0.720] | 0.221 [0.189, 0.251] | 2891–2895 | 375–386 |
| `tabular_logistic` | 0.787 [0.783, 0.789] | 0.530 [0.511, 0.544] | 2891–2895 | 375–386 |
| `tabular_xgboost` | 0.794 [0.768, 0.798] | 0.529 [0.474, 0.536] | 2891–2895 | 375–386 |
| `tabular_xgboost_embedding` | 0.779 [0.767, 0.793] | 0.503 [0.491, 0.525] | 2891–2895 | 375–386 |
| `history_window_raw` | 0.614 [0.560, 0.620] | 0.306 [0.265, 0.314] | 2885–3067 | 618–640 |
| `history_window_embedding` | 0.646 [0.607, 0.667] | 0.329 [0.303, 0.338] | 2885–3067 | 618–640 |
| `history_window_tabular_logistic` | 0.728 [0.711, 0.775] | 0.453 [0.416, 0.495] | 2885–3067 | 618–640 |
| `history_window_tabular_xgboost` | 0.816 [0.784, 0.821] | 0.586 [0.530, 0.592] | 2885–3067 | 618–640 |
| `history_window_tabular_xgboost_embedding` | 0.795 [0.778, 0.814] | 0.544 [0.520, 0.573] | 2885–3067 | 618–640 |
| `churn_6m_raw` | 0.716 [0.693, 0.716] | 0.269 [0.255, 0.295] | 2352–2398 | 295–299 |
| `churn_6m_embedding` | 0.743 [0.737, 0.759] | 0.306 [0.282, 0.334] | 2352–2398 | 295–299 |
| `lora_churn_6m` | 0.765 [0.763, 0.778] | 0.332 [0.322, 0.339] | 2352–2398 | 295–299 |
| `churn_6m_tabular_logistic` | 0.754 [0.749, 0.759] | 0.340 [0.305, 0.347] | 2352–2398 | 295–299 |
| `churn_6m_tabular_xgboost` | 0.779 [0.738, 0.779] | 0.341 [0.333, 0.354] | 2352–2398 | 295–299 |
| `churn_6m_tabular_xgboost_embedding` | 0.767 [0.762, 0.771] | 0.348 [0.329, 0.349] | 2352–2398 | 295–299 |
| `default_12m_raw` | 0.750 [0.689, 0.762] | 0.138 [0.123, 0.227] | 2810–2814 | 60–64 |
| `default_12m_embedding` | 0.650 [0.640, 0.713] | 0.152 [0.125, 0.175] | 2810–2814 | 60–64 |
| `lora_default_12m` | 0.735 [0.682, 0.743] | 0.258 [0.212, 0.273] | 2810–2814 | 60–64 |
| `default_12m_tabular_logistic` | 0.725 [0.655, 0.735] | 0.140 [0.138, 0.202] | 2810–2814 | 60–64 |
| `default_12m_tabular_xgboost` | 0.771 [0.739, 0.776] | 0.306 [0.304, 0.322] | 2810–2814 | 60–64 |
| `default_12m_tabular_xgboost_embedding` | 0.772 [0.688, 0.783] | 0.344 [0.262, 0.356] | 2810–2814 | 60–64 |
| `ltv_positive_raw` | 0.726 [0.712, 0.746] | 0.856 [0.845, 0.875] | 2352–2397 | 1679–1733 |
| `ltv_positive_embedding` | 0.832 [0.829, 0.835] | 0.915 [0.914, 0.917] | 2352–2397 | 1679–1733 |
| `lora_ltv_positive` | 0.848 [0.846, 0.870] | 0.925 [0.923, 0.935] | 2352–2397 | 1679–1733 |
| `ltv_positive_tabular_logistic` | 0.838 [0.838, 0.840] | 0.911 [0.911, 0.915] | 2352–2397 | 1679–1733 |
| `ltv_positive_tabular_xgboost` | 0.876 [0.871, 0.878] | 0.939 [0.938, 0.941] | 2352–2397 | 1679–1733 |
| `ltv_positive_tabular_xgboost_embedding` | 0.878 [0.863, 0.878] | 0.940 [0.932, 0.941] | 2352–2397 | 1679–1733 |

The JSON also has `combined`, `single_event_*`, and `aml_*` rows; with 1–2
evaluation positives per seed the AML rows are diagnostics, not results.

## Lift over `tabular_xgboost`

PR-AUC lift \(x / x_{\text{ref}} - 1\) per seed (42, 43, 44), from
`benchmarks/lift_table.py`:

| task | `embedding` | `lora` | `tabular_xgboost_embedding` |
|---|---|---|---|
| fraud (entity) | −52%, −54%, −55% | −53%, −60%, −59% | −7.1%, +6.0%, −2.0% |
| history window | −43%, −43%, −44% | – | −1.8%, −8.1%, −2.4% |
| `churn_6m` | −8%, −17%, −6% | −0%, −6%, −4% | +4.8%, +2.2%, −7.0% |
| `default_12m` | −46%, −59%, −50% | −20%, −31%, −10% | +10.8%, −14.5%, +13.2% |
| `ltv_positive` | −3%, −3%, −2% | −2%, −2%, −0% | −0.0%, −0.6%, +0.1% |

What the table supports:

- **No encoder arm reaches the tuned tabular model's PR-AUC** on any task at
  any seed. On fraud the gap is about half the PR-AUC. The one ROC-AUC
  exception is `lora_churn_6m` at seed 42 (0.778 against 0.738).
- **Appending the embedding adds nothing reproducible.** Its lift changes
  sign across seeds on fraud, churn, and default; it is negative at every
  seed on the history window and zero on LTV. The largest median, +10.8% on
  default, rests on 60–64 positives and has a −14.5% seed.
- **The tokenizer fix did not move the embedding.** Entity-level fraud
  ROC-AUC is 0.699 against 0.724 before the fix (splice record), and the
  forecast embeddings moved by 0.04 or less.
- **The five-aggregate `raw` arm overstated what the encoder adds.** Against
  it the embedding wins on fraud, churn, and LTV at every seed, as the splice
  record reports; against a tuned model on the same rows it loses.

## Limitations

- **This is not PRAGMA's setting.** PRAGMA pretrains far larger models on
  real Revolut histories and compares with Revolut's production models. Here
  it is 8.4M parameters, 3,000 steps, and 10,000 users with 128 events each,
  from a synthetic generator whose behaviour the hand-built features may
  capture directly. Which of model size, data scale, or the generator
  explains the gap is not tested here.
- **The history-window tabular features are generous.** Counts of new
  merchants and categories in the last 1, 7, and 30 days describe how the
  splice replaces a day of a victim's history. They are legal features (no
  label or future data), but the splice gives them a direct signal.
- **The embedding arm appends 256 dimensions** to the tabular features, with
  the same 30-trial search. A smaller projection or a longer search might
  change it; neither was tried.
- **Timings mix two runs.** Training and the first evaluation are from the
  sweep at `bb64ba2`; the embedding arm's cost is only in the re-evaluation's
  total. The sweep grew host memory from seed to seed in one process; the
  seed runner now starts one process per seed.
- Three pretraining seeds on one book at one size, and no multi-GPU run.
