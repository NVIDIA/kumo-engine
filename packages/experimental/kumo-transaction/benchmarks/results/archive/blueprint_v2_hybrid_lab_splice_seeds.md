# Blueprint v2 hybrid at lab size on spliced identity theft (one GPU, three seeds)

Milestone 2, criterion 7, rerun with the fraud probe on the spliced
identity-theft task
([dataset strategy](../../../docs/reference/datasets.md))
instead of pragmatiq's own fraud labels, which one transaction identifies.
The model, book, and schedule are those of
[`blueprint_v2_hybrid_lab_seeds.md`](blueprint_v2_hybrid_lab_seeds.md); the
user sample is not (see "What changed"). This shows the wiring at a size
where numbers stop being noise. It is not a quality claim.

Superseded by
[`blueprint_v2_hybrid_lab_kvt_seeds.md`](../blueprint_v2_hybrid_lab_kvt_seeds.md),
which reruns this config after the tokenizer fix, with tuned tabular baselines.

| field | value |
|---|---|
| config | `recipes/blueprint_v2_hybrid/config.lab.yaml`, no overrides; seeds 42, 43, 44 |
| code | commit `d8cc2a1`, no uncommitted changes to tracked files |
| data | pragmatiq Generator v2 book from `recipes/blueprint_v2_hybrid/pragmatiq_world.yaml`: 100,000 users, seed 0, 31 months (2023-01-01 to 2025-08-01), 128.5M events (`synth_100k_gen2_m31_seed0`); 10,000 users sampled uniformly, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01 |
| fraud task | spliced identity theft, matched donors (same home country, median amount within 1.5×); victim share 0.2 of each probe side; negatives are the victim's own card transactions in the 7 days before the theft day and on it; theft days in the 28 days before the side's cutoff |
| splice per seed | 9,636–9,650 eligible users; 996–1,004 train-side and 383–390 evaluation-side victims; 2,261–2,274 fraud and 8,923–9,230 legitimate labelled transactions; 536–547 intended victims found no matched donor and stay unlabelled; 34 users with pragmatiq's own fraud take no role |
| model | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA; 8,352,768 parameters |
| objective | next-event fields plus token/event/key MLM |
| training | eager, bf16, batch 32, 3,000 steps, cosine schedule with 150 warmup, AdamW, peak LR 5e-4 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare 314–321 s, train 765–769 s, evaluate 773–782 s; 1 h 33 min for three seeds |
| memory | 4.88–4.91 GiB peak GPU; peak host memory not measured in this run |
| loss | 51.4–59.9 at step 0, 4.09–4.17 at step 2,999 |
| probes | logistic regression; entity split fixed by the splice (70/30, victim and donor on one side); LoRA rank 8 on qkv and out, 300 steps, LR 1e-3, linear head |
| raw output | [`blueprint_v2_hybrid_lab_splice_seeds.json`](blueprint_v2_hybrid_lab_splice_seeds.json), from `benchmarks/seed_range.py`; each run also writes `identity_theft_manifest.json` |

Reproduce (build the book first; the command is in the header of
`pragmatiq_world.yaml`):

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v2_hybrid/config.lab.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v2_hybrid/lab_splice_seeds \
  --summary benchmarks/results/blueprint_v2_hybrid_lab_splice_seeds.json --device cuda
```

## Results

ROC-AUC and PR-AUC, median [min, max] over three seeds. The seed changes
pretraining, the user sample's split, the victims and donors, and
fine-tuning. It does not change the book.

| arm | ROC-AUC | PR-AUC | eval rows | eval positives |
|---|---|---|---|---|
| `raw` (fraud, one row per entity) | 0.568 [0.542, 0.571] | 0.146 [0.135, 0.154] | 2891–2895 | 375–386 |
| `embedding` | 0.724 [0.714, 0.740] | 0.254 [0.253, 0.280] | 2891–2895 | 375–386 |
| `combined` | 0.724 [0.718, 0.737] | 0.254 [0.252, 0.283] | 2891–2895 | 375–386 |
| `lora_fraud` | 0.689 [0.669, 0.712] | 0.208 [0.189, 0.219] | 2891–2895 | 375–386 |
| `history_window_raw` | 0.614 [0.560, 0.620] | 0.306 [0.265, 0.314] | 2885–3067 | 618–640 |
| `history_window_embedding` | 0.641 [0.610, 0.663] | 0.347 [0.307, 0.361] | 2885–3067 | 618–640 |
| `history_window_combined` | 0.641 [0.609, 0.661] | 0.347 [0.308, 0.356] | 2885–3067 | 618–640 |
| `single_event_embedding` (control) | 0.580 [0.570, 0.587] | 0.259 [0.257, 0.265] | 2885–3067 | 618–640 |
| `single_event_combined` (control) | 0.620 [0.584, 0.627] | 0.300 [0.269, 0.301] | 2885–3067 | 618–640 |
| `churn_6m_raw` | 0.716 [0.693, 0.716] | 0.269 [0.255, 0.295] | 2352–2398 | 295–299 |
| `churn_6m_embedding` | 0.740 [0.737, 0.764] | 0.329 [0.291, 0.349] | 2352–2398 | 295–299 |
| `churn_6m_combined` | 0.740 [0.737, 0.762] | 0.335 [0.287, 0.343] | 2352–2398 | 295–299 |
| `lora_churn_6m` | 0.766 [0.762, 0.770] | 0.328 [0.327, 0.337] | 2352–2398 | 295–299 |
| `default_12m_raw` | 0.750 [0.689, 0.762] | 0.138 [0.123, 0.227] | 2810–2814 | 60–64 |
| `default_12m_embedding` | 0.691 [0.648, 0.732] | 0.155 [0.133, 0.197] | 2810–2814 | 60–64 |
| `default_12m_combined` | 0.715 [0.671, 0.763] | 0.178 [0.121, 0.201] | 2810–2814 | 60–64 |
| `lora_default_12m` | 0.718 [0.708, 0.762] | 0.218 [0.199, 0.281] | 2810–2814 | 60–64 |
| `ltv_positive_raw` | 0.726 [0.712, 0.746] | 0.856 [0.845, 0.875] | 2352–2397 | 1679–1733 |
| `ltv_positive_embedding` | 0.829 [0.823, 0.832] | 0.915 [0.909, 0.917] | 2352–2397 | 1679–1733 |
| `ltv_positive_combined` | 0.829 [0.824, 0.834] | 0.915 [0.909, 0.917] | 2352–2397 | 1679–1733 |
| `lora_ltv_positive` | 0.845 [0.845, 0.875] | 0.924 [0.921, 0.940] | 2352–2397 | 1679–1733 |

`single_event_raw` equals `history_window_raw`, because the raw features are
computed from the history for both arms. The JSON also has `aml_*` rows: with
1–2 evaluation positives per seed they are diagnostics, not results.

What the table supports, as per-seed differences in ROC-AUC (seeds 42, 43, 44):

- **History adds to the frozen embedding on this fraud task.** The
  history-window embedding beats its single-event control at every seed
  (+0.040, +0.061, +0.076) and beats the raw history features at every seed
  (+0.050, +0.021, +0.048). On pragmatiq's own labels the control scored
  above the history (0.987 against 0.961), so this is the first comparison
  here that can show what history adds.
- **The effect is small.** 0.641 is far from what one transaction gave away
  on the book's own labels, and below what hand-built history features give a
  gradient-boosted classifier on the same kind of task (0.84–0.85 in
  [`pragmatiq_fraud_separability.md`](../pragmatiq_fraud_separability.md)). That
  figure uses different rows, a different split, and a different classifier,
  so it bounds what the history holds rather than being a baseline for this
  table.
- **Entity-level fraud.** The embedding beats the raw features at every seed
  (+0.145, +0.182, +0.169). The raw arm has only amount statistics, counts,
  hour, and weekday.
- **LoRA does not help fraud.** `lora_fraud` is below the frozen embedding at
  every seed (−0.001, −0.055, −0.051).
- **Churn and positive LTV** repeat the earlier record: the embedding beats
  the raw features at every seed (churn +0.020, +0.047, +0.048; LTV +0.106,
  +0.111, +0.083) and LoRA adds a little more (churn +0.034, +0.023, +0.002;
  LTV +0.012, +0.022, +0.046).
- **Default in 12 months.** The raw features beat the embedding at every seed
  (+0.030, +0.041, +0.059 for raw), on 60–64 positives.

## What changed from the previous record

[`blueprint_v2_hybrid_lab_seeds.md`](blueprint_v2_hybrid_lab_seeds.md) ran
the same config at commit `b54c3eb`, before the fraud probe moved to the
spliced task. Besides the fraud labels, one thing differs: the 10,000 users
are sampled uniformly. Before, the sampler filled the sample with every user
who has pragmatiq's own fraud (its fraud rows are therefore not comparable
with this record's). The forecast probes are scored on different users, and
moved by up to 0.03 ROC-AUC at the median.

## Limitations

- **The splice matches donors on home country and median amount only.**
  A stolen day can still differ from the victim in merchants, times, and
  categories, which is what the history-window probe is meant to detect;
  it can also differ in ways a real thief's spending would not.
- **28% of intended victims found no matched donor** in the donor search's
  limits (5 days, 200 donors per day). They stay on their side as unlabelled
  users, so each side has fewer victims than `victim_fraction` asks for.
- **Fraud is a fifth of the history-window rows** (618–640 of 2,885–3,067),
  far above any natural rate, so its PR-AUC is not comparable to one.
- **The raw arm uses only numbers**, no categorical fields, so it is a weak
  baseline. The tuned tabular baselines (`evaluation.tabular_baselines`)
  were added after this sweep.
- **The tokens cut fields.** Each event was capped at 24 key-value tokens in
  the adapter's order, so a long merchant name crowded out `amount` or `mcc`
  in 21–28% of transaction events, and `mcc` was binned as a number. Both
  are fixed after this sweep (CHANGELOG).
- **Peak host memory was not measured** in this sweep; the previous record
  measured 48.1 GB at the same sample size.
- Three pretraining seeds on one book at one size, and no multi-GPU run.
