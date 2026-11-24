# Blueprint v2 hybrid at lab size (one GPU, three seeds)

Milestone 2, criterion 7. An 8.4M-parameter `blueprint_v2_hybrid` model
pretrained on a synthetic pragmatiq book, then scored with the frozen probes,
the history-window probe (ADR 0013), and LoRA fine-tuning. This shows the
wiring at a size where numbers stop being noise. It is not a quality claim.

Superseded by
[`blueprint_v2_hybrid_lab_kvt_seeds.md`](../blueprint_v2_hybrid_lab_kvt_seeds.md),
by way of the spliced rerun
[`blueprint_v2_hybrid_lab_splice_seeds.md`](blueprint_v2_hybrid_lab_splice_seeds.md).

| field | value |
|---|---|
| config | `recipes/blueprint_v2_hybrid/config.lab.yaml`, no overrides; seeds 42, 43, 44 |
| code | commit `b54c3eb`, clean working tree |
| data | pragmatiq Generator v2 book from `recipes/blueprint_v2_hybrid/pragmatiq_world.yaml`: 100,000 users, seed 0, 31 months (2023-01-01 to 2025-08-01), 128.5M events (`synth_100k_gen2_m31_seed0`); 10,000 users sampled, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01 |
| model | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA; 8,352,768 parameters |
| objective | next-event fields plus token/event/key MLM |
| training | eager, bf16, batch 32, 3,000 steps, cosine schedule with 150 warmup, AdamW, peak LR 5e-4 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare 290–298 s, train 765–791 s (about 0.26 s per step), evaluate 774–830 s; about 31 min in total, 1 h 33 min for three seeds |
| memory | 4.86–4.92 GiB peak GPU; 48.1 GB peak host (resident set of the sweep process) |
| loss | 51.9–60.8 at step 0, 3.99–4.01 at step 2,999 |
| probes | logistic regression; fraud split entity-disjoint with per-side label windows (ADR 0013); LoRA rank 8 on qkv and out, 300 steps, LR 1e-3, linear head |
| raw output | [`blueprint_v2_hybrid_lab_seeds.json`](blueprint_v2_hybrid_lab_seeds.json), from `benchmarks/seed_range.py` |

Reproduce (build the book first; the command is in the header of
`pragmatiq_world.yaml`):

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v2_hybrid/config.lab.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v2_hybrid/lab_seeds \
  --summary benchmarks/results/blueprint_v2_hybrid_lab_seeds.json --device cuda
```

## Results

The fraud rows (`raw`, `embedding`, `combined`, `lora_fraud`, and the
`history_window_*` and `single_event_*` arms) are scored on pragmatiq's own
fraud labels, which one transaction identifies
([ADR 0015](../../../docs/architecture/0015-dataset-strategy-and-spliced-fraud.md)).
They are superseded by the spliced identity-theft task
([`blueprint_v2_hybrid_lab_splice_seeds.md`](blueprint_v2_hybrid_lab_splice_seeds.md))
and are not a fraud result.

Median [min, max] over three seeds. The seed changes the pretraining
initialization and order and the probe's row sample. It does not change the
book.

| arm | ROC-AUC | PR-AUC | eval rows | eval positives |
|---|---|---|---|---|
| `raw` (fraud, one row per entity) | 0.645 [0.639, 0.656] | 0.021 [0.019, 0.023] | 3000 | 33 |
| `embedding` | 0.686 [0.645, 0.717] | 0.069 [0.060, 0.121] | 3000 | 33 |
| `combined` | 0.678 [0.643, 0.717] | 0.059 [0.058, 0.115] | 3000 | 33 |
| `lora_fraud` | 0.671 [0.623, 0.690] | 0.052 [0.018, 0.062] | 3000 | 33 |
| `history_window_raw` | 0.631 [0.593, 0.646] | 0.275 [0.246, 0.276] | 4000 | 482–565 |
| `history_window_embedding` | 0.961 [0.925, 0.963] | 0.845 [0.790, 0.870] | 4000 | 482–565 |
| `history_window_combined` | 0.962 [0.924, 0.964] | 0.847 [0.785, 0.872] | 4000 | 482–565 |
| `single_event_embedding` (control) | 0.987 [0.980, 0.989] | 0.937 [0.914, 0.943] | 4000 | 482–565 |
| `single_event_combined` (control) | 0.987 [0.981, 0.988] | 0.937 [0.917, 0.942] | 4000 | 482–565 |
| `churn_6m_raw` | 0.707 [0.681, 0.713] | 0.272 [0.252, 0.284] | 2368 | 285–293 |
| `churn_6m_embedding` | 0.733 [0.722, 0.746] | 0.297 [0.292, 0.313] | 2368 | 285–293 |
| `churn_6m_combined` | 0.734 [0.724, 0.745] | 0.307 [0.307, 0.308] | 2368 | 285–293 |
| `lora_churn_6m` | 0.775 [0.750, 0.779] | 0.327 [0.323, 0.352] | 2368 | 285–293 |
| `default_12m_raw` | 0.719 [0.711, 0.736] | 0.093 [0.077, 0.106] | 2803 | 62–70 |
| `default_12m_embedding` | 0.688 [0.688, 0.711] | 0.143 [0.100, 0.144] | 2803 | 62–70 |
| `default_12m_combined` | 0.700 [0.690, 0.718] | 0.136 [0.106, 0.144] | 2803 | 62–70 |
| `lora_default_12m` | 0.689 [0.673, 0.715] | 0.172 [0.065, 0.229] | 2803 | 62–70 |
| `ltv_positive_raw` | 0.733 [0.713, 0.753] | 0.861 [0.847, 0.863] | 2368 | 1680–1689 |
| `ltv_positive_embedding` | 0.829 [0.811, 0.841] | 0.909 [0.898, 0.921] | 2368 | 1680–1689 |
| `ltv_positive_combined` | 0.829 [0.812, 0.839] | 0.909 [0.899, 0.920] | 2368 | 1680–1689 |
| `lora_ltv_positive` | 0.870 [0.837, 0.870] | 0.933 [0.915, 0.937] | 2368 | 1680–1689 |

`single_event_raw` equals `history_window_raw`, because the raw features are
computed the same way for both. AML has 2 evaluation positives per seed, so
its rows in the JSON are diagnostics, not results.

What the table supports:

- **Churn and positive LTV.** The frozen embedding beats the raw features at
  every seed, by +0.026 and +0.096 ROC-AUC at the median. LoRA beats the
  frozen embedding at every seed, by another +0.042 and +0.041. These are the
  tasks with enough positives to read.
- **Default in 12 months** is scored for the first time, because this book's
  default labels are dated after the train cutoff. The raw features lead on
  ROC-AUC (0.719 against 0.688) and the embedding leads on PR-AUC (0.143
  against 0.093). On 62–70 positives, with seed ranges that overlap, neither
  is a result.
- **Entity-level fraud.** The embedding leads the raw features at the median,
  but on 33 positives the seed ranges overlap. No conclusion.
- **Fraud history windows.** The single-event control scores above the full
  history (0.987 against 0.961). This book's fraud can be identified from one
  event, so the comparison cannot show what history adds (see below).

## What changed from the previous record

The record this replaces ran on commit `8a2de33` plus uncommitted changes, on
pragmatiq's default 25-month book (`synth_100k_gen2_seed0`). Three things
differ. The code is committed. Fine-tuning now reshuffles its rows every
epoch: before, rows arrived grouped by class, every minibatch held one class,
and the LoRA arms scored at chance (0.50 on churn and LTV). The book runs 31
months, so `default_12m` is scored instead of skipped. The history-window
probe now hits its 4,000-row cap. Frozen-probe numbers moved by up to 0.05
ROC-AUC between the two books, which is within what a different book can
explain.

## Limitations

- **The book's fraud is identifiable from a single event.** pragmatiq injects
  fraud as account-takeover drains with distinctive merchants, amounts, and
  locations. In a sample of the previous book, fraud events were 100% country
  PL (5% of other events), 80% MCC 5969 (6%), and online; this book comes
  from the same generator. None of these fields is label-derived. On this
  book the history-window comparison measures field recognition, not what
  history adds. It needs a book whose fraud is defined by behaviour over
  time.
- **The raw arm uses only numbers** (amount statistics, counts, hour,
  weekday), no categorical fields. That makes it a weak baseline when fraud is
  categorical.
- **The user sample is enriched for fraud.** Stratified sampling keeps every
  user with a fraud label, so fraud events are 12–14% of history-window rows,
  far above the book's rate. PR-AUC on those arms is not comparable to a
  natural base rate.
- **Loader memory limits the sample.** The pragmatiq adapter keeps each
  sampled user's full event history in memory: 48 GB at 10,000 users. A
  30,000-user run was killed by the kernel's out-of-memory killer at 89 GB, so
  a larger sample needs a columnar loader first.
- Three pretraining seeds on one book at one size, and no multi-GPU run. The
  v1 baseline model on the same rows (roadmap criterion 6) is recorded in
  [`blueprint_v2_hybrid_lab_v1_flat_seeds.md`](blueprint_v2_hybrid_lab_v1_flat_seeds.md).
