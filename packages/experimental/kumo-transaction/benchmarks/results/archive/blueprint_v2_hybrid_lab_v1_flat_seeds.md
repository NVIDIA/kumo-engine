# v1's flat decoder on v2's rows (one GPU, three seeds)

Milestone 2, criterion 6: the v1 baseline *model* on the same rows as the
hybrid. v1's design (one flat token stream, a causal decoder with no
positional encoding, next-token prediction) is pretrained on the same
pragmatiq book, split, schedule, and seeds as
[`blueprint_v2_hybrid_lab_seeds.md`](blueprint_v2_hybrid_lab_seeds.md), then
scored with the same probes. It is not a quality claim for either model.

Superseded by
[`blueprint_v2_hybrid_lab_v1_flat_splice_seeds.md`](../blueprint_v2_hybrid_lab_v1_flat_splice_seeds.md),
the same comparison with the fraud probe on spliced identity theft.

| field | value |
|---|---|
| config | `recipes/blueprint_v2_hybrid/config.lab_v1_flat.yaml`, no overrides; seeds 42, 43, 44 |
| code | commit `2af4905`. `0f2f5ff` was committed during the sweep and changes only documentation, notebooks, and one error message; the summary JSON's `commit` field was corrected to the commit the sweep started from |
| data | as the hybrid record: `synth_100k_gen2_m31_seed0`, 10,000 users sampled, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01 |
| tokens | `flat_key_value`: `[USR]` and profile tokens, then per event `[EVT]`, weekday, hour, log-gap bucket, and key-value tokens (about 25 tokens per event). Rows are capped at 2,048 tokens, keeping the profile and the most recent whole events |
| model | flat causal decoder, 8 layers, hidden 256, 8 heads, no positional encoding, SDPA; 6,603,009 parameters (hybrid: 8,352,768) |
| objective | next token |
| training | as the hybrid: eager, bf16, batch 32, 3,000 steps, cosine schedule with 150 warmup, AdamW, peak LR 5e-4 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare 243–252 s, train 1,555–1,558 s (about 0.52 s per step), evaluate 1,662–1,665 s; about 58 min in total, 2 h 54 min for three seeds |
| memory | 7.56 GiB peak GPU; 44.6 GB peak host (resident set of the sweep process) |
| loss | 6.46–6.54 at step 0, 0.37–0.41 at step 2,999 (per token; not comparable with the hybrid's summed multi-objective loss) |
| probes | as the hybrid record |
| raw output | [`blueprint_v2_hybrid_lab_v1_flat_seeds.json`](blueprint_v2_hybrid_lab_v1_flat_seeds.json), from `benchmarks/seed_range.py` |

Reproduce:

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v2_hybrid/config.lab_v1_flat.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v2_hybrid/lab_v1_flat_seeds \
  --summary benchmarks/results/blueprint_v2_hybrid_lab_v1_flat_seeds.json --device cuda
```

## Same rows

At every seed, every arm's evaluation row count and positive count equal the
hybrid run's at that seed, and every `raw` arm's metrics are identical to the
hybrid's. The seed draws the probe's row sample, so counts differ between
seeds (churn: 2,363–2,369 rows), not between models.

## Results

The fraud rows are scored on pragmatiq's own fraud labels, which one
transaction identifies
([ADR 0015](../../../docs/architecture/0015-dataset-strategy-and-spliced-fraud.md)).
They are superseded by the spliced identity-theft task and are not a fraud
result.

ROC-AUC, median [min, max] over three seeds. The last column is flat minus
hybrid at each seed (42, 43, 44). The `raw` arms are the same for both models
and are in the hybrid record.

| arm | flat (v1) | hybrid (v2) | flat − hybrid per seed |
|---|---|---|---|
| `embedding` (fraud, one row per entity) | 0.700 [0.656, 0.740] | 0.686 [0.645, 0.717] | −0.030, +0.055, +0.022 |
| `lora_fraud` | 0.670 [0.654, 0.690] | 0.671 [0.623, 0.690] | 0.000, −0.001, +0.032 |
| `history_window_embedding` | 0.981 [0.979, 0.982] | 0.961 [0.925, 0.963] | +0.057, +0.018, +0.017 |
| `single_event_embedding` (control) | 0.976 [0.970, 0.979] | 0.987 [0.980, 0.989] | −0.009, −0.008, −0.013 |
| `churn_6m_embedding` | 0.732 [0.727, 0.744] | 0.733 [0.722, 0.746] | −0.003, −0.001, +0.005 |
| `lora_churn_6m` | 0.756 [0.755, 0.772] | 0.775 [0.750, 0.779] | −0.004, −0.023, +0.004 |
| `ltv_positive_embedding` | 0.815 [0.792, 0.822] | 0.829 [0.811, 0.841] | −0.019, −0.019, −0.014 |
| `lora_ltv_positive` | 0.837 [0.833, 0.852] | 0.870 [0.837, 0.870] | −0.018, 0.000, −0.037 |
| `default_12m_embedding` | 0.633 [0.610, 0.649] | 0.688 [0.688, 0.711] | −0.101, −0.039, −0.055 |
| `lora_default_12m` | 0.675 [0.635, 0.697] | 0.689 [0.673, 0.715] | −0.054, −0.040, +0.024 |

Median `combined` arms are within 0.025 of median `embedding` arms for both models; they and
PR-AUC, precision@k, recall@k, and calibration are in the JSON.

What the table supports, on this book at this size:

- **Churn.** No difference: the frozen arms agree within 0.005 at every seed.
- **Positive LTV.** The hybrid's frozen embedding leads at every seed, by
  0.014–0.019. LoRA leads at two seeds and ties at one.
- **Default in 12 months.** The hybrid's frozen embedding leads at every seed,
  by 0.039–0.101, and the seed ranges do not overlap. At 62–70 positives this
  is the largest gap in the table and the least certain.
- **Fraud history windows.** The flat model leads on the full history
  (0.981 against 0.961, ranges disjoint) and trails on the single-event control
  (0.976 against 0.987). For the flat model the full history scores above the
  single event at every seed (+0.003 to +0.011); for the hybrid it scores
  below. Because this book's fraud can be identified from one event (see the
  hybrid record), this shows how each model's readout carries the scored
  event, not what history adds.
- **Entity-level fraud** (33 positives): no conclusion.

So on this synthetic book the hybrid does not beat v1's design across the
board. It leads on LTV and default, ties on churn, and trails on the fraud
history windows.

## Limitations

- **The flat model sees less history.** At about 25 tokens per event, a
  2,048-token row holds a median of 81 of the 128 events the hybrid reads
  (10th–90th percentile 69–83, measured on 300 users of this book); 87% of
  rows are cut. v1 used 4,096 tokens; halving it fits one GB10.
- **Different sizes and steps per token.** 6.6M against 8.4M parameters, and
  the same 3,000 steps of 32 rows means the flat model trains on more tokens
  per step but fewer events per row. Neither model was tuned.
- **The book's fraud is identifiable from a single event** (hybrid record,
  first limitation), so neither model's history-window score measures what
  history adds.
- One book, one size, three seeds, one GPU. Every limitation of the hybrid
  record applies.
