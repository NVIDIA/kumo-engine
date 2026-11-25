# v1's flat decoder on v2's rows, spliced identity theft (one GPU, three seeds)

Milestone 2, criterion 6, rerun with the fraud probe on the spliced
identity-theft task
([ADR 0015](../../docs/architecture/0015-dataset-strategy-and-spliced-fraud.md)).
v1's design (one flat token stream, a causal decoder with no positional
encoding, next-token prediction) is pretrained on the same book, rows,
schedule, and seeds as
[`blueprint_v2_hybrid_lab_splice_seeds.md`](archive/blueprint_v2_hybrid_lab_splice_seeds.md),
then scored with the same probes. It is not a quality claim for either model.

| field | value |
|---|---|
| config | `recipes/blueprint_v2_hybrid/config.lab_v1_flat.yaml` at commit `2acd68b`, no overrides; seeds 42, 43, 44 |
| code | commit `2acd68b`. Source edits made while the sweep ran were not loaded: the sweep runs every seed in one process, which imported the package before them |
| data | as the hybrid record: `synth_100k_gen2_m31_seed0`, 10,000 users sampled uniformly, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01; spliced identity theft with matched donors |
| tokens | `flat_key_value`: `[USR]` and profile tokens, then per event `[EVT]`, weekday, hour, log-gap bucket, and key-value tokens. Rows are capped at 2,048 tokens, keeping the profile and the most recent whole events |
| model | flat causal decoder, 8 layers, hidden 256, 8 heads, no positional encoding, SDPA; 6,603,009 parameters (hybrid: 8,352,768) |
| objective | next token |
| training | as the hybrid: eager, bf16, batch 32, 3,000 steps, cosine schedule with 150 warmup, AdamW, peak LR 5e-4 |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare 255–263 s, train 1,555–1,557 s (about 0.52 s per step), evaluate 1,639–1,642 s; 2 h 52 min for three seeds |
| memory | 7.56 GiB peak GPU; peak host memory not measured in this run |
| loss | 6.50–6.55 at step 0, 0.39–0.44 at step 2,999 (per token; not comparable with the hybrid's summed multi-objective loss) |
| probes | as the hybrid record |
| raw output | [`blueprint_v2_hybrid_lab_v1_flat_splice_seeds.json`](blueprint_v2_hybrid_lab_v1_flat_splice_seeds.json), from `benchmarks/seed_range.py` |

Reproduce (at commit `2acd68b`):

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v2_hybrid/config.lab_v1_flat.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v2_hybrid/lab_v1_flat_splice_seeds \
  --summary benchmarks/results/blueprint_v2_hybrid_lab_v1_flat_splice_seeds.json --device cuda
```

## Same rows

At every seed, every `raw` arm's metrics, row count, and positive count are
identical to the hybrid run's at that seed. The two sweeps scored the same
users, labels, and splices.

## Results

ROC-AUC, median [min, max] over three seeds. The last column is flat minus
hybrid at each seed (42, 43, 44). The `raw` arms are the same for both models
and are in the hybrid record.

| arm | flat (v1) | hybrid (v2) | flat − hybrid per seed |
|---|---|---|---|
| `embedding` (fraud, one row per entity) | 0.681 [0.673, 0.691] | 0.724 [0.714, 0.740] | −0.023, −0.051, −0.059 |
| `lora_fraud` | 0.664 [0.661, 0.671] | 0.689 [0.669, 0.712] | −0.048, −0.007, −0.018 |
| `history_window_embedding` | 0.601 [0.589, 0.606] | 0.641 [0.610, 0.663] | −0.021, −0.034, −0.061 |
| `single_event_embedding` (control) | 0.538 [0.525, 0.544] | 0.580 [0.570, 0.587] | −0.032, −0.036, −0.061 |
| `churn_6m_embedding` | 0.715 [0.715, 0.737] | 0.740 [0.737, 0.764] | +0.000, −0.025, −0.049 |
| `lora_churn_6m` | 0.755 [0.702, 0.771] | 0.766 [0.762, 0.770] | +0.001, −0.007, −0.064 |
| `ltv_positive_embedding` | 0.810 [0.802, 0.811] | 0.829 [0.823, 0.832] | −0.030, −0.013, −0.019 |
| `lora_ltv_positive` | 0.842 [0.841, 0.851] | 0.845 [0.845, 0.875] | −0.004, −0.003, −0.025 |
| `default_12m_embedding` | 0.679 [0.664, 0.683] | 0.691 [0.648, 0.732] | −0.049, +0.016, −0.012 |
| `lora_default_12m` | 0.719 [0.666, 0.733] | 0.718 [0.708, 0.762] | +0.015, −0.042, −0.043 |

PR-AUC, precision@k, recall@k, calibration, and the `combined` arms are in
the JSON.

What the table supports, on this book at this size:

- **Fraud.** The hybrid leads on every fraud arm at every seed: entity-level
  embedding by 0.023–0.059, the history window by 0.021–0.061, and LoRA by
  0.007–0.048. On pragmatiq's own labels (the superseded
  [`blueprint_v2_hybrid_lab_v1_flat_seeds.md`](archive/blueprint_v2_hybrid_lab_v1_flat_seeds.md)),
  the flat model led on the history window; that task was identifiable from
  one event.
- **History over the single event.** For both models the history-window
  embedding beats its single-event control at every seed (flat +0.051,
  +0.062, +0.076; hybrid +0.040, +0.061, +0.076). Against the raw history
  features the flat model is +0.029, −0.014, −0.013; the hybrid is ahead at
  every seed (+0.050, +0.021, +0.048).
- **Positive LTV.** The hybrid's frozen embedding leads at every seed, by
  0.013–0.030, as in the earlier record.
- **Churn and default in 12 months.** Mixed: the hybrid leads at two seeds
  and ties or trails at the third. Default has 60–64 positives.

So on this task the hybrid beats v1's design on fraud and positive LTV, and
neither wins on churn or default.

## Limitations

- **The flat model sees less history.** A 2,048-token row holds a median of
  81 of the 128 events the hybrid reads (measured on 300 users of this book
  for the earlier record).
- **Both models' tokens cut fields.** At commit `2acd68b` each event's
  key-value tokens were capped at 24 and emitted in the adapter's order, so a
  long merchant name crowded out `amount` and `mcc` in 21–28% of transaction
  events. The tokenizers now emit text fields last (CHANGELOG); both records
  predate that change.
- **Different sizes.** 6.6M against 8.4M parameters. Neither model was tuned.
- **The raw arm is weak** (amount statistics, count, hour, weekday). The tuned
  tabular baselines of `docs/concepts/evaluation-protocol.md` postdate this
  sweep.
- One book, one size, three seeds, one GPU. Every limitation of the hybrid
  record applies.
