# A flat BERT-style encoder against Blueprint v1 on TabFormer (seed 42)

Does a bidirectional encoder trained by masked field modelling, on v1's own
tokens, give TabFormer a history embedding that adds to raw features, where
v1's causal decoder's does not reliably
([`blueprint_v1_seeds`](blueprint_v1_seeds.md)) and `pragma_like`'s does not
at all ([`pragma_like_tabformer_seed42`](pragma_like_tabformer_seed42.md))?
[flat-encoder guide](../../recipes/blueprint_v1/README.md)'s encoder
differs from v1 only in its attention mask, objective, and readout. One seed,
an exploration: not a quality, fraud, or production claim.

| field | value |
|---|---|
| config | [`benchmarks/ablations/flat_encoder_tabformer.yaml`](../ablations/flat_encoder_tabformer.yaml), seed 42, with `evaluation.history_window.max_rows: 250000` |
| code | commit `124ab93` plus the uncommitted `flat_encoder` composition (flat-encoder guide) and `features.exclude_keys` |
| data | IBM TabFormer through `tfm prepare-tabformer`, v1's precomputed date split, all 2,000 users, 128-event probe histories, 341-event pretraining windows |
| rows | the rows of `pragma_like_tabformer_seed42`: train side, every training-window fraud with 5 negatives each; evaluation side, a seeded sample of 250,000 test rows, **277 frauds**. The raw and hand-built arms score identically in all three runs |
| encoder | 8 bidirectional blocks, d=512, 8 heads, GELU 2048, rotary positions by token index (base 5e5); 28.5M parameters. v1's 11 field tokens per event. Masked fields: 15% of field tokens and 10% of whole events, 80 / 10 / 10 `<mask>` / random same-field / kept, cross-entropy within the field. Readout: mean of the last (scored) event's field tokens |
| training | v1's schedule: 3,000 steps of 16 windows, AdamW at 2e-4, cosine, bf16. Masked-field loss 3.67 to 0.80 (last 100 steps), no non-finite gradients |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3; 16.1 GiB peak GPU memory |
| wall clock | prepare 302 s, train 4,008 s, evaluate 3,241 s |
| raw output | [`flat_encoder_tabformer_seed42.json`](flat_encoder_tabformer_seed42.json) |

## Results

`history_window` PR-AUC on the same 250,000 rows; lift over `xgboost_raw`.
The `pragma_like` and v1 columns are from
[`pragma_like_tabformer_seed42`](pragma_like_tabformer_seed42.md).

| arm | flat encoder | `pragma_like` | v1 (decoder) |
|---|---|---|---|
| `xgboost_raw` | 0.0414 | 0.0414 | 0.0414 |
| raw + history embedding | **0.2246 (+443%)** | 0.0395 (−5%) | 0.0566 (+37%) |
| raw + transaction-alone embedding | 0.1028 (+148%) | 0.0422 (+2%) | 0.0336 (−19%) |
| `tabular_xgboost` (hand-built) | 0.0650 (+57%) | 0.0650 (+57%) | 0.0650 (+57%) |
| hand-built + history embedding | **0.2708 (+554%)** | 0.0704 (+70%) | 0.0691 (+67%) |

ROC-AUC: raw 0.916; raw + history embedding 0.995 (encoder), 0.917
(`pragma_like`), 0.965 (v1); hand-built 0.975; hand-built + embedding 0.996
(encoder), 0.980 and 0.981.

## Reading

- **The encoder's history embedding lifts raw features fivefold** on these
  rows, and raw + embedding beats the hand-built features by 3.5×. It is the
  first TabFormer embedding in the kit to add more than the hand-built
  features do.
- **It also adds to the hand-built features** (+317% over `tabular_xgboost`),
  where v1's and `pragma_like`'s added 6–8%.
- **Context matters.** The same encoder reading the transaction alone lifts
  raw by 148%; with its 128-event history, by 443%. The readout is the scored
  transaction's tokens in both, so the difference is what the history adds.
- **The change from v1 is the mask, objective, and readout together.** Tokens,
  rows, windows, schedule, and probe are v1's; which of the three carries the
  lift is not separated here (the `mean` and `usr_token` readouts, or a
  decoder read out at the last event, would separate it).
- **Leakage review.** Each probe row is the user's events up to and including
  the scored one (`build_event_history_rows`), so bidirectional attention
  reaches no later event. Pretraining windows end at the train cutoff, the
  evaluation users are disjoint from the probe's training users, and no
  label-derived field is a token. A gap this large still needs the seeds
  below before it is a result.

## Limitations

One seed and 277 evaluation frauds: single arms at this support moved by
tens of percent between v1 runs. The comparison columns use a retrain of v1's
seed-42 config (see `pragma_like_tabformer_seed42`). The 250,000-row sample
is not the full test set of `blueprint_v1_seeds`. The encoder was not tuned:
it uses v1's size and schedule, and LoRA was off. Next: seeds 43 and 44, and
every test row (the encoder's flat tokens fit as v1's do).
