# Fraud separability on a pragmatiq book

Evidence for [dataset strategy](../../docs/reference/datasets.md).
How much of a fraud label one transaction gives away, for the book's own fraud
labels and for the spliced identity-theft task. No model from this repository
is involved: one gradient-boosted classifier is fit on three feature sets.

| field | value |
|---|---|
| script | [`benchmarks/fraud_separability.py`](../fraud_separability.py) |
| book | `synth_100k_gen2_m31_seed0` (pragmatiq 1.1.0, unmodified; world config `recipes/blueprint_v2_hybrid/pragmatiq_world.yaml`) |
| sample | native: every user with a fraud label plus 8,000 others; splice: 8,000 users without the book's own fraud, half victims and half donors |
| split | user-disjoint, 30% of users held out |
| classifier | XGBoost 3.4.1, 300 trees, depth 6, learning rate 0.1, subsample and column sample 0.8, categorical splits; scikit-learn 1.9.0 metrics |
| seeds | native: 0; splice: 0, 1, 2 (the seed changes the users, victims, donors, days, and held-out users) |
| raw output | [`pragmatiq_fraud_separability.json`](pragmatiq_fraud_separability.json) |

Feature sets:

- **event**: the labelled transaction's own fields (amount, hour, weekday, MCC,
  channel, country, currency, type).
- **event + profile**: plus the user's home country and whether the
  transaction's country and currency match it.
- **event + profile + history**: plus features that need the user's own past,
  all computed strictly before the transaction: time since the previous
  transaction, transactions in the previous hour and day, amount against the
  user's prior mean, prior payments at the same merchant, and whether a new
  device or an app session from abroad appears in the two hours before.

Reproduce:

```bash
B=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
uv run python benchmarks/fraud_separability.py --book $B --task native
uv run python benchmarks/fraud_separability.py --book $B --task splice --donors random --seed 0
uv run python benchmarks/fraud_separability.py --book $B --task splice --donors matched --seed 0
```

## Results

ROC-AUC on the held-out users; for the splice, the range over three seeds.
0.5 is chance.

| task | event | event + profile | event + profile + history | held-out rows (positives) |
|---|---|---|---|---|
| book's own fraud labels | 1.000 | 1.000 | 1.000 | 8,595 (1,035) |
| splice, random donors | 0.495–0.506 | 0.861–0.863 | 0.935–0.942 | 7,749–8,014 (1,951–2,018) |
| splice, matched donors | 0.497–0.501 | 0.497–0.503 | 0.840–0.850 | 7,868–7,951 (1,903–1,956) |

PR-AUC on the matched splice is 0.244–0.258 without history, which is the
positive share of the held-out rows, and 0.639–0.656 with it.

What the table supports:

- **The book's own fraud is identified from one transaction.** Every labelled
  fraud event is an online payment abroad; 3% of the labelled negatives are
  abroad and 30% are online. No history-based comparison can be read from it.
- **Spliced fraud with matched donors is not.** A thief's transaction is an
  ordinary transaction of a user from the same country with similar amounts,
  so the event and the profile score at chance. History recovers it: 80% of
  stolen payments are at a merchant the victim has never paid, against 29% of
  the victim's own payments.
- **Random donors leak through the profile.** A donor from another country
  pays abroad in another currency (66–67% of stolen payments are abroad).
  Matching the donor's home country removes that shortcut.

## Limitations

- One book and one classifier. The history features are a small hand-written
  set; a stronger baseline would score higher, and that is the point of a
  tuned baseline in the recipe record, not of this check.
- The rows are the victim's card transactions in the week before and the day
  of the theft, so a quarter of them are stolen. That base rate is far above a
  real card book's; the recipe task sets it through the negative window
  (dataset strategy).
- Donor matching uses the median amount and the home country only. It does
  not match age, income, or archetype, which the book's profiles also carry.
