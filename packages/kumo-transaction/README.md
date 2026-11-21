# Kumo Transaction (TxFM)

**Kumo Transaction**, or **TxFM**, brings transaction foundation models to Kumo Engine.

## Motivation

Payments, cards, and banking systems generate huge streams of transactions.
Teams use that data to predict things like fraud, customer churn, and spending
behavior. Today this usually means:

- **Hand-built features.** Analysts turn raw transaction history into
  aggregates (spend in the last 7 days, count of merchants, and so on), which
  is slow, specific to one task, and easy to get wrong.
- **One model per task.** Each new prediction problem repeats the same
  feature work from scratch.
- **Lost signal.** Summaries throw away the order, timing, and detail of the
  individual transactions.

## What problem it solves

A transaction foundation model learns directly from the raw sequence of
transactions and turns it into a compact, reusable representation (an
embedding). Kumo Transaction makes that practical in Kumo Engine:

- Learn from raw transaction history instead of hand-built features.
- Reuse one representation across many prediction tasks.
- Add the representation to the features you already use, to improve accuracy
  on tasks such as fraud detection.

