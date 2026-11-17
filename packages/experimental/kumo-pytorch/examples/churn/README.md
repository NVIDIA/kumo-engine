# Churn prediction from point-in-time features

This example predicts which customers of a retail store will stop buying. Every
week, for each customer, the model guesses whether they'll make no purchase at
all in the next seven days.

The model is a small network over hand-picked features: how much a customer
bought over the past few weeks, how long ago they last bought, how old their
recent purchases are, and the day of the week. The notebook declares each of
these features and how it's encoded, then trains the model and checks it for
leakage. It does that by comparing the model's score with the best score a model
could reach using only past data. Finally, it saves the model, scores a test
set, and serves one prediction from rows it passes in directly, the way an
online service would.

The data comes from the retail generator in `kumo.pytorch.datasets`, which makes
synthetic data modeled on the public H&M fashion-retail dataset. The example
uses its smallest size: 200 customers and about 8,000 purchases. It runs on the
DiskGraph engine and takes under a minute on a laptop CPU.

## Running it

See [running the examples](../README.md#running-them) for what to install. Then
open `walkthrough.ipynb` from this folder, or run it headless from
`packages/experimental/kumo-pytorch` with the test environment active:

```bash
python -m pytest -m notebook -k churn
```

## Files

- [`walkthrough.ipynb`](walkthrough.ipynb) is the notebook.
- [`churn_model.py`](churn_model.py) has the model, its label, and a helper the
  notebook uses to write JSON.

The notebook writes these, and they're committed so you can read them without
running it:

- [`contract.json`](contract.json) is the full contract, as the saved model
  stores it.
- [`output_schema.json`](output_schema.json) describes what the model receives
  for each input: types, vocabularies, buckets and fitted statistics.
- [`serving_request.json`](serving_request.json) is the request it serves,
  together with the rows passed in for it.
