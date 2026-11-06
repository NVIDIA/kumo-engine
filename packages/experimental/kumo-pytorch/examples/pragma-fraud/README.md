# Pre-train a transaction model, then fine-tune it to flag fraud

This example follows a common pattern with banking data: there are plenty of
transaction histories, but only a few of them are labeled as fraud. So we first
pre-train a model on all the histories without labels, then fine-tune it on the
small labeled set.

The model follows the design of PRAGMA, a published foundation model for
banking transactions. Each of a user's transactions is turned into a short run
of tokens, one per field (amount, merchant, country and so on), and
pre-training teaches the model to fill in values that have been masked
out. For fine-tuning, we keep the pre-trained model frozen, add a small head on
top, and train it to predict whether the user will have a fraudulent
transaction in the next seven days. The fine-tuned model is then saved, loaded
back, and used to score a test set.

The data comes from the banking generator in `kumo.pytorch.datasets`: about 400
users and 40,000 transactions, with some account takeovers mixed in. It runs on
the DiskGraph engine and takes under a minute on a laptop CPU.

## Running it

See [running the examples](../README.md#running-them) for what to install. Then
open `walkthrough.ipynb` from this folder, or run it headless from
`packages/experimental/kumo-pytorch` with the test environment active:

```bash
python -m pytest -m notebook -k pragma-fraud
```

## Files

- [`walkthrough.ipynb`](walkthrough.ipynb) is the notebook.
- [`transaction_model.py`](transaction_model.py) has the transaction model, the
  pre-training loss, the fraud head, and a helper the notebook uses to write
  JSON.

The notebook writes these, and they're committed so you can read them without
running it:

- [`contract.json`](contract.json) is the fine-tuning contract, as the saved
  model stores it. The pre-training contract is the same apart from its target.
- [`output_schema.json`](output_schema.json) describes what the model receives
  for each input: encoders, vocabularies, buckets and token IDs.
- [`id_spaces.json`](id_spaces.json) lists the shared token ID space, its size
  and its reserved IDs.
