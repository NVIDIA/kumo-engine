# Product recommendation with a graph neural network

This example recommends products to the customers of a retail store. Every
week, each customer gets a list of 20 candidate products, and the model ranks
them so that the ones they'll actually buy in the next seven days come first.

To score a customer, the model looks at their recent purchases and the products
in them, as a small two-hop graph. The candidate products, along with whether
each was bought, come from a table you supply. The model is a small graph
neural network built with PyTorch Geometric, and the notebook checks that it
beats simply recommending the most popular products.

The data comes from the retail generator in `kumo.pytorch.datasets`, which makes
synthetic data modeled on the public H&M fashion-retail dataset, at its
smallest size. It runs on the in-memory engine for now, because the
DiskGraph engine doesn't support candidate lists yet, and takes under a minute
on a laptop CPU.

## Running it

See [running the examples](../README.md#running-them) for what to install. Then
open `walkthrough.ipynb` from this folder, or run it headless from
`packages/experimental/kumo-pytorch` with the test environment active:

```bash
python -m pytest -m notebook -k hm-recommendation
```

## Files

- [`walkthrough.ipynb`](walkthrough.ipynb) is the notebook.
- [`recommender.py`](recommender.py) has the graph neural network, the recall
  metric, and a helper the notebook uses to write JSON.

The notebook writes these, and they're committed so you can read them without
running it:

- [`contract.json`](contract.json) is the full contract, as the saved model
  stores it.
- [`output_schema.json`](output_schema.json) describes what the model receives
  for each input: types, vocabularies and fitted statistics.
