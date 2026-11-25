# NVIDIA TFM Blueprint v1, step by step

Start here if you are new to transaction foundation models. These five
notebooks follow the public
[NVIDIA TFM Blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)
(v1), one notebook for each of its five, on the public IBM TabFormer dataset,
rebuilt on this kit. Each notebook ends with a "What changed from v1" box: the
fixes the kit made, such as the label shift v1's training was missing and
scoring each transaction with the customer's recent history, and why.

The canonical workflow is the recipe
[`blueprint_v1`](../../recipes/blueprint_v1); the notebooks
explain it and call the same `src/tfm` code.

## Run

From the repository root:

```bash
uv sync --extra native --extra dev --extra notebooks
uv run jupyter notebook notebooks/blueprint_v1/
```

Each notebook runs on CPU in about a minute on a small generated book, with
no download. Read them in order; each also runs on its own.

Optional extras add v1's tools: `--extra baselines` for its classifier,
`--extra umap` for its embedding plot, and `--extra huggingface` for the Llama
backbone. Without them the notebooks use logistic regression, PCA, and the
kit's eager decoder.

To see real TabFormer rows, build the splits once and set the root; the
notebooks then load 24 TabFormer users with the recipe's `config.small.yaml`:

```bash
uv run tfm prepare-tabformer --root ~/data/TabFormer   # after downloading transactions.tgz
export TFM_TABFORMER_ROOT=~/data/TabFormer
```

## Series

| Notebook | You learn | v1 counterpart |
|---|---|---|
| [01 · TabFormer, the temporal split, and a raw baseline](01_dataset_baseline.ipynb) | What a transaction and its label look like; v1's date split; one row per transaction, different customers per side; the baseline to beat; why average precision | `01_dataset_baseline` |
| [02 · Turning transactions into tokens](02_seq_preproc_tokenization.ipynb) | 11 field tokens per transaction and why `CUST` is dropped; the history row behind each scored transaction; pretraining windows | `02_seq_preproc_tokenization` |
| [03 · Pretraining by next-token prediction](03_foundation_model_training.ipynb) | Composing `TrxFM`; declared positions; a training loop; reproducing v1's unshifted-label bug | `03_foundation_model_training` |
| [04 · Embeddings from the pretrained model](04_inference_embedding_extraction.ipynb) | Each transaction embedded with its history, beside v1's transaction-alone embedding | `04_inference_embedding_extraction` |
| [05 · Do the embeddings help a fraud model?](05_xgboost_fraud_detection.ipynb) | Raw, raw plus either embedding, and hand-built history features; spread over seeds; results on the real book | `05_xgboost_fraud_detection` |

## Full-size runs

The full-size pretraining and fraud evaluation are one command on one GPU,
`uv run tfm report --recipe blueprint_v1 --device cuda` (the recipe's
`config.yaml`). The [recipe README](../../recipes/blueprint_v1/README.md)
gives its hardware, recorded times, and the multi-GPU variant.

Next series: [pragmatiq](../pragmatiq), which takes a multi-source book and
two architectures apart slot by slot.
