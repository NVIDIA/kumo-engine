# Migration

## From the v1 Blueprint to v2

v1 is the notebook-oriented developer example tagged `v1.0.0` on branch `v1`.
v2 is a breaking redesign. **Results, checkpoints, and configs are not
compatible.**

### Environment

| v1 | v2 |
|---|---|
| NeMo Framework container + Jupyter | Python 3.12 + `uv sync --extra dev` |
| NeMo AutoModel pretraining | The same HuggingFace Llama, trained by the SDK loop; `training.backend: fsdp2` for several GPUs |
| RAPIDS cuDF tokenizer | `transaction_field_tokenizer` (CPU), or `rapids_field_tokenizer` with cuDF installed |
| XGBoost on GPU | `evaluation.tabular_baselines.device: cuda` |
| `Pipfile` / notebook `%pip` | `pyproject.toml` / `uv.lock` (`uv sync --extra notebooks`) |
| Git LFS `models/decoder-foundation-model/` | Not used |

### Notebook mapping

The recipe is [`recipes/blueprint_v1`](../../recipes/blueprint_v1), and its
notebooks keep v1's five steps. They run on CPU on generated data, or on 24
TabFormer users when `TFM_TABFORMER_ROOT` is set.

| v1 | v2 notebook | v2 command |
|---|---|---|
| `01_dataset_baseline.ipynb` | `notebooks/blueprint_v1/01_dataset_baseline.ipynb` | `tfm prepare-tabformer` |
| `02_seq_preproc_tokenization.ipynb` | `notebooks/blueprint_v1/02_seq_preproc_tokenization.ipynb` | `tfm train --recipe blueprint_v1` (tokenizes first) |
| `03_foundation_model_training.ipynb` + `scripts/train_decoder_model.py` | `notebooks/blueprint_v1/03_foundation_model_training.ipynb` | `tfm train --recipe blueprint_v1` |
| `04_inference_embedding_extraction.ipynb` + `src/decoder_inference.py` | `notebooks/blueprint_v1/04_inference_embedding_extraction.ipynb` | `tfm infer --recipe blueprint_v1` |
| `05_xgboost_fraud_detection.ipynb` | `notebooks/blueprint_v1/05_xgboost_fraud_detection.ipynb` | `tfm evaluate --recipe blueprint_v1` |

### Behavioral differences

- v1 trained without shifting its labels; v2's `NextEvent` predicts the next
  token (ADR 0017).
- v1 embedded each transaction on its own. v2 embeds it with the customer's
  recent history and keeps the single-transaction embedding as a control.
- CI uses synthetic data only; TabFormer is not downloaded.
- CPU merchant hashing is not identical to cuDF `hash_values`.
- Metrics will not match v1's published notebook numbers.

## From 2.0.0a1 to 2.0.0a2

2.0.0a1 is the SDK the GTC Berlin lab ships. 2.0.0a2 renames its recipes and
groups its config keys; [the 2.0.0a1 to 2.0.0a2 guide](2.0.0a1-to-2.0.0a2.md)
maps each change, and `tfm migrate-config` rewrites a 2.0.0a1 config.

## From 2.0.0a2 to 2.0.0a3

2.0.0a2 configs load unchanged in 2.0.0a3. Code that calls the token-cache, tokenizer
and masking APIs changes, every token cache is rebuilt, and torch 2.9 or newer is
required; [the 2.0.0a2 to 2.0.0a3 guide](2.0.0a2-to-2.0.0a3.md) lists each change.
