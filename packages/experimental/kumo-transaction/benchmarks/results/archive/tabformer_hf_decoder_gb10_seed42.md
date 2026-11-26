# v1 Blueprint rebuilt on the SDK, TabFormer (one GPU, one seed)

Milestone 1, criterion 6, at **one seed**. The criterion asks for three seeds;
this record does not meet it on its own. It is the first run of
`tabformer_hf_decoder/config.gb10.yaml`
([Blueprint v1 guide](../../../recipes/blueprint_v1/README.md)): v1's
Llama decoder pretrained by next-event prediction on every TabFormer user,
then probed per transaction, with each scored transaction embedded either with
the customer's last 128 transactions or on its own. It is not a quality claim.

Superseded by [`blueprint_v1_seeds.md`](../blueprint_v1_seeds.md): the same
config at three seeds, scored on every test row with one fixed XGBoost setting.

| field | value |
|---|---|
| config | `recipes/tabformer_hf_decoder/config.gb10.yaml` at commit `1c985ec`, no overrides; seed 42 |
| code | training ran from a snapshot of the working tree that became `1c985ec`; its differences from the commit (flat-row truncation, which never triggers at these row sizes, and evaluation-only inference options) do not touch training. Evaluation re-ran from the checkpoint at `1c985ec` (`report.py --reuse-checkpoint`) |
| data | IBM TabFormer through `tfm prepare-tabformer` (v1's 80/10/10 date split); all 2,000 users loaded, 1,584 with training rows; train cutoff 2017-05-13, test window 2018-10-12 to 2020-02-28 |
| tokens | `transaction_field_tokenizer`, 11 fields (v1's layout without `CUST`) and `<sep>` per event; vocabulary 3,251 |
| model | HuggingFace Llama: 8 layers, hidden 512, 8 query / 2 KV heads, SwiGLU 1408, RoPE theta 5e5 (`position: rope`), SDPA; 25,882,625 parameters (v1: 28.95M, with a 6,251-token vocabulary) |
| pretraining | `NextEvent` on per-user windows of 341 events (4,093 tokens); eager, bf16, batch 16, 3,000 steps of a 10,000-step cosine schedule, 500 warmup, AdamW (0.9, 0.95), peak LR 2e-4, weight decay 0.077 |
| loss | 8.25 at step 0, 0.564 at step 2,999 (mean of the last 100 steps 0.573); median gradient norm 0.60, no non-finite steps |
| probe rows | `history_window_probe`: one row per labelled transaction, the user's last 128 events up to and including it (1,537 tokens), pooled at the last token; beside it the transaction alone (`single_event`). Users split 1,073 train / 460 evaluation by `event_entity_split`, stratified on fraud in the evaluation window |
| sizes | train: 100,000 rows, all 17,322 training-side frauds plus 82,678 sampled legitimate rows (the row cap binds before the ratio of 5). Evaluation: 100,000 rows sampled uniformly, **106 frauds** (0.106%) |
| arms | XGBoost, each arm with its own 30-trial Optuna search on a user-grouped inner split of the training rows; raw features exclude `user` (48 raw, 126 hand-built history features) |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock | prepare about 7 min and train about 80 min; the evaluation re-run took 2 h 24 min including data preparation, most of it in the XGBoost searches |
| memory | 63.2 GB peak host memory during evaluation; peak GPU memory not measured |
| raw output | [`tabformer_hf_decoder_gb10_seed42.json`](tabformer_hf_decoder_gb10_seed42.json) |

Reproduce (at commit `1c985ec`):

```bash
uv sync --extra huggingface --extra baselines
export TFM_TABFORMER_ROOT=~/data/TabFormer
uv run python recipes/tabformer_hf_decoder/report.py \
  --config recipes/tabformer_hf_decoder/config.gb10.yaml --device cuda \
  --run-dir outputs/tabformer_hf_decoder/gb10_seeds/seed_42
```

## Results

Test-side PR-AUC (average precision) over 106 frauds, seed 42. Lift is PR-AUC
over `history_window_xgboost_raw`, minus one.

| arm (`history_window_` prefix) | features | PR-AUC | ROC-AUC | P@50 | lift |
|---|---|---:|---:|---:|---:|
| `xgboost_raw` | the transaction's own fields (v1's raw arm) | 0.0219 | 0.890 | 0.10 | — |
| `xgboost_raw_embedding` | raw + history embedding | 0.0231 | 0.935 | 0.08 | +6% |
| `xgboost_raw_embedding:single_event` | raw + transaction-alone embedding (v1's combined arm) | **0.0823** | 0.944 | 0.20 | **+275%** |
| `tabular_xgboost` | hand-built history aggregates, no model | 0.0305 | 0.962 | 0.06 | +39% |
| `tabular_xgboost_embedding` | aggregates + history embedding | 0.0351 | 0.964 | 0.10 | +60% |

Logistic probes on the frozen embeddings alone: history 0.0359, transaction
alone 0.0575 (`history_window_embedding`, `single_event_embedding`).

What this run supports, at one seed and 106 test frauds:

- **Pretrained embeddings add to raw features.** Both embeddings raise ROC-AUC
  over raw (0.890 to 0.935 and 0.944), and the transaction-alone embedding
  nearly quadruples PR-AUC. That is v1's claim, now with a correctly shifted
  objective, no user id, disjoint users, and every arm tuned the same way.
- **The history embedding does not beat the single-transaction one.** This is
  the opposite of what the rebuild expected. With the last-token readout after
  128 events, the history embedding adds 6% PR-AUC to raw; the embedding of the
  transaction alone adds 275%. Alone, too, it trails (0.036 against 0.058).
  The history's value is real: hand-built aggregates of it lift raw by 39%.
  This readout just does not carry it into the embedding.
- **One seed is not a range.** With 106 positives, one or two rows move
  PR-AUC noticeably. The ordering above needs seeds 43 and 44
  (`benchmarks/seed_range.py`) before it is a result.

The entity-level `fraud` arms in `metrics.json` (unprefixed) have one
evaluation positive and are diagnostics only: few TabFormer users have a
fraud among their last 128 events.

## Not measured

- Why the history embedding underperforms. Candidates, none tested: the
  last-token state after 1,537 tokens dilutes the scored transaction; probe rows
  (128 events) are shorter than the 341-event pretraining windows; a pooling
  over the scored transaction's own tokens, or a shorter `sequence_length`,
  would separate the two.
- Seeds 43 and 44, and the spread they give.
- Peak GPU memory.
