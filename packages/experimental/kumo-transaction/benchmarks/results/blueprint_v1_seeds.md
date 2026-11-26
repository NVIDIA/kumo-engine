# Blueprint v1 on TabFormer, every test row (one GPU, three seeds)

Milestone 1's evidence criterion. It reruns
[`tabformer_hf_decoder_gb10_seed42`](archive/tabformer_hf_decoder_gb10_seed42.md) at
seeds 42, 43, and 44, with two protocol changes: every test-period
transaction of the evaluation users is scored, not a 100,000-row sample, and
every XGBoost arm uses one fixed setting. The headline is
`history_window_xgboost_raw_embedding` over `history_window_xgboost_raw`
([evaluation protocol](../../docs/concepts/evaluation-protocol.md)). It is
not a quality or fraud claim.

| field | value |
|---|---|
| config | `recipes/blueprint_v1/config.yaml`, no overrides; seeds 42, 43, 44 |
| code | a snapshot of the working tree on top of commit `e871249`, matching the committed recipe. Seed 42 re-used the earlier seed-42 checkpoint (`--reuse-checkpoint`), whose config differs only in key names; seeds 43 and 44 were trained in this run |
| data | IBM TabFormer through `tfm prepare-tabformer` (v1's 80/10/10 date split): train ends 2017-05-13, test window 2018-10-12 to 2020-02-28 |
| tokens | `transaction_field_tokenizer`, 11 fields and `<sep>` per event; vocabulary 3,251 |
| model | HuggingFace Llama: 8 layers, hidden 512, 8 query / 2 KV heads, SwiGLU 1408, RoPE theta 5e5, SDPA; 25,882,625 parameters |
| pretraining | `NextEvent` on per-user windows of 341 events (4,093 tokens); eager, bf16, batch 16, 3,000 steps, AdamW, peak LR 2e-4. Loss 8.21–8.25 at step 0, mean of the last 100 steps 0.563–0.573; median gradient norm 0.59–0.60, no non-finite steps |
| probe rows | one row per transaction, the user's last 128 events up to and including it (1,537 tokens), pooled at the last token; beside it the transaction alone. Users are split into train and evaluation per seed. Train side: every training-window fraud plus 5 legitimate rows per fraud. Evaluation side: every test-window row, **684,900–732,966 rows with 801–845 frauds** (0.12%) |
| arms | XGBoost with one fixed setting for every arm (`tabular_baselines.xgboost_params`: the median of seed 42's earlier 30-trial searches on the five history-window arms, which saw training rows only); early stopping picks the tree count. Raw features exclude `user` |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.3 |
| wall clock per seed | prepare about 300 s, train 4,862 s (seeds 43, 44), evaluate 6,823–7,171 s |
| memory | 21.4 GiB peak GPU in training; 63 GB host memory observed once during seed 42's evaluation (peak host memory not measured) |
| raw output | [`blueprint_v1_seeds.json`](blueprint_v1_seeds.json) |

Reproduce:

```bash
uv sync --extra huggingface --extra baselines
export TFM_TABFORMER_ROOT=~/data/TabFormer
uv run python benchmarks/seed_range.py \
  --config recipes/blueprint_v1/config.yaml --seeds 42 43 44 \
  --run-root outputs/blueprint_v1/seeds \
  --summary benchmarks/results/blueprint_v1_seeds.json --device cuda
```

## Results

Test-side PR-AUC per seed (42 / 43 / 44). Lift is PR-AUC over
`history_window_xgboost_raw`, minus one.

| arm (`history_window_` prefix) | features | PR-AUC | lift | ROC-AUC |
|---|---|---|---|---|
| `xgboost_raw` | the transaction's own fields (v1's raw arm) | 0.0225 / 0.0087 / 0.0119 | — | 0.905 / 0.890 / 0.877 |
| `xgboost_raw_embedding` | raw + history embedding (the headline) | 0.0222 / 0.0124 / 0.0124 | −1% / +42% / +4% | 0.941 / 0.948 / 0.924 |
| `xgboost_raw_embedding:single_event` | raw + transaction-alone embedding (v1's combined arm) | 0.0553 / 0.0214 / 0.0347 | **+146% / +145% / +192%** | 0.950 / 0.943 / 0.938 |
| `tabular_xgboost` | hand-built history aggregates, no model | 0.0278 / 0.0137 / 0.0198 | +23% / +57% / +66% | 0.957 / 0.955 / 0.955 |
| `tabular_xgboost_embedding` | aggregates + history embedding | 0.0338 / 0.0180 / 0.0235 | +50% / +106% / +98% | 0.969 / 0.964 / 0.962 |

What these runs support, at three seeds and about 800 test frauds each:

- **A pretrained embedding beats raw-feature XGBoost at every seed, but it is
  the single-transaction embedding.** Embedding the transaction alone lifts
  PR-AUC by 145–192%, which is v1's claim. The history embedding, read at the
  last token after 128 events, is the headline and the stated design. It
  raises ROC-AUC at every seed (+4% to +6%), but its PR-AUC lift ranges from
  −1% to +42%, so it is not a result.
- **The history carries signal the history embedding does not.** Hand-built
  aggregates of the same 128 events lift raw by 23–66%. The history embedding
  still adds to them at every seed (+50% to +106% together), so it holds
  information the aggregates lack.
- **Absolute PR-AUC depends on the user split.** Each seed draws different
  evaluation users. Raw PR-AUC moves from 0.0087 to 0.0225 across seeds, more
  than any arm differs within a seed, so compare arms within a seed.
- **Every test row moves the numbers, not the ordering.** At seed 42 the
  100,000-row sample of the earlier record (106 frauds) gave +6% for the
  history embedding and +275% for the single one. On every row (801 frauds) it
  gives −1% and +146%.

The entity-level `fraud` arms in `metrics.json` (unprefixed) have one
evaluation positive and are diagnostics only.

## Audit

- **Pretraining stops at the train cutoff.** Pretraining windows come from
  the train partition only, with the cutoff at its last timestamp
  (2017-05-13); later events are dropped before windowing. The split manifest
  records the precomputed partitions as `split_protocol`, so it no longer
  lists the unused `temporal_split.train_end` date.
- **Why the single-transaction embedding helps so much: the raw row is weak
  on categories.** The raw arm one-hot encodes only the ten most frequent
  training values of each categorical field. 90.6% of test frauds are at a
  merchant outside the top ten (legitimate rows: 69.4%), and 71.7% are at an
  MCC outside it (legitimate: 31.3%). The fraud is concentrated where the raw
  row is blind: 13 MCCs with a training fraud rate above 5% hold 7.6% of
  test frauds and 0.14% of legitimate rows, and 90 such merchants hold 5.0%
  and 0.12%. The transaction's own embedding sees every value through the
  tokenizer's vocabulary.
- **`Errors?` carries little of the signal.** 2,635 of the 2,724 test frauds
  have no error; `Bad PIN` has a tenfold fraud rate, but on 58 frauds.

## History length

Is the history embedding weak because the last token, 1,537 tokens after the
history starts, dilutes the scored transaction? The seed-42 checkpoint was
re-scored on the same rows with shorter histories
(`--reuse-checkpoint --set data.sequence_length=N`). PR-AUC, with lift over
`xgboost_raw` (0.0225, unchanged because it sees only the transaction):

| history events (tokens) | raw + history embedding | raw + single embedding | `tabular_xgboost` | `tabular_xgboost_embedding` |
|---|---|---|---|---|
| 16 (193) | 0.0236 (+5%) | 0.0553 (+146%) | 0.0363 (+61%) | 0.0499 (+122%) |
| 32 (385) | 0.0201 (−11%) | 0.0553 (+146%) | 0.0325 (+44%) | 0.0439 (+95%) |
| 128 (1,537) | 0.0222 (−1%) | 0.0553 (+146%) | 0.0278 (+23%) | 0.0338 (+50%) |

Shortening the history does not close the gap, so dilution over a long
window is not the explanation. Even at 16 events the history embedding adds
almost nothing to raw. The hand-built aggregates do better on shorter
histories at this seed, and the history embedding still adds to them at
every length. Raw output:
[`blueprint_v1_history_length_16.json`](blueprint_v1_history_length_16.json),
[`blueprint_v1_history_length_32.json`](blueprint_v1_history_length_32.json).

## Not measured

- A raw arm that encodes merchant and MCC with their training fraud rates
  instead of top-ten one-hots. That would separate "the embedding learns
  categories" from "the embedding learns fraud".
- A readout that keeps the scored transaction in focus, such as pooling over
  its own tokens or concatenating the transaction and history embeddings.
  The history-length check above makes this the next experiment: the last
  token after a history is a different state from the last token of the
  transaction alone, even at 16 events.
- History length at seeds 43 and 44.
- Multi-GPU (`config.multi_gpu.yaml`).
