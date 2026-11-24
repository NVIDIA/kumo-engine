# ADR 0017 — The v1 Blueprint on the SDK

## Status

Accepted; experimental. Supersedes ADRs 0009 and 0010, which are archived with
the code they describe on branch `archive/v1-replay`. Implemented by the
`blueprint_v1` recipe and its notebooks. Applies
[ADR 0013](0013-fraud-probe-split-integrity.md)'s split rule at event level
(decision 4) and changes none of its rules.

## Context

The first port of the NVIDIA TFM Blueprint v1 tried to replay it exactly: its
merchant buckets (recovered from its published corpus), its token ids, its
checkpoint, and its single-transaction XGBoost probe with per-arm hyperparameters.
That needed a `tfm.legacy` package, two replay tokenizers, a merchant-table
environment variable, validators under `scripts/legacy/`, and data options
that only this recipe set. Along the way the replay found defects in v1 that make
exact reproduction the wrong target:

1. **Unshifted training labels.** v1's `FinancialCLMDataset` set
   `labels = input_ids`, and its training loop popped the labels before the
   forward pass and scored them with a loss that does not shift. Position *i*
   was scored against token *i*, which a causal model can see, so the published
   checkpoint learned to copy its input: same-token cross-entropy 0.0000,
   next-token accuracy 0.87%. Shifting the labels in the dataset
   (`labels[:n-1] = tokens[1:]`, the rest `-100`) fixes it, provided nothing
   downstream shifts again.
2. **One id for two tokens.** v1's vocabulary gave `MONTH_12` and `CARD_0`
   the same id.
3. **A probe without history.** v1 pretrained on 315-transaction windows but
   embedded each transaction alone (`<bos>`, 12 fields, `<eos>`), so the probe
   never used what pretraining learned about sequences.
4. **Identifiers as raw features.** The raw arm ordinal-encoded `User` and
   `Merchant Name`, with the same users on both sides of the split, so part of
   the baseline is memorized per-user fraud rate.
5. **A constant customer token.** `CUST` repeats the user id in every event. It
   is trivially predictable, costs 1 token in 13, and hands the encoder the
   user's identity.
6. **One seed.** The published 42% lift is one XGBoost seed; its median over
   ten seeds is about 16%, and each arm had its own hand-tuned parameters.

The port had defects of its own: the Llama backbone was declared
`position: nope` although Llama applies rotary positions, pretraining windows
were cut per card while probe histories are per user, and over-long rows were
truncated from the end, dropping the most recent events. The event-level probe
also borrowed the entity-level probe's user split, which is stratified on each
user's last `sequence_length` events. On TabFormer 278 of 1,949 test-window
users have a fraud, but only 2 have one among their last 128 events, so the
split had almost no positive users to stratify; a TabFormer-only truncation
rule (`keep_recent_positive`) had papered over this.

## Decision

1. **Rebuild, don't replay.** v1's flow (date split, raw baseline, tokenizer,
   causal pretraining, embeddings, raw versus raw-plus-embedding XGBoost) is
   built from the SDK's ordinary components. Nothing reads v1's corpus,
   checkpoint, or merchant table. The replay code is on `archive/v1-replay`.
2. **`NextEvent` owns the shift.** Targets are the labels moved one position;
   the backend contract suite's anti-copy check guards every trainer.
3. **Transaction-level fraud with history.** Each scored transaction is one row:
   the user's most recent `data.sequence_length` events up to and including it,
   pooled at the last token (`evaluation.history_window`). The event on
   its own is kept as a control arm, which is v1's embedding. Rare positives are
   kept on the train side by `evaluation.history_window.train_negatives_per_positive`;
   the evaluation side keeps its natural rate.
4. **The event-level probe splits users on event labels.** `event_entity_split`
   stratifies on whether any of a user's evaluation-window events is labelled
   fraud, across the whole window, as ADR 0013 requires of the entity-level
   probe for its own rows. A spliced task keeps its fixed split.
5. **Identifiers are not raw features.** `evaluation.tabular_baselines.exclude_keys`
   drops them from the tabular arms, and the recipe tokenizes 11 fields
   (`features.fields` without `CUST`).
6. **Declared positions match the backbone.** `position: rope` names the rotary
   positions the HuggingFace Llama backbone applies, which requires it; the eager
   decoder refuses it. `composition.attention` selects Llama's attention kernel.
7. **Every arm is tuned the same way** (Optuna on an entity-disjoint inner
   split), and results are reported over seeds, each of which moves
   pretraining, the user split, and the negative sample.

## Consequences

- The published v1 numbers are no longer reproducible from this branch. Use
  `archive/v1-replay` or the frozen tag `v1.0.0`.
- The earlier retrain records (`v1_retrain_*`, `v1_published_probe_reproduction`)
  moved to the archive branch with the probe that produced them; their numbers
  are not comparable with this recipe's.
- `features.tokenizer: v1_blueprint_*`, `TFM_V1_MERCHANT_MAP`,
  `data.pretrain_window_split_by`, and `data.pretrain_observation_window` are
  removed. Configs that set them fail with a message naming the replacement.

## Alternatives rejected

- **Keep the replay quarantined in `tfm.legacy`.** It kept v1-only options on
  the public surface and a second probe beside the SDK's, and it only reproduces
  numbers whose cause is already understood.
- **Replace Llama with the eager decoder.** Correct by construction, but not
  v1's block, and much slower at a 4096-token context.
