# ADR 0019 — A flat BERT encoder on Blueprint v1's tokens

## Status

Accepted, experimental. Implemented as `benchmarks/ablations/flat_encoder_tabformer.yaml`.
Adds an experimental composition beside
[ADR 0017](0017-v1-blueprint-on-the-sdk.md)'s flat decoder; changes no
existing contract, split, or label rule.

## Context

Every flat model in the kit is a causal decoder: `Flat(Decoder(mask="causal"))`
trained on `NextEvent`. On TabFormer that model's history embedding does not
lift raw-feature XGBoost reliably (−1% to +42% PR-AUC over three seeds,
[`blueprint_v1_seeds`](../../benchmarks/results/blueprint_v1_seeds.md)), while
the embedding of the transaction alone lifts it by 145–192%. One explanation
is the readout: under a causal mask, the last token summarizes the history as
a predictor of the next event, not as a description of the scored one.

The other common design for tabular sequences is encoder-only. IBM's TabBERT,
published with TabFormer, masks field values and reads a bidirectional state.
The kit's encoder-only model, `pragma_like` (ADR 0007), needs key-value-time
tokens and a hierarchical stack, so it cannot be compared with v1 on v1's own
tokens. The question this ADR enables: on the same TabFormer rows, tokens, and
probes, does a bidirectional encoder trained by masked field modeling give a
history embedding that adds to raw features, where the causal decoder's does
not?

## Decision

1. **Composition.** `tfm.experimental.FlatEncoder`, selected by
   `model.composition.granularity.type: flat_encoder`, sized by the same
   `decoder:` section every stack reads (`depth`, `hidden_size`, `num_heads`,
   `dropout`, `max_seq_len`; `mask` does not apply). It reuses the
   bidirectional, rotary `EncoderBlock` of the key-value-time encoders (GELU
   feed-forward at 4 × width, so `ffn_size` must be unset). It is not added to
   the core `TrxFM` export surface; the Hugging Face Llama backbone is not
   used, because it is causal only.
2. **Tokens.** `transaction_field_tokenizer`: v1's 11 fields and a `<sep>`
   per event, `<bos>` first. The fraud label and `Errors?` are not tokens, as
   in v1. The tokenizer has no mask token; `FlatEncoder` appends one
   `<mask>` id after the fitted vocabulary, so v1's vocabulary, token ids, and
   checkpoints are unchanged. The tokenizer reports each field's id range as a
   `FieldVocabulary` (`tfm.contracts.tokenizers`), which the runtime passes to
   the builder.
3. **Objective.** `MaskedFields`, a new objective:
   - each field token (not `<bos>`, `<sep>`, `<eos>`, or padding) is chosen
     with `p_field` (default 0.15), and each whole event with `p_event`
     (default 0.10), all its fields at once;
   - a chosen token is replaced by `<mask>` 80% of the time, a random token of
     the same field 10%, and kept 10%;
   - cross-entropy over the chosen positions only, restricted to that field's
     id range, so a masked amount is predicted among amounts;
   - **label horizon:** none; the targets are values inside the window.
     **Cutoff semantics:** pretraining windows come from the train partition
     only, ending at or before the train cutoff, as for v1.
     **Outputs:** per-position logits and the loss. **Gradient:** masked
     positions only. **Leakage checks:** below.
4. **Position.** Rotary positions over token index (`position: rope`), with
   v1's base of 5e5. `NoPE` is refused: a bidirectional stream without
   positions has no order.
5. **Readout.** `last_event_mean`: the mean hidden state over the scored
   (last) event's field tokens. The scored transaction is in the window and
   every token sees the whole history, so this reads the transaction in
   context. `usr_token` (the `<bos>` state, BERT's `[CLS]`) and `mean` are the
   ablations; `last_token` is refused.
6. **Validation rules,** before any data is prepared:
   - `NextEvent` or `NextEventFields` on `FlatEncoder` breaks rule 1 (a
     causal objective needs a causal history model);
   - `MaskedFields` on `Flat(Decoder)` breaks rule 2 (masked modeling needs
     bidirectional attention inside its unit);
   - `NoPE` on `FlatEncoder` is refused;
   - a tokenizer other than `transaction_field_tokenizer` is refused, since
     masking needs its field id ranges; packed batches are refused;
   - next-event surprise needs a next-event head, so `evaluation.surprise` is
     refused for `FlatEncoder`.
7. **Evaluation.** Unchanged: every TabFormer arm of ADR 0018 and the
   single-event control, on the history-window rows of ADR 0017, so the raw
   and hand-built arms are identical to `blueprint_v1_seeds`.
8. **Where it lives.** A config under `benchmarks/ablations/` first, sized as
   v1 (8 layers, hidden 512, 28.5M parameters against v1's 25.9M; 341-event
   pretraining windows; 3,000 steps). It becomes a recipe config only if it shows a lift
   that holds over three seeds.

### Leakage

- A probe row's window ends at the scored transaction, so bidirectional
  attention reaches nothing after it; the encoder's reach is the row's history,
  the same as the decoder's.
- Pretraining windows end at the train cutoff, so no test-period event is a
  masked target or a context token in pretraining.
- No label-derived field is a token.

### Tests

- Contract: changing a later token changes earlier hidden states (the opposite
  of the decoder's causal-invariance test); padding never changes a real
  token's state; masking chooses field tokens only, and with `<mask>`
  replacement no chosen input holds its target; random replacements stay in
  their field; the loss ignores other fields' logits; `last_event_mean` reads
  only the last event's field tokens (`tests/contract/test_flat_encoder.py`).
- Validation: each refusal in decision 6.
- Recipe smoke: v1's smoke config with the encoder slots swapped, on CPU,
  and the ablation config differing from v1 only in the model and objective
  (`tests/recipe_smoke/test_smoke_flat_encoder.py`).

## Consequences

- One experimental composition and one objective, both in
  `tfm.experimental`, with rows in the compatibility matrix. No public batch,
  checkpoint, or output contract changes.
- Bidirectional attention over 4,096-token pretraining windows costs about
  what v1's causal attention does with SDPA; on one GB10 a seed should take
  about as long as v1's (about 80 min of training, about 2 h of evaluation).
- With no next-event head there is no surprise readout. A masked
  pseudo-likelihood of the last event could take its place, and is left for
  later.

## Alternatives considered

- **A Hugging Face `BertModel` or ModernBERT backbone.** Adds a dependency
  path and a second implementation of blocks the kit already has; a native
  encoder keeps the comparison with v1's decoder to the mask and objective.
- **`Decoder(mask="bidirectional")`.** Overloads a core export whose name and
  contract say causal.
- **Masked modeling over the whole vocabulary.** Lets the model spend capacity
  telling fields apart; field-restricted targets match TabBERT.
- **Only `pragma_like` on TabFormer.** It runs today (key-value-time tokens,
  `benchmarks/ablations/pragma_like_tabformer.yaml`), but differs from v1 in
  tokens, hierarchy, and readout at once, so it cannot isolate the mask.
