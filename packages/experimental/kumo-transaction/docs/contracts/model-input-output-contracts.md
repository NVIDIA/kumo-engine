# Model input/output contracts

`TransactionBatch` carries `input_ids`, `attention_mask` (1 = valid), optional
LM `labels`, `fraud_labels`, `numeric_features`, and alignment ids.

`ModelOutput` carries `hidden_states` `[B, T, H]`, pooled `embeddings` `[B, H]`,
next-token `logits` `[B, T, V]`, optional `loss`, optional `predictions`, and
optional `losses`.

`loss` is the total optimized scalar. When more than one objective is composed,
`losses` maps each objective to its already-weighted contribution, and those
contributions sum to `loss`. A single-objective model may leave `losses` unset.

Padding must not contribute to pooling or loss. Causal attention must not allow
position `t` to depend on tokens `> t`.
