# Objectives and embeddings

`NextEvent` is causal next-token prediction. Padding tokens are ignored in the
loss (`-100`). Embeddings are last-valid-token hidden states.

Binary heads are optional and forward-only by default. Adding a `SupervisedHead`
objective (`fraud_task.supervised_head.enabled: true`) composes sequence-level BCE alongside
`NextEvent` so the head receives gradient; `ModelOutput.losses` then reports each
contribution separately. The sklearn probe on frozen embeddings remains the
reference baseline, so enabling head training adds a comparison rather than
replacing one. `SupervisedHead` reads only `fraud_labels` already reduced at
`cutoff_time`, so it cannot widen the label window.

Experimental `TokenEventKeyMLM` reconstructs masked key-value tokens from the
concatenation of event-encoder, history `[EVT]`, and history `[USR]` states.
Readout is the history-encoder `[USR]` slot (`UsrTokenPool`). Downstream probes
are logistic (CI) or optional XGBoost; recipe `pragma_like` also reports a
`default_12m` forecast probe whose histories are rebuilt at each row's
`eval_ts` (not the MLM temporal-split cutoffs). Probe train/test is a seeded
entity split of labeled rows when the parquet book has no predefined folds.
