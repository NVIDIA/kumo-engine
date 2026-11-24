# Transaction sequences

Sequences group events by `entity_id`, order them, apply cutoff and optional
observation window, then encode with a **sequence tokenizer**. Tokenization is a
layout, not a fixed column list: the same `TransactionSequence` can be encoded
by different tokenizers. A tokenizer is compatible only with history models that
accept its `layout`.

The core export surface uses the `flat_fields` layout (`TransactionFieldTokenizer`):

```
<bos> field-tokens… <sep> field-tokens… <eos> <pad>…
```

Other layouts (for example key–value–time) are separate tokenizer
implementations and must not be silently coerced into `TransactionBatch`.

v2 canonical embeddings for the decoder recipe are **sequence-level** (last
valid token over the history). v1 mostly scored single transactions; see the
migration guide.
