# Industry landscape

This page maps published transaction foundation models onto the kit's five
[building-block slots](building-blocks.md), then shows which slot choices the
kit can already compose. Use it to explain a recipe's design choices by
pointing to the systems that made them.

Sources:

- [PRAGMA](https://arxiv.org/abs/2604.08649) (Revolut)
- [nuFormer](https://arxiv.org/abs/2507.23267) (Nubank)
- [TransactionGPT](https://arxiv.org/abs/2511.08939) (Visa)
- [NPPR](https://arxiv.org/abs/2401.01641) (Featurespace)
- [CoLES](https://arxiv.org/abs/2002.08232)
- [Stripe blog](https://stripe.com/blog/using-ai-optimize-payments-performance-payments-intelligence-suite)
- [Plaid blog](https://plaid.com/blog/sequential-foundation-model/)

Stripe and Plaid have published blog posts and talks, not papers. Treat their
entries as indicative only. No row below is a claim about any system's
performance.

## Systems by slot

### Revolut PRAGMA

- **Data and context:** multi-source events (transactions, app activity,
  trading) plus a profile of static attributes and dated lifelong milestones.
- **Event representation:** key/value/time tokens. Numbers are percentile
  bins, categories are tokens, and free text uses BPE or a frozen Nemotron
  text encoder. Time uses log Δt to the last event, calendar features, and
  TimeRoPE.
- **Architecture:** hierarchical, with bidirectional profile, event, and
  history encoders.
- **Objective:** masked modeling at token, event, and key level, with `[UNK]`
  dropout.
- **Readout and adaptation:** the `[USR]` and `[EVT]` slots; linear probes
  and LoRA at 2-4% of parameters.

### Nubank nuFormer

- **Data and context:** one customer's transaction history.
- **Event representation:** special tokens for amount sign, amount bucket,
  month, day, and weekday, plus BPE tokens of the description.
- **Architecture:** a flat token stream with event separators, read by a
  causal transformer with no positional encoding (NoPE).
- **Objective:** next-token prediction.
- **Readout and adaptation:** LoRA fine-tuning, plus fusion with existing
  tabular features. Its strongest variant, joint fusion, trains a DCNv2 over
  PLR-embedded tabular features together with the transformer's embedding,
  end to end.

### Visa TransactionGPT

- **Data and context:** transactions with metadata, entities, and engineered
  features as a separate modality.
- **Event representation:** field embeddings, with merchant category codes
  initialized from an LLM's text embeddings.
- **Architecture:** a 3D transformer over features, metadata, and time, fused
  through a virtual token layer. The temporal part is decoder-only.
- **Objective:** next-transaction generation, optimized jointly with
  supervised labels.

### Featurespace NPPR (TallierLTM)

- **Event representation:** continuous per-field embeddings.
- **Architecture:** a GRU.
- **Objective:** next-event prediction plus reconstruction of past events,
  conditioned on the time gap.
- **Readout and adaptation:** embeddings added as features to existing fraud
  models (early fusion), served through an embedding API.

### Stripe Payments Foundation Model

- **Data and context:** each payment with multi-entity context (buyer, card,
  device, merchant).
- **Event representation:** one dense embedding per payment. The objective
  has been described as BERT-style masking.
- **Readout and adaptation:** downstream classifiers over *sequences* of
  per-payment embeddings, for example for card-testing detection.

### Plaid

- **Architecture:** two stages. A contrastive transaction encoder learns
  per-event embeddings, and a sequence model runs over them.
- **Objective:** contrastive predictive coding, replaced-token detection, and
  temporal contrastive learning.

### CoLES

- **Architecture:** an RNN over event sequences.
- **Objective:** contrastive learning between sub-sequences of the same
  entity's history. It is a standard baseline in academic event-sequence
  benchmarks, where GRUs remain strong.

### NVIDIA Blueprint v1

- **Event representation:** one token per field (12 fields).
- **Architecture:** a flat stream read by a Llama causal decoder.
- **Objective:** next-token prediction.
- **Readout and adaptation:** XGBoost on raw features, embeddings, or both.

## What the landscape says about the slots

- The largest differences are in **sequence structure** (flat or
  hierarchical), **objective** (generative, masked, or contrastive, and often
  several at once), and **adaptation** (probe, then fine-tune, then fusion
  with tabular features).
- Token scheme and structure vary independently. nuFormer uses key-value-like
  tokens in a flat stream; TransactionGPT uses field inputs in a hierarchy.
- Fusion with existing tabular features recurs at Nubank, Visa, and
  Featurespace. It is usually the first question a fraud team asks.

## Kit coverage

The table shows the nearest composition each system maps to, and what the
kit does not cover. For support tiers, see the
[compatibility matrix](../reference/compatibility-matrix.md).

| System | Nearest kit composition | Not covered |
|---|---|---|
| PRAGMA | `pragma_like` recipe: `key_value_time` + `ProfileHistory` + `TokenEventKeyMLM` + `UsrTokenPool`, with LoRA | BPE or frozen-LLM text encoding |
| nuFormer | `flat_key_value` + `Flat(Decoder)` + `NextEvent`, with `fraud_task.finetune.head: joint_fusion` | BPE descriptions |
| TransactionGPT | `blueprint_v2` recipe: hierarchy with a causal history decoder | Engineered-feature modality, virtual tokens, LLM-initialized embeddings |
| Featurespace NPPR | `ProfileEventDecoder` + `NextEventFields` + `PastReconstruction`, or a `gru` backbone on a flat stream | Continuous per-field embeddings |
| Stripe | Frozen probe on per-entity embeddings | Multi-entity context, classifiers over sequences of event embeddings |
| Plaid | `SubsequenceContrastive` as an auxiliary objective | Pretrained frozen event encoder, CPC, replaced-token detection |
| CoLES | `SubsequenceContrastive` as an auxiliary objective, or a `gru` backbone | Contrastive-only pretraining |
| Blueprint v1 | `blueprint_v1` recipe | |

The [building-block ablation](../../benchmarks/ablations/building_blocks/README.md)
recipe runs several of these compositions on one shared synthetic book, so
you can compare slot choices under the same splits.
