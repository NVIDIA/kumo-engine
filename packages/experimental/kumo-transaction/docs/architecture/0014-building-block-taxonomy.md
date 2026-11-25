# 0014 — Five building-block slots, with semantic validation only

Status: accepted

## Context

A common way to describe a transaction foundation model, and this kit's
original storyline, uses three blocks: tokenization, model plus objective, and
downstream adaptation. A survey of published systems
([industry landscape](../concepts/industry-landscape.md)) shows that the three
blocks work for teaching but are too coarse for organizing composable code:

- **"Tokenization" is too narrow.** Featurespace's NPPR and Visa's
  TransactionGPT embed fields as continuous vectors, and nuFormer's tabular
  side uses periodic (PLR) numeric embeddings. What these share is *event
  representation*: how an event's fields become vectors. Discrete tokens are
  one option within it.
- **"Model plus objective" hides sequence structure.** nuFormer is a flat token
  stream, PRAGMA is a hierarchy (per-event encoder, then a history model), and
  TransactionGPT is a hierarchy with a causal temporal part. Token scheme and
  structure vary independently. Each objective also carries its own leakage
  semantics, and most systems combine several.
- **Readout is the hinge between model and adaptation.** Last token, a
  `[USR]`/CLS slot, a mean, and a sequence of event embeddings are all in use,
  and whether a readout is valid depends on the attention mask.
- **Data and context come first.** The choice of entity, event sources,
  profile state, windows, and cutoffs shapes every later block.

Before this ADR, the kit locked each granularity to one tokenizer layout, one
position scheme, one objective family, and one readout
(`tfm.api.validation`). Several config keys that looked like choices were
ignored. So a user could not, for example, run key-value-time tokens through a
flat decoder, or fit the hierarchical encoders on TabFormer's field tokens.

## Decision

### Slots

Code, configuration, docs, and notebook chapters are organized as five slots
plus two cross-cutting concerns. The mapping from every component to its slot
is in [building-blocks.md](../concepts/building-blocks.md).

| # | Slot | Question it answers | Config section | Code |
|---|---|---|---|---|
| 0 | Data and context | Which entity, which events and profile, which window and cutoff, which labels? | `data`, `splits`, `tasks` | `tfm.data` |
| 1 | Event representation | How do an event's fields and its time become vectors? | `features.tokenizer`, `model.composition.position` | `tfm.features`, `tfm.position` |
| 2 | Sequence architecture | How are events arranged (flat stream or hierarchy), and which backbone and attention kernel read them? | `model.granularity`, `model.backbone`, `model.composition.attention` | `tfm.compositions`, `tfm.layers`, `tfm.attention` |
| 3 | Pretraining objective | What is the model trained to predict, and from which states? | `model.composition.objective`, `objective` | `tfm.objectives` |
| 4 | Readout and adaptation | Which state becomes the embedding, and how is it adapted to a task? | `model.composition.readout`, `fraud_task` | `tfm.pooling`, `tfm.heads`, `tfm.adaptation` |

The cross-cutting concerns are not slots, because every composition must
satisfy them the same way:

- **Leakage and evaluation protocol**: temporal splits, label windows, and
  baselines (ADRs 0004, 0008, 0013, `tfm.evaluation`).
- **Systems**: training backends, precision, packing, and varlen (ADR 0011,
  `tfm.training`).

The teaching storyline keeps the three blocks. The Blueprint v2 notebooks
present slots 0-4 as chapters and tell learners which of the three blocks
each chapter belongs to.

### Validation checks semantics, not stacks

`validate_composition` keeps only the rules that follow from what a slot
means. Each rule names the property it protects:

1. **A causal objective needs a causal backbone and past-only time
   positions.** `NextEvent` and `NextEventFields` require causal history
   attention. Position encodings that depend on the sequence's last event
   (PRAGMA's `time_log`) are rejected under a causal history model.
2. **Masked modeling needs bidirectional attention within its unit.**
   `TokenEventKeyMLM` requires an event encoder that attends bidirectionally
   over one event's tokens.
3. **The readout must see every event up to the cutoff.** `LastTokenPool`
   and `MeanPool` are valid under a causal history model. `UsrTokenPool` is
   valid only under a bidirectional history model, because under a causal
   mask the `[USR]` slot sees only the profile.
4. **Variable-length events need a boundary.** Tokens from different events
   must be separated, either by a hierarchy that encodes one event per unit or
   by separator tokens in a flat stream. A tokenizer declares its layout, and
   a structure declares which layouts carry the boundary it needs.

Everything else is a free choice. In particular, token scheme and structure
are independent:

| Layout | Flat stream | Hierarchy |
|---|---|---|
| `flat_fields` (one token per field, `<sep>` between events) | yes | via `event_fields` |
| `key_value_time` (per-event key/value tokens) | via `flat_key_value` | yes |

`flat_key_value` serializes key-value-time vocabulary into one stream with
`[EVT]` separators and time tokens. `event_fields` groups the 12 flat-field
tokens per event into the key-value-time batch. Both are tokenizers, so
neither structure gains a special case.

Rules that protect an implementation limit rather than a meaning (for
example, "the HuggingFace backbone cannot honor packed segments") stay in
backend capability checks (ADR 0011), not in composition validation.

### Configuration

Every config key either changes behavior or is rejected. `RecipeConfig`
forbids unknown keys, and removed keys raise an error that names their
replacement. `model.composition` is the single place a recipe states its
slot 1-4 choices, and the runtime builds exactly what it declares.

### Public surface

The core export surface stays the eight names in AGENTS.md. Hierarchical
compositions, new objectives, readouts, and adaptation methods live in
`tfm.experimental` until they have recorded evidence. Growing the core
surface is a separate decision.

## Consequences

- Validation errors now name the violated rule, and more combinations pass.
  Each newly allowed combination still needs a smoke test before it is
  documented as supported in the
  [compatibility matrix](../reference/compatibility-matrix.md).
- Configs that set ignored keys stop loading, with a message naming the
  replacement key (see the CHANGELOG migration notes).
- The ablation harness `benchmarks/ablations/building_blocks` swaps one slot at a time
  on shared splits. That is only meaningful because the slots are now
  independent.
- Docs and notebooks share one vocabulary: "slot 1, event representation"
  means the same thing in the ADRs, the config reference, and the notebooks.

## Native recipe input loading

After 2.0.0a4 the recipe data slot defaults to the pinned experimental
`kumo-pytorch`/DiskGraph source on main. This is an input backend choice, not an
architecture or tokenizer choice. All sequence compositions retain their
existing encoders; native graph inputs pass through the backend-neutral batch
contracts. `data.loader: python` remains the explicit oracle/control. Explicit
caches retain their own batch builder. Future targets are still constructed
separately from historical input graphs. Installation and supported schema
limits are documented in the [migration guide](../migration/native-loader-default.md).
