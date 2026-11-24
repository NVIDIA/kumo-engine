# 0012 — Blueprint v2: PRAGMA-style encoders feeding a causal history decoder

Status: accepted

*Update, 2026-10-03:* the recipe is `blueprint_v2`, and its event and profile encoders are shared with `pragma_like` (`tfm.layers.kvt_encoders`). The `nemo_automodel` recipe below was not built; `training.backend: nemo` is removed.

## Context

Milestone 2 of the [roadmap](../reference/roadmap.md) composes a new reference
architecture from two that this kit already runs:

- v1 (`blueprint_v1`) is a flat causal decoder over field tokens. Its
  corrected retrain showed the embeddings carry signal (the record is on
  branch `archive/v1-replay`, ADR 0017),
  but it spends 13 tokens per event, and its downstream probe never uses
  history.
- PRAGMA (`pragma_like`, ADR 0007) encodes each event and the profile
  into one vector each, then runs a bidirectional history encoder trained with
  masked-token objectives. It has no causal readout, so an embedding at time
  `t` is not by construction a function of events up to `t` alone.

v2 keeps PRAGMA's per-event compression and v1's causal next-event training:
each event becomes one summary token, and a causal decoder reads the summary
tokens in time order. AGENTS.md requires that the objective's horizon, cutoff,
and leakage checks be fixed before model code, which is what this ADR does.

## Decision

The recipe is `blueprint_v2`.

### Composition

```text
profile tokens (as of cutoff) -> ProfileEncoder (bidirectional) -> z_a        [B, D]
event j key/value tokens      -> EventEncoder (bidirectional)   -> z_e[j]     [B, N, D]
slots [z_a, z_e[0..N-1]] + calendar, TimeRoPE(elapsed_log)
                              -> HistoryDecoder (causal)        -> h          [B, N+1, D]
h -> LastTokenPool (slot after the last valid event) -> embedding             [B, D]
h -> NextEventFields (+ TokenEventKeyMLM)            -> loss
```

- **Batch contract.** v2 consumes `KeyValueTimeBatch`. One optional field is
  added, `elapsed_log` (below); existing consumers ignore it. The tokenizer is
  `key_value_time`, fitted on the train partition only.
- **Token embedding.** Keys and values share one tied embedding table, which
  also serves as the output projection of both heads. Encoders receive
  embedded tokens, not ids.
- **Encoders.** Each encoder attends bidirectionally *within* its own unit: one
  event's fields, or the profile's attributes. Nothing crosses events before
  the history decoder, so within-unit bidirectionality cannot leak across time.
- **History decoder.** Causal self-attention over `N + 1` slots, with TimeRoPE
  and PRAGMA's calendar embedding added to each event summary. It is the
  SDK-native eager decoder, not HF Llama: Llama's rotary embedding takes
  integer positions and cannot express TimeRoPE.
- **Time positions.** PRAGMA's `time_log` is measured back from each
  sequence's *last* event. Under a causal mask that would tell slot `j` how far
  the future extends. The decoder instead uses `elapsed_log =
  8·ln(1 + Δt/8)`, with `Δt` in seconds since the sequence's *first* event,
  which depends only on the past. The profile slot sits at position 0.
- **Readout.** `LastTokenPool` at the slot after the last valid event, i.e. the
  state that has seen every event at or before the cutoff. PRAGMA's `[USR]`
  slot is not used: under a causal mask, slot 0 sees only the profile.

### Contracts

Three protocols are added to `tfm.contracts.protocols`, beside `HistoryModel`,
`PoolingStrategy`, and `Objective`:

- `EventEncoder.forward(tokens [B,N,T,D], token_mask, event_mask) ->
  (summaries [B,N,D], token_states [B,N,T,D])`, with padded events zeroed.
  Token states feed the MLM head.
- `ProfileEncoder.forward(tokens [B,P,D], mask, time_log) -> [B,D]`. An
  entity with no profile still gets its learned summary token.
- `SummaryHistoryModel.forward(summaries, attention_mask, time_positions) ->
  [B,S,D]`. `HistoryModel` takes token ids and is left unchanged.

All names live in `tfm.experimental` behind a `compose_blueprint_v2(...)`
builder and `validate_composition`, as ADR 0007 did for PRAGMA. None are added
to `tfm.__all__`. Validation requires TimeRoPE, an attention kernel that
supports both bidirectional and causal masks, exactly one `NextEventFields`
with at most one `TokenEventKeyMLM`, the `LastTokenPool` readout, and `Binary`
heads.

Configuration: `granularity.type: profile_event_decoder`, with
`profile_depth`, `event_depth`, and the decoder depth in
`granularity.decoder.depth`. `objective.name: next_event_fields`, and the new
`objective.auxiliary: [token_event_key_mlm]` adds the MLM loss.

### Objective: `NextEventFields`

Indices are 0-based. Slot 0 holds the profile and slot `j >= 1` holds event
`j - 1`.

| property | value |
|---|---|
| target | every value of event `j`, predicted from `h[j]`, which has seen the profile and events `0..j-1` |
| head | `logits = (W · [h[j]; embed(key)]) · Eᵀ` over the tied vocabulary `E`. The target event's keys are queries; only values are predicted |
| label alignment | shifted one *event* inside the objective. Backends never build labels (ADR 0011) |
| horizon | the next event in the same entity's history, at or before the cutoff |
| cutoff | all events and profile milestones are `<= cutoff`. The last slot has no target |
| required outputs | `ModelOutput.hidden_states` `[B, N+1, D]`, `embeddings` `[B, D]`, `loss`, `losses["next_event_fields"]` |
| gradient | flows through all three components. Padded events and fields are never selected as targets |
| event level vs sequence level | pretraining targets are event-level. Downstream fraud labels stay sequence-level at the cutoff and are never objective inputs |

Giving the head the target's keys means the loss is conditioned on which
fields event `j` has (for example, whether it is an app or a card event). The
decoder state, and so the embedding, never sees event `j`.

### Auxiliary objective: `TokenEventKeyMLM`

PRAGMA's within-event masking (token, whole event, key, and `[UNK]` dropout)
is applied to the *inputs* during training. Next-event targets always come from
the unmasked batch. A masked token at event `j` is predicted from its token
state and `h[j+1]`, the decoder state that includes event `j`'s masked summary.
The total loss is the unweighted sum of both terms.

### Profile as of cutoff

A profile snapshot's `as_of` can postdate the cutoff: the pragmatiq books are
dated 2025-02-01, after every training cutoff. `KeyValueTimeTokenizer` now
drops lifelong milestones dated after each sequence's cutoff, in both `fit` and
`encode`, and times the rest from the cutoff instead of from `as_of`. On a
2,000-user Generator v2 book at a 2024-06-01 cutoff, 140 `premium_upgrade`
and 151 `address_change` milestones were previously encoded. This also
changes `pragma_like`.

Static attributes carry no date. pragmatiq Generator v2 draws them before
simulation, so they do not depend on later events. Other data sources must
make the same guarantee or supply dated attributes.

### Leakage checks (contract tests)

In `tests/contract/test_blueprint_v2.py` and `test_key_value_time.py`:

1. **Causal invariance.** Changing event `k` leaves slots `0..k` unchanged and
   changes slots after it.
2. **Past-only time.** `elapsed_log` for a prefix equals the full sequence's
   values on that prefix. `time_log` does not, which is why it is not used.
3. **Anti-copy.** After training on shifted-cycle data, next-event loss from
   slot `j` is below the loss of the same head read from slot `j+1`, which
   holds event `j` (`alignment_losses`, ADR 0010, 0011).
4. **Profile as of cutoff.** A milestone after the cutoff is absent from the
   vocabulary and the encoded profile.
5. **Padding invariance.** Batching with a longer sequence changes no state at
   valid slots and not the embedding.

A bidirectional decoder fails check 1, and a head reading slot `j+1` fails
check 3.

### Data and size

The first scaled configuration, `config.pragmatiq.yaml`, uses a pragmatiq
Generator v2 book (tag v1.1.0): 100,000 users, events from 2023-01-01 to
2025-02-01. Generator v2 is an agent-based simulator with personas, latent
traits, and injected episodes. It is not LLM-driven. The book is generated or
downloaded outside this repository and located through `TFM_PRAGMATIQ_ROOT`,
as v1 treats TabFormer; the generator is not a dependency of this kit.

Size follows PRAGMA-M: `d = 512`, 8 heads, profile/event/decoder depths
3/16/12, about 99M parameters outside the embedding table. PRAGMA-M's history
encoder has 6 layers. v2's decoder has 12, so the model sits at ~100M
parameters without widening. TabFormer has no profile table and gets its own
recipe rather than a config here.

### Backends

`eager` is the baseline, with `sdpa` as the fused kernel. Multi-GPU training
uses `ddp`: each rank holds a full replica and DDP averages gradients during
backward. At ~100M parameters, fp32 weights, gradients, and AdamW state take
about 1.6 GB, so sharding them (`fsdp2`) would add per-layer all-gathers and
reduce-scatters for no memory benefit; activations, the real memory cost, are
not sharded by FSDP either. `ddp` reuses the eager loop, accepts both
backbones and every batch layout, and runs on CPU (Gloo) as well as CUDA
(NCCL), so its multi-rank behavior is tested in CI.

For DDP, every parameter must receive gradient on every step, so the MLM
projection exists only when `objective.auxiliary` enables MLM.

Sharded training is shown on v1 instead, where `fsdp2` and a planned
`nemo_automodel` recipe train the HuggingFace backbone that NeMo AutoModel
builds directly. v2's TimeRoPE decoder is a native module, not an HF model.
Sharding returns to v2 only if a larger rung needs it.

### Downstream evaluation

The recipe reuses the key-value-time runtime's probes: one embedding per entity
at the cutoff, with raw, embedding, and combined arms. The history-window
probe (one embedding per labelled row, rebuilt at that row's timestamp, with
v1's retrain as the baseline) is roadmap Milestone 2 criterion 6 and is not
yet implemented.

## Consequences

- v2 adds three protocols and one objective under `tfm.experimental`, and one
  optional batch field. No stable contract or core export changes.
- The profile cutoff fix changes `pragma_like` outputs on books whose
  snapshot postdates the cutoff.
- A core `ddp` backend is added, with `BackendCapabilities.shards_parameters`
  telling the runtime whether rank 0 holds the whole model. `fsdp2` and
  existing v1 configs are unaffected.
- The recipe starts `experimental` and is in the catalog,
  `recipes/README.md`, and the compatibility matrix.
- v2 trades v1's 13 tokens per event for one summary token, so a 4096-slot
  context covers about 4,000 events instead of 315. Whether that helps is what
  the history-window probe measures.
