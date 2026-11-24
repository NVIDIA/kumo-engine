# Extending the kit

How to add a component to each building-block slot. Read
[building blocks](../concepts/building-blocks.md) for what the slots are and
[architecture overview](architecture-overview.md) for how a run flows.

Every new component follows the same path:

1. Implement it behind the slot's existing contract. Do not add a branch for
   one architecture elsewhere.
2. Make it selectable from config, and add the value to the config schema's
   `Literal` so an unknown name fails at load time.
3. If it can be paired wrongly, add the rule to
   `tfm.api.validation` with a message that names the rule and the fix.
4. Add a unit test and a contract test for the property that makes it
   correct: shapes and masks, causality, label alignment, or leakage.
5. Export it from `tfm.experimental`, not `tfm.__all__`, and add an
   `experimental` row to the
   [compatibility matrix](compatibility-matrix.md).
6. Document it in [config schema](config-schema.md) and in the building-blocks
   table, and add a `CHANGELOG.md` entry.

A component moves from `experimental` to `reference` only with a documented
recipe, tests, a synthetic smoke, and recorded artifacts
([support policy](support-policy.md)).

## Slot 0: a data adapter

An adapter maps an external book onto the `TransactionEvent`,
`EntityProfile`, and label contracts
([data contract](../concepts/data-contract.md)).

```python
from tfm.data.adapters.registry import register_adapter
from tfm.data.adapters.result import AdapterResult

@register_adapter("my_book")
def load_my_book(config, *, seed, flat_fields=True, fixture_dir=None) -> AdapterResult:
    ...
```

- Timestamps are timezone-aware; events are ordered per entity.
- Nothing post-outcome or label-derived goes into `numeric`, `categorical`,
  or `text`. Labels go into `EventLabel` with an `available_at`, or into
  `AdapterResult.labels` with their own timestamp.
- Profile attributes must be known at each sequence's cutoff; drop or date
  anything that is not.
- Take the local path from `config.path` or an environment variable, never
  from a URL in the repository.
- Test that a small generated fixture round-trips, that ordering holds, and
  that no label reaches the inputs.

## Slot 1: a tokenizer

A tokenizer implements `SequenceTokenizer` (`tfm.contracts.tokenizers`) and
declares one layout.

```python
from tfm.features.tokenizers.registry import register_tokenizer

@register_tokenizer("my_tokens")
class MyTokenizer:
    name = "my_tokens"
    layout = "flat_fields"          # or flat_key_value, key_value_time
    requires_flat_fields = False    # True if it needs the 12-field event view
    uses_profiles = False
    def fit(self, sequences): ...   # train-side sequences only
    def encode(self, sequence): ...
    def collate(self, sequences, *, max_length): ...
    def manifest(self): ...
```

- Reuse an existing layout if you can; then every structure that accepts it
  works unchanged. A new layout needs a batch type, a structure that accepts
  it, and an entry in `accepted_layouts_for` (rule 4).
- To change only how one event is encoded in the key-value family, subclass
  `KeyValueTimeTokenizer` and override `_encode_event`, as `EventFieldsTokenizer`
  does.
- `collate` never attaches labels; `tfm.data.collate.collate_batch` does.
- Test truncation keeps whole, most-recent events, and that padding is masked.

## Slot 1: a position encoding

Position specs live in `tfm.position` and are named in
`model.composition.position` via `tfm.api.builders.build_position`. A causal
structure must only use time measured from the past (rule 1): see
`elapsed_log`, not `time_log`, in `ProfileEventDecoder`. Test that changing a
later event's time leaves earlier states unchanged.

## Slot 2: a structure or backbone

- **A new backbone for the flat stream** (like `GRUBackbone`) takes the
  decoder config and returns hidden states with the same shape and mask
  semantics as `DecoderModule`. Wire it in
  `tfm.runtime.pipeline._flat_backbone`, add the name to
  `ModelConfig.backbone`, and add it to the eager and DDP trainers'
  supported backbones. A causal backbone must raise if asked to run
  bidirectionally.
- **A new structure** is a spec in `tfm.compositions` (its accepted layouts,
  depths, and hidden size), a module in `tfm.layers`, a builder in
  `tfm.api.builders`, and a `granularity.type` value. Add its pairing rules to
  `validate_composition`.
- **A new attention kernel** is an `AttentionSpec` registered in
  `tfm.attention.registry`. `EagerAttention` stays the correctness baseline;
  add an equivalence test against it.

## Slot 3: an objective

An objective is a frozen dataclass that declares, as methods,
`required_outputs`, `target_level`, `horizon_semantics`, `cutoff_semantics`,
`gradient`, and `leakage_checks` (see `PastReconstruction`). The model that
supports it computes the loss and reports it in `ModelOutput.losses` under the
objective's `name`.

- Primary objectives are named by `model.composition.objective`; auxiliary
  ones are listed in `objective.auxiliary` with their own config section.
  Register both in `tfm.api.builders.build_objectives`.
- Add a contract test that the targets come only from events the predicting
  state may see, and one that the loss reaches the parameters it claims to
  train.
- New objectives must be reviewed against
  [leakage and temporal splits](../concepts/leakage-and-temporal-splits.md)
  before model code is written.

## Slot 4: a readout, head, or adaptation method

- **A readout** is a spec plus a module taking `(states [B, S, D], mask
  [B, S])` and returning `[B, D]`. Add both to `tfm.pooling.registry` and the
  name to `CompositionConfig.readout`. Rule 3 decides where it is valid.
- **A probe arm** is added in `tfm.adaptation.probe` and
  `tfm.evaluation.fraud.baselines`. The `raw` arm stays: it is the baseline
  every other arm is judged against.
- **A fine-tuning method or head** extends `FineTune` in
  `tfm.adaptation.finetune`. It must copy the pretrained model (the frozen
  arm is never modified), use the same entity-disjoint rows as the probe, and
  report under its own arm name beside the frozen probe.

## Systems: a training backend

Backends register through `tfm.training.backends.register_backend` and declare
`BackendCapabilities`; the runtime refuses what a backend does not declare.
Every backend must pass the backend contract suite (ADR 0011).
