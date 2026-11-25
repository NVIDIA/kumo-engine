# Extending the kit

How to add a component to each building-block slot. Read
[building blocks](../concepts/building-blocks.md) for what the slots are and
[architecture overview](architecture-overview.md) for how a run flows.

Every new component follows the same path:

1. Implement it behind the slot's existing contract. Do not add a branch for
   one architecture elsewhere.
2. Make it selectable from config. If the field is a `Literal`, add the value so
   an unknown name fails at load time; registered-name or import-path fields
   (`features.tokenizer`, `data.adapter`, `data.batch_builder`) need no schema change.
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
- To turn field values into token ids, hold a `ValueVocabulary`
  (`tfm.features.tokenizers`, experimental). Its `block_statistics`, `merge`,
  `value_ids` and `encode_events` give a token-cache tokenizer the same
  vectorised fit and encode as `KeyValueTimeTokenizer`. Each value id already
  names its key, except for text bytes, so a one-token-per-field layout needs
  nothing more; `encode_events` labels each token with its key for text.
- For the key-value layout (a `KEY_` token beside each value token), wrap the
  vocabulary in `KeyValuePairs` (`tfm.features.tokenizers`, experimental) and
  pass your event encoder to `encode_record` and `collate_records`
  (`tfm.features.tokenizers.key_value_time`), as `EventFieldsTokenizer` does.
- Per-field evaluation (surprise) groups tokens by the batch's `key_ids`.
  Implement `field_names()` (`tfm.contracts.tokenizers.FieldNamedTokenizer`)
  to name them; `KeyValuePairs.field_names` does it for the key-value layout.
- To read recipe keys as the built-ins do, give the class `feature_keys` and a
  classmethod `from_config(features, data)`
  (`tfm.features.tokenizers.registry.ConfiguredTokenizer`). A run refuses a
  `features` key the class does not list; without `from_config` the class is built
  from `features.tokenizer_options`.
- A token cache's rows reach the trainer through `data.batch_builder`: a
  registered name or `"package.module:attribute"` of a
  `tfm.data.token_cache.builders.BatchBuilder` (experimental). It receives the
  run's placement (`data.token_cache_on_device` on CUDA, else the CPU) and
  `event_arrays`, the per-event arrays the model reads; it need load no others.
- A tokenizer's cache uses the format its class names in `token_cache_format`
  (default `columnar`, any `ColumnarTokenizer`): a registered name or
  `"package.module:attribute"` of a `tfm.data.token_cache.formats.TokenCacheFormat`
  (experimental). The format checks the recipe, builds, opens and reads the cache.
- A per-event array the tokenizer emits (`EncodedBlock.events`, one number per
  event) reaches the batch under its name. The hierarchical models read a cyclic one
  through `model.composition.granularity.time_features`, e.g. `minute: 60`.
  `tfm.testing.conformance.check_columnar_tokenizer` checks each array's shape and
  dtype, that the cache's device batches keep one shape, and, given `sequences`,
  that a tokenizer with a per-sequence `collate` gives the cache's tokens. It raises
  `ConformanceError`.
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
Every backend must pass the backend contract suite (backend contracts).

## Graph composition extension points

Graph recipes use the same `RecipeConfig`, adapter registry and tokenizer
registry as other recipes. A registered adapter can return `AdapterResult.transfers`
alongside its events. User graph architectures receive the same public build
context as the built-ins:

```python
from torch import nn
from tfm.api.builders import model_from_config
from tfm.api.graph import GraphArchitecture, GraphBuildContext, graph_specs, register_graph_architecture
from tfm.config import RecipeConfig
from tfm.data.graph_recipe import prepare_graph_data
from tfm.heads.binary import Binary
from tfm.layers.gnn import GNNEncoderModel

def my_graph(config, context):
    return GNNEncoderModel(
        context.graph, objective=Binary(),
        head=nn.Sequential(nn.Linear(context.graph.hidden_size, 16), nn.Tanh(), nn.Linear(16, 1)),
    )

register_graph_architecture("my_graph", GraphArchitecture(my_graph, "graph"))
config = RecipeConfig.model_validate({
    "recipe_family": "graph",
    "model": {"composition": {"granularity": {"type": "my_graph"}}},
})
data = prepare_graph_data(config)
graph, sequence = graph_specs(config, data)
model = model_from_config(
    config, vocab_size=sequence.vocab_size,
    graph_context=GraphBuildContext(graph, sequence),
)
output = model(data.pretrain[0])
```

A module-level `GraphArchitecture` can instead be selected by import path in
`model.composition.granularity.type`. Its `layout` is `graph` or `graph_history`;
that declaration selects its inference input. The runner requires a head-free
`encoder` module for reuse. The `gnn_sequence` architecture uses the direct
Engine encoder by default. An encoder wrapper can be supplied as any module
with the declared graph spec and a `GraphBatch -> root embeddings` call.

### Context-aware history, readout and head

`GNNSequenceModel` accepts `readout=`, `head=` and `history_model=`. A custom
history module is called as `history_model(history, context=...)` and returns
`(states [B,S,D], mask [B,S])`. A custom `ContextHistorySpec` can instead build
that module from its `build(attention)` method. Only the width and callable
contract are shared; it need not subclass `ProfileEventDecoder`.

The reusable public halves of the built-in stack are:

```python
from tfm.layers.profile_event_decoder import encode_events, decode_history

# `stack` satisfies ProfileEventStack by callable signatures, not concrete classes.
summaries, token_states = encode_events(stack, history)
states, mask = decode_history(
    stack, history, summaries, history.event_mask, history.elapsed_log,
    context=graph_context,  # [B,D], or a callable returning it
)
```

The stack exposes `embed_tokens`, `event_encoder`, `calendar`,
`profile_encoder`, and `decoder`. The calendar consumes the batch's named
per-event arrays. A deferred graph field is evaluated after event encoding and
appended before profile encoding. Components must preserve masks, past-only
times, and the read-only batch contract. This hook is for cutoff-level tasks,
not prefix-level next-event targets.

YAML can name the same components using registered names or import paths:

```yaml
model:
  composition:
    granularity:
      type: gnn_sequence
      history_model: my_package.components:history_factory
      task_head: my_package.components:binary_head
    readout: my_package.components:ReadoutSpec
```

A history factory receives `(sequence_spec, attention, *, position)` and returns
a module. `position` is the selected `NoPE` or `TimeRoPE` spec; implement it or
reject unsupported choices explicitly. A direct `ContextHistorySpec.build`
accepts the same keyword after `attention`. Graph pretrain-only data has no
tokenizer and `graph_specs` returns `None` for the sequence spec; graph factories
must not require sequence state for that stage.
A binary head factory receives the embedding width and returns a module with
`[B,D] -> [B,1]` raw logits. Register them with
`tfm.api.graph.register_context_history` and `register_task_head` when selecting
by name. A custom readout spec has `name` and `build()`, with its module mapping
`(states, mask) -> [B,D]`; register its factory through
`tfm.pooling.registry.register_readout`. The supervised sequence control uses
the same factories and initialization as the hybrid.

A recipe with different orchestration can register a `RecipeRunner` through
`tfm.runtime.registry.register_recipe_runner` or select one directly with
`recipe_family: my_package.lifecycle:run_command`. It receives the already
parsed shared config; it should not read YAML a second time.
