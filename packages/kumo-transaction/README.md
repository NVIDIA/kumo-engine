# kumo-transaction

Transaction foundation model (TFM) support for Kumo Engine.

This package is based on the **Transaction Foundation Model kit**, a
reference-architecture kit for composing, training, and evaluating TFMs. The
kit is for education and solution architecture; it is not a production
authorization, fraud-decisioning, or model-serving system.

Source: <https://gitlab-master.nvidia.com/globalfsiteam/fsi_samples/transaction-foundation-model-dev> (branch `v2`)

## What it answers

Every recipe in the kit asks the same question: does adding a TFM embedding to
raw features beat XGBoost on raw features alone? Three reference architectures
answer it:

- **`blueprint_v1`**: NVIDIA's TFM Blueprint on public TabFormer card data, with
  flat field tokens and a causal decoder.
- **`pragma_like`**: a PRAGMA-style stack with bidirectional profile, event, and
  history encoders.
- **`blueprint_v2`**: PRAGMA's encoders under a causal history decoder.

`pragma_like` and `blueprint_v2` run on the same pragmatiq rows and tasks, so
they can be compared directly.

## Composing a model

A model is one choice in each of five slots: data, event representation,
sequence architecture, objective, and readout and adaptation. The kit refuses
pairings that would leak or cannot learn.

```python
from tfm import TrxFM, Flat, Decoder, NoPE, EagerAttention, LastTokenPool, NextEvent, Binary

model = TrxFM(
    granularity=Flat(Decoder(depth=2, mask="causal")),
    position=NoPE(),
    attention=EagerAttention(),
    objective=NextEvent(),
    readout=LastTokenPool(),
    heads={"fraud": Binary()},
)
```

A recipe's YAML is a serialization of this API. The hierarchical stacks are
experimental and import from `tfm.experimental`.

## Getting started with the upstream kit

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run tfm smoke --recipe blueprint_v1     # CPU, generated data, about a minute
```

Full-size runs need a local dataset (`TFM_TABFORMER_ROOT` or
`TFM_PRAGMATIQ_ROOT`) and, at reference size, a GPU.

## Upstream layout

| Path | Contents |
|---|---|
| `src/` | The `tfm` library |
| `recipes/` | Reference architectures and their commands |
| `notebooks/` | Step-by-step notebooks for `blueprint_v1` and pragmatiq |
| `benchmarks/` | Evaluation harness and recorded results |
| `configs/` | Recipe YAML |
| `docs/` | Concepts, reference, and migration guides |

## License

Apache 2.0.
