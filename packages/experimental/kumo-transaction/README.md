# Kumo Transaction (TxFM)

Kumo Transaction brings the transaction foundation model reference kit into
Kumo Engine. This snapshot retains the standalone `tfm` Python distribution,
imports, and CLI. Graph recipes use the optional Kumo Engine backend
described in the kit.

From the repository root, enter this directory before running the kit commands:

```bash
cd packages/experimental/kumo-transaction
```

The imported baseline and rules for later sync PRs are in [SYNC.md](SYNC.md).
Upstream contribution, ownership, release, and GitLab CI files are retained
as historical source context; Kumo Engine's repository review and contribution
rules govern changes here.

## Transaction Foundation Model kit

A **reference-architecture kit** for composing, training, and evaluating
transaction foundation models (TFMs). It is for education and solution
architecture; it is not a production authorization, fraud-decisioning, or
model-serving system.

The question every recipe answers is the same: does adding a TFM embedding to
raw features beat XGBoost on raw features alone? The original three sequence tracks that answer it are listed in
[recipes/README.md](recipes/README.md):

- **`blueprint_v1`**: NVIDIA's TFM Blueprint on public TabFormer card data,
  with flat field tokens and a causal decoder.
- **`pragma_like`**: a PRAGMA-style stack with bidirectional profile, event, and history encoders.
- **`blueprint_v2`**: PRAGMA's encoders under a causal history decoder.

`pragma_like` and `blueprint_v2` run on the same pragmatiq rows and tasks, so the two can be compared directly.

Experimental graph-only and graph-conditioned sequence examples add
future-subgraph pretraining; their commands and assumptions live in the
[recipe inventory](recipes/README.md).

## Start

Python 3.12+ and [uv](https://docs.astral.sh/uv/). The default native loader
builds from pinned source and needs rustup plus C/C++ build tools; see the
[installation guide](examples/kumo-engine/README.md#canonical-recipes). A cold
build can take tens of minutes. After installation, the smoke itself takes
about a minute on CPU:

```bash
uv sync --extra dev --extra native
uv run tfm smoke --recipe blueprint_v1     # CPU, generated data, about a minute
```

Then follow a notebook series (`uv sync --extra dev --extra native --extra notebooks`):

```bash
uv run jupyter notebook notebooks/blueprint_v1/   # the v1 Blueprint, step by step
uv run jupyter notebook notebooks/pragmatiq/      # PRAGMA-like and Blueprint v2
```

Full-size runs need a local dataset (`TFM_TABFORMER_ROOT` or
`TFM_PRAGMATIQ_ROOT`) and, at reference size, a GPU. Each recipe's README
gives its commands; [datasets](docs/reference/datasets.md) says how to get
each dataset and what it is for.

## Compose a model

A model is one choice in each of five slots: data, event representation,
sequence architecture, objective, and readout and adaptation
([building blocks](docs/concepts/building-blocks.md)). The kit refuses
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

A recipe's YAML is a serialization of this API
([config schema](docs/reference/config-schema.md)). The hierarchical stacks
are experimental and import from `tfm.experimental`.

## Where things are

| Question | Answer |
|---|---|
| What is supported, and at what tier? | [compatibility matrix](docs/reference/compatibility-matrix.md) |
| What has been reproduced? | [roadmap](docs/reference/roadmap.md), [recorded results](benchmarks/results/README.md) |
| How is a lift measured? | [evaluation protocol](docs/concepts/evaluation-protocol.md) |
| How do I add a component or recipe? | [extension guide](docs/reference/extending.md), [recipe template](docs/recipes/recipe-template.md) |
| How is the code organized? | [architecture overview](docs/reference/architecture-overview.md) |
| What changed? | [changelog](CHANGELOG.md), [v1-to-v2 migration](docs/migration/v1-to-v2.md) |

The documentation site (`uv sync --extra docs && make docs-serve`) renders
these pages with the notebooks and the API reference. CI runs only on
generated data; it never downloads TabFormer or checkpoints. The original
notebook-only v1 example is tagged `v1.0.0` on the `v1` branch.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md); report vulnerabilities as
[SECURITY.md](SECURITY.md) describes. See also the
[Code of Conduct](CODE_OF_CONDUCT.md). Licensed under Apache 2.0
([LICENSE](LICENSE)).

Run sequence, graph and hybrid recipes with the default native loader using the
[Kumo loader guide](examples/kumo-engine/README.md#canonical-recipes).
