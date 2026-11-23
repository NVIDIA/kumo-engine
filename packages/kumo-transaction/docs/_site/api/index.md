# API reference

Generated from the docstrings and type hints in `src/tfm`. Each page covers one
package and lists what it exports.

| Package | What it is for | Stability |
|---|---|---|
| [`tfm`](tfm.md) | The composition API: `TrxFM` and the eight core components | Public, stable |
| [`tfm.experimental`](experimental.md) | Components not yet in the core API: the hierarchical stacks, auxiliary objectives, other readouts | Experimental |
| [`tfm.config`](config.md) | The recipe configuration (`RecipeConfig`) every YAML file validates against | Public |
| [`tfm.contracts`](contracts.md) | Typed events, sequences, batches, and model outputs shared by every stack | Public, stable |
| [`tfm.runtime`](runtime.md) | Recipe orchestration: prepare, train, infer, evaluate, report | For recipes and notebooks |
| [`tfm.data`](data.md) | Adapters, labels, temporal and entity splits, sequence building | Public |
| [`tfm.features`](features.md) | Tokenizers | Public |
| [`tfm.adaptation`](adaptation.md) | Probe arms, fine-tuning, and the joint-fusion head | Experimental |
| [`tfm.evaluation`](evaluation.md) | Metrics, fraud baselines, reports | Public |

Which component belongs to which building-block slot is in
[Building blocks](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-transaction/docs/concepts/building-blocks.md). How to add one is in
[Extending the kit](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-transaction/docs/reference/extending.md).
