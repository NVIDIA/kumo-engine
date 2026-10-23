<!-- This README is also the PyPI project description: keep every link absolute. -->
<p align="center">
  <img height="180" src="https://raw.githubusercontent.com/NVIDIA/kumo-engine/main/ui/src/images/kumo_logo_green.svg" alt="Kumo" />
</p>

______________________________________________________________________

# Kumo Engine

Kumo Engine (`kumo.engine`) trains graph neural networks on relational data, in-process on your own machine. Define tables and the relationships between them, describe what to predict with a predictive query, then train a model and run batch prediction.

## Requirements

- Linux x86_64 with glibc 2.34 or newer, or macOS 14 or newer on Apple silicon
- CPython 3.11 or newer
- For production workloads, an NVIDIA GPU with a compatible driver; see [prerequisites](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/prerequisites) and [GPU sizing](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/gpu-sizing)

Kumo Engine uses a GPU when one is available: an NVIDIA GPU through CUDA, or an Apple silicon GPU through PyTorch's MPS backend. Without one, training and prediction run on CPU, which is enough for small jobs such as the quickstart below. To choose the device, pass `accelerator=` (`"auto"`, `"cpu"`, `"gpu"`, or `"mps"`) to `Trainer`.

## Installation

```bash
pip install kumo-engine
```

The base install reads local CSV and Parquet files. For other data sources, install the matching extra in the same environment:

| Data source | Command |
| --- | --- |
| Amazon S3 | `pip install "kumo-engine[s3]"` |
| Google Cloud Storage | `pip install "kumo-engine[gcs]"` |
| BigQuery | `pip install "kumo-engine[bigquery]"` |
| Snowflake | `pip install "kumo-engine[snowflake]"` |
| Databricks | `pip install "kumo-engine[databricks]"` |

Text columns are encoded with word vectors. The `text` extra installs the default model, GloVe 6B 50d:

```bash
pip install "kumo-engine[text]"
```

Extras combine, for example `pip install "kumo-engine[s3,text]"`.

The wheels on PyPI are for Linux x86_64 and for macOS on Apple silicon. To build and install on Linux ARM64 (DGX Spark), follow the [DGX Spark guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/dgx-spark.md).

### Optional: faster training on NVIDIA GPUs

On an NVIDIA GPU, PyTorch Geometric's optional `torch-scatter` extension accelerates the min and max aggregations used in training. Its wheels are built for a specific PyTorch and CUDA version, so install it from PyG's wheel index for your environment, for example for PyTorch 2.9 with CUDA 12.8:

```bash
pip install torch-scatter --only-binary torch-scatter -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

Check your versions with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`, and see [PyG's installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for the available combinations.

## Quickstart

Predict which users will place an order in the next 30 days, using a sample dataset of users and their orders.

```python
from kumo.engine import (Edge, FileConnector, Graph, PredictiveQuery, Table,
                         Trainer, datasets)

files = FileConnector(root_dir=datasets.shop())
users = Table.from_source_table(files["users"], primary_key="user_id")
orders = Table.from_source_table(files["orders"], primary_key="order_id",
                                 time_column="order_time")
graph = Graph(
    tables={"users": users, "orders": orders},
    edges=[Edge(src_table="orders", fkey="user_id", dst_table="users")],
)
snapshot = graph.snapshot()

pquery = PredictiveQuery(
    graph, "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
trainer = Trainer(pquery.suggest_model_plan(snapshot))
trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)

result = trainer.predict(pquery.generate_prediction_table(snapshot),
                         snapshot=snapshot, binary_classification_threshold=0.5)
print(result.frame.head())
```

To use your own data, point `FileConnector` at a directory of CSV or Parquet files instead of `datasets.shop()`.

`result.frame` is a pandas DataFrame with one row per user: `ENTITY`, the predicted label `TARGET_PRED`, and the class probabilities `False_PROB` and `True_PROB`.

`generate_prediction_table(snapshot)` predicts as of the latest `order_time` in the snapshot, so the 30 days start there. To predict from another time, pass `prediction_time=` a timezone-naive `datetime`, read as UTC.

`graph.snapshot()` ingests the tables into an on-disk graph index under the scratch directory, which also holds training runs, model bundles, and predictions. It defaults to `~/.cache/kumo-engine`; to put it on a large local disk, call `kumo.engine.configure(scratch_dir=...)` first.

## Documentation

- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview): prerequisites, GPU sizing, installation, and end-to-end workflows. Its User Code Guide covers connecting data, tables and graphs, training, and prediction.
- [Predictive query examples](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/workflows-and-solutions/predictive-query-examples): common prediction tasks written as predictive queries.

## Support

Report bugs and request features in [GitHub Issues](https://github.com/NVIDIA/kumo-engine/issues). To report a security vulnerability, follow [SECURITY.md](https://github.com/NVIDIA/kumo-engine/blob/main/SECURITY.md) instead of opening an issue.

## License

Kumo Engine is licensed under the [Apache License 2.0](https://github.com/NVIDIA/kumo-engine/blob/main/LICENSE). The wheel also bundles third-party components under their own licenses, listed in [THIRD-PARTY.txt](https://github.com/NVIDIA/kumo-engine/blob/main/THIRD-PARTY.txt).
