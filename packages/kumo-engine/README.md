<!-- This README is also the PyPI project description: keep every link absolute. -->
<p align="center">
  <img height="180" src="https://raw.githubusercontent.com/NVIDIA/kumo-engine/main/ui/src/images/kumo_logo_green.svg" alt="Kumo" />
</p>

# Kumo Engine

**Kumo Engine is an accelerator for context generation over structured data.** It stores your tables and their relationships as a graph on disk and, for each entity, samples the related records up to its prediction time. Describe what to predict in one query, and Kumo Engine uses that context to train and run predictive models directly on your relational data.

Kumo Engine is part of [NVIDIA Kumo](https://docs.nvidia.com/sdgm/about-nvidia-kumo/overview), a family of open models and accelerated libraries for predictive AI on tabular, relational, and time-series data. For an overview of its capabilities and design, see the [Kumo Engine repository](https://github.com/NVIDIA/kumo-engine).

## Installation

```bash
pip install kumo-engine
```

| Requirement | Details |
| --- | --- |
| Python | CPython 3.11 or later |
| Operating system | Linux x86-64 or ARM64 with glibc 2.34 or later, macOS 14 or later on Apple silicon, or Windows through WSL2 (x86-64) |
| CPU | AES and SSE2 instructions on x86-64, or AES and NEON on ARM64 |
| GPU | An NVIDIA GPU for production workloads. The quickstart and other small datasets also run on a CPU or an Apple silicon GPU. |
| Storage | A local NVMe SSD for production workloads. Network-attached volumes, such as Amazon EBS, make sampling much slower. |

The base package reads and writes local CSV and Parquet files. Optional extras add connectors and additional features; combine them as needed, for example `pip install "kumo-engine[snowflake,text]"`.

| Extra | Adds |
| --- | --- |
| `s3` | Amazon S3 |
| `gcs` | Google Cloud Storage |
| `adls` | Azure Data Lake Storage Gen2 |
| `bigquery` | BigQuery |
| `snowflake` | Snowflake, including Snowflake notebooks |
| `databricks` | Databricks, including Databricks notebooks |
| `text` | Encoding for free-text columns, such as descriptions or reviews, with GloVe 6B 50d word vectors |
| `mlflow` | MLflow run tracking and model registry |
| `sagemaker` | SageMaker managed MLflow |

Kumo Engine trains on an NVIDIA GPU or an Apple silicon GPU when one is available, and on the CPU otherwise. To choose the device, pass `accelerator="gpu"`, `"mps"`, or `"cpu"` to `Trainer`.

Recommendation tasks use Faiss for retrieval. On Linux x86-64 and macOS, `pip install kumo-engine` installs Faiss as a dependency. On Linux ARM64, Faiss is optional, and other tasks run without it. To run recommendation tasks there, install CPU Faiss with `pip install "kumo-engine[cpu]"` and pass `accelerator="cpu"` to `Trainer`, or follow the [DGX Spark guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/dgx-spark.md) to build GPU Faiss. The guide also covers CUDA training on ARM64.

On NVIDIA GPUs, the optional `torch-scatter` package speeds up some training workloads. Install the build for your PyTorch and CUDA versions as described in [PyG's installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Quickstart

This example trains a model that predicts which users will place an order in the next 30 days, using a sample dataset of users and orders that ships with the package.

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
training_result = trainer.fit(pquery.generate_training_table(snapshot),
                              snapshot=snapshot)

result = trainer.predict(pquery.generate_prediction_table(snapshot),
                         snapshot=snapshot, binary_classification_threshold=0.5)
print(result.frame.head())
```

When training completes, the example prints the first five predictions:

```text
   ENTITY  TARGET_PRED  False_PROB  True_PROB
0     625         True    0.118336   0.881664
1     631         True    0.110448   0.889552
2     505         True    0.172911   0.827089
3     519        False    0.939077   0.060923
4     752        False    0.939077   0.060923
```

`result.frame` has one row per user, with the predicted label `TARGET_PRED` and the class probabilities `False_PROB` and `True_PROB`. Exact values can differ between runs. To use your own data, point `FileConnector` at a directory of CSV or Parquet files.

`graph.snapshot()` writes the graph to a scratch directory, `~/.cache/kumo-engine` by default, which also holds training runs, model bundles, and predictions. If your home directory is on a network volume, such as an Amazon EBS boot volume, call `kumo.engine.configure(scratch_dir=...)` before taking the snapshot to use a directory on a local NVMe SSD instead.

## Next Steps

### Use a Coding Agent

The bundled agent skill teaches your coding agent the Kumo Engine API, so it can build, train, and run models on your data. To install it for the coding agents on your machine, run:

```bash
python -m kumo.engine.skills
```

In a Databricks notebook, call `kumo.engine.install_skills()` to install it for Genie Code. For Cortex Code in a Snowflake notebook, call `kumo.engine.install_skills("coco", scope="workspace", workspace="<workspace>")`. See [Get Started with Agents](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/agents).

### Visualize Table Relationships

Before training, check your tables, keys, and relationships in an interactive notebook diagram, or export the diagram as SVG, PDF, or Markdown:

```python
graph.visualize()
_ = graph.visualize("graph.pdf", show_cols=True)
```

See [Visualize Table Relationships](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/set-up-a-graph/graphs#visualize-table-relationships).

### Explain a Trained Model

`training_result.explanations` reads the column importance scores and other diagnostics recorded during training:

```python
explanations = training_result.explanations
if explanations.availability["gradient"].available:
    print(explanations.gradient_scores().to_pandas())
```

See the [training explanations guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/explainability.md) for plots, column cohorts, and sampled training context.

### Run in Production

For run tracking, model versioning, and drift monitoring with MLflow, Databricks, SageMaker, or Snowflake, see the [MLOps guide](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/docs/mlops.md).

## Documentation

- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview): prerequisites, setup guides, and end-to-end workflows.
- [User Code Guide](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/set-up-a-graph/tables): connect data, define tables and graphs, train, and predict.
- [Predictive Query Examples](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/workflows-and-solutions/predictive-query-examples): common prediction tasks written as predictive queries.
- [Release notes](https://docs.nvidia.com/sdgm/kumo-engine/resources/release-notes): changes, compatibility, and known issues for each release.

## Support

Report bugs and request features in [GitHub Issues](https://github.com/NVIDIA/kumo-engine/issues). To report a security vulnerability, follow [SECURITY.md](https://github.com/NVIDIA/kumo-engine/blob/main/SECURITY.md) instead of opening an issue.

## License

Kumo Engine is licensed under the [Apache License 2.0](https://github.com/NVIDIA/kumo-engine/blob/main/LICENSE). The wheel also bundles third-party components under their own licenses, listed in [THIRD-PARTY.txt](https://github.com/NVIDIA/kumo-engine/blob/main/THIRD-PARTY.txt).
