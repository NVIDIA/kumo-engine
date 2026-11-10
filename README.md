<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kumo-engine-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/kumo-engine-light.svg">
    <img src="assets/kumo-engine-light.svg" width="125" alt="Kumo Engine cube with a green foundation layer">
  </picture>
</p>

<h1 align="center">Kumo Engine</h1>

<p align="center">
  <a href="https://pypi.org/project/kumo-engine/"><img src="https://img.shields.io/pypi/v/kumo-engine?color=76B900" alt="PyPI version"></a>
  <a href="https://pypi.org/project/kumo-engine/"><img src="https://img.shields.io/pypi/pyversions/kumo-engine?color=76B900" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/NVIDIA/kumo-engine?color=76B900" alt="License"></a>
  <a href="https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview"><img src="https://img.shields.io/badge/docs-docs.nvidia.com-76B900" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview"><b>Documentation</b></a> ·
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="#what-makes-kumo-engine-fast"><b>Performance</b></a> ·
  <a href="examples/kumo-example-notebooks/README.md"><b>Examples</b></a>
</p>

**Kumo Engine is an accelerator for context generation over structured data.** It stores your tables and their relationships as a graph on disk and, for each entity, samples the related records up to its prediction time. Describe what to predict in one query, and Kumo Engine uses that context to train and run predictive models directly on your relational data.

<picture>
  <source media="(max-width: 600px)" srcset="assets/model-context-mobile.svg">
  <img src="assets/model-context.svg" alt="Connected tables become model context: customer and order tables define relationships, matching keys connect individual records, and selected historical records form model inputs. Future outcomes stay outside the inputs.">
</picture>

*Example: predicting whether a customer will order in the next 30 days. Keys link each customer to their orders, and the model sees only orders placed before the prediction time.*

## Key Capabilities

- **Scales beyond memory:** Kumo Engine samples context from graphs with billions of records on one machine, even when the graph is larger than memory.
- **Training data from one query:** A predictive query generates labeled training examples at points in each entity's history. Each example sees only records timestamped at or before its point in time, with no hand-written label logic or point-in-time joins.
- **Trains directly on relational data:** A built-in graph neural network (GNN) learns from the related records without requiring hand-built features. It handles classification, regression, forecasting, and recommendation tasks.
- **End-to-end workflow:** Hyperparameter search, explanations, embedding export, and batch prediction are built in. Kumo Engine also works with MLflow, Databricks, SageMaker, and Snowflake to log each run, version models so scheduled prediction jobs use the one you promote, and monitor prediction drift.
- **Runs locally or in your data platform:** Run Kumo Engine in Python on Linux, macOS, or Windows (WSL2), or in a Snowflake or Databricks notebook. Read data from, and write predictions to, local files, cloud storage, and data warehouses.
- **Works with coding agents:** The bundled agent skill teaches your coding agent the Kumo Engine API, so it can build, train, and run models on your data.

## Part of NVIDIA Kumo

Kumo Engine is part of [NVIDIA Kumo](https://docs.nvidia.com/sdgm/about-nvidia-kumo/overview), a family of open models and accelerated libraries for predictive AI on tabular, relational, and time-series data, and a core part of the [NVIDIA Agent Knowledge Foundation](https://developer.nvidia.com/topics/ai/agent-knowledge-foundation). To make predictions without training a model, use the pretrained Kumo foundation models (Kumo Tabular, Kumo Relational, Kumo Forecast, and Kumo Anomaly) with [Structured Data Models](https://github.com/NVIDIA/structured-data-models) or [Kumo-TS](https://github.com/NVIDIA/Kumo-TS).

## Installation

Install Kumo Engine from PyPI:

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

Choose your hardware and runtime environment:

- **Hardware:** [Prerequisites](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/prerequisites) · [GPU sizing](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/gpu-sizing)
- **Setup guides:** [Linux](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/linux-machine-deployment) · [macOS](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/macos) · [Windows (WSL2)](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/windows-wsl2) · [Snowflake notebook](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/snowflake-notebook) · [Databricks notebook](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/databricks-notebook)
- **Platform details:** the [package README](packages/kumo-engine/README.md#installation) covers ARM64 Linux setup, including Faiss for recommendation tasks.

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

Next steps:

- Follow a complete walkthrough in a [Google Colab](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/google-colab-notebook), [Snowflake](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/snowflake-notebook), or [Databricks](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/databricks-notebook) notebook.
- Explore more tasks in [Predictive Query Examples](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/workflows-and-solutions/predictive-query-examples), and browse the [example collection](examples/kumo-example-notebooks/README.md) for industry notebooks, standalone scripts, and synthetic-data generators.
- To have a coding agent build the workflow for you, install the agent skill by running `python -m kumo.engine.skills`. See [Get Started with Agents](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/agents).

## What Makes Kumo Engine Fast

Kumo Engine converts your tables into a graph once, at ingest. Its on-disk layout and its sampler are designed together for high-performance sampling:

- **Links resolved at ingest:** Minimal perfect hashing maps every primary key to a compact integer ID, and each foreign key is resolved once. Following a link while sampling is a direct lookup rather than a join.
- **Bounded context:** By default, each hop samples at most a set number of related records, so each entity's context has a fixed maximum size, however large the graph grows.
- **Contiguous recent history:** Each record's links are grouped by relationship and stored newest first. By default, sampling takes the most recent links: a binary search finds the prediction time, and the sampler reads the links that follow in order rather than gathering scattered rows.
- **Ordered reads for large batches:** Sampling expands a batch one hop at a time. For large batches, the sampler sorts each hop's records by their position on disk so reads move through each file in order, and spreads the work across CPU threads.
- **Larger than memory:** The graph files are read-only and memory-mapped. The operating system keeps frequently read pages in RAM and reads the rest from disk on demand, so the graph does not have to fit in memory, and many threads read it in parallel without locks.

See the [benchmark report](https://docs.nvidia.com/sdgm/kumo-engine/resources/benchmarks) for measured sampling and end-to-end results, with hardware, cache conditions, and methodology.

## Documentation

| Goal | Start Here |
| --- | --- |
| Connect data, define graphs, train, and predict | [User Code Guide](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/set-up-a-graph/tables) |
| Configure sampling and training | [Model Plan Guide](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/train-a-model/model-plan-guide) |
| Run batch predictions in production | [Production Workflow](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/deploy-a-model/overview) |
| Use the in-process Python API | [Local Python interface](packages/kumo-engine/README.md), imported as `kumo.engine` |
| Version, promote, and monitor models | [MLOps guide](packages/kumo-engine/docs/mlops.md) |
| Check compatibility before upgrading | [Release notes](https://docs.nvidia.com/sdgm/kumo-engine/resources/release-notes) |
| Explore the source code | [Architecture](ARCHITECTURE.md) and [engineering docs](docs/README.md) |

## Kumo Studio (Coming Soon)

> [!NOTE]
> This Kumo Studio information is prerelease and is not yet supported. Stay tuned for the Docker and Kubernetes releases.

Kumo Studio is a self-hosted application with a web UI, a REST API, and a Python client (`kumoai_studio`). Its source, [Docker Compose topology](docker/compose/README.md), and [Helm charts](helm/README.md) are in this repository. See the [preliminary documentation](https://docs.nvidia.com/sdgm/kumo-engine/kumo-studio-guide-coming-soon/preliminary-documentation/installation/requirements).

## Development and Support

Report bugs and request features in [GitHub Issues](https://github.com/NVIDIA/kumo-engine/issues). To report a security vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

This project is currently not accepting contributions.

<details>
<summary>Development guides and source-checkout validation</summary>

- [Setup and prerequisites](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/setup.md)
- [Testing and validation tiers](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/testing.md)
- [Public connector tests](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/public-tests.md)
- [Generated artifacts](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/generated-artifacts.md)
- [UI development](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/ui.md)
- [Agent instructions and setup](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/agents.md)
- [Review requirements](https://github.com/NVIDIA/kumo-engine/blob/main/docs/development/review.md)
- [In-tree local SDK validation example](https://github.com/NVIDIA/kumo-engine/blob/main/examples/sdk-local/README.md)

</details>

## License

Kumo-owned source is licensed under the [Apache License 2.0](https://github.com/NVIDIA/kumo-engine/blob/main/LICENSE). Vendored and incorporated third-party code retains its applicable licenses and attribution.

<details>
<summary>Component licenses and third-party attribution</summary>


Apache-licensed Python subprojects retain a copy of the root license at each package build root so it accompanies independently built wheels and source distributions. The nested component license files are:

| Component | License | License file |
| --- | --- | --- |
| `kumo-api` | Apache-2.0 | [packages/kumo-api/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-api/LICENSE) |
| `kumo-bigquery` | Apache-2.0 | [packages/kumo-bigquery/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-bigquery/LICENSE) |
| `kumo-core` | Apache-2.0 AND MIT | [packages/kumo-core/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-core/LICENSE), [third-party notice](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-core/THIRD-PARTY.txt) |
| `kumo-databricks` | Apache-2.0 | [packages/kumo-databricks/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-databricks/LICENSE) |
| `kumo-engine` (formerly `kumoai-local`) | Apache-2.0 | [packages/kumo-engine/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-engine/LICENSE) |
| `kumo-gcs` | Apache-2.0 | [packages/kumo-gcs/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-gcs/LICENSE) |
| `kumo-glove-6b-50d` | Apache-2.0 AND PDDL-1.0 | [tools/glove-6b-50d/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/tools/glove-6b-50d/LICENSE), [third-party notice](https://github.com/NVIDIA/kumo-engine/blob/main/tools/glove-6b-50d/THIRD-PARTY.txt) |
| `kumo-pql` | Apache-2.0 | [packages/kumo-pql/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-pql/LICENSE) |
| `kumo-s3` | Apache-2.0 | [packages/kumo-s3/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-s3/LICENSE) |
| `kumo-sdk` (`kumoai-studio`) | Apache-2.0 AND BSD-3-Clause | [packages/kumo-sdk/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-sdk/LICENSE), [third-party notice](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-sdk/THIRD-PARTY.txt) |
| `kumo-snowflake` | Apache-2.0 | [packages/kumo-snowflake/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-snowflake/LICENSE) |
| Vendored Temporal Helm chart | MIT | [helm/kumo-temporal/charts/temporal/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/helm/kumo-temporal/charts/temporal/LICENSE) |
| Vendored PtrHash | MIT | [packages/diskgraph/diskgraph-ptrhash/LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/diskgraph/diskgraph-ptrhash/LICENSE) |
| Fabric Toolbox | MIT | [3rdparty/notices/fabric-toolbox-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/fabric-toolbox-LICENSE) |
| PyTorch Forecasting | MIT | [3rdparty/notices/pytorch-forecasting-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/pytorch-forecasting-LICENSE) |
| scikit-learn | BSD-3-Clause | [3rdparty/notices/scikit-learn-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/scikit-learn-LICENSE) |
| spaCy | MIT | [3rdparty/notices/spacy-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/spacy-LICENSE) |
| Starlette | BSD-3-Clause | [3rdparty/notices/starlette-LICENSE.md](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/starlette-LICENSE.md) |
| Temporal Python samples | MIT | [3rdparty/notices/temporal-samples-python-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/temporal-samples-python-LICENSE) |
| torchtext | BSD-3-Clause | [3rdparty/notices/torchtext-LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/3rdparty/notices/torchtext-LICENSE) |
| react-codemirror | MIT | [ui/third_party/npm-licenses/react-codemirror-4.25.10.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/ui/third_party/npm-licenses/react-codemirror-4.25.10.LICENSE) |
| react-spacer | MIT | [ui/third_party/npm-licenses/react-spacer-1.3.0.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/ui/third_party/npm-licenses/react-spacer-1.3.0.LICENSE) |
| format | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/format-0.2.2.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/third_party/npm-licenses/format-0.2.2.LICENSE) |
| heap | PSF-2.0 | [packages/kumo-ui-catalog/third_party/npm-licenses/heap-0.2.5.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/third_party/npm-licenses/heap-0.2.5.LICENSE) |
| pathfinding | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/pathfinding-0.4.18.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/third_party/npm-licenses/pathfinding-0.4.18.LICENSE) |
| size-sensor | ISC | [packages/kumo-ui-catalog/third_party/npm-licenses/size-sensor-1.0.3.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/third_party/npm-licenses/size-sensor-1.0.3.LICENSE) |
| toggle-selection | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/toggle-selection-1.0.6.LICENSE](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/third_party/npm-licenses/toggle-selection-1.0.6.LICENSE) |

Third-party attribution is recorded in the [repository notice](https://github.com/NVIDIA/kumo-engine/blob/main/THIRD-PARTY.txt), [Python distribution notice](https://github.com/NVIDIA/kumo-engine/blob/main/KUMO-THIRD-PARTY.txt), [UI catalog notice](https://github.com/NVIDIA/kumo-engine/blob/main/packages/kumo-ui-catalog/THIRD-PARTY.txt), and [Helm stack notice](https://github.com/NVIDIA/kumo-engine/blob/main/helm/kumo-stack/THIRD-PARTY.txt).

</details>
