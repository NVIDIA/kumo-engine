<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/kumo-engine-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/kumo-engine-light.svg">
    <img src="assets/kumo-engine-light.svg" width="125" alt="Kumo Engine cube with a green foundation layer">
  </picture>
</p>

<h1 align="center">NVIDIA Kumo Engine</h1>

<p align="center">
  <a href="https://pypi.org/project/kumo-engine/"><img src="https://img.shields.io/pypi/v/kumo-engine?color=76B900" alt="PyPI version"></a>
  <a href="https://pypi.org/project/kumo-engine/"><img src="https://img.shields.io/pypi/pyversions/kumo-engine?color=76B900" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/NVIDIA/kumo-engine?color=76B900" alt="License"></a>
  <a href="https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview"><img src="https://img.shields.io/badge/docs-docs.nvidia.com-76B900" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview"><b>Documentation</b></a> ·
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="#benchmarks"><b>Benchmarks</b></a> ·
  <a href="examples/kumo-example-notebooks/README.md"><b>Examples</b></a>
</p>

**NVIDIA Kumo Engine is an open accelerator for context generation over structured data.** Its Rust-based DiskGraph engine indexes connected tables on local SSD and samples time-correct, multi-hop context for each entity, even when the index is larger than RAM. Declare what to predict in one query, and Kumo Engine uses that context to train and run predictive models directly on your relational data.

For example, to predict whether a customer will order in the next 30 days, Kumo Engine connects each customer to their order history and samples only the orders placed before the prediction time as model inputs.

<picture>
  <source media="(max-width: 600px)" srcset="assets/model-context-mobile.svg">
  <img src="assets/model-context.svg" alt="Connected tables become model context: customer and order tables define relationships, matching keys connect individual records, and selected historical records form model inputs. Future outcomes stay outside the inputs.">
</picture>

*Schematic example: the model plan determines sampling, and future outcomes remain outside the model inputs.*

## Key Capabilities

- **Context at scale, past RAM:** On one machine, Kumo Engine samples context from a 50-billion-node, 175.5-billion-edge graph whose 2.6 TiB index exceeds the host's 1 TiB of RAM. Warm sampling throughput stays nearly flat as the graph grows from 1 billion to 50 billion nodes.
- **Time-correct by construction:** Each training example and prediction sees only records timestamped before its prediction time, so future outcomes can't leak into model inputs.
- **Train directly on relational data:** One predictive query, such as `PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id`, defines the labels and the entities to score. A built-in temporal graph neural network learns from the connected records instead of hand-built cross-table features, for classification, regression, forecasting, and recommendation.
- **From experiment to production:** Automatic hyperparameter search, training explanations, embeddings, and reloadable model bundles that score new data without retraining. Track experiments, register models, and monitor drift with MLflow or Snowflake.
- **Run where your data lives:** Run in-process in Python on Linux, macOS, or Windows through WSL2, or in a Snowflake or Databricks notebook, with no Kumo account or API key. Read from and write predictions to local files, Amazon S3, Google Cloud Storage, Azure Data Lake Storage, BigQuery, Snowflake, and Databricks.
- **Agent-ready:** `kumo.install_skills()` teaches Claude Code, Codex, Snowflake Cortex Code, or Databricks Genie Code to build, train, and run Kumo Engine models from a plain-language request.

## Part of NVIDIA Kumo

NVIDIA Kumo is a family of open models, libraries, and tools for predictive AI across tabular, relational, and time-series data, and a core part of the NVIDIA Agent Knowledge Foundation.

- **NVIDIA Kumo Engine (this repository):** for teams that train and run their own predictive models on relational data in files or warehouses, at production scale.
- **NVIDIA Kumo Structured:** pretrained foundation models (Kumo Tabular, Kumo Relational, Kumo Forecast, and Kumo Anomaly) that make predictions without training a model per task. Refer to [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models).

## Installation

| Requirement | Supported |
| --- | --- |
| Python | CPython 3.11 or later |
| Operating system | Linux x86-64 or ARM64 with glibc 2.34 or later, macOS 14 or later on Apple silicon, or Windows through WSL2 (x86-64) |
| Compute | An NVIDIA GPU for production workloads. Small jobs, such as the quickstart, also run on CPU or an Apple silicon GPU. |
| Storage | Local SSD (NVMe or SATA) for the graph index and scratch data |

```bash
pip install kumo-engine
```

The base package reads local CSV and Parquet files. To read from a remote data source, install the matching optional dependency:

| Data Source | Installation |
| --- | --- |
| Amazon S3 | `pip install "kumo-engine[s3]"` |
| Google Cloud Storage | `pip install "kumo-engine[gcs]"` |
| Azure Data Lake Storage Gen2 | `pip install "kumo-engine[adls]"` |
| BigQuery | `pip install "kumo-engine[bigquery]"` |
| Snowflake | `pip install "kumo-engine[snowflake]"` |
| Databricks | `pip install "kumo-engine[databricks]"` |

Refer to [prerequisites](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/prerequisites) and [GPU sizing](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/gpu-sizing) for a compatible runtime and storage capacity. Follow the setup guide for [Linux](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/linux-machine-deployment), [macOS](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/macos), [Windows (WSL2)](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/windows-wsl2), [Snowflake notebooks](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/snowflake-notebook), or [Databricks notebooks](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/databricks-notebook). The [package README](packages/kumo-engine/README.md#installation) covers text columns, Faiss options, and ARM64 Linux.

## Quickstart

Predict which users will place an order in the next 30 days, using a bundled sample dataset of users and their orders. The example is small enough to train on CPU.

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

After the training progress output, it prints the first predictions. Values vary between runs:

```text
   ENTITY  TARGET_PRED  False_PROB  True_PROB
0     625         True    0.118336   0.881664
1     631         True    0.110448   0.889552
2     505         True    0.172911   0.827089
3     519        False    0.939077   0.060923
4     752        False    0.939077   0.060923
```

`result.frame` has one row per user, with the predicted label `TARGET_PRED` and the class probabilities `False_PROB` and `True_PROB`. To use your own data, point `FileConnector` at a directory of CSV or Parquet files instead of `datasets.shop()`.

Next steps:

- Follow a complete walkthrough in a [Google Colab](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/google-colab-notebook), [Snowflake](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/snowflake-notebook), or [Databricks](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/code-examples/databricks-notebook) notebook.
- Explore more tasks in [Predictive Query Examples](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/workflows-and-solutions/predictive-query-examples), and browse the [example collection](examples/kumo-example-notebooks/README.md) for industry notebooks, standalone scripts, and synthetic-data generators.
- To have a coding agent build the workflow for you, install the agent skill with `kumo.install_skills()`. Refer to [Get Started with Agents](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/agents).

## Benchmarks

### Context Generation on a 50-Billion-Node Graph

Time to sample **one million starting entities**, normalized from warm sampling throughput, on a synthetic graph with **50 billion nodes and 175.5 billion edges**:

| Sampling Depth | Kumo Engine | Prepared DuckDB 1.5.6 |
| --- | --- | --- |
| 2 hops | **5 seconds** | 4.0 minutes |
| 3 hops | **19 seconds** | 5.3 minutes |
| 4 hops | **39 seconds** | 6.5 minutes |

Both ran with warm caches on one AWS i4i.32xlarge: 128 vCPUs, 1 TiB RAM, and 30 TB local NVMe. Kumo Engine's 2.6 TiB index exceeded RAM. Timings exclude preparation and model training. Initial preparation took 49.8 minutes for DiskGraph and 40.1 minutes for DuckDB.

### Training on Matched Hardware

On a separate 111-million-node citation graph, Kumo Engine completed 15 training/validation epochs in **41 minutes 14 seconds**, compared with **64 minutes 13 seconds** for matched-model PyG. Test accuracy was **63.26% versus 63.93%**; both used one g5.16xlarge with one A10G GPU.

### Complete Workflow Versus Feature Engineering

From raw Parquet to a trained model on the 50-billion-node dataset, Kumo Engine took **5.2 hours**, compared with **9.9 hours** for a pipeline that engineered features with DuckDB 1.5.5. Each ran on one AWS g6e.48xlarge. The workflows used different model families and training budgets, so this compares complete workflows rather than matched models.

Refer to the [full benchmark report](https://docs.nvidia.com/sdgm/kumo-engine/resources/benchmarks) for experimental scope, preparation costs, memory measurements, and additional comparisons. Results depend on the workload and configuration.

## Documentation

| Goal | Start Here |
| --- | --- |
| Connect data, define graphs, train, and predict | [User Code Guide](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/set-up-a-graph/tables) |
| Configure sampling and training | [Model Plan Guide](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/train-a-model/model-plan-guide) |
| Run batch predictions in production | [Production Workflow](https://docs.nvidia.com/sdgm/kumo-engine/user-code-guide/deploy-a-model/overview) |
| Use the in-process Python API | [Local Python interface](packages/kumo-engine/README.md), imported as `kumo.engine` |
| Track experiments and register models | [MLOps guide](packages/kumo-engine/docs/mlops.md) |
| Check compatibility before upgrading | [Release notes](https://docs.nvidia.com/sdgm/kumo-engine/resources/release-notes) |

For repository internals, refer to the [architecture](ARCHITECTURE.md) and [engineering documentation index](docs/README.md).

## Kumo Studio (Coming Soon)

> [!NOTE]
> This Kumo Studio information is prerelease and is not yet supported. Stay tuned for the Docker and Kubernetes releases.

Kumo Studio is a self-hosted application with a web UI, a REST API, and a Python client (`kumoai_studio`). Its source, [Docker Compose topology](docker/compose/README.md), and [Helm charts](helm/README.md) are in this repository. Refer to the [preliminary documentation](https://docs.nvidia.com/sdgm/kumo-engine/kumo-studio-guide-coming-soon/preliminary-documentation/installation/requirements).

## Releases

Kumo Engine is published to [PyPI](https://pypi.org/project/kumo-engine/) and [GitHub Releases](https://github.com/NVIDIA/kumo-engine/releases). The release type identifies compatible fixes (patch), compatible additions (minor), or breaking changes (major).

## Development and Support

Report bugs or documentation problems in [GitHub Issues](https://github.com/NVIDIA/kumo-engine/issues), including the affected path and observed behavior. Keep credentials and private deployment details out of reports. To report a security vulnerability, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

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
