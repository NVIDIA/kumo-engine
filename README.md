<p align="center">
  <img height="180" src="https://kumo-sdk-public.s3.us-west-2.amazonaws.com/kumo-logo-pink.svg" />
</p>

---

![Full Tests](https://github.com/kumo-ai/kumo/actions/workflows/testing.yml/badge.svg?branch=master) ![UI Tests](https://github.com/kumo-ai/kumo/actions/workflows/testing_ui.yml/badge.svg?branch=master) ![Linting](https://github.com/kumo-ai/kumo/actions/workflows/linting.yml/badge.svg?branch=master) ![Install](https://github.com/kumo-ai/kumo/actions/workflows/install.yml/badge.svg?branch=master)

# Kumo

Kumo applies graph machine learning to relational data. Define tables and their
relationships, express a prediction task with a predictive query, and train a
model for batch prediction.

This repository contains the Python interfaces, core execution engine, REST
service, web UI, connectors, and deployment definitions. Start with the
[architecture](ARCHITECTURE.md) or the [documentation index](docs/README.md).

## Choose an interface

| Workflow | Source and documentation |
| --- | --- |
| Execute through Python in-process | [Local Python interface](packages/kumo-engine/README.md), imported as `kumo.engine` |
| Use the service from Python | [Service client](packages/kumo-sdk/README.md), imported as `kumoai_studio` |
| Use REST and the web UI on a single host | [Compose topology](docker/compose/README.md) |
| Deploy the application on Kubernetes | [Application Helm chart](helm/kumo-helm/README.md) |
| Explore example infrastructure deployments | [Helm reference examples and support boundary](helm/README.md) |

## Local SDK example

[Run the in-tree SDK example](examples/sdk-local/README.md).

## Development

- [Setup and prerequisites](docs/development/setup.md)
- [Testing and validation tiers](docs/development/testing.md)
- [Public connector tests](docs/development/public-tests.md)
- [Generated artifacts](docs/development/generated-artifacts.md)
- [UI development](docs/development/ui.md)
- [Agent instructions and setup](docs/development/agents.md)
- [Review requirements](docs/development/review.md)

For a documentation problem, report the affected path and the behavior you
observed in a repository issue. Keep credentials and private deployment details
out of reports.

## Contributing

This project is currently not accepting contributions.

## License

Kumo-owned source is licensed under [Apache-2.0](LICENSE). Vendored and incorporated third-party code retains its applicable licenses and attribution.

Apache-licensed Python subprojects retain a copy of the root license at each package build root so it accompanies independently built wheels and source distributions. The nested component license files are:

| Component | License | License file |
| --- | --- | --- |
| `kumo-api` | Apache-2.0 | [packages/kumo-api/LICENSE](packages/kumo-api/LICENSE) |
| `kumo-bigquery` | Apache-2.0 | [packages/kumo-bigquery/LICENSE](packages/kumo-bigquery/LICENSE) |
| `kumo-core` | Apache-2.0 AND MIT | [packages/kumo-core/LICENSE](packages/kumo-core/LICENSE), [third-party notice](packages/kumo-core/THIRD-PARTY.txt) |
| `kumo-databricks` | Apache-2.0 | [packages/kumo-databricks/LICENSE](packages/kumo-databricks/LICENSE) |
| `kumo-engine` (formerly `kumoai-local`) | Apache-2.0 | [packages/kumo-engine/LICENSE](packages/kumo-engine/LICENSE) |
| `kumo-gcs` | Apache-2.0 | [packages/kumo-gcs/LICENSE](packages/kumo-gcs/LICENSE) |
| `kumo-glove-6b-50d` | Apache-2.0 AND PDDL-1.0 | [tools/glove-6b-50d/LICENSE](tools/glove-6b-50d/LICENSE), [third-party notice](tools/glove-6b-50d/THIRD-PARTY.txt) |
| `kumo-pql` | Apache-2.0 | [packages/kumo-pql/LICENSE](packages/kumo-pql/LICENSE) |
| `kumo-s3` | Apache-2.0 | [packages/kumo-s3/LICENSE](packages/kumo-s3/LICENSE) |
| `kumo-sdk` (`kumoai-studio`) | Apache-2.0 AND BSD-3-Clause | [packages/kumo-sdk/LICENSE](packages/kumo-sdk/LICENSE), [third-party notice](packages/kumo-sdk/THIRD-PARTY.txt) |
| `kumo-snowflake` | Apache-2.0 | [packages/kumo-snowflake/LICENSE](packages/kumo-snowflake/LICENSE) |
| Vendored Temporal Helm chart | MIT | [helm/kumo-temporal/charts/temporal/LICENSE](helm/kumo-temporal/charts/temporal/LICENSE) |
| Vendored PtrHash | MIT | [packages/diskgraph/diskgraph-ptrhash/LICENSE](packages/diskgraph/diskgraph-ptrhash/LICENSE) |
| Fabric Toolbox | MIT | [3rdparty/notices/fabric-toolbox-LICENSE](3rdparty/notices/fabric-toolbox-LICENSE) |
| PyTorch Forecasting | MIT | [3rdparty/notices/pytorch-forecasting-LICENSE](3rdparty/notices/pytorch-forecasting-LICENSE) |
| scikit-learn | BSD-3-Clause | [3rdparty/notices/scikit-learn-LICENSE](3rdparty/notices/scikit-learn-LICENSE) |
| spaCy | MIT | [3rdparty/notices/spacy-LICENSE](3rdparty/notices/spacy-LICENSE) |
| Starlette | BSD-3-Clause | [3rdparty/notices/starlette-LICENSE.md](3rdparty/notices/starlette-LICENSE.md) |
| Temporal Python samples | MIT | [3rdparty/notices/temporal-samples-python-LICENSE](3rdparty/notices/temporal-samples-python-LICENSE) |
| torchtext | BSD-3-Clause | [3rdparty/notices/torchtext-LICENSE](3rdparty/notices/torchtext-LICENSE) |
| react-codemirror | MIT | [ui/third_party/npm-licenses/react-codemirror-4.25.10.LICENSE](ui/third_party/npm-licenses/react-codemirror-4.25.10.LICENSE) |
| react-spacer | MIT | [ui/third_party/npm-licenses/react-spacer-1.3.0.LICENSE](ui/third_party/npm-licenses/react-spacer-1.3.0.LICENSE) |
| format | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/format-0.2.2.LICENSE](packages/kumo-ui-catalog/third_party/npm-licenses/format-0.2.2.LICENSE) |
| heap | PSF-2.0 | [packages/kumo-ui-catalog/third_party/npm-licenses/heap-0.2.5.LICENSE](packages/kumo-ui-catalog/third_party/npm-licenses/heap-0.2.5.LICENSE) |
| pathfinding | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/pathfinding-0.4.18.LICENSE](packages/kumo-ui-catalog/third_party/npm-licenses/pathfinding-0.4.18.LICENSE) |
| size-sensor | ISC | [packages/kumo-ui-catalog/third_party/npm-licenses/size-sensor-1.0.3.LICENSE](packages/kumo-ui-catalog/third_party/npm-licenses/size-sensor-1.0.3.LICENSE) |
| toggle-selection | MIT | [packages/kumo-ui-catalog/third_party/npm-licenses/toggle-selection-1.0.6.LICENSE](packages/kumo-ui-catalog/third_party/npm-licenses/toggle-selection-1.0.6.LICENSE) |

Third-party attribution is recorded in the [repository notice](THIRD-PARTY.txt), [Python distribution notice](KUMO-THIRD-PARTY.txt), [UI catalog notice](packages/kumo-ui-catalog/THIRD-PARTY.txt), and [Helm stack notice](helm/kumo-stack/THIRD-PARTY.txt).
