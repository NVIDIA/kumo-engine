# Helm deployment

`kumo-helm` is the supported Kumo application chart. The accompanying stack,
dependency, edge, and provider charts are reference deployment examples outside
that support boundary. Operators are responsible for provisioning, securing,
upgrading, and operating the surrounding infrastructure. Kumo owns the
application's documented integration requirements and validation.

| Component | Role |
| --- | --- |
| [kumo-helm](kumo-helm/README.md) | Supported application workloads and their external-service connections |
| [kumo-stack](kumo-stack/README.md) | Reference composed deployment and editable installation bundle |
| [kumo-temporal](kumo-temporal/README.md) | Reference Temporal deployment configuration |
| [kumo-edge](kumo-edge/README.md) | Reference ingress/edge integration |
| [kumo-platform-eks](kumo-platform-eks/README.md) | Reference EKS infrastructure configuration |
| [kumo-platform-aks](kumo-platform-aks/README.md) | Reference AKS infrastructure configuration |
| [kumo-platform-aro](kumo-platform-aro/README.md) | Reference ARO infrastructure configuration |

Start with the [application requirements](kumo-helm/docs/requirements.md) whether
you use an example or supply infrastructure yourself. Existing example tests
remain useful; their results do not extend support to operating that platform.

## Application chart versus bundle

The source directories distinguish two artifacts that currently both declare
`name: kumo-helm` in `Chart.yaml`. `helm/kumo-helm` is the application chart.
`helm/kumo-stack` is the source for an editable bundle containing charts,
configuration, documentation, and an installer; it declares `type: library`.
Inspect the contents and source before following a download/install recipe.
This documentation change does not rename artifacts or change publication.

Public release locations and tested version compatibility remain release
inputs. A reference example is not evidence that its configured image registry is
publicly accessible.
