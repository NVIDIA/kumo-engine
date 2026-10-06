# Kumo application chart

This is the supported chart for Kumo application workloads. It renders the
REST/API and UI services, data/ML workflow workers, pool controller, associated
configuration and namespaced access, and installation hooks. The controller
also creates worker resources at runtime.

The chart connects to operator-supplied infrastructure. Read the
[requirements](docs/requirements.md) and [configuration contract](docs/configuration.md)
before preparing values. The [Helm index](https://github.com/kumo-ai/kumo/blob/master/helm/README.md) explains the support
boundary and optional reference examples.

## Prepare and validate

1. Provision the required services, storage, worker nodes, identity, and
   namespace access. Select compatible application images and an explicit tag.
2. Prepare a values file using [values.yaml](values.yaml) and
   [values.schema.json](values.schema.json). Supply references to existing
   Secrets rather than putting secret values into committed configuration.
3. Review the rendered manifests locally:

   ```bash
   helm lint helm/kumo-helm -f /path/to/values.yaml
   helm template kumo helm/kumo-helm --namespace YOUR_NAMESPACE -f /path/to/values.yaml
   ```

   Run from the repository root. Rendered manifests can contain sensitive
   values supplied by the operator; do not publish their output indiscriminately.
4. Install only after reviewing the generated resources and prerequisites.
   Rendering alone does not verify service connectivity, storage, or GPU access.

The current values include release-specific image defaults. Supply image
locations you can access; verified public artifacts and version compatibility
must be recorded before a public launch installation is advertised.

See [pool resources](docs/pool-resources.md) and
[pool tolerations](docs/pool-tolerations.md) before changing an existing pool.
