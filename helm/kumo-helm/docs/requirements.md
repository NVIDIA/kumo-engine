# Application requirements

These requirements belong to the Kumo application chart, independently of any
reference installer. Use the [configuration contract](configuration.md) and
[values](../values.yaml) for exact field names. The chart does not provision
PostgreSQL, Temporal servers, object storage, a CSI driver/StorageClass, node
pools, workload identity, ingress, or a service mesh.

| Requirement | Operator supplies | Application integration |
| --- | --- | --- |
| Images | Accessible registry, selected release/tag, and pull credentials when needed | `image.*`; image-pull configuration also reaches worker service accounts |
| Metadata database | PostgreSQL database, role, network/TLS access, and authentication | `postgres.*`; StateDB reads its connection from workload identity or a mounted Secret |
| Workflow service | Reachable Temporal endpoint, namespace, and applicable TLS/client credentials | `temporal.*`; namespace registration is enabled by default and requires permission |
| Durable data | Object-storage location and read/write permissions appropriate to its use | `kumoConfig.storage.*`, provider identity, and the documented storage contract |
| Worker capacity | Suitable GPU/CPU nodes, device/runtime setup, scheduling labels/taints, and capacity | `kumoConfig.diskgraph.nodeTypes`, pool scheduling/resource settings |
| Worker cache | A CSI driver and StorageClass satisfying the [cache storage requirements](#csi-storageclass-and-scratch-storage) | `kumoConfig.diskgraph.storageClass`; pool controller creates runtime storage resources |
| Kubernetes access | Target namespace and permissions to create the chart's resources and bind its roles | Namespaced service accounts/RBAC; hooks have their own access requirements |
| Application identity | Login configuration and required Secret objects | `auth.*` and stable `tenantSecretKey` |
| Routing and network | DNS/TLS/ingress as needed; connectivity among application, workers, and external services | `exposure.*`, Services, operator network policies/mesh configuration |

## Authentication and secrets

The leaf chart defaults `auth.enabled` to `false`. Select and configure the
intended login mode explicitly for an installation; the example stack's
choices are not the leaf chart's defaults. OIDC and password login refer to
pre-existing Secrets. Production requires a stable valid Fernet key through
`tenantSecretKey`; changing it affects encrypted credentials and API-key hashes.

PostgreSQL password authentication and workload-identity authentication are
alternative configurations. Include the server's CA chain where it is not
already trusted by the image. Credentials and CA mounts must reach dynamically
created workers as well as chart-rendered pods.

## Temporal

`temporal.registerNamespace` defaults to `true`. Set it to `false` when the
namespace is managed separately and supply an existing usable namespace.
The runtime supports configured client mTLS material. An endpoint string alone
does not prove that a managed Temporal offering is compatible: validate the
namespace, authentication, TLS, and required API operations with the selected
service and release.

## CSI, StorageClass, and scratch storage

The operator installs and maintains the CSI driver, its provider permissions,
and the StorageClass. Set `kumoConfig.diskgraph.storageClass` to that existing
class; the application chart does not create it. The pool controller creates
cache PVCs at runtime, so a successful Helm render does not verify storage.

The cache storage contract requires:

- `volumeBindingMode: WaitForFirstConsumer`, so binding follows worker placement,
  and `reclaimPolicy: Delete`, so deleted cache claims release their volumes.
- A CSI provisioner supporting `ReadWriteOnce` filesystem volumes shared by
  the cache daemon and activity pods on the same node.
- Driver registration in `CSINode` on every eligible cache node, including GPU
  nodes with taints. Match the `kumo.ai/diskgraph-ssd-size` and
  `kumo.ai/diskgraph-gpu-count` labels to the configured
  `kumoConfig.diskgraph.nodeTypes`, and supply enough provisionable storage
  for those tiers. Cache nodes also require
  `kumo.ai/diskgraph-pool-type: cache`; overflow nodes require
  `kumo.ai/diskgraph-pool-type: ephemeral`.
- With `kumoConfig.diskgraph.storageCapacityTracking: required` (the default),
  conformance requires a `CSIDriver` advertising `spec.storageCapacity: true`.
  Missing `CSIStorageCapacity` objects are advisory, including with pools
  scaled to zero. Use `advisory` for a compatible provisioner without capacity
  tracking. This setting controls conformance checks, not runtime controller
  admission; the explicitly enabled pool initialization canary validates
  actual PVC binding and scheduling.

The [CSI conformance helpers](../conformance/csi_contract.sh) check class/driver
settings and node registration. Driver-specific installation and filesystem
parameters are covered by the [reference deployment guides](https://github.com/kumo-ai/kumo/tree/master/helm/kumo-stack);
those examples do not extend the application support boundary.

Databackend ingest scratch is configured separately through
`workloads.databackend.ingestScratch`. Its default `emptyDir` uses node disk;
size the node and `workloads.databackend.resources.requests.ephemeral-storage`
for ingestion. To use a generic ephemeral PVC, set `type: ephemeral`,
`storageClass`, and `size`, using a suitable `WaitForFirstConsumer` CSI class.
See [values.yaml](../values.yaml) for scratch limits and volume permissions.
Neither ingest scratch nor cache volumes replace the configured durable object
storage.

## Runtime-created resources

Review `pool/` as well as chart manifests. Pool workers require their configured
storage, identity, secret mounts, labels, and scheduling constraints.
`global.podAnnotations` is not propagated to runtime pool pods; arrange required
mesh/other admission behavior through operator-controlled mechanisms.
See [resource inheritance](pool-resources.md) and
[toleration rollout](pool-tolerations.md).

The configured worker soft file-descriptor limit requires a sufficiently high
container-runtime hard limit. Helm cannot raise that hard limit. GPU model,
CUDA/driver, memory, and storage sizing must be validated for the selected
application release and workload; this document does not infer a production
support matrix from development fallbacks.

## Validation and compatibility

Installation preflight checks only part of this contract: inspect its
[implementation](../templates/) and [conformance tools](https://github.com/kumo-ai/kumo/tree/master/helm/kumo-helm/test/conformance)
for the exact checks. Passing a local render or TCP check is not a complete
installation test. Run any live conformance probes only against explicitly
authorized infrastructure.

Record the application image/chart version and the tested external-service
versions with each release. Versions pinned by a reference example describe
that example; they are not a general compatibility guarantee.
