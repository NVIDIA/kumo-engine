# Kumo editable installation bundle

> Reference deployment example: this bundle and its infrastructure installer
> are outside the supported Kumo application-chart boundary. Operators own
> the surrounding infrastructure. The supported application chart is the
> bundled `charts/kumo-helm`; its requirements are authoritative.


This bundle contains configuration, the installer, documentation, Kumo charts,
and an unpacked upstream Temporal chart. Obtain a bundle compatible with the
selected application release from the release owner; public distribution
locations still require release verification. The source application chart
and this bundle both currently declare the name `kumo-helm`.

After obtaining and unpacking the bundle, edit `config.yaml` and inspect the
reference recipe for the target environment before running `bash install.sh`.
The installer changes infrastructure; it is not a read-only validation command.

For a bundle distributed through NVIDIA NGC, replace the organization, team,
API key, and version placeholders below. Pull it with Helm, edit one file, and
run one script:

```bash
export NGC_API_KEY='<ngc-api-key>'
helm repo add kumo \
  'https://helm.ngc.nvidia.com/<org>/<team>' \
  --username '$oauthtoken' \
  --password "$NGC_API_KEY"
helm pull kumo/kumo-helm --version '<kumo-version>' --untar
cd kumo-helm
${EDITOR:-vi} config.yaml
bash install.sh
```

Use the same configuration to reconcile just one edited chart:

```bash
bash install.sh platform
bash install.sh temporal
bash install.sh core
```

The bundle chart uses the Kumo release version rather than introducing a
separate stack version. Published chart versions are immutable. The exact
default runtime image references remain with the source they describe. The
chart is a download format; `install.sh` creates the independent platform,
Temporal, Kumo application, and edge releases.

The top-level layout is deliberately small:

```text
kumo-helm/
  Chart.yaml
  README.md
  config.yaml
  install.sh
  charts/
  docs/
  bin/                 # advanced operations and troubleshooting
  renderer/ providers/ stages/ files/ sql/
```

Choose the prerequisite recipe for the target platform:

- [Install Kumo on EKS](docs/install-eks.md)
- [Install Kumo on AKS](docs/install-aks.md)
- [Install Kumo on ARO](docs/install-aro.md)

Those three recipes are reference configurations, outside the supported
application-chart boundary. They state
the cloud prerequisites and complete container inventory before making
changes, then provide the commands for a fresh cluster. If an installation
gate fails, use [troubleshooting by stage](docs/troubleshooting.md).

## Advanced installer behavior

`install.sh` delegates to the advanced installer. Operators who need to
render, diagnose, or resume a bounded stage can invoke it directly:

```bash
bash ./bin/kumo render --values config.yaml --output-dir ./rendered
bash ./bin/kumo check cloud --values config.yaml
bash ./bin/kumo install --from database --through temporal --values config.yaml
```

The complete stage sequence is:

```bash
bash ./bin/kumo check cloud -f config.yaml
bash ./bin/kumo install platform -f config.yaml
bash ./bin/kumo verify platform -f config.yaml
bash ./bin/kumo bootstrap database -f config.yaml
bash ./bin/kumo verify database -f config.yaml
bash ./bin/kumo install temporal -f config.yaml
bash ./bin/kumo verify temporal -f config.yaml
bash ./bin/kumo install core -f config.yaml
bash ./bin/kumo verify core -f config.yaml
```

`install --through platform` and `install --from database --through temporal`
run bounded portions of the same ordered stage functions. Already-converged
stages are safe to retry. The installer always uses `cluster.context`; it
never relies on the current kube context.

The commands used by the recipes are:

- `render`: validate the one-file schema and write the exact leaf-chart
  values without contacting a cluster.
- `check cloud`: verify cloud, cluster, PostgreSQL, storage, identity, image,
  and provider prerequisites before mutation.
- `install`: reconcile all stages in dependency order.
- `conformance`: run the values-driven Kumo prerequisite and health probe.
- `verify gpu-lifecycle`: create every configured GPU tier, verify storage,
  then remove the canaries and wait for pools to return to zero.
- `verify container-policy`: reject any running image outside the Kumo,
  Temporal, NVIDIA, and selected provider allowlist. Run this only on a
  dedicated fresh acceptance cluster because it scans every namespace.
- `status`: report the independent releases.
- `uninstall-tenant`: remove only the tenant release and its runtime cache
  resources.
- `decommission`: remove the tenant, bundled Temporal, platform release,
  and stack-owned capacity.

`bash ./bin/e2e-install --values config.yaml` runs install, conformance,
GPU lifecycle, and container closure in that order.

## Ownership and credentials

The installer reconciles the reference components described below. Operators
supply the prerequisites in the selected provider guide. It adopts
customer-owned PostgreSQL, object storage, workload identities, private
networking, DNS, certificates, registries, and cloud IAM. `decommission`
retains those adopted resources and database data.

Kumo's application-metadata PostgreSQL database, called **StateDB** in
configuration and diagnostics, uses provider-native passwordless
authentication where it exists: AWS IAM on EKS and Microsoft Entra on AKS/ARO,
selected with `postgres.authMode`. Prefer it — no long-lived credential exists,
so there is nothing to rotate or leak. Elsewhere, use password authentication
as described below. Bundled stock Temporal uses one password role for schema
migration and runtime access to `temporal` and `temporal_visibility`.

### StateDB password authentication

The following settings apply directly to the [application chart](https://github.com/kumo-ai/kumo/blob/master/helm/kumo-helm/docs/configuration.md),
not the bundle's top-level `config.yaml`. The bundle supports EKS, AKS, and
ARO and renders their provider identity settings; this section does not add
another stack provider.

Set `postgres.dsnSecret` instead of `authMode` for self-managed PostgreSQL, or
anywhere cloud workload identity is unavailable — including GCP, where
`authMode` offers no option. The two are mutually exclusive; the application
rejects a config that supplies both.

The Secret you create holds one key, named by `postgres.dsnSecret.key`, whose
value is the connection JSON the application parses. `username`, `host`, and
`database` are required; the `tls` block is optional, but if present `sslmode`
must be `verify-ca` or `verify-full`.

```json
{
  "username": "kumo",
  "password": "...",
  "host": "postgres.example.com",
  "port": "5432",
  "database": "kumo",
  "tls": {
    "sslmode": "verify-full",
    "sslrootcert": "/opt/kumo/certs/aws-rds-roots.pem"
  }
}
```

```bash
kubectl -n <ns> create secret generic kumo-statedb --from-file=statedb=./conn.json
```

The chart mounts that Secret and passes only its key name in the deploy config,
so the password never appears in values, in a ConfigMap, or in Helm metadata.

`sslrootcert` is a path libpq opens inside the container, not inline PEM. Two
CAs are baked into the images: `/opt/kumo/certs/aws-rds-roots.pem` for AWS RDS,
and `/etc/ssl/certs/ca-certificates.crt` for Azure Database for PostgreSQL or
any publicly-trusted certificate. Those need nothing further.

If the database's server CA is not in either trust bundle, provide its CA
explicitly. For example, when connecting directly to a Cloud SQL instance
using its instance CA, put that certificate in a Secret and name it in
`postgres.caSecret`:

```bash
gcloud sql instances describe INSTANCE \
  --format='value(serverCaCert.cert)' > ca.pem
kubectl -n <ns> create secret generic cloudsql-ca --from-file=ca.pem=./ca.pem
```

```yaml
postgres:
  authMode: ""
  dsnSecret: {secretName: kumo-statedb, key: statedb}
  caSecret: {secretName: cloudsql-ca, key: ca.pem}
```

`caSecret.key` selects which key of your Secret holds the bundle; the chart
remaps it to a fixed filename, so a cert-manager Secret keyed `ca.crt` works
unchanged. It is mounted on the chart's workloads and on the pods the pool
controller creates, so set `sslrootcert` to exactly
`/var/run/secrets/kumo/ca/ca.pem`. Helm cannot verify that, because the path
lives inside your Secret; preflight reads it and fails the install with
`POSTGRES_TLS_CA` before any workload starts.

Keeping the certificate in its own Secret lets it rotate without rewriting the
credential. A proxy placed only beside chart-managed workloads would not
serve the runtime pool pods, which the controller creates separately. The
configuration above uses a direct database connection; any operator-provided
proxy topology must also be reachable from those pool pods.

With `bootstrapTemporal=true`, the installer:

1. creates the named Kubernetes Secret when it is absent;
2. runs a short-lived Job in the Temporal namespace that mints its own AWS IAM
   or Azure Entra PostgreSQL token from the workload identity named by
   `cloud.postgres.admin.annotations`;
3. creates the Temporal role and databases from that Job; and
4. passes only the Secret name and key to the Temporal chart.

The generated password is never accepted in values, passed on a command line,
rendered into a manifest, written to a ConfigMap, or stored in Helm metadata;
the Job reads it from the Secret and the admin token never leaves the pod. The
installer host therefore needs no PostgreSQL client and no network route to the
database, which is what makes the bootstrap work against PostgreSQL that only
the cluster can reach. Set `bootstrapTemporal=false` only when an enterprise
secret process has already created the matching Secret and database role.

Bootstrapping needs PostgreSQL 15 or newer, because it hands each Temporal
database's `public` schema to the Temporal role and earlier releases reserve
that for a superuser. StateDB and a customer-managed Temporal database carry no
such floor; the example stack also accepts 14 for both.

The default recipes terminate HTTPS at the provider edge. Bundled Temporal
installs an ingress NetworkPolicy; operators must supply Kumo namespace
policies appropriate to their CNI and routing. The application chart does not
install them. The recipes do not mesh-enroll Kumo or Temporal workloads. EKS
uses an ALB with ACM, AKS uses private Application Gateway v2
with the managed AGIC add-on, and ARO routes directly from the built-in
OpenShift router to the Kumo UI.

On first core installation, the stack creates the stable tenant Fernet/API-key
Secret when absent and adopts an enterprise-precreated Secret without rotating
it. The platform recipes separately create the temporary username/password
login Secret without putting its value in the values file.

## Capacity and lifecycle

The CPU pool carries Kumo's control plane, whose six Deployments are meant to
share a single node and request 4.6 CPU and approximately 14 GiB between them.
On EKS it defaults to `m7a.2xlarge` under a 32-vCPU ceiling, preserving the previous
pool's ceiling of four nodes. One node holds the default control plane; the
remaining ceiling leaves room for install-time hook Jobs, stack workloads that
select no pool, and a replacement Karpenter pre-spins before gracefully
retiring a drifted or consolidated node.
Kumo's Deployment rollouts use `maxSurge: 0`, so they do not require an extra
application pod during an upgrade.
`cluster.providers.eks.cpuNodePool` resizes it, and every instance type named
there must still hold that whole aggregate once the kubelet reservation and the
node's DaemonSets are taken out. The reference configuration uses an
8-vCPU/32-GiB type. The three cache tiers are one-T4 each:

| Tier | EKS | AKS/ARO | Cache PVC |
|---|---|---|---|
| Small | `g4dn.xlarge` | `Standard_NC4as_T4_v3` | 256 GiB |
| Medium | `g4dn.2xlarge` | `Standard_NC8as_T4_v3` | 512 GiB |
| Large | `g4dn.8xlarge` | `Standard_NC16as_T4_v3` | 1024 GiB |

`ssdGiB` is the managed cache PVC request and node-selection label. It is not
a VM root disk or local NVMe/temp-disk size. GPU pools have a minimum of zero.
AKS uses scale-down mode `Delete`; EKS Karpenter terminates empty instances;
ARO MachineAutoscalers return the adopted MachineSets to zero. The platform
StorageClasses use `reclaimPolicy: Delete`.

The class behind that PVC is a performance decision, not a detail: the
DiskGraph index is built on and served from it, so a network-attached class
caps index build and query I/O at the volume's throughput ceiling while the
instance's own NVMe sits idle. `storage.diskgraph.storageClass` names the class
and says who owns it. Unset, each platform keeps the class it already used —
`ebs-gp3-xfs` on EKS, `managed-csi-xfs` on Azure — created and owned by the
platform release. With `create: false` the stack adopts a class the cluster
already provides, the same way it adopts the database, object storage and the
CSI drivers, which is how the index is kept on each node's local NVMe through a
customer-installed local-volume driver. Such a driver also needs the raw disks,
so that install writes `cluster.providers.eks.cacheGpuInstanceStorePolicy: ""`
— with the quotes, since a bare key is YAML null and the setting is required,
which makes that spelling a validation error rather than a silent `RAID0` on
the very nodes it was meant to leave raw. `RAID0`, the shipped value, gives the
NVMe to kubelet instead. The setting reaches the cache tiers only. The
ephemeral overflow pool runs on its own EC2NodeClass, always `RAID0`, and takes
no setting: its jobs write to an `emptyDir` on kubelet's filesystem, so raw
devices would leave them on the AMI-backed kubelet filesystem and leave jobs
that exceed its capacity Pending.
Handing the NVMe to a local-volume driver also moves kubelet and containerd
onto the cache node's root disk, which the AMI sizes at 20 GiB — too small for
the GPU image, which fills it and puts the node into `DiskPressure`, where it
rejects every pod including the DiskGraph daemon it exists to run.
`cluster.providers.eks.cacheGpuRootVolumeGiB` sizes that disk and defaults to
200 GiB, which also has to cover the cache daemon's ingest spill — raise it for
large ingests. It applies whichever way the policy above is set, so that
changing the policy has one consequence rather than two.
Whoever provides the class, it must bind `WaitForFirstConsumer`, format xfs and
reclaim `Delete`; verification asserts those properties rather than the
identity of the driver behind them.
`storage.diskgraph.capacityTracking` controls conformance checks, not runtime
controller admission. The reference cloud-disk configuration uses `advisory`;
`required` requires the CSI driver to advertise storage-capacity support.
Missing capacity objects are advisory. The explicitly enabled pool canary
tests actual PVC binding and scheduling. The stack installs no local-volume
driver and prepares no devices for one.

Those GPU pools are labelled and not tainted, so Kumo and bundled Temporal
workloads reach CPU capacity through `cluster.cpuNodeSelector` rather than
landing on a GPU node and holding it up. Unset, it keeps each platform's
default: `kumo.ai/node-pool: cpu` on EKS and AKS, nothing on ARO. The fresh ARO
recipe labels its general MachineSet templates and current workers
`kumo.ai/node-pool: cpu` and sets the
selector explicitly. Set on EKS or AKS it is not advisory — EKS stamps it on
the Karpenter CPU NodePool and AKS on the CPU agent pool it reconciles, so the
capacity the stack creates always satisfies it — and it has to keep
`kumo.ai/node-pool: cpu`, the one label those GPU pools never carry, so that no
GPU pool can satisfy the selector however the rest is written. ARO creates no
CPU capacity, so `check cloud` tests the selector against the cluster instead:
it refuses one that no existing node carries, and one that the configured GPU
MachineSets' own labels would also satisfy. That check reads labels declared
before any GPU node exists, so it catches a plainly wrong selector rather than
proving a right one. Whichever platform, the selector may not be built from
labels every node carries — `kubernetes.io/os`,
`topology.kubernetes.io/region` and the like say where a node is, not what it
is for, and select GPU capacity just as readily.

The Helm default sets a one-hour idle eviction deadline for local SSD
snapshot copies after their final lease, controlled by
`kumoConfig.diskgraph.snapshotCacheTtlSeconds`. Retention sweeps perform
eviction; retiring an emptied worker removes its daemon and PVC, and the
StorageClass reclaim policy removes the cloud disk. The explicit GPU lifecycle
gate disables caching and cleans up immediately.

Helm release state and the customer's edited source tree are the installation
record. The installer does not create a second stack-version ConfigMap.
