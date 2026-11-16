# Troubleshooting by stage

> Reference deployment example, outside the supported Kumo application-chart
> boundary. Operators own the surrounding infrastructure. See the
> [bundle guide](../README.md) for scope and prerequisites.


## Cloud

No Kubernetes or node-pool mutation has occurred. Fix the reported cluster,
StorageClass, PostgreSQL auth/extension, network, identity, or CSI
prerequisite and rerun `kumo check cloud`.

An image-manifest 401/403 means the configured pull Secret cannot read the
exact Kumo tag. Refresh a short-lived ECR token or correct the mirrored
registry credentials before retrying. An Azure GPU-family quota failure reports
non-stack use, configured maximum demand, current limit, and the minimum new
limit; do not proceed by silently substituting a different GPU family.

## Platform

With `storage.diskgraph.storageClass.create: true`, a StorageClass that already
uses the configured name stops the install unless the platform release itself
renders it: a class that merely carries the release's Helm annotations would be
imported into the release and then deleted by `decommission`. Compare its
provisioner, XFS parameter, reclaim policy, and binding mode, then either adopt
it with `create: false` or give the stack's class another name. Remove the
existing one only when no PV uses it. With `create: false` the named class must
already exist, because nothing in the stack creates it. Verification requires
`WaitForFirstConsumer`, `reclaimPolicy: Delete`, and xfs from whichever driver
provides it. Already-converged AKS pools are skipped on retry.

A Kumo pod that stays Pending on a cluster whose install passed usually means
`cluster.cpuNodeSelector` was changed without reinstalling the platform stage:
EKS carries it on the Karpenter CPU NodePool and AKS on the CPU agent pool, so
run `install platform` before `install core` after changing it. On ARO the
selector reaches nothing the stack creates, and `check cloud` refuses a
selector that no node already carries, and one the configured GPU MachineSets would also satisfy.

On AKS, an `ingress-appgw` enablement failure usually means the installer
identity cannot create the add-on's role assignment on Application Gateway.
An Ingress that remains addressless after the add-on is healthy usually means
the selected gateway has no private frontend IP, cannot reach AKS pod
addresses, or the TLS Secret is invalid. The installer refuses to retarget an
already-enabled AGIC add-on because that could disrupt another application.

## Database

An Azure `pg_hba` rejection for a Temporal password role commonly means
password authentication is disabled even when the firewall is correct.
`permission denied for schema public` means the role does not own the public
schema. The bootstrap stage corrects database and schema ownership.

## Temporal

The installer prints the default and visibility schema container logs when
Helm fails. Azure visibility schema requires the `btree_gin` extension to be
allowlisted. Retry `install temporal` after correcting the database; schema
updates are idempotent.

## Core

Core generates its tenant Fernet/API-key Secret when absent, but never
generates image-pull or OIDC Secrets. An existing tenant Secret with the wrong
key name fails before Helm runs and is never overwritten. On failure, inspect
the printed jobs, pods, deployments, and recent namespace events, then rerun
`install core`.

A deployment that is otherwise healthy but fails every local file upload with
`Could not upload the file!` almost always has no bucket CORS policy, or one
that does not name the browser's origin. The browser `PUT`s each part straight
to object storage, so no server-side check sees the failure. The variant
`Error: ETag, Could not upload the file!` means the policy exists but omits
`ETag` from `ExposeHeaders`. Both are fixed by the CORS rule in the `S3 access`
section of `install-eks.md`.

Normal stack installation keeps the costly GPU canary out of core readiness.
`verify gpu-lifecycle` explicitly enables it. A Pending
`dg-pool-canary-*` with an autoscaler `cloudProviderError` is a cloud
quota/capacity problem, not a label mismatch when its selector matches the
rendered pool labels. The command restores the CPU-ready configuration after
the canary completes.

## Acceptance

The canonical `kumo conformance` probe reports `EDGE_HTTPS` separately from
application health. A deployment whose internal UI, REST, API, StateDB, and
isolation checks pass but whose external route returns 404 needs the
chart-managed edge resource, provider ingress, certificate, and DNS checked
in that order.
