# Install Kumo on EKS

> Reference deployment example, outside the supported Kumo application-chart
> boundary. Operators own the surrounding infrastructure. See the
> [bundle guide](../README.md) for scope and prerequisites.


This recipe installs one Kumo environment on a fresh standard EKS cluster.
An internal AWS Application Load Balancer (ALB) terminates HTTPS using AWS
Certificate Manager (ACM), and Kumo initially uses email/password login.
Operators supply Kumo namespace NetworkPolicies; the bundled Temporal chart
installs its own ingress policy. No service mesh is installed.

## Prerequisites

Prepare all of the following before starting:

- An amd64 EKS 1.34 cluster, not EKS Auto Mode, with a reachable private API
  endpoint and at least one 4-vCPU/16-GiB system node such as `m7a.xlarge`.
- Active EKS managed add-ons `vpc-cni:v1.22.3-eksbuild.1`,
  `coredns:v1.12.4-eksbuild.18`, and
  `kube-proxy:v1.34.6-eksbuild.17`. Enable VPC CNI network-policy
  enforcement.
- An IAM OIDC provider for the cluster and IAM roles for Karpenter, EBS CSI,
  AWS Load Balancer Controller, and the Kumo `control`, `worker`, and
  `controller` Kubernetes ServiceAccounts.
- A Karpenter EC2 node role. If Kumo images are mirrored to ECR, the
  system-node and Karpenter-node roles must be able to pull that mirror.
- Node network paths to each configured registry, or private registry mirrors
  and VPC endpoints that eliminate public registry access.
- Private subnets and a node security group tagged
  `karpenter.sh/discovery=<cluster-name>`. At least two Availability Zones
  must contain subnets tagged `kubernetes.io/role/internal-elb=1`.
- A private RDS PostgreSQL 14 or newer instance with IAM database
  authentication enabled and `btree_gin` available. PostgreSQL 15 or newer is
  required when `cloud.postgres.bootstrapTemporal` is true; the bootstrap
  refuses to run against 14.
- A private S3 bucket and prefix for Kumo-managed data, plus a bucket CORS
  policy for browser uploads (see [S3 access](#s3-access)).
- An issued ACM certificate in the EKS region for the Kumo hostname and a
  Route 53 private hosted zone containing that hostname.
- AWS CLI v2, Helm 3, `kubectl`, `jq`, `openssl`, and `curl` on the installer
  host. That host must reach the EKS API. It never connects to RDS: every
  database step runs from inside the cluster.
- Access to a registry containing all three compatible Kumo release images.
  Public artifact locations remain a release prerequisite.
- A versioned editable bundle compatible with the selected application release;
  see the artifact caveat in the [bundle README](../README.md).

The installer uses Karpenter CRD/chart `1.12.0`, AWS Load Balancer Controller
chart `1.14.0`, EBS CSI managed add-on `v1.63.0-eksbuild.1`, and the Temporal
chart already included as unpacked source in the bundle.

Set `images.kumo.registry`, `images.kumo.repoPath`, and the compatible
`images.kumo.tag` explicitly. The current source defaults reference a restricted
registry; they are not public download instructions. Use a namespaced pull
Secret when your registry requires credentials.

## End-to-end network architecture

```text
Customer private network
  User browser
      |
      | HTTPS:443; private DNS resolves the Kumo hostname
      | Direct Connect, VPN, or VPC peering/transit
      v
AWS VPC
  Internal Application Load Balancer  [ACM TLS termination]
      |
      | HTTP:80 to EKS pod addresses
      v
  EKS cluster in private subnets
    AWS Load Balancer Controller
         - - programs the ALB from the Kumo Ingress - -^

    kumo-ui Service -> UI pod and built-in API reverse proxy
                           |
                           +--> Kumo REST/API and control pods
                                   |
                                   +--> Temporal frontend
                                   |
                                   +--> CPU/GPU worker pods

  Private managed services
    Kumo pods ----- TLS + IAM database auth ------> RDS PostgreSQL metadata DB
    Temporal ------ TLS + Kubernetes password ----> RDS Temporal databases
    Kumo workers -- HTTPS + IAM role -------------> S3
```

For inbound application traffic, client TLS terminates at the ALB and the
private ALB-to-UI hop uses HTTP. Connections to RDS and S3 use their own TLS.
Traffic behind the ALB stays on private addresses; restricting it requires
operator-provided Kumo NetworkPolicies. The AWS Load Balancer Controller is a control-plane
component: requests pass through the ALB directly to Kumo pods, not through
the controller pod.

## PostgreSQL access

Kumo stores application metadata in a PostgreSQL database. The configuration
and diagnostics call this database **StateDB**. Before installation:

1. Create the database named by `cloud.postgres.stateDatabase` (the example
   uses `kumo`).
2. Create the login named by `cloud.postgres.stateUser` (the example uses
   `kumo_state`), grant it `rds_iam`, and make it the owner of that database
   and its `public` schema. Kumo runs its own schema migrations with this
   login, so it needs normal owner privileges.
3. Give each Kumo IAM role `rds-db:connect` permission for that exact RDS
   resource ID and database login:

   `arn:aws:rds-db:<region>:<account-id>:dbuser:<db-resource-id>/kumo_state`

4. Create an IAM-authenticated PostgreSQL administrator login with `CREATEROLE`
   and `CREATEDB`, named by `cloud.postgres.admin.username`. It creates one
   password login and the `temporal` and `temporal_visibility` databases for
   the bundled stock Temporal server.
5. Create an IAM role for that administrator login, trusted by the cluster's
   OIDC provider for the `temporal` namespace ServiceAccount
   `kumo-temporal-bootstrap`, and grant it `rds-db:connect` for
   `arn:aws:rds-db:<region>:<account-id>:dbuser:<db-resource-id>/postgres_admin`.
   Put its ARN in `cloud.postgres.admin.annotations` under
   `eks.amazonaws.com/role-arn`. The bootstrap Job assumes this role from
   inside the cluster and mints its own database token, so no operator machine
   needs a route to RDS. The stack creates and deletes that ServiceAccount
   around each database step.

PostgreSQL 15 or newer is required when `bootstrapTemporal` is true: the
bootstrap reassigns each Temporal database's `public` schema, which earlier
releases allow only for a superuser.

AWS documents the exact `rds-db:connect` policy format in
[Creating and using an IAM policy for IAM database access](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.IAMPolicy.html).

## S3 access

For the bucket and prefix configured in `cloud.objectStorage.uri`, grant all
three Kumo IAM roles:

- `s3:ListBucket` on the bucket, limited to the configured prefix;
- `s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject` on objects below that
  prefix.

Also grant the worker role `s3:ListBucket` and `s3:GetObject` for every
customer source-data prefix that Kumo is allowed to ingest.

Local file upload is the one flow where the browser `PUT`s to S3 itself, so the
bucket also needs a CORS policy naming the Kumo hostname. Without it the browser
blocks the upload before any bytes are sent, while every server-side operation
keeps working. `ExposeHeaders` must list `ETag`: the UI reads it from each part
to complete the multipart upload, and browsers hide unexposed response headers.
`AllowedOrigins` must name every origin the browser loads Kumo from, which for
this recipe is `https://` plus the configured `tenant.exposureHost`.

```bash
S3_BUCKET="<bucket>"
aws s3api put-bucket-cors --bucket "${S3_BUCKET}" --cors-configuration '{"CORSRules": [{
  "AllowedOrigins": ["https://<kumo-hostname>"],
  "AllowedMethods": ["PUT", "POST", "GET", "HEAD"],
  "AllowedHeaders": ["*"],
  "ExposeHeaders": ["ETag"],
  "MaxAgeSeconds": 3600
}]}'
```

`put-bucket-cors` replaces the entire CORS configuration rather than adding to
it. On a bucket shared with anything else, read the current rules with
`aws s3api get-bucket-cors --bucket "${S3_BUCKET}"` first and submit the merged
set, otherwise this command deletes rules another workload depends on.

## Container ingredients

The reference recipe selects amd64 images. Supply the Kumo image registry,
repository prefix, and compatible release tag through `images.kumo` in
`config.yaml`. The placeholders below are not public artifact locations. When
mirroring third-party images, keep the listed versions and digests.

| Repository and tag field | Purpose |
|---|---|
| `<registry>/<repoPath>/kumo-backend-cpu` + `images.kumo.tag` | Kumo API, workers, and controllers |
| `<registry>/<repoPath>/kumo-backend-gpu` + `images.kumo.tag` | Kumo GPU work |
| `<registry>/<repoPath>/kumo-ui` + `images.kumo.tag` | Kumo web UI |
| `docker.io/temporalio/server:1.31.2@sha256:6b02e5176631c8d28f010735a3a73c69423a9adbcfad7e5dfe23836c904b7e26` | Temporal server |
| `docker.io/temporalio/admin-tools:1.31.2@sha256:7e5820112475b3490f011b28d86ca9fd8348f1640b8dd04c62adb906e6b28cb2` | Temporal schema and administration jobs |
| `docker.io/postgres:17.10-alpine@sha256:af194ccf3e2d7fe367012c7b88ce8b816c5c889b18a5b316799a1f0d7eac746a` | psql for the Temporal database bootstrap Job |
| `nvcr.io/nvidia/k8s-device-plugin:v0.19.0@sha256:7bf6ab18378be099493c9fe50f9cd2d559e0d81f17742d7ac188365408579d5d` | NVIDIA device discovery |
| `public.ecr.aws/karpenter/controller:1.12.0@sha256:8128bc8cdd0cc8ca2d2cf68ec61033eceb8cfb24834a08ab278784924013cfd0` | EKS node provisioning |
| `public.ecr.aws/eks/aws-load-balancer-controller:v2.14.0@sha256:58d83491b3eb25080e478bdd9a245b773fd9ceb29d9182fa4b198dee5ff31d22` | ALB reconciliation |

When mirroring, use the same compatible Kumo release tag for all three
repositories. The system-node and Karpenter-node roles need pull permission on
the mirror.

EKS additionally runs AWS-published images for the managed VPC CNI, CoreDNS,
kube-proxy, and `aws-ebs-csi-driver:v1.63.0-eksbuild.1` add-ons. AWS selects
those regional image names from the pinned add-on versions; Kumo does not
select or mirror them. The NVIDIA AL2023 Karpenter AMI supplies the host GPU
driver.

This installation does not require Istio, ingress-nginx, cert-manager,
ExternalDNS, External Secrets, Secrets Store CSI, TopoLVM, MinIO, an
in-cluster PostgreSQL server, or bundled observability.

## Installation

Run all commands from the root of the extracted bundle directory.

### 1. Connect to the cluster

```bash
VALUES_FILE="$PWD/config.yaml"
KUMO_STACK="$PWD/bin/kumo"

aws eks update-kubeconfig \
  --region <aws-region> \
  --name <eks-cluster-name> \
  --alias <kube-context-name>

kubectl --context <kube-context-name> get nodes
```

`VALUES_FILE` and `KUMO_STACK` are shell conveniences used repeatedly below;
they do not configure the installer.

### 2. Edit the customer configuration

```bash
${EDITOR:-vi} "$VALUES_FILE"
```

Replace every example cluster name, Kubernetes context, AWS account ID, VPC
ID, role ARN, database host and login, bucket, certificate ARN, tenant name,
hostname, and administrator email.

Set the Kumo registry, repository prefix, and release tag to your image source.
For password/token authentication, set `pullSecretName: kumo-registry`; the next
step creates that Secret. For an ECR mirror, leave `pullSecretName` empty when
node roles provide access. Change third-party repositories only when mirroring
them. Internal service-mesh TLS is disabled by default. Do not store registry
credentials in `config.yaml`.

### 3. Render and inspect the complete configuration

```bash
bash "$KUMO_STACK" render \
  --values "$VALUES_FILE" \
  --output-dir ./rendered-eks

runtime_value() {
  sed -n "s/^${1}=//p" ./rendered-eks/runtime.env
}
CONTEXT=$(runtime_value CONTEXT)
TENANT_NAMESPACE=$(runtime_value TENANT_NAMESPACE)
TENANT_RELEASE=$(runtime_value TENANT_RELEASE)
EDGE_HOST=$(runtime_value EDGE_HOST)
PROVIDER_REGION=$(runtime_value PROVIDER_REGION)
TEMPORAL_NAMESPACE=$(runtime_value TEMPORAL_NAMESPACE)
```

Inspect `rendered-eks/runtime.env` and the four rendered Helm values files
(platform, Temporal, core, and edge).
They contain image references and Kubernetes Secret names, but no Secret
values. The assignments above read the non-secret names used below without
executing the file as a shell script.

### 4. Create the registry and initial-login Secrets

```bash
read -r -p "Registry hostname (match images.kumo.registry): " KUMO_REGISTRY
read -r -p "Registry username: " KUMO_REGISTRY_USER
read -r -s -p "Registry password or pull token: " KUMO_REGISTRY_PASSWORD
printf '\n'

for namespace in "$TENANT_NAMESPACE" "$TEMPORAL_NAMESPACE"; do
  kubectl --context "$CONTEXT" create namespace "$namespace" \
    --dry-run=client -o yaml |
    kubectl --context "$CONTEXT" apply -f -
  kubectl --context "$CONTEXT" --namespace "$namespace" \
    create secret docker-registry kumo-registry \
    --docker-server="$KUMO_REGISTRY" \
    --docker-username="$KUMO_REGISTRY_USER" \
    --docker-password="$KUMO_REGISTRY_PASSWORD" \
    --dry-run=client -o yaml |
    kubectl --context "$CONTEXT" --namespace "$namespace" apply -f -
done
unset KUMO_REGISTRY_PASSWORD

kubectl --context "$CONTEXT" create namespace "$TENANT_NAMESPACE" \
  --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" apply -f -

read -r -s -p "Temporary Kumo admin password: " KUMO_BOOTSTRAP_PASSWORD
printf '\n'
printf '%s' "$KUMO_BOOTSTRAP_PASSWORD" |
  kubectl --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" \
    create secret generic kumo-bootstrap-password \
    --from-file=password=/dev/stdin \
    --dry-run=client -o yaml |
  kubectl --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" apply -f -
unset KUMO_BOOTSTRAP_PASSWORD
```

The commands above cover a registry requiring credentials with bundled
Temporal. For an ECR mirror accessible through the node roles, skip the
registry-secret loop and leave `images.kumo.pullSecretName` empty. With
external Temporal, create the registry Secret only in `TENANT_NAMESPACE`.

Before installation, supply Kumo namespace network policies appropriate to
this cluster's CNI and ingress path. The application chart does not install
them. To start from the bundled backend-ingress example:

```bash
cp charts/kumo-helm/examples/networkpolicy.yaml ./kumo-networkpolicy.yaml
${EDITOR:-vi} ./kumo-networkpolicy.yaml
# Replace <release> and <ns> with TENANT_RELEASE and TENANT_NAMESPACE.
# Review selectors, permitted traffic, and CNI enforcement before applying.
kubectl --context "$CONTEXT" --namespace "$TENANT_NAMESPACE" \
  apply -f ./kumo-networkpolicy.yaml
```

This example covers backend ingress only; operators own the complete network
policy for their deployment.

### 5. Install

```bash
bash install.sh
```

The installer reconciles EBS CSI, Karpenter, AWS Load Balancer Controller,
the CPU and T4 pools, Temporal, and Kumo in dependency order. It generates the
Temporal database password directly into a Kubernetes Secret without placing
the value in Helm values or command-line arguments.

### 6. Create private DNS for the internal ALB

```bash
ROUTE53_ZONE_ID=<private-hosted-zone-id>

ALB_DNS=$(
  kubectl --context "$CONTEXT" \
    --namespace "$TENANT_NAMESPACE" \
    get "ingress/${TENANT_RELEASE}-edge" \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
)

ALB_ZONE_ID=$(
  aws elbv2 describe-load-balancers \
    --region "$PROVIDER_REGION" \
    --query "LoadBalancers[?DNSName=='${ALB_DNS}'].CanonicalHostedZoneId | [0]" \
    --output text
)

CHANGE_ID=$(
  aws route53 change-resource-record-sets \
    --hosted-zone-id "$ROUTE53_ZONE_ID" \
    --change-batch "$(
      jq -n \
        --arg host "$EDGE_HOST" \
        --arg dns "dualstack.${ALB_DNS}" \
        --arg zone "$ALB_ZONE_ID" \
        '{Changes:[{Action:"UPSERT",ResourceRecordSet:{
          Name:$host,Type:"A",AliasTarget:{
            HostedZoneId:$zone,DNSName:$dns,EvaluateTargetHealth:false
          }}}]}'
    )" \
    --query 'ChangeInfo.Id' \
    --output text
)

aws route53 wait resource-record-sets-changed --id "$CHANGE_ID"
curl --fail --show-error --head "https://${EDGE_HOST}/"
```

### 7. Run the acceptance checks

```bash
bash "$KUMO_STACK" conformance --values "$VALUES_FILE"
bash "$KUMO_STACK" verify gpu-lifecycle --values "$VALUES_FILE"
bash "$KUMO_STACK" verify container-policy --values "$VALUES_FILE"
bash "$KUMO_STACK" status --values "$VALUES_FILE"
```

The GPU check creates each configured T4 tier and returns it to zero. The
container-policy check examines every namespace, so run it only on the fresh
acceptance cluster described by this recipe.

Log in at `https://$EDGE_HOST` with the configured administrator email and
temporary password, then replace that password through the Kumo UI.

For command and lifecycle details, see the
[shared installer reference](../README.md). For failures, use
[troubleshooting by stage](troubleshooting.md).
