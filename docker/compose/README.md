# Kumo Compose Stack

This directory contains the complete Docker Compose topology for a single-host
Kumo stack. Choose the workflow that matches what you are doing:

| Goal | Start here |
|------|------------|
| Build and provision an Azure single-host deployment | [Azure deployment guide](../../deploy/azure/README.md) |
| Run the Azure Compose reference deployment | [Azure Compose guide](azure/README.md) |
| Test compatible prepared images on a single host | [Prepared-image quick start](#prepared-image-quick-start) |
| Build and test changes from a Kumo checkout | [Local development](#local-production-style-run-no-ami-no-gpu-required) |
| Operate a prepared single-host installation | [Production GPU run](#production-gpu-run) |

These source deployment paths require prepared images and dependencies. Public
artifact availability and a clean installation still need release verification.

## Compose files

| File | Purpose |
|------|---------|
| `docker-compose.singlehost.yml` | Base service definitions — all environments |
| `docker-compose.singlehost.postgres.yml` | Bundled Postgres overlay — local StateDB |
| `docker-compose.singlehost.gpu.yml` | GPU image and NVIDIA device overlay |
| `docker-compose.production-local.override.yml` | Local image selection and nginx/Prometheus config mounts |

`.env.compose-local` and `.env.compose-local.local` configure the repository
developer workflow below.

---

## Prepared-image quick start

### Prerequisites

- Docker Engine with Docker Compose v2
- A Kumo checkout containing `deploy/shared`; the bootstrap container mounts
  that directory through `KUMO_BOOTSTRAP_DIR`
- Access to compatible `kumo-ui`, `kumo-backend-cpu`, and (for GPU mode)
  `kumo-backend-gpu` images in your chosen registry
- Port `8888` available. A different `NGINX_PORT` also requires matching
  `listen` ports in the selected nginx configuration
- Recommended capacity: 32 vCPU, 128 GB RAM, and at least 100 GB free disk
- For GPU mode: an NVIDIA GPU, host driver, and NVIDIA Container Toolkit

Run these commands from `docker/compose`:

```bash
cp env.internal-testing.example .env
mkdir -p data/dataplane data/diskgraph data/glove data/secrets data/tls
```

Edit `.env` and replace `REPLACE_WITH_RANDOM_PASSWORD`. Also replace the
template's image settings with compatible prepared images you can access:

```dotenv
KUMO_IMAGE_REGISTRY=registry.example.com/team
KUMO_IMAGE_TAG=REPLACE_WITH_IMAGE_TAG
```

Use the registry hostname for `docker login` below; the image registry setting
includes the namespace containing the product images.

Generate a password with `openssl rand -hex 24`, or an equivalent password
generator. The `.env` file and `data/` runtime directory are ignored by Git.

Authenticate, render-check the configuration, pull images, and start:

```bash
docker login registry.example.com
./kumo-compose config --quiet
./kumo-compose pull
./kumo-compose up -d
./kumo-compose ps --all
```

Open `http://localhost:8888`. Cold startup can take several minutes because
REST waits for Postgres, Temporal, and the workers.

`kumo-compose` automatically uses the selected GPU backend and NVIDIA device
overlay when both the NVIDIA driver and Docker NVIDIA runtime are available.
Otherwise it uses the selected CPU backend. No `.env` changes are needed. To
deliberately disable a GPU, prefix a command with `KUMO_BACKEND_VARIANT=cpu`.

### Validation

The bootstrap, Temporal schema, and Temporal namespace containers are one-shot
jobs and should exit with status 0. Other services should be running, and
those with configured health checks should report healthy:

```bash
./kumo-compose ps --all
./kumo-compose logs --no-color kumo-bootstrap temporal-schema temporal-namespace
curl -fsS http://localhost:8888/ >/dev/null
```

For GPU mode, also verify device visibility:

```bash
./kumo-compose exec -T kumo-pool-worker nvidia-smi
```

Stop the stack while retaining data:

```bash
./kumo-compose down --remove-orphans
```

Use `./kumo-compose down -v --remove-orphans` only when you also intend to
delete Docker-managed Postgres, observability, and generated-configuration
volumes. Compose never
deletes the host-mounted `data/` directory.

Authentication is disabled in the internal-testing template. Use it only on a
trusted host and do not expose nginx to an untrusted network. GloVe data is not
included; tests using text features require data under `data/glove`.

---

## Local production-style run (no AMI, no GPU required)

Use this to test diskgraph and pipeline changes locally in a
production-topology stack. Skips the 1-3 hour AMI build cycle.

Run the following `make` commands from the repository root. The Makefile for
this workflow is not inside `docker/compose`.

**Prerequisites:**
- Docker Engine with Compose
- A prepared backend dependency/image environment. The current backend build
  tooling still requires a GitHub-token build secret; [public dependency setup](../../docs/development/setup.md#setup-prerequisites-and-side-effects)
  remains unfinished. The UI uses the in-tree catalog.

```bash
make compose-local        # build if needed, then start the stack
make compose-local-logs
make compose-local-down
```

With `KUMO_IMAGE_TAG=local` or `latest`, `make compose-local` builds missing
images and compares recorded hashes for
its backend/UI source allowlists before reusing them. Build time depends on
the host and cache. The hash uses Git commits/status and the DiskGraph source
hash, so it is not a content hash of every uncommitted file; run
`make compose-local-build` explicitly when reusing a dirty checkout whose
changes are not detected. Other tags use the Makefile’s configured ECR pull
path and require authorized registry access.

By default the backend image stages `packages/diskgraph`, builds a wheel
in a throwaway builder stage, and installs only that wheel in the runtime
image. The package bundles the native `ingest` binary used at runtime.

To run a second development stack on the same host, set `STACK`. This prefixes
container names, shifts ports, and
uses stack-specific generated config, runtime, DiskGraph, and dataplane paths.
Image tags are selected separately by `KUMO_IMAGE_TAG`; the checked-in default
is `latest`, so differently named stacks can share images:

```bash
STACK=kumo1 make compose-local-build
STACK=kumo1 make compose-local
STACK=kumo1 make compose-local-logs
STACK=kumo1 make compose-local-down
```

For stack names ending in a number, the default `PORT_OFFSET` is that number
times 100. Only nginx is published on the host, so `STACK=kumo1` publishes
nginx on `8988`; the other service ports remain internal to the Compose
network. Override `PORT_OFFSET` or `NGINX_PORT` when that port is already in
use.

Configuration for this flow lives in `.env.compose-local`. Edit that file
for shared defaults, create `.env.compose-local.local` for machine-specific
overrides that should not be committed, or override a value for one run from
the shell. Values in `.env.compose-local.local` take precedence when running
through `make compose-local`.

**Custom DiskGraph data directory** (e.g. a real SSD):

```bash
LOCAL_DISKGRAPH_DATA=/fast/ssd make compose-local
```

`LOCAL_DISKGRAPH_DATA` and `DISKGRAPH_DATA_ROOT` are base directories. The
preflight step appends `$STACK/diskgraph`, then mounts that host directory at
`/mnt/localssd` in every container. For example, `STACK=kumo1` with the default
base uses `/mnt/localssd/kumo1/diskgraph` on the host and `/mnt/localssd` in the
containers.

**Custom dataplane or artifact storage:**

Every stack gets an isolated dataplane root. The preflight step inserts
`$STACK` before `{instance_id}`, leaving that placeholder for runtime expansion,
or appends `$STACK` when there is no placeholder. This applies
to local paths, S3, ADLS, and GCS roots.

```bash
PRODUCTION_LOCAL_DATAPLANE_ROOT='abfss://container@acct.dfs.core.windows.net/dev/{instance_id}/' \
PRODUCTION_LOCAL_ARTIFACT_ROOT='abfss://container@acct.dfs.core.windows.net/artifacts/' \
make compose-local
```

For durable VM-local settings, put the same values in
`.env.compose-local.local`:

```bash
PRODUCTION_LOCAL_DATAPLANE_ROOT=abfss://container@acct.dfs.core.windows.net/dev/{instance_id}/
PRODUCTION_LOCAL_ARTIFACT_ROOT=abfss://container@acct.dfs.core.windows.net/artifacts/
AZURE_STORAGE_ACCOUNT=acct
```

S3 roots work the same way:

```bash
PRODUCTION_LOCAL_DATAPLANE_ROOT='s3://my-test-bucket/dev/{instance_id}/' \
PRODUCTION_LOCAL_ARTIFACT_ROOT='s3://my-artifact-bucket' \
make compose-local
```

GCS roots work the same way:

```bash
PRODUCTION_LOCAL_DATAPLANE_ROOT='gs://my-test-bucket/dev/{instance_id}/' \
make compose-local
```

**What runs automatically on each `make compose-local`:**
- Storage setup reuses an existing `/mnt/localssd` mount or Azure resource
  disk. Otherwise it can format the first recognized instance-store NVMe
  device as XFS; it does not create RAID0. Review
  `deploy/shared/scripts/mount-nvme.sh` before running on a host with existing
  scratch data. When no usable scratch mount is found, preflight can fall back
  to repo-local storage.
- API, bootstrap, controller, and workflow services use the CPU backend image.
  The pool worker selects CPU or GPU based on visible NVIDIA devices and the
  Docker NVIDIA runtime; set
  `KUMO_BACKEND_VARIANT=cpu` or `KUMO_BACKEND_VARIANT=gpu` to force a variant.
- GPU compose-local runs include the NVIDIA device-reservation overlay and use
  the CUDA 13 GPU backend for the pool worker. CPU remains the fallback when no
  NVIDIA runtime is available.
- DiskGraph uses a stack-scoped local SSD path derived from
  `LOCAL_DISKGRAPH_DATA` / `DISKGRAPH_DATA_ROOT`
- Dataplane roots are stack-scoped and written into the generated deploy config

Internal service ports are the same as production. nginx is the only service
published on the host:

| Service | Internal port |
|---------|---------------|
| Web UI | 3000 (proxied via nginx on 8888) |
| REST API | 8000 |
| Public/SDK API | 8001 |
| Temporal | 7233 |
| Prometheus | 9090 |
| Jaeger | 16686 |

Namespaced stacks apply `PORT_OFFSET` to the published nginx port only.

---

## Production GPU run

Production singlehost uses three compose files. The base file defines the core
services, the Postgres overlay adds the bundled database, and the GPU overlay
adds the NVIDIA device reservation to the pool worker.

The stack requires `KUMO_IMAGE_TAG`, and bundled-Postgres installs additionally
require `POSTGRES_PASSWORD` (compose fails fast
otherwise). Firstboot generates a complete `.env`; compose-local uses its
selected environment files and preflight exports. Preserve the prepared
installation’s identity, storage, authentication, and path settings. For a
bundled-Postgres GPU installation, run from its Compose directory:

```bash
test -s .env  # Existing installation configuration, not a two-variable template
docker compose \
  -f docker-compose.singlehost.yml \
  -f docker-compose.singlehost.postgres.yml \
  -f docker-compose.singlehost.gpu.yml \
  up -d
```

The Postgres overlay is required for bundled-Postgres installs and for
bundled Temporal persistence. Omitting it is intended only for
`KUMO_POSTGRES_MODE=external`, after preflight has stored the StateDB and
Temporal DSNs in mounted secrets. The GPU overlay is required for GPU-backed
train and predict activities. Omitting it starts all services normally but the
pool worker runs without a GPU device, so training will fall back to CPU or
fail.

The boot, restart, and watchdog scripts in `deploy/shared/scripts/` select the
base and GPU files, adding the Postgres overlay in bundled mode. The health
script inspects running containers; it does not assemble a Compose command.

For external/BYO Postgres, provide structured connection inputs to the
preflight/firstboot step:

```bash
KUMO_POSTGRES_MODE=external
POSTGRES_HOST=postgres.example.com
POSTGRES_PORT=5432
POSTGRES_USER=kumo
POSTGRES_PASSWORD_FILE=/path/to/postgres-password
KUMO_STATE_DB_NAME=kumo
TEMPORAL_DB_NAME=temporal
TEMPORAL_VISIBILITY_DB_NAME=temporal_visibility
POSTGRES_SSLMODE=verify-full
POSTGRES_SSLROOTCERT=/opt/kumo/secrets/postgres-ca.pem
TEMPORAL_SERVER_IMAGE=registry.example.com/temporalio/server:1.31.1
TEMPORAL_ADMIN_TOOLS_IMAGE=registry.example.com/temporalio/admin-tools:1.31.1
```

The external Postgres deployment must already have the target databases and
grants. Temporal follows its upstream Postgres model: `TEMPORAL_DB_NAME` backs
the default store and `TEMPORAL_VISIBILITY_DB_NAME` backs visibility. In
external mode, compose-local and singlehost boot scripts write structured
connection secrets for preflight validation and omit
`docker-compose.singlehost.postgres.yml`.

Do not put raw DSNs in the steady-state compose environment. Preflight/firstboot
validates the structured bootstrap inputs, writes mounted StateDB secrets, and
passes Temporal the stock Postgres environment variables expected by the
upstream Temporal images. For TLS verification, copy the customer CA into
`KUMO_SECRETS_DIR` and set `POSTGRES_SSLMODE=verify-ca|verify-full` plus the
in-container `POSTGRES_SSLROOTCERT` path.

Stock Temporal's Postgres driver still requires the server certificate to be
valid for `POSTGRES_HOST` when a CA file is configured. Use a Postgres endpoint
whose certificate includes that hostname for both `verify-ca` and
`verify-full`.

Air-gapped environments can mirror the upstream Temporal images and set
`TEMPORAL_SERVER_IMAGE` / `TEMPORAL_ADMIN_TOOLS_IMAGE` to those mirrored tags.

External mode runs a preflight check against all configured Postgres DSNs before
starting Kumo. The check uses `postgres:16-alpine`'s `psql` client, connects
with a short timeout, creates/inserts/updates/deletes/drops a scratch table,
and fails fast if any database is missing or lacks usable grants. Set
`KUMO_POSTGRES_DOCKER_NETWORK` if the probe container must run on a specific
Docker network.

In external mode, `temporal-schema` does not have a compose-level Postgres
dependency because the Postgres service is outside the stack. The firstboot or
compose-local preflight is the startup gate; bypassing it means
`temporal-schema` will fail fast if the external DB is unreachable or missing
grants.

---

## Configuration

Path overrides for the base compose file are controlled by env vars.
Compose-local defaults auto-detect the cloud provider and include internal test
storage. Set explicit operator-controlled dataplane/artifact roots; the current
`auto` defaults are not a public storage service:

| Variable | Default | Description |
|----------|---------|-------------|
| `DISKGRAPH_DATA_ROOT` | `/mnt/localssd` | Base host path for DiskGraph data; preflight maps `$DISKGRAPH_DATA_ROOT/$STACK/diskgraph` to `/mnt/localssd` in containers |
| `LOCAL_DISKGRAPH_DATA` | `/mnt/localssd` | Production-local preflight override for the DiskGraph base host path |
| `COMPOSE_LOCAL_CLOUD_PROVIDER` | `auto` | Cloud provider for `auto` storage roots; supports `auto`, `aws`, `gcp`, and `azure` |
| `PRODUCTION_LOCAL_DATAPLANE_ROOT` | `auto` | Dataplane base root written into generated deploy config after stack scoping; `auto` selects provider-specific test storage with a repo-local `.cache/kumo` fallback; supply a writable root for your deployment |
| `PRODUCTION_LOCAL_ARTIFACT_ROOT` | `auto` | Artifact root written into generated deploy config; `auto` resolves to provider-specific artifact storage requiring verified access, or local artifacts when dataplane falls back locally |
| `AZURE_STORAGE_ACCOUNT` | Internal test account in current Compose defaults | Fallback for ADLS client calls that omit an account; explicit `abfss://` URIs carry their own account. Set operator-controlled storage explicitly |
| `LOCAL_DATAPLANE_HOST_MOUNT` | `<repo>/.cache/kumo/compose-local/...` | Host source for the optional local dataplane bind mount |
| `LOCAL_DATAPLANE_MOUNT` | `<repo>/.cache/kumo/compose-local/...` | Container destination for the optional local dataplane bind mount |
| `DEPLOY_ENV_CONFIG_PATH` | `kumo_config` named volume | Deploy config source; compose-local preflight replaces it with `config/generated/$STACK/deploy_env_config.production-local.json` |
| `KUMO_SECRETS_DIR` | `kumo_secrets` named volume | Secrets source; compose-local preflight replaces it with a stack-scoped runtime directory |
| `KUMO_GLOVE_DIR` | `/opt/kumo/glove` | GloVe embeddings directory |
| `NGINX_CONFIG` | `kumo_nginx` named volume | nginx config source |
| `KUMO_TLS_DIR` | `kumo_tls` named volume | TLS certificates source |
| `PROMETHEUS_CONFIG` | `kumo_prometheus` named volume | Prometheus scrape config source |
| `TEMPORAL_SERVER_IMAGE` | `temporalio/server:1.31.1` | Temporal server image |
| `TEMPORAL_ADMIN_TOOLS_IMAGE` | `temporalio/admin-tools:1.31.1` | Temporal schema/namespace init image |
| `KUMO_IMAGE_REGISTRY` | `kumo` | Registry/namespace prefix for the `kumo-backend-cpu`, `kumo-backend-gpu`, and `kumo-ui` images |
| `PROMETHEUS_IMAGE` | `prom/prometheus:v2.53.4` | Prometheus image |
| `NGINX_IMAGE` | `nginx:1.29.7` | nginx reverse-proxy image |
| `JAEGER_IMAGE` | `jaegertracing/all-in-one:1.76.0` | Jaeger all-in-one tracing image |
| `POSTGRES_IMAGE` | `postgres:16-alpine` | Bundled Postgres image (Postgres overlay) |
| `TEMPORAL_NAMESPACE_RETENTION` | `30d` | Retention for newly created Temporal namespace |

## S3-compatible connector identities

Project-created S3 Connectors may use administrator-owned access-key
identities for MinIO and other S3-compatible services. Connector configuration
is independent of deployment storage: endpoints, regions, addressing styles,
private CAs, keys, buckets, and policies may all differ. Connector credentials
are encrypted in Kumo StateDB and never belong in Compose or deployment secret
mounts.

Create a normal S3 identity through the administrator API or UI. In a
production-like stage the endpoint must use HTTPS and must be the S3 API origin
without credentials, a path, query, fragment, or MinIO Console URL. A custom
endpoint requires a SigV4 signing region; `path` addressing is the usual MinIO
choice. The optional session token and private CA PEM are stored with the
encrypted credentials.

```json
{
  "display_name": "Customer MinIO reader",
  "config": {
    "type": "s3",
    "authentication": {"mode": "access_key"},
    "connection": {
      "endpoint_url": "https://minio.customer.example:9000",
      "region": "us-east-1",
      "addressing_style": "path"
    }
  },
  "credentials": {
    "kind": "s3_access_key",
    "access_key_id": "DEDICATED_CONNECTOR_ACCESS_KEY",
    "secret_access_key": "REDACTED",
    "session_token": null,
    "ca_certificate": "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----"
  }
}
```

Use the returned identity ID when creating an ordinary rooted S3 Connector.
Grant that identity only the required list/location and object operations for
the Connector bucket and prefix; do not grant access to Kumo dataplane or
artifact buckets. Kumo stages access-key-backed DiskGraph inputs inside the
authorized Connector scope, then runs DiskGraph under the independent
deployment connection so a native child never receives both keys.

Test and rotate credentials through the identity and Connector APIs or UI.
Rotation replaces access key, secret key, optional token, and CA atomically;
responses expose only non-secret connection metadata and rotation timestamps.
Existing AWS ambient and AssumeRole identities omit `connection` and retain
their existing behavior. MinIO OIDC/LDAP STS, OAuth exchanges, and MinIO
`AssumeRole` are not supported.
