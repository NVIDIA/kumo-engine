# Devserver

The devserver coordinator (`rest/devserver/multi_main.py`) launches one or more
tmux-based child stacks (`rest/devserver/main.py`) for the
full local Kumo stack for development. It starts 8+ services in parallel panes, validates
system dependencies before startup, and provides actionable error messages if anything is
missing.

## Quick Start

Use a prepared checkout; see [setup](development/setup.md). For one local
devserver with an explicit choice to run without cgroup fencing:

```bash
UNFENCED=1 NUMS=1 make devserver-local
```

For cgroup fencing, use `CGROUP_CONFIG=auto NUMS=1` on a configured host,
or a JSON `CGROUP_CONFIG` with per-child CPU and memory allocations. The
Makefile requires a resource policy; inspect `rest/devserver/planning.py`
for its schema.

The local variant uses repo-local data-plane storage, local DiskGraph data
compute, local ML, and local Temporal. Auth is disabled by default
(`DISABLE_AUTH=true` in `rest/devserver/.env`).

## Resetting local state

`make reset-devserver` and `make reset-devserver-local` are not child-specific:
they delete configured state, terminate workflows, and may delete Kubernetes
jobs. The reset script's `--dry_run` still terminates workflows. Stop the
intended child and verify its paths and endpoints before any cleanup.

## Services Started

| Service | Description |
|---------|-------------|
| Web UI | React dev server; native Node on macOS unless Docker is selected, Docker elsewhere. Coordinated children require Docker even on macOS. |
| REST API | FastAPI with Swagger docs at `/docs` |
| Public REST API | Secondary API with auth disabled |
| t-worker-dataworkflow | Temporal worker for data ingestion/processing |
| t-worker-mlworkflow | Temporal worker for ML orchestration |
| Log viewer | Local dev watcher or k8s stern logs (variant-dependent) |
| Memory monitor | Prints memory usage periodically |
| Monitoring | Jaeger + Grafana (default on Linux; opt-in on macOS) + SQLite console |

## Browser and API Topology

The devserver uses one browser origin. With base port `N`, the Vite UI server
serves the application and proxies its internal `/v0/*` and `/v1/*` requests to
the private REST listener:

| Port | Purpose |
|---:|---|
| `N` | Browser-facing UI and internal API origin |
| `N + 1` | Direct private REST diagnostics and Swagger documentation |
| `N + 2` | Public REST API and SDK endpoint |

Use the base port assigned in the coordinator launch output. The UI must not be configured to call the
private REST port directly. That listener remains available at `N + 1` for
diagnostics; SDK clients continue to use the public API at `N + 2`.

Use `localhost` consistently in browser URLs. For Direct Connect OIDC, register
the exact callback for the selected base port:

```text
http://localhost:<N>/v0/auth/callback
```

For example, base port `12000` uses
`http://localhost:12000/v0/auth/callback`. Backend proxy targets and health
checks may use `127.0.0.1`; it is not interchangeable with `localhost` in an
OIDC redirect URI.

## Variants

Deploy config for each variant is in `rest/devserver/configs/<variant>.json`:

Use `variant=local` for local Temporal. Other variant files can reference
internal services or cloud resources; they require operator-supplied
configuration and are not public launch recipes.

All variants use the pooled training and prediction execution path; there is no separate ML compute backend setting.

`make devserver-local` (with the resource policy above) starts a local Temporal cluster automatically
before launching the devserver. If Temporal is already
running, you can use `UNFENCED=1 NUMS=1 make devserver variant=local` directly.

The default coordinator home is `<repo>/.cache/kumo/devserver`, which is ignored
by git; children have isolated state below it. Override it with
`KUMO_HOME=/path/to/cache` when you need a different local root.

## Storage Configuration

For the tmux devserver, copy or edit a variant file under
`rest/devserver/configs/`. The commonly changed fields are in the `storage`
block:

```json
{
  "storage": {
    "data_plane_storage_root": "abfss://container@account.dfs.core.windows.net/dev/{instance_id}/",
    "artifact_root": "abfss://container@account.dfs.core.windows.net/artifacts/"
  }
}
```

S3 roots use the same shape:

```json
{
  "storage": {
    "data_plane_storage_root": "s3://my-test-bucket/dev/{instance_id}/",
    "artifact_root": "s3://my-artifact-bucket"
  }
}
```

`{instance_id}` and `{kumo_home}` are expanded by the app at
runtime, which keeps shared buckets isolated per developer VM and lets local
variants share the centralized cache default. For the Docker compose-local
flow, use `docker/compose/.env.compose-local` instead; preflight generates a
stack-scoped config under `docker/compose/config/generated/$STACK/`.

## Makefile Flags

```bash
CGROUP_CONFIG=auto NUMS=2 make devserver-local      # Two fenced child stacks
UNFENCED=1 NUMS=1 BASE_PORTS=13000 make devserver-local  # Explicit base port
UNFENCED=1 NUMS=1 DEVSERVER_MONITORING=1 make devserver-local
UNFENCED=1 NUMS=1 UI_CATALOG_DEV=1 make devserver-local  # In-tree catalog hot reload
```

`BASE_PORT` and `RANDOM_PORT` are replaced by `BASE_PORTS` for the coordinator.
Use `DEVSERVER_IDS` to name children; its count must match the selected policy.
Print a child's ports and paths with `make devserver-env DEVSERVER_ID=CHILD_ID`.
If the launch used a custom `KUMO_HOME`, pass that path as `DEVSERVER_KUMO_HOME`
for this lookup: `DEVSERVER_KUMO_HOME=/path/to/cache make devserver-env DEVSERVER_ID=CHILD_ID`.

## Environment Variables

Global settings live in `rest/devserver/.env`:

- **Identity**: `DEPLOY_STAGE`, `INSTANCE_ID`
- **Feature flags**: Multilabel ranking, BigQuery, column analysis, Databricks enablement
- **Auth**: `DISABLE_AUTH=true` by default (allows random credentials)

Port allocation is automatic unless `BASE_PORTS` is supplied. Use the launch
output for each child's assigned port range.
The same-origin UI proxy derives its private REST target from the selected
range, so no source change is required.

## Dependency Checks

The devserver validates dependencies before launch and will refuse to start if any are
missing. Checks include:

- **System**: tmux, `temporalio` at the version specified in `3rdparty/BUILD`, kumo package; Docker when a Docker-backed panel is enabled
- **Ports**: All allocated ports are available (checked via `lsof`)
- **Cloud credentials**: S3/GCS access, Temporal Cloud connectivity
- **K8s context** (k8s-backed variants): EKS config and stern for log viewing
- **Local resources**: DiskGraph package and repo-local cache paths

On macOS, the core API and workers run natively. Coordinated child UI startup
still requires Docker for its dependency volume, even when npm runs natively.
The containerized monitoring pane is disabled by default; enable it with
`UNFENCED=1 NUMS=1 DEVSERVER_MONITORING=1 make devserver-local`. Linux retains monitoring and its
Docker dependency by default. `KUMO_UI_DEVSERVER_DOCKER=1` also explicitly
selects the Docker-backed UI on macOS. To run the API and job workers without
the UI (for example on a memory-constrained development host), set
`DEVSERVER_UI=0`; the REST APIs, Temporal workers, and DiskGraph pool remain
enabled. With both UI and monitoring disabled, the native macOS core does not
require Docker.

When monitoring is enabled, Grafana and Prometheus use `--network host`.
Docker Desktop 4.34 and later
supports this as an opt-in feature: open **Settings > Resources > Network**,
enable **Host networking**, then apply and restart Docker Desktop. Without it,
Jaeger remains reachable through its explicitly published ports but Grafana and
Prometheus are isolated inside Docker Desktop's Linux VM.

Failures include specific instructions for resolution (e.g. "run `mise install`" or
"run `make install-diskgraph`").

## Safety

The devserver enforces hard-fail checks to prevent accidental production damage:

- `DEPLOY_STAGE` must be `dev` or `dev_local` (never `prod`)
- `instance_id` cannot resolve to `'prod'`
- Control plane storage root cannot contain `/test/prod/` path

## Debugging

All services run in tmux panes. Substitute the child ID from the coordinator
launch output; both its tmux socket and session are named `devserver-CHILD_ID`:

```bash
tmux -L devserver-CHILD_ID attach -t devserver-CHILD_ID
```

Navigate between panes with `Ctrl-b` + arrow keys. Each pane shows live stdout/stderr
from its service. The REST API Swagger UI is available at `http://localhost:<port>/docs`.

## Reset

Reset and cleanup commands can remove databases, jobs, caches, or local
processes. Inspect the current Makefile/script and the child's lifecycle
manifest before using them; do not assume a single session named `devserver`.

## Troubleshooting

**Port conflicts**: Inspect the owning child/process and choose another
`BASE_PORTS` value if needed. Do not kill a guessed tmux session.

**Temporal not running** (local variant): Use `UNFENCED=1 NUMS=1 make devserver-local` to start
Temporal automatically. If using
`UNFENCED=1 NUMS=1 make devserver variant=local` directly, run `make up-local-temporal` first.

**Dependency check failures**: The devserver validates dependencies on startup and
prints specific instructions for resolution. Follow the printed guidance — common fixes
include `mise install`, `make install-diskgraph`, or configuring cloud credentials.

## DiskGraph Backend

See [diskgraph_integration.md](diskgraph_integration.md) for details on the DiskGraph
Rust engine, which provides local SSD-backed graph processing. DiskGraph is the
default and only supported graph backend.

## Databricks Backend

See [devserver-databricks.md](devserver-databricks.md) for instructions on running
with a Databricks backend.
