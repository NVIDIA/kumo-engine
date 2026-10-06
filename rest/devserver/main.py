# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import ipaddress
import json
import os
import os.path as osp
import sys
import zlib
from shlex import quote
from typing import List, Optional
from urllib.parse import urlsplit

from service_framework import developer_credentials


def _load_devserver_env() -> None:
    """Apply tracked devserver defaults for an explicit CLI launch."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key in os.environ:
                        # Callers may supply variant-specific local settings
                        # without editing the tracked .env defaults.
                        continue
                    os.environ[key] = value


def _enable_dev_gcp_user_credentials() -> None:
    """Enable Google developer credentials for this devserver process tree."""
    os.environ.setdefault(
        developer_credentials.ENV_DEV_GCP_USER_CREDENTIALS_ENABLED, 'true')


# The script entry point needs these defaults before importing modules that may
# resolve deploy configuration. Importers (tests and tooling) must remain free
# of process-wide environment side effects.
if __name__ == '__main__':
    _load_devserver_env()
    # A devserver is a single-developer environment. Let its existing ambient
    # identity flow resolve the developer's standard Google ADC automatically;
    # the runtime resolver independently refuses this source outside dev stages.
    _enable_dev_gcp_user_credentials()

# yapf: disable
# isort: off


from io_utils import fsutil  # noqa: E402
from rest.constants import (DATA_BACKEND_WORKER_NAME,  # noqa: E402
                            ML_WORKFLOW_WORKER)
from service_framework._raw_deploy_config import (  # noqa: E402
    CustomTemporalConfig,
    SqliteMetadataStorageConfig,
)
from service_framework.deploy_env_config import (DeployEnvConfig,  # noqa: E402
                                                 deploy_env as deploy_env_v2)
from service_framework.deploy_stage import DeployStage  # noqa: E402
from tmux_tools import config  # noqa: E402
from rest.devserver.cgroup import CgroupLimits, scope_argv  # noqa: E402
from rest.devserver.instance import (DEFAULT_BASE_PORT,  # noqa: E402
                                     DevserverInstance)
# isort: on
# yapf: enable


def is_gcp_deploy() -> bool:
    """Check if running in GCP deployment mode.

    Derives from the data-plane storage root rather than a standalone env var.
    """
    return fsutil.is_gcs(deploy_env_v2().data_plane_storage_root)


DEVSERVER_BASE_PORT = DEFAULT_BASE_PORT


def generate_base_port(random_port=False,
                       base_port: Optional[int] = None) -> int:
    """Generate a random port number from the current username, to prevent port
    conflicts when different users are running the devserver on the same VM.
    """
    if base_port is not None:
        return base_port

    configured_base_port = os.environ.get('PHASE1B_DEVSERVER_BASE_PORT')
    if configured_base_port:
        return int(configured_base_port)

    if not random_port:
        return DEVSERVER_BASE_PORT

    try:
        username = os.getlogin()
        return DEVSERVER_BASE_PORT + 100 * (zlib.adler32(username.encode()) %
                                            100)
    except Exception:
        # In the case where os.getlogin() fails, return a hard-coded address
        return DEVSERVER_BASE_PORT


async def _initialize_devserver_sqlite_state(
    deploy_config: DeployEnvConfig,
    *,
    db_storage_root: Optional[str] = None,
) -> None:
    """Create shared SQLite metadata before devserver panes start.

    REST, public REST, and the Temporal workers start concurrently. On a cold
    devserver, allowing those processes to bootstrap SQLite independently can
    leave a late-opening worker attached to a different first-created database
    state. Creating the schema and canonical Default resources in the parent
    process gives every pane one existing database to open.

    Postgres-backed variants already coordinate bootstrap in the database and
    do not need this local preflight.
    """
    if not isinstance(deploy_config.metadata_storage,
                      SqliteMetadataStorageConfig):
        return

    from rest.settings import get_db_storage_root
    from statedb.db import StateServices, open_state_db_from_deploy_config

    root = db_storage_root or get_db_storage_root()
    state_db = await open_state_db_from_deploy_config(
        deploy_config,
        state_root=root,
    )
    try:
        services = StateServices(state_db)
        default_group, _ = await services.groups.ensure_default_resources()
        # ``variant=ambient`` is an explicit local-admin configuration. The
        # generic ambient bootstrap intentionally leaves its system identity
        # unbound, but an auth-disabled app grants its canonical Default
        # group that configured workload role before SDK users create their
        # first S3 connector. This preflight does the same before the devserver
        # processes start.
        await services.identities.ensure_ambient_s3_identity_bound_to_group(
            default_group.id)
    finally:
        await state_db.close()


def port_is_available(port: int) -> config.Dependency:
    return config.Dependency(
        name=f"port {port} is available",
        command=f'bash -c "! lsof -nP -iTCP:{port} -sTCP:LISTEN"',
        instructions=f"""
Port {port} is currently in use by another program. Consider running the
following commands to kill the offending program or use the RANDOM PORT
option (`RANDOM_PORT=true make devserver-*`).

    kill <PID>
""")


def temporal_workflow_ui_url(instance_id: str) -> str:
    return ('https://cloud.temporal.io/namespaces/devtest.put0h/workflows'
            f'?search=advanced&query=instance_id+%3D"{instance_id}"')


def _run_k8s_pool_preflight(tenant: str) -> None:
    """Interactive setup wizard for ``variant=k8s``.

    devserver resolves the canonical tenant from ``deploy_env_v2()`` once
    and passes it here so the shell wizard is a "dumb" provisioner — no
    duplicate deploy_env logic in bash. The wizard checks AWS SSO,
    kubectl + EKS access, the pool worker image tag in ECR, and that
    ``tenant``'s cluster resources (namespace + ServiceAccount +
    Karpenter NodePools) exist, provisioning anything missing via the
    kubectl-apply script. Idempotent — fully-set-up tenants no-op every
    check.
    """
    import subprocess
    script = osp.join(osp.dirname(__file__), '..', '..', 'scripts', 'dev',
                      'devserver_k8s_setup.sh')
    r = subprocess.run(['bash', osp.abspath(script), tenant])
    if r.returncode != 0:
        raise SystemExit(
            f'k8s pool preflight failed (exit {r.returncode}). '
            f'Re-run `make devserver variant=k8s` after addressing the '
            f'issue above.')


def ui_panel(web_ui_port: int, rest_api_port: int, docker_label: str,
             ui_node_modules_volume: Optional[str] = None) -> config.Panel:
    # Check if catalog development mode is enabled via UI_CATALOG_PATH env var
    ui_catalog_path = os.environ.get('UI_CATALOG_PATH', '')
    ui_catalog_arg = f' {quote(ui_catalog_path)}' if ui_catalog_path else ''
    # The shell script accepts a dependency volume as its fifth argument,
    # after the optional catalog path. Keep legacy's original command shape.
    ui_volume_arg = ''
    if ui_node_modules_volume:
        if not ui_catalog_path:
            ui_volume_arg += " ''"
        ui_volume_arg += f' {quote(ui_node_modules_volume)}'
    return config.Panel(
        title="Web UI",
        command=("./scripts/run_ui_devserver.sh "
                 f"{web_ui_port} {rest_api_port} {docker_label}{ui_catalog_arg}"
                 f'{ui_volume_arg}'),
        ssh_ports=[web_ui_port],
        web_urls=[
            ("Web UI", f"http://localhost:{web_ui_port}/"),
        ],
    )


def _browser_topology_instructions(
    web_ui_port: int,
    rest_api_port: int,
    public_rest_api_port: int,
    *,
    direct_connect_oidc: bool,
) -> str:
    instructions = f"""
Browser UI and internal API origin:
    http://localhost:{web_ui_port}/

Direct internal REST diagnostics:
    http://localhost:{rest_api_port}/docs

Public REST / SDK endpoint:
    http://localhost:{public_rest_api_port}/docs
"""
    if direct_connect_oidc:
        instructions += f"""
Register this exact Direct Connect OIDC callback URI (use localhost):
    http://localhost:{web_ui_port}/v0/auth/callback
"""
    return instructions


def agent_panel(agent_port: int) -> config.Panel:
    return config.Panel(
        title="Agent",
        command=(f"./scripts/run_agent_devserver.sh {agent_port}"),
        ssh_ports=[agent_port],
        web_urls=[
            ("Agent", f"http://localhost:{agent_port}/"),
        ],
    )


def sqlite_db_panel() -> config.Panel:
    return config.Panel(
        title="SQLite DB",
        command="./scripts/sqlite_sql_console.sh",
        ssh_ports=[],
        web_urls=[],
    )


def docker_deps() -> List[config.Dependency]:
    return [
        config.Dependency(
            name="docker is installed and running",
            command="docker ps",
            instructions="""
Please ensure that docker is installed and running.

Mac:
    Installation instructions are here:
        https://docs.docker.com/desktop/install/mac-install/

    It is highly recommended to enable virtiofs for better performance:
        https://www.docker.com/blog/speed-boost-achievement-unlocked-on-docker-desktop-4-6-for-mac/

Ubuntu:
    Installation instructions are here:
        https://docs.docker.com/engine/install/ubuntu/

    If Docker was just installed and `docker ps` fails with:
        permission denied while trying to connect to the docker API

    your current shell likely does not have the updated `docker` group yet.
    Start a new login shell and try again.

""",
        ),
    ]


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; "
        f"got {value!r}")


def temporal_deps(deploy_stage: str, is_local: bool) -> List[config.Dependency]:
    python = quote(sys.executable)
    if is_local:
        return [
            config.Dependency(
                name="Local Temporal Cluster is running", command=(
                    f'DEPLOY_STAGE={deploy_stage} TEMPORAL_SERVER=localhost '
                    f'{python} ./temporalio_commons/wait_server_ready.py -t 10'
                ), instructions="""
Please make sure that you have local Temporal Cluster running by executing:

    cd ~/code/kumo && make up-local-temporal

To tear down the local temporal cluster afterwards, run:
    cd ~/code/kumo && make down-local-temporal
""")
        ]
    return [
        config.Dependency(
            name="temporal is initalized",
            command=f'{python} ./rest/scripts/init_temporal.py {deploy_stage}',
            instructions="""
Please make sure that you have access to temporal cloud. This may involve
configuring the Temporal endpoint and certificate for your deployment.
""",
        )
    ]


def _stern_dep() -> config.Dependency:
    """Stern is the log-tailing CLI used in every variant — cheap to check
    (no kubeconfig required), so it's appended unconditionally."""
    return config.Dependency(
        name="stern is installed", command="stern -v", instructions="""
Please make sure that you have installed stern by running:

    mise install

""")


def _eks_context_dep(pool_strategy: Optional[str]) -> config.Dependency:
    """AWS EKS context setup. Pool+k8s variant targets the testing-uw4
    cluster where pool worker pods are provisioned; legacy default
    variants stay on dev-uw2.

    Context-name convention is split: testing-uw4 is added with
    ``aws eks update-kubeconfig --alias testing-uw4`` (short name);
    dev-uw2 entries are stamped with the full ARN per legacy setup.

    The pool controller pins its Python kubernetes client to the
    right cluster regardless of what kubectl's current-context is
    (see pool/service.py — POOL_K8S_CLUSTER), so the controller can't
    drift. But the operator's interactive ``kubectl`` commands DO
    follow current-context, so set it here for terminal convenience.
    Failing this dependency only blocks devserver startup if the
    context entry doesn't exist in the kubeconfig at all — operator
    needs to ``aws eks update-kubeconfig`` first.
    """
    cluster = ('testing-uw4' if pool_strategy == 'k8s' else 'dev-uw2')
    context = (cluster if cluster == 'testing-uw4' else
               f'arn:aws:eks:us-west-2:926922431314:cluster/{cluster}')
    return config.Dependency(
        name=f"k8s context '{context}' available + selected",
        command=(f"kubectl config get-contexts {context} >/dev/null "
                 f"&& kubectl config use-context {context}"), instructions=f"""
Add the {cluster} context to your kubeconfig:

    aws eks --region us-west-2 update-kubeconfig --name {cluster}

""")


def _needs_eks_context(data_compute: object) -> bool:
    """Whether the AWS EKS context dependency is required for this variant.

    The EKS context check is meaningful iff a Diskgraph pool with
    ``pool_strategy='k8s'`` is provisioned — that's the only path where
    devserver-launched processes (the pool controller's operator-facing
    ``kubectl`` commands, log-tailing via stern, etc.) target the EKS
    cluster from this box.

    Snowflake / Databricks data-compute configs have no diskgraph pool
    and thus no EKS context to set up; their boxes never need kubectl.
    GCP deployments handle K8s via gcloud, not aws-eks, so they're also
    skipped (see ``is_gcp_deploy``).
    """
    return getattr(data_compute, 'pool_strategy', None) == 'k8s'


def k8s_deps(pool_strategy: Optional[str] = None) -> List[config.Dependency]:
    # For GCP deployments, skip AWS EKS context setup; only require
    # stern for log viewing.
    if is_gcp_deploy():
        return [_stern_dep()]
    return [_eks_context_dep(pool_strategy), _stern_dep()]


def _build_dependencies(
    reserved_ports: List[int],
    s3_cmd: str,
    *,
    requires_docker: bool,
) -> List[config.Dependency]:
    """Build the list of dependencies, skipping AWS-specific ones for GCP."""
    dev_deploy_env = deploy_env_v2()
    deploy_stage = dev_deploy_env.deploy_mode.value

    # Base dependencies (required for all deployments)
    python = quote(sys.executable)
    deps: List[config.Dependency] = (
        docker_deps() if requires_docker else []
    ) + [port_is_available(port) for port in reserved_ports] + [
        config.Dependency(
            name="tmux is installed",
            command="tmux -V",
            instructions="""
Please install tmux by running one of the following commands:

    mac: brew install tmux
    ubuntu: sudo apt-get install tmux
""",
        ),
        config.Dependency(
            name="temporalio is installed",
            command=(f"{python} -c "
                     "\"import importlib.metadata as m; "
                     "print('Version:', m.version('temporalio'))\""),
            # temporalio is pinned as a range (>=1.18,<2) in 3rdparty/BUILD;
            # match the major so this gate doesn't rot on every minor bump.
            stdout_search_pattern=r"Version: 1\.",
            instructions="""
Please make sure temporalio is installed at a supported version (the pin lives
in setup.py / 3rdparty/BUILD). If not, (re)install it by running:

    python -m pip uninstall -y temporalio ray && make update-packages

Note: you can safely ignore the warnings about dependency version conflicts.
""",
        ),
        config.Dependency(
            name="kumo is installed",
            command=f'{python} -c "import kumo"',
            instructions="""
Please make sure that kumo and all of its dependencies are installed, and that
your conda/venv/virtualenv is activated. Installation instructions are here:

    ./README.md
""",
        ),
    ]

    # Cloud access checks based on storage backend
    storage_root = dev_deploy_env.data_plane_storage_root
    if fsutil.is_s3(storage_root):
        s3_cmd = ('import boto3; '
                  'boto3.client("s3").head_bucket('
                  'Bucket="kumo-dev-dataplane-bucket")')
        deps.append(
            config.Dependency(
                name="s3 is available",
                command=f"{python} -c {quote(s3_cmd)}",
                instructions="""
Please make sure that you have credentials to access S3.
""",
            ))
    elif fsutil.is_gcs(storage_root):
        deps.append(
            config.Dependency(
                name="gcs is available",
                command=(f"{python} -c "
                         "\"from google.cloud import storage; "
                         "storage.Client().get_bucket('kumo-test-bucket')\""),
                instructions="""
Please make sure that you have credentials to access GCS.
Run: gcloud auth application-default login
""",
            ))
    # local path: no cloud storage check needed

    deps.append(
        config.Dependency(
            name="diskgraph package is installed from the in-tree source",
            command=(f"./scripts/diskgraph verify --python {python}"),
            instructions="""
DiskGraph is not installed from the repository's in-tree source.

Reinstall with:

    make install-diskgraph
""",
        ))

    # Add k8s and temporal deps.
    # The EKS context check (kubectl + dev-uw2/testing-uw4 kubeconfig
    # entry) is only meaningful when this box hosts a diskgraph pool
    # with ``pool_strategy='k8s'``. Singlehost diskgraph variants and
    # Databricks (which has no diskgraph pool) skip the check. GCP variants
    # skip EKS regardless (handled in ``k8s_deps``).
    # Stern is needed only for k8s-backed pools. In particular, do not invoke
    # tool-manager shims for it during a local singlehost launch: no local pane
    # uses stern, and dependency resolution may perform expensive network/tool
    # checks before any listener can become ready.
    if is_gcp_deploy() or _needs_eks_context(dev_deploy_env.data_compute):
        pool_strategy = getattr(dev_deploy_env.data_compute, 'pool_strategy',
                                None)
        deps.extend(k8s_deps(pool_strategy))
    deps.extend(
        temporal_deps(
            deploy_stage,
            isinstance(dev_deploy_env.orchestration, CustomTemporalConfig),
        ))

    return deps


def _validate_devserver_non_prod_env() -> None:
    """Hard-fail if devserver resolves a prod-like environment."""
    dev_deploy_env = deploy_env_v2()
    problems: List[str] = []

    if dev_deploy_env.deploy_mode not in (DeployStage.DEV,
                                          DeployStage.DEV_LOCAL):
        problems.append(f'deploy_mode must be dev/dev_local, got: '
                        f'{dev_deploy_env.deploy_mode.value}')

    if dev_deploy_env.instance_id == 'prod':
        problems.append("instance_id resolves to 'prod', which is not allowed")

    control_root = dev_deploy_env.control_plane_storage_root
    if '/test/prod/' in control_root:
        problems.append(
            f'control_plane_storage_root points to prod path: {control_root}')

    if not problems:
        return

    lines = '\n'.join([f'  - {line}' for line in problems])
    raise RuntimeError(
        'Invalid devserver environment.\n'
        f'Parsed deploy mode: {dev_deploy_env.deploy_mode.value}\n'
        f'Parsed instance_id: {dev_deploy_env.instance_id}\n'
        f'Parsed control_plane_storage_root: {control_root}\n'
        f'Problems:\n{lines}\n'
        'How to fix:\n'
        '  1. Use dev/dev_local DEPLOY_STAGE for devserver.\n'
        '  2. Ensure config roots do not target /test/prod/.\n'
        '  3. Restart the devserver.')


def _optional_env_assignments(names: List[str]) -> str:
    assignments = []
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            assignments.append(f'{name}={quote(value)}')
    if not assignments:
        return ''
    return ' '.join(assignments) + ' '


def _local_worker_metrics_env(port: int) -> str:
    # Local devserver panels should not bind Temporal SDK metrics on the
    # IPv6 wildcard. On Linux, that can collide with an IPv4 listener for the
    # same port and leave a worker pane at a shell while REST health remains OK.
    return f'WORKER_METRICS_HOST=127.0.0.1 WORKER_METRICS_PORT={port} '


# Safe, non-secret provider controls that local devserver child processes need
# in order to exercise workload identity.  Keep this an explicit allowlist:
# access keys, session tokens, provider credentials, and secret values must
# never be copied into pane command lines.
_DEVSERVER_RUNTIME_ENV_NAMES = (
    'AWS_DEFAULT_REGION',
    'DB_STORAGE_ROOT',
    'DEPLOY_ENV_CONFIG_FILE',
    'DEPLOY_STAGE',
    'DEVSERVER_INSTANCE_SUFFIX',
    'INSTANCE_ID',
    'JAEGER_PORT',
    'KUMO_AUTORPC_SERVER_NAME_PREFIX',
    'KUMO_GRAFANA_HOME',
    'KUMO_HOME',
    'KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED',
    'KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED',
    'KUMO_ENABLE_UNAUTHENTICATED_API_KEY_STAGING',
    # Dev-only: trust configured admin_emails for OIDC first-admin even when
    # the IdP (e.g. Entra ID) omits email_verified. Set by the oidc variant.
    'KUMO_DEV_TRUST_ADMIN_EMAILS',
    'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED',
    developer_credentials.ENV_DEV_GCP_USER_CREDENTIALS_ENABLED,
    'MISE_CACHE_DIR',
    'MISE_DATA_DIR',
    # Opt-in, non-secret TT-NG diagnostics/IPC/sampler controls; forward to
    # worker panes.
    'KUMO_TTNG_DEBUG_BATCH_TIMING',
    'KUMO_TTNG_EXPERIMENT_FORCE_SAMPLER_MODE',
    'KUMO_TORCH_SHARING_STRATEGY',
    # REST creates project upload roots and presigns GCS upload parts.  Local
    # authorized-user/external-account ADC does not infer a project the way
    # GCE metadata and service-account JSON do, so every storage-I/O pane
    # needs this safe project selector.
    'GOOGLE_CLOUD_PROJECT',
    'PYTHONPATH',
    'TMPDIR',
    'TEMPORAL_ADDRESS',
    'TEMPORAL_NAMESPACE',
    'TEMPORAL_SERVER',
    'TORCH_HOME',
    'XDG_CACHE_HOME',
    'XDG_CONFIG_HOME',
    'XDG_DATA_HOME',
)
# Safe, non-secret transfer selector used only by connector-I/O workers.  The
# pool controller also forwards it to k8s pool workers.
_CONNECTOR_IO_RUNTIME_ENV_NAMES = ('GCP_STORAGE_TRANSFER_BUCKET', )
_AWS_WORKLOAD_RUNTIME_ENV_NAMES = (
    'AWS_CONTAINER_CREDENTIALS_FULL_URI',
    'AWS_ENDPOINT_URL_S3',
    'AWS_ENDPOINT_URL_STS',
)
_AZURE_WORKLOAD_RUNTIME_ENV_NAMES = ('AZURE_FEDERATED_TOKEN_FILE', )
_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME = (
    'KUMO_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS')
_DEVSERVER_SECRET_ENV_NAMES = (
    'AWS_ACCESS_KEY_ID',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN',
    'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_SECURITY_TOKEN',
    'AWS_SESSION_TOKEN',
    'AZURE_CLIENT_CERTIFICATE_PASSWORD',
    'AZURE_CLIENT_SECRET',
    'AZURE_PASSWORD',
    'AZURE_STORAGE_CONNECTION_STRING',
    'DATABRICKS_CLIENT_SECRET',
    'GOOGLE_CREDENTIALS_STR_JSON',
    'GOOGLE_APPLICATION_CREDENTIALS',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_PRIVATE_KEY',
)
_DEVSERVER_INHERITED_ENV_SCRUB_NAMES = (
    *_DEVSERVER_SECRET_ENV_NAMES,
    # Connector-I/O controls are removed from every pane first and re-added
    # only to the data/ML/pool commands that consume them.
    *_CONNECTOR_IO_RUNTIME_ENV_NAMES,
    # The safe base runtime allowlist re-adds this after scrubbing inheritance.
    'GOOGLE_CLOUD_PROJECT',
    # AWS credential-chain selectors are scrubbed first, then only validated
    # loopback values are re-added when exchange is explicitly enabled.
    'AWS_CONFIG_FILE',
    'AWS_CONTAINER_CREDENTIALS_FULL_URI',
    'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
    'AWS_DEFAULT_PROFILE',
    'AWS_ENDPOINT_URL',
    'AWS_ENDPOINT_URL_S3',
    'AWS_ENDPOINT_URL_STS',
    'AWS_PROFILE',
    'AWS_ROLE_ARN',
    'AWS_ROLE_SESSION_NAME',
    'AWS_SDK_LOAD_CONFIG',
    'AWS_SHARED_CREDENTIALS_FILE',
    'AWS_WEB_IDENTITY_TOKEN_FILE',
    # GCP local static/file credential selectors are deliberately unavailable;
    # GCE metadata remains the selector-free workload identity path.
    'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE',
    'CLOUDSDK_CONFIG',
    # This dev-only selector is validated in the parent process and is never
    # inherited by panes. Only GOOGLE_APPLICATION_CREDENTIALS pointing at the
    # validated external-account document is re-added below.
    _DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME,
    # Azure local static selectors are scrubbed. The dedicated helper re-adds
    # only an absolute existing regular AZURE_FEDERATED_TOKEN_FILE when Azure
    # workload exchange is explicitly enabled.
    'AZURE_AUTHORITY_HOST',
    'AZURE_CLIENT_CERTIFICATE_PATH',
    'AZURE_CLIENT_ID',
    'AZURE_CREDENTIALS_PATH',
    'AZURE_FEDERATED_TOKEN_FILE',
    'AZURE_TENANT_ID',
    'AZURE_USERNAME',
    'IDENTITY_ENDPOINT',
    'IMDS_ENDPOINT',
    'MSI_ENDPOINT',
)


def _explicit_env_bool(name: str) -> bool:
    value = os.environ.get(name, '').strip().lower()
    if value in {'1', 'true', 'yes'}:
        return True
    if value in {'', '0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be an explicit true/false value')


def _validate_loopback_http_url(name: str, value: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f'{name} must be a valid loopback HTTP URL') from error
    if (parsed.scheme != 'http' or not parsed.hostname or port is None
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError(
            f'{name} must be loopback HTTP with an explicit port and no '
            'userinfo, query, or fragment')
    hostname = parsed.hostname.casefold()
    if hostname == 'localhost':
        return
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError(f'{name} must target a loopback address')


def _aws_workload_runtime_env_assignments() -> str:
    """Isolate AWS config and render explicitly enabled local selectors."""
    isolated_config = (f'AWS_CONFIG_FILE={quote(os.devnull)} '
                       f'AWS_SHARED_CREDENTIALS_FILE={quote(os.devnull)} ')
    if not _explicit_env_bool('KUMO_AWS_WORKLOAD_EXCHANGE_ENABLED'):
        return isolated_config
    for name in _AWS_WORKLOAD_RUNTIME_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            _validate_loopback_http_url(name, value)
    return isolated_config + _optional_env_assignments(
        list(_AWS_WORKLOAD_RUNTIME_ENV_NAMES))


def _azure_workload_runtime_env_assignments() -> str:
    """Re-add only a validated local Azure workload token-file path."""
    if not _explicit_env_bool('KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED'):
        return ''
    name = _AZURE_WORKLOAD_RUNTIME_ENV_NAMES[0]
    value = os.environ.get(name)
    if value is None:
        return ''
    if not os.path.isabs(value) or not os.path.isfile(value):
        raise ValueError(f'{name} must be an absolute existing regular file')
    return _optional_env_assignments([name])


def _gcp_workload_runtime_env_assignments() -> str:
    """Re-add one validated external-account file for local GCP exchange."""
    if not _explicit_env_bool('KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED'):
        return ''
    value = os.environ.get(_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME)
    if value is None:
        return ''
    if (not osp.isabs(value) or osp.islink(value) or not osp.isfile(value)):
        raise ValueError(
            f'{_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME} must be an '
            'absolute existing regular non-symlink file')
    try:
        with open(value, encoding='utf-8') as credential_file:
            document = json.load(credential_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(
            f'{_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME} must contain '
            'valid JSON') from None
    if not isinstance(document, dict):
        raise ValueError(
            f'{_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME} must contain '
            'a JSON object')
    if document.get('type') != 'external_account':
        raise ValueError(
            f'{_DEV_GCP_EXTERNAL_ACCOUNT_CREDENTIALS_ENV_NAME} requires '
            'exact credential type external_account')
    return f'GOOGLE_APPLICATION_CREDENTIALS={quote(value)} '


def _devserver_runtime_env(
        *,
        additional_names: tuple[str, ...] = (),
) -> str:
    """Build a pane command prefix that cannot inherit static secrets."""
    scrub = 'env ' + ' '.join(
        f'-u {name}' for name in _DEVSERVER_INHERITED_ENV_SCRUB_NAMES) + ' '
    safe = _optional_env_assignments(
        list((*_DEVSERVER_RUNTIME_ENV_NAMES, *additional_names)))
    return (scrub + safe + _aws_workload_runtime_env_assignments() +
            _azure_workload_runtime_env_assignments() +
            _gcp_workload_runtime_env_assignments())


def _connector_io_runtime_env() -> str:
    """Build the safe environment for data/ML connector-I/O workers."""
    return _devserver_runtime_env(
        additional_names=_CONNECTOR_IO_RUNTIME_ENV_NAMES)


def _ensure_devserver_repo_on_pythonpath() -> None:
    """Prepend this checkout to PYTHONPATH so every pane imports the
    coordinator's revision, not a script-dir or another worktree's editable
    install.  Absolute, because panes may not keep the launch cwd.
    """
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    current = os.environ.get('PYTHONPATH', '')
    entries = [entry for entry in current.split(os.pathsep) if entry]
    entries = [entry for entry in entries if osp.abspath(entry) != repo_root]
    os.environ['PYTHONPATH'] = os.pathsep.join([repo_root, *entries])


def _cgroup_enabled_from_env() -> bool:
    return _explicit_env_bool('PHASE1B_DEVSERVER_CGROUP')


def web_devserver(
    rest_make_target: str,
    random_port: bool = False,
    name: Optional[str] = None,
    base_port: Optional[int] = None,
    instance: Optional[DevserverInstance] = None,
) -> config.Driver:
    python = quote(sys.executable)
    instance = instance or DevserverInstance(
        generate_base_port(random_port=random_port, base_port=base_port))
    instance.apply_environment()
    _ensure_devserver_repo_on_pythonpath()
    base_port = instance.base_port
    ports = instance.ports()
    if not instance.is_legacy:
        os.environ['JAEGER_PORT'] = str(ports['jaeger_otlp'])
    web_ui_port = ports['web_ui']
    rest_api_port = ports['rest_api']
    public_rest_api_port = ports['public_rest_api']
    jaeger_port = ports['jaeger_ui']
    jaeger_otlp_port = ports['jaeger_otlp']
    prometheus_port = ports['prometheus']
    grafana_port = ports['grafana']

    minimal_phase1b_devserver = os.environ.get('PHASE1B_DEVSERVER_MINIMAL',
                                               '').strip().lower() in {
                                                   '1',
                                                   'true',
                                                   'yes',
                                               }
    ui_enabled = (not minimal_phase1b_devserver
                  and _env_bool('DEVSERVER_UI', default=True))
    # The core macOS stack is native. Monitoring remains available as an
    # explicit Docker-backed add-on; Linux retains its historical default.
    monitoring_enabled = (not minimal_phase1b_devserver and _env_bool(
        'DEVSERVER_MONITORING', default=sys.platform != 'darwin'))
    # Child UIs use a per-child Docker dependency volume.  Keep the legacy
    # macOS native UI behavior, but do not let concurrent native ``npm ci``
    # processes mutate one checkout's node_modules tree.
    ui_uses_docker = (not instance.is_legacy or sys.platform != 'darwin'
                      or _env_bool('KUMO_UI_DEVSERVER_DOCKER', default=False))
    requires_docker = monitoring_enabled or (ui_enabled and ui_uses_docker)

    # All docker containers should be given the following label, so that they
    # can be killed gracefully when tmux closes
    docker_label = instance.docker_label
    session_closed_callback: Optional[str] = None
    if requires_docker:
        docker_stop = (
            f'docker stop $(docker ps -q --filter label={docker_label})')
        session_closed_callback = f'bash -c {quote(docker_stop)}'

    disable_auth = os.environ.get('DISABLE_AUTH', 'false')
    batch_prediction_retention_flags = _optional_env_assignments([
        'DELETE_OLD_BP_JOBS',
        'MAX_BP_JOBS_PER_WORKFLOW',
    ])
    kumo_timing_flags = _optional_env_assignments([
        'KUMO_TIMING',
        'KUMO_TIMING_FILE',
    ])
    devserver_runtime_flags = _devserver_runtime_env()
    connector_io_runtime_flags = _connector_io_runtime_env()

    # Inline the REST API launch command based on make target.
    if rest_make_target == 'rest-api-autoreload':
        rest_api_launch_command = (
            'exec watchmedo auto-restart --directory=./ --pattern=*.py '
            f'--recursive -- {python} rest/main.py')
    else:
        rest_api_launch_command = f'{python} rest/main.py'

    rest_api_flags = (f"REST_API_PORT={rest_api_port} "
                      f"DISABLE_AUTH={disable_auth} "
                      f"{kumo_timing_flags}"
                      f"{devserver_runtime_flags}"
                      f"{batch_prediction_retention_flags}")

    dev_deploy_env = deploy_env_v2()
    _validate_devserver_non_prod_env()
    from rest.config.direct_connect import resolve_auth_settings_config
    resolved_auth = resolve_auth_settings_config(dev_deploy_env)
    direct_connect_oidc = (resolved_auth.enabled and
                           resolved_auth.direct_connect.login_method == 'oidc')
    # Diskgraph is the only supported data compute path. The pool
    # controller always runs locally; the singlehost variant adds a
    # local pool-worker panel, the k8s variant relies on cluster pool
    # worker pods provisioned by the controller.
    dc = dev_deploy_env.data_compute
    if getattr(dc, 'pool_strategy', None) == 'k8s':
        # Derive the canonical tenant from the same deploy_env the
        # controller will use, and hand it to the wizard so it
        # provisions exactly what the controller expects.
        import re
        tenant = re.sub(r'[^a-z0-9-]', '-',
                        dev_deploy_env.id_suffix.lower()).lstrip('-')
        _run_k8s_pool_preflight(tenant)
    pool_controller_panel = config.Panel(
        title='pool-controller',
        # The controller runs two metrics endpoints on two distinct ports:
        # the Temporal SDK exporter (WORKER_METRICS_PORT) and its own
        # prometheus_client server (POOL_METRICS_PORT). POOL_METRICS_PORT
        # previously went unset and defaulted to 9090, which the monitoring
        # panel's Prometheus already holds -> EADDRINUSE on boot, taking the
        # diskgraph pool (and any diskgraph training) down. Assign both.
        command=(f"{_local_worker_metrics_env(ports['pool_worker_metrics'])}"
                 f"POOL_METRICS_PORT={ports['pool_metrics']} "
                 f"{kumo_timing_flags}"
                 f"{connector_io_runtime_flags}"
                 f"{python} -m pool.controller_main"))

    pool_worker_panel: Optional[config.Panel] = None
    if getattr(dc, 'pool_strategy', None) == 'singlehost':
        from pool.task_queues import SINGLEHOST_DAEMON_TQ
        pool_task_queue = dev_deploy_env.task_queue(SINGLEHOST_DAEMON_TQ)
        pool_worker_metrics_env = _local_worker_metrics_env(
            ports['pool_worker_worker_metrics'])
        pool_worker_panel = config.Panel(
            title='pool-worker', command=(f"{pool_worker_metrics_env}"
                                          f"POOL_WORKER_UID=singlehost "
                                          f"POOL_TASK_QUEUE={pool_task_queue} "
                                          f"POOL_STORAGE_MODE=daemon-on-pvc "
                                          f"{kumo_timing_flags}"
                                          f"{connector_io_runtime_flags}"
                                          f"{python} -m pool.worker_main"))

    # command for checking permissions to S3
    s3_cmd = (
        'import boto3; '
        'boto3.client("s3").head_bucket(Bucket="kumo-dev-dataplane-bucket")')

    auth_enabled_instructions = """
When logging into the website, you must use a valid username/password or sign
in with SSO.

When using the API, click "Authorize" in the top right corner, and enter a
valid access token. To get a valid token, log into the UI, click CMD + D, and
copy the access token.
"""

    auth_disabled_instructions = """
When logging into the website, you may use a random username or password.

When using the API, click "Authorize" in the top right corner, and enter a
random authentication token.
"""

    disable_auth_bool = disable_auth.lower() == 'true'
    public_rest_worker_metrics_env = _local_worker_metrics_env(
        ports['public_rest_worker_metrics'])
    panels: List[config.Panel] = []
    if ui_enabled:
        panels.append(
            ui_panel(web_ui_port, rest_api_port, docker_label,
                     instance.ui_node_modules_volume()))
    panels.extend([
        config.Panel(
            title="Rest API",
            command=(
                f"{rest_api_flags}"
                f"{_local_worker_metrics_env(ports['rest_worker_metrics'])}"
                f"METRICS_PORT={ports['rest_metrics']} "
                f"{rest_api_launch_command}"),
            ssh_ports=[rest_api_port],
            web_urls=[
                ("Rest API", f"http://localhost:{rest_api_port}/docs"),
            ],
        ),
        config.Panel(
            title="t-worker-dataworkflow",
            command=(
                f"WORKERS={DATA_BACKEND_WORKER_NAME} "
                f"{_local_worker_metrics_env(ports['data_worker_metrics'])}"
                f"{kumo_timing_flags}"
                f"{connector_io_runtime_flags}"
                f"{python} temporalio_commons/worker_main.py"),
            ssh_ports=[],
            web_urls=[
                ("Temporal workflows",
                 temporal_workflow_ui_url(dev_deploy_env.instance_id)),
            ],
        ),
        config.Panel(
            title="t-worker-mlworkflow",
            command=(f"WORKERS={ML_WORKFLOW_WORKER} "
                     f"{_local_worker_metrics_env(ports['ml_worker_metrics'])}"
                     "FAST_TRAIN=1 "
                     f"{kumo_timing_flags}"
                     f"{connector_io_runtime_flags}"
                     f"{python} temporalio_commons/worker_main.py"),
            ssh_ports=[],
            web_urls=[],
        ),
        config.Panel(
            title="Public Rest API",
            command=(f"REST_API_PORT={public_rest_api_port} "
                     f"{public_rest_worker_metrics_env}"
                     f"METRICS_PORT={ports['public_rest_metrics']} "
                     f"DISABLE_AUTH={disable_auth} "
                     f"{kumo_timing_flags}"
                     f"{devserver_runtime_flags}"
                     f"{batch_prediction_retention_flags}"
                     f"{python} rest/public/main.py"),
            ssh_ports=[public_rest_api_port],
            web_urls=[
                ("Public Rest API",
                 f"http://localhost:{public_rest_api_port}/docs"),
            ],
        ),
    ])
    if not minimal_phase1b_devserver:
        panels.append(pool_controller_panel)
        if pool_worker_panel is not None:
            panels.append(pool_worker_panel)
        panels.extend([
            # Print memory usage every 10 seconds
            config.Panel(
                title="memusage",
                command=f"{python} ./tools/print_memory_usage.py",
                ssh_ports=[],
            ),
        ])
        if monitoring_enabled:
            monitoring_command = (f"JAEGER_PORT={jaeger_port} "
                                  f"JAEGER_RESTART=y "
                                  f"GRAFANA_PORT={grafana_port} "
                                  f"REST_API_PORT={rest_api_port} "
                                  f"bash ./monitoring/monitoring.sh")
            if not instance.is_legacy:
                monitoring_command = (f"JAEGER_PORT={jaeger_otlp_port} "
                                      f"JAEGER_RESTART=y "
                                      f"GRAFANA_PORT={grafana_port} "
                                      f"JAEGER_UI_PORT={jaeger_port} "
                                      f"JAEGER_OTLP_PORT={jaeger_otlp_port} "
                                      f"PROMETHEUS_PORT={prometheus_port} "
                                      f"REST_API_PORT={rest_api_port} "
                                      f"{devserver_runtime_flags}"
                                      f"bash ./monitoring/monitoring.sh")
            panels.append(
                config.Panel(
                    title="monitoring",
                    command=monitoring_command,
                    ssh_ports=[jaeger_port, grafana_port],
                    web_urls=[
                        ("Grafana (metrics)",
                         f"http://localhost:{grafana_port}/"),
                        ("Jaeger (tracing)",
                         f"http://localhost:{jaeger_port}/"),
                    ],
                ))
    panels.append(sqlite_db_panel())

    settings = config.Driver(
        name=name
        or os.environ.get('PHASE1B_DEVSERVER_NAME',
                          'website' if instance.is_legacy else instance.name),
        tmux_server_id=instance.name,
        session_closed_callback=session_closed_callback,
        additional_instructions=(_browser_topology_instructions(
            web_ui_port,
            rest_api_port,
            public_rest_api_port,
            direct_connect_oidc=direct_connect_oidc,
        ) + (auth_disabled_instructions
             if disable_auth_bool else auth_enabled_instructions)),
        panels=panels,
        dependencies=_build_dependencies(
            list(
                range(base_port,
                      base_port + (14 if instance.is_legacy else 17))),
            s3_cmd,
            requires_docker=requires_docker,
        ),
        reserved_ports=list(
            range(base_port, base_port + (14 if instance.is_legacy else 17))),
    )
    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoreload', action='store_true',
                        help='Auto-restart REST server on file changes')
    parser.add_argument('--random_port', action='store_true',
                        help='Use a random base port')
    parser.add_argument('--detached', action='store_true',
                        help='Start detached')
    parser.add_argument('--name',
                        default=os.environ.get('PHASE1B_DEVSERVER_NAME'),
                        help='tmux session name for detached devserver')
    parser.add_argument('--base_port', type=int, default=None,
                        help='base port for devserver panels')
    parser.add_argument(
        '--child-id', default=None,
        help='stable child devserver identity for a '
        'coordinator-managed launch')
    parser.add_argument(
        '--cgroup', action='store_true', default=_cgroup_enabled_from_env(),
        help='fence this devserver in a managed cgroup v2 scope')
    parser.add_argument('--cgroup-cpus', type=float,
                        default=os.environ.get('PHASE1B_DEVSERVER_CGROUP_CPUS'),
                        help='CPU quota in cores when --cgroup is set')
    parser.add_argument(
        '--cgroup-memory-gib', type=float,
        default=os.environ.get('PHASE1B_DEVSERVER_CGROUP_MEMORY_GIB'),
        help='memory cap in GiB when --cgroup is set')

    args = parser.parse_args()
    rest_make_target = 'rest-api-autoreload' if args.autoreload else 'rest-api'
    instance = DevserverInstance(
        generate_base_port(random_port=args.random_port,
                           base_port=args.base_port), child_id=args.child_id)
    instance.apply_environment()
    if not instance.is_legacy:
        os.environ['JAEGER_PORT'] = str(instance.ports()['jaeger_otlp'])
    if args.cgroup:
        if args.cgroup_cpus is None or args.cgroup_memory_gib is None:
            raise SystemExit(
                '--cgroup requires --cgroup-cpus and --cgroup-memory-gib '
                '(or PHASE1B_DEVSERVER_CGROUP_CPUS and '
                'PHASE1B_DEVSERVER_CGROUP_MEMORY_GIB)')
        try:
            limits = CgroupLimits(float(args.cgroup_cpus),
                                  float(args.cgroup_memory_gib))
            argv = scope_argv(instance.name, limits)
        except ValueError as error:
            raise SystemExit(f'failed to configure cgroup: {error}') from error
        if not os.environ.get('_DEVSERVER_CGROUP_SCOPE_WRAPPED'):
            os.environ['_DEVSERVER_CGROUP_SCOPE_WRAPPED'] = '1'
            # Re-exec as ``-m rest.devserver.main`` (not sys.argv) so the
            # package root survives inside the scope.
            reexec = ([sys.executable, '-m', 'rest.devserver.main'] +
                      sys.argv[1:] if __package__ else [sys.executable] +
                      sys.argv)
            os.execvp(argv[0], argv + reexec)

    # Step 1: .env is already loaded by _load_env_file_early() at import time.
    # Reject legacy env vars that are now in the JSON config.
    _DEPRECATED_ENV_VARS = [
        'TEMPORAL_SERVER',
    ]
    truthy_values = {'1', 'true', 'yes', 'on', 'enabled'}
    stale = [
        v for v in _DEPRECATED_ENV_VARS
        if os.environ.get(v, '').strip().lower() in truthy_values
    ]
    if stale:
        raise SystemExit(
            f"Legacy env vars detected: {', '.join(stale)}\n"
            f"These are now provided via the deploy config JSON file.\n"
            f"Please unset them (and remove from .env if present): "
            f"unset {' '.join(stale)}")

    # Step 2: DEPLOY_ENV_CONFIG_FILE must already be set (by Makefile).
    if 'DEPLOY_ENV_CONFIG_FILE' not in os.environ:
        raise SystemExit("DEPLOY_ENV_CONFIG_FILE not set. "
                         "Use 'make devserver' or set it manually.")

    # Step 3: Initialize local metadata before concurrently launching panes.
    dev_deploy_env = deploy_env_v2()
    asyncio.run(_initialize_devserver_sqlite_state(dev_deploy_env))

    # Step 4: Build and run devserver using deploy_env() for all config.
    driver = web_devserver(rest_make_target=rest_make_target,
                           random_port=args.random_port, name=args.name,
                           base_port=args.base_port, instance=instance)
    driver.run(detached=args.detached)
