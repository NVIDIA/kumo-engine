# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPERS = REPO / "deploy/shared/scripts/cloud-storage-preflight.sh"
TEMPLATE = REPO / "deploy/shared/configs/deploy_env_config.template.json"
ASSET_INSTALLER = REPO / "deploy/shared/scripts/install-singlehost-assets.sh"


def _bash(
    command: str,
    *,
    env: dict[str, str] | None = None,
    strict: bool = False,
):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {HELPERS!s}; "
            f"{'set -euo pipefail; ' if strict else ''}{command}",
        ],
        text=True,
        capture_output=True,
        env=merged,
        check=False,
    )


def _fake_command(tmp_path: Path, name: str, body: str) -> Path:
    command = tmp_path / name
    command.write_text(f"#!/bin/bash\n{body}")
    command.chmod(0o755)
    return command


def _install_singlehost_assets(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    install_root = tmp_path / "root"
    staging.mkdir()
    for name in (
            "docker-compose.singlehost.yml",
            "docker-compose.singlehost.postgres.yml",
            "docker-compose.singlehost.gpu.yml",
    ):
        shutil.copy(REPO / "docker/compose" / name, staging / name)
    for source, destination in (
        (REPO / "docker/compose/config/postgres-initdb",
         staging / "postgres-initdb"),
        (REPO / "docker/compose/config/temporal", staging / "temporal"),
        (REPO / "deploy/shared", staging / "deploy-shared"),
    ):
        shutil.copytree(source, destination)

    subprocess.run(
        ["bash", str(ASSET_INSTALLER),
         str(staging),
         str(install_root)],
        check=True,
    )
    return install_root


def test_singlehost_asset_installer_preserves_common_layout(tmp_path):
    install_root = _install_singlehost_assets(tmp_path)
    kumo_root = install_root / "opt/kumo"
    assert all((kumo_root / "compose" / name).is_file() for name in (
        "docker-compose.singlehost.yml",
        "docker-compose.singlehost.postgres.yml",
        "docker-compose.singlehost.gpu.yml",
    ))
    assert (kumo_root /
            "compose/config/temporal/dynamicconfig/docker.yaml").is_file()
    # Bootstrap selects and writes the nginx config.
    assert (kumo_root / "nginx").is_dir()
    assert not (kumo_root / "nginx/nginx.conf").exists()
    prometheus = (kumo_root / "prometheus/prometheus.yml").read_text()
    assert 'targets: ["kumo-rest:8007"]' in prometheus
    assert 'targets: ["localhost:8007"]' not in prometheus
    assert (kumo_root / "configs/deploy_env_config.template.json").is_file()
    assert (kumo_root / "deploy-shared/scripts/kumo-bootstrap.py").is_file()
    assert (kumo_root / "deploy-shared/nginx/nginx.conf").is_file()
    assert (kumo_root / "deploy-shared/nginx/nginx.http.conf").is_file()
    for nginx in (kumo_root / "deploy-shared/nginx").glob("*.conf"):
        text = nginx.read_text()
        assert "X-Forwarded-For $remote_addr" in text
        assert "$proxy_add_x_forwarded_for" not in text
    assert all(
        os.access(script, os.X_OK)
        for script in (kumo_root / "scripts").glob("*.sh"))
    assert (install_root /
            "etc/systemd/system/kumo-mount-nvme.service").is_file()
    assert (install_root / "etc/systemd/system/docker.service.d/"
            "kumo-mount-nvme.conf").is_file()


def test_singlehost_packers_use_common_asset_bundle():
    for template in (
            REPO / "deploy/ami/kumo-singlehost.pkr.hcl",
            REPO / "deploy/gcp/kumo-singlehost-gcp.pkr.hcl",
            REPO / "deploy/azure/kumo-singlehost-azure.pkr.hcl",
    ):
        text = template.read_text()
        assert text.count('source      = "../shared/"') == 1
        assert text.count('destination = "/tmp/deploy-shared"') == 1
        assert text.count("sudo bash /tmp/deploy-shared/scripts/"
                          "install-singlehost-assets.sh") == 1


@pytest.mark.parametrize(
    "enable_tls,generate_tls,variant,expect_cert",
    [(False, True, "nginx.http.conf", False), (True, True, "nginx.conf", True),
     (True, False, "nginx.conf", False)],
)
def test_installed_singlehost_bootstrap_materializes_assets(
    tmp_path,
    enable_tls,
    generate_tls,
    variant,
    expect_cert,
):
    install_root = _install_singlehost_assets(tmp_path)
    kumo_root = install_root / "opt/kumo"
    output_root = tmp_path / "output"
    nginx_file = output_root / "nginx/nginx.conf"
    tls_dir = output_root / "tls"
    runtime_config_root = REPO / "docker/compose/config"
    (kumo_root / "deploy-shared/runtime-config").symlink_to(
        runtime_config_root, target_is_directory=True)
    postgres_init_file = output_root / "postgres-initdb/init.sh"
    temporal_config_file = output_root / "temporal/dynamicconfig/docker.yaml"
    env = os.environ.copy()
    env.update({
        "DISABLE_AUTH":
        "true",
        "ENABLE_TLS":
        str(enable_tls).lower(),
        "KUMO_BOOTSTRAP_GENERATE_TLS":
        str(generate_tls).lower(),
        "KUMO_POSTGRES_MODE":
        "bundled",
        "POSTGRES_PASSWORD":
        "test-password",
        "DATAPLANE_ROOT":
        "/tmp/dataplane",
        "ARTIFACT_ROOT":
        "/tmp/artifacts",
        "KUMO_BOOTSTRAP_OWNER_UID":
        str(os.getuid()),
        "KUMO_BOOTSTRAP_OWNER_GID":
        str(os.getgid()),
        "KUMO_BOOTSTRAP_SECRETS_DIR":
        str(output_root / "secrets"),
        "KUMO_BOOTSTRAP_DEPLOY_CONFIG_FILE":
        str(output_root / "config/deploy_env_config.json"),
        "KUMO_BOOTSTRAP_DEPLOY_CONFIG_SOURCE":
        "kumo_config",
        "KUMO_BOOTSTRAP_NGINX_FILE":
        str(nginx_file),
        "KUMO_BOOTSTRAP_PROMETHEUS_FILE":
        str(output_root / "prometheus/prometheus.yml"),
        "KUMO_BOOTSTRAP_TLS_DIR":
        str(tls_dir),
        "KUMO_BOOTSTRAP_POSTGRES_INIT_FILE":
        str(postgres_init_file),
        "KUMO_BOOTSTRAP_TEMPORAL_DYNAMIC_CONFIG_FILE":
        str(temporal_config_file),
    })
    result = subprocess.run(
        [
            sys.executable,
            str(kumo_root / "deploy-shared/scripts/kumo-bootstrap.py"),
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    nginx_config = nginx_file.read_text()
    assert nginx_config == (kumo_root / "deploy-shared/nginx" /
                            variant).read_text()
    assert "resolver 127.0.0.11" in nginx_config
    assert nginx_config.count(" resolve;") == 3
    assert (tls_dir / "server.crt").exists() is expect_cert
    assert (tls_dir / "server.key").exists() is expect_cert
    assert postgres_init_file.read_text() == (
        runtime_config_root /
        "postgres-initdb/01-create-kumo-databases.sh").read_text()
    assert temporal_config_file.read_text() == (
        runtime_config_root / "temporal/dynamicconfig/docker.yaml").read_text()
    assert (postgres_init_file.stat().st_mode & 0o777) == 0o755
    assert (temporal_config_file.stat().st_mode & 0o777) == 0o644

    deploy_config_file = output_root / "config/deploy_env_config.json"
    initial_config = deploy_config_file.read_bytes()
    unchanged = subprocess.run(result.args, text=True, capture_output=True,
                               env=env, check=False)
    assert unchanged.returncode == 0, unchanged.stderr
    assert deploy_config_file.read_bytes() == initial_config

    env["DATAPLANE_ROOT"] = "/tmp/changed-dataplane"
    updated = subprocess.run(result.args, text=True, capture_output=True,
                             env=env, check=False)
    assert updated.returncode == 0, updated.stderr
    assert "updated generated deploy config" in updated.stdout
    assert "/tmp/dataplane" not in updated.stdout
    assert "/tmp/changed-dataplane" not in updated.stdout
    assert json.loads(
        deploy_config_file.read_text(encoding="utf-8")
    )["storage"]["data_plane_storage_root"] == "/tmp/changed-dataplane"

    env.pop("KUMO_BOOTSTRAP_DEPLOY_CONFIG_SOURCE")
    env["DATAPLANE_ROOT"] = "/tmp/custom-source-is-not-rewritten"
    custom_source = deploy_config_file.read_bytes()
    skipped = subprocess.run(result.args, text=True, capture_output=True,
                             env=env, check=False)
    assert skipped.returncode == 0, skipped.stderr
    assert deploy_config_file.read_bytes() == custom_source


@pytest.mark.parametrize(
    "uri,expected",
    [
        (
            "abfss://data@acct.dfs.core.windows.net/a/b",
            "data|acct|dfs.core.windows.net|a/b|AzureCloud|"
            "abfss://data@acct.dfs.core.windows.net/a/b",
        ),
        (
            "abfs://data@acct.dfs.core.usgovcloudapi.net/",
            "data|acct|dfs.core.usgovcloudapi.net||AzureUSGovernment|"
            "abfss://data@acct.dfs.core.usgovcloudapi.net",
        ),
        (
            "abfss://data@acct.dfs.core.chinacloudapi.cn/path",
            "data|acct|dfs.core.chinacloudapi.cn|path|AzureChinaCloud|"
            "abfss://data@acct.dfs.core.chinacloudapi.cn/path",
        ),
    ],
)
def test_parse_adls_root_routes_sovereign_cloud(uri, expected):
    result = _bash(
        f"parse_adls_root {uri!r} && "
        'printf "%s|%s|%s|%s|%s|%s" "$ADLS_FILESYSTEM" "$ADLS_ACCOUNT" '
        '"$ADLS_ENDPOINT_SUFFIX" "$ADLS_PREFIX" "$ADLS_CLOUD" '
        '"$ADLS_CANONICAL_ROOT"')
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/path",
        "abfss://missing-at/path",
        "abfss://data@acct.dfs.example.com/path",
        "abfss://@acct.dfs.core.windows.net/path",
        "abfss://data@acct.dfs.core.windows.net/a/../b",
        "abfss://data@acct.dfs.core.windows.net/a//b",
        "abfss://data@acct.dfs.core.windows.net/a%2Fb",
        "abfss://data@acct.dfs.core.windows.net/path?query=x",
        "abfss://data@acct.dfs.core.windows.net/path#fragment",
        "abfss://data@acct@other.dfs.core.windows.net/path",
    ],
)
def test_parse_adls_root_rejects_invalid_or_unsupported_uri(uri):
    assert _bash(f"parse_adls_root {uri!r}").returncode != 0


@pytest.mark.parametrize(
    "root,remote",
    [
        ("s3://bucket/p", True),
        ("gs://bucket/p", True),
        ("abfs://data@acct.dfs.core.windows.net/p", True),
        ("abfss://data@acct.dfs.core.windows.net/p", True),
        ("/data/kumo", False),
        ("relative", False),
    ],
)
def test_remote_storage_classification(root, remote):
    assert (_bash(f"is_remote_storage_root {root!r}").returncode == 0) is remote


def test_legacy_abfs_dataplane_is_canonicalized_before_runtime():
    legacy = "abfs://data@acct.dfs.core.usgovcloudapi.net/kumo/root"
    result = _bash(f"canonicalize_adls_root {legacy!r}")
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "abfss://data@acct.dfs.core.usgovcloudapi.net/kumo/root")

    firstboot = (REPO / "deploy/shared/scripts/kumo-firstboot.sh").read_text()
    assert 'DATAPLANE_ROOT="$canonical_dataplane_root"' in firstboot
    assert "abfss://*)" in firstboot


@pytest.mark.parametrize(
    "value,expected",
    [
        ("public", "AzureCloud"),
        ("AzureCloud", "AzureCloud"),
        ("usgovernment", "AzureUSGovernment"),
        ("china", "AzureChinaCloud"),
    ],
)
def test_normalize_azure_cloud(value, expected):
    result = _bash(f"normalize_azure_cloud {value!r}")
    assert result.returncode == 0
    assert result.stdout.strip() == expected


def test_azure_metadata_detection_config_and_region(tmp_path):
    encoded = base64.b64encode(b"dataplane_root: abfss://data\n").decode()
    _fake_command(
        tmp_path, "curl", "case \"$*\" in\n"
        "  *latest/meta-data/instance-id*|*metadata.google.internal*) exit 1 ;;\n"
        f"  *compute/userData*) echo {encoded!r} ;;\n"
        "  *compute/location*) echo westus2 ;;\n"
        "  *compute/vmId*) echo vm-id ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n")
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}

    result = _bash(
        'printf "%s|" "$(detect_cloud_provider)"; '
        'printf "%s|" "$(read_cloud_config azure)"; '
        'cloud_region azure',
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "azure|dataplane_root: abfss://data|westus2\n"


def _install_fake_az(
    tmp_path: Path, ) -> tuple[dict[str, str], Path, Path, Path]:
    log = tmp_path / "az.log"
    blob = tmp_path / "blob"
    directory = tmp_path / "directory"
    _fake_command(
        tmp_path, "az", 'echo "$*" >> "$AZ_LOG"\n'
        'if [[ "$*" == *"$AZ_FAIL_ON"* ]] && [[ -n "$AZ_FAIL_ON" ]]; then exit 9; fi\n'
        'if [[ "$*" == *"directory exists"* ]]; then\n'
        '  [[ -f "$AZ_DIRECTORY" ]] && echo true || echo false\n'
        'elif [[ "$*" == *"directory create"* ]]; then touch "$AZ_DIRECTORY"\n'
        'elif [[ "$*" == *"file list"* || "$*" == *"file upload"* ]]; then\n'
        '  args=("$@"); target=""\n'
        '  for ((i=0; i<${#args[@]}; i++)); do\n'
        '    [[ "${args[i]}" == --path ]] && target="${args[i+1]}"\n'
        '  done\n'
        '  [[ "$target" == */* && ! -f "$AZ_DIRECTORY" ]] && exit 12\n'
        '  if [[ "$*" == *"file upload"* ]]; then\n'
        '  while [[ $# -gt 0 ]]; do [[ "$1" == --source ]] && { cp "$2" "$AZ_BLOB"; exit; }; shift; done\n'
        '  fi\n'
        'elif [[ "$*" == *"file download"* ]]; then\n'
        '  while [[ $# -gt 0 ]]; do [[ "$1" == --destination ]] && { cp "$AZ_BLOB" "$2"; exit; }; shift; done\n'
        'elif [[ "$*" == *"generate-sas"* ]]; then echo signed-token\n'
        'elif [[ "$*" == *"file delete"* ]]; then rm -f "$AZ_BLOB"\n'
        "fi\n")
    return {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "AZ_LOG": str(log),
        "AZ_BLOB": str(blob),
        "AZ_DIRECTORY": str(directory),
        "AZURE_CLIENT_ID": "runtime-identity-client-id",
    }, log, blob, directory


def test_adls_preflight_proves_all_operations_and_cleans_probe(tmp_path):
    env, log, blob, directory = _install_fake_az(tmp_path)
    result = _bash(
        "check_adls_storage_access "
        "abfss://data@acct.dfs.core.usgovcloudapi.net/kumo/root",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    for operation in (
            "cloud set --name AzureUSGovernment",
            "login --identity --allow-no-subscriptions --only-show-errors "
            "--client-id runtime-identity-client-id",
            "file list",
            "directory exists",
            "directory create",
            "file upload",
            "file download",
            "generate-sas",
            "file delete",
    ):
        assert operation in calls
    assert "kumo/root/.kumo-firstboot-access-check" in calls
    assert not blob.exists()
    assert directory.exists()


def test_adls_preflight_fails_closed_and_cleanup_runs(tmp_path):
    env, log, blob, directory = _install_fake_az(tmp_path)
    env["AZ_FAIL_ON"] = "generate-sas"
    result = _bash(
        "check_cloud_storage_access "
        "abfss://data@acct.dfs.core.windows.net/kumo/root",
        env=env,
        strict=True,
    )
    assert result.returncode != 0
    assert "file delete" in log.read_text()
    assert not blob.exists()
    assert directory.exists()


def test_deploy_template_renders_explicit_lifecycle_without_credentials(
        tmp_path):
    env = {
        "DATAPLANE_ROOT": "abfss://data@acct.dfs.core.windows.net/kumo",
        "DISKGRAPH_DATA_ROOT": "/mnt/localssd",
        "TEMPORAL_HOST": "temporal",
        "TEMPORAL_PORT": "7233",
        "TEMPORAL_NAMESPACE": "prod",
        "METADATA_STORAGE_TYPE": "postgres",
        "STATE_DB_DSN_SECRET_NAME": "STATE_DB_DSN",
        "ARTIFACT_ROOT": "abfss://artifacts@acct.dfs.core.windows.net/kumo",
        "LIFECYCLE_READY": "true",
    }
    result = subprocess.run(
        ["envsubst"],
        input=TEMPLATE.read_text(),
        text=True,
        capture_output=True,
        env={
            **os.environ,
            **env
        },
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["storage"]["lifecycle_ready"] is True
    assert "secret" not in json.dumps(config["storage"]).lower()


def test_azure_iac_user_data_contains_only_secret_references():
    terraform = (REPO / "deploy/azure/terraform/main.tf").read_text()
    assert '"oidc_client_secret: ' not in terraform
    assert '"direct_connect_admin_temp_password: ' not in terraform
    assert '"oidc_client_secret_uri: ' in terraform
    assert '"direct_connect_admin_temp_password_uri: ' in terraform
    assert ('"azure_client_id: '
            '${azurerm_user_assigned_identity.kumo.client_id}"' in terraform)
    assert 'role_definition_name = "Key Vault Secrets User"' in terraform
    assert 'check "storage_account_name_is_valid"' in terraform
    assert 'regex("^[a-z0-9]{3,24}$", local.storage_account_name)' in terraform
    assert "boot_diagnostics {}" in terraform
    assert all(prefix in terraform for prefix in (
        "tmp_payloads/",
        "temporary-uploaded-datasets/",
        "tmp_archives/",
    ))
    gcp = (REPO / "deploy/gcp/terraform/main.tf").read_text()
    assert all(prefix in gcp for prefix in (
        "tmp_payloads/",
        "temporary-uploaded-datasets/",
        "tmp_archives/",
    ))


def test_azure_sovereign_runtime_and_registry_settings_are_wired():
    compose = (REPO /
               "docker/compose/docker-compose.singlehost.yml").read_text()
    assert "AZURE_AUTHORITY_HOST: ${AZURE_AUTHORITY_HOST:-}" in compose
    assert "AZURE_CLIENT_ID: ${AZURE_CLIENT_ID:-}" in compose

    packer = (REPO / "deploy/azure/kumo-singlehost-azure.pkr.hcl").read_text()
    compact_packer = "".join(packer.split())
    for environment, suffix in (
        ("public", "azurecr.io"),
        ("usgovernment", "azurecr.us"),
        ("china", "azurecr.cn"),
    ):
        assert f'{environment}="{suffix}"' in compact_packer
    assert 'acr="${var.acr_name}.${local.acr_suffix}"' in compact_packer


def _install_fake_aws(
    tmp_path: Path, ) -> tuple[dict[str, str], Path, Path, None]:
    log = tmp_path / "aws.log"
    blob = tmp_path / "blob"
    _fake_command(
        tmp_path, "aws", 'echo "$*" >> "$AWS_LOG"\n'
        'if [[ -n "$AWS_FAIL_ON" && "$*" == *"$AWS_FAIL_ON"* ]]; then exit 8; fi\n'
        'if [[ "$*" == *"put-object"* ]]; then\n'
        '  while [[ $# -gt 0 ]]; do [[ "$1" == --body ]] && { cp "$2" "$AWS_BLOB"; exit; }; shift; done\n'
        'elif [[ "$*" == *"get-object"* ]]; then cp "$AWS_BLOB" "${@: -1}"\n'
        'elif [[ "$*" == *"s3 presign"* ]]; then echo signed-url\n'
        'elif [[ "$*" == *"delete-object"* ]]; then rm -f "$AWS_BLOB"\n'
        "fi\n")
    return {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "AWS_LOG": str(log),
        "AWS_BLOB": str(blob),
        "AWS_FAIL_ON": "",
        "AWS_REGION": "us-west-2",
    }, log, blob, None


def _install_fake_gcs(
    tmp_path: Path, ) -> tuple[dict[str, str], Path, Path, None]:
    log = tmp_path / "gcs.log"
    blob = tmp_path / "blob"
    _fake_command(
        tmp_path, "gsutil", 'echo "gsutil $*" >> "$GCS_LOG"\n'
        'if [[ -n "$GCS_FAIL_ON" && "$*" == *"$GCS_FAIL_ON"* ]]; then exit 7; fi\n'
        'if [[ "$*" == *" cp "* ]]; then\n'
        '  args=("$@"); src="${args[-2]}"; dst="${args[-1]}"\n'
        '  if [[ "$src" == gs://* ]]; then cp "$GCS_BLOB" "$dst"; else cp "$src" "$GCS_BLOB"; fi\n'
        'elif [[ "$*" == *" rm "* ]]; then rm -f "$GCS_BLOB"\n'
        "fi\n")
    _fake_command(
        tmp_path, "gcloud", 'echo "gcloud $*" >> "$GCS_LOG"\n'
        'if [[ -n "$GCS_FAIL_ON" && "$*" == *"$GCS_FAIL_ON"* ]]; then exit 7; fi\n'
        "echo signed-url\n")
    return {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GCS_LOG": str(log),
        "GCS_BLOB": str(blob),
        "GCS_FAIL_ON": "",
    }, log, blob, None


@pytest.mark.parametrize(
    ('provider', 'root', 'failure', 'operations'),
    [
        ('s3', 's3://bucket/root', 's3 presign',
         ('list-objects-v2', 'put-object', 'get-object', 's3 presign',
          'delete-object')),
        ('gcs', 'gs://bucket/root', 'storage sign-url',
         ('gsutil ls', 'gsutil -q cp', 'gcloud storage sign-url',
          'gsutil -q rm')),
    ],
)
def test_object_preflight_reads_signs_and_cleans_on_failure(
        tmp_path, provider, root, failure, operations):
    install = _install_fake_aws if provider == 's3' else _install_fake_gcs
    env, log, blob, _ = install(tmp_path)
    env['AWS_FAIL_ON' if provider == 's3' else 'GCS_FAIL_ON'] = failure
    result = _bash(f'check_cloud_storage_access {root}', env=env, strict=True)

    assert result.returncode != 0
    calls = log.read_text()
    assert all(operation in calls for operation in operations)
    if provider == 'gcs':
        assert calls.count('gsutil -q cp') == 2
    assert not blob.exists()


@pytest.mark.parametrize("provider", ["s3", "gcs", "adls"])
def test_storage_preflight_wrapper_is_strict_shell_safe(tmp_path, provider):
    if provider == "s3":
        env, _log, blob, directory = _install_fake_aws(tmp_path)
        root = "s3://bucket/root"
    elif provider == "gcs":
        env, _log, blob, directory = _install_fake_gcs(tmp_path)
        root = "gs://bucket/root"
    else:
        env, _log, blob, directory = _install_fake_az(tmp_path)
        root = "abfss://data@acct.dfs.core.windows.net/kumo/root"

    result = _bash(
        f"check_cloud_storage_access {root}; printf success",
        env=env,
        strict=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("success")
    assert not blob.exists()
    if directory is not None:
        assert directory.exists()


def test_adls_rootless_preflight_does_not_create_or_remove_root(tmp_path):
    env, log, blob, directory = _install_fake_az(tmp_path)

    result = _bash(
        "check_cloud_storage_access "
        "abfss://data@acct.dfs.core.windows.net; printf success",
        env=env,
        strict=True,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "directory exists" not in calls
    assert "directory create" not in calls
    assert not directory.exists()
    assert not blob.exists()
