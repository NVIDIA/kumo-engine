#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$REPO_ROOT/docker/compose"
CONFIG_DIR="$COMPOSE_DIR/config"
REPO_LOCAL_CACHE_ROOT="$REPO_ROOT/.cache/kumo"
COMPOSE_ENV_FILE="${COMPOSE_LOCAL_BASE_ENV_FILE:-$COMPOSE_DIR/.env.compose-local}"
if [[ "$COMPOSE_ENV_FILE" != /* ]]; then
    COMPOSE_ENV_FILE="$REPO_ROOT/$COMPOSE_ENV_FILE"
fi
COMPOSE_LOCAL_ENV_FILE="$COMPOSE_DIR/.env.compose-local.local"
COMPOSE_LOCAL_EXTRA_ENV_FILES="${COMPOSE_LOCAL_EXTRA_ENV_FILES:-}"
COMPOSE_LOCAL_EXTRA_COMPOSE_FILES="${COMPOSE_LOCAL_EXTRA_COMPOSE_FILES:-}"
if [[ -x /usr/bin/python3 ]]; then
    PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
# Caller env wins over env-files (matches docker compose's own precedence).
# Keep this Bash 3.2 compatible for macOS.
_CALLER_ENV_VARS="$(compgen -e)"

caller_env_has() {
    local key="$1"
    case $'\n'"$_CALLER_ENV_VARS"$'\n' in
        *$'\n'"$key"$'\n'*) return 0 ;;
        *) return 1 ;;
    esac
}

export_temporal_postgres_env() {
    export TEMPORAL_SQL_TLS=false
    export TEMPORAL_SQL_TLS_ENABLED=false
    export TEMPORAL_SQL_TLS_CA_FILE="${POSTGRES_SSLROOTCERT:-}"
    export TEMPORAL_SQL_TLS_SERVER_NAME=""
    export TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION=false
    export TEMPORAL_SQL_HOST_VERIFICATION=false

    case "${POSTGRES_SSLMODE:-}" in
        "")
            ;;
        verify-ca)
            export TEMPORAL_SQL_TLS=true
            export TEMPORAL_SQL_TLS_ENABLED=true
            export TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION=true
            ;;
        verify-full)
            export TEMPORAL_SQL_TLS=true
            export TEMPORAL_SQL_TLS_ENABLED=true
            export TEMPORAL_SQL_TLS_SERVER_NAME="${POSTGRES_HOST:-}"
            export TEMPORAL_SQL_HOST_VERIFICATION=true
            ;;
        *)
            echo "ERROR: POSTGRES_SSLMODE must be verify-ca or verify-full" >&2
            exit 1
            ;;
    esac
}

DOCKER_COMPOSE_CMD=()

resolve_docker_compose() {
    if [[ -n "${DOCKER_COMPOSE:-}" ]]; then
        DOCKER_COMPOSE_CMD=($DOCKER_COMPOSE)
    elif docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE_CMD=(docker-compose)
    elif [[ -x /opt/homebrew/lib/docker/cli-plugins/docker-compose ]]; then
        DOCKER_COMPOSE_CMD=(/opt/homebrew/lib/docker/cli-plugins/docker-compose)
    else
        echo "ERROR: docker compose is required for compose-local. Install the Docker Compose plugin or docker-compose." >&2
        exit 1
    fi
}

DEFAULT_PRODUCTION_LOCAL_DATAPLANE_ROOT='auto'
DEFAULT_PRODUCTION_LOCAL_ARTIFACT_ROOT='auto'
DEFAULT_LOCAL_DATAPLANE_ROOT="$REPO_LOCAL_CACHE_ROOT/compose-local/dataplane/{instance_id}/"
DEFAULT_LOCAL_ARTIFACT_ROOT="$REPO_LOCAL_CACHE_ROOT/compose-local/dataplane/artifacts/{instance_id}/"
DEFAULT_AWS_DATAPLANE_ROOT='s3://kumo-test-dataplane-bucket/compose-local/{instance_id}/'
DEFAULT_AWS_ARTIFACT_ROOT='s3://kumo-public-datasets'
DEFAULT_GCP_DATAPLANE_ROOT='gs://kumo-test-bucket/compose-local/{instance_id}/'
DEFAULT_GCP_ARTIFACT_ROOT='gs://kumo-public-datasets-bigquery'
DEFAULT_AZURE_DATAPLANE_ROOT='abfss://test-container@kumotestdataplane.dfs.core.windows.net/devserver2/{instance_id}/'
DEFAULT_AZURE_ARTIFACT_ROOT='abfss://kumopublic@kumopublicdatasets.dfs.core.windows.net/'

default_port_offset() {
    local stack="$1"
    if [[ "$stack" == "kumo" ]]; then
        echo 0
    elif [[ "$stack" =~ ([0-9]+)$ ]]; then
        echo $((${BASH_REMATCH[1]} * 100))
    else
        echo 100
    fi
}

stack_scoped_root() {
    local root="${1%/}"
    local stack="$2"
    local marker="{instance_id}"
    if [[ "$root" == *"/$stack" || "$root" == *"/$stack/"* ]]; then
        echo "$root/"
    elif [[ "$root" == *"$marker"* ]]; then
        local prefix="${root%%"$marker"*}"
        local suffix="${root#*"$marker"}"
        echo "${prefix%/}/$stack/$marker$suffix/"
    else
        echo "$root/$stack/"
    fi
}

stack_scoped_local_subdir() {
    local root="${1%/}"
    local stack="$2"
    local leaf="$3"
    if [[ "$root" == *"/$stack/$leaf" || "$root" == *"/$stack/$leaf/" ]]; then
        echo "$root"
    else
        echo "$root/$stack/$leaf"
    fi
}

# Reclaim ownership of the script-managed default secrets dir in case a prior
# bootstrap container left it unwritable for this unprivileged host user.
# A user-supplied KUMO_SECRETS_DIR is never chowned here; it is guarded
# at write time via secrets_dir_writable instead.
reclaim_default_secrets_dir() {
    local secrets_dir="$1"
    [[ -e "$secrets_dir" ]] || return 0
    if [[ -O "$secrets_dir" && -w "$secrets_dir" && -x "$secrets_dir" ]]; then
        return 0
    fi
    echo "==> $secrets_dir is owned by another user (stale uid 999 from a prior" >&2
    echo "    kumo-bootstrap run) — reclaiming ownership with sudo" >&2
    if ! sudo -n chown -R "$(id -u):$(id -g)" "$secrets_dir" 2>/dev/null; then
        die_secrets_not_writable "$secrets_dir"
    fi
}

# Returns true when we can create and replace the Postgres secret files in
# $1: the dir must be writable and traversable, and any pre-existing
# DSN/password file must be writable too.
secrets_dir_writable() {
    local dir="$1" name path
    [[ -d "$dir" && -w "$dir" && -x "$dir" ]] || return 1
    for name in STATE_DB_DSN TEMPORAL_DB_DSN TEMPORAL_VISIBILITY_DB_DSN \
                kumo-postgres-password; do
        path="$dir/$name"
        [[ ! -e "$path" || -w "$path" ]] || return 1
    done
    return 0
}

die_secrets_not_writable() {
    local secrets_dir="$1"
    echo "ERROR: secrets dir '$secrets_dir' is not writable by $(id -un) (uid $(id -u))." >&2
    echo "       A previous kumo-bootstrap run likely chowned it to uid 999. Fix with:" >&2
    echo "         sudo chown -R \"\$(id -u):\$(id -g)\" '$secrets_dir'" >&2
    echo "       or remove it to regenerate:" >&2
    echo "         sudo rm -rf '$secrets_dir'" >&2
    echo "       then re-run 'make compose-local'." >&2
    exit 1
}

is_remote_root() {
    [[ "$1" =~ ^[A-Za-z][A-Za-z0-9+.-]*:// ]]
}

detect_cloud_provider() {
    if [[ "${COMPOSE_LOCAL_CLOUD_PROVIDER:-auto}" != "auto" ]]; then
        echo "$COMPOSE_LOCAL_CLOUD_PROVIDER"
        return
    fi
    local aws_token
    aws_token=$(curl -sf -m 1 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
        http://169.254.169.254/latest/api/token 2>/dev/null || true)
    if [[ -n "$aws_token" ]] && curl -sf -m 1 -H "X-aws-ec2-metadata-token: $aws_token" \
        http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
        echo "aws"
    elif curl -sf -m 1 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
        echo "aws"
    elif curl -sf -m 1 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/id >/dev/null 2>&1; then
        echo "gcp"
    elif curl -sf -m 1 -H "Metadata:true" \
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01" >/dev/null 2>&1; then
        echo "azure"
    else
        echo "unknown"
    fi
}

cloud_default_dataplane_root() {
    case "$1" in
        aws) echo "$DEFAULT_AWS_DATAPLANE_ROOT" ;;
        gcp) echo "$DEFAULT_GCP_DATAPLANE_ROOT" ;;
        azure) echo "$DEFAULT_AZURE_DATAPLANE_ROOT" ;;
        *) echo "" ;;
    esac
}

cloud_default_artifact_root() {
    case "$1" in
        aws) echo "$DEFAULT_AWS_ARTIFACT_ROOT" ;;
        gcp) echo "$DEFAULT_GCP_ARTIFACT_ROOT" ;;
        azure) echo "$DEFAULT_AZURE_ARTIFACT_ROOT" ;;
        *) echo "$DEFAULT_LOCAL_ARTIFACT_ROOT" ;;
    esac
}

detect_aws_region() {
    if [[ -n "${AWS_REGION:-}" ]]; then
        echo "$AWS_REGION"
        return
    fi
    local token region
    token=$(curl -sf -m 1 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
        http://169.254.169.254/latest/api/token 2>/dev/null || true)
    if [[ -n "$token" ]]; then
        region=$(curl -sf -m 1 -H "X-aws-ec2-metadata-token: $token" \
            http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
    fi
    if [[ -z "${region:-}" ]]; then
        region=$(curl -sf -m 1 \
            http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
    fi
    echo "${region:-us-west-2}"
}

check_s3_root_access() {
    command -v aws >/dev/null 2>&1 || return 1
    local root="${1#s3://}"
    local bucket="${root%%/*}"
    local prefix=""
    if [[ "$root" != "$bucket" ]]; then
        prefix="${root#*/}"
    fi
    prefix="${prefix%/}"
    [[ -n "$bucket" ]] || return 1

    local probe_key=".kumo-compose-local-access-check"
    if [[ -n "$prefix" ]]; then
        probe_key="$prefix/$probe_key"
    fi
    local region
    region="$(detect_aws_region)"

    aws s3api list-objects-v2 \
        --bucket "$bucket" \
        --prefix "${prefix:+${prefix}/}" \
        --max-items 1 \
        --region "$region" >/dev/null 2>&1 &&
    aws s3api put-object \
        --bucket "$bucket" \
        --key "$probe_key" \
        --body /dev/null \
        --region "$region" >/dev/null 2>&1 &&
    aws s3api delete-object \
        --bucket "$bucket" \
        --key "$probe_key" \
        --region "$region" >/dev/null 2>&1
}

check_gcs_root_access() {
    local root="${1#gs://}"
    local bucket="${root%%/*}"
    local prefix=""
    if [[ "$root" != "$bucket" ]]; then
        prefix="${root#*/}"
    fi
    prefix="${prefix%/}"
    [[ -n "$bucket" ]] || return 1

    local probe_key=".kumo-compose-local-access-check"
    if [[ -n "$prefix" ]]; then
        probe_key="$prefix/$probe_key"
    fi
    local probe_path="gs://$bucket/$probe_key"

    if command -v gsutil >/dev/null 2>&1; then
        gsutil -q ls -b "gs://$bucket" >/dev/null 2>&1 &&
        gsutil -q cp /dev/null "$probe_path" >/dev/null 2>&1 &&
        gsutil -q rm "$probe_path" >/dev/null 2>&1
    elif command -v gcloud >/dev/null 2>&1; then
        gcloud storage buckets describe "gs://$bucket" >/dev/null 2>&1 &&
        gcloud storage cp /dev/null "$probe_path" >/dev/null 2>&1 &&
        gcloud storage rm "$probe_path" >/dev/null 2>&1
    else
        return 1
    fi
}

check_adls_root_access() {
    command -v az >/dev/null 2>&1 || return 1
    local rest="${1#abfss://}"
    local container="${rest%@*}"
    local account_host="${rest#*@}"
    local account="${account_host%%.*}"
    local prefix=""
    if [[ "$account_host" == *".net/"* ]]; then
        prefix="${account_host#*.net/}"
    fi
    prefix="${prefix%/}"
    [[ -n "$container" && -n "$account" ]] || return 1

    local tmp_file probe_path
    tmp_file="$(mktemp)"
    probe_path=".kumo-compose-local-access-check"
    if [[ -n "$prefix" ]]; then
        probe_path="$prefix/$probe_path"
    fi
    : > "$tmp_file"

    az storage fs file upload \
        --account-name "$account" \
        --file-system "$container" \
        --path "$probe_path" \
        --source "$tmp_file" \
        --auth-mode login \
        --overwrite true >/dev/null 2>&1 &&
    az storage fs file delete \
        --account-name "$account" \
        --file-system "$container" \
        --path "$probe_path" \
        --auth-mode login \
        --yes >/dev/null 2>&1
    local status=$?
    rm -f "$tmp_file"
    return "$status"
}

cloud_root_is_writable() {
    case "$1" in
        s3://*) check_s3_root_access "$1" ;;
        gs://*) check_gcs_root_access "$1" ;;
        abfss://*) check_adls_root_access "$1" ;;
        *) return 1 ;;
    esac
}

resolve_auto_dataplane_root() {
    local provider="$1"
    local candidate
    candidate="$(cloud_default_dataplane_root "$provider")"
    if [[ -n "$candidate" ]] && cloud_root_is_writable "$candidate"; then
        echo "$candidate"
    else
        if [[ -n "$candidate" ]]; then
            echo "==> Cloud dataplane unavailable for $provider; using local fallback" >&2
        else
            echo "==> Cloud provider unknown; using local dataplane fallback" >&2
        fi
        echo "$DEFAULT_LOCAL_DATAPLANE_ROOT"
    fi
}

local_root_prefix() {
    local root="${1%/}"
    local marker="{instance_id}"
    if [[ "$root" == *"$marker"* ]]; then
        root="${root%%"$marker"*}"
    fi
    echo "${root%/}"
}

write_nginx_config() {
    local path="$1"
    local nginx_port="$2"
    local ui_port="$3"
    local rest_port="$4"
    local public_rest_port="$5"
    cat > "$path" << NGINXEOF
events {
    worker_connections 1024;
}

http {
    proxy_connect_timeout 75s;
    proxy_send_timeout    300s;
    proxy_read_timeout    600s;
    send_timeout          300s;
    client_max_body_size  128m;

    server {
        listen ${nginx_port};
        listen [::]:${nginx_port} ipv6only=on;

        location /v0/ {
            proxy_pass http://kumo-rest:${rest_port}/v0/;
            proxy_redirect  default;
            proxy_set_header Host              \$host;
            proxy_set_header X-Real-IP         \$remote_addr;
            proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location /v1/ {
            proxy_pass http://kumo-rest:${rest_port}/v1/;
            proxy_redirect  default;
            proxy_set_header Host              \$host;
            proxy_set_header X-Real-IP         \$remote_addr;
            proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location /api/ {
            proxy_pass http://kumo-rest-public:${public_rest_port}/;
            proxy_redirect ~^https?://[^/]+(/.*)$ /api\$1;
            proxy_set_header Host              \$host;
            proxy_set_header X-Real-IP         \$remote_addr;
            proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        location / {
            proxy_pass http://kumo-ui:${ui_port};
            proxy_set_header Host              \$host;
            proxy_set_header X-Real-IP         \$remote_addr;
            proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
NGINXEOF
}

write_prometheus_config() {
    local path="$1"
    cat > "$path" << PROMEOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: kumo-production-local

scrape_configs:
  - job_name: rest-api
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-rest:${REST_METRICS_PORT}"]

  - job_name: rest-api-public
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-rest-public:${PUBLIC_REST_METRICS_PORT}"]

  - job_name: worker-data
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-worker-data:${DATA_WORKER_METRICS_PORT}"]

  - job_name: worker-data-app
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-worker-data:${DATA_WORKER_APP_METRICS_PORT}"]

  - job_name: worker-mlworkflow
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-worker-mlworkflow:${MLWORKFLOW_WORKER_METRICS_PORT}"]

  - job_name: worker-mlworkflow-app
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-worker-mlworkflow:${MLWORKFLOW_WORKER_APP_METRICS_PORT}"]

  - job_name: pool-controller
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-pool-controller:${POOL_CONTROLLER_METRICS_PORT}"]

  - job_name: pool-worker
    scrape_interval: 10s
    static_configs:
      - targets: ["kumo-pool-worker:${POOL_WORKER_METRICS_PORT}"]

  - job_name: prometheus
    scrape_interval: 10s
    static_configs:
      - targets: ["localhost:${PROMETHEUS_PORT}"]
PROMEOF
}

load_compose_env_file() {
    local env_file="$1"
    local override="${2:-false}"
    local line key value
    if [[ ! -f "$env_file" ]]; then
        return
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        [[ -z "$key" || ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && continue
        caller_env_has "$key" && continue
        if [[ "$override" == "true" || -z "${!key+x}" ]]; then
            export "$key=$value"
        fi
    done < "$env_file"
}

resolve_repo_path() {
    case "$1" in
        /*) echo "$1" ;;
        *) echo "$REPO_ROOT/$1" ;;
    esac
}

for_each_extra_path() {
    local raw="$1"
    local callback="$2"
    local path
    local -a extra_paths
    [[ -n "$raw" ]] || return 0
    IFS=':' read -r -a extra_paths <<< "$raw"
    for path in "${extra_paths[@]}"; do
        [[ -n "$path" ]] || continue
        "$callback" "$(resolve_repo_path "$path")"
    done
}

load_extra_env_file() {
    load_compose_env_file "$1" true
}

append_extra_compose_file() {
    compose_files+=(-f "$1")
}

append_extra_compose_env_file() {
    compose_env_files+=(--env-file "$1")
}

load_compose_env_defaults() {
    load_compose_env_file "$COMPOSE_ENV_FILE" false
    load_compose_env_file "$COMPOSE_LOCAL_ENV_FILE" true
    for_each_extra_path "$COMPOSE_LOCAL_EXTRA_ENV_FILES" load_extra_env_file
}

check_prereqs() {
    local missing=()
    command -v docker &>/dev/null || missing+=("docker")
    if ! docker info &>/dev/null 2>&1; then
        local docker_info_output
        docker_info_output="$(docker info 2>&1 || true)"
        if [[ "$(uname -s)" == "Linux" ]] && printf '%s\n' "$docker_info_output" | grep -q "permission denied"; then
            echo "==> Adding $USER to docker group and re-launching..." >&2
            sudo usermod -aG docker "$USER"
            exec sg docker -c "$(printf '%q ' "$0" "$@")"
        fi
        echo "ERROR: docker is installed but not accessible. Start Docker Desktop and verify 'docker info' succeeds." >&2
        printf '%s\n' "$docker_info_output" >&2
        exit 1
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: Missing prerequisites:" >&2
        printf "  - %s\n" "${missing[@]}" >&2
        exit 1
    fi
}

setup_nvme() {
    local mount_point="/mnt/localssd"
    if mountpoint -q "$mount_point" 2>/dev/null; then
        local write_probe="$mount_point/.kumo-compose-write-probe.$$"
        if (umask 077; : > "$write_probe") 2>/dev/null; then
            rm -f "$write_probe"
            echo "==> NVMe already mounted at $mount_point" >&2
            echo "$mount_point"
        else
            echo "==> $mount_point is mounted read-only — using repo-local storage" >&2
            echo "$REPO_LOCAL_CACHE_ROOT/compose-local/diskgraph"
        fi
        return
    fi
    if ! ls /dev/nvme*n1 &>/dev/null 2>&1; then
        if mountpoint -q /mnt 2>/dev/null; then
            local azure_mount="/mnt/localssd"
            if sudo mkdir -p "$azure_mount" 2>/dev/null && sudo chmod 777 "$azure_mount" 2>/dev/null; then
                echo "==> Using /mnt for local diskgraph data" >&2
                echo "$azure_mount"
                return
            fi
        fi
        echo "$REPO_LOCAL_CACHE_ROOT/compose-local/diskgraph"
        return
    fi
    echo "==> NVMe device(s) detected — mounting to $mount_point" >&2
    sudo bash "$REPO_ROOT/deploy/shared/scripts/mount-nvme.sh" >&2
    if mountpoint -q "$mount_point" 2>/dev/null; then
        echo "$mount_point"
    else
        echo "$REPO_LOCAL_CACHE_ROOT/compose-local/diskgraph"
    fi
}

detect_gpu() {
    if [[ "${BUILD_GPU:-}" == "0" ]]; then
        echo "no"
        return
    fi
    if [[ "${BUILD_GPU:-}" == "1" ]]; then
        echo "yes"
        return
    fi
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null 2>&1; then
        echo "no"
        return
    fi
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        echo "yes"
    else
        echo "==> NVIDIA driver detected, but Docker NVIDIA runtime is unavailable — using CPU pool worker" >&2
        echo "no"
    fi
}

generate_deploy_config() {
    local dataplane="$1"
    local artifact_root="$2"
    local temporal_port="$3"
    local output_path="$4"
    local template="$REPO_ROOT/deploy/shared/configs/deploy_env_config.template.json"

    DATAPLANE_ROOT="$dataplane" \
    ARTIFACT_ROOT="$artifact_root" \
    DISKGRAPH_DATA_ROOT="/mnt/localssd" \
    METADATA_STORAGE_TYPE="${METADATA_STORAGE_TYPE:-postgres}" \
    STATE_DB_DSN_SECRET_NAME="${STATE_DB_DSN_SECRET_NAME:-STATE_DB_DSN}" \
    LIFECYCLE_READY="${LIFECYCLE_READY:-false}" \
    TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-prod}" \
    TEMPORAL_HOST="${TEMPORAL_HOST:-temporal}" \
    TEMPORAL_PORT="$temporal_port" \
    "$PYTHON_BIN" - "$template" "$output_path" <<'PY'
import os
import json
import string
import sys

template_path, output_path = sys.argv[1:3]
with open(template_path, encoding='utf-8') as template_file:
    rendered = string.Template(template_file.read()).safe_substitute(os.environ)
config = json.loads(rendered)
if os.environ.get("DISABLE_AUTH", "false").lower() != "true":
    login_method = os.environ.get(
        "DIRECT_CONNECT_LOGIN_METHOD",
        "oidc",
    ).lower()
    direct_connect = {
        "login_method": login_method,
        "allowed_return_origins": [
            origin
            for origin in [os.environ.get("PUBLIC_URL", "").rstrip("/")]
            if origin
        ],
        "admin_emails": [
            email.strip()
            for email in os.environ.get("ADMIN_EMAILS", "").split(",")
            if email.strip()
        ],
    }
    secret_name = "kumo-secret-key"
    if login_method == "password":
        direct_connect["password"] = {
            "temporary_password_ref": {
                "secret_name": secret_name,
                "secret_key": "local_admin_temp_password",
            },
        }
    else:
        direct_connect["oidc"] = {
            "issuer": os.environ.get("OIDC_ISSUER", "").rstrip("/"),
            "client_id": os.environ.get("OIDC_CLIENT_ID", ""),
            "groups_claim": os.environ.get("OIDC_GROUPS_CLAIM", "groups"),
            "client_secret_ref": {
                "secret_name": secret_name,
                "secret_key": "oidc_client_secret",
            },
        }
    config["auth"] = {
        "enabled": True,
        "backend": "direct_connect",
        "direct_connect": direct_connect,
    }
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(config, output_file, indent=2)
    output_file.write("\n")
PY
}

prepare_glove_dir() {
    local glove_dir="$1"
    local level="${COMPOSE_LOCAL_GLOVE_LEVEL:-basic}"

    mkdir -p "$glove_dir"
    if [[ "$level" == "none" ]]; then
        echo "==> GloVe dir      : $glove_dir (download skipped)" >&2
        return
    fi

    if [[ "$level" == "basic" \
        && -f "$glove_dir/glove_6B_index.parquet" \
        && -f "$glove_dir/glove_6B_50d_emb.npy" ]]; then
        echo "==> GloVe dir      : $glove_dir" >&2
        return
    fi

    echo "==> Preparing GloVe: $glove_dir ($level)" >&2
    GLOVE_DEST="$glove_dir" BAKE_GLOVE_LEVEL="$level" "$REPO_ROOT/docker/install_glove_files"
}

main() {
    check_prereqs
    local user_kumo_image_tag="${KUMO_IMAGE_TAG:-}"
    local user_diskgraph_data="${LOCAL_DISKGRAPH_DATA:-}"
    local user_diskgraph_root="${DISKGRAPH_DATA_ROOT:-}"
    local user_dataplane_host_mount="${LOCAL_DATAPLANE_HOST_MOUNT:-}"
    local user_dataplane_mount="${LOCAL_DATAPLANE_MOUNT:-}"
    local user_secrets_dir="${KUMO_SECRETS_DIR:-}"
    load_compose_env_defaults
    local user_glove_dir="${KUMO_GLOVE_DIR:-}"

    local nvme_root backend_variant postgres_mode diskgraph_root dataplane dataplane_host_mount_root dataplane_container_mount_root artifact_root compose_files compose_env_files
    local cloud_provider auto_dataplane_root_used
    local stack port_offset host_runtime_root runtime_root generated_config_dir deploy_config_path nginx_config_path prometheus_config_path

    nvme_root=$(setup_nvme)
    if [[ -n "${KUMO_BACKEND_VARIANT:-}" ]]; then
        backend_variant="$KUMO_BACKEND_VARIANT"
    elif [[ "$(detect_gpu)" == "yes" ]]; then
        backend_variant="gpu"
    else
        backend_variant="cpu"
    fi
    postgres_mode="$("$REPO_ROOT/scripts/lib/resolve-postgres-mode.sh" "$COMPOSE_ENV_FILE" "$COMPOSE_LOCAL_ENV_FILE" $COMPOSE_LOCAL_EXTRA_ENV_FILES)"
    case "$backend_variant" in
        cpu)
            compose_files=(
                -f "$COMPOSE_DIR/docker-compose.singlehost.yml"
                -f "$COMPOSE_DIR/docker-compose.production-local.override.yml"
            )
            ;;
        gpu)
            if [[ "$(detect_gpu)" != "yes" ]]; then
                echo "ERROR: KUMO_BACKEND_VARIANT=gpu requires a visible GPU and Docker NVIDIA runtime" >&2
                exit 1
            fi
            # Default: one GPU worker slot per GPU.
            if [[ -z "${ML_MAX_GPU_WORKER_SLOTS:-}" ]]; then
                ML_MAX_GPU_WORKER_SLOTS="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
                [[ "$ML_MAX_GPU_WORKER_SLOTS" -ge 1 ]] || ML_MAX_GPU_WORKER_SLOTS=1
                export ML_MAX_GPU_WORKER_SLOTS
            fi
            compose_files=(
                -f "$COMPOSE_DIR/docker-compose.singlehost.yml"
                -f "$COMPOSE_DIR/docker-compose.singlehost.gpu.yml"
                -f "$COMPOSE_DIR/docker-compose.production-local.override.yml"
            )
            ;;
        *)
            echo "ERROR: KUMO_BACKEND_VARIANT must be cpu or gpu, got '$backend_variant'" >&2
            exit 1
            ;;
    esac
    if [[ "$postgres_mode" != "external" ]]; then
        compose_files=(
            "${compose_files[@]:0:2}"
            -f "$COMPOSE_DIR/docker-compose.singlehost.postgres.yml"
            "${compose_files[@]:2}"
        )
    fi
    for_each_extra_path "$COMPOSE_LOCAL_EXTRA_COMPOSE_FILES" \
        append_extra_compose_file

    stack="${STACK:-${COMPOSE_PROJECT_NAME:-kumo}}"
    [[ "$stack" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        echo "ERROR: STACK must contain only letters, numbers, '.', '_' or '-'" >&2
        exit 1
    }
    export COMPOSE_PROJECT_NAME="$stack"
    export COMPOSE_CONTAINER_PREFIX="${COMPOSE_CONTAINER_PREFIX:-$stack}"
    if [[ -z "${PORT_OFFSET+x}" ]]; then
        port_offset="$(default_port_offset "$stack")"
    else
        port_offset="$PORT_OFFSET"
    fi
    [[ "$port_offset" =~ ^[0-9]+$ ]] || {
        echo "ERROR: PORT_OFFSET must be a non-negative integer" >&2
        exit 1
    }

    # Internal ports live in each stack's private docker bridge network, so
    # they don't need per-stack offsets. Only NGINX_PORT is published to the
    # host and must differ between concurrent stacks.
    export REST_API_PORT="${REST_API_PORT:-8000}"
    export PUBLIC_REST_API_PORT="${PUBLIC_REST_API_PORT:-8001}"
    export REST_WORKER_METRICS_PORT="${REST_WORKER_METRICS_PORT:-8002}"
    export DATA_WORKER_METRICS_PORT="${DATA_WORKER_METRICS_PORT:-8003}"
    export MLWORKFLOW_WORKER_METRICS_PORT="${MLWORKFLOW_WORKER_METRICS_PORT:-8005}"
    export PUBLIC_REST_WORKER_METRICS_PORT="${PUBLIC_REST_WORKER_METRICS_PORT:-8006}"
    export REST_METRICS_PORT="${REST_METRICS_PORT:-8007}"
    export PUBLIC_REST_METRICS_PORT="${PUBLIC_REST_METRICS_PORT:-8008}"
    export POOL_CONTROLLER_METRICS_PORT="${POOL_CONTROLLER_METRICS_PORT:-8009}"
    export POOL_WORKER_METRICS_PORT="${POOL_WORKER_METRICS_PORT:-8010}"
    export DATA_WORKER_APP_METRICS_PORT="${DATA_WORKER_APP_METRICS_PORT:-8103}"
    export MLWORKFLOW_WORKER_APP_METRICS_PORT="${MLWORKFLOW_WORKER_APP_METRICS_PORT:-8105}"
    export TEMPORAL_PORT="${TEMPORAL_PORT:-7233}"
    export UI_PORT="${UI_PORT:-3000}"
    export NGINX_PORT="${NGINX_PORT:-$((8888 + port_offset))}"
    export PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
    export JAEGER_ADMIN_PORT="${JAEGER_ADMIN_PORT:-14269}"
    export JAEGER_COLLECTOR_GRPC_PORT="${JAEGER_COLLECTOR_GRPC_PORT:-14250}"
    export JAEGER_COLLECTOR_HTTP_PORT="${JAEGER_COLLECTOR_HTTP_PORT:-14268}"
    export JAEGER_OTLP_GRPC_PORT="${JAEGER_OTLP_GRPC_PORT:-4317}"
    export JAEGER_OTLP_HTTP_PORT="${JAEGER_OTLP_HTTP_PORT:-4318}"
    export JAEGER_QUERY_GRPC_PORT="${JAEGER_QUERY_GRPC_PORT:-16685}"
    export JAEGER_QUERY_PORT="${JAEGER_QUERY_PORT:-16686}"
    export JAEGER_ZIPKIN_PORT="${JAEGER_ZIPKIN_PORT:-9411}"
    export DEPLOY_STAGE="${DEPLOY_STAGE:-prod}"
    export TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-prod}"

    if [[ "$stack" != "kumo" && -z "$user_kumo_image_tag" && "${KUMO_IMAGE_TAG:-local}" == "local" ]]; then
        export KUMO_IMAGE_TAG="$stack"
    fi

    if [[ -n "$user_diskgraph_data" ]]; then
        diskgraph_root="$LOCAL_DISKGRAPH_DATA"
    elif [[ -n "$user_diskgraph_root" ]]; then
        diskgraph_root="$DISKGRAPH_DATA_ROOT"
    else
        diskgraph_root="$nvme_root"
    fi
    diskgraph_root="$(stack_scoped_local_subdir "$diskgraph_root" "$stack" diskgraph)"
    host_runtime_root="$(stack_scoped_local_subdir "$nvme_root" "$stack" runtime)"
    if ! mkdir -p "$host_runtime_root"/{secrets,tls,dataplane} 2>/dev/null; then
        local fallback_root="$REPO_LOCAL_CACHE_ROOT/compose-local/diskgraph"
        echo "==> $host_runtime_root is not writable — using repo-local storage" >&2
        if [[ -z "$user_diskgraph_data" && -z "$user_diskgraph_root" ]]; then
            diskgraph_root="$(stack_scoped_local_subdir "$fallback_root" "$stack" diskgraph)"
        fi
        host_runtime_root="$(stack_scoped_local_subdir "$fallback_root" "$stack" runtime)"
    fi
    runtime_root="$REPO_LOCAL_CACHE_ROOT/compose-local/runtime-$stack"

    cloud_provider="$(detect_cloud_provider)"
    auto_dataplane_root_used="false"
    dataplane="${PRODUCTION_LOCAL_DATAPLANE_ROOT:-$DEFAULT_PRODUCTION_LOCAL_DATAPLANE_ROOT}"
    if [[ "$dataplane" == "auto" ]]; then
        dataplane="$(resolve_auto_dataplane_root "$cloud_provider")"
        auto_dataplane_root_used="true"
    fi
    dataplane="$(stack_scoped_root "$dataplane" "$stack")"
    dataplane_host_mount_root="$host_runtime_root/dataplane"
    dataplane_container_mount_root="$runtime_root/dataplane"
    if ! is_remote_root "$dataplane"; then
        if [[ -n "$user_dataplane_host_mount" || -n "$user_dataplane_mount" ]]; then
            dataplane_host_mount_root="${user_dataplane_host_mount:-$(local_root_prefix "$dataplane")}"
            dataplane_container_mount_root="${user_dataplane_mount:-$dataplane_host_mount_root}"
        else
            dataplane_host_mount_root="$(local_root_prefix "$dataplane")"
            dataplane_container_mount_root="$dataplane_host_mount_root"
        fi
    fi

    artifact_root="${PRODUCTION_LOCAL_ARTIFACT_ROOT:-$DEFAULT_PRODUCTION_LOCAL_ARTIFACT_ROOT}"
    if [[ "$artifact_root" == "auto" ]]; then
        if [[ "$auto_dataplane_root_used" == "true" ]] && ! is_remote_root "$dataplane"; then
            artifact_root="$DEFAULT_LOCAL_ARTIFACT_ROOT"
        else
            artifact_root="$(cloud_default_artifact_root "$cloud_provider")"
        fi
    fi
    if ! is_remote_root "$artifact_root"; then
        artifact_root="$(stack_scoped_root "$artifact_root" "$stack")"
    fi
    generated_config_dir="$CONFIG_DIR/generated/$stack"
    mkdir -p "$generated_config_dir"
    deploy_config_path="$generated_config_dir/deploy_env_config.production-local.json"
    nginx_config_path="$generated_config_dir/nginx.production-local.conf"
    prometheus_config_path="$generated_config_dir/prometheus.production-local.yml"

    mkdir -p "$diskgraph_root"
    mkdir -p "$host_runtime_root"/{secrets,tls,dataplane}
    export METADATA_STORAGE_TYPE="${METADATA_STORAGE_TYPE:-postgres}"
    export STATE_DB_DSN_SECRET_NAME="${STATE_DB_DSN_SECRET_NAME:-STATE_DB_DSN}"
    export KUMO_BOOTSTRAP_OWNER_UID="${KUMO_BOOTSTRAP_OWNER_UID:-999}"
    export KUMO_BOOTSTRAP_OWNER_GID="${KUMO_BOOTSTRAP_OWNER_GID:-999}"
    [[ -n "$user_glove_dir" ]] || export KUMO_GLOVE_DIR="$host_runtime_root/glove"
    prepare_glove_dir "$KUMO_GLOVE_DIR"

    generate_deploy_config "$dataplane" "$artifact_root" "$TEMPORAL_PORT" "$deploy_config_path"
    write_nginx_config "$nginx_config_path" "$NGINX_PORT" "$UI_PORT" "$REST_API_PORT" "$PUBLIC_REST_API_PORT"
    write_prometheus_config "$prometheus_config_path"
    chmod 644 "$deploy_config_path" "$nginx_config_path" "$prometheus_config_path"

    mkdir -p "$dataplane_host_mount_root"
    [[ -n "$user_secrets_dir" ]] || reclaim_default_secrets_dir "$host_runtime_root/secrets"
    chmod 755 "$host_runtime_root" "$host_runtime_root/tls" 2>/dev/null || true
    chmod 750 "$host_runtime_root/secrets"
    find "$host_runtime_root/secrets" -type f \
        -exec chmod 640 {} + 2>/dev/null || true
    chmod 777 "$dataplane_host_mount_root" 2>/dev/null || true
    chmod 777 "$diskgraph_root" 2>/dev/null || true

    [[ -n "$user_dataplane_host_mount" ]] || export LOCAL_DATAPLANE_HOST_MOUNT="$dataplane_host_mount_root"
    [[ -n "$user_dataplane_mount" ]] || export LOCAL_DATAPLANE_MOUNT="$dataplane_container_mount_root"
    [[ -n "$user_secrets_dir" ]] || export KUMO_SECRETS_DIR="$host_runtime_root/secrets"

    export DEPLOY_ENV_CONFIG_PATH="$deploy_config_path"
    # File-bind the generated configs at the neutral bootstrap mount targets so
    # the base compose (backend deploy config, nginx -c, prometheus
    # --config.file) and the kumo-bootstrap no-op agree on the paths.
    export DEPLOY_ENV_CONFIG_MOUNT=/etc/kumo/config/deploy_env_config.json
    export NGINX_CONFIG_MOUNT=/etc/kumo/nginx/nginx.conf
    export PROMETHEUS_CONFIG_MOUNT=/etc/kumo/prometheus/prometheus.yml
    export KUMO_POSTGRES_MODE="$postgres_mode"
    export LOCAL_DATAPLANE_HOST_MOUNT="${LOCAL_DATAPLANE_HOST_MOUNT:-$host_runtime_root/dataplane}"
    export LOCAL_DATAPLANE_MOUNT="${LOCAL_DATAPLANE_MOUNT:-$runtime_root/dataplane}"
    export KUMO_SECRETS_DIR="${KUMO_SECRETS_DIR:-$host_runtime_root/secrets}"
    if [[ -z "$user_secrets_dir" && ! -w "$KUMO_SECRETS_DIR" ]]; then
        sudo chown -R "$(id -u):$(id -g)" "$KUMO_SECRETS_DIR"
    fi
    if [[ ! -w "$KUMO_SECRETS_DIR" ]]; then
        echo "ERROR: compose-local secrets directory is not writable: $KUMO_SECRETS_DIR" >&2
        return 1
    fi
    export KUMO_STATE_DB_NAME="${KUMO_STATE_DB_NAME:-kumo}"
    export TEMPORAL_DB_NAME="${TEMPORAL_DB_NAME:-temporal}"
    export TEMPORAL_VISIBILITY_DB_NAME="${TEMPORAL_VISIBILITY_DB_NAME:-temporal_visibility}"
    if [[ -z "${POSTGRES_PASSWORD:-}" && -n "${POSTGRES_PASSWORD_FILE:-}" ]]; then
        export POSTGRES_PASSWORD="$(cat "$POSTGRES_PASSWORD_FILE")"
    fi
    if ! secrets_dir_writable "$KUMO_SECRETS_DIR"; then
        die_secrets_not_writable "$KUMO_SECRETS_DIR"
    fi
    if [[ "$postgres_mode" == "external" ]]; then
        STATE_DB_DSN_FILE="$KUMO_SECRETS_DIR/STATE_DB_DSN" \
        TEMPORAL_DB_DSN_FILE="$KUMO_SECRETS_DIR/TEMPORAL_DB_DSN" \
        TEMPORAL_VISIBILITY_DB_DSN_FILE="$KUMO_SECRETS_DIR/TEMPORAL_VISIBILITY_DB_DSN" \
        "$PYTHON_BIN" "$REPO_ROOT/deploy/shared/scripts/write-postgres-secrets.py"
        KUMO_POSTGRES_SECRETS_DIR="$KUMO_SECRETS_DIR" \
        "$REPO_ROOT/deploy/shared/scripts/check-postgres-dsns.sh" \
            "$KUMO_SECRETS_DIR/STATE_DB_DSN" \
            "$KUMO_SECRETS_DIR/TEMPORAL_DB_DSN" \
            "$KUMO_SECRETS_DIR/TEMPORAL_VISIBILITY_DB_DSN"
    else
        # Bundled Postgres credentials for compose-local.
        export POSTGRES_USER="${POSTGRES_USER:-kumo}"
        export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-kumo}"
        export POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
        export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
        STATE_DB_DSN_FILE="$KUMO_SECRETS_DIR/STATE_DB_DSN" \
        TEMPORAL_DB_DSN_FILE="$KUMO_SECRETS_DIR/TEMPORAL_DB_DSN" \
        TEMPORAL_VISIBILITY_DB_DSN_FILE="$KUMO_SECRETS_DIR/TEMPORAL_VISIBILITY_DB_DSN" \
        "$PYTHON_BIN" "$REPO_ROOT/deploy/shared/scripts/write-postgres-secrets.py"
    fi
    export_temporal_postgres_env
    for postgres_secret in STATE_DB_DSN TEMPORAL_DB_DSN TEMPORAL_VISIBILITY_DB_DSN; do
        [[ -f "$KUMO_SECRETS_DIR/$postgres_secret" ]] || continue
        chmod 640 "$KUMO_SECRETS_DIR/$postgres_secret"
        if [[ -z "$user_secrets_dir" ]]; then
            sudo -n chown \
                "$KUMO_BOOTSTRAP_OWNER_UID:$KUMO_BOOTSTRAP_OWNER_GID" \
                "$KUMO_SECRETS_DIR/$postgres_secret"
        fi
    done
    export KUMO_TLS_DIR="${KUMO_TLS_DIR:-$host_runtime_root/tls}"
    export NGINX_CONFIG="$nginx_config_path"
    export PROMETHEUS_CONFIG="$prometheus_config_path"
    export KUMO_BACKEND_VARIANT="$backend_variant"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        export KUMO_WORKER_HEALTHCHECK_TIMEOUT="${KUMO_WORKER_HEALTHCHECK_TIMEOUT:-30s}"
    fi
    export KUMO_CPU_BACKEND_IMAGE="${KUMO_CPU_BACKEND_IMAGE:-kumo/kumo-backend-cpu:${KUMO_IMAGE_TAG:-local}}"
    if [[ "$backend_variant" == "gpu" ]]; then
        export KUMO_POOL_WORKER_IMAGE="${KUMO_POOL_WORKER_IMAGE:-kumo/kumo-backend-gpu:${KUMO_IMAGE_TAG:-local}}"
    else
        export KUMO_POOL_WORKER_IMAGE="${KUMO_POOL_WORKER_IMAGE:-$KUMO_CPU_BACKEND_IMAGE}"
    fi

    compose_env_files=(--env-file "$COMPOSE_ENV_FILE")
    if [[ -f "$COMPOSE_LOCAL_ENV_FILE" ]]; then
        compose_env_files+=(--env-file "$COMPOSE_LOCAL_ENV_FILE")
    fi
    for_each_extra_path "$COMPOSE_LOCAL_EXTRA_ENV_FILES" \
        append_extra_compose_env_file

    echo "==> Stack          : $stack" >&2
    echo "==> Image tag      : ${KUMO_IMAGE_TAG:-local}" >&2
    echo "==> Port offset    : $port_offset" >&2
    echo "==> Diskgraph data : $diskgraph_root" >&2
    echo "==> Dataplane      : $dataplane" >&2
    echo "==> Artifact root  : $artifact_root" >&2
    echo "==> Metadata DB    : postgres ($postgres_mode)" >&2
    echo "==> CPU backend    : $KUMO_CPU_BACKEND_IMAGE" >&2
    echo "==> Pool worker    : $KUMO_POOL_WORKER_IMAGE" >&2
    echo "==> GPU slots      : ${ML_MAX_GPU_WORKER_SLOTS:-1}" >&2

    # Export so the env snapshot below captures it for callers running
    # in KUMO_PREFLIGHT_NO_UP mode. Used inline in the default up path too.
    export DISKGRAPH_DATA_ROOT="$diskgraph_root"

    # compose_files looks like (-f a.yml -f b.yml -f c.yml); pull just the
    # file paths so KUMO_PREFLIGHT_NO_UP callers know which set we resolved
    # (CPU vs GPU) without mirroring detect_gpu themselves.
    local resolved_files=()
    local i
    for (( i=1; i<${#compose_files[@]}; i+=2 )); do
        resolved_files+=("${compose_files[i]}")
    done
    # Join with ':' in a subshell to avoid mutating IFS globally.
    KUMO_PREFLIGHT_RESOLVED_COMPOSE_FILES="$(IFS=:; printf '%s' "${resolved_files[*]}")"
    export KUMO_PREFLIGHT_RESOLVED_COMPOSE_FILES

    if [[ "${KUMO_PREFLIGHT_NO_UP:-0}" == "1" ]]; then
        # Caller (kumo-tests compose_stack fixture) issues ``up`` via
        # testcontainers. Snapshot the resolved env so it can hydrate
        # os.environ before doing so — preflight's exports die with us.
        # Caller may pin the snapshot path via KUMO_PREFLIGHT_ENV_OUT
        # (e.g. a pytest tmp dir).
        local env_snapshot="${KUMO_PREFLIGHT_ENV_OUT:-$generated_config_dir/preflight.env}"
        mkdir -p "$(dirname "$env_snapshot")"
        env > "$env_snapshot"
        echo "==> KUMO_PREFLIGHT_NO_UP=1 — config ready; env at $env_snapshot" >&2
        return 0
    fi

    write_stack_manifest "$stack" "$diskgraph_root" "$host_runtime_root" "$dataplane" "$generated_config_dir"

    resolve_docker_compose

    "${DOCKER_COMPOSE_CMD[@]}" \
        -p "$stack" \
        "${compose_files[@]}" \
        "${compose_env_files[@]}" \
        up -d

    # Nginx resolves compose service names at startup. Refresh it after `up`
    # because backend containers may have been recreated with new bridge IPs.
    "${DOCKER_COMPOSE_CMD[@]}" \
        -p "$stack" \
        "${compose_files[@]}" \
        "${compose_env_files[@]}" \
        restart nginx
}

write_stack_manifest() {
    local stack="$1" diskgraph_root="$2" host_runtime_root="$3" dataplane="$4" generated_config_dir="$5"
    local manifest_dir="${KUMO_STACK_MANIFEST_DIR:-$REPO_ROOT/../kumo-tests/stacks}"
    mkdir -p "$manifest_dir"
    python3 - <<EOF
import json, os
nginx_port = int(os.environ["NGINX_PORT"])
manifest = {
    "stack": "$stack",
    "instance_id": "$stack",
    "api_url": f"http://localhost:{nginx_port}/api",
    "ui_url": f"http://localhost:{nginx_port}",
    "repo_root": "$REPO_ROOT",
    "diskgraph_data": "$diskgraph_root",
    "runtime_root": "$host_runtime_root",
    "dataplane": "$dataplane",
    "generated_config_dir": "$generated_config_dir",
    "ports": {
        "public_rest": nginx_port,
    },
}
path = os.path.join("$manifest_dir", "$stack.json")
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"==> Stack manifest : {path}", flush=True)
EOF
}

if [[ "${COMPOSE_LOCAL_PREFLIGHT_SOURCE_ONLY:-0}" != "1" ]]; then
    main "$@"
fi
