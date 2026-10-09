#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kumo single-host first-boot script. Runs once via cloud-init per-once.
# Reads deployment config from AWS, GCP, or Azure instance metadata.
set -euo pipefail

echo "=== Kumo first-boot script starting ==="

PREFLIGHT_HELPERS=${KUMO_PREFLIGHT_HELPERS:-/opt/kumo/scripts/cloud-storage-preflight.sh}
if [ ! -f "$PREFLIGHT_HELPERS" ]; then
    PREFLIGHT_HELPERS="$(dirname "$0")/cloud-storage-preflight.sh"
fi
# shellcheck source=deploy/shared/scripts/cloud-storage-preflight.sh disable=SC1091
source "$PREFLIGHT_HELPERS"

CLOUD_PROVIDER=$(detect_cloud_provider)
echo "Detected cloud: $CLOUD_PROVIDER"

# Read deployment config from instance metadata
read_config() {
    local cloud="$CLOUD_PROVIDER"

    local userdata=""
    userdata=$(read_cloud_config "$cloud")

    # Parse key: value pairs from user-data. Supported keys:
    #   dataplane_root   - Dataplane root: S3, GCS, ADLS, or local
    #   diskgraph_data_root - NVMe/SSD mount for diskgraph index (default: /mnt/localssd)
    #   enable_tls       - Enable TLS with self-signed cert (default: false)
    #   public_url       - Browser-facing Kumo origin, e.g. https://kumo.example.com
    #   allow_local_public_url - Allow localhost public_url for auth (default: false)
    #   auth_backend     - Legacy selector; only direct_connect is supported
    #   disable_auth     - Disable auth for testing only (default: false)
    #   oidc_issuer      - Direct Connect OIDC issuer URL
    #   oidc_client_id   - Direct Connect OIDC client id
    #   oidc_client_secret - Direct Connect OIDC client secret
    #   oidc_client_secret_uri - Azure Key Vault secret ID (preferred on Azure)
    #   oidc_groups_claim - Direct Connect OIDC groups claim (default: groups)
    #   direct_connect_login_method - Direct Connect login mode: oidc or password (default: oidc)
    #   admin_emails     - Comma-separated Direct Connect bootstrap admin emails
    #   direct_connect_admin_temp_password - Temporary first-login password for Direct Connect password mode
    #   direct_connect_admin_temp_password_uri - Azure Key Vault secret ID
    #   deploy_stage     - Deploy stage passed to DEPLOY_STAGE env var (default: prod).
    #                      Set to 'preprod' to disable auth for testing.
    #   temporal_namespace - Temporal namespace for the single-host stack (default: prod)
    #   artifact_root    - Shared prebuilt artifact root for non-GloVe assets (default: cloud-specific public dataset root).
    #   lifecycle_ready  - Temporary-prefix expiry policy is provisioned (default: false).
    #   azure_cloud      - public, usgovernment, or china
    #   azure_client_id  - user-assigned managed identity client ID
    #                      Singlehost GloVe is always served from the AMI-local /opt/kumo/glove mount.
    #   postgres_mode    - "bundled" to run the local Postgres container, or "external" for BYO Postgres.
    #   postgres_*       - External Postgres inputs used to write structured secrets.
    #   state_db_name    - Optional StateDB database name for structured external inputs.
    #   temporal_db_name - Optional Temporal database name.
    #   temporal_visibility_db_name - Optional Temporal visibility database name.
    #   temporal_server_image - Optional mirrored Temporal server image.
    #   temporal_admin_tools_image - Optional mirrored Temporal admin-tools image.
    if [ -n "$userdata" ]; then
        DATAPLANE_ROOT=$(echo "$userdata" | grep -oP 'dataplane_root:\s*\K\S+' || echo "")
        DISKGRAPH_DATA_ROOT=$(echo "$userdata" | grep -oP 'diskgraph_data_root:\s*\K\S+' || echo "")
        ENABLE_TLS=$(echo "$userdata" | grep -oP 'enable_tls:\s*\K\S+' || echo "")
        PUBLIC_URL=$(echo "$userdata" | grep -oP 'public_url:\s*\K\S+' || echo "")
        ALLOW_LOCAL_PUBLIC_URL=$(echo "$userdata" | grep -oP 'allow_local_public_url:\s*\K\S+' || echo "")
        AUTH_BACKEND=$(echo "$userdata" | grep -oP 'auth_backend:\s*\K\S+' || echo "")
        DISABLE_AUTH=$(echo "$userdata" | grep -oP 'disable_auth:\s*\K\S+' || echo "")
        OIDC_ISSUER=$(echo "$userdata" | grep -oP 'oidc_issuer:\s*\K\S+' || echo "")
        OIDC_CLIENT_ID=$(echo "$userdata" | grep -oP 'oidc_client_id:\s*\K\S+' || echo "")
        OIDC_CLIENT_SECRET=$(echo "$userdata" | grep -oP 'oidc_client_secret:\s*\K\S+' || echo "")
        OIDC_CLIENT_SECRET_URI=$(echo "$userdata" | grep -oP 'oidc_client_secret_uri:\s*\K\S+' || echo "")
        OIDC_GROUPS_CLAIM=$(echo "$userdata" | grep -oP 'oidc_groups_claim:\s*\K\S+' || echo "")
        DIRECT_CONNECT_LOGIN_METHOD=$(echo "$userdata" | grep -oP 'direct_connect_login_method:\s*\K\S+' || echo "")
        ADMIN_EMAILS=$(echo "$userdata" | grep -oP 'admin_emails:\s*\K\S+' || echo "")
        DIRECT_CONNECT_ADMIN_TEMP_PASSWORD=$(echo "$userdata" | grep -oP 'direct_connect_admin_temp_password:\s*\K\S+' || echo "")
        DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI=$(echo "$userdata" | grep -oP 'direct_connect_admin_temp_password_uri:\s*\K\S+' || echo "")
        DEPLOY_STAGE=$(echo "$userdata" | grep -oP 'deploy_stage:\s*\K\S+' || echo "")
        TEMPORAL_NAMESPACE=$(echo "$userdata" | grep -oP 'temporal_namespace:\s*\K\S+' || echo "")
        ARTIFACT_ROOT=$(echo "$userdata" | grep -oP 'artifact_root:\s*\K\S+' || echo "")
        LIFECYCLE_READY=$(echo "$userdata" | grep -oP 'lifecycle_ready:\s*\K\S+' || echo "")
        AZURE_CLOUD_SETTING=$(echo "$userdata" | grep -oP 'azure_cloud:\s*\K\S+' || echo "")
        AZURE_CLIENT_ID=$(echo "$userdata" | grep -oP 'azure_client_id:\s*\K\S+' || echo "")
        KUMO_POSTGRES_MODE=$(echo "$userdata" | grep -oP 'postgres_mode:\s*\K\S+' || echo "")
        POSTGRES_HOST=$(echo "$userdata" | grep -oP 'postgres_host:\s*\K\S+' || echo "")
        POSTGRES_PORT=$(echo "$userdata" | grep -oP 'postgres_port:\s*\K\S+' || echo "")
        POSTGRES_USER=$(echo "$userdata" | grep -oP 'postgres_user:\s*\K\S+' || echo "")
        POSTGRES_PASSWORD=$(echo "$userdata" | grep -oP 'postgres_password:\s*\K\S+' || echo "")
        POSTGRES_PASSWORD_FILE=$(echo "$userdata" | grep -oP 'postgres_password_file:\s*\K\S+' || echo "")
        POSTGRES_SSLMODE=$(echo "$userdata" | grep -oP 'postgres_sslmode:\s*\K\S+' || echo "")
        POSTGRES_SSLROOTCERT=$(echo "$userdata" | grep -oP 'postgres_sslrootcert:\s*\K\S+' || echo "")
        KUMO_STATE_DB_NAME=$(echo "$userdata" | grep -oP 'state_db_name:\s*\K\S+' || echo "")
        TEMPORAL_DB_NAME=$(echo "$userdata" | grep -oP 'temporal_db_name:\s*\K\S+' || echo "")
        TEMPORAL_VISIBILITY_DB_NAME=$(echo "$userdata" | grep -oP 'temporal_visibility_db_name:\s*\K\S+' || echo "")
        TEMPORAL_SERVER_IMAGE=$(echo "$userdata" | grep -oP 'temporal_server_image:\s*\K\S+' || echo "")
        TEMPORAL_ADMIN_TOOLS_IMAGE=$(echo "$userdata" | grep -oP 'temporal_admin_tools_image:\s*\K\S+' || echo "")
    fi

    DATAPLANE_ROOT=${DATAPLANE_ROOT:-/data/kumo/dataplane}
    DATAPLANE_ROOT="${DATAPLANE_ROOT%/}"  # strip trailing slash
    DISKGRAPH_DATA_ROOT=${DISKGRAPH_DATA_ROOT:-/mnt/localssd}
    DISKGRAPH_DATA_ROOT="${DISKGRAPH_DATA_ROOT%/}"

    # Validate dataplane path
    case "$DATAPLANE_ROOT" in
        s3://*) ;;  # S3 paths are fine
        gs://*) ;;  # GCS paths are fine
        abfs://*|abfss://*)
            local canonical_dataplane_root
            if ! canonical_dataplane_root=$(canonicalize_adls_root "$DATAPLANE_ROOT"); then
                echo "ERROR: Invalid or unsupported ADLS dataplane root: $DATAPLANE_ROOT"
                exit 1
            fi
            # `abfs://` is accepted only as deployment input. Runtime storage
            # coordinates are always canonical encrypted `abfss://` URIs.
            DATAPLANE_ROOT="$canonical_dataplane_root"
            ;;
        /tmp|/tmp/*)
            echo "ERROR: dataplane_root=$DATAPLANE_ROOT is on /tmp which gets cleaned. Use a persistent path like /data/kumo/dataplane"
            exit 1 ;;
        /mnt/localssd|/mnt/localssd/*)
            echo "ERROR: dataplane_root=$DATAPLANE_ROOT is on NVMe which is wiped on stop/start. Use a persistent disk path like /data/kumo/dataplane"
            exit 1 ;;
        /*)
            ;;  # absolute local path, OK
        *)
            echo "ERROR: dataplane_root=$DATAPLANE_ROOT must be absolute or use s3://, gs://, abfs://, or abfss://"
            exit 1 ;;
    esac
    ENABLE_TLS=${ENABLE_TLS:-false}
    AUTH_BACKEND=${AUTH_BACKEND:-direct_connect}
    AUTH_BACKEND=$(echo "$AUTH_BACKEND" | tr '[:upper:]' '[:lower:]')
    if [ "$AUTH_BACKEND" != "direct_connect" ]; then
        echo "ERROR: Direct Connect is the only supported auth backend. Got: $AUTH_BACKEND"
        exit 1
    fi
    DISABLE_AUTH=${DISABLE_AUTH:-false}
    DISABLE_AUTH=$(echo "$DISABLE_AUTH" | tr '[:upper:]' '[:lower:]')
    OIDC_ISSUER=${OIDC_ISSUER:-}
    OIDC_CLIENT_ID=${OIDC_CLIENT_ID:-}
    OIDC_CLIENT_SECRET=${OIDC_CLIENT_SECRET:-}
    OIDC_CLIENT_SECRET_URI=${OIDC_CLIENT_SECRET_URI:-}
    OIDC_GROUPS_CLAIM=${OIDC_GROUPS_CLAIM:-groups}
    DIRECT_CONNECT_LOGIN_METHOD=${DIRECT_CONNECT_LOGIN_METHOD:-oidc}
    DIRECT_CONNECT_LOGIN_METHOD=$(echo "$DIRECT_CONNECT_LOGIN_METHOD" | tr '[:upper:]' '[:lower:]')
    DIRECT_CONNECT_ADMIN_TEMP_PASSWORD=${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD:-}
    DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI=${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI:-}
    ADMIN_EMAILS=${ADMIN_EMAILS:-}
    TEMPORAL_NAMESPACE=${TEMPORAL_NAMESPACE:-prod}
    if [ -z "${ARTIFACT_ROOT:-}" ]; then
        case "$CLOUD_PROVIDER" in
            gcp) ARTIFACT_ROOT="gs://kumo-public-datasets-bigquery" ;;
            azure)
                if parse_adls_root "$DATAPLANE_ROOT"; then
                    ARTIFACT_ROOT="abfss://artifacts@${ADLS_ACCOUNT}.${ADLS_ENDPOINT_SUFFIX}/kumo"
                else
                    echo "ERROR: Azure deployments require an explicit artifact_root when dataplane_root is not ADLS"
                    exit 1
                fi
                ;;
            *) ARTIFACT_ROOT="s3://kumo-public-datasets" ;;
        esac
    fi
    LIFECYCLE_READY=$(printf '%s' "${LIFECYCLE_READY:-false}" | tr '[:upper:]' '[:lower:]')
    AZURE_CLOUD_SETTING=${AZURE_CLOUD_SETTING:-}
    AZURE_CLIENT_ID=${AZURE_CLIENT_ID:-}
    case "$LIFECYCLE_READY" in
        true|false) ;;
        *) echo "ERROR: lifecycle_ready must be true or false"; exit 1 ;;
    esac
    ALLOW_LOCAL_PUBLIC_URL=${ALLOW_LOCAL_PUBLIC_URL:-false}
    PUBLIC_URL=${PUBLIC_URL:-}
    KUMO_POSTGRES_MODE=$(printf '%s' "${KUMO_POSTGRES_MODE:-bundled}" | tr '[:upper:]' '[:lower:]')
    case "$KUMO_POSTGRES_MODE" in
        bundled|external) ;;
        *)
            echo "ERROR: postgres_mode must be 'bundled' or 'external', got '$KUMO_POSTGRES_MODE'"
            exit 1
            ;;
    esac
}

read_config

CLOUD_REGION=$(cloud_region "$CLOUD_PROVIDER")
AWS_REGION=$([ "$CLOUD_PROVIDER" = aws ] && echo "$CLOUD_REGION" || echo us-west-2)
echo "Cloud region: $CLOUD_REGION"
AZURE_CLOUD_NAME=""
AZURE_AUTHORITY_HOST=""
if [ "$CLOUD_PROVIDER" = azure ] && parse_adls_root "$DATAPLANE_ROOT"; then
    AZURE_CLOUD_NAME="$ADLS_CLOUD"
    AZURE_AUTHORITY_HOST=$(azure_authority_for_cloud "$AZURE_CLOUD_NAME")
fi
if [ "$CLOUD_PROVIDER" = azure ] && [ -n "$AZURE_CLOUD_SETTING" ]; then
    CONFIGURED_AZURE_CLOUD=$(normalize_azure_cloud "$AZURE_CLOUD_SETTING") || {
        echo "ERROR: azure_cloud must be public, usgovernment, or china"
        exit 1
    }
    if [ -n "$AZURE_CLOUD_NAME" ] && [ "$CONFIGURED_AZURE_CLOUD" != "$AZURE_CLOUD_NAME" ]; then
        echo "ERROR: azure_cloud does not match the ADLS endpoint suffix"
        exit 1
    fi
    AZURE_CLOUD_NAME="$CONFIGURED_AZURE_CLOUD"
    AZURE_AUTHORITY_HOST=$(azure_authority_for_cloud "$AZURE_CLOUD_NAME")
fi
if [ "$CLOUD_PROVIDER" = azure ] && [ -z "$AZURE_CLIENT_ID" ]; then
    echo "ERROR: Azure deployments require azure_client_id for the user-assigned managed identity"
    exit 1
fi

if [ "$CLOUD_PROVIDER" = azure ] && { [ -n "$OIDC_CLIENT_SECRET_URI" ] || [ -n "$DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI" ]; }; then
    az cloud set --name "$AZURE_CLOUD_NAME" >/dev/null
    azure_managed_identity_login
    if [ -n "$OIDC_CLIENT_SECRET_URI" ]; then
        OIDC_CLIENT_SECRET=$(az keyvault secret show --id "$OIDC_CLIENT_SECRET_URI" \
            --query value -o tsv --only-show-errors)
    fi
    if [ -n "$DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI" ]; then
        DIRECT_CONNECT_ADMIN_TEMP_PASSWORD=$(az keyvault secret show \
            --id "$DIRECT_CONNECT_ADMIN_TEMP_PASSWORD_URI" \
            --query value -o tsv --only-show-errors)
    fi
fi

if [ "$DISABLE_AUTH" != "true" ]; then
    if [ -z "$PUBLIC_URL" ] && [ "$ALLOW_LOCAL_PUBLIC_URL" != "true" ]; then
        echo "ERROR: Direct Connect auth requires public_url in instance user-data."
        echo "Set public_url to the browser-facing ALB/DNS origin, for example: https://kumo.example.com"
        exit 1
    fi
    case "$PUBLIC_URL" in
        http://localhost|https://localhost|http://localhost:*|https://localhost:*|http://127.0.0.1|https://127.0.0.1|http://127.0.0.1:*|https://127.0.0.1:*)
            if [ "$ALLOW_LOCAL_PUBLIC_URL" != "true" ]; then
                echo "ERROR: public_url=$PUBLIC_URL is local-only and is not valid for Direct Connect auth behind ALB/DNS."
                echo "Set allow_local_public_url: true only for local smoke tests."
                exit 1
            fi
            ;;
    esac
fi

if [ "$DISABLE_AUTH" != "true" ]; then
    missing_direct_connect_config=""
    [ -z "${ADMIN_EMAILS:-}" ] && missing_direct_connect_config="${missing_direct_connect_config} admin_emails"
    case "$DIRECT_CONNECT_LOGIN_METHOD" in
        oidc)
            [ -z "${OIDC_ISSUER:-}" ] && missing_direct_connect_config="${missing_direct_connect_config} oidc_issuer"
            [ -z "${OIDC_CLIENT_ID:-}" ] && missing_direct_connect_config="${missing_direct_connect_config} oidc_client_id"
            [ -z "${OIDC_CLIENT_SECRET:-}" ] && missing_direct_connect_config="${missing_direct_connect_config} oidc_client_secret"
            ;;
        password)
            [ -z "${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD:-}" ] && missing_direct_connect_config="${missing_direct_connect_config} direct_connect_admin_temp_password"
            ;;
        *)
            echo "ERROR: direct_connect_login_method must be oidc or password, got: $DIRECT_CONNECT_LOGIN_METHOD"
            exit 1
            ;;
    esac
    if [ -n "$missing_direct_connect_config" ]; then
        echo "ERROR: Direct Connect auth missing required user-data fields:${missing_direct_connect_config}"
        exit 1
    fi
fi

# Derived single-host state (credential secret, bundled Postgres password + DSN
# secrets, deploy_env_config.json, self-signed TLS, nginx variant) is generated
# by the shared bootstrap module so the AMI host path and the compose bootstrap
# container run one implementation. Firstboot only supplies the cloud-derived
# inputs and points the module at the AMI host roots the compose binds expect.
KUMO_STATE_DB_NAME="${KUMO_STATE_DB_NAME:-kumo}"
TEMPORAL_DB_NAME="${TEMPORAL_DB_NAME:-temporal}"
TEMPORAL_VISIBILITY_DB_NAME="${TEMPORAL_VISIBILITY_DB_NAME:-temporal_visibility}"
TEMPORAL_SERVER_IMAGE="${TEMPORAL_SERVER_IMAGE:-temporalio/server:1.31.1}"
TEMPORAL_ADMIN_TOOLS_IMAGE="${TEMPORAL_ADMIN_TOOLS_IMAGE:-temporalio/admin-tools:1.31.1}"

set_temporal_postgres_env() {
    TEMPORAL_SQL_TLS=false
    TEMPORAL_SQL_TLS_ENABLED=false
    TEMPORAL_SQL_TLS_CA_FILE="${POSTGRES_SSLROOTCERT:-}"
    TEMPORAL_SQL_TLS_SERVER_NAME=""
    TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION=false
    TEMPORAL_SQL_HOST_VERIFICATION=false

    case "${POSTGRES_SSLMODE:-}" in
        "")
            ;;
        verify-ca)
            TEMPORAL_SQL_TLS=true
            TEMPORAL_SQL_TLS_ENABLED=true
            TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION=true
            ;;
        verify-full)
            TEMPORAL_SQL_TLS=true
            TEMPORAL_SQL_TLS_ENABLED=true
            TEMPORAL_SQL_TLS_SERVER_NAME="${POSTGRES_HOST:-}"
            TEMPORAL_SQL_HOST_VERIFICATION=true
            ;;
        *)
            echo "ERROR: POSTGRES_SSLMODE must be verify-ca or verify-full" >&2
            exit 1
            ;;
    esac
}

if [ -z "${POSTGRES_PASSWORD:-}" ] && [ -n "${POSTGRES_PASSWORD_FILE:-}" ]; then
    POSTGRES_PASSWORD="$(cat "$POSTGRES_PASSWORD_FILE")"
fi

# Bundled Postgres connection defaults shared by the .env and the DSN secrets.
if [ "$KUMO_POSTGRES_MODE" != "external" ]; then
    POSTGRES_USER="${POSTGRES_USER:-kumo}"
    POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
    POSTGRES_PORT="${POSTGRES_PORT:-5432}"
fi

# Deploy-config render inputs (defaults match the template's expectations).
TEMPORAL_PORT="${TEMPORAL_PORT:-7233}"
TEMPORAL_HOST="${TEMPORAL_HOST:-temporal}"
METADATA_STORAGE_TYPE="${METADATA_STORAGE_TYPE:-postgres}"
STATE_DB_DSN_SECRET_NAME="${STATE_DB_DSN_SECRET_NAME:-STATE_DB_DSN}"

# Browser-facing origin used for the Direct Connect return-origin allowlist.
# Defaulted before the module runs so the rendered auth config is complete.
if [ -z "$PUBLIC_URL" ]; then
    if [ "$ENABLE_TLS" = "true" ]; then
        PUBLIC_URL="https://localhost"
    else
        PUBLIC_URL="http://localhost"
    fi
fi
PUBLIC_URL="${PUBLIC_URL%/}"

# Firstboot is installed under /var/lib/cloud/scripts, so the module is not a
# sibling on the running instance; fall back to the staged deploy/shared tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_MODULE="$SCRIPT_DIR/kumo-bootstrap.py"
[ -f "$BOOTSTRAP_MODULE" ] || BOOTSTRAP_MODULE="/opt/kumo/deploy-shared/scripts/kumo-bootstrap.py"
export ENABLE_TLS DISABLE_AUTH DIRECT_CONNECT_LOGIN_METHOD ADMIN_EMAILS
export OIDC_ISSUER OIDC_CLIENT_ID OIDC_CLIENT_SECRET OIDC_GROUPS_CLAIM PUBLIC_URL
export DIRECT_CONNECT_ADMIN_TEMP_PASSWORD KUMO_POSTGRES_MODE
export POSTGRES_HOST POSTGRES_PORT POSTGRES_USER POSTGRES_PASSWORD
export POSTGRES_PASSWORD_FILE POSTGRES_SSLMODE POSTGRES_SSLROOTCERT
export KUMO_STATE_DB_NAME TEMPORAL_DB_NAME TEMPORAL_VISIBILITY_DB_NAME
export DATAPLANE_ROOT DISKGRAPH_DATA_ROOT ARTIFACT_ROOT TEMPORAL_NAMESPACE LIFECYCLE_READY
export TEMPORAL_HOST TEMPORAL_PORT METADATA_STORAGE_TYPE STATE_DB_DSN_SECRET_NAME
KUMO_BOOTSTRAP_SECRETS_DIR=/opt/kumo/secrets \
KUMO_BOOTSTRAP_DEPLOY_CONFIG_FILE=/etc/kumo/deploy_env_config.json \
KUMO_BOOTSTRAP_NGINX_FILE=/opt/kumo/nginx/nginx.conf \
KUMO_BOOTSTRAP_PROMETHEUS_FILE=/opt/kumo/prometheus/prometheus.yml \
KUMO_BOOTSTRAP_TLS_DIR=/etc/kumo/tls \
    python3 "$BOOTSTRAP_MODULE"
echo "Bootstrap module generated secrets, deploy config, TLS, and nginx config"

# External Postgres: prove connectivity from the same Docker surface Kumo uses.
# Bundled Postgres: read the generated password into the compose .env.
if [ "$KUMO_POSTGRES_MODE" = "external" ]; then
    KUMO_POSTGRES_SECRETS_DIR=/opt/kumo/secrets \
    /opt/kumo/scripts/check-postgres-dsns.sh \
        /opt/kumo/secrets/STATE_DB_DSN \
        /opt/kumo/secrets/TEMPORAL_DB_DSN \
        /opt/kumo/secrets/TEMPORAL_VISIBILITY_DB_DSN
else
    POSTGRES_PASSWORD="$(cat /opt/kumo/secrets/kumo-postgres-password)"
fi
set_temporal_postgres_env
POSTGRES_ENV=$(cat << PGENV
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
KUMO_STATE_DB_NAME=${KUMO_STATE_DB_NAME}
TEMPORAL_DB_NAME=${TEMPORAL_DB_NAME}
TEMPORAL_VISIBILITY_DB_NAME=${TEMPORAL_VISIBILITY_DB_NAME}
TEMPORAL_SERVER_IMAGE=${TEMPORAL_SERVER_IMAGE}
TEMPORAL_ADMIN_TOOLS_IMAGE=${TEMPORAL_ADMIN_TOOLS_IMAGE}
TEMPORAL_SQL_TLS=${TEMPORAL_SQL_TLS}
TEMPORAL_SQL_TLS_ENABLED=${TEMPORAL_SQL_TLS_ENABLED}
TEMPORAL_SQL_TLS_CA_FILE=${TEMPORAL_SQL_TLS_CA_FILE}
TEMPORAL_SQL_TLS_SERVER_NAME=${TEMPORAL_SQL_TLS_SERVER_NAME}
TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION=${TEMPORAL_SQL_TLS_DISABLE_HOST_VERIFICATION}
TEMPORAL_SQL_HOST_VERIFICATION=${TEMPORAL_SQL_HOST_VERIFICATION}
PGENV
)

# Write .env for docker-compose (image tag is baked into AMI)
KUMO_IMAGE_TAG=$(cat /etc/kumo/image_tag 2>/dev/null || echo "latest")
# Determine local mount path for containers
case "$DATAPLANE_ROOT" in
    s3://*|gs://*|abfs://*|abfss://*) LOCAL_DATAPLANE_MOUNT="/data/kumo" ;;
    *) LOCAL_DATAPLANE_MOUNT="$(dirname "$DATAPLANE_ROOT")" ;;
esac

if [ "$CLOUD_PROVIDER" = azure ]; then
    case "$ARTIFACT_ROOT" in
        abfss://*)
            echo "Checking Azure artifact-root read access: $ARTIFACT_ROOT"
            if ! check_adls_list_access "$ARTIFACT_ROOT"; then
                echo "ERROR: Cannot list Azure artifact root at $ARTIFACT_ROOT"
                echo "Check Storage Blob Data Reader RBAC and private DNS/network access."
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Azure artifact_root must use provider-native ADLS storage"
            exit 1
            ;;
    esac
fi
# The compose dataplane bind mount is HOST_MOUNT:MOUNT. On the AMI the host
# path equals the container path (firstboot mkdir/chowns it to uid 999 below),
# so set HOST_MOUNT=MOUNT.
LOCAL_DATAPLANE_HOST_MOUNT="$LOCAL_DATAPLANE_MOUNT"

AUTH_ENV=$(cat << DCEOF
DISABLE_AUTH=${DISABLE_AUTH}
DCEOF
)

NGINX_PORT=$([ "$ENABLE_TLS" = "true" ] && echo "443" || echo "80")

# Only an explicit slot count is persisted; otherwise kumo-boot.sh counts the
# GPUs on every boot, so a resized instance never keeps a stale count.
GPU_WORKER_SLOTS_LINE=""
if [ -n "${ML_MAX_GPU_WORKER_SLOTS:-}" ]; then
    GPU_WORKER_SLOTS_LINE="ML_MAX_GPU_WORKER_SLOTS=${ML_MAX_GPU_WORKER_SLOTS}"
fi

cat > /opt/kumo/compose/.env << EOF
KUMO_IMAGE_TAG=${KUMO_IMAGE_TAG}
DEPLOY_STAGE=${DEPLOY_STAGE:-prod}
TEMPORAL_NAMESPACE=${TEMPORAL_NAMESPACE}
DEPLOY_ENV_CONFIG_FILE=/etc/kumo/deploy_env_config.json
LOCAL_DATAPLANE_MOUNT=${LOCAL_DATAPLANE_MOUNT}
LOCAL_DATAPLANE_HOST_MOUNT=${LOCAL_DATAPLANE_HOST_MOUNT}
AWS_REGION=${AWS_REGION}
AZURE_CLOUD_NAME=${AZURE_CLOUD_NAME}
AZURE_AUTHORITY_HOST=${AZURE_AUTHORITY_HOST}
AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
KUMO_POSTGRES_MODE=${KUMO_POSTGRES_MODE}
NGINX_PORT=${NGINX_PORT}
${GPU_WORKER_SLOTS_LINE}
KUMO_SECRETS_DIR=/opt/kumo/secrets
KUMO_TLS_DIR=/etc/kumo/tls
DEPLOY_ENV_CONFIG_PATH=/etc/kumo/deploy_env_config.json
DEPLOY_ENV_CONFIG_MOUNT=/etc/kumo/deploy_env_config.json
NGINX_CONFIG=/opt/kumo/nginx/nginx.conf
NGINX_CONFIG_MOUNT=/etc/nginx/nginx.conf
NGINX_CONFIG_FILE=/etc/nginx/nginx.conf
PROMETHEUS_CONFIG=/opt/kumo/prometheus/prometheus.yml
PROMETHEUS_CONFIG_MOUNT=/etc/prometheus/prometheus.yml
PROMETHEUS_CONFIG_FILE=/etc/prometheus/prometheus.yml
KUMO_BOOTSTRAP_DIR=/opt/kumo/deploy-shared
${POSTGRES_ENV}
${AUTH_ENV}
EOF
chmod 600 /opt/kumo/compose/.env

# Verify dataplane access and create directory if local
case "$DATAPLANE_ROOT" in
    s3://*|gs://*|abfs://*|abfss://*)
        echo "Checking cloud dataplane access: $DATAPLANE_ROOT"
        if ! check_cloud_storage_access "$DATAPLANE_ROOT"; then
            echo "ERROR: Dataplane preflight failed for $DATAPLANE_ROOT"
            echo "Required scoped permissions: list, write, read, delete, and sign/download."
            exit 1
        fi
        ;;
    *)
        echo "Creating local dataplane directory: $DATAPLANE_ROOT"
        mkdir -p "$DATAPLANE_ROOT"
        chown -R 999:999 "$DATAPLANE_ROOT"
        ;;
esac

echo "=== Kumo first-boot complete ==="
