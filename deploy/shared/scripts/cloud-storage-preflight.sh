#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Testable cloud metadata, storage-root, and access-preflight helpers used by
# kumo-firstboot.sh. This file has no top-level side effects so offline tests
# can source it with fake provider CLIs on PATH.

azure_metadata() {
    local path="$1"
    curl -sf -m 2 -H "Metadata: true" \
        "http://169.254.169.254/metadata/instance/${path}?api-version=2021-02-01&format=text"
}

detect_cloud_provider() {
    if curl -sf -m 2 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
        echo "aws"
    elif curl -sf -m 2 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/id >/dev/null 2>&1; then
        echo "gcp"
    elif azure_metadata "compute/vmId" >/dev/null 2>&1; then
        echo "azure"
    else
        echo "unknown"
    fi
}

read_cloud_config() {
    case "$1" in
        aws)
            curl -sf http://169.254.169.254/latest/user-data 2>/dev/null || true
            ;;
        gcp)
            curl -sf -H "Metadata-Flavor: Google" \
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/kumo-config" 2>/dev/null || true
            ;;
        azure)
            # Azure IMDS returns customData/userData as base64-encoded text.
            local encoded
            encoded=$(azure_metadata "compute/userData" 2>/dev/null || true)
            [ -z "$encoded" ] || printf '%s' "$encoded" | base64 --decode
            ;;
    esac
}

cloud_region() {
    case "$1" in
        aws)
            local token region
            token=$(curl -sf -m 2 -X PUT \
                -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
                http://169.254.169.254/latest/api/token 2>/dev/null || true)
            if [ -n "$token" ]; then
                region=$(curl -sf -m 2 -H "X-aws-ec2-metadata-token: $token" \
                    http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
            fi
            if [ -z "${region:-}" ]; then
                region=$(curl -sf -m 2 \
                    http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
            fi
            echo "${region:-us-west-2}"
            ;;
        gcp)
            local zone
            zone=$(curl -sf -m 2 -H "Metadata-Flavor: Google" \
                http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null || true)
            echo "$zone" | sed 's|.*/||; s|-[a-z]$||'
            ;;
        azure)
            azure_metadata "compute/location"
            ;;
        *) echo "us-west-2" ;;
    esac
}

is_remote_storage_root() {
    case "$1" in
        s3://*|gs://*|abfs://*|abfss://*) return 0 ;;
        *) return 1 ;;
    esac
}

azure_cloud_for_suffix() {
    case "$1" in
        dfs.core.windows.net) echo "AzureCloud" ;;
        dfs.core.usgovcloudapi.net) echo "AzureUSGovernment" ;;
        dfs.core.chinacloudapi.cn) echo "AzureChinaCloud" ;;
        *) return 1 ;;
    esac
}

azure_authority_for_cloud() {
    case "$1" in
        AzureCloud) echo "https://login.microsoftonline.com/" ;;
        AzureUSGovernment) echo "https://login.microsoftonline.us/" ;;
        AzureChinaCloud) echo "https://login.chinacloudapi.cn/" ;;
        *) return 1 ;;
    esac
}

normalize_azure_cloud() {
    case "${1,,}" in
        public|azurecloud) echo "AzureCloud" ;;
        usgovernment|azureusgovernment) echo "AzureUSGovernment" ;;
        china|azurechinacloud) echo "AzureChinaCloud" ;;
        *) return 1 ;;
    esac
}

azure_managed_identity_login() {
    local args=(--identity --allow-no-subscriptions --only-show-errors)
    if [ -n "${AZURE_CLIENT_ID:-}" ]; then
        args+=(--client-id "$AZURE_CLIENT_ID")
    fi
    az login "${args[@]}" >/dev/null
}

parse_adls_root() {
    local uri="$1" rest authority path account_host
    case "$uri" in
        abfs://*|abfss://*) ;;
        *) return 1 ;;
    esac
    case "$uri" in
        *\?*|*\#*|*\\*|*%2f*|*%2F*|*%5c*|*%5C*) return 1 ;;
    esac
    rest="${uri#*://}"
    authority="${rest%%/*}"
    [ "$rest" = "$authority" ] && path="" || path="${rest#*/}"
    case "$authority" in
        *@*) ;;
        *) return 1 ;;
    esac
    ADLS_FILESYSTEM="${authority%%@*}"
    account_host="${authority#*@}"
    ADLS_ACCOUNT="${account_host%%.*}"
    ADLS_ENDPOINT_SUFFIX="${account_host#*.}"
    ADLS_PREFIX="${path%/}"
    if [[ "$ADLS_FILESYSTEM" =~ ^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$ ]] &&
        [[ "$ADLS_ACCOUNT" =~ ^[a-z0-9]{3,24}$ ]] &&
        [ "$account_host" = "${ADLS_ACCOUNT}.${ADLS_ENDPOINT_SUFFIX}" ] &&
        { [ -z "$ADLS_PREFIX" ] || [[ "/$ADLS_PREFIX/" != *"//"* ]]; } &&
        { [ -z "$ADLS_PREFIX" ] || [[ "/$ADLS_PREFIX/" != *"/./"* ]]; } &&
        { [ -z "$ADLS_PREFIX" ] || [[ "/$ADLS_PREFIX/" != *"/../"* ]]; } &&
        ADLS_CLOUD=$(azure_cloud_for_suffix "$ADLS_ENDPOINT_SUFFIX"); then
        ADLS_CANONICAL_ROOT="abfss://${ADLS_FILESYSTEM}@${ADLS_ACCOUNT}.${ADLS_ENDPOINT_SUFFIX}${ADLS_PREFIX:+/${ADLS_PREFIX}}"
        return 0
    fi
    return 1
}

canonicalize_adls_root() {
    parse_adls_root "$1" || return 1
    printf '%s' "$ADLS_CANONICAL_ROOT"
}

check_s3_storage_access() {
    local root="$1" s3_path bucket prefix probe_key probe_body probe_download
    local status=0 delete_status=0
    s3_path="${root#s3://}"
    bucket="${s3_path%%/*}"
    [ "$s3_path" = "$bucket" ] && prefix="" || prefix="${s3_path#*/}"
    prefix="${prefix%/}"
    [ -n "$bucket" ] || return 1
    probe_key="${prefix:+${prefix}/}.kumo-firstboot-access-check"
    probe_body=$(mktemp)
    probe_download=$(mktemp)
    printf 'kumo-storage-preflight' > "$probe_body"
    aws s3api list-objects-v2 --bucket "$bucket" \
        --prefix "${prefix:+${prefix}/}" --max-items 1 --region "$AWS_REGION" >/dev/null &&
    aws s3api put-object --bucket "$bucket" --key "$probe_key" \
        --body "$probe_body" --region "$AWS_REGION" >/dev/null &&
    aws s3api get-object --bucket "$bucket" --key "$probe_key" \
        --region "$AWS_REGION" "$probe_download" >/dev/null &&
    cmp -s "$probe_body" "$probe_download" &&
    aws s3 presign "s3://${bucket}/${probe_key}" --region "$AWS_REGION" >/dev/null || status=$?
    aws s3api delete-object --bucket "$bucket" --key "$probe_key" \
        --region "$AWS_REGION" >/dev/null 2>&1 || delete_status=$?
    rm -f "$probe_body" "$probe_download"
    if [ "$status" -eq 0 ] && [ "$delete_status" -ne 0 ]; then
        status="$delete_status"
    fi
    return "$status"
}

check_gcs_storage_access() {
    local root="$1" bucket probe_uri probe_body probe_download
    local status=0 delete_status=0
    bucket="${root#gs://}"
    bucket="${bucket%%/*}"
    probe_uri="${root%/}/.kumo-firstboot-access-check"
    probe_body=$(mktemp)
    probe_download=$(mktemp)
    printf 'kumo-storage-preflight' > "$probe_body"
    (gsutil ls "${root%/}/" >/dev/null 2>&1 || gsutil ls "gs://${bucket}" >/dev/null) &&
    gsutil -q cp "$probe_body" "$probe_uri" &&
    gsutil -q cp "$probe_uri" "$probe_download" &&
    cmp -s "$probe_body" "$probe_download" &&
    gcloud storage sign-url "$probe_uri" --duration=5m >/dev/null || status=$?
    gsutil -q rm "$probe_uri" >/dev/null 2>&1 || delete_status=$?
    rm -f "$probe_body" "$probe_download"
    if [ "$status" -eq 0 ] && [ "$delete_status" -ne 0 ]; then
        status="$delete_status"
    fi
    return "$status"
}

check_adls_storage_access() {
    local root="$1" probe_path probe_body probe_download expiry sas
    local directory_exists status=0 delete_status=0
    parse_adls_root "$root" || return 1
    az cloud set --name "$ADLS_CLOUD" >/dev/null
    azure_managed_identity_login
    if [ -n "$ADLS_PREFIX" ]; then
        directory_exists=$(az storage fs directory exists \
            --account-name "$ADLS_ACCOUNT" \
            --file-system "$ADLS_FILESYSTEM" --name "$ADLS_PREFIX" \
            --auth-mode login --only-show-errors --query exists -o tsv)
        if [ "$directory_exists" != "true" ]; then
            az storage fs directory create --account-name "$ADLS_ACCOUNT" \
                --file-system "$ADLS_FILESYSTEM" --name "$ADLS_PREFIX" \
                --auth-mode login --only-show-errors >/dev/null
        fi
    fi
    probe_path="${ADLS_PREFIX:+${ADLS_PREFIX}/}.kumo-firstboot-access-check"
    probe_body=$(mktemp)
    probe_download=$(mktemp)
    printf 'kumo-storage-preflight' > "$probe_body"
    az storage fs file list --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" --path "$ADLS_PREFIX" \
        --auth-mode login --only-show-errors --num-results 1 >/dev/null &&
    az storage fs file upload --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" --path "$probe_path" \
        --source "$probe_body" --overwrite --auth-mode login --only-show-errors >/dev/null &&
    az storage fs file download --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" --path "$probe_path" \
        --destination "$probe_download" --overwrite --auth-mode login --only-show-errors >/dev/null &&
    cmp -s "$probe_body" "$probe_download" &&
    expiry=$(date -u -d '+5 minutes' '+%Y-%m-%dT%H:%MZ') &&
    sas=$(az storage blob generate-sas --account-name "$ADLS_ACCOUNT" \
        --container-name "$ADLS_FILESYSTEM" --name "$probe_path" \
        --permissions r --expiry "$expiry" --auth-mode login --as-user \
        --only-show-errors -o tsv) &&
    [ -n "$sas" ] || status=$?
    az storage fs file delete --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" --path "$probe_path" \
        --auth-mode login --yes --only-show-errors >/dev/null 2>&1 || delete_status=$?
    rm -f "$probe_body" "$probe_download"
    if [ "$status" -eq 0 ] && [ "$delete_status" -ne 0 ]; then
        status="$delete_status"
    fi
    return "$status"
}

check_adls_list_access() {
    parse_adls_root "$1" || return 1
    az cloud set --name "$ADLS_CLOUD" >/dev/null &&
    azure_managed_identity_login &&
    az storage fs file list --account-name "$ADLS_ACCOUNT" \
        --file-system "$ADLS_FILESYSTEM" --path "$ADLS_PREFIX" \
        --auth-mode login --only-show-errors --num-results 1 >/dev/null
}

check_cloud_storage_access() {
    case "$1" in
        s3://*) check_s3_storage_access "$1" ;;
        gs://*) check_gcs_storage_access "$1" ;;
        abfs://*|abfss://*) check_adls_storage_access "$1" ;;
        *) return 1 ;;
    esac
}
