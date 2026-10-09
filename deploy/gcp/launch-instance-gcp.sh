#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch a GCP VM from a Kumo singlehost image.
# Usage: ./launch-instance-gcp.sh <deployment-name> <image-name> [instance-name]
#
#   <image-name>: exact GCP image name, e.g. kumo-singlehost-861aa1b8
#                 (find with: gcloud compute images list --project=<project> --filter="family=kumo-singlehost")
#   [instance-name]: defaults to <deployment-name>
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project <project-id>
#
# Common overrides (env vars):
#   DATAPLANE_ROOT=gs://<bucket>/kumo/dataplane  # default: gs://kumo-<deployment-name>/kumo/dataplane
#   GCP_ZONE=us-west1-b                          # default
#   MACHINE_TYPE=n1-standard-32                  # default
#   ACCELERATOR_TYPE=nvidia-tesla-t4             # default (nvidia-l4 for g2, nvidia-tesla-a100 for a2)
#   ACCELERATOR_COUNT=1                          # default
#   LOCAL_SSD_COUNT=1                            # default (375GB each, NVME)
#   LOCAL_SSD_INTERFACE=nvme                     # default (nvme or scsi)
#   BOOT_DISK_SIZE_GB=100                        # default
#   BOOT_DISK_TYPE=pd-ssd                        # default
#   ENABLE_TLS=false                             # optional; default false set by kumo-firstboot.sh
#   PUBLIC_URL=https://kumo.example.com          # required for Direct Connect auth unless local URLs are allowed
#   DIRECT_CONNECT_LOGIN_METHOD=password         # optional; oidc or password, default oidc
#   OIDC_ISSUER=https://idp.example.com          # required for Direct Connect OIDC mode
#   OIDC_CLIENT_ID=kumo                          # required for Direct Connect OIDC mode
#   OIDC_CLIENT_SECRET=...                       # required for Direct Connect OIDC mode
#   OIDC_GROUPS_CLAIM=groups                     # optional; default groups
#   ADMIN_EMAILS=admin@example.com               # required for Direct Connect
#   DIRECT_CONNECT_ADMIN_TEMP_PASSWORD=...       # required for Direct Connect password mode
#   SUBNET=projects/PROJECT/regions/REGION/subnetworks/NAME  # optional
#   SERVICE_ACCOUNT=sa@project.iam.gserviceaccount.com       # optional; auto-created if unset
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <deployment-name> <image-name> [instance-name]"
    echo "  Find available images:"
    echo "    gcloud compute images list --project=<project> --filter=\"family=kumo-singlehost\""
    exit 1
fi

DEPLOYMENT_NAME="$1"
IMAGE_NAME="$2"
INSTANCE_NAME="${3:-${DEPLOYMENT_NAME}}"
DATAPLANE_ROOT="${DATAPLANE_ROOT:-gs://kumo-${DEPLOYMENT_NAME}/kumo/dataplane}"

if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "ERROR: gcloud user credentials not found. Run: gcloud auth login"
    exit 1
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}"
if [ -z "$GCP_PROJECT" ]; then
    echo "ERROR: GCP project not set. Either:"
    echo "  export GCP_PROJECT=<project-id>"
    echo "  gcloud config set project <project-id>"
    exit 1
fi

# Instance parameters
GCP_ZONE="${GCP_ZONE:-us-west1-b}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-32}"
# gcloud uses lowercase-hyphen format: nvidia-tesla-t4, nvidia-l4, nvidia-tesla-a100
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-tesla-t4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-100}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
LOCAL_SSD_COUNT="${LOCAL_SSD_COUNT:-1}"
LOCAL_SSD_INTERFACE="${LOCAL_SSD_INTERFACE:-nvme}"
SUBNET="${SUBNET:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"

# Optional Kumo config — if unset, kumo-firstboot.sh applies its own defaults
ENABLE_TLS="${ENABLE_TLS:-}"
PUBLIC_URL="${PUBLIC_URL:-}"
ALLOW_LOCAL_PUBLIC_URL="${ALLOW_LOCAL_PUBLIC_URL:-}"
DISABLE_AUTH="${DISABLE_AUTH:-}"
OIDC_ISSUER="${OIDC_ISSUER:-}"
OIDC_CLIENT_ID="${OIDC_CLIENT_ID:-}"
OIDC_CLIENT_SECRET="${OIDC_CLIENT_SECRET:-}"
OIDC_GROUPS_CLAIM="${OIDC_GROUPS_CLAIM:-}"
DIRECT_CONNECT_LOGIN_METHOD="${DIRECT_CONNECT_LOGIN_METHOD:-}"
ADMIN_EMAILS="${ADMIN_EMAILS:-}"
DIRECT_CONNECT_ADMIN_TEMP_PASSWORD="${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD:-}"

GCP_REGION="${GCP_ZONE%-?}"

# --- GCP resource setup: service account, bucket, IAM ---

# Create or accept a per-deployment service account
if [ -z "$SERVICE_ACCOUNT" ]; then
    # GCP SA IDs are 6-30 chars; truncate if needed
    KUMO_SA_NAME="${DEPLOYMENT_NAME:0:24}"
    KUMO_SA_NAME="kumo-${KUMO_SA_NAME}"
    SERVICE_ACCOUNT="${KUMO_SA_NAME}@${GCP_PROJECT}.iam.gserviceaccount.com"

    if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" \
            --project="${GCP_PROJECT}" >/dev/null 2>&1; then
        echo "Creating service account: ${SERVICE_ACCOUNT}"
        gcloud iam service-accounts create "${KUMO_SA_NAME}" \
            --project="${GCP_PROJECT}" \
            --display-name="Kumo singlehost (${DEPLOYMENT_NAME})"
    else
        echo "Service account: ${SERVICE_ACCOUNT} (exists)"
    fi
fi

# Grant project-level roles (idempotent)
echo "Binding IAM roles to ${SERVICE_ACCOUNT}..."
for role in \
    roles/secretmanager.secretAccessor \
    roles/logging.logWriter \
    roles/monitoring.metricWriter \
    roles/iam.serviceAccountTokenCreator; do
    gcloud projects add-iam-policy-binding "${GCP_PROJECT}" \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="${role}" \
        --condition=None \
        --quiet >/dev/null
done

# Create dataplane bucket and grant object access (GCS only)
case "$DATAPLANE_ROOT" in
    gs://*)
        GCS_BUCKET=$(echo "$DATAPLANE_ROOT" | sed 's|gs://||; s|/.*||')
        if ! gcloud storage buckets describe "gs://${GCS_BUCKET}" \
                --project="${GCP_PROJECT}" >/dev/null 2>&1; then
            echo "Creating GCS bucket: gs://${GCS_BUCKET} (${GCP_REGION})"
            gcloud storage buckets create "gs://${GCS_BUCKET}" \
                --project="${GCP_PROJECT}" \
                --location="${GCP_REGION}" \
                --uniform-bucket-level-access
        else
            echo "GCS bucket: gs://${GCS_BUCKET} (exists)"
        fi
        gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
            --member="serviceAccount:${SERVICE_ACCOUNT}" \
            --role=roles/storage.objectAdmin \
            --quiet >/dev/null
        echo "Granted storage.objectAdmin on gs://${GCS_BUCKET}"
        ;;
esac

# Repeat --local-ssd once per disk
LOCAL_SSD_ARGS=()
for _ in $(seq 1 "$LOCAL_SSD_COUNT"); do
    LOCAL_SSD_ARGS+=("--local-ssd=interface=${LOCAL_SSD_INTERFACE}")
done

# Optional flags
OPTIONAL_ARGS=()
if [ -n "$SUBNET" ]; then
    OPTIONAL_ARGS+=("--subnet=${SUBNET}")
fi

echo ""
echo "GCP project:     ${GCP_PROJECT}"
echo "GCP zone:        ${GCP_ZONE}"
echo "Instance name:   ${INSTANCE_NAME}"
echo "Image:           ${IMAGE_NAME} (project: ${GCP_PROJECT})"
echo "Machine type:    ${MACHINE_TYPE}"
echo "Accelerator:     ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo "Boot disk:       ${BOOT_DISK_SIZE_GB}GB ${BOOT_DISK_TYPE}"
echo "Local SSDs:      ${LOCAL_SSD_COUNT}x 375GB (${LOCAL_SSD_INTERFACE})"
echo "Service account: ${SERVICE_ACCOUNT}"
echo "Deployment:      ${DEPLOYMENT_NAME}"
echo "Dataplane:       ${DATAPLANE_ROOT}"
echo ""

# Write kumo-config to a temp file; gcloud reads it as the metadata value.
# Optional keys are omitted when unset so kumo-firstboot.sh uses its own defaults.
KUMO_CONFIG_FILE=$(mktemp)
trap 'rm -f "${KUMO_CONFIG_FILE}"' EXIT
{
    echo "dataplane_root: ${DATAPLANE_ROOT}"
    [ -n "${ENABLE_TLS}" ] && echo "enable_tls: ${ENABLE_TLS}"
    [ -n "${PUBLIC_URL}" ] && echo "public_url: ${PUBLIC_URL}"
    [ -n "${ALLOW_LOCAL_PUBLIC_URL}" ] && echo "allow_local_public_url: ${ALLOW_LOCAL_PUBLIC_URL}"
    [ -n "${DISABLE_AUTH}" ] && echo "disable_auth: ${DISABLE_AUTH}"
    [ -n "${OIDC_ISSUER}" ] && echo "oidc_issuer: ${OIDC_ISSUER}"
    [ -n "${OIDC_CLIENT_ID}" ] && echo "oidc_client_id: ${OIDC_CLIENT_ID}"
    [ -n "${OIDC_CLIENT_SECRET}" ] && echo "oidc_client_secret: ${OIDC_CLIENT_SECRET}"
    [ -n "${OIDC_GROUPS_CLAIM}" ] && echo "oidc_groups_claim: ${OIDC_GROUPS_CLAIM}"
    [ -n "${DIRECT_CONNECT_LOGIN_METHOD}" ] && echo "direct_connect_login_method: ${DIRECT_CONNECT_LOGIN_METHOD}"
    [ -n "${ADMIN_EMAILS}" ] && echo "admin_emails: ${ADMIN_EMAILS}"
    [ -n "${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD}" ] && echo "direct_connect_admin_temp_password: ${DIRECT_CONNECT_ADMIN_TEMP_PASSWORD}"
} > "${KUMO_CONFIG_FILE}"

gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --image="${IMAGE_NAME}" \
    --image-project="${GCP_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type="${BOOT_DISK_TYPE}" \
    --no-address \
    --service-account="${SERVICE_ACCOUNT}" \
    --scopes=cloud-platform \
    "${LOCAL_SSD_ARGS[@]}" \
    "${OPTIONAL_ARGS[@]}" \
    --metadata=serial-port-logging-enable=true \
    --metadata-from-file=kumo-config="${KUMO_CONFIG_FILE}"

echo ""
echo "Connect via IAP tunnel:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --project=${GCP_PROJECT} --zone=${GCP_ZONE} --tunnel-through-iap"
echo ""
echo "Check first-boot progress (cloud-init log):"
echo "  gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --project=${GCP_PROJECT} --zone=${GCP_ZONE}"
