#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <kumo-image-tag>" >&2
    exit 2
fi
: "${AZURE_SUBSCRIPTION_ID:?Set AZURE_SUBSCRIPTION_ID}"
: "${AZURE_IMAGE_RESOURCE_GROUP:?Set AZURE_IMAGE_RESOURCE_GROUP}"
: "${AZURE_ACR_NAME:?Set AZURE_ACR_NAME}"
: "${AZURE_BUILDER_IDENTITY_ID:?Set AZURE_BUILDER_IDENTITY_ID}"
: "${AZURE_BUILDER_IDENTITY_CLIENT_ID:?Set AZURE_BUILDER_IDENTITY_CLIENT_ID}"
: "${AZURE_GLOVE_SOURCE_URL:?Set AZURE_GLOVE_SOURCE_URL (no SAS token)}"

packer init kumo-singlehost-azure.pkr.hcl
packer build \
    -var "subscription_id=${AZURE_SUBSCRIPTION_ID}" \
    -var "resource_group_name=${AZURE_IMAGE_RESOURCE_GROUP}" \
    -var "acr_name=${AZURE_ACR_NAME}" \
    -var "builder_identity_id=${AZURE_BUILDER_IDENTITY_ID}" \
    -var "builder_identity_client_id=${AZURE_BUILDER_IDENTITY_CLIENT_ID}" \
    -var "glove_source_url=${AZURE_GLOVE_SOURCE_URL}" \
    -var "location=${AZURE_LOCATION:-westus2}" \
    -var "azure_environment=${AZURE_ENVIRONMENT:-public}" \
    -var "kumo_image_tag=$1" \
    kumo-singlehost-azure.pkr.hcl
