#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

VERSION=$1
DOCKER_REGISTRY_ID="926922431314"
AWS_REGISTRY="${DOCKER_REGISTRY_ID}.dkr.ecr.us-west-2.amazonaws.com"
SNOWFLAKE_REGISTRY="lfwgwbp-kumo-snowservices.registry.snowflakecomputing.com"
SNOWFLAKE_REGISTRY_PATH="${SNOWFLAKE_REGISTRY}/kumo_app_db/image_repo/kumo_repo"
MISSING_IMAGES=()

# Login to registry

check_image() {
    local repo_name=$1

    if [[ "${REGISTRY}" == "${SNOWFLAKE_REGISTRY}" ]]; then
        echo "Checking ${repo_name} in ${REGISTRY}"
        if ! docker buildx imagetools inspect "${SNOWFLAKE_REGISTRY_PATH}/${repo_name}:${VERSION}" > /dev/null 2>&1; then
            echo "Error: Image ${repo_name}:${VERSION} not found in ${REGISTRY}"
            MISSING_IMAGES+=("${repo_name}")
        fi
    else
        repo_name="kumo/${repo_name}"
        echo "Checking ${repo_name} in ${REGISTRY}"
        if ! aws ecr describe-images --region "us-west-2" --registry-id "${DOCKER_REGISTRY_ID}" \
            --repository-name "${repo_name}" --image-ids imageTag="${VERSION}" &> /dev/null; then
            echo "Error: Image ${repo_name}:${VERSION} not found in ${REGISTRY}"
            MISSING_IMAGES+=("${repo_name}")
        fi
    fi
}

echo "Checking all required images"

IMAGES=("kumo-ui" "kumo-backend-cpu" "kumo-backend-gpu")

# Check all images
for image in "${IMAGES[@]}"; do
    check_image "${image}"
done

# Report results
if [ ${#MISSING_IMAGES[@]} -ne 0 ]; then
    echo -e "\nMissing images/artifacts:"
    printf '%s\n' "${MISSING_IMAGES[@]}"
    exit 1
fi

echo -e "\nAll required images and artifacts exist"
exit 0
