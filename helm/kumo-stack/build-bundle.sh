#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

STACK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HELM_ROOT=$(cd "${STACK_ROOT}/.." && pwd)
REPO_ROOT=$(cd "${HELM_ROOT}/.." && pwd)
OUTPUT_DIR="${1:-${PWD}/dist}"
BUNDLE_ROOT="${OUTPUT_DIR}/kumo-stack"

[[ ! -e "${BUNDLE_ROOT}" ]] || {
  printf 'error: bundle destination already exists: %s\n' "${BUNDLE_ROOT}" >&2
  exit 1
}

[[ -f "${HELM_ROOT}/kumo-temporal/files/aws-rds-roots.pem" ]] || {
  printf '%s\n' \
    'error: missing staged AWS RDS roots; run kumo-stack/release/prepare-aws-rds-ca.sh first' >&2
  exit 1
}

staging=$(mktemp -d "${TMPDIR:-/tmp}/kumo-stack-source.XXXXXX")
cleanup() {
  rm -rf "${staging}"
}
trap cleanup EXIT

source_root="${staging}/kumo-stack"
mkdir -p "${source_root}/charts" "${OUTPUT_DIR}"

install -m 0644 "${STACK_ROOT}/Chart.yaml" "${source_root}/Chart.yaml"
install -m 0644 "${REPO_ROOT}/LICENSE" "${source_root}/LICENSE"
install -m 0644 "${STACK_ROOT}/README.md" "${source_root}/README.md"
install -m 0644 "${STACK_ROOT}/THIRD-PARTY.txt" "${source_root}/THIRD-PARTY.txt"
install -m 0644 "${STACK_ROOT}/config.yaml" "${source_root}/config.yaml"
install -m 0755 "${STACK_ROOT}/install.sh" "${source_root}/install.sh"
for directory in bin docs files providers renderer sql stages; do
  cp -R "${STACK_ROOT}/${directory}" "${source_root}/${directory}"
done

for component in \
  kumo-helm kumo-edge kumo-temporal kumo-platform-eks kumo-platform-aks \
  kumo-platform-aro; do
  cp -R "${HELM_ROOT}/${component}" "${source_root}/charts/${component}"
  rm -rf "${source_root}/charts/${component}/test"
done

[[ -f "${source_root}/charts/kumo-temporal/charts/temporal/Chart.yaml" ]] || {
  printf 'error: vendored Temporal chart is missing\n' >&2
  exit 1
}
[[ -s "${source_root}/charts/kumo-temporal/charts/temporal/LICENSE" ]] || {
  printf 'error: vendored Temporal license is missing\n' >&2
  exit 1
}

mv "${source_root}" "${BUNDLE_ROOT}"
printf '%s\n' "${BUNDLE_ROOT}"
