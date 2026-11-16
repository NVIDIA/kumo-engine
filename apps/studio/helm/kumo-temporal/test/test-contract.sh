#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELM_BIN="${HELM_BIN:-helm}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kumo-temporal-test.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local file="$1"
  local needle="$2"
  grep -Fq -- "${needle}" "${file}" ||
    fail "${file} does not contain: ${needle}"
}

assert_not_contains() {
  local file="$1"
  local needle="$2"
  if grep -Fq -- "${needle}" "${file}"; then
    fail "${file} unexpectedly contains: ${needle}"
  fi
}

render() {
  local provider="$1"
  "${HELM_BIN}" template temporal "${CHART_DIR}" \
    --namespace temporal \
    --values "${CHART_DIR}/profiles/${provider}.yaml" \
    --values "${CHART_DIR}/test/values-${provider}.yaml"
}

render aws >"${TMP_DIR}/aws.yaml"
render azure >"${TMP_DIR}/azure.yaml"

for provider in aws azure; do
  manifest="${TMP_DIR}/${provider}.yaml"

  assert_not_contains "${manifest}" "kind: Secret"
  assert_not_contains "${manifest}" "name: temporal-admintools"
  assert_not_contains "${manifest}" "name: temporal-web"
  assert_not_contains "${manifest}" "create-default-store"
  assert_not_contains "${manifest}" "create-visibility-store"
  assert_not_contains "${manifest}" "temporalio/ui"
  assert_not_contains "${manifest}" "cassandra"
  assert_not_contains "${manifest}" "elasticsearch"

  assert_contains "${manifest}" 'image: "docker.io/temporalio/server:1.31.2"'
  assert_contains "${manifest}" 'image: "docker.io/temporalio/admin-tools:1.31.2"'
  assert_contains "${manifest}" "name: manage-schema-default-store"
  assert_contains "${manifest}" "name: manage-schema-visibility-store"
  assert_contains "${manifest}" "name: temporal-postgres"
  assert_contains "${manifest}" "key: password"
  assert_contains "${manifest}" "name: temporal-frontend"
  if [[ "${provider}" == aws ]]; then
    assert_not_contains "${manifest}" "name: kumo-temporal-strict-mtls"
  else
    assert_contains "${manifest}" "name: kumo-temporal-strict-mtls"
  fi
  assert_contains "${manifest}" 'value: "true"'
  assert_contains "${manifest}" 'value: "false"'

  host_verification_count=$(grep -Fc "enableHostVerification: true" "${manifest}")
  [[ "${host_verification_count}" -eq 2 ]] ||
    fail "${provider} should enable hostname verification for both stores"
  awk '
    /- name: SQL_TLS$/ {
      getline
      if ($0 !~ /value: "true"$/) exit 1
      tls_count++
    }
    /- name: SQL_TLS_DISABLE_HOST_VERIFICATION$/ {
      getline
      if ($0 !~ /value: "false"$/) exit 1
      verify_count++
    }
    END { exit !(tls_count == 2 && verify_count == 2) }
  ' "${manifest}" ||
    fail "${provider} schema Jobs must use TLS with hostname verification"

  secret_ref_count="$(
    grep -A3 -F "secretKeyRef:" "${manifest}" |
      grep -Fc "name: temporal-postgres"
  )"
  [[ "${secret_ref_count}" -eq 10 ]] ||
    fail "${provider} should have ten server/schema password Secret refs, got ${secret_ref_count}"

  deployment_count="$(
    grep -Ec '^kind: Deployment$' "${manifest}"
  )"
  [[ "${deployment_count}" -eq 4 ]] ||
    fail "${provider} should render four Temporal Deployments, got ${deployment_count}"

  awk '
    BEGIN { RS = "---" }
    /kind: NetworkPolicy/ && /name: kumo-temporal-ingress/ { print }
  ' "${manifest}" >"${TMP_DIR}/${provider}-networkpolicy.yaml"
  assert_not_contains "${TMP_DIR}/${provider}-networkpolicy.yaml" "port: 9090"
done

assert_not_contains "${TMP_DIR}/aws.yaml" "name: kumo-temporal-ingress"
assert_not_contains "${TMP_DIR}/aws.yaml" "kumo.ai/temporal-client: \"true\""
assert_contains "${TMP_DIR}/azure.yaml" "name: kumo-temporal-ingress"
assert_contains "${TMP_DIR}/azure.yaml" "kumo.ai/temporal-client: \"true\""

assert_contains "${TMP_DIR}/aws.yaml" "name: kumo-temporal-aws-rds-ca"
assert_contains "${TMP_DIR}/aws.yaml" "name: kumo-temporal-aws-rds-ca-schema"
assert_contains "${TMP_DIR}/aws.yaml" '"helm.sh/hook-weight": "-5"'
assert_contains "${TMP_DIR}/aws.yaml" "value: /etc/temporal/provider-certs/aws-rds-roots.pem"
assert_contains "${TMP_DIR}/aws.yaml" "name: aws-rds-ca"

ca_sha=$(openssl dgst -sha256 \
  "${CHART_DIR}/files/aws-rds-roots.pem" | awk '{print $NF}')
annotation_count=$(grep -Fc \
  "kumo.ai/aws-rds-ca-sha256: ${ca_sha}" \
  "${CHART_DIR}/profiles/aws.yaml")
all_annotation_count=$(grep -Fc \
  "kumo.ai/aws-rds-ca-sha256:" \
  "${CHART_DIR}/profiles/aws.yaml")
[[ "${annotation_count}" -eq 2 && "${all_annotation_count}" -eq 2 ]] ||
  fail "AWS server and schema CA annotations must match the packaged bundle"

assert_not_contains "${TMP_DIR}/azure.yaml" "aws-rds-roots.pem"
assert_not_contains "${TMP_DIR}/azure.yaml" "SQL_TLS_CA_FILE"
assert_not_contains "${TMP_DIR}/azure.yaml" "port: 15008"

if "${HELM_BIN}" template invalid "${CHART_DIR}" \
  --namespace temporal \
  --values "${CHART_DIR}/profiles/aws.yaml" \
  --values "${CHART_DIR}/test/values-aws.yaml" \
  --set temporal.server.config.persistence.datastores.default.sql.createDatabase=true \
  >"${TMP_DIR}/invalid.yaml" 2>"${TMP_DIR}/invalid.log"; then
  fail "createDatabase=true unexpectedly rendered"
fi
assert_contains "${TMP_DIR}/invalid.log" "createDatabase must remain false"

if "${HELM_BIN}" template invalid "${CHART_DIR}" \
  --namespace temporal \
  --values "${CHART_DIR}/profiles/aws.yaml" \
  --values "${CHART_DIR}/test/values-aws.yaml" \
  --set temporal.server.config.persistence.datastores.visibility.sql.existingSecret=other-secret \
  >"${TMP_DIR}/invalid.yaml" 2>"${TMP_DIR}/invalid.log"; then
  fail "mismatched Secret unexpectedly rendered"
fi
assert_contains "${TMP_DIR}/invalid.log" "must use the same existingSecret"

if "${HELM_BIN}" template invalid "${CHART_DIR}" \
  --namespace temporal \
  --values "${CHART_DIR}/profiles/aws.yaml" \
  --values "${CHART_DIR}/test/values-aws.yaml" \
  --set-string temporal.server.config.persistence.datastores.default.sql.password=unsafe \
  >"${TMP_DIR}/invalid.yaml" 2>"${TMP_DIR}/invalid.log"; then
  fail "plaintext password unexpectedly rendered"
fi
assert_contains "${TMP_DIR}/invalid.log" "password and passwordCommand values are forbidden"

if "${HELM_BIN}" template invalid "${CHART_DIR}" \
  --namespace temporal \
  --values "${CHART_DIR}/profiles/aws.yaml" \
  --values "${CHART_DIR}/test/values-aws.yaml" \
  --set-string temporal.server.config.persistence.datastores.default.sql.passwordCommand=unsafe \
  >"${TMP_DIR}/invalid.yaml" 2>"${TMP_DIR}/invalid.log"; then
  fail "passwordCommand unexpectedly rendered"
fi
assert_contains "${TMP_DIR}/invalid.log" "password and passwordCommand values are forbidden"

"${HELM_BIN}" template mirror "${CHART_DIR}" \
  --namespace temporal \
  --values "${CHART_DIR}/profiles/azure.yaml" \
  --values "${CHART_DIR}/test/values-azure.yaml" \
  --set temporal.server.image.repository=registry.example.com/approved/temporal-server \
  --set temporal.admintools.image.repository=registry.example.com/approved/temporal-admin-tools \
  >"${TMP_DIR}/mirror.yaml"
assert_contains "${TMP_DIR}/mirror.yaml" 'image: "registry.example.com/approved/temporal-server:1.31.2"'
assert_contains "${TMP_DIR}/mirror.yaml" 'image: "registry.example.com/approved/temporal-admin-tools:1.31.2"'

echo "kumo-temporal contract tests passed"
