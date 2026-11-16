#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-static-gcp-identity.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

BIN_DIR="${TMP_DIR}/bin"
KUBECTL_LOG="${TMP_DIR}/kubectl.log"
mkdir -p "$BIN_DIR"

cat >"${BIN_DIR}/kubectl" <<'FAKE_KUBECTL'
#!/usr/bin/env bash
set -u

printf '%s\n' "$*" >>"${FAKE_KUBECTL_LOG}"

if [ "${1:-}" = version ]; then
  printf '%s\n' '{"serverVersion":{"major":"1","minor":"28"}}'
  exit 0
fi

if [ "${1:-}" = get ] && [ "${2:-}" = secret ] &&
  [ "${3:-}" = gcp-application-credentials ]; then
  case "${FAKE_GCP_SECRET_CASE:-configured}" in
    configured) printf '%s\n' credentials.json ;;
    wrong_key) printf '%s\n' other.json ;;
    query_error) exit 42 ;;
    *) exit 43 ;;
  esac
  exit 0
fi

exit 1
FAKE_KUBECTL
chmod +x "${BIN_DIR}/kubectl"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

strip_ansi() {
  sed $'s/\033\\[[0-9;]*m//g'
}

output=$(FAKE_KUBECTL_LOG="$KUBECTL_LOG" \
  PATH="${BIN_DIR}:${PATH}" \
  bash "${ROOT}/conformance/check.sh" \
    --chart "$ROOT" \
    -f "${ROOT}/test/manifests/values-gcp-static.yaml" \
    -n static-gcp prereq 2>&1) || :
output=$(printf '%s\n' "$output" | strip_ansi)

printf '%s\n' "$output" |
  grep -Eq '^[[:space:]]*PASS[[:space:]]+STORAGE_IDENTITY([[:space:]]|$)' ||
  fail "static GCP ADC did not satisfy STORAGE_IDENTITY"
[[ "$output" == *"static GCP ADC Secret 'gcp-application-credentials'"* ]] ||
  fail "static GCP identity result omitted the configured Secret"
[[ "$output" != *"GCP workload identity has missing or invalid fields"* ]] ||
  fail "static GCP ADC still depended on GKE workload identity"

calls=$(<"$KUBECTL_LOG")
[[ "$calls" == *"get secret gcp-application-credentials -n static-gcp"* ]] ||
  fail "conformance did not query the configured GCP ADC Secret"
[[ "$calls" == *"go-template="* ]] ||
  fail "conformance did not request key names without Secret values"

missing_key_output=$(FAKE_GCP_SECRET_CASE=wrong_key \
  FAKE_KUBECTL_LOG="$KUBECTL_LOG" \
  PATH="${BIN_DIR}:${PATH}" \
  bash "${ROOT}/conformance/check.sh" \
    --chart "$ROOT" \
    -f "${ROOT}/test/manifests/values-gcp-static.yaml" \
    -n static-gcp prereq 2>&1) || :
missing_key_output=$(printf '%s\n' "$missing_key_output" | strip_ansi)

printf '%s\n' "$missing_key_output" |
  grep -Eq '^[[:space:]]*FAIL[[:space:]]+STORAGE_IDENTITY([[:space:]]|$)' ||
  fail "missing configured GCP ADC key did not fail STORAGE_IDENTITY"
[[ "$missing_key_output" == *"does not contain configured key 'credentials.json'"* ]] ||
  fail "missing-key result omitted the configured key name"
[[ "$missing_key_output" != *"GCP workload identity has missing or invalid fields"* ]] ||
  fail "missing static GCP ADC key incorrectly fell back to GKE workload identity"

query_error_output=$(FAKE_GCP_SECRET_CASE=query_error \
  FAKE_KUBECTL_LOG="$KUBECTL_LOG" \
  PATH="${BIN_DIR}:${PATH}" \
  bash "${ROOT}/conformance/check.sh" \
    --chart "$ROOT" \
    -f "${ROOT}/test/manifests/values-gcp-static.yaml" \
    -n static-gcp prereq 2>&1) || :
query_error_output=$(printf '%s\n' "$query_error_output" | strip_ansi)

printf '%s\n' "$query_error_output" |
  grep -Eq '^[[:space:]]*FAIL[[:space:]]+STORAGE_IDENTITY([[:space:]]|$)' ||
  fail "GCP ADC Secret query error did not fail STORAGE_IDENTITY"
[[ "$query_error_output" == *"could not query static GCP ADC Secret"* ]] ||
  fail "query-error result did not distinguish a failed Secret query"

echo "Static GCP identity conformance contract: passed"
