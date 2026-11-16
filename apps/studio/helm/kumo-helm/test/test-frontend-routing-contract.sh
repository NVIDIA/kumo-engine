#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELM_BIN=${HELM:-helm}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

command -v "$HELM_BIN" >/dev/null 2>&1 || fail "Helm is required"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-frontend-routing.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

base_fixture="${ROOT}/test/manifests/values-aws-default.yaml"
chart_args=(
  -f "$base_fixture"
  --set-string 'workloads.rest.extraEnv.FORWARDED_ALLOW_IPS=198.51.100.0/24'
  --set-string 'workloads.rest.extraEnv.DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED=true'
)
plaintext_args=(
  "${chart_args[@]}"
  --set-string temporal.clientMtlsSecret=
)

"$HELM_BIN" lint "$ROOT" "${chart_args[@]}" >/dev/null
"$HELM_BIN" template kumo "$ROOT" "${chart_args[@]}" \
  >"${TMP_DIR}/frontend.yaml"
"$HELM_BIN" template kumo "$ROOT" "${chart_args[@]}" \
  --show-only templates/configmap-ui-config.yaml \
  >"${TMP_DIR}/frontend-ui-config.yaml"
for component in ui rest api; do
  "$HELM_BIN" template kumo "$ROOT" "${chart_args[@]}" \
    --show-only "templates/deployment-kumo-${component}.yaml" \
    >"${TMP_DIR}/frontend-${component}.yaml"
done
"$HELM_BIN" template kumo "$ROOT" "${plaintext_args[@]}" \
  --show-only templates/deployment-kumo-ui.yaml \
  >"${TMP_DIR}/plaintext-ui.yaml"
"$HELM_BIN" template kumo "$ROOT" "${plaintext_args[@]}" \
  >"${TMP_DIR}/plaintext.yaml"
"$HELM_BIN" template kumo "$ROOT" "${plaintext_args[@]}" \
  --set-string exposure.host= \
  --show-only templates/deployment-kumo-ui.yaml \
  >"${TMP_DIR}/local-ui.yaml"
"$HELM_BIN" template kumo "$ROOT" "${plaintext_args[@]}" \
  --set-string exposure.host= \
  --show-only templates/configmap-ui-config.yaml \
  >"${TMP_DIR}/local-ui-config.yaml"
"$HELM_BIN" template kumo "$ROOT" "${plaintext_args[@]}" \
  --set-string exposure.host= >"${TMP_DIR}/local.yaml"
main_container() {
  awk '
    /^      containers:$/ { capture = 1 }
    capture && /^      volumes:$/ { exit }
    capture { print }
  ' "$1"
}

main_container "${TMP_DIR}/frontend-ui.yaml" \
  >"${TMP_DIR}/frontend-main.yaml"
main_container "${TMP_DIR}/plaintext-ui.yaml" \
  >"${TMP_DIR}/plaintext-main.yaml"
main_container "${TMP_DIR}/local-ui.yaml" \
  >"${TMP_DIR}/local-main.yaml"

if grep -Eq '^[[:space:]]+(command|args):' "${TMP_DIR}/frontend-main.yaml"; then
  fail "the fixed frontend bypasses the image-owned entrypoint"
fi
if grep -Fq 'initContainers:' "${TMP_DIR}/frontend-ui.yaml"; then
  fail "the frontend must not require an init container"
fi
[ "$(grep -c '^[[:space:]]*image:' "${TMP_DIR}/frontend-ui.yaml")" -eq 1 ] ||
  fail "the frontend must render exactly one container image"
grep -Fq '          name: ui-config' "${TMP_DIR}/frontend-ui.yaml" ||
  fail "the frontend does not mount the Helm-rendered UI config"
grep -Fq '          readOnly: true' "${TMP_DIR}/frontend-ui.yaml" ||
  fail "the frontend UI config mount must be read-only"
if grep -Fq envsubst "${TMP_DIR}/frontend.yaml"; then
  fail "the chart retained an envsubst dependency"
fi

assert_config() {
  local manifest="$1" expected="$2"
  grep -Fq "$expected" "$manifest" ||
    fail "${manifest} does not contain expected UI config: ${expected}"
}

assert_config "${TMP_DIR}/frontend-ui-config.yaml" \
  'REACT_APP_BASE_URL_DEV: "https://test.example.com/internal",'
assert_config "${TMP_DIR}/frontend-ui-config.yaml" \
  'REACT_APP_BASE_URL_PROD: "https://test.example.com/internal",'
assert_config "${TMP_DIR}/frontend-ui-config.yaml" \
  'KUMO_AUTH_BACKEND: "auth0",'
assert_config "${TMP_DIR}/frontend-ui-config.yaml" 'AUTH_DISABLED: true,'
assert_config "${TMP_DIR}/local-ui-config.yaml" \
  'REACT_APP_BASE_URL_DEV: "/internal",'
if grep -Fq '$REACT_APP_' "${TMP_DIR}/frontend-ui-config.yaml"; then
  fail "the frontend ConfigMap retained runtime substitution placeholders"
fi

env_value() {
  local manifest="$1" wanted="$2"
  awk -v wanted="$wanted" '
    $1 == "-" && $2 == "name:" && $3 == wanted {
      found++
      if (getline <= 0 || $1 != "value:") exit 3
      sub(/^[[:space:]]*value:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      result = $0
    }
    END {
      if (found != 1) exit 2
      print result
    }
  ' "$manifest"
}

assert_env() {
  local manifest="$1" name="$2" expected="$3" actual
  actual=$(env_value "$manifest" "$name") ||
    fail "${manifest} must set ${name} exactly once"
  [ "$actual" = "$expected" ] ||
    fail "${name}=${actual}; expected ${expected}"
}

assert_env "${TMP_DIR}/frontend-main.yaml" KUMO_UI_ROUTING_MODE frontend
assert_env "${TMP_DIR}/frontend-main.yaml" KUMO_PRIVATE_UPSTREAM kumo-rest:80
assert_env "${TMP_DIR}/frontend-main.yaml" KUMO_PUBLIC_UPSTREAM kumo-api:80
assert_env "${TMP_DIR}/frontend-main.yaml" KUMO_EXTERNAL_HOST test.example.com
assert_env "${TMP_DIR}/frontend-main.yaml" KUMO_EXTERNAL_SCHEME https
assert_env "${TMP_DIR}/plaintext-main.yaml" KUMO_EXTERNAL_SCHEME https
assert_env "${TMP_DIR}/local-main.yaml" KUMO_EXTERNAL_SCHEME http

assert_count() {
  local manifest="$1" pattern="$2" expected="$3" count
  count=$(grep -Fc -- "$pattern" "$manifest" || true)
  [ "$count" -eq "$expected" ] ||
    fail "${manifest} contains ${count} copies of ${pattern}; expected ${expected}"
}

assert_count "${TMP_DIR}/frontend-ui.yaml" \
  '        app.kubernetes.io/instance: "kumo"' 1
assert_count "${TMP_DIR}/frontend-ui.yaml" \
  '        app.kubernetes.io/component: "frontend"' 1
for component in rest api; do
  assert_count "${TMP_DIR}/frontend-${component}.yaml" \
    '        app.kubernetes.io/instance: "kumo"' 1
  assert_count "${TMP_DIR}/frontend-${component}.yaml" \
    '        app.kubernetes.io/component: "backend"' 1
done

# Network and mesh policy are the operator's; the chart must render neither.
for kind in NetworkPolicy AuthorizationPolicy PeerAuthentication; do
  if grep -Fq "kind: ${kind}" "${TMP_DIR}/plaintext.yaml"; then
    fail "chart rendered ${kind}; see examples/networkpolicy.yaml instead"
  fi
done

actual_gateway_env=$(
  sed -n 's/^[[:space:]]*- name: \(KUMO_[A-Z_]*\)$/\1/p' \
    "${TMP_DIR}/frontend-main.yaml" | sort
)
expected_gateway_env=$'KUMO_EXTERNAL_HOST\nKUMO_EXTERNAL_SCHEME\nKUMO_PRIVATE_UPSTREAM\nKUMO_PUBLIC_UPSTREAM\nKUMO_UI_ROUTING_MODE'
[ "$actual_gateway_env" = "$expected_gateway_env" ] ||
  fail "frontend gateway environment drifted: ${actual_gateway_env}"

assert_env "${TMP_DIR}/frontend-rest.yaml" FORWARDED_ALLOW_IPS '*'
assert_env "${TMP_DIR}/frontend-api.yaml" FORWARDED_ALLOW_IPS '*'
assert_env "${TMP_DIR}/frontend-rest.yaml" \
  DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED false
if grep -Fq '  FORWARDED_ALLOW_IPS: "198.51.100.0/24"' \
  "${TMP_DIR}/frontend.yaml"; then
  fail "ignored frontend proxy-trust override leaked through the shared ConfigMap"
fi
if grep -Fq '  DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED: "true"' \
  "${TMP_DIR}/frontend.yaml"; then
  fail "ignored frontend limiter override leaked through the shared ConfigMap"
fi
if grep -Fq DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED \
  "${TMP_DIR}/frontend-api.yaml"; then
  fail "kumo-api received an explicit REST-only password-limiter override"
fi
[ "$(grep -c -- '- name: FORWARDED_ALLOW_IPS' "${TMP_DIR}/frontend.yaml")" -eq 2 ] ||
  fail "proxy trust must be explicit only on REST and API"
[ "$(grep -c -- '- name: DIRECT_CONNECT_PASSWORD_LOGIN_IP_RATE_LIMIT_ENABLED' \
  "${TMP_DIR}/frontend.yaml")" -eq 1 ] ||
  fail "the password source-IP limiter must be explicit only on REST"

service_names=$(
  awk '
    $0 == "kind: Service" { service = 1; metadata = 0; next }
    service && $0 == "metadata:" { metadata = 1; next }
    service && metadata && /^  name: / {
      print $2
      service = 0
      metadata = 0
    }
  ' "${TMP_DIR}/frontend.yaml" | sort
)
[ "$service_names" = $'kumo-api\nkumo-rest\nkumo-ui' ] ||
  fail "the chart must retain exactly the three Services: ${service_names}"

bad_service_types=$(
  awk '
    $0 == "kind: Service" { service = 1; next }
    service && /^  type: / && $2 != "ClusterIP" { print $2 }
    service && /^---$/ { service = 0 }
  ' "${TMP_DIR}/frontend.yaml"
)
[ -z "$bad_service_types" ] ||
  fail "the chart rendered non-ClusterIP Services: ${bad_service_types}"
if grep -Eq '^kind: (Ingress|Gateway|HTTPRoute|Route)$' \
  "${TMP_DIR}/frontend.yaml"; then
  fail "the application chart must stop at ClusterIP Services"
fi

auth_args=(
  --set auth.enabled=true
  --set-string auth.method=password
  --set-string auth.password.temporaryPassword.secretName=kumo-login
  --set-string tenantSecretKey.secretName=kumo-secret-key
  --set-string tenantSecretKey.secretKey=credentials_encryption_key
)
"$HELM_BIN" template kumo "$ROOT" -f "$base_fixture" "${auth_args[@]}" \
  --show-only templates/configmap-ui-config.yaml \
  >"${TMP_DIR}/frontend-auth-ui-config.yaml"
assert_config "${TMP_DIR}/frontend-auth-ui-config.yaml" \
  'KUMO_AUTH_BACKEND: "direct_connect",'
assert_config "${TMP_DIR}/frontend-auth-ui-config.yaml" 'AUTH_DISABLED: false,'
"$HELM_BIN" template kumo "$ROOT" -f "$base_fixture" "${auth_args[@]}" \
  --show-only templates/configmap-kumo-config.yaml \
  >"${TMP_DIR}/frontend-auth-config.yaml"
if grep -Fq '  FORWARDED_ALLOW_IPS:' \
  "${TMP_DIR}/frontend-auth-config.yaml"; then
  fail "frontend auth leaked proxy trust through the shared ConfigMap"
fi

failure_id=0
expect_failure() {
  local description="$1" expected="$2"
  shift 2
  failure_id=$((failure_id + 1))
  if "$HELM_BIN" template kumo "$ROOT" -f "$base_fixture" "$@" \
    >"${TMP_DIR}/failure-${failure_id}.out" \
    2>"${TMP_DIR}/failure-${failure_id}.err"; then
    fail "accepted ${description}"
  fi
  grep -Fq "$expected" "${TMP_DIR}/failure-${failure_id}.err" ||
    fail "${description} failed for the wrong reason"
}

expect_failure "the removed routing override" \
  'routing is not configurable' \
  --set-string routing.mode=invalid
expect_failure "an invalid frontend hostname" 'lowercase DNS-1123 hostname' \
  --set-string exposure.host=Not_DNS
expect_failure "frontend auth without a host" \
  'auth.enabled=true requires a nonempty exposure.host' \
  "${auth_args[@]}" \
  --set-string 'auth.allowedReturnOrigins[0]=https://return.example' \
  --set-string exposure.host=
expect_failure "a frontend instance-label collision" \
  'global.podLabels["app.kubernetes.io/instance"] is reserved by the chart' \
  --set-string 'global.podLabels.app\.kubernetes\.io/instance=caller'
expect_failure "a frontend component-label collision" \
  'global.podLabels["app.kubernetes.io/component"] is reserved by the chart' \
  --set-string 'global.podLabels.app\.kubernetes\.io/component=caller'

"$HELM_BIN" template kumo "$ROOT" -f "$base_fixture" "${auth_args[@]}" \
  >"${TMP_DIR}/plaintext-auth.yaml"
grep -Fq 'https://test.example.com' "${TMP_DIR}/plaintext-auth.yaml" ||
  fail "plaintext authenticated install did not retain HTTPS external URLs"

echo "Frontend routing render contract: passed"
