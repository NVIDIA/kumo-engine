#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

SMOKE_IMAGE=${KUMO_GATEWAY_SMOKE_IMAGE:-docker.io/library/nginx:1.29.7-alpine-slim@sha256:0848ca84c476868cbeb6a5c2c009a98821b8540f96c44b1ba06820db50262e35}

if [ -n "${CONTAINER_RUNTIME:-}" ]; then
  RUNTIME_BIN=$CONTAINER_RUNTIME
elif command -v docker >/dev/null 2>&1; then
  RUNTIME_BIN=$(command -v docker)
elif command -v nerdctl.lima >/dev/null 2>&1; then
  RUNTIME_BIN=$(command -v nerdctl.lima)
elif command -v nerdctl >/dev/null 2>&1; then
  RUNTIME_BIN=$(command -v nerdctl)
elif command -v podman >/dev/null 2>&1; then
  RUNTIME_BIN=$(command -v podman)
else
  fail "docker, nerdctl, or podman is required"
fi

runtime() {
  env -u GITHUB_PAT_TOKEN -u GH_TOKEN -u GITHUB_TOKEN "$RUNTIME_BIN" "$@"
}

GATEWAY_IMAGE=${KUMO_UI_IMAGE:-$SMOKE_IMAGE}
SOURCE_MODE=false
SOURCE_MOUNTS=()
if [ -z "${KUMO_UI_IMAGE:-}" ]; then
  SOURCE_MODE=true
  for artifact in \
    docker/nginx.frontend.conf.template \
    docker/render-nginx-ui.sh \
    docker/docker-entrypoint-ui.sh \
    docker/nginx.ui.conf; do
    [ -f "${ROOT}/${artifact}" ] || fail "missing source artifact: $artifact"
  done
  SOURCE_MOUNTS=(
    -v "${ROOT}/docker/nginx.frontend.conf.template:/opt/kumo-ui/nginx.frontend.conf.template:ro"
    -v "${ROOT}/docker/render-nginx-ui.sh:/opt/kumo-ui/bin/render-nginx:ro"
    -v "${ROOT}/docker/docker-entrypoint-ui.sh:/docker-entrypoint-ui.sh:ro"
    -v "${ROOT}/docker/nginx.ui.conf:/etc/nginx/conf.d/default.conf:ro"
  )
else
  runtime run --rm --entrypoint /bin/sh "$GATEWAY_IMAGE" -c '
    test -r /opt/kumo-ui/nginx.frontend.conf.template
    test -x /opt/kumo-ui/bin/render-nginx
    test -x /docker-entrypoint-ui.sh
    test -r /usr/share/nginx/html/index.html
  ' || fail "KUMO_UI_IMAGE is not a complete built Kumo UI image"
fi

gateway_run() {
  if [ "$SOURCE_MODE" = true ]; then
    runtime run "${SOURCE_MOUNTS[@]}" "$@"
  else
    runtime run "$@"
  fi
}

smoke_wget() {
  local target=$1 url=$2
  runtime run --rm --network "container:${target}" \
    --entrypoint wget "$SMOKE_IMAGE" -qO- "$url"
}

smoke_nc() {
  local timeout=$1
  runtime exec -i "$PROBE_CONTAINER" \
    nc -w "$timeout" 127.0.0.1 3000
}

TMP_DIR=$(mktemp -d "${ROOT}/.gateway-smoke.XXXXXX")
NETWORK="kumo-gateway-smoke-$$"
PRIVATE_CONTAINER="kumo-smoke-private-$$"
PUBLIC_CONTAINER="kumo-smoke-public-$$"
GATEWAY_CONTAINER="kumo-smoke-ui-$$"
PROBE_CONTAINER="kumo-smoke-probe-$$"
EXTERNAL_CONTAINER="kumo-smoke-external-$$"
RESPONSE_FILE=""
response_number=0

cleanup() {
  runtime rm -f "$PROBE_CONTAINER" "$EXTERNAL_CONTAINER" "$GATEWAY_CONTAINER" \
    "$PUBLIC_CONTAINER" "$PRIVATE_CONTAINER" >/dev/null 2>&1 || true
  runtime network rm "$NETWORK" >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "${TMP_DIR}/config"
printf '%s\n' 'window.__GATEWAY_SMOKE__ = "frontend";' \
  >"${TMP_DIR}/config/env.js"
chmod 755 "$TMP_DIR" "${TMP_DIR}/config"
chmod 644 "${TMP_DIR}/config/env.js"

cat >"${TMP_DIR}/backend.conf" <<'NGINX'
pid /tmp/nginx.pid;
error_log stderr notice;
events {}
http {
  access_log off;
  client_max_body_size 3m;
  server {
    listen 8080;
    default_type text/plain;
    location = / {
      return 307 redoc;
    }
    location = /docs {
      return 200 '<html>swagger-ui ./openapi.json</html>';
    }
    location = /redoc {
      return 200 '<html>ReDoc ./openapi.json</html>';
    }
    location = /openapi.json {
      default_type application/json;
      return 200 '{"openapi":"3.1.0"}';
    }
    location = /absolute-redirect {
      return 307 https://$http_host/v1/redirected?source=api;
    }
    location = /backend-not-found {
      default_type application/problem+json;
      return 404 '{"backend":"not_found"}';
    }
    location / {
      return 200 "backend=$hostname|uri=$request_uri|host=$http_host|proto=$http_x_forwarded_proto|xff=$http_x_forwarded_for|real_ip=$http_x_real_ip|request_id=$http_x_request_id|forwarded=$http_forwarded|xf_host=$http_x_forwarded_host|connection=$http_connection|upgrade=$http_upgrade|origin=$http_origin|cookie=$http_cookie|authorization=$http_authorization|traceparent=$http_traceparent";
    }
  }
}
NGINX

runtime network create "$NETWORK" >/dev/null
for fixture in "$PRIVATE_CONTAINER" "$PUBLIC_CONTAINER"; do
  fixture_hostname=private-backend
  [ "$fixture" = "$PUBLIC_CONTAINER" ] && fixture_hostname=public-backend
  runtime run -d --name "$fixture" --hostname "$fixture_hostname" \
    --network "$NETWORK" \
    -v "${TMP_DIR}/backend.conf:/tmp/backend.conf:ro" \
    --entrypoint nginx "$SMOKE_IMAGE" \
    -c /tmp/backend.conf -g "daemon off;" >/dev/null
  ready=false
  for _ in $(seq 1 50); do
    if smoke_wget "$fixture" \
      http://127.0.0.1:8080/ready >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 0.1
  done
  [ "$ready" = true ] || fail "backend fixture did not become ready: $fixture"
done

start_external() {
  local command=()
  if [ "$SOURCE_MODE" = true ]; then
    command=(--entrypoint /bin/sh "$GATEWAY_IMAGE" \
      /docker-entrypoint-ui.sh nginx -g "daemon off;")
  else
    command=("$GATEWAY_IMAGE")
  fi

  gateway_run -d --name "$EXTERNAL_CONTAINER" --network "$NETWORK" \
    -v "${TMP_DIR}/config:/tmp/config:ro" \
    "${command[@]}" >/dev/null
  for _ in $(seq 1 50); do
    if smoke_wget "$EXTERNAL_CONTAINER" \
      http://127.0.0.1:3000/ >/dev/null 2>&1; then
      if runtime exec "$EXTERNAL_CONTAINER" grep -Fq __GATEWAY_SMOKE__ \
        /usr/share/nginx/html/env-config.js 2>/dev/null; then
        fail "default external mode consumed frontend-only /tmp/config/env.js"
      fi
      runtime rm -f "$EXTERNAL_CONTAINER" >/dev/null
      return
    fi
    sleep 0.1
  done
  runtime logs "$EXTERNAL_CONTAINER" >&2 || true
  fail "default external mode did not start"
}

expect_startup_failure() {
  local public_upstream=$1 host=$2 scheme=$3 description=$4 output status

  set +e
  output=$(gateway_run --rm --network "$NETWORK" \
    -e KUMO_UI_ROUTING_MODE=frontend \
    -e KUMO_PRIVATE_UPSTREAM="${PRIVATE_CONTAINER}:8080" \
    -e KUMO_PUBLIC_UPSTREAM="$public_upstream" \
    -e KUMO_EXTERNAL_HOST="$host" \
    -e KUMO_EXTERNAL_SCHEME="$scheme" \
    --entrypoint /bin/sh "$GATEWAY_IMAGE" -c '
      /bin/sh /docker-entrypoint-ui.sh
      status=$?
      test "$status" -ne 0 || exit 90
      test ! -e /tmp/kumo-ui/ready || exit 91
      echo NO_READY_AFTER_FAILED_STARTUP
      exit "$status"
    ' 2>&1)
  status=$?
  set -e

  [ "$status" -ne 0 ] || fail "renderer accepted $description"
  [[ "$output" == *"frontend renderer:"* ]] ||
    fail "renderer rejected $description without a bounded error: $output"
  [[ "$output" == *"NO_READY_AFTER_FAILED_STARTUP"* ]] ||
    fail "failed startup published readiness for $description: $output"
}

start_probe() {
  runtime rm -f "$PROBE_CONTAINER" >/dev/null 2>&1 || true
  runtime run -d --name "$PROBE_CONTAINER" \
    --network "container:${GATEWAY_CONTAINER}" \
    --entrypoint tail "$SMOKE_IMAGE" -f /dev/null >/dev/null

  for _ in $(seq 1 20); do
    if runtime exec "$PROBE_CONTAINER" true >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
  runtime logs --tail 100 "$PROBE_CONTAINER" >&2 || true
  fail "frontend gateway probe did not become ready"
}

start_gateway() {
  local host=$1 scheme=$2 command=()
  runtime rm -f "$PROBE_CONTAINER" >/dev/null 2>&1 || true
  runtime rm -f "$GATEWAY_CONTAINER" >/dev/null 2>&1 || true
  if [ "$SOURCE_MODE" = true ]; then
    command=(--entrypoint /bin/sh "$GATEWAY_IMAGE" \
      /docker-entrypoint-ui.sh nginx -g "daemon off;")
  else
    command=("$GATEWAY_IMAGE")
  fi

  gateway_run -d --name "$GATEWAY_CONTAINER" --network "$NETWORK" \
    -e KUMO_UI_ROUTING_MODE=frontend \
    -e KUMO_PRIVATE_UPSTREAM="${PRIVATE_CONTAINER}:8080" \
    -e KUMO_PUBLIC_UPSTREAM="${PUBLIC_CONTAINER}:8080" \
    -e KUMO_EXTERNAL_HOST="$host" \
    -e KUMO_EXTERNAL_SCHEME="$scheme" \
    -v "${TMP_DIR}/config:/tmp/config:ro" \
    "${command[@]}" >/dev/null

  for _ in $(seq 1 50); do
    if smoke_wget "$GATEWAY_CONTAINER" \
      http://127.0.0.1:3000/-/ready 2>/dev/null |
      grep -Fq '"status":"ready"'; then
      runtime exec "$GATEWAY_CONTAINER" \
        nginx -t -c /tmp/kumo-ui/nginx.conf >/dev/null 2>&1
      start_probe
      return
    fi
    sleep 0.1
  done
  runtime logs "$GATEWAY_CONTAINER" >&2 || true
  fail "frontend gateway did not become ready"
}

dump_transport_diagnostics() {
  local container
  echo "Frontend gateway transport diagnostics:" >&2
  for container in "$PROBE_CONTAINER" "$GATEWAY_CONTAINER" \
    "$PUBLIC_CONTAINER" "$PRIVATE_CONTAINER"; do
    echo "container=$container" >&2
    runtime inspect --format \
      'status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}}' \
      "$container" >&2 || true
    runtime logs --tail 100 "$container" >&2 || true
  done
}

emit_request() {
  local method=$1 target=$2 inbound_host=$3
  local producer_status=0
  {
    printf '%s %s HTTP/1.1\r\n' "$method" "$target"
    printf 'Host: %s\r\n' "$inbound_host"
    printf 'Connection: close, Upgrade\r\nUpgrade: websocket\r\n'
    printf 'Forwarded: for=198.51.100.10;proto=ftp;host=hostile.invalid\r\n'
    printf 'X-Forwarded-Host: hostile.invalid\r\nX-Forwarded-Proto: ftp\r\n'
    printf 'X-Forwarded-For: 198.51.100.10, 203.0.113.8\r\n'
    printf 'X-Real-IP: 198.51.100.10\r\nX-Request-ID: hostile-request-id\r\n'
    printf 'Origin: https://browser.example\r\nCookie: session=preserved\r\n'
    printf 'Authorization: Bearer-preserved\r\n'
    printf 'Traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01\r\n'
    printf 'Content-Length: 0\r\n\r\n'
  } || producer_status=$?
  sleep 0.5
  return "$producer_status"
}

emit_request_with_body() {
  local method=$1 target=$2 body_file=$3 body_bytes=$4
  local producer_status=0
  {
    printf '%s %s HTTP/1.1\r\n' "$method" "$target"
    printf 'Host: hostile.invalid:9443\r\n'
    printf 'Connection: close\r\n'
    printf 'Content-Type: application/octet-stream\r\n'
    printf 'Content-Length: %s\r\n\r\n' "$body_bytes"
    cat "$body_file"
  } || producer_status=$?
  sleep 0.5
  return "$producer_status"
}

capture_response() {
  local raw=$1 timeout=$2 description=$3 producer=$4
  shift 4
  local attempt byte_count producer_status transport_status
  local -a statuses

  for attempt in 1 2; do
    : >"$raw"
    set +e
    "$producer" "$@" | smoke_nc "$timeout" >"$raw"
    statuses=("${PIPESTATUS[@]}")
    set -e
    producer_status=${statuses[0]}
    transport_status=${statuses[1]}
    byte_count=$(wc -c <"$raw" | tr -d ' ')
    if [ "$byte_count" -gt 0 ]; then
      return
    fi
    if [ "$attempt" -eq 1 ]; then
      echo "WARN: empty response; retrying once: $description" \
        "(bytes=$byte_count producer=$producer_status transport=$transport_status)" \
        >&2
    fi
  done

  dump_transport_diagnostics
  fail "request returned no response: $description" \
    "(bytes=$byte_count producer=$producer_status transport=$transport_status)"
}

request() {
  local method=$1 target=$2 inbound_host=${3:-hostile.invalid:9443}
  local raw="${TMP_DIR}/response-$((++response_number)).raw"
  local normalized="${raw}.txt"

  capture_response "$raw" 5 "$method $target" \
    emit_request "$method" "$target" "$inbound_host"
  tr -d '\r' <"$raw" >"$normalized"
  RESPONSE_FILE=$normalized
}

request_with_body() {
  local method=$1 target=$2 body_bytes=$3
  local raw="${TMP_DIR}/response-$((++response_number)).raw"
  local normalized="${raw}.txt"
  local body_file="${raw}.body"

  dd if=/dev/zero of="$body_file" bs=1024 count=$((body_bytes / 1024)) \
    2>/dev/null
  [ "$(wc -c <"$body_file" | tr -d ' ')" = "$body_bytes" ] ||
    fail "failed to create request body with $body_bytes bytes"

  capture_response "$raw" 10 "$method $target" \
    emit_request_with_body "$method" "$target" "$body_file" "$body_bytes"
  tr -d '\r' <"$raw" >"$normalized"
  RESPONSE_FILE=$normalized
}

response_status() { awk 'NR == 1 { print $2; exit }' "$RESPONSE_FILE"; }
response_header() {
  awk -v wanted="$1" 'BEGIN { wanted=tolower(wanted) } /^$/ { exit }
    tolower($0) ~ "^" wanted ":" { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }' \
    "$RESPONSE_FILE"
}
response_body() { awk 'body { print } /^$/ { body=1 }' "$RESPONSE_FILE"; }
body_field() {
  awk -F'|' -v wanted="$2" '{ for (i=1; i<=NF; i++) {
    if (index($i, wanted "=") == 1) { print substr($i, length(wanted)+2); exit }
  }}' <<<"$1"
}

assert_status() {
  local expected=$1 description=$2 actual
  actual=$(response_status)
  [ "$actual" = "$expected" ] ||
    fail "$description returned ${actual:-no status}; expected $expected"
}

assert_backend() {
  local target=$1 expected_backend=$2 expected_uri=$3
  local expected_host=$4 expected_scheme=${5:-https}
  local inbound_host=${6:-hostile.invalid:9443}
  local body
  request GET "$target" "$inbound_host"
  assert_status 200 "$target"
  body=$(response_body)
  [ "$(body_field "$body" backend)" = "$expected_backend" ] ||
    fail "$target selected the wrong backend: $body"
  [ "$(body_field "$body" uri)" = "$expected_uri" ] ||
    fail "$target rewrote the URI incorrectly: $body"
  [ "$(body_field "$body" host)" = "$expected_host" ] ||
    fail "$target forwarded the wrong Host: $body"
  [ "$(body_field "$body" proto)" = "$expected_scheme" ] ||
    fail "$target forwarded the wrong scheme: $body"
  [ "$(body_field "$body" xff)" = 127.0.0.1 ] ||
    fail "$target trusted inbound X-Forwarded-For: $body"
  [ -z "$(body_field "$body" xf_host)" ] ||
    fail "$target preserved hostile X-Forwarded-Host: $body"
}

start_external
expect_startup_failure 'invalid/upstream:8080' customer.example.com https \
  'an invalid upstream'
expect_startup_failure "${PUBLIC_CONTAINER}:8080" 'bad_host.example' https \
  'an invalid hostname'

start_gateway customer.example.com https
rendered_config=$(runtime exec "$GATEWAY_CONTAINER" \
  nginx -T -c /tmp/kumo-ui/nginx.conf 2>/dev/null)
for directive in \
  'proxy_connect_timeout 75s;' \
  'proxy_send_timeout 300s;' \
  'proxy_read_timeout 600s;' \
  'send_timeout 300s;' \
  'client_max_body_size 128m;'; do
  [[ "$rendered_config" == *"$directive"* ]] ||
    fail "rendered frontend config is missing: $directive"
done
request GET /-/live
assert_status 200 liveness
[ "$(response_body)" = '{"status":"live"}' ] || fail "wrong liveness body"
request GET /-/ready
assert_status 200 readiness
[ "$(response_body)" = '{"status":"ready"}' ] || fail "wrong readiness body"

assert_backend '/v0/projects?x=1' private-backend '/v0/projects?x=1' \
  "${PRIVATE_CONTAINER}:8080"
assert_backend '/v1?x=2' private-backend '/v1?x=2' \
  "${PRIVATE_CONTAINER}:8080"
assert_backend '/internal/projects?x=3' private-backend '/projects?x=3' \
  "${PRIVATE_CONTAINER}:8080"
assert_backend '/api/projects?x=4' public-backend '/projects?x=4' \
  "${PUBLIC_CONTAINER}:8080"

request GET '/internal?x=3'
assert_status 308 '/internal redirect'
[ "$(response_header Location)" = '/internal/?x=3' ] || fail "wrong /internal redirect"
request GET '/api?x=4'
assert_status 308 '/api redirect'
[ "$(response_header Location)" = '/api/?x=4' ] || fail "wrong /api redirect"
request GET /api/absolute-redirect
assert_status 307 'upstream API redirect'
[ "$(response_header Location)" = '/api/v1/redirected?source=api' ] ||
  fail "upstream API redirect escaped /api: $(response_header Location)"
request GET /api/
assert_status 307 'relative upstream API root redirect'
[ "$(response_header Location)" = redoc ] ||
  fail "relative upstream API redirect was rewritten: $(response_header Location)"
for docs_path in /api/docs /api/redoc; do
  request GET "$docs_path"
  assert_status 200 "$docs_path"
  [[ "$(response_body)" == *'./openapi.json'* ]] ||
    fail "$docs_path escaped the /api schema prefix"
done
request GET /api/openapi.json
assert_status 200 'public API schema'
[ "$(response_body)" = '{"openapi":"3.1.0"}' ] ||
  fail "public API schema route escaped /api"
request GET /api/backend-not-found
assert_status 404 'backend 404 pass-through'
[ "$(response_body)" = '{"backend":"not_found"}' ] || fail "backend 404 was replaced"
request_with_body POST /api/large-request $((2 * 1024 * 1024))
assert_status 200 '2 MiB API request'

request GET /
assert_status 200 'SPA root'
spa_body=$(response_body)
request GET /apiary
assert_status 200 'namespace boundary SPA fallback'
[ "$(response_body)" = "$spa_body" ] || fail "/apiary escaped SPA fallback"
request POST /missing
assert_status 404 'SPA method gate'
request GET /public_api/projects
assert_status 404 'blocked public API namespace'

for unsafe_path in '/api//projects' '/api/%2e%2E/projects' \
  '/api%2Fprojects' '/api%5Cprojects' '/api\projects'; do
  request GET "$unsafe_path"
  assert_status 400 "unsafe path $unsafe_path"
done
assert_backend '/api/projects?next=/api%2Fsecret//..%5C' \
  public-backend '/projects?next=/api%2Fsecret//..%5C' \
  "${PUBLIC_CONTAINER}:8080"

request GET /api/headers
assert_status 200 'header normalization'
headers_body=$(response_body)
for field in forwarded xf_host connection upgrade; do
  [ -z "$(body_field "$headers_body" "$field")" ] ||
    fail "gateway preserved hostile $field: $headers_body"
done
[ "$(body_field "$headers_body" real_ip)" = 127.0.0.1 ] ||
  fail "gateway trusted hostile X-Real-IP: $headers_body"
request_id=$(body_field "$headers_body" request_id)
[[ "$request_id" =~ ^[0-9A-Fa-f]{32}$ ]] || fail "gateway did not mint request ID"
[ "$(body_field "$headers_body" origin)" = https://browser.example ] || fail "Origin was lost"
[ "$(body_field "$headers_body" cookie)" = session=preserved ] || fail "Cookie was lost"
[ "$(body_field "$headers_body" authorization)" = Bearer-preserved ] || fail "Authorization was lost"
[ -n "$(body_field "$headers_body" traceparent)" ] || fail "Traceparent was lost"

request GET /env-config.js
assert_status 200 'runtime UI config'
[[ "$(response_body)" == *'__GATEWAY_SMOKE__'* ]] || fail "frontend config handoff failed"
[ "$(response_header Cache-Control)" = \
  'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0' ] ||
  fail "runtime UI config lost no-cache policy"

start_gateway '' http
assert_backend '/api/local?x=1' public-backend '/local?x=1' \
  "${PUBLIC_CONTAINER}:8080" http local.gateway:31337

mode=source-mounted
[ "$SOURCE_MODE" = false ] && mode=actual-image
echo "Frontend gateway smoke: passed ($mode, $GATEWAY_IMAGE)"
