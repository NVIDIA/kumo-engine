#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

template_path=${1:-/opt/kumo-ui/nginx.frontend.conf.template}
output_path=${2:-/tmp/kumo-ui/nginx.conf}

die() {
  echo "frontend renderer: $*" >&2
  exit 1
}

[ "${KUMO_UI_ROUTING_MODE:-}" = frontend ] ||
  die "KUMO_UI_ROUTING_MODE must be frontend"
[ -r "$template_path" ] ||
  die "Nginx template is not readable: $template_path"

validate_upstream() {
  upstream_name=$1
  upstream_value=$2
  printf '%s\n' "$upstream_value" |
    grep -Eq '^[A-Za-z0-9][A-Za-z0-9.-]*:[1-9][0-9]{0,4}$' ||
    die "$upstream_name must be a DNS name and port"
  upstream_port=${upstream_value##*:}
  [ "$upstream_port" -le 65535 ] ||
    die "$upstream_name port is outside 1..65535"
}

private_upstream=${KUMO_PRIVATE_UPSTREAM:-}
public_upstream=${KUMO_PUBLIC_UPSTREAM:-}
validate_upstream KUMO_PRIVATE_UPSTREAM "$private_upstream"
validate_upstream KUMO_PUBLIC_UPSTREAM "$public_upstream"

[ "${KUMO_EXTERNAL_HOST+x}" = x ] ||
  die "KUMO_EXTERNAL_HOST must be set (empty selects local fallback)"
[ "${KUMO_EXTERNAL_SCHEME+x}" = x ] ||
  die "KUMO_EXTERNAL_SCHEME must be set"

external_host=$KUMO_EXTERNAL_HOST
external_scheme=$KUMO_EXTERNAL_SCHEME
case "$external_scheme" in
  http|https) ;;
  *) die "KUMO_EXTERNAL_SCHEME must be http or https" ;;
esac

if [ -n "$external_host" ]; then
  [ "${#external_host}" -le 253 ] &&
    printf '%s\n' "$external_host" |
      grep -Eq '^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$' ||
    die "KUMO_EXTERNAL_HOST must be a lowercase DNS-1123 hostname"
  if printf '%s\n' "$external_host" |
    grep -Eq '^[0-9]+(\.[0-9]+){3}$'; then
    die "KUMO_EXTERNAL_HOST must be a DNS hostname, not an IP address"
  fi
  proxy_host=$external_host
else
  [ "$external_scheme" = http ] ||
    die "host-empty local fallback requires KUMO_EXTERNAL_SCHEME=http"
  proxy_host='$http_host'
fi

output_dir=${output_path%/*}
[ "$output_dir" != "$output_path" ] || output_dir=.
mkdir -p "$output_dir"
rendered_path="${output_path}.rendered.$$"
trap 'rm -f "$rendered_path"' EXIT HUP INT TERM

sed \
  -e "s|@@PRIVATE_UPSTREAM@@|$private_upstream|g" \
  -e "s|@@PUBLIC_UPSTREAM@@|$public_upstream|g" \
  -e "s|@@EXTERNAL_HOST@@|$proxy_host|g" \
  -e "s|@@EXTERNAL_SCHEME@@|$external_scheme|g" \
  "$template_path" >"$rendered_path"

mv "$rendered_path" "$output_path"
trap - EXIT HUP INT TERM
