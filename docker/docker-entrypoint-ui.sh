#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

mkdir -p /usr/share/nginx/html

routing_mode=${KUMO_UI_ROUTING_MODE:-external}

if [ "$routing_mode" = frontend ] && [ -f /tmp/config/env.js ]; then
  cp /tmp/config/env.js /usr/share/nginx/html/env-config.js
elif [ -f /opt/app/env.js ]; then
  sed_script="s/__UI_LOCAL__/${UI_LOCAL:-false}/g"

  if [ -n "${API_BASE_URL:-}" ]; then
    # Escape sed replacement metacharacters for the | delimiter: | & \
    api_base_url_escaped="$(printf '%s' "$API_BASE_URL" | sed 's/[|&\\]/\\&/g')"
    sed_script="s|\`http://127.0.0.1:\${port}/\`|\"${api_base_url_escaped}\"|g; s|http://127.0.0.1:8000/|${api_base_url_escaped}|g; ${sed_script}"
  fi

  auth_disabled_value="${AUTH_DISABLED:-true}"
  sed_script="s/__AUTH_DISABLED__/${auth_disabled_value}/g; s/AUTH_DISABLED: true/AUTH_DISABLED: ${auth_disabled_value}/; ${sed_script}"

  sed "$sed_script" /opt/app/env.js > /usr/share/nginx/html/env-config.js
fi

case "$routing_mode" in
  external)
    if [ -n "${KUMO_UI_PORT:-}" ]; then
      sed \
        -e "s/listen 3000/listen ${KUMO_UI_PORT}/g" \
        -e "s/listen \\[::\\]:3000/listen [::]:${KUMO_UI_PORT}/g" \
        /etc/nginx/conf.d/default.conf > /tmp/default.conf
      cp /tmp/default.conf /etc/nginx/conf.d/default.conf
    fi

    exec "$@"
    ;;
  frontend)
    state_dir=/tmp/kumo-ui
    config_path="${state_dir}/nginx.conf"
    ready_path="${state_dir}/ready"

    mkdir -p "$state_dir"
    rm -f "$ready_path"
    /bin/sh /opt/kumo-ui/bin/render-nginx \
      /opt/kumo-ui/nginx.frontend.conf.template "$config_path"
    nginx -t -c "$config_path"
    printf '%s\n' '{"status":"ready"}' >"$ready_path"
    exec nginx -c "$config_path" -g "daemon off;"
    ;;
  *)
    echo "unsupported KUMO_UI_ROUTING_MODE: ${KUMO_UI_ROUTING_MODE}" >&2
    exit 1
    ;;
esac
