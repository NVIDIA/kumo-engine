#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare the Direct Connect OIDC devserver variant.
#
# Prompts for the admin email (the bootstrap admin) and renders an untracked
# deploy config from rest/devserver/configs/oidc.json into /tmp. The OIDC
# client secret is NOT handled here: it is pulled from AWS Secrets Manager
# (secret name "devserver-oidc-client-secret") at server startup, so no local
# secret needs to be shared or seeded.
set -euo pipefail

WORK_DIR="/tmp/kumo-oidc-secrets"
TEMPLATE="rest/devserver/configs/oidc.json"
RENDERED="${WORK_DIR}/oidc.rendered.json"
EMAIL_CACHE="${WORK_DIR}/admin-email"

mkdir -p "${WORK_DIR}"

default_email=""
if [ -f "${EMAIL_CACHE}" ]; then
  default_email="$(cat "${EMAIL_CACHE}")"
fi

prompt="Direct Connect admin email"
if [ -n "${default_email}" ]; then
  prompt="${prompt} [${default_email}]"
fi
read -r -p "${prompt}: " admin_email
admin_email="${admin_email:-${default_email}}"
if [ -z "${admin_email}" ]; then
  echo "ERROR: an admin email is required to bootstrap the first admin." >&2
  exit 1
fi
printf '%s' "${admin_email}" > "${EMAIL_CACHE}"

# Resolve the base port the devserver will actually bind to (the web UI is on
# base_port + 0). `devserver-oidc` passes no --base_port/--random_port, so this
# matches generate_base_port(): PHASE1B_DEVSERVER_BASE_PORT or the 10000 default.
base_port="${PHASE1B_DEVSERVER_BASE_PORT:-10000}"

# Render the config with the chosen admin email and the resolved return origin.
ADMIN_EMAIL="${admin_email}" BASE_PORT="${base_port}" \
  TEMPLATE="${TEMPLATE}" RENDERED="${RENDERED}" python3 - <<'PY'
import json, os
with open(os.environ["TEMPLATE"]) as f:
    cfg = json.load(f)
dc = cfg["auth"]["direct_connect"]
dc["admin_emails"] = [os.environ["ADMIN_EMAIL"]]
dc["allowed_return_origins"] = [f'http://localhost:{os.environ["BASE_PORT"]}']
with open(os.environ["RENDERED"], "w") as f:
    json.dump(cfg, f, indent=2)
PY

echo "OIDC variant ready:"
echo "  admin email : ${admin_email}"
echo "  config      : ${RENDERED}"
echo "  client secret source: AWS Secrets Manager (devserver-oidc-client-secret)"
echo "  callback URI to register in Entra: http://localhost:${base_port}/v0/auth/callback"
echo "  (attached single devserver on base port ${base_port}; exact URI also printed at startup)"
