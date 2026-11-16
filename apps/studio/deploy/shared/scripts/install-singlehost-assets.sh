#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install the provider-neutral files staged by the single-host Packer builds.
set -euo pipefail

STAGING_ROOT="${1:-/tmp}"
INSTALL_ROOT="${2:-}"
KUMO_ROOT="$INSTALL_ROOT/opt/kumo"
BOOTSTRAP_STAGING_ROOT="$STAGING_ROOT/deploy-shared"
BOOTSTRAP_ROOT="$KUMO_ROOT/deploy-shared"

mkdir -p \
    "$KUMO_ROOT/compose/config/postgres-initdb" \
    "$KUMO_ROOT/compose/config/temporal" \
    "$KUMO_ROOT/scripts" \
    "$KUMO_ROOT/configs" \
    "$KUMO_ROOT/secrets" \
    "$KUMO_ROOT/nginx" \
    "$KUMO_ROOT/prometheus" \
    "$INSTALL_ROOT/etc/kumo" \
    "$INSTALL_ROOT/etc/systemd/system/docker.service.d"
mkdir -p "$BOOTSTRAP_ROOT"
cp -R "$BOOTSTRAP_STAGING_ROOT"/. "$BOOTSTRAP_ROOT/"
cp "$STAGING_ROOT"/docker-compose.singlehost*.yml "$KUMO_ROOT/compose/"
cp -r "$STAGING_ROOT"/postgres-initdb/* "$KUMO_ROOT/compose/config/postgres-initdb/"
cp -r "$STAGING_ROOT"/temporal/* "$KUMO_ROOT/compose/config/temporal/"
cp "$BOOTSTRAP_ROOT"/scripts/* "$KUMO_ROOT/scripts/"
chmod +x "$KUMO_ROOT"/scripts/*.sh
# Bootstrap writes nginx.conf after selecting the TLS or HTTP variant.
cp "$BOOTSTRAP_ROOT"/prometheus/* "$KUMO_ROOT/prometheus/"
cp "$BOOTSTRAP_ROOT/configs/deploy_env_config.template.json" "$KUMO_ROOT/configs/"
cp "$BOOTSTRAP_ROOT/configs/kumo-mount-nvme.service" "$INSTALL_ROOT/etc/systemd/system/"
cp "$BOOTSTRAP_ROOT/configs/docker-kumo-mount-nvme.conf" \
    "$INSTALL_ROOT/etc/systemd/system/docker.service.d/kumo-mount-nvme.conf"
