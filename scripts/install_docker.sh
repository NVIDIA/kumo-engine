#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install docker.
if command -v docker &> /dev/null; then
    echo "Docker is already installed!"
    exit 0
fi

if [ "$(uname -s)" = "Linux" ]; then
    if [ "$EUID" -ne 0 ]; then
        echo "Please run as root. 'sudo ./scripts/install_docker.sh'"
        exit 1
    fi

    if [ ! -r /etc/os-release ]; then
        echo "Unable to detect Linux distribution: /etc/os-release is missing."
        echo "Please install Docker manually: https://docs.docker.com/engine/install/"
        exit 1
    fi

    # shellcheck disable=SC1091
    . /etc/os-release

    if [ "${ID:-}" != "ubuntu" ]; then
        echo "Unsupported Linux distribution: ${ID:-unknown}"
        echo "Please install Docker manually: https://docs.docker.com/engine/install/"
        exit 1
    fi

    case "${VERSION_CODENAME:-}" in
        jammy|noble) ;;
        *)
            echo "Unsupported Ubuntu release: ${PRETTY_NAME:-unknown}"
            echo "Supported Ubuntu releases: 22.04 (jammy), 24.04 (noble)"
            echo "Please install Docker manually: https://docs.docker.com/engine/install/ubuntu/"
            exit 1
            ;;
    esac

    echo "Attempting to install Docker Engine on ${PRETTY_NAME}."
    echo "If any of the steps fail, follow the official instructions:"
    echo "https://docs.docker.com/engine/install/ubuntu/"

    apt-get update
    apt-get install -y ca-certificates curl

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    arch="$(dpkg --print-architecture)"
    echo \
        "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "Installed Docker: $(docker --version)"
    exit 0
fi

if [ "$(uname -s)" = "Darwin" ]; then
    echo "Docker Desktop installation is intentionally not automated on macOS." >&2
    echo "Install and start it using the official instructions:" >&2
    echo "https://docs.docker.com/desktop/setup/install/mac-install/" >&2
    exit 1
fi

echo "Unsupported operating system: $(uname -s)"
echo "Please install Docker manually: https://docs.docker.com/engine/install/"
exit 1
