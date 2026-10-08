#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
fi
export PATH="${HOME}/.cargo/bin:${PATH}"

if ! command -v rustup >/dev/null 2>&1; then
  rustup_installer_commit="d95a37b6ab92cc1e455d1576039333c97ca3e2c5"
  curl --proto '=https' --tlsv1.2 -sSf \
    "https://raw.githubusercontent.com/rust-lang/rustup/${rustup_installer_commit}/rustup-init.sh" |
    sh -s -- -y
  # shellcheck disable=SC1090
  source "${HOME}/.cargo/env"
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

rustup self update
