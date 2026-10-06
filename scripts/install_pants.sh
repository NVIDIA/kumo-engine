#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

if ! command -v pants >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://static.pantsbuild.org/setup/get-pants.sh | bash
fi

if command -v pants >/dev/null 2>&1; then
  echo "Pants version:"
  pants --version
else
  echo "Pants was installed but is not on PATH in this shell."
  echo "Ensure ~/.local/bin is on PATH, restart your shell, then run: pants --version"
  exit 1
fi
