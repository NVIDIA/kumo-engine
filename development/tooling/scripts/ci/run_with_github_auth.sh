#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GitHub token is required to resolve private dependencies" >&2
  exit 1
fi

github_basic_auth="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  echo "::add-mask::${github_basic_auth}"
fi

# Pass authentication only to this command and its children. Ignore any
# persistent user config on self-hosted runners, and normalize legacy SSH
# dependency URLs to credential-free HTTPS before applying the scoped header.
# In particular, do not persist the token in ~/.gitconfig,
# ~/.git-credentials, or ~/.netrc.
GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_COUNT=3 \
  GIT_CONFIG_KEY_0=http.https://github.com/.extraheader \
  GIT_CONFIG_VALUE_0="AUTHORIZATION: basic ${github_basic_auth}" \
  GIT_CONFIG_KEY_1=url.https://github.com/.insteadOf \
  GIT_CONFIG_VALUE_1=ssh://git@github.com/ \
  GIT_CONFIG_KEY_2=url.https://github.com/.insteadOf \
  GIT_CONFIG_VALUE_2=git@github.com: \
  "$@"
