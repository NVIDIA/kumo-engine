#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


diskgraph_portable_target_feature_flags() {
  case "$(uname -m)" in
    x86_64|amd64)
      printf '%s' "-C target-feature=+aes,+sse2"
      ;;
    aarch64|arm64)
      printf '%s' "-C target-feature=+aes,+neon"
      ;;
    *)
      return 0
      ;;
  esac
}

diskgraph_release_rustflags() {
  local rustflags="${RUSTFLAGS:-}"
  local target_feature_flags
  target_feature_flags="$(diskgraph_portable_target_feature_flags)"

  if [[ -n "${target_feature_flags}" ]]; then
    rustflags="${rustflags} ${target_feature_flags}"
  fi
  rustflags="${rustflags} $*"

  # Trim leading/trailing whitespace without depending on non-portable shell
  # extensions beyond bash, which these scripts already require.
  rustflags="${rustflags#"${rustflags%%[![:space:]]*}"}"
  rustflags="${rustflags%"${rustflags##*[![:space:]]}"}"
  printf '%s' "${rustflags}"
}
