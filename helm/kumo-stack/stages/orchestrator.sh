#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

INSTALL_STAGES=(cloud platform database temporal core)

stage_index() {
  local wanted="$1" index
  for index in "${!INSTALL_STAGES[@]}"; do
    [[ "${INSTALL_STAGES[$index]}" == "${wanted}" ]] && {
      printf '%s' "${index}"
      return
    }
  done
  die "unknown install stage: ${wanted}"
}

run_stage() {
  local stage="$1"
  printf '→ %s\n' "${stage}"
  case "${stage}" in
    cloud)
      stage_cloud_check
      ;;
    platform)
      stage_platform_apply
      stage_platform_verify
      ;;
    database)
      stage_database_apply
      stage_database_verify
      ;;
    temporal)
      stage_temporal_apply
      stage_temporal_verify
      ;;
    core)
      stage_core_apply
      stage_core_verify
      ;;
    *)
      die "unknown install stage: ${stage}"
      ;;
  esac
  printf '✓ %s\n' "${stage}"
}

run_named_install() {
  local stage="$1"
  case "${stage}" in
    platform)
      stage_cloud_check
      stage_platform_apply
      stage_platform_verify
      ;;
    temporal)
      stage_database_verify
      stage_temporal_apply
      stage_temporal_verify
      ;;
    core)
      # Temporal health proves its runtime database path; the core chart's
      # in-cluster preflight proves StateDB. Re-running the operator-side
      # database bootstrap check here breaks private PostgreSQL deployments.
      stage_temporal_verify
      stage_core_apply
      stage_core_verify
      ;;
    *)
      die "install supports platform, temporal, or core"
      ;;
  esac
}

run_database_bootstrap() {
  stage_platform_verify
  stage_database_apply
  stage_database_verify
}

run_install_range() {
  local from="${FROM_STAGE:-cloud}" through="${THROUGH_STAGE:-core}"
  local first last index
  first=$(stage_index "${from}")
  last=$(stage_index "${through}")
  (( first <= last )) ||
    die "--from ${from} is after --through ${through}"
  for ((index = first; index <= last; index++)); do
    run_stage "${INSTALL_STAGES[$index]}"
  done
}
