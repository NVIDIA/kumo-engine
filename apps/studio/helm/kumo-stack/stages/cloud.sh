#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

validate_rendered_image_policy() {
  local component="$1" manifest="$2"
  shift 2
  local image permitted allowed
  while IFS= read -r image; do
    [[ -n "${image}" ]] || continue
    allowed=false
    for permitted in "$@"; do
      if [[ "${image}" == "${permitted}" ]]; then
        allowed=true
        break
      fi
    done
    [[ "${allowed}" == true ]] ||
      die "${component} chart renders non-permitted image ${image}"
  done < <(
    sed -nE \
      's/^[[:space:]]+image:[[:space:]]*"?([^"[:space:]]+)"?.*$/\1/p' \
      "${manifest}" | sort -u
  )
}

verify_registry_image_with_secret() {
  local image="$1" docker_config="$2"
  local registry repository_tag repository reference auth manifest_url
  local response status challenge realm service scope token_json token
  registry="${image%%/*}"
  repository_tag="${image#*/}"
  if [[ "${repository_tag}" == *@* ]]; then
    repository="${repository_tag%%@*}"
    repository="${repository%:*}"
    reference="${repository_tag#*@}"
  else
    repository="${repository_tag%:*}"
    reference="${repository_tag##*:}"
  fi
  manifest_url="https://${registry}/v2/${repository}/manifests/${reference}"
  auth=$("${JQ}" -r --arg registry "${registry}" '
    .auths[$registry].auth //
    .auths["https://" + $registry].auth //
    ([.auths | to_entries[] |
      select(.key | contains($registry)) | .value.auth][0]) //
    empty
  ' <<<"${docker_config}")
  [[ -n "${auth}" ]] ||
    die "Secret ${TENANT_NAMESPACE}/${KUMO_PULL_SECRET} has no credentials for ${registry}"

  response=$(printf 'header = "Authorization: Basic %s"\n' "${auth}" |
    curl --config - --head --silent --show-error --location \
      --connect-timeout 10 --max-time 30 \
      --header 'Accept: application/vnd.oci.image.index.v1+json' \
      --header 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
      --header 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
      --write-out $'\n%{http_code}' "${manifest_url}") ||
    die "registry request failed for ${image}"
  status="${response##*$'\n'}"
  [[ "${status}" != 2* && "${status}" != 3* ]] || return 0
  [[ "${status}" == 401 ]] ||
    die "image pull Secret ${TENANT_NAMESPACE}/${KUMO_PULL_SECRET} cannot read ${image}"

  challenge=$(sed -n \
    's/^[Ww][Ww][Ww]-[Aa]uthenticate:[[:space:]]*[Bb]earer[[:space:]]*//p' \
    <<<"${response}" | tr -d '\r')
  [[ -n "${challenge}" ]] ||
    die "registry ${registry} rejected ${image} without a Bearer challenge"
  realm=$(sed -n 's/.*realm="\([^"]*\)".*/\1/p' <<<"${challenge}")
  service=$(sed -n 's/.*service="\([^"]*\)".*/\1/p' <<<"${challenge}")
  scope=$(sed -n 's/.*scope="\([^"]*\)".*/\1/p' <<<"${challenge}")
  [[ -n "${realm}" ]] ||
    die "registry ${registry} returned an invalid Bearer challenge"
  scope="${scope:-repository:${repository}:pull}"

  token_json=$(printf 'header = "Authorization: Basic %s"\n' "${auth}" |
    curl --config - --get --fail --silent --show-error --location \
      --connect-timeout 10 --max-time 30 \
      --data-urlencode "service=${service}" \
      --data-urlencode "scope=${scope}" "${realm}") ||
    die "registry token exchange failed for ${image}"
  token=$("${JQ}" -r '.token // .access_token // empty' <<<"${token_json}")
  [[ -n "${token}" ]] ||
    die "registry token exchange returned no token for ${image}"

  if ! printf 'header = "Authorization: Bearer %s"\n' "${token}" |
    curl --config - --head --fail --silent --show-error --location \
      --connect-timeout 10 --max-time 30 \
      --header 'Accept: application/vnd.oci.image.index.v1+json' \
      --header 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
      --header 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
      "${manifest_url}" >/dev/null; then
    die "image pull Secret ${TENANT_NAMESPACE}/${KUMO_PULL_SECRET} cannot read ${image}"
  fi
}

verify_kumo_image_access() {
  [[ -n "${KUMO_PULL_SECRET}" ]] || return 0
  require_command curl

  local secret_json docker_config image
  secret_json=$(kubectl_cmd -n "${TENANT_NAMESPACE}" get secret \
    "${KUMO_PULL_SECRET}" -o json) ||
    die "missing image pull Secret ${TENANT_NAMESPACE}/${KUMO_PULL_SECRET}"
  docker_config=$("${JQ}" -r '
    .data[".dockerconfigjson"] // empty | @base64d
  ' <<<"${secret_json}")
  [[ -n "${docker_config}" ]] ||
    die "Secret ${TENANT_NAMESPACE}/${KUMO_PULL_SECRET} has no .dockerconfigjson"

  for image in "${KUMO_CPU_IMAGE}" "${KUMO_GPU_IMAGE}" "${KUMO_UI_IMAGE}"; do
    verify_registry_image_with_secret "${image}" "${docker_config}"
  done
}

stage_cloud_check() {
  validate_leaf_charts
  kubectl_cmd version -o json >/dev/null
  load_provider
  provider_preflight
  verify_kumo_image_access

  # The Temporal bootstrap opens no database connection from here: it runs as a
  # Job whose pod mints its own admin token, so this host needs neither psql nor
  # a network route to private PostgreSQL. The cloud CLI stays required because
  # provider preflight drives it. The admin login itself is proved by the
  # database stage, which is the first place a cluster-side identity exists.
  if [[ "${TEMPORAL_ENABLED}" == true &&
        "${POSTGRES_BOOTSTRAP}" == true ]]; then
    case "${POSTGRES_ADMIN_MODE}" in
      aws_iam) require_command aws ;;
      azure_entra) require_command az ;;
      *) die "unsupported cloud.postgres.admin.mode: ${POSTGRES_ADMIN_MODE}" ;;
    esac
  elif [[ "${TEMPORAL_ENABLED}" == true ]] &&
       ! kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get secret \
         "${POSTGRES_SECRET}" >/dev/null 2>&1; then
    die "bootstrapTemporal=false requires pre-created Secret ${TEMPORAL_NAMESPACE}/${POSTGRES_SECRET}"
  fi
}
