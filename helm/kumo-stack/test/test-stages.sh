#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRACE=()

die() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

stage_cloud_check() { TRACE+=(cloud); }
stage_platform_apply() { TRACE+=(platform-apply); }
stage_platform_verify() { TRACE+=(platform-verify); }
stage_database_apply() { TRACE+=(database-apply); }
stage_database_verify() { TRACE+=(database-verify); }
stage_temporal_apply() { TRACE+=(temporal-apply); }
stage_temporal_verify() { TRACE+=(temporal-verify); }
stage_core_apply() { TRACE+=(core-apply); }
stage_core_verify() { TRACE+=(core-verify); }

# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/orchestrator.sh"

run_install_range >/dev/null
expected=(
  cloud
  platform-apply platform-verify
  database-apply database-verify
  temporal-apply temporal-verify
  core-apply core-verify
)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "full install stage order is ${TRACE[*]}"

TRACE=()
FROM_STAGE=database
THROUGH_STAGE=temporal
run_install_range >/dev/null
expected=(database-apply database-verify temporal-apply temporal-verify)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "bounded install stage order is ${TRACE[*]}"

TRACE=()
run_named_install platform
expected=(cloud platform-apply platform-verify)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "named platform install order is ${TRACE[*]}"

TRACE=()
run_database_bootstrap
expected=(platform-verify database-apply database-verify)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "database bootstrap order is ${TRACE[*]}"

TRACE=()
run_named_install temporal
expected=(database-verify temporal-apply temporal-verify)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "named Temporal install order is ${TRACE[*]}"

TRACE=()
run_named_install core
expected=(temporal-verify core-apply core-verify)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "named core install order is ${TRACE[*]}"

# Every Temporal database check reaches PostgreSQL from inside the cluster:
# customer-managed databases through the stock schema hooks, installer-managed
# ones through the stack's own Job. Neither may require operator-side psql.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/database.sh"
JQ="${JQ:-jq}"
TEMPORAL_ENABLED=true
TEMPORAL_NAMESPACE=temporal
POSTGRES_SECRET=temporal-postgres
POSTGRES_SECRET_KEY=password
verify_state_database_from_cluster() { TRACE+=(state-database); }
require_command() { TRACE+=("require-$1"); }
kubectl_cmd() { printf '{"data":{"password":"cGFzc3dvcmQ="}}\n'; }
verify_temporal_database() { TRACE+=(temporal-database); }

TRACE=()
POSTGRES_BOOTSTRAP=false
stage_database_verify
expected=(state-database)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "customer-managed database verification is ${TRACE[*]}"

TRACE=()
POSTGRES_BOOTSTRAP=true
stage_database_verify
expected=(state-database temporal-database)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "installer-managed database verification is ${TRACE[*]}"

if grep -Fq 'require_command psql' "${ROOT}/kumo-stack/stages/cloud.sh"; then
  die "the cloud gate still requires psql on the operator host"
fi
# psql may be named in the rendered Job manifest, but never invoked as a
# command by this stage, and no cloud token may be minted here.
if grep -v '^[[:space:]]*#' "${ROOT}/kumo-stack/stages/database.sh" |
  grep -Eq 'PGPASSWORD|generate-db-auth-token|get-access-token|(^|[;&|]|\$\()[[:space:]]*psql[[:space:]]'; then
  die "the database stage still opens PostgreSQL from the operator host"
fi

# The Temporal database Job carries the SQL and both entry points, mints its
# admin token in-pod, and reads the Temporal password from the Secret. No
# credential may appear in the manifest the API server stores.
unset -f verify_temporal_database
STACK_ROOT="${ROOT}/kumo-stack"
PLATFORM=eks
TIMEOUT=20m
VERIFY_TIMEOUT=5m
PROVIDER_REGION=us-west-2
POSTGRES_HOST=state.example.us-west-2.rds.amazonaws.com
POSTGRES_PORT=5432
POSTGRES_ADMIN_MODE=aws_iam
POSTGRES_ADMIN_USER=postgres_admin
POSTGRES_ADMIN_DATABASE=postgres
POSTGRES_ADMIN_ANNOTATION_KEY=eks.amazonaws.com/role-arn
POSTGRES_ADMIN_ANNOTATION_VALUE=arn:aws:iam::000000000000:role/kumo-postgres-admin
POSTGRES_TEMPORAL_USER=temporal_runtime
POSTGRES_CLIENT_IMAGE=registry.example.com/postgres:17.10-alpine@sha256:client
POSTGRES_CLIENT_PULL_SECRET=postgres-pull
KUMO_CPU_IMAGE=registry.example.com/kumo-backend-cpu:sha
KUMO_PULL_SECRET=kumo-pull
JOB_MANIFEST=""
SERVICE_ACCOUNT_MANIFEST=""
# The ConfigMap is applied through a pipeline, whose stages run in subshells, so
# it can only be captured in a file.
APPLIED_LOG=$(mktemp "${TMPDIR:-/tmp}/kumo-applied.XXXXXX")
KUBECTL_CALLS=""
DELETED=""
SERVICE_ACCOUNT_EXISTS=false
SERVICE_ACCOUNT_LABELLED=false
SERVICE_ACCOUNT_ERROR='Error from server (NotFound): serviceaccounts "kumo-temporal-bootstrap" not found'
duration_seconds() { printf '300'; }
wait_for_job_terminal() { return 0; }
record_delete() {
  local -a words=("$@")
  local index
  for index in "${!words[@]}"; do
    if [[ "${words[index]}" == delete ]]; then
      DELETED+="${words[index + 1]}/${words[index + 2]} "
      return
    fi
  done
}
kubectl_cmd() {
  local args=" $* " applied
  KUBECTL_CALLS+="$* "$'\n'
  case "${args}" in
    *" get serviceaccount "*)
      if [[ "${SERVICE_ACCOUNT_EXISTS}" != true ]]; then
        printf '%s\n' "${SERVICE_ACCOUNT_ERROR}" >&2
        return 1
      fi
      "${JQ}" -nc \
        --arg key "${POSTGRES_ADMIN_ANNOTATION_KEY}" \
        --arg value "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
        --argjson labelled "${SERVICE_ACCOUNT_LABELLED}" '{
          metadata: {
            annotations: {($key): $value},
            labels: (if $labelled
              then {"kumo.ai/stage-probe": "database"}
              else {} end)
          }
        }'
      ;;
    *" delete "*) record_delete "$@" ;;
    *" apply -f - "*)
      applied=$(cat)
      printf '%s\n' "${applied}" >>"${APPLIED_LOG}"
      case "${applied}" in
        *"kind: Job"*) JOB_MANIFEST="${applied}" ;;
        *"kind: ServiceAccount"*) SERVICE_ACCOUNT_MANIFEST="${applied}" ;;
      esac
      ;;
    *" create configmap "*)
      printf '{"kind":"ConfigMap","metadata":{"name":"kumo-temporal-bootstrap"}}\n'
      ;;
  esac
}

run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database bootstrap Job did not run"
expected_identity="${POSTGRES_ADMIN_ANNOTATION_KEY}: \"${POSTGRES_ADMIN_ANNOTATION_VALUE}\""
[[ "${SERVICE_ACCOUNT_MANIFEST}" == *"${expected_identity}"* ]] ||
  die "the bootstrap ServiceAccount is not created with ${expected_identity}"
for required in \
  "image: ${KUMO_CPU_IMAGE}" \
  "image: ${POSTGRES_CLIENT_IMAGE}" \
  "- name: kumo-pull" \
  "- name: postgres-pull" \
  "serviceAccountName: kumo-temporal-bootstrap" \
  "automountServiceAccountToken: false" \
  "- bootstrap" \
  "secretKeyRef" \
  "activeDeadlineSeconds: 360" \
  "runAsUser: 1000" \
  "fsGroup: 1000" \
  "medium: Memory"; do
  [[ "${JOB_MANIFEST}" == *"${required}"* ]] ||
    die "the Temporal database Job manifest lacks ${required}"
done
# The psql image is pulled and proved before a credential exists, so a cold
# pull cannot spend the token's lifetime before the first connection.
[[ "$(grep -n 'name: check-psql' <<<"${JOB_MANIFEST}" | cut -d: -f1)" -lt \
   "$(grep -n 'name: mint-admin-credential' <<<"${JOB_MANIFEST}" | cut -d: -f1)" ]] ||
  die "the Temporal database Job mints its token before proving the psql image"
[[ "${JOB_MANIFEST}" != *"KUMO_TEMPORAL_PASSWORD, value"* &&
   "${JOB_MANIFEST}" != *"PGPASSWORD"* ]] ||
  die "the Temporal database Job manifest carries a credential"

# Everything this invocation created is removed, most exposed first: the Job
# whose pod holds the minted admin token, then the ServiceAccount that could
# mint another, then the artifact ConfigMap.
[[ "${DELETED}" == *"job/kumo-temporal-database-bootstrap serviceaccount/kumo-temporal-bootstrap configmap/kumo-temporal-bootstrap "* ]] ||
  die "the Temporal database stage left artifacts behind: ${DELETED}"
# A decommission of an install that was killed outright finds them by label.
for labelled in "${SERVICE_ACCOUNT_MANIFEST}" "${JOB_MANIFEST}"; do
  [[ "${labelled}" == *"kumo.ai/stage-probe: database"* ]] ||
    die "a Temporal database object is created without the stage label"
done
grep -Fq '"kumo.ai/stage-probe": "database"' "${APPLIED_LOG}" ||
  die "the Temporal database ConfigMap is created without the stage label"

# A ServiceAccount the customer created for this identity is reused, not
# adopted: deleting it would remove configuration the stack does not own, and
# labelling it would hand it to the decommission sweep instead.
DELETED=""
SERVICE_ACCOUNT_EXISTS=true
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database Job did not run with a pre-existing ServiceAccount"
[[ "${DELETED}" != *serviceaccount/* ]] ||
  die "a pre-existing admin ServiceAccount was deleted"
[[ "${SERVICE_ACCOUNT_MANIFEST}" != *"kumo.ai/stage-probe"* ]] ||
  die "a customer's ServiceAccount was labelled as stack-owned"

# One this stage created and was killed before deleting still carries the stage
# label, and is reclaimed rather than left for a decommission.
DELETED=""
SERVICE_ACCOUNT_LABELLED=true
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database Job did not run with a remnant ServiceAccount"
[[ "${DELETED}" == *"serviceaccount/kumo-temporal-bootstrap"* ]] ||
  die "a labelled remnant ServiceAccount was not reclaimed"
SERVICE_ACCOUNT_EXISTS=false
SERVICE_ACCOUNT_LABELLED=false

# A ServiceAccount that cannot be read is not a ServiceAccount that is absent.
# Reading it as absent is what would let the stage adopt, and then delete, one
# the customer created.
SERVICE_ACCOUNT_ERROR='Error from server (Forbidden): serviceaccounts "kumo-temporal-bootstrap" is forbidden'
if (
  run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m
) 2>/dev/null; then
  die "an unreadable admin ServiceAccount was treated as absent"
fi
SERVICE_ACCOUNT_ERROR='Error from server (NotFound): serviceaccounts "kumo-temporal-bootstrap" not found'

# The EXIT trap removes the same artifacts when an operator interrupts the
# install while the Job is live. Nothing may survive that path, and the normal
# path afterwards must not repeat or fail on work the trap already did.
DELETED=""
INTERRUPTED_DELETIONS=""
wait_for_job_terminal() {
  remove_temporal_database_artifacts ||
    die "interrupt cleanup could not remove the Temporal database artifacts"
  INTERRUPTED_DELETIONS="${DELETED}"
  return 0
}
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database Job reported failure after an interrupt cleanup"
[[ "${INTERRUPTED_DELETIONS}" == *"job/kumo-temporal-database-bootstrap serviceaccount/kumo-temporal-bootstrap configmap/kumo-temporal-bootstrap "* ]] ||
  die "an interrupt left Temporal database artifacts behind: ${INTERRUPTED_DELETIONS}"
[[ "${DELETED}" == "${INTERRUPTED_DELETIONS}" ]] ||
  die "the Temporal database stage deleted its artifacts twice"
wait_for_job_terminal() { return 0; }

# Decommission reclaims what no shell was left alive to clean up.
KUBECTL_CALLS=""
remove_temporal_database_remnants
[[ "${KUBECTL_CALLS}" == *"delete job,configmap,serviceaccount -l kumo.ai/stage-probe=database"* ]] ||
  die "decommission does not reclaim labelled Temporal database remnants"
grep -Fq 'remove_temporal_database_remnants' "${ROOT}/kumo-stack/bin/kumo" ||
  die "decommission never calls the Temporal database remnant sweep"
# An interrupt must reach that cleanup instead of killing the shell where it
# stands, and the exit path must run it before anything else it does.
for trap_line in \
  "trap cleanup EXIT" \
  "trap 'exit 130' INT" \
  "trap 'exit 143' TERM" \
  "trap 'exit 129' HUP"; do
  grep -Fq "${trap_line}" "${ROOT}/kumo-stack/bin/kumo" ||
    die "the installer does not install ${trap_line}"
done
grep -A9 '^cleanup()' "${ROOT}/kumo-stack/bin/kumo" |
  grep -Fq 'remove_temporal_database_artifacts' ||
  die "the installer exit path does not remove Temporal database artifacts"
grep -A9 '^cleanup()' "${ROOT}/kumo-stack/bin/kumo" |
  grep -Fq 'remove_state_database_artifacts' ||
  die "the installer exit path does not remove state database artifacts"

JOB_MANIFEST=""
run_temporal_database_job kumo-temporal-database-verify verify 5m ||
  die "the Temporal database verification Job did not run"
[[ "${JOB_MANIFEST}" == *"- verify"* ]] ||
  die "the verification Job does not select the read-only entry point"

# OpenShift allocates the UID and fsGroup per namespace and rejects a pod that
# pins its own, so ARO must leave both to the SCC.
JOB_MANIFEST=""
PLATFORM=aro
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database Job did not run on ARO"
[[ "${JOB_MANIFEST}" != *"runAsUser"* && "${JOB_MANIFEST}" != *"fsGroup"* ]] ||
  die "the Temporal database Job pins a UID that OpenShift will not admit"
[[ "${JOB_MANIFEST}" == *"runAsNonRoot: true"* ]] ||
  die "the Temporal database Job allows root on ARO"
PLATFORM=eks

# A ServiceAccount that cannot be created or annotated has no cloud identity,
# so setup failures must stop the stage rather than run the Job without one.
# errexit is suppressed inside a function called in a `||` list, so this is the
# path that would otherwise fail open.
SETUP_FAILS=false
JOB_APPLIES=0
kubectl_cmd() {
  local args=" $* " applied
  case "${args}" in
    *" get serviceaccount "*)
      printf '%s\n' "${SERVICE_ACCOUNT_ERROR}" >&2
      return 1
      ;;
    *" apply -f - "*)
      applied=$(cat)
      if [[ "${applied}" == *"kind: ServiceAccount"* ]]; then
        [[ "${SETUP_FAILS}" != true ]] || return 1
        [[ "${applied}" == *"${POSTGRES_ADMIN_ANNOTATION_VALUE}"* ]] ||
          die "the bootstrap ServiceAccount is created without its identity"
      elif [[ "${applied}" == *"kind: Job"* ]]; then
        JOB_APPLIES=$((JOB_APPLIES + 1))
      fi
      ;;
    *" create configmap "*)
      printf '{"kind":"ConfigMap","metadata":{"name":"kumo-temporal-bootstrap"}}\n'
      ;;
  esac
}
SETUP_FAILS=true
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m &&
  die "a failed ServiceAccount apply still reported success"
[[ "${JOB_APPLIES}" == 0 ]] ||
  die "the Temporal database Job ran without a ServiceAccount identity"
# Control: the same mock without the injected failure must reach the Job, so the
# assertion above cannot pass for an unrelated reason.
SETUP_FAILS=false
run_temporal_database_job kumo-temporal-database-bootstrap bootstrap 20m ||
  die "the Temporal database Job did not run with a healthy ServiceAccount"
[[ "${JOB_APPLIES}" == 1 ]] ||
  die "the control run applied ${JOB_APPLIES} Jobs"

# A namespace that is genuinely gone holds no remnants. An unreadable one is a
# different answer, and must not be mistaken for an empty namespace.
NAMESPACE_ERROR='Error from server (NotFound): namespaces "temporal" not found'
kubectl_cmd() {
  case " $* " in
    *" get namespace "*) printf '%s\n' "${NAMESPACE_ERROR}" >&2; return 1 ;;
    *) die "the remnant sweep ran against a namespace it could not read" ;;
  esac
}
remove_temporal_database_remnants
NAMESPACE_ERROR='Error from server (Forbidden): namespaces "temporal" is forbidden'
if (remove_temporal_database_remnants) 2>/dev/null; then
  die "the remnant sweep read an unreadable namespace as an empty one"
fi

# Both pull Secrets are attached once each; an empty name adds nothing.
pull_secrets=$(temporal_database_pull_secrets)
[[ "${pull_secrets}" == *"- name: kumo-pull"*"- name: postgres-pull"* ]] ||
  die "the Temporal database Job drops a configured pull Secret"
POSTGRES_CLIENT_PULL_SECRET="${KUMO_PULL_SECRET}"
pull_secrets=$(temporal_database_pull_secrets)
[[ "$(grep -c 'name: kumo-pull' <<<"${pull_secrets}")" == 1 ]] ||
  die "the Temporal database Job repeats a shared pull Secret"
# One name being a prefix of the other must not be mistaken for a duplicate.
KUMO_PULL_SECRET=kumo-pull-private
POSTGRES_CLIENT_PULL_SECRET=kumo-pull
pull_secrets=$(temporal_database_pull_secrets)
[[ "$(grep -c 'name: kumo-pull' <<<"${pull_secrets}")" == 2 ]] ||
  die "the Temporal database Job drops a pull Secret whose name is a prefix"
KUMO_PULL_SECRET=""
POSTGRES_CLIENT_PULL_SECRET=""
[[ -z "$(temporal_database_pull_secrets)" ]] ||
  die "the Temporal database Job invents an empty pull Secret"

# Exercise real state-database cleanup after the orchestration stubs above.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/database.sh"
TENANT_NAMESPACE=kumo-acme
KUMO_PULL_SECRET=kumo-pull
KUMO_CPU_IMAGE=registry.example.com/kumo-backend-cpu:sha
VERIFY_TIMEOUT=5m
POSTGRES_STATE_AUTH_MODE=aws_iam
POSTGRES_STATE_DATABASE=kumo
POSTGRES_STATE_USER=kumo_state
STATE_CONTROL_ANNOTATION_KEY=eks.amazonaws.com/role-arn
STATE_CONTROL_ANNOTATION_VALUE=arn:aws:iam::000000000000:role/kumo-control
SERVICE_ACCOUNT_EXISTS=false
SERVICE_ACCOUNT_ERROR='Error from server (NotFound): serviceaccounts "kumo-control" not found'
kubectl_cmd() {
  case " $* " in
    *" get serviceaccount "*)
      if [[ "${SERVICE_ACCOUNT_EXISTS}" != true ]]; then
        printf '%s\n' "${SERVICE_ACCOUNT_ERROR}" >&2
        return 1
      fi
      "${JQ}" -nc --arg key "${STATE_CONTROL_ANNOTATION_KEY}" \
        --arg value "${STATE_CONTROL_ANNOTATION_VALUE}" \
        '{metadata: {annotations: {($key): $value}}}'
      ;;
    *" delete "*) record_delete "$@" ;;
    *" apply -f - "*) cat >/dev/null ;;
  esac
}

DELETED=""
INTERRUPTED_DELETIONS=""
wait_for_job_terminal() {
  # Only what the interrupt itself removes, so the check's own opening delete
  # of a stale Job cannot stand in for a Job the trap failed to record.
  DELETED=""
  remove_state_database_artifacts ||
    die "interrupt cleanup could not remove the state database artifacts"
  INTERRUPTED_DELETIONS="${DELETED}"
  return 0
}
verify_state_database_from_cluster ||
  die "the state database check reported failure after an interrupt cleanup"
[[ "${INTERRUPTED_DELETIONS}" == "job/kumo-statedb-database-check serviceaccount/kumo-control " ]] ||
  die "an interrupt did not remove the state database Job and then its ServiceAccount: ${INTERRUPTED_DELETIONS}"
[[ "${DELETED}" == "${INTERRUPTED_DELETIONS}" ]] ||
  die "the state database check deleted its artifacts twice"
wait_for_job_terminal() { return 0; }

# kumo-control is the core release's ServiceAccount as well. One that already
# carries the control identity is reused, and deleting it would strip that
# release of the identity it runs as.
DELETED=""
SERVICE_ACCOUNT_EXISTS=true
verify_state_database_from_cluster ||
  die "the state database check did not run with a pre-existing ServiceAccount"
[[ "${DELETED}" != *serviceaccount/* ]] ||
  die "the state database check deleted the core release's ServiceAccount"

# A ServiceAccount that cannot be read is not one that is absent: reading it as
# absent is what would record the core release's kumo-control as this check's
# own and then delete it.
DELETED=""
SERVICE_ACCOUNT_EXISTS=false
SERVICE_ACCOUNT_ERROR='Error from server (Forbidden): serviceaccounts "kumo-control" is forbidden'
if (verify_state_database_from_cluster) 2>/dev/null; then
  die "an unreadable kumo-control ServiceAccount was treated as absent"
fi

# A create race must not let cleanup delete another owner's kumo-control.
DELETED_LOG=$(mktemp "${TMPDIR:-/tmp}/kumo-statedb-deleted.XXXXXX")
DELETED=""
SERVICE_ACCOUNT_ERROR='Error from server (NotFound): serviceaccounts "kumo-control" not found'
kubectl_cmd() {
  case " $* " in
    *" get serviceaccount "*)
      printf '%s\n' "${SERVICE_ACCOUNT_ERROR}" >&2
      return 1
      ;;
    *" create serviceaccount "*)
      printf 'Error from server (AlreadyExists): serviceaccounts "kumo-control" already exists\n' >&2
      return 1
      ;;
    *" delete "*) record_delete "$@"; printf '%s' "${DELETED}" >"${DELETED_LOG}" ;;
    *" apply -f - "*) cat >/dev/null ;;
  esac
}
RACE_STATUS=0
(
  trap 'remove_state_database_artifacts || true' EXIT
  verify_state_database_from_cluster
) 2>/dev/null || RACE_STATUS=$?
! grep -q serviceaccount/ "${DELETED_LOG}" ||
  die "losing the kumo-control create race deleted the winner's ServiceAccount"
[[ "${RACE_STATUS}" != 0 ]] ||
  die "a kumo-control ServiceAccount that could not be created was verified"
rm -f "${DELETED_LOG}"

# Registry checks accept both direct Basic auth (ECR-style) and the standard
# Bearer challenge/token exchange used by registries such as ACR.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/cloud.sh"
TENANT_NAMESPACE=kumo-acme
KUMO_PULL_SECRET=kumo-pull
registry_config='{"auths":{"registry.example.com":{"auth":"dXNlcjpwYXNz"}}}'
curl() {
  local args=" $* "
  cat >/dev/null
  if [[ "${args}" == *"/manifests/"* &&
        "${args}" != *"https://registry.example.com/v2/kumo/backend/manifests/${EXPECTED_REGISTRY_REFERENCE}"* ]]; then
    return 1
  fi
  if [[ "${args}" == *" --write-out "* ]]; then
    if [[ "${REGISTRY_TEST_MODE}" == direct ]]; then
      printf 'HTTP/1.1 200 OK\r\n\n200'
    else
      printf 'HTTP/1.1 401 Unauthorized\r\n'
      printf 'Www-Authenticate: Bearer realm="https://auth.example.com/token",service="registry.example.com",scope="repository:kumo/backend:pull"\r\n\n401'
    fi
  elif [[ "${args}" == *" --get "* ]]; then
    printf '{"token":"registry-token"}'
  fi
}
for REGISTRY_TEST_MODE in direct bearer; do
  EXPECTED_REGISTRY_REFERENCE=immutable
  verify_registry_image_with_secret \
    "registry.example.com/kumo/backend:immutable" "${registry_config}"
  EXPECTED_REGISTRY_REFERENCE=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  verify_registry_image_with_secret \
    "registry.example.com/kumo/backend:immutable@${EXPECTED_REGISTRY_REFERENCE}" \
    "${registry_config}"
done
unset -f curl

# The explicit GPU gate must run the real controller canary and restore the
# normal CPU-ready release afterward.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/gpu.sh"
TRACE=()
HELM_ARGS=""
TENANT_RELEASE=acme
TENANT_NAMESPACE=kumo-acme
REPO_ROOT="${ROOT}"
GPU_VERIFY_TIMEOUT=60m
GPU_CANARY_ATTEMPT_TIMEOUT=20m
GPU_CANARY_RETRY_DELAY=1s
GPU_CANARY_RETRY_MAX_DELAY=1s
# Supplied by bin/kumo in a real run; the stages are sourced bare here.
duration_seconds() {
  case "$1" in
    *s) printf '%s' "${1%s}" ;;
    *m) printf '%s' "$(( ${1%m} * 60 ))" ;;
    *h) printf '%s' "$(( ${1%h} * 3600 ))" ;;
  esac
}
PLATFORM=eks
# Exercise the real capacity diagnostic. Record reads in a file because command
# substitutions run in subshells.
NODECLAIM_READS=$(mktemp "${TMPDIR:-/tmp}/kumo-nodeclaims.XXXXXX")
NODECLAIMS_READABLE=false
NODECLAIM_JSON='{"items":[]}'
gpu_nodeclaim_read() {
  printf '%s\n' "$*" >>"${NODECLAIM_READS}"
  if [[ "${NODECLAIMS_READABLE}" != true ]]; then
    printf 'error: the server doesn'\''t have a resource type "nodeclaims"\n' >&2
    return 1
  fi
  printf '%s' "${NODECLAIM_JSON}"
}
stage_core_verify() { TRACE+=(core-verify); }
core_profile_args() { CORE_PROFILE_ARGS=(-f core-values.yaml); }
helm_upgrade() {
  TRACE+=(gpu-canary)
  HELM_ARGS="$*"
}
kubectl_cmd() {
  case " $* " in
    *" get nodeclaims "*) gpu_nodeclaim_read "$@" || return 1; return 0 ;;
  esac
  printf 'hash@1'
}
stage_core_apply() { TRACE+=(core-restore); }
verify_canary_storage_deleted() { TRACE+=(storage-deleted); }
wait_for_gpu_scale_zero() { TRACE+=(scale-zero); }

stage_gpu_lifecycle_verify >/dev/null
expected=(core-verify gpu-canary core-restore storage-deleted scale-zero)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "GPU lifecycle order is ${TRACE[*]}"
[[ "${HELM_ARGS}" == *"POOL_INIT_CANARY_ENABLED=true"* ]] ||
  die "GPU lifecycle did not enable the authoritative canary"
[[ "${HELM_ARGS}" == *"POOL_CANARY_RESULT_TTL_HOURS=0"* ]] ||
  die "GPU lifecycle did not force a fresh validation"

# Retry empty validation on EKS and ARO; diagnostic failures must not abort.
# File-backed counters survive annotation reads in command substitutions.
ATTEMPTS_FILE=$(mktemp)
CLEARED_FILE=$(mktemp)
kubectl_cmd() {
  local n
  # Count validation reads, not annotation clearing.
  case " $* " in
    *" get nodeclaims "*) gpu_nodeclaim_read "$@" || return 1; return 0 ;;
    *" annotate deployment "*) printf 'cleared' > "${CLEARED_FILE}"; return 0 ;;
    *" get deployment "*) ;;
    *) return 0 ;;
  esac
  n=$(( $(cat "${ATTEMPTS_FILE}") + 1 ))
  printf '%s' "${n}" > "${ATTEMPTS_FILE}"
  if (( n >= 2 )); then printf 'hash@1'; fi
}
for PLATFORM in eks aro; do
  TRACE=()
  printf '0' > "${ATTEMPTS_FILE}"
  : > "${CLEARED_FILE}"
  : > "${NODECLAIM_READS}"
  stage_gpu_lifecycle_verify >/dev/null
  expected=(core-verify gpu-canary gpu-canary core-restore storage-deleted scale-zero)
  [[ "${TRACE[*]}" == "${expected[*]}" ]] ||
    die "GPU canary did not retry an empty validation on ${PLATFORM}: ${TRACE[*]}"
  [[ "$(cat "${CLEARED_FILE}")" == cleared ]] ||
    die "GPU canary did not drop the previous run's validation annotation"
  case "${PLATFORM}" in
    # Non-EKS retries must not query Karpenter NodeClaims.
    aro) [[ ! -s "${NODECLAIM_READS}" ]] ||
      die "the capacity diagnostic read Karpenter NodeClaims on ARO: $(cat "${NODECLAIM_READS}")" ;;
    # EKS queries only stack-labelled GPU NodeClaims.
    eks)
      [[ -s "${NODECLAIM_READS}" ]] ||
        die "the capacity diagnostic never read a NodeClaim on EKS"
      ! grep -qvF -- '-l kumo.ai/gpu-node=true' "${NODECLAIM_READS}" ||
        die "the capacity diagnostic read NodeClaims outside the stack's GPU pools: $(cat "${NODECLAIM_READS}")"
      ;;
  esac
done
PLATFORM=eks
rm -f "${ATTEMPTS_FILE}" "${CLEARED_FILE}"

# A failed annotation clear must stop before any canary sweep.
STALE_SWEEP_FILE=$(mktemp)
if (
  TRACE=()
  kubectl_cmd() {
    case " $* " in
      *" annotate deployment "*)
        printf 'Error from server (Forbidden): deployments.apps "kumo-poolcontroller" is forbidden\n' >&2
        return 1
        ;;
      *" get deployment "*) printf 'cache-small,cache-medium,cache-large,ephemeral' ;;
      *) return 0 ;;
    esac
  }
  helm_upgrade() { TRACE+=(gpu-canary); printf '%s\n' "${TRACE[*]}" >"${STALE_SWEEP_FILE}"; }
  stage_gpu_lifecycle_verify
) >/dev/null 2>&1; then
  die "an unclearable validation annotation let a previous install's result pass the GPU gate"
fi
[[ ! -s "${STALE_SWEEP_FILE}" ]] ||
  die "the GPU gate swept before proving the previous run's validation was cleared"
rm -f "${STALE_SWEEP_FILE}"

# Exhaustion restores core and reports the last EKS capacity error.
TRACE_FILE=$(mktemp)
FAILURE_FILE=$(mktemp)
(
  TRACE=()
  GPU_VERIFY_TIMEOUT=2s
  NODECLAIMS_READABLE=true
  NODECLAIM_JSON='{"items":[{"metadata":{"labels":{"karpenter.sh/nodepool":"kumo-cache-large"}},"status":{"conditions":[{"type":"Launched","status":"False","message":"InsufficientInstanceCapacity"}]}}]}'
  kubectl_cmd() {
    case " $* " in
      *" get nodeclaims "*) gpu_nodeclaim_read "$@" || return 1; return 0 ;;
    esac
    printf ''
  }
  stage_core_apply() { TRACE+=(core-restore); printf '%s\n' "${TRACE[*]}" > "${TRACE_FILE}"; }
  stage_gpu_lifecycle_verify >/dev/null 2>"${FAILURE_FILE}"
) || true
grep -q 'core-restore' "${TRACE_FILE}" ||
  die "GPU canary must restore the release even when the budget is exhausted"
grep -Fq 'last capacity error: kumo-cache-large: InsufficientInstanceCapacity' "${FAILURE_FILE}" ||
  die "the exhausted GPU canary did not name the pool that could not launch"
rm -f "${TRACE_FILE}" "${FAILURE_FILE}" "${NODECLAIM_READS}"

# A first core install generates one stable Fernet-compatible tenant key,
# while a pre-created enterprise Secret is validated and never overwritten.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/core.sh"
TENANT_NAMESPACE=kumo-acme
TENANT_SECRET=kumo-acme-secret-key
TENANT_SECRET_KEY=key
JQ="${JQ:-jq}"
OPENSSL="${OPENSSL:-openssl}"
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-tenant-secret-test.XXXXXX")
SECRET_EXISTS=false
CREATED_SECRET_FILE=""
kubectl_cmd() {
  case "${3:-}:${4:-}" in
    get:secret)
      [[ "${SECRET_EXISTS}" == true ]] || return 1
      printf '{"data":{"key":"c3RhYmxl"}}\n'
      ;;
    create:secret)
      local arg
      for arg in "$@"; do
        if [[ "${arg}" == --from-file="${TENANT_SECRET_KEY}"=* ]]; then
          CREATED_SECRET_FILE="${arg#--from-file=}"
          CREATED_SECRET_FILE="${CREATED_SECRET_FILE#*=}"
        fi
      done
      [[ -n "${CREATED_SECRET_FILE}" ]] ||
        die "tenant Secret was not created from a private file"
      grep -Eq '^[A-Za-z0-9_-]{43}=$' "${CREATED_SECRET_FILE}" ||
        die "generated tenant Secret is not a Fernet-compatible key"
      ;;
    *)
      die "unexpected tenant Secret kubectl call: $*"
      ;;
  esac
}

ensure_tenant_secret
[[ -n "${CREATED_SECRET_FILE}" ]] ||
  die "missing tenant Secret was not generated"
SECRET_EXISTS=true
CREATED_SECRET_FILE=""
ensure_tenant_secret
[[ -z "${CREATED_SECRET_FILE}" ]] ||
  die "existing tenant Secret was overwritten"
rm -rf "${WORK_DIR}"

grep -A8 '^stage_core_verify()' "${ROOT}/kumo-stack/stages/core.sh" |
  grep -Fq 'rollout status deployment' ||
  die "core verification does not wait for tenant Deployment rollouts"
grep -A8 '^stage_core_verify()' "${ROOT}/kumo-stack/stages/core.sh" |
  grep -Fq -- '--selector "tenant=${TENANT}"' ||
  die "core rollout wait is not scoped to the tenant"
! grep -A8 '^stage_core_verify()' "${ROOT}/kumo-stack/stages/core.sh" |
  grep -Fq 'get deployment -o name' ||
  die "core verification retained per-Deployment enumeration"
grep -A14 '^run_conformance()' "${ROOT}/kumo-stack/bin/kumo" |
  grep -Fq '"KUMO_CONFORMANCE_INSECURE_TLS=true"' ||
  die "AKS conformance does not disable edge certificate verification"
! grep -A14 '^run_conformance()' "${ROOT}/kumo-stack/bin/kumo" |
  grep -Fq 'CURL_CA_BUNDLE' ||
  die "AKS conformance retained the private-CA bundle workaround"

# Fresh-cluster acceptance permits only product images and the selected
# provider's official registries.
PLATFORM=aks
KUMO_CPU_IMAGE=registry.example.com/kumo-backend-cpu:sha
KUMO_GPU_IMAGE=registry.example.com/kumo-backend-gpu:sha
KUMO_UI_IMAGE=registry.example.com/kumo-ui:sha
TEMPORAL_SERVER_IMAGE=docker.io/temporalio/server:1.31.2
TEMPORAL_ADMIN_TOOLS_IMAGE=docker.io/temporalio/admin-tools:1.31.2
POSTGRES_CLIENT_IMAGE=docker.io/postgres:17.10-alpine@sha256:client
NVIDIA_DEVICE_PLUGIN_IMAGE=nvcr.io/nvidia/k8s-device-plugin:v0.19.0
KARPENTER_IMAGE=public.ecr.aws/karpenter/controller:1.12.0@sha256:test
AWS_LOAD_BALANCER_CONTROLLER_IMAGE=public.ecr.aws/eks/aws-load-balancer-controller:v2.14.0@sha256:58d83491b3eb25080e478bdd9a245b773fd9ceb29d9182fa4b198dee5ff31d22
POLICY_IMAGE=mcr.microsoft.com/oss/azure/workload-identity/webhook:v1
kubectl_cmd() {
  printf '{"items":[{"spec":{"containers":[{"image":"%s"}]}}]}\n' \
    "${POLICY_IMAGE}"
}
stage_container_policy_verify
# The pinned psql image is part of the stack. A policy check taken while the
# Temporal bootstrap Job is still running must not condemn the cluster for it.
POLICY_IMAGE="${POSTGRES_CLIENT_IMAGE}"
stage_container_policy_verify
POLICY_IMAGE=quay.io/jetstack/cert-manager-controller:v1.21.0
if (stage_container_policy_verify) 2>/dev/null; then
  die "fresh-cluster policy accepted a non-provider image"
fi

# The DiskGraph StorageClass is adopted like the database and the CSI drivers,
# so ownership is only asserted for a class the stack claims to create, and an
# adopted name that does not exist has to stop the install.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/common.sh"
JQ="${JQ:-jq}"
PLATFORM_NAMESPACE=kumo-system
STORAGE_CLASS=diskgraph-class
STORAGE_CLASS_JSON=""
STORAGE_CLASS_READABLE=true
STORAGE_CAPACITY_TRACKING=advisory
CSI_DRIVER_JSON='{"spec":{"storageCapacity":false}}'
CPU_NODE_SELECTOR=""
NODE_LIST_JSON='{"items":[]}'
NODES_READABLE=true
# Mirrors `kubectl get --ignore-not-found`: absent prints nothing and succeeds,
# while an unreadable API fails.
kubectl_cmd() {
  case "$2" in
    ingressclass)
      # Upstream's edge wiring, not the CPU pool contract these tests assert.
      ;;
    network.operator.openshift.io)
      # ARO reconciliation settles the OVN gateway before it touches any
      # MachineSet; that path is upstream's, not what these tests assert.
      printf '{"spec":{"defaultNetwork":{"ovnKubernetesConfig":{"gatewayConfig":{"routingViaHost":true}}}}}'
      ;;
    storageclass)
      [[ "$*" == *--ignore-not-found* ]] ||
        die "reading a StorageClass without --ignore-not-found cannot tell absent from unreadable"
      [[ "${STORAGE_CLASS_READABLE}" == true ]] || return 1
      printf '%s' "${STORAGE_CLASS_JSON}"
      ;;
    csidriver)
      [[ "$*" == *--ignore-not-found* ]] ||
        die "reading a CSIDriver without --ignore-not-found cannot tell absent from unreadable"
      printf '%s' "${CSI_DRIVER_JSON}"
      ;;
    nodes)
      [[ "$*" == *"-l ${CPU_NODE_SELECTOR// /,}"* ]] ||
        die "node lookup did not use the configured CPU selector: $*"
      [[ "${NODES_READABLE}" == true ]] || return 1
      printf '%s' "${NODE_LIST_JSON}"
      ;;
    openshift-machine-api)
      [[ "$4" == machineset ]] ||
        die "unexpected ARO kubectl call: $*"
      [[ "${MACHINESETS_READABLE}" == true ]] || return 1
      # Callers read these through command substitutions, so the record of what
      # was asked for and what was patched has to outlive those subshells.
      case "$3" in
        patch)
          local arg previous=""
          for arg in "$@"; do
            [[ "${previous}" != -p ]] ||
              printf '%s\n' "${arg}" >>"${MACHINESET_PATCHES}"
            previous="${arg}"
          done
          ;;
        get)
          printf '%s\n' "$5" >>"${SEEN_MACHINESETS}"
          if [[ "$*" == *jsonpath* ]]; then
            printf '%s' "${MACHINESET_VM_SIZE}"
          else
            "${JQ}" -nc --argjson labels "${MACHINESET_LABELS}" \
              --arg vm_size "${MACHINESET_VM_SIZE}" '{spec:{template:{spec:{
                metadata: {labels: $labels},
                providerSpec: {value: {vmSize: $vm_size}}
              }}}}'
          fi
          ;;
        *) die "unexpected ARO kubectl verb: $*" ;;
      esac
      ;;
    *) die "unexpected storage kubectl call: $*" ;;
  esac
}
MACHINESET_LABELS='{}'
MACHINESET_VM_SIZE=Standard_NC4as_T4_v3
MACHINESETS_READABLE=true
SEEN_MACHINESETS=$(mktemp "${TMPDIR:-/tmp}/kumo-machinesets.XXXXXX")
MACHINESET_PATCHES=$(mktemp "${TMPDIR:-/tmp}/kumo-patches.XXXXXX")
trap 'rm -f "${SEEN_MACHINESETS}" "${MACHINESET_PATCHES}"' EXIT
RELEASE_LISTABLE=true
RELEASE_STATUS=""
RELEASE_VALUES_READABLE=true
RELEASE_VALUES='{}'
# Mirrors bin/kumo: an unlistable release history is fatal rather than absent,
# and an absent release prints no status at all.
helm_release_status() {
  [[ "${RELEASE_LISTABLE}" == true ]] ||
    die "cannot list Helm releases in $2"
  printf '%s' "${RELEASE_STATUS}"
}
helm_release_values() {
  [[ "${RELEASE_VALUES_READABLE}" == true ]] || return 1
  printf '%s' "${RELEASE_VALUES}"
}
release_creates() {
  "${JQ}" -nc --arg name "$1" '{storageClass: {create: true, name: $name}}'
}
# Adversarial YAML ensures ownership never regresses to parsing rendered
# manifests; deployed release values are authoritative.
RELEASE_MANIFESTS=(
  "$(printf -- 'kind: StorageClass\nparameters:\n  name: %s\nmetadata:\n  name: other-class\n' "${STORAGE_CLASS}")"
  "$(printf -- 'apiVersion: storage.k8s.io/v1\nkind: StorageClass\nmetadata:\n  name: %s\nmetadata:\n  name: other-class\n' "${STORAGE_CLASS}")"
  "$(printf -- 'apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: %s\nkind: StorageClass\nmetadata:\n  name: other-class\n' "${STORAGE_CLASS}")"
)
helm_release_manifest() {
  die "StorageClass ownership must not be decided by parsing a rendered manifest: ${RELEASE_MANIFESTS[0]}"
}
conforming_class='"provisioner":"topolvm.io","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete","parameters":{"csi.storage.k8s.io/fstype":"xfs"}'
owned_by() {
  printf '{%s,"metadata":{"annotations":{"meta.helm.sh/release-name":"%s","meta.helm.sh/release-namespace":"%s"}}}' \
    "${conforming_class}" "$1" "$2"
}
assert_rejected() {
  local description="$1"
  shift
  if ("$@") >/dev/null 2>&1; then
    die "${description}"
  fi
}

STORAGE_CLASS_CREATE=false
STORAGE_CLASS_JSON=""
assert_rejected "a missing adopted StorageClass was accepted" \
  verify_storage_class_ownership kumo-platform-eks
STORAGE_CLASS_JSON=$(owned_by customer-platform infra)
verify_storage_class_ownership kumo-platform-eks ||
  die "an operator-owned StorageClass was not adoptable"
# Adopting the class the platform release still renders would let the next
# upgrade delete it out from under the tenant.
STORAGE_CLASS_JSON=$(owned_by kumo-platform-eks kumo-system)
assert_rejected "the platform release's own StorageClass was adopted" \
  verify_storage_class_ownership kumo-platform-eks
# The annotations that carry that signal are writable by anything that can edit
# the class, so stripping them must not turn a class the release still creates
# into an adoptable one. The upgrade that drops it from the release's manifest
# deletes it either way.
STORAGE_CLASS_JSON=$(printf '{%s}' "${conforming_class}")
RELEASE_STATUS=deployed
RELEASE_VALUES=$("${JQ}" -nc --arg name "${STORAGE_CLASS}" \
  '{storageClass: {create: true, name: $name}}')
assert_rejected "an unannotated class the release still creates was adopted" \
  verify_storage_class_ownership kumo-platform-eks
# An interrupted release could render anything, so adoption cannot proceed
# against one either.
RELEASE_STATUS=pending-upgrade
assert_rejected "a class was adopted against an interrupted platform release" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_LISTABLE=false
assert_rejected "a class was adopted without reading the release state" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_LISTABLE=true
# Adoption reads a false predicate as "the release creates nothing of that
# name", so values that cannot be read as one document must not reach it.
RELEASE_STATUS=deployed
for RELEASE_VALUES in \
  '' \
  '   ' \
  'not json' \
  '{"storageClass":{"create":true,"name":"diskgraph-class"}}{"other":1}' \
  '{"other":1}{"storageClass":{"create":true,"name":"diskgraph-class"}}'; do
  assert_rejected "unreadable release values let a class be adopted" \
    verify_storage_class_ownership kumo-platform-eks
done
# A release that genuinely creates no such class is adoptable.
RELEASE_VALUES='null'
verify_storage_class_ownership kumo-platform-eks ||
  die "a class was not adoptable against a release holding no values"
RELEASE_VALUES='{}'
verify_storage_class_ownership kumo-platform-eks ||
  die "a class was not adoptable against a release that creates none"
RELEASE_STATUS=""
RELEASE_VALUES='{}'
STORAGE_CLASS_JSON='{"provisioner":"topolvm.io","volumeBindingMode":"Immediate","reclaimPolicy":"Delete","parameters":{"csi.storage.k8s.io/fstype":"xfs"}}'
assert_rejected "an adopted StorageClass skipped the DiskGraph contract" \
  verify_storage_class_ownership kumo-platform-eks
STORAGE_CLASS_JSON=$(owned_by customer-platform infra)
CSI_DRIVER_JSON=""
assert_rejected "an adopted StorageClass with no registered CSI driver was accepted" \
  verify_storage_class_ownership kumo-platform-eks
CSI_DRIVER_JSON='{"spec":{"storageCapacity":false}}'
# capacityTracking=required makes the pool controller wait for capacity objects,
# so the driver behind the class has to publish them.
STORAGE_CAPACITY_TRACKING=required
assert_rejected "capacityTracking=required accepted a driver without capacity" \
  verify_storage_class_ownership kumo-platform-eks
CSI_DRIVER_JSON='{"spec":{"storageCapacity":true}}'
verify_storage_class_ownership kumo-platform-eks ||
  die "capacityTracking=required rejected a capacity-publishing driver"
# The class a platform release is about to create does not exist yet, so its
# cloud driver is proven directly rather than through the class.
CSI_DRIVER_JSON='{"spec":{"storageCapacity":false}}'
assert_rejected "a cloud driver without capacity backed capacityTracking=required" \
  verify_csi_driver ebs.csi.aws.com
CSI_DRIVER_JSON=""
assert_rejected "an unregistered cloud CSI driver was accepted" \
  verify_csi_driver ebs.csi.aws.com
STORAGE_CAPACITY_TRACKING=advisory
CSI_DRIVER_JSON='{"spec":{"storageCapacity":false}}'
verify_csi_driver ebs.csi.aws.com ||
  die "capacityTracking=advisory demanded a capacity-publishing driver"
# A denied or unreachable API must never read as an absent class.
STORAGE_CLASS_READABLE=false
assert_rejected "an unreadable StorageClass was treated as missing" \
  verify_storage_class_ownership kumo-platform-eks
STORAGE_CLASS_READABLE=true

STORAGE_CLASS_CREATE=true
STORAGE_CLASS_JSON=""
verify_storage_class_ownership kumo-platform-eks ||
  die "a StorageClass the platform release will create was rejected"
STORAGE_CLASS_READABLE=false
assert_rejected "an unreadable StorageClass passed the ownership guard" \
  verify_storage_class_ownership kumo-platform-eks
STORAGE_CLASS_READABLE=true
STORAGE_CLASS_JSON=$(owned_by customer-platform infra)
assert_rejected "a foreign StorageClass was claimed as stack-created" \
  verify_storage_class_ownership kumo-platform-eks
# Mutable Helm annotations alone must not prove StorageClass ownership.
STORAGE_CLASS_JSON=$(owned_by kumo-platform-eks kumo-system)
RELEASE_STATUS=""
assert_rejected "an annotated StorageClass was adopted before its release existed" \
  verify_storage_class_ownership kumo-platform-eks
# Only a deployed release owns anything definite. An uninstalled one already had
# its resources deleted, and an interrupted one has an unknown resource set, so
# neither can prove the live class is the one it created.
RELEASE_VALUES=$(release_creates "${STORAGE_CLASS}")
for RELEASE_STATUS in \
  uninstalled failed uninstalling superseded unknown \
  pending-install pending-upgrade pending-rollback; do
  assert_rejected "a ${RELEASE_STATUS} release proved StorageClass ownership" \
    verify_storage_class_ownership kumo-platform-eks
done
RELEASE_STATUS=deployed
RELEASE_VALUES=$(release_creates other-class)
assert_rejected "a release creating another class was claimed as this one's owner" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_VALUES=$("${JQ}" -nc --arg name "${STORAGE_CLASS}" \
  '{storageClass: {create: false, name: $name}}')
assert_rejected "a release that adopts its class was claimed to create it" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_VALUES='null'
assert_rejected "a release with no values proved StorageClass ownership" \
  verify_storage_class_ownership kumo-platform-eks
# `jq -e` exits 0 on empty input, so an empty values document would otherwise
# read as a satisfied predicate.
RELEASE_VALUES=''
assert_rejected "empty release values proved StorageClass ownership" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_VALUES='   '
assert_rejected "blank release values proved StorageClass ownership" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_VALUES=$(release_creates "${STORAGE_CLASS}")
verify_storage_class_ownership kumo-platform-eks ||
  die "the platform release's own StorageClass was rejected"
RELEASE_VALUES_READABLE=false
assert_rejected "unreadable release values passed the ownership guard" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_VALUES_READABLE=true
RELEASE_LISTABLE=false
assert_rejected "an unreadable release history passed the ownership guard" \
  verify_storage_class_ownership kumo-platform-eks
RELEASE_LISTABLE=true

verify_cpu_node_selector ||
  die "an empty CPU node selector was rejected"
CPU_NODE_SELECTOR="kumo.ai/node-pool=cpu topology.kubernetes.io/zone=a"
assert_rejected "a CPU node selector no node carries was accepted" \
  verify_cpu_node_selector
NODE_LIST_JSON='{"items":[{"metadata":{"name":"worker-0"}}]}'
verify_cpu_node_selector ||
  die "a CPU node selector the cluster already carries was rejected"
NODES_READABLE=false
assert_rejected "an unreadable node list passed the CPU selector check" \
  verify_cpu_node_selector
NODES_READABLE=true

# Some CPU node matching is not enough: capacity the selector also matches is
# capacity core can be scheduled onto.
verify_cpu_selector_excludes "GPU capacity" \
  '{"kumo.ai/node-pool":"cpu"}' ||
  die "a selector only partly matched by GPU labels was rejected"
verify_cpu_selector_excludes "GPU capacity" \
  '{"kumo.ai/node-pool":"cpu","topology.kubernetes.io/zone":"b"}' ||
  die "a selector with one differing label value was rejected"
assert_rejected "a CPU selector every GPU label satisfies was accepted" \
  verify_cpu_selector_excludes "GPU capacity" \
  '{"kumo.ai/node-pool":"cpu","topology.kubernetes.io/zone":"a","kumo.ai/gpu-node":"true"}'
# jq exits 1 for a false comparison and 2 or more for a real error. Only the
# first may read as "this capacity is out of the selector's reach".
assert_rejected "unreadable labels read as GPU capacity the selector excludes" \
  verify_cpu_selector_excludes "GPU capacity" 'not json'
CPU_NODE_SELECTOR=""
verify_cpu_selector_excludes "GPU capacity" '{"kumo.ai/gpu-node":"true"}' ||
  die "an empty CPU selector, which targets nothing, was checked for exclusion"

# ARO adopts its GPU MachineSets, so they keep labels this stack never chose. A
# CPU selector those labels satisfy puts core on GPU capacity however many CPU
# nodes also match it, which the live-node check alone cannot see.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/aro.sh"
ARO_CACHE_MACHINESETS=aro-gpu-small,aro-gpu-large
ARO_EPHEMERAL_MACHINESET=aro-gpu-ephemeral
GPU_CACHE_TIERS='small,Standard_NC4as_T4_v3,256,1;large,Standard_NC16as_T4_v3,1024,1'

CPU_NODE_SELECTOR="node-role.kubernetes.io/worker="
MACHINESET_LABELS='{"node-role.kubernetes.io/worker":""}'
assert_rejected "a CPU selector every GPU MachineSet label satisfies was accepted" \
  verify_aro_gpu_exclusion
MACHINESET_LABELS='{"node-role.kubernetes.io/infra":""}'
: >"${SEEN_MACHINESETS}"
verify_aro_gpu_exclusion ||
  die "a CPU selector no GPU MachineSet satisfies was rejected"
checked=$(tr '\n' ' ' <"${SEEN_MACHINESETS}")
[[ "${checked}" == "aro-gpu-small aro-gpu-large aro-gpu-ephemeral " ]] ||
  die "GPU exclusion checked ${checked}"
# The labels this module is about to stamp count too, not just the ones the
# MachineSet already carries, so the exclusion check and the reconcile patch
# have to agree on exactly what those are.
[[ "$(aro_gpu_labels 'small,Standard_NC4as_T4_v3,256,1')" == \
  '{"kumo.ai/gpu-node":"true","kumo.ai/diskgraph-pool-type":"cache","kumo.ai/diskgraph-ssd-size":"256","kumo.ai/diskgraph-gpu-count":"1","cluster-api/accelerator":"nvidia-t4"}' ]] ||
  die "ARO cache labels are $(aro_gpu_labels 'small,Standard_NC4as_T4_v3,256,1')"
[[ "$(aro_gpu_labels '')" == \
  '{"kumo.ai/gpu-node":"true","kumo.ai/diskgraph-pool-type":"ephemeral","cluster-api/accelerator":"nvidia-t4"}' ]] ||
  die "ARO ephemeral labels are $(aro_gpu_labels '')"
for stamped in kumo.ai/gpu-node=true kumo.ai/diskgraph-pool-type=cache \
  kumo.ai/diskgraph-ssd-size=256 kumo.ai/diskgraph-gpu-count=1 \
  cluster-api/accelerator=nvidia-t4; do
  CPU_NODE_SELECTOR="${stamped}"
  MACHINESET_LABELS='{}'
  assert_rejected "a CPU selector matching stamped label ${stamped} was accepted" \
    verify_aro_gpu_exclusion
done
CPU_NODE_SELECTOR="kumo.ai/diskgraph-pool-type=ephemeral"
MACHINESET_LABELS='{}'
assert_rejected "a CPU selector matching a label the stack stamps was accepted" \
  verify_aro_gpu_exclusion
# kubelet populates these on the GPU nodes too, so they cannot be the thing
# that makes a selector look like it excludes GPU capacity.
for kubelet_label in kubernetes.io/os=linux kubernetes.io/arch=amd64 \
  node.kubernetes.io/instance-type=Standard_NC4as_T4_v3; do
  CPU_NODE_SELECTOR="${kubelet_label}"
  MACHINESET_LABELS='{}'
  assert_rejected "a CPU selector of universal label ${kubelet_label} was accepted" \
    verify_aro_gpu_exclusion
done
# OpenShift gives every RHCOS worker this one, GPU workers included.
CPU_NODE_SELECTOR="node.openshift.io/os_id=rhcos"
MACHINESET_LABELS='{}'
assert_rejected "a CPU selector of the RHCOS node label was accepted" \
  verify_aro_gpu_exclusion
CPU_NODE_SELECTOR="node.kubernetes.io/instance-type=Standard_D4as_v5"
verify_aro_gpu_exclusion ||
  die "a CPU selector naming a non-GPU instance type was rejected"

# The exclusion check is only as good as its claim about what reconciliation
# stamps, so drive the real reconcile path and read the patches it sends.
ARO_CACHE_MACHINESETS=aro-gpu-small
ARO_EPHEMERAL_MACHINESET=aro-gpu-ephemeral
GPU_CACHE_TIERS='small,Standard_NC4as_T4_v3,256,1'
GPU_EPHEMERAL_INSTANCE_TYPE=Standard_NC4as_T4_v3
: >"${MACHINESET_PATCHES}"
provider_prepare_cloud ||
  die "ARO reconciliation of its GPU MachineSets failed"
patched=$("${JQ}" -sc 'map(.spec.template.spec.metadata.labels)' \
  "${MACHINESET_PATCHES}")
[[ "${patched}" == '[{"kumo.ai/gpu-node":"true","kumo.ai/diskgraph-pool-type":"cache","kumo.ai/diskgraph-ssd-size":"256","kumo.ai/diskgraph-gpu-count":"1","cluster-api/accelerator":"nvidia-t4"},{"kumo.ai/gpu-node":"true","kumo.ai/diskgraph-pool-type":"ephemeral","cluster-api/accelerator":"nvidia-t4"}]' ]] ||
  die "ARO reconciliation stamps ${patched}"
MACHINESETS_READABLE=false
CPU_NODE_SELECTOR="node-role.kubernetes.io/worker="
assert_rejected "an unreadable MachineSet passed the GPU exclusion check" \
  verify_aro_gpu_exclusion
MACHINESETS_READABLE=true
CPU_NODE_SELECTOR=""
verify_aro_gpu_exclusion ||
  die "an empty CPU selector was checked against GPU MachineSets"

# Platform verification proves the properties the DiskGraph cache depends on,
# not the identity of the driver that provides them.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/stages/platform.sh"
[[ "$(grep -c -- '--for=create' "${ROOT}/kumo-stack/stages/platform.sh")" == 2 ]] ||
  die "ARO platform installation does not wait for both operator CRDs to be created"
helm_release_is_deployed() { :; }
load_provider() { :; }
provider_verify_platform() { :; }
PLATFORM=eks
STORAGE_CLASS_JSON=$(printf '{%s}' "${conforming_class}")
stage_platform_verify ||
  die "a conforming local-volume StorageClass failed platform verification"
STORAGE_CLASS_JSON='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete","parameters":{"fsType":"xfs"}}'
stage_platform_verify ||
  die "a conforming cloud-disk StorageClass failed platform verification"
STORAGE_CLASS_JSON='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"Immediate","reclaimPolicy":"Delete","parameters":{"csi.storage.k8s.io/fstype":"xfs"}}'
assert_rejected "an eagerly bound StorageClass passed platform verification" \
  stage_platform_verify
STORAGE_CLASS_JSON='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Retain","parameters":{"csi.storage.k8s.io/fstype":"xfs"}}'
assert_rejected "a retaining StorageClass passed platform verification" \
  stage_platform_verify
STORAGE_CLASS_JSON='{"provisioner":"ebs.csi.aws.com","volumeBindingMode":"WaitForFirstConsumer","reclaimPolicy":"Delete","parameters":{"csi.storage.k8s.io/fstype":"ext4"}}'
assert_rejected "a non-xfs StorageClass passed platform verification" \
  stage_platform_verify
STORAGE_CLASS_JSON=""
assert_rejected "a missing StorageClass passed platform verification" \
  stage_platform_verify
STORAGE_CLASS_JSON=$(printf '{%s}' "${conforming_class}")
STORAGE_CLASS_READABLE=false
assert_rejected "an unreadable StorageClass passed platform verification" \
  stage_platform_verify
STORAGE_CLASS_READABLE=true

# AKS builds its own CPU capacity, so the pool it reconciles and the pool it
# verifies both carry the selector core is rendered with. A literal here would
# leave core Pending on any cluster whose CPU nodes are labelled differently.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/aks.sh"
CPU_NODE_SELECTOR="kumo.ai/instance-type=kumo-default"
AKS_CPU_POOL_NAME=kumocpu
AKS_CPU_POOL_INSTANCE_TYPE=Standard_D8as_v5
PROVIDER_RESOURCE_GROUP=test-aks-rg
PROVIDER_CLUSTER=test-aks
AKS_APPLICATION_GATEWAY_ID=/subscriptions/0/gw
AKS_CACHE_POOL_NAMES=kgcsmall
AKS_EPHEMERAL_POOL_NAME=kumogeph
GPU_CACHE_TIERS='small,Standard_NC4as_T4_v3,256,1'
GPU_EPHEMERAL_INSTANCE_TYPE=Standard_NC4as_T4_v3
GPU_MAX_NODES=2
CPU_POOL_LABELS=""
ensure_aks_application_gateway_addon() { :; }
ensure_aks_edge_tls_secret() { :; }
verify_aks_application_gateway() { :; }
verify_aks_application_gateway_addon() { :; }
az() { printf 'false\n'; }
ensure_aks_pool() {
  [[ "$1" != "${AKS_CPU_POOL_NAME}" ]] || CPU_POOL_LABELS="$5"
}
verify_aks_pool() { ensure_aks_pool "$@"; }
verify_no_gpu_nodes() { :; }

provider_prepare_cloud
[[ "${CPU_POOL_LABELS}" == "${CPU_NODE_SELECTOR}" ]] ||
  die "AKS creates its CPU pool with labels ${CPU_POOL_LABELS}"
CPU_POOL_LABELS=""
provider_verify_platform
[[ "${CPU_POOL_LABELS}" == "${CPU_NODE_SELECTOR}" ]] ||
  die "AKS verifies its CPU pool against labels ${CPU_POOL_LABELS}"

# Both CPU-selector gates have to be wired into ARO preflight, not merely
# defined: they run before any mutation, which is the only point where a
# selector that GPU capacity also satisfies can still be refused cheaply.
# shellcheck source=/dev/null
source "${ROOT}/kumo-stack/providers/aro.sh"
TRACE=()
az() { :; }
kubectl_cmd() {
  # Upstream's own preflight prerequisites, answered so that the wiring
  # assertions below are what this test turns on.
  case "$2" in
    clusterversion) printf '{"status":{"desired":{"version":"4.20.3"}}}' ;;
    *) printf '{"status":{"availableReplicas":2}}' ;;
  esac
}
STORAGE_CLASS_CREATE=false
ARO_OPERATOR_MODE=dedicated
# An ARO install renders the Azure shape of the Temporal bootstrap identity, and
# preflight shape-checks it ahead of the gates asserted below.
POSTGRES_ADMIN_ANNOTATION_KEY=azure.workload.identity/client-id
POSTGRES_ADMIN_ANNOTATION_VALUE=11111111-2222-3333-4444-555555555555
verify_storage_class_ownership() { TRACE+=(ownership); }
verify_cpu_node_selector() { TRACE+=(cpu-selector); }
verify_aro_gpu_exclusion() { TRACE+=(gpu-exclusion); }
verify_azure_postgres() { TRACE+=(postgres); }
provider_preflight
expected=(ownership cpu-selector gpu-exclusion postgres)
[[ "${TRACE[*]}" == "${expected[*]}" ]] ||
  die "ARO preflight order is ${TRACE[*]}"

# Exercise the real release readers against fake Helm; assert behavior rather
# than source text.
FAKE_BIN=$(mktemp -d "${TMPDIR:-/tmp}/kumo-fake-helm.XXXXXX")
trap 'rm -rf "${SEEN_MACHINESETS}" "${MACHINESET_PATCHES}" "${FAKE_BIN}"' EXIT
cat >"${FAKE_BIN}/helm" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"${FAKE_HELM_CALLS}"
case "$1 $2" in
  "version --short")
    [[ -n "${FAKE_HELM_VERSION}" ]] || exit 1
    printf '%s' "${FAKE_HELM_VERSION}"
    ;;
  "list --kube-context")
    [[ "${FAKE_HELM_LIST}" != unreadable ]] || exit 1
    printf '%s' "${FAKE_HELM_LIST}"
    ;;
  "get values")
    [[ "${FAKE_HELM_VALUES}" != unreadable ]] || exit 1
    printf '%s' "${FAKE_HELM_VALUES}"
    ;;
  "upgrade --kube-context") ;;
  *) exit 1 ;;
esac
SH
chmod +x "${FAKE_BIN}/helm"
FAKE_HELM_CALLS="${FAKE_BIN}/calls"
export FAKE_HELM_LIST FAKE_HELM_VALUES FAKE_HELM_VERSION FAKE_HELM_CALLS
HELM="${FAKE_BIN}/helm"
CONTEXT=test-context
# shellcheck source=/dev/null
eval "$(sed -n '/^helm_release_status() {/,/^}/p' "${ROOT}/kumo-stack/bin/kumo")"
eval "$(sed -n '/^helm_release_values() {/,/^}/p' "${ROOT}/kumo-stack/bin/kumo")"

# The whole invocation is asserted, not just the flag that differs by major:
# the context and namespace decide which cluster and release answer at all.
assert_helm_called() {
  local description="$1" invocation="$2"
  grep -Fxq -- "${invocation}" "${FAKE_HELM_CALLS}" ||
    die "${description}; helm was called as: $(tr '\n' '|' <"${FAKE_HELM_CALLS}")"
}
# Helm 3 hides everything but deployed and failed unless asked for `--all`;
# Helm 4 lists that same set by default and rejects the flag outright.
FAKE_HELM_LIST='[{"name":"kumo-platform-eks","status":"pending-upgrade"}]'
FAKE_HELM_VERSION='v3.19.0+g3d8990f'
: >"${FAKE_HELM_CALLS}"
[[ "$(helm_release_status kumo-platform-eks kumo-system)" == pending-upgrade ]] ||
  die "the release reader did not report a listed status"
assert_helm_called "the Helm 3 listing is wrong" \
  'list --kube-context test-context --namespace kumo-system --all --filter ^kumo-platform-eks$ --output json'
FAKE_HELM_VERSION='v4.2.1+gd591a19'
: >"${FAKE_HELM_CALLS}"
[[ "$(helm_release_status kumo-platform-eks kumo-system)" == pending-upgrade ]] ||
  die "the release reader did not report a listed status on Helm 4"
assert_helm_called "the Helm 4 listing is wrong" \
  'list --kube-context test-context --namespace kumo-system --filter ^kumo-platform-eks$ --output json'
# A neighbouring release must not answer for this one; --filter is a regex.
FAKE_HELM_LIST='[{"name":"kumo-platform-eks-canary","status":"deployed"}]'
[[ -z "$(helm_release_status kumo-platform-eks kumo-system)" ]] ||
  die "the release reader accepted a different release's status"
FAKE_HELM_LIST='[]'
[[ -z "$(helm_release_status kumo-platform-eks kumo-system)" ]] ||
  die "a release absent from an exhaustive listing did not read as absent"
FAKE_HELM_LIST=unreadable
assert_rejected "an unlistable release history read as a definite answer" \
  helm_release_status kumo-platform-eks kumo-system
# A listing that is not one JSON array is no answer at all, and must not read
# as a release that was never installed.
for FAKE_HELM_LIST in '' '   ' '{}' '{"releases":[]}' '[] []' 'not json'; do
  assert_rejected "release listing '${FAKE_HELM_LIST}' read as a definite answer" \
    helm_release_status kumo-platform-eks kumo-system
done
FAKE_HELM_LIST='[]'
FAKE_HELM_LIST='[]'
FAKE_HELM_VERSION=""
assert_rejected "an unreadable Helm version read as a definite answer" \
  helm_release_status kumo-platform-eks kumo-system
# A version string that cannot be placed as major 3 or 4 must stop the install
# rather than fall through to a branch that lists fewer statuses than it needs.
for FAKE_HELM_VERSION in 'helm v3.19.0' '3.19.0' 'v5.0.0' 'weird'; do
  assert_rejected "Helm version '${FAKE_HELM_VERSION}' was guessed at" \
    helm_release_status kumo-platform-eks kumo-system
done
FAKE_HELM_VERSION='v4.2.1+gd591a19'

# The values reader is what decides ownership, so prove it reads the release's
# computed values as JSON rather than merely mentioning the flags that do.
FAKE_HELM_VALUES='{"storageClass":{"create":true,"name":"diskgraph-class"}}'
: >"${FAKE_HELM_CALLS}"
[[ "$(helm_release_values kumo-platform-eks kumo-system)" == "${FAKE_HELM_VALUES}" ]] ||
  die "the values reader did not return the release's own values"
assert_helm_called "the values reader invocation is wrong" \
  'get values --kube-context test-context kumo-platform-eks --namespace kumo-system --all --output json'
FAKE_HELM_VALUES=unreadable
assert_rejected "unreadable release values read as a definite answer" \
  helm_release_values kumo-platform-eks kumo-system
FAKE_HELM_VALUES='{}'

# Exercise the shipped helm_upgrade wrapper and assert whether fake Helm runs.
# shellcheck source=/dev/null
eval "$(sed -n '/^require_wait_capable_helm() {/,/^}/p' "${ROOT}/kumo-stack/bin/kumo")"
# shellcheck source=/dev/null
eval "$(sed -n '/^helm_upgrade() {/,/^}/p' "${ROOT}/kumo-stack/bin/kumo")"
# Run each case in a subshell to reset the memoized version check.
# shellcheck disable=SC2034  # read by the eval'd require_wait_capable_helm
HELM_WAIT_CHECKED=""
helm_wait_gate_verdict() {
  FAKE_HELM_VERSION="$1"
  : >"${FAKE_HELM_CALLS}"
  if (helm_upgrade acme chart --namespace kumo-acme --wait) >/dev/null 2>&1; then
    grep -q '^upgrade ' "${FAKE_HELM_CALLS}" && printf 'ran' || printf 'no-helm'
  else
    printf 'refused'
  fi
}
# Reject only 4.2.1; allow adjacent releases, Helm 3, and unparseable versions.
for gate_case in \
  "4.2.0:ran" \
  "4.2.1:refused" \
  "v4.2.1+gd591a19:refused" \
  "4.2.2:ran" \
  "4.2.10:ran" \
  "v3.19.0+g3d8990f:ran" \
  "helm version 4.2.1:ran" \
  ":ran"; do
  gate_version="${gate_case%:*}"
  gate_expected="${gate_case##*:}"
  gate_verdict=$(helm_wait_gate_verdict "${gate_version}")
  [[ "${gate_verdict}" == "${gate_expected}" ]] ||
    die "the wait gate ${gate_verdict} Helm '${gate_version}', expected ${gate_expected}"
done
# Unparseable versions proceed with a warning.
FAKE_HELM_VERSION='helm version 4.2.1'
: >"${FAKE_HELM_CALLS}"
gate_warning=$(helm_upgrade acme chart --wait 2>&1 >/dev/null) ||
  die "an unreadable Helm version refused the install"
[[ "${gate_warning}" == *"helm version 4.2.1"* &&
   "${gate_warning}" == *"without the Helm 4.2.1 check"* ]] ||
  die "an unreadable Helm version proceeded without saying so: ${gate_warning}"
FAKE_HELM_VERSION='v4.2.1+gd591a19'

# The schema and the template have to agree on the CPU discriminator, or a
# validator that only reads the schema accepts a selector Helm then refuses.
"${JQ}" -e '
  [.allOf[] | select(.if.properties.platform.enum == ["eks", "aks"])][0]
  | .then.properties.cluster.properties.cpuNodeSelector
  | .required == ["kumo.ai/node-pool"]
    and .properties["kumo.ai/node-pool"].const == "cpu"
' "${ROOT}/kumo-stack/renderer/values.schema.json" >/dev/null ||
  die "the renderer schema does not require the CPU node-pool discriminator"

grep -A18 '^uninstall_tenant()' "${ROOT}/kumo-stack/bin/kumo" |
  grep -Fq 'diskgraph-activity-pod' ||
  die "tenant uninstall does not select runtime-created resources"

rm -f "${APPLIED_LOG}"

echo "Kumo staged orchestration tests: passed"
