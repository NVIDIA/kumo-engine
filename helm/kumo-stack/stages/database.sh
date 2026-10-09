#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared non-secret Job assets; credentials are injected separately.
TEMPORAL_DATABASE_ARTIFACTS=kumo-temporal-bootstrap
TEMPORAL_DATABASE_SERVICE_ACCOUNT=kumo-temporal-bootstrap
TEMPORAL_DATABASE_ARTIFACT_DIR=/opt/kumo/bootstrap
TEMPORAL_DATABASE_CREDENTIAL_DIR=/run/kumo
# Label stage-owned artifacts so decommission can reclaim interrupted installs.
TEMPORAL_DATABASE_LABEL_KEY=kumo.ai/stage-probe
TEMPORAL_DATABASE_LABEL_VALUE=database

# Resources owned by this invocation and removed by the EXIT trap.
TEMPORAL_DATABASE_ACTIVE_JOB=""
TEMPORAL_DATABASE_OWNED_CONFIGMAP=""
TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT=""

# Record only a state DB ServiceAccount created here; kumo-control may pre-exist.
STATE_DATABASE_ACTIVE_JOB=""
STATE_DATABASE_OWNED_SERVICE_ACCOUNT=""

ensure_temporal_secret() {
  if kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get secret \
    "${POSTGRES_SECRET}" >/dev/null 2>&1; then
    return
  fi
  [[ "${POSTGRES_BOOTSTRAP}" == true ]] ||
    die "Secret ${TEMPORAL_NAMESPACE}/${POSTGRES_SECRET} must be pre-created"

  # Piped straight into the Secret: the generated password never lands in a
  # shell variable, an argument list, or a file on the operator's machine.
  "${OPENSSL}" rand -base64 48 | tr -d '\n' |
    kubectl_cmd -n "${TEMPORAL_NAMESPACE}" create secret generic \
      "${POSTGRES_SECRET}" \
      "--from-file=${POSTGRES_SECRET_KEY}=/dev/stdin" >/dev/null
}

require_temporal_pull_secrets() {
  local secret
  for secret in "${KUMO_PULL_SECRET}" "${POSTGRES_CLIENT_PULL_SECRET}"; do
    [[ -n "${secret}" ]] || continue
    kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get secret "${secret}" \
      >/dev/null 2>&1 ||
      die "Temporal database Jobs need image pull Secret ${TEMPORAL_NAMESPACE}/${secret}"
  done
}

temporal_database_pull_secrets() {
  local secret fragment="" seen=""
  for secret in "${KUMO_PULL_SECRET}" "${POSTGRES_CLIENT_PULL_SECRET}"; do
    if [[ -n "${secret}" && " ${seen} " != *" ${secret} "* ]]; then
      seen+=" ${secret}"
      fragment+=$(printf '\n        - name: %s' "${secret}")
    fi
  done
  [[ -z "${fragment}" ]] || printf '      imagePullSecrets:%s' "${fragment}"
}

# Distinguish NotFound from an unreadable API so customer-owned objects are
# never adopted accidentally.
kubectl_object_exists() {
  local error
  if error=$(kubectl_cmd "$@" -o name 2>&1 >/dev/null); then
    return 0
  fi
  [[ "${error}" == *NotFound* ]] || die "cannot read $*: ${error}"
  return 1
}

# Delete credentials-bearing resources first. Retain failed names so the EXIT
# trap can retry them; successful cleanup remains idempotent.
remove_temporal_database_artifacts() {
  local status=0
  if [[ -n "${TEMPORAL_DATABASE_ACTIVE_JOB}" ]]; then
    if kubectl_cmd -n "${TEMPORAL_NAMESPACE}" delete job \
      "${TEMPORAL_DATABASE_ACTIVE_JOB}" --ignore-not-found --wait=true \
      >/dev/null; then
      TEMPORAL_DATABASE_ACTIVE_JOB=""
    else
      status=1
    fi
  fi
  if [[ -n "${TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT}" ]]; then
    if kubectl_cmd -n "${TEMPORAL_NAMESPACE}" delete serviceaccount \
      "${TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT}" --ignore-not-found \
      --wait=true >/dev/null; then
      TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT=""
    else
      status=1
    fi
  fi
  if [[ -n "${TEMPORAL_DATABASE_OWNED_CONFIGMAP}" ]]; then
    if kubectl_cmd -n "${TEMPORAL_NAMESPACE}" delete configmap \
      "${TEMPORAL_DATABASE_OWNED_CONFIGMAP}" --ignore-not-found --wait=true \
      >/dev/null; then
      TEMPORAL_DATABASE_OWNED_CONFIGMAP=""
    else
      status=1
    fi
  fi
  return "${status}"
}

# State DB counterpart; nonempty ownership records make it safe before runtime
# namespace values are loaded.
remove_state_database_artifacts() {
  local status=0
  if [[ -n "${STATE_DATABASE_ACTIVE_JOB}" ]]; then
    if kubectl_cmd -n "${TENANT_NAMESPACE}" delete job \
      "${STATE_DATABASE_ACTIVE_JOB}" --ignore-not-found --wait=true \
      >/dev/null; then
      STATE_DATABASE_ACTIVE_JOB=""
    else
      status=1
    fi
  fi
  if [[ -n "${STATE_DATABASE_OWNED_SERVICE_ACCOUNT}" ]]; then
    if kubectl_cmd -n "${TENANT_NAMESPACE}" delete serviceaccount \
      "${STATE_DATABASE_OWNED_SERVICE_ACCOUNT}" --ignore-not-found \
      --wait=true >/dev/null; then
      STATE_DATABASE_OWNED_SERVICE_ACCOUNT=""
    else
      status=1
    fi
  fi
  return "${status}"
}

# Decommission removes labelled remnants left when no shell survived to clean up.
remove_temporal_database_remnants() {
  kubectl_object_exists get namespace "${TEMPORAL_NAMESPACE}" || return 0
  kubectl_cmd -n "${TEMPORAL_NAMESPACE}" delete \
    job,configmap,serviceaccount \
    -l "${TEMPORAL_DATABASE_LABEL_KEY}=${TEMPORAL_DATABASE_LABEL_VALUE}" \
    --ignore-not-found --wait=true >/dev/null
}

# Run inside the cluster for private-network reachability and workload identity.
# A Kumo init container mints credentials for the SDK-free psql image.
run_temporal_database_job() {
  local job="$1" mode="$2" timeout="$3"
  local service_account="${TEMPORAL_DATABASE_SERVICE_ACCOUNT}"
  local service_account_json existing_annotation existing_label
  local service_account_label="" pod_identity_label image_pull_secrets deadline
  local pod_security_context client_user
  local status=0

  [[ -n "${POSTGRES_ADMIN_ANNOTATION_VALUE}" ]] ||
    die "cloud.postgres.admin.annotations must name the workload identity that logs in as ${POSTGRES_ADMIN_USER}"
  require_temporal_pull_secrets

  # Reclaim only stage-labelled leftovers; never label or delete a customer-owned
  # ServiceAccount.
  if kubectl_object_exists -n "${TEMPORAL_NAMESPACE}" get serviceaccount \
    "${service_account}"; then
    service_account_json=$(kubectl_cmd -n "${TEMPORAL_NAMESPACE}" \
      get serviceaccount "${service_account}" -o json) ||
      die "cannot read ServiceAccount ${TEMPORAL_NAMESPACE}/${service_account}"
    existing_annotation=$("${JQ}" -r \
      --arg key "${POSTGRES_ADMIN_ANNOTATION_KEY}" \
      '.metadata.annotations[$key] // empty' <<<"${service_account_json}")
    [[ "${existing_annotation}" == "${POSTGRES_ADMIN_ANNOTATION_VALUE}" ]] ||
      die "ServiceAccount ${TEMPORAL_NAMESPACE}/${service_account} has the wrong ${POSTGRES_ADMIN_ANNOTATION_KEY}"
    existing_label=$("${JQ}" -r --arg key "${TEMPORAL_DATABASE_LABEL_KEY}" \
      '.metadata.labels[$key] // empty' <<<"${service_account_json}")
    if [[ "${existing_label}" == "${TEMPORAL_DATABASE_LABEL_VALUE}" ]]; then
      TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT="${service_account}"
    fi
  else
    # Record ownership before apply so interruption during creation still cleans up.
    TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT="${service_account}"
  fi
  if [[ -n "${TEMPORAL_DATABASE_OWNED_SERVICE_ACCOUNT}" ]]; then
    service_account_label=$(printf '  labels:\n    %s: %s' \
      "${TEMPORAL_DATABASE_LABEL_KEY}" "${TEMPORAL_DATABASE_LABEL_VALUE}")
  fi

  pod_identity_label=""
  if [[ "${POSTGRES_ADMIN_MODE}" == azure_entra ]]; then
    pod_identity_label='        azure.workload.identity/use: "true"'
  fi
  image_pull_secrets=$(temporal_database_pull_secrets)
  # ARO's SCC assigns UID/fsGroup; elsewhere pin them so both containers can read
  # the shared credential.
  pod_security_context=""
  client_user=""
  if [[ "${PLATFORM}" != aro ]]; then
    pod_security_context=$(printf '      securityContext:\n        fsGroup: 1000')
    client_user=$(printf '            runAsUser: 1000\n            runAsGroup: 1000')
  fi
  # Bound the pod even if the operator process dies before cleanup.
  deadline=$(( $(duration_seconds "${timeout}") + 60 ))

  # Callers suppress errexit with ||, so collect failures and always run cleanup.
  TEMPORAL_DATABASE_OWNED_CONFIGMAP="${TEMPORAL_DATABASE_ARTIFACTS}"
  TEMPORAL_DATABASE_ACTIVE_JOB="${job}"

  # Apply the workload-identity annotation atomically with ServiceAccount creation.
  if ! kubectl_cmd -n "${TEMPORAL_NAMESPACE}" apply -f - >/dev/null <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${service_account}
${service_account_label}
  annotations:
    ${POSTGRES_ADMIN_ANNOTATION_KEY}: "${POSTGRES_ADMIN_ANNOTATION_VALUE}"
EOF
  then
    status=1
  elif ! kubectl_cmd -n "${TEMPORAL_NAMESPACE}" create configmap \
    "${TEMPORAL_DATABASE_ARTIFACTS}" \
    "--from-file=${STACK_ROOT}/sql/bootstrap-temporal.sql" \
    "--from-file=${STACK_ROOT}/sql/verify-temporal.sql" \
    "--from-file=${STACK_ROOT}/files/postgres-admin-token.py" \
    "--from-file=${STACK_ROOT}/files/temporal-database.sh" \
    --dry-run=client -o json |
    "${JQ}" --arg key "${TEMPORAL_DATABASE_LABEL_KEY}" \
      --arg value "${TEMPORAL_DATABASE_LABEL_VALUE}" \
      '.metadata.labels[$key] = $value' |
    kubectl_cmd -n "${TEMPORAL_NAMESPACE}" apply -f - >/dev/null; then
    status=1
  elif ! kubectl_cmd -n "${TEMPORAL_NAMESPACE}" delete job "${job}" \
    --ignore-not-found --wait=true >/dev/null; then
    status=1
  elif ! kubectl_cmd -n "${TEMPORAL_NAMESPACE}" apply -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job}
  labels:
    ${TEMPORAL_DATABASE_LABEL_KEY}: ${TEMPORAL_DATABASE_LABEL_VALUE}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: ${deadline}
  ttlSecondsAfterFinished: 300
  template:
    metadata:
      labels:
        ${TEMPORAL_DATABASE_LABEL_KEY}: ${TEMPORAL_DATABASE_LABEL_VALUE}
${pod_identity_label}
    spec:
      restartPolicy: Never
      serviceAccountName: ${service_account}
      automountServiceAccountToken: false
${image_pull_secrets}
${pod_security_context}
      volumes:
        - name: artifacts
          configMap: {name: ${TEMPORAL_DATABASE_ARTIFACTS}}
        - name: credentials
          emptyDir: {medium: Memory, sizeLimit: 1Mi}
      initContainers:
        # Pull psql before minting the short-lived database token.
        - name: check-psql
          image: ${POSTGRES_CLIENT_IMAGE}
          command: [psql, --version]
          securityContext:
            runAsNonRoot: true
${client_user}
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
            seccompProfile: {type: RuntimeDefault}
        - name: mint-admin-credential
          image: ${KUMO_CPU_IMAGE}
          command:
            - python3
            - ${TEMPORAL_DATABASE_ARTIFACT_DIR}/postgres-admin-token.py
          env:
            - {name: POSTGRES_ADMIN_MODE, value: "${POSTGRES_ADMIN_MODE}"}
            - {name: AWS_REGION, value: "${PROVIDER_REGION:-}"}
            - {name: PGHOST, value: "${POSTGRES_HOST}"}
            - {name: PGPORT, value: "${POSTGRES_PORT}"}
            - {name: PGUSER, value: "${POSTGRES_ADMIN_USER}"}
            - name: KUMO_CREDENTIAL_DIR
              value: ${TEMPORAL_DATABASE_CREDENTIAL_DIR}
          volumeMounts:
            - name: artifacts
              mountPath: ${TEMPORAL_DATABASE_ARTIFACT_DIR}
              readOnly: true
            - name: credentials
              mountPath: ${TEMPORAL_DATABASE_CREDENTIAL_DIR}
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
            seccompProfile: {type: RuntimeDefault}
      containers:
        - name: temporal-database
          image: ${POSTGRES_CLIENT_IMAGE}
          command:
            - /bin/sh
            - ${TEMPORAL_DATABASE_ARTIFACT_DIR}/temporal-database.sh
            - ${mode}
          env:
            - {name: PGHOST, value: "${POSTGRES_HOST}"}
            - {name: PGPORT, value: "${POSTGRES_PORT}"}
            - {name: PGUSER, value: "${POSTGRES_ADMIN_USER}"}
            - {name: PGDATABASE, value: "${POSTGRES_ADMIN_DATABASE}"}
            - {name: PGSSLMODE, value: verify-full}
            - {name: PGCONNECT_TIMEOUT, value: "15"}
            - name: PGSSLROOTCERT
              value: ${TEMPORAL_DATABASE_CREDENTIAL_DIR}/ca.pem
            - {name: KUMO_TEMPORAL_ROLE, value: "${POSTGRES_TEMPORAL_USER}"}
            - name: KUMO_TEMPORAL_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: ${POSTGRES_SECRET}
                  key: ${POSTGRES_SECRET_KEY}
          volumeMounts:
            - name: artifacts
              mountPath: ${TEMPORAL_DATABASE_ARTIFACT_DIR}
              readOnly: true
            - name: credentials
              mountPath: ${TEMPORAL_DATABASE_CREDENTIAL_DIR}
              readOnly: true
          securityContext:
            runAsNonRoot: true
${client_user}
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
            seccompProfile: {type: RuntimeDefault}
EOF
  then
    status=1
  elif ! wait_for_job_terminal \
    "${TEMPORAL_NAMESPACE}" "${job}" "${timeout}"; then
    kubectl_cmd -n "${TEMPORAL_NAMESPACE}" logs "job/${job}" \
      --all-containers --tail=120 >&2 || true
    status=1
  fi

  # Deleting the Job discards the credential tmpfs; EXIT does the same on interrupt.
  remove_temporal_database_artifacts || status=1
  return "${status}"
}

bootstrap_temporal_database() {
  run_temporal_database_job \
    kumo-temporal-database-bootstrap bootstrap "${TIMEOUT}" ||
    die "Temporal PostgreSQL bootstrap failed inside the cluster"
}

verify_temporal_database() {
  run_temporal_database_job \
    kumo-temporal-database-verify verify "${VERIFY_TIMEOUT}" ||
    die "Temporal PostgreSQL databases do not match ${POSTGRES_TEMPORAL_USER}"
}

stage_database_apply() {
  ensure_namespace "${TENANT_NAMESPACE}"
  label_namespace "${TENANT_NAMESPACE}" true
  if [[ "${TEMPORAL_ENABLED}" != true ]]; then
    return
  fi

  ensure_namespace "${TEMPORAL_NAMESPACE}"
  label_namespace "${TEMPORAL_NAMESPACE}" false
  if [[ "${POSTGRES_BOOTSTRAP}" == true ]]; then
    ensure_temporal_secret
    bootstrap_temporal_database
  fi
}

verify_state_database_from_cluster() {
  local job=kumo-statedb-database-check
  local service_account=kumo-control
  local existing_annotation pod_identity_label image_pull_secret ca_path
  local status=0

  # Reuse an existing kumo-control. Record a newly created one before creation so
  # interrupts clean it up without deleting the core release's identity.
  if kubectl_object_exists -n "${TENANT_NAMESPACE}" get serviceaccount \
    "${service_account}"; then
    existing_annotation=$(kubectl_cmd -n "${TENANT_NAMESPACE}" \
      get serviceaccount "${service_account}" -o json |
      "${JQ}" -r --arg key "${STATE_CONTROL_ANNOTATION_KEY}" \
        '.metadata.annotations[$key] // empty')
    [[ "${existing_annotation}" == "${STATE_CONTROL_ANNOTATION_VALUE}" ]] ||
      die "ServiceAccount ${TENANT_NAMESPACE}/${service_account} has the wrong ${STATE_CONTROL_ANNOTATION_KEY}"
  else
    STATE_DATABASE_OWNED_SERVICE_ACCOUNT="${service_account}"
    # Use create so losing a race cannot silently adopt another owner's ServiceAccount.
    if ! kubectl_cmd -n "${TENANT_NAMESPACE}" create serviceaccount \
      "${service_account}" >/dev/null; then
      STATE_DATABASE_OWNED_SERVICE_ACCOUNT=""
      die "cannot create ServiceAccount ${TENANT_NAMESPACE}/${service_account}"
    fi
    kubectl_cmd -n "${TENANT_NAMESPACE}" annotate serviceaccount \
      "${service_account}" \
      "${STATE_CONTROL_ANNOTATION_KEY}=${STATE_CONTROL_ANNOTATION_VALUE}" \
      --overwrite >/dev/null
  fi

  pod_identity_label=""
  if [[ "${POSTGRES_STATE_AUTH_MODE}" == azure_entra ]]; then
    pod_identity_label='        azure.workload.identity/use: "true"'
    ca_path=/etc/ssl/certs/ca-certificates.crt
  else
    ca_path=/opt/kumo/certs/aws-rds-roots.pem
  fi
  image_pull_secret=""
  if [[ -n "${KUMO_PULL_SECRET}" ]]; then
    image_pull_secret=$(printf \
      '      imagePullSecrets:\n        - name: %s' "${KUMO_PULL_SECRET}")
  fi

  STATE_DATABASE_ACTIVE_JOB="${job}"
  kubectl_cmd -n "${TENANT_NAMESPACE}" delete job "${job}" \
    --ignore-not-found --wait=true >/dev/null
  if ! kubectl_cmd -n "${TENANT_NAMESPACE}" apply -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job}
  labels:
    kumo.ai/stage-probe: database
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 300
  template:
    metadata:
      labels:
        kumo.ai/stage-probe: database
${pod_identity_label}
    spec:
      restartPolicy: Never
      serviceAccountName: ${service_account}
      automountServiceAccountToken: false
${image_pull_secret}
      containers:
        - name: database-check
          image: ${KUMO_CPU_IMAGE}
          command: [python3, -m, statedb.db.validate]
          env:
            - {name: STATE_AUTH_MODE, value: "${POSTGRES_STATE_AUTH_MODE}"}
            - {name: AWS_REGION, value: "${PROVIDER_REGION:-}"}
            - {name: PGHOST, value: "${POSTGRES_HOST}"}
            - {name: PGPORT, value: "${POSTGRES_PORT}"}
            - {name: PGDATABASE, value: "${POSTGRES_STATE_DATABASE}"}
            - {name: PGUSER, value: "${POSTGRES_STATE_USER}"}
            - {name: PGSSLROOTCERT, value: "${ca_path}"}
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {drop: [ALL]}
            seccompProfile: {type: RuntimeDefault}
EOF
  then
    status=1
  elif ! wait_for_job_terminal \
    "${TENANT_NAMESPACE}" "${job}" "${VERIFY_TIMEOUT}"; then
    kubectl_cmd -n "${TENANT_NAMESPACE}" logs "job/${job}" \
      -c database-check --tail=120 >&2 || true
    status=1
  fi

  # Deleting the Job discards its database token; EXIT handles interrupts.
  remove_state_database_artifacts || status=1
  return "${status}"
}

stage_database_verify() {
  verify_state_database_from_cluster
  [[ "${TEMPORAL_ENABLED}" == true ]] || return 0
  kubectl_cmd -n "${TEMPORAL_NAMESPACE}" get secret \
    "${POSTGRES_SECRET}" -o json |
    "${JQ}" -e --arg key "${POSTGRES_SECRET_KEY}" \
      '.data[$key] | type == "string" and length > 0' >/dev/null

  # A customer-managed database/Secret is tested by Temporal's stock schema
  # hooks from inside the cluster. Do not require the operator to reach private
  # PostgreSQL merely to repeat that connection test.
  if [[ "${POSTGRES_BOOTSTRAP}" == true ]]; then
    verify_temporal_database
  fi
}
