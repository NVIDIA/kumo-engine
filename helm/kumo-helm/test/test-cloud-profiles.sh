#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELM_BIN=${HELM:-helm}
KUMO_REGISTRY=nvcr.io

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local file=$1 text=$2
  grep -Fq -- "$text" "$file" || fail "$file does not contain: $text"
}

assert_absent() {
  local file=$1 text=$2
  if grep -Fq -- "$text" "$file"; then
    fail "$file unexpectedly contains: $text"
  fi
}

assert_env_pair() {
  local file=$1 name=$2 value=$3
  awk -v name="$name" -v value="$value" '
    $0 ~ "^[[:space:]]*- name: " name "$" {
      getline
      line = $0
      sub(/^[[:space:]]*/, "", line)
      if (line == "value: \"" value "\"") found = 1
    }
    END { exit !found }
  ' "$file" || fail "$file does not set ${name}=${value}"
}

assert_all_contains() {
  local file=$1 text
  shift
  for text in "$@"; do assert_contains "$file" "$text"; done
}

resource_doc() {
  local manifest=$1 kind=$2 name=$3 output=$4
  awk -v kind="$kind" -v name="$name" '
    BEGIN { RS = "---" }
    $0 ~ ("kind: " kind "([[:space:]]|$)") &&
      $0 ~ ("name: " name "([[:space:]]|$)") {
      print $0
    }
  ' "$manifest" >"$output"
  [ -s "$output" ] || fail "$kind $name was not rendered"
}

command -v "$HELM_BIN" >/dev/null 2>&1 || fail "Helm is required"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kumo-cloud-profiles.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

RENDER_FAILURE_COUNT=0

assert_render_failure() {
  local failure_message=$1 expected=$2
  shift 2
  RENDER_FAILURE_COUNT=$((RENDER_FAILURE_COUNT + 1))
  local log="${TMP_DIR}/render-failure-${RENDER_FAILURE_COUNT}.log"

  if "$HELM_BIN" template "$@" >"$log" 2>&1; then
    fail "$failure_message"
  fi
  assert_contains "$log" "$expected"
}

assert_rejected_setting() {
  local setting=$1 expected=$2
  assert_render_failure \
    "chart accepted removed setting ${setting%%=*}" "$expected" \
    profile-test "$ROOT" \
    -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
    --set-string "$setting"
}

render_profile() {
  local provider=$1
  local profile="${ROOT}/profiles/${provider}.yaml"
  local fixture="${ROOT}/test/profiles/values-${provider}.yaml"
  local manifest="${TMP_DIR}/${provider}.yaml"

  assert_render_failure \
    "$provider profile rendered without customer values" \
    "global.tenant must replace the chart placeholder" \
    profile-test "$ROOT" -f "$profile"
  "$HELM_BIN" lint "$ROOT" -f "$profile" -f "$fixture" >/dev/null
  "$HELM_BIN" template profile-test "$ROOT" \
    -f "$profile" -f "$fixture" --kube-version 1.35.0 >"$manifest"

  local service_account_count service_account_manifest
  service_account_manifest="${TMP_DIR}/${provider}-serviceaccounts.yaml"
  "$HELM_BIN" template profile-test "$ROOT" \
    -f "$profile" -f "$fixture" --kube-version 1.35.0 \
    --show-only templates/serviceaccount.yaml >"$service_account_manifest"
  service_account_count=$(grep -c '^kind: ServiceAccount$' "$service_account_manifest")
  [ "$service_account_count" -eq 4 ] ||
    fail "$provider rendered $service_account_count ServiceAccounts, want 4"

  assert_absent "$manifest" REPLACE_
  assert_absent "$manifest" "privileged: true"
  assert_contains "$manifest" "seccompProfile:"
  assert_contains "$manifest" "allowPrivilegeEscalation: false"
  for deployment in kumo-ui kumo-rest kumo-api databackend mlworkflow kumo-poolcontroller; do
    local runtime_doc="${TMP_DIR}/${provider}-${deployment}-runtime.yaml"
    resource_doc "$manifest" Deployment "$deployment" "$runtime_doc"
    assert_all_contains "$runtime_doc" "readOnlyRootFilesystem: true" \
      "mountPath: /tmp" "emptyDir: {}"
    if [ "$deployment" != kumo-ui ]; then
      assert_contains "$runtime_doc" "mountPath: /home/kumouser/.cache"
    fi
  done
  assert_absent "$manifest" "secretName: temporal-mtls"
  assert_absent "$manifest" "kumo-postgres-migration"
  assert_absent "$manifest" "POOL_WORKER_SECRETS_SECRET"
  assert_contains "$manifest" "auth_mode"
  case "$provider" in
    aws)
      assert_contains "$manifest" "aws_iam"
      assert_contains "$manifest" "/opt/kumo/certs/aws-rds-roots.pem"
      ;;
    azure)
      assert_contains "$manifest" "azure_entra"
      assert_contains "$manifest" "/etc/ssl/certs/ca-certificates.crt"
      ;;
  esac
  assert_contains "$manifest" "name: KUMO_IMAGE_TAG"
  # The chart owns no mesh, edge, or network policy on any provider.
  assert_absent "$manifest" "kind: PeerAuthentication"
  assert_absent "$manifest" "kind: AuthorizationPolicy"
  assert_absent "$manifest" "kind: NetworkPolicy"

  local bad_images
  bad_images=$(sed -n 's/^[[:space:]]*image:[[:space:]]*"\{0,1\}\([^"[:space:]]*\).*/\1/p' \
    "$manifest" | grep -Fv "${KUMO_REGISTRY}/" || true)
  [ -z "$bad_images" ] || fail "$provider rendered images outside the Kumo registry: $bad_images"

  local role deployment expected_automount doc
  for role in frontend control worker controller; do
    doc="${TMP_DIR}/${provider}-sa-${role}.yaml"
    resource_doc "$service_account_manifest" ServiceAccount "kumo-${role}" "$doc"
    case "$role" in
      frontend|control) expected_automount=false ;;
      worker|controller) expected_automount=true ;;
    esac
    assert_contains "$doc" "automountServiceAccountToken: ${expected_automount}"
  done

  resource_doc "$manifest" Deployment kumo-ui \
    "${TMP_DIR}/${provider}-kumo-ui.yaml"
  assert_absent "${TMP_DIR}/${provider}-kumo-ui.yaml" "nodeSelector:"
  assert_contains "${TMP_DIR}/${provider}-kumo-ui.yaml" \
    "serviceAccountName: kumo-frontend"
  assert_contains "${TMP_DIR}/${provider}-kumo-ui.yaml" \
    "automountServiceAccountToken: false"
  for deployment in kumo-rest kumo-api; do
    resource_doc "$manifest" Deployment "$deployment" \
      "${TMP_DIR}/${provider}-${deployment}.yaml"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "serviceAccountName: kumo-control"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "automountServiceAccountToken: false"
  done
  assert_contains "${TMP_DIR}/${provider}-kumo-rest.yaml" 'cpu: "1"'
  assert_contains "${TMP_DIR}/${provider}-kumo-rest.yaml" "memory: 4Gi"
  assert_contains "${TMP_DIR}/${provider}-kumo-rest.yaml" "memory: 8Gi"
  assert_contains "${TMP_DIR}/${provider}-kumo-api.yaml" "memory: 2Gi"
  for deployment in databackend mlworkflow; do
    resource_doc "$manifest" Deployment "$deployment" \
      "${TMP_DIR}/${provider}-${deployment}.yaml"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "serviceAccountName: kumo-worker"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      'cpu: "1"'
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "memory: 4Gi"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "cpu: 1500m"
    assert_contains "${TMP_DIR}/${provider}-${deployment}.yaml" \
      "memory: 5Gi"
  done
  resource_doc "$manifest" Deployment kumo-poolcontroller \
    "${TMP_DIR}/${provider}-poolcontroller.yaml"
  assert_env_pair "${TMP_DIR}/${provider}-poolcontroller.yaml" \
    POOL_WORKER_READ_ONLY_ROOT_FILESYSTEM true
  for read_only in false null; do
    local readonly_manifest="${TMP_DIR}/${provider}-readonly-${read_only}.yaml"
    "$HELM_BIN" template profile-test "$ROOT" -f "$profile" -f "$fixture" \
      --set "global.containerSecurityContext.readOnlyRootFilesystem=${read_only}" \
      >"$readonly_manifest"
    assert_absent "$readonly_manifest" "readOnlyRootFilesystem: true"
    assert_env_pair "$readonly_manifest" POOL_WORKER_READ_ONLY_ROOT_FILESYSTEM false
  done
  assert_contains "${TMP_DIR}/${provider}-poolcontroller.yaml" \
    "serviceAccountName: kumo-controller"
  assert_contains "${TMP_DIR}/${provider}-poolcontroller.yaml" "memory: 3Gi"
  assert_contains "${TMP_DIR}/${provider}-poolcontroller.yaml" \
    'value: "kumo-worker"'
  assert_contains "${TMP_DIR}/${provider}-poolcontroller.yaml" \
    'test -f /tmp/kumo-poolcontroller-ready'

  resource_doc "$manifest" Job profile-test-register-temporal-ns \
    "${TMP_DIR}/${provider}-register-namespace.yaml"
  assert_absent "${TMP_DIR}/${provider}-register-namespace.yaml" "nodeSelector:"
  assert_contains "${TMP_DIR}/${provider}-register-namespace.yaml" \
    '"helm.sh/hook": pre-install,pre-upgrade'
  assert_contains "${TMP_DIR}/${provider}-register-namespace.yaml" \
    "automountServiceAccountToken: false"
  assert_contains "${TMP_DIR}/${provider}-register-namespace.yaml" \
    "name: DEPLOY_ENV_CONFIG_JSON"
  assert_contains "${TMP_DIR}/${provider}-register-namespace.yaml" \
    "cpu: 100m"
  assert_contains "${TMP_DIR}/${provider}-register-namespace.yaml" \
    "memory: 256Mi"
  assert_absent "${TMP_DIR}/${provider}-register-namespace.yaml" \
    "configMapRef:"
}

render_profile aws
render_profile azure

AWS_MANIFEST="${TMP_DIR}/aws.yaml"
AZURE_MANIFEST="${TMP_DIR}/azure.yaml"

GCP_STATIC_MANIFEST="${TMP_DIR}/gcp-static.yaml"
"$HELM_BIN" lint "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" >/dev/null
"$HELM_BIN" template gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  >"$GCP_STATIC_MANIFEST"

resource_doc "$GCP_STATIC_MANIFEST" ConfigMap kumo-config \
  "${TMP_DIR}/gcp-static-configmap.yaml"
assert_contains "${TMP_DIR}/gcp-static-configmap.yaml" \
  'GOOGLE_APPLICATION_CREDENTIALS: "/var/run/secrets/kumo/gcp/application-credentials.json"'
assert_contains "${TMP_DIR}/gcp-static-configmap.yaml" \
  'GOOGLE_CLOUD_PROJECT: "gcp-test-project"'
assert_absent "$GCP_STATIC_MANIFEST" "kind: Secret"

for deployment in kumo-rest kumo-api databackend mlworkflow kumo-poolcontroller; do
  doc="${TMP_DIR}/gcp-static-${deployment}.yaml"
  resource_doc "$GCP_STATIC_MANIFEST" Deployment "$deployment" "$doc"
  assert_contains "$doc" 'secretName: "gcp-application-credentials"'
  assert_contains "$doc" 'key: "credentials.json"'
  assert_contains "$doc" "path: application-credentials.json"
  assert_contains "$doc" "defaultMode: 292"
  assert_contains "$doc" "mountPath: /var/run/secrets/kumo/gcp"
  assert_contains "$doc" "readOnly: true"
  assert_contains "$doc" "configMapRef:"
  assert_contains "$doc" "name: kumo-config"
done
assert_env_pair "${TMP_DIR}/gcp-static-kumo-poolcontroller.yaml" \
  POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET \
  gcp-application-credentials
assert_env_pair "${TMP_DIR}/gcp-static-kumo-poolcontroller.yaml" \
  POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY \
  credentials.json

resource_doc "$GCP_STATIC_MANIFEST" Deployment kumo-ui \
  "${TMP_DIR}/gcp-static-kumo-ui.yaml"
assert_absent "${TMP_DIR}/gcp-static-kumo-ui.yaml" \
  "gcp-application-credentials"
assert_absent "${TMP_DIR}/gcp-static-kumo-ui.yaml" \
  "GOOGLE_APPLICATION_CREDENTIALS"
for job in gcp-static-test-preflight gcp-static-test-register-temporal-ns; do
  doc="${TMP_DIR}/${job}.yaml"
  resource_doc "$GCP_STATIC_MANIFEST" Job "$job" "$doc"
  assert_all_contains "$doc" "readOnlyRootFilesystem: true" \
    "mountPath: /tmp" "mountPath: /home/kumouser/.cache" "emptyDir: {}"
  assert_absent "$doc" "gcp-application-credentials"
  assert_absent "$doc" "GOOGLE_APPLICATION_CREDENTIALS"
done

assert_render_failure \
  "GCP static credentials accepted an empty Secret key" \
  "kumoConfig.gsaKey.key is required when existingSecret is set" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string kumoConfig.gsaKey.key=
assert_render_failure \
  "GCP static credentials accepted inline credential JSON" \
  "credentialsJson" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string kumoConfig.gsaKey.credentialsJson=must-not-render
assert_render_failure \
  "GCP project ID was accepted without enabling static credentials" \
  "kumoConfig.gsaKey.projectId requires existingSecret" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string kumoConfig.gsaKey.existingSecret=
assert_render_failure \
  "GCP static credentials accepted a GKE workload-identity annotation" \
  "cannot be combined with GCP workload-identity annotations" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string 'serviceAccounts.worker.annotations.iam\.gke\.io/gcp-service-account=worker@example.iam.gserviceaccount.com'
assert_render_failure \
  "GCP static credentials accepted an empty-valued GKE workload-identity annotation key" \
  "cannot be combined with GCP workload-identity annotations" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string 'serviceAccounts.worker.annotations.iam\.gke\.io/gcp-service-account='
assert_render_failure \
  "GCP static credentials accepted a webhook workload-identity annotation" \
  "cannot be combined with GCP workload-identity annotations" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string 'serviceAccounts.control.annotations.cloud\.google\.com/workload-identity-provider=projects/123/locations/global/workloadIdentityPools/test/providers/test'
assert_render_failure \
  "GCS accepted neither static credentials nor workload identity" \
  "object storage requires a nonempty workload-identity annotation on serviceAccounts.control" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string kumoConfig.gsaKey.projectId= \
  --set-string kumoConfig.gsaKey.existingSecret=
assert_render_failure \
  "S3 accepted GCP static credentials instead of workload identity" \
  "kumoConfig.gsaKey.existingSecret applies only to a gs:// kumoConfig.storage.dataBucket" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string kumoConfig.storage.dataBucket=s3://test-bucket/disk-graph
assert_render_failure \
  "AWS profile accepted GCP static credentials alongside valid AWS identity" \
  "kumoConfig.gsaKey.existingSecret applies only to a gs:// kumoConfig.storage.dataBucket" \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  --set-string kumoConfig.gsaKey.existingSecret=gcp-application-credentials \
  --set-string kumoConfig.gsaKey.key=credentials.json \
  --set-string kumoConfig.gsaKey.projectId=gcp-test-project
assert_render_failure \
  "GCP static credentials accepted a duplicate ADC environment override" \
  "workloads.rest.extraEnv.GOOGLE_APPLICATION_CREDENTIALS cannot override kumoConfig.gsaKey" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string workloads.rest.extraEnv.GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials.json
assert_render_failure \
  "GCP static credentials accepted a duplicate project environment override" \
  "workloads.rest.extraEnv.GOOGLE_CLOUD_PROJECT cannot override kumoConfig.gsaKey.projectId" \
  gcp-static-test "$ROOT" \
  -f "$ROOT/test/manifests/values-gcp-static.yaml" \
  --set-string workloads.rest.extraEnv.GOOGLE_CLOUD_PROJECT=other-project
for variable in GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT \
  POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET \
  POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY; do
  assert_render_failure \
    "GCP static credentials accepted a pool-controller override of ${variable}" \
    "workloads.poolcontroller.extraEnv.${variable} cannot override kumoConfig.gsaKey" \
    gcp-static-test "$ROOT" \
    -f "$ROOT/test/manifests/values-gcp-static.yaml" \
    --set-string "workloads.poolcontroller.extraEnv.${variable}=override"
done

assert_render_failure \
  "object-store profile accepted an empty worker identity" \
  "object storage requires a nonempty workload-identity annotation on serviceAccounts.worker" \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  --set-string 'serviceAccounts.worker.annotations.eks\.amazonaws\.com/role-arn='

assert_render_failure \
  "Azure profile accepted unencrypted abfs storage" \
  "must use abfss:// for encrypted Azure Data Lake traffic" \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/azure.yaml" -f "$ROOT/test/profiles/values-azure.yaml" \
  --set-string 'kumoConfig.storage.dataBucket=abfs://profile@profiletest.dfs.core.windows.net/disk-graph'

assert_render_failure \
  "object-store profile accepted an unsupported storage scheme" \
  "kumoConfig.storage.dataBucket must use s3://, gs://, or abfss://" \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  --set-string 'kumoConfig.storage.dataBucket=https://storage.example.com/disk-graph'

assert_rejected_setting "serviceAccount.name=kumo" \
  "serviceAccount is not supported"
assert_rejected_setting "platform.provider=azure" \
  "platform is not supported"
assert_rejected_setting "image.uiTag=test-version" \
  "image.uiTag is not supported"
assert_rejected_setting "workloads.ui.privileged=true" \
  "workloads.ui.privileged is not supported"
assert_rejected_setting "global.containerSecurityContext.privileged=true" \
  "global.containerSecurityContext.privileged cannot be enabled"
assert_rejected_setting "global.transport.mode=insecure" \
  "additional properties 'transport' not allowed"
assert_rejected_setting "global.deployStage=prod" \
  "additional properties 'deployStage' not allowed"
assert_rejected_setting "security.containerSecurityContext.privileged=false" \
  "security is not supported"
assert_rejected_setting "exposure.tls=false" \
  "additional properties 'tls' not allowed"
assert_rejected_setting "temporal.tls.enabled=false" \
  "additional properties 'tls' not allowed"
assert_rejected_setting "temporal.certDir=/tmp/temporal" \
  "additional properties 'certDir' not allowed"
assert_rejected_setting "kumoConfig.extra.auth.enabled=false" \
  "kumoConfig.extra.auth cannot override chart-managed authentication wiring"
assert_rejected_setting "kumoConfig.extra.external_host=bypass.example.com" \
  "kumoConfig.extra.external_host cannot override chart-managed exposure wiring"
for identity_key in ambient_s3 ambient_gcp azure_storage; do
  assert_rejected_setting "kumoConfig.extra.${identity_key}.enabled=true" \
    "kumoConfig.extra.${identity_key} cannot override chart-managed object-store identity wiring"
done
assert_rejected_setting "kumoConfig.extra.orchestration.server_address=temporal.example.com" \
  "kumoConfig.extra.orchestration.server_address cannot override chart-managed Temporal wiring"
assert_rejected_setting "kumoConfig.extra.orchestration.tls_cert_dir=/tmp/temporal" \
  "kumoConfig.extra.orchestration.tls_cert_dir cannot override chart-managed Temporal wiring"
assert_rejected_setting "kumoConfig.extra.metadata_storage.dsn_secret_name=custom-statedb" \
  "kumoConfig.extra.metadata_storage.dsn_secret_name cannot override chart-managed StateDB wiring"
assert_rejected_setting "kumoConfig.extra.secrets.backend=aws_sm" \
  "kumoConfig.extra.secrets.backend cannot override chart-managed Secret wiring"
assert_rejected_setting "kumoConfig.extra.secrets.mount_path=/tmp/secrets" \
  "kumoConfig.extra.secrets.mount_path cannot override chart-managed Secret wiring"
assert_rejected_setting \
  "kumoConfig.extra.storage.data_plane_storage_root=abfs://container@account/path" \
  "kumoConfig.extra.storage.data_plane_storage_root cannot override chart-managed object storage wiring"
assert_rejected_setting \
  "kumoConfig.extra.storage.control_plane_storage_root=http://storage.example.com/path" \
  "kumoConfig.extra.storage.control_plane_storage_root must use s3://, gs://, or abfss://"
assert_rejected_setting \
  "kumoConfig.extra.storage.artifact_root=abfs://container@account/path" \
  "kumoConfig.extra.storage.artifact_root must use s3://, gs://, or abfss://"
assert_rejected_setting "observability.slackUrl=http://hooks.example.com/test" \
  "observability.slackUrl must use https://"

"$HELM_BIN" template profile-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string observability.slackUrl=https://hooks.example.com/test \
  >/dev/null

AUX_STORAGE_MANIFEST="${TMP_DIR}/aux-storage.yaml"
"$HELM_BIN" template profile-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string kumoConfig.extra.storage.control_plane_storage_root=s3://control/path \
  --set-string kumoConfig.extra.storage.artifact_root=abfss://artifacts@account/path \
  >"$AUX_STORAGE_MANIFEST"
assert_contains "$AUX_STORAGE_MANIFEST" \
  '\"control_plane_storage_root\":\"s3://control/path\"'
assert_contains "$AUX_STORAGE_MANIFEST" \
  '\"artifact_root\":\"abfss://artifacts@account/path\"'

assert_absent "$ROOT/profiles/openshift.yaml" "istio.io/"

for provider in aws azure; do
  OPENSHIFT_MANIFEST="${TMP_DIR}/${provider}-openshift.yaml"
  "$HELM_BIN" lint "$ROOT" \
    -f "$ROOT/profiles/${provider}.yaml" \
    -f "$ROOT/test/profiles/values-${provider}.yaml" \
    -f "$ROOT/test/profiles/values-fixed-security-context.yaml" \
    -f "$ROOT/profiles/openshift.yaml" >/dev/null
  "$HELM_BIN" template profile-test "$ROOT" \
    -f "$ROOT/profiles/${provider}.yaml" \
    -f "$ROOT/test/profiles/values-${provider}.yaml" \
    -f "$ROOT/test/profiles/values-fixed-security-context.yaml" \
    -f "$ROOT/profiles/openshift.yaml" \
    >"$OPENSHIFT_MANIFEST"
  assert_absent "$OPENSHIFT_MANIFEST" "fsGroup:"
  assert_absent "$OPENSHIFT_MANIFEST" "supplementalGroups:"
  assert_absent "$OPENSHIFT_MANIFEST" "runAsUser:"
  assert_absent "$OPENSHIFT_MANIFEST" "runAsGroup:"
  assert_absent "$OPENSHIFT_MANIFEST" "asm-1-29"
  assert_contains "$OPENSHIFT_MANIFEST" "runAsNonRoot: true"
  resource_doc "$OPENSHIFT_MANIFEST" Deployment kumo-poolcontroller \
    "${TMP_DIR}/${provider}-openshift-poolcontroller.yaml"
  assert_contains "${TMP_DIR}/${provider}-openshift-poolcontroller.yaml" \
    'name: POOL_WORKER_FS_GROUP'
  assert_contains "${TMP_DIR}/${provider}-openshift-poolcontroller.yaml" \
    'value: ""'
done

assert_render_failure \
  "cloud profile accepted a null diskgraph node type" \
  'kumoConfig.diskgraph.nodeTypes["test-gpu"] must be a map with a positive ssd_gib' \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  --set-json 'kumoConfig.diskgraph.nodeTypes.test-gpu=null'

assert_render_failure \
  "cloud profile accepted a non-integer runtimeFsGroup" \
  "/workloads/poolcontroller/runtimeFsGroup': got string, want null or integer" \
  profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  --set-string workloads.poolcontroller.runtimeFsGroup=not-an-integer

AWS_PLAINTEXT_MANIFEST="${TMP_DIR}/aws-plaintext.yaml"
"$HELM_BIN" template profile-test "$ROOT" \
  -f "$ROOT/profiles/aws.yaml" -f "$ROOT/test/profiles/values-aws.yaml" \
  >"$AWS_PLAINTEXT_MANIFEST"
assert_absent "$AWS_PLAINTEXT_MANIFEST" "kind: PeerAuthentication"
resource_doc "$AWS_PLAINTEXT_MANIFEST" Job profile-test-preflight \
  "${TMP_DIR}/aws-plaintext-preflight.yaml"

PLAINTEXT_AUTH_MANIFEST="${TMP_DIR}/plaintext-auth.yaml"
assert_render_failure \
  "prod accepted a missing deployment secret key" \
  "tenantSecretKey is required in prod" \
  transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set auth.enabled=true \
  --set-string auth.method=password \
  --set-string auth.password.temporaryPassword.secretName=kumo-login \
  --set-string exposure.host=kumo.example.com

"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set auth.enabled=true \
  --set-string auth.method=password \
  --set-string auth.password.temporaryPassword.secretName=kumo-login \
  --set-string exposure.host=kumo.example.com \
  --set-string tenantSecretKey.secretName=kumo-secret-key \
  --set-string tenantSecretKey.secretKey=credentials_encryption_key \
  >"$PLAINTEXT_AUTH_MANIFEST"
assert_contains "$PLAINTEXT_AUTH_MANIFEST" "https://kumo.example.com"
assert_absent "$PLAINTEXT_AUTH_MANIFEST" "kind: PeerAuthentication"

"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string observability.otlpHostname=collector.example.com \
  --set-string observability.pushGateway.address=http://push.example.com \
  >/dev/null

"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string temporal.address=temporal-frontend.temporal.svc:7233 \
  >"${TMP_DIR}/transport-explicit-port.yaml"
assert_contains "${TMP_DIR}/transport-explicit-port.yaml" \
  '\"temporalAddress\":\"temporal-frontend.temporal.svc:7233\"'
assert_contains "${TMP_DIR}/transport-explicit-port.yaml" \
  '\"server_address\":\"temporal-frontend.temporal.svc:7233\"'

TRANSPORT_MTLS_MANIFEST="${TMP_DIR}/transport-mtls.yaml"
"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string temporal.address=temporal.example.com \
  --set-string temporal.clientMtlsSecret=temporal-mtls \
  >"$TRANSPORT_MTLS_MANIFEST"
assert_contains "$TRANSPORT_MTLS_MANIFEST" \
  '\"tls_cert_dir\":\"/etc/temporal/certs\"'
for deployment in kumo-rest kumo-api databackend mlworkflow kumo-poolcontroller; do
  doc="${TMP_DIR}/mtls-${deployment}.yaml"
  resource_doc "$TRANSPORT_MTLS_MANIFEST" Deployment "$deployment" "$doc"
  assert_contains "$doc" "mountPath: /etc/temporal/certs"
  assert_contains "$doc" "secretName: temporal-mtls"
done
resource_doc "$TRANSPORT_MTLS_MANIFEST" Job transport-test-register-temporal-ns \
  "${TMP_DIR}/mtls-register-namespace.yaml"
assert_contains "${TMP_DIR}/mtls-register-namespace.yaml" \
  "mountPath: /etc/temporal/certs"
assert_contains "${TMP_DIR}/mtls-register-namespace.yaml" \
  "secretName: temporal-mtls"
assert_contains "${TMP_DIR}/mtls-kumo-poolcontroller.yaml" \
  "name: POOL_WORKER_TEMPORAL_CERT_SECRET"
assert_contains "${TMP_DIR}/mtls-kumo-poolcontroller.yaml" \
  'value: "temporal-mtls"'

OIDC_ARGS=(
  --set auth.enabled=true
  --set-string exposure.host=kumo.example.com
  --set-string auth.oidc.clientId=kumo
  --set-string auth.oidc.clientSecret.secretName=kumo-oidc
  --set-string auth.oidc.clientSecret.secretKey=client-secret
  --set-string tenantSecretKey.secretName=kumo-secret-key
  --set-string tenantSecretKey.secretKey=credentials_encryption_key
)
assert_render_failure \
  "OIDC accepted a plaintext issuer URL" \
  "auth.oidc.issuerUrl must use https://" \
  transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  "${OIDC_ARGS[@]}" \
  --set-string auth.oidc.issuerUrl=http://idp.example.com

"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  "${OIDC_ARGS[@]}" \
  --set-string auth.oidc.issuerUrl=https://idp.example.com \
  --set-string 'auth.allowedReturnOrigins[0]=https://return.example.com' \
  >/dev/null

assert_render_failure \
  "encrypted auth accepted a plaintext return origin" \
  'auth.allowedReturnOrigins entry "http://return.example.com" must use https://' \
  transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  "${OIDC_ARGS[@]}" \
  --set-string auth.oidc.issuerUrl=https://idp.example.com \
  --set-string 'auth.allowedReturnOrigins[0]=http://return.example.com'

assert_render_failure \
  "encrypted auth accepted multiple comma-separated return origins in one entry" \
  "must contain exactly one origin" \
  transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  "${OIDC_ARGS[@]}" \
  --set-string auth.oidc.issuerUrl=https://idp.example.com \
  --set-literal 'auth.allowedReturnOrigins[0]=https://return.example.com,http://bypass.example.com'

# No Secret means no cert dir and no mount. Temporal mTLS follows the Secret
# alone now; there is no separate transport flag that can suppress it.
"$HELM_BIN" template transport-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string temporal.clientMtlsSecret= \
  >"${TMP_DIR}/no-temporal-mtls.yaml"
assert_contains "${TMP_DIR}/no-temporal-mtls.yaml" \
  '\"tls_cert_dir\":\"\"'
assert_absent "${TMP_DIR}/no-temporal-mtls.yaml" \
  "secretName: temporal-mtls"

assert_render_failure \
  "base chart accepted a global app-label collision" \
  'global.podLabels["app"] is reserved by the chart' \
  label-test "$ROOT" \
  -f "$ROOT/test/manifests/values-aws-minimal.yaml" \
  --set-string global.podLabels.app=caller

assert_contains "$AWS_MANIFEST" \
  "s3://profile-test/disk-graph/profile-test"
assert_absent "$AWS_MANIFEST" "CUSTOMER_ID"
assert_contains "$AWS_MANIFEST" \
  '\"storageCapacityTracking\":\"advisory\"'
assert_contains "$AWS_MANIFEST" \
  '\"ambient_s3\":{\"approved_role_arns\":[\"arn:aws:iam::000000000000:role/kumo-worker-test\"]}'
for role in control worker controller; do
  doc="${TMP_DIR}/aws-sa-${role}.yaml"
  assert_contains "$doc" "eks.amazonaws.com/role-arn: arn:aws:iam::000000000000:role/kumo-${role}-test"
done
assert_absent "${TMP_DIR}/aws-sa-frontend.yaml" "eks.amazonaws.com/role-arn"
assert_contains "${TMP_DIR}/aws-poolcontroller.yaml" 'value: "ebs-gp3-xfs"'
assert_absent "$AWS_MANIFEST" "name: kumo-registry"
assert_absent "$AWS_MANIFEST" "KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED"
assert_absent "$AWS_MANIFEST" "GOOGLE_APPLICATION_CREDENTIALS"
assert_absent "$AWS_MANIFEST" "gcp-application-credentials"

assert_contains "$AZURE_MANIFEST" \
  "abfss://profile@profiletest.dfs.core.windows.net/disk-graph/profile-test"
assert_absent "$AZURE_MANIFEST" "CUSTOMER_ID"
assert_contains "$AZURE_MANIFEST" \
  '\"storageCapacityTracking\":\"advisory\"'
assert_contains "$AZURE_MANIFEST" \
  'KUMO_AZURE_WORKLOAD_EXCHANGE_ENABLED: "true"'
for role in control worker controller; do
  doc="${TMP_DIR}/azure-sa-${role}.yaml"
  assert_contains "$doc" "azure.workload.identity/client-id:"
  assert_contains "$doc" "- name: kumo-registry"
done
assert_absent "${TMP_DIR}/azure-sa-frontend.yaml" \
  "azure.workload.identity/client-id"
assert_contains "${TMP_DIR}/azure-sa-frontend.yaml" "- name: kumo-registry"
assert_absent "${TMP_DIR}/azure-kumo-ui.yaml" \
  'azure.workload.identity/use: "true"'
for deployment in kumo-rest kumo-api databackend mlworkflow poolcontroller; do
  assert_contains "${TMP_DIR}/azure-${deployment}.yaml" \
    'azure.workload.identity/use: "true"'
done
assert_contains "${TMP_DIR}/azure-poolcontroller.yaml" \
  'value: "{\"azure.workload.identity/use\":\"true\"}"'
assert_contains "${TMP_DIR}/azure-poolcontroller.yaml" \
  "name: POOL_INIT_EPHEMERAL_CANARY_ENABLED"
assert_contains "${TMP_DIR}/azure-poolcontroller.yaml" 'value: "false"'
assert_contains "${TMP_DIR}/azure-poolcontroller.yaml" 'value: "managed-csi-xfs"'
assert_contains "${TMP_DIR}/azure-databackend.yaml" \
  "storageClassName: managed-csi-xfs"
assert_contains "${TMP_DIR}/azure-databackend.yaml" "storage: 256Gi"
assert_contains "${TMP_DIR}/azure-databackend.yaml" "fsGroup: 999"
assert_contains "${TMP_DIR}/azure-databackend.yaml" \
  'value: "/mnt/localssd/tmp"'
assert_contains "${TMP_DIR}/azure-databackend.yaml" \
  "mountPath: /mnt/localssd"
assert_contains "${TMP_DIR}/azure-databackend.yaml" "ephemeral:"

for provider in aws azure; do
  manifest="${TMP_DIR}/${provider}-neutral.yaml"
  "$HELM_BIN" template profile-test "$ROOT" \
    -f "$ROOT/profiles/${provider}.yaml" \
    -f "$ROOT/test/profiles/values-${provider}.yaml" \
    --set-string exposure.host=kumo.internal.example.com >"$manifest"
  if grep -Eq '^kind: (Ingress|Gateway|HTTPRoute|Route|PeerAuthentication|AuthorizationPolicy|NetworkPolicy)$' "$manifest"; then
    fail "$provider rendered infrastructure resources the chart must not own"
  fi
done

AZURE_HOOKS="${TMP_DIR}/azure-hooks.yaml"
"$HELM_BIN" template profile-test "$ROOT" \
  -f "$ROOT/profiles/azure.yaml" -f "$ROOT/test/profiles/values-azure.yaml" \
  --show-only templates/job-register-namespace.yaml \
  --show-only templates/preflight/job-preflight.yaml >"$AZURE_HOOKS"
assert_absent "$AZURE_HOOKS" 'azure.workload.identity/use: "true"'

echo "AWS and Azure profile contracts passed."
