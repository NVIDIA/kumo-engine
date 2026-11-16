#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Kumo conformance probe (READ-ONLY) — values-driven + dependency-ordered.
#
# It derives every expectation from YOUR chart values by rendering the chart
# (helm template -f your-values.yaml), so it checks the config you will actually
# install — not a fixed environment. Checks run in dependency order; when a
# prerequisite fails, dependents report BLOCK by default instead of a misleading
# error; explicitly advisory checks can report ADV without blocking the probe.
#
# Usage:
#   check.sh [-f values.yaml]... [--chart DIR] [--profile AWS_PROFILE] -n NAMESPACE [--kube-context CTX] [prereq|health|all]
#     -f                  your chart values file (repeatable). global.tenant must be set.
#     --chart             chart directory (default: the chart this bundle ships in)
#     --profile           AWS_PROFILE to export (AWS storage backends)
#     -n, --namespace NS  target namespace to probe (required; falls back to $HELM_NAMESPACE).
#     --kube-context CTX  kube context used for every kubectl/helm invocation (default: current context)
#     phase               prereq | health | all   (default: all)
#
# Exit code = number of FAIL + BLOCK (SKIP and ADV do not count). 0 = conformant.
set -uo pipefail

# Requires bash >= 4 (associative arrays). macOS ships 3.2 — run with a newer bash
# (e.g. `brew install bash`), otherwise the status table errors out.
if [ "${BASH_VERSINFO:-0}" -lt 4 ]; then
  echo "check.sh requires bash >= 4 (found ${BASH_VERSION:-unknown}). On macOS: brew install bash, then run with it."
  exit 2
fi
# Required tools (jq is validated below; the rest fail loudly if absent):
#   kubectl, helm, jq, curl  (+ aws / gcloud|gsutil / az for the STORAGE_ACCESS check)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pure JSON validation used by the read-only CSI probe.
source "${SCRIPT_DIR}/csi_contract.sh"
CHART_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PHASE=all
VAL_ARGS=()
NAMESPACE=""
KUBE_CONTEXT=""
HELM_COMMAND="${HELM_BIN:-helm}"
KUBECTL_COMMAND="${KUBECTL_BIN:-kubectl}"
EDGE_CURL_TLS_ARGS=()
if [ "${KUMO_CONFORMANCE_INSECURE_TLS:-false}" = true ]; then
  EDGE_CURL_TLS_ARGS=(--insecure)
fi
# need_val <flag> [value...]: FATAL (not an unbound-variable crash) when a flag is given
# without its required value, i.e. it was the last argument.
need_val(){ [ "$#" -ge 2 ] || { echo "FATAL: $1 requires a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    -f)              need_val "$@"; VAL_ARGS+=(-f "$2"); shift 2 ;;
    --chart)         need_val "$@"; CHART_DIR="$2"; shift 2 ;;
    --profile)       need_val "$@"; AWS_PROFILE="$2"; export AWS_PROFILE; shift 2 ;;
    -n|--namespace)  need_val "$@"; NAMESPACE="$2"; shift 2 ;;
    --kube-context)  need_val "$@"; KUBE_CONTEXT="$2"; shift 2 ;;
    prereq|health|all) PHASE="$1"; shift ;;
    -h|--help) awk 'NR==1{next} /^# SPDX-|^# All rights reserved\.$/{next} /^#/{print; next} {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; awk '/^# Usage:/{f=1} f{ if (/^#/) print; else exit }' "$0"; exit 2 ;;
  esac
done

# ── Resolve target namespace (highest precedence: -n/--namespace, then HELM_NAMESPACE env) ───
# No implicit default: this used to fall through to helm's own "default" namespace resolution,
# which produced a FALSE GREEN "namespace default exists" for anyone probing before choosing a
# namespace. An unset namespace is a config gap, not "default".
if [ -z "$NAMESPACE" ]; then
  if [ -n "${HELM_NAMESPACE:-}" ]; then
    NAMESPACE="$HELM_NAMESPACE"
  else
    echo "FATAL: no target namespace given. Pass -n/--namespace <namespace> (or set HELM_NAMESPACE)."
    echo "  Refusing to silently probe the 'default' namespace."
    exit 2
  fi
fi

# ── kube context (applies to every kubectl/helm invocation below) ────────────────────────────
HELM_CTX_ARGS=()
KUBECTL_CTX_ARGS=()
if [ -n "$KUBE_CONTEXT" ]; then
  HELM_CTX_ARGS=(--kube-context "$KUBE_CONTEXT")
  KUBECTL_CTX_ARGS=(--context "$KUBE_CONTEXT")
fi
kubectl(){
  command "${KUBECTL_COMMAND}" \
    ${KUBECTL_CTX_ARGS[@]+"${KUBECTL_CTX_ARGS[@]}"} "$@"
}

# Start a service port-forward on a kubectl-selected local port. Fixed ports
# make simultaneous checks against different clusters race or probe each other.
PORT_FORWARD_PID=""
PORT_FORWARD_PORT=""
PORT_FORWARD_LOG=""
start_port_forward(){
  local service="$1" attempt
  PORT_FORWARD_PID=""
  PORT_FORWARD_PORT=""
  PORT_FORWARD_LOG=$(mktemp "${TMPDIR:-/tmp}/kumo-port-forward.XXXXXX")
  command "${KUBECTL_COMMAND}" \
    ${KUBECTL_CTX_ARGS[@]+"${KUBECTL_CTX_ARGS[@]}"} \
    port-forward -n "$NS" "svc/${service}" :80 \
    >"$PORT_FORWARD_LOG" 2>&1 &
  PORT_FORWARD_PID=$!
  for attempt in {1..50}; do
    PORT_FORWARD_PORT=$(sed -nE \
      's/^Forwarding from .*:([0-9]+) -> [0-9]+$/\1/p' \
      "$PORT_FORWARD_LOG" | head -n 1)
    [ -n "$PORT_FORWARD_PORT" ] && return 0
    kill -0 "$PORT_FORWARD_PID" 2>/dev/null || break
    sleep 0.1
  done
  return 1
}
stop_port_forward(){
  if [ -n "$PORT_FORWARD_PID" ]; then
    kill "$PORT_FORWARD_PID" 2>/dev/null || true
    wait "$PORT_FORWARD_PID" 2>/dev/null || true
  fi
  [ -z "$PORT_FORWARD_LOG" ] || rm -f -- "$PORT_FORWARD_LOG"
  PORT_FORWARD_PID=""
  PORT_FORWARD_PORT=""
  PORT_FORWARD_LOG=""
}
trap stop_port_forward EXIT

# ── Resolve expectations from the chart (single source of truth) ─────────────
RENDER=$("${HELM_COMMAND}" template kumo "$CHART_DIR" \
  ${VAL_ARGS[@]+"${VAL_ARGS[@]}"} \
  --namespace "$NAMESPACE" ${HELM_CTX_ARGS[@]+"${HELM_CTX_ARGS[@]}"} \
  --show-only templates/tests/probe-inputs.yaml 2>/dev/null) || true
INPUTS=$(printf '%s\n' "$RENDER" | grep 'inputs.json:' \
  | sed -E 's/^[[:space:]]*inputs\.json:[[:space:]]*//' | jq -c 'fromjson' 2>/dev/null) || true
if [ -z "${INPUTS:-}" ]; then
  echo "FATAL: could not render probe inputs from the chart."
  echo "  Pass your values with -f (global.tenant must be set), e.g.:"
  echo "    $0 -f my-values.yaml prereq"
  echo "  Tried: helm template kumo $CHART_DIR ${VAL_ARGS[*]:-} --namespace $NAMESPACE ${HELM_CTX_ARGS[*]:-} --show-only templates/tests/probe-inputs.yaml"
  exit 2
fi
jget(){ printf '%s' "$INPUTS" | jq -r ".$1 // empty"; }

TENANT=$(jget tenant)
DEPLOY_STAGE=$(jget deployStage)
# Match DeployEnvConfig.id_suffix and the pool controller's Kubernetes
# normalization. DEV modes also prefix the runtime container username, which a
# values-only probe cannot derive.
case "$DEPLOY_STAGE" in
  prod|staging|preprod) POOL_ID_SUFFIX="$TENANT" ;;
  test) POOL_ID_SUFFIX="${TENANT}-test" ;;
  *) POOL_ID_SUFFIX="" ;;
esac
POOL_TENANT=$(printf '%s' "$POOL_ID_SUFFIX" |
  tr '[:upper:]' '[:lower:]' |
  sed -E 's/[^a-z0-9-]/-/g; s/^-+//')
NS=$(jget namespace)
SC=$(jget storageClass)
CSI_CAPACITY_POLICY=$(jget storageCapacityTracking)
CSI_CAPACITY_POLICY=${CSI_CAPACITY_POLICY:-required}
IMAGE_REGISTRY=$(jget imageRegistry); IMAGE_PULL_SECRET=$(jget imagePullSecret)
TQ_ADDR=$(jget temporalAddress)
TQ_SECRET=$(jget temporalClientMtlsSecret)
REG=$(jget registerNamespace)
STORAGE_ROOT=$(jget storageRoot); SCHEME="${STORAGE_ROOT%%://*}"
GCP_ADC_SECRET=$(jget gcpApplicationCredentialsSecret)
GCP_ADC_KEY=$(jget gcpApplicationCredentialsKey)
IDENTITY_ANNOTATIONS=$(printf '%s' "$INPUTS" | jq -c '.identityAnnotations // {}')
IDENTITY_POD_LABELS=$(printf '%s' "$INPUTS" | jq -c '.identityPodLabels // {}')
PG_AUTH_MODE=$(jget postgresAuthMode); PG_HOST=$(jget postgresHost)
PG_PORT=$(jget postgresPort)
NODE_SEL=$(printf '%s' "$INPUTS" | jq -c '.nodeSelector // {}')
REST_SVC=$(jget restSvc); API_SVC=$(jget apiSvc); UI_SVC=$(jget uiSvc)
EXPOSURE_HOST=$(jget exposureHost)

FAILS=0
declare -A ST
ok(){   ST[$1]=PASS;  printf '  \033[32mPASS \033[0m %-16s %s\n' "$1" "$2"; }
bad(){  ST[$1]=FAIL;  FAILS=$((FAILS+1)); printf '  \033[31mFAIL \033[0m %-16s %s\n' "$1" "$2"; }
skip(){ ST[$1]=SKIP;  printf '  \033[33mSKIP \033[0m %-16s %s\n' "$1" "$2"; }
blk(){  ST[$1]=BLOCK; FAILS=$((FAILS+1)); printf '  \033[35mBLOCK\033[0m %-16s %s\n' "$1" "$2"; }
# advisory: visible but NON-blocking (does not count toward the exit code) — used for
# materialize/train-time prerequisites that aren't required just to deploy the tenant.
adv(){  ST[$1]=ADV;   printf '  \033[36mADV  \033[0m %-16s %s\n' "$1" "$2"; }
# need <id>...: succeed only if every listed check already PASSed
need(){ local d; for d in "$@"; do [ "${ST[$d]:-}" = PASS ] || return 1; done; }

# resolve_incluster_svc <host>: best-effort map a short host to an in-cluster Service.
# Echoes "ns/svc" and returns 0 if a Service is found; returns 1 otherwise.
#   *.svc / *.svc.cluster.local  → svc.ns from the FQDN (caller's strict branch handles FAIL)
#   2-label a.b                  → Service 'a' in namespace 'b'
#   bare a                       → Service 'a' in the release namespace
# 3+ label FQDNs and IPs are not treated as in-cluster (return 1).
resolve_incluster_svc(){
  local h="$1" ns
  case "$h" in
    *.svc|*.svc.cluster.local)
      local svc="${h%%.*}" rest="${h#*.}"; ns="${rest%%.*}"
      kubectl get svc "$svc" -n "$ns" >/dev/null 2>&1 && { printf '%s/%s' "$ns" "$svc"; return 0; }
      return 1 ;;
    *.*.*) return 1 ;;                                  # 3+ labels → external FQDN
    *.*)                                                # 2-label a.b → svc 'a' in ns 'b'
      local svc="${h%%.*}"; ns="${h#*.}"
      kubectl get svc "$svc" -n "$ns" >/dev/null 2>&1 && { printf '%s/%s' "$ns" "$svc"; return 0; }
      return 1 ;;
    *)                                                  # bare single-label → try the release ns
      kubectl get svc "$h" -n "$NS" >/dev/null 2>&1 && { printf '%s/%s' "$NS" "$h"; return 0; }
      return 1 ;;
  esac
}

filter_ready_nodes(){
  jq '{
    items: [.items[]
      | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))]
  }'
}

load_current_kumo_nodes(){
  KUMO_NODE_QUERY_ERROR=""
  if [ -n "$sel" ]; then
    CPU_NODES_JSON=$(kubectl get nodes -l "$sel" -o json 2>/dev/null |
      filter_ready_nodes) ||
      KUMO_NODE_QUERY_ERROR="could not list CPU/hook nodes for selector {$sel}"
  else
    # With no selector, target ordinary Ready workers rather than control-plane
    # or cordoned nodes where chart workloads do not schedule by default.
    CPU_NODES_JSON=$(kubectl get nodes -o json 2>/dev/null |
      filter_ready_nodes |
      jq '{
        items: [.items[]
          | select((.spec.unschedulable // false) == false)
          | select((.metadata.labels // {} | has("node-role.kubernetes.io/control-plane")) | not)
          | select((.metadata.labels // {} | has("node-role.kubernetes.io/master")) | not)]
      }') ||
      KUMO_NODE_QUERY_ERROR="could not list CPU/hook nodes"
  fi
  CACHE_NODES_JSON=$(kubectl get nodes \
    -l "kumo.ai/diskgraph-pool-type=cache" -o json 2>/dev/null |
    filter_ready_nodes) ||
    KUMO_NODE_QUERY_ERROR="could not list DiskGraph cache nodes"
  EPH_NODES_JSON=$(kubectl get nodes \
    -l "kumo.ai/diskgraph-pool-type=ephemeral" -o json 2>/dev/null |
    filter_ready_nodes) ||
    KUMO_NODE_QUERY_ERROR="could not list DiskGraph ephemeral nodes"
  [ -z "$KUMO_NODE_QUERY_ERROR" ] || return 1

  KUMO_NODES_JSON=$(printf '%s\n%s\n%s\n' \
    "$CPU_NODES_JSON" "$CACHE_NODES_JSON" "$EPH_NODES_JSON" |
    jq -sc '{items: ([.[].items[]] | unique_by(.metadata.uid // .metadata.name))}') ||
    {
      KUMO_NODE_QUERY_ERROR="could not combine current Kumo target nodes"
      return 1
    }
  KUMO_NODE_GAPS=""
  [ "$(printf '%s' "$CPU_NODES_JSON" | jq '.items | length')" -gt 0 ] ||
    KUMO_NODE_GAPS="${KUMO_NODE_GAPS} CPU/hook"
  [ "$(printf '%s' "$CACHE_NODES_JSON" | jq '.items | length')" -gt 0 ] ||
    KUMO_NODE_GAPS="${KUMO_NODE_GAPS} cache"
  [ "$(printf '%s' "$EPH_NODES_JSON" | jq '.items | length')" -gt 0 ] ||
    KUMO_NODE_GAPS="${KUMO_NODE_GAPS} ephemeral"
}

prereq(){
  echo "── PREREQ (tenant=${TENANT}, storage=${SCHEME}, temporal=custom) ──"

  # root: cluster reachable. Parse BOTH major and minor (strip GKE/EKS '28+' suffixes with
  # grep -oE '[0-9]+'); require major == 1 AND minor >= 28.
  KVER=$(kubectl version -o json 2>/dev/null)
  vmaj=$(printf '%s' "$KVER" | jq -r '.serverVersion.major // empty' | grep -oE '[0-9]+' | head -1)
  vmin=$(printf '%s' "$KVER" | jq -r '.serverVersion.minor // empty' | grep -oE '[0-9]+' | head -1)
  if [ -n "$vmaj" ] && [ -n "$vmin" ] && [ "$vmaj" -eq 1 ] && [ "$vmin" -ge 28 ]; then
    ok K8S "API reachable, server ${vmaj}.${vmin} (>=1.28)"
  else bad K8S "Kubernetes API unreachable or server ${vmaj:-?}.${vmin:-?} (<1.28)"; fi

  # WFFC cache CSI configuration. Live PVC binding remains the
  # pool controller init canary's responsibility.
  if need K8S; then
    if ! SC_JSON=$(kubectl get sc "$SC" --ignore-not-found -o json 2>/dev/null); then
      bad CSI "could not query configured StorageClass '${SC}' — check Kubernetes API access/RBAC"
    elif [ -z "$SC_JSON" ]; then
      bad CSI "configured StorageClass '${SC}' not found (kumoConfig.diskgraph.storageClass)"
    else
      CSI_DRIVER=$(printf '%s' "$SC_JSON" | jq -r '.provisioner // empty')
      DRIVER_JSON=""
      CSI_CAPACITY_ENABLED=false
      CSI_DRIVER_QUERY_FAILED=false
      if [ -n "$CSI_DRIVER" ] &&
        ! DRIVER_JSON=$(kubectl get csidriver "$CSI_DRIVER" --ignore-not-found -o json 2>/dev/null); then
        CSI_DRIVER_QUERY_FAILED=true
      fi
      if [ "$CSI_DRIVER_QUERY_FAILED" = true ]; then
        bad CSI "could not query CSIDriver '${CSI_DRIVER}' — check Kubernetes API access/RBAC"
      elif CSI_ERROR=$(csi_validate_config "$SC_JSON" "$DRIVER_JSON" "$CSI_CAPACITY_POLICY"); then
        CSI_CAPACITY_ENABLED=$(printf '%s' "$DRIVER_JSON" | jq -r '.spec.storageCapacity // false')
        if [ "$CSI_CAPACITY_ENABLED" = true ]; then
          ok CSI "StorageClass '${SC}' uses WaitForFirstConsumer and reclaimPolicy=Delete; CSIDriver '${CSI_DRIVER}' advertises storageCapacity"
        else
          ok CSI "StorageClass '${SC}' uses WaitForFirstConsumer and reclaimPolicy=Delete; CSIDriver '${CSI_DRIVER}' is present and capacity tracking is advisory"
        fi
      else
        bad CSI "StorageClass '${SC}' does not satisfy storageCapacityTracking=${CSI_CAPACITY_POLICY}: ${CSI_ERROR}"
      fi
    fi
  else
    blk CSI "needs K8S"
  fi

  if need CSI; then
    if [ "$CSI_CAPACITY_POLICY" = advisory ] && [ "$CSI_CAPACITY_ENABLED" != true ]; then
      adv CSI_CAPACITY "CSIDriver '${CSI_DRIVER}' does not advertise storageCapacity; allowed by the advisory policy for dynamically provisioned storage, and the pool controller init canary is authoritative"
    elif ! CAPACITY_JSON=$(kubectl get csistoragecapacities.storage.k8s.io -A -o json 2>/dev/null); then
      adv CSI_CAPACITY "capacity unknown/unqueryable for '${SC}' — could not query CSIStorageCapacity; check RBAC/API availability"
    elif csi_has_published_capacity "$SC" "$CAPACITY_JSON"; then
      ok CSI_CAPACITY "nonzero CSIStorageCapacity published for '${SC}'"
    else
      adv CSI_CAPACITY "no nonzero CSIStorageCapacity currently published for '${SC}' — expected while a node-local Karpenter pool is scaled to zero; the pool controller init canary is authoritative"
    fi
  else
    adv CSI_CAPACITY "needs CSI"
  fi

  # Node targeting applies to every workload and install hook. A generic
  # autoscaler signal cannot prove it can create the requested labels, so an
  # explicit selector requires at least one Ready match before installation.
  # We do NOT compute aggregate or per-node capacity. Size steady-state aggregate
  # capacity from resolved workloads.* replicas/requests, and per-node fit from
  # every rendered Pod, including pre-install hooks and ephemeral-storage requests.
  # An undersized matching pool passes here but leaves pods Pending.
  # (Taints must also be tolerated via nodeTargeting.tolerations — not asserted here.)
  if need K8S; then
    sel=$(printf '%s' "$NODE_SEL" | jq -r 'to_entries|map("\(.key)=\(.value)")|join(",")')
    if [ -z "$sel" ]; then ok NODE_TARGETING "no nodeSelector set — no label restriction; verify steady-state workload capacity and per-node fit for rendered workloads and hooks"
    else
      n=$(kubectl get nodes -l "$sel" --no-headers 2>/dev/null | grep -c ' Ready ')
      if { [ -n "$n" ] && [ "$n" -ge 1 ]; } 2>/dev/null; then
        ok NODE_TARGETING "${n} Ready node(s) match {${sel}} — verify steady-state workload capacity and per-node fit for every rendered Pod, including hooks and ephemeral storage"
      else
        bad NODE_TARGETING "no Ready node matches {${sel}} — add matching install capacity or remove nodeTargeting.nodeSelector; generic autoscaler presence cannot prove that it creates these labels"
      fi
    fi
  else blk NODE_TARGETING "needs K8S"; fi

  # release namespace — gate for everything namespaced
  if need K8S; then
    kubectl get ns "$NS" >/dev/null 2>&1 && ok NAMESPACES "namespace ${NS} exists" \
      || bad NAMESPACES "missing namespace ${NS}"
  else blk NAMESPACES "needs K8S"; fi

  # Private-registry credentials. Presence/type is enforceable; token freshness
  # is registry-specific (not encoded in dockerconfigjson), so ECR rotation
  # remains an operator responsibility.
  if [ -z "$IMAGE_PULL_SECRET" ]; then
    if [ "$SCHEME" = abfss ] &&
      printf '%s' "$IMAGE_REGISTRY" | grep -q '\.dkr\.ecr\.'; then
      bad IMAGE_PULL_SECRET "ADLS profile pulls from ECR but image.pullSecretName is empty; configure and refresh the ECR dockerconfigjson Secret"
    else
      skip IMAGE_PULL_SECRET "image.pullSecretName is empty — relying on node/kubelet registry credentials"
    fi
  elif need NAMESPACES; then
    PULL_SECRET_JSON=""
    if ! PULL_SECRET_JSON=$(kubectl get secret "$IMAGE_PULL_SECRET" -n "$NS" --ignore-not-found -o json 2>/dev/null); then
      bad IMAGE_PULL_SECRET "could not query image pull Secret '${IMAGE_PULL_SECRET}' in ${NS} — check Kubernetes API access/RBAC"
    elif [ -z "$PULL_SECRET_JSON" ]; then
      bad IMAGE_PULL_SECRET "image pull Secret '${IMAGE_PULL_SECRET}' is missing in ${NS}"
    elif printf '%s' "$PULL_SECRET_JSON" | jq -e \
      '.type == "kubernetes.io/dockerconfigjson" and (.data | has(".dockerconfigjson"))' >/dev/null; then
      if printf '%s' "$IMAGE_REGISTRY" | grep -q '\.dkr\.ecr\.'; then
        ok IMAGE_PULL_SECRET "ECR pull Secret '${IMAGE_PULL_SECRET}' has the required type/key; refresh it immediately before install because token expiry cannot be inferred"
      else
        ok IMAGE_PULL_SECRET "image pull Secret '${IMAGE_PULL_SECRET}' has the required type/key"
      fi
    else
      bad IMAGE_PULL_SECRET "image pull Secret '${IMAGE_PULL_SECRET}' must be type kubernetes.io/dockerconfigjson with data[.dockerconfigjson]"
    fi
  else
    blk IMAGE_PULL_SECRET "needs NAMESPACES"
  fi


  host="${TQ_ADDR%%:*}"
  temporal_port=7233
  case "$TQ_ADDR" in
    *:*) temporal_port="${TQ_ADDR##*:}" ;;
  esac

  if [ -z "$EXPOSURE_HOST" ]; then
    skip EDGE_HTTPS "exposure.host is empty; diagnostic port-forward only"
  else
    edge_code=$(curl "${EDGE_CURL_TLS_ARGS[@]}" \
      --max-time 15 -sS -o /dev/null -w '%{http_code}' \
      "https://${EXPOSURE_HOST}/-/ready" 2>/dev/null || true)
    if [ -z "$edge_code" ] || [ "$edge_code" = "000" ]; then
      if [ "$PHASE" = prereq ]; then
        adv EDGE_HTTPS "public HTTPS is not reachable yet; finish the cloud edge after the kumo-ui Service exists, then run the all phase before exposure"
      else
        bad EDGE_HTTPS "https://${EXPOSURE_HOST} failed DNS resolution, TLS handshake, or connection"
      fi
    elif [ "$PHASE" = all ]; then
      edge_rest_code=$(curl "${EDGE_CURL_TLS_ARGS[@]}" \
        --max-time 15 -sS -o /dev/null -w '%{http_code}' \
        "https://${EXPOSURE_HOST}/internal/health" 2>/dev/null || true)
      edge_api_code=$(curl "${EDGE_CURL_TLS_ARGS[@]}" \
        --max-time 15 -sS -o /dev/null -w '%{http_code}' \
        "https://${EXPOSURE_HOST}/api/health" 2>/dev/null || true)
      if [ "$edge_code" = "200" ] &&
        [ "$edge_rest_code" = "200" ] &&
        [ "$edge_api_code" = "200" ]; then
        ok EDGE_HTTPS "HTTPS catch-all reaches UI, REST, and API through the frontend router"
      else
        bad EDGE_HTTPS "external HTTPS checks returned ready=${edge_code:-no-response}, REST=${edge_rest_code:-no-response}, API=${edge_api_code:-no-response}; expected 200 through the frontend router"
      fi
    else
      ok EDGE_HTTPS "https://${EXPOSURE_HOST} completed an HTTPS request (HTTP ${edge_code})"
    fi
  fi

  # A nonempty clientMtlsSecret selects native Temporal mTLS.
  if [ -n "$TQ_SECRET" ]; then
    if need NAMESPACES; then
      TQ_SECRET_JSON=$(kubectl get secret "$TQ_SECRET" -n "$NS" \
        --ignore-not-found -o json 2>/dev/null) || TQ_SECRET_JSON=""
      if [ -n "$TQ_SECRET_JSON" ] && printf '%s' "$TQ_SECRET_JSON" | jq -e '
        .type == "kubernetes.io/tls"
        and (.data | has("tls.crt") and has("tls.key") and has("ca.crt"))
      ' >/dev/null 2>&1
      then
        ok TEMPORAL_MTLS "client mTLS Secret '${TQ_SECRET}' has tls.crt, tls.key, and ca.crt"
      else
        bad TEMPORAL_MTLS "client mTLS Secret '${TQ_SECRET}' must be kubernetes.io/tls with tls.crt, tls.key, and ca.crt"
      fi
    else
      blk TEMPORAL_MTLS "needs NAMESPACES"
    fi
  else
    skip TEMPORAL_MTLS "temporal.clientMtlsSecret is not set; Temporal transport is the operator's to secure"
  fi

  # Native mTLS may target an external endpoint, so a non-resolving address is
  # informational rather than a failure.
  if found=$(resolve_incluster_svc "$host"); then
    ok TEMPORAL_ACCESS "in-cluster Temporal Service '${found}' is present"
  else
    skip TEMPORAL_ACCESS "Temporal '${TQ_ADDR}' does not resolve to an in-cluster Service; verify reachability separately"
  fi

  # temporal namespace registration — handled by the chart
  skip TEMPORAL_NS "registered by the chart's pre-install hook Job (registerNamespace=${REG})"

  # Provider profiles remain values-only. Validate their resolved identity
  # metadata here so the chart templates stay cloud-neutral.
  if need K8S; then
    if [ "$SCHEME" = gs ] && [ -n "$GCP_ADC_SECRET" ]; then
      # Ask kubectl's Go template for Secret key names only. Never print or
      # retain the service-account JSON value in conformance output.
      # $key and $_ below are Go-template variables, not shell expressions.
      # shellcheck disable=SC2016
      if ! gcp_adc_keys=$(kubectl get secret "$GCP_ADC_SECRET" -n "$NS" \
        --ignore-not-found \
        -o 'go-template={{range $key, $_ := .data}}{{println $key}}{{end}}' \
        2>/dev/null); then
        bad STORAGE_IDENTITY "could not query static GCP ADC Secret '${GCP_ADC_SECRET}' in ${NS} — check Kubernetes API access/RBAC"
      elif printf '%s\n' "$gcp_adc_keys" | grep -Fxq -- "$GCP_ADC_KEY"; then
        ok STORAGE_IDENTITY "static GCP ADC Secret '${GCP_ADC_SECRET}' contains configured key '${GCP_ADC_KEY}'"
      else
        bad STORAGE_IDENTITY "static GCP ADC Secret '${GCP_ADC_SECRET}' was not found in ${NS} or does not contain configured key '${GCP_ADC_KEY}'"
      fi
    else
      case "$SCHEME" in
        s3|gs|abfss)
          case "$SCHEME" in
            s3)    identity_key="eks.amazonaws.com/role-arn"; identity_kind=AWS ;;
            gs)    identity_key="iam.gke.io/gcp-service-account"; identity_kind=GCP ;;
            abfss) identity_key="azure.workload.identity/client-id"; identity_kind=Azure ;;
          esac
          identity_errors=""
          for role in control worker controller; do
            identity_value=$(printf '%s' "$IDENTITY_ANNOTATIONS" |
              jq -r --arg role "$role" --arg key "$identity_key" \
                '.[$role][$key] // empty')
            case "$SCHEME" in
              s3)
                [[ "$identity_value" =~ ^arn:(aws|aws-us-gov|aws-cn):iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+$ ]] ||
                  identity_errors="${identity_errors} ${role}:${identity_key}" ;;
              gs)
                [[ "$identity_value" == *@*.iam.gserviceaccount.com ]] ||
                  identity_errors="${identity_errors} ${role}:${identity_key}" ;;
              abfss)
                [[ "$identity_value" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] ||
                  identity_errors="${identity_errors} ${role}:${identity_key}"
                identity_label=$(printf '%s' "$IDENTITY_POD_LABELS" |
                  jq -r --arg role "$role" \
                    '.[$role]["azure.workload.identity/use"] // empty')
                [ "$identity_label" = true ] ||
                  identity_errors="${identity_errors} ${role}:azure.workload.identity/use=true" ;;
            esac
          done
          if [ -z "$identity_errors" ]; then
            ok STORAGE_IDENTITY "${identity_kind} workload identity is configured for control, worker, and controller"
          else
            bad STORAGE_IDENTITY "${identity_kind} workload identity has missing or invalid fields:${identity_errors}"
          fi ;;
        *) skip STORAGE_IDENTITY "non-object backend ('${SCHEME}') — supply workload identity via the SA annotations" ;;
      esac
    fi
  else blk STORAGE_IDENTITY "needs K8S"; fi

  # Storage reachability from the operator is only a proxy: workload identity
  # deliberately uses a different principal. A successful list is useful, but
  # an operator authorization failure cannot prove the workload will fail.
  bkt="${STORAGE_ROOT#*://}"; bkt="${bkt%%/*}"
  case "$SCHEME" in
    s3)
      if command -v aws >/dev/null 2>&1; then
        aws s3 ls "s3://${bkt}/" >/dev/null 2>&1 && ok STORAGE_ACCESS "s3 bucket '${bkt}' reachable (root ${STORAGE_ROOT})" \
          || adv STORAGE_ACCESS "cannot list s3://${bkt} with operator credentials — verify from a worker-identity pod after install"
      else skip STORAGE_ACCESS "s3 backend but awscli not installed — install it to verify '${STORAGE_ROOT}'"; fi ;;
    gs)
      if command -v gcloud >/dev/null 2>&1; then
        gcloud storage ls "gs://${bkt}/" >/dev/null 2>&1 && ok STORAGE_ACCESS "gcs bucket '${bkt}' reachable" \
          || adv STORAGE_ACCESS "cannot list gs://${bkt} with operator credentials — verify from a worker-identity pod after install"
      elif command -v gsutil >/dev/null 2>&1; then
        gsutil ls "gs://${bkt}/" >/dev/null 2>&1 && ok STORAGE_ACCESS "gcs bucket '${bkt}' reachable" \
          || adv STORAGE_ACCESS "cannot list gs://${bkt} with operator credentials — verify from a worker-identity pod after install"
      else skip STORAGE_ACCESS "gcs backend but gcloud/gsutil not installed"; fi ;;
    abfss)
      # abfss://<container>@<account>.dfs.core.windows.net/<path> — probe with the operator's AAD creds.
      if command -v az >/dev/null 2>&1; then
        cont="${bkt%%@*}"; host="${bkt#*@}"; acct="${host%%.*}"
        if [ "$bkt" = "$cont" ] || [ -z "$acct" ]; then
          skip STORAGE_ACCESS "azure backend '${STORAGE_ROOT}' — cannot parse <container>@<account> from the URI to probe"
        elif az storage fs show --auth-mode login --account-name "$acct" --name "$cont" >/dev/null 2>&1; then
          ok STORAGE_ACCESS "ADLS filesystem '${cont}' reachable on account '${acct}' (root ${STORAGE_ROOT})"
        else
          adv STORAGE_ACCESS "cannot access ADLS filesystem '${cont}' on account '${acct}' with operator credentials — verify from an Azure-workload-identity pod after install"
        fi
      else skip STORAGE_ACCESS "azure backend but az CLI not installed — install it to verify '${STORAGE_ROOT}'"; fi ;;
    *) skip STORAGE_ACCESS "unrecognized storage scheme '${SCHEME}' in '${STORAGE_ROOT}'" ;;
  esac

  case "$PG_AUTH_MODE" in
    aws_iam) ok POSTGRES_CA "AWS RDS roots are baked into the Kumo image" ;;
    azure_entra) ok POSTGRES_CA "Azure Postgres uses the Kumo image system trust store" ;;
    *) bad POSTGRES_CA "postgres.authMode must be aws_iam or azure_entra" ;;
  esac
  if need NAMESPACES; then
    case "$PG_AUTH_MODE" in
      aws_iam|azure_entra)
        ok STATE_DB_RUNTIME "passwordless Postgres auth mode '${PG_AUTH_MODE}' configured" ;;
      *) bad STATE_DB_RUNTIME "postgres.authMode must be aws_iam or azure_entra" ;;
    esac
  else blk STATE_DB_RUNTIME "needs NAMESPACES"; fi

  need POSTGRES_CA \
    && ok POSTGRES_TLS "chart enforces verify-full TLS with the configured CA" \
    || blk POSTGRES_TLS "needs POSTGRES_CA"

  if need STATE_DB_RUNTIME; then
      case "$PG_HOST" in
        *.svc|*.svc.cluster.local)
          if found=$(resolve_incluster_svc "$PG_HOST"); then
            ok PG_ACCESS "in-cluster Postgres svc '${found}' present"
          else
            bad PG_ACCESS "declared in-cluster Postgres '${PG_HOST}' does not resolve to a Service"
          fi ;;
        *.*.*) skip PG_ACCESS "external Postgres '${PG_HOST}:${PG_PORT}' — in-cluster preflight performs the authoritative dial" ;;
        *)
          if found=$(resolve_incluster_svc "$PG_HOST"); then
            ok PG_ACCESS "in-cluster Postgres svc '${found}' present"
          else
            skip PG_ACCESS "Postgres host '${PG_HOST}' has no matching Service"
          fi ;;
      esac
  else blk PG_ACCESS "needs STATE_DB_RUNTIME"; fi

  # ── Materialize/train-time prerequisites (NON-blocking advisories) ───────────
  # Read-only PRE-FLIGHT for the common label/plugin mistakes. The pool controller
  # authoritatively validates these at startup (init canaries use a
  # kumoConfig.diskgraph-sized PVC + GPU pod request; POOL_INIT_CANARY_ENABLED defaults on;
  # POOL_CANARY_GRACE_SECS defaults to 600s). An expected cache-tier failure excludes only
  # that tier from new provisioning; existing routable workers remain usable. An expected
  # ephemeral failure disables overflow. Startup is fatal only with zero available cache tiers
  # or an unexpected error. Reported as ADV so this read-only pre-flight stays green while
  # flagging gaps loudly. The kumo-stack platform guides provision the pools.

  # GPU capability — DiskGraph cache/eph activity pods request nvidia.com/gpu. GPU can be
  # provided by the standalone NVIDIA k8s-device-plugin, the NVIDIA GPU Operator (which installs
  # the device plugin + driver and a clusterpolicies.nvidia.com CR), or a cloud-managed GPU stack
  # (GKE/EKS). Rather than match one DaemonSet name, accept ANY positive signal. This stays a
  # best-effort advisory: the pool controller's init canary (which actually requests
  # nvidia.com/gpu) is the authoritative check, so a missed signal here only loses a pre-flight hint.
  if need K8S; then
    gpu_sig=""
    # (1) Definitive — a node already advertises allocatable nvidia.com/gpu (any install method;
    #     absent when the GPU pools are scale-to-zero / not yet provisioned).
    kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null \
      | grep -qE '[1-9]' && gpu_sig="a node advertises allocatable nvidia.com/gpu"
    # (2) NVIDIA GPU Operator present (it owns the device plugin + driver).
    [ -z "$gpu_sig" ] && kubectl get crd clusterpolicies.nvidia.com >/dev/null 2>&1 \
      && gpu_sig="NVIDIA GPU Operator (clusterpolicies.nvidia.com) installed"
    # (3) A device-plugin / GPU-operator DaemonSet by common name (standalone plugin or operator).
    [ -z "$gpu_sig" ] && kubectl get ds -A --no-headers 2>/dev/null \
      | grep -qiE 'nvidia.*device.?plugin|gpu-?operator|nvidia-gpu-device-plugin' \
      && gpu_sig="an NVIDIA device-plugin / GPU-operator DaemonSet"
    if [ -n "$gpu_sig" ]; then adv GPU "GPU capability detected ($gpu_sig)"
    else adv GPU "no GPU signal (no node advertising nvidia.com/gpu, no GPU Operator ClusterPolicy, no device-plugin DaemonSet) — ensure GPU capability before materialize/train. This advisory can miss scale-to-zero or cloud-managed capacity; the pool controller's init canary is authoritative, and startup is fatal only if zero cache tiers validate."; fi
  else adv GPU "needs K8S"; fi

  # DiskGraph cache + ephemeral node pools — the pool controller schedules cache daemons onto
  # nodes labeled {kumo.ai/diskgraph-pool-type=cache, kumo.ai/diskgraph-ssd-size=<ssd_gib>,
  # kumo.ai/diskgraph-gpu-count=<gpu_count>} (one tier per effective runtime
  # pool_diskgraph_node_types entry, normally assembled from kumoConfig.diskgraph.nodeTypes;
  # gpu_count defaults to 1) and ephemeral activity pods onto
  # {kumo.ai/diskgraph-pool-type=ephemeral} (no size/gpu-count label). Cache pods
  # bind the configured WFFC CSI class; ephemeral jobs use node-local storage.
  # Taints are optional (cache tolerates dedicated=diskgraph-<tenant>, eph
  # dedicated=diskgraph-eph-<tenant>; <tenant> is the stage-aware runtime pool
  # tenant normalized to [a-z0-9-]), plus configured poolExtraTolerations. We check
  # for a Karpenter NodePool template (or a Ready node) carrying the labels.
  TIERS=$(printf '%s' "$INPUTS" | jq -r '
    def runtime_int:
      if type == "boolean" then
        if . then 1 else 0 end
      elif type == "number" then
        select(. == floor)
      elif type == "string" and
          test("^[[:space:]]*[+-]?[0-9]+(?:_[0-9]+)*(?:\\.0+)?[[:space:]]*$") then
        gsub("_"; "") | tonumber
      else
        empty
      end;
    reduce (.diskgraphTiers[]?
      | {ssd: (.ssd | runtime_int), gpu: (.gpu | runtime_int)}
      | select(.ssd > 0 and .gpu > 0)) as $tier
      ([];
        if any(.[]; .ssd == $tier.ssd and .gpu == $tier.gpu) then
          .
        else
          . + [$tier]
        end)
    | .[]
    | "\(.ssd | tostring) \(.gpu | tostring)"
  ' 2>/dev/null)
  if need K8S; then
    NP=$(kubectl get nodepools.karpenter.sh -o json 2>/dev/null)
    dg_selector(){ # $1=pool-type, $2=ssd-size (optional), $3=gpu-count (optional)
      local sel="kumo.ai/diskgraph-pool-type=$1"
      [ -n "${2:-}" ] && sel="${sel},kumo.ai/diskgraph-ssd-size=$2"
      [ -n "${3:-}" ] && sel="${sel},kumo.ai/diskgraph-gpu-count=$3"
      printf '%s' "$sel"
    }
    dg_has(){ # $1=pool-type, $2=ssd-size (optional), $3=gpu-count (optional)
      local sel
      sel=$(dg_selector "$@")
      kubectl get nodes -l "$sel" --no-headers 2>/dev/null | grep -q ' Ready ' && return 0
      [ -n "$NP" ] && printf '%s' "$NP" | jq -e \
        --arg pt "$1" --arg s "${2:-}" --arg g "${3:-}" \
        '.items[]?.spec.template.metadata.labels // {} | select(.["kumo.ai/diskgraph-pool-type"]==$pt and ($s=="" or .["kumo.ai/diskgraph-ssd-size"]==$s) and ($g=="" or .["kumo.ai/diskgraph-gpu-count"]==$g))' >/dev/null 2>&1
    }
    miss=""; shown=""
    while read -r s g; do
      [ -n "$s" ] || continue
      dg_has cache "$s" "$g" || miss="${miss} cache(ssd-size=${s},gpu-count=${g})"
      shown="${shown}${s}Gi/${g}gpu "
    done <<<"$TIERS"
    dg_has ephemeral "" "" || miss="${miss} ephemeral"
    if [ -z "$TIERS" ]; then adv DISKGRAPH_POOLS "no valid diskgraph nodeTypes configured (entries must be maps with ssd_gib) — controller startup has zero cache tiers and is fatal when init-canary validation is enabled"
    elif [ -z "$miss" ]; then adv DISKGRAPH_POOLS "cache pools for tiers [${shown}] + ephemeral pool present (labels match)"
    else adv DISKGRAPH_POOLS "MATERIALIZE-TIME: missing diskgraph pool(s):${miss} — create GPU node pool(s) (Karpenter NodePool / EKS managed nodegroup / GKE or AKS node pool) labeled kumo.ai/diskgraph-pool-type=cache|ephemeral (cache also needs kumo.ai/diskgraph-ssd-size=<ssd_gib> AND kumo.ai/diskgraph-gpu-count=<gpu_count> per tier, matching kumoConfig.diskgraph.nodeTypes). Cache nodes must bind the configured WFFC CSI class; ephemeral nodes need sufficient fast kubelet storage. Helm install may complete, but controller startup requires at least one available cache tier; unavailable cache tiers are excluded from new provisioning and a missing ephemeral pool disables overflow. See RUNBOOK."
    fi
  else adv DISKGRAPH_POOLS "needs K8S"; fi

  # A cache PVC can bind only where the StorageClass driver's node plugin is
  # registered. Check the exact cache selectors used by the pool controller;
  # CSINode is authoritative for registration (CSIStorageCapacity is not).
  # Judge only nodes where the cache pod can schedule: Ready, non-cordoned, and
  # not blocked by an untolerated hard taint.
  POOL_EXTRA_TOLERATIONS=$(printf '%s' "$INPUTS" | jq -c '.poolExtraTolerations // []')
  if need CSI; then
    if [ -z "$TIERS" ]; then
      adv CSI_NODE_COVERAGE "no valid effective DiskGraph cache tiers to inspect"
    elif [ -z "$POOL_TENANT" ]; then
      adv CSI_NODE_COVERAGE "cannot derive the runtime DiskGraph pool tenant for deploy stage '${DEPLOY_STAGE}'; coverage unknown. The pool controller init canary is authoritative"
    elif ! CSINODES_JSON=$(kubectl get csinodes.storage.k8s.io -o json 2>/dev/null); then
      adv CSI_NODE_COVERAGE "could not list CSINodes for driver '${CSI_DRIVER}' — coverage unknown; check Kubernetes API access/RBAC"
    else
      cov_live=0; cov_empty=""; cov_ineligible=""; cov_uncovered=""; cov_errors=""
      while read -r s g; do
        [ -n "$s" ] || continue
        cov_sel=$(dg_selector cache "$s" "$g")
        if ! TIER_NODES=$(kubectl get nodes -l "$cov_sel" -o json 2>/dev/null); then
          cov_errors="${cov_errors}${cov_errors:+; }could not list nodes for {${cov_sel}}"
        elif ! tier_cov=$(printf '{"nodes":%s,"csinodes":%s}' "$TIER_NODES" "$CSINODES_JSON" |
          csi_node_coverage "$CSI_DRIVER" dedicated "diskgraph-${POOL_TENANT}" "$POOL_EXTRA_TOLERATIONS" 2>/dev/null); then
          cov_errors="${cov_errors}${cov_errors:+; }could not evaluate nodes for {${cov_sel}}"
        else
          tier_matched=$(printf '%s' "$tier_cov" | jq -r '.matched')
          tier_live=$(printf '%s' "$tier_cov" | jq -r '.eligible')
          tier_uncovered=$(printf '%s' "$tier_cov" | jq -r '.uncovered | join(", ")')
          cov_live=$((cov_live + tier_live))
          if [ "$tier_matched" -eq 0 ]; then
            cov_empty="${cov_empty}${cov_empty:+, }${s}Gi/${g}gpu"
          elif [ "$tier_live" -eq 0 ]; then
            cov_ineligible="${cov_ineligible}${cov_ineligible:+, }${s}Gi/${g}gpu"
          fi
          [ -n "$tier_uncovered" ] && cov_uncovered="${cov_uncovered}${cov_uncovered:+; }${tier_uncovered} (${s}Gi/${g}gpu)"
        fi
      done <<<"$TIERS"

      if [ -n "$cov_uncovered" ]; then
        bad CSI_NODE_COVERAGE "cache node(s) ${cov_uncovered} do not register CSI driver '${CSI_DRIVER}'. Extend the CSI node DaemonSet's scheduling rules so it runs on every DiskGraph cache pool; then verify the driver appears in each node's CSINode.${cov_errors:+ Also: ${cov_errors}.}"
      elif [ -n "$cov_errors" ]; then
        adv CSI_NODE_COVERAGE "${cov_errors} — coverage unknown; check Kubernetes API access/RBAC"
      elif [ -n "$cov_ineligible" ]; then
        if [ "$cov_live" -eq 0 ]; then
          adv CSI_NODE_COVERAGE "matching cache nodes exist for tier(s) [${cov_ineligible}], but none are currently eligible (Ready, non-cordoned, and compatible with the cache pod's hard-taint tolerations); CSI coverage is limited to eligible nodes${cov_empty:+. Tier(s) [${cov_empty}] have no matching cache nodes (scale-to-zero)}"
        else
          adv CSI_NODE_COVERAGE "driver '${CSI_DRIVER}' is registered on ${cov_live} eligible cache node(s); tier(s) [${cov_ineligible}] have matching nodes but none currently eligible (Ready, non-cordoned, and compatible with the cache pod's hard-taint tolerations)${cov_empty:+. Tier(s) [${cov_empty}] have no matching cache nodes (scale-to-zero)}"
        fi
      elif [ "$cov_live" -eq 0 ]; then
        skip CSI_NODE_COVERAGE "no matching cache nodes for effective tier(s) [${cov_empty}] — scale-to-zero"
      elif [ -n "$cov_empty" ]; then
        adv CSI_NODE_COVERAGE "driver '${CSI_DRIVER}' is registered on ${cov_live} eligible cache node(s); tier(s) [${cov_empty}] have no matching cache nodes (scale-to-zero)"
      else
        ok CSI_NODE_COVERAGE "driver '${CSI_DRIVER}' is registered on all ${cov_live} eligible cache node(s) targeted by the effective tiers"
      fi
    fi
  else
    adv CSI_NODE_COVERAGE "needs CSI (the configured StorageClass/CSIDriver check did not pass)"
  fi

  # Pool controller RBAC — now NAMESPACED + chart-provided (the workload-manager Role grants
  # poddisruptionbudgets for cache-daemon protection and persistentvolumeclaims for the
  # cache-daemon + init-canary volumes); no cluster-scoped RBAC is required.
  adv POOL_RBAC "chart-provided (namespaced PDB + PVC RBAC on the workload-manager Role) — no cluster RBAC needed"
}

health(){
  echo "── HEALTH (tenant=${TENANT}) ──"
  local pods_ok=1
  # SUSTAINED readiness: a single snapshot can pass a crash-looping pod during a brief
  # "Running" blip between restarts. Sample twice ~20s apart and require, in the release
  # namespace: all Ready both times, no bad phase (Error/CrashLoopBackOff), and NO NEW restarts in
  # between. We use the restart DELTA (not absolute count) so the expected ~1-2 min
  # Temporal-namespace-registration flap doesn't fail once it has settled.
  _restarts(){ kubectl get pods -n "$1" --no-headers 2>/dev/null | awk '{s+=($4+0)} END{print s+0}'; }
  r0=$(_restarts "$NS")
  sleep 20
  if ! kubectl get ns "$NS" >/dev/null 2>&1; then
    bad PODS "namespace ${NS} not found (chart installed?)"; pods_ok=0
  else
    notready=$(kubectl get pods -n "$NS" --no-headers 2>/dev/null | awk '$3!="Completed" && $3!="Succeeded" {split($2,a,"/"); if(a[1]!=a[2]) print $1}')
    bad_pods=$(kubectl get pods -n "$NS" --no-headers 2>/dev/null | awk '$3!="Running" && $3!="Completed" && $3!="Succeeded" {print $1":"$3}')
    any=$(kubectl get pods -n "$NS" --no-headers 2>/dev/null)
    delta=$(( $(_restarts "$NS") - r0 ))
    if [ "$delta" -gt 0 ] 2>/dev/null; then
      bad PODS "${NS}: ${delta} new restart(s) over 20s — crash loop (not stably Ready): [${notready:-} ${bad_pods:-}]"; pods_ok=0
    elif [ -z "$notready" ] && [ -z "$bad_pods" ] && [ -n "$any" ]; then ok PODS "all Ready in ${NS} (stable — no new restarts over 20s)"
    else bad PODS "${NS}: not-ready=[${notready:-} ${bad_pods:-}]"; pods_ok=0; fi
  fi
  # REST/API/UI reachability is only meaningful once pods are Ready (depends_on PODS).
  if [ "$pods_ok" != 1 ]; then
    blk REST_HEALTH "needs PODS"; blk API "needs PODS"; blk UI "needs PODS"
    blk BACKEND_ISOLATION "needs PODS"
  else
    if kubectl get svc "$UI_SVC" -n "$NS" >/dev/null 2>&1; then
      if start_port_forward "$UI_SVC"; then
        code=$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' \
          "http://127.0.0.1:${PORT_FORWARD_PORT}/" 2>/dev/null || true)
        api_code=$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' \
          "http://127.0.0.1:${PORT_FORWARD_PORT}/api/health" 2>/dev/null || true)
        rest_code=$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' \
          "http://127.0.0.1:${PORT_FORWARD_PORT}/internal/health" 2>/dev/null || true)
        stop_port_forward
        [ "$code" = "200" ] &&
          ok UI "served (200)" ||
          bad UI "returned '${code:-no-response}'"
        [ "$rest_code" = "200" ] &&
          ok REST_HEALTH "/internal/health 200 through frontend (StateDB ready)" ||
          bad REST_HEALTH "/internal/health returned '${rest_code:-no-response}'"
        [ "$api_code" = "200" ] &&
          ok API "/api/health 200 through frontend" ||
          bad API "/api/health returned '${api_code:-no-response}'"
      else
        stop_port_forward
        bad UI "could not start port-forward to svc ${UI_SVC}"
        blk REST_HEALTH "needs UI"
        blk API "needs UI"
      fi
    else
      bad UI "svc ${UI_SVC} absent"
      blk REST_HEALTH "needs UI"
      blk API "needs UI"
    fi
    caller=$(kubectl get pod -n "$NS" -l app=mlworkflow \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    _worker_http_200(){
      kubectl exec -n "$NS" "$caller" -- python3 -c \
        'import sys, urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=5).status == 200 else 1)' \
        "$1" >/dev/null 2>&1
    }
    if [ -z "$caller" ]; then
      bad BACKEND_ISOLATION "no kumo-mlworkflow pod is available for the direct-backend denial probe"
    elif ! _worker_http_200 "http://${UI_SVC}/"; then
      bad BACKEND_ISOLATION "worker probe cannot reach the frontend baseline"
    elif ! _worker_http_200 "http://${REST_SVC}:9123/metrics"; then
      bad BACKEND_ISOLATION "worker probe cannot reach backend metrics port 9123"
    elif _worker_http_200 "http://${REST_SVC}/health" ||
      _worker_http_200 "http://${API_SVC}/health"; then
      adv BACKEND_ISOLATION "a non-frontend workload can reach backend port 8000; apply examples/networkpolicy.yaml to restrict it"
    else
      ok BACKEND_ISOLATION "frontend is reachable and a non-frontend workload is denied on backend port 8000"
    fi
  fi
}

case "$PHASE" in
  prereq) prereq ;;
  health) health ;;
  all)    prereq; health ;;
esac
echo "── ${FAILS} FAIL/BLOCK ──"
exit "$FAILS"
