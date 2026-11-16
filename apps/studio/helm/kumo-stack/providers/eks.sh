#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

readonly EKS_KUBERNETES_VERSION=1.34
readonly EKS_KARPENTER_CHART_VERSION=1.12.0
readonly EKS_AWS_LOAD_BALANCER_CONTROLLER_CHART_VERSION=1.14.0
readonly EKS_VPC_CNI_ADDON_VERSION=v1.22.3-eksbuild.1
readonly EKS_COREDNS_ADDON_VERSION=v1.12.4-eksbuild.18
readonly EKS_KUBE_PROXY_ADDON_VERSION=v1.34.6-eksbuild.17
readonly EKS_EBS_CSI_ADDON_VERSION=v1.63.0-eksbuild.1

verify_aws_role() {
  local arn="$1" label="$2" role_name
  role_name="${arn##*/}"
  aws iam get-role --role-name "${role_name}" >/dev/null ||
    die "${label} IAM role ${arn} is not accessible"
}

# Validate the common IRSA trust-policy shape for the Temporal bootstrap
# ServiceAccount. This is a partial IAM evaluator: it recognizes StringEquals
# and StringLike subject/audience conditions and ignores Deny statements.
# Policies using other constructs may be rejected or fail later in the Job.
verify_eks_postgres_admin_identity() {
  local cluster_json="$1"
  local issuer subject role_name role condition pattern trusted=false
  issuer=$("${JQ}" -r '.identity.oidc.issuer // empty' <<<"${cluster_json}")
  issuer="${issuer#https://}"
  [[ -n "${issuer}" ]] ||
    die "EKS cluster ${PROVIDER_CLUSTER} publishes no OIDC issuer for workload identity"
  subject=$(temporal_database_identity_subject)

  [[ "${POSTGRES_ADMIN_ANNOTATION_VALUE}" =~ ^arn:aws[a-z-]*:iam::[0-9]{12}:role/.+$ ]] ||
    die "cloud.postgres.admin.annotations[${POSTGRES_ADMIN_ANNOTATION_KEY}] must be an IAM role ARN, not ${POSTGRES_ADMIN_ANNOTATION_VALUE}"
  # A role's path is not part of the name IAM looks it up by, so the lookup can
  # answer for a different role than the ARN names.
  role_name="${POSTGRES_ADMIN_ANNOTATION_VALUE##*/}"
  role=$(aws iam get-role --role-name "${role_name}" --query Role --output json) ||
    die "PostgreSQL admin IAM role ${POSTGRES_ADMIN_ANNOTATION_VALUE} is not accessible"
  "${JQ}" -e --arg arn "${POSTGRES_ADMIN_ANNOTATION_VALUE}" \
    '.Arn == $arn' <<<"${role}" >/dev/null ||
    die "no IAM role ${POSTGRES_ADMIN_ANNOTATION_VALUE} exists; ${role_name} is $("${JQ}" -r '.Arn' <<<"${role}")"

  # Require the cluster OIDC provider, compatible audience, and matching
  # ServiceAccount subject. Subject globs are evaluated in the shell.
  while IFS=$'\t' read -r condition pattern; do
    if [[ "${condition}" == StringLike ]]; then
      # Escape bash pattern syntax that IAM StringLike does not support.
      pattern=${pattern//\\/\\\\}
      pattern=${pattern//\[/\\[}
      # shellcheck disable=SC2053
      if [[ "${subject}" == ${pattern} ]]; then
        trusted=true
      fi
    elif [[ "${subject}" == "${pattern}" ]]; then
      trusted=true
    fi
    [[ "${trusted}" != true ]] || break
  done < <("${JQ}" -r \
    --arg issuer "${issuer,,}" \
    --arg sub_key "${issuer,,}:sub" \
    --arg aud_key "${issuer,,}:aud" '
    def condition_values($operator; $key):
      ((.Condition[$operator]? // {}) | to_entries[] |
        select((.key | ascii_downcase) == $key) | .value |
        if type == "array" then .[] else . end);
    def federated_issuer:
      if type == "string"
        then (ascii_downcase | split("oidc-provider/"))
        else [] end |
      if length == 2 then .[1] else empty end;

    .AssumeRolePolicyDocument.Statement[]? |
    select(.Effect == "Allow") |
    select([.Action] | flatten | map(ascii_downcase) |
      index("sts:assumerolewithwebidentity")) |
    select([.Principal.Federated?] | flatten |
      any(federated_issuer == $issuer)) |
    select([condition_values("StringEquals"; $aud_key),
            condition_values("StringLike"; $aud_key)] as $audiences |
      ($audiences | length) == 0 or
      ($audiences | any(. == "sts.amazonaws.com" or . == "*"))) |
    (
      (condition_values("StringEquals"; $sub_key) | ["StringEquals", .]),
      (condition_values("StringLike"; $sub_key) | ["StringLike", .])
    ) |
    @tsv
  ' <<<"${role}")
  [[ "${trusted}" == true ]] ||
    die "IAM role ${POSTGRES_ADMIN_ANNOTATION_VALUE} has no sts:AssumeRoleWithWebIdentity statement for ${issuer}, subject ${subject}, audience sts.amazonaws.com"
}

eks_addon_is_active() {
  aws eks describe-addon \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --region "${PROVIDER_REGION}" \
    --addon-name "$1" \
    --query 'addon.status' --output text 2>/dev/null |
    grep -Fxq ACTIVE
}

verify_eks_addon_version() {
  local name="$1" version="$2" addon_json
  addon_json=$(aws eks describe-addon \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --region "${PROVIDER_REGION}" \
    --addon-name "${name}" \
    --query addon --output json 2>/dev/null) ||
    die "required EKS managed add-on ${name}:${version} is not installed"
  "${JQ}" -e --arg version "${version}" '
    .status == "ACTIVE" and .addonVersion == $version
  ' <<<"${addon_json}" >/dev/null ||
    die "EKS managed add-on ${name} must be ACTIVE at ${version}"
  printf '%s' "${addon_json}"
}

ensure_eks_addon() {
  local name="$1" role_arn="$2" version="$3" addon_json
  if addon_json=$(aws eks describe-addon \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --region "${PROVIDER_REGION}" \
    --addon-name "${name}" \
    --query addon --output json 2>/dev/null); then
    if "${JQ}" -e --arg role "${role_arn}" --arg version "${version}" '
      .status == "ACTIVE" and
      .serviceAccountRoleArn == $role and
      .addonVersion == $version
    ' <<<"${addon_json}" >/dev/null; then
      return
    fi
    aws eks update-addon \
      --cluster-name "${PROVIDER_CLUSTER}" \
      --region "${PROVIDER_REGION}" \
      --addon-name "${name}" \
      --addon-version "${version}" \
      --service-account-role-arn "${role_arn}" \
      --resolve-conflicts PRESERVE >/dev/null
  else
    aws eks create-addon \
      --cluster-name "${PROVIDER_CLUSTER}" \
      --region "${PROVIDER_REGION}" \
      --addon-name "${name}" \
      --addon-version "${version}" \
      --service-account-role-arn "${role_arn}" \
      --resolve-conflicts OVERWRITE >/dev/null
  fi
  aws eks wait addon-active \
    --cluster-name "${PROVIDER_CLUSTER}" \
    --region "${PROVIDER_REGION}" \
    --addon-name "${name}"
}

install_karpenter() {
  helm_upgrade karpenter-crd \
    oci://public.ecr.aws/karpenter/karpenter-crd \
    --namespace kube-system \
    --version "${EKS_KARPENTER_CHART_VERSION}" \
    --atomic --wait --timeout "${TIMEOUT}"
  helm_upgrade karpenter \
    oci://public.ecr.aws/karpenter/karpenter \
    --namespace kube-system \
    --version "${EKS_KARPENTER_CHART_VERSION}" \
    --atomic --wait --timeout "${TIMEOUT}" \
    --set "controller.image.repository=${KARPENTER_IMAGE_REPOSITORY}" \
    --set "controller.image.tag=${KARPENTER_IMAGE_TAG}" \
    --set "controller.image.digest=${KARPENTER_IMAGE_DIGEST}" \
    --set "settings.clusterName=${PROVIDER_CLUSTER}" \
    --set "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn=${EKS_KARPENTER_CONTROLLER_ROLE_ARN}" \
    --set controller.resources.requests.cpu=250m \
    --set controller.resources.requests.memory=256Mi
}

install_aws_load_balancer_controller() {
  local chart
  ensure_helm_repo eks https://aws.github.io/eks-charts
  chart=eks/aws-load-balancer-controller

  # The upstream chart installs CRDs on a first install but Helm does not
  # update CRDs during an upgrade. Applying the chart's own stock CRDs keeps
  # retries and upgrades on the same supported artifact.
  "${HELM}" show crds "${chart}" \
    --version "${EKS_AWS_LOAD_BALANCER_CONTROLLER_CHART_VERSION}" |
    kubectl_cmd apply --server-side --force-conflicts -f - >/dev/null
  helm_upgrade aws-load-balancer-controller "${chart}" \
    --namespace kube-system \
    --version "${EKS_AWS_LOAD_BALANCER_CONTROLLER_CHART_VERSION}" \
    --atomic --wait --timeout "${TIMEOUT}" \
    --set "clusterName=${PROVIDER_CLUSTER}" \
    --set "region=${PROVIDER_REGION}" \
    --set "vpcId=${EKS_VPC_ID}" \
    --set "image.repository=${AWS_LOAD_BALANCER_CONTROLLER_IMAGE_REPOSITORY}" \
    --set "image.tag=${AWS_LOAD_BALANCER_CONTROLLER_IMAGE_TAG}" \
    --set serviceAccount.create=true \
    --set serviceAccount.name=aws-load-balancer-controller \
    --set "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn=${EKS_LOAD_BALANCER_CONTROLLER_ROLE_ARN}"
}

provider_preflight() {
  require_command aws
  local cluster_json addon name version vpc_cni_json
  cluster_json=$(aws --cli-connect-timeout 10 --cli-read-timeout 30 \
    eks describe-cluster \
    --name "${PROVIDER_CLUSTER}" \
    --region "${PROVIDER_REGION}" \
    --query cluster --output json)
  "${JQ}" -e --arg vpc "${EKS_VPC_ID}" \
    --arg version "${EKS_KUBERNETES_VERSION}" '
    .status == "ACTIVE" and
    .resourcesVpcConfig.vpcId == $vpc and
    .version == $version
  ' <<<"${cluster_json}" >/dev/null ||
    die "EKS cluster must be ACTIVE, Kubernetes ${EKS_KUBERNETES_VERSION}, and in the configured VPC"

  for addon in \
    "vpc-cni:${EKS_VPC_CNI_ADDON_VERSION}" \
    "coredns:${EKS_COREDNS_ADDON_VERSION}" \
    "kube-proxy:${EKS_KUBE_PROXY_ADDON_VERSION}"; do
    name="${addon%%:*}"
    version="${addon#*:}"
    if [[ "${name}" == vpc-cni ]]; then
      vpc_cni_json=$(verify_eks_addon_version "${name}" "${version}")
    else
      verify_eks_addon_version "${name}" "${version}" >/dev/null
    fi
  done
  "${JQ}" -e '
    try (.configurationValues | fromjson | .enableNetworkPolicy | tostring)
    catch "false" | . == "true"
  ' <<<"${vpc_cni_json}" >/dev/null ||
    die "EKS VPC CNI add-on must set enableNetworkPolicy=true"

  aws ec2 describe-subnets --region "${PROVIDER_REGION}" \
    --filters \
      "Name=vpc-id,Values=${EKS_VPC_ID}" \
      "Name=tag:karpenter.sh/discovery,Values=${PROVIDER_CLUSTER}" \
    --query 'length(Subnets)' --output text |
    grep -Eq '^[1-9][0-9]*$' ||
    die "EKS VPC has no subnet tagged karpenter.sh/discovery=${PROVIDER_CLUSTER}"
  aws ec2 describe-security-groups --region "${PROVIDER_REGION}" \
    --filters \
      "Name=vpc-id,Values=${EKS_VPC_ID}" \
      "Name=tag:karpenter.sh/discovery,Values=${PROVIDER_CLUSTER}" \
    --query 'length(SecurityGroups)' --output text |
    grep -Eq '^[1-9][0-9]*$' ||
    die "EKS VPC has no security group tagged karpenter.sh/discovery=${PROVIDER_CLUSTER}"
  aws iam get-role --role-name "${EKS_KARPENTER_NODE_ROLE}" >/dev/null ||
    die "Karpenter node role ${EKS_KARPENTER_NODE_ROLE} is not accessible"
  verify_aws_role "${EKS_KARPENTER_CONTROLLER_ROLE_ARN}" Karpenter
  verify_aws_role "${EKS_EBS_CSI_ROLE_ARN}" "EBS CSI"
  verify_aws_role "${EKS_LOAD_BALANCER_CONTROLLER_ROLE_ARN}" \
    "AWS Load Balancer Controller"
  if temporal_database_bootstrap_enabled; then
    verify_eks_postgres_admin_identity "${cluster_json}"
  fi
  aws acm describe-certificate \
    --region "${PROVIDER_REGION}" \
    --certificate-arn "${EKS_EDGE_CERTIFICATE_ARN}" \
    --query 'Certificate.Status' --output text |
    grep -Fxq ISSUED ||
    die "EKS edge ACM certificate is not ISSUED"

  local edge_subnet_tag edge_subnets
  if [[ "${EKS_EDGE_SCHEME}" == internet-facing ]]; then
    edge_subnet_tag=kubernetes.io/role/elb
  else
    edge_subnet_tag=kubernetes.io/role/internal-elb
  fi
  edge_subnets=$(aws ec2 describe-subnets \
    --region "${PROVIDER_REGION}" \
    --filters \
      "Name=vpc-id,Values=${EKS_VPC_ID}" \
      "Name=tag:${edge_subnet_tag},Values=1" \
    --query 'Subnets[].AvailabilityZone' --output json)
  "${JQ}" -e 'unique | length >= 2' <<<"${edge_subnets}" >/dev/null ||
    die "EKS ALB requires ${edge_subnet_tag}=1 subnets in at least two Availability Zones"

  verify_storage_class_ownership kumo-platform-eks
}

provider_prepare_cloud() {
  ensure_eks_addon aws-ebs-csi-driver "${EKS_EBS_CSI_ROLE_ARN}" \
    "${EKS_EBS_CSI_ADDON_VERSION}"
  install_karpenter
  install_aws_load_balancer_controller
}

provider_verify_platform() {
  local pool tier tier_name instance_type ssd_gib gpu_count
  local -a tiers
  eks_addon_is_active aws-ebs-csi-driver ||
    die "EKS managed EBS CSI add-on is not ACTIVE"
  wait_for_deployment kube-system karpenter
  wait_for_deployment kube-system aws-load-balancer-controller

  kubectl_cmd get nodepool kumo-cpu >/dev/null
  IFS=';' read -r -a tiers <<<"${GPU_CACHE_TIERS}"
  for tier in "${tiers[@]}"; do
    IFS=',' read -r tier_name instance_type ssd_gib gpu_count <<<"${tier}"
    pool="kumo-cache-${tier_name}"
    kubectl_cmd get nodepool "${pool}" >/dev/null
  done
  kubectl_cmd get nodepool kumo-ephemeral >/dev/null
  verify_no_gpu_nodes
}

provider_decommission() {
  # The leaf release owns capacity. Cluster-wide add-ons and the ingress
  # remain because they can be shared by another Kumo tenant.
  :
}
