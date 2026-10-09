#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Interactive setup wizard for the diskgraph pool+k8s devserver.
#
# Walks every prerequisite (AWS SSO, kubectl + EKS, INSTANCE_ID,
# pool worker image version, dev tenant provisioning) and resolves
# each in place — prompting the user where input is required.
#
# Idempotent: re-running is safe and just verifies the steps that
# have already been done.
#
# Provisioning bypasses the normal deployment automation (this is OPTION B —
# REST + controller run locally; only pool worker pods land in the
# cluster, so we don't need a full eph deployment). See
# scripts/dev/provision-pool-k8s-namespace.sh.
#
# When this script completes successfully, `make devserver-pool-k8s`
# is ready to run.

set -u
set -o pipefail

KUMO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EKS_CLUSTER='testing-uw4'
EKS_REGION='us-west-2'
AWS_ACCOUNT_ID='926922431314'
# Fallback profile only used if no other AWS credentials work; hosts with
# an IAM instance profile (e.g. devapp_dk) don't need it.
FALLBACK_AWS_PROFILE="PowerUserAccess-${AWS_ACCOUNT_ID}"
BASHRC="${HOME}/.bashrc"

# Canonical tenant passed by devserver/main.py (resolved from
# deploy_env() — including any .env file overrides — so the wizard
# provisions for the exact tenant the controller will look up).
WIZARD_TENANT="${1:-}"

# ── Output helpers ────────────────────────────────────────────────────────
ok()      { printf '  \033[32m✓\033[0m  %s\n' "$*"; }
fail()    { printf '  \033[31m✗\033[0m  %s\n' "$*"; }
note()    { printf '     %s\n' "$*"; }
heading() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
prompt_yn() {
    local q="$1"; local default="${2:-y}"
    local prompt
    if [[ "$default" == "y" ]]; then prompt="[Y/n]"; else prompt="[y/N]"; fi
    read -r -p "  $q $prompt " ans
    ans="${ans:-$default}"
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ── Step 1: AWS credentials ──────────────────────────────────────────────
# Accept whatever's already configured — IAM instance profile (devapp_dk),
# exported AWS_ACCESS_KEY_ID env, an existing AWS_PROFILE, or an SSO
# session under our fallback profile. Only prompt for SSO login if
# nothing works.
check_aws_credentials() {
    heading "AWS credentials"

    # 1. Whatever the shell already has — covers exported env vars or
    # an existing valid AWS_PROFILE.
    if aws sts get-caller-identity >/dev/null 2>&1; then
        local arn
        arn=$(aws sts get-caller-identity --query Arn --output text)
        ok "Logged in as $arn"
        return 0
    fi

    # 2. If AWS_PROFILE was exported but expired, the host may still
    # have an IAM instance profile (devapp / devapp_dk both do). Try
    # unsetting AWS_PROFILE so the SDK falls through to instance creds.
    if [[ -n "${AWS_PROFILE:-}" ]]; then
        if env -u AWS_PROFILE aws sts get-caller-identity >/dev/null 2>&1; then
            local arn
            arn=$(env -u AWS_PROFILE aws sts get-caller-identity \
                --query Arn --output text)
            ok "Logged in as $arn (via instance profile; was AWS_PROFILE=" \
               "$AWS_PROFILE)"
            unset AWS_PROFILE
            return 0
        fi
    fi

    # 3. Try the fallback SSO profile.
    if AWS_PROFILE="$FALLBACK_AWS_PROFILE" \
        aws sts get-caller-identity >/dev/null 2>&1; then
        local arn
        arn=$(AWS_PROFILE="$FALLBACK_AWS_PROFILE" \
            aws sts get-caller-identity --query Arn --output text)
        ok "Logged in as $arn (profile: $FALLBACK_AWS_PROFILE)"
        export AWS_PROFILE="$FALLBACK_AWS_PROFILE"
        return 0
    fi

    # 3. Nothing works — prompt to log in via SSO under the fallback profile.
    fail "No working AWS credentials in shell, env, or profile " \
         "'$FALLBACK_AWS_PROFILE'."
    note "Run: aws sso login --profile $FALLBACK_AWS_PROFILE"
    if prompt_yn "Run it now (opens device-code URL)?"; then
        aws sso login --profile "$FALLBACK_AWS_PROFILE" --no-browser
        if AWS_PROFILE="$FALLBACK_AWS_PROFILE" \
            aws sts get-caller-identity >/dev/null 2>&1; then
            ok "SSO login successful"
            export AWS_PROFILE="$FALLBACK_AWS_PROFILE"
            return 0
        fi
    fi
    fail "AWS credentials not configured. Re-run this script when ready."
    return 1
}

# ── Step 2: kubectl + EKS ────────────────────────────────────────────────
check_kubectl_eks() {
    heading "kubectl + EKS access"

    if ! command -v kubectl >/dev/null 2>&1; then
        fail "kubectl not installed."
        if prompt_yn "Install via 'sudo snap install kubectl --classic'?"; then
            sudo snap install kubectl --classic || return 1
        else
            return 1
        fi
    fi

    if ! kubectl config get-contexts "$EKS_CLUSTER" >/dev/null 2>&1; then
        note "Adding kubeconfig context for $EKS_CLUSTER ..."
        aws eks update-kubeconfig --region "$EKS_REGION" \
            --name "$EKS_CLUSTER" --alias "$EKS_CLUSTER" || return 1
    fi

    if kubectl --context "$EKS_CLUSTER" get ns --request-timeout=5s \
        >/dev/null 2>&1; then
        ok "kubectl can list namespaces on $EKS_CLUSTER"
        return 0
    fi

    # Auth failed — possible causes: stale kubeconfig user entry, expired
    # AWS creds (already handled in step 1), or the principal isn't
    # mapped in the cluster's access entries. Refresh the kubeconfig
    # entry once and re-check; if still failing, surface to the operator.
    note "kubectl auth failed against $EKS_CLUSTER; refreshing kubeconfig..."
    aws eks update-kubeconfig --region "$EKS_REGION" \
        --name "$EKS_CLUSTER" --alias "$EKS_CLUSTER" || true
    if kubectl --context "$EKS_CLUSTER" get ns --request-timeout=5s \
        >/dev/null 2>&1; then
        ok "kubectl can list namespaces on $EKS_CLUSTER"
        return 0
    fi
    fail "kubectl still cannot reach $EKS_CLUSTER."
    note "Check that your principal is granted EKS access. For an IAM"
    note "role like ec2-not-root, ask a cluster admin to run:"
    note "  aws eks create-access-entry --cluster-name $EKS_CLUSTER \\"
    note "    --principal-arn <role-arn>"
    note "  aws eks associate-access-policy --cluster-name $EKS_CLUSTER \\"
    note "    --principal-arn <role-arn> \\"
    note "    --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \\"
    note "    --access-scope type=cluster"
    return 1
}

# ── Step 3: Pool worker image version (~/kumo_version.txt) ───────────────
# daemon_image_uri() reads ~/kumo_version.txt for the tag used when
# spawning pool worker pods on the cluster. Those pods run the GPU image
# (pool/images.py resolves both daemon and activity images to
# kumo-backend-gpu), while the deployed services run the CPU one, so a
# usable tag must exist in both repositories.
_KUMO_BACKEND_REPOS=(kumo/kumo-backend-cpu kumo/kumo-backend-gpu)

# Sets _MISSING_REPOS to the repositories lacking the tag.
_verify_ecr_tag_exists() {
    local tag="$1" repo
    _MISSING_REPOS=()
    for repo in "${_KUMO_BACKEND_REPOS[@]}"; do
        aws ecr describe-images --region "$EKS_REGION" \
            --repository-name "$repo" \
            --image-ids imageTag="$tag" >/dev/null 2>&1 \
            || _MISSING_REPOS+=("$repo")
    done
    (( ${#_MISSING_REPOS[@]} == 0 ))
}

check_image_version() {
    heading "Pool worker image version (~/kumo_version.txt)"
    local version_file="${HOME}/kumo_version.txt"
    local current=""
    if [[ -f "$version_file" ]]; then
        current=$(tr -d '[:space:]' < "$version_file")
    fi
    local head_sha
    head_sha=$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse HEAD 2>/dev/null || true)

    local -a tags=() labels=()
    tags+=("master"); labels+=("master (latest nightly)")
    if [[ -n "$head_sha" ]]; then
        tags+=("$head_sha"); labels+=("${head_sha:0:12} (current HEAD)")
    fi
    if [[ -n "$current" && "$current" != "master" && "$current" != "$head_sha" ]]; then
        tags+=("$current"); labels+=("${current:0:40} (current ~/kumo_version.txt)")
    fi
    tags+=(""); labels+=("Enter custom tag")

    echo
    echo "  Available image tags:"
    local i
    for (( i=0; i<${#tags[@]}; i++ )); do
        printf '    %d) %s\n' "$((i+1))" "${labels[$i]}"
    done

    local choice
    read -r -p "  Choose [1]: " choice
    choice="${choice:-1}"
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#tags[@]} )); then
        fail "Invalid choice: '$choice'"
        return 1
    fi

    local pin="${tags[$((choice-1))]}"
    if [[ -z "$pin" ]]; then
        read -r -p "  Image tag: " pin
        [[ -n "$pin" ]] || { fail "No tag entered."; return 1; }
    fi
    if ! _verify_ecr_tag_exists "$pin"; then
        fail "Tag '$pin' missing from: ${_MISSING_REPOS[*]}"
        local registry='926922431314.dkr.ecr.us-west-2.amazonaws.com'
        local repo
        note "Build and push it with:"
        printf '\n     KUMO_BACKEND_VARIANT=gpu KUMO_IMAGE_TAG=%s make compose-local-build-backend\n' "$pin"
        printf '     aws ecr get-login-password --region us-west-2 \\\n'
        printf '       | docker login --username AWS --password-stdin %s\n' "$registry"
        for repo in "${_MISSING_REPOS[@]}"; do
            printf '     docker tag %s:%s %s/%s:%s\n' "$repo" "$pin" "$registry" "$repo" "$pin"
            printf '     docker push %s/%s:%s\n' "$registry" "$repo" "$pin"
        done
        printf '\n'
        return 1
    fi
    echo "$pin" > "$version_file"
    ok "Wrote '$pin' to $version_file"
}

# ── Step 4: Provision dev tenant (kubectl-apply) ─────────────────────────
# Tenant name is resolved by devserver/main.py from the same
# ``deploy_env()`` the controller will use, and passed in as ``$1``.
# We don't re-derive it here — keeping a parallel Python invocation in
# bash invariably drifts (different env, different .env loading) and
# produces a name that doesn't match what the controller looks up.
check_tenant() {
    heading "Dev tenant (namespace + SA + NodePools)"

    local tenant="${WIZARD_TENANT:-}"
    if [[ -z "$tenant" ]]; then
        fail "No tenant supplied to the wizard."
        note "devserver should resolve the canonical tenant from"
        note "deploy_env() and pass it as the first positional arg."
        return 1
    fi
    note "Canonical tenant: $tenant"
    local ns="kumo-dataplane-${tenant}"
    local exists=0
    if kubectl --context "$EKS_CLUSTER" get ns "$ns" >/dev/null 2>&1; then
        exists=1
    fi

    if (( exists )) && \
        kubectl --context "$EKS_CLUSTER" -n "$ns" get sa kumo-temporal \
            >/dev/null 2>&1 && \
        kubectl --context "$EKS_CLUSTER" get nodepool \
            "diskgraph-${tenant}" >/dev/null 2>&1 && \
        kubectl --context "$EKS_CLUSTER" get nodepool \
            "diskgraph-eph-${tenant}" >/dev/null 2>&1; then
        ok "Tenant fully provisioned: $ns + SA + NodePools"
        return 0
    fi

    if (( exists )); then
        fail "Namespace $ns exists but is missing SA / NodePools — re-applying."
    else
        fail "Namespace $ns + SA + NodePools not yet provisioned."
    fi

    # Pre-check the principal can create what the provisioning script
    # needs. Failing here surfaces an actionable error before
    # `kubectl apply` half-creates resources or fails with a generic
    # 403. ``auth can-i`` returns 0 for yes, 1 for no — capture so a
    # 'no' doesn't take down the script.
    local can_ns can_sa can_np
    can_ns=$(kubectl --context "$EKS_CLUSTER" auth can-i \
        create namespace 2>/dev/null || true)
    can_sa=$(kubectl --context "$EKS_CLUSTER" auth can-i \
        create serviceaccount --namespace "$ns" 2>/dev/null || true)
    can_np=$(kubectl --context "$EKS_CLUSTER" auth can-i \
        create nodepools.karpenter.sh 2>/dev/null || true)
    if [[ "$can_ns" != "yes" || "$can_sa" != "yes" || "$can_np" != "yes" ]]; then
        fail "Your principal can't create the required resources on $EKS_CLUSTER:"
        note "  create namespace: $can_ns"
        note "  create serviceaccount (in $ns): $can_sa"
        note "  create nodepools.karpenter.sh: $can_np"
        note "Ask a cluster admin to grant cluster-admin or pre-provision the tenant."
        return 1
    fi

    if ! prompt_yn "kubectl-apply the manifests now?"; then
        return 1
    fi
    if ! bash "$KUMO_REPO_ROOT/scripts/dev/provision-pool-k8s-namespace.sh" \
        "$tenant"; then
        fail "Provisioning script failed."
        return 1
    fi
    ok "Tenant provisioned"
}

# ── Main flow ────────────────────────────────────────────────────────────
main() {
    cd "$KUMO_REPO_ROOT" || exit 1

    printf '\033[1m=== devserver-pool-k8s setup ===\033[0m\n'

    local steps=(
        check_aws_credentials
        check_kubectl_eks
        check_image_version
        check_tenant
    )

    for step in "${steps[@]}"; do
        if ! "$step"; then
            printf '\n\033[31mSetup halted at %s. Re-run after addressing the issue above.\033[0m\n' "$step"
            exit 1
        fi
    done

    printf '\n\033[32m=== All checks passed. ===\033[0m\n'
}

main "$@"
