#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
migration_fail() { echo "$*" >&2; exit 1; }
json_same() {
  jq -S . "$1" > "$migration_tmp/compare-a.json"
  jq -S . "$2" > "$migration_tmp/compare-b.json"
  cmp -s "$migration_tmp/compare-a.json" "$migration_tmp/compare-b.json"
}
role_lookup() {
  if aws iam get-role --role-name "$1" --output json > "$migration_tmp/role.json" 2> "$migration_tmp/iam-error"; then return 0; fi
  case "$(cat "$migration_tmp/iam-error")" in
    *'(NoSuchEntity)'*) return 1 ;;
    *) cat "$migration_tmp/iam-error" >&2; migration_fail 'Role lookup failed.' ;;
  esac
}
inline_check() {
  inline_present=false
  inline_upgrade=false
  if aws iam get-role-policy --role-name "$1" --policy-name "$2" --query PolicyDocument --output json > "$migration_tmp/inline.json" 2> "$migration_tmp/iam-error"; then
    if ! json_same "$migration_tmp/inline.json" "$3"; then
      shift 3
      inline_previous=false
      for inline_expected in "$@"; do
        if json_same "$migration_tmp/inline.json" "$inline_expected"; then inline_previous=true; break; fi
      done
      [ "$inline_previous" = true ] || migration_fail 'Existing inline policy differs from the current or approved previous policy.'
      inline_upgrade=true
    fi
    inline_present=true
  else
    case "$(cat "$migration_tmp/iam-error")" in
      *'(NoSuchEntity)'*) ;;
      *) cat "$migration_tmp/iam-error" >&2; migration_fail 'Inline policy lookup failed.' ;;
    esac
  fi
}
previous_space_policy() {
  new_space_name=$(jq -r .studio.space.SpaceName "$migration_tmp/access.json")
  studio_base="arn:aws:sagemaker:${migration_region}:${migration_account}"
  studio_log="arn:aws:logs:${migration_region}:${migration_account}:log-group:/aws/sagemaker/studio:log-stream:${migration_domain}"
  # Recognize only the policy from the original profile/space name collision.
  jq --arg new_space "$studio_base:space/$migration_domain/$new_space_name" \
     --arg old_space "$studio_base:space/$migration_domain/kumo-test-validation" \
     --arg new_app "$studio_base:app/$migration_domain/$new_space_name/JupyterLab/default" \
     --arg old_app "$studio_base:app/$migration_domain/kumo-test-validation/JupyterLab/default" \
     --arg new_log "$studio_log/$new_space_name/JupyterLab/default*" \
     --arg old_log "$studio_log/kumo-test-validation/JupyterLab/default*" '
    walk(if type == "string" then
      if . == $new_space then $old_space
      elif . == $new_app then $old_app
      elif . == $new_log then $old_log
      else . end
    else . end)
  ' "$1" > "$2"
}
new_role_check() {
  new_key=$1
  new_name=$(jq -r --arg key "$new_key" '.roles[$key].arn | split("/")[-1]' "$migration_tmp/access.json")
  new_duration=3600; [ "$new_key" != RUNNER ] || new_duration=10800
  role_present=false
  if role_lookup "$new_name"; then
    role_present=true
    jq -e --arg name "$new_name" --arg account "$migration_account" --argjson duration "$new_duration" '.Role.Arn == ("arn:aws:iam::" + $account + ":role/" + $name) and .Role.Path == "/" and .Role.MaxSessionDuration == $duration and (.Role.PermissionsBoundary == null) and any(.Role.Tags[]?; .Key == "KumoTestSuite" and .Value == "kumo-engine-sagemaker-validation")' "$migration_tmp/role.json" >/dev/null || migration_fail 'Dedicated role identity, ownership or session settings differ.'
    jq '.Role.AssumeRolePolicyDocument' "$migration_tmp/role.json" > "$migration_tmp/current-trust.json"
    json_same "$migration_tmp/current-trust.json" "$migration_tmp/$new_key-trust.json" || migration_fail 'Dedicated role trust differs.'
    aws iam list-attached-role-policies --role-name "$new_name" --output json > "$migration_tmp/attached.json"
    jq -e '.AttachedPolicies == []' "$migration_tmp/attached.json" >/dev/null || migration_fail 'Dedicated role has unexpected managed policies.'
    aws iam list-role-policies --role-name "$new_name" --output json > "$migration_tmp/inline-names.json"
    jq -e 'all(.PolicyNames[]; . == "kumo-engine-sagemaker-validation")' "$migration_tmp/inline-names.json" >/dev/null || migration_fail 'Dedicated role has unexpected inline policies.'
    inline_check "$new_name" kumo-engine-sagemaker-validation "$migration_tmp/$new_key-policy.json" "$migration_tmp/$new_key-previous-policy.json" "$migration_tmp/$new_key-validation-previous.json" "$migration_tmp/$new_key-validation-space-previous.json" "$migration_tmp/$new_key-model-previous.json"
  fi
}
legacy_inline_check() {
  if role_lookup "$1"; then inline_check "$1" "$2" "$3"; else inline_present=false; fi
}
legacy_trust_check() {
  legacy_trust_present=false
  if ! role_lookup gha-kumo-workflow-tests; then return; fi
  jq '.Role.AssumeRolePolicyDocument | .Statement |= (if type == "array" then . else [.] end)' "$migration_tmp/role.json" > "$migration_tmp/legacy-trust-current.json"
  jq --slurpfile want "$migration_tmp/legacy.json" '[.Statement[] | select(. == $want[0].runner_trust)] | length' "$migration_tmp/legacy-trust-current.json" > "$migration_tmp/trust-count.json"
  legacy_count=$(cat "$migration_tmp/trust-count.json")
  [ "$legacy_count" -le 1 ] || migration_fail 'Legacy trust addition is duplicated.'
  if [ "$legacy_count" -eq 1 ]; then
    legacy_trust_present=true
    jq --slurpfile want "$migration_tmp/legacy.json" '.Statement |= map(select(. != $want[0].runner_trust))' "$migration_tmp/legacy-trust-current.json" > "$migration_tmp/legacy-trust-after.json"
    jq -e '.Statement | length > 0' "$migration_tmp/legacy-trust-after.json" >/dev/null || migration_fail 'Refusing to leave the shared role without any trust statements.'
  else
    jq -e '[.Statement[] | .. | strings | select(contains("sagemaker-mlops-test"))] == []' "$migration_tmp/legacy-trust-current.json" >/dev/null || migration_fail 'Legacy environment trust differs from the known addition.'
  fi
}
legacy_studio_check() {
  legacy_studio_present=false
  if ! role_lookup gha-kumo-workflow-tests; then return; fi
  aws iam list-attached-role-policies --role-name gha-kumo-workflow-tests --output json > "$migration_tmp/legacy-attached.json"
  legacy_policy_arn="arn:aws:iam::${migration_account}:policy/kumo-engine-sagemaker-studio"
  if jq -e --arg arn "$legacy_policy_arn" 'any(.AttachedPolicies[]; .PolicyArn == $arn)' "$migration_tmp/legacy-attached.json" >/dev/null; then
    aws iam get-policy --policy-arn "$legacy_policy_arn" --output json > "$migration_tmp/legacy-policy.json"
    jq -e --arg arn "$legacy_policy_arn" '.Policy.Arn == $arn' "$migration_tmp/legacy-policy.json" >/dev/null || migration_fail 'Legacy managed policy identity differs.'
    legacy_version=$(jq -r .Policy.DefaultVersionId "$migration_tmp/legacy-policy.json")
    aws iam get-policy-version --policy-arn "$legacy_policy_arn" --version-id "$legacy_version" --query PolicyVersion.Document --output json > "$migration_tmp/legacy-policy-document.json"
    json_same "$migration_tmp/legacy-policy-document.json" "$migration_tmp/legacy-studio-policy.json" || migration_fail 'Legacy Studio managed policy differs.'
    aws iam list-policy-tags --policy-arn "$legacy_policy_arn" --output json > "$migration_tmp/legacy-policy-tags.json"
    jq -e 'any(.Tags[]; .Key == "KumoTestSuite" and .Value == "kumo-engine-sagemaker-studio")' "$migration_tmp/legacy-policy-tags.json" >/dev/null || migration_fail 'Legacy managed policy lacks suite ownership.'
    legacy_studio_present=true
  fi
}
legacy_preflight() {
  legacy_inline_check gha-kumo-workflow-tests kumo-engine-sagemaker-notebook "$migration_tmp/legacy-runner-policy.json"
  legacy_inline_check gha-kumo-workflow-tests kumo-engine-sagemaker-ui-read "$migration_tmp/legacy-ui-policy.json"
  legacy_inline_check kumo-mlops-validation-sagemaker kumo-engine-sagemaker-notebook "$migration_tmp/legacy-execution-policy.json"
  legacy_inline_check kumo-mlops-validation-sagemaker pipeline-steps "$migration_tmp/legacy-steps-policy.json"
  legacy_inline_check kumo-mlops-validation-sagemaker kumo-engine-sagemaker-studio "$migration_tmp/legacy-execution_studio-policy.json"
  legacy_studio_check
  legacy_trust_check
}
legacy_retire() {
  for legacy_kind in runner ui execution steps execution_studio; do
    case "$legacy_kind" in
      runner) legacy_role=gha-kumo-workflow-tests; legacy_name=kumo-engine-sagemaker-notebook ;;
      ui) legacy_role=gha-kumo-workflow-tests; legacy_name=kumo-engine-sagemaker-ui-read ;;
      execution) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=kumo-engine-sagemaker-notebook ;;
      execution_studio) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=kumo-engine-sagemaker-studio ;;
      steps) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=pipeline-steps ;;
    esac
    legacy_inline_check "$legacy_role" "$legacy_name" "$migration_tmp/legacy-$legacy_kind-policy.json"
    if [ "$inline_present" = true ]; then aws iam delete-role-policy --role-name "$legacy_role" --policy-name "$legacy_name"; fi
  done
  legacy_studio_check
  if [ "$legacy_studio_present" = true ]; then aws iam detach-role-policy --role-name gha-kumo-workflow-tests --policy-arn "$legacy_policy_arn"; fi
  legacy_trust_check
  if [ "$legacy_trust_present" = true ]; then
    cp "$migration_tmp/legacy-trust-current.json" "$migration_tmp/legacy-trust-before-write.json"
    legacy_trust_check
    json_same "$migration_tmp/legacy-trust-before-write.json" "$migration_tmp/legacy-trust-current.json" || migration_fail 'Shared role trust changed during migration.'
    aws iam update-assume-role-policy --role-name gha-kumo-workflow-tests --policy-document "file://$migration_tmp/legacy-trust-after.json"
  fi
}
apply_access() (
  set -eu
  migration_tmp=$(mktemp -d)
  trap 'rm -rf "$migration_tmp"' EXIT
  trap 'exit 1' HUP INT TERM
  migration_account=$(aws sts get-caller-identity --query Account --output text)
  [ "$migration_account" = 926922431314 ] || migration_fail 'Refusing migration outside the dedicated test account.'
  migration_region=us-west-2
  aws iam get-open-id-connect-provider --open-id-connect-provider-arn "arn:aws:iam::${migration_account}:oidc-provider/token.actions.githubusercontent.com" --output json > "$migration_tmp/oidc.json"
  jq -e '(.Url == "token.actions.githubusercontent.com" or .Url == "https://token.actions.githubusercontent.com") and (.ClientIDList | index("sts.amazonaws.com") != null)' "$migration_tmp/oidc.json" >/dev/null || migration_fail 'GitHub OIDC provider URL or audience differs.'
  studio_prepare
  for migration_key in RUNNER PIPELINE EXECUTION; do
    jq --arg key "$migration_key" '.roles[$key].trust' "$migration_tmp/access.json" > "$migration_tmp/$migration_key-trust.json"
    jq --arg key "$migration_key" '.roles[$key].permissions + {Statement: (.roles[$key].permissions.Statement + (if $key == "RUNNER" then .studio.runner_permissions.Statement elif $key == "EXECUTION" then .studio.execution_permissions.Statement else [] end))}' "$migration_tmp/access.json" > "$migration_tmp/$migration_key-policy.json"
    previous_space_policy "$migration_tmp/$migration_key-policy.json" "$migration_tmp/$migration_key-previous-policy.json"
    jq --arg key "$migration_key" '.dedicated_previous_policies[$key]' "$migration_tmp/legacy.json" > "$migration_tmp/$migration_key-validation-previous.json"
    previous_space_policy "$migration_tmp/$migration_key-validation-previous.json" "$migration_tmp/$migration_key-validation-space-previous.json"
    if [ "$migration_key" = EXECUTION ]; then
      jq '.dedicated_logged_model_previous_policy' "$migration_tmp/legacy.json" > "$migration_tmp/$migration_key-model-previous.json"
    else
      cp "$migration_tmp/$migration_key-policy.json" "$migration_tmp/$migration_key-model-previous.json"
    fi
  done
  for legacy_kind in runner ui execution steps execution_studio studio; do jq --arg key "${legacy_kind}_policy" '.[$key]' "$migration_tmp/legacy.json" > "$migration_tmp/legacy-$legacy_kind-policy.json"; done
  for migration_key in RUNNER PIPELINE EXECUTION; do new_role_check "$migration_key"; done
  studio_preflight
  legacy_preflight
  # All known state has passed read-only admission before the first mutation.
  for migration_key in RUNNER PIPELINE EXECUTION; do
    new_role_check "$migration_key"
    if [ "$role_present" = false ]; then
      aws iam create-role --role-name "$new_name" --assume-role-policy-document "file://$migration_tmp/$migration_key-trust.json" --max-session-duration "$new_duration" --tags Key=KumoTestSuite,Value=kumo-engine-sagemaker-validation >/dev/null
    fi
    new_role_check "$migration_key"
    inline_check "$new_name" kumo-engine-sagemaker-validation "$migration_tmp/$migration_key-policy.json" "$migration_tmp/$migration_key-previous-policy.json" "$migration_tmp/$migration_key-validation-previous.json" "$migration_tmp/$migration_key-validation-space-previous.json" "$migration_tmp/$migration_key-model-previous.json"
    if [ "$inline_present" = false ] || [ "$inline_upgrade" = true ]; then aws iam put-role-policy --role-name "$new_name" --policy-name kumo-engine-sagemaker-validation --policy-document "file://$migration_tmp/$migration_key-policy.json"; fi
  done
  setup_studio
  legacy_preflight
  legacy_retire
  for migration_key in RUNNER PIPELINE EXECUTION; do
    new_role_check "$migration_key"
    [ "$role_present" = true ] && [ "$inline_present" = true ] && [ "$inline_upgrade" = false ] || migration_fail 'Dedicated role admission changed after provisioning; rerun after IAM propagation.'
  done
  for legacy_kind in runner ui execution steps execution_studio; do
    case "$legacy_kind" in
      runner) legacy_role=gha-kumo-workflow-tests; legacy_name=kumo-engine-sagemaker-notebook ;;
      ui) legacy_role=gha-kumo-workflow-tests; legacy_name=kumo-engine-sagemaker-ui-read ;;
      execution) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=kumo-engine-sagemaker-notebook ;;
      execution_studio) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=kumo-engine-sagemaker-studio ;;
      steps) legacy_role=kumo-mlops-validation-sagemaker; legacy_name=pipeline-steps ;;
    esac
    legacy_inline_check "$legacy_role" "$legacy_name" "$migration_tmp/legacy-$legacy_kind-policy.json"
    [ "$inline_present" = false ] || migration_fail 'Legacy inline policy is still present; rerun after IAM propagation.'
  done
  legacy_studio_check
  [ "$legacy_studio_present" = false ] || migration_fail 'Legacy Studio attachment is still present.'
  legacy_trust_check
  [ "$legacy_trust_present" = false ] || migration_fail 'Legacy environment trust is still present.'
  printf 'Dedicated roles and private Studio resources verified. Set SAGEMAKER_TEST_DOMAIN_ID=%s in the protected environment.\n' "$migration_domain"
)
