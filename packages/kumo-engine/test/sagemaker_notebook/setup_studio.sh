#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
studio_sm() { aws --region "$migration_region" sagemaker "$@"; }
studio_owned() {
  studio_sm list-tags --resource-arn "$1" --output json > "$migration_tmp/studio-tags.json"
  jq -e 'any(.Tags[]; .Key == "KumoTestSuite" and .Value == "kumo-engine-sagemaker-studio")' "$migration_tmp/studio-tags.json" >/dev/null || migration_fail 'Studio resource ownership differs.'
}
studio_prepare() {
  studio_sm list-domains --output json > "$migration_tmp/domains.json"
  jq '[.Domains[] | select(.DomainName == "kumo-test-sagemaker")]' "$migration_tmp/domains.json" > "$migration_tmp/domains-matched.json"
  [ "$(jq length "$migration_tmp/domains-matched.json")" -eq 1 ] || migration_fail 'Expected exactly one existing test domain; no domain will be created.'
  migration_domain=$(jq -r '.[0].DomainId' "$migration_tmp/domains-matched.json")
  printf '%s\n' "$migration_domain" | jq -Re 'test("^d-[a-z0-9]+$")' >/dev/null
  studio_sm describe-domain --domain-id "$migration_domain" --output json > "$migration_tmp/domain.json"
  migration_domain_arn="arn:aws:sagemaker:${migration_region}:${migration_account}:domain/${migration_domain}"
  jq -e --arg arn "$migration_domain_arn" --arg id "$migration_domain" '.DomainArn == $arn and .DomainId == $id and .DomainName == "kumo-test-sagemaker" and .Status == "InService" and .AuthMode == "IAM"' "$migration_tmp/domain.json" >/dev/null || migration_fail 'Existing test domain identity/status differs.'
  studio_owned "$migration_domain_arn"
  jq --arg domain "$migration_domain" --arg account "$migration_account" 'walk(if type == "string" then gsub("__KUMO_TEST_DOMAIN_ID__"; $domain) | gsub("__KUMO_TEST_ACCOUNT__"; $account) else . end)' "$handoff_tmp/access.json" > "$migration_tmp/access.json"
  for studio_kind in profile space; do
    jq --arg kind "$studio_kind" '.studio[$kind]' "$migration_tmp/access.json" > "$migration_tmp/$studio_kind-request.json"
  done
  jq --arg domain "$migration_domain" --arg account "$migration_account" 'walk(if type == "string" then gsub("__KUMO_TEST_DOMAIN_ID__"; $domain) | gsub("__KUMO_TEST_ACCOUNT__"; $account) else . end)' "$handoff_tmp/legacy.json" > "$migration_tmp/legacy.json"
}
studio_lookup() {
  studio_kind=$1
  if [ "$studio_kind" = profile ]; then
    studio_action=describe-user-profile; studio_flag=--user-profile-name
    studio_name=$(jq -r .UserProfileName "$migration_tmp/profile-request.json")
  else
    studio_action=describe-space; studio_flag=--space-name
    studio_name=$(jq -r .SpaceName "$migration_tmp/space-request.json")
  fi
  if studio_sm "$studio_action" --domain-id "$migration_domain" "$studio_flag" "$studio_name" --output json > "$migration_tmp/$studio_kind.json" 2> "$migration_tmp/studio-error"; then
    return 0
  fi
  case "$(cat "$migration_tmp/studio-error")" in
    *'(ResourceNotFound)'*) return 1 ;;
    *) cat "$migration_tmp/studio-error" >&2; migration_fail 'Studio lookup failed.' ;;
  esac
}
studio_validate() {
  studio_kind=$1
  if [ "$studio_kind" = profile ]; then
    studio_name=$(jq -r .UserProfileName "$migration_tmp/profile-request.json")
    studio_arn="arn:aws:sagemaker:${migration_region}:${migration_account}:user-profile/${migration_domain}/${studio_name}"
    jq -e --arg arn "$studio_arn" --slurpfile want "$migration_tmp/profile-request.json" '.UserProfileArn == $arn and .DomainId == $want[0].DomainId and .UserProfileName == $want[0].UserProfileName and (.UserSettings as $actual | all($want[0].UserSettings | paths(scalars); . as $path | ($actual | getpath($path)) == ($want[0].UserSettings | getpath($path))))' "$migration_tmp/profile.json" >/dev/null || migration_fail 'Existing validation profile differs.'
  else
    studio_name=$(jq -r .SpaceName "$migration_tmp/space-request.json")
    studio_arn="arn:aws:sagemaker:${migration_region}:${migration_account}:space/${migration_domain}/${studio_name}"
    jq -e --arg arn "$studio_arn" --slurpfile want "$migration_tmp/space-request.json" '.SpaceArn == $arn and .DomainId == $want[0].DomainId and .SpaceName == $want[0].SpaceName and .OwnershipSettings == $want[0].OwnershipSettings and .SpaceSharingSettings == $want[0].SpaceSharingSettings and (.SpaceSettings as $actual | all($want[0].SpaceSettings | paths(scalars); . as $path | ($actual | getpath($path)) == ($want[0].SpaceSettings | getpath($path))))' "$migration_tmp/space.json" >/dev/null || migration_fail 'Existing validation space differs.'
  fi
  studio_owned "$studio_arn"
  jq -e '.Status as $state | ["InService", "Creating", "Pending", "Updating"] | index($state) != null' "$migration_tmp/$studio_kind.json" >/dev/null || migration_fail 'Validation Studio resource is failed or deleting.'
}
studio_preflight() {
  for studio_item in profile space; do
    if studio_lookup "$studio_item"; then studio_validate "$studio_item"; fi
  done
}
studio_wait() {
  studio_wait_kind=$1
  studio_count=0
  while [ "$studio_count" -lt 120 ]; do
    if studio_lookup "$studio_wait_kind"; then
      studio_validate "$studio_wait_kind"
      [ "$(jq -r .Status "$migration_tmp/$studio_wait_kind.json")" != InService ] || return 0
    fi
    studio_count=$((studio_count + 1))
    sleep 5
  done
  migration_fail 'Studio provisioning did not finish; rerun this script to validate/reuse it.'
}
setup_studio() {
  for studio_item in profile space; do
    if studio_lookup "$studio_item"; then
      studio_validate "$studio_item"
    else
      if [ "$studio_item" = profile ]; then studio_action=create-user-profile; else studio_action=create-space; fi
      studio_sm "$studio_action" --cli-input-json "file://$migration_tmp/$studio_item-request.json" >/dev/null
    fi
    studio_wait "$studio_item"
  done
}
