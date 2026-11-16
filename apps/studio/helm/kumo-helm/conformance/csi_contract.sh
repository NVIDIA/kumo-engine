#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pure JSON helpers for read-only CSI conformance checks.

csi_validate_config() {
  local sc_json="$1" driver_json="$2" capacity_policy="${3:-required}"
  local mode provisioner reclaim_policy capacity_enabled
  mode=$(printf '%s' "$sc_json" | jq -r '.volumeBindingMode // empty')
  provisioner=$(printf '%s' "$sc_json" | jq -r '.provisioner // empty')
  reclaim_policy=$(printf '%s' "$sc_json" | jq -r '.reclaimPolicy // empty')

  if [ "$mode" != "WaitForFirstConsumer" ]; then
    printf 'volumeBindingMode is %s, want WaitForFirstConsumer' "${mode:-unset}"
    return 1
  fi
  if [ "$reclaim_policy" != "Delete" ]; then
    printf 'reclaimPolicy is %s, want Delete to release cache volumes' \
      "${reclaim_policy:-unset}"
    return 1
  fi
  if [ -z "$provisioner" ]; then
    printf 'StorageClass provisioner is unset'
    return 1
  fi
  if [ -z "$driver_json" ]; then
    printf 'CSIDriver %s not found' "$provisioner"
    return 1
  fi

  case "$capacity_policy" in
    required|advisory) ;;
    *)
      printf 'storage capacity tracking policy is %s, want required or advisory' \
        "${capacity_policy:-unset}"
      return 1
      ;;
  esac

  capacity_enabled=$(printf '%s' "$driver_json" |
    jq -r '.spec.storageCapacity // false')
  if [ "$capacity_policy" = required ] && [ "$capacity_enabled" != "true" ]; then
    printf 'CSIDriver %s spec.storageCapacity is not true' "$provisioner"
    return 1
  fi
}

csi_has_published_capacity() {
  local storage_class="$1" capacities_json="$2"
  printf '%s' "$capacities_json" | jq -e --arg sc "$storage_class" '
    any(
      .items[]?;
      .storageClassName == $sc
      and (.capacity // "") != ""
      and ((.capacity // "") | test("^0([EPTGMK]i?|m)?$") | not)
    )
  ' >/dev/null
}

# csi_node_coverage <driver> <toleration-key> <toleration-value> [extra-tolerations-json] reads
# {nodes,csinodes} list objects from stdin and emits
# {matched,eligible,uncovered}. Eligible means Ready, non-cordoned, and
# schedulable by a cache pod with its built-in and extra tolerations;
# PreferNoSchedule is soft and remains eligible. Registration comes from
# CSINode, the kubelet-published source of truth.
# CSIStorageCapacity is irrelevant: a registered driver may omit capacity
# objects, which are checked separately.
csi_node_coverage() {
  local driver="$1" toleration_key="$2" toleration_value="$3"
  local extra_tolerations="${4:-[]}"
  jq -ce \
    --arg driver "$driver" \
    --arg toleration_key "$toleration_key" \
    --arg toleration_value "$toleration_value" \
    --argjson extras "$extra_tolerations" '
    ([{key: $toleration_key, value: $toleration_value,
       operator: "Equal", effect: "NoSchedule"}] + ($extras // [])) as $tolerations
    |
    def cache_pod_tolerates($taint):
      ($taint.effect // "") as $effect
      | if $effect == "PreferNoSchedule" then true
        elif $effect == "NoSchedule" or $effect == "NoExecute" then
          any($tolerations[];
            ((.effect // "") == "" or .effect == $effect)
            and ((.key // "") == "" or .key == $taint.key)
            and (if .operator == "Exists" then true
                 else (.value // "") == ($taint.value // "") end))
        else false
        end;
    (.nodes.items | arrays) as $nodes
    | (.csinodes.items | arrays) as $csinodes
    | [$nodes[]
        | select((.spec.unschedulable // false) != true)
        | select(any(.status.conditions[]?;
            .type == "Ready" and .status == "True"))
        | select(all(.spec.taints[]?; cache_pod_tolerates(.)))] as $eligible
    | {matched: ($nodes | length), eligible: ($eligible | length), uncovered: [
        $eligible[] | .metadata.name as $node
        | select(any($csinodes[]?; .metadata.name == $node and
            any(.spec.drivers[]?; .name == $driver)) | not)
        | $node]}
  '
}
