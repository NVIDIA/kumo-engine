#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
: "${REPOSITORY:?}" "${SOURCE_SHA:?}" "${LABEL:?}" "${SCRATCH_GIB:?}" "${INSTANCE_TYPE:?}" "${TIER:?}" "${CAPACITY_FUNCTION:?}" "${AWS_REGION:?}" "${CAPACITY_DEADLINE:?}" "${POOL_RUN_ID:?}" "${POOL_ATTEMPT:?}" "${POOL_CASES:?}"
bootstrapping=1 boot_check='runner image'
invoke() {
  local payload function_error
  payload=$(python3 - "$REPOSITORY" "$POOL_RUN_ID" "$POOL_ATTEMPT" "$SOURCE_SHA" "$CAPACITY_DEADLINE" "$INSTANCE_TYPE" "$LABEL" "$POOL_CASES" "$1" "${2:-}" <<'PY'
import json, sys
repo, run, attempt, sha, deadline, instance, label, cases, kind, error = sys.argv[1:]
payload = dict(kind=kind, repo=repo, run=run, attempt=attempt, sha=sha,
    deadline=int(deadline), instance=instance, label=label, cases=json.loads(cases))
if error:
    payload['error'] = error
print(json.dumps(payload))
PY
) || return
  if [[ -z ${token_file:-} ]]; then token_file=$(mktemp /run/engine-e2e-token.XXXXXX) || return; fi
  function_error=$(aws lambda invoke --function-name "$CAPACITY_FUNCTION" --region "$AWS_REGION" --cli-connect-timeout 10 --cli-read-timeout 30 --cli-binary-format raw-in-base64-out --payload "$payload" --query FunctionError --output text "$token_file") || return
  [[ $function_error == None ]]
}
finish() {
  local status=$? reason
  trap - EXIT
  set +e
  if (( status != 0 && bootstrapping )); then
    reason="Bootstrap failed: $boot_check (exit $status)"
    printf '%s\n' "$reason" >&2
    invoke bootstrap "$reason" || printf '%s\n' 'Bootstrap failure notification failed' >&2
  fi
  rm -f "${token_file:-}"
  shutdown -h now
  exit "$status"
}
trap finish EXIT
[[ $EUID -eq 0 && $REPOSITORY == NVIDIA/kumo-engine && $SOURCE_SHA =~ ^[0-9a-f]{40}$ && $SCRATCH_GIB =~ ^[1-9][0-9]{0,5}$ ]]
[[ $TIER == nightly || $TIER == weekly ]]
[[ $CAPACITY_DEADLINE =~ ^[1-9][0-9]{0,10}$ ]]
[[ -x /opt/actions-runner/config.sh && -x /opt/actions-runner/run.sh && ! -e /opt/actions-runner/.runner ]]
boot_check='capacity deadline'
(( $(date +%s) < CAPACITY_DEADLINE ))
boot_check='instance type'
metadata_token=$(curl --retry 5 --retry-all-errors --retry-max-time 30 --fail --silent --show-error --max-time 10 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)
actual_instance=$(curl --retry 5 --retry-all-errors --retry-max-time 30 --fail --silent --show-error --max-time 10 -H "X-aws-ec2-metadata-token: $metadata_token" http://169.254.169.254/latest/meta-data/instance-type)
[[ $actual_instance == "$INSTANCE_TYPE" ]]
unset metadata_token
boot_check='local SSD setup'
curl --retry 5 --retry-all-errors --retry-max-time 120 --connect-timeout 10 --max-time 30 --fail --silent --show-error --location "https://raw.githubusercontent.com/$REPOSITORY/$SOURCE_SHA/packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh" -o /run/engine-e2e-localssd.sh
# The formatter writes to /dev/tty even with --yes.
DISKGRAPH_DATA_ROOT=/mnt/localssd script --quiet --return --command 'bash /run/engine-e2e-localssd.sh --yes' /dev/null </dev/null
rm -f /run/engine-e2e-localssd.sh
mountpoint -q /mnt/localssd
[[ $(findmnt -n -o FSTYPE /mnt/localssd) == xfs ]]
boot_check='local SSD free space'
free_bytes=$(df -B1 --output=avail /mnt/localssd | tail -n 1)
(( free_bytes >= SCRATCH_GIB * 1024**3 ))
boot_check='runner user setup'
useradd --create-home --user-group --shell /bin/bash engine-e2e
boot_check='shared memory'
loginctl enable-linger engine-e2e
boot_check='instance metadata isolation'
runner_uid=$(id -u engine-e2e)
iptables -I OUTPUT -d 169.254.169.254 -m owner --uid-owner "$runner_uid" -j REJECT
ip6tables -I OUTPUT -d fd00:ec2::254 -m owner --uid-owner "$runner_uid" -j REJECT
boot_check='runner permissions'
install -d -o engine-e2e -g engine-e2e -m 700 /mnt/localssd/engine-e2e
chown -R engine-e2e:engine-e2e /opt/actions-runner
boot_check=GPU
runuser -u engine-e2e -- nvidia-smi
boot_check='Python 3.12'
runuser -u engine-e2e -- python3.12 -c 'import sys, venv; assert sys.version_info[:2] == (3, 12)'
boot_check='scratch write access'
runuser -u engine-e2e -- test -w /mnt/localssd/engine-e2e
echo 'engine-e2e boot prerequisites ready: GPU, Python 3.12, and local SSD'
bootstrapping=0 boot_check='runner registration'
registration_status=1
for delay in 0 5 10; do
  (( $(date +%s) + delay < CAPACITY_DEADLINE )) || break
  (( delay == 0 )) || sleep "$delay"
  (( $(date +%s) < CAPACITY_DEADLINE )) || break
  if invoke registration; then registration_status=0; break; else registration_status=$?; fi
done
(( registration_status == 0 )) || exit "$registration_status"
RUNNER_TOKEN=$(python3 -c 'import json,sys; token=json.load(open(sys.argv[1])); assert token is None or (isinstance(token,str) and token); print(token or "")' "$token_file")
rm -f "$token_file"
unset token_file
[[ -n $RUNNER_TOKEN ]] || exit 0
boot_check='runner configuration'
cd /opt/actions-runner
mode=()
if [[ $TIER == weekly ]]; then mode+=(--ephemeral); fi
runuser -u engine-e2e -- env -i HOME=/home/engine-e2e PATH=/usr/local/bin:/usr/bin:/bin ./config.sh --unattended --disableupdate --url "https://github.com/$REPOSITORY" --token "$RUNNER_TOKEN" --name "$LABEL-$(hostname)" --labels "$LABEL" --no-default-labels --work /mnt/localssd/engine-e2e/_work "${mode[@]}"
unset RUNNER_TOKEN
printf 'ENGINE_E2E_INSTANCE_TYPE=%s\nENGINE_E2E_CAPACITY_DEADLINE=%s\n' "$actual_instance" "$CAPACITY_DEADLINE" >> .env
chown engine-e2e:engine-e2e .env
runuser -u engine-e2e -- env -i HOME=/home/engine-e2e PATH=/usr/local/bin:/usr/bin:/bin ./run.sh
