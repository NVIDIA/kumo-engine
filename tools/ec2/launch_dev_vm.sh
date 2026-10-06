#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Launch an engineering EC2 dev VM from the approved AWS Launch Templates.
#
# Usage:
#   ./tools/ec2/launch_dev_vm.sh
#
# Prerequisites:
#   - AWS CLI v2 is installed and authenticated with access to the target account.
#   - AWS Session Manager plugin is installed locally.
#   - A local SSH key pair exists, for example ~/.ssh/id_ed25519 and
#     ~/.ssh/id_ed25519.pub. The public key is installed on the VM.
#
# Defaults:
#   - Region: current AWS CLI region, falling back to us-west-2.
#   - Name tag and SSH alias: <local-user>-eng-vm.
#   - If that name conflicts with an active EC2 instance or an existing SSH
#     config Host, the script uses <local-user>-eng-vm-2, then -3, etc.
#   - Template: CPU, with menu options for CPU, SSD, and GPU.
#   - Auto-shutdown: daily at 00:00 America/Los_Angeles.
#
# What this script does:
#   - Shows the available engineering launch templates.
#   - Lets you review and edit region, name, template, SSH user, and key paths.
#   - Validates with an AWS dry-run before launch.
#   - Launches the instance with the configured Name tag.
#   - Waits for EC2 running state and Systems Manager registration.
#   - Creates the remote Linux user, installs your public SSH key, grants
#     passwordless sudo, and enables the auto-shutdown systemd timer.
#   - Prepends a Host entry to ~/.ssh/config so `ssh <alias>` works through SSM.
#
# The script intentionally does not commit, push, or install repo aliases.
set -euo pipefail

log()  { printf "\033[1;32m[+]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[!]\033[0m %s\n" "$*" >&2; }
err()  { printf "\033[1;31m[x]\033[0m %s\n" "$*" >&2; }

CONFIG_FILE="${SSH_CONFIG_FILE:-$HOME/.ssh/config}"
SSH_DIR="$(dirname "$CONFIG_FILE")"
KEY_PAIR_NAME="${DEV_VM_KEY_PAIR_NAME:-kumo_init}"
AUTO_SHUTDOWN_SERVICE_NAME="rc-vm-auto-shutdown.service"
AUTO_SHUTDOWN_TIMER_NAME="rc-vm-auto-shutdown.timer"
AUTO_SHUTDOWN_TIMEZONE="America/Los_Angeles"

install_hint() {
  local tool="$1"
  case "$tool" in
    aws)
      printf 'Install the AWS CLI and rerun this script.\n' >&2
      printf 'macOS: brew install awscli\n' >&2
      printf 'Linux: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html\n' >&2
      ;;
    aws-session-manager-plugin)
      printf 'Install the AWS Session Manager plugin and rerun this script.\n' >&2
      printf 'macOS: brew install --cask session-manager-plugin\n' >&2
      printf 'Linux: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html\n' >&2
      ;;
  esac
}

prompt_with_default() {
  local prompt="$1" default="$2" value
  read -r -p "$prompt [$default]: " value
  printf '%s\n' "${value:-$default}"
}

prompt_yes_no() {
  local prompt="$1" default="$2" value normalized suffix
  case "$default" in
    yes) suffix="[Y/n]" ;;
    no) suffix="[y/N]" ;;
    *) err "prompt_yes_no default must be yes or no."; exit 1 ;;
  esac
  while true; do
    read -r -p "$prompt $suffix: " value
    normalized="${value:-$default}"
    normalized="$(printf '%s' "$normalized" | tr '[:upper:]' '[:lower:]')"
    case "$normalized" in
      y|yes) printf 'yes\n'; return 0 ;;
      n|no) printf 'no\n'; return 0 ;;
    esac
    warn "Please answer yes or no."
  done
}

sanitize_name_part() {
  printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g'
}

default_region() {
  local configured=""
  configured="$(aws configure get region 2>/dev/null || true)"
  printf '%s\n' "${configured:-us-west-2}"
}

default_identity_file() {
  if [[ -f "$HOME/.ssh/id_ed25519" ]]; then
    printf '%s\n' "$HOME/.ssh/id_ed25519"
  elif [[ -f "$HOME/.ssh/id_rsa" ]]; then
    printf '%s\n' "$HOME/.ssh/id_rsa"
  elif [[ -f "$HOME/.ssh/kumo-init-key.pem" ]]; then
    printf '%s\n' "$HOME/.ssh/kumo-init-key.pem"
  else
    printf '%s\n' "$HOME/.ssh/id_ed25519"
  fi
}

default_public_key_file() {
  local private_key="$1"
  if [[ -f "${private_key}.pub" ]]; then
    printf '%s\n' "${private_key}.pub"
  elif [[ -f "$HOME/.ssh/id_ed25519.pub" ]]; then
    printf '%s\n' "$HOME/.ssh/id_ed25519.pub"
  elif [[ -f "$HOME/.ssh/id_rsa.pub" ]]; then
    printf '%s\n' "$HOME/.ssh/id_rsa.pub"
  else
    printf '%s\n' "${private_key}.pub"
  fi
}

template_for_kind() {
  case "$1" in
    cpu) printf 'Engineering-CPU-template\n' ;;
    ssd) printf 'Engineering-SSD-template\n' ;;
    gpu) printf 'Engineering-GPU-template\n' ;;
    *) return 1 ;;
  esac
}

kind_label() {
  case "$1" in
    cpu) printf 'CPU\n' ;;
    ssd) printf 'SSD\n' ;;
    gpu) printf 'GPU\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

select_kind() {
  local value normalized
  while true; do
    printf '\nInstance template:\n' >&2
    printf '  1) cpu  Engineering-CPU-template [default]\n' >&2
    printf '  2) ssd  Engineering-SSD-template\n' >&2
    printf '  3) gpu  Engineering-GPU-template\n' >&2
    read -r -p "Select instance template (number or name) [cpu]: " value
    normalized="$(printf '%s' "${value:-cpu}" | tr '[:upper:]' '[:lower:]')"
    case "$normalized" in
      1|cpu) printf 'cpu\n'; return 0 ;;
      2|ssd) printf 'ssd\n'; return 0 ;;
      3|gpu) printf 'gpu\n'; return 0 ;;
    esac
    warn "Choose 1, 2, 3, cpu, ssd, or gpu."
  done
}

alias_exists() {
  local alias_name="$1"
  [[ -f "$CONFIG_FILE" ]] || return 1
  awk -v alias_name="$alias_name" '
    $1 == "Host" {
      for (i = 2; i <= NF; i++) {
        if ($i == alias_name) found = 1
      }
    }
    END { exit found ? 0 : 1 }
  ' "$CONFIG_FILE"
}

ensure_aws_ready() {
  if ! command -v aws >/dev/null 2>&1; then
    err "aws CLI is required to launch dev VMs."
    install_hint aws
    exit 1
  fi

  if ! command -v session-manager-plugin >/dev/null 2>&1; then
    err "AWS Session Manager plugin is required to SSH into dev VMs."
    install_hint aws-session-manager-plugin
    exit 1
  fi

  if aws sts get-caller-identity >/dev/null 2>&1; then
    return 0
  fi

  warn "AWS credentials are not active. Running aws sso login."
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    aws sso login --profile "$AWS_PROFILE"
  else
    aws sso login
  fi
  aws sts get-caller-identity >/dev/null
}

print_aws_identity() {
  local identity
  identity="$(aws sts get-caller-identity --query '[Account,Arn]' --output text)"
  printf 'AWS identity: %s\n' "$identity"
}

template_exists() {
  local region="$1" template="$2"
  aws ec2 describe-launch-templates \
    --region "$region" \
    --launch-template-names "$template" \
    --query 'LaunchTemplates[0].LaunchTemplateName' \
    --output text >/dev/null 2>&1
}

print_available_templates() {
  local region="$1"
  aws ec2 describe-launch-templates \
    --region "$region" \
    --query 'LaunchTemplates[?LaunchTemplateName==`Engineering-CPU-template` || LaunchTemplateName==`Engineering-SSD-template` || LaunchTemplateName==`Engineering-GPU-template`].[LaunchTemplateName,LaunchTemplateId,DefaultVersionNumber]' \
    --output table
}

key_pair_exists() {
  local region="$1"
  aws ec2 describe-key-pairs \
    --region "$region" \
    --key-names "$KEY_PAIR_NAME" \
    --query 'KeyPairs[0].KeyName' \
    --output text >/dev/null 2>&1
}

duplicate_instances() {
  local region="$1" vm_name="$2"
  aws ec2 describe-instances \
    --region "$region" \
    --filters "Name=tag:Name,Values=$vm_name" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,Tags[?Key==`Name`]|[0].Value]' \
    --output text
}

validate_no_duplicate_name() {
  local region="$1" vm_name="$2" duplicates
  duplicates="$(duplicate_instances "$region" "$vm_name")"
  if [[ -n "$duplicates" ]]; then
    warn "An active EC2 instance already uses Name=$vm_name in $region:"
    printf '%s\n' "$duplicates" >&2
    return 1
  fi
}

name_is_available() {
  local region="$1" vm_name="$2"
  [[ -z "$(duplicate_instances "$region" "$vm_name")" ]]
}

default_vm_name_for_region() {
  local region="$1" user_part="$2" base candidate index
  base="${user_part}-eng-vm"
  candidate="$base"
  index=2

  while ! name_is_available "$region" "$candidate" || alias_exists "$candidate"; do
    candidate="${user_part}-eng-vm-${index}"
    index=$((index + 1))
  done

  printf '%s\n' "$candidate"
}

validate_inputs() {
  local template
  template="$(template_for_kind "$instance_kind")"

  if [[ -z "${vm_name// }" ]]; then
    warn "VM name cannot be empty."
    return 1
  fi
  if [[ ! "$vm_name" =~ ^[a-z0-9-]+-eng-vm(-[0-9]+)?$ ]]; then
    warn "VM name should follow <username>-eng-vm or <username>-eng-vm-<number>."
    return 1
  fi
  if [[ -z "${ssh_alias// }" ]]; then
    warn "SSH alias cannot be empty."
    return 1
  fi
  if alias_exists "$ssh_alias"; then
    warn "SSH alias '$ssh_alias' already exists in $CONFIG_FILE."
    return 1
  fi
  if [[ -z "${ssh_user// }" ]]; then
    warn "SSH user cannot be empty."
    return 1
  fi
  if [[ "$ssh_user" =~ [^a-z_][^a-z0-9_-]*$ || ! "$ssh_user" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    warn "SSH user must be a valid Linux username."
    return 1
  fi
  if [[ ! -f "$public_key_file" ]]; then
    warn "Public key file not found: $public_key_file"
    return 1
  fi
  if ! template_exists "$region" "$template"; then
    warn "Launch template $template was not found in $region."
    return 1
  fi
  if ! key_pair_exists "$region"; then
    warn "Key pair $KEY_PAIR_NAME was not found in $region."
    return 1
  fi
  validate_no_duplicate_name "$region" "$vm_name" || return 1
}

dry_run_launch() {
  local template="$1" output
  set +e
  output="$(aws ec2 run-instances \
    --region "$region" \
    --dry-run \
    --launch-template "LaunchTemplateName=$template,Version=\$Default" \
    --key-name "$KEY_PAIR_NAME" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$vm_name},{Key=Type,Value=Development}]" \
    --count 1 2>&1)"
  local status=$?
  set -e

  if [[ "$output" == *"DryRunOperation"* ]]; then
    return 0
  fi

  printf '%s\n' "$output" >&2
  return "$status"
}

launch_instance() {
  local template="$1"
  aws ec2 run-instances \
    --region "$region" \
    --launch-template "LaunchTemplateName=$template,Version=\$Default" \
    --key-name "$KEY_PAIR_NAME" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$vm_name},{Key=Type,Value=Development}]" \
    --count 1 \
    --query 'Instances[0].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone]' \
    --output text
}

wait_for_instance_running() {
  local instance_id="$1"
  log "Waiting for $instance_id to reach running state..."
  aws ec2 wait instance-running --region "$region" --instance-ids "$instance_id"
}

wait_for_ssm_online() {
  local instance_id="$1" deadline now status
  deadline=$((SECONDS + 600))
  log "Waiting for $instance_id to register with Systems Manager..."
  while true; do
    status="$(aws ssm describe-instance-information \
      --region "$region" \
      --filters "Key=InstanceIds,Values=$instance_id" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text 2>/dev/null || true)"
    if [[ "$status" == "Online" ]]; then
      return 0
    fi
    now="$SECONDS"
    if (( now >= deadline )); then
      err "Timed out waiting for Systems Manager to report $instance_id online."
      exit 1
    fi
    sleep 10
  done
}

json_escape() {
  local input="$1"
  input="${input//\\/\\\\}"
  input="${input//\"/\\\"}"
  input="${input//$'\n'/\\n}"
  printf '%s\n' "$input"
}

base64_one_line() {
  base64 | tr -d '\n'
}

run_remote_shell() {
  local instance_id="$1" commands="$2" encoded wrapped escaped command_id status deadline now
  encoded="$(printf '%s' "$commands" | base64_one_line)"
  wrapped="$(cat <<EOF
set -eu
encoded='$encoded'
printf '%s' "\$encoded" | base64 -d | bash
EOF
)"
  escaped="$(json_escape "$wrapped")"
  command_id="$(aws ssm send-command \
    --region "$region" \
    --instance-ids "$instance_id" \
    --document-name "AWS-RunShellScript" \
    --parameters "{\"commands\":[\"$escaped\"]}" \
    --query 'Command.CommandId' \
    --output text)"

  deadline=$((SECONDS + 600))
  while true; do
    status="$(aws ssm get-command-invocation \
      --region "$region" \
      --command-id "$command_id" \
      --instance-id "$instance_id" \
      --query 'Status' \
      --output text 2>/dev/null || true)"
    case "$status" in
      Success)
        return 0
        ;;
      Failed|Cancelled|TimedOut|Cancelling)
        aws ssm get-command-invocation \
          --region "$region" \
          --command-id "$command_id" \
          --instance-id "$instance_id" \
          --query '{Status:Status,Stdout:StandardOutputContent,Stderr:StandardErrorContent}' \
          --output json >&2 || true
        return 1
        ;;
    esac
    now="$SECONDS"
    if (( now >= deadline )); then
      err "Timed out waiting for remote command $command_id."
      return 1
    fi
    sleep 5
  done
}

remote_single_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

bootstrap_instance() {
  local instance_id="$1" public_key quoted_user quoted_public_key commands
  public_key="$(<"$public_key_file")"
  quoted_user="$(remote_single_quote "$ssh_user")"
  quoted_public_key="$(remote_single_quote "$public_key")"

  commands="$(cat <<EOF
set -euo pipefail
devuser=${quoted_user}
pubkey=${quoted_public_key}

if ! id "\$devuser" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "\$devuser"
fi

install -d -m 700 -o "\$devuser" -g "\$devuser" "/home/\$devuser/.ssh"
touch "/home/\$devuser/.ssh/authorized_keys"
grep -qxF "\$pubkey" "/home/\$devuser/.ssh/authorized_keys" || printf '%s\n' "\$pubkey" >> "/home/\$devuser/.ssh/authorized_keys"
chown "\$devuser:\$devuser" "/home/\$devuser/.ssh/authorized_keys"
chmod 600 "/home/\$devuser/.ssh/authorized_keys"

printf '%s ALL=(ALL) NOPASSWD: ALL\n' "\$devuser" > "/etc/sudoers.d/\$devuser"
chmod 440 "/etc/sudoers.d/\$devuser"
visudo -cf "/etc/sudoers.d/\$devuser" >/dev/null

cat > /etc/systemd/system/${AUTO_SHUTDOWN_SERVICE_NAME} <<'SERVICE'
[Unit]
Description=Shut down VM

[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now
SERVICE

cat > /etc/systemd/system/${AUTO_SHUTDOWN_TIMER_NAME} <<'TIMER'
[Unit]
Description=Daily VM auto-shutdown

[Timer]
OnCalendar=*-*-* 00:00:00 ${AUTO_SHUTDOWN_TIMEZONE}
Persistent=true

[Install]
WantedBy=timers.target
TIMER

systemctl daemon-reload
systemctl enable --now ${AUTO_SHUTDOWN_TIMER_NAME}
EOF
)"

  log "Bootstrapping $instance_id: user, SSH key, sudo, and 00:00 ${AUTO_SHUTDOWN_TIMEZONE} auto-shutdown..."
  run_remote_shell "$instance_id" "$commands"
}

write_ssh_config() {
  local instance_id="$1" tmp_config config_block aws_proxy_command

  printf -v aws_proxy_command '    ProxyCommand sh -c "aws ssm start-session --target %%h --region %s --document-name AWS-StartSSHSession --parameters \\"portNumber=%%p\\""' "$region"
  config_block="$(cat <<EOF
Host $ssh_alias
    HostName $instance_id
    User $ssh_user
    IdentityFile $identity_file
    ForwardAgent yes
    IdentityAgent \$SSH_AUTH_SOCK
${aws_proxy_command}
EOF
)"

  mkdir -p "$SSH_DIR"
  chmod 700 "$SSH_DIR" || true
  touch "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE" || true

  tmp_config="$(mktemp)"
  if [[ -s "$CONFIG_FILE" ]]; then
    {
      printf '%s\n\n' "$config_block"
      cat "$CONFIG_FILE"
    } > "$tmp_config"
  else
    printf '%s\n' "$config_block" > "$tmp_config"
  fi
  mv "$tmp_config" "$CONFIG_FILE"
  log "Prepended Host $ssh_alias to $CONFIG_FILE"
}

print_summary() {
  local template
  template="$(template_for_kind "$instance_kind")"
  printf '\nDev VM launch summary:\n'
  printf '  Region:        %s\n' "$region"
  printf '  Name tag:      %s\n' "$vm_name"
  printf '  Instance kind: %s\n' "$(kind_label "$instance_kind")"
  printf '  Template:      %s\n' "$template"
  printf '  Key pair:      %s\n' "$KEY_PAIR_NAME"
  printf '  SSH alias:     %s\n' "$ssh_alias"
  printf '  SSH user:      %s\n' "$ssh_user"
  printf '  Identity file: %s\n' "$identity_file"
  printf '  Public key:    %s\n' "$public_key_file"
  printf '  Auto-shutdown: 00:00 %s\n' "$AUTO_SHUTDOWN_TIMEZONE"
}

edit_loop() {
  local choice
  while true; do
    print_summary
    printf '\n'
    printf '  1) Launch\n'
    printf '  2) Edit name\n'
    printf '  3) Edit region\n'
    printf '  4) Edit instance kind\n'
    printf '  5) Edit SSH alias\n'
    printf '  6) Edit SSH user\n'
    printf '  7) Edit identity file\n'
    printf '  8) Edit public key file\n'
    printf '  9) Show available templates\n'
    printf ' 10) Abort\n'
    read -r -p "Choose [1]: " choice
    case "${choice:-1}" in
      1) return 0 ;;
      2)
        vm_name="$(prompt_with_default "EC2 Name tag" "$vm_name")"
        ssh_alias="$vm_name"
        ;;
      3)
        region="$(prompt_with_default "AWS region" "$region")"
        ;;
      4)
        instance_kind="$(select_kind)"
        ;;
      5)
        ssh_alias="$(prompt_with_default "SSH alias" "$ssh_alias")"
        ;;
      6)
        ssh_user="$(prompt_with_default "SSH user" "$ssh_user")"
        ;;
      7)
        identity_file="$(prompt_with_default "Private key path" "$identity_file")"
        public_key_file="$(default_public_key_file "$identity_file")"
        ;;
      8)
        public_key_file="$(prompt_with_default "Public key path" "$public_key_file")"
        ;;
      9)
        print_available_templates "$region" || true
        ;;
      10)
        log "Aborted without launching."
        exit 0
        ;;
      *)
        warn "Choose a number between 1 and 10."
        ;;
    esac
  done
}

ensure_aws_ready
print_aws_identity

default_user="$(sanitize_name_part "${USER:-$(id -un)}")"
region="$(default_region)"
if [[ "$region" != "us-west-2" && "$region" != "ap-south-1" ]]; then
  region="us-west-2"
fi
vm_name="$(default_vm_name_for_region "$region" "$default_user")"
ssh_alias="$vm_name"
ssh_user="$default_user"
identity_file="$(default_identity_file)"
public_key_file="$(default_public_key_file "$identity_file")"
instance_kind="$(select_kind)"

printf '\nDiscovered Engineering launch templates in %s:\n' "$region"
print_available_templates "$region" || true

while true; do
  edit_loop
  if validate_inputs; then
    break
  fi
  warn "Fix the validation issue above, then try launching again."
done

if [[ ! -f "$identity_file" ]]; then
  warn "Private key file not found locally: $identity_file"
  warn "SSH config will still be written, but connect will fail until the key exists."
fi

template="$(template_for_kind "$instance_kind")"
log "Validating launch request with AWS dry-run..."
dry_run_launch "$template" || {
  err "AWS dry-run launch validation failed."
  exit 1
}

if [[ "$(prompt_yes_no "Launch $vm_name from $template in $region" "yes")" != "yes" ]]; then
  log "Aborted without launching."
  exit 0
fi

validate_no_duplicate_name "$region" "$vm_name" || {
  err "Duplicate instance name appeared before launch. Aborting."
  exit 1
}

log "Launching $vm_name..."
launch_output="$(launch_instance "$template")"
read -r instance_id instance_type instance_state availability_zone <<<"$launch_output"

log "Launched $instance_id ($instance_type, $instance_state, $availability_zone)."
wait_for_instance_running "$instance_id"
wait_for_ssm_online "$instance_id"
bootstrap_instance "$instance_id"
write_ssh_config "$instance_id"

printf '\nSSH config updated:\n'
printf '  File:         %s\n' "$CONFIG_FILE"
printf '  Host:         %s\n' "$ssh_alias"
printf '  HostName:     %s\n' "$instance_id"
printf '  User:         %s\n' "$ssh_user"
printf '  IdentityFile: %s\n' "$identity_file"
printf '\nConnect:\n'
printf '  ssh %s\n' "$ssh_alias"
printf '\n'
printf 'If you have the rc SSH helpers installed, you can also use:\n'
printf '  vm %s\n' "$ssh_alias"
printf '\n'
printf 'SSH helper setup notes:\n'
printf '  open https://github.com/babymastodon/rc/blob/master/ssh/README.md\n'
printf '  mkdir -p ~/code\n'
printf '  git clone git@github.com:babymastodon/rc.git ~/code/rc\n'
printf '  ~/code/rc/install.sh\n'
printf '  source ~/.bashrc\n'
