# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packer template for Kumo single-host AMI with diskgraph.
#
# Target instance: g4dn.8xlarge (32 vCPU, 128GB RAM, 1x T4 GPU, 900GB NVMe)
#
# Usage:
#   packer build -var kumo_image_tag=<commit-sha> kumo-singlehost.pkr.hcl
#
# Prerequisites:
#   - AWS credentials configured
#   - Docker images already built and pushed to ECR

packer {
  required_plugins {
    amazon = {
      version = ">= 1.2.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "kumo_image_tag" {
  type        = string
  description = "Docker image tag (full 40-char commit SHA)"
}

variable "ami_name_tag" {
  type        = string
  description = "AMI name suffix. Defaults to kumo_image_tag when empty."
  default     = ""
}

variable "force_deregister" {
  type        = bool
  description = "Deregister an existing AMI with the same name before building."
  default     = false
}

locals {
  ami_tag = var.ami_name_tag != "" ? var.ami_name_tag : var.kumo_image_tag
}

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "instance_type" {
  type    = string
  default = "g4dn.8xlarge"
}

variable "docker_registry" {
  type    = string
  default = "926922431314.dkr.ecr.us-west-2.amazonaws.com"
}

variable "temporal_server_image" {
  type    = string
  default = "temporalio/server:1.31.1"
}

variable "temporal_admin_tools_image" {
  type    = string
  default = "temporalio/admin-tools:1.31.1"
}

# AWS credentials for ECR access during build.
# Pass via env vars: PKR_VAR_aws_access_key_id, etc.
# or via -var flags. These are temporary SSO session creds.
variable "aws_access_key_id" {
  type      = string
  sensitive = true
  default   = ""
}

variable "aws_secret_access_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "aws_session_token" {
  type      = string
  sensitive = true
  default   = ""
}

# Source from kumo-base AMI (built by kumo-base.pkr.hcl) instead of
# the AWS DLAMI. kumo-base ships Ubuntu 22.04 + NVIDIA driver + Docker
# + nvidia-container-toolkit only – install-base.sh strips the ~40 GB
# host CUDA toolkit the DLAMI ships and we don't use (CUDA lives in the
# kumo-backend-gpu container image).
#
# Net baseline: the DLAMI's ~49 GB used drops to ~9 GB on kumo-base.
source "amazon-ebs" "kumo-singlehost" {
  ami_name              = "kumo-singlehost-${local.ami_tag}"
  force_deregister      = var.force_deregister
  force_delete_snapshot = var.force_deregister
  instance_type         = var.instance_type
  region                = var.aws_region

  source_ami_filter {
    filters = {
      name                = "kumo-base-ubuntu22-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
      architecture        = "x86_64"
    }
    most_recent = true
    owners      = ["self"]
  }

  ssh_username = "ubuntu"

  # AMI creation can take a long time for large images with cached Docker layers.
  # Default waiter timeout is too short (~40min); increase to ~90min.
  aws_polling {
    delay_seconds = 30
    max_attempts  = 180
  }

  # For ECR access during build, use one of:
  # 1. iam_instance_profile = "your-ecr-read-profile" (needs iam:GetInstanceProfile)
  # 2. temporary_iam_instance_profile_policy_document (needs iam:CreateInstanceProfile)
  # 3. Pass AWS creds via environment_vars (current approach, uses builder's creds)
  #
  # TODO: Switch to option 1 or 2 when IAM permissions are available.
  run_tags = {
    Name        = "packer-kumo-singlehost-${local.ami_tag}"
    ManagedBy   = "github-actions-packer"
    Application = "kumo"
    Component   = "singlehost"
  }

  # 100GB for AMI build off kumo-base. Parallel docker pulls spike disk
  # usage well above settled image sizes because compressed + extracted
  # layers coexist briefly per layer per image. With all images pulling
  # simultaneously, the transient peak on a 50GB volume exceeded 45GB
  # of available space (observed nightly-20260517 build failure).
  # The slimmed kumo-base snapshot is 75GB (its host CUDA toolkit is
  # stripped) and its baked root uses only ~9GB, so this must be >= 75GB.
  # 100GB leaves ~90GB usable; the measured settled peak is ~56GB (docker
  # images + 7.5GB glove), comfortable headroom even as images grow.
  # Build provisioner emits `=== DISK [NN ...] ===` checkpoints in
  # the Packer log so we can verify peak usage at every phase.
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    volume_type           = "gp3"
    throughput            = 500
    iops                  = 4000
    delete_on_termination = true
  }

  tags = {
    Name        = "kumo-singlehost-${local.ami_tag}"
    ManagedBy   = "github-actions-packer"
    Application = "kumo"
    Component   = "singlehost"
    ImageTag    = var.kumo_image_tag
  }

  snapshot_tags = {
    Name        = "kumo-singlehost-${local.ami_tag}"
    ManagedBy   = "github-actions-packer"
    Application = "kumo"
    Component   = "singlehost"
    ImageTag    = local.ami_tag
  }
}

build {
  sources = ["source.amazon-ebs.kumo-singlehost"]

  # 1. System packages. Stop + mask unattended-upgrades first so the
  # systemd service doesn't grab the dpkg lock mid-build (the prior
  # `fuser`-based wait was racy: lock could be re-acquired between
  # the wait clearing and apt-get starting).
  provisioner "shell" {
    inline = [
      "echo '=== DISK [00 baseline pre-apt] ==='; df -h /",
      "sudo systemctl stop unattended-upgrades || true",
      "sudo systemctl mask unattended-upgrades || true",
      "while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo 'Waiting for residual apt lock...'; sleep 5; done",
      # DLAMI ships with duplicate CUDA repo entries (one from cuda-keyring,
      # one from add-apt-repository). Remove the add-apt-repository copies
      # to silence apt warnings. Also ensure universe is enabled for nvme-cli.
      "sudo rm -f /etc/apt/sources.list.d/archive_uri-*nvidia*.list",
      "sudo add-apt-repository -y universe",
      "sudo apt-get update",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y",
      "sudo apt-get install -y python3.10-venv xfsprogs nvme-cli gettext-base",
      # Log kernel/NVIDIA holds after the single-host apt upgrade.
      "apt-mark showhold | sort",
      "echo '=== DISK [01 after apt upgrade] ==='; df -h /",
    ]
  }

  # 1a. Reboot into the final kernel and verify host + container GPU support
  # after the single-host apt layer.
  provisioner "shell" {
    inline            = ["sudo reboot"]
    expect_disconnect = true
  }

  # smoke-test.sh checks host driver state and Docker GPU passthrough. It
  # removes its CUDA smoke image before snapshotting.
  provisioner "shell" {
    pause_before = "60s"
    script       = "../shared/base-ami/smoke-test.sh"
  }

  # 2. Create kumouser (UID 999, matches Docker container user)
  provisioner "shell" {
    inline = [
      "sudo groupadd -g 999 kumouser || true",
      "sudo useradd -m -u 999 -g 999 -s /bin/bash kumouser || true",
      "echo 'kumouser ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/kumouser",
    ]
  }

  # 3. Add ubuntu user to docker group
  provisioner "shell" {
    inline = [
      "sudo usermod -aG docker ubuntu",
    ]
  }

  # 4. Stage provider-neutral deployment assets.
  provisioner "file" {
    sources = [
      "../../docker/compose/docker-compose.singlehost.yml",
      "../../docker/compose/docker-compose.singlehost.postgres.yml",
      "../../docker/compose/docker-compose.singlehost.gpu.yml",
      "../../docker/compose/config/postgres-initdb",
      "../../docker/compose/config/temporal",
    ]
    # Multi-source destinations must end with a trailing slash.
    destination = "/tmp/"
  }

  # Directory destinations must exist before uploads.
  provisioner "shell" {
    inline = ["mkdir -p /tmp/deploy-shared"]
  }

  # Stage the provider-neutral runtime bundle consumed by the common installer.
  provisioner "file" {
    source      = "../shared/"
    destination = "/tmp/deploy-shared"
  }

  provisioner "shell" {
    inline = [
      "sudo bash /tmp/deploy-shared/scripts/install-singlehost-assets.sh",
      "sudo test -f /opt/kumo/deploy-shared/scripts/kumo-bootstrap.py",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable kumo-mount-nvme.service",
    ]
  }

  # 8. Pull and cache all Docker images from ECR (including Temporal)
  # Uses builder's AWS credentials passed via environment_vars
  provisioner "shell" {
    execute_command = "chmod +x {{ .Path }}; {{ .Vars }} bash {{ .Path }}"
    environment_vars = [
      "AWS_ACCESS_KEY_ID=${var.aws_access_key_id}",
      "AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}",
      "AWS_SESSION_TOKEN=${var.aws_session_token}",
    ]
    inline = [
      "echo '=== DISK [20 before docker pulls] ==='; df -h /",
      "( while true; do echo \"DISK-SAMPLE $(df -B1 --output=used / | tail -1)\"; sleep 5; done ) & echo $! > /tmp/dfsampler.pid",
      "aws ecr get-login-password --region ${var.aws_region} | sudo docker login --username AWS --password-stdin ${var.docker_registry}",
      # Pull all images in parallel. PIDs are captured individually so that
      # wait returns the correct exit status for each job.
      "sudo docker pull ${var.docker_registry}/kumo/kumo-backend-cpu:${var.kumo_image_tag} & pids=($!)",
      "sudo docker pull ${var.docker_registry}/kumo/kumo-backend-gpu:${var.kumo_image_tag} & pids+=($!)",
      "sudo docker pull ${var.docker_registry}/kumo/kumo-ui:${var.kumo_image_tag} & pids+=($!)",
      "sudo docker pull ${var.temporal_server_image} & pids+=($!)",
      "sudo docker pull ${var.temporal_admin_tools_image} & pids+=($!)",
      "sudo docker pull prom/prometheus:v2.53.4 & pids+=($!)",
      "sudo docker pull jaegertracing/all-in-one:1.76.0 & pids+=($!)",
      "sudo docker pull nginx:1.29.7 & pids+=($!)",
      "sudo docker pull postgres:16-alpine & pids+=($!)",
      "failed=0; for pid in $${pids[@]}; do wait $pid || { echo \"docker pull PID $pid failed\" >&2; failed=1; }; done; [ $failed -eq 0 ] || exit 1",
      "echo '=== DISK [23 after all docker pulls] ==='; df -h /",
      "kill \"$(cat /tmp/dfsampler.pid)\" 2>/dev/null || true; rm -f /tmp/dfsampler.pid",
      # Re-tag to short names (strip ECR registry prefix from baked AMI)
      "sudo docker tag ${var.docker_registry}/kumo/kumo-backend-cpu:${var.kumo_image_tag} kumo/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker tag ${var.docker_registry}/kumo/kumo-backend-gpu:${var.kumo_image_tag} kumo/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker tag ${var.docker_registry}/kumo/kumo-ui:${var.kumo_image_tag} kumo/kumo-ui:${var.kumo_image_tag}",
      # Remove ECR-prefixed tags and logout (don't bake registry references into AMI)
      "sudo docker rmi ${var.docker_registry}/kumo/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker rmi ${var.docker_registry}/kumo/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker rmi ${var.docker_registry}/kumo/kumo-ui:${var.kumo_image_tag}",
      "sudo docker logout ${var.docker_registry}",
      "echo '=== DISK [24 after docker tag/rmi cleanup] ==='; df -h /",
    ]
  }

  # 8a. Download the full GloVe artifact set to /opt/kumo/glove on the
  # host. Runs AFTER docker pulls so the 7.5GB on-disk footprint
  # doesn't compound with the gpu-image pull peak (compressed +
  # extracted layers double up briefly during pull). The compose
  # stack mounts this directory into every kumo container at
  # /opt/app/artifacts/glove (read-only). Backend images may include the
  # basic GloVe set for standard deployments; the host mount is the
  # singlehost source of truth for the full offline artifact set. PySpark
  # JARs are still baked into the backend images (install_dep_jars).
  #
  # Reuses docker/install_glove_files (same script the image build
  # uses) with GLOVE_DEST overridden to the host path.
  provisioner "file" {
    source      = "../../docker/install_glove_files"
    destination = "/tmp/install_glove_files"
  }

  provisioner "shell" {
    environment_vars = [
      "GLOVE_DEST=/opt/kumo/glove",
      "BAKE_GLOVE_LEVEL=all",
      "AWS_ACCESS_KEY_ID=${var.aws_access_key_id}",
      "AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}",
      "AWS_SESSION_TOKEN=${var.aws_session_token}",
    ]
    inline = [
      "echo '=== DISK [30 before glove download] ==='; df -h /",
      "sudo mkdir -p /opt/kumo/glove",
      # AWS CLI is a snap on /snap/bin (on the ssh user's PATH, not root's
      # sudo secure_path), so run the download as the ssh user like the ECR
      # provisioner does. Pre-chown to write, restore root ownership after.
      "sudo chown $(whoami):$(whoami) /opt/kumo/glove",
      "chmod +x /tmp/install_glove_files",
      "/tmp/install_glove_files",
      "sudo chown -R root:root /opt/kumo/glove",
      "ls -la /opt/kumo/glove | head -20",
      "du -sh /opt/kumo/glove",
      "echo '=== DISK [31 after glove download] ==='; df -h /",
    ]
  }

  # 8b. Verify every image in the compose file is cached locally.
  # Uses `docker compose config --images` (proper YAML parsing with variable
  # interpolation) then checks each image exists via `docker image inspect`.
  # No registry calls — purely local. Fails the Packer build if any image
  # was forgotten, catching the problem here instead of on the customer's
  # airgapped instance at boot time.
  provisioner "shell" {
    inline = [
      "echo 'KUMO_IMAGE_TAG=${var.kumo_image_tag}' | sudo tee /opt/kumo/compose/.env",
      "echo 'POSTGRES_PASSWORD=packer-build' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'LOCAL_DATAPLANE_HOST_MOUNT=/tmp' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'LOCAL_DATAPLANE_MOUNT=/tmp' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'PRODUCTION_LOCAL_DATAPLANE_ROOT=/tmp/{instance_id}/' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'PRODUCTION_LOCAL_ARTIFACT_ROOT=/tmp/artifacts/{instance_id}/' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'TEMPORAL_SERVER_IMAGE=${var.temporal_server_image}' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'TEMPORAL_ADMIN_TOOLS_IMAGE=${var.temporal_admin_tools_image}' | sudo tee -a /opt/kumo/compose/.env",
      "echo 'KUMO_BOOTSTRAP_DIR=/opt/kumo/deploy-shared' | sudo tee -a /opt/kumo/compose/.env",
      "cd /opt/kumo/compose && sudo docker compose -f docker-compose.singlehost.yml -f docker-compose.singlehost.postgres.yml -f docker-compose.singlehost.gpu.yml config --images > /tmp/compose-images.txt",
      "for img in $(cat /tmp/compose-images.txt); do sudo docker image inspect \"$img\" > /dev/null 2>&1 || { echo \"FATAL: image $img referenced in compose but not cached in AMI\"; exit 1; }; done",
      "sudo rm /opt/kumo/compose/.env /tmp/compose-images.txt",
    ]
  }

  # 8. Install boot scripts
  provisioner "shell" {
    inline = [
      # NVMe mount script used by systemd and cloud-init fallback
      "sudo cp /opt/kumo/scripts/mount-nvme.sh /usr/local/bin/",
      "sudo chmod +x /usr/local/bin/mount-nvme.sh",

      # cloud-init script for first boot
      "sudo cp /opt/kumo/scripts/kumo-firstboot.sh /var/lib/cloud/scripts/per-once/",
      "sudo chmod +x /var/lib/cloud/scripts/per-once/kumo-firstboot.sh",

      # boot script (every boot)
      "sudo cp /opt/kumo/scripts/kumo-boot.sh /var/lib/cloud/scripts/per-boot/",
      "sudo chmod +x /var/lib/cloud/scripts/per-boot/kumo-boot.sh",

      # Customer-facing tools
      "sudo cp /opt/kumo/scripts/kumo-restart.sh /usr/local/bin/kumo-restart",
      "sudo cp /opt/kumo/scripts/kumo-diag.sh /usr/local/bin/kumo-diag",
      "sudo cp /opt/kumo/scripts/kumo-health.sh /usr/local/bin/kumo-health",
      "sudo cp /opt/kumo/scripts/kumo-watchdog.sh /usr/local/bin/kumo-watchdog",
      "sudo cp /opt/kumo/scripts/kumo-autostop.sh /usr/local/bin/kumo-autostop",
      "sudo chmod +x /usr/local/bin/kumo-restart /usr/local/bin/kumo-diag /usr/local/bin/kumo-health /usr/local/bin/kumo-watchdog /usr/local/bin/kumo-autostop",
      # Install watchdog and autostop cron jobs (every 5 minutes)
      "echo '*/5 * * * * root /usr/local/bin/kumo-watchdog >> /var/log/kumo-watchdog.log 2>&1' | sudo tee /etc/cron.d/kumo-watchdog > /dev/null",
      "echo '*/5 * * * * root /usr/local/bin/kumo-autostop >> /var/log/kumo-autostop.log 2>&1' | sudo tee /etc/cron.d/kumo-autostop > /dev/null",
      # Log rotation for cron job logs
      "sudo cp /opt/kumo/deploy-shared/configs/logrotate-kumo-cron /etc/logrotate.d/kumo-cron",
    ]
  }

  # 9. Write image tag for boot scripts
  provisioner "shell" {
    inline = [
      "echo '${var.kumo_image_tag}' | sudo tee /etc/kumo/image_tag",
    ]
  }

  # 10. Clean up
  provisioner "shell" {
    inline = [
      "sudo docker system prune -f",
      "sudo apt-get clean",
      "sudo rm -rf /tmp/* /var/tmp/*",
      # Remove cloud-init per-once semaphore so firstboot script runs on
      # the real instance. Don't use `cloud-init clean` — it deletes
      # everything under /var/lib/cloud/ including our installed scripts.
      "sudo rm -f /var/lib/cloud/sem/config_scripts_per_once.once",
    ]
  }
}
