# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packer template for Kumo single-host GCP custom image.
#
# Usage:
#   packer build -var kumo_image_tag=<commit-sha> -var gcp_project_id=<project> kumo-singlehost-gcp.pkr.hcl
#
# Common overrides:
#   -var accelerator_type=NVIDIA_L4        # L4 GPU (g2 family instances)
#   -var accelerator_type=NVIDIA_TESLA_A100
#   -var accelerator_count=2
#   -var local_ssd_count=2                 # attach 2x 375GB local SSDs at launch
#   -var local_ssd_interface=NVME          # NVME (default) or SCSI
#   -var boot_disk_size_gb=150
#
# Prerequisites:
#   - gcloud auth application-default login
#   - Build VM service account needs roles/artifactregistry.reader
#   - Docker images already pushed to Artifact Registry

packer {
  required_plugins {
    googlecompute = {
      version = ">= 1.1.0"
      source  = "github.com/hashicorp/googlecompute"
    }
  }
}

variable "kumo_image_tag" {
  type        = string
  description = "Docker image tag — full commit SHA or a version tag like v2.20.3_gcp"
}

variable "gcp_project_id" {
  type        = string
  description = "GCP project ID where the image will be built and stored"
}

variable "gcp_zone" {
  type    = string
  default = "us-west1-b"
}

variable "machine_type" {
  type        = string
  default     = "n1-standard-8"
  description = "Build VM machine type. n1-standard-8 is sufficient for the image build; use n1-standard-32 to match the runtime instance."
}

variable "instance_name" {
  type        = string
  default     = ""
  description = "Name for the Packer build VM. Defaults to kumo-packer-<uuid>. GCP requires lowercase alphanumeric + hyphens, max 63 chars."
}

variable "accelerator_type" {
  type        = string
  default     = "NVIDIA_TESLA_T4"
  description = "GPU accelerator type for launch-time instance_config.json. Common values: NVIDIA_TESLA_T4 (n1), NVIDIA_L4 (g2), NVIDIA_TESLA_A100 (a2). Does not affect the build VM."
}

variable "accelerator_count" {
  type        = number
  default     = 1
  description = "Number of GPU accelerators for launch-time instance_config.json. Does not affect the build VM."
}

variable "nvidia_driver_version" {
  type        = string
  default     = "580"
  description = "NVIDIA headless server driver major version to install. CUDA 13 containers require driver >= 580."
}

variable "boot_disk_size_gb" {
  type        = number
  default     = 100
  description = "Boot disk size in GB. Ubuntu 22.04 base ~10GB + Docker images ~20GB; 100GB is comfortable."
}

variable "boot_disk_type" {
  type        = string
  default     = "pd-ssd"
  description = "Boot disk type: pd-ssd (default), pd-balanced, or pd-standard."
}

# Local SSDs are ephemeral — they are NOT part of the image and must be attached
# at instance launch time. These variables are written into /etc/kumo/instance_config.json
# so boot scripts and operators know the intended configuration.
variable "local_ssd_count" {
  type        = number
  default     = 1
  description = "Number of local SSD scratch disks to attach at launch time. Each disk is 375GB."
}

variable "local_ssd_interface" {
  type        = string
  default     = "NVME"
  description = "Local SSD interface: NVME (default, higher throughput) or SCSI."
}

variable "artifact_registry" {
  type        = string
  default     = "us-west1-docker.pkg.dev"
  description = "Artifact Registry host (e.g. us-west1-docker.pkg.dev). Must match the repository's region."
}

variable "artifact_registry_repo" {
  type        = string
  default     = "kumo"
  description = "Artifact Registry repository name within the project (e.g. kumo, kumo-images)."
}

variable "artifact_registry_namespace" {
  type        = string
  default     = "kumo"
  description = "Image namespace (directory prefix) within the repository. Set to empty string if images are at the repo root."
}

variable "temporal_server_image" {
  type    = string
  default = "temporalio/server:1.31.1"
}

variable "temporal_admin_tools_image" {
  type    = string
  default = "temporalio/admin-tools:1.31.1"
}

locals {
  registry        = var.artifact_registry
  registry_prefix = var.artifact_registry_namespace != "" ? "${local.registry}/${var.gcp_project_id}/${var.artifact_registry_repo}/${var.artifact_registry_namespace}" : "${local.registry}/${var.gcp_project_id}/${var.artifact_registry_repo}"
  instance_name   = var.instance_name != "" ? var.instance_name : "kumo-packer-${uuidv4()}"
  # GCP image names must be lowercase alphanumeric + hyphens, max 63 chars.
  # Sanitize the tag: lowercase, replace dots/underscores with hyphens, truncate.
  # "kumo-singlehost-" is 16 chars, leaving 47 for the tag.
  # Examples:
  #   "abc123def456..."  (SHA)         → unchanged,     "kumo-singlehost-abc123..."
  #   "v2.20.3_gcp"      (version tag) → "v2-20-3-gcp", "kumo-singlehost-v2-20-3-gcp"
  image_tag_sanitized = substr(replace(replace(lower(var.kumo_image_tag), ".", "-"), "_", "-"), 0, 47)
  image_name          = "kumo-singlehost-${local.image_tag_sanitized}"
}

source "googlecompute" "kumo-singlehost" {
  project_id   = var.gcp_project_id
  zone         = var.gcp_zone
  machine_type = var.machine_type

  # Plain Ubuntu 22.04 LTS — minimal base, no pre-installed ML frameworks to conflict.
  # NVIDIA driver and nvidia-container-toolkit are installed in provisioners below.
  source_image_family     = "ubuntu-2204-lts"
  source_image_project_id = ["ubuntu-os-cloud"]

  # Boot disk. 100GB minimum; increase if pulling extra images or models.
  disk_size = var.boot_disk_size_gb
  disk_type = var.boot_disk_type

  image_name        = local.image_name
  image_description = "Kumo single-host image (${var.kumo_image_tag})"
  image_labels = {
    application = "kumo"
    component   = "singlehost"
    image_tag   = local.image_tag_sanitized
  }

  # Image family groups builds for easy lookup of the latest image:
  #   gcloud compute images describe-from-family kumo-singlehost
  image_family = "kumo-singlehost"

  instance_name = local.instance_name
  ssh_username  = "ubuntu"

  # No external IP — required by the compute.vmExternalIpAccess org policy.
  # use_iap tunnels SSH through GCP Identity-Aware Proxy instead.
  # Requires the caller to have roles/iap.tunnelResourceAccessor on the project.
  omit_external_ip = true
  use_internal_ip  = true
  use_iap          = true

  # The build VM uses the project's default Compute Engine SA.
  # That SA must have roles/artifactregistry.reader to pull images.
  # To use a dedicated SA instead:
  #   service_account_email = "kumo-packer@${var.gcp_project_id}.iam.gserviceaccount.com"

  # state_timeout controls how long Packer waits for GCE state transitions
  # (instance start, stop, image creation). Default 5m; increase for slow regions.
  # Note: total build time is 60-90min due to large Docker image pulls — that is
  # not controlled here but by the provisioner steps themselves.
  state_timeout = "10m"
}

build {
  sources = ["source.googlecompute.kumo-singlehost"]

  # 1. System packages
  provisioner "shell" {
    inline = [
      "while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo 'Waiting for apt lock...'; sleep 5; done",
      "sudo add-apt-repository -y universe",
      "sudo apt-get update",
      # Keep the launched kernel aligned with the kernel used to build the
      # NVIDIA module.
      "sudo apt-mark hold linux-gcp linux-image-gcp linux-headers-gcp || true",
      "dpkg-query -W -f='$${db:Status-Status} $${binary:Package}\\n' \"linux-image-$(uname -r)\" \"linux-headers-$(uname -r)\" \"linux-modules-$(uname -r)\" \"linux-modules-extra-$(uname -r)\" 2>/dev/null | awk '$1 == \"installed\" { print $2 }' | xargs -r sudo apt-mark hold",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y",
      "sudo apt-get install -y python3.10-venv xfsprogs nvme-cli curl gettext-base",
      "uname -r",
      "apt-mark showhold | sort",
    ]
  }

  # 2. Docker (not pre-installed on plain Ubuntu 22.04)
  provisioner "shell" {
    inline = [
      "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
      "echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable\" | sudo tee /etc/apt/sources.list.d/docker.list",
      "sudo apt-get update",
      "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
      "sudo systemctl enable docker",
      "sudo systemctl start docker",
    ]
  }

  # 3. NVIDIA driver + container toolkit.
  # Kumo containers bring their own CUDA runtime (nvidia/cuda:13.0.2-runtime-ubuntu22.04).
  # The host only needs: (a) the kernel driver so /dev/nvidia* devices exist,
  # (b) nvidia-container-toolkit so Docker can expose those devices to containers.
  # Using headless server driver — no display components, smaller footprint.
  # nvidia-cdi-refresh.service will warn during install (no GPU on build VM) — harmless.
  provisioner "shell" {
    inline = [
      "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
      "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
      "sudo apt-get update",
      "sudo apt-get install -y nvidia-headless-${var.nvidia_driver_version}-server nvidia-utils-${var.nvidia_driver_version}-server",
      # The GCP build VM has no accelerator, so modinfo is the build-time GPU
      # driver gate.
      "modinfo nvidia",
      # Freeze NVIDIA packages with the held kernel module.
      "dpkg-query -W -f='$${db:Status-Status} $${binary:Package}\\n' 'nvidia-*' 'libnvidia-*' 2>/dev/null | awk '$1 == \"installed\" { print $2 }' | xargs -r sudo apt-mark hold",
      "sudo apt-get install -y nvidia-container-toolkit",
      "sudo nvidia-ctk runtime configure --runtime=docker",
      "sudo systemctl restart docker",
    ]
  }

  # 4. Create kumouser (UID 999, matches Docker container user).
  # On Ubuntu 22.04 + Docker CE, GID 999 is the docker group so we skip creating
  # a separate kumouser group. --non-unique allows sharing UID 999 with lxd.
  provisioner "shell" {
    inline = [
      "sudo useradd -m -u 999 -g docker --non-unique -s /bin/bash kumouser || true",
      "echo 'kumouser ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/kumouser",
    ]
  }

  # 5. Add ubuntu user to docker group
  provisioner "shell" {
    inline = [
      "sudo usermod -aG docker ubuntu",
    ]
  }

  # 6. Stage provider-neutral deployment assets.
  provisioner "file" {
    sources = [
      "../../docker/compose/docker-compose.singlehost.yml",
      "../../docker/compose/docker-compose.singlehost.postgres.yml",
      "../../docker/compose/docker-compose.singlehost.gpu.yml",
      "../../docker/compose/config/postgres-initdb",
      "../../docker/compose/config/temporal",
    ]
    destination = "/tmp"
  }

  # Stage the provider-neutral runtime bundle consumed by the common installer.
  provisioner "file" {
    source      = "../shared/"
    destination = "/tmp/deploy-shared"
  }

  provisioner "shell" {
    inline = [
      "sudo bash /tmp/deploy-shared/scripts/install-singlehost-assets.sh",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable kumo-mount-nvme.service",
    ]
  }

  # 9. Pull and cache all Docker images from Artifact Registry.
  # Auth uses the build VM's GCE metadata service token — more reliable than
  # sudo gcloud auth configure-docker, which depends on gcloud being in root's
  # secure_path and may fail silently if Docker calls the helper as root.
  # SA must have roles/artifactregistry.reader on the registry.
  provisioner "shell" {
    inline = [
      # Fetch a fresh OAuth2 token from the metadata service and docker login.
      # The token has a 3600s TTL; all pulls complete well within that window.
      "TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])')",
      "echo \"$TOKEN\" | sudo docker login -u oauth2accesstoken --password-stdin ${local.registry}",
      # Pull the consolidated Kumo backend images from Artifact Registry.
      "sudo docker pull ${local.registry_prefix}/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker pull ${local.registry_prefix}/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker pull ${local.registry_prefix}/kumo-ui:${var.kumo_image_tag}",
      "sudo docker pull ${var.temporal_server_image}",
      "sudo docker pull ${var.temporal_admin_tools_image}",

      # Pull third-party public images

      "sudo docker pull prom/prometheus:v2.53.4",
      "sudo docker pull jaegertracing/all-in-one:1.76.0",
      "sudo docker pull nginx:1.29.7",
      "sudo docker pull postgres:16-alpine",
      # Re-tag to short names (strip registry prefix so the image runs air-gapped)
      "sudo docker tag ${local.registry_prefix}/kumo-backend-cpu:${var.kumo_image_tag} kumo/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker tag ${local.registry_prefix}/kumo-backend-gpu:${var.kumo_image_tag} kumo/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker tag ${local.registry_prefix}/kumo-ui:${var.kumo_image_tag} kumo/kumo-ui:${var.kumo_image_tag}",
      # Remove registry-prefixed tags (don't bake registry references into image)
      "sudo docker rmi ${local.registry_prefix}/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker rmi ${local.registry_prefix}/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker rmi ${local.registry_prefix}/kumo-ui:${var.kumo_image_tag}",
      "sudo docker logout ${local.registry}",
    ]
  }

  # 9a. Download the full GloVe artifact set to /opt/kumo/glove on the host.
  # The compose stack mounts this directory into every kumo container at
  # /opt/app/artifacts/glove (read-only). Backend images are built with
  # BAKE_GLOVE_LEVEL=none so they ship without GloVe; the host mount is the
  # single source of truth.
  #
  # Reuses docker/install_glove_files (same script the image build uses) with
  # GLOVE_DEST overridden to the host path and BAKE_GLOVE_LEVEL=all (singlehost
  # is air-gapped vis-à-vis cloud-side glove fallback).
  #
  # Auth: install_glove_files fetches from s3://kumo-public-datasets, which is
  # now private and requires signed `aws s3 cp`. This GCP builder does NOT
  # forward AWS credentials to the glove provisioner, so the S3 fetch needs
  # separate handling (AWS creds wired onto the GCP builder) before it will
  # authenticate. Out of scope for the AWS singlehost bake.
  provisioner "shell" {
    inline = [
      "sudo apt-get install -y awscli",
    ]
  }

  provisioner "file" {
    source      = "../../docker/install_glove_files"
    destination = "/tmp/install_glove_files"
  }

  provisioner "shell" {
    environment_vars = [
      "GLOVE_DEST=/opt/kumo/glove",
      "BAKE_GLOVE_LEVEL=all",
    ]
    inline = [
      "sudo mkdir -p /opt/kumo/glove",
      "sudo chmod +x /tmp/install_glove_files",
      "sudo -E /tmp/install_glove_files",
      "ls -la /opt/kumo/glove | head -20",
      "du -sh /opt/kumo/glove",
    ]
  }

  # 9b. Verify every compose image is cached locally before finalizing the image.
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
      "for img in $(cat /tmp/compose-images.txt); do sudo docker image inspect \"$img\" > /dev/null 2>&1 || { echo \"FATAL: image $img referenced in compose but not cached in GCP image\"; exit 1; }; done",
      "sudo rm /opt/kumo/compose/.env /tmp/compose-images.txt",
    ]
  }

  # 10. Install boot scripts
  provisioner "shell" {
    inline = [
      # NVMe mount script (detects GCP local SSDs via "nvme_card" model string)
      "sudo cp /opt/kumo/scripts/mount-nvme.sh /usr/local/bin/",
      "sudo chmod +x /usr/local/bin/mount-nvme.sh",
      # cloud-init: runs once on first boot
      "sudo cp /opt/kumo/scripts/kumo-firstboot.sh /var/lib/cloud/scripts/per-once/",
      "sudo chmod +x /var/lib/cloud/scripts/per-once/kumo-firstboot.sh",
      # cloud-init: runs on every boot
      "sudo cp /opt/kumo/scripts/kumo-boot.sh /var/lib/cloud/scripts/per-boot/",
      "sudo chmod +x /var/lib/cloud/scripts/per-boot/kumo-boot.sh",
      # Customer-facing tools
      "sudo cp /opt/kumo/scripts/kumo-restart.sh /usr/local/bin/kumo-restart",
      "sudo cp /opt/kumo/scripts/kumo-diag.sh /usr/local/bin/kumo-diag",
      "sudo cp /opt/kumo/scripts/kumo-health.sh /usr/local/bin/kumo-health",
      "sudo cp /opt/kumo/scripts/kumo-watchdog.sh /usr/local/bin/kumo-watchdog",
      "sudo cp /opt/kumo/scripts/kumo-autostop.sh /usr/local/bin/kumo-autostop",
      "sudo chmod +x /usr/local/bin/kumo-restart /usr/local/bin/kumo-diag /usr/local/bin/kumo-health /usr/local/bin/kumo-watchdog /usr/local/bin/kumo-autostop",
      # Cron jobs
      "echo '*/5 * * * * root /usr/local/bin/kumo-watchdog >> /var/log/kumo-watchdog.log 2>&1' | sudo tee /etc/cron.d/kumo-watchdog > /dev/null",
      "echo '*/5 * * * * root /usr/local/bin/kumo-autostop >> /var/log/kumo-autostop.log 2>&1' | sudo tee /etc/cron.d/kumo-autostop > /dev/null",
      "sudo cp /opt/kumo/deploy-shared/configs/logrotate-kumo-cron /etc/logrotate.d/kumo-cron",
    ]
  }

  # 11. Write image tag and instance configuration for boot scripts and operators.
  # instance_config.json documents the recommended launch configuration so that
  # kumo-health, kumo-diag, and launch docs all agree on the intended setup.
  # Local SSDs are ephemeral — attached at launch time, not baked into the image.
  # This file tells operators what to attach when launching from this image.
  provisioner "shell" {
    inline = [
      "echo '${var.kumo_image_tag}' | sudo tee /etc/kumo/image_tag",
      "python3 -c \"import json; open('/tmp/instance_config.json','w').write(json.dumps({'image_tag':'${var.kumo_image_tag}','accelerator_type':'${var.accelerator_type}','accelerator_count':${var.accelerator_count},'boot_disk_type':'${var.boot_disk_type}','boot_disk_size_gb':${var.boot_disk_size_gb},'local_ssd_count':${var.local_ssd_count},'local_ssd_interface':'${var.local_ssd_interface}'},indent=2))\"",
      "sudo mv /tmp/instance_config.json /etc/kumo/instance_config.json",
    ]
  }

  # 12. Clean up
  provisioner "shell" {
    inline = [
      "sudo docker system prune -f",
      "sudo apt-get clean",
      "sudo rm -rf /tmp/* /var/tmp/*",
      # Remove per-once semaphore so kumo-firstboot.sh runs on the real instance.
      "sudo rm -f /var/lib/cloud/sem/config_scripts_per_once.once",
    ]
  }
}
