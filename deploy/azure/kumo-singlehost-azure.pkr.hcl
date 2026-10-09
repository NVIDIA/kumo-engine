# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

packer {
  required_plugins {
    azure = {
      source  = "github.com/hashicorp/azure"
      version = "= 2.6.2"
    }
  }
}

variable "subscription_id" {
  type = string
}
variable "resource_group_name" {
  type = string
}
variable "location" {
  type    = string
  default = "westus2"
}
variable "azure_environment" {
  type    = string
  default = "public"
}
variable "kumo_image_tag" {
  type = string
}
variable "managed_image_name" {
  type    = string
  default = ""
}
variable "build_vm_size" {
  type    = string
  default = "Standard_NC16as_T4_v3"
}
variable "acr_name" {
  type = string
}
variable "acr_namespace" {
  type    = string
  default = ""
}
variable "builder_identity_id" {
  type = string
}
variable "builder_identity_client_id" {
  type = string
}
variable "glove_source_url" {
  type        = string
  description = "HTTPS URL of the immutable GloVe directory in Azure Blob Storage; no SAS token."
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
  image_name = var.managed_image_name != "" ? var.managed_image_name : "kumo-singlehost-${replace(var.kumo_image_tag, ".", "-")}"
  acr_suffix = {
    public       = "azurecr.io"
    usgovernment = "azurecr.us"
    china        = "azurecr.cn"
  }[var.azure_environment]
  acr  = "${var.acr_name}.${local.acr_suffix}"
  repo = var.acr_namespace != "" ? "${local.acr}/${var.acr_namespace}" : local.acr
  packer_cloud = {
    public       = "Public"
    usgovernment = "USGovernment"
    china        = "China"
  }[var.azure_environment]
  cli_cloud = {
    public       = "AzureCloud"
    usgovernment = "AzureUSGovernment"
    china        = "AzureChinaCloud"
  }[var.azure_environment]
}

source "azure-arm" "kumo" {
  use_azure_cli_auth                = true
  subscription_id                   = var.subscription_id
  managed_image_resource_group_name = var.resource_group_name
  managed_image_name                = local.image_name
  location                          = var.location
  cloud_environment_name            = local.packer_cloud
  user_assigned_managed_identities  = [var.builder_identity_id]
  os_type                           = "Linux"
  image_publisher                   = "Canonical"
  image_offer                       = "0001-com-ubuntu-server-jammy"
  image_sku                         = "22_04-lts-gen2"
  vm_size                           = var.build_vm_size
  azure_tags = {
    application = "kumo-singlehost"
    version     = var.kumo_image_tag
  }
}

build {
  sources = ["source.azure-arm.kumo"]

  provisioner "file" {
    source      = "../shared/base-ami/install-base.sh"
    destination = "/tmp/install-base.sh"
  }
  provisioner "shell" {
    inline = [
      "chmod +x /tmp/install-base.sh",
      "sudo /tmp/install-base.sh",
      "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash",
    ]
  }

  provisioner "shell" {
    inline = [
      "sudo useradd -m -u 999 -g docker --non-unique -s /bin/bash kumouser || true",
      "echo 'kumouser ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/kumouser",
    ]
  }

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
      "sudo mkdir -p /opt/kumo/glove",
    ]
  }

  provisioner "shell" {
    inline = [
      "az cloud set --name ${local.cli_cloud}",
      "az login --identity --username ${var.builder_identity_client_id} --allow-no-subscriptions >/dev/null",
      "ACR_TOKEN=$(az acr login --name ${var.acr_name} --expose-token --query accessToken -o tsv)",
      "echo \"$ACR_TOKEN\" | sudo docker login ${local.acr} --username 00000000-0000-0000-0000-000000000000 --password-stdin",
      "sudo chown ubuntu:docker /opt/kumo/glove",
      "az storage copy --source '${var.glove_source_url}' --destination /opt/kumo/glove --recursive --auth-mode login --only-show-errors",
      "test -n \"$(find /opt/kumo/glove -type f -print -quit)\" || { echo 'FATAL: GloVe artifact source was empty'; exit 1; }",
      "sudo chown -R root:root /opt/kumo/glove && sudo chmod -R a-w /opt/kumo/glove",
      "sudo docker pull ${local.repo}/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker pull ${local.repo}/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker pull ${local.repo}/kumo-ui:${var.kumo_image_tag}",
      "sudo docker pull ${local.repo}/prometheus:latest",
      "sudo docker pull ${var.temporal_server_image}",
      "sudo docker pull ${var.temporal_admin_tools_image}",
      "sudo docker pull jaegertracing/all-in-one:1.76.0",
      "sudo docker pull nginx:1.29.7",
      "sudo docker pull postgres:16-alpine",
      "sudo docker tag ${local.repo}/kumo-backend-cpu:${var.kumo_image_tag} kumo/kumo-backend-cpu:${var.kumo_image_tag}",
      "sudo docker tag ${local.repo}/kumo-backend-gpu:${var.kumo_image_tag} kumo/kumo-backend-gpu:${var.kumo_image_tag}",
      "sudo docker tag ${local.repo}/kumo-ui:${var.kumo_image_tag} kumo/kumo-ui:${var.kumo_image_tag}",
      "sudo docker tag ${local.repo}/prometheus:latest kumo/prometheus:latest",
      "sudo docker logout ${local.acr}",
    ]
  }

  provisioner "shell" {
    inline = [
      "sudo cp /opt/kumo/scripts/mount-nvme.sh /usr/local/bin/",
      "sudo cp /opt/kumo/scripts/kumo-firstboot.sh /var/lib/cloud/scripts/per-once/",
      "sudo cp /opt/kumo/scripts/kumo-boot.sh /var/lib/cloud/scripts/per-boot/",
      "for tool in restart diag health watchdog autostop; do sudo cp /opt/kumo/scripts/kumo-$tool.sh /usr/local/bin/kumo-$tool; done",
      "sudo chmod +x /usr/local/bin/kumo-* /var/lib/cloud/scripts/per-once/kumo-firstboot.sh /var/lib/cloud/scripts/per-boot/kumo-boot.sh",
      "echo '*/5 * * * * root /usr/local/bin/kumo-watchdog >> /var/log/kumo-watchdog.log 2>&1' | sudo tee /etc/cron.d/kumo-watchdog >/dev/null",
      "echo '*/5 * * * * root /usr/local/bin/kumo-autostop >> /var/log/kumo-autostop.log 2>&1' | sudo tee /etc/cron.d/kumo-autostop >/dev/null",
      "sudo cp /opt/kumo/deploy-shared/configs/logrotate-kumo-cron /etc/logrotate.d/kumo-cron",
      "echo '${var.kumo_image_tag}' | sudo tee /etc/kumo/image_tag",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable kumo-mount-nvme.service",
      "sudo rm -f /var/lib/cloud/sem/config_scripts_per_once.once",
      "sudo docker system prune -f",
      "sudo apt-get clean",
    ]
  }
}
