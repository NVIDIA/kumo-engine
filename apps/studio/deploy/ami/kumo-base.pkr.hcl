# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packer template for kumo-base AMI (AWS variant).
#
# Builds a minimal Ubuntu 22.04 + NVIDIA driver + Docker +
# nvidia-container-toolkit AMI suitable as a base for any kumo
# container deployment. install-base.sh strips the DLAMI's unused host
# CUDA toolkit stack, so the downstream kumo-singlehost.pkr.hcl can source
# this AMI (instead of the AWS DLAMI) with a baked root of ~9 GB used
# instead of the DLAMI's ~49 GB.
#
# Cloud portability: the install-base.sh + smoke-test.sh scripts
# are cloud-agnostic. Future kumo-base-{gcp,azure}.pkr.hcl files
# would source the same scripts; only the cloud builder block
# differs.
#
# Usage:
#   packer build -var nvidia_driver_version=580-open kumo-base.pkr.hcl

packer {
  required_plugins {
    amazon = {
      version = ">= 1.2.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "instance_type" {
  type = string
  # g4dn.xlarge is enough for the smoke test and cheaper than larger.
  default = "g4dn.xlarge"
}

variable "nvidia_driver_version" {
  type        = string
  description = "NVIDIA server-edition driver major version (passed to apt-get install)"
  default     = "580-open"
}

# AWS credentials for AMI registration. Pass via env vars or -var flags.
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

source "amazon-ebs" "kumo-base" {
  ami_name      = "kumo-base-ubuntu22-{{timestamp}}"
  instance_type = var.instance_type
  region        = var.aws_region

  # AWS Deep Learning Base GPU AMI (Ubuntu 22.04): ships a prebuilt NVIDIA
  # driver whose version varies per snapshot (we have observed both 575 and
  # 595). install-base.sh keeps that prebuilt driver only when it is >= 580
  # (the CUDA-13 floor) and otherwise force-installs the pinned driver; then
  # smoke-test.sh fails the build if the driver is < 580.
  #
  # most_recent is intentional: each base rebuild inherits the latest
  # AWS-patched DLAMI (driver + OS security fixes); the >= 580 smoke guard
  # fails the build if a snapshot ever regresses below the CUDA-13 floor.
  source_ami_filter {
    filters = {
      name                = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
      architecture        = "x86_64"
    }
    most_recent = true
    owners      = ["amazon"]
  }

  ssh_username = "ubuntu"

  # Do not set volume_size: the root then defaults to the source DLAMI's root
  # snapshot size (75 GB today, the EC2 floor since a volume cannot be smaller
  # than its snapshot). This auto-tracks DLAMI root growth (it has grown 65 GB
  # -> 75 GB across snapshots) instead of a hardcoded size that would fail the
  # build with InvalidBlockDeviceMapping the day AWS ships a larger DLAMI root.
  # install-base.sh strips the DLAMI's ~40 GB host CUDA toolkit, so the baked
  # root settles at ~9 GB used regardless of the provisioned size.
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name        = "kumo-base-ubuntu22"
    Application = "kumo"
    Component   = "base"
    SourceDLAMI = "{{ .SourceAMI }}"
    # NvidiaDriverPin is the driver we force-install on a <580 DLAMI; the driver
    # actually shipped (the kept DLAMI driver on the common path) is recorded in
    # /etc/kumo/nvidia-driver-version and can vary per DLAMI snapshot.
    NvidiaDriverPin = var.nvidia_driver_version
  }
}

build {
  sources = ["source.amazon-ebs.kumo-base"]

  # 1. Cloud-agnostic provisioning: NVIDIA driver, Docker, toolkit.
  provisioner "shell" {
    environment_vars = [
      "NVIDIA_DRIVER_VERSION=${var.nvidia_driver_version}",
    ]
    script = "../shared/base-ami/install-base.sh"
  }

  # 2. Reboot to pick up the new NVIDIA kernel module.
  provisioner "shell" {
    inline            = ["sudo reboot"]
    expect_disconnect = true
  }

  # 3. Wait for instance back up + GPU smoke test.
  provisioner "shell" {
    pause_before = "60s"
    script       = "../shared/base-ami/smoke-test.sh"
  }
}
