#!/bin/bash -ex
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Cloud-agnostic base provisioning for kumo container hosts.
#
# Runs on plain Ubuntu 22.04 LTS sources from any major CSP (Canonical
# publishes the same image family on AWS, GCP, Azure). Installs the
# minimum required to host kumo's container stack:
#
#   1. NVIDIA driver         — kernel module + user-space libs
#   2. Docker engine         — official docker.com installer
#   3. nvidia-container-toolkit — Docker GPU runtime hook
#   4. kumo standard tools   — python3-venv, xfsprogs, nvme-cli, awscli
#
# Notably NOT used on the host: CUDA toolkit, cuDNN, NCCL. Those live in
# the kumo-backend-gpu container image; the host only needs the driver +
# nvidia-container-toolkit. The AWS DLAMI ships ~40 GB of host CUDA
# toolkits under /usr/local/cuda-* regardless, so step 5 strips them and
# the baked root settles at ~9 GB used instead of ~49 GB.
#
# Pin NVIDIA_DRIVER_VERSION to a specific server-edition major version
# for ABI compatibility with the CUDA version in kumo-backend-gpu.
# Bump deliberately when the image's CUDA version changes.

# Driver branch from NVIDIA's CUDA keyring repo (open kernel module
# variant, recommended for Linux >= 5.x). Bump deliberately when CUDA
# version in kumo-backend-gpu image changes.
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-580-open}"

# Wait for cloud-init to finish its boot-time apt operations. Without
# this, Packer's SSH can connect mid-boot and our apt-get update races
# with cloud-init's, leaving /var/lib/apt/lists in an inconsistent
# state ("fopen: No such file or directory" mid-read).
sudo cloud-init status --wait

# Stop unattended-upgrades to avoid dpkg-lock races during Packer-driven
# apt operations. Mask so it doesn't auto-restart.
sudo systemctl stop unattended-upgrades || true
sudo systemctl mask unattended-upgrades || true
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "Waiting for residual apt lock..."
    sleep 5
done

sudo apt-get update

# Freeze the kernel + NVIDIA package set. Filter to installed packages
# only: the broad globs can also match virtual / not-installed names in
# the dpkg db (Provides aliases, leftover `rc` stanzas), which apt-mark
# can't hold and would fail this `set -e` script.
hold_kernel_nvidia() {
  local held
  held="$(dpkg-query -W -f='${db:Status-Status} ${binary:Package}\n' \
    linux-aws linux-image-aws linux-headers-aws \
    'linux-image-*-aws' 'linux-headers-*-aws' 'linux-modules-*-aws' \
    'nvidia-driver-*' 'nvidia-dkms-*' 'nvidia-kernel-*' \
    'nvidia-utils-*' 'nvidia-compute-utils-*' 'libnvidia-*' \
    'xserver-xorg-video-nvidia-*' 2>/dev/null \
    | awk '$1 == "installed" { print $2 }' || true)"
  if [ -n "${held}" ]; then
    echo "${held}" | xargs -r sudo apt-mark hold
  fi
}
# Hold the DLAMI's kernel + driver before the general upgrade so `upgrade`
# cannot bump them out from under the smoke-validated set.
hold_kernel_nvidia
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# 1. NVIDIA driver.
#    On AWS we source the Deep Learning Base GPU AMI, which ships a prebuilt
#    driver – but the exact version drifts per DLAMI snapshot (we have observed
#    both 575 and 595 within days), and CUDA 13 / Blackwell (sm_120) needs
#    >= 580: a 575 driver silently falls back to CPU on Blackwell. So keep the
#    DLAMI's prebuilt driver only when it is already >= 580 (no DKMS needed);
#    otherwise force-install the pinned open driver from NVIDIA's CUDA keyring
#    repo (DKMS-compiles against the booted kernel). smoke-test.sh re-asserts
#    the >= 580 floor so a bad snapshot fails the build instead of shipping.
DRIVER_FLOOR=580
cur_major=""
if command -v nvidia-smi >/dev/null 2>&1; then
    cur_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
      | head -1 | cut -d. -f1)
fi
if [ -n "${cur_major}" ] && [ "${cur_major}" -ge "${DRIVER_FLOOR}" ] 2>/dev/null; then
    echo "NVIDIA driver ${cur_major}.x present (>= ${DRIVER_FLOOR}); using the DLAMI's prebuilt driver."
else
    echo "NVIDIA driver missing or below ${DRIVER_FLOOR} (found '${cur_major:-none}'); installing nvidia-driver-${NVIDIA_DRIVER_VERSION}."
    # A <580 DLAMI ships an older driver that is often apt-mark held; unhold the
    # NVIDIA + kernel packages so the pinned driver can replace it (set -e safe).
    held_nv="$(apt-mark showhold 2>/dev/null | grep -E '^(nvidia-|libnvidia-|linux-(image|headers|modules)-)' || true)"
    if [ -n "${held_nv}" ]; then
        echo "${held_nv}" | xargs -r sudo apt-mark unhold
    fi
    # shellcheck disable=SC1091  # /etc/os-release only exists on the build host
    DISTRIBUTION=$(. /etc/os-release; echo "$ID$VERSION_ID" | tr -d '.')
    wget -qO /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRIBUTION}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
    sudo apt-get update

    sudo apt-get install -y \
        "linux-headers-$(uname -r)" \
        "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
    # Re-freeze the newly installed kernel + driver set (the branch unheld it
    # above so the pinned driver could replace the DLAMI's).
    hold_kernel_nvidia
fi

# 2. Docker (use docker.com's installer for the latest stable)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu

# 3. nvidia-container-toolkit. The DLAMI already ships it; only install on a
#    base that lacks it. (`gpg --dearmor` to an existing keyring on the DLAMI
#    prompts to overwrite and fails on the TTY-less packer session, so we skip
#    rather than re-run it.)
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "nvidia-container-toolkit already present (nvidia-ctk found); skipping install."
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
fi
# Wire docker -> nvidia runtime (idempotent; docker is (re)installed above).
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Kumo standard tools (referenced by kumo-singlehost provisioner)
sudo apt-get install -y \
    python3.10-venv \
    xfsprogs \
    nvme-cli \
    awscli

# 5. Shrink the baked AMI.
#    (a) Strip the DLAMI's unused host CUDA toolkit. The Deep Learning Base
#        AMI lays down several full CUDA toolkits under /usr/local/cuda-*
#        (runfile installs, not dpkg-managed; ~40 GB across versions). Kumo
#        never uses a host CUDA toolkit – its CUDA ships in the
#        kumo-backend-gpu container image, and the host only needs the driver
#        + nvidia-container-toolkit (both live in /usr/bin + /usr/lib and the
#        kernel-module tree, untouched here). This drops the baked root from
#        ~49 GB used to ~9 GB. No-op on a base without these dirs.
if ls -d /usr/local/cuda /usr/local/cuda-* >/dev/null 2>&1; then
    echo "Stripping unused host CUDA toolkit:"
    sudo du -shc /usr/local/cuda /usr/local/cuda-* 2>/dev/null | tail -1 || true
fi
sudo rm -rf /usr/local/cuda /usr/local/cuda-*
# Drop stale CUDA entries from the dynamic linker so ldconfig doesn't warn
# about the removed toolkit (harmless if the file is absent).
sudo rm -f /etc/ld.so.conf.d/*cuda*.conf
# Drop DLAMI-provided PATH/LD_LIBRARY_PATH exports for the removed toolkit
# (harmless if absent).
sudo rm -f /etc/profile.d/*cuda*.sh
sudo ldconfig

#    (b) Clear apt cache + lists.
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
