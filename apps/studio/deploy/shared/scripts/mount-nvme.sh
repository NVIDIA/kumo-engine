#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mount NVMe instance store SSD for diskgraph.
# Runs on every boot (NVMe is ephemeral — lost on stop/start).
#
# Cloud-agnostic: works on AWS g4dn (NVMe instance store), GCP local SSD,
# and Azure GPU VM temporary/resource disks.
#
# On g4dn.8xlarge: /dev/nvme1n1 is the 900GB instance store.

set -euo pipefail

MOUNT_POINT="/mnt/localssd"
MOUNT_UID=999   # kumouser in Docker containers
MOUNT_GID=999
DISCOVERY_RETRIES=5
DISCOVERY_SLEEP_SEC=1

# Already mounted?
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "$MOUNT_POINT already mounted"
    chown "$MOUNT_UID:$MOUNT_GID" "$MOUNT_POINT"
    exit 0
fi

# Azure Linux images normally format and mount the ephemeral resource disk at
# /mnt before this service runs. Keep that provider-owned mount intact and put
# DiskGraph beneath it instead of reformatting the backing device.
if [ -e /dev/disk/azure/resource-part1 ] && mountpoint -q /mnt 2>/dev/null; then
    echo "Using Azure resource disk mounted at /mnt"
    mkdir -p "$MOUNT_POINT"
    chown "$MOUNT_UID:$MOUNT_GID" "$MOUNT_POINT"
    chmod 1777 "$MOUNT_POINT"
    exit 0
fi

# Find NVMe instance store (not root EBS)
# Get the whole-disk device backing /. lsblk PKNAME handles both NVMe
# (/dev/nvme0n1p1 → nvme0n1) and traditional (/dev/sda1 → sda) names.
ROOT_DEVICE="/dev/$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)" | head -1)"
NVME_DEVICE=""

# Read the NVMe Model Number (the identify-controller MN field), using sysfs
# to avoid nvme-cli dependency.
read_nvme_model() {
    local model_file="/sys/block/${1#/dev/}/device/model"
    [ -r "$model_file" ] || return 1
    local model=""
    read -r model < "$model_file" || true
    [ -n "$model" ] || return 1
    printf '%s\n' "$model"
}

log_nvme_discovery_snapshot() {
    local dev root_match model
    echo "NVMe discovery snapshot:"
    lsblk -d -o NAME,MODEL
    for dev in /dev/nvme*n1; do
        [ -b "$dev" ] || continue
        if [ "$(readlink -f "$dev")" = "$(readlink -f "$ROOT_DEVICE")" ]; then
            root_match=yes
        else
            root_match=no
        fi
        model="$(read_nvme_model "$dev" || true)"
        echo "  candidate=$dev root=$root_match model=${model:-unknown}"
    done
}

find_nvme_instance_store() {
    local dev model
    for dev in /dev/nvme*n1; do
        [ -b "$dev" ] || continue
        [ "$(readlink -f "$dev")" = "$(readlink -f "$ROOT_DEVICE")" ] && continue
        model="$(read_nvme_model "$dev" || true)"
        case "$model" in
            # AWS instance store.
            "Amazon EC2 NVMe Instance Storage"*)
                echo "$dev"
                return 0
                ;;
            # GCP local SSD. Match exactly: persistent disks report
            # "nvme_card-pd" and must not be claimed as scratch space.
            "nvme_card")
                echo "$dev"
                return 0
                ;;
            # Azure: local NVMe on VM families that expose the temporary disk as
            # a raw NVMe namespace rather than a waagent-mounted resource disk.
            "Microsoft NVMe Direct Disk"*)
                echo "$dev"
                return 0
                ;;
        esac
    done
    return 1
}

for attempt in $(seq 1 "$DISCOVERY_RETRIES"); do
    if NVME_DEVICE="$(find_nvme_instance_store)"; then
        break
    fi
    if [ "$attempt" -lt "$DISCOVERY_RETRIES" ]; then
        echo "Waiting for NVMe instance store discovery (attempt ${attempt}/${DISCOVERY_RETRIES})..."
        sleep "$DISCOVERY_SLEEP_SEC"
    fi
done

if [ -z "$NVME_DEVICE" ]; then
    echo "WARNING: No NVMe instance store found. Creating $MOUNT_POINT on root disk."
    log_nvme_discovery_snapshot
    mkdir -p "$MOUNT_POINT"
    chown "$MOUNT_UID:$MOUNT_GID" "$MOUNT_POINT"
    exit 0
fi

echo "Found NVMe instance store: $NVME_DEVICE"

# Tear down DLAMI LVM if present (AWS DLAMI default)
if lvs 2>/dev/null | grep -q "lv_ephemeral"; then
    echo "Tearing down DLAMI LVM..."
    umount /opt/dlami/nvme 2>/dev/null || true
    lvremove -f vg.01/lv_ephemeral 2>/dev/null || true
    vgremove -f vg.01 2>/dev/null || true
    pvremove -f "$NVME_DEVICE" 2>/dev/null || true
fi

# Format as XFS if needed
if ! blkid -s TYPE "$NVME_DEVICE" | grep -q xfs; then
    echo "Formatting $NVME_DEVICE as XFS..."
    wipefs -a "$NVME_DEVICE"
    mkfs.xfs -f -l size=1g "$NVME_DEVICE"
fi

# Mount with diskgraph-optimized options
mkdir -p "$MOUNT_POINT"
mount -o noatime,nodiratime,allocsize=1m,logbufs=8,logbsize=256k "$NVME_DEVICE" "$MOUNT_POINT"
chown "$MOUNT_UID:$MOUNT_GID" "$MOUNT_POINT"
chmod 1777 "$MOUNT_POINT"

echo "NVMe mounted at $MOUNT_POINT ($(df -h "$MOUNT_POINT" | tail -1 | awk '{print $2}'))"
