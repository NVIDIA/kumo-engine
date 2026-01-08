#!/usr/bin/env bash
# localssd-raid0-ext4-tui.sh
#
# Interactive setup for initializing cloud VMs with multiple attached local SSDs
# into a single ext4 volume tuned for heavy random reads.
# - Detects NVMe disks and prompts for selection.
# - Optional destructive reset of selected disks (unmount + wipe).
# - Creates RAID0 (mdadm) with a user-chosen chunk size.
# - Formats ext4 with matching stripe geometry and mounts it.
# - Does not persist RAID0 across reboots (ephemeral local SSDs).
#
# Usage:
#   sudo scripts/localssd-raid0-ext4-tui.sh
#
# Notes:
# - All selected disks are wiped.
# - For a single disk, the script skips RAID0 and formats directly.
# - Mount persistence is manual; this script does not modify /etc/fstab.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -- "$0" "$@"
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd lsblk
require_cmd findmnt
require_cmd mdadm
require_cmd mkfs.ext4
require_cmd wipefs

root_disk_name() {
  local root_src parent
  root_src="$(findmnt -n -o SOURCE /)"
  parent="$(lsblk -no PKNAME "$root_src")"
  if [[ -n "$parent" ]]; then
    echo "$parent"
  else
    echo "$(basename "$root_src")"
  fi
}

root_device_path() {
  local root_disk
  root_disk="$(root_disk_name)"
  readlink -f "/sys/block/$root_disk/device"
}

is_boot_disk() {
  local disk="$1"
  local root_path disk_path
  root_path="$(root_device_path)"
  disk_path="$(readlink -f "/sys/block/$(basename "$disk")/device")"
  [[ -n "$root_path" && "$disk_path" == "$root_path" ]]
}

list_nvme_disks() {
  local d
  while IFS= read -r d; do
    d="/dev/$d"
    if ! is_boot_disk "$d"; then
      echo "$d"
    fi
  done < <(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1 ~ /^nvme/ {print $1}')
}

disk_info() {
  local disk="$1"
  local size model mps sigs
  size="$(lsblk -dn -o SIZE "$disk")"
  model="$(lsblk -dn -o MODEL "$disk" | sed 's/[[:space:]]*$//')"
  mps="$(lsblk -n -o MOUNTPOINT "$disk" | awk 'NF' | paste -sd ',' -)"
  sigs="$(wipefs -n "$disk" 2>/dev/null | wc -l | tr -d ' ' || true)"
  printf "%s | %s | %s | mounts: %s | sigs: %s\n" "$disk" "$size" "${model:-unknown}" "${mps:-none}" "$sigs"
}

disk_in_use() {
  local disk="$1"
  local disk_name
  disk_name="$(basename "$disk")"
  if lsblk -n -o MOUNTPOINT "$disk" | awk 'NF{found=1} END{exit !found}'; then
    return 0
  fi
  if lsblk -nr -o NAME,TYPE,PKNAME | awk -v d="$disk_name" '$3==d && $2 ~ /^raid/ {found=1} END{exit !found}'; then
    return 0
  fi
  if wipefs -n "$disk" 2>/dev/null | awk 'NF{found=1} END{exit !found}'; then
    return 0
  fi
  return 1
}

reset_disk() {
  local disk="$1"
  local mp part md disk_name
  disk_name="$(basename "$disk")"
  while IFS= read -r md; do
    [[ -z "$md" ]] && continue
    while IFS= read -r mp; do
      [[ -n "$mp" ]] && umount "$mp"
    done < <(lsblk -nr -o MOUNTPOINT "/dev/$md" | awk 'NF')
    mdadm --stop "/dev/$md" >/dev/null 2>&1 || true
    mdadm --remove "/dev/$md" >/dev/null 2>&1 || true
  done < <(lsblk -nr -o NAME,TYPE,PKNAME | awk -v d="$disk_name" '$2=="raid0" && $3==d {print $1}')
  while IFS= read -r mp; do
    [[ -n "$mp" ]] && umount "$mp"
  done < <(lsblk -nr -o MOUNTPOINT "$disk" | awk 'NF')
  while IFS= read -r part; do
    wipefs -a "/dev/$part" >/dev/null 2>&1 || true
  done < <(lsblk -nr -o NAME "$disk" | awk 'NR>1')
  mdadm --zero-superblock --force "$disk" >/dev/null 2>&1 || true
  wipefs -a "$disk"
}

choose_disks() {
  local disks=("$@")
  local ans
  echo "Eligible NVMe disks:" >/dev/tty
  for d in "${disks[@]}"; do
    disk_info "$d" >/dev/tty
  done
  echo >/dev/tty
  printf "Use all listed disks for the volume? [Y/n] " >/dev/tty
  read -r ans </dev/tty
  case "${ans:-Y}" in
    y|Y) printf "%s\n" "${disks[@]}" ;;
    *) ;;
  esac
}

main() {
  mapfile -t disks < <(list_nvme_disks)
  if [[ "${#disks[@]}" -eq 0 ]]; then
    echo "No NVMe disks found."
    exit 1
  fi

  mapfile -t selected < <(choose_disks "${disks[@]}")
  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "No disks selected. Exiting."
    exit 1
  fi

  local in_use=()
  for d in "${selected[@]}"; do
    if disk_in_use "$d"; then
      in_use+=("$d")
    fi
  done

  if [[ "${#in_use[@]}" -gt 0 ]]; then
    echo "Selected disks appear in use or have signatures:"
    for d in "${in_use[@]}"; do
      disk_info "$d"
    done
    local ans
    read -r -p "Reset (unmount + wipe) these disks? This is destructive. [y/N] " ans
    case "${ans:-N}" in
      y|Y)
        for d in "${in_use[@]}"; do
          reset_disk "$d"
        done
        ;;
      *)
        echo "Aborting."
        exit 1
        ;;
    esac
  fi

  # Random-read oriented defaults.
  local chunk_k="1024"
  local mountpoint="/mnt/localssd"
  local stripe_cache="4096"
  local block_k="4"

  local target
  if [[ "${#selected[@]}" -eq 1 ]]; then
    target="${selected[0]}"
  else
    local md=""
    local i
    for i in $(seq 0 127); do
      if [[ ! -e "/dev/md$i" ]]; then
        md="/dev/md$i"
        break
      fi
    done
    if [[ -z "$md" ]]; then
      echo "No free /dev/mdX device found."
      exit 1
    fi
    echo "Creating RAID0 device $md from: ${selected[*]}"
    mdadm --create "$md" --level=0 --raid-devices="${#selected[@]}" --chunk="$chunk_k" "${selected[@]}"
    target="$md"
  fi

  for d in "${selected[@]}"; do
    if [[ -w "/sys/block/$(basename "$d")/queue/scheduler" ]]; then
      echo none > "/sys/block/$(basename "$d")/queue/scheduler" || true
    fi
  done

  if [[ "$target" == /dev/md* && -w "/sys/block/$(basename "$target")/md/stripe_cache_size" ]]; then
    echo "$stripe_cache" > "/sys/block/$(basename "$target")/md/stripe_cache_size" || true
  fi

  local mkfs_opts=(-F -b 4096 -E "lazy_itable_init=0,lazy_journal_init=0")
  if [[ "${#selected[@]}" -gt 1 ]]; then
    local stride=$((chunk_k / block_k))
    local stripe_width=$((stride * ${#selected[@]}))
    mkfs_opts=(-F -b 4096 -E "stride=${stride},stripe-width=${stripe_width},lazy_itable_init=0,lazy_journal_init=0")
  fi

  mkfs.ext4 "${mkfs_opts[@]}" "$target"

  mkdir -p "$mountpoint"
  if findmnt -n "$mountpoint" >/dev/null 2>&1; then
    umount "$mountpoint"
  fi
  if command -v udevadm >/dev/null 2>&1; then
    udevadm settle || true
  fi
  sync
  mount_opts="noatime,nodiratime"
  if ! mount -o "$mount_opts" "$target" "$mountpoint"; then
    echo "error: failed to mount $target on $mountpoint" >&2
    echo "Check: dmesg | tail -n 20" >&2
    exit 1
  fi
  chmod 0777 "$mountpoint"

  echo "Done."
  echo "  Target: $target"
  echo "  Disks: ${selected[*]}"
  echo "  Chunk: ${chunk_k}KiB"
  echo "  Block: ${block_k}KiB"
  if [[ "$target" == /dev/md* ]]; then
    echo "  Stripe cache: ${stripe_cache}"
  fi
  if [[ "${#selected[@]}" -gt 1 ]]; then
    echo "  Stride: ${stride}"
    echo "  Stripe width: ${stripe_width}"
  fi
  echo "  Mount opts: ${mount_opts}"
  echo "  Mount: $mountpoint"
}

main "$@"
