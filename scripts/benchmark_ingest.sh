#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Pre-defined list of config JSONs (edit this list in-order).
CONFIGS=(
  "${SCRIPT_DIR}/benchmark_configs/alt_detection_5pct.json"
)

GIT_BRANCH="main"
MOUNTPOINT="/mnt/localssd"
BENCHMARK_ROOT="${REPO_ROOT}/benchmark"
STATE_FILE="${BENCHMARK_ROOT}/benchmark_state.txt"
RESULTS_FILE="${BENCHMARK_ROOT}/benchmark_results.tsv"
LOG_DIR="${BENCHMARK_ROOT}/logs"

ALLOW_DIRTY="${BENCHMARK_ALLOW_DIRTY:-0}"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf "[%s] %s\n" "$(timestamp_utc)" "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

load_state_index() {
  if [[ -f "${STATE_FILE}" ]]; then
    local idx
    idx="$(tr -d '[:space:]' < "${STATE_FILE}" || true)"
    if [[ "${idx}" =~ ^[0-9]+$ ]]; then
      echo "${idx}"
      return 0
    fi
  fi
  echo "-1"
}

save_state_index() {
  local idx="$1"
  printf "%s\n" "${idx}" > "${STATE_FILE}"
}

ensure_clean_git() {
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    return 0
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Git working tree is dirty. Commit/stash changes or set BENCHMARK_ALLOW_DIRTY=1." >&2
    exit 1
  fi
}

update_repo() {
  log "Updating repo (branch: ${GIT_BRANCH})..."
  ensure_clean_git
  git fetch --prune origin "${GIT_BRANCH}"
  git merge --ff-only "origin/${GIT_BRANCH}"
}

current_git_rev() {
  git rev-parse --short HEAD
}

reset_xfs_mount() {
  require_cmd findmnt
  require_cmd mkfs.xfs
  require_cmd mount
  require_cmd umount

  local source fstype
  source="$(findmnt -n -o SOURCE -- "${MOUNTPOINT}" || true)"
  fstype="$(findmnt -n -o FSTYPE -- "${MOUNTPOINT}" || true)"
  if [[ -z "${source}" || -z "${fstype}" ]]; then
    log "Mountpoint ${MOUNTPOINT} is not mounted; invoking localssd-raid0-xfs-tui.sh -y..."
    sudo "${REPO_ROOT}/scripts/localssd-raid0-xfs-tui.sh" -y
    source="$(findmnt -n -o SOURCE -- "${MOUNTPOINT}" || true)"
    fstype="$(findmnt -n -o FSTYPE -- "${MOUNTPOINT}" || true)"
    if [[ -z "${source}" || -z "${fstype}" ]]; then
      echo "Mountpoint ${MOUNTPOINT} still not mounted after setup; cannot reset." >&2
      exit 1
    fi
  fi
  if [[ "${fstype}" != "xfs" ]]; then
    echo "Mountpoint ${MOUNTPOINT} is not XFS (fstype=${fstype}); refusing to reset." >&2
    exit 1
  fi

  log "Resetting XFS mount ${MOUNTPOINT} on ${source}..."
  sudo umount "${MOUNTPOINT}"
  sudo mkfs.xfs -f "${source}"
  sudo mount -o noatime,nodiratime,allocsize=1m,logbufs=8,logbsize=256k "${source}" "${MOUNTPOINT}"
  sudo chmod 1777 "${MOUNTPOINT}"
}

run_ingest() {
  local config="$1"
  local log_file="$2"
  "${REPO_ROOT}/diskgraph-ingest/scripts/ingest_s3.sh" "${config}" 2>&1 | tee "${log_file}"
}

main() {
  require_cmd git
  require_cmd date

  mkdir -p "${LOG_DIR}"
  if [[ ! -f "${RESULTS_FILE}" ]]; then
    printf "timestamp_utc\tindex\tconfig\tstatus\tgit_rev\tlog_file\n" > "${RESULTS_FILE}"
  fi

  local last_completed
  last_completed="$(load_state_index)"
  log "Last completed index: ${last_completed}"

  local failures=0
  local total="${#CONFIGS[@]}"
  local i
  for i in "${!CONFIGS[@]}"; do
    if (( i <= last_completed )); then
      continue
    fi
    local config="${CONFIGS[$i]}"
    if [[ ! -f "${config}" ]]; then
      log "Config missing: ${config}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${i}" "${config}" "missing" "$(current_git_rev)" "-" >> "${RESULTS_FILE}"
      failures=$((failures + 1))
      continue
    fi

    update_repo
    local git_rev
    git_rev="$(current_git_rev)"

    reset_xfs_mount

    local log_file="${LOG_DIR}/$(basename "${config}").${git_rev}.$(date -u +%Y%m%dT%H%M%SZ).log"
    log "Running ingest (${i}/${total}): ${config}"
    if run_ingest "${config}" "${log_file}"; then
      log "Success: ${config}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${i}" "${config}" "ok" "${git_rev}" "${log_file}" >> "${RESULTS_FILE}"
      save_state_index "${i}"
    else
      log "Failure: ${config} (see ${log_file})"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${i}" "${config}" "fail" "${git_rev}" "${log_file}" >> "${RESULTS_FILE}"
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    log "Completed with ${failures} failure(s)."
    exit 1
  fi
  log "All benchmark ingests completed successfully."
}

main "$@"
