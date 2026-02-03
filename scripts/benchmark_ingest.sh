#!/usr/bin/env bash
# Benchmark ingest runner
#
# What it does:
# - Iterates a pre-defined, ordered list of diskgraph_config.json files (config_list.txt).
# - Before each run:
#   - Pulls the latest code from origin/main (fast-forward only).
#   - Ensures the /mnt/localssd XFS volume exists; if not, it runs the
#     local SSD setup script in non-interactive mode to create/mount it.
#   - Resets the XFS filesystem to clear prior outputs.
# - Runs diskgraph-ingest (S3 input) for the config and logs output.
# - Records results and minimal state for resume after interruption.
#
# How to use:
# - Edit benchmark_configs/config_list.txt to include the configs you want, in order.
# - Run: ./scripts/benchmark_ingest.sh
# - Designed to be left running continuously; it resumes from the last
#   completed index in benchmark/benchmark_state.txt if restarted.
#
# Notes:
# - Config list is read from benchmark_configs/config_list.txt. When you add new configs,
#   commit that file update and the script will pick up the new list after the next git pull.
# - All logs, results, and state live under ./benchmark/ (git-ignored).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Pre-defined list of config JSONs (edit config_list.txt in-order).
CONFIG_LIST_FILE="${SCRIPT_DIR}/benchmark_configs/config_list.txt"
CONFIGS=()

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

load_configs() {
  CONFIGS=()
  if [[ -f "${CONFIG_LIST_FILE}" ]]; then
    while IFS= read -r line; do
      [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
      CONFIGS+=("${REPO_ROOT}/${line#./}")
    done < "${CONFIG_LIST_FILE}"
  fi
}

update_repo() {
  log "Updating repo (branch: ${GIT_BRANCH})..."
  ensure_clean_git
  git fetch --prune origin "${GIT_BRANCH}"
  git merge --ff-only "origin/${GIT_BRANCH}"
  load_configs
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
  load_configs
  if [[ ! -f "${RESULTS_FILE}" ]]; then
    printf "timestamp_utc\tindex\tconfig\tstatus\tgit_rev\tlog_file\n" > "${RESULTS_FILE}"
  fi

  if (( ${#CONFIGS[@]} == 0 )); then
    echo "No configs listed in CONFIGS; nothing to run." >&2
    exit 1
  fi

  local total="${#CONFIGS[@]}"
  while true; do
    local last_completed
    last_completed="$(load_state_index)"
    log "Last completed index: ${last_completed}"

    local failures=0
    local max_index=$((total - 1))
    if (( last_completed >= max_index )); then
      log "All configs completed; restarting from index 0."
      save_state_index "-1"
      last_completed="-1"
    fi

    local idx
    for idx in "${!CONFIGS[@]}"; do
      if (( idx <= last_completed )); then
        continue
      fi
      local config="${CONFIGS[$idx]}"
      if [[ ! -f "${config}" ]]; then
        log "Config missing: ${config}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${idx}" "${config}" "missing" "$(current_git_rev)" "-" >> "${RESULTS_FILE}"
        failures=$((failures + 1))
        continue
      fi

      update_repo
      local git_rev
      git_rev="$(current_git_rev)"

      reset_xfs_mount

      local log_file="${LOG_DIR}/$(basename "${config}").${git_rev}.$(date -u +%Y%m%dT%H%M%SZ).log"
      log "Running ingest (${idx}/${total}): ${config}"
      if run_ingest "${config}" "${log_file}"; then
        log "Success: ${config}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${idx}" "${config}" "ok" "${git_rev}" "${log_file}" >> "${RESULTS_FILE}"
        save_state_index "${idx}"
      else
        log "Failure: ${config} (see ${log_file})"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(timestamp_utc)" "${idx}" "${config}" "fail" "${git_rev}" "${log_file}" >> "${RESULTS_FILE}"
        failures=$((failures + 1))
      fi
    done

    if (( failures > 0 )); then
      log "Completed with ${failures} failure(s)."
    else
      log "All benchmark ingests completed successfully."
    fi
  done
}

main "$@"
