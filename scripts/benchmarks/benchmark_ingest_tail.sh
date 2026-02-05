#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

use_journal=1
if ! command -v journalctl >/dev/null 2>&1; then
  use_journal=0
fi

latest_benchmark_log() {
  local log
  log="$(ls -t "${REPO_ROOT}/benchmark/logs"/*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${log}" ]]; then
    echo "${log}"
  fi
}

bench_log="$(latest_benchmark_log)"
if [[ -z "${bench_log}" && ${use_journal} -eq 0 ]]; then
  echo "No logs found yet under ${REPO_ROOT}/benchmark/logs and journalctl is unavailable." >&2
  exit 1
fi

if (( use_journal == 1 )); then
  journalctl --user -u benchmark_ingest.service -n 200 -f &
  journal_pid=$!
  trap 'kill "${journal_pid}" 2>/dev/null || true' EXIT INT TERM
fi

if [[ -n "${bench_log}" ]]; then
  exec tail -n 200 -F "${bench_log}"
fi

wait
