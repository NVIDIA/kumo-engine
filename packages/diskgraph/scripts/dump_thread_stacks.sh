#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Dump Rust thread stacks from a running process (default: diskgraph ingest).
#
# Usage (recommended with sudo):
#   sudo ./scripts/dump_thread_stacks.sh       # auto-detect ingest PID
#   sudo ./scripts/dump_thread_stacks.sh <pid> # explicit PID
#
# Notes:
# - Requires gdb.
# - If attach fails, run with sudo.
# - For readable Rust stacks in release, build with:
#   RUSTFLAGS="-C debuginfo=1 -C force-frame-pointers=yes" \
#   cargo build -p diskgraph-ingest --release

PID="${1:-}"

if [[ -z "$PID" ]]; then
  if command -v pgrep >/dev/null 2>&1; then
    PID="$(pgrep -n -f '(^|/| )ingest( |$)')"
  fi
  if [[ -z "${PID}" ]]; then
    echo "Usage: $0 <pid>"
    echo "No PID provided and could not auto-detect ingest process."
    exit 1
  fi
fi

if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  echo "PID must be a number"
  exit 1
fi

if [[ ! -d "/proc/$PID" ]]; then
  echo "PID $PID not found"
  exit 1
fi

if ! command -v gdb >/dev/null 2>&1; then
  echo "gdb not found; install gdb or use pstack/eu-stack"
  exit 1
fi

OUT_FILE="$(mktemp -t rust-stacks.XXXXXX.txt)"
chmod 0644 "$OUT_FILE"

gdb -p "$PID" -batch -ex "thread apply all bt" >"$OUT_FILE" 2>&1 || true

if grep -Eqi "could not attach|ptrace:" "$OUT_FILE"; then
  if grep -Eqi "ptrace_scope" "$OUT_FILE"; then
    echo "Failed to attach (ptrace_scope restriction)."
    if [[ -r /proc/sys/kernel/yama/ptrace_scope ]]; then
      echo "Current /proc/sys/kernel/yama/ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope)"
    fi
    echo "Try: sudo $0 $PID"
    echo "Or temporarily: sudo sysctl kernel.yama.ptrace_scope=0"
  else
    echo "Failed to collect stacks. Try: sudo $0 $PID"
  fi
  echo "Output saved to $OUT_FILE"
  exit 1
fi

echo "Saved stacks to $OUT_FILE"
echo
echo "Top frame summary (best-effort):"
awk '
  /^Thread / { in_thread=1; next }
  in_thread && /^#0 / {
    in_thread=0
    frame="unknown"
    line=$0
    if (sub(/^.* in /, "", line)) {
      split(line, parts, " ")
      if (length(parts[1]) > 0) {
        frame=parts[1]
      }
    }
    count[frame]++
  }
  END {
    for (f in count) {
      printf("  %-40s %d\n", f, count[f])
    }
  }
' "$OUT_FILE" | sort -k2,2nr -k1,1

echo
echo "Likely blocked threads (best-effort):"
awk '
  /^Thread / { in_thread=1; next }
  in_thread && /^#0 / {
    in_thread=0
    frame="unknown"
    line=$0
    if (sub(/^.* in /, "", line)) {
      split(line, parts, " ")
      if (length(parts[1]) > 0) {
        frame=parts[1]
      }
    }
    if (frame ~ /(futex|epoll_wait|poll|select|pthread_cond_wait|nanosleep|clock_nanosleep|read|write|open|stat|fsync|pread|pwrite|lseek|__lll_lock_wait|pthread_mutex_lock|sem_wait|accept|connect)/) {
      print "  " frame
      blocked[frame]++
    }
  }
  END {
    if (length(blocked) == 0) {
      print "  (no obvious blocking syscalls in #0 frames)"
    }
  }
' "$OUT_FILE"

echo
echo "Top frame summary (frames #0-2, best-effort):"
awk '
  /^Thread / { in_thread=1; next }
  in_thread && /^#([0-2]) / {
    frame="unknown"
    line=$0
    if (sub(/^.* in /, "", line)) {
      split(line, parts, " ")
      if (length(parts[1]) > 0) {
        frame=parts[1]
      }
    }
    count[frame]++
  }
  in_thread && /^#3 / { in_thread=0 }
  END {
    for (f in count) {
      printf("  %-40s %d\n", f, count[f])
    }
  }
' "$OUT_FILE" | sort -k2,2nr -k1,1

echo
echo "Likely blocked threads (frames #0-2, best-effort):"
awk '
  /^Thread / { in_thread=1; next }
  in_thread && /^#([0-2]) / {
    frame="unknown"
    line=$0
    if (sub(/^.* in /, "", line)) {
      split(line, parts, " ")
      if (length(parts[1]) > 0) {
        frame=parts[1]
      }
    }
    if (frame ~ /(futex|epoll_wait|poll|select|pthread_cond_wait|nanosleep|clock_nanosleep|read|write|open|stat|fsync|pread|pwrite|lseek|__lll_lock_wait|pthread_mutex_lock|sem_wait|accept|connect)/) {
      print "  " frame
      blocked[frame]++
    }
  }
  in_thread && /^#3 / { in_thread=0 }
  END {
    if (length(blocked) == 0) {
      print "  (no obvious blocking syscalls in frames #0-2)"
    }
  }
' "$OUT_FILE"

echo
echo "If you need richer symbols in release, rebuild with:"
echo "  RUSTFLAGS=\"-C debuginfo=1\" cargo build --release"
