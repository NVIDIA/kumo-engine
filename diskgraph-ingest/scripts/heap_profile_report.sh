#!/usr/bin/env bash
set -euo pipefail

# Generate jemalloc heap profile reports (text + SVG).
#
# Usage:
#   ./diskgraph-ingest/scripts/heap_profile_report.sh /path/to/heap-123.heap
#   ./diskgraph-ingest/scripts/heap_profile_report.sh /path/to/heap-123.heap /path/to/ingest
#
# Outputs:
#   heap-123.txt
#   heap-123.svg

HEAP_FILE="${1:-}"
INGEST_BIN="${2:-./target/release/ingest}"

if [[ -z "${HEAP_FILE}" ]]; then
  echo "Usage: $0 /path/to/heap-*.heap [/path/to/ingest]" >&2
  exit 1
fi

if [[ ! -f "${HEAP_FILE}" ]]; then
  echo "heap file not found: ${HEAP_FILE}" >&2
  exit 1
fi

if [[ ! -x "${INGEST_BIN}" ]]; then
  echo "ingest binary not executable: ${INGEST_BIN}" >&2
  exit 1
fi

JEPROF="${JEPROF:-}"
if [[ -z "${JEPROF}" ]]; then
  if command -v jeprof >/dev/null 2>&1; then
    JEPROF="jeprof"
  else
    JEPROF="$(find ./target -name jeprof -type f 2>/dev/null | head -n 1)"
  fi
fi

if [[ -z "${JEPROF}" ]]; then
  echo "jeprof not found; set JEPROF=/path/to/jeprof" >&2
  exit 1
fi

if [[ ! -x "${JEPROF}" ]]; then
  chmod +x "${JEPROF}" || true
fi

BASE="${HEAP_FILE%.heap}"
TXT_OUT="${BASE}.txt"
SVG_OUT="${BASE}.svg"

HAS_DOT="true"
if ! command -v dot >/dev/null 2>&1; then
  HAS_DOT="false"
  echo "warning: graphviz 'dot' not found; SVG may be incomplete." >&2
  echo "install:" >&2
  echo "  macOS: brew install graphviz" >&2
  echo "  Ubuntu/Debian: sudo apt-get install graphviz" >&2
  echo "  Fedora: sudo dnf install graphviz" >&2
  echo "  Arch: sudo pacman -S graphviz" >&2
fi

"${JEPROF}" --show_bytes --lines "${INGEST_BIN}" "${HEAP_FILE}" > "${TXT_OUT}"
if [[ "${HAS_DOT}" == "true" ]]; then
  "${JEPROF}" --show_bytes --lines --svg "${INGEST_BIN}" "${HEAP_FILE}" > "${SVG_OUT}"
fi

echo "Wrote ${TXT_OUT}"
if [[ "${HAS_DOT}" == "true" ]]; then
  echo "Wrote ${SVG_OUT}"
else
  echo "Skipped ${SVG_OUT} (graphviz 'dot' missing)"
fi
