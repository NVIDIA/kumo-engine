#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_python_wheel.sh --python <python> --out <dir> [--version <version>] [--zig]

Build the diskgraph Python wheel, including the standalone ingest binary.

Options:
  --python    Python interpreter to build for.
  --out       Output directory for the wheel.
  --version   Optional Python package version to stamp before building.
  --zig       Build a manylinux2014 Linux wheel and standalone ingest binary
              with Zig/cargo-zigbuild.

This script prepares an optional stamped source tree and invokes the root
Python package build. The package backend owns ingest binary staging so
downstream repos can install from a plain Git URL.
EOF
}

PYTHON_BIN=""
OUT_DIR=""
VERSION=""
USE_ZIG=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --zig)
      USE_ZIG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$PYTHON_BIN" ] || [ -z "$OUT_DIR" ]; then
  usage >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="$REPO_ROOT"
TEMP_DIR=""

cleanup() {
  if [ -n "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
  fi
}
trap cleanup EXIT

if [ -n "$VERSION" ]; then
  git -C "$REPO_ROOT" cat-file -e "HEAD^{commit}"
  TEMP_DIR="$(mktemp -d)"
  git -C "$REPO_ROOT" archive HEAD | tar -x -C "$TEMP_DIR"
  BUILD_ROOT="$TEMP_DIR"
fi

mkdir -p "$OUT_DIR"

if [ -n "$VERSION" ]; then
  "$PYTHON_BIN" - "$BUILD_ROOT/pyproject.toml" "$VERSION" <<'PYEOF'
import pathlib
import re
import sys

pyproject = pathlib.Path(sys.argv[1])
version = sys.argv[2]
text = pyproject.read_text()
text, count = re.subn(
    r'(?m)^version = "([^"]+)"$',
    f'version = "{version}"',
    text,
)
if count != 1:
    raise SystemExit(
        f"Expected to replace exactly one project version in {pyproject}, "
        f"replaced {count}"
    )
pyproject.write_text(text)
PYEOF
fi

if [ "$USE_ZIG" = "1" ] && ! cargo zigbuild --version >/dev/null 2>&1; then
  cargo install cargo-zigbuild --locked >&2
fi

DISKGRAPH_BUILD_ZIG="$USE_ZIG" "$PYTHON_BIN" -m pip wheel \
  --no-deps \
  --wheel-dir "$OUT_DIR" \
  "$BUILD_ROOT" >&2

WHEEL_PATH="$(find "$OUT_DIR" -maxdepth 1 -type f -name 'diskgraph-*.whl' | sort | tail -n 1)"
if [ -z "$WHEEL_PATH" ]; then
  echo "No diskgraph wheel found in $OUT_DIR" >&2
  exit 1
fi

if [ "$USE_ZIG" = "1" ] && ! basename "$WHEEL_PATH" | grep -q 'manylinux'; then
  echo "ERROR: expected a manylinux wheel from --zig, got $(basename "$WHEEL_PATH")" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m zipfile -l "$WHEEL_PATH" | grep -q 'diskgraph/bin/ingest'; then
  echo "ERROR: wheel does not contain diskgraph/bin/ingest" >&2
  "$PYTHON_BIN" -m zipfile -l "$WHEEL_PATH" | grep -E 'diskgraph/(bin|_c)' >&2 || true
  exit 1
fi

echo "$WHEEL_PATH"
