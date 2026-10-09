#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds the kumo-glove-6b-50d wheel from Stanford's official glove.6B.zip.
#
#   build_wheel.sh OUT_DIR
#
# Without $PYTHON it builds in a venv with pinned dependencies (Python 3.12
# through uv when available, else python3 >= 3.11), so the wheel is
# reproducible; a given $PYTHON needs pip, numpy, pandas, pyarrow, filelock and
# setuptools>=77. The zip is downloaded to $GLOVE_ZIP unless already there.
set -euo pipefail

OUT="$(mkdir -p "${1:?usage: build_wheel.sh OUT_DIR}" && cd "$1" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ZIP_URL=https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
ZIP_SHA256=617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb
# Byte-identical to the glove_6B_50d_emb.npy Kumo has always shipped.
VECTORS_SHA256=2d85b9db2ad8ddeb2e07474af9c9fb45a925d21c40878d4b365d189f13eb5a15
# SHA-256 of the 400,000 words joined by newlines.
VOCAB_SHA256=029a7168ebc71f71e23e1f6ae94967234a13dcd64717754746c171be20d31a77
export SOURCE_DATE_EPOCH=315532800

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
zip="${GLOVE_ZIP:-${XDG_CACHE_HOME:-$HOME/.cache}/kumo/glove.6B.zip}"
if [ ! -f "$zip" ]; then
  curl -fsSL --retry 3 --create-dirs -o "$zip.part" "$ZIP_URL"
  mv "$zip.part" "$zip"
fi
echo "$ZIP_SHA256  $zip" | sha256sum --check --strict -

if [ -z "${PYTHON:-}" ]; then
  PYTHON="$work/venv/bin/python"
  pins=(pip numpy==2.5.3 pandas==3.0.6 pyarrow==25.0.1 filelock==4.0.1 setuptools==84.0.0)
  if command -v uv >/dev/null; then
    uv venv -q --python 3.12 "$work/venv"
    uv pip install -q --python "$PYTHON" "${pins[@]}"
  else
    python3 -m venv "$work/venv"
    "$PYTHON" -m pip install -q "${pins[@]}"
  fi
fi

"$PYTHON" "$REPO/scripts/gen_third_party.py" \
  --artifact glove-6b-50d-bundled-data --check "$HERE/THIRD-PARTY.txt"

cp -r "$HERE" "$work/pkg"
data="$work/pkg/src/kumo_glove_6b_50d/data"
mkdir -p "$data"
PYTHONPATH="$REPO/packages/kumo-core/src" "$PYTHON" - "$zip" "$work/cache" "$data" "$VOCAB_SHA256" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

from kumo_core.embedding_models import GloVeVectors

zip_path, cache, data, vocab_sha256 = sys.argv[1:]

model = GloVeVectors(zip_path, dim=50, cache_dir=cache)
model.vectors()
digest = hashlib.sha256("\n".join(model.vocabulary()).encode()).hexdigest()
assert digest == vocab_sha256, f"vocabulary digest {digest}"
for name in ("glove_6B_index.parquet", "glove_6B_50d_emb.npy"):
    shutil.copy(next(Path(cache).rglob(name)), data)
PY
(cd "$data" && echo "$VECTORS_SHA256  glove_6B_50d_emb.npy" | sha256sum --check --strict -)

"$PYTHON" -m pip wheel --no-deps --no-build-isolation -q -w "$work/dist" "$work/pkg"
"$PYTHON" "$REPO/scripts/check_python_license_metadata.py" --wheel-dir "$work/dist"
(cd "$work/dist" && for wheel in kumo_glove_6b_50d-*.whl; do sha256sum "$wheel" > "$wheel.sha256"; done)
cp "$work/dist/"* "$OUT/"
