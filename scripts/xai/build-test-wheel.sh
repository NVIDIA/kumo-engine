#!/usr/bin/env bash
# Build and validate a committed candidate without changing the prototype.
set -euo pipefail
ENGINE="$(git rev-parse --show-toplevel)"
PROTO="${XAI_PROTOTYPE:-/home/joliveira/kumo-work/supply-chain-xai-prototype}"
PY312="${XAI_PYTHON:-/home/joliveira/.local/share/mise/installs/python/3.12.13/bin/python3}"
RUST_BIN="${XAI_RUST_BIN:-/home/joliveira/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin}"
test "$(git -C "$ENGINE" branch --show-current)" = supply-chain-xai
git -C "$ENGINE" diff --exit-code
git -C "$ENGINE" diff --cached --exit-code
UNCOMMITTED_SOURCE="$(git -C "$ENGINE" ls-files --others --exclude-standard -- packages/kumo-core packages/kumo-engine scripts/xai)"
if test -n "$UNCOMMITTED_SOURCE"; then
    printf 'Uncommitted source must be reviewed and committed before building:\n%s\n' "$UNCOMMITTED_SOURCE" >&2
    exit 1
fi
REV="$(git -C "$ENGINE" rev-parse HEAD)"
RUN="$(mktemp -d /tmp/engine-xai-wheel.XXXXXX)"
printf '%s\n' "$RUN"
printf '%s\n' "$REV" > "$RUN/source-revision.txt"
git -C "$ENGINE" status --porcelain > "$RUN/source-status.txt"
git clone --no-hardlinks --no-checkout "$ENGINE" "$RUN/source"
git -C "$RUN/source" checkout --detach "$REV"
"$PY312" -m venv "$RUN/build-venv"
"$PY312" -m venv "$RUN/test-venv"
cp "$PROTO/build-requirements.txt" "$RUN/build-requirements.txt"
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$RUN/cache/pip"
export UV_CACHE_DIR="$RUN/cache/uv"
export CARGO_HOME="$RUN/cache/cargo"
export CARGO_TARGET_DIR="$RUN/cache/native-target"
export XDG_CACHE_HOME="$RUN/cache"
export CARGO_BUILD_JOBS=2
export PATH="$RUN/build-venv/bin:$RUN/cargo-tools/bin:$RUST_BIN:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME DISKGRAPH_INGEST_BIN
"$RUN/build-venv/bin/python" -m pip --isolated install --index-url https://pypi.org/simple -r "$RUN/build-requirements.txt"
cargo install cargo-zigbuild --version 0.24.1 --locked --root "$RUN/cargo-tools"
export PYTHON="$RUN/build-venv/bin/python"
export PIP_CONSTRAINT="$RUN/build-requirements.txt"
export KUMO_REPO="$RUN/source"
unset BUNDLE_VERSION OUTPUT_DIR
/usr/bin/time -v -o "$RUN/build.resources.txt" timeout 7200 bash "$RUN/source/packages/kumo-engine/scripts/build_wheel.sh" --output-dir "$RUN/wheels" > "$RUN/build.stdout.log" 2> "$RUN/build.stderr.log"
unset PIP_CONSTRAINT
set -- "$RUN"/wheels/kumo_engine-*.whl
test "$#" -eq 1
WHEEL="$1"
sha256sum "$WHEEL" > "$RUN/wheel.sha256"
"$RUN/build-venv/bin/uv" export --project "$PROTO" --frozen --no-emit-project --no-emit-package kumo-engine --no-hashes --output-file "$RUN/runtime-requirements.txt"
"$RUN/test-venv/bin/python" -m pip --isolated install --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
"$RUN/test-venv/bin/python" -m pip --isolated install --index-url https://pypi.org/simple -r "$RUN/runtime-requirements.txt" "$WHEEL"
"$RUN/test-venv/bin/python" -I -m pip --isolated check
"$RUN/test-venv/bin/python" -I -m pip freeze --all > "$RUN/installed-requirements.txt"
"$PYTHON" --version > "$RUN/python-version.txt"
rustc --version > "$RUN/rust-version.txt"
cargo --version > "$RUN/cargo-version.txt"
cargo-zigbuild --version > "$RUN/cargo-zigbuild-version.txt"
"$PYTHON" -m pip freeze --all > "$RUN/build-tool-versions.txt"
uname -a > "$RUN/platform.txt"
ldd --version > "$RUN/glibc-version.txt"
sha256sum "$RUN/build-requirements.txt" "$RUN/runtime-requirements.txt" "$RUN/installed-requirements.txt" "$RUN/build-tool-versions.txt" > "$RUN/inputs.sha256"
mkdir "$RUN/acceptance"
cp "$RUN/source/packages/kumo-engine/test/wheel_xai_smoke.py" "$RUN/acceptance/runner.py"
cp "$RUN/source/scripts/xai/walk_native_gate.py" "$RUN/acceptance/walk_native_gate.py"
mkdir "$RUN/acceptance/tests"
for FILE in packages/kumo-core/tests/test_local_sensitivity.py packages/kumo-core/tests/test_xai_capture_validation.py packages/kumo-core/tests/test_prediction_explanations.py packages/kumo-core/tests/test_record_removal.py packages/kumo-core/tests/explain/test_subgraph_collector_keys.py packages/kumo-engine/test/test_local_prediction_sensitivity.py packages/kumo-engine/test/test_prediction_output_columns.py packages/kumo-engine/test/test_prediction_evidence_bundle.py packages/kumo-engine/test/test_xai_integration_validation.py packages/kumo-engine/test/test_uncertainty.py packages/kumo-engine/test/test_xai_interval_validation.py scripts/xai/test_streaming_resources.py; do
    cp "$RUN/source/$FILE" "$RUN/acceptance/tests/"
done
for FILE in packages/kumo-core/tests/test_walk_attribution.py packages/kumo-engine/test/test_walk_evidence.py scripts/xai/test_walk_reference_gate.py; do
    cp "$RUN/source/$FILE" "$RUN/acceptance/tests/"
done
cd "$RUN/acceptance"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v -o "$RUN/unit.resources.txt" timeout 1800 "$RUN/test-venv/bin/python" -I -m pytest tests --import-mode=prepend -p no:cacheprovider -q -s --junitxml="$RUN/unit.xml" > "$RUN/unit.log" 2>&1
/usr/bin/time -v -o "$RUN/native.resources.txt" timeout 1800 "$RUN/test-venv/bin/python" -I runner.py --suite native --wheel "$WHEEL" --source-revision "$REV" --output "$RUN/results/native" > "$RUN/native.log" 2>&1
/usr/bin/time -v -o "$RUN/walk-native.resources.txt" timeout 1800 "$RUN/test-venv/bin/python" -I walk_native_gate.py --wheel "$WHEEL" --output "$RUN/results/walk-native" > "$RUN/walk-native.log" 2>&1
/usr/bin/time -v -o "$RUN/supply-chain.resources.txt" timeout 3600 "$RUN/test-venv/bin/python" -I runner.py --suite supply-chain --wheel "$WHEEL" --source-revision "$REV" --prototype-root "$PROTO" --output "$RUN/results/supply-chain" > "$RUN/supply-chain.log" 2>&1
sha256sum "$RUN"/*.log "$RUN"/*.resources.txt "$RUN"/results/*/acceptance.json > "$RUN/acceptance-evidence.sha256"
printf 'Candidate accepted: %s\n' "$RUN"
