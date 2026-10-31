#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the one published `kumo-engine` wheel: the local training closure plus
# every provider package, merged into a single native-platform wheel.
#
#     pip install kumo-engine                 # engine + provider *code*
#     pip install 'kumo-engine[snowflake]'    # ... + snowflake-connector, ...
#
# Provider packages (kumo-snowflake, kumo-databricks, ...) are always bundled,
# but their third-party dependencies are only required by the matching extra
# (see scripts/merge_wheels.py), so `kumo.engine.FileConnector` works on a bare
# install and `SnowflakeConnector` raises ImportError until the extra is added.
#
# Versioning: the root VERSION file is the source of truth, shared with kumo-sdk,
# the platform and the charts (scripts/stamp_version.py copies it into
# kumo.engine._version, which the wheel's metadata reads). A plain run stamps `<version>+<short commit>`
# (a local version, which PyPI rejects, so dev builds cannot be published by
# accident); `--release` stamps exactly `<version>`. CI only passes --release
# for a published GitHub release tagged `kumo-engine-v<version>`.
#
# Wheel tag: DiskGraph is the only native component and is a stable-ABI
# (abi3) extension, so one wheel serves every CPython at or above its floor,
# currently 3.11. The merger intersects every component Requires-Python
# constraint for the resulting wheel. Linux uses manylinux_<MAX_GLIBC>,
# backed by check_glibc_floor. macOS inherits DiskGraph's native platform tag.
#
# Bundled first-party and native components:
#
#     every            ${KUMO_REPO}/packages/* -- kumo-engine, kumo-core,
#     packages/ dist   kumo-api, kumo-pql, kumo-snowflake, kumo-s3,
#                      kumo-databricks, kumo-bigquery, kumo-gcs, ... every
#                      directory with a pyproject.toml or setup.py, EXCEPT
#                      kumo-sdk: it ships the hosted `kumoai_studio` SDK,
#                      and this wheel is the local-only closure.
#     diskgraph        packages/diskgraph -- native abi3 extension plus the
#                      standalone `ingest` binary
#
# kumo-api, kumo-pql, kumo-core and every provider are always bundled.
# kumo-glove-6b-50d stays a PyPI dependency of the `text` extra.
#
# Everything else (torch, torch-geometric, pytorch-frame, pytorch-lightning,
# tensorboard, optuna, ...) stays a normal Requires-Dist that pip resolves
# from PyPI at install time; --offline-report lists exactly what that is.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

# --------------------------------------------------------------------------
# Defaults. Every one of these is overridable by flag or environment.
# --------------------------------------------------------------------------
KUMO_REPO="${KUMO_REPO:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
BUNDLE_NAME="kumo-engine"
BUNDLE_VERSION="${BUNDLE_VERSION:-}"
RELEASE=0
# Build DiskGraph through cargo-zigbuild so the extension and the ingest binary
# link against a manylinux2014 glibc rather than this host's. A native build
# links against the build host's glibc and will not load on the runtime.
HOST_OS="$(uname -s)"
if [ "${HOST_OS}" = "Darwin" ]; then
    USE_ZIG="${USE_ZIG:-0}"
    # Matches the minimum of the macOS ARM64 Faiss dependency. Both the
    # extension and standalone ingest must use the same deployment target.
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
else
    USE_ZIG="${USE_ZIG:-1}"
fi
# Highest glibc the DiskGraph binaries may require, and the manylinux tag the
# wheel claims. Container Runtime is Ubuntu-based and provides 2.35, so 2.34
# leaves headroom while still catching a binary accidentally linked against a
# modern build host. See check_glibc_floor.
MAX_GLIBC="${MAX_GLIBC:-2.34}"
PLATFORM_TAG="${PLATFORM_TAG:-}"
if [ "${HOST_OS}" != "Darwin" ]; then
    : "${PLATFORM_TAG:=manylinux_${MAX_GLIBC//./_}_x86_64}"
fi
VERIFY_IMAGE="${VERIFY_IMAGE:-}"
THIRD_PARTY_NOTICE="${THIRD_PARTY_NOTICE:-}"
GENERATE_THIRD_PARTY_NOTICE=0
OFFLINE_REPORT=0
KEEP_WORK=0

usage() {
    awk 'NR >= 6 && /^#/ { sub(/^# ?/, ""); print; next } NR >= 6 { exit }' \
        "${BASH_SOURCE[0]}"
    cat <<'EOF'

Usage: scripts/build_wheel.sh [options]

Options:
  --output-dir DIR      Where the wheel is written.       [dist/]
  --release             Stamp exactly the VERSION file's version, for publishing.
                        Without it: <version>+<short commit>.
  --version VERSION     Override the version entirely (PEP 440).
  --platform-tag TAG    Wheel platform tag. [manylinux_2_34_x86_64 on Linux;
                        native DiskGraph tag on macOS]
  --kumo-repo PATH      kumo monorepo checkout.           [this repo]
  --no-zig              Build DiskGraph natively instead of through
                        cargo-zigbuild. Default on macOS. On Linux, the host's
                        glibc must be no newer than the runtime's.
  --verify-image IMAGE  After building, install the wheel into IMAGE with
                        Docker and import the closure as a smoke test.
  --offline-report      Print the requirements pip must still resolve at
                        install time.
  --keep-work           Leave the scratch directory in place for inspection.
  -h, --help            This message.

Environment:
  MACOSX_DEPLOYMENT_TARGET  Minimum macOS version for both native binaries.
                           Must be >= 14.0. [14.0]
  MAX_GLIBC             Highest glibc the DiskGraph binaries may require.
                        [2.34; Container Runtime provides 2.35]
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --release) RELEASE=1; shift ;;
        --version) BUNDLE_VERSION="$2"; shift 2 ;;
        --platform-tag) PLATFORM_TAG="$2"; shift 2 ;;
        --kumo-repo) KUMO_REPO="$2"; shift 2 ;;
        --no-zig) USE_ZIG=0; shift ;;
        --verify-image) VERIFY_IMAGE="$2"; shift 2 ;;
        --offline-report) OFFLINE_REPORT=1; shift ;;
        --keep-work) KEEP_WORK=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

LOCAL_ROOT="${KUMO_REPO}/packages/kumo-engine"
: "${OUTPUT_DIR:=${LOCAL_ROOT}/dist}"
DISKGRAPH_SOURCE="${KUMO_REPO}/packages/diskgraph"

log() { echo "[wheel] $*" >&2; }
die() { echo "[wheel] ERROR: $*" >&2; exit 1; }

if [ "${HOST_OS}" = "Darwin" ]; then
    if ! [[ "${MACOSX_DEPLOYMENT_TARGET}" =~ ^[0-9]+(\.[0-9]+)*$ ]] \
        || ! [ "${MACOSX_DEPLOYMENT_TARGET%%.*}" -ge 14 ]; then
        die "macOS wheels require MACOSX_DEPLOYMENT_TARGET >= 14.0 (got ${MACOSX_DEPLOYMENT_TARGET})"
    fi
fi

if [ "${HOST_OS}" = "Darwin" ] && [ "${USE_ZIG}" != "0" ]; then
    die "macOS wheels require a native DiskGraph build (USE_ZIG=0)"
fi

# The build host's interpreter only drives pip and the merge. Every pure-Python
# component is built with `pip wheel`; DiskGraph's wheel tag comes from its
# abi3 build config, not from this interpreter.
PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || die "no ${PYTHON} on PATH"
"${PYTHON}" -m pip --version >/dev/null 2>&1 \
    || die "${PYTHON} has no pip; set PYTHON to an interpreter that does\
 (a virtualenv works)"

[ -d "${KUMO_REPO}/packages" ] \
    || die "no kumo monorepo at ${KUMO_REPO} (pass --kumo-repo)"
[ -d "${LOCAL_ROOT}" ] || die "${KUMO_REPO} has no packages/kumo-engine"
FIRST_PARTY_LICENSE="${KUMO_REPO}/LICENSE"
[ -f "${FIRST_PARTY_LICENSE}" ] \
    || die "no first-party license at ${FIRST_PARTY_LICENSE}"
if [ -z "${THIRD_PARTY_NOTICE}" ]; then
    THIRD_PARTY_NOTICE="${KUMO_REPO}/THIRD-PARTY.txt"
    GENERATE_THIRD_PARTY_NOTICE=1
fi
[ -f "${THIRD_PARTY_NOTICE}" ] \
    || die "no third-party notice at ${THIRD_PARTY_NOTICE}\
 (set THIRD_PARTY_NOTICE)"
if [ "${GENERATE_THIRD_PARTY_NOTICE}" = "1" ] \
    && [ "$(cd "${KUMO_REPO}" && pwd -P)" != "${REPO_ROOT}" ]; then
    die "automatic notice generation requires this script's checkout; run the script from ${KUMO_REPO} or set THIRD_PARTY_NOTICE to a reviewed notice"
fi

KUMO_REPO_HEAD="$(git -C "${KUMO_REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -z "${BUNDLE_VERSION}" ]; then
    # The root VERSION file is the release version. A
    # non-release build appends the commit as a PEP 440 local version, so it
    # sorts above the release it was cut from but can never be uploaded to
    # PyPI, which rejects local versions.
    [ -f "${KUMO_REPO}/VERSION" ] || die "no VERSION file at ${KUMO_REPO}/VERSION"
    BUNDLE_VERSION="$(tr -d '[:space:]' < "${KUMO_REPO}/VERSION")"
    if [ "${RELEASE}" != "1" ]; then
        BUNDLE_VERSION="${BUNDLE_VERSION}+${KUMO_REPO_HEAD}"
    fi
fi

if [ "${RELEASE}" = "1" ] && [[ "${BUNDLE_VERSION}" == *+* ]]; then
    die "--release needs a public version, got ${BUNDLE_VERSION}"
fi

WORK="$(mktemp -d -t kumo-wheel-XXXXXX)"
cleanup() {
    if [ "${KEEP_WORK}" = "1" ]; then
        log "scratch kept at ${WORK}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup EXIT

WHEELS="${WORK}/wheels"
mkdir -p "${WHEELS}"
# Provenance, one `<wheel filename>\t<classification>\t<origin>` row per
# component. The merge step turns it into kumo_bundle_manifest.json, and
# scripts/gen_third_party.py checks the classifications.
COMPONENTS="${WORK}/components.tsv"
: > "${COMPONENTS}"

# Record the newest wheel matching a glob as a component. Fails if the build
# step that was supposed to produce it did not.
record() {
    local pattern="$1" classification="$2" origin="$3" found
    case "${classification}" in
        kumo-first-party|diskgraph-native|third-party) ;;
        *) die "unknown component classification: ${classification}" ;;
    esac
    # shellcheck disable=SC2012  # names are wheel filenames, no odd characters
    found="$(ls -1 "${WHEELS}"/${pattern} 2>/dev/null | sort | tail -1)"
    [ -n "${found}" ] || die "expected a wheel matching ${pattern} in ${WHEELS}"
    printf '%s\t%s\t%s\n' "$(basename "${found}")" "${classification}" \
        "${origin}" >> "${COMPONENTS}"
}

# --------------------------------------------------------------------------
# 1) The first-party pure-Python distributions: every packages/* directory
#    except kumo-sdk (the hosted `kumoai_studio` SDK, not part of the local
#    closure) and the native DiskGraph build below. A new provider package
#    ships the moment it lands under packages/, but merge_wheels.py refuses
#    to build until kumo-engine declares an extra that reaches it.
# --------------------------------------------------------------------------
KUMO_REPO_DIRTY=""
if [ -n "$(git -C "${KUMO_REPO}" status --porcelain -- packages 2>/dev/null)" ]; then
    KUMO_REPO_DIRTY="+local changes"
fi
for pkg_dir in "${KUMO_REPO}"/packages/*/; do
    pkg_name="$(basename "${pkg_dir}")"
    case "${pkg_name}" in kumo-sdk|diskgraph) continue ;; esac
    [ -f "${pkg_dir}pyproject.toml" ] || [ -f "${pkg_dir}setup.py" ] || continue
    dist_glob="${pkg_name//-/_}-*.whl"
    log "building ${pkg_name} from ${pkg_dir}"
    "${PYTHON}" -m pip wheel --no-deps -q -w "${WHEELS}" "${pkg_dir}"
    record "${dist_glob}" kumo-first-party \
        "${KUMO_REPO}/packages/${pkg_name} @ ${KUMO_REPO_HEAD}${KUMO_REPO_DIRTY}"
done
# --------------------------------------------------------------------------
# 2) DiskGraph, from packages/diskgraph. The wheel is abi3, so one build
#    serves every CPython at or above its floor, and it is not linked against
#    torch, so it is torch-version-independent.
# --------------------------------------------------------------------------
command -v cargo >/dev/null 2>&1 || die "cargo is required to build DiskGraph"

ZIG_ENV=(DISKGRAPH_BUILD_ZIG=0)
if [ "${USE_ZIG}" = "1" ]; then
    # Probe for the binary, not `cargo zigbuild --version`: the zigbuild
    # subcommand rejects --version, so that probe always fails and would
    # reinstall on every run. This is the check the build backend makes.
    if ! command -v cargo-zigbuild >/dev/null 2>&1; then
        log "installing cargo-zigbuild (needed for the manylinux build)"
        cargo install cargo-zigbuild --locked >&2
    fi
    ZIG_ENV=(DISKGRAPH_BUILD_ZIG=1)
fi

[ -f "${DISKGRAPH_SOURCE}/diskgraph-pybind/Cargo.toml" ] \
    || die "no DiskGraph source at ${DISKGRAPH_SOURCE}"

log "building diskgraph from ${DISKGRAPH_SOURCE}"
env "${ZIG_ENV[@]}" "${PYTHON}" -m pip wheel --no-deps -q \
    -w "${WHEELS}" "${DISKGRAPH_SOURCE}" >&2
record 'diskgraph-*.whl' diskgraph-native "in-tree source ${DISKGRAPH_SOURCE}"

if [ "${HOST_OS}" = "Darwin" ]; then
    DG_WHEEL="$(ls -1 "${WHEELS}"/diskgraph-*.whl | sort | tail -1)"
    DG_PLATFORM="$(basename "${DG_WHEEL}" .whl)"
    DG_PLATFORM="${DG_PLATFORM##*-}"
    case "${DG_PLATFORM}" in
        macosx_*_"$(uname -m)") ;;
        *) die "expected a native macOS DiskGraph wheel, got ${DG_WHEEL}" ;;
    esac
    if [ -n "${PLATFORM_TAG}" ] && [ "${PLATFORM_TAG}" != "${DG_PLATFORM}" ]; then
        die "platform tag ${PLATFORM_TAG} does not match DiskGraph's ${DG_PLATFORM}"
    fi
    PLATFORM_TAG="${DG_PLATFORM}"
fi

if [ "${USE_ZIG}" = "1" ]; then
    DG_WHEEL="$(ls -1 "${WHEELS}"/diskgraph-*.whl | sort | tail -1)"
    case "$(basename "${DG_WHEEL}")" in
        *manylinux*) ;;
        *) die "expected a manylinux DiskGraph wheel from the zig build,\
 got $(basename "${DG_WHEEL}")" ;;
    esac
fi

# The wheel's platform tag only describes the extension module: maturin applies
# --compatibility manylinux2014 to what it builds, while the ingest binary is a
# separate cargo-zigbuild invocation with no glibc-version suffix on its target,
# so it picks up whatever glibc zig defaults to. Measure both rather than trust
# the filename -- a binary above the runtime's glibc fails at exec time, long
# after `pip install` reported success.
check_glibc_floor() {
    local dumper=""
    if command -v objdump >/dev/null 2>&1; then
        dumper="objdump -T"
    elif command -v readelf >/dev/null 2>&1; then
        dumper="readelf -W --dyn-syms"
    else
        log "WARNING: no objdump/readelf; skipping the glibc floor check"
        return 0
    fi

    local probe="${WORK}/glibc-probe"
    rm -rf "${probe}"
    mkdir -p "${probe}"
    # shellcheck disable=SC2012
    "${PYTHON}" -m zipfile -e \
        "$(ls -1 "${WHEELS}"/diskgraph-*.whl | sort | tail -1)" "${probe}"

    local worst="0.0" path floor
    for path in "${probe}/diskgraph/bin/ingest" "${probe}"/diskgraph/_c*.so; do
        [ -f "${path}" ] || continue
        floor="$(${dumper} "${path}" 2>/dev/null \
            | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' \
            | sort -uV | tail -1)"
        [ -n "${floor}" ] || continue
        log "  $(basename "${path}") needs glibc >= ${floor}"
        worst="$(printf '%s\n%s\n' "${worst}" "${floor}" | sort -V | tail -1)"
    done

    if [ "$(printf '%s\n%s\n' "${worst}" "${MAX_GLIBC}" | sort -V | tail -1)" \
            != "${MAX_GLIBC}" ]; then
        die "DiskGraph needs glibc ${worst}, above the ${MAX_GLIBC} ceiling.\
 The target runtime must provide at least ${worst}; raise MAX_GLIBC if it does."
    fi
    log "DiskGraph glibc floor: ${worst} (ceiling ${MAX_GLIBC})"
}
if [ "${HOST_OS}" != "Darwin" ]; then
    check_glibc_floor
fi

# --------------------------------------------------------------------------
# 3) Generate the legal payload and combined license expression.
# --------------------------------------------------------------------------
NOTICE_GENERATOR="${KUMO_REPO}/scripts/gen_third_party.py"
[ -f "${NOTICE_GENERATOR}" ] || die "no notice generator at ${NOTICE_GENERATOR}"
NOTICE_ARTIFACT_ARGS=(
    --artifact diskgraph-extension
    --artifact diskgraph-ingest
    --artifact kumo-core-bundled-source
)

if [ "${GENERATE_THIRD_PARTY_NOTICE}" = "1" ]; then
    "${PYTHON}" "${NOTICE_GENERATOR}" --check "${THIRD_PARTY_NOTICE}" \
        || die "the committed third-party notice is stale"
    GENERATED_THIRD_PARTY_NOTICE="${WORK}/THIRD-PARTY.txt"
    "${PYTHON}" "${NOTICE_GENERATOR}" --wheels "${WHEELS}" \
        --components "${COMPONENTS}" \
        "${NOTICE_ARTIFACT_ARGS[@]}" \
        --output "${GENERATED_THIRD_PARTY_NOTICE}" \
        || die "the collected wheels do not match the third-party inventory"
    THIRD_PARTY_NOTICE="${GENERATED_THIRD_PARTY_NOTICE}"
fi

LICENSE_EXPRESSION="$(
    "${PYTHON}" "${NOTICE_GENERATOR}" \
        "${NOTICE_ARTIFACT_ARGS[@]}" \
        --base-license Apache-2.0 \
        --license-expression
)" || die "could not derive the wheel license expression"

# --------------------------------------------------------------------------
# 4) Merge everything into one wheel.
# --------------------------------------------------------------------------
merge_args=(--version "${BUNDLE_VERSION}" --platform-tag "${PLATFORM_TAG}"
            --notice "${THIRD_PARTY_NOTICE}"
            --license "${FIRST_PARTY_LICENSE}"
            --license-expression "${LICENSE_EXPRESSION}")
if [ "${OFFLINE_REPORT}" = "1" ]; then
    merge_args+=(--offline-report)
fi
WHEEL_PATH="$("${PYTHON}" "${SCRIPT_DIR}/merge_wheels.py" "${WHEELS}" \
    "${COMPONENTS}" "${OUTPUT_DIR}" "${merge_args[@]}")"

SIZE_MIB="$("${PYTHON}" -c 'import os, sys; print(os.path.getsize(sys.argv[1]) // 1024**2)' "${WHEEL_PATH}")"
log "wrote ${WHEEL_PATH} (${SIZE_MIB} MiB)"

# --------------------------------------------------------------------------
# 5) Optional smoke test: install the finished wheel into a real base image and
#    import the closure.
# --------------------------------------------------------------------------
if [ -n "${VERIFY_IMAGE}" ]; then
    log "verifying in ${VERIFY_IMAGE}"
    # Written to a file rather than passed as `bash -c`, so the Python source
    # below needs no shell quoting.
    cat > "${WORK}/verify.py" <<'PYEOF'
import importlib.util
import os
import pathlib
import sys
from importlib.metadata import distribution

import diskgraph
import kumo_core  # noqa: F401
import kumoapi  # noqa: F401
import torch
import torch_geometric
from kumo.engine import FileConnector, Graph, Table, Trainer  # noqa: F401
from kumo.engine import SnowflakeConnector  # noqa: F401  (import is lazy)
from kumo.engine import __version__
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.version import get_kumo_version

dist = distribution(sys.argv[1])
assert dist.version == __version__ == get_kumo_version(), (
    dist.version, __version__, get_kumo_version())
extras = set(dist.metadata.get_all('Provides-Extra') or [])
print('extras:', ', '.join(sorted(extras)))

# Provider code ships in every install, their third-party dependencies only
# with the matching extra: the entry points are registered, but loading one
# must fail on the missing dependency rather than succeed.
sources = {e.name for e in dist.entry_points
           if e.group == 'kumoai_studio.sources'}
if 'snowflake' in sources:
    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        pass
    else:
        raise SystemExit('snowflake-connector-python is installed without '
                         'the snowflake extra')
    try:
        import kumo_snowflake.backend  # noqa: F401
    except ImportError:
        pass
    else:
        raise SystemExit('kumo_snowflake.backend imported without the '
                         'snowflake extra')

ingest = pathlib.Path(diskgraph.__file__).parent / 'bin' / 'ingest'
assert ingest.is_file() and os.access(ingest, os.X_OK), ingest
if importlib.util.find_spec('kumo_glove_6b_50d'):  # only with the `text` extra
    EmbeddingModelStore().get('glove.6B-50d').vectors()
print('wheel OK', __version__, '| torch', torch.__version__, '| pyg',
      torch_geometric.__version__)
PYEOF
    cat > "${WORK}/verify.sh" <<EOF
set -eux
prefix=\$(python -c 'import sys; print(sys.prefix)')
# The Container Runtime entrypoint expects these to exist; a bare \`docker run\`
# skips the step that creates them.
mkdir -p /var/log/managedservices/user/mlrs /var/log/managedservices/system/mlrs
python -m pip install --no-cache-dir --prefix "\$prefix" \\
    /wheel/$(basename "${WHEEL_PATH}")
python /work/verify.py "${BUNDLE_NAME}"
EOF
    docker run --rm \
        -v "$(cd "$(dirname "${WHEEL_PATH}")" && pwd):/wheel:ro" \
        -v "${WORK}:/work:ro" \
        --entrypoint bash "${VERIFY_IMAGE}" /work/verify.sh >&2
fi

echo "${WHEEL_PATH}"
