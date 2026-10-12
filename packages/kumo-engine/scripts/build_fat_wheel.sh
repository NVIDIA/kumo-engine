#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build one fat wheel carrying the whole Kumo training closure, so a Snowflake
# notebook on Container Runtime can install it with a single `pip install` --
# no private index, no GitHub access, no wheelhouse:
#
#     PUT file://dist/kumo_fat_bundle-*.whl @DB.SCHEMA.WHEELS
#     # ...then, in the notebook:
#     session.file.get("@DB.SCHEMA.WHEELS/kumo_fat_bundle-...whl", "/tmp/w")
#     !python -m pip install --prefix "$(python -c 'import sys; print(sys.prefix)')" \
#         /tmp/w/kumo_fat_bundle-*.whl
#
# FIRST, probe the interpreter you are targeting. Container Runtime versions
# differ in interpreter, and every native wheel in the bundle is keyed to it.
# In a notebook on the target runtime:
#
#     import sys; print(sys.version_info[:2])
#     # -> (3, 12)
#
# then pass `--python-tag cp312` (the default). The bundle does not pin torch:
# the target's own torch/CUDA build satisfies kumo-core's `torch>=2.7.0,<3`.
# DBR 17.3: --extra-components kumo-databricks (or kumo-databricks,kumo-s3).
# The build fails loudly if any component has no wheel for the interpreter.
#
# Bundled first-party and native components:
#
#     every            ${KUMO_REPO}/packages/* -- kumo-engine, kumo-core,
#     packages/ dist   kumo-api, kumo-snowflake, kumo-s3, kumo-databricks,
#                       kumo-bigquery, ... every
#                       directory with a pyproject.toml or setup.py, EXCEPT
#                       kumo-sdk: this bundle ships the local-only
#                       `kumo-engine` package instead of the hosted
#                       `kumoai_studio` SDK, which previously collided with it
#                       on the top-level `kumoai` import name (resolved by the
#                       kumoai_studio rename, but kumo-sdk stays excluded here
#                       since this bundle is for the local-only closure).
#     diskgraph        packages/diskgraph -- native abi3 extension plus the
#                      standalone `ingest` binary
#     glove.6B 50d     Stanford GloVe data in the first-party
#                      kumo-glove-6b-50d wrapper
#     torch-geometric  released pure-Python wheel from PyPI
#
# Other Python dependencies remain Requires-Dist entries; --offline-report lists them.
#
# Adapted from kumo's scripts/build_kumo_fat_wheel.py: this bundles
# kumo-engine (the local-only SDK) as well as every reader package under
# packages/, but not kumo-sdk (the hosted `kumoai_studio` SDK).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

# --------------------------------------------------------------------------
# Defaults. Every one of these is overridable by flag or environment.
# --------------------------------------------------------------------------
KUMO_REPO="${KUMO_REPO:-${REPO_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
BUNDLE_NAME="${BUNDLE_NAME:-kumo-fat-bundle}"
BUNDLE_VERSION="${BUNDLE_VERSION:-}"
PYTHON_TAG="${PYTHON_TAG:-cp312}"
# Conservative Linux tag covering the bundled DiskGraph binaries.
PLATFORM_TAG="${PLATFORM_TAG:-linux_x86_64}"
TORCH_GEOMETRIC_VERSION="${TORCH_GEOMETRIC_VERSION:-2.8.0.post1}"
EXTRA_COMPONENTS="${EXTRA_COMPONENTS:-}"
# Build DiskGraph through cargo-zigbuild so the extension and the ingest binary
# link against a manylinux2014 glibc rather than this host's. A native build
# links against the build host's glibc and will not load on the runtime.
USE_ZIG="${USE_ZIG:-1}"
# Highest glibc the DiskGraph binaries may require. Container Runtime is
# Ubuntu-based and provides 2.35, so 2.34 leaves headroom while still catching a
# binary accidentally linked against a modern build host. Note that this is NOT
# what the wheel's manylinux tag claims: see check_glibc_floor.
MAX_GLIBC="${MAX_GLIBC:-2.34}"
VERIFY_IMAGE="${VERIFY_IMAGE:-}"
THIRD_PARTY_NOTICE="${THIRD_PARTY_NOTICE:-}"
GENERATE_THIRD_PARTY_NOTICE=0
OFFLINE_REPORT=0
KEEP_WORK=0
MANYLINUX_IMAGE="quay.io/pypa/manylinux2014_x86_64"

usage() {
    sed -n '6,59p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Usage: scripts/build_fat_wheel.sh [options]

Options:
  --output-dir DIR      Where the wheel is written.       [dist/]
  --name NAME           Bundle distribution name.         [kumo-fat-bundle]
  --version VERSION     Bundle version.        [today's date+short commit]
  --python-tag TAG      Target interpreter, e.g. cp312.   [cp312]
  --extra-components LIST Optional packages, comma-separated. [all]
                        E.g. kumo-databricks,kumo-s3; engine is always included.
  --platform-tag TAG    Target platform tag.              [linux_x86_64]
  --torch-geometric VER torch-geometric version.          [2.8.0.post1]
  --kumo-repo PATH      kumo monorepo checkout.           [this repo]
  --no-zig              Build DiskGraph natively instead of through
                        cargo-zigbuild. Only safe when this host's glibc is no
                        newer than the runtime's.
  --verify-image IMAGE  After building, install the wheel into IMAGE with
                        Docker and import the closure as a smoke test.
  --offline-report      Print the requirements pip must still resolve at
                        install time.
  --keep-work           Leave the scratch directory in place for inspection.
  -h, --help            This message.

Environment:
  MAX_GLIBC             Highest glibc the DiskGraph binaries may require.
                        [2.34; Container Runtime provides 2.35]
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --name) BUNDLE_NAME="$2"; shift 2 ;;
        --version) BUNDLE_VERSION="$2"; shift 2 ;;
        --python-tag) PYTHON_TAG="$2"; shift 2 ;;
        --extra-components) EXTRA_COMPONENTS="$2"; shift 2 ;;
        --platform-tag) PLATFORM_TAG="$2"; shift 2 ;;
        --torch-geometric) TORCH_GEOMETRIC_VERSION="$2"; shift 2 ;;
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

log() { echo "[fat-wheel] $*" >&2; }
die() { echo "[fat-wheel] ERROR: $*" >&2; exit 1; }

EXTRA_COMPONENTS="${EXTRA_COMPONENTS//[[:space:]]/}"
if [ -n "${EXTRA_COMPONENTS}" ]; then
    IFS=',' read -r -a requested_components <<< "${EXTRA_COMPONENTS}"
    for component in "${requested_components[@]}"; do
        case "${component}" in
            kumo-sdk|diskgraph|*/*|.|..|'') die "invalid extra component: ${component}" ;;
        esac
        [[ -f "${KUMO_REPO}/packages/${component}/pyproject.toml"
           || -f "${KUMO_REPO}/packages/${component}/setup.py" ]] \
            || die "unknown extra component: ${component}"
    done
fi

# The build host's interpreter only drives pip; it need not match --python-tag.
# Every pure-Python component is built with `pip wheel`, DiskGraph's wheel tag
# comes from its abi3 build config.
PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 || die "no ${PYTHON} on PATH"
"${PYTHON}" -m pip --version >/dev/null 2>&1 \
    || die "${PYTHON} has no pip; set PYTHON to an interpreter that does\
 (a virtualenv works)"

# The interpreter used to drive pip is irrelevant to the pure-Python
# components, but `pip download` needs a --python-version, so derive the
# target's from the tag: cp310 -> 3.10.
case "${PYTHON_TAG}" in
    cp3??) TARGET_PY="3.${PYTHON_TAG#cp3}" ;;
    *) die "--python-tag must look like cp312, got '${PYTHON_TAG}'" ;;
esac

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
if [ -z "${BUNDLE_VERSION}" ]; then
    # No --version: stamp today's date and the monorepo's short commit, e.g.
    # 20260103+a1b2c3d. Distinguishes builds from the same kumoai_studio version
    # without requiring a version bump for every fat-wheel build. `+`, not
    # `-`: PEP 440 has no plain hyphen segment, so `20260103-a1b2c3d` is not
    # a valid version and pip rejects it; `+` is the local-version separator.
    KUMO_REPO_SHORT_HEAD="$(git -C "${KUMO_REPO}" rev-parse --short HEAD \
        2>/dev/null || echo unknown)"
    BUNDLE_VERSION="$(date +%Y%m%d)+${KUMO_REPO_SHORT_HEAD}"
fi

WHEEL_TAG="${PYTHON_TAG}-${PYTHON_TAG}-${PLATFORM_TAG}"

WORK="$(mktemp -d -t kumo-fat-wheel-XXXXXX)"
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
# component. The merge step turns it into kumo_bundle_manifest.json.
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
#    except kumo-sdk and the native DiskGraph build below (kumo-core,
#    kumo-api, kumo-pql, kumo-snowflake, kumo-s3,
#    kumo-databricks, kumo-bigquery,
#    kumo-engine, ...). kumo-sdk is excluded on purpose: it ships the
#    hosted `kumoai_studio` SDK, which this bundle replaces with the local-only
#    `kumo-engine` package -- the two previously both provided a top-level
#    `kumoai` module (before the kumoai_studio rename), so bundling both would
#    have collided at merge time; kumo-sdk stays excluded since this bundle is
#    for the local-only closure. The build
#    interpreter is irrelevant. A new reader package needs no change here
#    -- it ships in the default bundle the moment it lands under packages/.
#    --extra-components restricts optional packages on any Torch target.
# --------------------------------------------------------------------------
KUMO_REPO_HEAD="$(git -C "${KUMO_REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
KUMO_REPO_DIRTY=""
if [ -n "$(git -C "${KUMO_REPO}" status --porcelain -- packages 2>/dev/null)" ]; then
    KUMO_REPO_DIRTY="+local changes"
fi
for pkg_dir in "${KUMO_REPO}"/packages/*/; do
    pkg_name="$(basename "${pkg_dir}")"
    case "${pkg_name}" in kumo-sdk|diskgraph) continue ;; esac
    if [ -n "${EXTRA_COMPONENTS}" ]; then
        case "${pkg_name}" in
            kumo-api|kumo-core|kumo-pql|kumo-engine) ;;
            *) [[ ",${EXTRA_COMPONENTS}," = *",${pkg_name},"* ]] || continue ;;
        esac
    fi
    [ -f "${pkg_dir}pyproject.toml" ] || [ -f "${pkg_dir}setup.py" ] || continue
    dist_glob="${pkg_name//-/_}-*.whl"
    log "building ${pkg_name} from ${pkg_dir}"
    "${PYTHON}" -m pip wheel --no-deps -q -w "${WHEELS}" "${pkg_dir}"
    record "${dist_glob}" kumo-first-party \
        "${KUMO_REPO}/packages/${pkg_name} @ ${KUMO_REPO_HEAD}${KUMO_REPO_DIRTY}"
done
# The default text-embedding model, so text columns work with no upload.
log "building kumo-glove-6b-50d from Stanford's glove.6B.zip"
env -u PYTHON "${KUMO_REPO}/tools/glove-6b-50d/build_wheel.sh" "${WORK}/glove"
cp "${WORK}"/glove/kumo_glove_6b_50d-*.whl "${WHEELS}/"
record "kumo_glove_6b_50d-*.whl" kumo-first-party \
    "${KUMO_REPO}/tools/glove-6b-50d @ ${KUMO_REPO_HEAD}, from glove.6B.zip"

# --------------------------------------------------------------------------
# 2) DiskGraph, from packages/diskgraph.
#    The wheel is cp310-abi3, so one build
#    serves every CPython at or above 3.10, and it is not linked against
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
check_glibc_floor

# --------------------------------------------------------------------------
# 3) torch-geometric (pure Python), bundled from its released PyPI wheel. The
#    fat-wheel target is Python 3.12, where the released config-store behavior
#    is valid; the Python 3.14 monorepo separately pins an upstream fix.
# --------------------------------------------------------------------------
log "building torch-geometric ${TORCH_GEOMETRIC_VERSION}"
"${PYTHON}" -m pip wheel --no-deps -q -w "${WHEELS}" \
    "torch-geometric==${TORCH_GEOMETRIC_VERSION}"
record 'torch_geometric-*.whl' third-party \
    "torch-geometric==${TORCH_GEOMETRIC_VERSION} (PyPI)"

# --------------------------------------------------------------------------
# 4) Generate the legal payload and combined license expression.
# --------------------------------------------------------------------------
NOTICE_GENERATOR="${KUMO_REPO}/scripts/gen_third_party.py"
[ -f "${NOTICE_GENERATOR}" ] || die "no notice generator at ${NOTICE_GENERATOR}"
NOTICE_ARTIFACT_ARGS=(
    --artifact fat-wheel-bundled-python
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

FAT_WHEEL_LICENSE_EXPRESSION="$(
    "${PYTHON}" "${NOTICE_GENERATOR}" \
        "${NOTICE_ARTIFACT_ARGS[@]}" \
        --base-license Apache-2.0 \
        --license-expression
)" || die "could not derive the fat-wheel license expression"

# --------------------------------------------------------------------------
# 5) Merge everything into one wheel.
# --------------------------------------------------------------------------
MERGE_PY="${WORK}/merge.py"
cat > "${MERGE_PY}" <<'PYEOF'
"""Merge the collected component wheels into one installable wheel."""

import base64
import configparser
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

(wheels_dir, components_tsv, out_dir, name, version, tag, python_tag,
 target_py, offline_report, notice_path, license_path,
 license_expression) = sys.argv[1:]

wheels_dir = Path(wheels_dir)
components = []
for row in Path(components_tsv).read_text(encoding='utf-8').splitlines():
    if not row.strip():
        continue
    filename, classification, origin = row.split('\t', 2)
    components.append({
        'wheel': wheels_dir / filename,
        'classification': classification,
        'origin': origin,
    })


def log(message):
    print(f'[fat-wheel] {message}', file=sys.stderr, flush=True)


def canonical(dist):
    return re.sub(r'[-_.]+', '-', dist).lower()


def wheel_tags(wheel):
    """Parse the compressed tag set out of a wheel filename."""
    stem = wheel.name[:-len('.whl')]
    python, abi, platform = stem.rsplit('-', 3)[1:]
    return [(py, ab, pl) for py in python.split('.')
            for ab in abi.split('.') for pl in platform.split('.')]


def is_compatible(wheel):
    """Would the target interpreter accept this wheel?

    Only the interpreter/ABI half is checked: every platform tag allowed here
    is linux x86_64, and pip on the target accepts all of them.
    """
    target_minor = int(python_tag.removeprefix('cp'))
    for python, abi, _ in wheel_tags(wheel):
        if python.startswith('py3') and abi == 'none':
            return True  # universal
        if abi == 'abi3' and python.startswith('cp'):
            # abi3 is forward compatible: built for <= target runs on target.
            if int(python.removeprefix('cp')) <= target_minor:
                return True
        if python == python_tag and abi in (python_tag, 'abi3'):
            return True
    return False


# Read each component's own metadata, then refuse to ship a wheel whose
# contents cannot load on the target.
for component in components:
    with zipfile.ZipFile(component['wheel']) as archive:
        entry = next(n for n in archive.namelist()
                     if n.endswith('.dist-info/METADATA'))
        message = Parser().parsestr(archive.read(entry).decode('utf-8'))
    component['name'] = message['Name']
    component['version'] = message['Version']
    component['requires'] = message.get_all('Requires-Dist') or []

incompatible = [c for c in components if not is_compatible(c['wheel'])]
if incompatible:
    listing = '\n'.join(f"    {c['name']}: {c['wheel'].name}"
                        for c in incompatible)
    raise SystemExit(
        f'[fat-wheel] ERROR: these components have no {python_tag} build:\n'
        f'{listing}\nMerging them would ship extension modules the target '
        f'cannot import. Rebuild them for the target interpreter, or target '
        f'the interpreter they were built for with --python-tag.')

# Unpack every component into one tree, refusing silent overwrites. A
# component's own .dist-info is dropped -- the bundle ships one of its own --
# and its provenance is kept in kumo_bundle_manifest.json instead.
staging = Path(tempfile.mkdtemp(prefix='kumo-fat-staging-', dir=str(wheels_dir.parent)))
bundle_data = f"{name.replace('-', '_')}-{version}.data"
owners = {}
entry_points = configparser.ConfigParser(interpolation=None,
                                         delimiters=('=', ))
entry_points.optionxform = str
for component in components:
    with zipfile.ZipFile(component['wheel']) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target_name = info.filename
            if re.match(r'[^/]+\.dist-info/entry_points\.txt$', target_name):
                # Plugins are found through entry points, so keep them.
                found = configparser.ConfigParser(interpolation=None,
                                                  delimiters=('=', ))
                found.optionxform = str
                found.read_string(archive.read(info).decode('utf-8'))
                for group in found.sections():
                    if not entry_points.has_section(group):
                        entry_points.add_section(group)
                    for key, value in found.items(group):
                        if entry_points.get(group, key, fallback=value) != value:
                            raise SystemExit(
                                f'[fat-wheel] ERROR: entry point {group}:'
                                f'{key} is declared twice, differently')
                        entry_points.set(group, key, value)
                continue
            if re.match(r'[^/]+\.dist-info/', target_name):
                continue
            # A component's .data/ tree is keyed to that component's own
            # name-version, which the installer will not recognise under the
            # bundle's dist-info. Re-key it to the bundle's.
            data_match = re.match(r'[^/]+\.data/(.*)', target_name)
            if data_match:
                target_name = f'{bundle_data}/{data_match.group(1)}'
            previous = owners.get(target_name)
            if previous is not None and previous != component['name']:
                raise SystemExit(
                    f'[fat-wheel] ERROR: {target_name} is provided by both '
                    f"{previous} and {component['name']}; refusing to merge")
            owners[target_name] = component['name']
            destination = staging / target_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
            # The zip's executable bit matters: DiskGraph's bundled ingest
            # binary is spawned as a subprocess.
            mode = info.external_attr >> 16
            if mode & 0o111:
                destination.chmod(destination.stat().st_mode | 0o111)
    log(f"merged {component['name']} {component['version']}")

# Union the components' requirements, minus what the bundle itself contains.
bundled = {canonical(c['name']) for c in components}
# No seeded torch pin: kumo-core's own `torch>=2.7.0,<3` (an unconditional,
# non-extra dependency) flows through the union below unfiltered, same as
# any other plain third-party requirement -- letting the install target's
# own already-present torch/CUDA build satisfy it.
requirements = []
seen = set(requirements)
# No CPU/GPU branch for Faiss: kumo-engine's install_requires pulls
# kumo-core's `faiss-gpu` extra unconditionally, on every target, because
# faiss-gpu works as a CPU-only superset too (see the rationale in
# packages/kumo-engine/setup.py) -- so `faiss-gpu` is what every bundle
# target needs promoted here, not just CUDA builds.
#
# `faiss-gpu` is promoted because kumo-engine's own Requires-Dist is
# `kumo-core[faiss-gpu]>=...` (see packages/kumo-engine/setup.py): kumo-core
# is bundled, so that reference -- extras included -- is dropped by the
# `key in bundled` check below, and kumo-core's own `.dist-info` is dropped
# too. The only way its `faiss-gpu==1.14.3; extra == "faiss-gpu"`
# requirement survives into the merged bundle is by promoting the
# `faiss-gpu` marker here, same as the provider extras.
provider_extras = {'sdk', 'faiss-gpu'} | {
    key.removeprefix('kumo-') for key in bundled
}
for component in components:
    for requirement in component['requires']:
        extra = re.search(r';\s*extra\s*==\s*["\']([^"\']+)["\']\s*$',
                          requirement)
        if extra:
            if extra.group(1) not in provider_extras:
                continue
            requirement = requirement[:extra.start()].strip()
        dist = re.split(r'[<>=!~;\[ @]', requirement.strip(), maxsplit=1)[0]
        key = canonical(dist)
        if key in bundled:
            continue
        requirement = requirement.strip()
        if requirement in seen:
            continue
        # A direct URL for something the bundle carries is already satisfied;
        # for anything else it would break an offline install.
        if '@' in requirement.split(';')[0]:
            log(f'dropping direct-URL requirement: {requirement}')
            continue
        seen.add(requirement)
        requirements.append(requirement)

dist_info = staging / f"{name.replace('-', '_')}-{version}.dist-info"
dist_info.mkdir(parents=True)
if entry_points.sections():
    with open(dist_info / 'entry_points.txt', 'w', encoding='utf-8') as handle:
        entry_points.write(handle)

description = ['Bundled distributions:', '']
description += [f"  {c['name']} {c['version']}  ({c['origin']})"
                for c in components]
description += [
    '',
    f'Built for CPython {target_py}.',
    'Assembled by scripts/build_fat_wheel.sh. The bundled distributions\' own '
    'metadata is not installed, so importlib.metadata.version("kumo-core") '
    'will not resolve; see kumo_bundle_manifest.json for what is inside.',
]
next_minor = int(target_py.split('.')[1]) + 1
metadata = [
    'Metadata-Version: 2.4',
    f'Name: {name}',
    f'Version: {version}',
    'Summary: Kumo local training bundle',
    f'Requires-Python: >={target_py},<3.{next_minor}',
    f'License-Expression: {license_expression}',
    'License-File: LICENSE',
    'License-File: THIRD-PARTY.txt',
    *(f'Requires-Dist: {requirement}' for requirement in requirements),
    '',
    '\n'.join(description),
]
(dist_info / 'METADATA').write_text('\n'.join(metadata) + '\n',
                                    encoding='utf-8')
notice = Path(notice_path)
license = Path(license_path)
if not license.is_file():
    raise SystemExit(f'[fat-wheel] ERROR: no first-party license at {license}')
if not notice.is_file():
    raise SystemExit(f'[fat-wheel] ERROR: no third-party notice at {notice}')
(dist_info / 'licenses').mkdir(parents=True, exist_ok=True)
(dist_info / 'licenses' / 'LICENSE').write_bytes(license.read_bytes())
(dist_info / 'licenses' / 'THIRD-PARTY.txt').write_bytes(notice.read_bytes())
log(f'included first-party license ({license.stat().st_size // 1024} KiB)')
log(f'included third-party notice ({notice.stat().st_size // 1024} KiB)')

(dist_info / 'WHEEL').write_text(
    'Wheel-Version: 1.0\n'
    'Generator: build_fat_wheel.sh\n'
    'Root-Is-Purelib: false\n'
    f'Tag: {tag}\n', encoding='utf-8')
(staging / 'kumo_bundle_manifest.json').write_text(json.dumps({
    'bundle': {'name': name, 'version': version, 'tag': tag},
    'python': target_py,
    'components': [{
        'name': c['name'],
        'version': c['version'],
        'classification': c['classification'],
        'origin': c['origin'],
    } for c in components],
    'requires_at_install': requirements,
}, indent=2) + '\n', encoding='utf-8')

# Zip the staging tree, writing RECORD last: it lists every other file.
output = Path(out_dir) / f"{name.replace('-', '_')}-{version}-{tag}.whl"
output.parent.mkdir(parents=True, exist_ok=True)
rows = []
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(p for p in staging.rglob('*') if p.is_file()):
        relative = path.relative_to(staging).as_posix()
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
        rows.append([relative, f'sha256={digest}', str(len(data))])
        info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, data)
    record_name = (dist_info.relative_to(staging) / 'RECORD').as_posix()
    rows.append([record_name, '', ''])
    buffer = tempfile.NamedTemporaryFile('w', newline='', suffix='.csv',
                                         delete=False)
    with buffer:
        csv.writer(buffer).writerows(rows)
    archive.write(buffer.name, record_name)
    os.unlink(buffer.name)

log('bundled: ' + ', '.join(f"{c['name']} {c['version']}"
                            for c in components))
if offline_report == '1':
    log('still resolved from an index at install time:')
    for requirement in requirements:
        print(f'    {requirement}', file=sys.stderr)
print(output)
PYEOF

WHEEL_PATH="$("${PYTHON}" "${MERGE_PY}" "${WHEELS}" "${COMPONENTS}" \
    "${OUTPUT_DIR}" "${BUNDLE_NAME}" "${BUNDLE_VERSION}" "${WHEEL_TAG}" \
    "${PYTHON_TAG}" "${TARGET_PY}" "${OFFLINE_REPORT}" \
    "${THIRD_PARTY_NOTICE}" "${FIRST_PARTY_LICENSE}" \
    "${FAT_WHEEL_LICENSE_EXPRESSION}")"

SIZE_MIB="$(( $(stat -c %s "${WHEEL_PATH}") / 1024 / 1024 ))"
log "wrote ${WHEEL_PATH} (${SIZE_MIB} MiB)"

# --------------------------------------------------------------------------
# 6) Optional smoke test: install the finished wheel into a real base image and
#    import the closure.
# --------------------------------------------------------------------------
if [ -n "${VERIFY_IMAGE}" ]; then
    log "verifying in ${VERIFY_IMAGE}"
    # Written to a file rather than passed as `bash -c`, so the Python source
    # below needs no shell quoting.
    cat > "${WORK}/verify.py" <<'PYEOF'
import os
import pathlib
import sys
from importlib.metadata import distribution

import diskgraph
import kumo_core  # noqa: F401
import kumoapi  # noqa: F401
import torch
import torch_geometric
from kumo.engine import Graph, Table, Trainer  # noqa: F401
from kumo_core.embedding_models import EmbeddingModelStore

for entry in distribution(sys.argv[1]).entry_points:
    if entry.group in {'kumoai_studio.sources', 'kumoai_studio.sinks',
                       'kumoai_studio.identities'}:
        entry.load()

ingest = pathlib.Path(diskgraph.__file__).parent / 'bin' / 'ingest'
assert ingest.is_file() and os.access(ingest, os.X_OK), ingest
EmbeddingModelStore().get('glove.6B-50d').vectors()
print('bundle OK | torch', torch.__version__, '| pyg', torch_geometric.__version__)
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
