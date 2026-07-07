#!/usr/bin/env bash
# Build + publish a kumo-diskgraph wheel from a feature branch to S3.
#
# Mirrors production main's nightly publish pipeline but writes under
# `packages-dev/<branch-slug>/` instead of `packages/`, so kumo-side
# branches that need the in-flight rework wheel can resolve a stable
# prefix while production main stays untouched.
#
# Bucket / prefix layout (same shape as production):
#
#   s3://kumo-diskgraph/packages-dev/<branch-slug>/
#       ubuntu-<UBUNTU>/py<PY>/{commit/<SHA>/, latest/}/
#       diskgraph-0.1.0+g<short-sha>-cp<py>-...-...whl
#
# Slug default: `tt-ng` (the integration line for the rework).
# Override with --branch-slug for ad-hoc layouts.
#
# Usage:
#   ./diskgraph-pybind/scripts/publish_dev_wheel.sh [options]
#
# Options:
#   --branch-slug <slug>   Override the default tt-ng branch slug.
#   --ubuntu <ver>         Ubuntu version slot (default: 22.04).
#   --python <ver>         Python version slot (default: 3.10).
#   --dry-run              Print the upload commands without running them.
#
# Requires: `cargo` (Rust toolchain) on PATH for maturin build,
# `python3` with `pip`, `aws` CLI with creds for the kumo-diskgraph
# bucket.  Run on Linux for manylinux-shaped wheels.
set -euo pipefail

UBUNTU_VERSION="22.04"
PYTHON_VERSION="3.10"
BRANCH_SLUG=""
DRY_RUN="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --branch-slug)
            BRANCH_SLUG="$2"
            shift 2
            ;;
        --ubuntu)
            UBUNTU_VERSION="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PYBIND_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [ -z "${BRANCH_SLUG}" ]; then
    # Default to the integration line slug `tt-ng`.  Wheels for the
    # task-dataset rework — including PRs that target tt-ng — publish
    # into the same `packages-dev/tt-ng/` pipeline so kumo-side
    # consumers (kumo-9 in dev flavor) resolve a single, stable prefix
    # while colleagues iterate on parallel feature branches.
    # Override with `--branch-slug <slug>` for ad-hoc layouts.
    BRANCH_SLUG="tt-ng"
fi

SHA=$(git rev-parse HEAD)
if [ -z "${SHA}" ]; then
    echo "Failed to resolve git HEAD." >&2
    exit 2
fi

BUILD_DIR="$(mktemp -d -t kumo-diskgraph-publish-XXXXXX)"
VENV_DIR="${BUILD_DIR}/venv"
trap 'rm -rf "${BUILD_DIR}"' EXIT

# Build via maturin (same as production main).  Requires `cargo` on
# PATH; pyproject.toml declares maturin as the PEP 517 build backend.
# Throwaway venv so site-packages stays untouched on Linux and PEP 668
# doesn't bite on macOS Homebrew.
echo "Setting up build venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip build "maturin>=1.0,<2.0"

# Embed the SHA into the wheel version (mirrors `.github/workflows/nightly.yml`)
# so each publish lands at a unique S3 commit slot and `pip list` shows
# which commit is installed.  Restored after build via the EXIT trap.
SHORT_SHA=${SHA:0:8}
VERSION_WITH_SHA="0.1.0+g${SHORT_SHA}"
PYPROJECT="${PYBIND_DIR}/pyproject.toml"
PYPROJECT_BACKUP="${BUILD_DIR}/pyproject.toml.bak"
cp "${PYPROJECT}" "${PYPROJECT_BACKUP}"
trap 'cp "${PYPROJECT_BACKUP}" "${PYPROJECT}"; rm -rf "${BUILD_DIR}"' EXIT
sed -i.tmp "s/^version = .*/version = \"${VERSION_WITH_SHA}\"/" "${PYPROJECT}" && rm -f "${PYPROJECT}.tmp"

echo "Building wheel from ${PYBIND_DIR} (version=${VERSION_WITH_SHA})..."
( cd "${PYBIND_DIR}" \
    && "${VENV_DIR}/bin/python" -m build --wheel --outdir "${BUILD_DIR}" )

WHEEL_PATH=$(find "${BUILD_DIR}" -maxdepth 1 -type f -name 'diskgraph-*.whl' | head -n 1)
if [ -z "${WHEEL_PATH}" ]; then
    echo "No wheel produced under ${BUILD_DIR}." >&2
    exit 1
fi
WHEEL_BASENAME=$(basename "${WHEEL_PATH}")
echo "Built ${WHEEL_BASENAME}"

PREFIX="s3://kumo-diskgraph/packages-dev/${BRANCH_SLUG}/ubuntu-${UBUNTU_VERSION}/py${PYTHON_VERSION}"
COMMIT_DST="${PREFIX}/commit/${SHA}/${WHEEL_BASENAME}"
LATEST_DST="${PREFIX}/latest/${WHEEL_BASENAME}"

run_or_print() {
    if [ "${DRY_RUN}" = "true" ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

echo "Publishing to ${PREFIX}"
run_or_print aws s3 cp "${WHEEL_PATH}" "${COMMIT_DST}" --no-progress
run_or_print aws s3 cp "${WHEEL_PATH}" "${LATEST_DST}" --no-progress

echo
echo "Wheel published:"
echo "  branch slug:  ${BRANCH_SLUG}"
echo "  SHA:          ${SHA}"
echo "  ubuntu:       ${UBUNTU_VERSION}"
echo "  python:       ${PYTHON_VERSION}"
echo "  commit slot:  ${COMMIT_DST}"
echo "  latest slot:  ${LATEST_DST}"
echo
echo "Pin from a kumo branch:"
echo "  3rdparty/BUILD: KUMO_DISKGRAPH_VERSION = \"${SHA}\""
echo "  scripts/install_diskgraph.sh / .github/actions/setup/action.yml:"
echo "      DISKGRAPH_S3_PREFIX / diskgraph-s3-prefix = packages-dev/${BRANCH_SLUG}"
