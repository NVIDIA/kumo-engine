#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install of the python dependencies required for running the devserver.
#
# Under the hood, it creates a virtual environment at the location `.venv`, and
# installs all of the necessary python packages to this location. Each time this
# script is run, it will delete the virtual environment, so you are guarenteed
# to have a fresh environment each time.
#
# Prerequisites:
#
#   Please install dev tools via mise first:
#     mise install
#   It is discouraged to use conda simultaneously, as there may be conflicts.
#
# Usage
#
#   ./scripts/install_venv.sh
#   source ./.venv/bin/activate
#
# For convenience, you can add the following alias to your .bashrc to make
# reinstallation and activation easy (with the appropriate modifications).
#
#   alias kumo="cd $HOME/code/kumo; source ./.venv/bin/activate"
#   alias reinstall_kumo="kumo; ./scripts/install_venv.sh; kumo"
#
# Then, whenever you open a new terminal, just type "kumo" to enter the dev
# environment.
#
# This script needs to be run from the root of the Kumo repository. Only Ubuntu
# 22.04 is officially tested and supported.

set -xe

# This script deletes and recreates ./.venv. If the caller's shell is sitting
# inside a venv (commonly ./.venv itself -- e.g. the `reinstall_kumo` alias
# below), warn and continue: we still build on the correct interpreter (uv,
# resolved from .python-version, never the active venv), and remind the
# caller at the end to re-activate so they aren't left pointing at the
# now-recreated env.
ACTIVE_VENV_AT_START="${VIRTUAL_ENV:-}"
if [[ -n "${ACTIVE_VENV_AT_START}" ]]; then
  echo "WARNING: a virtualenv is active (${ACTIVE_VENV_AT_START}); ./.venv will be recreated. Re-activate it when this finishes (see the note below)." >&2
fi

if ! command -v uv >/dev/null 2>&1; then
  cat << EOF
uv is not available.

Install project toolchains with mise, then retry:

  mise install

EOF
  exit 1
fi

# Recreate the canonical virtualenv path and compatibility symlink. With no
# --python flag, uv resolves (and downloads, if missing) the interpreter
# pinned in .python-version itself -- PYTHON_BIN, if set, overrides that.
rm -rf ./.venv ./venv
uv venv ${PYTHON_BIN:+--python "${PYTHON_BIN}"} ./.venv
ln -s .venv ./venv

# Enable parallel build, if any packages need to be compiled
export MAKEFLAGS=-j8
export CMAKE_BUILD_PARALLEL_LEVEL=8

# Install all dependencies
make update-packages

# Install packages that are convenient for writing code
uv pip install --python ./.venv/bin/python flake8 pyright autopep8 isort yapf autoflake pip

# If a venv was active when this ran (override path), the caller's shell still
# has the OLD venv on PATH. Remind them to re-activate the freshly built one.
if [[ -n "${ACTIVE_VENV_AT_START}" ]]; then
  set +x
  echo ""
  echo "================================================================"
  echo "Your shell still has the OLD venv active (${ACTIVE_VENV_AT_START})."
  echo "Run: deactivate && source ./.venv/bin/activate"
  echo "================================================================"
fi
