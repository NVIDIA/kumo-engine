#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

setup_platform() {
  if [[ -n "${KUMO_SETUP_PLATFORM:-}" ]]; then
    printf '%s\n' "${KUMO_SETUP_PLATFORM}"
  else
    uname -s
  fi
}

setup_architecture() {
  if [[ -n "${KUMO_SETUP_ARCHITECTURE:-}" ]]; then
    printf '%s\n' "${KUMO_SETUP_ARCHITECTURE}"
  else
    uname -m
  fi
}

install_system_packages_linux() {
  echo
  echo "=============================="
  echo "Installing system packages"

  sudo apt-get update -y
  # redshift-connector caps lxml below its first Python 3.14 wheel, so local
  # installs need the XML headers. ARM64 native extensions also need OpenBLAS
  # and SWIG when PyG and FAISS wheels are built from source.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-utils cmake curl g++ gcc gdb git gnupg2 graphviz libkrb5-dev \
    libopenblas-dev libxml2-dev libxslt1-dev pass sqlite3 ssh swig vim make
}

install_system_packages_macos() {
  echo
  echo "=============================="
  echo "Installing system packages with Homebrew"

  if [[ "$(setup_architecture)" != arm64 ]]; then
    cat <<EOF >&2
Native Kumo setup currently supports Apple Silicon macOS only.
Detected architecture: $(setup_architecture)

Continue using the supported Ubuntu development environment on this machine.
EOF
    return 1
  fi

  if ! command -v brew >/dev/null 2>&1; then
    cat <<'EOF' >&2
Homebrew is required for Kumo development on macOS, but it was not found.

Install Homebrew from:
  https://brew.sh/

Then retry:
  make install-all-dependencies
EOF
    return 1
  fi

  brew bundle --file "${REPO_ROOT}/Brewfile"
}

ensure_docker_linux() {
  echo
  echo "=============================="
  echo "Installing Docker"

  sudo "${SCRIPT_DIR}/install_docker.sh"
  sudo usermod -aG docker "${USER}"
  if ! id -nG | grep -qw docker; then
    echo
    echo "Docker group membership was updated for ${USER}, but this shell session"
    echo "does not have the new group yet."
    echo
    echo "Before running docker-based commands, start a new login shell."
  fi
}

install_platform_dependencies() {
  local platform
  platform="$(setup_platform)"
  case "${platform}" in
    Linux)
      install_system_packages_linux
      ensure_docker_linux
      ;;
    Darwin)
      install_system_packages_macos
      ;;
    *)
      echo "Unsupported operating system: ${platform}" >&2
      echo "Supported systems are Ubuntu Linux and Apple Silicon macOS." >&2
      return 1
      ;;
  esac
}

main() {
  local no_modify_shell=false
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-modify-shell) no_modify_shell=true; shift ;;
      *) echo "Unknown option: $1" >&2; return 1 ;;
    esac
  done

  if [[ ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  cd "${REPO_ROOT}"
  install_platform_dependencies

  echo
  echo "=============================="
  echo "Installing mise"
  local install_mise_args=()
  if [[ "${no_modify_shell}" == true ]]; then
    install_mise_args+=("--no-modify-shell")
  fi
  "${SCRIPT_DIR}/install_mise.sh" "${install_mise_args[@]}"

  echo
  echo "=============================="
  echo "Installing rustup"
  "${SCRIPT_DIR}/install_rustup.sh"

  echo
  echo "=============================="
  echo "Installing pants"
  "${SCRIPT_DIR}/install_pants.sh"

  echo
  echo "=============================="
  echo "Installing dev tools via mise"
  mise trust
  mise install

  echo
  echo "=============================="
  echo "Setting up virtual environment"
  mise exec -- "${SCRIPT_DIR}/install_venv.sh"

  local shell_name
  shell_name="$(basename "${SHELL:-bash}")"

  echo
  echo "=============================="
  echo "All dependencies installed successfully."
  echo
  if [[ "${shell_name}" == bash || "${shell_name}" == zsh ]]; then
    echo "To activate mise in your current shell, run:"
    echo
    echo "  eval \"\$(\"\$HOME/.local/bin/mise\" activate ${shell_name})\""
  else
    echo "Configure mise activation for '${shell_name}' using:"
    echo "  https://mise.jdx.dev/getting-started.html#activate-mise"
  fi
  echo
  echo "Or start a new login shell."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
