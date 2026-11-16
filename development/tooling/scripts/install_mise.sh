#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

MODIFY_SHELL=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-modify-shell) MODIFY_SHELL=false; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

MISE_BIN="${HOME}/.local/bin/mise"

write_managed_mise_block() {
  local target_file="$1"
  local tmp_existing
  local tmp_block
  local start_marker="# >>> mise >>>"
  local end_marker="# <<< mise <<<"
  local start_count
  local end_count
  local start_line
  local end_line

  tmp_existing="$(mktemp)"
  tmp_block="$(mktemp)"
  cat > "${tmp_block}"

  touch "${target_file}"
  start_count="$(grep -cF "${start_marker}" "${target_file}" || true)"
  end_count="$(grep -cF "${end_marker}" "${target_file}" || true)"

  # Safety check: only allow no existing block, or one complete, ordered block.
  # Any other marker state is treated as corruption and hard-fails.
  if [[ "${start_count}" -eq 0 && "${end_count}" -eq 0 ]]; then
    :
  elif [[ "${start_count}" -eq 1 && "${end_count}" -eq 1 ]]; then
    start_line="$(grep -nF "${start_marker}" "${target_file}" | cut -d: -f1)"
    end_line="$(grep -nF "${end_marker}" "${target_file}" | cut -d: -f1)"
    if [[ "${start_line}" -ge "${end_line}" ]]; then
      echo "Invalid mise marker order in ${target_file}: start marker must appear before end marker." >&2
      rm -f "${tmp_existing}" "${tmp_block}"
      return 1
    fi
  else
    echo "Invalid mise marker state in ${target_file}: expected 0/0 or 1/1 start/end markers, got ${start_count}/${end_count}." >&2
    rm -f "${tmp_existing}" "${tmp_block}"
    return 1
  fi

  awk -v start="${start_marker}" -v end="${end_marker}" '
    $0 == start {in_block = 1; next}
    $0 == end {in_block = 0; next}
    !in_block {print}
  ' "${target_file}" > "${tmp_existing}"

  {
    cat "${tmp_existing}"
    if [[ -s "${tmp_existing}" ]]; then
      echo
    fi
    echo "${start_marker}"
    cat "${tmp_block}"
    echo "${end_marker}"
  } > "${target_file}"

  rm -f "${tmp_existing}" "${tmp_block}"
}

if [[ ! -x "${MISE_BIN}" ]]; then
  curl https://mise.run | sh
else
  "${MISE_BIN}" self-update -y
fi

# Verify install succeeded without loading an untrusted project config from the
# caller's current directory. The top-level installer explicitly trusts the
# repository before it runs project-aware mise commands.
(cd "${HOME}" && "${MISE_BIN}" --version)

SHELL_NAME="${KUMO_SETUP_SHELL:-$(basename "${SHELL:-bash}")}"
if [[ "${MODIFY_SHELL}" == true && "${SHELL_NAME}" != bash && "${SHELL_NAME}" != zsh ]]; then
  echo "Cannot automatically configure unsupported shell '${SHELL_NAME}'." >&2
  echo "Re-run with --no-modify-shell and configure mise activation manually." >&2
  exit 1
fi

PROFILE_TARGET="${HOME}/.profile"
if [[ -f "${HOME}/.bash_profile" ]]; then
  PROFILE_TARGET="${HOME}/.bash_profile"
fi

if [[ "${MODIFY_SHELL}" == true ]]; then
  echo
  echo "=============================="
  echo "Configuring shell startup files for mise"
  if [[ "${SHELL_NAME}" == zsh ]]; then
    write_managed_mise_block "${HOME}/.zshrc" <<'EOF'
eval "$("$HOME/.local/bin/mise" activate zsh)"
EOF

    write_managed_mise_block "${HOME}/.zprofile" <<'EOF'
if [[ -o login ]] && ! typeset -f mise >/dev/null 2>&1; then
  eval "$("$HOME/.local/bin/mise" activate zsh --shims)"
fi
EOF
  elif [[ "${SHELL_NAME}" == bash ]]; then
    write_managed_mise_block "${HOME}/.bashrc" <<'EOF'
eval "$("$HOME/.local/bin/mise" activate bash)"
EOF

    write_managed_mise_block "${PROFILE_TARGET}" <<'EOF'
if [ -n "$BASH_VERSION" ] && ! declare -F mise >/dev/null 2>&1; then
  # .bash_profile/.profile often source .bashrc first for login shells.
  # If mise is already a shell function from the .bashrc activation,
  # skip shims so PATH-based activation is not replaced.
  # Reference: https://mise.jdx.dev/dev-tools/shims.html#how-to-add-mise-shims-to-path
  eval "$("$HOME/.local/bin/mise" activate bash --shims)"
fi
EOF
  fi
else
  echo
  echo "=============================="
  echo "Skipping shell configuration (--no-modify-shell)"
  echo "Run the following commands to configure shell startup files manually:"
  echo
  if [[ "${SHELL_NAME}" == zsh ]]; then
    echo "  cat <<'EOF' >> \"${HOME}/.zshrc\""
    echo 'eval "$("$HOME/.local/bin/mise" activate zsh)"'
    echo 'EOF'
    echo
    echo "  cat <<'EOF' >> \"${HOME}/.zprofile\""
    echo 'if [[ -o login ]] && ! typeset -f mise >/dev/null 2>&1; then'
    echo '  eval "$("$HOME/.local/bin/mise" activate zsh --shims)"'
    echo 'fi'
    echo 'EOF'
  elif [[ "${SHELL_NAME}" == bash ]]; then
    echo "  cat <<'EOF' >> \"${HOME}/.bashrc\""
    echo 'eval "$("$HOME/.local/bin/mise" activate bash)"'
    echo 'EOF'
    echo
    echo "  cat <<'EOF' >> \"${PROFILE_TARGET}\""
    echo 'if [ -n "$BASH_VERSION" ] && ! declare -F mise >/dev/null 2>&1; then'
    echo '  eval "$("$HOME/.local/bin/mise" activate bash --shims)"'
    echo 'fi'
    echo 'EOF'
  else
    echo "Shell '${SHELL_NAME}' is not modified automatically."
    echo "Configure activation using: https://mise.jdx.dev/getting-started.html#activate-mise"
  fi
fi
