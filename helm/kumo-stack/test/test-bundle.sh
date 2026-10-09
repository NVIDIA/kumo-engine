#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

STACK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/kumo-stack-bundle-test.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

HELM_BIN="${HELM_BIN:-helm}"

built_bundle=$(bash "${STACK_ROOT}/build-bundle.sh" "${TMP_ROOT}/output")
bundle_root=$(cd "${built_bundle}" && pwd)

[[ -f "${bundle_root}/Chart.yaml" ]] || fail "Chart.yaml is missing"
grep -Fqx 'type: library' "${bundle_root}/Chart.yaml" ||
  fail "bundle chart is installable"
[[ -r "${bundle_root}/install.sh" ]] || fail "install.sh is not readable"
[[ -f "${bundle_root}/config.yaml" ]] || fail "config.yaml is missing"
[[ -f "${bundle_root}/README.md" ]] || fail "README.md is missing"
cmp "${STACK_ROOT}/../../LICENSE" "${bundle_root}/LICENSE" >/dev/null ||
  fail "root LICENSE is missing or changed"
cmp "${STACK_ROOT}/THIRD-PARTY.txt" "${bundle_root}/THIRD-PARTY.txt" >/dev/null ||
  fail "third-party notice is missing or changed"
for chart in kumo-helm kumo-temporal kumo-platform-eks \
  kumo-platform-aks kumo-platform-aro; do
  [[ -f "${bundle_root}/charts/${chart}/Chart.yaml" ]] ||
    fail "${chart} source is missing"
  [[ ! -d "${bundle_root}/charts/${chart}/test" ]] ||
    fail "${chart} test fixtures were shipped"
done
[[ -f "${bundle_root}/charts/kumo-temporal/charts/temporal/Chart.yaml" ]] ||
  fail "Temporal dependency source is not unpacked"
[[ -s "${bundle_root}/charts/kumo-temporal/charts/temporal/LICENSE" ]] ||
  fail "Temporal dependency license is missing"
if find "${bundle_root}" \( -name '*.tgz' -o -name bom.yaml \) \
  -print -quit | grep -q .; then
  fail "bundle contains an archive or version catalog"
fi

package_dir="${TMP_ROOT}/package"
mkdir -p "${package_dir}"
"${HELM_BIN}" package "${bundle_root}" --destination "${package_dir}" \
  >/dev/null
bundle_version=$(awk '$1 == "version:" { print $2; exit }' \
  "${bundle_root}/Chart.yaml")
package="${package_dir}/kumo-helm-${bundle_version}.tgz"
[[ -f "${package}" ]] || fail "Helm package is missing"

template_error="${TMP_ROOT}/template-error"
if "${HELM_BIN}" template bundle-test "${package}" \
  > /dev/null 2>"${template_error}"; then
  fail "bundle chart is installable"
fi
grep -Fq 'library charts are not installable' "${template_error}" ||
  fail "bundle chart did not reject installation as a library"

mkdir -p "${TMP_ROOT}/pulled"
tar -xzf "${package}" -C "${TMP_ROOT}/pulled"
bundle_root=$(cd "${TMP_ROOT}/pulled/kumo-helm" && pwd)
cmp "${STACK_ROOT}/../../LICENSE" "${bundle_root}/LICENSE" >/dev/null ||
  fail "packaged root LICENSE is missing or changed"
cmp "${STACK_ROOT}/THIRD-PARTY.txt" "${bundle_root}/THIRD-PARTY.txt" >/dev/null ||
  fail "packaged third-party notice is missing or changed"
[[ -f "${bundle_root}/charts/kumo-temporal/charts/temporal/Chart.yaml" ]] ||
  fail "packaged Temporal dependency source is missing"

HELM_BIN="${HELM_BIN}" bash "${bundle_root}/bin/kumo" render \
  --values "${bundle_root}/config.yaml" \
  --output-dir "${TMP_ROOT}/rendered" >/dev/null
[[ -f "${TMP_ROOT}/rendered/core-values.yaml" ]] ||
  fail "bundle did not render the core chart values"
grep -Fxq 'TENANT_RELEASE=replace-me' "${TMP_ROOT}/rendered/runtime.env" ||
  fail "tenant release name was not derived from tenant.name"
grep -Fxq 'TENANT_NAMESPACE=kumo-replace-me' \
  "${TMP_ROOT}/rendered/runtime.env" ||
  fail "tenant namespace was not derived from tenant.name"

# Verify the convenience wrapper's routing without contacting a cluster.
calls="${TMP_ROOT}/installer-arguments"
mv "${bundle_root}/bin/kumo" "${bundle_root}/bin/kumo-real"
cat >"${bundle_root}/bin/kumo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"${KUMO_TEST_CALLS}"
EOF
chmod -x "${bundle_root}/install.sh" "${bundle_root}/bin/kumo"
export KUMO_TEST_CALLS="${calls}"

bash "${bundle_root}/install.sh"
[[ "$(sed -n '1p' "${calls}")" == install &&
   "$(sed -n '2p' "${calls}")" == --values &&
   "$(sed -n '3p' "${calls}")" == "${bundle_root}/config.yaml" &&
   "$(wc -l <"${calls}" | tr -d ' ')" == 3 ]] ||
  fail "full install wrapper arguments are wrong"

bash "${bundle_root}/install.sh" core
[[ "$(sed -n '1p' "${calls}")" == install &&
   "$(sed -n '2p' "${calls}")" == core &&
   "$(sed -n '3p' "${calls}")" == --values &&
   "$(sed -n '4p' "${calls}")" == "${bundle_root}/config.yaml" &&
   "$(wc -l <"${calls}" | tr -d ' ')" == 4 ]] ||
  fail "component install wrapper arguments are wrong"

printf '%s\n' "Kumo editable bundle tests: passed"
