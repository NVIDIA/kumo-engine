#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly OUTPUT_FILE="${1:?usage: prepare-aws-rds-ca.sh OUTPUT_FILE}"
readonly COMMERCIAL_URL="https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"
readonly COMMERCIAL_SHA256="fe45bbebf92ad3e27a583bbb2ddd1553c521ed4d49af5514dc0a40372ea5395c"
readonly GOVCLOUD_URL="https://truststore.pki.us-gov-west-1.rds.amazonaws.com/global/global-bundle.pem"
readonly GOVCLOUD_SHA256="bae59f78f2e2ba789e734cdcac78c13a0f0e99aa3f7bd49f1f37477c815b9b33"
readonly COMBINED_SHA256="3904eb87884e7f062f6634d4e52f30cca4f6ad0c10142a960184b60cc3852a60"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

require_digest() {
  local file="$1" expected="$2" actual
  actual=$(sha256_file "${file}")
  [[ "${actual}" == "${expected}" ]] || {
    printf 'error: SHA-256 mismatch for %s: got %s, want %s\n' \
      "${file}" "${actual}" "${expected}" >&2
    exit 1
  }
}

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/kumo-rds-ca.XXXXXX")
cleanup() {
  rm -rf "${work_dir}"
}
trap cleanup EXIT

curl --fail --silent --show-error --location \
  --connect-timeout 10 --max-time 60 \
  "${COMMERCIAL_URL}" --output "${work_dir}/commercial.pem"
curl --fail --silent --show-error --location \
  --connect-timeout 10 --max-time 60 \
  "${GOVCLOUD_URL}" --output "${work_dir}/govcloud.pem"
require_digest "${work_dir}/commercial.pem" "${COMMERCIAL_SHA256}"
require_digest "${work_dir}/govcloud.pem" "${GOVCLOUD_SHA256}"

cat "${work_dir}/commercial.pem" "${work_dir}/govcloud.pem" \
  >"${work_dir}/combined.pem"
require_digest "${work_dir}/combined.pem" "${COMBINED_SHA256}"

subjects=$(
  openssl crl2pkcs7 -nocrl -certfile "${work_dir}/combined.pem" |
    openssl pkcs7 -print_certs -noout
)
for family in \
  "Root CA RSA2048 G1" \
  "Root CA RSA4096 G1" \
  "Root CA ECC384 G1"; do
  grep -Fq "${family}" <<<"${subjects}" || {
    printf 'error: generated RDS bundle lacks %s\n' "${family}" >&2
    exit 1
  }
done
mkdir -p "$(dirname "${OUTPUT_FILE}")"
install -m 0444 "${work_dir}/combined.pem" "${OUTPUT_FILE}"
