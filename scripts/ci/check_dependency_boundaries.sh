#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Replaces Pants's per-package dependency-boundary enforcement (removed when
# Pants was retired). A uv workspace shares one lock and one environment, so
# uv itself cannot catch a package importing a dependency that's only
# installed because a sibling workspace member happens to need it (see
# https://docs.astral.sh/uv/concepts/projects/workspaces/). This runs deptry
# per independently-installable distribution, against that distribution's
# own manifest, to catch imports with no matching declared dependency
# (DEP001) or a declared-but-unused one (DEP004). DEP002 (unused
# dependency) is ignored below; too noisy for optional/platform-specific
# deps to block on yet.
#
# Scope: only packages with static PEP 621 `[project.dependencies]` are
# covered. The root `kumo` distribution, kumo-engine, and kumo-sdk declare
# their dependencies dynamically from setup.py, which deptry's PEP621 parser
# cannot read; diskgraph is a native Rust/pybind package with no
# conventional Python src/ to scan. None of these four are checked here.
#
# Each PACKAGES entry's extra_ignores field is a reviewed baseline of
# pre-existing findings (mostly: importing a module that's only a
# transitive dependency via kumo-core, not declared directly). It must
# shrink over time, not grow -- a new entry here needs a reason, not just a
# passing CI run.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# package_dir:src_root:known_first_party:extra_exclude:extra_ignores
# extra_ignores is a `|`-separated list of modules pre-existing DEP001
# findings reference -- `|` because --per-rule-ignores already uses `,` to
# separate different rule=value mappings.
PACKAGES=(
  "packages/kumo-api:kumoapi:kumoapi::typing_extensions|pydantic_core"
  "packages/kumo-pql:kumopql:kumopql:kumopql/grammar:"
  "packages/kumo-core:src:kumo_core::google"
  "packages/kumo-snowflake:src:kumo_snowflake::kumoapi|snowflake|cryptography|pydantic|numpy"
  "packages/kumo-bigquery:src:kumo_bigquery::kumoapi|google"
  "packages/kumo-databricks:src:kumo_databricks::databricks|kumoapi|pydantic"
  "packages/kumo-gcs:src:kumo_gcs::kumoapi|google"
  "packages/kumo-adls:src:kumo_adls::"
  "packages/kumo-s3:src:kumo_s3::botocore|kumoapi|pydantic"
)

# Module names deptry can't infer correctly from the distribution name.
# pkg=a|b means the distribution provides multiple top-level modules.
MODULE_MAP="azure-core=azure,azure-identity=azure,azure-storage-file-datalake=azure,kumo-api=kumoapi,kumo-pql=kumopql,pytorch-frame=torch_frame,faiss-cpu=faiss,faiss-gpu=faiss,pytorch-lightning=pytorch_lightning|lightning_fabric,antlr4-python3-runtime=antlr4,pyyaml=yaml,python-dateutil=dateutil"

overall_status=0

for entry in "${PACKAGES[@]}"; do
  IFS=':' read -r pkg_dir src_root known_first_party extra_exclude extra_ignores <<< "$entry"
  echo "::group::deptry $pkg_dir"

  args=(uvx deptry "$src_root" --ignore DEP002 --known-first-party "$known_first_party"
        --package-module-name-map "$MODULE_MAP")
  if [[ -n "$extra_exclude" ]]; then
    args+=(--exclude "$extra_exclude")
  fi
  if [[ -n "$extra_ignores" ]]; then
    args+=(--per-rule-ignores "DEP001=${extra_ignores}")
  fi

  if ! (cd "$pkg_dir" && "${args[@]}"); then
    echo "::error::dependency-boundary violations in $pkg_dir (see above)"
    overall_status=1
  fi
  echo "::endgroup::"
done

exit "$overall_status"
