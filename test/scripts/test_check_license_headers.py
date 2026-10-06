# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path

from scripts import check_license_headers

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY = check_license_headers.load_policy(REPOSITORY_ROOT /
                                           "scripts/license_header_policy.json")
HASH_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SLASH_HEADER = HASH_HEADER.replace("#", "//")
MIT_HASH_HEADER = HASH_HEADER.replace("Apache-2.0", "MIT")
SHADCN_HEADER = """// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT
"""


def test_header_validation_covers_supported_syntax_and_preambles() -> None:
    valid = {
        "example.py":
        HASH_HEADER + "print('ok')\n",
        "example.ts":
        SLASH_HEADER + "export {};\n",
        "example.rs":
        SLASH_HEADER + "#![allow(dead_code)]\nfn main() {}\n",
        "example.g4":
        SLASH_HEADER + "grammar Example;\n",
        "example.css":
        """/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
body {}
""",
        "example.sql":
        HASH_HEADER.replace("#", "--") + "SELECT 1;\n",
        "templates/example.tpl":
        """{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{- define "example" -}}
""",
        "example.html":
        """<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
<html></html>
""",
        "script.sh":
        "#!/usr/bin/env bash\n" + HASH_HEADER + "set -eu\n",
        "component.tsx":
        '"use client";\n' + SLASH_HEADER + "export {};\n",
        "tools/faiss-gpu-amd64/build.Dockerfile":
        "# syntax=docker/dockerfile:1\n" + HASH_HEADER + "FROM scratch\n",
        "packages/kumo-sdk/kumoai_studio/_version.py":
        "__version__ = '1.2.3'\n" + HASH_HEADER,
    }
    for path, text in valid.items():
        assert check_license_headers.validate_text(path, text,
                                                   POLICY) is None, path

    invalid = {
        "missing.py":
        "print('body')\n",
        "body-string.py":
        "print('SPDX-License-Identifier: Apache-2.0')\n",
        "wrong.py":
        MIT_HASH_HEADER + "print('wrong license')\n",
        "late.py":
        "print('body')\n" + HASH_HEADER,
        "duplicate.py":
        HASH_HEADER + HASH_HEADER + "print('body')\n",
        "malformed.html":
        valid["example.html"].replace("-->", "--!>"),
        "build.rs":
        "#![allow(dead_code)]\n" + SLASH_HEADER,
        "doc.rs":
        "//! SPDX-License-Identifier: Apache-2.0\n",
        "tools/faiss-gpu-amd64/build.Dockerfile":
        HASH_HEADER + "# syntax=docker/dockerfile:1\n",
        "packages/kumo-sdk/kumoai_studio/_version.py":
        "print('runs first')\n" + HASH_HEADER,
    }
    for path, text in invalid.items():
        assert check_license_headers.validate_text(path, text,
                                                   POLICY) is not None, path


def test_scoped_and_derived_sources_require_their_actual_licenses() -> None:
    api = "packages/kumo-api/kumoapi/example.py"
    assert check_license_headers.validate_text(api,
                                               MIT_HASH_HEADER + "value = 1\n",
                                               POLICY) is None
    assert check_license_headers.validate_text(api, HASH_HEADER + "value = 1\n",
                                               POLICY) is not None
    assert check_license_headers.validate_text("example.py",
                                               MIT_HASH_HEADER + "value = 1\n",
                                               POLICY) is not None

    shadcn = "packages/kumo-ui-catalog/src/components/ui/button.tsx"
    assert check_license_headers.validate_text(shadcn,
                                               SHADCN_HEADER + "export {};\n",
                                               POLICY) is None
    assert check_license_headers.validate_text(shadcn,
                                               SLASH_HEADER + "export {};\n",
                                               POLICY) is not None
    assert check_license_headers.validate_text("example.ts",
                                               SHADCN_HEADER + "export {};\n",
                                               POLICY) is not None

    upstream = """# SPDX-FileCopyrightText: Copyright (c) 2019, Example Project.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
    assert check_license_headers.validate_text(
        "fork.py", HASH_HEADER + upstream + "print('derived')\n",
        POLICY) is not None


def test_repository_generated_and_vendored_policy_is_applied() -> None:
    generated = [
        ("ui/src/gen/v1/types/tableKind.ts",
         "@generated by Orval — DO NOT EDIT."),
        ("packages/kumo-pql/kumopql/grammar/PQLGrammarParser.py",
         "@generated by ANTLR"),
        ("ui/src/utils/PQLParsing/grammar/PQLGrammarParser.ts",
         "@generated by ANTLR"),
    ]
    for path, marker in generated:
        assert check_license_headers.matching_exclusion(path, POLICY) is None
        text = (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        assert marker in text
        assert check_license_headers.validate_text(path, text, POLICY) is None

    for path in (
            "helm/kumo-temporal/charts/temporal/templates/server-service.yaml",
            "packages/diskgraph/diskgraph-ptrhash/src/lib.rs",
    ):
        exclusion = check_license_headers.matching_exclusion(path, POLICY)
        assert exclusion is not None
        assert exclusion["kind"] == "vendored"


def test_exclusions_are_exact_and_self_cleaning(tmp_path: Path) -> None:
    policy = deepcopy(POLICY)
    policy["license_scopes"] = []
    policy["derived_sources"] = []
    policy["exclusions"] = [{
        "pattern": "generated/**",
        "kind": "generated",
        "license": "Apache-2.0",
        "reason": "Generated fixture.",
    }]
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "missing.py").write_text("print('generated')\n",
                                          encoding="utf-8")
    (generated / "binary.py").write_bytes(b"\xff\xfe")
    errors, scanned, excluded = check_license_headers.check_paths(
        tmp_path, policy, ["generated/missing.py", "generated/binary.py"])
    assert errors == []
    assert scanned == 0
    assert {path
            for path, _ in excluded
            } == {"generated/missing.py", "generated/binary.py"}

    upstream_header = HASH_HEADER.replace("NVIDIA CORPORATION & AFFILIATES",
                                          "Example Project")
    (generated / "late.py").write_text(
        upstream_header + HASH_HEADER + "print('derived')\n", encoding="utf-8")
    (generated / "mit.py").write_text(MIT_HASH_HEADER + "print('vendored')\n",
                                      encoding="utf-8")
    errors, _, _ = check_license_headers.check_paths(
        tmp_path, policy, ["generated/late.py", "generated/mit.py"])
    assert errors == [
        "generated/late.py: excluded source contains an NVIDIA SPDX copyright line outside a valid repository header",
        "generated/mit.py: excluded source contains an NVIDIA SPDX copyright line outside a valid repository header",
    ]

    (generated / "licensed.py").write_text(HASH_HEADER + "print('ok')\n",
                                           encoding="utf-8")
    errors, _, _ = check_license_headers.check_paths(tmp_path, policy,
                                                     ["generated/licensed.py"])
    assert errors == [
        "generated/licensed.py: excluded source satisfies repository header policy; remove the repository header or narrow the exclusion"
    ]

    overlap = deepcopy(policy)
    overlap["exclusions"].append({
        "pattern": "generated/missing.py",
        "kind": "generated",
        "license": "Apache-2.0",
        "reason": "Specific fixture.",
    })
    errors, _, _ = check_license_headers.check_paths(tmp_path, overlap,
                                                     ["generated/missing.py"])
    assert errors == [
        "generated/missing.py: matches multiple license-header exclusions: generated/**, generated/missing.py"
    ]

    errors, _, _ = check_license_headers.check_paths(tmp_path, policy, [])
    assert errors == [
        "license-header exclusion is stale or unnecessary: generated/**"
    ]


def test_scopes_and_derived_classifications_cannot_go_stale_or_overlap(
        tmp_path: Path) -> None:
    scope_policy = deepcopy(POLICY)
    scope_policy["license_scopes"] = [{
        "pattern": "moved-package/**",
        "license_identifier": "MIT",
        "reason": "Fixture.",
    }]
    scope_policy["derived_sources"] = []
    scope_policy["exclusions"] = []
    errors, _, _ = check_license_headers.check_paths(tmp_path, scope_policy, [])
    assert errors == [
        "license scope is stale or matches no candidate: moved-package/**"
    ]

    source = {
        "paths": ["derived.ts"],
        "license_identifier":
        "Apache-2.0 AND MIT",
        "required_copyright_lines":
        ["SPDX-FileCopyrightText: Copyright (c) 2023 shadcn"],
        "provenance":
        "https://example.com/source",
    }
    derived_policy = deepcopy(POLICY)
    derived_policy["license_scopes"] = []
    derived_policy["exclusions"] = []
    derived_policy["derived_sources"] = [dict(source, name="fixture")]
    errors, _, _ = check_license_headers.check_paths(tmp_path, derived_policy,
                                                     [])
    assert errors == [
        "derived-source classification is stale or not a candidate: derived.ts"
    ]

    (tmp_path / "derived.ts").write_text(SHADCN_HEADER + "export {};\n",
                                         encoding="utf-8")
    derived_policy["derived_sources"].append(dict(source, name="duplicate"))
    errors, _, _ = check_license_headers.check_paths(tmp_path, derived_policy,
                                                     ["derived.ts"])
    assert errors == [
        "derived.ts: matches multiple derived-source classifications: fixture, duplicate"
    ]


def test_repository_candidate_tree_passes() -> None:
    errors, scanned, _ = check_license_headers.check_paths(
        REPOSITORY_ROOT,
        POLICY,
        check_license_headers.tracked_files(REPOSITORY_ROOT),
    )
    assert errors == []
    assert scanned > 3_000
