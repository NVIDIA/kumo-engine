# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path

from scripts import gen_third_party as notice


class ThirdPartyNoticeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = notice.load_json(notice.DEFAULT_INVENTORY)
        cls.policy = notice.load_json(notice.DEFAULT_POLICY)

    def test_committed_notice_is_current_and_contains_only_legal_content(
            self) -> None:
        first = notice.render(self.inventory, self.policy)
        self.assertEqual(first, notice.DEFAULT_NOTICE.read_bytes())
        text = first.decode("utf-8")
        self.assertIn("GloVe pretrained word vectors", text)
        self.assertIn("Fabric Toolbox execution-plan visualizer", text)
        self.assertIn("PyTorch Forecasting", text)
        self.assertIn("spaCy English stop words", text)
        self.assertIn("Starlette HTTPException", text)
        self.assertIn("torchtext basic English tokenizer", text)
        for forbidden in (
                "Generation inputs:",
                "Review    :",
                "Evidence  :",
                "Obtained  :",
                "OSRB-",
                "Kumo PR #",
                "input fingerprint",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(notice.PERSONAL_PATH_RE.search(text))
        for digest in self.inventory["texts"]:
            self.assertIn(f"Legal text {digest[:12]}:", text, digest)
        self.assertLess(len(first), 2_000_000)

        backend = notice.render(
            self.inventory,
            self.policy,
            artifact_ids=[
                "backend-antlr4-runtime",
                "backend-baked-glove",
                "diskgraph-extension",
                "diskgraph-ingest",
                "kumo-core-bundled-source",
            ],
        ).decode("utf-8")
        self.assertIn("antlr4-python3-runtime 4.9.3", backend)
        self.assertIn("GloVe pretrained word vectors", backend)
        self.assertIn("PyTorch Forecasting", backend)
        rust_component = min(
            (component for component in self.inventory["components"]
             if "diskgraph-ingest" in component["artifact_ids"]),
            key=lambda component: component["id"])
        self.assertIn(f"{rust_component['name']} {rust_component['version']}",
                      backend)
        self.assertNotIn("pyg_lib 0.7.0", backend)
        self.assertNotIn("Starlette HTTPException", backend)
        self.assertNotIn("CPython 3.12 Linux x86_64 fat wheel", backend)

        sdk = notice.render(
            self.inventory,
            self.policy,
            artifact_ids=["kumo-sdk-bundled-source"],
        ).decode("utf-8")
        self.assertIn("Starlette HTTPException", sdk)
        self.assertNotIn("PyTorch Forecasting", sdk)

        glove = notice.render(
            self.inventory,
            self.policy,
            artifact_ids=["glove-6b-50d-bundled-data"],
        ).decode("utf-8")
        self.assertIn("GloVe pretrained word vectors", glove)
        self.assertIn("Public Domain Dedication and License v1.0", glove)
        self.assertNotIn("Starlette HTTPException", glove)
        self.assertNotIn("backend container", glove)
        self.assertNotIn("fat wheels", glove)

    def test_committed_inventory_closes_artifacts_policy_and_provenance(
            self) -> None:
        self.assertEqual(notice.validate(self.inventory, self.policy), [])
        artifacts = {
            artifact["id"]: set(artifact["components"])
            for artifact in self.inventory["artifacts"]
        }
        self.assertLess(artifacts["diskgraph-ingest"],
                        artifacts["diskgraph-extension"])
        self.assertEqual(artifacts["glove-6b-50d-bundled-data"],
                         {"data:glove-pretrained-vectors@1.0.0"})
        fat_wheel = artifacts["fat-wheel-bundled-python"]
        self.assertEqual(
            {
                notice.canonical_name(component["name"])
                for component in self.inventory["components"]
                if component["id"] in fat_wheel
                and component["ecosystem"] == "python"
            }, {"torch-geometric"})

    def test_fat_wheel_license_expression_covers_selected_components(
            self) -> None:
        # Not pinned: DiskGraph dependency changes alter the Rust obligations,
        # and each new crate is gated by policy.json and human review.
        expression = notice.selected_license_expression(
            self.inventory,
            self.policy,
            artifact_ids=[
                "fat-wheel-bundled-python",
                "diskgraph-extension",
                "diskgraph-ingest",
                "kumo-core-bundled-source",
            ],
            base_licenses=["Apache-2.0", "MIT"],
        )
        terms = expression.split(" AND ")
        # The GloVe data in the fat wheel carries PDDL-1.0.
        self.assertLessEqual({"Apache-2.0", "MIT", "PDDL-1.0"}, set(terms))
        self.assertLessEqual(set(terms), set(self.policy["allowed_licenses"]))

    def test_policy_validation_fails_closed(self) -> None:
        cases = []

        bad_license = copy.deepcopy(self.inventory)
        bad_license["components"][0]["license_expression"] = "GPL-3.0-only"
        bad_license["components"][0]["elected_licenses"] = []
        cases.append((bad_license, self.policy, "disallowed"))

        missing_text = copy.deepcopy(self.inventory)
        missing_text["components"][0]["files"] = []
        cases.append((missing_text, self.policy, "no upstream text"))

        broken_scope = copy.deepcopy(self.inventory)
        broken_scope["components"][0]["artifact_ids"] = []
        cases.append((broken_scope, self.policy, "artifact_ids disagree"))

        stale_exception = copy.deepcopy(self.policy)
        stale_exception["exceptions"].append({
            "review_reference": "TEST",
            "ecosystem": "python",
            "license_expression": "MPL-2.0",
            "name": "gone",
            "obligations": "retain source",
            "reason": "test",
            "version": "1",
        })
        cases.append(
            (self.inventory, stale_exception, "stale policy exceptions"))

        for inventory, policy, expected in cases:
            with self.subTest(expected=expected):
                errors = notice.validate(inventory, policy)
                self.assertTrue(any(expected in error for error in errors),
                                errors)

    def test_derived_source_policy_requires_provenance_or_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "3rdparty" / "notices").mkdir(parents=True)
            policy_path = root / "scripts" / "license_header_policy.json"
            provenance_path = (root / "3rdparty" / "notices" /
                               "source-provenance.json")
            derived_source = {
                "name":
                "sample",
                "paths": ["sample.py"],
                "license_identifier":
                "Apache-2.0 AND MIT",
                "required_copyright_lines":
                ["SPDX-FileCopyrightText: Copyright (c) Sample"],
            }
            provenance = {
                "schema": 1,
                "sources": [{
                    "path": "sample.py",
                    "license_expression": "MIT",
                }],
            }
            policy_path.write_text(
                json.dumps({
                    "version": 1,
                    "derived_sources": [derived_source],
                }))
            provenance_path.write_text(json.dumps(provenance))
            self.assertEqual(notice._derived_source_policy_errors(root), [])

            provenance_path.write_text(
                json.dumps({
                    "schema": 1,
                    "sources": [],
                }))
            errors = notice._derived_source_policy_errors(root)
            self.assertTrue(
                any("lacks workspace provenance" in error for error in errors),
                errors)

            derived_source["notice_file"] = "NOTICE.txt"
            (root /
             "NOTICE.txt").write_text("License: MIT\nCopyright (c) Sample\n")
            policy_path.write_text(
                json.dumps({
                    "version": 1,
                    "derived_sources": [derived_source],
                }))
            self.assertEqual(notice._derived_source_policy_errors(root), [])

    def test_wheel_validation_covers_legal_native_and_merge_inputs(
            self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            third_party = root / "sample-1.0-py3-none-any.whl"
            first_party = root / "kumo_core-1.0-py3-none-any.whl"
            with zipfile.ZipFile(third_party, "w") as archive:
                archive.writestr(
                    "sample-1.0.dist-info/METADATA",
                    "Name: sample\nVersion: 1.0+pt27cu126\nLicense-Expression: MIT\n",
                )
                archive.writestr("sample-1.0.dist-info/licenses/LICENSE",
                                 "terms\n")
                archive.writestr("sample/legal/NOTICE.md", "notice\n")
                archive.writestr("sample/native.so", b"native")
            with zipfile.ZipFile(first_party, "w") as archive:
                archive.writestr("kumo_core-1.0.dist-info/METADATA",
                                 "Name: kumo-core\nVersion: 1.0\n")

            (_, _, files, metadata_sha256,
             native_members) = notice.wheel_record(third_party)
            inventory = {
                "components": [{
                    "name":
                    "sample",
                    "version":
                    "1.0",
                    "ecosystem":
                    "python",
                    "artifact_ids": ["fat-wheel-bundled-python"],
                    "files": [{
                        "path": path,
                        "sha256": digest,
                        "source": "python-wheel",
                    } for path, digest in notice.normalized_legal_files(
                        files).items()],
                    "wheel_metadata_sha256":
                    metadata_sha256,
                    "native_audit": {
                        "members": [{
                            "path": path,
                        } for path in native_members],
                    },
                }]
            }
            components = root / "components.tsv"
            components.write_text(
                f"{third_party.name}\tthird-party\tPyPI\n"
                f"{first_party.name}\tkumo-first-party\tlocal checkout\n",
                encoding="utf-8",
            )
            self.assertEqual(
                notice.validate_wheels(inventory, root, components), [])
            self.assertEqual(set(files), {
                "sample-1.0.dist-info/licenses/LICENSE",
                "sample/legal/NOTICE.md",
            })
            self.assertEqual(native_members, {"sample/native.so"})

            components.write_text(
                f"{third_party.name}\tthird-party\tPyPI\n",
                encoding="utf-8",
            )
            errors = notice.validate_wheels(inventory, root, components)
            self.assertTrue(
                any("unclassified wheels" in error for error in errors), errors)

            inventory["components"][0]["native_audit"]["members"][0][
                "path"] = "sample/missing.so"
            errors = notice.validate_wheels(inventory, root)
            self.assertTrue(
                any("native member inventory differs" in error
                    for error in errors), errors)

    def test_check_mode_rejects_stale_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "THIRD-PARTY.txt"
            path.write_bytes(b"stale\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = notice.main([
                    "--artifact",
                    "kumo-bundled-source",
                    "--artifact",
                    "kumo-core-bundled-source",
                    "--check",
                    str(path),
                ])
            self.assertEqual(result, 1)
            self.assertIn(
                "--artifact kumo-bundled-source --artifact kumo-core-bundled-source --output",
                stderr.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
