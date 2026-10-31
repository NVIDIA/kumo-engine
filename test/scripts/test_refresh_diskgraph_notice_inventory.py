# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts import gen_third_party as notice
from scripts import refresh_diskgraph_notice_inventory as refresher

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
APACHE = "Apache License\nVersion 2.0\n\nCopyright [yyyy] [name of copyright owner]\n"
MIT = "MIT License\n\nCopyright (c) 2024 Example Authors.\n\nPermission is granted.\n"


def _no_metadata() -> dict[str, Any]:
    raise AssertionError("Cargo metadata is needed only for new components")


def _package(name: str, version: str, license_expression: str) -> dict:
    return {
        "license": license_expression,
        "name": name,
        "source": REGISTRY,
        "source_kind": "registry-or-git",
        "version": version,
    }


def _manifest(packages: list[dict]) -> dict:
    artifact = {
        "default_features": True,
        "dependency_kinds": ["normal"],
        "features": [],
        "root": "diskgraph-ingest",
        "targets": ["x86_64-unknown-linux-gnu"],
    }
    return {
        "artifacts": [
            dict(artifact, id=artifact_id, label=artifact_id, packages=packages)
            for artifact_id in refresher.DISKGRAPH_ARTIFACTS
        ],
        "cargo_lock_sha256":
        notice.sha256_text("lock"),
        "schema":
        1,
    }


def _inventory(components: list[dict], texts: dict[str, str]) -> dict:
    artifact_fields = {
        "default_features": True,
        "dependency_kinds": ["normal"],
        "features": [],
        "root": "diskgraph-ingest",
        "targets": ["x86_64-unknown-linux-gnu"],
    }
    return {
        "artifacts": [
            dict(artifact_fields, id=artifact_id, label=artifact_id,
                 components=[item["id"] for item in components])
            for artifact_id in refresher.DISKGRAPH_ARTIFACTS
        ],
        "components":
        components,
        "schema":
        1,
        "texts":
        texts,
        "tools": {
            "spdx_license_list_data": "v3.28.0"
        },
    }


class RefreshDiskgraphNoticeInventoryTest(unittest.TestCase):
    def test_committed_inventory_is_current(self) -> None:
        inventory = notice.load_json(notice.DEFAULT_INVENTORY)
        refreshed = refresher.refresh(
            inventory,
            notice.load_json(notice.DEFAULT_DISKGRAPH_MANIFEST),
            notice.load_json(notice.DEFAULT_POLICY),
            _no_metadata,
        )
        self.assertEqual(refresher.serialize(refreshed),
                         notice.DEFAULT_INVENTORY.read_bytes())

    def test_new_crate_version_replaces_the_old_component(self) -> None:
        old_text = MIT.replace("2024", "2023")
        old = {
            "artifact_ids":
            list(refresher.DISKGRAPH_ARTIFACTS),
            "copyright": ["Copyright (c) 2023 Example Authors"],
            "declared_expression":
            "MIT",
            "ecosystem":
            "rust",
            "elected_licenses": [],
            "expression_source":
            "index/sample-1.0.0/LICENSE",
            "files": [{
                "path": "LICENSE",
                "sha256": notice.sha256_text(old_text),
                "source": "cargo-package",
            }],
            "id":
            "rust:sample@1.0.0",
            "license_expression":
            "MIT",
            "name":
            "sample",
            "origin":
            "https://example.invalid/sample",
            "version":
            "1.0.0",
            "review_reference":
            "OSRB-6649311",
        }
        kept = dict(
            old, id="rust:kept-crate@2.0.0", name="kept_crate", version="2.0.0",
            files=[{
                "path": "LICENSE",
                "sha256": notice.sha256_text(MIT),
                "source": "cargo-package",
            }])
        inventory = _inventory([kept, old], {
            notice.sha256_text(old_text): old_text,
            notice.sha256_text(MIT): MIT,
        })
        manifest = _manifest([
            _package("kept_crate", "2.0.0", "MIT"),
            _package("sample", "1.1.0", "MIT/Apache-2.0"),
        ])
        manifest["artifacts"][1]["packages"] = manifest["artifacts"][1][
            "packages"][:1]

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory) / "index.crates.io-test" / "sample-1.1.0"
            crate.mkdir(parents=True)
            (crate / "Cargo.toml").write_text("[package]\n")
            (crate / "LICENSE-APACHE").write_bytes(
                APACHE.replace("\n", "\r\n").encode())
            (crate / "LICENSE-MIT").write_text(MIT)
            (crate / "NOTICE").write_text("Copyright 2024 Notice Holder\n")
            (crate / "COPYRIGHT").write_text("Not a license text\n")
            metadata = {
                "packages": [{
                    "manifest_path": str(crate / "Cargo.toml"),
                    "name": "sample",
                    "repository": "https://example.invalid/sample",
                    "source": REGISTRY,
                    "version": "1.1.0",
                }]
            }
            refreshed = refresher.refresh(inventory, manifest,
                                          {"allowed_licenses": ["MIT"]},
                                          lambda: metadata)

            root = Path(directory) / "repo"
            third_party = root / "packages" / "diskgraph" / "third_party"
            third_party.mkdir(parents=True)
            (root / "packages" / "diskgraph" / "Cargo.lock").write_text("lock")
            (third_party / "artifact-manifest.json").write_text(
                json.dumps(manifest))
            self.assertEqual(notice._diskgraph_manifest_errors(refreshed, root),
                             [])
            stale = notice._diskgraph_manifest_errors(inventory, root)
            self.assertTrue(stale)
            self.assertTrue(
                all(notice.DISKGRAPH_REFRESH_COMMANDS in error
                    for error in stale), stale)

        components = {item["id"]: item for item in refreshed["components"]}
        self.assertEqual(list(components),
                         ["rust:kept-crate@2.0.0", "rust:sample@1.1.0"])
        self.assertEqual(components["rust:kept-crate@2.0.0"], kept)
        added = components["rust:sample@1.1.0"]
        self.assertEqual(added["artifact_ids"], ["diskgraph-extension"])
        self.assertEqual(added["license_expression"], "MIT OR Apache-2.0")
        self.assertEqual(added["elected_licenses"], ["MIT"])
        self.assertEqual([item["path"] for item in added["files"]],
                         ["LICENSE-APACHE", "LICENSE-MIT", "NOTICE"])
        self.assertEqual(added["expression_source"],
                         "index.crates.io-test/sample-1.1.0/LICENSE-APACHE")
        self.assertEqual(added["copyright"], [
            "Copyright (c) 2024 Example Authors",
            "Copyright 2024 Notice Holder",
        ])
        self.assertEqual(added["review_reference"],
                         refresher.AUTOMATED_REVIEW_REFERENCE)
        self.assertEqual(refreshed["texts"][added["files"][0]["sha256"]],
                         APACHE)
        self.assertNotIn(notice.sha256_text(old_text), refreshed["texts"])
        artifacts = {item["id"]: item for item in refreshed["artifacts"]}
        self.assertEqual(artifacts["diskgraph-extension"]["components"],
                         ["rust:kept-crate@2.0.0", "rust:sample@1.1.0"])
        self.assertEqual(artifacts["diskgraph-ingest"]["components"],
                         ["rust:kept-crate@2.0.0"])

    def test_crate_without_license_file_reuses_reviewed_spdx_text(self) -> None:
        backfilled = {
            "artifact_ids":
            list(refresher.DISKGRAPH_ARTIFACTS),
            "files": [{
                "path": "SPDX-MIT.txt",
                "provenance": {
                    "declared_in": "other-1.0.0/Cargo.toml",
                    "license_id": "MIT",
                    "version": "v3.28.0",
                },
                "sha256": notice.sha256_text(MIT),
                "source": "spdx-backfill",
            }],
            "ecosystem":
            "rust",
            "id":
            "rust:other@1.0.0",
            "name":
            "other",
            "version":
            "1.0.0",
        }
        inventory = _inventory([backfilled], {notice.sha256_text(MIT): MIT})
        manifest = _manifest([
            _package("other", "1.0.0", "MIT"),
            _package("bare", "0.1.0", "MIT"),
            _package("unlicensed", "0.1.0", "Zlib"),
            _package("partial", "0.1.0", "MIT AND Zlib"),
            dict(_package("vendored", "0.1.0", "MIT"),
                 source_kind="vendored-third-party"),
        ])
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            packages = []
            for name in ("bare", "unlicensed", "partial"):
                crate = Path(directory) / "index" / f"{name}-0.1.0"
                crate.mkdir(parents=True)
                packages.append({
                    "manifest_path": str(crate / "Cargo.toml"),
                    "name": name,
                    "source": REGISTRY,
                    "version": "0.1.0",
                })
            with contextlib.redirect_stderr(stderr):
                refreshed = refresher.refresh(
                    inventory, manifest, {"allowed_licenses": ["MIT", "Zlib"]},
                    lambda: {"packages": packages})
        components = {item["id"]: item for item in refreshed["components"]}
        self.assertEqual(components["rust:bare@0.1.0"]["files"],
                         [{
                             "path": "SPDX-MIT.txt",
                             "provenance": {
                                 "declared_in": "bare-0.1.0/Cargo.toml",
                                 "license_id": "MIT",
                                 "version": "v3.28.0",
                             },
                             "sha256": notice.sha256_text(MIT),
                             "source": "spdx-backfill",
                         }])
        self.assertEqual(components["rust:bare@0.1.0"]["expression_source"],
                         "Cargo.toml: license")
        self.assertEqual(components["rust:bare@0.1.0"]["origin"],
                         "https://crates.io/crates/bare")
        # No reviewed Zlib text exists, so validation rejects the components.
        self.assertEqual(components["rust:unlicensed@0.1.0"]["files"], [])
        self.assertEqual(components["rust:partial@0.1.0"]["files"], [])
        self.assertNotIn("rust:vendored@0.1.0", components)
        self.assertEqual(
            stderr.getvalue(),
            "skipping rust:vendored@0.1.0: vendored-third-party source\n")

    def test_crate_without_license_expression_fails_the_refresh(self) -> None:
        package = _package("bare", "0.1.0", "MIT")
        del package["license"]
        with tempfile.TemporaryDirectory() as directory:
            metadata = {
                "packages": [{
                    "manifest_path":
                    str(Path(directory) / "Cargo.toml"),
                    "name":
                    "bare",
                    "source":
                    REGISTRY,
                    "version":
                    "0.1.0",
                }]
            }
            with self.assertRaisesRegex(refresher.RefreshError,
                                        "bare 0.1.0: .*no license expression"):
                refresher.refresh(_inventory([], {}), _manifest([package]),
                                  {"allowed_licenses": ["MIT"]},
                                  lambda: metadata)

    def test_elections_prefer_reviewed_choices_and_fail_closed(self) -> None:
        allowed = {"Apache-2.0", "BSD-2-Clause", "ISC", "MIT", "Unlicense"}
        self.assertEqual(
            refresher.elected_licenses("MIT OR Apache-2.0", allowed),
            ["Apache-2.0"])
        self.assertEqual(
            refresher.elected_licenses("Unlicense OR MIT", allowed), ["MIT"])
        self.assertEqual(
            refresher.elected_licenses("ISC OR BSD-2-Clause", allowed),
            ["BSD-2-Clause"])
        self.assertEqual(
            refresher.elected_licenses("GPL-3.0-only OR LGPL-2.1-only",
                                       allowed), [])
        self.assertEqual(refresher.elected_licenses("MIT", allowed), [])

    def test_copyright_lines_skip_license_prose_and_templates(self) -> None:
        self.assertEqual(
            refresher.copyright_lines([
                ("Copyright [yyyy] [name of copyright owner]\n"
                 "copyright notice that is included in or attached\n"
                 "2. Grant of Copyright License. Subject to the terms\n"
                 "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM\n"
                 "// Copyright 2015 Brian Smith.\n"
                 "Copyright [2015] [Dan Burkert]\n"
                 "© 2016 Bertram Truong\n"),
                "Copyright 2015 Brian Smith\n",
            ]),
            [
                "Copyright 2015 Brian Smith",
                "Copyright [2015] [Dan Burkert]",
                "© 2016 Bertram Truong",
            ],
        )


if __name__ == "__main__":
    unittest.main()
