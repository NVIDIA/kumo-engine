# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = (Path(__file__).resolve().parents[2] /
           "scripts/check_cargo_license_metadata.py")


@unittest.skipUnless(shutil.which("cargo"), "Cargo is required")
class CargoLicenseMetadataTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/license_header_policy.json").write_text(
            json.dumps({
                "version": 1,
                "license_identifier": "Apache-2.0",
                "exclusions": [{
                    "pattern": "crates/vendor/**",
                    "kind": "vendored",
                    "license": "MIT",
                }],
            }))
        for directory, name, license_line in (
            ("app", "first-party", "license.workspace = true"),
            ("vendor", "external-library", 'license = "MIT"'),
        ):
            crate = self.root / "crates" / directory
            (crate / "src").mkdir(parents=True)
            (crate / "src/lib.rs").write_text("")
            (crate / "Cargo.toml").write_text(
                f'[package]\nname = "{name}"\nversion = "0.1.0"\n'
                f'edition = "2021"\n{license_line}\n')
        (self.root / "crates/Cargo.toml").write_text(
            '[workspace]\nmembers = ["app", "vendor"]\nresolver = "2"\n'
            '[workspace.package]\nlicense = "Apache-2.0"\n')

    def check(self):
        return subprocess.run(
            [
                sys.executable, str(CHECKER), "--root", str(self.root),
                "--manifest-path", "crates/Cargo.toml"
            ],
            capture_output=True,
            text=True,
        )

    def test_inherited_and_vendored_licenses_follow_policy(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_or_incorrect_declarations_fail(self):
        for crate, old, new, expected in (
            ("app", "license.workspace = true", "", "Apache-2.0"),
            ("app", "license.workspace = true", 'license = "MIT"', "Apache-2.0"),
            ("vendor", 'license = "MIT"', 'license = "Apache-2.0"', "MIT"),
        ):
            with self.subTest(crate=crate, declaration=new):
                manifest = self.root / "crates" / crate / "Cargo.toml"
                original = manifest.read_text()
                manifest.write_text(original.replace(old, new))
                try:
                    result = self.check()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"crates/{crate}/Cargo.toml", result.stderr)
                    self.assertIn(f"expected license {expected!r}", result.stderr)
                finally:
                    manifest.write_text(original)
