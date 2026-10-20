# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

SCRIPTS = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/scripts'
RUNNER = SCRIPTS / 'smoke_wheel.py'


class WheelSmokeTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(RUNNER.is_file(), 'The wheel smoke runner is missing')
        sys.path.insert(0, str(SCRIPTS))
        self.runner = importlib.import_module('smoke_wheel')
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def wheel(self, name='kumo-engine', version='1.2.3', requires=''):
        distribution = f'{name.replace("-", "_")}-{version}'
        wheel = self.root / f'{distribution}-py3-none-any.whl'
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
        if requires:
            metadata += f'Requires-Dist: {requires}\n'
        with ZipFile(wheel, 'w') as archive:
            archive.writestr(f'{distribution}.dist-info/METADATA', metadata)
            archive.writestr(
                f'{distribution}.dist-info/WHEEL',
                'Wheel-Version: 1.0\nRoot-Is-Purelib: true\n'
                'Tag: py3-none-any\n')
            archive.writestr(f'{distribution}.dist-info/RECORD', '')
        return wheel

    def test_reads_selected_version_and_rejects_another_distribution(self):
        wheel = self.wheel()
        identity = self.runner.wheel_identity(wheel)
        self.assertEqual(identity['version'], '1.2.3')
        self.assertEqual(identity['sha256'],
                         hashlib.sha256(wheel.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'kumo-engine'):
            self.runner.wheel_identity(self.wheel(name='another-package'))

    def test_install_report_must_match_both_version_and_fingerprint(self):
        identity = {'version': '1.2.3', 'sha256': 'a' * 64}
        engine = {
            'metadata': {
                'name': 'kumo-engine',
                'version': '1.2.3'
            },
            'download_info': {
                'archive_info': {
                    'hashes': {
                        'sha256': 'a' * 64
                    }
                }
            }
        }
        report = {'install': [engine]}
        self.runner.verify_install(report, identity)
        engine['download_info']['archive_info']['hashes']['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.runner.verify_install(report, identity)
        engine['download_info']['archive_info']['hashes']['sha256'] = 'a' * 64
        engine['metadata']['version'] = '9.9.9'
        with self.assertRaisesRegex(ValueError, 'version'):
            self.runner.verify_install(report, identity)

    def test_subprocess_failure_preserves_output_and_exit_code(self):
        log = self.root / 'failed.log'
        with self.assertRaises(subprocess.CalledProcessError) as error:
            self.runner.run_command([
                sys.executable, '-c',
                'print("deliberate failure"); raise SystemExit(7)'
            ], self.root, log)
        self.assertEqual(error.exception.returncode, 7)
        self.assertIn('deliberate failure', log.read_text())

    def test_pip_cannot_resolve_dependencies_from_host_configuration(self):
        self.wheel(name='smoke-local-only-dependency')
        wheel = self.wheel(requires='smoke-local-only-dependency==1.2.3')
        config = self.root / 'pip.conf'
        config.write_text(f'[global]\nfind-links = {self.root}\n')
        venv = self.root / 'venv'
        subprocess.run([sys.executable, '-I', '-m', 'venv',
                        str(venv)], check=True, capture_output=True)
        log = self.root / 'install.log'
        # Stay offline. Only the host configuration can supply this dependency.
        with patch.dict(os.environ, {'PIP_CONFIG_FILE': str(config)}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.runner.run_command([
                    str(venv / 'bin/python'), '-I', '-m', 'pip', '--isolated',
                    'install', '--no-cache-dir', '--no-index',
                    str(wheel)
                ], self.root, log)
        self.assertIn(
            'No matching distribution found for '
            'smoke-local-only-dependency', log.read_text())

    def test_bad_wheel_fails_the_command_and_saves_a_failure_report(self):
        output = self.root / 'evidence'
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel(name='wrong')), '--output-dir',
            str(output)
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['stage'], 'wheel_identity')
        self.assertIn('kumo-engine', report['error'])


if __name__ == '__main__':
    unittest.main()
