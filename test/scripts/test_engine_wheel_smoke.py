# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
import shutil
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

        # Build requirements use another pip process. Keep them isolated too.
        source = self.root / 'source'
        source.mkdir()
        shutil.copyfile(self.wheel(),
                        source / 'kumo_engine-1.2.3-py3-none-any.whl')
        (source / 'pyproject.toml').write_text(
            '[build-system]\nrequires = ["smoke-local-only-dependency==1.2.3"]\n'
            'build-backend = "backend"\nbackend-path = ["."]\n')
        (source / 'backend.py').write_text(
            'import importlib.metadata, pathlib, shutil\n'
            'assert importlib.metadata.version("smoke-local-only-dependency") == "1.2.3"\n'
            'def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n'
            '    wheel = next(pathlib.Path(__file__).parent.glob("*.whl"))\n'
            '    shutil.copyfile(wheel, pathlib.Path(wheel_directory) / wheel.name)\n'
            '    return wheel.name\n')
        log = self.root / 'source-build.log'
        with patch.dict(os.environ, {'PIP_FIND_LINKS': str(self.root)}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.runner.run_command([
                    str(venv / 'bin/python'), '-I', '-m', 'pip', '--isolated',
                    'install', '--no-cache-dir', '--no-index',
                    str(source)
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

    def test_learning_keeps_the_wheel_identity_guard(self):
        output = self.root / 'learning-evidence'
        completed = subprocess.run([
            sys.executable,
            str(RUNNER),
            str(self.wheel(name='wrong')), '--learning', '--output-dir',
            str(output)
        ], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue((output / 'summary.json').is_file(), completed.stderr)
        result = json.loads((output / 'summary.json').read_text())
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['stage'], 'wheel_identity')

    def test_reference_mode_rejects_a_different_wheel_before_installation(self):
        reference_name = json.loads(self.runner.REQUIREMENTS.read_text()
                                    )['reference']['wheel']['filename']
        for version in ('1.2.3', '1.0.0'):
            with self.subTest(version=version):
                output = self.root / f'reference-evidence-{version}'
                wheel = self.wheel(version=version).rename(self.root /
                                                           reference_name)
                completed = subprocess.run([
                    sys.executable,
                    str(RUNNER),
                    str(wheel), '--reference', '--output-dir',
                    str(output)
                ], capture_output=True, text=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertTrue((output / 'summary.json').is_file(),
                                completed.stderr)
                report = json.loads((output / 'summary.json').read_text())
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['stage'], 'reference')
                self.assertIn('approved reference', report['error'])
                self.assertFalse((output / 'install.log').exists())

    def test_missing_or_invalid_requirements_are_rejected(self):
        original = json.loads(self.runner.REQUIREMENTS.read_text())
        invalid = [
            ('fixture', 'seed', None),
            ('fixture', 'entities', 0),
            ('configuration', 'device', 'cuda'),
            ('required_checks', None, []),
            ('reference', 'dependencies', {}),
        ]
        for section, key, value in invalid:
            with self.subTest(section=section, key=key):
                requirements = json.loads(json.dumps(original))
                if key is None:
                    requirements[section] = value
                elif value is None:
                    del requirements[section][key]
                else:
                    requirements[section][key] = value
                path = self.root / 'requirements.json'
                path.write_text(json.dumps(requirements))
                with self.assertRaises(ValueError):
                    self.runner.load_requirements(path)

    def test_reference_evidence_must_pass_and_have_an_exact_fingerprint(self):
        original = json.loads(self.runner.REQUIREMENTS.read_text())
        for section, field, value in [
            ('evidence', 'status', 'FAIL'),
            ('wheel', 'sha256', 'not-a-fingerprint'),
        ]:
            with self.subTest(field=field):
                requirements = json.loads(json.dumps(original))
                requirements['reference'][section][field] = value
                path = self.root / 'requirements.json'
                path.write_text(json.dumps(requirements))
                with self.assertRaises(ValueError):
                    self.runner.load_requirements(path)

    def test_reference_dependency_versions_must_match_the_complete_report(self):
        report = {
            'install': [
                dict(metadata=dict(name='kumo-engine', version='1.0.0')),
                dict(metadata=dict(name='Some_Package', version='1.2.3')),
            ]
        }
        self.runner.verify_reference_dependencies(report,
                                                  {'some-package': '1.2.3'})
        for expected in ({}, {
                'some-package': '9.9.9'
        }, {
                'some-package': '1.2.3',
                'missing': '1.0'
        }):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, 'dependencies'):
                    self.runner.verify_reference_dependencies(report, expected)

    def test_missing_or_failed_workflow_checks_cannot_report_success(self):
        self.runner.verify_workflow_result(
            dict(status='PASS', checks=dict(training='PASS')), ['training'])
        for status, checks in [
            ('FAIL', dict(training='PASS')),
            ('PASS', {}),
            ('PASS', dict(training='FAIL')),
        ]:
            with self.subTest(status=status, checks=checks):
                with self.assertRaisesRegex(ValueError, 'missing or failed'):
                    self.runner.verify_workflow_result(
                        dict(status=status, checks=checks), ['training'])


if __name__ == '__main__':
    unittest.main()
