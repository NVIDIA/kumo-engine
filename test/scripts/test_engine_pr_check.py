# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise changed-file selection and fail-closed required-check reporting."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.agent_local

SCRIPT = (Path(__file__).resolve().parents[2] /
          '.github/scripts/engine-pr-check.py')
REVISION = 'a' * 40


class EnginePrCheckTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_cli(self, command, values, success=True):
        args = [sys.executable, str(SCRIPT), command]
        for name, value in values.items():
            path = self.root / (name + '.json')
            path.write_text(json.dumps(value))
            args.extend(['--' + name, str(path)])
        output = self.root / (command + '-result.json')
        args.extend(['--output', str(output)])
        if command == 'select':
            args.extend(['--revision', REVISION])
        result = subprocess.run(args, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return json.loads(output.read_text()) if output.exists() else None

    def select(self, files, metadata=None, success=True):
        return self.run_cli(
            'select', {
                'files-json': files,
                'pr-json': metadata or {
                    'head': {
                        'sha': REVISION
                    },
                    'changed_files': len(files)
                },
            }, success=success)

    def test_documentation_exemption_does_not_select_gpu(self):
        selection = self.select([
            {
                'filename': 'docs/testing.md',
                'status': 'modified'
            },
            {
                'filename': 'README.md',
                'status': 'modified'
            },
        ])
        self.assertTrue(selection['docs_only'])
        for key in ('learning', 'install', 'example', 'behavior', 'source'):
            self.assertIs(selection[key], False, key)
        result = self.run_cli('gate', {
            'selection': selection,
            'needs-json': {}
        })
        self.assertEqual(result['status'], 'PASS')

    def test_runnable_documentation_still_requires_example_and_wheel(self):
        selection = self.select([
            {
                'filename': 'packages/kumo-engine/README.md',
                'status': 'modified'
            },
        ])
        self.assertFalse(selection['learning'])
        self.assertTrue(selection['example'])
        self.run_cli('gate', {
            'selection': selection,
            'needs-json': {}
        }, success=False)
        result = self.run_cli(
            'gate', {
                'selection': selection,
                'needs-json':
                {job: {
                    'result': 'success'
                }
                 for job in ('build', 'gpu')},
            })
        self.assertEqual(result['required_jobs'], ['build', 'gpu'])

    def test_unknown_code_and_shared_dependencies_select_full_checks(self):
        for path in ('new_component/handler.py', 'uv.lock',
                     'packages/kumo-core/src/kumo_core/train/trainer.py',
                     'docs/generate.py', 'ui/src/App.tsx'):
            with self.subTest(path=path):
                selection = self.select([{
                    'filename': path,
                    'status': 'modified'
                }])
                for key in ('learning', 'install', 'behavior', 'source'):
                    self.assertIs(selection[key], True, key)

    def test_packaging_selects_install_and_source(self):
        selection = self.select([{
            'filename': 'packages/kumo-engine/scripts/build_wheel.sh',
            'status': 'modified',
        }])
        self.assertTrue(selection['learning'])
        self.assertTrue(selection['install'])
        self.assertTrue(selection['source'])
        self.assertFalse(selection['behavior'])

    def test_test_only_owner_checks_and_shared_fixture_users(self):
        owner = self.select([{
            'filename': 'test/scripts/test_engine_wheel_smoke.py',
            'status': 'modified',
        }])
        self.assertTrue(owner['learning'])
        self.assertTrue(owner['source'])
        self.assertFalse(owner['behavior'])
        fixture = self.select([{
            'filename': 'packages/kumo-engine/test/conftest.py',
            'status': 'modified',
        }])
        self.assertTrue(fixture['behavior'])
        helper = self.select([{
            'filename': 'packages/kumo-engine/test/wheel_gpu_learning.py',
            'status': 'modified',
        }])
        self.assertTrue(helper['behavior'])

    def test_removed_code_and_code_renamed_to_docs_keep_learning(self):
        for record in ({
                'filename': 'kumo/runtime.py',
                'status': 'removed'
        }, {
                'filename': 'docs/old.md',
                'previous_filename': 'kumo/old.py',
                'status': 'renamed'
        }):
            selection = self.select([record])
            self.assertTrue(selection['learning'])
            self.assertTrue(selection['behavior'])

    def test_stale_revision_and_incomplete_or_invalid_inventory_fail(self):
        files = [{'filename': 'docs/testing.md', 'status': 'modified'}]
        cases = [
            (files, {
                'head': {
                    'sha': 'b' * 40
                },
                'changed_files': 1
            }),
            (files, {
                'head': {
                    'sha': REVISION
                },
                'changed_files': 2
            }),
            (files, {
                'head': {
                    'sha': REVISION
                },
                'changed_files': 3001
            }),
            (files, {
                'head': {
                    'sha': REVISION
                },
                'changed_files': True
            }),
            ([], {
                'head': {
                    'sha': REVISION
                },
                'changed_files': 0
            }),
            (files * 2, {
                'head': {
                    'sha': REVISION
                },
                'changed_files': 2
            }),
            ([{
                'filename': '../README.md',
                'status': 'modified'
            }], None),
            ([{
                'filename': 'README.md',
                'status': 'renamed'
            }], None),
            ([{
                'filename': 'README.md',
                'status': []
            }], None),
        ]
        for records, metadata in cases:
            with self.subTest(records=records, metadata=metadata):
                self.select(records, metadata, success=False)

    def test_selected_jobs_must_all_succeed(self):
        selection = self.select([{
            'filename': 'kumo/runtime.py',
            'status': 'modified'
        }])
        needs = {
            job: {
                'result': 'success'
            }
            for job in ('build', 'gpu', 'source')
        }
        result = self.run_cli('gate', {
            'selection': selection,
            'needs-json': needs
        })
        self.assertEqual(result['required_jobs'], ['build', 'gpu', 'source'])
        for job in needs:
            for state in ('failure', 'skipped', 'cancelled', None):
                with self.subTest(job=job, state=state):
                    changed = dict(needs)
                    changed[job] = {'result': state}
                    report = self.run_cli('gate', {
                        'selection': selection,
                        'needs-json': changed
                    }, success=False)
                    self.assertEqual(report['status'], 'FAIL')
            changed = dict(needs)
            del changed[job]
            self.run_cli('gate', {
                'selection': selection,
                'needs-json': changed
            }, success=False)

    def test_missing_detection_and_tampered_exemption_cannot_pass(self):
        self.run_cli('gate', {'selection': {}, 'needs-json': {}}, success=False)
        selection = self.select([{
            'filename': 'kumo/runtime.py',
            'status': 'modified'
        }])
        selection.update(learning=False, install=False, behavior=False,
                         source=False, docs_only=True)
        self.run_cli('gate', {
            'selection': selection,
            'needs-json': {}
        }, success=False)


if __name__ == '__main__':
    unittest.main()
