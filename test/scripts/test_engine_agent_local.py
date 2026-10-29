# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/agent_local_tests.py'


class AgentLocalTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'Agent-local entry point is missing')
        self.runner = runpy.run_path(str(SCRIPT))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def execute(self, source, limit=300):
        test = self.root / 'test_selection.py'
        test.write_text('import pytest\n' + source)
        return self.runner['run_tests']([test], self.root / 'results', limit)

    def test_only_marked_tests_run_and_exact_selection_is_recorded(self):
        report = self.execute('''
@pytest.mark.agent_local
def test_selected():
    assert True

def test_unselected():
    raise AssertionError('Unmarked test ran')
''')
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['selected_nodeids']), 1)
        self.assertTrue(
            report['selected_nodeids'][0].endswith('::test_selected'))
        self.assertLess(report['elapsed_seconds'], 300)

    def test_failed_selected_test_cannot_pass(self):
        report = self.execute('''
@pytest.mark.agent_local
def test_failure():
    assert False
''')
        self.assertEqual(report['status'], 'FAIL')
        self.assertGreater(report['failures'], 0)

    def test_skipped_selected_test_cannot_pass(self):
        report = self.execute('''
@pytest.mark.agent_local
def test_skipped():
    pytest.skip('Optional dependency is missing')
''')
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['skipped'], 1)

    def test_empty_marked_selection_cannot_pass(self):
        report = self.execute('def test_not_marked():\n    assert True\n')
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['selected_nodeids'], [])

    def test_early_zero_exit_cannot_hide_an_unfinished_selected_test(self):
        report = self.execute('''
@pytest.mark.agent_local
def test_a_complete():
    assert True

@pytest.mark.agent_local
def test_b_unfinished():
    pytest.exit('Stopped before completing the selection', returncode=0)
''')
        self.assertEqual(report['returncode'], 0)
        self.assertEqual(report['status'], 'FAIL')

    def test_receipt_reporting_time_is_inside_the_deadline(self):
        reported = []
        original = Path.write_text

        def write(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name == 'result.json':
                reported.append(True)
            return result

        clock = SimpleNamespace(monotonic=lambda: 300 if reported else 0)
        with patch.dict(self.runner['run_tests'].__globals__, time=clock), \
                patch.object(Path, 'write_text', write):
            report = self.execute('''
@pytest.mark.agent_local
def test_complete():
    assert True
''')
        self.assertEqual(report['status'], 'FAIL')

    def test_deadline_includes_collection_and_kills_the_whole_run(self):
        report = self.execute('import time\ntime.sleep(10)\n', limit=.5)
        self.assertEqual(report['status'], 'FAIL')
        self.assertTrue(report['timed_out'])
        self.assertLess(report['elapsed_seconds'], 5)

    def test_exact_deadline_is_rejected(self):
        report = dict(returncode=0, selected_nodeids=['test.py::test_ok'],
                      completed_nodeids=['test.py::test_ok'],
                      unmarked_nodeids=[], tests=1, failures=0, skipped=0,
                      timed_out=False, elapsed_seconds=299.999)
        self.assertTrue(self.runner['accepted'](report, 300))
        report['elapsed_seconds'] = 300
        self.assertFalse(self.runner['accepted'](report, 300))
