# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finished-run timing includes queues, final jobs, and exact attempts."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.agent_local

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / '.github/scripts/engine-pr-timing.py'
REVISION = 'a' * 40


def recorded_run():
    return {
        'id': 42,
        'head_sha': REVISION,
        'run_attempt': 2,
        'status': 'completed',
        'conclusion': 'success',
        'created_at': '2026-10-29T00:00:00Z',
        'run_started_at': '2026-10-29T00:01:00Z',
        'html_url': 'https://github.com/NVIDIA/kumo-engine/actions/runs/42',
    }


def recorded_jobs():
    return {
        'total_count':
        2,
        'jobs': [
            {
                'id': 101,
                'run_id': 42,
                'head_sha': REVISION,
                'run_attempt': 2,
                'name': 'build',
                'status': 'completed',
                'conclusion': 'success',
                'created_at': '2026-10-29T00:00:00Z',
                'started_at': '2026-10-29T00:05:00Z',
                'completed_at': '2026-10-29T00:12:00Z'
            },
            {
                'id': 102,
                'run_id': 42,
                'head_sha': REVISION,
                'run_attempt': 2,
                'name': 'required',
                'status': 'completed',
                'conclusion': 'success',
                'created_at': '2026-10-29T00:12:00Z',
                'started_at': '2026-10-29T00:19:00Z',
                'completed_at': '2026-10-29T00:20:01Z'
            },
        ]
    }


class TimingTests(unittest.TestCase):
    def run_audit(self, run=None, jobs=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run.json').write_text(
                json.dumps(recorded_run() if run is None else run))
            (root / 'jobs.json').write_text(
                json.dumps(recorded_jobs() if jobs is None else jobs))
            output = root / 'timing.json'
            completed = subprocess.run([
                sys.executable,
                str(SCRIPT), '--run',
                str(root / 'run.json'), '--jobs',
                str(root / 'jobs.json'), '--revision', REVISION, '--output',
                str(output)
            ], capture_output=True, text=True)
            self.assertTrue(output.exists(), completed.stderr)
            return completed, json.loads(output.read_text())

    def test_queue_and_last_job_can_miss_preferred_but_meet_accepted_target(
            self):
        process, result = self.run_audit()
        self.assertEqual(result['elapsed_seconds'], 1201)
        self.assertEqual(result['budget_seconds'], 1320)
        self.assertTrue(result['within_budget'])
        self.assertFalse(result.get('within_preferred_budget'))
        self.assertTrue(result['passed'])
        self.assertEqual(result['jobs'][0]['queue_seconds'], 300)
        self.assertEqual(result['jobs'][0]['execution_seconds'], 420)
        self.assertEqual(result['run_attempt'], 2)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_exact_preferred_budget_equality_passes(self):
        jobs = recorded_jobs()
        jobs['jobs'][1]['completed_at'] = '2026-10-29T00:20:00Z'
        process, result = self.run_audit(jobs=jobs)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result['passed'])
        self.assertEqual(result['elapsed_seconds'], 1200)
        self.assertTrue(result.get('within_preferred_budget'))
        self.assertEqual(result.get('timing_classification'),
                         'preferred_target')

    def test_1277_seconds_passes_accepted_target_without_claiming_twenty(self):
        jobs = recorded_jobs()
        jobs['jobs'][1]['completed_at'] = '2026-10-29T00:21:17Z'
        process, result = self.run_audit(jobs=jobs)
        self.assertTrue(result['passed'])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(result['elapsed_seconds'], 1277)
        self.assertEqual(result['budget_seconds'], 1320)
        self.assertTrue(result['within_budget'])
        self.assertEqual(result.get('schema'), 2)
        self.assertEqual(result.get('preferred_budget_seconds'), 1200)
        self.assertEqual(result.get('guardrail_seconds'), 1500)
        self.assertFalse(result.get('within_preferred_budget'))
        self.assertTrue(result.get('within_guardrail'))
        self.assertEqual(result.get('timing_classification'), 'accepted_target')

    def test_twenty_two_minute_boundary_is_accepted(self):
        jobs = recorded_jobs()
        jobs['jobs'][1]['completed_at'] = '2026-10-29T00:22:00Z'
        process, result = self.run_audit(jobs=jobs)
        self.assertTrue(result['passed'])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result['within_budget'])
        self.assertTrue(result.get('within_guardrail'))
        self.assertEqual(result.get('timing_classification'), 'accepted_target')

    def test_below_guardrail_does_not_claim_the_accepted_target(self):
        for finished in ('2026-10-29T00:22:01Z', '2026-10-29T00:24:59Z'):
            with self.subTest(finished=finished):
                jobs = recorded_jobs()
                jobs['jobs'][1]['completed_at'] = finished
                process, result = self.run_audit(jobs=jobs)
                self.assertFalse(result['passed'])
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(result['within_budget'])
                self.assertTrue(result.get('within_guardrail'))
                self.assertEqual(result.get('timing_classification'),
                                 'above_accepted_target')

    def test_twenty_five_minutes_or_more_fails_the_strict_guardrail(self):
        for finished in ('2026-10-29T00:25:00Z', '2026-10-29T00:25:01Z'):
            with self.subTest(finished=finished):
                jobs = recorded_jobs()
                jobs['jobs'][1]['completed_at'] = finished
                process, result = self.run_audit(jobs=jobs)
                self.assertFalse(result['passed'])
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(result.get('within_guardrail'))
                self.assertEqual(result.get('timing_classification'),
                                 'guardrail_exceeded')

    def test_pending_or_mismatched_records_cannot_claim_a_time(self):
        cases = []
        run = recorded_run()
        run['status'] = 'in_progress'
        cases.append((run, recorded_jobs()))
        run = recorded_run()
        run['head_sha'] = 'b' * 40
        cases.append((run, recorded_jobs()))
        for key, value in [('run_attempt', 1), ('status', 'in_progress'),
                           ('head_sha', 'b' * 40), ('completed_at', None),
                           ('started_at', '2026-10-29T00:30:00Z')]:
            jobs = recorded_jobs()
            jobs['jobs'][0][key] = value
            cases.append((recorded_run(), jobs))
        for run, jobs in cases:
            with self.subTest(run=run, jobs=jobs):
                process, result = self.run_audit(run, jobs)
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(result['status'], 'FAIL')
                self.assertFalse(result['passed'])
                self.assertNotIn('elapsed_seconds', result)

    def test_incomplete_or_repeated_paginated_jobs_fail(self):
        jobs = recorded_jobs()
        jobs['jobs'].pop()
        process, result = self.run_audit(jobs=jobs)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('inventory', result['error'])
        jobs = recorded_jobs()
        jobs['jobs'][1] = copy.deepcopy(jobs['jobs'][0])
        process, result = self.run_audit(jobs=jobs)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Duplicate', result['error'])

    def test_truncated_bare_jobs_cannot_approve_the_budget(self):
        # Omitting the final job cannot approve an unverified inventory.
        first_job_only = recorded_jobs()['jobs'][:1]
        process, result = self.run_audit(jobs=first_job_only)
        self.assertEqual(result['elapsed_seconds'], 720)
        self.assertFalse(result['inventory_count_verified'])
        self.assertFalse(result['passed'])
        self.assertNotEqual(process.returncode, 0)

    def test_all_paginated_job_pages_are_measured(self):
        jobs = recorded_jobs()
        pages = [{'total_count': 2, 'jobs': [job]} for job in jobs['jobs']]
        _, result = self.run_audit(jobs=pages)
        self.assertEqual(result['job_count'], 2)
        self.assertEqual(result['elapsed_seconds'], 1201)

    def test_failed_job_cannot_turn_a_green_workflow_into_a_pass(self):
        jobs = recorded_jobs()
        jobs['jobs'][0]['conclusion'] = 'failure'
        jobs['jobs'][1]['completed_at'] = '2026-10-29T00:20:00Z'
        process, result = self.run_audit(jobs=jobs)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(result['passed'])
        self.assertEqual(result['failed_jobs'], ['build'])


if __name__ == '__main__':
    unittest.main()
