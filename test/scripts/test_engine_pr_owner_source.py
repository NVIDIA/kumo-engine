# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner CI must not hide allowed failures behind a green workflow."""

import copy
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.agent_local

SCRIPT = (Path(__file__).resolve().parents[2] /
          '.github/scripts/engine-pr-owner-source.py')
REVISION = '54d73d45aa48a4c841cd8a8cdc9b5a8e8af5b5c5'

# Exact fields from retained GitHub run40830051338 and its failed matrix job.
# The source workflow reported success despite this failed connector suite.
RETAINED_RUN = dict(id=40830051338, path='.github/workflows/testing.yml',
                    head_sha=REVISION, event='pull_request', status='completed',
                    conclusion='success', run_attempt=1,
                    created_at='2026-10-29T06:34:43Z',
                    updated_at='2026-10-29T06:56:26Z',
                    run_started_at='2026-10-29T06:34:43Z')
RETAINED_FAILURE = dict(id=123302914542, run_id=40830051338, run_attempt=1,
                        head_sha=REVISION, status='completed',
                        conclusion='failure',
                        name='pytest (test/config/ test/connector/)',
                        started_at='2026-10-29T06:37:50Z',
                        completed_at='2026-10-29T06:43:06Z')


class OwnerSourceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(),
                        'The owner-source observer is missing')
        self.worker = runpy.run_path(str(SCRIPT))
        self.run = copy.deepcopy(RETAINED_RUN)
        self.job = dict(RETAINED_FAILURE, conclusion='success')
        self.jobs = dict(total_count=1, jobs=[self.job])

    def evaluate(self, run=None, jobs=None):
        return self.worker['evaluate_report'](
            self.run if run is None else run,
            self.jobs if jobs is None else jobs, REVISION)

    def test_saved_green_workflow_with_failed_matrix_is_rejected(self):
        result = self.evaluate(
            jobs=dict(total_count=1, jobs=[RETAINED_FAILURE]))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failed_jobs'][0]['id'], 123302914542)
        self.assertEqual(self.run['conclusion'], 'success')

    def test_success_reports_complete_owner_timeline(self):
        self.job['completed_at'] = '2026-10-29T06:56:25Z'
        result = self.evaluate()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['owner_workflow_seconds'], 1302)
        self.assertEqual(result['owner_workflow_updated_seconds'], 1303)
        self.assertEqual(result['job_count'], 1)

    def test_queued_and_running_owner_workflows_remain_pending(self):
        for state in ('queued', 'in_progress', 'waiting'):
            with self.subTest(state=state):
                self.run.update(status=state, conclusion=None)
                self.assertEqual(self.evaluate()['status'], 'PENDING')

    def test_revision_event_and_workflow_mismatches_are_rejected(self):
        for field, value in (('head_sha', 'a' * 40), ('event', 'push'),
                             ('path', '.github/workflows/another.yml')):
            with self.subTest(field=field):
                changed = dict(self.run, **{field: value})
                self.assertEqual(self.evaluate(run=changed)['status'], 'FAIL')

    def test_failed_or_cancelled_jobs_cannot_be_exempted(self):
        for state in ('failure', 'cancelled', 'timed_out', 'action_required',
                      'neutral', None):
            with self.subTest(state=state):
                self.job['conclusion'] = state
                self.assertEqual(self.evaluate()['status'], 'FAIL')

    def test_hidden_failed_step_is_rejected(self):
        self.job['steps'] = [
            dict(name='Tests', status='completed', conclusion='failure')
        ]
        self.assertEqual(self.evaluate()['status'], 'FAIL')

    def test_skipped_conditional_job_is_allowed_but_all_skipped_is_not(self):
        skipped = dict(self.job, id=self.job['id'] + 1, conclusion='skipped')
        self.jobs = dict(total_count=2, jobs=[self.job, skipped])
        self.assertEqual(self.evaluate()['status'], 'PASS')
        self.assertEqual(
            self.evaluate(jobs=dict(total_count=1, jobs=[skipped]))['status'],
            'FAIL')

    def test_incomplete_duplicate_and_wrong_attempt_metadata_is_rejected(self):
        for jobs in (dict(total_count=2, jobs=[self.job]),
                     dict(total_count=2, jobs=[self.job, self.job]),
                     dict(total_count=1, jobs=[dict(self.job, run_attempt=2)]),
                     dict(total_count=1, jobs=[dict(self.job, run_id=1)]),
                     dict(total_count=1,
                          jobs=[dict(self.job, head_sha='b' * 40)]),
                     dict(total_count=0, jobs=[])):
            with self.subTest(jobs=jobs):
                self.assertEqual(self.evaluate(jobs=jobs)['status'], 'FAIL')

    def test_unfinished_job_in_completed_workflow_is_rejected(self):
        self.job.update(status='in_progress', conclusion=None)
        self.assertEqual(self.evaluate()['status'], 'FAIL')

    def test_missing_timestamps_and_malformed_steps_produce_failure_reports(
            self):
        for job in (dict(self.job,
                         completed_at=None), dict(self.job, steps=['invalid'])):
            with self.subTest(job=job):
                self.assertEqual(
                    self.evaluate(
                        jobs=dict(total_count=1, jobs=[job]))['status'], 'FAIL')

    def observe(self, service, timeout=15):
        clock = [0.0]

        def sleep(seconds):
            self.assertLessEqual(seconds, 10)
            clock[0] += seconds

        observe = self.worker['observe']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'owner.json'
            fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
            with patch.dict(observe.__globals__, gh_api=service,
                            time=fake_time):
                report = observe('NVIDIA/kumo-engine', REVISION, output,
                                 timeout)
            self.assertEqual(json.loads(output.read_text()), report)
        self.assertLessEqual(report['waited_seconds'], timeout)
        return report

    def test_observer_collects_every_job_page_and_preserves_raw_metadata(self):
        rows = [
            dict(self.job, id=self.job['id'] + index) for index in range(101)
        ]

        def service(endpoint, deadline):
            if '/workflows/testing.yml/runs?' in endpoint:
                return dict(workflow_runs=[self.run])
            if '/attempts/1/jobs?' in endpoint:
                page = int(endpoint.rsplit('page=', 1)[-1])
                return dict(total_count=101,
                            jobs=rows[(page - 1) * 100:page * 100])
            if endpoint.endswith('/runs/40830051338'):
                return self.run
            raise AssertionError(endpoint)

        result = self.observe(service)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['job_count'], 101)
        self.assertEqual(len(result['job_pages']), 2)
        self.assertEqual(result['job_pages'][1]['jobs'][0], rows[100])

    def test_missing_run_and_access_denial_fail_without_an_exemption(self):
        missing = self.observe(lambda *args: dict(workflow_runs=[]))
        self.assertEqual(missing['status'], 'FAIL')
        self.assertIn('No owner source PR workflow', missing['error'])

        def forbidden(*args):
            raise RuntimeError('HTTP403: secret-value denied')

        with patch.dict('os.environ', GH_TOKEN='secret-value'):
            denied = self.observe(forbidden)
        self.assertEqual(denied['status'], 'FAIL')
        self.assertIn('HTTP403', denied['error'])
        self.assertNotIn('secret-value', denied['error'])

    def test_pending_run_timeout_is_bounded_and_keeps_last_run(self):
        self.run.update(status='in_progress', conclusion=None)
        result = self.observe(lambda *args: dict(workflow_runs=[self.run]))
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('timeout', result['error'])
        self.assertEqual(result['waited_seconds'], 15)
        self.assertEqual(result['run']['head_sha'], REVISION)


if __name__ == '__main__':
    unittest.main()
