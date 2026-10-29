#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wait for existing owner-source CI and reject allowed job failures."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKFLOW = '.github/workflows/testing.yml'


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Owner timing lacks a timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Owner timestamp lacks its time zone')
    return parsed


def evaluate_report(run, jobs, revision):
    """Evaluate a complete latest-attempt GitHub run and job inventory."""
    report = dict(status='FAIL', failed_jobs=[])
    try:
        if (not isinstance(run, dict) or run.get('head_sha') != revision
                or run.get('path') != WORKFLOW
                or run.get('event') != 'pull_request'
                or type(run.get('id')) is not int
                or type(run.get('run_attempt')) is not int):
            raise ValueError(
                'Owner workflow identity differs from the PR revision')
        report.update(run_id=run['id'], run_attempt=run['run_attempt'],
                      revision=revision, url=run.get('html_url'))
        if run.get('status') != 'completed':
            report.update(status='PENDING',
                          reason='Owner workflow is unfinished')
            return report
        if run.get('conclusion') != 'success':
            raise ValueError('Owner workflow did not succeed')
        rows = jobs.get('jobs') if isinstance(jobs, dict) else None
        count = jobs.get('total_count') if isinstance(jobs, dict) else None
        if (type(count) is not int or count <= 0 or not isinstance(rows, list)
                or len(rows) != count):
            raise ValueError('Owner job inventory is incomplete or empty')
        seen, completions, executed = set(), [], 0
        for job in rows:
            if (not isinstance(job, dict) or type(job.get('id')) is not int
                    or job['id'] in seen or job.get('run_id') != run['id']
                    or job.get('run_attempt') != run['run_attempt']
                    or job.get('head_sha') != revision):
                raise ValueError(
                    'Owner jobs are duplicate or from another attempt')
            seen.add(job['id'])
            steps = job.get('steps', [])
            if (not isinstance(steps, list)
                    or any(not isinstance(step, dict) for step in steps)):
                raise ValueError('Invalid owner job steps')
            failed_steps = [
                step.get('name') for step in steps
                if step.get('status') != 'completed'
                or step.get('conclusion') not in ('success', 'skipped')
            ]
            if (job.get('status') != 'completed'
                    or job.get('conclusion') not in ('success', 'skipped')
                    or failed_steps):
                report['failed_jobs'].append(
                    dict(id=job['id'], name=job.get('name'),
                         status=job.get('status'),
                         conclusion=job.get('conclusion'),
                         failed_steps=failed_steps))
            if job.get('conclusion') != 'skipped':
                executed += 1
                completions.append(timestamp(job.get('completed_at')))
        if not executed:
            raise ValueError('Owner workflow executed no source jobs')
        created = timestamp(run.get('created_at'))
        updated = timestamp(run.get('updated_at'))
        last = max(completions)
        if last < created or updated < last:
            raise ValueError('Owner workflow timing is inconsistent')
        report.update(
            job_count=count, created_at=run['created_at'],
            run_started_at=run.get('run_started_at'),
            last_job_completed_at=last.isoformat(),
            owner_workflow_seconds=(last - created).total_seconds(),
            owner_workflow_updated_seconds=(updated - created).total_seconds())
        if report['failed_jobs']:
            raise ValueError(
                'Owner jobs or steps failed despite workflow summary')
        report['status'] = 'PASS'
    except (ValueError, TypeError, KeyError) as error:
        report['error'] = str(error)
    return report


def redact(message):
    for key in ('GH_TOKEN', 'GITHUB_TOKEN'):
        if os.environ.get(key):
            message = message.replace(os.environ[key], '[REDACTED]')
    return message[:2000]


def gh_api(endpoint, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Owner source observation reached its deadline')
    completed = subprocess.run(['gh', 'api', '--method', 'GET', endpoint],
                               capture_output=True, text=True,
                               timeout=min(30, remaining))
    if completed.returncode:
        raise RuntimeError(
            redact(completed.stderr.strip()) or 'GitHub API access failed')
    return json.loads(completed.stdout)


def observe(repository, revision, output, timeout_seconds):
    started = time.monotonic()
    deadline = started + timeout_seconds
    evidence = dict(status='RUNNING', repository=repository, revision=revision,
                    workflow=WORKFLOW, polls=[], run=None, job_pages=[])

    def save():
        evidence['waited_seconds'] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2) + '\n')

    save()
    try:
        while time.monotonic() < deadline:
            inventory = gh_api(
                f'repos/{repository}/actions/workflows/testing.yml/runs'
                f'?event=pull_request&head_sha={revision}&per_page=100',
                deadline)
            evidence['run_inventory'] = inventory
            runs = inventory.get('workflow_runs', [])
            if not isinstance(runs, list):
                raise ValueError('Invalid owner workflow inventory')
            matching = [
                run for run in runs
                if run.get('head_sha') == revision and run.get('event') ==
                'pull_request' and run.get('path') == WORKFLOW
            ]
            if not matching:
                raise ValueError(
                    'No owner source PR workflow exists at this revision')
            run = max(matching, key=lambda row: row['id'])
            evidence['run'] = run
            evidence['polls'].append(
                dict(id=run['id'], attempt=run.get('run_attempt'),
                     status=run.get('status'), conclusion=run.get('conclusion'),
                     waited_seconds=time.monotonic() - started))
            if run.get('status') == 'completed':
                rows, total, page = [], None, 1
                evidence['job_pages'] = []
                while total is None or len(rows) < total:
                    response = gh_api(
                        f'repos/{repository}/actions/runs/{run["id"]}'
                        f'/attempts/{run["run_attempt"]}/jobs'
                        f'?per_page=100&page={page}', deadline)
                    evidence['job_pages'].append(response)
                    if (type(response.get('total_count')) is not int
                            or not isinstance(response.get('jobs'), list)):
                        raise ValueError('Invalid owner job page')
                    if total is None:
                        total = response['total_count']
                    if response['total_count'] != total:
                        raise ValueError(
                            'Owner job inventory changed during collection')
                    if not response['jobs']:
                        raise ValueError(
                            'Owner job pagination ended before completion')
                    rows.extend(response['jobs'])
                    page += 1
                # A rerun must not approve an older completed attempt.
                current = gh_api(f'repos/{repository}/actions/runs/{run["id"]}',
                                 deadline)
                evidence['run_after_jobs'] = current
                if current.get('run_attempt') == run['run_attempt']:
                    decision = evaluate_report(
                        current, dict(total_count=total, jobs=rows), revision)
                    evidence.update(decision)
                    if decision['status'] != 'PENDING':
                        break
            save()
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        else:
            raise TimeoutError(
                'Owner source workflow did not finish within the timeout')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError) as error:
        evidence.update(status='FAIL', error=redact(str(error)))
    finally:
        save()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=900)
    options = parser.parse_args()
    if (not re.fullmatch(r'[\w.-]+/[\w.-]+', options.repository)
            or not re.fullmatch('[0-9a-f]{40}', options.revision)
            or not math.isfinite(options.timeout_seconds)
            or options.timeout_seconds <= 0):
        parser.error(
            'Require OWNER/REPO, a full revision and a positive timeout')
    report = observe(options.repository, options.revision, options.output,
                     options.timeout_seconds)
    print(
        json.dumps({
            key: report.get(key)
            for key in ('status', 'run_id', 'waited_seconds',
                        'owner_workflow_seconds', 'error')
        }))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
