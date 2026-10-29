#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit the complete timeline of one finished GitHub Actions run."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PREFERRED_BUDGET_SECONDS = 1200
BUDGET_SECONDS = 1320
GUARDRAIL_SECONDS = 1500
CONCLUSIONS = {
    'success', 'failure', 'cancelled', 'skipped', 'timed_out',
    'action_required', 'neutral', 'stale', 'startup_failure'
}


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Missing timestamp')
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if date.tzinfo is None:
        raise ValueError('Timestamp lacks a timezone')
    return date.astimezone(timezone.utc)


def job_inventory(payload):
    if isinstance(payload, dict):
        pages = [payload]
    elif isinstance(payload, list) and payload:
        if all(isinstance(row, dict) and 'jobs' in row for row in payload):
            pages = payload
        elif all(isinstance(row, dict) and 'id' in row for row in payload):
            return payload, False
        else:
            raise ValueError('Malformed job inventory')
    else:
        raise ValueError('A nonempty complete job inventory is required')
    counts = [page.get('total_count') for page in pages]
    if (any(type(count) is not int or count < 1 for count in counts)
            or len(set(counts)) != 1
            or any(not isinstance(page.get('jobs'), list) for page in pages)):
        raise ValueError('Malformed paginated job inventory')
    jobs = [job for page in pages for job in page['jobs']]
    if len(jobs) != counts[0]:
        raise ValueError('Incomplete job inventory')
    return jobs, True


def audit(run, payload, revision):
    if not isinstance(revision, str) or not re.fullmatch(
            '[0-9a-f]{40}', revision):
        raise ValueError('A full source revision is required')
    if (not isinstance(run, dict) or run.get('head_sha') != revision
            or run.get('status') != 'completed'
            or run.get('conclusion') not in CONCLUSIONS):
        raise ValueError('The exact revision needs a completed workflow run')
    run_id, attempt = run.get('id'), run.get('run_attempt')
    if (type(run_id) is not int or run_id < 1 or type(attempt) is not int
            or attempt < 1 or not isinstance(run.get('html_url'), str)
            or not run['html_url'].startswith('https://')):
        raise ValueError('Missing run identity or attempt')
    created = timestamp(run.get('created_at'))
    started = timestamp(run.get('run_started_at'))
    if started < created:
        raise ValueError('Workflow started before its creation')
    jobs, counted = job_inventory(payload)
    seen, rows, finished = set(), [], []
    for job in jobs:
        if (not isinstance(job, dict) or type(job.get('id')) is not int
                or job['id'] < 1 or job.get('run_id') != run_id
                or job.get('run_attempt') != attempt
                or job.get('head_sha') != revision
                or job.get('status') != 'completed'
                or job.get('conclusion') not in CONCLUSIONS
                or not isinstance(job.get('name'), str) or not job['name']):
            raise ValueError(
                'Every job must be terminal for this exact attempt')
        if job['id'] in seen:
            raise ValueError('Duplicate job identity')
        seen.add(job['id'])
        queued = timestamp(job.get('created_at'))
        running = timestamp(job.get('started_at'))
        complete = timestamp(job.get('completed_at'))
        if queued < created or complete < created:
            raise ValueError('Job timestamps precede workflow creation')
        skipped = job['conclusion'] == 'skipped'
        if not skipped and not queued <= running <= complete:
            raise ValueError('Job timestamps are out of order')
        # GitHub can stamp a skipped job's completion before its start.
        # It executed no work. Preserve those timestamps without inventing work.
        row = dict(
            id=job['id'], name=job['name'], conclusion=job['conclusion'],
            created_at=job['created_at'], started_at=job['started_at'],
            completed_at=job['completed_at'], queue_seconds=0 if skipped else
            (running - queued).total_seconds(),
            execution_seconds=0 if skipped else
            (complete - running).total_seconds(),
            creation_offset_seconds=(queued - created).total_seconds())
        rows.append(row)
        finished.append(complete)
    final = max(finished)
    if final < started:
        raise ValueError('Jobs finished before the workflow started')
    elapsed = (final - created).total_seconds()
    failures = [
        row['name'] for row in rows
        if row['conclusion'] not in ('success', 'skipped')
    ]
    success = run['conclusion'] == 'success' and not failures
    within_preferred = elapsed <= PREFERRED_BUDGET_SECONDS
    within_budget = elapsed <= BUDGET_SECONDS
    within_guardrail = elapsed < GUARDRAIL_SECONDS
    if within_preferred:
        classification = 'preferred_target'
    elif within_budget:
        classification = 'accepted_target'
    elif within_guardrail:
        classification = 'above_accepted_target'
    else:
        classification = 'guardrail_exceeded'
    return dict(
        schema=2, status='MEASURED', revision=revision, run_id=run_id,
        run_url=run['html_url'], run_attempt=attempt,
        run_conclusion=run['conclusion'], created_at=run['created_at'],
        run_started_at=run['run_started_at'],
        final_job_completed_at=final.isoformat(), elapsed_seconds=elapsed,
        budget_seconds=BUDGET_SECONDS, within_budget=within_budget,
        preferred_budget_seconds=PREFERRED_BUDGET_SECONDS,
        within_preferred_budget=within_preferred,
        guardrail_seconds=GUARDRAIL_SECONDS, within_guardrail=within_guardrail,
        timing_classification=classification,
        timing_policy='Twenty minutes is preferred; twenty-two minutes is '
        'accepted. Above twenty-two minutes misses the accepted target, even '
        'below the strict under-twenty-five-minute guardrail. A pass also '
        'requires a verified complete inventory and successful jobs.',
        passed=counted and success and within_budget and within_guardrail,
        failed_jobs=failures,
        job_count=len(rows), inventory_count_verified=counted, jobs=rows,
        timing_scope='Workflow creation through completion of every job '
        'in the supplied attempt, including queues and final reporting. '
        'Other workflow runs require separate audits.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(json.loads(args.run.read_text()),
                       json.loads(args.jobs.read_text()), args.revision)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = dict(schema=2, status='FAIL', passed=False, error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result),
          file=sys.stdout if result['passed'] else sys.stderr)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
