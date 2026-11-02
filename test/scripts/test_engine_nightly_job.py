# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the managed job's trust, capacity, and cleanup boundaries."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_job.py'
SPEC = importlib.util.spec_from_file_location('nightly_job', SOURCE)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def config():
    return dict(
        format='kumo-engine-nightly-job-v1', min_free_bytes=1024,
        input_manifest_sha256='a' * 64, **{
            key:
            dict(bucket='shared-fixtures', region='us-west-2',
                 key='engine/nightly/' + key, sha256='b' * 64)
            for key in ('inputs', 'reference', 'policy')
        })


@pytest.mark.parametrize('ref,head,accepted', [
    ('refs/heads/main', None, True),
    ('refs/heads/pull-request/137', 'a' * 40, True),
    ('refs/heads/pull-request/137', 'b' * 40, False),
    ('refs/heads/feature', 'a' * 40, False),
    ('refs/pull/137/merge', 'a' * 40, False),
])
def test_only_main_or_the_current_approved_copy_can_reach_private_work(
        ref, head, accepted):
    if accepted:
        job.trusted_ref(ref, 'a' * 40, head, state='open',
                        target='NVIDIA/kumo-engine')
    else:
        with pytest.raises(ValueError):
            job.trusted_ref(ref, 'a' * 40, head, state='open',
                            target='NVIDIA/kumo-engine')


@pytest.mark.parametrize('state,target', [('closed', 'NVIDIA/kumo-engine'),
                                          ('open', 'somewhere/else'),
                                          (None, None)])
def test_copied_ref_also_requires_an_open_pr_targeting_this_repository(
        state, target):
    with pytest.raises(ValueError):
        job.trusted_ref('refs/heads/pull-request/137', 'a' * 40, 'a' * 40,
                        state=state, target=target)


def test_configuration_rejects_missing_and_unsafe_object_bindings():
    value = config()
    assert job.configuration(json.dumps(value)) == value
    for key, damage in [('min_free_bytes', 0), ('input_manifest_sha256', 'bad'),
                        ('inputs', {}), ('policy', None)]:
        with pytest.raises(ValueError):
            job.configuration(json.dumps(dict(value, **{key: damage})))
    value['inputs']['key'] = '../other'
    with pytest.raises(ValueError):
        job.configuration(json.dumps(value))


def test_capacity_uses_free_bytes_even_when_total_disk_is_large(tmp_path):
    free = shutil.disk_usage(tmp_path).free
    with pytest.raises(ValueError, match='free'):
        job.create_work(tmp_path, free + 1)
    assert list(tmp_path.iterdir()) == []


def test_cleanup_removes_only_the_marked_job_and_preserves_evidence(tmp_path):
    work = job.create_work(tmp_path, 1)
    (work / 'env').mkdir()
    (work / 'env/temporary').write_text('temporary')
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'result.json').write_text('{"status":"FAIL"}')
    job.cleanup_work(work, tmp_path)
    assert not work.exists()
    assert (evidence / 'result.json').read_text() == '{"status":"FAIL"}'
    for unsafe in [tmp_path, evidence, tmp_path.parent]:
        with pytest.raises(ValueError):
            job.cleanup_work(unsafe, tmp_path)


def test_cleanup_rejects_a_symlink_to_another_job(tmp_path):
    first = job.create_work(tmp_path, 1)
    second = tmp_path / 'engine-nightly-link'
    second.symlink_to(first, target_is_directory=True)
    with pytest.raises(ValueError):
        job.cleanup_work(second, tmp_path)
    assert first.is_dir()


def test_cleanup_checks_job_ownership_before_removing_any_evidence(tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    evidence = tmp_path / 'other-evidence'
    evidence.mkdir()
    (evidence / '.nightly-evidence').write_text(str(other))
    (evidence / 'valuable.json').write_text('keep')
    result = subprocess.run(
        [sys.executable, str(SOURCE), 'cleanup'],
        env=dict(os.environ, RUNNER_TEMP=str(tmp_path),
                 NIGHTLY_JOB_ROOT=str(other), NIGHTLY_EVIDENCE_UPLOADED='true'),
        capture_output=True, text=True)
    assert result.returncode != 0
    assert (evidence / 'valuable.json').read_text() == 'keep'
