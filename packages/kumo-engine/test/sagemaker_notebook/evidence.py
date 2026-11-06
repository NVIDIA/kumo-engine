# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind retained notebook outputs to their actual submitted execution."""
import hashlib
import json
import re
from pathlib import Path

MARKER = 'KUMO_SAGEMAKER_RECEIPT='


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False,
                   default=lambda v: v.isoformat()) + '\n')


def receipt(executed, submitted, spec):
    expected = [c for c in submitted['cells'] if c['cell_type'] == 'code']
    actual = [c for c in executed['cells'] if c['cell_type'] == 'code']
    if len(actual) != len(expected):
        raise ValueError('Executed notebook has different code cells')
    found = []
    for before, after in zip(expected, actual):
        if ''.join(before['source']) != ''.join(after['source']):
            raise ValueError('Executed notebook source differs from submission')
        if after.get('execution_count') is None:
            raise ValueError('Notebook contains an unexecuted code cell')
        for output in after.get('outputs', []):
            if output.get('output_type') == 'error':
                raise ValueError('Notebook execution contains a cell error')
            if output.get('output_type') == 'stream':
                for line in ''.join(output.get('text', '')).splitlines():
                    if line.startswith(MARKER):
                        found.append(json.loads(line[len(MARKER):]))
    if len(found) != 1:
        raise ValueError('Expected one actual notebook execution receipt')
    record = found[0]
    for key in ('attempt', 'source_commit', 'wheel_sha256', 'output_prefix'):
        if record.get(key) != spec[key]:
            raise ValueError('Receipt does not identify this execution: ' + key)
    if not re.fullmatch('[a-f0-9]{12}', record.get('execution_id', '')):
        raise ValueError('Missing kernel execution identity')
    files = record.get('files')
    if not isinstance(files, dict) or 'result.json' not in files:
        raise ValueError('Missing notebook evidence inventory')
    for name, sha in files.items():
        if (not re.fullmatch('[A-Za-z0-9_.-]+', name)
                or not isinstance(sha, str)
                or not re.fullmatch('[a-f0-9]{64}', sha)):
            raise ValueError('Unsafe evidence inventory')
    return record


def collect(s3, bucket, executed, submitted, spec, directory):
    record = receipt(executed, submitted, spec)
    destination = Path(directory) / 'outputs'
    destination.mkdir()
    for name, expected in record['files'].items():
        body = s3.get_object(Bucket=bucket,
                             Key=spec['output_prefix'] + '/' + name)['Body']
        try:
            content = body.read()
        finally:
            body.close()
        if digest(content) != expected:
            raise ValueError('Evidence checksum mismatch: ' + name)
        (destination / name).write_bytes(content)
    report = json.loads((destination / 'result.json').read_text())
    for key in ('attempt', 'source_commit', 'wheel_sha256', 'output_prefix'):
        if report.get(key) != spec[key]:
            raise ValueError('Stored report identifies another execution: ' +
                             key)
    if (report.get('execution_id') != record['execution_id']
            or report.get('test_sources') != spec['test_sources']
            or report.get('payload') != spec['assets']):
        raise ValueError('Stored report differs from executed source/payload')
    save(Path(directory) / 'receipt.json', record)
    return report
