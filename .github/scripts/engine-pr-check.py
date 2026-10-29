#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select PR wheel checks and require all selected jobs to pass."""

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

FLAGS = ('learning', 'install', 'example', 'behavior', 'source', 'docs_only')
STATUSES = {
    'added', 'removed', 'modified', 'renamed', 'copied', 'changed', 'unchanged'
}
PACKAGING = {
    'VERSION',
    'packages/kumo-engine/pyproject.toml',
    'packages/kumo-engine/scripts/build_wheel.sh',
    'packages/kumo-engine/scripts/merge_wheels.py',
    'packages/kumo-engine/scripts/check_pypi_requirements.py',
    'scripts/stamp_version.py',
}


def valid_path(value):
    if (not isinstance(value, str) or not value or value.startswith('/')
            or '\\' in value or any(ord(c) < 32 for c in value)
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('Invalid changed-file path')
    return value


def selected_paths(paths):
    selected = dict.fromkeys(FLAGS, False)
    selected['docs_only'] = True
    for name in paths:
        path = PurePosixPath(name)
        documentation = path.suffix.lower() in ('.md', '.rst', '.adoc')
        examples = ('examples/', 'packages/kumo-engine/examples/')
        if (name == 'packages/kumo-engine/README.md'
                or name.startswith(examples)):
            selected['example'] = True
        if documentation:
            continue
        selected.update(learning=True, source=True, docs_only=False)
        if name in PACKAGING:
            selected['install'] = True
        elif (path.name.startswith('test_') and path.suffix == '.py'
              and any(part in ('test', 'tests') for part in path.parts)):
            # Owner tests still get the eight-task check for code PRs.
            continue
        else:
            # Shared helpers, fixtures and unknown impact include consumers.
            selected.update(install=True, behavior=True)
    return selected


def select(files, metadata, revision):
    if (not isinstance(revision, str)
            or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ValueError('A full source revision is required')
    head = metadata.get('head', {}) if isinstance(metadata, dict) else {}
    count = (metadata.get('changed_files')
             if isinstance(metadata, dict) else None)
    if not isinstance(head, dict) or head.get('sha') != revision:
        raise ValueError('The PR head differs from the selected revision')
    if (type(count) is not int or not 1 <= count <= 3000
            or not isinstance(files, list) or len(files) != count):
        raise ValueError('A complete nonempty PR file inventory is required')
    paths, filenames = set(), set()
    for record in files:
        if (not isinstance(record, dict)
                or not isinstance(record.get('status'), str)
                or record['status'] not in STATUSES):
            raise ValueError('Invalid changed-file record')
        name = valid_path(record.get('filename'))
        if name in filenames:
            raise ValueError('Duplicate changed-file record')
        filenames.add(name)
        paths.add(name)
        previous = record.get('previous_filename')
        if record['status'] == 'renamed' and previous is None:
            raise ValueError('A renamed file needs its previous path')
        if previous is not None:
            paths.add(valid_path(previous))
    flags = selected_paths(paths)
    reason = ('Documentation changes only' if flags['docs_only'] else
              'Code changes require all eight learning tasks')
    return dict(schema=1, status='SELECTED', revision=revision, reason=reason,
                changed_files=count, paths=sorted(paths), **flags)


def gate(selection, needs):
    if (not isinstance(selection, dict)
            or type(selection.get('schema')) is not int
            or selection.get('schema') != 1
            or selection.get('status') != 'SELECTED'
            or not isinstance(needs, dict)):
        raise ValueError('A successful change-selection receipt is required')
    revision = selection.get('revision')
    if (not isinstance(revision, str)
            or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ValueError('Selection lacks a full source revision')
    paths = selection.get('paths')
    count = selection.get('changed_files')
    if (type(count) is not int or not 1 <= count <= 3000
            or not isinstance(paths, list)
            or not count <= len(paths) <= count * 2):
        raise ValueError('Selection lacks a complete file inventory')
    paths = [valid_path(path) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError('Selection contains duplicate paths')
    expected = selected_paths(paths)
    if any(
            type(selection.get(key)) is not bool or selection[key] != value
            for key, value in expected.items()):
        raise ValueError('Selection does not match its recorded paths')
    required = []
    gpu_flags = ('learning', 'install', 'example', 'behavior')
    if any(selection[key] for key in gpu_flags):
        required.extend(('build', 'gpu'))
    if selection['source']:
        required.append('source')
    failed = [
        job for job in required if not isinstance(needs.get(job), dict)
        or needs[job].get('result') != 'success'
    ]
    # Unselected jobs can skip, but their failures must remain visible.
    failed.extend(
        job for job, result in needs.items()
        if job not in required and (not isinstance(result, dict) or result.get(
            'result') not in ('success', 'skipped')))
    return dict(schema=1, status='FAIL' if failed else 'PASS',
                revision=revision, required_jobs=required, failed_jobs=failed,
                docs_only=selection['docs_only'], learning_scores_block=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    selection = commands.add_parser('select')
    selection.add_argument('--files-json', type=Path, required=True)
    selection.add_argument('--pr-json', type=Path, required=True)
    selection.add_argument('--revision', required=True)
    selection.add_argument('--output', type=Path, required=True)
    aggregate = commands.add_parser('gate')
    aggregate.add_argument('--selection', type=Path, required=True)
    aggregate.add_argument('--needs-json', type=Path, required=True)
    aggregate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'select':
            files = json.loads(args.files_json.read_text())
            result = select(files, json.loads(args.pr_json.read_text()),
                            args.revision)
        else:
            result = gate(json.loads(args.selection.read_text()),
                          json.loads(args.needs_json.read_text()))
    except (OSError, ValueError) as error:
        result = dict(schema=1, status='FAIL', error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] == 'FAIL':
        print(json.dumps(result), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
