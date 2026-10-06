#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys

MARKER = '<!-- kumo-pr-change-map -->'


def gh(*args: str, payload: dict | None = None) -> object:
    result = subprocess.run(
        ['gh', 'api', *args],
        check=True,
        input=json.dumps(payload) if payload is not None else None,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: write_sync_change_map.py PR_NUMBER')
    repository = os.environ['GITHUB_REPOSITORY']
    number = sys.argv[1]
    pages = gh(f'repos/{repository}/pulls/{number}/files?per_page=100',
               '--paginate', '--slurp')
    files = [item for page in pages for item in page]
    paths = [item['filename'] for item in files]

    groups: dict[str, list[str]] = {}
    for path in paths:
        if path.startswith('packages/diskgraph/'):
            relative = path.removeprefix('packages/diskgraph/')
            group = relative.split('/', 1)[0] if '/' in relative else 'root'
        else:
            group = 'provenance'
        groups.setdefault(group, []).append(path)

    lines = [
        MARKER,
        '## Change map',
        '',
        '| File | Function / scope | What changed | Why |',
        '| --- | --- | --- | --- |',
    ]
    for group, group_paths in sorted(groups.items()):
        rendered = '<br>'.join(f'`{path}`' for path in group_paths)
        if group == 'provenance':
            what = 'Update the mirrored source revision and file manifest.'
            why = 'Keep synchronization identity independently verifiable.'
        else:
            what = f'Synchronize upstream DiskGraph {group} files.'
            why = 'Keep the in-tree build source identical to its authority.'
        scope = ('Revision and manifest data' if group == 'provenance' else
                 f'Upstream {group} files; mechanical import')
        lines.append(f'| {rendered} | {scope} | {what} | {why} |')
    lines.extend([
        '',
        'Necessity: kumo-diskgraph/main advanced and Kumo builds its internal wheel from the mirror.',
        '',
        'Focused verification: every mirrored path, mode, and Git blob matches the recorded upstream revision.',
        '',
        'Stopping condition: the mirror and provenance manifest are exact and no Kumo integration file changes.',
    ])
    body = '\n'.join(lines) + '\n'

    user = gh('user')
    pages = gh(f'repos/{repository}/issues/{number}/comments?per_page=100',
               '--paginate', '--slurp')
    comments = [item for page in pages for item in page]
    existing = next(
        (comment for comment in comments if comment['user']['id'] == user['id']
         and comment['body'].lstrip().startswith(MARKER)), None)
    if existing is None:
        comment = gh(
            f'repos/{repository}/issues/{number}/comments',
            '--method',
            'POST',
            '--input',
            '-',
            payload={'body': body},
        )
    else:
        comment = gh(
            f'repos/{repository}/issues/comments/{existing["id"]}',
            '--method',
            'PATCH',
            '--input',
            '-',
            payload={'body': body},
        )

    pull = gh(f'repos/{repository}/pulls/{number}')
    link = comment['html_url']
    description = pull['body'] or ''
    description = '\n'.join(line for line in description.splitlines()
                            if not line.startswith('Change map: ')).rstrip()
    gh(
        f'repos/{repository}/pulls/{number}',
        '--method',
        'PATCH',
        '--input',
        '-',
        payload={'body': f'{description}\n\nChange map: {link}\n'},
    )


if __name__ == '__main__':
    main()
