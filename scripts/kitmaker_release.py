#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish wheels attached to a GitHub release to PyPI through Kitmaker.

Kitmaker fetches each wheel from its release-asset URL and publishes it to
pypi.org and pypi.nvidia.com. A publication cannot be recalled, so nothing is
uploaded until every check has passed: the Portal project carries the
expected name and owner, each wheel belongs to it, has the expected platform
tag, and is attached to the release with the same SHA-256, and a dry run of
the batch completes. Right before the upload the attachments must still be
the same release assets, by id as well as digest: replacing an asset gives it
a new id, so a file swapped during the dry run is never published. A release
URL names a file rather than its bytes, so once Kitmaker reports the
publication, pypi.org and pypi.nvidia.com must both serve each wheel with the
SHA-256 this run built.

    kitmaker_release.py --project-id 6903 --project kumo-engine \\
        --platform manylinux_2_34_x86_64 --repo NVIDIA/kumo-engine \\
        --tag kumo-engine-v1.0.0 dist/kumo_engine-*.whl

``KITMAKER_API_TOKEN``, ``KITMAKER_PIC`` and ``GITHUB_TOKEN`` come from the
environment.
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote

PORTAL = 'https://kitmaker-portal.nvidia.com/api/v0'
GITHUB = 'https://api.github.com'
INDEXES = ('https://pypi.org/simple', 'https://pypi.nvidia.com')
TERMINAL_OK = {'completed', 'success', 'succeeded'}
TERMINAL_BAD = {'failed', 'error', 'cancelled', 'canceled'}
POLL_SECONDS = 30
POLL_ATTEMPTS = 120
INDEX_ATTEMPTS = 40


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def sha256(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


class Api:
    def __init__(self, base, token):
        self.base = base.rstrip('/')
        self.token = token

    def call(self, method, path, body=None):
        request = urllib.request.Request(
            f'{self.base}{path}',
            data=None if body is None else json.dumps(body).encode(),
            method=method, headers={
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json',
            })
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read() or b'{}')
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')[:500]
            raise SystemExit(f'{method} {path}: HTTP {error.code}: {detail}')
        except urllib.error.URLError as error:
            raise SystemExit(f'{method} {path}: {error.reason}')


def release_uuids(response):
    """The release ids a submission returned, under any of the keys Portal
    has used for them."""
    uuids = []
    for key in ('uuid', 'release_uuid', 'release_uuids'):
        value = response.get(key)
        if isinstance(value, list):
            uuids += value
        elif value:
            uuids.append(value)
    uuids += [
        release['release_uuid'] for release in response.get('releases', [])
        if isinstance(release, dict) and release.get('release_uuid')
    ]
    if not uuids:
        raise SystemExit(f'Portal returned no release id: {response}')
    return sorted(set(map(str, uuids)))


def wait(portal, uuids, what):
    """Poll each release to a terminal state; running out of attempts is a
    failure, since an unfinished release has proved nothing."""
    for uuid in uuids:
        for _ in range(POLL_ATTEMPTS):
            response = portal.call('GET', f'/status/{uuid}')
            status = str(response.get('status', '')).lower()
            if status in TERMINAL_OK:
                print(f'{what}: {status} ({uuid})')
                break
            if status in TERMINAL_BAD:
                raise SystemExit(f'{what}: {status} ({uuid})')
            time.sleep(POLL_SECONDS)
        else:
            raise SystemExit(f'{what}: not finished after '
                             f'{POLL_ATTEMPTS * POLL_SECONDS}s ({uuid})')


def submit(portal, project_id, project, pic, urls, upload):
    payload = [{
        'pic': pic,
        'job_type': 'wheel-release-job',
        'url': url,
        'mirroring_strategy': 'upload-both',
        'upload': upload,
    } for url in urls]
    body = {'project_name': project, 'payload': payload}
    return release_uuids(
        portal.call('POST', f'/projects/{project_id}/releases', body))


def attached(github, repo, tag):
    release = github.call('GET', f'/repos/{repo}/releases/tags/{tag}')
    return {
        asset['name']:
        (asset['id'], asset['browser_download_url'], asset.get('digest'))
        for asset in release.get('assets', [])
        if asset.get('state') == 'uploaded'
    }


def asset_urls(wheels, assets, project, platform):
    """The release-asset URL of each wheel, refusing any wheel that belongs
    to another project, has another platform tag, is not attached, or
    differs from the attached file."""
    urls = []
    for wheel in wheels:
        fields = wheel.name.removesuffix('.whl').split('-')
        if wheel.suffix != '.whl' or len(fields) < 5:
            raise SystemExit(f'{wheel.name} is not a wheel')
        if canonical(fields[0]) != canonical(project):
            raise SystemExit(f'{wheel.name} does not belong to {project}')
        if fields[-1] != platform:
            raise SystemExit(f'{wheel.name} is not tagged {platform}')
        if wheel.name not in assets:
            raise SystemExit(f'{wheel.name} is not attached to the release')
        _, url, digest = assets[wheel.name]
        if digest != sha256(wheel):
            raise SystemExit(f'the attached {wheel.name} is not the file '
                             f'this run built ({digest or "no digest"})')
        urls.append(url)
    return urls


def index_digest(index, project, filename):
    try:
        with urllib.request.urlopen(f'{index}/{canonical(project)}/',
                                    timeout=120) as response:
            page = response.read().decode()
    except urllib.error.URLError:
        return None
    for href in re.findall(r'href="([^"]+)"', page):
        path, _, fragment = href.partition('#')
        if (unquote(path.rsplit('/', 1)[-1]) == filename
                and fragment.startswith('sha256=')):
            return 'sha256:' + fragment.removeprefix('sha256=')
    return None


def verify_published(wheels, project):
    """Wait for each index to serve each wheel, and require the bytes this
    run built."""
    for index in INDEXES:
        for wheel in wheels:
            digest = None
            for _ in range(INDEX_ATTEMPTS):
                digest = index_digest(index, project, wheel.name)
                if digest:
                    break
                time.sleep(POLL_SECONDS)
            if digest != sha256(wheel):
                raise SystemExit(f'{index} serves {wheel.name} as '
                                 f'{digest or "nothing"}, but this run built '
                                 f'{sha256(wheel)}')
            print(f'{index}: {wheel.name} matches')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--project-id', type=int, required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--platform', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--portal', default=PORTAL)
    parser.add_argument('--github', default=GITHUB)
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args(argv)

    env = {
        name: os.environ.get(name, '')
        for name in ('KITMAKER_API_TOKEN', 'KITMAKER_PIC', 'GITHUB_TOKEN')
    }
    missing = [name for name, value in env.items() if not value]
    if missing:
        raise SystemExit(f'not set: {", ".join(missing)}')
    portal = Api(args.portal, env['KITMAKER_API_TOKEN'])
    github = Api(args.github, env['GITHUB_TOKEN'])
    pic = env['KITMAKER_PIC']

    record = portal.call('GET', f'/projects/{args.project_id}')
    if record.get('name') != args.project:
        raise SystemExit(f'Portal project {args.project_id} is '
                         f'{record.get("name")!r}, not {args.project!r}')
    if str(record.get('owner_email', '')).lower() != pic.lower():
        raise SystemExit(f'KITMAKER_PIC is {pic}, but Portal project '
                         f'{args.project_id} is owned by '
                         f'{record.get("owner_email")}')

    def checked():
        assets = attached(github, args.repo, args.tag)
        urls = asset_urls(args.wheels, assets, args.project, args.platform)
        return urls, {wheel.name: assets[wheel.name] for wheel in args.wheels}

    urls, pinned = checked()
    for url in urls:
        print(f'{args.project}: {url}')
    uuids = submit(portal, args.project_id, args.project, pic, urls, False)
    wait(portal, uuids, f'{args.project} dry run')
    if checked()[1] != pinned:
        raise SystemExit('the release assets changed during the dry run')
    uuids = submit(portal, args.project_id, args.project, pic, urls, True)
    wait(portal, uuids, f'{args.project} publication')
    verify_published(args.wheels, args.project)


if __name__ == '__main__':
    main()
