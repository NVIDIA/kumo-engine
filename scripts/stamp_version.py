#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the repository version from ``VERSION`` into every artifact.

The kumo-engine wheel, the Studio images, and the Helm charts release
together under the single PEP 440 version in ``VERSION``. Python sources
take it as-is. Helm needs SemVer, so ``X.Y.Z.devN`` becomes ``X.Y.Z-dev.N``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_NUMBER = r'(?:0|[1-9][0-9]*)'
VERSION_PATTERN = re.compile(
    rf'(?P<release>{_NUMBER}\.{_NUMBER}\.{_NUMBER})(?:\.dev(?P<dev>{_NUMBER}))?'
)

# Charts whose version and appVersion are the release version. kumo-temporal
# only takes the chart version: its appVersion names the Temporal server.
RELEASE_CHARTS = (
    'kumo-stack',
    'kumo-stack/renderer',
    'kumo-helm',
    'kumo-edge',
    'kumo-platform-eks',
    'kumo-platform-aks',
    'kumo-platform-aro',
)

YAML_VERSION = re.compile(r'^version: (?P<value>\S+)$', re.MULTILINE)
YAML_APP_VERSION = re.compile(r'^appVersion: "(?P<value>[^"]*)"$', re.MULTILINE)


@dataclass(frozen=True)
class Site:
    r"""A version field: the ``value`` group of the single match of
    :obj:`pattern` in :obj:`path`, written in SemVer when :obj:`semver`."""
    path: str
    field: str
    pattern: re.Pattern[str]
    semver: bool


SITES = (
    Site(
        'packages/kumo-engine/kumo/engine/_version.py',
        '__version__',
        re.compile(r'^__version__ = "(?P<value>[^"]*)"$', re.MULTILINE),
        semver=False,
    ),
    *(Site(f'helm/{chart}/Chart.yaml', field, pattern, semver=True)
      for chart in RELEASE_CHARTS
      for field, pattern in (('version', YAML_VERSION),
                             ('appVersion', YAML_APP_VERSION))),
    Site('helm/kumo-temporal/Chart.yaml', 'version', YAML_VERSION, semver=True),
    Site('helm/kumo-helm/conformance/contract.yaml', 'version', YAML_VERSION,
         semver=True),
    Site(
        'helm/kumo-stack/renderer/values.yaml',
        'images.kumo.tag',
        re.compile(
            r'^images:\n  kumo:\n(?:    (?!tag:).*\n)*    tag: '
            r'(?P<value>\S+)$', re.MULTILINE),
        semver=True,
    ),
)


def read_versions() -> tuple[str, str]:
    r"""Returns the PEP 440 and SemVer forms of ``VERSION``."""
    version = (REPO_ROOT / 'VERSION').read_text(encoding='utf-8').strip()
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise SystemExit(
            f'VERSION must be X.Y.Z or X.Y.Z.devN, found {version!r}')
    if match['dev'] is None:
        return version, version
    return version, f"{match['release']}-dev.{match['dev']}"


def stamp(check: bool) -> list[str]:
    r"""Rewrites every out-of-date site, or only reports them if
    :obj:`check`. Returns a description of each out-of-date site."""
    pep440, semver = read_versions()
    stale = []
    for site in SITES:
        path = REPO_ROOT / site.path
        text = path.read_text(encoding='utf-8')
        matches = list(site.pattern.finditer(text))
        if len(matches) != 1:
            raise SystemExit(f'{site.path}: expected one {site.field} field, '
                             f'found {len(matches)}')
        match = matches[0]
        expected = semver if site.semver else pep440
        if match['value'] == expected:
            continue
        stale.append(
            f'{site.path} {site.field}: {match["value"]} -> {expected}')
        if not check:
            path.write_text(
                text[:match.start('value')] + expected +
                text[match.end('value'):],
                encoding='utf-8',
            )
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='fail if any site differs from VERSION instead of rewriting it',
    )
    args = parser.parse_args()
    stale = stamp(args.check)
    for line in stale:
        print(line)
    if args.check and stale:
        raise SystemExit('Versions differ from VERSION; run '
                         'scripts/stamp_version.py to update them.')


if __name__ == '__main__':
    main()
