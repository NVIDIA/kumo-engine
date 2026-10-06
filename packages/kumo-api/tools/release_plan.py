# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT
"""Plan a Kumo API major or minor release.

The release workflow uses this module to validate its requested release before
creating commits, branches, tags, or GitHub releases. Keeping the decision
logic here makes the destructive workflow deterministic and unit-testable.
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_VERSION_RE = re.compile(
    r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
_RELEASE_BRANCH_RE = re.compile(r'^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
_RELEASE_TAG_RE = re.compile(
    r'^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')


@dataclass(frozen=True, order=True)
class Version:
    """A strict three-component release version."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> 'Version':
        match = _VERSION_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f'Invalid package version {value!r}; expected X.Y.Z without '
                'leading zeroes.')
        return cls(*(int(part) for part in match.groups()))

    @classmethod
    def parse_tag(cls, value: str) -> 'Version | None':
        match = _RELEASE_TAG_RE.fullmatch(value)
        if match is None:
            return None
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}.{self.patch}'

    @property
    def branch(self) -> str:
        return f'v{self.major}.{self.minor}'

    @property
    def tag(self) -> str:
        return f'v{self}'


# Kumo API intentionally moved its next public contract from 0.92 directly to
# 3.0. Keep this exception exact so the planner does not silently permit other
# skipped-major releases.
_APPROVED_MAJOR_RELEASE_PREDECESSORS = {
    Version(3, 0, 0): Version(0, 92, 0),
}


@dataclass(frozen=True)
class ReleasePlan:
    """Validated operations for a major or minor release."""

    current_version: Version
    target_version: Version
    previous_tag: str
    needs_version_update: bool
    dry_run: bool

    @property
    def target_branch(self) -> str:
        return self.target_version.branch

    @property
    def target_tag(self) -> str:
        return self.target_version.tag

    @property
    def mutate_remote(self) -> bool:
        return not self.dry_run

    def github_outputs(self) -> dict[str, str]:
        """Return values consumed by subsequent GitHub Actions steps."""
        return {
            'current_version': str(self.current_version),
            'target_version': str(self.target_version),
            'target_branch': self.target_branch,
            'target_tag': self.target_tag,
            'previous_tag': self.previous_tag,
            'needs_version_update': str(self.needs_version_update).lower(),
            'mutate_remote': str(self.mutate_remote).lower(),
        }


def _parse_requested_branch(value: str) -> Version:
    match = _RELEASE_BRANCH_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f'Invalid requested release {value!r}; expected vX.Y without '
            'leading zeroes.')
    major, minor = (int(part) for part in match.groups())
    return Version(major, minor, 0)


def _release_tags(tags: Iterable[str]) -> dict[Version, str]:
    result = {}
    for tag in tags:
        version = Version.parse_tag(tag)
        if version is not None:
            result[version] = tag
    return result


def _previous_major_tag(target: Version, tags: dict[Version, str]) -> str:
    previous_major = target.major - 1
    candidates = [(version, tag) for version, tag in tags.items()
                  if version.major == previous_major and version.patch == 0]
    if not candidates:
        raise ValueError(
            f'Cannot find a v{previous_major}.Y.0 tag for release notes.')
    return max(candidates)[1]


def plan_release(*, current_version: str, requested_branch: str,
                 existing_branches: Iterable[str],
                 existing_tags: Iterable[str], dry_run: bool) -> ReleasePlan:
    """Validate and plan one major or minor release.

    A normal release advances the version on ``master`` by one minor or one
    major version. An already-bumped major release is also accepted, allowing
    a breaking contract PR to put ``X.0.0`` on ``master`` before its release
    branch and tag exist.
    """
    current = Version.parse(current_version)
    target = _parse_requested_branch(requested_branch)
    branches = set(existing_branches)
    tags = _release_tags(existing_tags)

    if target.branch in branches:
        raise ValueError(f'Release branch {target.branch} already exists.')
    if target in tags:
        raise ValueError(f'Release tag {target.tag} already exists.')

    if current == target:
        if target.major == 0 or target.minor != 0:
            raise ValueError(
                'Only an already-bumped major version X.0.0 can be released '
                'without updating pyproject.toml.')
        if not tags:
            raise ValueError(
                'Cannot validate a major release without prior tags.')
        latest_version = max(tags)
        approved_predecessor = _APPROVED_MAJOR_RELEASE_PREDECESSORS.get(target)
        if approved_predecessor is not None:
            if latest_version != approved_predecessor:
                raise ValueError(
                    f'{latest_version} is not an approved predecessor for '
                    f'already-bumped {target}; expected '
                    f'{approved_predecessor}.')
            previous_tag = tags[approved_predecessor]
        else:
            previous_major = target.major - 1
            if (latest_version >= target
                    or latest_version.major != previous_major):
                raise ValueError(
                    f'Already-bumped {target} is not the next major version '
                    f'after {latest_version}.')
            previous_tag = _previous_major_tag(target, tags)
        needs_version_update = False
    else:
        current_branch = current.branch
        if current_branch not in branches:
            raise ValueError(
                f'Current release branch {current_branch} does not exist.')

        is_next_minor = (target.major == current.major
                         and target.minor == current.minor + 1)
        is_next_major = (target.major == current.major + 1
                         and target.minor == 0)
        if not (is_next_minor or is_next_major):
            raise ValueError(
                f'Requested {target.branch} is not the next minor or major '
                f'release after {current}.')

        if is_next_major:
            previous_tag = _previous_major_tag(target, tags)
        else:
            previous_version = Version(current.major, current.minor, 0)
            try:
                previous_tag = tags[previous_version]
            except KeyError as error:
                raise ValueError(
                    f'Cannot find {previous_version.tag} for release notes.'
                ) from error
        needs_version_update = True

    return ReleasePlan(current_version=current, target_version=target,
                       previous_tag=previous_tag,
                       needs_version_update=needs_version_update,
                       dry_run=dry_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current-version', required=True)
    parser.add_argument('--requested-branch', required=True)
    parser.add_argument('--existing-branch', action='append', default=[])
    parser.add_argument('--existing-tag', action='append', default=[])
    parser.add_argument('--dry-run', choices=('true', 'false'), required=True)
    parser.add_argument('--github-output', type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments and emit GitHub Actions step outputs."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        plan = plan_release(current_version=args.current_version,
                            requested_branch=args.requested_branch,
                            existing_branches=args.existing_branch,
                            existing_tags=args.existing_tag,
                            dry_run=args.dry_run == 'true')
    except ValueError as error:
        parser.error(str(error))

    output_path = args.github_output
    if output_path is None and os.environ.get('GITHUB_OUTPUT'):
        output_path = Path(os.environ['GITHUB_OUTPUT'])
    output = ''.join(f'{key}={value}\n'
                     for key, value in plan.github_outputs().items())
    if output_path is None:
        print(output, end='')
    else:
        with output_path.open('a', encoding='utf-8') as file:
            file.write(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
