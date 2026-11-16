# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the identity and metadata of one SDK release wheel."""

import argparse
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version

EXPECTED_API_REQUIREMENT = Requirement('kumo-api==3.0.0')


def validate_wheel(path: Path, expected_version: str) -> None:
    distribution, filename_version, _, _ = parse_wheel_filename(path.name)
    expected = Version(expected_version.removeprefix('kumo-sdk-v'))
    if distribution != 'kumoai-studio':
        raise ValueError(f'unexpected wheel distribution: {distribution}')
    if filename_version != expected:
        raise ValueError(
            f'{path.name} version {filename_version} != {expected}')

    with zipfile.ZipFile(path) as wheel:
        metadata_files = [
            name for name in wheel.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f'{path.name} has {len(metadata_files)} METADATA files')
        metadata = Parser().parsestr(
            wheel.read(metadata_files[0]).decode('utf-8'))

    if Version(metadata['Version']) != expected:
        raise ValueError(
            f"METADATA version {metadata['Version']} != {expected}")

    api_requirements = [
        requirement for value in metadata.get_all('Requires-Dist', [])
        if (requirement := Requirement(value)).name == 'kumo-api'
    ]
    if len(api_requirements) != 1:
        raise ValueError('release wheel must have one kumo-api requirement')
    if api_requirements[0] != EXPECTED_API_REQUIREMENT:
        raise ValueError(f'kumo-api requirement {api_requirements[0]} != '
                         f'{EXPECTED_API_REQUIREMENT}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    validate_wheel(args.wheel, args.expected_version)


if __name__ == '__main__':
    main()
