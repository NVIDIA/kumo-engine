#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared Engine checks at 10B and 25B with weekly policy."""

import argparse
import importlib.util
import sys
from pathlib import Path


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    path = Path(__file__).with_name('nightly_job.py')
    spec = importlib.util.spec_from_file_location('weekly_shared_job', path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    if arguments and arguments[0] not in ('build', 'cleanup'):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '--configuration', type=Path,
            default=Path(__file__).parents[1] / 'test/weekly/inputs.json')
        parser.add_argument('--tier', choices=('weekly', ), default='weekly')
        selected, remainder = parser.parse_known_args(arguments[1:])
        value = shared.configuration(selected.configuration.read_text())
        if value['format'] != 'kumo-engine-weekly-job-v2':
            raise ValueError(
                'The weekly caller requires a weekly configuration')
        arguments = [
            arguments[0], '--configuration',
            str(selected.configuration), *remainder
        ]
        if arguments[0] in ('run', 'combine'):
            arguments += ['--tier', 'weekly']
    return shared.main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
