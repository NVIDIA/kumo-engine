# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a standalone migration script without contacting AWS."""
import argparse
import copy
import json
from pathlib import Path

from policies import plan, studio_plan

SCRIPT = 'update-sagemaker-test-permissions.sh'
DESCRIPTION = (
    'Run `sh update-sagemaker-test-permissions.sh` in AWS CloudShell to create '
    'dedicated test roles and a private validation profile/space in the existing '
    'test domain, then revoke the exact legacy suite grants from shared roles. '
    'The script rejects drift before changes, preserves unrelated permissions '
    'and domain defaults, and launches no app or notebook. It can resume the '
    'original partial deployment by updating only the exact old space-scoped '
    'policies; all other policy differences are rejected.\n')


def standalone(directory, configuration):
    source = Path(__file__).resolve().parent
    configuration = copy.deepcopy(configuration)
    if 'studio' not in configuration:
        configuration['studio'] = studio_plan()
    legacy = json.loads((source / 'legacy_permissions.json').read_text())
    code = (
        '#!/bin/sh\n'
        '# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.\n'
        '# All rights reserved.\n'
        '# SPDX-License-Identifier: Apache-2.0\n'
        '# Run: sh update-sagemaker-test-permissions.sh\n'
        'set -eu\numask 077\n'
        '[ "$#" -eq 0 ] || { echo "Usage: sh $0" >&2; exit 1; }\n'
        'command -v aws >/dev/null\ncommand -v jq >/dev/null\n'
        'handoff_tmp=$(mktemp -d)\n'
        'trap \'rm -rf "$handoff_tmp"\' EXIT\n'
        "trap 'exit 1' HUP INT TERM\n")
    for name, value, delimiter in [('access', configuration,
                                    'KUMO_ACCESS_CONFIG'),
                                   ('legacy', legacy, 'KUMO_LEGACY_CONFIG')]:
        code += ('cat > "$handoff_tmp/' + name + '.json" <<\'' + delimiter +
                 '\'\n' + json.dumps(value, indent=2) + '\n' + delimiter + '\n')
    for name in ('apply_access.sh', 'setup_studio.sh'):
        code += '\n' + (source / name).read_text()
    code += '\napply_access\n'
    target = Path(directory) / SCRIPT
    target.write_text(code)
    target.chmod(0o700)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    target = standalone(directory, plan())
    (directory / 'README.md').write_text(DESCRIPTION)
    print(
        json.dumps(
            dict(status='prepared', script=str(target), aws_mutations=False)))


if __name__ == '__main__':
    main()
