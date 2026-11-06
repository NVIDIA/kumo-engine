# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a portable admin handoff without contacting AWS."""
import argparse
import json
import shutil
from pathlib import Path

from evidence import save
from policies import plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    proposal = plan()
    save(directory / 'plan.json', proposal)
    for name, record in proposal['roles'].items():
        save(directory / (name.lower() + '-permissions.json'),
             record['permissions'])
    save(directory / 'runner-trust-addition.json',
         proposal['runner_trust_addition'])
    shutil.copyfile(
        Path(__file__).with_name('apply_access.sh'), directory / 'apply.sh')
    (directory / 'README.md').write_text(
        '# SageMaker test IAM additions\n\n'
        'Use AWS admin credentials for account 926922431314. '
        'AWS CloudShell includes AWS CLI and jq. Review the three policy JSON files, '
        'then run `sh apply.sh` from this directory. No Python, repository checkout, '
        'or files on Elvin\'s machine are required.\n\n'
        'The script reads the live runner trust, adds only the protected '
        'sagemaker-mlops-test environment, checks for intervening trust changes, '
        'and adds the kumo-engine-sagemaker-notebook inline policy to each existing '
        'test role. Other inline policies and trust entries are preserved. '
        'Re-running the same bundle is supported.\n\n'
        'Resources: existing test S3 prefix mlops-validation/sagemaker/, '
        'kumo-test-notebook-* jobs and Pipelines, kumo-mlops-validation managed '
        'MLflow server, and the AWS SageMaker Distribution image. Writes to '
        'the suite\'s denied/ prefix are explicitly denied for the negative test. '
        'The added permissions apply to every session of these shared roles.\n\n'
        'These changes enable protected GitHub execution. They do not grant '
        'Engineer SSO PassRole access or native Studio notebook UI access. '
        'No AWS access keys or other credentials are in this bundle.\n')
    archive = shutil.make_archive(str(directory), 'zip', directory.parent,
                                  directory.name)
    print(
        json.dumps(
            dict(status='prepared', directory=str(directory), archive=archive,
                 aws_mutations=False)))


if __name__ == '__main__':
    main()
