# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the real notebook job and fresh-process AWS execution identity."""
import json
import os
import platform
import re
import sys
import time
from pathlib import Path


def receipt_reference(config, record):
    if (not isinstance(record, dict) or set(record)
            != {'attempt', 'source_commit', 'job_arn', 'execution_arn'}
            or not all(isinstance(value, str) for value in record.values())):
        raise ValueError('Job identity receipt has invalid fields')
    attempt = config['attempt']
    pipeline = 'kumo-test-notebook-' + attempt
    prefix = f'arn:aws:sagemaker:{config["region"]}:{config["account"]}'
    if (not re.fullmatch(r'[a-f0-9]{12}', attempt)
            or not re.fullmatch(r'[a-f0-9]{40}', config['source_commit'])
            or config['pipeline_name'] != pipeline or config['job_receipt_key']
            != config['root'] + '/runs/' + attempt + '/job-identity.json'
            or record.get('attempt') != attempt
            or record.get('source_commit') != config['source_commit']
            or not re.fullmatch(
                re.escape(prefix + ':training-job/' + pipeline) +
                r'(?:-[A-Za-z0-9-]+)?', record.get('job_arn', ''))
            or not re.fullmatch(
                re.escape(prefix + ':pipeline/' + pipeline + '/execution/') +
                r'[A-Za-z0-9_-]+', record.get('execution_arn', ''))):
        raise ValueError(
            'Job identity receipt differs from this notebook attempt')
    return record['job_arn']


def discover(config, session):
    from botocore.exceptions import ClientError

    s3 = session.client('s3')
    deadline = time.monotonic() + 90
    while True:
        try:
            body = s3.get_object(Bucket=config['bucket'],
                                 Key=config['job_receipt_key'])['Body']
        except ClientError as error:
            if error.response['Error']['Code'] not in ('404', 'NoSuchKey'):
                raise
        else:
            try:
                content = body.read(8193)
            finally:
                body.close()
            if len(content) > 8192:
                raise ValueError('Job identity receipt exceeds its size budget')
            record = json.loads(content)
            return dict(config, job_arn=receipt_reference(config, record))
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'Controller did not publish this notebook job identity')
        time.sleep(3)


def qualify(config, session=None, *, kernel=False):
    if session is None:
        import boto3
        session = boto3.Session(region_name=config['region'])
    identity = session.client('sts').get_caller_identity()
    account = config['account']
    role = config['execution_role']
    if (identity.get('Account') != account or not re.fullmatch(
            rf'arn:aws:sts::{account}:assumed-role/' + re.escape(
                role.rsplit('/', 1)[-1]) + r'/[^/]+', identity.get('Arn', ''))):
        raise ValueError('Fresh process resolved another AWS execution role')
    if (platform.machine() != 'x86_64' or sys.version_info < (3, 11)
            or platform.libc_ver()[0] != 'glibc'
            or tuple(map(int,
                         platform.libc_ver()[1].split('.'))) < (2, 34)):
        raise ValueError(
            'Notebook requires x86_64, Python >=3.11 and glibc >=2.34')
    if (os.environ.get('SM_JOB_DEF_VERSION') != '1.0' or
            os.environ.get('SM_INPUT_NOTEBOOK_NAME') != config['notebook_name']
            or not Path('/opt/ml/input/config/resourceconfig.json').is_file()):
        raise ValueError('Expected the actual SageMaker NotebookJob runtime')
    input_root = Path(os.environ.get('SM_EXECUTION_INPUT_PATH', '/invalid'))
    source = input_root / config['notebook_name']
    if not source.is_file():
        raise ValueError('Notebook job did not mount its submitted notebook')
    if kernel:
        from IPython import get_ipython
        if get_ipython() is None:
            raise ValueError('Graph execution must use the notebook kernel')
    sm = session.client('sagemaker')
    name = config['job_arn'].rsplit('/', 1)[-1]
    job = sm.describe_training_job(TrainingJobName=name)
    tags = {
        t['Key']: t['Value']
        for t in sm.list_tags(ResourceArn=job['TrainingJobArn'])['Tags']
    }
    if (job['TrainingJobArn'] != config['job_arn'] or job['RoleArn'] != role
            or job['TrainingJobStatus'] != 'InProgress'
            or tags.get('KumoTestAttempt') != config['attempt']
            or tags.get('KumoTestSource') != config['source_commit']
            or job['AlgorithmSpecification']['TrainingImage'] != config['image']
            or job.get('Environment',
                       {}).get('SM_KERNEL_NAME') != config['kernel']):
        raise ValueError('Notebook execution differs from its submitted job')
    expected = f's3://{config["bucket"]}/{config["job_prefix"]}/'
    channels = job.get('InputDataConfig', [])
    if len(channels) != 1 or not channels[0]['DataSource']['S3DataSource'][
            'S3Uri'].startswith(expected):
        raise ValueError('Notebook job inputs are outside this attempt')
    notebook = json.loads(source.read_text())
    if notebook.get('metadata',
                    {}).get('kumo_test_attempt') != config['attempt']:
        raise ValueError('Mounted notebook identifies another attempt')
    return dict(context='sagemaker_notebook', job=name,
                job_arn=job['TrainingJobArn'], user=identity['Arn'],
                account=identity['Account'], execution_role=role,
                python=sys.version, glibc=platform.libc_ver()[1],
                architecture=platform.machine(), kernel=config['kernel'],
                image=config['image'], notebook=config['notebook_name'])
