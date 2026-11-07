# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated resources shared by local and protected GitHub executions."""
import re

ACCOUNT = '926922431314'
REGION = 'us-west-2'
BRANCH = 'egu/sagemaker-platform-validation'
ENVIRONMENT = 'sagemaker-mlops-test'
PREFIX = 'kumo-test-sagemaker'
BUCKET = 'kumo-test-dataplane-bucket'
ROOT = 'mlops-validation/sagemaker'
DENIED_ROOT = ROOT + '-denied'
TRACKING_SERVER = 'kumo-mlops-validation'
DOMAIN_NAME = PREFIX
USER_PROFILE = 'kumo-test-platform'
SPACE_NAME = 'kumo-test-notebook'
# Unchanged SageMaker Distribution 3.6.1 CPU image in the test repository.
IMAGE = (f'{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/'
         'kumo-mlops-validation@sha256:'
         'c67c9ea12dcf5faa14496248553f6b441567137a75d533d0733a7671f00cf986')
INSTANCE = 'ml.m5.2xlarge'


def role(name):
    if name not in ('RUNNER', 'PIPELINE', 'EXECUTION', 'MLFLOW'):
        raise ValueError('Unknown test role')
    names = dict(RUNNER='gha-kumo-workflow-tests',
                 PIPELINE='kumo-mlops-validation-sagemaker',
                 EXECUTION='kumo-mlops-validation-sagemaker',
                 MLFLOW='kumo-mlops-validation-mlflow')
    return f'arn:aws:iam::{ACCOUNT}:role/{names[name]}'


def configuration():
    return dict(
        account=ACCOUNT, region=REGION, bucket=BUCKET, root=ROOT,
        execution_role=role('EXECUTION'), pipeline_role=role('PIPELINE'),
        tracking_role=role('MLFLOW'), runner_role=role('RUNNER'),
        tracking_arn=f'arn:aws:sagemaker:{REGION}:{ACCOUNT}:'
        f'mlflow-tracking-server/{TRACKING_SERVER}',
        tracking_server=TRACKING_SERVER, domain_name=DOMAIN_NAME,
        user_profile=USER_PROFILE, space=SPACE_NAME, image=IMAGE,
        kernel='kumo-test-engine', instance=INSTANCE, volume_gib=100,
        tracking_artifact_root=f's3://{BUCKET}/{ROOT}/mlflow-artifacts-final',
        artifact_root=f's3://{BUCKET}/{ROOT}/artifacts',
        denied_artifact_root=f's3://{BUCKET}/{DENIED_ROOT}')


def execution_identity(identity, expected_role):
    match = re.fullmatch(
        rf'arn:aws:sts::{ACCOUNT}:assumed-role/([^/]+)/([^/]+)',
        identity.get('Arn', ''))
    if (identity.get('Account') != ACCOUNT or match is None
            or expected_role != role('EXECUTION')
            or match[1] != expected_role.rsplit('/', 1)[-1]):
        raise ValueError('Notebook must use the dedicated execution role')
    return dict(account=identity['Account'], arn=identity['Arn'],
                expected_role=expected_role)


def test_job(name):
    if not re.fullmatch(r'kumo-test-notebook-[a-f0-9]{12}(?:-[a-zA-Z0-9-]+)?',
                        name):
        raise ValueError('Expected this suite\'s dedicated notebook job')
    return name
