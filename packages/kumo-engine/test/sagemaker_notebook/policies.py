# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IAM documents for dedicated test resources; generating them makes no API calls."""
import re

from config import (
    ACCOUNT,
    BRANCH,
    BUCKET,
    ENVIRONMENT,
    REGION,
    ROOT,
    TRACKING_SERVER,
    role,
)


def statement(actions, resources, *, effect='Allow', condition=None):
    value = dict(Effect=effect, Action=actions, Resource=resources)
    if condition:
        value['Condition'] = condition
    return value


def document(statements):
    return dict(Version='2012-10-17', Statement=statements)


def trust(name):
    if name == 'RUNNER':
        return document([
            dict(
                Effect='Allow', Principal={
                    'Federated': f'arn:aws:iam::{ACCOUNT}:oidc-provider/token.actions.githubusercontent.com'
                }, Action='sts:AssumeRoleWithWebIdentity', Condition=dict(
                    StringEquals={
                        'token.actions.githubusercontent.com:aud':
                        'sts.amazonaws.com',
                        'token.actions.githubusercontent.com:sub':
                        f'repo:NVIDIA@1728152/kumo-engine@1402052651:environment:{ENVIRONMENT}',
                    }))
        ])
    return document([
        dict(Effect='Allow', Principal=dict(Service='sagemaker.amazonaws.com'),
             Action='sts:AssumeRole',
             Condition=dict(StringEquals={'aws:SourceAccount': ACCOUNT}))
    ])


def permissions(name, domain_id=None):
    if domain_id is not None and not re.fullmatch(r'd-[a-z0-9]+', domain_id):
        raise ValueError('Invalid dedicated Studio domain ID')
    base = f'arn:aws:sagemaker:{REGION}:{ACCOUNT}'
    bucket = f'arn:aws:s3:::{BUCKET}'
    jobs = base + ':training-job/kumo-test-notebook-*'
    pipeline = base + ':pipeline/kumo-test-notebook-*'
    server = base + ':mlflow-tracking-server/' + TRACKING_SERVER
    result = []
    if name in ('RUNNER', 'PIPELINE', 'EXECUTION', 'MLFLOW'):
        result += [
            statement(['s3:GetBucketLocation', 's3:ListBucket'], [bucket]),
            statement(['s3:GetObject', 's3:GetObjectAttributes'],
                      [bucket + '/' + ROOT + '/*'])
        ]
    if name in ('RUNNER', 'EXECUTION'):
        result += [
            statement([
                's3:PutObject', 's3:DeleteObject', 's3:AbortMultipartUpload',
                's3:ListMultipartUploadParts'
            ], [bucket + '/' + ROOT + '/*']),
            statement(
                ['s3:PutObject', 's3:DeleteObject', 's3:AbortMultipartUpload'],
                [bucket + '/' + ROOT + '/denied/*'], effect='Deny')
        ]
    if name == 'MLFLOW':
        result += [
            statement(['s3:PutObject', 's3:DeleteObject'],
                      [bucket + '/' + ROOT + '/tracking/*']),
            statement([
                'sagemaker:CreateModelPackageGroup',
                'sagemaker:DescribeModelPackageGroup',
                'sagemaker:CreateModelPackage',
                'sagemaker:DescribeModelPackage',
                'sagemaker:UpdateModelPackage', 'sagemaker:AddTags'
            ], [
                base + ':model-package-group/kumo-test-*',
                base + ':model-package/kumo-test-*/*'
            ])
        ]
    if name == 'EXECUTION':
        result += [
            statement(['sagemaker-mlflow:*'], [server]),
            statement(['sagemaker:DescribeTrainingJob', 'sagemaker:ListTags'],
                      [jobs]),
            statement(['sagemaker:ListTrainingJobs'], ['*']),
            statement([
                'sagemaker:DescribeModelPackage',
                'sagemaker:DescribeModelPackageGroup'
            ], [
                base + ':model-package-group/kumo-test-*',
                base + ':model-package/kumo-test-*/*'
            ]),
            statement(['ecr:GetAuthorizationToken'], ['*']),
            statement([
                'ecr:BatchGetImage', 'ecr:GetDownloadUrlForLayer',
                'ecr:BatchCheckLayerAvailability'
            ], [
                f'arn:aws:ecr:{REGION}:542918446943:'
                'repository/sagemaker-distribution-prod'
            ]),
            statement(['logs:CreateLogStream', 'logs:PutLogEvents'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:'
                'log-group:/aws/sagemaker/TrainingJobs:log-stream:kumo-test-notebook-*'
            ])
        ]
    if name in ('PIPELINE', 'RUNNER'):
        passed = role('EXECUTION') if name == 'PIPELINE' else role('PIPELINE')
        result += [
            statement(['iam:PassRole'], [passed], condition=dict(
                StringEquals={'iam:PassedToService': 'sagemaker.amazonaws.com'
                              }))
        ]
    if name == 'PIPELINE':
        result += [
            statement([
                'sagemaker:CreateTrainingJob', 'sagemaker:DescribeTrainingJob',
                'sagemaker:StopTrainingJob', 'sagemaker:AddTags'
            ], [jobs])
        ]
    if name == 'RUNNER':
        result += [
            statement([
                'sagemaker:CreatePipeline', 'sagemaker:UpdatePipeline',
                'sagemaker:DescribePipeline',
                'sagemaker:StartPipelineExecution',
                'sagemaker:DescribePipelineExecution',
                'sagemaker:ListPipelineExecutionSteps',
                'sagemaker:StopPipelineExecution', 'sagemaker:AddTags'
            ], [pipeline, pipeline + '/*']),
            statement([
                'sagemaker:DescribeTrainingJob', 'sagemaker:ListTags',
                'sagemaker:StopTrainingJob'
            ], [jobs]),
            statement([
                'sagemaker:DescribeMlflowTrackingServer',
                'sagemaker:CreatePresignedMlflowTrackingServerUrl',
                'sagemaker-mlflow:AccessUI'
            ], [server]),
            statement(['logs:DescribeLogStreams'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:'
                'log-group:/aws/sagemaker/TrainingJobs:*'
            ]),
            statement(['logs:GetLogEvents'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:'
                'log-group:/aws/sagemaker/TrainingJobs:'
                'log-stream:kumo-test-notebook-*'
            ])
        ]
        if domain_id:
            result += [
                statement(['sagemaker:DescribeDomain', 'sagemaker:ListTags'],
                          [base + ':domain/' + domain_id]),
                statement(['sagemaker:CreatePresignedDomainUrl'], [
                    base + ':user-profile/' + domain_id + '/kumo-test-platform'
                ])
            ]
    if name not in ('RUNNER', 'PIPELINE', 'EXECUTION', 'MLFLOW'):
        raise ValueError('Unknown test role')
    return document(result)


def plan():
    execution = permissions('EXECUTION')['Statement'] + permissions(
        'PIPELINE')['Statement']
    return dict(
        account=ACCOUNT, region=REGION, bucket=BUCKET,
        github_environment=ENVIRONMENT, branch=BRANCH,
        operation='Add scoped policies to existing roles; preserve existing policies',
        runner_trust_addition=trust('RUNNER')['Statement'][0], roles={
            'RUNNER':
            dict(arn=role('RUNNER'), permissions=permissions('RUNNER')),
            'EXECUTION':
            dict(arn=role('EXECUTION'), permissions=document(execution)),
        })
