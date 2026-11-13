# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IAM documents for dedicated test resources; generating them makes no API calls."""
import json
import re
from pathlib import Path

from config import (
    ACCOUNT,
    BUCKET,
    DENIED_ROOT,
    DOMAIN_NAME,
    ENVIRONMENT,
    INSTANCE,
    REGION,
    ROOT,
    SPACE_NAME,
    TRACKING_SERVER,
    USER_PROFILE,
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
                    'Federated':
                    f'arn:aws:iam::{ACCOUNT}:oidc-provider/token.actions.githubusercontent.com'
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
             Condition=dict(
                 StringEquals={'aws:SourceAccount': ACCOUNT},
                 ArnLike={'aws:SourceArn': (
                     f'arn:aws:sagemaker:{REGION}:{ACCOUNT}:pipeline/kumo-test-notebook-*/*'
                     if name == 'PIPELINE' else
                     f'arn:aws:sagemaker:{REGION}:{ACCOUNT}:*')}))
    ])


def mlflow_ui_reads():
    return ['sagemaker-mlflow:' + action for action in (
        'GetRun', 'GetExperiment', 'GetMetricHistory', 'GetRegisteredModel',
        'GetModelVersion', 'SearchModelVersions', 'SearchRuns',
        'SearchExperiments', 'SearchRegisteredModels', 'ListArtifacts')]


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
            statement(['s3:GetBucketLocation'], [bucket]),
            statement(['s3:ListBucket'], [bucket], condition={
                'StringLike': {'s3:prefix': ROOT + '/*'}
            }),
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
                [bucket + '/' + DENIED_ROOT + '/*'], effect='Deny')
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
            statement(
                mlflow_ui_reads() + [
                    'sagemaker-mlflow:' + action for action in
                    ('CreateExperiment', 'GetExperimentByName',
                     'SetExperimentTag', 'CreateRun', 'UpdateRun', 'LogBatch',
                     'LogInputs', 'LogOutputs', 'GetLoggedModel',
                     'SearchLoggedModels', 'FinalizeLoggedModel', 'LogModel',
                     'UpdateModelVersion', 'LogMetric', 'LogParam', 'SetTag',
                     'CreateRegisteredModel', 'CreateModelVersion',
                     'SetRegisteredModelTag', 'SetModelVersionTag',
                     'SetRegisteredModelAlias', 'GetModelVersionByAlias',
                     'GetDownloadURIForModelVersionArtifacts')
                ], [server]),
            statement(['sagemaker:DescribeTrainingJob', 'sagemaker:ListTags'],
                      [jobs]),
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
                f'arn:aws:ecr:{REGION}:{ACCOUNT}:'
                'repository/kumo-mlops-validation'
            ]),
            statement(['logs:CreateLogStream', 'logs:PutLogEvents'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:'
                'log-group:/aws/sagemaker/TrainingJobs:log-stream:kumo-test-notebook-*'
            ]),
            statement(['logs:CreateLogGroup', 'logs:DescribeLogStreams'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:'
                'log-group:/aws/sagemaker/TrainingJobs:*'
            ]),
            statement(['cloudwatch:PutMetricData'], ['*'], condition={
                'StringLike': {
                    'cloudwatch:namespace': '/aws/sagemaker/*'
                }
            })
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
                'sagemaker:DescribeTrainingJob', 'sagemaker:StopTrainingJob',
                'sagemaker:AddTags'
            ], [jobs]),
            statement(
                ['sagemaker:CreateTrainingJob'], [jobs], condition={
                    'ForAllValues:StringEquals': {
                        'sagemaker:InstanceTypes': [INSTANCE]
                    },
                    'NumericLessThanEquals': {
                        'sagemaker:MaxRuntimeInSeconds': 3600
                    },
                    'Null': {
                        'sagemaker:InstanceTypes': 'false',
                        'sagemaker:MaxRuntimeInSeconds': 'false'
                    },
                })
        ]
    if name == 'RUNNER':
        fixture = json.loads((Path(__file__).parent.parent / 'pr172/inputs.json').read_text())
        result += [statement(['s3:GetObject'], [
            'arn:aws:s3:::' + fixture['bucket'] + '/' + fixture['key']])]
        result += [statement(
            mlflow_ui_reads() + ['sagemaker-mlflow:' + action for action in (
                'GetExperimentByName', 'GetExperiment', 'CreateExperiment',
                'CreateRegisteredModel', 'DeleteExperiment', 'DeleteRegisteredModel')],
            [server])]
        result += [
            statement(['s3:GetObject'],
                      [bucket + '/' + DENIED_ROOT + '/README.txt'])
        ]
        result += [
            statement([
                'sagemaker:CreatePipeline', 'sagemaker:UpdatePipeline',
                'sagemaker:DescribePipeline', 'sagemaker:DeletePipeline',
                'sagemaker:ListPipelineExecutions', 'sagemaker:ListTags',
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
                    base + ':user-profile/' + domain_id + '/' + USER_PROFILE
                ])
            ]
    if name not in ('RUNNER', 'PIPELINE', 'EXECUTION', 'MLFLOW'):
        raise ValueError('Unknown test role')
    return document(result)


def plan():
    return dict(
        account=ACCOUNT, region=REGION, bucket=BUCKET,
        github_environment=ENVIRONMENT,
        operation='Create dedicated environment-only roles and revoke legacy suite grants',
        mlflow_ui_permissions=document([statement(mlflow_ui_reads(), [
            f'arn:aws:sagemaker:{REGION}:{ACCOUNT}:mlflow-tracking-server/{TRACKING_SERVER}'])]),
        roles={name: dict(arn=role(name), trust=trust(name),
                          permissions=permissions(name))
               for name in ('RUNNER', 'PIPELINE', 'EXECUTION')})


def studio_plan(vpc_id=None, subnet_ids=None):
    if (vpc_id is not None
            and not re.fullmatch(r'vpc-(?:[a-f0-9]{8}|[a-f0-9]{17})', vpc_id)):
        raise ValueError('Invalid Studio VPC ID')
    if ((vpc_id is None) != (subnet_ids is None)
            or (subnet_ids is not None and (not subnet_ids
                or len(set(subnet_ids)) != len(subnet_ids)
                or any(not re.fullmatch(r'subnet-(?:[a-f0-9]{8}|[a-f0-9]{17})', s)
                       for s in subnet_ids)))):
        raise ValueError('Provide distinct Studio subnet IDs')
    domain_id = '__KUMO_TEST_DOMAIN_ID__'
    base = f'arn:aws:sagemaker:{REGION}:{ACCOUNT}'
    domain = base + ':domain/' + domain_id
    profile = base + ':user-profile/' + domain_id + '/' + USER_PROFILE
    space = base + ':space/' + domain_id + '/' + SPACE_NAME
    app = base + ':app/' + domain_id + '/' + SPACE_NAME + '/JupyterLab/default'
    read = [
        statement(['sagemaker:DescribeDomain'], [domain]),
        statement(['sagemaker:DescribeUserProfile'], [profile]),
        statement(['sagemaker:DescribeSpace'], [space]),
        statement(['sagemaker:DescribeApp'], [app]),
        statement(['sagemaker:ListTags'], [domain, profile, space, app]),
    ]
    runner = document(read + [
        statement(['sagemaker:CreatePresignedDomainUrl'], [profile]),
        statement(['sagemaker:CreateApp'], [app], condition={
            'ForAllValues:StringEquals': {'sagemaker:InstanceTypes': [INSTANCE]},
            'Null': {'sagemaker:InstanceTypes': 'false'},
        }),
        statement(['sagemaker:DeleteApp', 'sagemaker:AddTags'], [app]),
        statement(['sagemaker:DescribeMlflowTrackingServer',
                   'sagemaker:CreatePresignedMlflowTrackingServerUrl',
                   'sagemaker-mlflow:AccessUI'],
                  [base + ':mlflow-tracking-server/' + TRACKING_SERVER]),
    ])
    tags = [dict(Key='KumoTestSuite', Value='kumo-engine-sagemaker-studio')]
    user = dict(ExecutionRole=role('EXECUTION'), AutoMountHomeEFS='Disabled',
                DefaultLandingUri='studio::', StudioWebPortal='ENABLED',
                JupyterLabAppSettings=dict(AppLifecycleManagement=dict(
                    IdleSettings=dict(LifecycleManagement='ENABLED',
                                      IdleTimeoutInMinutes=60,
                                      MinIdleTimeoutInMinutes=60,
                                      MaxIdleTimeoutInMinutes=60))),
                SharingSettings=dict(NotebookOutputOption='Disabled'),
                SpaceStorageSettings=dict(DefaultEbsStorageSettings=dict(
                    DefaultEbsVolumeSizeInGb=100, MaximumEbsVolumeSizeInGb=100)))
    return dict(
        region=REGION,
        placeholder=domain_id, runner_permissions=runner,
        execution_permissions=document(read + [
            statement(['logs:DescribeLogStreams'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/sagemaker/studio:*']),
            statement(['logs:CreateLogStream', 'logs:PutLogEvents'], [
                f'arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/sagemaker/studio:'
                f'log-stream:{domain_id}/{SPACE_NAME}/JupyterLab/default*'])]),
        domain=dict(DomainName=DOMAIN_NAME, AuthMode='IAM',
                    AppNetworkAccessType='PublicInternetOnly',
                    **(dict(VpcId=vpc_id, SubnetIds=subnet_ids) if vpc_id else {}),
                    DefaultUserSettings=user,
                    DefaultSpaceSettings=dict(ExecutionRole=role('EXECUTION')),
                    Tags=tags),
        profile=dict(DomainId=domain_id, UserProfileName=USER_PROFILE,
                     UserSettings=user, Tags=tags),
        space=dict(DomainId=domain_id, SpaceName=SPACE_NAME,
                   OwnershipSettings=dict(OwnerUserProfileName=USER_PROFILE),
                   SpaceSharingSettings=dict(SharingType='Private'),
                   SpaceSettings=dict(
                       AppType='JupyterLab', JupyterLabAppSettings=dict(
                           DefaultResourceSpec=dict(InstanceType=INSTANCE)),
                       SpaceStorageSettings=dict(EbsStorageSettings=dict(
                           EbsVolumeSizeInGb=100))), Tags=tags))
