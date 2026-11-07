# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS API access restricted to the suite's dedicated account and resources."""
import base64
import hashlib
import json
import re
import time
from pathlib import Path

from config import ACCOUNT, DENIED_ROOT, REGION, test_job
from evidence import digest, save
from runtime import receipt_reference


class Client:
    def __init__(self, config, session=None):
        if config['account'] != ACCOUNT or config['region'] != REGION:
            raise ValueError('Use the dedicated SageMaker test account/region')
        if session is None:
            import boto3
            session = boto3.Session(region_name=REGION)
        self.session, self.config = session, config
        self.identity = session.client('sts').get_caller_identity()
        if self.identity['Account'] != ACCOUNT:
            raise ValueError('AWS credentials resolve to another account')
        self.s3, self.sm = session.client('s3'), session.client('sagemaker')

    def preflight(self):
        cfg = self.config
        self.s3.head_bucket(Bucket=cfg['bucket'], ExpectedBucketOwner=ACCOUNT)
        region = self.s3.get_bucket_location(
            Bucket=cfg['bucket'])['LocationConstraint']
        if region != REGION:
            raise ValueError('Test bucket is in another region')
        server = self.sm.describe_mlflow_tracking_server(
            TrackingServerName=cfg['tracking_server'])
        if (server['TrackingServerArn'] != cfg['tracking_arn']
                or server['RoleArn'] != cfg['tracking_role']
                or server['ArtifactStoreUri'].rstrip('/')
                != cfg['tracking_artifact_root']):
            raise ValueError(
                'Managed MLflow server is not the dedicated test server')
        if (server['TrackingServerStatus'] not in ('Created', 'Started')
                or server.get('IsActive') != 'Active'):
            raise ValueError('Dedicated managed MLflow server is not ready')
        # An existing readable sentinel distinguishes permission tests from 404s.
        probe = self.s3.get_object(Bucket=cfg['bucket'],
                                   Key=DENIED_ROOT + '/README.txt')
        probe['Body'].close()
        return dict(
            account=self.identity['Account'], caller=self.identity['Arn'],
            tracking_arn=server['TrackingServerArn'],
            tracking_version=server['MlflowVersion'],
            registry_sync=server.get('AutomaticModelRegistration',
                                     False), denied_sentinel='readable')

    def asset(self, name, expected, local=None):
        from botocore.exceptions import ClientError

        cfg = self.config
        if (not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.+-]*', name)
                or not re.fullmatch(r'[a-f0-9]{64}', expected)):
            raise ValueError('Invalid content-addressed asset')
        key = f'{cfg["root"]}/payload-cache/{expected}/{name}'
        try:
            head = self.s3.head_object(Bucket=cfg['bucket'], Key=key,
                                       ChecksumMode='ENABLED')
        except ClientError as error:
            if error.response['Error']['Code'] not in ('404', 'NoSuchKey'):
                raise
            if local is None:
                raise ValueError('Seed the missing pinned asset: ' +
                                 name) from None
            content = Path(local).read_bytes()
            if digest(content) != expected:
                raise ValueError('Local asset differs from pinned input: ' +
                                 name)
            try:
                self.s3.put_object(
                    Bucket=cfg['bucket'], Key=key, Body=content,
                    ChecksumSHA256=base64.b64encode(
                        hashlib.sha256(content).digest()).decode(),
                    Metadata={'sha256': expected}, IfNoneMatch='*')
            except ClientError as error:
                if error.response['Error']['Code'] not in (
                        '412', 'PreconditionFailed'):
                    raise
            head = self.s3.head_object(Bucket=cfg['bucket'], Key=key,
                                       ChecksumMode='ENABLED')
        checksum = head.get('ChecksumSHA256')
        if checksum != base64.b64encode(bytes.fromhex(expected)).decode():
            body = self.s3.get_object(Bucket=cfg['bucket'], Key=key)['Body']
            try:
                if digest(body.read()) != expected:
                    raise ValueError('Cached asset checksum mismatch: ' + name)
            finally:
                body.close()
        return dict(bucket=cfg['bucket'], key=key, sha256=expected,
                    size=head['ContentLength'])

    def publish_job(self, execution, arn, spec, root):
        from botocore.exceptions import ClientError

        name = test_job(arn.rsplit('/', 1)[-1])
        if not name.startswith(spec['pipeline_name']):
            raise ValueError('Job belongs to another pipeline attempt')
        job = self.sm.describe_training_job(TrainingJobName=name)
        tags = {
            t['Key']: t['Value']
            for t in self.sm.list_tags(ResourceArn=arn)['Tags']
        }
        if (job['TrainingJobArn'] != arn
                or job['RoleArn'] != self.config['execution_role']
                or tags.get('KumoTestAttempt') != spec['attempt']
                or tags.get('KumoTestSource') != spec['source_commit']):
            raise ValueError(
                'Actual training job differs from the submitted attempt')
        record = dict(attempt=spec['attempt'],
                      source_commit=spec['source_commit'], job_arn=arn,
                      execution_arn=execution.arn)
        receipt_reference(dict(self.config, **spec), record)
        content = (json.dumps(record, sort_keys=True) + '\n').encode()
        save(Path(root) / 'training-job-live.json', job)
        (Path(root) / 'job-identity.json').write_bytes(content)
        try:
            response = self.s3.put_object(Bucket=self.config['bucket'],
                                          Key=spec['job_receipt_key'],
                                          Body=content, IfNoneMatch='*')
            save(
                Path(root) / 'job-identity-publication.json',
                dict(key=spec['job_receipt_key'], sha256=digest(content),
                     response=response))
        except ClientError as error:
            if error.response['Error']['Code'] not in ('412',
                                                       'PreconditionFailed'):
                raise
            body = self.s3.get_object(Bucket=self.config['bucket'],
                                      Key=spec['job_receipt_key'])['Body']
            try:
                if body.read() != content:
                    raise ValueError(
                        'Job identity receipt already identifies another execution'
                    )
            finally:
                body.close()
            save(
                Path(root) / 'job-identity-publication.json',
                dict(key=spec['job_receipt_key'], sha256=digest(content),
                     already_present=True))

    def wait(self, execution, root, timeout, spec):
        deadline = time.monotonic() + timeout
        published = None
        try:
            while True:
                record = execution.describe()
                save(Path(root) / 'pipeline-execution.json', record)
                steps = execution.list_steps()
                save(Path(root) / 'pipeline-steps.json', steps)
                state = record['PipelineExecutionStatus']
                jobs = [
                    s['Metadata']['TrainingJob']['Arn'] for s in steps
                    if 'TrainingJob' in s.get('Metadata', {})
                ]
                if len(jobs) > 1:
                    raise ValueError('Expected only one actual notebook job')
                if jobs and state in ('Executing', 'Stopping'):
                    if published is None:
                        self.publish_job(execution, jobs[0], spec, root)
                        published = jobs[0]
                    elif published != jobs[0]:
                        raise ValueError(
                            'Pipeline changed its notebook job identity')
                print('SageMaker notebook pipeline:', state, flush=True)
                if state not in ('Executing', 'Stopping'):
                    if len(jobs) != 1:
                        raise ValueError(
                            'Expected one actual notebook training job; '
                            'pipeline state: ' + state)
                    name = test_job(jobs[0].rsplit('/', 1)[-1])
                    pipeline_name = test_job(record['PipelineArn'].rsplit(
                        '/', 1)[-1])
                    if name != pipeline_name and not name.startswith(
                            pipeline_name + '-'):
                        raise ValueError(
                            'Pipeline returned another suite attempt')
                    return self.sm.describe_training_job(TrainingJobName=name)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        'Notebook pipeline exceeded its deadline')
                time.sleep(15)
        except BaseException:
            try:
                if execution.describe(
                )['PipelineExecutionStatus'] == 'Executing':
                    execution.stop()
            except Exception:
                pass
            raise

    def submit(self, notebook, initialization, spec, root, timeout):
        from sagemaker.session import Session
        from sagemaker.workflow.notebook_job_step import NotebookJobStep
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.pipeline_definition_config import (
            PipelineDefinitionConfig, )

        cfg = self.config

        class ExistingBucketSession(Session):
            def default_bucket(self):
                return cfg['bucket']

        class SingleAttemptNotebook(NotebookJobStep):
            @property
            def arguments(self):
                request = super().arguments
                request.pop('RetryStrategy', None)
                return request

        session = ExistingBucketSession(
            boto_session=self.session, sagemaker_client=self.sm,
            default_bucket=cfg['bucket'],
            sagemaker_config={'SchemaVersion': '1.0'})
        step = SingleAttemptNotebook(
            input_notebook=str(notebook),
            initialization_script=str(initialization), image_uri=cfg['image'],
            kernel_name=cfg['kernel'], name='KumoTestNotebook',
            notebook_job_name=spec['pipeline_name'], role=cfg['execution_role'],
            s3_root_uri=f's3://{cfg["bucket"]}/{spec["job_prefix"]}',
            instance_type=cfg['instance'], volume_size=cfg['volume_gib'],
            max_runtime_in_seconds=3600, tags=[
                dict(Key='KumoTestSuite', Value='sagemaker-platform'),
                dict(Key='KumoTestAttempt', Value=spec['attempt']),
                dict(Key='KumoTestSource', Value=spec['source_commit'])
            ])
        pipeline = Pipeline(
            name=spec['pipeline_name'], steps=[step], sagemaker_session=session,
            pipeline_definition_config=PipelineDefinitionConfig(
                use_custom_job_prefix=True))
        definition = json.loads(pipeline.definition())
        save(Path(root) / 'pipeline-definition.json', definition)
        pipeline.upsert(
            role_arn=cfg['pipeline_role'],
            tags=[dict(Key='KumoTestSuite', Value='sagemaker-platform')])
        execution = pipeline.start()
        save(
            Path(root) / 'submitted-run.json',
            dict(pipeline=spec['pipeline_name'], execution_arn=execution.arn))
        return self.wait(execution, root, timeout, spec)
