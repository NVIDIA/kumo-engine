# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove only resources acquired by one fresh, completed test attempt."""
import re
import time
from urllib.parse import urlsplit

from botocore.exceptions import ClientError
from evidence import save


class AttemptResources:
    def __init__(self, client, spec, root, tracking=None):
        self.client, self.spec, self.root = client, spec, root
        if not re.fullmatch('[a-f0-9]{12}', spec['attempt']):
            raise ValueError('Invalid cleanup attempt')
        self.name = 'kumo-test-mlops-' + spec['attempt']
        if spec['pipeline_name'] != 'kumo-test-notebook-' + spec['attempt']:
            raise ValueError('Invalid cleanup pipeline')
        self.prefixes = [spec['root'] + '/runs/' + spec['attempt'] + '/',
                         spec['root'] + '/notebook-jobs/' + spec['attempt'] + '/',
                         spec['root'] + '/artifacts/' + spec['attempt'] + '/']
        self.tracking = tracking
        self.acquired = False
        self.experiment_id = None
        self.model_created = None
        self.tags = {'kumo.test.attempt': spec['attempt'],
                     'kumo.test.source': spec['source_commit']}

    def mlflow(self):
        if self.tracking is None:
            import mlflow
            mlflow.set_tracking_uri(self.spec['tracking_arn'])
            mlflow.set_registry_uri(self.spec['tracking_arn'])
            self.tracking = mlflow.MlflowClient(
                tracking_uri=self.spec['tracking_arn'],
                registry_uri=self.spec['tracking_arn'])
        return self.tracking

    def __enter__(self):
        for prefix in self.prefixes:
            if self.client.s3.list_objects_v2(Bucket=self.spec['bucket'],
                                              Prefix=prefix, MaxKeys=1).get('Contents'):
                raise ValueError('Attempt prefix already exists; no cleanup ownership acquired')
        try:
            self.client.sm.describe_pipeline(PipelineName=self.spec['pipeline_name'])
        except ClientError as error:
            if error.response['Error']['Code'] != 'ResourceNotFound':
                raise
        else:
            raise ValueError('Attempt pipeline already exists')
        if self.spec['scenario'] != 'graph':
            from mlflow.exceptions import MlflowException
            tracking = self.mlflow()
            if tracking.get_experiment_by_name(self.name) is not None:
                raise ValueError('Attempt experiment already exists')
            try:
                tracking.get_registered_model(self.name)
            except MlflowException as error:
                if error.error_code != 'RESOURCE_DOES_NOT_EXIST':
                    raise
            else:
                raise ValueError('Attempt model already exists')
        self.acquired = True
        return self

    def prepare_tracking(self):
        if self.spec['scenario'] == 'graph':
            return
        tracking = self.mlflow()
        self.experiment_id = str(tracking.create_experiment(self.name, tags=self.tags))
        save(self.root / 'resources.json', dict(experiment_id=self.experiment_id))
        model = tracking.create_registered_model(self.name, tags=self.tags)
        self.model_created = model.creation_timestamp
        save(self.root / 'resources.json',
             dict(experiment_id=self.experiment_id, model_created=self.model_created))

    def wait_terminal(self, timeout=600):
        deadline = time.monotonic() + timeout
        while True:
            response = self.client.sm.list_pipeline_executions(
                PipelineName=self.spec['pipeline_name'])
            if response.get('NextToken'):
                raise ValueError('Cleanup refuses an ambiguous pipeline')
            active = [e for e in response.get('PipelineExecutionSummaries', [])
                      if e['PipelineExecutionStatus'] in ('Executing', 'Stopping')]
            pending_jobs = False
            for execution in response.get('PipelineExecutionSummaries', []):
                steps = self.client.sm.list_pipeline_execution_steps(
                    PipelineExecutionArn=execution['PipelineExecutionArn'])
                if steps.get('NextToken'):
                    raise ValueError('Cleanup refuses ambiguous notebook steps')
                for step in steps.get('PipelineExecutionSteps', []):
                    arn = step.get('Metadata', {}).get('TrainingJob', {}).get('Arn')
                    if not arn:
                        continue
                    prefix = (f'arn:aws:sagemaker:{self.spec["region"]}:'
                              f'{self.spec["account"]}:training-job/' + self.spec['pipeline_name'])
                    if arn != prefix and not arn.startswith(prefix + '-'):
                        raise ValueError('Cleanup pipeline references another job')
                    job = self.client.sm.describe_training_job(TrainingJobName=arn.rsplit('/', 1)[1])
                    if job['TrainingJobArn'] != arn or job['RoleArn'] != self.spec['execution_role']:
                        raise ValueError('Cleanup job execution identity changed')
                    pending_jobs |= job['TrainingJobStatus'] not in ('Completed', 'Stopped', 'Failed')
            if not active and not pending_jobs:
                return
            for execution in active:
                if execution['PipelineExecutionStatus'] == 'Executing':
                    try:
                        self.client.sm.stop_pipeline_execution(
                            PipelineExecutionArn=execution['PipelineExecutionArn'])
                    except ClientError as error:
                        if error.response['Error']['Code'] != 'ConflictException':
                            raise
                        current = self.client.sm.describe_pipeline_execution(
                            PipelineExecutionArn=execution['PipelineExecutionArn'])
                        if current['PipelineExecutionStatus'] not in ('Stopping', 'Stopped'):
                            raise
            if time.monotonic() >= deadline:
                raise TimeoutError('Cleanup pipeline did not stop within its deadline')
            time.sleep(10)

    def remove_prefix(self, prefix):
        while True:
            response = self.client.s3.list_objects_v2(
                Bucket=self.spec['bucket'], Prefix=prefix, MaxKeys=1000)
            keys = [item['Key'] for item in response.get('Contents', [])]
            if not keys:
                return
            if any(not key.startswith(prefix) for key in keys):
                raise ValueError('S3 returned an object outside the attempt prefix')
            response = self.client.s3.delete_objects(
                Bucket=self.spec['bucket'],
                Delete=dict(Objects=[dict(Key=key) for key in keys], Quiet=True))
            if response.get('Errors'):
                raise RuntimeError('Attempt object cleanup failed')

    def require_pipeline_owner(self, pipeline):
        expected = (f'arn:aws:sagemaker:{self.spec["region"]}:{self.spec["account"]}:'
                    'pipeline/' + self.spec['pipeline_name'])
        tags = {t['Key']: t['Value'] for t in self.client.sm.list_tags(ResourceArn=expected)['Tags']}
        if (pipeline['PipelineArn'] != expected
                or pipeline.get('RoleArn') != self.spec['pipeline_role']
                or tags.get('KumoTestSuite') != 'sagemaker-platform'
                or tags.get('KumoTestAttempt') != self.spec['attempt']
                or tags.get('KumoTestSource') != self.spec['source_commit']):
            raise ValueError('Cleanup pipeline ownership changed')

    def close(self):
        if not self.acquired:
            return
        sm, spec = self.client.sm, self.spec
        try:
            pipeline = sm.describe_pipeline(PipelineName=spec['pipeline_name'])
        except ClientError as error:
            if error.response['Error']['Code'] != 'ResourceNotFound':
                raise
            pipeline = None
        if pipeline:
            self.require_pipeline_owner(pipeline)
            self.wait_terminal()
            self.require_pipeline_owner(sm.describe_pipeline(PipelineName=spec['pipeline_name']))
            sm.delete_pipeline(PipelineName=spec['pipeline_name'])
        if spec['scenario'] != 'graph':
            from mlflow.exceptions import MlflowException
            tracking = self.mlflow()
            try:
                model = tracking.get_registered_model(self.name)
            except MlflowException as error:
                if error.error_code != 'RESOURCE_DOES_NOT_EXIST':
                    raise
                model = None
            if model:
                if (self.model_created is None
                        or model.name != self.name
                        or model.creation_timestamp != self.model_created
                        or any(model.tags.get(k) != v for k, v in self.tags.items())):
                    raise ValueError('Attempt registered model ownership changed')
                tracking.delete_registered_model(self.name)
            experiment = tracking.get_experiment_by_name(self.name)
            if experiment:
                if (self.experiment_id is None
                        or str(experiment.experiment_id) != self.experiment_id
                        or experiment.name != self.name
                        or any(experiment.tags.get(k) != v for k, v in self.tags.items())):
                    raise ValueError('Attempt experiment ownership changed')
                location = urlsplit(experiment.artifact_location)
                store = urlsplit(spec['tracking_artifact_root'].rstrip('/') + '/')
                prefix = location.path.lstrip('/').rstrip('/') + '/'
                if (location.scheme != 's3' or location.netloc != spec['bucket']
                        or store.netloc != location.netloc
                        or prefix != store.path.lstrip('/') + str(experiment.experiment_id) + '/'):
                    raise ValueError('Attempt experiment has an unexpected artifact location')
                tracking.delete_experiment(experiment.experiment_id)
                self.remove_prefix(prefix)
        for prefix in self.prefixes:
            self.remove_prefix(prefix)

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except Exception:
            save(self.root / 'cleanup.json', dict(status='failed'))
            if error is None:
                raise
        else:
            save(self.root / 'cleanup.json',
                 dict(status='passed', mlflow_metadata='soft-deleted'))
