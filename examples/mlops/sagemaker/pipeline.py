# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upsert a SageMaker Pipeline (SageMaker Python SDK v3) that fits and scores
with kumo-engine; see README.md."""

import argparse
from typing import Any, Optional, Sequence

PROCESS_CODE = '/opt/ml/processing/input/code'


def build_pipeline(
    *,
    name: str,
    role: str,
    image: str,
    code: str,
    data_root: str,
    artifact_root: str,
    tracking_arn: str,
    experiment: str,
    fit_instance_type: str,
    score_instance_type: str,
    alias: str,
    model_name: str = 'users-churn',
    table_prefix: str = '',
    on_error: str = 'raise',
) -> Any:
    from sagemaker.core.processing import (
        ProcessingInput,
        ProcessingS3Input,
        Processor,
    )
    from sagemaker.core.workflow.pipeline_context import PipelineSession
    from sagemaker.mlops.workflow.pipeline import Pipeline
    from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.train.configs import Compute, OutputDataConfig, SourceCode
    from sagemaker.train.model_trainer import ModelTrainer

    # The SDK uploads its training-container drivers next to the artifacts,
    # where the role already has rights, not to a default bucket.
    bucket, _, prefix = artifact_root.removeprefix('s3://').rstrip(
        '/').partition('/')
    session = PipelineSession(
        default_bucket=bucket,
        default_bucket_prefix='/'.join(filter(None, [prefix, 'sagemaker'])))
    env = {
        'MLFLOW_TRACKING_URI': tracking_arn,
        'MLFLOW_EXPERIMENT_NAME': experiment,
    }
    flags = [
        '--artifact-root', artifact_root, '--source', data_root,
        '--registry', 'mlflow', '--model-name', model_name, '--alias',
        alias, f'--table-prefix={table_prefix}', '--on-error', on_error
    ]

    trainer = ModelTrainer(
        training_image=image,
        role=role,
        compute=Compute(instance_type=fit_instance_type, instance_count=1),
        source_code=SourceCode(
            source_dir=code, command='python fit_job.py',
            args=[*flags, '--scratch-dir', '/tmp/kumo']),
        environment=env,
        output_data_config=OutputDataConfig(
            s3_output_path=f'{artifact_root.rstrip("/")}/sagemaker-output'),
        sagemaker_session=session,
    )
    fit = TrainingStep(name='Fit', step_args=trainer.train())

    processor = Processor(
        role=role,
        image_uri=image,
        instance_count=1,
        instance_type=score_instance_type,
        entrypoint=['python', f'{PROCESS_CODE}/score_job.py'],
        env=env,
        sagemaker_session=session,
    )
    predict = ProcessingStep(
        name='Predict',
        step_args=processor.run(
            inputs=[ProcessingInput(
                input_name='code', s3_input=ProcessingS3Input(
                    s3_uri=code, s3_data_type='S3Prefix',
                    local_path=PROCESS_CODE, s3_input_mode='File'))],
            arguments=[
                *flags, '--scratch-dir', '/opt/ml/processing/scratch'
            ],
        ),
        depends_on=[fit],
    )
    return Pipeline(name=name, steps=[fit, predict],
                    sagemaker_session=session)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='kumo-engine-users-churn')
    for flag in ('--role', '--image', '--code', '--data-root',
                 '--artifact-root', '--tracking-arn'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--experiment', default='kumo-engine-users-churn')
    parser.add_argument('--fit-instance-type', required=True)
    parser.add_argument('--score-instance-type', required=True)
    parser.add_argument('--alias', default='champion')
    parser.add_argument('--model-name', default='users-churn')
    parser.add_argument('--table-prefix', default='')
    parser.add_argument('--on-error', choices=('raise', 'warn'),
                        default='raise')
    parser.add_argument('--start', action='store_true')
    args = vars(parser.parse_args(argv))
    start = args.pop('start')
    pipeline = build_pipeline(**args)
    pipeline.upsert(role_arn=args['role'])
    if start:
        print(pipeline.start().arn)


if __name__ == '__main__':
    main()
