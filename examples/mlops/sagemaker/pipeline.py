# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upsert a SageMaker Pipeline (SDK v2) that fits and scores with kumo-engine;
see README.md."""

import argparse
from typing import Any, Optional, Sequence

TRAIN_CODE = '/opt/ml/input/data/code'
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
    instance_type: str,
) -> Any:
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    from sagemaker.processing import ProcessingInput, Processor
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    session = PipelineSession()
    env = {
        'MLFLOW_TRACKING_URI': tracking_arn,
        'MLFLOW_EXPERIMENT_NAME': experiment,
    }
    flags = [
        '--artifact-root', artifact_root, '--source', data_root,
        '--registry', 'mlflow', '--model-name', 'users-churn'
    ]

    estimator = Estimator(
        image_uri=image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        container_entry_point=['python', f'{TRAIN_CODE}/fit_job.py'],
        container_arguments=[*flags, '--scratch-dir', '/tmp/kumo'],
        environment=env,
        output_path=f'{artifact_root.rstrip("/")}/sagemaker-output',
        sagemaker_session=session,
    )
    fit = TrainingStep(
        name='Fit',
        step_args=estimator.fit(inputs={'code': TrainingInput(code)}),
    )

    processor = Processor(
        role=role,
        image_uri=image,
        instance_count=1,
        instance_type=instance_type,
        entrypoint=['python', f'{PROCESS_CODE}/score_job.py'],
        env=env,
        sagemaker_session=session,
    )
    predict = ProcessingStep(
        name='Predict',
        step_args=processor.run(
            inputs=[ProcessingInput(source=code, destination=PROCESS_CODE)],
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
    parser.add_argument('--instance-type', required=True)
    parser.add_argument('--start', action='store_true')
    args = vars(parser.parse_args(argv))
    start = args.pop('start')
    pipeline = build_pipeline(**args)
    pipeline.upsert(role_arn=args['role'])
    if start:
        print(pipeline.start().arn)


if __name__ == '__main__':
    main()
