# SageMaker Pipelines

`pipeline.py` defines and upserts a two-step pipeline:

1. **`Fit`** (training job) runs `python fit_job.py` on `users.parquet`,
   `items.parquet` and `orders.parquet` under `--data-root`, inside
   `kumo.mlops(...)` with `--artifact-root`: the snapshot, training table and
   model are stored under it. `--registry mlflow` logs the fit to the managed
   MLflow tracking server, registers the model as a new version of
   `users-churn` and moves the `--alias` alias (`champion` by default) to it.
2. **`Predict`** (processing job, after `Fit`) runs `python score_job.py`,
   loading the model its `Fit` step trained and scoring a fresh snapshot.
   `Fit` writes `models/<bundle id>` to `/opt/ml/model/model_path`; the
   training job's uncompressed model output is `Predict`'s `model` input,
   so a later execution's `Fit` moving the alias does not change what this
   execution scores.

`--fit-instance-type` and `--score-instance-type` size the two steps
separately. `--model-name` (default `users-churn`) and `--table-prefix`
(prefix of the three Parquet file names) are passed to both steps, as is
`--on-error` (default `raise`: a failed tracking call or registration fails
the step, so `Predict` never runs on the previous version unnoticed). With `--alias candidate`, the pipeline registers and scores a
candidate and leaves `champion` to be moved by hand (see
[Promote a model](../../../packages/kumo-engine/docs/mlops.md#promote-a-model)).
On a shared tracking server, pass your own `--name` and `--model-name`: the
defaults would update another team's pipeline or registered model.

To score on a schedule without retraining, put `Predict` in its own pipeline
with a `PipelineSchedule` trigger (`pipeline.put_triggers`), without the
`model` input and `--model-path-file`, so it loads
`models:/users-churn@<alias>`.

## Before running

1. Build and push the image. [`Dockerfile`](Dockerfile) installs
   `kumo-engine[mlflow,sagemaker,text]` (MLflow, S3, `sagemaker-mlflow` and the default word vectors for text columns) and torch 2.12 for CUDA 12.6
   on `python:3.12-slim`, from a directory holding the kumo-engine wheel and
   its kumo-core, kumo-api, kumo-pql, kumo-s3 and diskgraph wheels
   (`--find-links`; the engine wheel goes in by path, since PyPI's `kumo-engine`
   1.0.0 release, which predates the MLOps API, would otherwise win). Add `--build-arg TORCH=cpu` for a CPU
   image. The images are about 8 GB (GPU) and 3.5 GB (CPU).
   ```bash
   docker build -f examples/mlops/sagemaker/Dockerfile -t kumo-engine-sagemaker <wheel dir>
   aws ecr create-repository --repository-name kumo-engine-sagemaker
   aws ecr get-login-password | docker login --username AWS \
     --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag kumo-engine-sagemaker <account>.dkr.ecr.<region>.amazonaws.com/kumo-engine-sagemaker:<tag>
   docker push <account>.dkr.ecr.<region>.amazonaws.com/kumo-engine-sagemaker:<tag>
   ```
2. Upload the code channel: `aws s3 sync examples/mlops s3://<bucket>/kumo/code/`.
3. With the SageMaker Python SDK v3 (`pip install sagemaker`, Python 3.10 or later):
   ```bash
   python examples/mlops/sagemaker/pipeline.py --role <role-arn> \
     --image <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag> \
     --code s3://<bucket>/kumo/code/ --data-root s3://<bucket>/kumo/data/ \
     --artifact-root s3://<bucket>/kumo/artifacts/ \
     --tracking-arn <mlflow-tracking-server-or-app-arn> \
     --fit-instance-type ml.g5.xlarge --score-instance-type ml.g5.xlarge \
     --start
   ```
   The SDK reads the region from `AWS_DEFAULT_REGION` (or the AWS config
   file). The Training step runs `fit_job.py` through the SDK's
   training-container drivers; the Processing step runs `score_job.py`
   directly. SageMaker's own files (the drivers, the training job's output)
   go to `sagemaker/`, a sibling of the artifact root (for
   `s3://<bucket>/kumo/artifacts/`, `s3://<bucket>/kumo/sagemaker/`), so the
   write-once root holds only kumo-engine's artifacts. `--query` overrides
   `fit_job.py`'s default query.
   The script sets `SAGEMAKER_VALIDATE_ROLE=false` unless it is already set:
   SDK v3 checks an explicitly passed execution role client-side for VPC
   network-interface (`ec2:*NetworkInterface*`, `ec2:Describe*`) and
   `cloudwatch:PutMetricData` rights, which this pipeline (no VPC config)
   does not use and the least-privilege role below does not grant.
   SageMaker still checks the role when it starts each job.
   The SDK's own INFO lines (telemetry, `Popping out 'TrainingJobName'`,
   `Running within a PipelineSession, there will be No Wait`) are expected.
   The name (`--name`, default `kumo-engine-users-churn`) may not contain
   `-aws-`: SageMaker reserves that token, and the execution then fails with
   no steps (`The token -aws- is reserved`).

## Privileges

- **Caller:** `sagemaker:CreatePipeline`, `UpdatePipeline`,
  `StartPipelineExecution`, and `iam:PassRole` on `<role-arn>`; S3 write under
  the `sagemaker/` sibling prefix and `s3:ListAllMyBuckets` or `s3:ListBucket` on
  the bucket (the SDK checks the bucket before uploading its drivers).
- **Execution role** (`--role` also runs the steps): `sagemaker:CreateTrainingJob`,
  `DescribeTrainingJob`, `StopTrainingJob`, `CreateProcessingJob`,
  `DescribeProcessingJob`, `StopProcessingJob` and `AddTags`; `iam:PassRole`
  on itself; S3 read on the code and data prefixes, read/write on
  the artifact prefix, and read/write on the `sagemaker/` sibling prefix (the
  drivers and the training job's output); ECR pull; CloudWatch Logs; the `sagemaker-mlflow:*`
  actions for tracking and the registry (`CreateRun`, `LogBatch`,
  `CreateRegisteredModel`, `CreateModelVersion`, `SetRegisteredModelAlias`,
  `GetModelVersionByAlias`, ...).

## Notes

- **GPU quota for `Predict`.** The `Predict` step is a SageMaker Processing
  job, which draws on the `<instance type> for processing job usage` quota
  (for example `ml.g5.xlarge for processing job usage`), separate from the
  training job quota and often 0 by default. Request an increase in Service
  Quotas before scoring on a GPU type, or pass a CPU type as
  `--score-instance-type` with a CPU image. A GPU `Predict` step was verified
  on `ml.g5.xlarge` with that quota at 2.
- **Flags.** Both steps pass the job scripts' flags as plain strings
  (`container_arguments` for the training step, `arguments` for the
  processing step); only MLflow's own settings are environment variables.
- **MLflow.** `MLFLOW_TRACKING_URI` is the tracking server or MLflow App ARN.
  Managed MLflow servers run MLflow 3.0; the `mlflow` extra installs
  mlflow-skinny 3.5 or later, which works against them, so do not pin the
  client to the server's version. Aliases are letters, digits and underscores, so
  SageMaker's reserved `SageMakerLifecycle-` aliases cannot be set. Model names
  are letters, digits and hyphens (no `_` or `.`), checked when `kumo.MLOps`
  is built.
- **Scratch.** The fit step uses `/tmp/kumo` (on NVMe instance types
  SageMaker puts `/tmp` on the instance store; on EBS-only types raise
  `volume_size`); the score step uses `/opt/ml/processing/scratch`.
- **Retries.** Step `retry_policies` key on SageMaker failure reasons, never
  on the exit code; a non-zero exit is not retried.

## Verified

This pipeline was upserted and run on `ml.m5.xlarge` with a CPU image, the
tracking server ARN as `MLFLOW_TRACKING_URI` and 180 days of synthetic orders
(the 30-day query needs about 150 days of history): both steps exit 0, the
fit registers a new `users-churn` version and moves `champion`, and the score
step loads it. It also ran with both steps on `ml.g5.xlarge` with a GPU image
built like `Dockerfile` (python 3.12-slim, torch 2.12 for CUDA 12.6, the
wheels and `sagemaker-mlflow`), using the GPU in both. The separate
instance-type flags, `--alias` and this exact `Dockerfile` have not been run
on SageMaker. An MLflow App ARN as tracking URI is unverified.
`test_job_templates.py` checks the scripts locally.

## References

- Pipeline steps: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html
- Retry policies: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-retry-policy.html
- Instance store and `/tmp`:
  https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-tips-considerations.html
- Managed MLflow and IAM:
  https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html,
  https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-iam.html
