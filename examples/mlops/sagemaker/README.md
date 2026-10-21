# SageMaker Pipelines

`pipeline.py` defines and upserts a two-step pipeline:

1. **`Fit`** (training job) runs `python fit_job.py` on `users.parquet`,
   `items.parquet` and `orders.parquet` under `KUMO_DATA_ROOT`.
   `KUMO_REGISTRY=mlflow` logs the fit to the managed MLflow tracking server,
   registers the model as a new version of `users-churn` and moves the
   `champion` alias to it.
2. **`Predict`** (processing job, after `Fit`) runs `python score_job.py`,
   loading `models:/users-churn@champion` and writing
   `<artifact root>/runs/<execution id>/predictions/`.

To score on a schedule without retraining, put `Predict` in its own pipeline
with a `PipelineSchedule` trigger (`pipeline.put_triggers`).

## Before running

1. Build an ECR image with `kumo-engine[mlflow,s3]`, its dependency wheels
   and `sagemaker-mlflow` (plus the CUDA stack for GPU instance types).
2. Upload the code channel: `aws s3 sync examples/mlops s3://<bucket>/kumo/code/`.
3. With the SageMaker Python SDK v2 (`pip install 'sagemaker<3'`):
   ```bash
   python examples/mlops/sagemaker/pipeline.py --role <role-arn> \
     --image <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag> \
     --code s3://<bucket>/kumo/code/ --data-root s3://<bucket>/kumo/data/ \
     --artifact-root s3://<bucket>/kumo/artifacts/ \
     --tracking-arn <mlflow-tracking-server-or-app-arn> \
     --instance-type <gpu_instance_type> --start
   ```
   SDK v3 replaced `Estimator` with `ModelTrainer` and moved the workflow
   classes (`sagemaker.mlops.workflow.*`); this script is not ported.

## Privileges

- **Caller:** `sagemaker:CreatePipeline`, `UpdatePipeline`,
  `StartPipelineExecution`, and `iam:PassRole` on `<role-arn>`.
- **Execution role** (`--role` also runs the steps): `sagemaker:CreateTrainingJob`,
  `DescribeTrainingJob`, `StopTrainingJob`, `CreateProcessingJob`,
  `DescribeProcessingJob`, `StopProcessingJob` and `AddTags`; `iam:PassRole`
  on itself; S3 read on the code and data prefixes and read/write on
  the artifact prefix; ECR pull; CloudWatch Logs; the `sagemaker-mlflow:*`
  actions for tracking and the registry (`CreateRun`, `LogBatch`,
  `CreateRegisteredModel`, `CreateModelVersion`, `SetRegisteredModelAlias`,
  `GetModelVersionByAlias`, ...).

## Notes

- **Job ids.** The training step passes the pipeline name and execution ARN
  as `KUMO_JOB_ID`/`KUMO_JOB_RUN_ID` (`container_arguments` takes only plain
  strings); the processing step passes them as `--job-id`/`--job-run-id`.
- **MLflow.** `MLFLOW_TRACKING_URI` is the tracking server or MLflow App ARN.
  Managed MLflow servers run MLflow 3.0; the engine needs a newer client
  (3.11 or later), which works against them, so do not pin the client to the
  server's version. Aliases are letters, digits and underscores, so
  SageMaker's reserved `SageMakerLifecycle-` aliases cannot be set.
- **Scratch.** The fit step uses `/tmp/kumo` (on NVMe instance types
  SageMaker puts `/tmp` on the instance store; on EBS-only types raise
  `volume_size`); the score step uses `/opt/ml/processing/scratch`.
- **Retries.** Step `retry_policies` key on SageMaker failure reasons, never
  on the exit code; a non-zero exit is not retried.

## Verified

The pipeline was upserted and run on `ml.m5.xlarge` with a CPU image, the
tracking server ARN as `MLFLOW_TRACKING_URI` and 180 days of synthetic orders
(the 30-day query needs about 150 days of history): both steps exit 0, the
fit registers a new `users-churn` version and moves `champion`, and the score
step loads it. An MLflow App ARN as tracking URI is unverified.

## References

- Pipeline steps: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html
- Retry policies: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-retry-policy.html
- Instance store and `/tmp`:
  https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage-tips-considerations.html
- Managed MLflow and IAM:
  https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html,
  https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-create-tracking-server-iam.html
