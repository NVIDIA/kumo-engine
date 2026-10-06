# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any, Dict, Optional

from io_utils import kumo_dataclass
from service_framework.env_var import EnvVar

_ONE_GB = 1000 * 1000 * 1000  # bytes.
_ONE_MILLION = 1000000

# DEV/TEST training-config override (shrink epochs/experiments so dev & test
# jobs finish fast), applied by MLBackendSettings.parse_from_env in non-prod
# deploy modes. Mirrors TRAINING_CFG_FAST_DEV/TRAINING_CFG_CPU_TEST in
# rest/backends/ml_backend/worker.py.
_DEV_TRAINING_CFG_OVERRIDE = {
    'max_epochs': 5,
    'num_experiments': 2,
}


@kumo_dataclass
class MLJobSettings:
    train_cfg_overrides: Dict[str, Any] = field(default_factory=dict)
    # Inert compatibility field for MLJobSettings serialized by older workers.
    # Job artifacts are resolved through get_job_base_dir().
    file_storage_path: str = ''

    max_total_table_size: int = 18 * _ONE_GB
    max_total_graph_edges: int = 140 * _ONE_MILLION

    # Limits to decide when/how to partition a prediction table.
    pred_max_num_entities_per_partition: int = 10 * _ONE_MILLION
    pred_max_num_rows_per_partition: int = 300 * _ONE_MILLION

    # TODO(deprecated): unused — global-materialize-dir selection was removed
    # (edges are served from the DiskGraph index, so the path-layout flag is
    # inert). Field retained only so in-flight/serialized MLJobSettings still
    # deserialize; delete after all prod releases are past this change.
    use_global_materialize_dir: bool = False

    # Test-only settings.
    raise_fake_error: bool = False
    use_fake_job_backend: bool = False

    # DEPRECATED: use BatchPredictionJobConfig.num_parallel_workers instead.
    max_concurrent_prediction_partitions: int = 1


@kumo_dataclass
class MLJobQueueSettings:
    """Configurations for job concurrency control and queueing."""

    max_concurrent_jobs_total: int = 10
    max_concurrent_gpu_worker_slots: int = 10
    use_job_queue: bool = True
    max_job_queue_length: int = 10
    job_queue_poll_seconds: float = 40.0

    @staticmethod
    def parse_from_env() -> 'MLJobQueueSettings':
        settings = MLJobQueueSettings()
        settings.max_concurrent_jobs_total = EnvVar(
            'ML_MAX_CONCURRENT_JOBS_TOTAL',
            type=int,
            default_value=settings.max_concurrent_jobs_total,
        ).parse_value()
        settings.max_concurrent_gpu_worker_slots = EnvVar(
            'ML_MAX_GPU_WORKER_SLOTS',
            type=int,
            default_value=settings.max_concurrent_gpu_worker_slots,
        ).parse_value()
        settings.use_job_queue = EnvVar(
            'ML_USE_JOB_QUEUE',
            type=bool,
            default_value=settings.use_job_queue,
        ).parse_value()
        settings.max_job_queue_length = EnvVar(
            'ML_JOB_QUEUE_SIZE',
            type=int,
            default_value=settings.max_job_queue_length,
        ).parse_value()
        settings.job_queue_poll_seconds = EnvVar(
            'ML_JOB_QUEUE_POLL_SECONDS',
            type=float,
            default_value=settings.job_queue_poll_seconds,
        ).parse_value()
        return settings


@kumo_dataclass
class MLBackendSettings:
    slack_url: Optional[str] = None
    # py3.14 stdlib dataclasses reject mutable instance defaults; use
    # field(default_factory=...).
    ml_job_settings: MLJobSettings = field(default_factory=MLJobSettings)
    job_queue_settings: MLJobQueueSettings = field(
        default_factory=MLJobQueueSettings)

    # TODO: possibly remove this if we always need prediction stats.
    prediction_col_stats: bool = True

    @staticmethod
    def parse_from_env() -> 'MLBackendSettings':
        """Build MLBackendSettings from process env vars + the deploy-config
        singleton, with no WorkerEnv.

        This lets an in-process MLJobBackend be constructed in the REST/public
        apps (which have no WorkerEnv). It deliberately MIRRORS the worker's
        ``rest/backends/ml_backend/worker.py:generate_ml_backend_settings`` and
        must be kept in sync with it; the EnvVar names/types/defaults match
        ``_ml_backend_env_variables`` in ``temporalio_commons/worker_main.py``.
        (The worker keeps its own WorkerEnv-based builder so it still honors
        per-test ``env_var_test_override`` injection; unifying the two behind a
        shared assembler is a follow-up once integ-validated.)
        """
        # Imported lazily: rest.settings pulls in heavier deploy machinery.
        from rest.settings import DataScaleLimits
        from service_framework.deploy_env_config import deploy_env
        from service_framework.deploy_stage import DeployStage

        deploy_mode = deploy_env().deploy_mode
        data_limits = DataScaleLimits.parse_from_env()

        slack_url = EnvVar('SLACK_URL', str, default_value='').parse_value()
        if not slack_url or not slack_url.startswith('https'):
            slack_url = ''

        is_prod = deploy_mode in (DeployStage.PROD, DeployStage.STAGING)
        settings = MLBackendSettings(
            slack_url=slack_url,
            ml_job_settings=MLJobSettings(
                pred_max_num_entities_per_partition=EnvVar(
                    'ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION', int,
                    default_value=10 * _ONE_MILLION).parse_value(),
                pred_max_num_rows_per_partition=EnvVar(
                    'ML_PRED_MAX_NUM_ROWS_PER_PARTITION', int,
                    default_value=300 * _ONE_MILLION).parse_value(),
                max_total_table_size=data_limits.max_graph_total_size_bytes,
                max_total_graph_edges=data_limits.max_graph_total_edges,
            ),
            job_queue_settings=MLJobQueueSettings.parse_from_env(),
            prediction_col_stats=EnvVar('PRED_COL_STATS', bool,
                                        default_value=True).parse_value(),
        )
        if is_prod:
            return settings

        # DEV/TEST: shrink the training config so jobs finish fast. The worker
        # splits this into FAST_DEV (cuda) vs CPU_TEST, but both dicts are
        # currently identical, so we apply the single override whenever
        # FAST_TRAIN is set or we're in TEST mode (no torch.cuda probe needed).
        if (EnvVar('FAST_TRAIN', bool, default_value=False).parse_value()
                or deploy_mode == DeployStage.TEST):
            settings.ml_job_settings.train_cfg_overrides = (
                _DEV_TRAINING_CFG_OVERRIDE)
        return settings
