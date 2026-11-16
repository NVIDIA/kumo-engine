# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from kumo_core.training.options import allow_array_targets
from pydantic import Field

from io_utils import fsutil, kumo_dataclass
from rest.config.batch_prediction_job import BatchPredictionJobOutput
from rest.config.training_job import TrainingJobOutput
from rest.utils.file_upload import supports_backend_file_upload
from service_framework import developer_credentials
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from statedb.config.common import PredictiveQueryID


@kumo_dataclass
class FeatureFlags:
    ENABLE_COLUMN_ANALYSIS: bool = True
    MAX_TABLES_IN_A_GRAPH: int = 12
    IS_DATABRICKS_DEPLOY: bool = False
    IS_GCP_DEPLOY: bool = False
    ENABLE_UI_SCHEMA_VALIDATION: bool = False
    ALLOW_ARRAY_TARGETS: bool = allow_array_targets()
    IS_DISKGRAPH_DEPLOY: bool = False
    ENABLE_FILE_UPLOAD: bool = True
    GCP_WORKLOAD_EXCHANGE_ENABLED: bool = False
    # Ambient GCP identity: whether the deployment allows using Kumo's own
    # Workload Identity service account directly (no impersonation, no key),
    # plus the service-account email the customer grants IAM access to.
    GCP_AMBIENT_ENABLED: bool = False
    GCP_AMBIENT_SERVICE_ACCOUNT_EMAIL: Optional[str] = None
    GCP_AMBIENT_DEVELOPER_CREDENTIALS: bool = False

    @classmethod
    def parse_from_env_vars(cls):
        ENABLE_COLUMN_ANALYSIS = EnvVar(
            'ENABLE_COLUMN_ANALYSIS',
            bool,
            required=False,
            default_value=True,
        ).parse_value()

        MAX_TABLES_IN_A_GRAPH = EnvVar(
            'MAX_TABLES_IN_A_GRAPH',
            int,
            required=False,
            default_value=12,
        ).parse_value()

        DATABRICKS_HOST = EnvVar(
            'DATABRICKS_HOST',
            str,
            required=False,
            default_value='',
        ).parse_value()

        ENABLE_UI_SCHEMA_VALIDATION = EnvVar(
            'ENABLE_UI_SCHEMA_VALIDATION',
            bool,
            required=False,
            default_value=False,
        ).parse_value()

        GCP_WORKLOAD_EXCHANGE_ENABLED = EnvVar(
            'KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED',
            bool,
            required=False,
            default_value=False,
        ).parse_value()

        ambient_gcp = getattr(deploy_env(), 'ambient_gcp', None)
        gcp_ambient_emails = (ambient_gcp.approved_service_account_emails
                              if ambient_gcp is not None else ())
        ambient_gcp_developer_credentials = (
            developer_credentials.dev_gcp_user_credentials_enabled())

        return cls(
            ENABLE_COLUMN_ANALYSIS=ENABLE_COLUMN_ANALYSIS,
            MAX_TABLES_IN_A_GRAPH=MAX_TABLES_IN_A_GRAPH,
            IS_DATABRICKS_DEPLOY=bool(DATABRICKS_HOST),
            # TODO: overhaul feature flags — IS_GCP_DEPLOY and other
            # deployment-derived flags should move to a proper feature flag
            # system rather than being derived from storage paths.
            IS_GCP_DEPLOY=fsutil.is_gcs(deploy_env().data_plane_storage_root),
            ENABLE_UI_SCHEMA_VALIDATION=ENABLE_UI_SCHEMA_VALIDATION,
            ALLOW_ARRAY_TARGETS=allow_array_targets(),
            IS_DISKGRAPH_DEPLOY=True,
            ENABLE_FILE_UPLOAD=supports_backend_file_upload(
                deploy_env().data_plane_storage_root, ),
            GCP_WORKLOAD_EXCHANGE_ENABLED=GCP_WORKLOAD_EXCHANGE_ENABLED,
            GCP_AMBIENT_ENABLED=(ambient_gcp is not None
                                 or ambient_gcp_developer_credentials),
            GCP_AMBIENT_SERVICE_ACCOUNT_EMAIL=(gcp_ambient_emails[0]
                                               if gcp_ambient_emails else None),
            GCP_AMBIENT_DEVELOPER_CREDENTIALS=(
                ambient_gcp_developer_credentials),
        )


@kumo_dataclass
class ConfigFile:
    title: str = ''
    subtitle: str = ''
    content: Optional[str] = None


@kumo_dataclass
class JobDebugInfo:
    # Metadata
    query_id: Optional[PredictiveQueryID] = None
    job_id: Optional[str] = None
    job_status: Optional[Union[TrainingJobOutput,
                               BatchPredictionJobOutput]] = None

    # Debug Information
    files: List[ConfigFile] = Field(default_factory=list)


@kumo_dataclass
class DebugInfo:
    db_path: str
    release_version: str
    temporal_url: str
    # py3.14 stdlib dataclasses reject mutable instance defaults; use a
    # default_factory.
    feature_flags: FeatureFlags = Field(
        default_factory=FeatureFlags.parse_from_env_vars)
    job_queue_status: str = ''


TrainingJobInputJSON = Dict[Any, Any]


@kumo_dataclass
class ParsedReadMeDoc:
    title: str
    subtitle: str
    hint: str
