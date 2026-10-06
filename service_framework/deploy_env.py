# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Common utility for ENV variable based application configurations and settings.
"""

import logging
import os
import os.path as osp
import socket
from enum import Enum
from pathlib import Path
from typing import NewType, Optional

import requests
from ec2_metadata import ec2_metadata

from async_utils.sync import blocking_func
from service_framework.deploy_stage import DeployStage
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

# Environment variables populated when running in Snowpark container services.
# See https://docs.snowflake.com/en/developer-guide/snowpark-container-services/additional-considerations-services-jobs#connecting-to-snowflake  # noqa
SNOWFLAKE_ACCOUNT: Optional[str] = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_HOST: Optional[str] = os.getenv('SNOWFLAKE_HOST')
SNOWFLAKE_DATABASE: Optional[str] = os.getenv('SNOWFLAKE_DATABASE')
# Schema that the kumo service is running from, usually USER_SCHEMA
# Customer-set with stored procedures, var provided by SPCS
SNOWFLAKE_SCHEMA: Optional[str] = os.getenv('SNOWFLAKE_SCHEMA')
# Schema owned by native app, set in kumo-snow-integration
SNOWFLAKE_APP_SCHEMA: Optional[str] = os.getenv('SNOWFLAKE_APP_SCHEMA')
SNOWFLAKE_SERVICE: Optional[str] = os.getenv('SNOWFLAKE_SERVICE')

# Environment variables used for databricks credentials
DATABRICKS_TOKEN = EnvVar('DATABRICKS_TOKEN', type=str, default_value='')
DATABRICKS_CLIENT_ID = EnvVar('DATABRICKS_CLIENT_ID', type=str,
                              default_value='')
DATABRICKS_CLIENT_SECRET = EnvVar('DATABRICKS_CLIENT_SECRET', type=str,
                                  default_value='')
# Environment variables used for databricks environment config
DATABRICKS_HOST = EnvVar('DATABRICKS_HOST', type=str, default_value='')
DATABRICKS_CLUSTER_ID = EnvVar('DATABRICKS_CLUSTER_ID', type=str,
                               default_value='')
DATABRICKS_WAREHOUSE_ID = EnvVar('DATABRICKS_WAREHOUSE_ID', type=str,
                                 default_value='')
# Environment variable indicating if running tests
KUMO_TESTING = EnvVar('KUMO_TESTING', type=bool, default_value=False)

# Cloud provider flag - indicates if deployed on GCP (vs AWS default)
IS_GCP_DEPLOY = EnvVar(
    name="IS_GCP_DEPLOY", type=bool, required=False, default_value=False,
    description="Whether deployed on GCP (default is AWS)").parse_value()


@blocking_func()
def get_instance_id() -> str:
    try:
        return ec2_metadata.instance_id
    except Exception:  # pylint: disable=broad-except
        if SNOWFLAKE_ACCOUNT:
            id = f'{SNOWFLAKE_ACCOUNT}_{SNOWFLAKE_DATABASE}_{SNOWFLAKE_SCHEMA}'
            if SNOWFLAKE_SERVICE:
                id += f'_{SNOWFLAKE_SERVICE}'
            return id
        return socket.gethostname()


# Cache the machine instance ID at import time (sync context) so that
# downstream code (e.g. deploy_env_config.deploy_env() singleton) can read
# it without calling the @blocking_func get_instance_id() at runtime.
MACHINE_INSTANCE_ID: str = get_instance_id()


class TemporalServerLocation(Enum):
    # Temporal Cloud.
    CLOUD = 'cloud'

    # "127.0.0.1:7233", used for pytest on github runner
    LOCALHOST = 'localhost'


_TEMPORAL_SERVER = EnvVar(
    'TEMPORAL_SERVER',
    TemporalServerLocation,
    required=False,
    default_value=TemporalServerLocation.CLOUD,
)
REGION_SUFFIX = EnvVar('REGION_SUFFIX', str, required=False, default_value='')
_INSTANCE_ID = EnvVar('INSTANCE_ID', str, required=False,
                      default_value=get_instance_id())
_TEMPORAL_CERT_DIR = EnvVar('TEMPORAL_CERT_DIR', str, required=False,
                            default_value='')

_CUSTOM_S3_DATA_PLANE_BUCKET = EnvVar(
    name='CUSTOM_S3_DATA_PLANE_BUCKET', type=str, required=False,
    default_value='', description='Custom S3 bucket for customers '
    'who need physical isolation')

_CUSTOM_REST_PERSIST_DIR = EnvVar(
    name='CUSTOM_REST_PERSIST_DIR', type=str, required=False, default_value='',
    description='Custom dir to keep persistent data (DB, models, etc)')

DEVTEST_NAMESPACE = 'devtest'

# Mapping from deploy stage to Temporal namespaces.
_DEPLOY_STAGE_TO_TEMPORAL_NAMESPACE = {
    DeployStage.PROD: 'prod',
    DeployStage.STAGING: 'staging',
    DeployStage.PREPROD: 'preprod',
    DeployStage.DEV: DEVTEST_NAMESPACE,
    DeployStage.DEV_LOCAL: DEVTEST_NAMESPACE,
    DeployStage.TEST: DEVTEST_NAMESPACE,
}

# Registries of docker images.
ECR_REGISTRY = "926922431314.dkr.ecr.us-west-2.amazonaws.com/kumo/"

DEFAULT_REGISTRY = EnvVar(
    name="DEFAULT_REGISTRY",
    type=str,
    default_value=ECR_REGISTRY,
    description="Default registry for docker images",
).parse_value()
_CUSTOM_TEMPORAL_NAMESPACE = EnvVar('TEMPORAL_NAMESPACE', type=str,
                                    required=False, default_value='',
                                    description='Temporal namespace')

# A string that can be used as a kubernetes label, to describe the deploy
# environment of a particular pod/job/etc.
K8sDeployEnvLabel = NewType("K8sDeployEnvLabel", str)

_SLACK_URL = EnvVar('SLACK_URL', str, default_value='')
_CUSTOMER_SLACK_URL = EnvVar('CUSTOMER_SLACK_URL', str, default_value='')

ENV_KUMO_HOME = 'KUMO_HOME'
DEFAULT_CACHE_RELATIVE_DIR = osp.join('.cache', 'kumo')


def _source_repo_root() -> Optional[str]:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').exists() and (
                parent / 'service_framework').is_dir():
            return str(parent)
    return None


def _default_cache_dir() -> str:
    return osp.join(_source_repo_root() or os.getcwd(),
                    DEFAULT_CACHE_RELATIVE_DIR)


def _is_local_filesystem_path(path: str) -> bool:
    if path.startswith('@'):
        return False

    from io_utils.fsutil.uri import LocalURI, ParseURI
    try:
        return isinstance(ParseURI(path), LocalURI)
    except ValueError:
        return False


def get_default_home_dir() -> str:
    home_dir = os.getenv(ENV_KUMO_HOME)
    if home_dir is None:
        return _default_cache_dir()

    home_dir = os.path.expanduser(home_dir)
    if not osp.isabs(home_dir) and _is_local_filesystem_path(home_dir):
        home_dir = osp.abspath(home_dir)
    return home_dir


def get_snowflake_connection_string() -> str:
    r"""
    When running in SaaS, set it to KUMOAI_SAAS and when running in SPCS,
    set it to KUMOAI_SPCS.
    """
    # SNOWFLAKE_ACCOUNT is only set in SPCS:
    if SNOWFLAKE_ACCOUNT is not None:
        return 'KUMOAI_SPCS'
    return 'KUMOAI_SAAS'


@blocking_func()
def send_slack_message(message: str, url: Optional[str] = None,
                       customer_specific_url: Optional[str] = None,
                       timeout: float = 2.0):
    """
    This function makes a blocking call to post a slack message for notifying
    failures, be careful with using this in async functions.
    """
    urls = [
        url or _SLACK_URL.parse_value(), customer_specific_url
        or _CUSTOMER_SLACK_URL.parse_value()
    ]
    for url in urls:
        if url:
            try:
                data = {"text": message, "link_names": True}
                headers = {'Content-type': 'application/json'}
                requests.post(url, json=data, headers=headers, timeout=timeout)
            except Exception:
                log.exception(f"Failed to send slack message to {url}.")
