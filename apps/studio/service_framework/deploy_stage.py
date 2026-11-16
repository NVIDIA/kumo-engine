# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from service_framework.env_var import EnvVar


class DeployStage(str, Enum):
    # Production deploy mode.
    PROD = 'prod'

    # Staging deploy mode (similar deployment env as prod but without access to
    # customer data, mainly for dev & testing in a prod-like environment.)
    # namespace = 'staging'
    STAGING = 'staging'

    # Pre-production deploy mode.
    # task_queue_suffix = '-<instance_id>'
    # workflow_id_suffix = '-<instance_id>'
    PREPROD = 'preprod'

    # In development mode: application and worker run by the same developer on
    # a single machine. We append hostname to task queue and workflow_id to
    # ensure developers can run devserver on multiple machines without having
    # to worry about interference, e.g.:
    # https://github.com/kumo-ai/kumo/issues/2367
    #
    # namespace = 'devtest'
    # task_queue_suffix = '-<instance_id>'
    # workflow_id_suffix = '-<instance_id>'
    # k8s_deploy_env_label = 'dev-<instance_id>'
    DEV = 'dev'
    # Local development mode: REST API server, temporal workers, k8s ML worker
    # etc, all running locally on dev VM and storing data in local filesystem
    # instead of S3.
    # k8s_deploy_env_label = 'dev_local-<instance_id>'
    DEV_LOCAL = 'dev_local'

    # In test mode: many tests may concurrently run by local pytest as well as
    # github test runners. Given that we've setup github runner to spin local
    # Temporal Cluster, TEST mode behaves very similar to DEV mode for local
    # run of pytest.
    # namespace = 'devtest'
    # task_queue_suffix = '-<instance_id>-test'
    # workflow_id_suffix = '-<instance_id>-test'
    # k8s_deploy_env_label = 'test-<instance_id>'
    TEST = 'test'

    @property
    def is_dev(self) -> bool:
        return self in [DeployStage.DEV, DeployStage.DEV_LOCAL]

    @property
    def is_test(self) -> bool:
        return self == DeployStage.TEST

    @property
    def is_preprod(self) -> bool:
        return self == DeployStage.PREPROD


_DEPLOY_STAGE = EnvVar(
    'DEPLOY_STAGE',
    type=DeployStage,
    required=False,
    default_value=DeployStage.PROD,
)


def deploy_stage() -> DeployStage:
    """Read ``DEPLOY_STAGE`` from the environment.

    Public, lightweight accessor — does not trigger the
    ``deploy_env()`` singleton init chain, so it's safe to call
    before ``initialize_logging()`` runs.
    """
    return _DEPLOY_STAGE.parse_value()
