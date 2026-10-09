# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import field
from datetime import timedelta
from typing import Dict, Optional

from temporalio.common import RetryPolicy

from io_utils import kumo_dataclass
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.auto_rpc_exception import _AutoRpcNonRetriableError

DEFAULT_RETRY_POLICY = RetryPolicy(
    maximum_interval=timedelta(seconds=1.0), maximum_attempts=3,
    non_retryable_error_types=[_AutoRpcNonRetriableError.__name__])

_DEFAULT_TIMEOUT_SEC = 30
_SERVER_NAME_PREFIX_ENV_VAR = 'KUMO_AUTORPC_SERVER_NAME_PREFIX'


def parse_auto_rpc_server_name_prefix_from_env() -> Optional[str]:
    server_name_prefix = EnvVar(
        _SERVER_NAME_PREFIX_ENV_VAR,
        str,
        required=False,
        default_value='',
    ).parse_value().strip()
    return server_name_prefix or None


@kumo_dataclass
class AutoRpcTimeoutConfig:
    """Configuration for timeouts and retry policy."""
    # Total timeout of a RPC call, including all retry attempts.
    rpc_timeout: timedelta = timedelta(seconds=_DEFAULT_TIMEOUT_SEC)
    # Total timeout for a single attempt of RPC call. If not specified, will
    # use the rpc_timeout (above) without retry.
    attempt_timeout: Optional[timedelta] = None
    # Custom retry policy. see:
    # https://docs.temporal.io/concepts/what-is-a-retry-policy
    # default_factory: RetryPolicy is a mutable dataclass instance, which v2
    # (stdlib dataclass) rejects as a bare default.
    retry_policy: Optional[RetryPolicy] = field(
        default_factory=lambda: DEFAULT_RETRY_POLICY)


@kumo_dataclass
class AutoRpcClientConfig:
    temporal_config: DeployEnvConfig

    # Configuration for timeouts and retry policy.
    timeout_config: AutoRpcTimeoutConfig = field(
        default_factory=AutoRpcTimeoutConfig)

    # Configuration of timeout/retry at per method level. This takes precedence
    # over above default timeout_config at service level.
    method_timeout_config: Dict[str, AutoRpcTimeoutConfig] = field(
        default_factory=dict)

    # A service may be configured to deploy at multiple "clusters", with each
    # cluster serving different clients or purposes. In that case, set the
    # server_name_prefix to distinguish.
    server_name_prefix: Optional[str] = None

    @staticmethod
    def parse_from_env(
            service_cls: type, config: Optional[DeployEnvConfig] = None,
            env_var_prefix: str = 'RPC_TIMEOUT_') -> 'AutoRpcClientConfig':
        timeout_env_var = f'{env_var_prefix}{service_cls.__name__.upper()}'
        service_timeout = EnvVar(timeout_env_var, int, True)
        if service_timeout.is_set():
            rpc_timeout = timedelta(seconds=service_timeout.parse_value())
        else:
            rpc_timeout_sec = EnvVar(
                'RPC_TIMEOUT_DEFAULT', type=int, required=False,
                default_value=_DEFAULT_TIMEOUT_SEC).parse_value()
            rpc_timeout = timedelta(seconds=rpc_timeout_sec)

        # Per-method rpc timeout is configured via "RPC_TIMEOUT_MYSERVICE__FOO"
        per_method_timeout_prefix = timeout_env_var + '__'
        method_timeout_config: Dict[str, AutoRpcTimeoutConfig] = dict()
        for env_var in os.environ:
            if env_var.startswith(per_method_timeout_prefix):
                method = env_var[len(per_method_timeout_prefix):].lower()
                method_timeout_config[method] = AutoRpcTimeoutConfig(
                    rpc_timeout=timedelta(
                        seconds=EnvVar(env_var, int, True).parse_value()))

        if config is None:
            config = deploy_env()
        return AutoRpcClientConfig(
            config,
            timeout_config=AutoRpcTimeoutConfig(rpc_timeout=rpc_timeout),
            method_timeout_config=method_timeout_config,
            server_name_prefix=parse_auto_rpc_server_name_prefix_from_env())
