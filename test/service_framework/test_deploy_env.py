# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from enum import Enum
from unittest.mock import PropertyMock, patch

import pytest

from service_framework import deploy_env
from service_framework.deploy_env import DeployStage
from service_framework.env_var import EnvVar


class IntEnumValue(Enum):
    UNDEFINED = 0
    VALUE_1 = 1
    VALUE_2 = 2


@pytest.mark.parametrize('stage', list(DeployStage))
def test_instance_discovery_respects_deployment_stage(monkeypatch, stage):
    monkeypatch.setenv('DEPLOY_STAGE', stage.value)
    monkeypatch.setattr(deploy_env.socket, 'gethostname', lambda: 'local-host')
    with patch.object(type(deploy_env.ec2_metadata), 'instance_id',
                      new_callable=PropertyMock,
                      return_value='i-test-instance') as metadata:
        loaded = runpy.run_path(deploy_env.__file__)
        local = stage in (DeployStage.TEST, DeployStage.DEV_LOCAL)
        expected = 'local-host' if local else 'i-test-instance'
        assert loaded['MACHINE_INSTANCE_ID'] == expected
        assert loaded['get_instance_id']() == expected
        if local:
            metadata.assert_not_called()
        else:
            metadata.assert_called()


@pytest.mark.parametrize('snowflake', [False, True])
def test_instance_discovery_keeps_cloud_fallbacks(monkeypatch, snowflake):
    monkeypatch.setenv('DEPLOY_STAGE', 'prod')
    monkeypatch.setattr(deploy_env.socket, 'gethostname', lambda: 'local-host')
    monkeypatch.setattr(deploy_env, 'SNOWFLAKE_ACCOUNT',
                        'account' if snowflake else None)
    monkeypatch.setattr(deploy_env, 'SNOWFLAKE_DATABASE', 'database')
    monkeypatch.setattr(deploy_env, 'SNOWFLAKE_SCHEMA', 'schema')
    monkeypatch.setattr(deploy_env, 'SNOWFLAKE_SERVICE', 'service')
    with patch.object(type(deploy_env.ec2_metadata), 'instance_id',
                      new_callable=PropertyMock,
                      side_effect=RuntimeError('metadata unavailable')):
        expected = 'account_database_schema_service' if snowflake else 'local-host'
        assert deploy_env.get_instance_id() == expected


def test_env_var_default(monkeypatch: pytest.MonkeyPatch):
    name = "TEST_VAR"
    val = "val1"
    default1 = "default_val1"
    default2 = "default_val2"

    var = EnvVar(name=name, type=str, default_value=default1)

    assert default2 == var.parse_value(default=default2)
    assert default1 == var.parse_value()

    monkeypatch.setenv(name, val)
    assert val == var.parse_value(default=default2)
    assert val == var.parse_value()


def test_env_var_parsing(monkeypatch: pytest.MonkeyPatch):
    required_bool = EnvVar('BOOLEAN_VAR', type=bool, required=True)
    with pytest.raises(ValueError):
        required_bool.parse_value()

    monkeypatch.setenv('BOOLEAN_VAR', '')
    with pytest.raises(ValueError):
        required_bool.parse_value()

    monkeypatch.setenv('BOOLEAN_VAR', '123')
    with pytest.raises(ValueError):
        required_bool.parse_value()

    monkeypatch.setenv('BOOLEAN_VAR', '1')
    assert required_bool.parse_value()
    monkeypatch.setenv('BOOLEAN_VAR', '0')
    assert not required_bool.parse_value()
    monkeypatch.setenv('BOOLEAN_VAR', 'True')
    assert required_bool.parse_value()
    monkeypatch.setenv('BOOLEAN_VAR', 'falSE')
    assert not required_bool.parse_value()

    optional_int = EnvVar('INTEGER_VAR', int, False, default_value=123)
    assert optional_int.parse_value() == 123
    monkeypatch.setenv('INTEGER_VAR', '456')
    assert optional_int.parse_value() == 456
    monkeypatch.setenv('INTEGER_VAR', '')
    assert optional_int.parse_value() == 123
    monkeypatch.setenv('INTEGER_VAR', 'invalid')
    with pytest.raises(ValueError):
        optional_int.parse_value()

    optional_float = EnvVar('FLOAT_VAR', float, False, default_value=1.05)
    assert optional_float.parse_value() == 1.05
    monkeypatch.setenv('FLOAT_VAR', '')
    assert optional_float.parse_value() == 1.05
    monkeypatch.setenv('FLOAT_VAR', '10.01')
    assert optional_float.parse_value() == 10.01
    monkeypatch.setenv('FLOAT_VAR', '-3.14159E3')
    assert optional_float.parse_value() == -3141.59
    monkeypatch.setenv('FLOAT_VAR', 'invalid')
    with pytest.raises(ValueError):
        optional_float.parse_value()

    int_enum_var = EnvVar('INT_ENUM_VAR', IntEnumValue, False,
                          default_value=IntEnumValue.UNDEFINED)
    assert int_enum_var.parse_value() == IntEnumValue.UNDEFINED
    monkeypatch.setenv('INT_ENUM_VAR', 'VALUE_1')
    assert int_enum_var.parse_value() == IntEnumValue.VALUE_1
    monkeypatch.setenv('INT_ENUM_VAR', '1')
    with pytest.raises(ValueError):
        int_enum_var.parse_value()

    # str enum type
    deploy_stage_var = EnvVar('DEPLOY_STAGE_VAR', DeployStage, False,
                              default_value=DeployStage.TEST)
    assert deploy_stage_var.parse_value() == DeployStage.TEST
    monkeypatch.setenv('DEPLOY_STAGE_VAR', '')
    assert deploy_stage_var.parse_value() == DeployStage.TEST
    monkeypatch.setenv('DEPLOY_STAGE_VAR', 'prod')
    assert deploy_stage_var.parse_value() == DeployStage.PROD
    monkeypatch.setenv('DEPLOY_STAGE_VAR', 'PROD')
    assert deploy_stage_var.parse_value() == DeployStage.PROD
    monkeypatch.setenv('DEPLOY_STAGE_VAR', 'invalid')
    with pytest.raises(ValueError):
        deploy_stage_var.parse_value()
