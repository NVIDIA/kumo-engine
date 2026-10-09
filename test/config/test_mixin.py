# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import marshal
from dataclasses import asdict
from typing import Any

import pytest
from kumo_core.utils.registry import register
from pydantic.dataclasses import dataclass

from io_utils import kumo_dataclass
from kumo.config import ConfigMixin
from kumo.config.mixin import (
    _recursive_config,
    _recursive_from_config,
    construct_object_string_from_config,
)
from kumo.stats import Stats


@dataclass(config=dict(arbitrary_types_allowed=True))  # type: ignore
class MyDataclass:
    _target_: str = 'test.config.test_mixin.MyDataclass'
    attr: int = 0


@register(with_target=True)
class MyNestedClass(ConfigMixin):
    def __init__(self, attr):
        self.attr = attr


@kumo_dataclass
class MyClassConfig:
    attr0: int
    attr1: Any
    attr2: Any


@register(data_cls=MyClassConfig)
class MyClass(ConfigMixin):
    def __init__(self, attr0, attr1, attr2):
        self.attr0 = attr0
        self.attr1 = attr1
        self.attr2 = attr2


def _increment(value):
    return value + 1


@pytest.mark.parametrize('config_as_dict', [True, False])
def test_config(config_as_dict):
    expected_cfg_dict = {
        'attr0': 1,
        'attr1': {
            'attr': 3,
            '_target_': 'test.config.test_mixin.MyNestedClass'
        },
        'attr2': {
            'attr': 5,
            '_target_': 'test.config.test_mixin.MyDataclass',
        },
    }
    my_instance = MyClass(1, MyNestedClass(attr=3), MyDataclass(attr=5))
    cfg = my_instance.config()
    cfg_dict = asdict(cfg)
    assert cfg_dict == expected_cfg_dict

    out = MyClass.from_config(cfg_dict if config_as_dict else cfg)
    assert out.attr0 == 1
    assert isinstance(out.attr1, MyNestedClass)
    assert out.attr1.attr == 3
    assert isinstance(out.attr2, MyDataclass)
    assert out.attr2.attr == 5


@pytest.mark.parametrize('config_as_dict', [True, False])
def test_bad_config(config_as_dict):
    bad_cfg_dict = {
        'attr0': 1,
        'attr1': {
            'attr': 3,
            '_target_': 'test.config.test_mixin.MyNestedClass'
        },
        'attr2': {
            'attr': 5,
            '_target_': 'test.config.test_mixin.MyDataclass',
        },
        'bad_attr': 7
    }
    with pytest.warns(UserWarning, match='bad_attr'):
        out = MyClass.from_config(bad_cfg_dict)

    assert out.attr0 == 1
    assert isinstance(out.attr1, MyNestedClass)
    assert out.attr1.attr == 3
    assert isinstance(out.attr2, MyDataclass)
    assert out.attr2.attr == 5


@pytest.mark.parametrize(
    'value',
    [
        b'\x00binary\xff',
        bytearray(b'bytes'),
        {
            'raw': b'\x01\x02',
            'nested': [bytearray(b'category'), {
                'value': b'\xff'
            }],
        },
    ],
)
def test_binary_values_are_not_treated_as_marshaled_functions(value):
    assert _recursive_from_config(_recursive_config(value)) == value


def test_function_config_uses_explicit_marshaled_code_marker():
    config = _recursive_config(_increment)

    assert isinstance(config, dict)
    assert config['__kumo_config_type__'] == 'marshalled_function_v1'
    assert set(config) == {'__kumo_config_type__', 'code'}
    restored = _recursive_from_config(config)
    assert restored(2) == 3


@pytest.mark.parametrize(
    ('config', 'error'),
    [
        ({
            '__kumo_config_type__': 'marshalled_function_v1'
        }, ValueError),
        ({
            '__kumo_config_type__': 'unknown',
            'code': b''
        }, ValueError),
        ({
            '__kumo_config_type__': 'marshalled_function_v1',
            'code': b'',
            'extra': True,
        }, ValueError),
        ({
            '__kumo_config_type__': 'marshalled_function_v1',
            'code': 'bad'
        }, TypeError),
        ({
            '__kumo_config_type__': 'marshalled_function_v1',
            'code': b'not-bytecode',
        }, ValueError),
        ({
            '__kumo_config_type__': 'marshalled_function_v1',
            'code': marshal.dumps(123),
        }, TypeError),
    ],
)
def test_malformed_marshaled_function_payload_is_rejected(config, error):
    with pytest.raises(error):
        _recursive_from_config(config)


def test_binary_categorical_stats_config_round_trip():
    stats = Stats(
        value=[(b'\x00category', 4), (b'\xffcategory', 2)],
        approx=False,
        data_version='binary-v1',
    )

    restored = Stats.from_config(stats.config())

    assert restored.value == [[b'\x00category', 4], [b'\xffcategory', 2]]
    assert restored.approx is False
    assert restored.data_version == 'binary-v1'


def test_construct_object_string_from_config():
    # config as a dataclass
    config = MyDataclass(attr=123)
    assert (
        construct_object_string_from_config(config) == 'MyDataclass(attr=123)')

    # config as a dict
    config = {'attr': 123, '_target_': 'test.config.test_mixin.MyDataclass'}
    assert (
        construct_object_string_from_config(config) == 'MyDataclass(attr=123)')

    # config as a ConfigMixin
    config = MyNestedClass(attr=123)
    assert (construct_object_string_from_config(config) ==
            'MyNestedClass(attr=123)')


@pytest.mark.parametrize(
    'load_config',
    [
        lambda config: MyClass.from_config(config),
        _recursive_from_config,
        construct_object_string_from_config,
    ],
)
def test_unknown_config_target_fails_explicitly(load_config):
    config = {
        '_target_': 'missing.module.Owner',
        'attr0': 1,
        'attr1': None,
        'attr2': None,
    }

    with pytest.raises(
            ValueError,
            match=r'unsupported config target: missing\.module\.Owner'):
        load_config(config)
