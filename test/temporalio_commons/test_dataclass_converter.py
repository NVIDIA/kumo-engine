# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import pytest

from io_utils import fsutil
from io_utils.json import load_json, to_json_dict
from rest.config.api_command import FixtureDefinition
from statedb.config.common import (
    ConnectorID,
    ResourceTimestamps,
    TableIDv1,
    now_utc,
)
from temporalio_commons.dataclass_converter import KumoPayloadConverter


class StringSubclass(str):
    pass


def scan_test_cases() -> List[str]:
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       '../rest/integ/test_cases')
    return fsutil.listdir(dir, suffix='.json')


@pytest.fixture(params=scan_test_cases())
def create_args_fixture(request) -> FixtureDefinition:
    return load_json(request.param, FixtureDefinition)


def _validate(obj):
    assert _round_trip_encode(obj) == obj
    assert _round_trip_encode_type_hint(obj) == obj


def _round_trip_encode(obj):
    converter = KumoPayloadConverter()
    return converter.from_payloads(converter.to_payloads([obj]))[0]


def _round_trip_encode_type_hint(obj):
    converter = KumoPayloadConverter()
    return converter.from_payloads(converter.to_payloads([obj]), [type(obj)])[0]


def test_convert_complex_dataclass(create_args_fixture: FixtureDefinition):
    value = _round_trip_encode(create_args_fixture)
    assert to_json_dict(value) == to_json_dict(
        to_json_dict(create_args_fixture))

    value = _round_trip_encode_type_hint(create_args_fixture)
    assert to_json_dict(value) == to_json_dict(
        to_json_dict(create_args_fixture))


def test_convert_simple_dataclass():
    timestamps = ResourceTimestamps(created_at=now_utc(), updated_at=now_utc())
    _validate(timestamps)


def test_convert_set_list(create_args_fixture: FixtureDefinition):
    _validate(set([1, 2, 3, 4, 5]))
    connector_id = ConnectorID('connector_' + '1' * 32)
    _validate(set([connector_id, TableIDv1(f'{connector_id}.table2')]))
    _validate([4, 5, 6, 7, 8])
    _validate(list([create_args_fixture, create_args_fixture]))


def test_convert_dict(create_args_fixture: FixtureDefinition):
    _validate(_round_trip_encode({}))
    _validate(_round_trip_encode({1: 1}))
    _validate(_round_trip_encode({'a': None}))
    _validate(_round_trip_encode({'a': 'b', 'c': 'd'}))

    _validate({1: create_args_fixture, 2: create_args_fixture})
    _validate({'a': create_args_fixture, 'b': create_args_fixture})


def test_convert_string_subtype():
    s = StringSubclass("abc")
    _validate(s)
