# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typing

import pytest
from pydantic import model_validator

from io_utils import kumo_dataclass
from io_utils.dataclasses import before_validator_dict
from io_utils.json import from_json, to_json


@kumo_dataclass
class Cls:
    x: int
    y: str


@kumo_dataclass
class MissingY:
    x: int


@kumo_dataclass
class WithBeforeValidator:
    x: int = 0

    @model_validator(mode='before')
    @classmethod
    def _normalize(cls, data: typing.Any) -> typing.Any:
        return before_validator_dict(data)


def test_remove_field():
    # Ensure that the dataclass is forwards-compatible (old code should ignore
    # newly added field).
    x = Cls(x=1, y="b")
    data = to_json(x)
    y = from_json(data, MissingY)
    assert y.x == x.x


def test_before_validator_dict_keyword_ok():
    # Keyword / mapping construction is normalized to a dict and round-trips.
    assert WithBeforeValidator(x=5).x == 5
    assert from_json(to_json(WithBeforeValidator(x=5)),
                     WithBeforeValidator).x == 5


def test_before_validator_dict_rejects_positional():
    # Positional args used to be silently dropped (``WithBeforeValidator(5)``
    # became ``x=0``); they must now raise instead of losing the value.
    with pytest.raises(Exception):
        WithBeforeValidator(5)
