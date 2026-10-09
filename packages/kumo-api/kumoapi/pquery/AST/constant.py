# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import datetime
import json
import logging
from typing import Any, List, Union

import pandas as pd
from pydantic.dataclasses import dataclass

from kumoapi.pquery.AST.ast_node import ArrayDtype, ASTNode
from kumoapi.typing import _FORBID_EXTRA_CONFIG, Dtype

_KUMO_INTERNAL_TIMESTAMP_PREFIX = 'kumo_timestamp_'

logger = logging.getLogger(__name__)


@dataclass(repr=False, config=_FORBID_EXTRA_CONFIG)
class Constant(ASTNode):
    r"""Creates an atomic description of a constant.
    Args:
        value: Value representation as it appeared in the input query.
    """
    value: Union[List['Constant'], str, None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value is None:
            self.value = 'NULL'
        try:
            self.typed_value()
        except Exception as e:
            raise TypeError(
                f'{self.value} is not a valid instance of {self.dtype_maybe}'
            ) from e

    @staticmethod
    def value_type_cast_to_internal_value(
        value: Union[str, int, float, List[Any], Any]
    ) -> Union[str, List['Constant']]:
        r"""If value is int or float, cast value type to string.
            If value is string, cast value to its appropriate format.
            If value is a list, cast each element in value to its appropriate
            format.
            """
        if isinstance(value, (list, tuple)):
            value = [Constant.from_value(x) for x in value]
            return value
        elif isinstance(value, str):
            value = json.dumps(value)  # Escape special characters.
        elif isinstance(value, datetime.datetime):
            return str(pd.Timestamp(value))
        return str(value)

    @classmethod
    def from_value(cls, value: Any) -> 'Constant':
        dtype = None
        if value is None:
            return cls(value='NULL', dtype_maybe=None)
        if isinstance(value, bool):
            dtype = Dtype.bool
        elif isinstance(value, int):
            dtype = Dtype.int
        elif isinstance(value, float):
            dtype = Dtype.float
        elif isinstance(value,
                        (pd.Timestamp, datetime.datetime, datetime.date)):
            dtype = Dtype.time
        elif (isinstance(value, str)
              and value.startswith(_KUMO_INTERNAL_TIMESTAMP_PREFIX)):
            # This is for backward compatibility with old configs
            value = value[len(_KUMO_INTERNAL_TIMESTAMP_PREFIX):]
            try:
                value = pd.Timestamp(value)
            except TypeError:
                logger.warning(
                    f"Time value {value} in config is invalid, using pd.NaT")
                value = pd.NaT
            dtype = Dtype.time
        elif isinstance(value, list):
            if all(isinstance(x, int) for x in value):
                dtype = Dtype.intlist
            elif all(isinstance(x, (int, float)) for x in value):
                dtype = Dtype.floatlist
            elif all(isinstance(x, str) for x in value):
                dtype = Dtype.stringlist
        elif isinstance(value, str):
            dtype = Dtype.string
        if dtype is None:
            raise TypeError(
                f'Unsupported constant {value} of type {type(value)}.')
        return cls(value=cls.value_type_cast_to_internal_value(value),
                   dtype_maybe=dtype)

    def typed_value(
            self,  #
    ) -> Union[int, float, str, pd.Timestamp, List[Any], None]:
        if self.dtype_maybe is None:
            return None
        assert self.value is not None
        assert isinstance(self.dtype_maybe, (Dtype, ArrayDtype))
        if (isinstance(self.dtype_maybe, ArrayDtype)
                or self.dtype_maybe.is_list()):
            assert isinstance(self.value, list)
            return [v.typed_value() for v in self.value]
        assert isinstance(self.dtype_maybe, Dtype)
        if self.dtype_maybe.is_bool():
            return str(self.value).lower() == 'true'
        if self.dtype_maybe.is_int():
            return int(str(self.value))
        if self.dtype_maybe.is_float():
            return float(str(self.value))
        if self.dtype_maybe.is_string():
            return str(self.value)[1:-1].encode(
                'latin-1', 'backslashreplace').decode('unicode-escape')
        if self.dtype_maybe.is_timestamp():
            value = pd.to_datetime(self.value)
            assert isinstance(value, pd.Timestamp)
            return value
        return self.value

    def to_string(self, rich: bool = False) -> str:
        assert self.value is not None
        if isinstance(self.value, list):
            return '(' + ', '.join([v.to_string() for v in self.value]) + ')'
        return self.value
