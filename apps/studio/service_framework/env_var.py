# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Generic, Optional, Type, TypeVar, Union

from pydantic import model_validator
from pydantic.dataclasses import dataclass

# NOTE smart-union ordering is preserved by member order. pydantic v2 always
# uses "smart" union matching, so the v1 `smart_union` config flag is gone.
T = TypeVar('T', bound=Union[bool, int, float, Enum, str])


@dataclass
class EnvVar(Generic[T]):
    name: str
    type: Type[T]
    # Env var can be either require with None default value; or optional with
    # a default value that's not None.
    required: bool = False
    default_value: Optional[T] = None
    description: str = ''

    # v1 @root_validator (post-validation, read `values`) ->
    # @model_validator(mode='after') reading `self`.
    @model_validator(mode='after')
    def _check_invariants(self) -> 'EnvVar':
        is_required = bool(self.required)
        default_value_exists = self.default_value is not None
        if not is_required ^ default_value_exists:
            raise ValueError(
                f'{self.name} is required: {is_required} but has default '
                f'value: {default_value_exists}')
        return self

    def is_set(self) -> bool:
        return os.environ.get(self.name)

    def parse_value(self, default: Optional[T] = None) -> T:
        raw_value = os.environ.get(self.name)
        if self.required and not raw_value:
            raise ValueError(f'{self.name} must be set in Env variable.')
        if raw_value:
            return self._evaluate_raw_value(raw_value)
        if default is not None:
            return default
        assert self.default_value is not None
        return self.default_value

    def _evaluate_raw_value(self, raw_value: str) -> T:
        if self.type == bool:
            bool_val_str = raw_value.strip().lower()
            if bool_val_str in ('1', 'enabled', 'true', 'yes', 'on'):
                return True
            if bool_val_str in ('0', 'disabled', 'false', 'no', 'off'):
                return False
            raise ValueError(f'Unable to parse bool value from {raw_value}')
        if issubclass(self.type, Enum):
            try:
                raw_value = raw_value.strip("\"")  # support string values
                return self.type[raw_value]  # by enum member name
            except KeyError:
                # Pass through to convert to enum by value (which only works
                # for str enum).
                pass
        return self.type(raw_value)
