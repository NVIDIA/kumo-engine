# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Type, TypeVar

AnyEnum = TypeVar('AnyEnum', bound=Type[Enum])


def extends_enum(base_enum_type: Type[Enum]):
    """
    Decorator to annotate and check if an enum type is extending another enum
    type, because python does not allow extending enum type directly.

    For example:
    class FooEnum(Enum):
        X = 1
        Y = 2

    @extends_enum(FooEnum)
    class ExtendedFooEnum(Enum):  # This should work
        X = 1
        Y = 2
        Z = 3

    @extends_enum(FooEnum)
    class BadExtendedFoo(Enum):  # RAISE assert error!
        X = 1
        Y = 999   # mismatched enum value
        Z = 4

    @extends_enum(FooEnum)
    class BadExtendedFoo(Enum):  # RAISE assert error!
        # Missing X = 1 enum value
        Y = 2
        Z = 3

    """
    def _decorator(extended_enum_type: AnyEnum) -> AnyEnum:
        enum_values = [(e.name, e.value) for e in extended_enum_type]
        for e in base_enum_type:
            assert (e.name, e.value) in enum_values, (
                f"{(e.name, e.value)} not in {enum_values}.")
        return extended_enum_type

    return _decorator
