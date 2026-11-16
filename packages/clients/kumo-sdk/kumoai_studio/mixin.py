# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Optional, Type, TypeVar

from kumoapi.json_serde import dataclass_to_dict

T = TypeVar('T')


class CastMixin:
    @classmethod
    def _cast(
        cls: Type[T],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[T]:
        # TODO clean up type hints
        # TODO can we apply this recursively?
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, cls):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
            if dataclasses.is_dataclass(elem):
                return cls(**dataclass_to_dict(elem))
        return cls(*args, **kwargs)
