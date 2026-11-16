# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

_T = TypeVar('_T')


def only_instance(results, expected_type: type[_T]) -> _T:
    matches = [result for result in results if isinstance(result, expected_type)]
    assert len(matches) == 1
    return matches[0]
