# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.utils import format_bytes


def test_format_bytes():
    assert format_bytes(2000000000) == '1.86 GiB'
    assert format_bytes(20000000000) == '18.63 GiB'
    assert format_bytes(2000000000, 'm') == '1907.35 MiB'
    assert format_bytes(2000000) == '1.91 MiB'

    with pytest.raises(ValueError, match='Invalid output unit value x'):
        format_bytes(100, 'x')
