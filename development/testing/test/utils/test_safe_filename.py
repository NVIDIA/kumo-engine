# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.utils.safe_filename import safe_filename


def test_safe_filename():
    file_name = 'a' * 252 + ' // ' + 'bbb'
    safe_name = safe_filename(file_name)
    assert '%20' in safe_name
    assert ' ' not in safe_name
    assert '/' not in safe_name
    assert 'b' not in safe_name
