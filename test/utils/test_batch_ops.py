# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.utils.batch_ops import raise_errors


def test_raise_errors():
    assert raise_errors([1, None], "Message") == [1, None]
    with pytest.raises(Exception, match="Message"):
        raise_errors([1, Exception("exc")], "Message")
