# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from async_utils.caching import EventLoopBlockedError
from io_utils.exceptions import catch_and_log


def test_catch_and_log(caplog):
    # This should not raise an error
    with catch_and_log():
        raise ValueError("This will be caught and logged")
    assert (caplog.records[0].message ==
            "Encountered exception: This will be caught and logged")
    caplog.clear()

    # This also should not raise an error
    with catch_and_log(ex_to_catch=(ValueError, TypeError)):
        raise TypeError("This will be caught and logged")
    assert (caplog.records[0].message ==
            "Encountered exception: This will be caught and logged")
    caplog.clear()

    # This should raise an error
    with pytest.raises(AssertionError):
        with catch_and_log((ValueError, TypeError)):
            raise AssertionError("This exception will still be raised")

    with pytest.raises(ValueError):
        with catch_and_log(ex_to_raise=(ValueError, )):
            raise ValueError('This will be raised')

    with pytest.raises(EventLoopBlockedError):
        with catch_and_log():
            raise EventLoopBlockedError('This will be raised')
