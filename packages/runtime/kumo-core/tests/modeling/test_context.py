# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

import kumo_core.modeling.context as modeling_context
def test_explaining() -> None:
    assert not modeling_context.is_explaining()
    modeling_context.set_explaining()
    assert modeling_context.is_explaining()
    modeling_context.set_explaining(False)
    assert not modeling_context.is_explaining()

    with modeling_context.explaining():
        assert modeling_context.is_explaining()
    assert not modeling_context.is_explaining()


def test_explaining_restores_on_exception() -> None:
    # Must restore the global even if the body raises, else it leaks into
    # later requests.
    assert not modeling_context.is_explaining()
    with pytest.raises(RuntimeError):
        with modeling_context.explaining():
            assert modeling_context.is_explaining()
            raise RuntimeError("boom")
    assert not modeling_context.is_explaining()


def test_debugging() -> None:
    assert not modeling_context.is_debugging()
    modeling_context.set_debugging()
    assert modeling_context.is_debugging()
    modeling_context.set_debugging(False)
    assert not modeling_context.is_debugging()

    with modeling_context.debugging():
        assert modeling_context.is_debugging()
    assert not modeling_context.is_debugging()
