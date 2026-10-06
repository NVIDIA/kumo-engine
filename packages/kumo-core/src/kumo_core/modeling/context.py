# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_is_explaining: bool = False
_is_debugging: bool = False


def is_explaining() -> bool:
    r"""Returns :obj:`True` during computation of gradient explanations and
    :obj:`False` otherwise. This is useful whenever modules need to store or
    behave differently during explainability.

    .. testcode::

        if modeling_context.is_explaining():
            print("Do something in explaining mode.")
        else:
            print("Do something in non-explaining mode.")
    """
    return _is_explaining


@contextmanager
def explaining(flag: bool = True) -> Iterator[None]:
    r"""Context manager to enable explainability mode.

    Args:
        flag: Whether to turn explainability on or off.

    .. testcode::

        with modeling_context.explaining():
            model(...)  # saves inputs for calculating gradients
    """
    global _is_explaining
    _old_state = _is_explaining
    _is_explaining = flag
    try:
        yield
    finally:  # Restore on exception too, else it leaks into later requests.
        _is_explaining = _old_state


def set_explaining(flag: bool = True) -> None:
    r"""Set the process-wide explainability mode.

    Args:
        flag: Whether to turn explainability on or off.

    .. testcode::

        modeling_context.set_explaining(True)
        assert modeling_context.is_explaining()
        modeling_context.set_explaining(False)
        assert not modeling_context.is_explaining()
    """
    global _is_explaining
    _is_explaining = flag


def is_debugging() -> bool:
    r"""Returns :obj:`True` when in debug mode and :obj:`False` otherwise.

    .. testcode::

        if modeling_context.is_debugging():
            print("Do something in debug mode.")
        else:
            print("Do something in non-debug mode.")
    """
    return _is_debugging


@contextmanager
def debugging(flag: bool = True) -> Iterator[None]:
    r"""Context manager to enable debug mode.

    Args:
        flag: Whether to turn debug mode on or off.

    .. testcode::

        with modeling_context.debugging():
            model(...)
    """
    global _is_debugging
    _old_state = _is_debugging
    _is_debugging = flag
    yield
    _is_debugging = _old_state


def set_debugging(flag: bool = True) -> None:
    r"""Set the process-wide debug mode.

    Args:
        flag: Whether to turn debug mode on or off.

    .. testcode::

        modeling_context.set_debugging(True)
        assert modeling_context.is_debugging()
        modeling_context.set_debugging(False)
        assert not modeling_context.is_debugging()
    """
    global _is_debugging
    _is_debugging = flag


@contextmanager
def manual_seed(seed: int | None) -> Iterator[None]:
    r"""Context manager to temporarily set the seed for generating random
    numbers.

    Args:
        seed: The desired seed.

    .. testcode::

        with modeling_context.manual_seed(42):
            Model(...)
    """
    rng_state = torch.get_rng_state()

    if seed is not None:
        torch.manual_seed(seed)

    try:
        yield

    finally:
        if seed is not None:
            torch.set_rng_state(rng_state)
