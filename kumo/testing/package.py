# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from importlib.util import find_spec
from shutil import which


def withPackage(*args, executables=()) -> Callable:
    r"""Skip tests unless their Python packages and binaries are available."""
    na_packages = {arg for arg in args if find_spec(arg) is None}
    na_executables = {
        executable
        for executable in executables if which(executable) is None
    }

    def decorator(func: Callable) -> Callable:
        import pytest

        return pytest.mark.skipif(
            bool(na_packages or na_executables),
            reason=(f"Package(s) {na_packages} and executable(s) "
                    f"{na_executables} are not installed"),
        )(func)

    return decorator


def withRegisteredOp(*args) -> Callable:
    r"""A decorator to skip tests if a certain op is not registered."""
    def is_registered(op: str) -> bool:
        import torch
        module = torch.ops
        for attr in op.split('.'):
            try:
                module = getattr(module, attr)
            except RuntimeError:
                return False
        return True

    na_ops = {arg for arg in args if not is_registered(arg)}

    def decorator(func: Callable) -> Callable:
        import pytest

        return pytest.mark.skipif(
            len(na_ops) > 0,
            reason=f"Operator(s) {na_ops} are not registered",
        )(func)

    return decorator
