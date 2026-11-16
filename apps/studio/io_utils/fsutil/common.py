# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import logging
import os
import os.path as osp
import tempfile
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator, Optional, TypeVar

from typing_extensions import ParamSpec

from .credentials import ContextPropagatingThreadPoolExecutor
from .uri import UCURI, LocalURI, ParseURI

logger = logging.getLogger(__name__)

DEFAULT_DATA_VERSION = -1.0

P = ParamSpec("P")
R = TypeVar("R")


def fsExceptionHandler(func: Callable[P, R]) -> Callable[P, R]:
    r"""Decorator to handle exceptions from filesystem operations."""
    @wraps(func)
    def _fsutil_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: E722
            # NOTE Do not log `e` here!!! It contains a reference to the
            # traceback which will result in a circular dependency and a
            # memory leak!
            # https://cosmicpercolator.com/2016/01/13/exception-leaks-in-python-2-and-3/
            logger.error("%s triggers %s error with args: %s, kwargs: %s",
                         func.__name__, type(e), args, kwargs)
            raise e

    return _fsutil_wrapper


class FsutilContext:
    THREAD_POOL_SIZE = 50

    def __init__(self):
        self.thread_pool = ContextPropagatingThreadPoolExecutor(
            thread_name_prefix='FSUtil-Threadpool',
            max_workers=FsutilContext.THREAD_POOL_SIZE)


fsutil_ctx = FsutilContext()


def match_any(path: str, patterns: Optional[list[str]] = None) -> bool:
    if patterns is None:
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def should_exclude(path: str, exclude: Optional[list[str]] = None) -> bool:
    if exclude is None or len(exclude) == 0:
        return False
    return match_any(path, exclude)


def should_include(path: str, include: Optional[list[str]] = None) -> bool:
    if include is None or len(include) == 0:
        return True
    return match_any(path, include)


def is_local(path: str) -> bool:
    parsed = ParseURI(path)
    if isinstance(parsed, LocalURI):
        return True
    return False


def is_ucv(path: str) -> bool:
    return isinstance(ParseURI(path), UCURI)


def get_scheme(path: str) -> str:
    """Return the URI scheme for the given path.

    The value comes from the ``scheme`` attribute on the parsed URI object
    (defined in ``uri.py``).  Returns one of: ``'s3'``, ``'gs'``, ``'dbfs'``,
    ``'abfss'``, ``'local'``.
    """
    parsed = ParseURI(path)
    return parsed.scheme


def support_fs(path: str) -> bool:
    r"""Returns true if the given path can be used via fs API.
    Certain functions require this to be true, such as:
    * df.to_parquet
    * df.to_csv
    * pyarrow.parquet.ParquetWriter
    """
    from io_utils.fsutil.generic import is_s3
    return is_local(path) or is_s3(path)


def _temp_prefix() -> Optional[str]:
    # EC2 Pushdown case
    if osp.isdir("/mnt/data/"):
        temp_root = "/mnt/data/tmp/"
        os.makedirs(temp_root, exist_ok=True)
        return temp_root

    return None


@contextmanager
def TemporaryDirectory() -> Generator[str, None, None]:
    """Create and return a unique temporary directory.
    Usage:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """
    with tempfile.TemporaryDirectory(prefix=_temp_prefix()) as temp_dir:
        yield temp_dir


def mkdtemp() -> str:
    """Create and return a unique temporary directory.
    The return value is the pathname of the directory.
    Caller is responsible for deleting the directory when done with it.
    """
    return tempfile.mkdtemp(dir=_temp_prefix())
