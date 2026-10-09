# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A ``__getattr__`` / ``__dir__`` pair for a package that exports some of
its names lazily.

Every plugin package (``kumo_snowflake``, ``kumo_databricks``,
``kumo_bigquery``, ``kumo_s3``, ``kumo_gcs``) keeps its platform-facing reader
importable with no vendor client library installed, and exposes its
``LocalBackend`` plugin half -- which does need one -- only when a caller
actually asks for it by name. Each package used to carry an identical
``__getattr__`` / ``__dir__`` copy to do this; this module is that copy,
written once.
"""

import sys
from importlib import import_module
from typing import Any, Callable, Dict, List, Sequence


def module_getattr(
    package: str,
    exports: Dict[str, str],
) -> Callable[[str], Any]:
    r"""Builds the ``__getattr__`` a package assigns to resolve
    :obj:`exports` lazily.

    A resolved name is written back into :obj:`package`'s own namespace, so
    this runs once per name and every later lookup is ordinary attribute
    access: PEP 562 consults a module ``__getattr__`` only for a name the
    module dict does not already hold.

    Args:
        package: The importing module's own ``__name__``.
        exports: Maps an exported name to the submodule (relative to
            :obj:`package`) that defines it.
    """
    def __getattr__(name: str) -> Any:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f'module {package!r} has no attribute '
                                 f'{name!r}')
        value = getattr(import_module(f'{package}.{module}'), name)
        importing = sys.modules.get(package)
        if importing is not None:
            setattr(importing, name, value)
        return value

    return __getattr__


def module_dir(all_names: Sequence[str]) -> Callable[[], List[str]]:
    r"""Builds the ``__dir__`` matching a :func:`module_getattr` package's
    own ``__all__``.
    """
    def __dir__() -> List[str]:
        return sorted(all_names)

    return __dir__
