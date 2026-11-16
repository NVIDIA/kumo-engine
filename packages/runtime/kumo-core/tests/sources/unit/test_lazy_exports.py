# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest
from kumo_core.sources.lazy_exports import module_dir, module_getattr


@pytest.fixture
def package():
    r"""A package and one submodule, wired the way a plugin distribution
    wires itself: ``__getattr__`` assigned on the package, the exported name
    defined only on the submodule.

    Registered in :obj:`sys.modules` because that is where
    :func:`module_getattr` writes a resolved name back to, and reached
    through ordinary attribute access so PEP 562's own dispatch is what the
    assertions go through -- calling the returned closure directly would
    prove the closure works and leave the wiring untested.
    """
    submodule = types.ModuleType('scratch_pkg.backend')
    submodule.Thing = object()
    package = types.ModuleType('scratch_pkg')
    package.__getattr__ = module_getattr('scratch_pkg',
                                         {'Thing': 'backend'})
    sys.modules['scratch_pkg'] = package
    sys.modules['scratch_pkg.backend'] = submodule
    try:
        yield package, submodule
    finally:
        del sys.modules['scratch_pkg.backend']
        del sys.modules['scratch_pkg']


def test_a_listed_name_resolves_from_its_submodule(package):
    imported, submodule = package
    assert imported.Thing is submodule.Thing


def test_a_resolved_name_is_cached_in_the_package_namespace(package):
    r"""The second lookup must not re-enter ``__getattr__``: a plugin binds
    its backend once per name, and re-importing on every access would make
    an attribute read cost an ``importlib`` round trip forever.
    """
    imported, submodule = package
    assert imported.Thing is submodule.Thing
    assert vars(imported)['Thing'] is submodule.Thing

    def fail(name):
        raise AssertionError(f'__getattr__ ran again for {name!r}')

    imported.__getattr__ = fail
    assert imported.Thing is submodule.Thing


def test_an_unlisted_name_raises_attribute_error_naming_the_package(package):
    imported, _ = package
    with pytest.raises(AttributeError, match="'scratch_pkg'.*'nothing'"):
        imported.nothing


def test_a_package_with_no_module_entry_still_resolves():
    r"""A package mid-import is not yet in :obj:`sys.modules` on every
    import machinery; resolving must not depend on the write-back finding it.
    """
    submodule = types.ModuleType('unregistered_pkg.backend')
    submodule.Thing = object()
    sys.modules['unregistered_pkg.backend'] = submodule
    try:
        getattr_ = module_getattr('unregistered_pkg', {'Thing': 'backend'})
        assert getattr_('Thing') is submodule.Thing
    finally:
        del sys.modules['unregistered_pkg.backend']


def test_module_dir_returns_all_names_sorted():
    dir_ = module_dir(['c', 'a', 'b'])
    assert dir_() == ['a', 'b', 'c']
