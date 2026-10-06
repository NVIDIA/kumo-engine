# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import shutil

from kumo_core.training.config import LoaderKwargs, SamplerKwargs
from torch_geometric.sampler import NumNeighbors

from kumo.home import get_home_dir


def teardown_function():
    # Delete everything in tasks, graphs directory too:
    folder = osp.join(get_home_dir(), 'graphs')
    shutil.rmtree(folder, ignore_errors=True)

    folder = osp.join(get_home_dir(), 'tasks')
    shutil.rmtree(folder, ignore_errors=True)


def test_sampler_kwargs_defaults_and_coercion():
    # No-arg path: supplies the default `num_neighbors` of [-1, -1].
    default = SamplerKwargs()
    assert isinstance(default.num_neighbors, NumNeighbors)
    assert default.num_neighbors.values == [-1, -1]

    # Explicit ``num_neighbors=None`` resolves to the same default.
    assert SamplerKwargs(num_neighbors=None).num_neighbors.values == [-1, -1]

    # A plain list is coerced into ``NumNeighbors``.
    coerced = SamplerKwargs(num_neighbors=[10, 10])
    assert isinstance(coerced.num_neighbors, NumNeighbors)
    assert coerced.num_neighbors.values == [10, 10]

    # An existing ``NumNeighbors`` is preserved as-is.
    nn = NumNeighbors([5, 5])
    assert SamplerKwargs(num_neighbors=nn).num_neighbors.values == [5, 5]

    # ``cast`` from a partial dict resolves defaults + coercion.
    cast = SamplerKwargs.cast({'num_neighbors': [3, 3]})
    assert cast.num_neighbors.values == [3, 3]
    # Other defaults survive casting.
    assert cast.disjoint is True
    assert cast.temporal_strategy == 'last'


def test_loader_kwargs_defaults():
    # No-arg path: supplies default ``batch_size`` and resolves
    # ``num_workers`` (0 under pytest, via the ``PYTEST_CURRENT_TEST`` rule).
    default = LoaderKwargs()
    assert default.batch_size == 1024
    assert default.num_workers == 0
    assert default.filter_per_worker is True
    assert default.pin_memory is False

    # Partial override keeps the resolved ``num_workers`` default.
    partial = LoaderKwargs(batch_size=512)
    assert partial.batch_size == 512
    assert partial.num_workers == 0

    # ``cast`` from an empty dict yields the full default instance.
    cast = LoaderKwargs.cast({})
    assert cast.batch_size == 1024
    assert cast.num_workers == 0
