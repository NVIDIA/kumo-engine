# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import shutil

import pytest
from kumoapi.typing import Dtype

from kumo.home import get_home_dir
from kumo.testing.generator import ColumnSpec, EdgeSpec, GraphSpec, TableSpec

TEST_GRAPH_NAME = 'kumo_generator_test_graph'


def teardown_function():
    folder = osp.join(get_home_dir(), 'graphs', TEST_GRAPH_NAME)
    shutil.rmtree(folder, ignore_errors=True)


def test_generate_graph():
    graph_spec = GraphSpec(
        table_specs={
            0: TableSpec(100,
                         [Dtype.int, (Dtype.float, 0.8), (Dtype.string, 0.4)]),
            1: TableSpec(50, [Dtype.date, Dtype.time]),
            2: TableSpec(25, [Dtype.int, Dtype.bool])
        },
        edge_specs={
            (0, 1): EdgeSpec(0.5),
            (1, 0): EdgeSpec(1),
            (0, 2): EdgeSpec(1),
            (1, 2): EdgeSpec(2)
        },
    )

    graph = graph_spec.generate(TEST_GRAPH_NAME)
    assert graph.num_tables == 3

    graph.infer_metadata()
    graph.infer_encoders()
    graph.validate()

    # High NA fraction
    assert graph['t0', '1'].na_fraction() >= 0.8


def test_validation_errors():
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        _ = ColumnSpec(dtype=None, na_frac=2)

    with pytest.raises(ValueError, match="must be integral"):
        _ = EdgeSpec(2.5)


def test_generate_graph_multiedges():
    graph_spec = GraphSpec(
        table_specs={
            0: TableSpec(100,
                         [Dtype.int, (Dtype.float, 0.8), (Dtype.string, 0.4)]),
            1: TableSpec(50, [Dtype.date, Dtype.time]),
            2: TableSpec(25, [Dtype.int, Dtype.bool]),
            3: TableSpec(15, [Dtype.int])
        },
        edge_specs={
            (0, 1): EdgeSpec(0.5),
            (1, 2): EdgeSpec(2),
            (0, 1, 3): EdgeSpec(3),
            (0, 1, 2): EdgeSpec(25)
        },
    )

    graph = graph_spec.generate(TEST_GRAPH_NAME)
    assert graph.num_tables == 4
    graph.infer_metadata()
    graph.infer_encoders()
    graph.validate()
