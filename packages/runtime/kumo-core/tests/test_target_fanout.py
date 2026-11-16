# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.automl.target_fanout import (
    clamp_inferred_fanout,
    infer_target_edge_fanout,
)

PATHS = [[("orders", "user_id", "users")]]


def rows(**counts):
    """A row-count accessor that records what it was asked for."""
    asked = []

    def row_count(table_name):
        asked.append(table_name)
        return counts.get(table_name)

    row_count.asked = asked
    return row_count


def test_the_degree_formula_is_sublinear_in_the_entity_degree():
    # 40000 / 100 -> degree 400 -> ceil(sqrt(400) * 8) == 160. A hundredfold
    # degree buys a tenfold fanout, not a hundredfold one.
    assert infer_target_edge_fanout(
        PATHS, row_count=rows(orders=40_000, users=100)) == 160


@pytest.mark.parametrize("degree, expected", [
    (0, 64),
    (1, 64),
    (64, 64),
    (65, 65),
    (400, 160),
    (4096, 512),
    (10**9, 512),
])
def test_the_fanout_stays_within_the_inferred_bounds(degree, expected):
    assert clamp_inferred_fanout(degree) == expected


def test_a_measured_entity_degree_is_preferred_over_the_row_estimate():
    row_count = rows(orders=40_000, users=100)

    result = infer_target_edge_fanout(PATHS, row_count=row_count,
                                      avg_entity_degree=lambda: 4)

    # ceil(sqrt(4) * 8) == 16, clamped up to the floor.
    assert result == 64
    # The row counts cost a table-statistics computation on the monolith, so
    # the branch that does not need them must not ask for them.
    assert row_count.asked == []


def test_no_accessor_is_consulted_without_a_target_to_entity_path():
    row_count = rows(orders=40_000, users=100)
    degree_reads = []

    def avg_entity_degree():
        degree_reads.append(True)
        return 400

    assert infer_target_edge_fanout([], row_count=row_count,
                                    avg_entity_degree=avg_entity_degree) \
        is None
    assert row_count.asked == []
    assert degree_reads == []


@pytest.mark.parametrize("counts", [
    {
        "orders": 400
    },
    {
        "users": 100
    },
    {},
])
def test_unreadable_row_counts_report_no_estimate(counts):
    # `None` means "leave the edge at its hop default", never "sample zero".
    assert infer_target_edge_fanout(PATHS, row_count=rows(**counts)) is None


@pytest.mark.parametrize("counts", [
    {
        "orders": 0,
        "users": 100
    },
    {
        "orders": 400,
        "users": 0
    },
    {
        "orders": 0,
        "users": 0
    },
])
def test_an_empty_table_estimates_the_lower_bound_not_no_estimate(counts):
    # A caller that has already marked the edge `inferred` from these same
    # paths cannot absorb `None`; the monolith writes it straight into the
    # hop entry, where it is rejected as a non-integer. Zero rows is a
    # measurement, so it resolves to the floor -- and an entity table with
    # no rows never divides.
    assert infer_target_edge_fanout(PATHS, row_count=rows(**counts)) == 64


def test_a_multi_edge_path_reports_no_estimate():
    paths = [[("orders", "user_id", "users"), ("users", "org_id", "orgs")]]

    assert infer_target_edge_fanout(paths,
                                    row_count=rows(orders=400,
                                                   users=100)) is None


def test_an_absent_degree_statistic_falls_through_to_the_row_estimate():
    result = infer_target_edge_fanout(PATHS,
                                      row_count=rows(orders=40_000,
                                                     users=100),
                                      avg_entity_degree=lambda: None)

    assert result == 160
