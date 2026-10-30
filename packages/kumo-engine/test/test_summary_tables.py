# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from kumo_core.summaries import Histogram, Table

from kumo.engine._summary_tables import SummaryTables, table_json


def test_a_histogram_table_closes_its_end_bins_at_the_values_extent():
    tables = SummaryTables()
    tables.add_histogram(Histogram('holdout/histogram/actual', (0.0, 1.0),
                                   (1, 2, 3), (-2.5, 4.0)))
    tables.add_histogram(Histogram('holdout/histogram/predicted', (0.0, 1.0),
                                   (0, 5, 1), (-1.0, 6.0)))

    table, = tables.tables()

    assert table.columns == ('from', 'to', 'actual', 'predicted',
                             'actual cumulative share',
                             'predicted cumulative share')
    assert [row[:4] for row in table.rows] == [
        (-2.5, 0.0, 1, 0), (0.0, 1.0, 2, 5), (1.0, 6.0, 3, 1)]


def test_tables_round_measurements_and_keep_definitions_and_counts():
    table = Table('holdout/by_threshold', ('threshold', 'tp', 'precision'), (
        (0.07 + 1e-17 * 3, 12, 0.48774387193112),
        (1 / 3, 7, 27.610799999999998),
    ))

    assert json.loads(table_json(table))['data'] == [
        [0.07, 12, 0.487744], [0.3333333333, 7, 27.6108]]


def test_a_histogram_table_leaves_out_empty_bins():
    tables = SummaryTables()
    tables.add_histogram(Histogram('pred/histogram/score', (0.0, 0.5, 1.0),
                                   (0, 4, 0, 1), (0.1, 1.0)))

    table, = tables.tables()

    assert table.rows == ((0.0, 0.5, 4, 0.8), (1.0, 1.0, 1, 1.0))
