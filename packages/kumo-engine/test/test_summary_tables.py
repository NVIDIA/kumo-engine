# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from kumo_core.summaries import Table

from kumo.engine._summary_tables import Routed, route, table_json


def test_tables_round_measurements_and_keep_definitions_and_counts():
    table = Table('holdout/by_threshold', ('threshold', 'tp', 'precision'), (
        (0.07 + 1e-17 * 3, 12, 0.48774387193112),
        (1 / 3, 7, 27.610799999999998),
    ))

    assert json.loads(table_json(table))['data'] == [
        [0.07, 12, 0.487744], [0.3333333333, 7, 27.6108]]


def test_a_summary_type_no_tracker_knows_is_skipped():
    assert route([object()]) == Routed()
