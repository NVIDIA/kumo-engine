# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sample datasets are installed with the package and read in place."""

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from kumoapi.typing import Stype

from kumo.engine import FileConnector, Table, datasets


def test_shop_tables_resolve_through_file_connector(tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    root = Path(datasets.shop())
    assert root.is_absolute() and root.is_dir()
    files = FileConnector(root_dir=datasets.shop())
    assert files.table_names() == ['orders', 'users']

    users = Table.from_source_table(files['users'], primary_key='user_id')
    orders = Table.from_source_table(files['orders'], primary_key='order_id',
                                     time_column='order_time')
    assert [column.name for column in users.definition.cols] == ['user_id']
    columns = {column.name: column for column in orders.definition.cols}
    assert sorted(columns) == ['order_id', 'order_time', 'user_id']
    assert columns['order_time'].stype == Stype.timestamp

    # The shop() docstring quotes these.
    assert pq.read_metadata(root / 'users.parquet').num_rows == 1000
    times = pq.read_table(root / 'orders.parquet')['order_time']
    assert len(times) == 12000
    assert pc.min(times).as_py().year == pc.max(times).as_py().year == 2025
