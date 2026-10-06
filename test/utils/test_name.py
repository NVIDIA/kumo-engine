# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.utils.names import (
    fully_qualified_table_name,
    get_data_source,
    get_table_name,
    is_fully_qualified_name,
)


def test_is_fully_qualified_name():
    assert not is_fully_qualified_name("table")
    assert is_fully_qualified_name("csv.table")
    assert not is_fully_qualified_name("csv.table.col")


def test_fully_qualified_table_name():
    # Table name, data source specified.
    assert fully_qualified_table_name("table", "csv") == "csv.table"
    # Already FQN should return the same
    assert fully_qualified_table_name("csv.table") == "csv.table"
    # Tuple, list format.
    assert fully_qualified_table_name(["csv", "table"]) == "csv.table"
    assert fully_qualified_table_name(("csv", "table")) == "csv.table"

    with pytest.raises(AssertionError, match="length 2"):
        fully_qualified_table_name(["csv", "table1", "table2"])
        fully_qualified_table_name(["csv"])

    # No data source specified should throw error.
    with pytest.raises(NotImplementedError, match="table_name"):
        fully_qualified_table_name("table_name")

    # Fully qualified name with matching data source.
    assert fully_qualified_table_name("csv.table", "csv") == "csv.table"
    # Fully qualified name with different data source should throw error.
    with pytest.raises(AssertionError):
        fully_qualified_table_name("csv.table", "other")


def test_get_table_name():
    assert get_table_name("csv.table") == "table"


def test_get_data_source():
    assert get_data_source("csv.table") == "csv"
