# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a reader may claim about the Parquet it produced."""

import pytest
from kumo_core.sources.types import ReadMode, StagedTable


def staged(**overrides) -> StagedTable:
    fields = dict(
        directory="/tmp/orders",
        paths=["/tmp/orders/part-00000.parquet"],
        version="1698343200",
        source_size=1024,
        mode_used=ReadMode.STREAMED,
    )
    fields.update(overrides)
    return StagedTable(**fields)


@pytest.mark.parametrize("field,value", [
    ("paths", "/tmp/orders/part-00000.parquet"),
    ("paths", ["/a/x.parquet", "/a/x.parquet"]),
    ("paths", ["/a/x.parquet", ""]),
    ("version", ""),
    ("source_size", -1),
    ("num_rows", -1),
    ("scanned_bytes", -1),
])
def test_a_claim_no_read_could_have_produced_is_refused(field, value):
    with pytest.raises(ValueError):
        staged(**{field: value})


def test_a_read_that_produced_no_paths_is_refused_unless_it_read_no_rows():
    """A warehouse that writes no file for an empty table is reporting an
    empty table; one that writes none for a table it says holds rows has
    lost them."""
    assert staged(paths=[], num_rows=0).paths == ()
    for rows in (None, 1):
        with pytest.raises(ValueError):
            staged(paths=[], num_rows=rows)
