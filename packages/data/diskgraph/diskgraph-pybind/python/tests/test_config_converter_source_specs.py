# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from diskgraph import DiskgraphConfigConverter


def test_converter_passes_an_explicit_source_spec_through_verbatim():
    """An explicit ``source_specs`` entry replaces path building entirely.

    The converter must not touch the filesystem for such a table, so the
    root below deliberately does not exist.
    """
    config = {
        "name": "orders_graph",
        "tables": [{
            "table_name": "orders",
            "data_source": "warehouse",
            "source_name": "orders",
            "data_version": "v1",
            "pkey": {
                "name": "id",
                "dtype": "int64",
                "stype": "ID",
            },
            "cols": [],
        }],
    }
    source = {
        "type": "object_store_parquet",
        "objects": [{
            "path": "s3://staging-bucket/orders/part-00000.parquet",
            "e_tag": "d41d8cd98f00b204e9800998ecf8427e",
            "size": 123_456,
        }],
        "auth_profile": "ambient",
    }

    converted = DiskgraphConfigConverter(
        config,
        "/path/that/does/not/exist",
        {"orders": "v1"},
        source_specs={
            "orders": {
                "source": source,
                "source_size": 123_456,
            },
        },
    ).convert()

    assert converted["tables"][0]["source"] == source
    assert converted["tables"][0]["source_size"] == 123_456
