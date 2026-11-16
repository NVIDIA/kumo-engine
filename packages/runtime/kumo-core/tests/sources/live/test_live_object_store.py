# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier: the object-store reader against real S3, row-verified.

``s3://kumo-public-datasets/`` is what the hosted SDK example asserts against,
so reading it here is the portability evidence. Multi-part directory reads are
covered against synthetic objects in ``test_object_store.py``; no fixture in a
customer or production bucket is named here.
"""

from __future__ import annotations

import os

import pytest
from kumo_core.sources import layout
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumoapi.typing import Dtype
from tests.sources.conformance import SourceReaderConformance

PUBLIC_BUCKET = "kumo-public-datasets"
PUBLIC_ROOT = f"{PUBLIC_BUCKET}/customerltv_mini"
REGION = "us-west-2"

#: Established with `pyarrow.dataset.Dataset.count_rows`, independently of the
#: reader under test.
PUBLIC_ROWS = {"customer": 4373, "stock": 4070, "transaction": 541909}

#: What the bucket held when this was written; the hosted SDK example asserts
#: the three source tables alone.
PUBLIC_TABLES = [
    "customer",
    "kumo_predictions_embeddings",
    "kumo_predictions_predictions",
    "stock",
    "transaction",
]

if not os.environ.get("KUMO_S3_LIVE"):
    pytest.skip(
        "LIVE TIER NOT RUN: nothing in this file was covered. Set "
        "KUMO_S3_LIVE=1 with AWS credentials that can read "
        f"s3://{PUBLIC_ROOT}/; AWS_PROFILE must be unset for the ambient "
        "instance role to be used.",
        allow_module_level=True,
    )


def filesystem():
    from pyarrow.fs import S3FileSystem
    return S3FileSystem(region=REGION)


def public_reader() -> ObjectStoreSourceReader:
    return ObjectStoreSourceReader(filesystem(), root=PUBLIC_ROOT)


def test_the_public_bucket_lists_the_tables_the_hosted_example_names():
    found = layout.discover_tables(
        layout.scan(filesystem(), PUBLIC_ROOT,
                    max_entries=layout.MAX_SUPPORTED_FILES))
    assert set(PUBLIC_ROWS) <= set(found)
    assert found == PUBLIC_TABLES


def test_the_public_customer_table_reports_the_column_the_example_pins():
    assert public_reader().dtypes("customer") == {"CustomerID": Dtype.float64}


def test_a_single_object_table_resolves_without_a_directory():
    reader = ObjectStoreSourceReader(filesystem(), root=PUBLIC_BUCKET)
    path, kind = reader.resolve("metr_la/sensors.parquet")
    assert path == f"{PUBLIC_BUCKET}/metr_la/sensors.parquet"
    assert kind == layout.PARQUET
    assert reader.dtypes("metr_la/sensors") == {
        "sensor_id": Dtype.string,
        "sensor_index": Dtype.int32,
    }


class TestLiveObjectStoreConformance(SourceReaderConformance):
    """The shared contract against a real bucket."""
    def reader(self) -> ObjectStoreSourceReader:
        return public_reader()

    def table(self) -> str:
        return "transaction"

    def columns(self):
        return {
            "InvoiceNo": Dtype.string,
            "Quantity": Dtype.int64,
            "UnitPrice": Dtype.float64,
        }

    def truth_rows(self) -> int:
        return PUBLIC_ROWS["transaction"]
