# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Connecting data on Amazon S3.

Set ``KUMO_API_KEY``. ``KUMO_PROJECT`` is optional and defaults to the
canonical Default project. ``KUMO_S3_IDENTITY_ID`` may select an
administrator-managed S3 identity assigned to the project's group.

When ``KUMO_S3_IDENTITY_ID`` is omitted, Kumo uses its canonical ambient AWS
workload identity. An administrator must assign that identity to the project's
group, and the deployment must explicitly enable ambient S3 and allow its
workload role. Ambient S3 is disabled by default; it is never an anonymous
credential fallback.
"""
import os

import pandas as pd
from kumoapi.source_table import SourceColumn
from kumoapi.typing import Dtype, Stype

import kumoai_studio as kumo
from kumoai_studio.connector import S3Connector

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
DATA_ROOT = "s3://kumo-public-datasets/customerltv_mini/"
SENSOR_URI = "s3://kumo-public-datasets/metr_la/sensors.parquet"


def _create_any_uri_connector(identity_id: str | None) -> S3Connector:
    """Use an explicit identity when supplied, otherwise request ambient."""
    if identity_id:
        return kumo.S3Connector(identity=identity_id)
    return kumo.S3Connector()


def main() -> None:
    """Create rooted and any-URI S3 connectors and inspect sample tables."""
    identity_id = os.getenv("KUMO_S3_IDENTITY_ID")
    project_id = os.getenv("KUMO_PROJECT")
    kumo.init(
        url=API_URL,
        api_key=os.environ["KUMO_API_KEY"],
        project=project_id,
    )

    # 1. Create a rooted connector. Tables may be at any depth below the
    # prefix.
    connector = kumo.S3Connector(
        name="customer_ltv_source",
        identity=identity_id,
        root_dir=DATA_ROOT,
    )

    public_data = kumo.S3Connector(
        name="public_data",
        identity=identity_id,
        root_dir="s3://kumo-public-datasets/",
    )
    public_data.table("metr_la/sensors.parquet")
    public_data.table_from_uri(SENSOR_URI)

    # 2. An any-URI connector can address unrelated buckets and prefixes. It
    # requires a full URI for every table and cannot list tables. Supplying an
    # explicit identity keeps this example usable when ambient S3 is disabled.
    # Omitting KUMO_S3_IDENTITY_ID deliberately exercises the no-argument
    # ambient flow described above. The sensor metadata object is a bounded
    # smoke input.
    any_s3 = _create_any_uri_connector(identity_id)
    any_s3.table(SENSOR_URI)

    # Applications can explicitly select both resources without changing the
    # process-global project. This optional branch runs when both IDs are set.
    if project_id and identity_id:
        kumo.S3Connector(
            name="explicit_s3",
            identity=identity_id,
            project_id=project_id,
        )

    # 3. Retrieve the same rooted connector by its project-local name.
    connector = kumo.S3Connector.get_by_name("customer_ltv_source")

    # 4. List immediate tables behind a rooted connector's configured prefix.
    # Prediction outputs are written back beside the source tables, so
    # discovery finds more than the three below.
    table_names = connector.table_names()
    assert {'customer', 'stock', 'transaction'} <= set(table_names)

    # 5. Read the schema information from a raw table in this connector.
    customer = connector['customer']
    columns = customer.columns
    assert columns == [
        SourceColumn(name='CustomerID', stype=Stype.numerical,
                     dtype=Dtype.float64, is_primary=False)
    ]
    # Error handling if a table name is wrong:
    try:
        connector['nonexistent']
    except ValueError as error:
        assert "does not exist" in str(error)
    else:
        raise RuntimeError("Exception not raised.")

    # 6. Read a limited number of samples from a raw table.
    customer = connector['customer']
    head_rows = customer.head(num_rows=10)
    assert isinstance(head_rows, pd.DataFrame)
    assert len(head_rows) == 10


if __name__ == "__main__":
    main()
