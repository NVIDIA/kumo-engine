# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Generate and upload tiny local Parquet and CSV tables to Kumo.

Set ``KUMO_API_KEY`` and ``KUMO_PROJECT``. All example data is generated in a
temporary directory, so this script has no external dataset or download.
"""
import os
import tempfile

import pandas as pd

import kumoai_studio as kumo
from kumoai_studio.connector import (
    FileUploadConnector,
    delete_uploaded_table,
    upload_table,
)
from kumoai_studio.graph import Table

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]


def main():
    """Demonstrate file upload functionality."""
    kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "articles.parquet")
        csv_path = os.path.join(tmpdir, "sensors.csv")
        example_data = pd.DataFrame({
            'sensor_id': [101, 102, 103],
            'reading': [1.5, 2.0, 2.5],
            'observed_at':
            pd.to_datetime([
                '2026-07-01T00:00:00Z',
                '2026-07-01T00:05:00Z',
                '2026-07-01T00:10:00Z',
            ]),
        })
        example_data.to_parquet(parquet_path, index=False)
        example_data.to_csv(csv_path, index=False)

        parquet_table_name = "articles"
        csv_table_name = "transactions"

        uploaded = []
        try:
            # Upload local files
            upload_table(name=parquet_table_name, path=parquet_path)
            uploaded.append((parquet_table_name, 'parquet'))
            print(f"Successfully uploaded table '{parquet_table_name}'.")
            upload_table(name=csv_table_name, path=csv_path)
            uploaded.append((csv_table_name, 'csv'))
            print(f"Successfully uploaded table '{csv_table_name}'.")

            # Access the uploaded tables
            parquet_connector = FileUploadConnector(file_type="parquet")
            csv_connector = FileUploadConnector(file_type="csv")

            # Create a Kumo Tables
            kumo_parquet_table = Table.from_source_table(
                source_table=parquet_connector[parquet_table_name])
            kumo_csv_table = Table.from_source_table(
                source_table=csv_connector[csv_table_name])

            print(f"Tables accessed: {kumo_parquet_table.source_name} "
                  f"and {kumo_csv_table.source_name}")

        finally:
            for table_name, file_type in reversed(uploaded):
                delete_uploaded_table(name=table_name, file_type=file_type)
                print(f"Deleted '{table_name}'.")


if __name__ == "__main__":
    main()
